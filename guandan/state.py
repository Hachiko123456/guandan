"""State primitives for the independent single-hand GuanDan engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable
import numpy as np
from .cards import Card, Rank, Suit, decode_cards, encode_cards, full_deck, sort_cards, validate_level_rank
from .combos import Combination, CombinationKind
class GuandanError(Exception): pass
class IllegalActionError(GuandanError): pass
class EpisodeTerminatedError(GuandanError): pass
class StateInvariantError(GuandanError): pass
class Phase(str,Enum): PLAY="play"; TRIBUTE="tribute"; RETURN="return"; TERMINAL="terminal"
TEAM_OF=(0,1,0,1)
@dataclass(frozen=True,slots=True)
class PreviousHandResult:
    ranking:tuple[int,int,int,int]; winner_team:int; outcome_class:str; tribute_cancelled:bool=False
    def __post_init__(self):
        r=tuple(int(x) for x in self.ranking);w=int(self.winner_team);o=str(self.outcome_class).upper().replace("-","_")
        if len(r)!=4 or set(r)!={1,2,3,4}:raise ValueError("ranking must contain each finishing rank 1..4 exactly once")
        if w not in (0,1) or o not in {"DOUBLE_DOWN","HEAD_THIRD","HEAD_LAST"}:raise ValueError("invalid previous hand result")
        p=[r.index(i) for i in (1,2,3,4)]
        if TEAM_OF[p[0]]!=w:raise ValueError("winner_team does not match rank-1 seat")
        if o=="DOUBLE_DOWN" and TEAM_OF[p[0]]!=TEAM_OF[p[1]]:raise ValueError("DOUBLE_DOWN requires ranks 1 and 2 on one team")
        if o=="HEAD_THIRD" and TEAM_OF[p[0]]!=TEAM_OF[p[2]]:raise ValueError("HEAD_THIRD requires ranks 1 and 3 on one team")
        if o=="HEAD_LAST" and TEAM_OF[p[0]]!=TEAM_OF[p[3]]:raise ValueError("HEAD_LAST requires ranks 1 and 4 on one team")
        object.__setattr__(self,"ranking",r);object.__setattr__(self,"winner_team",w);object.__setattr__(self,"outcome_class",o);object.__setattr__(self,"tribute_cancelled",bool(self.tribute_cancelled))
    def seat_with_rank(self,rank):return self.ranking.index(int(rank))
@dataclass(frozen=True,slots=True)
class CommittedAction:
    player_id:int;kind:CombinationKind|str;cards:tuple[Card,...]=();declaration:Combination|None=None
    def __post_init__(self):
        object.__setattr__(self,"cards",tuple(self.cards))
        if isinstance(self.kind,str) and self.kind not in {"pass","tribute","return"}:object.__setattr__(self,"kind",CombinationKind(self.kind))
    @property
    def is_pass(self):return self.kind=="pass"
@dataclass(slots=True)
class RoundState:
    hands:list[list[Card]];level_rank:Rank=Rank.TWO;active_seat:int=0;leader_seat:int=0;phase:Phase=Phase.PLAY;current_winning:Combination|None=None;current_winner_seat:int|None=None;consecutive_passes:int=0;finished_ranks:list[int]=field(default_factory=list);done:bool=False;rewards:np.ndarray=field(default_factory=lambda:np.zeros(4,dtype=np.float32));committed_step:int=0;history:list[CommittedAction]=field(default_factory=list);previous_result:PreviousHandResult|None=None;tribute_donors:list[int]=field(default_factory=list);tribute_order:list[int]=field(default_factory=list);tribute_selections:dict[int,Card]=field(default_factory=dict);tribute_escrow:list[Card]=field(default_factory=list);tribute_transfers:list[tuple[int,int,Card]]=field(default_factory=list);return_obligations:list[tuple[int,int]]=field(default_factory=list);return_transfers:list[tuple[int,int,Card]]=field(default_factory=list);anti_tribute:bool=False;emptied_seats:list[int]=field(default_factory=list);outcome_class:str|None=None;winner_team:int|None=None;team_rewards:np.ndarray=field(default_factory=lambda:np.zeros(2,dtype=np.float32));finish_committed_step:int|None=None;finish_token_step:int|None=None
    def __post_init__(self):
        self.level_rank=validate_level_rank(self.level_rank)
        if len(self.hands)!=4:raise ValueError("a GuanDan round requires four hands")
        self.hands=[list(sort_cards(h)) for h in self.hands];self.phase=Phase(self.phase)
        if not 0<=self.active_seat<4 or not 0<=self.leader_seat<4:raise ValueError("seat must be in [0,4)")
        self.rewards=np.asarray(self.rewards,dtype=np.float32).copy();self.team_rewards=np.asarray(self.team_rewards,dtype=np.float32).copy();self.finished_ranks=list(self.finished_ranks);self.emptied_seats=list(self.emptied_seats);self.history=list(self.history);self.tribute_donors=list(self.tribute_donors);self.tribute_order=list(self.tribute_order);self.tribute_selections=dict(self.tribute_selections);self.tribute_escrow=list(self.tribute_escrow);self.tribute_transfers=list(self.tribute_transfers);self.return_obligations=list(self.return_obligations);self.return_transfers=list(self.return_transfers)
        if self.previous_result is not None and not self.tribute_donors:self.tribute_donors=self._donors()
        fresh=self.previous_result is not None and self.committed_step==0 and not self.history and not self.tribute_selections and not self.tribute_transfers and not self.return_transfers and not self.anti_tribute
        if fresh:
            if self._anti():self.anti_tribute=True;self.phase=Phase.PLAY;self.leader_seat=self.previous_result.seat_with_rank(1);self.active_seat=self.leader_seat
            else:self.phase=Phase.TRIBUTE;self.active_seat=min(self.tribute_donors);self.leader_seat=self.active_seat
        self.check_invariants()
    @classmethod
    def deal(cls,*,seed=0,level_rank=Rank.TWO,leader_seat=0,previous_result=None,previous_hand_result=None):
        if previous_result is not None and previous_hand_result is not None and previous_result!=previous_hand_result:raise ValueError("pass only one previous-hand result")
        prev=previous_result if previous_result is not None else previous_hand_result;cards=list(full_deck());np.random.default_rng(seed).shuffle(cards);return cls([cards[i*27:(i+1)*27] for i in range(4)],level_rank=level_rank,active_seat=leader_seat,leader_seat=leader_seat,previous_result=prev)
    @classmethod
    def from_hands(cls,hands,*,level_rank=Rank.TWO,leader_seat=0,previous_result=None,previous_hand_result=None):
        if previous_result is not None and previous_hand_result is not None and previous_result!=previous_hand_result:raise ValueError("pass only one previous-hand result")
        return cls([list(h) for h in hands],level_rank=level_rank,active_seat=leader_seat,leader_seat=leader_seat,previous_result=previous_result if previous_result is not None else previous_hand_result)
    def clone(self):return RoundState([list(h) for h in self.hands],self.level_rank,self.active_seat,self.leader_seat,self.phase,self.current_winning,self.current_winner_seat,self.consecutive_passes,list(self.finished_ranks),self.done,self.rewards.copy(),self.committed_step,list(self.history),self.previous_result,list(self.tribute_donors),list(self.tribute_order),dict(self.tribute_selections),list(self.tribute_escrow),list(self.tribute_transfers),list(self.return_obligations),list(self.return_transfers),self.anti_tribute,list(self.emptied_seats),self.outcome_class,self.winner_team,self.team_rewards.copy(),self.finish_committed_step,self.finish_token_step)
    def overwrite_from(self,o):
        self.hands=[list(h) for h in o.hands]
        for n in ("level_rank","active_seat","leader_seat","phase","current_winning","current_winner_seat","consecutive_passes","done","committed_step","previous_result","anti_tribute","outcome_class","winner_team","finish_committed_step","finish_token_step"):setattr(self,n,getattr(o,n))
        for n in ("finished_ranks","history","tribute_donors","tribute_order","tribute_escrow","tribute_transfers","return_obligations","return_transfers","emptied_seats"):setattr(self,n,list(getattr(o,n)))
        self.tribute_selections=dict(o.tribute_selections);self.rewards=o.rewards.copy();self.team_rewards=o.team_rewards.copy()
    @property
    def ranking(self):
        r=[0]*4
        for i,s in enumerate(self.finished_ranks,1):r[s]=i
        return tuple(r)
    @property
    def team_reward(self):return self.team_rewards.copy()
    @property
    def tribute_cancelled(self):return self.anti_tribute
    @property
    def pending_tribute_donors(self):return tuple(s for s in self.tribute_donors if s not in self.tribute_selections)
    @property
    def pending_return(self):return self.return_obligations[0] if self.return_obligations else None
    def _donors(self):return sorted(self.previous_result.seat_with_rank(r) for r in ((3,4) if self.previous_result.outcome_class=="DOUBLE_DOWN" else (4,)))
    def _anti(self):return self.previous_result is not None and {c.card_id for c in full_deck() if c.rank is Rank.BIG_JOKER}.issubset({c.card_id for s in self.tribute_donors for c in self.hands[s] if c.rank is Rank.BIG_JOKER})
    def check_invariants(self):
        if len(self.hands)!=4 or self.rewards.shape!=(4,) or self.team_rewards.shape!=(2,):raise StateInvariantError("invalid state shape")
        if not 0<=self.active_seat<4 or not 0<=self.leader_seat<4 or self.committed_step<0:raise StateInvariantError("invalid seat or committed-step metadata")
        hand=[c.card_id for h in self.hands for c in h];played=[c.card_id for a in self.history if isinstance(a.kind,CombinationKind) for c in a.cards];esc=[c.card_id for c in self.tribute_escrow];ids=hand+played+esc
        if len(ids)!=len(set(ids)) or set(ids)!={c.card_id for c in full_deck()}:raise StateInvariantError("physical cards are not a lossless partition")
        if len(self.finished_ranks)!=len(set(self.finished_ranks)) or any(s not in range(4) for s in self.finished_ranks):raise StateInvariantError("invalid finished ranks")
        if len(self.emptied_seats)!=len(set(self.emptied_seats)) or not set(self.emptied_seats).issubset(self.finished_ranks) or any(self.hands[s] for s in self.emptied_seats):raise StateInvariantError("invalid emptied seats")
        if self.done and (self.phase is not Phase.TERMINAL or len(self.finished_ranks)!=4 or self.outcome_class not in {"DOUBLE_DOWN","HEAD_THIRD","HEAD_LAST"} or self.winner_team not in (0,1) or self.finish_committed_step is None or self.finish_token_step is None):raise StateInvariantError("invalid terminal state")
        if self.done:
            expected=np.asarray([1.0 if TEAM_OF[x]==self.winner_team else -1.0 for x in range(4)],dtype=np.float32)
            if not np.array_equal(self.rewards,expected):raise StateInvariantError("terminal player reward disagrees with winner team")
        if not self.done and (self.phase is Phase.TERMINAL or np.any(self.rewards!=0) or np.any(self.team_rewards!=0)):raise StateInvariantError("invalid nonterminal state")
        if self.previous_result is None and (self.phase in {Phase.TRIBUTE,Phase.RETURN} or self.tribute_donors):raise StateInvariantError("tribute without previous result")
        if self.previous_result is not None:
            if tuple(self.tribute_donors)!=tuple(self._donors()):raise StateInvariantError("tribute donors are not deterministic")
            if self.phase is Phase.TRIBUTE and (self.anti_tribute or not self.pending_tribute_donors or self.active_seat not in self.pending_tribute_donors):raise StateInvariantError("invalid tribute phase")
            if self.phase is Phase.RETURN and (self.anti_tribute or not self.return_obligations or self.active_seat!=self.return_obligations[0][0]):raise StateInvariantError("invalid return phase")
        selected=[c.card_id for c in self.tribute_selections.values()]
        if len(selected)!=len(set(selected)) or set(self.tribute_selections)-set(self.tribute_donors):raise StateInvariantError("invalid tribute selections")
        if self.phase is Phase.TRIBUTE and set(selected)!=set(esc):raise StateInvariantError("tribute escrow mismatch")
        if self.phase is not Phase.TRIBUTE and esc:raise StateInvariantError("resolved tribute has escrow")
    @staticmethod
    def _dec(x):
        if x is None:return None
        return {"kind":x.kind.value,"cards":list(encode_cards(x.cards)),"level_rank":int(x.level_rank),"comparison_rank":int(x.comparison_rank),"comparison_key":list(x.comparison_key),"wild_assignments":[[i,int(r),int(s)] for i,(r,s) in sorted(x.wild_assignments.items())],"natural_wild_ids":list(x.natural_wild_ids)}
    @staticmethod
    def _undec(x):
        if x is None:return None
        return Combination(kind=CombinationKind(x["kind"]),cards=decode_cards(x["cards"]),level_rank=Rank(x["level_rank"]),comparison_rank=Rank(x["comparison_rank"]),comparison_key=tuple(x["comparison_key"]),_wild_assignment_items=tuple((int(i),Rank(r),Suit(s)) for i,r,s in x.get("wild_assignments",[])),_natural_wild_ids=tuple(x.get("natural_wild_ids",[])))
    def serialize(self):
        def a(x):return {"player_id":x.player_id,"kind":x.kind.value if isinstance(x.kind,CombinationKind) else str(x.kind),"cards":list(encode_cards(x.cards)),"declaration":self._dec(x.declaration)}
        return {"level_rank":int(self.level_rank),"active_seat":self.active_seat,"leader_seat":self.leader_seat,"phase":self.phase.value,"hands":[list(encode_cards(h)) for h in self.hands],"finished_ranks":list(self.finished_ranks),"ranking":list(self.ranking),"emptied_seats":list(self.emptied_seats),"done":self.done,"outcome_class":self.outcome_class,"winner_team":self.winner_team,"rewards":self.rewards.tolist(),"team_rewards":self.team_rewards.tolist(),"finish_committed_step":self.finish_committed_step,"finish_token_step":self.finish_token_step,"committed_step":self.committed_step,"current_winning":self._dec(self.current_winning),"current_winner_seat":self.current_winner_seat,"consecutive_passes":self.consecutive_passes,"previous_result":None if self.previous_result is None else {"ranking":list(self.previous_result.ranking),"winner_team":self.previous_result.winner_team,"outcome_class":self.previous_result.outcome_class,"tribute_cancelled":self.previous_result.tribute_cancelled},"tribute_donors":list(self.tribute_donors),"tribute_order":list(self.tribute_order),"tribute_selections":{str(s):c.card_id for s,c in self.tribute_selections.items()},"tribute_escrow":list(encode_cards(self.tribute_escrow)),"tribute_transfers":[{"donor":a,"recipient":b,"card":c.card_id} for a,b,c in self.tribute_transfers],"return_obligations":[list(x) for x in self.return_obligations],"return_transfers":[{"recipient":a,"donor":b,"card":c.card_id} for a,b,c in self.return_transfers],"anti_tribute":self.anti_tribute,"history":[a(x) for x in self.history]}
    @classmethod
    def from_serialized(cls,p):
        prev=p.get("previous_result");prev=None if prev is None else PreviousHandResult(tuple(prev["ranking"]),prev["winner_team"],prev["outcome_class"],prev.get("tribute_cancelled",False));by={c.card_id:c for c in full_deck()};hist=[]
        for x in p.get("history",[]):
            k=x["kind"];k=CombinationKind(k) if k in {q.value for q in CombinationKind} else k;hist.append(CommittedAction(x["player_id"],k,decode_cards(x.get("cards",[])),cls._undec(x.get("declaration"))))
        return cls([list(decode_cards(h)) for h in p["hands"]],level_rank=Rank(p["level_rank"]),active_seat=p["active_seat"],leader_seat=p["leader_seat"],phase=Phase(p["phase"]),current_winning=cls._undec(p.get("current_winning")),current_winner_seat=p.get("current_winner_seat"),consecutive_passes=p.get("consecutive_passes",0),finished_ranks=list(p.get("finished_ranks",[])),done=p.get("done",False),rewards=np.asarray(p.get("rewards",[0,0,0,0]),dtype=np.float32),committed_step=p.get("committed_step",0),history=hist,previous_result=prev,tribute_donors=list(p.get("tribute_donors",[])),tribute_order=list(p.get("tribute_order",[])),tribute_selections={int(k):by[int(v)] for k,v in p.get("tribute_selections",{}).items()},tribute_escrow=list(decode_cards(p.get("tribute_escrow",[]))),tribute_transfers=[(x["donor"],x["recipient"],by[x["card"]]) for x in p.get("tribute_transfers",[])],return_obligations=[tuple(x) for x in p.get("return_obligations",[])],return_transfers=[(x["recipient"],x["donor"],by[x["card"]]) for x in p.get("return_transfers",[])],anti_tribute=p.get("anti_tribute",False),emptied_seats=list(p.get("emptied_seats",[])),outcome_class=p.get("outcome_class"),winner_team=p.get("winner_team"),team_rewards=np.asarray(p.get("team_rewards",[0,0]),dtype=np.float32),finish_committed_step=p.get("finish_committed_step"),finish_token_step=p.get("finish_token_step"))
__all__=["CommittedAction","EpisodeTerminatedError","GuandanError","IllegalActionError","Phase","PreviousHandResult","RoundState","StateInvariantError","TEAM_OF"]
