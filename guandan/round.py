"""Atomic single-hand GuanDan rules engine for A02."""
from __future__ import annotations
from itertools import combinations
from typing import Iterable, Iterator
import numpy as np
from .cards import Card, Rank, effective_rank_value, sort_cards
from .combos import Combination, CombinationKind, can_beat, recognize_combinations, require_combination
from .state import CommittedAction, EpisodeTerminatedError, GuandanError, IllegalActionError, Phase, RoundState, StateInvariantError, TEAM_OF
_BOMBS={CombinationKind.RANK_BOMB,CombinationKind.STRAIGHT_FLUSH,CombinationKind.FOUR_KINGS}
def _active(s):return tuple(x for x in range(4) if x not in s.finished_ranks)
def _next(s,seat):
    for i in range(1,5):
        x=(seat+i)%4
        if x not in s.finished_ranks:return x
    raise StateInvariantError("no active seat remains")
def _key(a):
    d=a.declaration;k=a.kind.value if isinstance(a.kind,CombinationKind) else str(a.kind);dk=() if d is None else (d.kind.value,d.physical_ids(),d.comparison_rank,d.comparison_key,tuple(sorted((i,r,su) for i,(r,su) in d.wild_assignments.items())),d.natural_wild_ids)
    return (a.player_id,k,tuple(c.card_id for c in a.cards),dk)
def _dk(d):return (d.kind,d.physical_ids(),d.level_rank,d.comparison_rank,d.comparison_key,tuple(sorted((i,r,s) for i,(r,s) in d.wild_assignments.items())),d.natural_wild_ids)
def enumerate_plays(hand, *, level_rank, current=None):
    from itertools import product
    from .cards import NON_JOKER_RANKS, WINDOW_RANKS
    h=tuple(sort_cards(hand)); out={}; cache={}
    if current is not None:
        if not isinstance(current,Combination) or current.level_rank is not level_rank or not any(_dk(x)==_dk(current) for x in recognize_combinations(current.cards,level_rank=level_rank)):
            raise ValueError("current combination declaration is not valid for its physical cards and level")
    allowed=set(_BOMBS | {CombinationKind.SINGLE,CombinationKind.PAIR,CombinationKind.TRIPLE,CombinationKind.FULL_HOUSE,CombinationKind.STRAIGHT,CombinationKind.PAIR_SEQUENCE,CombinationKind.TRIPLE_SEQUENCE}) if current is None else (_BOMBS if current.kind in _BOMBS else {current.kind}|_BOMBS)
    wilds=tuple(c for c in h if c.is_level_wild(level_rank)); natural={r:tuple(c for c in h if c.rank is r and not c.is_level_wild(level_rank) and not c.is_joker) for r in NON_JOKER_RANKS}; jokers={r:tuple(c for c in h if c.rank is r) for r in (Rank.SMALL_JOKER,Rank.BIG_JOKER)}
    def emit(cards):
        ids=tuple(sorted(c.card_id for c in cards)); ds=cache.get(ids)
        if ds is None:ds=recognize_combinations(tuple(sorted(cards,key=lambda c:c.card_id)),level_rank=level_rank);cache[ids]=ds
        for d in ds:
            if d.kind in allowed and (current is None or can_beat(d,current)):out[_dk(d)]=d
    if CombinationKind.SINGLE in allowed:
        for c in h:emit((c,))
    for r in NON_JOKER_RANKS:
        pool=natural[r]
        for n,k in ((2,CombinationKind.PAIR),(3,CombinationKind.TRIPLE)):
            if k not in allowed:continue
            for nc in range(max(0,n-len(wilds)),min(n,len(pool))+1):
                for a in combinations(pool,nc):
                    for b in combinations(wilds,n-nc):emit(a+b)
        if CombinationKind.RANK_BOMB in allowed:
            for n in range(4,min(10,len(pool)+len(wilds))+1):
                for nc in range(max(0,n-len(wilds)),min(n,len(pool))+1):
                    for a in combinations(pool,nc):
                        for b in combinations(wilds,n-nc):emit(a+b)
    if CombinationKind.PAIR in allowed:
        for r in (Rank.SMALL_JOKER,Rank.BIG_JOKER):
            for a in combinations(jokers[r],2):emit(a)
    if CombinationKind.FULL_HOUSE in allowed:
        for tr in NON_JOKER_RANKS:
            for pr in NON_JOKER_RANKS:
                if tr is pr:continue
                for tc in range(4):
                    for pc in range(3):
                        wc=5-tc-pc
                        if not 0<=wc<=min(2,len(wilds)) or tc>len(natural[tr]) or pc>len(natural[pr]):continue
                        for a in combinations(natural[tr],tc):
                            for b in combinations(natural[pr],pc):
                                for w in combinations(wilds,wc):emit(a+b+w)
    def seq(window,copies):
        total=len(window)*copies; pools=[tuple(c for c in h if c.rank is r) for r in window]; wild_ids={c.card_id for c in wilds}
        for counts in product(range(copies+1),repeat=len(window)):
            wc=total-sum(counts)
            if not 0<=wc<=min(2,len(wilds)) or any(n>len(pool) for n,pool in zip(counts,pools)):continue
            for groups in product(*(combinations(pool,n) for pool,n in zip(pools,counts))):
                cards=tuple(c for g in groups for c in g); ids={c.card_id for c in cards}; used=sum(c.card_id in wild_ids for c in cards); extra=tuple(c for c in wilds if c.card_id not in ids)
                if used+wc>2:continue
                for x in combinations(extra,wc):emit(cards+x)
    def ws(n):
        base=tuple(tuple(WINDOW_RANKS[i:i+n]) for i in range(len(WINDOW_RANKS)-n+1));extra={5:(Rank.ACE,Rank.TWO,Rank.THREE,Rank.FOUR,Rank.FIVE),3:(Rank.ACE,Rank.TWO,Rank.THREE),2:(Rank.ACE,Rank.TWO)}[n];return base+(extra,)
    if CombinationKind.STRAIGHT in allowed or CombinationKind.STRAIGHT_FLUSH in allowed:
        for w in ws(5):seq(w,1)
    if CombinationKind.PAIR_SEQUENCE in allowed:
        for w in ws(3):seq(w,2)
    if CombinationKind.TRIPLE_SEQUENCE in allowed:
        for w in ws(2):seq(w,3)
    if CombinationKind.FOUR_KINGS in allowed:
        for a in combinations(jokers[Rank.SMALL_JOKER],2):
            for b in combinations(jokers[Rank.BIG_JOKER],2):emit(a+b)
    return tuple(sorted(out.values(),key=lambda d:(d.kind.value,d.comparison_key,d.physical_ids(),tuple(sorted(d.wild_assignments.items())),d.natural_wild_ids)))

def enumerate_legal_actions(s):
    if s.done or s.phase is Phase.TERMINAL:return ()
    if s.phase is Phase.TRIBUTE:
        c=tribute_card_for(s,s.active_seat);return () if c is None else (CommittedAction(s.active_seat,"tribute",(c,)),)
    if s.phase is Phase.RETURN:return tuple(CommittedAction(s.active_seat,"return",(c,)) for c in return_cards_for(s,s.active_seat))
    if s.phase is not Phase.PLAY:raise StateInvariantError("unsupported phase")
    a=[CommittedAction(s.active_seat,"pass")] if s.current_winning is not None else []
    a.extend(CommittedAction(s.active_seat,d.kind,d.cards,d) for d in enumerate_plays(s.hands[s.active_seat],level_rank=s.level_rank,current=s.current_winning))
    return tuple(sorted({_key(x):x for x in a}.values(),key=_key))
def legal_actions(s):return enumerate_legal_actions(s)
def iter_legal_actions(s)->Iterator[CommittedAction]:yield from enumerate_legal_actions(s)
def tribute_card_for(s,donor):
    if donor not in s.tribute_donors:return None
    e=[c for c in s.hands[donor] if not c.is_joker and not c.is_level_wild(s.level_rank)]
    return min(e,key=lambda c:(-effective_rank_value(c.rank,s.level_rank),c.card_id)) if e else None
def return_cards_for(s,recipient):
    if s.pending_return is None or s.pending_return[0]!=recipient:return ()
    ranks={Rank.TWO,Rank.THREE,Rank.FOUR,Rank.FIVE,Rank.SIX,Rank.SEVEN,Rank.EIGHT,Rank.NINE,Rank.TEN}
    return tuple(sorted((c for c in s.hands[recipient] if not c.is_joker and not c.is_level_wild(s.level_rank) and c.rank in ranks),key=lambda c:(effective_rank_value(c.rank,s.level_rank),c.card_id)))
def _held(s,a):
    have={c.card_id for c in s.hands[a.player_id]};ids=[c.card_id for c in a.cards]
    if len(ids)!=len(set(ids)) or not set(ids).issubset(have):raise IllegalActionError("action selects a missing or duplicated physical card")
def _valid(s,a,d):return d.level_rank is s.level_rank and d.kind is a.kind and tuple(sorted(c.card_id for c in d.cards))==tuple(sorted(c.card_id for c in a.cards)) and any(_dk(x)==_dk(d) for x in recognize_combinations(a.cards,level_rank=s.level_rank))
def _term(s):
    s.done=True;s.phase=Phase.TERMINAL;s.current_winning=None;s.current_winner_seat=None;s.consecutive_passes=0;s.winner_team=TEAM_OF[s.finished_ranks[0]];s.outcome_class="DOUBLE_DOWN" if TEAM_OF[s.finished_ranks[0]]==TEAM_OF[s.finished_ranks[1]] else ("HEAD_THIRD" if TEAM_OF[s.finished_ranks[0]]==TEAM_OF[s.finished_ranks[2]] else "HEAD_LAST");s.rewards=np.asarray([1.0 if TEAM_OF[x]==s.winner_team else -1.0 for x in range(4)],dtype=np.float32);s.team_rewards=np.asarray([1.0 if x==s.winner_team else -1.0 for x in range(2)],dtype=np.float32);s.finish_committed_step=s.committed_step;s.finish_token_step=s.committed_step
def _append(s,a):
    for i in range(1,5):
        x=(a+i)%4
        if x not in s.finished_ranks:s.finished_ranks.append(x)
def _finish(s,x):
    if x not in s.emptied_seats:s.emptied_seats.append(x)
    if x not in s.finished_ranks:s.finished_ranks.append(x)
    if len(s.finished_ranks)>=2 and TEAM_OF[s.finished_ranks[0]]==TEAM_OF[s.finished_ranks[1]]:_append(s,s.finished_ranks[1]);_term(s)
    elif len(s.finished_ranks)==3:_append(s,s.finished_ranks[-1]);_term(s)
def _resolve(s):
    if s.previous_result is None or set(s.tribute_selections)!=set(s.tribute_donors):raise StateInvariantError("tribute selections incomplete")
    p=[(d,s.tribute_selections[d]) for d in s.tribute_donors]
    if s.previous_result.outcome_class=="DOUBLE_DOWN":p.sort(key=lambda x:(-effective_rank_value(x[1].rank,s.level_rank),x[0],x[1].card_id));rec=[s.previous_result.seat_with_rank(1),s.previous_result.seat_with_rank(2)]
    else:rec=[s.previous_result.seat_with_rank(1)]
    s.tribute_order[:]=[d for d,_ in p];s.tribute_escrow.clear();s.tribute_transfers.clear()
    for i,(d,c) in enumerate(p):r=rec[min(i,len(rec)-1)];s.hands[r].append(c);s.tribute_transfers.append((d,r,c))
    s.return_obligations[:]=[(r,d) for d,r,_ in s.tribute_transfers];s.phase=Phase.RETURN;s.active_seat=s.return_obligations[0][0];s.leader_seat=s.active_seat
def _tribute(s,a):
    if a.player_id!=s.active_seat or a.player_id not in s.pending_tribute_donors or len(a.cards)!=1:raise IllegalActionError("invalid tribute action")
    _held(s,a);e=tribute_card_for(s,a.player_id)
    if e is None or a.cards[0].card_id!=e.card_id:raise IllegalActionError("tribute must use highest eligible card")
    c=a.cards[0];s.hands[a.player_id].remove(c);s.tribute_selections[a.player_id]=c;s.tribute_escrow.append(c);s.history.append(CommittedAction(a.player_id,"tribute",(c,)));s.committed_step+=1
    if s.pending_tribute_donors:s.active_seat=min(s.pending_tribute_donors)
    else:_resolve(s)
def _return(s,a):
    if s.pending_return is None or a.player_id!=s.pending_return[0] or len(a.cards)!=1:raise IllegalActionError("invalid return action")
    _held(s,a);c=a.cards[0]
    if c not in return_cards_for(s,a.player_id):raise IllegalActionError("return card is not legal")
    r,d=s.pending_return;s.hands[a.player_id].remove(c);s.hands[d].append(c);s.return_transfers.append((a.player_id,d,c));s.history.append(CommittedAction(a.player_id,"return",(c,)));s.committed_step+=1;s.return_obligations.pop(0)
    if s.return_obligations:s.active_seat=s.return_obligations[0][0]
    else:s.phase=Phase.PLAY;s.leader_seat=s.tribute_order[0] if s.tribute_order else s.previous_result.seat_with_rank(1);s.active_seat=s.leader_seat
def _pass(s,a):
    if s.current_winning is None or a.cards:raise IllegalActionError("invalid pass")
    s.history.append(CommittedAction(a.player_id,"pass"));s.committed_step+=1;s.consecutive_passes+=1;act=_active(s);need=len(act)-(1 if s.current_winner_seat in act else 0)
    if s.consecutive_passes>=max(1,need):
        w=s.current_winner_seat;s.current_winning=None;s.current_winner_seat=None;s.consecutive_passes=0
        if w in s.emptied_seats:s.leader_seat=(w+2)%4 if (w+2)%4 not in s.finished_ranks else _next(s,w)
        else:s.leader_seat=w
        s.active_seat=s.leader_seat
    else:s.active_seat=_next(s,a.player_id)
def _play(s,a):
    if not isinstance(a.kind,CombinationKind) or not a.cards:raise IllegalActionError("invalid play")
    _held(s,a)
    try:d=a.declaration or require_combination(a.cards,level_rank=s.level_rank,kind=a.kind)
    except (TypeError,ValueError) as exc:raise IllegalActionError("cards do not form declared combination") from exc
    if not _valid(s,a,d):raise IllegalActionError("forged or mismatched declaration")
    if s.current_winning is not None and not can_beat(d,s.current_winning):raise IllegalActionError("play does not beat current winning play")
    ids={c.card_id for c in a.cards};s.hands[a.player_id]=[c for c in s.hands[a.player_id] if c.card_id not in ids];s.history.append(CommittedAction(a.player_id,d.kind,tuple(sort_cards(a.cards)),d));s.committed_step+=1;s.current_winning=d;s.current_winner_seat=a.player_id;s.consecutive_passes=0
    if not s.hands[a.player_id]:_finish(s,a.player_id)
    if not s.done:s.active_seat=_next(s,a.player_id)
def _inplace(s,a):
    if s.done or s.phase is Phase.TERMINAL:raise EpisodeTerminatedError("round is already terminal")
    if not isinstance(a,CommittedAction):raise TypeError("action must be a CommittedAction")
    if a.player_id!=s.active_seat:raise IllegalActionError("action is from non-active seat")
    if a.player_id in s.finished_ranks:raise IllegalActionError("finished seat cannot act")
    if s.phase is Phase.TRIBUTE:_tribute(s,a)
    elif s.phase is Phase.RETURN:_return(s,a)
    elif s.phase is Phase.PLAY:_pass(s,a) if a.is_pass else _play(s,a)
    else:raise StateInvariantError("unsupported phase")
def apply_action(s,action):
    if not isinstance(s,RoundState):raise TypeError("state must be a RoundState")
    if s.done or s.phase is Phase.TERMINAL:raise EpisodeTerminatedError("round is already terminal")
    t=s.clone();_inplace(t,action);t.check_invariants();s.overwrite_from(t);return s
def legal_follow(s,cards,*,kind):
    try:t=s.clone();_inplace(t,CommittedAction(s.active_seat,kind,tuple(cards)));t.check_invariants();return True
    except (GuandanError,TypeError,ValueError):return False
__all__=["apply_action","enumerate_legal_actions","enumerate_plays","iter_legal_actions","legal_actions","legal_follow","return_cards_for","tribute_card_for","RoundState","CommittedAction"]
