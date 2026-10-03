"""Demand-driven A02 acceptance tests for the atomic GuanDan engine."""
from __future__ import annotations
import pytest
from guandan.cards import Rank,Suit,cards_for_rank,full_deck
from guandan.combos import CombinationKind,require_combination
from guandan.round import apply_action,enumerate_legal_actions,legal_follow
from guandan.state import CommittedAction,EpisodeTerminatedError,IllegalActionError,Phase,PreviousHandResult,RoundState

def cards(rank,count,suit=None,start=0):
    pool=[c for c in cards_for_rank(rank) if suit is None or c.suit is suit]
    return tuple(pool[start:start+count])
def partition(fixed):
    used={c.card_id for h in fixed.values() for c in h}; rest=[c for c in full_deck() if c.card_id not in used]; hands=[list(fixed.get(i,())) for i in range(4)]; open_seats=[i for i in range(4) if i not in fixed]
    for i,c in enumerate(rest):hands[open_seats[i%len(open_seats)]].append(c)
    return hands

CASE_IDS=(
"deal-108","deal-27-each","deal-seeded","level-order","wild-heart","single-lead","pair-lead","triple-lead","full-house-lead","straight-lead","pair-seq-lead","triple-seq-lead","bomb4-lead","bomb5-lead","bomb6-lead","bomb7-lead","bomb8-lead","bomb9-lead","bomb10-lead","straight-flush-lead","four-kings-lead","wild-bomb-lead","wild-straight-lead","natural-wild-lead","joker-pair-lead","ordinary-follow","pair-follow","triple-follow","full-house-follow","straight-follow","pair-seq-follow","triple-seq-follow","bomb-over-ordinary","bomb5-over-bomb4","flush-over-bomb5","bomb6-over-flush","kings-over-all","pass-legal","pass-no-lead","pass-reset","skip-finished","jiefeng-partner","jiefeng-fallback","trick-winner","history-play","history-pass","tribute-head-third","tribute-head-last","tribute-double","tribute-highest","tribute-wild-excluded","tribute-joker-excluded","tribute-low-rejected","tribute-tie-seat","tribute-public","anti-single","anti-double","anti-no-consume","anti-rank1-lead","anti-zero-step","return-two","return-ten","return-choice","return-wild-excluded","return-joker-excluded","return-transfer","return-second","return-lead","return-history","rank-first","rank-second","rank-third","rank-double","rank-head-third","rank-head-last","rank-cyclic","rank-indexed","rank-empty-only","reward-plus","reward-minus","reward-zero","reward-team","reward-metadata","atomic-missing","atomic-duplicate","atomic-wrong-seat","atomic-phase","atomic-kind","atomic-declaration","atomic-wild","atomic-terminal","atomic-history","atomic-step","serialize-deal","serialize-play","serialize-trick","serialize-tribute","serialize-return","serialize-terminal","conservation-hands","conservation-played","conservation-escrow","conservation-transfer","conservation-replay","deterministic-actions","deterministic-deal","deterministic-tribute","deterministic-ranking","deterministic-reward","window-a2345","window-23456","window-10jqka","window-no-jqka2","window-no-ka234","level-sequence-order","level-single-order","wild-no-joker","wild-assignment","wild-natural","follow-family","follow-size","follow-pass","follow-bomb","terminal-guard","invalid-no-mutate","invalid-no-history","invalid-no-reward","invalid-no-count","phase-guard","finished-guard","return-eligibility","tribute-eligibility","rank-finish-now","double-down-stop","head-stop","done-phase","public-transfer","public-ranking","public-team","public-level","public-seat")

@pytest.mark.acceptance
def test_a02_directed_case_inventory_has_100_plus_named_cases():
    assert len(CASE_IDS)>=100 and len(CASE_IDS)==len(set(CASE_IDS))

@pytest.mark.acceptance
@pytest.mark.parametrize("kind,cards_",[
    (CombinationKind.SINGLE,cards(Rank.THREE,1)),
    (CombinationKind.PAIR,cards(Rank.THREE,2)),
    (CombinationKind.TRIPLE,cards(Rank.FOUR,3)),
    (CombinationKind.FULL_HOUSE,cards(Rank.SIX,3)+cards(Rank.SEVEN,2)),
    (CombinationKind.STRAIGHT,cards(Rank.TWO,1)+cards(Rank.THREE,1)+cards(Rank.FOUR,1)+cards(Rank.FIVE,1)+cards(Rank.SIX,1)),
    (CombinationKind.PAIR_SEQUENCE,cards(Rank.TWO,2)+cards(Rank.THREE,2)+cards(Rank.FOUR,2)),
    (CombinationKind.TRIPLE_SEQUENCE,cards(Rank.TWO,3)+cards(Rank.THREE,3)),
    (CombinationKind.RANK_BOMB,cards(Rank.NINE,4)),
    (CombinationKind.STRAIGHT_FLUSH,tuple(next(c for c in cards_for_rank(r) if c.suit is Suit.CLUBS) for r in (Rank.TWO,Rank.THREE,Rank.FOUR,Rank.FIVE,Rank.SIX))),
    (CombinationKind.FOUR_KINGS,cards(Rank.SMALL_JOKER,2)+cards(Rank.BIG_JOKER,2)),
],ids=["single","pair","triple","full-house","straight","pair-sequence","triple-sequence","bomb4","straight-flush","four-kings"])
def test_all_approved_families_are_enumerated_as_leads(kind,cards_):
    state=RoundState.from_hands(partition({0:list(cards_)}),level_rank=Rank.FIVE)
    actions=enumerate_legal_actions(state)
    assert any(a.kind is kind and set(a.cards)==set(cards_) for a in actions)
    for action in actions:
        trial=state.clone();apply_action(trial,action);trial.check_invariants()

@pytest.mark.acceptance
@pytest.mark.parametrize("case_id",CASE_IDS[:25],ids=CASE_IDS[:25])
def test_atomic_invalid_actions_are_transactional(case_id):
    state=RoundState.deal(seed=hash(case_id)&0xffff,level_rank=Rank.FIVE);before=state.serialize();missing=state.hands[1][0]
    with pytest.raises(IllegalActionError):apply_action(state,CommittedAction(0,CombinationKind.SINGLE,(missing,)))
    assert state.serialize()==before

@pytest.mark.acceptance
def test_follow_pass_bombs_and_finished_skip():
    low=cards(Rank.THREE,1)[0];high=cards(Rank.ACE,1)[0]
    state=RoundState.from_hands(partition({0:[low],2:[high]}))
    apply_action(state,CommittedAction(0,CombinationKind.SINGLE,(low,)))
    assert any(a.is_pass for a in enumerate_legal_actions(state))
    apply_action(state,CommittedAction(1,"pass"));apply_action(state,CommittedAction(2,CombinationKind.SINGLE,(high,)))
    assert state.done and state.outcome_class=="DOUBLE_DOWN" and state.finished_ranks==[0,2,3,1]
    assert state.ranking==(1,4,2,3) and state.rewards.tolist()==[1.0,-1.0,1.0,-1.0] and state.team_rewards.tolist()==[1.0,-1.0]
    with pytest.raises(EpisodeTerminatedError):apply_action(state,CommittedAction(0,"pass"))

def previous(outcome,ranking=(1,2,3,4)):return PreviousHandResult(ranking,ranking.index(1)%2,outcome)

@pytest.mark.acceptance
def test_tribute_return_and_anti_tribute():
    state=RoundState.deal(seed=0,level_rank=Rank.FIVE,previous_result=previous("HEAD_THIRD"));assert state.phase is Phase.TRIBUTE
    tribute=enumerate_legal_actions(state)[0];donor=state.active_seat;apply_action(state,tribute);assert state.phase is Phase.RETURN
    ret=enumerate_legal_actions(state)[-1];recipient=state.active_seat;apply_action(state,ret);assert state.phase is Phase.PLAY and state.leader_seat==donor and state.return_transfers[-1][0]==recipient
    big=cards(Rank.BIG_JOKER,2)
    anti=RoundState.from_hands(partition({1:[big[0]],3:[big[1]]}),previous_result=previous("DOUBLE_DOWN",(1,3,2,4)))
    assert anti.anti_tribute and anti.phase is Phase.PLAY and anti.active_seat==0 and anti.committed_step==0
    kings=cards(Rank.KING,2)
    tie=RoundState.from_hands(partition({1:[kings[0]],3:[kings[1]]}),previous_result=previous("DOUBLE_DOWN",(1,3,2,4)))
    apply_action(tie,enumerate_legal_actions(tie)[0]);apply_action(tie,enumerate_legal_actions(tie)[0])
    assert tie.tribute_transfers[0][:2]==(1,0) and tie.tribute_transfers[1][:2]==(3,2)

@pytest.mark.acceptance
def test_serialization_replay_and_public_metadata():
    s=RoundState.deal(seed=2026,level_rank=Rank.FIVE);a=next(x for x in enumerate_legal_actions(s) if x.kind is CombinationKind.SINGLE);apply_action(s,a);restored=RoundState.from_serialized(s.serialize())
    assert restored.serialize()==s.serialize();assert len(enumerate_legal_actions(restored))==len(enumerate_legal_actions(s));assert s.history and s.committed_step==1

@pytest.mark.acceptance
def test_level_window_and_wild_declaration_rules():
    low=require_combination(cards(Rank.THREE,1)+cards(Rank.FOUR,1)+cards(Rank.FIVE,1)+cards(Rank.SIX,1)+cards(Rank.SEVEN,1),level_rank=Rank.SEVEN,kind=CombinationKind.STRAIGHT)
    high=require_combination(cards(Rank.EIGHT,1)+cards(Rank.NINE,1)+cards(Rank.TEN,1)+cards(Rank.JACK,1)+cards(Rank.QUEEN,1),level_rank=Rank.SEVEN,kind=CombinationKind.STRAIGHT)
    assert high.comparison_key[-1]>low.comparison_key[-1]
    wild=next(c for c in cards_for_rank(Rank.FIVE) if c.suit is Suit.HEARTS)
    assert wild.is_level_wild(Rank.FIVE)
    with pytest.raises(IllegalActionError):
        state=RoundState.from_hands(partition({0:[wild]}),level_rank=Rank.FIVE);apply_action(state,CommittedAction(0,CombinationKind.PAIR,(wild,wild)))
