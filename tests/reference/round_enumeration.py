"""Frozen pre-optimization enumeration oracle, test-only (source 5e70f6a).
The recognizer is shared; independent recognizer parity is tested separately.
"""
from itertools import combinations
from guandan.cards import Rank,sort_cards
from guandan.combos import Combination,CombinationKind,recognize_combinations,can_beat
from guandan.round import _dk
_BOMBS={CombinationKind.RANK_BOMB,CombinationKind.STRAIGHT_FLUSH,CombinationKind.FOUR_KINGS}
def enumerate_plays(hand, *, level_rank, current=None):
    from itertools import product
    from guandan.cards import NON_JOKER_RANKS, WINDOW_RANKS
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

