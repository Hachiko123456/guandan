"""Golden rule regression corpus and opt-in reproducible microbenchmark.

The digests were captured from commit 5e70f6a before the bounded optimization.
No timing threshold is enforced by pytest. For paired, unprofiled measurements:
    python -c "import runpy; runpy.run_path('tests/unit/test_combo_performance.py', run_name='__main__')" --baseline-ref 5e70f6a --repeat 5
Run from the repository root; ordinary pytest never requires Git or measures time.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random

import pytest

from guandan import cards, combos

BASELINE_REF = "5e70f6a0ea482c5a3861021405566593843b7ac5"
# Per-level digest includes ordered declarations and validated comparisons.
GOLDEN = {
    3: '184ea29e6d467ebb80d5ef99b1bc3fff516fd4bd8eb467dc5240c01ba2ec8483',
    4: '1a026c9baff51fc7ca7ea31bd8ddb43006df0da7f6bd4d4b2bd2d101475ed14c',
    5: '2c87f7316f467e7937f9192649d810bb635fa8b7be13089d5f758ae917c504ff',
    6: '4218ac09636cc5b86f0fecca1cb6f44be9f15bbda620ed4c566c571f580d0baf',
    7: '9c179a45e948271bd4c455350d01380c1fcd1fcdc077cc344bc3d59deaf96ee8',
    8: 'a05bd39bbd19a43ca6142069392f26515d5d5f7affed1de2dbb87c40b19d38a1',
    9: '25a048d5f0a31278300c5b064aa72fc97d0cccf0d6673c55ac1283b3349d691d',
    10: 'f096dc232469473df28e7cdf8c40282c6ab0c8515afd6c0d3fbfbfa44de5f758',
    11: 'fbe4591b1bb238d214e897128ff495b6bedcecd5de311d25bf06617a416ace52',
    12: '2f161a035986bfc32faf260df70cd3e4480d11a4f345106d96fc01b3fcc2c530',
    13: '1f466d8aa421fb89457b2d098903f2ec8500fb8d5b3c13c2645d6157ccfb9918',
    14: 'be6d191e777bc07483a06ebf3b58aed1ef259bf68c73edd159d851d6f73565ad',
    15: 'db67a115d62fed080d7e73e3cdce6e3f93e7841b21fb04ab52ee395eb57e2529',
}


def _corpus(level: int) -> tuple[tuple[int, ...], ...]:
    """Both decks, every rank/level, low windows, wild variants and invalid shapes."""
    ranks = tuple(range(3, 16))
    windows = (15, *range(3, 15))
    wilds = ((level - 3) * 4 + 2, (level - 3) * 4 + 56)

    def group(rank, count, suit=None):
        ids = tuple((rank - 3) * 4 + s + d * 54 for d in range(2) for s in range(4)
                    if (suit is None or s == suit))
        return ids[:count]

    cases: list[tuple[int, ...]] = [(), (0, 0), (52, 106), (53, 107), (52, 53), (52, 53, 106, 107)]
    cases.extend((i,) for i in range(108))

    def add_variants(ids):
        cases.append(tuple(ids))
        # Replacement (not insertion) preserves intended family size.
        for n in (1, 2):
            if n > len(ids):
                continue
            for positions in (tuple(range(n)), tuple(range(len(ids) - n, len(ids)))):
                result = list(ids)
                for pos, wild in zip(positions, wilds[:n]):
                    result[pos] = wild
                cases.append(tuple(result))

    for rank in ranks:
        for count in range(2, 9):
            add_variants(group(rank, count))
        # Nine- and ten-card bombs must use off-rank heart-level wilds.
        cases.append(group(rank, 8) + wilds[:1])
        cases.append(group(rank, 8) + wilds)
    for triple in ranks:
        for pair in ranks:
            if triple != pair:
                add_variants(group(triple, 3) + group(pair, 2))
    for length, copies, low in ((5, 1, (14, 15, 3, 4, 5)), (3, 2, (14, 15, 3)), (2, 3, (14, 15))):
        for window in [windows[i:i + length] for i in range(14 - length)] + [low]:
            for suit in range(4):
                # For triples use three distinct cards rather than only the two same-suit copies.
                ids = tuple(i for rank in window for i in (group(rank, copies, suit) if copies < 3 else group(rank, copies)))
                add_variants(ids)
    # Forbidden wraps and oversized runs.
    for window in ((11, 12, 13, 14, 15), (13, 14, 15, 3, 4), windows[:6]):
        add_variants(tuple(group(r, 1)[0] for r in window))
    rng = random.Random(70100 + level)
    for size in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 27):
        for _ in range(24):
            cases.append(tuple(rng.sample(range(108), size)))
    # A canonical recognizer must not depend on input order.
    cases.extend(tuple(reversed(ids)) for ids in cases[::31])
    return tuple(dict.fromkeys(cases))


def _signature(declaration):
    return (
        declaration.kind.value, declaration.physical_ids(), int(declaration.level_rank),
        int(declaration.comparison_rank), declaration.comparison_key,
        tuple(tuple(int(v) for v in item) for item in declaration._wild_assignment_items),
        declaration.natural_wild_ids,
    )


def _digest(card_module, combo_module, level):
    outputs = []
    samples = {}
    for ids in _corpus(level):
        result = combo_module.recognize_combinations(
            tuple(card_module.ALL_CARDS[i] for i in ids), level_rank=card_module.Rank(level))
        outputs.append(tuple(_signature(d) for d in result))
        for declaration in result:
            bucket = samples.setdefault(declaration.kind.value, [])
            if len(bucket) < 4:
                bucket.append(declaration)
    comparisons = []
    for left in itertools.chain.from_iterable(samples.values()):
        for right in itertools.chain.from_iterable(samples.values()):
            try:
                value = combo_module.compare_combinations(left, right)
            except ValueError as exc:
                value = type(exc).__name__
            comparisons.append(value)
    return hashlib.sha256(json.dumps((outputs, comparisons), separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("level", range(3, 16))
def test_ordered_declarations_and_comparisons_match_preoptimization(level):
    assert _digest(cards, combos, level) == GOLDEN[level]


def test_card_tables_match_physical_formula_and_all_level_orders():
    for card_id in range(108):
        c = cards.Card(card_id)
        offset = card_id % 54
        rank = cards.Rank(3 + offset // 4) if offset < 52 else cards.Rank(16 + offset - 52)
        suit = cards.Suit(offset % 4) if offset < 52 else cards.Suit.JOKER
        assert (c.rank, c.suit, c.deck_index, c.deck_offset) == (rank, suit, card_id // 54, offset)
        assert c.is_joker == (offset >= 52)
        assert cards.card_sort_key(c) == (int(rank) - 3, int(suit), card_id // 54, card_id)
        for level in cards.NORMAL_RANKS:
            assert c.is_level_wild(level) == (suit is cards.Suit.HEARTS and rank is level)
            order = tuple(r for r in cards.WINDOW_RANKS if r is not level) + (level, *cards.JOKER_RANKS)
            assert cards.effective_rank_order(level) == order
            assert cards.effective_rank_value(rank, level) == order.index(rank)
    for rank in cards.Rank:
        assert cards.cards_for_rank(rank) == tuple(c for c in cards.ALL_CARDS if c.rank is rank)


@pytest.mark.parametrize("bad", [None, True, 2, 18, "5", [], cards.Rank.SMALL_JOKER, cards.Rank.BIG_JOKER])
def test_invalid_level_rejected_before_table_lookup(bad):
    with pytest.raises(ValueError):
        cards.validate_level_rank(bad)
    with pytest.raises(ValueError):
        cards.effective_rank_order(bad)
    with pytest.raises(ValueError):
        cards.ALL_CARDS[52].is_level_wild(bad)
    assert combos.recognize_combinations((cards.ALL_CARDS[0],), level_rank=bad) == ()



def test_wild_partition_is_computed_once_per_recognition(monkeypatch):
    original = combos._wild_and_natural
    calls = []

    def count_calls(physical, level):
        calls.append((physical, level))
        return original(physical, level)

    monkeypatch.setattr(combos, "_wild_and_natural", count_calls)
    for physical in _bench_cases(cards):
        calls.clear()
        combos.recognize_combinations(physical, level_rank=cards.Rank.FIVE)
        assert len(calls) == 1


def test_impossible_sequence_multiplicities_rejected_before_windows(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("impossible sequence entered window enumeration")

    monkeypatch.setattr(combos, "_sequence_declarations", unexpected)
    for ids in ((0, 1, 2, 3, 54), (0, 1, 2, 3, 54, 55), (0, 4, 8, 12, 52)):
        combos.recognize_combinations(tuple(cards.ALL_CARDS[i] for i in ids), level_rank=cards.Rank.FIVE)


def _load_reference(ref):
    """Explicit developer benchmark only: load the two committed modules in isolation."""
    import subprocess
    import sys
    import types
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    package = types.ModuleType("_combo_reference")
    package.__path__ = []
    sys.modules[package.__name__] = package
    modules = []
    for name in ("cards", "combos"):
        source = subprocess.check_output(["git", "show", f"{ref}:guandan/{name}.py"], cwd=root, text=True, encoding="utf-8")
        module = types.ModuleType(f"{package.__name__}.{name}")
        module.__package__ = package.__name__
        sys.modules[module.__name__] = module
        exec(compile(source, f"{ref}/guandan/{name}.py", "exec"), module.__dict__)
        modules.append(module)
    return tuple(modules)


def _bench_cases(cm):
    def by(rank, suit=0, deck=0):
        return cm.ALL_CARDS[(rank - 3) * 4 + suit + deck * 54]
    def rr(rank, n):
        return cm.cards_for_rank(cm.Rank(rank))[:n]
    return (
        tuple(by(r) for r in (15, 3, 4, 6, 7)),
        tuple(by(r) for r in (15, 3, 4, 5, 6)),
        (by(5, 2), by(6), by(7), by(8), by(9)),
        rr(7, 5), rr(7, 8), rr(6, 3) + rr(7, 2), rr(6, 3) + rr(7, 3),
        tuple(cm.ALL_CARDS[i] for i in (52, 53, 106, 107)),
    )


def _benchmark():
    import argparse
    import cProfile
    import pstats
    import statistics
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default=BASELINE_REF)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--loops", type=int, default=1000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--golden", action="store_true")
    args = parser.parse_args()
    reference_cards, reference_combos = _load_reference(args.baseline_ref)
    if args.golden:
        print(repr({level: _digest(reference_cards, reference_combos, level) for level in range(3, 16)}))
        return
    times = {"baseline": [], "optimized": []}
    modules = {"baseline": (reference_cards, reference_combos), "optimized": (cards, combos)}
    for label, (cm, bm) in modules.items():
        cases = _bench_cases(cm)
        for _ in range(30):
            for case in cases:
                bm.recognize_combinations(case, level_rank=cm.Rank.FIVE)
    for repeat in range(args.repeat):
        # Alternate execution order to reduce drift bias.
        for label in (tuple(modules) if repeat % 2 == 0 else tuple(reversed(modules))):
            cm, bm = modules[label]
            cases = _bench_cases(cm)
            start = time.perf_counter()
            for _ in range(args.loops):
                for case in cases:
                    bm.recognize_combinations(case, level_rank=cm.Rank.FIVE)
            times[label].append(time.perf_counter() - start)
    result = {label: {"seconds": values, "median": statistics.median(values)} for label, values in times.items()}
    result["calls_per_repeat"] = 8 * args.loops
    result["speedup"] = result["baseline"]["median"] / result["optimized"]["median"]
    print(json.dumps(result, indent=2))
    if args.profile:
        for label, (cm, bm) in modules.items():
            profiler = cProfile.Profile()
            cases = _bench_cases(cm)
            profiler.enable()
            for _ in range(args.loops):
                for case in cases:
                    bm.recognize_combinations(case, level_rank=cm.Rank.FIVE)
            profiler.disable()
            print(label)
            pstats.Stats(profiler).sort_stats("cumtime").print_stats(18)


if __name__ == "__main__":
    _benchmark()
