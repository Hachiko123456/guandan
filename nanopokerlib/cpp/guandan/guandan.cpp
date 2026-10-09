#include "guandan.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace nanopokerlib::guandan {
namespace {

constexpr std::array<int, 4> TEAM_OF{0, 1, 0, 1};
constexpr std::array<Rank, 13> PRINTED_RANKS{
    Rank::THREE, Rank::FOUR, Rank::FIVE, Rank::SIX, Rank::SEVEN,
    Rank::EIGHT, Rank::NINE, Rank::TEN, Rank::JACK, Rank::QUEEN,
    Rank::KING, Rank::ACE, Rank::TWO,
};
constexpr std::array<Rank, 13> WINDOW_RANKS{
    Rank::TWO, Rank::THREE, Rank::FOUR, Rank::FIVE, Rank::SIX,
    Rank::SEVEN, Rank::EIGHT, Rank::NINE, Rank::TEN, Rank::JACK,
    Rank::QUEEN, Rank::KING, Rank::ACE,
};
constexpr std::array<Rank, 15> RANK_ORDER{
    Rank::THREE, Rank::FOUR, Rank::FIVE, Rank::SIX, Rank::SEVEN,
    Rank::EIGHT, Rank::NINE, Rank::TEN, Rank::JACK, Rank::QUEEN,
    Rank::KING, Rank::ACE, Rank::TWO, Rank::SMALL_JOKER, Rank::BIG_JOKER,
};
constexpr std::array<const char*, 10> KIND_NAMES{
    "single", "pair", "triple", "full_house", "straight",
    "pair_sequence", "triple_sequence", "rank_bomb",
    "straight_flush", "four_kings",
};
constexpr std::array<const char*, 8> TOKEN_KIND_NAMES{
    "PASS", "COMMIT", "FAMILY", "RANK", "SUIT", "LENGTH", "CARD", "WILD_ASSIGNMENT",
};

bool is_normal_rank(Rank rank) {
    const int value = static_cast<int>(rank);
    return value >= static_cast<int>(Rank::THREE) && value <= static_cast<int>(Rank::TWO);
}

bool is_joker_rank(Rank rank) {
    return rank == Rank::SMALL_JOKER || rank == Rank::BIG_JOKER;
}

void validate_level(Rank level) {
    if (!is_normal_rank(level)) {
        throw std::invalid_argument("level_rank must be a printed non-joker rank");
    }
}

void validate_card_id(int card_id) {
    if (card_id < 0 || card_id >= NUM_PHYSICAL_CARDS) {
        throw std::invalid_argument("card_id must be in [0,108)");
    }
}

bool contains(const std::vector<int>& values, int value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

template<class T>
bool contains_value(const std::vector<T>& values, const T& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::tuple<int, int, int, int> card_sort_key(int card_id) {
    const Card card{card_id};
    return {rank_order_index(card.rank()), static_cast<int>(card.suit()), card.deck_index(), card.id};
}

std::string vector_key(const std::vector<int>& values) {
    std::ostringstream out;
    for (int value : values) {
        out << value << ',';
    }
    return out.str();
}

std::string combination_key(const Combination& combo) {
    std::ostringstream out;
    out << static_cast<int>(combo.kind) << '|';
    for (int id : combo.card_ids) out << id << ',';
    out << '|' << static_cast<int>(combo.level_rank) << '|' << static_cast<int>(combo.comparison_rank) << '|';
    for (int value : combo.comparison_key) out << value << ',';
    out << '|';
    for (const auto& assignment : combo.wild_assignments) {
        out << assignment.card_id << ':' << static_cast<int>(assignment.rank) << ':'
            << static_cast<int>(assignment.suit) << ':' << assignment.natural << ',';
    }
    out << '|';
    for (int id : combo.natural_wild_ids) out << id << ',';
    return out.str();
}

std::string action_key(const Action& action) {
    std::ostringstream out;
    out << action.player_id << '|' << action.kind << '|';
    for (int id : action.card_ids) out << id << ',';
    if (action.declaration) out << '|' << combination_key(*action.declaration);
    return out.str();
}

bool lexicographic_vector_less(const std::vector<int>& left, const std::vector<int>& right) {
    return std::lexicographical_compare(left.begin(), left.end(), right.begin(), right.end());
}

std::vector<std::vector<int>> choose_ids(const std::vector<int>& pool, int count) {
    std::vector<std::vector<int>> result;
    if (count < 0 || count > static_cast<int>(pool.size())) return result;
    std::vector<int> current;
    std::function<void(int, int)> rec = [&](int start, int remaining) {
        if (remaining == 0) {
            result.push_back(current);
            return;
        }
        for (int index = start; index <= static_cast<int>(pool.size()) - remaining; ++index) {
            current.push_back(pool[index]);
            rec(index + 1, remaining - 1);
            current.pop_back();
        }
    };
    rec(0, count);
    return result;
}

std::vector<std::vector<int>> sequence_windows(int length) {
    std::vector<std::vector<int>> windows;
    for (int index = 0; index + length <= static_cast<int>(WINDOW_RANKS.size()); ++index) {
        std::vector<int> window;
        for (int offset = 0; offset < length; ++offset) {
            window.push_back(static_cast<int>(WINDOW_RANKS[index + offset]));
        }
        windows.push_back(std::move(window));
    }
    if (length == 5) {
        windows.push_back({14, 15, 3, 4, 5});
    } else if (length == 3) {
        windows.push_back({14, 15, 3});
    } else if (length == 2) {
        windows.push_back({14, 15});
    } else {
        throw std::invalid_argument("unsupported sequence window length");
    }
    return windows;
}

struct Facts {
    std::vector<int> wilds;
    std::vector<int> natural;
    std::map<Rank, int> counts;
    bool has_joker{};
    std::set<Suit> suits;
};

Facts card_facts(const std::vector<int>& cards, Rank level) {
    Facts facts;
    for (int id : cards) {
        const Card card{id};
        if (card.is_level_wild(level)) {
            facts.wilds.push_back(id);
        } else {
            facts.natural.push_back(id);
            facts.counts[card.rank()] += 1;
            facts.suits.insert(card.suit());
            facts.has_joker = facts.has_joker || card.is_joker();
        }
    }
    return facts;
}

struct AssignmentVariant {
    Rank rank;
    Suit suit;
    bool natural;
};

std::vector<AssignmentVariant> assignment_variants(int card_id, Rank target_rank, Rank level, std::optional<Suit> target_suit) {
    const Card card{card_id};
    if (!card.is_level_wild(level)) throw std::invalid_argument("assignment requires a level wild");
    if (is_joker_rank(target_rank)) return {};
    std::vector<AssignmentVariant> result;
    const bool natural_same = target_rank == level && (!target_suit || *target_suit == Suit::HEARTS);
    if (natural_same) result.push_back({target_rank, Suit::HEARTS, true});
    if (!(natural_same && target_suit && *target_suit == Suit::HEARTS)) {
        result.push_back({target_rank, target_suit.value_or(Suit::CLUBS), false});
    }
    return result;
}

std::vector<std::vector<WildAssignment>> assignment_products(
    const std::vector<int>& wilds,
    const std::vector<Rank>& targets,
    Rank level,
    std::optional<Suit> target_suit = std::nullopt) {
    if (wilds.size() != targets.size()) return {};
    std::vector<std::vector<WildAssignment>> result;
    std::vector<WildAssignment> current;
    std::function<void(std::size_t)> rec = [&](std::size_t index) {
        if (index == wilds.size()) {
            result.push_back(current);
            return;
        }
        for (const auto& variant : assignment_variants(wilds[index], targets[index], level, target_suit)) {
            current.push_back({wilds[index], variant.rank, variant.suit, variant.natural});
            rec(index + 1);
            current.pop_back();
        }
    };
    rec(0);
    return result;
}

std::vector<int> ordinary_key(CombinationKind kind, int size, Rank rank, Rank level) {
    const bool sequence = kind == CombinationKind::STRAIGHT || kind == CombinationKind::PAIR_SEQUENCE || kind == CombinationKind::TRIPLE_SEQUENCE;
    return {0, static_cast<int>(kind), size, sequence ? window_rank_value(rank) : effective_rank_value(rank, level)};
}

std::vector<int> bomb_key(CombinationKind kind, int size, Rank rank, Rank level) {
    if (kind == CombinationKind::FOUR_KINGS) return {9};
    if (kind == CombinationKind::STRAIGHT_FLUSH) return {3, window_rank_value(rank)};
    int bomb_class = 0;
    if (size == 4) bomb_class = 1;
    else if (size == 5) bomb_class = 2;
    else bomb_class = size - 2;
    return {bomb_class, effective_rank_value(rank, level)};
}

Combination build_combo(
    CombinationKind kind,
    const std::vector<int>& cards,
    Rank level,
    Rank comparison_rank,
    std::vector<WildAssignment> assignments = {}) {
    Combination combo;
    combo.kind = kind;
    combo.card_ids = sort_card_ids(cards);
    combo.level_rank = level;
    combo.comparison_rank = comparison_rank;
    combo.comparison_key = combo.is_bomb() ? bomb_key(kind, static_cast<int>(cards.size()), comparison_rank, level)
                                           : ordinary_key(kind, static_cast<int>(cards.size()), comparison_rank, level);
    combo.wild_assignments = std::move(assignments);
    std::sort(combo.wild_assignments.begin(), combo.wild_assignments.end(), [](const auto& a, const auto& b) {
        return a.card_id < b.card_id;
    });
    for (const auto& assignment : combo.wild_assignments) {
        if (assignment.natural) combo.natural_wild_ids.push_back(assignment.card_id);
    }
    std::sort(combo.natural_wild_ids.begin(), combo.natural_wild_ids.end());
    return combo;
}

std::vector<Combination> group_declarations(
    CombinationKind kind,
    const std::vector<int>& cards,
    Rank level,
    Rank target,
    int required,
    const Facts& facts) {
    if (static_cast<int>(cards.size()) != required || facts.wilds.size() > 2 || facts.has_joker) return {};
    for (const auto& [rank, _count] : facts.counts) {
        if (rank != target) return {};
    }
    std::vector<Rank> targets(facts.wilds.size(), target);
    std::vector<Combination> result;
    for (auto assignments : assignment_products(facts.wilds, targets, level)) {
        result.push_back(build_combo(kind, cards, level, target, std::move(assignments)));
    }
    return result;
}

std::vector<Combination> sequence_declarations(
    CombinationKind kind,
    const std::vector<int>& cards,
    Rank level,
    const std::vector<int>& window,
    int copies,
    const Facts& facts,
    std::optional<Suit> declared_suit = std::nullopt) {
    if (facts.wilds.size() > 2 || facts.has_joker) return {};
    if (declared_suit) {
        for (Suit suit : facts.suits) if (suit != *declared_suit) return {};
    }
    std::set<Rank> allowed;
    for (int rank : window) allowed.insert(static_cast<Rank>(rank));
    for (const auto& [rank, count] : facts.counts) {
        if (!allowed.contains(rank) || count > copies) return {};
    }
    std::vector<Rank> missing;
    for (int raw : window) {
        const Rank rank = static_cast<Rank>(raw);
        const auto found = facts.counts.find(rank);
        const int count = found == facts.counts.end() ? 0 : found->second;
        for (int n = count; n < copies; ++n) missing.push_back(rank);
    }
    if (missing.size() != facts.wilds.size()) return {};
    std::vector<Combination> result;
    for (auto assignments : assignment_products(facts.wilds, missing, level, declared_suit)) {
        result.push_back(build_combo(kind, cards, level, static_cast<Rank>(window.back()), std::move(assignments)));
    }
    return result;
}

std::vector<Combination> full_house_declarations(
    const std::vector<int>& cards,
    Rank level,
    const Facts& facts) {
    if (cards.size() != 5 || facts.wilds.size() > 2 || facts.has_joker || facts.counts.size() > 2) return {};
    std::vector<Combination> result;
    for (Rank triple : PRINTED_RANKS) {
        if (!facts.counts.contains(triple)) continue;
        for (Rank pair : PRINTED_RANKS) {
            if (pair == triple) continue;
            bool valid = true;
            for (const auto& [rank, count] : facts.counts) {
                if (rank != triple && rank != pair) { valid = false; break; }
                if (count > (rank == triple ? 3 : 2)) { valid = false; break; }
            }
            if (!valid) continue;
            std::vector<Rank> targets;
            const int triple_count = facts.counts.contains(triple) ? facts.counts.at(triple) : 0;
            const int pair_count = facts.counts.contains(pair) ? facts.counts.at(pair) : 0;
            for (int n = triple_count; n < 3; ++n) targets.push_back(triple);
            for (int n = pair_count; n < 2; ++n) targets.push_back(pair);
            if (targets.size() != facts.wilds.size()) continue;
            for (auto assignments : assignment_products(facts.wilds, targets, level)) {
                result.push_back(build_combo(CombinationKind::FULL_HOUSE, cards, level, triple, std::move(assignments)));
            }
        }
    }
    return result;
}

std::vector<Combination> rank_bomb_declarations(
    const std::vector<int>& cards,
    Rank level,
    const Facts& facts) {
    if (cards.size() < 4 || cards.size() > 10 || facts.wilds.size() > 2 || facts.has_joker || facts.counts.size() != 1) return {};
    const Rank target = facts.counts.begin()->first;
    for (int id : facts.natural) if (Card{id}.rank() != target) return {};
    std::vector<Rank> targets(facts.wilds.size(), target);
    std::vector<Combination> result;
    for (auto assignments : assignment_products(facts.wilds, targets, level)) {
        result.push_back(build_combo(CombinationKind::RANK_BOMB, cards, level, target, std::move(assignments)));
    }
    return result;
}

bool combo_less(const Combination& left, const Combination& right) {
    if (left.kind != right.kind) return static_cast<int>(left.kind) < static_cast<int>(right.kind);
    if (left.comparison_key != right.comparison_key) return lexicographic_vector_less(left.comparison_key, right.comparison_key);
    if (left.card_ids != right.card_ids) return lexicographic_vector_less(left.card_ids, right.card_ids);
    auto assignment_tuple = [](const WildAssignment& a) {
        return std::tuple{a.card_id, static_cast<int>(a.rank), static_cast<int>(a.suit), a.natural};
    };
    if (left.wild_assignments != right.wild_assignments) {
        return std::lexicographical_compare(
            left.wild_assignments.begin(), left.wild_assignments.end(),
            right.wild_assignments.begin(), right.wild_assignments.end(),
            [&](const auto& a, const auto& b) { return assignment_tuple(a) < assignment_tuple(b); });
    }
    return lexicographic_vector_less(left.natural_wild_ids, right.natural_wild_ids);
}

std::vector<int> represented_ranks(const Combination& combo) {
    std::vector<int> result;
    for (int id : combo.card_ids) {
        auto found = std::find_if(combo.wild_assignments.begin(), combo.wild_assignments.end(), [&](const auto& value) {
            return value.card_id == id;
        });
        result.push_back(static_cast<int>(found == combo.wild_assignments.end() ? Card{id}.rank() : found->rank));
    }
    return result;
}

Rank represented_rank(int card_id, const Combination& combo) {
    const auto found = std::find_if(combo.wild_assignments.begin(), combo.wild_assignments.end(), [&](const auto& value) {
        return value.card_id == card_id;
    });
    return found == combo.wild_assignments.end() ? Card{card_id}.rank() : found->rank;
}

std::vector<int> window_for(const Combination& combo) {
    int length = 0;
    switch (combo.kind) {
        case CombinationKind::STRAIGHT:
        case CombinationKind::STRAIGHT_FLUSH: length = 5; break;
        case CombinationKind::PAIR_SEQUENCE: length = 3; break;
        case CombinationKind::TRIPLE_SEQUENCE: length = 2; break;
        default: throw std::invalid_argument("declaration is not a window family");
    }
    const auto ranks = represented_ranks(combo);
    std::set<int> represented(ranks.begin(), ranks.end());
    std::vector<std::vector<int>> candidates;
    for (const auto& window : sequence_windows(length)) {
        if (window.back() == static_cast<int>(combo.comparison_rank)) candidates.push_back(window);
    }
    for (const auto& window : candidates) {
        std::set<int> values(window.begin(), window.end());
        if (std::includes(values.begin(), values.end(), represented.begin(), represented.end())) return window;
    }
    if (!candidates.empty()) return candidates.front();
    throw std::invalid_argument("cannot derive canonical sequence window");
}

bool same_combo(const Combination& left, const Combination& right) {
    return combination_key(left) == combination_key(right);
}

void append_unique(std::vector<Combination>& target, std::vector<Combination> values) {
    target.insert(target.end(), std::make_move_iterator(values.begin()), std::make_move_iterator(values.end()));
}

}  // namespace

const std::array<Rank, 13>& printed_ranks() { return PRINTED_RANKS; }
const std::array<Rank, 13>& window_ranks() { return WINDOW_RANKS; }
const std::array<Rank, 15>& rank_order() { return RANK_ORDER; }

int Card::deck_index() const { validate_card_id(id); return id / 54; }
int Card::deck_offset() const { validate_card_id(id); return id % 54; }
Rank Card::rank() const {
    const int offset = deck_offset();
    if (offset < 52) return PRINTED_RANKS[offset / 4];
    return offset == 52 ? Rank::SMALL_JOKER : Rank::BIG_JOKER;
}
Suit Card::suit() const {
    const int offset = deck_offset();
    return offset < 52 ? static_cast<Suit>(offset % 4) : Suit::JOKER;
}
bool Card::is_joker() const { return suit() == Suit::JOKER; }
bool Card::is_level_wild(Rank level) const { validate_level(level); return suit() == Suit::HEARTS && rank() == level; }

bool Combination::is_bomb() const {
    return kind == CombinationKind::RANK_BOMB || kind == CombinationKind::STRAIGHT_FLUSH || kind == CombinationKind::FOUR_KINGS;
}

int PreviousResult::seat_with_rank(int rank_value) const {
    for (int seat = 0; seat < 4; ++seat) if (ranking[seat] == rank_value) return seat;
    throw std::invalid_argument("finishing rank absent");
}

void PreviousResult::validate() const {
    std::set<int> ranks(ranking.begin(), ranking.end());
    if (ranks != std::set<int>{1, 2, 3, 4}) throw std::invalid_argument("ranking must contain 1..4");
    if (winner_team < 0 || winner_team > 1) throw std::invalid_argument("winner_team must be 0 or 1");
    if (outcome_class != "DOUBLE_DOWN" && outcome_class != "HEAD_THIRD" && outcome_class != "HEAD_LAST") {
        throw std::invalid_argument("invalid outcome_class");
    }
    const int first = seat_with_rank(1);
    if (TEAM_OF[first] != winner_team) throw std::invalid_argument("winner_team does not match rank-1 seat");
    const int teammate_rank = outcome_class == "DOUBLE_DOWN" ? 2 : outcome_class == "HEAD_THIRD" ? 3 : 4;
    if (TEAM_OF[seat_with_rank(teammate_rank)] != winner_team) throw std::invalid_argument("outcome_class does not match ranking");
}

int effective_rank_value(Rank rank, Rank level) {
    validate_level(level);
    std::vector<Rank> order;
    if (level == Rank::TWO) {
        for (Rank value : PRINTED_RANKS) if (value != Rank::TWO) order.push_back(value);
        order.push_back(Rank::TWO);
    } else {
        order.push_back(Rank::TWO);
        for (Rank value : PRINTED_RANKS) {
            if (value != Rank::TWO && value != level) order.push_back(value);
        }
        order.push_back(level);
    }
    order.push_back(Rank::SMALL_JOKER);
    order.push_back(Rank::BIG_JOKER);
    const auto found = std::find(order.begin(), order.end(), rank);
    if (found == order.end()) throw std::invalid_argument("invalid rank");
    return static_cast<int>(std::distance(order.begin(), found));
}

int rank_order_index(Rank rank) {
    const auto found = std::find(RANK_ORDER.begin(), RANK_ORDER.end(), rank);
    if (found == RANK_ORDER.end()) throw std::invalid_argument("invalid rank");
    return static_cast<int>(std::distance(RANK_ORDER.begin(), found));
}

int window_rank_value(Rank rank) {
    const auto found = std::find(WINDOW_RANKS.begin(), WINDOW_RANKS.end(), rank);
    if (found == WINDOW_RANKS.end()) throw std::invalid_argument("rank is not in sequence window order");
    return static_cast<int>(std::distance(WINDOW_RANKS.begin(), found));
}

std::vector<int> sort_card_ids(const std::vector<int>& ids) {
    std::vector<int> result = ids;
    for (int id : result) validate_card_id(id);
    std::sort(result.begin(), result.end(), [](int a, int b) { return card_sort_key(a) < card_sort_key(b); });
    return result;
}

std::string combination_kind_name(CombinationKind kind) {
    const int index = static_cast<int>(kind);
    if (index < 0 || index >= static_cast<int>(KIND_NAMES.size())) throw std::invalid_argument("invalid combination kind");
    return KIND_NAMES[index];
}

CombinationKind combination_kind_from_name(const std::string& name) {
    for (int index = 0; index < static_cast<int>(KIND_NAMES.size()); ++index) {
        if (name == KIND_NAMES[index]) return static_cast<CombinationKind>(index);
    }
    throw std::invalid_argument("unknown combination kind: " + name);
}

std::string phase_name(Phase phase) {
    switch (phase) {
        case Phase::PLAY: return "play";
        case Phase::TRIBUTE: return "tribute";
        case Phase::RETURN: return "return";
        case Phase::TERMINAL: return "terminal";
    }
    throw std::invalid_argument("invalid phase");
}

std::vector<Combination> recognize_combinations(const std::vector<int>& input_ids, Rank level) {
    try { validate_level(level); } catch (...) { return {}; }
    if (input_ids.empty() || input_ids.size() > 10) return {};
    std::set<int> unique;
    for (int id : input_ids) {
        if (id < 0 || id >= NUM_PHYSICAL_CARDS || !unique.insert(id).second) return {};
    }
    const std::vector<int> cards = sort_card_ids(input_ids);
    const int size = static_cast<int>(cards.size());
    std::vector<Combination> declarations;
    if (size == 1) {
        const Card card{cards.front()};
        if (card.is_level_wild(level)) {
            declarations.push_back(build_combo(CombinationKind::SINGLE, cards, level, card.rank(), {{card.id, card.rank(), card.suit(), true}}));
        } else {
            declarations.push_back(build_combo(CombinationKind::SINGLE, cards, level, card.rank()));
        }
        return declarations;
    }
    const Facts facts = card_facts(cards, level);
    std::vector<Rank> group_ranks;
    if (facts.counts.size() == 1) group_ranks.push_back(facts.counts.begin()->first);
    else if (facts.counts.empty()) group_ranks.assign(PRINTED_RANKS.begin(), PRINTED_RANKS.end());

    if (size == 2) {
        const Rank first = Card{cards[0]}.rank();
        const Rank second = Card{cards[1]}.rank();
        if ((first == Rank::SMALL_JOKER && second == Rank::SMALL_JOKER) || (first == Rank::BIG_JOKER && second == Rank::BIG_JOKER)) {
            declarations.push_back(build_combo(CombinationKind::PAIR, cards, level, first));
        } else {
            for (Rank rank : group_ranks) append_unique(declarations, group_declarations(CombinationKind::PAIR, cards, level, rank, 2, facts));
        }
    }
    if (size == 3) {
        for (Rank rank : group_ranks) append_unique(declarations, group_declarations(CombinationKind::TRIPLE, cards, level, rank, 3, facts));
    }
    if (size == 5) append_unique(declarations, full_house_declarations(cards, level, facts));
    if (size == 4) {
        std::map<Rank, int> counts;
        for (int id : cards) counts[Card{id}.rank()] += 1;
        if (counts[Rank::SMALL_JOKER] == 2 && counts[Rank::BIG_JOKER] == 2) {
            declarations.push_back(build_combo(CombinationKind::FOUR_KINGS, cards, level, Rank::BIG_JOKER));
        }
    }
    if (size >= 4 && size <= 10) append_unique(declarations, rank_bomb_declarations(cards, level, facts));

    bool counts_are_one = !facts.has_joker;
    for (const auto& [_rank, count] : facts.counts) counts_are_one = counts_are_one && count == 1;
    if (size == 5 && counts_are_one) {
        for (const auto& window : sequence_windows(5)) {
            append_unique(declarations, sequence_declarations(CombinationKind::STRAIGHT, cards, level, window, 1, facts));
            if (facts.suits.size() == 1) {
                append_unique(declarations, sequence_declarations(CombinationKind::STRAIGHT_FLUSH, cards, level, window, 1, facts, *facts.suits.begin()));
            }
        }
    }
    if (size == 6 && !facts.has_joker) {
        for (const auto& [kind, copies] : std::array<std::pair<CombinationKind, int>, 2>{{
                 {CombinationKind::PAIR_SEQUENCE, 2}, {CombinationKind::TRIPLE_SEQUENCE, 3}}}) {
            if (static_cast<int>(facts.counts.size()) > 6 / copies) continue;
            bool too_many = false;
            for (const auto& [_rank, count] : facts.counts) too_many = too_many || count > copies;
            if (too_many) continue;
            for (const auto& window : sequence_windows(6 / copies)) {
                append_unique(declarations, sequence_declarations(kind, cards, level, window, copies, facts));
            }
        }
    }

    std::map<std::string, Combination> unique_declarations;
    for (auto& declaration : declarations) unique_declarations[combination_key(declaration)] = std::move(declaration);
    declarations.clear();
    for (auto& [_key, declaration] : unique_declarations) declarations.push_back(std::move(declaration));
    std::sort(declarations.begin(), declarations.end(), combo_less);
    return declarations;
}

int compare_combinations(const Combination& left, const Combination& right) {
    const auto left_candidates = recognize_combinations(left.card_ids, left.level_rank);
    const auto right_candidates = recognize_combinations(right.card_ids, right.level_rank);
    if (std::none_of(left_candidates.begin(), left_candidates.end(), [&](const auto& value) { return same_combo(value, left); }) ||
        std::none_of(right_candidates.begin(), right_candidates.end(), [&](const auto& value) { return same_combo(value, right); })) {
        throw std::invalid_argument("combination declaration is not valid for its cards and level");
    }
    if (left.level_rank != right.level_rank) throw std::invalid_argument("combinations use different level contexts");
    if (left.is_bomb() || right.is_bomb()) {
        if (left.is_bomb() != right.is_bomb()) return left.is_bomb() ? 1 : -1;
        if (left.comparison_key == right.comparison_key) return 0;
        return lexicographic_vector_less(left.comparison_key, right.comparison_key) ? -1 : 1;
    }
    if (left.kind != right.kind || left.size() != right.size()) throw std::invalid_argument("ordinary combinations are incomparable");
    return left.comparison_key.back() == right.comparison_key.back() ? 0 : left.comparison_key.back() < right.comparison_key.back() ? -1 : 1;
}

bool can_beat(const Combination& candidate, const Combination& current) {
    try { return compare_combinations(candidate, current) > 0; }
    catch (const std::invalid_argument&) { return false; }
}

namespace {

bool recognized_beats(const Combination& candidate, const std::optional<Combination>& current) {
    if (!current) return true;
    const bool left_bomb = candidate.is_bomb();
    const bool right_bomb = current->is_bomb();
    if (left_bomb || right_bomb) {
        if (left_bomb != right_bomb) return left_bomb;
        return lexicographic_vector_less(current->comparison_key, candidate.comparison_key);
    }
    return candidate.kind == current->kind && candidate.size() == current->size() &&
           candidate.comparison_key.back() > current->comparison_key.back();
}

std::vector<int> active_seats(const RoundState& state) {
    std::vector<int> result;
    for (int seat = 0; seat < 4; ++seat) if (!contains(state.finished_ranks, seat)) result.push_back(seat);
    return result;
}

int next_active(const RoundState& state, int seat) {
    for (int offset = 1; offset <= 4; ++offset) {
        const int next = (seat + offset) % 4;
        if (!contains(state.finished_ranks, next)) return next;
    }
    throw StateInvariantError("no active seat remains");
}

std::vector<int> natural_pool(const std::vector<int>& hand, Rank rank, Rank level) {
    std::vector<int> result;
    for (int id : hand) {
        const Card card{id};
        if (card.rank() == rank && !card.is_level_wild(level) && !card.is_joker()) result.push_back(id);
    }
    return result;
}

std::vector<int> rank_pool(const std::vector<int>& hand, Rank rank) {
    std::vector<int> result;
    for (int id : hand) if (Card{id}.rank() == rank) result.push_back(id);
    return result;
}

bool action_less(const Action& left, const Action& right) {
    return action_key(left) < action_key(right);
}

void remove_card(std::vector<int>& hand, int card_id) {
    const auto found = std::find(hand.begin(), hand.end(), card_id);
    if (found == hand.end()) throw IllegalActionError("action selects a missing physical card");
    hand.erase(found);
}

void validate_held(const RoundState& state, const Action& action) {
    std::set<int> selected;
    for (int id : action.card_ids) {
        if (!selected.insert(id).second || !contains(state.hands[action.player_id], id)) {
            throw IllegalActionError("action selects a missing or duplicated physical card");
        }
    }
}

int tribute_card_for(const RoundState& state, int donor) {
    if (!contains(state.tribute_donors, donor)) return -1;
    std::vector<int> eligible;
    for (int id : state.hands[donor]) if (!Card{id}.is_level_wild(state.level_rank)) eligible.push_back(id);
    if (eligible.empty()) return -1;
    return *std::min_element(eligible.begin(), eligible.end(), [&](int left, int right) {
        const auto left_key = std::tuple{-effective_rank_value(Card{left}.rank(), state.level_rank), left};
        const auto right_key = std::tuple{-effective_rank_value(Card{right}.rank(), state.level_rank), right};
        return left_key < right_key;
    });
}

std::vector<int> return_cards_for(const RoundState& state, int recipient) {
    const auto pending = state.pending_return();
    if (!pending || pending->first != recipient) return {};
    const std::set<Rank> ranks{
        Rank::TWO, Rank::THREE, Rank::FOUR, Rank::FIVE, Rank::SIX,
        Rank::SEVEN, Rank::EIGHT, Rank::NINE, Rank::TEN,
    };
    std::vector<int> result;
    for (int id : state.hands[recipient]) {
        const Card card{id};
        if (!card.is_joker() && !card.is_level_wild(state.level_rank) && card.rank() != state.level_rank && ranks.contains(card.rank())) {
            result.push_back(id);
        }
    }
    std::sort(result.begin(), result.end(), [&](int left, int right) {
        return std::tuple{effective_rank_value(Card{left}.rank(), state.level_rank), left} <
               std::tuple{effective_rank_value(Card{right}.rank(), state.level_rank), right};
    });
    return result;
}

void terminalize(RoundState& state) {
    state.done = true;
    state.phase = Phase::TERMINAL;
    state.current_winning.reset();
    state.current_winner_seat = -1;
    state.consecutive_passes = 0;
    state.winner_team = TEAM_OF[state.finished_ranks.front()];
    if (TEAM_OF[state.finished_ranks[0]] == TEAM_OF[state.finished_ranks[1]]) state.outcome_class = "DOUBLE_DOWN";
    else if (TEAM_OF[state.finished_ranks[0]] == TEAM_OF[state.finished_ranks[2]]) state.outcome_class = "HEAD_THIRD";
    else state.outcome_class = "HEAD_LAST";
    for (int seat = 0; seat < 4; ++seat) state.rewards[seat] = TEAM_OF[seat] == state.winner_team ? 1.0f : -1.0f;
    for (int team = 0; team < 2; ++team) state.team_rewards[team] = team == state.winner_team ? 1.0f : -1.0f;
    state.finish_committed_step = state.committed_step;
    state.finish_token_step = state.committed_step;
}

void append_remaining_ranks(RoundState& state, int after) {
    for (int offset = 1; offset <= 4; ++offset) {
        const int seat = (after + offset) % 4;
        if (!contains(state.finished_ranks, seat)) state.finished_ranks.push_back(seat);
    }
}

void finish_seat(RoundState& state, int seat) {
    if (!contains(state.emptied_seats, seat)) state.emptied_seats.push_back(seat);
    if (!contains(state.finished_ranks, seat)) state.finished_ranks.push_back(seat);
    if (state.finished_ranks.size() >= 2 && TEAM_OF[state.finished_ranks[0]] == TEAM_OF[state.finished_ranks[1]]) {
        append_remaining_ranks(state, state.finished_ranks[1]);
        terminalize(state);
    } else if (state.finished_ranks.size() == 3) {
        append_remaining_ranks(state, state.finished_ranks.back());
        terminalize(state);
    }
}

void resolve_tribute(RoundState& state) {
    if (!state.previous_result || state.tribute_selections.size() != state.tribute_donors.size()) {
        throw StateInvariantError("tribute selections incomplete");
    }
    std::vector<std::pair<int, int>> selected;
    for (int donor : state.tribute_donors) selected.push_back({donor, state.tribute_selections.at(donor)});
    std::vector<int> recipients;
    if (state.previous_result->outcome_class == "DOUBLE_DOWN") {
        std::sort(selected.begin(), selected.end(), [&](const auto& left, const auto& right) {
            return std::tuple{-effective_rank_value(Card{left.second}.rank(), state.level_rank), left.first, left.second} <
                   std::tuple{-effective_rank_value(Card{right.second}.rank(), state.level_rank), right.first, right.second};
        });
        recipients = {state.previous_result->seat_with_rank(1), state.previous_result->seat_with_rank(2)};
    } else {
        recipients = {state.previous_result->seat_with_rank(1)};
    }
    state.tribute_order.clear();
    for (const auto& [donor, _card] : selected) state.tribute_order.push_back(donor);
    state.tribute_escrow.clear();
    state.tribute_transfers.clear();
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto [donor, card] = selected[index];
        const int recipient = recipients[std::min(index, recipients.size() - 1)];
        state.hands[recipient].push_back(card);
        state.tribute_transfers.push_back({donor, recipient, card});
    }
    state.return_obligations.clear();
    for (const auto& transfer : state.tribute_transfers) state.return_obligations.push_back({transfer.target, transfer.source});
    state.phase = Phase::RETURN;
    state.active_seat = state.return_obligations.front().first;
    state.leader_seat = state.active_seat;
}

void apply_tribute(RoundState& state, const Action& action) {
    const auto pending = state.pending_tribute_donors();
    if (action.player_id != state.active_seat || !contains(pending, action.player_id) || action.card_ids.size() != 1) {
        throw IllegalActionError("invalid tribute action");
    }
    validate_held(state, action);
    const int expected = tribute_card_for(state, action.player_id);
    if (expected < 0 || action.card_ids[0] != expected) throw IllegalActionError("tribute must use highest eligible card");
    const int card = action.card_ids[0];
    remove_card(state.hands[action.player_id], card);
    state.tribute_selections[action.player_id] = card;
    state.tribute_escrow.push_back(card);
    state.history.push_back({action.player_id, "tribute", {card}, std::nullopt});
    ++state.committed_step;
    const auto remaining = state.pending_tribute_donors();
    if (!remaining.empty()) state.active_seat = *std::min_element(remaining.begin(), remaining.end());
    else resolve_tribute(state);
}

void apply_return(RoundState& state, const Action& action) {
    const auto pending = state.pending_return();
    if (!pending || action.player_id != pending->first || action.card_ids.size() != 1) throw IllegalActionError("invalid return action");
    validate_held(state, action);
    const int card = action.card_ids[0];
    const auto legal = return_cards_for(state, action.player_id);
    if (!contains(legal, card)) throw IllegalActionError("return card is not legal");
    const auto [recipient, donor] = *pending;
    remove_card(state.hands[recipient], card);
    state.hands[donor].push_back(card);
    state.return_transfers.push_back({recipient, donor, card});
    state.history.push_back({recipient, "return", {card}, std::nullopt});
    ++state.committed_step;
    state.return_obligations.erase(state.return_obligations.begin());
    if (!state.return_obligations.empty()) state.active_seat = state.return_obligations.front().first;
    else {
        state.phase = Phase::PLAY;
        state.leader_seat = !state.tribute_order.empty() ? state.tribute_order.front() : state.previous_result->seat_with_rank(1);
        state.active_seat = state.leader_seat;
    }
}

void apply_pass(RoundState& state, const Action& action) {
    if (!state.current_winning || !action.card_ids.empty()) throw IllegalActionError("invalid pass");
    state.history.push_back({action.player_id, "pass", {}, std::nullopt});
    ++state.committed_step;
    ++state.consecutive_passes;
    const auto active = active_seats(state);
    const int need = static_cast<int>(active.size()) - (contains(active, state.current_winner_seat) ? 1 : 0);
    if (state.consecutive_passes >= std::max(1, need)) {
        const int winner = state.current_winner_seat;
        state.current_winning.reset();
        state.current_winner_seat = -1;
        state.consecutive_passes = 0;
        if (contains(state.emptied_seats, winner)) {
            const int partner = (winner + 2) % 4;
            state.leader_seat = !contains(state.finished_ranks, partner) ? partner : next_active(state, winner);
        } else state.leader_seat = winner;
        state.active_seat = state.leader_seat;
    } else state.active_seat = next_active(state, action.player_id);
}

void apply_play(RoundState& state, const Action& action) {
    if (action.kind == "pass" || action.kind == "tribute" || action.kind == "return" || action.card_ids.empty()) {
        throw IllegalActionError("invalid play");
    }
    validate_held(state, action);
    CombinationKind kind;
    try { kind = combination_kind_from_name(action.kind); }
    catch (const std::invalid_argument&) { throw IllegalActionError("invalid play family"); }
    std::optional<Combination> declaration = action.declaration;
    if (!declaration) {
        for (const auto& candidate : recognize_combinations(action.card_ids, state.level_rank)) {
            if (candidate.kind == kind) { declaration = candidate; break; }
        }
    }
    if (!declaration) throw IllegalActionError("cards do not form declared combination");
    const auto candidates = recognize_combinations(action.card_ids, state.level_rank);
    const bool valid = declaration->level_rank == state.level_rank && declaration->kind == kind &&
        std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) { return same_combo(candidate, *declaration); });
    if (!valid) throw IllegalActionError("forged or mismatched declaration");
    if (state.current_winning && !can_beat(*declaration, *state.current_winning)) throw IllegalActionError("play does not beat current winning play");
    for (int card : action.card_ids) remove_card(state.hands[action.player_id], card);
    state.history.push_back({action.player_id, combination_kind_name(declaration->kind), sort_card_ids(action.card_ids), *declaration});
    ++state.committed_step;
    state.current_winning = *declaration;
    state.current_winner_seat = action.player_id;
    state.consecutive_passes = 0;
    if (state.hands[action.player_id].empty()) finish_seat(state, action.player_id);
    if (!state.done) state.active_seat = next_active(state, action.player_id);
}

}  // namespace

std::vector<Combination> enumerate_plays(const RoundState& state, int seat) {
    if (seat < 0 || seat >= 4) throw std::out_of_range("seat out of range");
    const auto hand = sort_card_ids(state.hands[seat]);
    const auto current = state.current_winning;
    std::set<CombinationKind> allowed;
    const std::set<CombinationKind> bombs{CombinationKind::RANK_BOMB, CombinationKind::STRAIGHT_FLUSH, CombinationKind::FOUR_KINGS};
    if (!current) {
        for (int kind = 0; kind < 10; ++kind) allowed.insert(static_cast<CombinationKind>(kind));
    } else if (current->is_bomb()) allowed = bombs;
    else {
        allowed = bombs;
        allowed.insert(current->kind);
    }
    std::vector<int> wilds;
    for (int id : hand) if (Card{id}.is_level_wild(state.level_rank)) wilds.push_back(id);
    std::map<Rank, std::vector<int>> natural;
    for (Rank rank : PRINTED_RANKS) natural[rank] = natural_pool(hand, rank, state.level_rank);
    std::map<Rank, std::vector<int>> jokers;
    jokers[Rank::SMALL_JOKER] = rank_pool(hand, Rank::SMALL_JOKER);
    jokers[Rank::BIG_JOKER] = rank_pool(hand, Rank::BIG_JOKER);

    std::set<std::string> emitted_sets;
    std::map<std::string, Combination> out;
    auto emit = [&](std::vector<int> cards) {
        std::sort(cards.begin(), cards.end());
        const std::string ids_key = vector_key(cards);
        if (!emitted_sets.insert(ids_key).second) return;
        for (auto declaration : recognize_combinations(cards, state.level_rank)) {
            if (allowed.contains(declaration.kind) && recognized_beats(declaration, current)) out[combination_key(declaration)] = std::move(declaration);
        }
    };

    if (allowed.contains(CombinationKind::SINGLE)) for (int id : hand) emit({id});
    for (Rank rank : PRINTED_RANKS) {
        const auto& pool = natural[rank];
        for (const auto& [count, kind] : std::array<std::pair<int, CombinationKind>, 2>{{{2, CombinationKind::PAIR}, {3, CombinationKind::TRIPLE}}}) {
            if (!allowed.contains(kind)) continue;
            const int min_natural = std::max(0, count - static_cast<int>(wilds.size()));
            const int max_natural = std::min(count, static_cast<int>(pool.size()));
            for (int natural_count = min_natural; natural_count <= max_natural; ++natural_count) {
                for (const auto& left : choose_ids(pool, natural_count)) {
                    for (const auto& right : choose_ids(wilds, count - natural_count)) {
                        auto cards = left; cards.insert(cards.end(), right.begin(), right.end()); emit(std::move(cards));
                    }
                }
            }
        }
        if (allowed.contains(CombinationKind::RANK_BOMB)) {
            const int maximum = std::min(10, static_cast<int>(pool.size() + wilds.size()));
            for (int count = 4; count <= maximum; ++count) {
                const int min_natural = std::max(0, count - static_cast<int>(wilds.size()));
                const int max_natural = std::min(count, static_cast<int>(pool.size()));
                for (int natural_count = min_natural; natural_count <= max_natural; ++natural_count) {
                    for (const auto& left : choose_ids(pool, natural_count)) {
                        for (const auto& right : choose_ids(wilds, count - natural_count)) {
                            auto cards = left; cards.insert(cards.end(), right.begin(), right.end()); emit(std::move(cards));
                        }
                    }
                }
            }
        }
    }
    if (allowed.contains(CombinationKind::PAIR)) {
        for (Rank rank : {Rank::SMALL_JOKER, Rank::BIG_JOKER}) for (const auto& cards : choose_ids(jokers[rank], 2)) emit(cards);
    }
    if (allowed.contains(CombinationKind::FULL_HOUSE)) {
        for (Rank triple : PRINTED_RANKS) for (Rank pair : PRINTED_RANKS) {
            if (triple == pair) continue;
            for (int triple_count = 0; triple_count <= 3; ++triple_count) for (int pair_count = 0; pair_count <= 2; ++pair_count) {
                const int wild_count = 5 - triple_count - pair_count;
                if (wild_count < 0 || wild_count > std::min(2, static_cast<int>(wilds.size())) ||
                    triple_count > static_cast<int>(natural[triple].size()) || pair_count > static_cast<int>(natural[pair].size())) continue;
                for (const auto& a : choose_ids(natural[triple], triple_count)) {
                    for (const auto& b : choose_ids(natural[pair], pair_count)) {
                        for (const auto& w : choose_ids(wilds, wild_count)) {
                            auto cards = a; cards.insert(cards.end(), b.begin(), b.end()); cards.insert(cards.end(), w.begin(), w.end()); emit(std::move(cards));
                        }
                    }
                }
            }
        }
    }

    auto seq = [&](const std::vector<int>& window, int copies, const std::vector<int>& sequence_hand) {
        const int total = static_cast<int>(window.size()) * copies;
        std::vector<std::vector<int>> pools;
        for (int raw_rank : window) pools.push_back(rank_pool(sequence_hand, static_cast<Rank>(raw_rank)));
        std::set<int> wild_ids(wilds.begin(), wilds.end());
        std::vector<int> counts(window.size());
        std::function<void(std::size_t)> count_rec;
        count_rec = [&](std::size_t index) {
            if (index < counts.size()) {
                for (int count = 0; count <= copies; ++count) { counts[index] = count; count_rec(index + 1); }
                return;
            }
            const int sum = std::accumulate(counts.begin(), counts.end(), 0);
            const int wild_count = total - sum;
            if (wild_count < 0 || wild_count > std::min(2, static_cast<int>(wilds.size()))) return;
            for (std::size_t i = 0; i < pools.size(); ++i) if (counts[i] > static_cast<int>(pools[i].size())) return;
            std::vector<std::vector<std::vector<int>>> choices;
            for (std::size_t i = 0; i < pools.size(); ++i) choices.push_back(choose_ids(pools[i], counts[i]));
            std::vector<int> selected;
            std::function<void(std::size_t)> group_rec;
            group_rec = [&](std::size_t group) {
                if (group < choices.size()) {
                    for (const auto& choice : choices[group]) {
                        const auto old_size = selected.size();
                        selected.insert(selected.end(), choice.begin(), choice.end());
                        group_rec(group + 1);
                        selected.resize(old_size);
                    }
                    return;
                }
                std::set<int> selected_ids(selected.begin(), selected.end());
                const int used_wilds = static_cast<int>(std::count_if(selected.begin(), selected.end(), [&](int id) { return wild_ids.contains(id); }));
                std::vector<int> extra;
                for (int id : wilds) if (!selected_ids.contains(id)) extra.push_back(id);
                if (used_wilds + wild_count > 2) return;
                for (const auto& add : choose_ids(extra, wild_count)) {
                    auto cards = selected; cards.insert(cards.end(), add.begin(), add.end()); emit(std::move(cards));
                }
            };
            group_rec(0);
        };
        count_rec(0);
    };

    if (allowed.contains(CombinationKind::STRAIGHT)) {
        for (const auto& window : sequence_windows(5)) seq(window, 1, hand);
    } else if (allowed.contains(CombinationKind::STRAIGHT_FLUSH)) {
        std::set<int> wild_ids(wilds.begin(), wilds.end());
        for (Suit suit : {Suit::CLUBS, Suit::DIAMONDS, Suit::HEARTS, Suit::SPADES}) {
            std::vector<int> suited;
            for (int id : hand) if (Card{id}.suit() == suit || wild_ids.contains(id)) suited.push_back(id);
            if (suited.size() >= 5) for (const auto& window : sequence_windows(5)) seq(window, 1, suited);
        }
    }
    if (allowed.contains(CombinationKind::PAIR_SEQUENCE)) for (const auto& window : sequence_windows(3)) seq(window, 2, hand);
    if (allowed.contains(CombinationKind::TRIPLE_SEQUENCE)) for (const auto& window : sequence_windows(2)) seq(window, 3, hand);
    if (allowed.contains(CombinationKind::FOUR_KINGS)) {
        for (const auto& small : choose_ids(jokers[Rank::SMALL_JOKER], 2)) for (const auto& big : choose_ids(jokers[Rank::BIG_JOKER], 2)) {
            auto cards = small; cards.insert(cards.end(), big.begin(), big.end()); emit(std::move(cards));
        }
    }

    std::vector<Combination> result;
    for (auto& [_key, declaration] : out) result.push_back(std::move(declaration));
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        const auto lk = std::tuple{combination_kind_name(left.kind), left.comparison_key, left.card_ids, combination_key(left)};
        const auto rk = std::tuple{combination_kind_name(right.kind), right.comparison_key, right.card_ids, combination_key(right)};
        return lk < rk;
    });
    return result;
}

std::vector<Action> enumerate_legal_actions(const RoundState& state) {
    if (state.done || state.phase == Phase::TERMINAL) return {};
    std::vector<Action> actions;
    if (state.phase == Phase::TRIBUTE) {
        const int card = tribute_card_for(state, state.active_seat);
        if (card >= 0) actions.push_back({state.active_seat, "tribute", {card}, std::nullopt});
    } else if (state.phase == Phase::RETURN) {
        for (int card : return_cards_for(state, state.active_seat)) actions.push_back({state.active_seat, "return", {card}, std::nullopt});
    } else if (state.phase == Phase::PLAY) {
        if (state.current_winning) actions.push_back({state.active_seat, "pass", {}, std::nullopt});
        for (auto declaration : enumerate_plays(state, state.active_seat)) {
            actions.push_back({state.active_seat, combination_kind_name(declaration.kind), declaration.card_ids, declaration});
        }
    } else throw StateInvariantError("unsupported phase");
    std::map<std::string, Action> unique;
    for (auto& action : actions) unique[action_key(action)] = std::move(action);
    actions.clear();
    for (auto& [_key, action] : unique) actions.push_back(std::move(action));
    std::sort(actions.begin(), actions.end(), action_less);
    return actions;
}

void apply_action(RoundState& state, const Action& action) {
    if (state.done || state.phase == Phase::TERMINAL) throw EpisodeTerminatedError("round is already terminal");
    if (action.player_id != state.active_seat) throw IllegalActionError("action is from non-active seat");
    if (contains(state.finished_ranks, action.player_id)) throw IllegalActionError("finished seat cannot act");
    RoundState trial = state;
    if (trial.phase == Phase::TRIBUTE) apply_tribute(trial, action);
    else if (trial.phase == Phase::RETURN) apply_return(trial, action);
    else if (trial.phase == Phase::PLAY) action.is_pass() ? apply_pass(trial, action) : apply_play(trial, action);
    else throw StateInvariantError("unsupported phase");
    trial.check_invariants();
    state = std::move(trial);
}

std::array<int, 4> RoundState::ranking() const {
    std::array<int, 4> result{};
    for (std::size_t index = 0; index < finished_ranks.size(); ++index) result[finished_ranks[index]] = static_cast<int>(index) + 1;
    return result;
}

std::vector<int> RoundState::pending_tribute_donors() const {
    std::vector<int> result;
    for (int donor : tribute_donors) if (!tribute_selections.contains(donor)) result.push_back(donor);
    return result;
}

std::optional<std::pair<int, int>> RoundState::pending_return() const {
    if (return_obligations.empty()) return std::nullopt;
    return return_obligations.front();
}

void RoundState::initialize_exchange() {
    if (!previous_result) return;
    previous_result->validate();
    tribute_donors.clear();
    if (previous_result->outcome_class == "DOUBLE_DOWN") {
        tribute_donors = {previous_result->seat_with_rank(3), previous_result->seat_with_rank(4)};
    } else tribute_donors = {previous_result->seat_with_rank(4)};
    std::sort(tribute_donors.begin(), tribute_donors.end());
    std::set<int> donor_big_jokers;
    for (int donor : tribute_donors) for (int id : hands[donor]) if (Card{id}.rank() == Rank::BIG_JOKER) donor_big_jokers.insert(id);
    anti_tribute = donor_big_jokers.contains(53) && donor_big_jokers.contains(107);
    if (anti_tribute) {
        phase = Phase::PLAY;
        leader_seat = previous_result->seat_with_rank(1);
        active_seat = leader_seat;
    } else {
        phase = Phase::TRIBUTE;
        active_seat = *std::min_element(tribute_donors.begin(), tribute_donors.end());
        leader_seat = active_seat;
    }
}

void RoundState::check_invariants() const {
    if (active_seat < 0 || active_seat >= 4 || leader_seat < 0 || leader_seat >= 4 || committed_step < 0) {
        throw StateInvariantError("invalid seat or committed-step metadata");
    }
    std::vector<int> ids;
    for (const auto& hand : hands) ids.insert(ids.end(), hand.begin(), hand.end());
    for (const auto& action : history) {
        if (action.kind != "pass" && action.kind != "tribute" && action.kind != "return") {
            ids.insert(ids.end(), action.card_ids.begin(), action.card_ids.end());
        }
    }
    ids.insert(ids.end(), tribute_escrow.begin(), tribute_escrow.end());
    std::set<int> unique(ids.begin(), ids.end());
    if (ids.size() != NUM_PHYSICAL_CARDS || unique.size() != NUM_PHYSICAL_CARDS || *unique.begin() != 0 || *unique.rbegin() != 107) {
        throw StateInvariantError("physical cards are not a lossless partition");
    }
    std::set<int> finished(finished_ranks.begin(), finished_ranks.end());
    if (finished.size() != finished_ranks.size() || std::any_of(finished.begin(), finished.end(), [](int seat) { return seat < 0 || seat >= 4; })) {
        throw StateInvariantError("invalid finished ranks");
    }
    std::set<int> emptied(emptied_seats.begin(), emptied_seats.end());
    if (emptied.size() != emptied_seats.size()) throw StateInvariantError("invalid emptied seats");
    for (int seat : emptied_seats) {
        if (!finished.contains(seat) || !hands[seat].empty()) throw StateInvariantError("invalid emptied seats");
    }
    if (done) {
        if (phase != Phase::TERMINAL || finished_ranks.size() != 4 ||
            (outcome_class != "DOUBLE_DOWN" && outcome_class != "HEAD_THIRD" && outcome_class != "HEAD_LAST") ||
            winner_team < 0 || winner_team > 1 || finish_committed_step < 0 || finish_token_step < 0) {
            throw StateInvariantError("invalid terminal state");
        }
        for (int seat = 0; seat < 4; ++seat) {
            const float expected = TEAM_OF[seat] == winner_team ? 1.0f : -1.0f;
            if (rewards[seat] != expected) throw StateInvariantError("terminal reward disagrees with winner team");
        }
    } else {
        if (phase == Phase::TERMINAL) throw StateInvariantError("invalid nonterminal phase");
        for (float reward : rewards) if (reward != 0.0f) throw StateInvariantError("nonterminal reward is nonzero");
        for (float reward : team_rewards) if (reward != 0.0f) throw StateInvariantError("nonterminal team reward is nonzero");
    }
    if (!previous_result && (phase == Phase::TRIBUTE || phase == Phase::RETURN || !tribute_donors.empty())) {
        throw StateInvariantError("tribute without previous result");
    }
    if (previous_result) {
        std::vector<int> expected;
        if (previous_result->outcome_class == "DOUBLE_DOWN") expected = {previous_result->seat_with_rank(3), previous_result->seat_with_rank(4)};
        else expected = {previous_result->seat_with_rank(4)};
        std::sort(expected.begin(), expected.end());
        if (tribute_donors != expected) throw StateInvariantError("tribute donors are not deterministic");
        if (phase == Phase::TRIBUTE) {
            const auto pending = pending_tribute_donors();
            if (anti_tribute || pending.empty() || !contains(pending, active_seat)) throw StateInvariantError("invalid tribute phase");
        }
        if (phase == Phase::RETURN) {
            if (anti_tribute || return_obligations.empty() || active_seat != return_obligations.front().first) throw StateInvariantError("invalid return phase");
        }
    }
    std::set<int> selected;
    for (const auto& [donor, card] : tribute_selections) {
        if (!selected.insert(card).second || !contains(tribute_donors, donor)) throw StateInvariantError("invalid tribute selections");
    }
    const std::set<int> escrow(tribute_escrow.begin(), tribute_escrow.end());
    if (phase == Phase::TRIBUTE && selected != escrow) throw StateInvariantError("tribute escrow mismatch");
    if (phase != Phase::TRIBUTE && !escrow.empty()) throw StateInvariantError("resolved tribute has escrow");
    if (!done && phase == Phase::PLAY && (contains(finished_ranks, active_seat) || hands[active_seat].empty())) {
        throw StateInvariantError("nonterminal acting seat must be live");
    }
}

namespace {

std::vector<int> declaration_ranks(const Combination& combo) {
    if (combo.kind == CombinationKind::PAIR || combo.kind == CombinationKind::TRIPLE || combo.kind == CombinationKind::RANK_BOMB) {
        return {token_rank(combo.comparison_rank)};
    }
    if (combo.kind == CombinationKind::FULL_HOUSE) {
        std::map<Rank, int> counts;
        for (int card : combo.card_ids) counts[represented_rank(card, combo)] += 1;
        Rank pair_rank = combo.comparison_rank;
        for (const auto& [rank, count] : counts) if (count == 2) { pair_rank = rank; break; }
        return {token_rank(combo.comparison_rank), token_rank(pair_rank)};
    }
    if (combo.kind == CombinationKind::STRAIGHT || combo.kind == CombinationKind::PAIR_SEQUENCE ||
        combo.kind == CombinationKind::TRIPLE_SEQUENCE || combo.kind == CombinationKind::STRAIGHT_FLUSH) {
        return {token_rank(static_cast<Rank>(window_for(combo).front()))};
    }
    return {};
}

std::optional<int> declaration_length(const Combination& combo) {
    if (combo.kind == CombinationKind::RANK_BOMB) return combo.size();
    if (combo.kind == CombinationKind::PAIR_SEQUENCE || combo.kind == CombinationKind::TRIPLE_SEQUENCE) return static_cast<int>(window_for(combo).size());
    return std::nullopt;
}

std::optional<Suit> declaration_suit(const Combination& combo) {
    if (combo.kind != CombinationKind::STRAIGHT_FLUSH) return std::nullopt;
    for (int id : combo.card_ids) {
        const Card card{id};
        if (!card.is_joker() && !card.is_level_wild(combo.level_rank)) return card.suit();
    }
    if (!combo.wild_assignments.empty()) return combo.wild_assignments.front().suit;
    throw ProtocolError("straight flush has no represented suit");
}

std::vector<int> ordered_action_cards(const Action& action) {
    if (!action.declaration) {
        auto result = action.card_ids;
        std::sort(result.begin(), result.end());
        return result;
    }
    const Combination& combo = *action.declaration;
    if (combo.kind == CombinationKind::FULL_HOUSE) {
        std::map<Rank, int> counts;
        for (int card : combo.card_ids) counts[represented_rank(card, combo)] += 1;
        Rank pair_rank = combo.comparison_rank;
        for (const auto& [rank, count] : counts) if (rank != combo.comparison_rank && count == 2) { pair_rank = rank; break; }
        std::vector<int> ordered;
        for (Rank target : {combo.comparison_rank, pair_rank}) {
            std::vector<int> group;
            for (int card : action.card_ids) if (represented_rank(card, combo) == target) group.push_back(card);
            std::sort(group.begin(), group.end());
            ordered.insert(ordered.end(), group.begin(), group.end());
        }
        if (ordered.size() == action.card_ids.size()) return ordered;
    }
    if (combo.kind == CombinationKind::STRAIGHT || combo.kind == CombinationKind::STRAIGHT_FLUSH ||
        combo.kind == CombinationKind::PAIR_SEQUENCE || combo.kind == CombinationKind::TRIPLE_SEQUENCE) {
        const auto window = window_for(combo);
        std::map<Rank, int> order;
        for (std::size_t index = 0; index < window.size(); ++index) order[static_cast<Rank>(window[index])] = static_cast<int>(index);
        auto result = action.card_ids;
        std::sort(result.begin(), result.end(), [&](int left, int right) {
            return std::tuple{order.contains(represented_rank(left, combo)) ? order.at(represented_rank(left, combo)) : 999, left} <
                   std::tuple{order.contains(represented_rank(right, combo)) ? order.at(represented_rank(right, combo)) : 999, right};
        });
        return result;
    }
    auto result = action.card_ids;
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace

int token_rank(Rank rank) { return RANK_TOKEN_BASE + rank_order_index(rank); }
Rank rank_from_token(int token) {
    const int index = token - RANK_TOKEN_BASE;
    if (index < 0 || index >= static_cast<int>(RANK_ORDER.size())) throw std::invalid_argument("token is not a rank token");
    return RANK_ORDER[index];
}
int token_suit(Suit suit) { return SUIT_TOKEN_BASE + static_cast<int>(suit); }
int token_length(int length) {
    if (length < 0 || length > 10) throw std::invalid_argument("length must be in 0..10");
    return LENGTH_TOKEN_BASE + length;
}
int token_card(int card_id) { validate_card_id(card_id); return CARD_TOKEN_BASE + card_id; }
int token_wild(Rank rank, Suit suit) {
    if (!is_normal_rank(rank) || suit == Suit::JOKER) throw std::invalid_argument("wild assignment must target non-joker rank/suit");
    const auto found = std::find(PRINTED_RANKS.begin(), PRINTED_RANKS.end(), rank);
    return WILD_TOKEN_BASE + static_cast<int>(std::distance(PRINTED_RANKS.begin(), found)) * 4 + static_cast<int>(suit);
}

std::vector<int> canonical_tokens(const Action& action) {
    if (action.kind == "pass") return {PASS_TOKEN};
    if (action.kind == "tribute" || action.kind == "return") {
        if (action.card_ids.size() != 1) throw ProtocolError("tribute/return action must contain one card");
        return {token_card(action.card_ids.front()), COMMIT_TOKEN};
    }
    if (!action.declaration) throw ProtocolError("play action requires a declaration");
    const Combination& combo = *action.declaration;
    std::vector<int> tokens{FAMILY_TOKEN_BASE + static_cast<int>(combo.kind)};
    const auto ranks = declaration_ranks(combo);
    const auto suit = declaration_suit(combo);
    if (combo.kind == CombinationKind::STRAIGHT_FLUSH) {
        tokens.push_back(token_suit(*suit));
        tokens.insert(tokens.end(), ranks.begin(), ranks.end());
    } else tokens.insert(tokens.end(), ranks.begin(), ranks.end());
    if (const auto length = declaration_length(combo)) tokens.push_back(token_length(*length));
    for (int card : ordered_action_cards(action)) tokens.push_back(token_card(card));
    auto assignments = combo.wild_assignments;
    std::sort(assignments.begin(), assignments.end(), [](const auto& a, const auto& b) { return a.card_id < b.card_id; });
    for (const auto& assignment : assignments) tokens.push_back(token_wild(assignment.rank, assignment.suit));
    tokens.push_back(COMMIT_TOKEN);
    if (tokens.size() > MAX_PREFIX_TOKENS) throw ProtocolError("canonical action exceeds max_action_tokens");
    return tokens;
}

std::string token_kind(int token_id) {
    if (token_id == PASS_TOKEN) return "PASS";
    if (token_id == COMMIT_TOKEN) return "COMMIT";
    if (token_id >= FAMILY_TOKEN_BASE && token_id < FAMILY_TOKEN_BASE + 10) return "FAMILY";
    if (token_id >= RANK_TOKEN_BASE && token_id < RANK_TOKEN_BASE + 15) return "RANK";
    if (token_id >= SUIT_TOKEN_BASE && token_id <= SUIT_TOKEN_BASE + 4) return "SUIT";
    if (token_id >= LENGTH_TOKEN_BASE && token_id <= LENGTH_TOKEN_BASE + 10) return "LENGTH";
    if (token_id >= CARD_TOKEN_BASE && token_id < CARD_TOKEN_BASE + 108) return "CARD";
    if (token_id >= WILD_TOKEN_BASE && token_id < WILD_TOKEN_BASE + 52) return "WILD_ASSIGNMENT";
    throw ProtocolError("V1 exposed unsupported legal token " + std::to_string(token_id));
}

int phase_token(const Game& game) {
    if (game.state.phase == Phase::TRIBUTE) return TRIBUTE_TOKEN;
    if (game.state.phase == Phase::RETURN) return RETURN_TOKEN;
    if (game.state.phase == Phase::PLAY) return game.state.current_winning ? FOLLOW_PLAY_TOKEN : LEAD_PLAY_TOKEN;
    throw EpisodeTerminatedError("terminal states have no V2 action phase");
}

core::ActionRow action_row(const Game& game, int token_id) {
    const int stage = static_cast<int>(game.prefix_tokens.size());
    if (stage < 0 || stage > MAX_PREFIX_TOKENS) throw CapacityError("prefix exceeds structured-row capacity");
    const std::string kind = token_kind(token_id);
    const auto found = std::find(TOKEN_KIND_NAMES.begin(), TOKEN_KIND_NAMES.end(), kind);
    if (found == TOKEN_KIND_NAMES.end()) throw ProtocolError("unsupported token kind");
    const int kind_code = static_cast<int>(std::distance(TOKEN_KIND_NAMES.begin(), found));
    const bool commit = token_id == PASS_TOKEN || token_id == COMMIT_TOKEN;
    core::ActionRow row;
    row.tokens = {ACTION_ROW_BEGIN, phase_token(game), ACTION_STAGE_BASE + stage,
                  ACTION_KIND_BASE + kind_code, ACTION_PAYLOAD, token_id,
                  commit ? ACTION_COMMIT : ACTION_NON_COMMIT, ACTION_ROW_END};
    row.token_id = token_id;
    row.phase_token = row.tokens[1];
    row.prefix_stage = stage;
    row.kind = kind;
    row.is_commit = commit;
    return row;
}

Game::Game(GameConfig cfg) : config(std::move(cfg)), rng(config.seed) {
    validate_level(config.level_rank);
    if (config.leader_seat < 0 || config.leader_seat >= 4) throw std::invalid_argument("leader_seat must be in [0,4)");
    if (config.max_token_steps <= 0 || config.max_token_steps > MAX_GAME_HORIZON) throw std::invalid_argument("invalid max_token_steps");
    if (config.previous_result) config.previous_result->validate();
}

void Game::reset(std::uint64_t seed) {
    rng.seed(seed);
    config.seed = seed;
    std::array<std::vector<int>, 4> hands;
    std::vector<int> deck(NUM_PHYSICAL_CARDS);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    for (int seat = 0; seat < 4; ++seat) hands[seat] = std::vector<int>(deck.begin() + seat * 27, deck.begin() + (seat + 1) * 27);
    reset_from_hands(hands);
    ++reset_count;
}

void Game::reset_auto() {
    std::array<std::vector<int>, 4> hands;
    std::vector<int> deck(NUM_PHYSICAL_CARDS);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    for (int seat = 0; seat < 4; ++seat) hands[seat] = std::vector<int>(deck.begin() + seat * 27, deck.begin() + (seat + 1) * 27);
    reset_from_hands(hands);
    ++reset_count;
}

void Game::reset_from_hands(const std::array<std::vector<int>, 4>& hands) {
    std::vector<int> all;
    for (const auto& hand : hands) all.insert(all.end(), hand.begin(), hand.end());
    std::set<int> unique;
    for (int id : all) { validate_card_id(id); unique.insert(id); }
    if (all.size() != NUM_PHYSICAL_CARDS || unique.size() != NUM_PHYSICAL_CARDS) throw std::invalid_argument("hands must partition all 108 physical cards");
    state = RoundState{};
    state.hands = hands;
    for (auto& hand : state.hands) hand = sort_card_ids(hand);
    state.level_rank = config.level_rank;
    state.active_seat = config.leader_seat;
    state.leader_seat = config.leader_seat;
    state.phase = Phase::PLAY;
    state.previous_result = config.previous_result;
    if (state.previous_result) state.initialize_exchange();
    prefix_tokens.clear();
    candidate_cache.clear();
    candidate_cache_valid = false;
    token_step = 0;
    truncated = false;
    state.check_invariants();
}

std::vector<std::pair<std::vector<int>, Action>> Game::candidate_sequences() const {
    if (state.done || state.phase == Phase::TERMINAL) return {};
    if (candidate_cache_valid) return candidate_cache;
    std::map<std::vector<int>, Action> result;
    for (auto action : enumerate_legal_actions(state)) {
        auto sequence = canonical_tokens(action);
        if (sequence.size() > MAX_PREFIX_TOKENS) throw ProtocolError("canonical action exceeds configured capacity");
        const auto found = result.find(sequence);
        if (found != result.end() && action_key(found->second) != action_key(action)) {
            throw ProtocolError("two distinct actions share one canonical token sequence");
        }
        result[std::move(sequence)] = std::move(action);
    }
    std::vector<std::pair<std::vector<int>, Action>> values;
    for (auto& [sequence, action] : result) values.push_back({sequence, action});
    candidate_cache = values;
    candidate_cache_valid = true;
    return values;
}

std::vector<int> Game::legal_next_tokens() const {
    if (state.done || truncated) return {};
    std::set<int> next;
    for (const auto& [sequence, _action] : candidate_sequences()) {
        if (sequence.size() > prefix_tokens.size() && std::equal(prefix_tokens.begin(), prefix_tokens.end(), sequence.begin())) {
            next.insert(sequence[prefix_tokens.size()]);
        }
    }
    if (next.size() > MAX_AVAILABLE_ACTIONS) throw CapacityError("legal action capacity overflow");
    return {next.begin(), next.end()};
}

std::vector<core::ActionRow> Game::candidate_rows() const {
    std::vector<core::ActionRow> rows;
    for (int token : legal_next_tokens()) rows.push_back(action_row(*this, token));
    std::set<std::array<int, 8>> unique;
    for (const auto& row : rows) if (!unique.insert(row.tokens).second) throw ProtocolError("V2 action rows are not unique");
    return rows;
}

ProtocolResult Game::step_token(int token_id) {
    if (state.done || truncated) throw EpisodeTerminatedError("stepwise protocol is terminal or truncated");
    const auto legal = legal_next_tokens();
    if (!contains(legal, token_id)) throw IllegalActionError("token is not in legal-next-token set");
    Game trial = *this;
    std::vector<std::pair<std::vector<int>, Action>> matching;
    for (const auto& candidate : trial.candidate_sequences()) {
        const auto& sequence = candidate.first;
        if (sequence.size() > trial.prefix_tokens.size() &&
            std::equal(trial.prefix_tokens.begin(), trial.prefix_tokens.end(), sequence.begin()) &&
            sequence[trial.prefix_tokens.size()] == token_id) matching.push_back(candidate);
    }
    if (matching.empty()) throw IllegalActionError("token does not extend a live canonical prefix");
    ++trial.token_step;
    ProtocolResult result;
    result.is_commit = token_id == PASS_TOKEN || token_id == COMMIT_TOKEN;
    if (token_id == PASS_TOKEN) {
        const Action action = matching.front().second;
        apply_action(trial.state, action);
        trial.prefix_tokens.clear();
        trial.candidate_cache.clear();
        trial.candidate_cache_valid = false;
        result.committed_action = action;
    } else if (token_id == COMMIT_TOKEN) {
        std::vector<Action> complete;
        for (const auto& [sequence, action] : matching) {
            if (sequence.size() == trial.prefix_tokens.size() + 1 && sequence.back() == COMMIT_TOKEN) complete.push_back(action);
        }
        if (complete.size() != 1) throw ProtocolError("COMMIT does not identify exactly one canonical action");
        apply_action(trial.state, complete.front());
        trial.prefix_tokens.clear();
        trial.candidate_cache.clear();
        trial.candidate_cache_valid = false;
        result.committed_action = complete.front();
    } else {
        trial.prefix_tokens.push_back(token_id);
        if (trial.legal_next_tokens().empty()) throw ProtocolError("non-commit token created a dead prefix");
    }
    if (trial.state.done) trial.state.finish_token_step = trial.token_step;
    if (trial.token_step >= trial.config.max_token_steps && !trial.state.done) {
        trial.truncated = true;
        result.termination_reason = "max_token_steps";
    }
    trial.check_invariants();
    result.done = trial.state.done;
    result.truncated = trial.truncated;
    result.rewards = trial.state.done ? trial.state.rewards : std::array<float, 4>{};
    result.token_step = trial.token_step;
    result.committed_step = trial.state.committed_step;
    *this = std::move(trial);
    return result;
}

void Game::check_invariants() const {
    state.check_invariants();
    if (token_step < 0 || token_step > config.max_token_steps) throw StateInvariantError("token step outside configured range");
    if (state.committed_step > token_step || state.committed_step != static_cast<int>(state.history.size())) throw StateInvariantError("committed/token step mismatch");
    if (truncated != (token_step == config.max_token_steps && !state.done)) throw StateInvariantError("truncation flag mismatch");
    if (state.done && !prefix_tokens.empty()) throw StateInvariantError("terminal state has private prefix");
    if (!state.done && !truncated && legal_next_tokens().empty()) throw StateInvariantError("nonterminal state has no legal action");
}

namespace {

void set_one(std::array<std::uint8_t, NUM_CHANNELS>& channels, int index, int value = 1) {
    if (index < 0 || index >= NUM_CHANNELS) throw CapacityError("channel index overflow");
    channels[index] = static_cast<std::uint8_t>(value);
}

void set_bits(std::array<std::uint8_t, NUM_CHANNELS>& channels, int start, int value, int width) {
    if (value < 0 || value >= (1 << width)) throw CapacityError("counter exceeds channel bit capacity");
    for (int bit = 0; bit < width; ++bit) channels[start + bit] = static_cast<std::uint8_t>((value >> bit) & 1);
}

int phase_index(Phase phase) { return static_cast<int>(phase); }

int outcome_index(const std::string& outcome) {
    if (outcome == "DOUBLE_DOWN") return 0;
    if (outcome == "HEAD_THIRD") return 1;
    if (outcome == "HEAD_LAST") return 2;
    throw std::invalid_argument("invalid outcome");
}

std::array<std::uint8_t, NUM_CHANNELS> encode_channels(const Game& game, int observer) {
    const RoundState& state = game.state;
    std::array<std::uint8_t, NUM_CHANNELS> channels{};
    std::set<int> visible(state.hands[observer].begin(), state.hands[observer].end());
    if (state.phase == Phase::RETURN) {
        for (const auto& transfer : state.return_transfers) if (transfer.target == observer) visible.erase(transfer.card_id);
    }
    for (int id : visible) set_one(channels, id);
    if (state.current_winning) for (int id : state.current_winning->card_ids) set_one(channels, 108 + id);
    if (state.phase == Phase::TRIBUTE || state.phase == Phase::RETURN) {
        if (state.tribute_selections.contains(observer)) set_one(channels, 216 + state.tribute_selections.at(observer));
        for (const auto& transfer : state.return_transfers) if (transfer.source == observer) set_one(channels, 216 + transfer.card_id);
    }
    const std::vector<int> prefix = observer == state.active_seat ? game.prefix_tokens : std::vector<int>{};
    if (prefix.size() > MAX_PREFIX_TOKENS) throw CapacityError("private prefix exceeds channel capacity");
    for (std::size_t index = 0; index < prefix.size(); ++index) set_one(channels, 324 + static_cast<int>(index), prefix[index]);
    for (int seat = 0; seat < 4; ++seat) set_one(channels, 356 + seat, static_cast<int>(state.hands[seat].size()));
    const auto ranking = state.ranking();
    for (int seat = 0; seat < 4; ++seat) set_one(channels, 360 + seat, ranking[seat]);
    set_one(channels, 364 + observer);
    set_one(channels, 368 + state.active_seat);
    set_one(channels, 372 + state.leader_seat);
    if (state.current_winner_seat >= 0) set_one(channels, 376 + state.current_winner_seat);
    for (int seat : state.emptied_seats) set_one(channels, 380 + seat);
    set_one(channels, 384 + phase_index(state.phase));
    const auto level_found = std::find(PRINTED_RANKS.begin(), PRINTED_RANKS.end(), state.level_rank);
    set_one(channels, 388 + static_cast<int>(std::distance(PRINTED_RANKS.begin(), level_found)));
    set_one(channels, 401 + std::min(state.consecutive_passes, 3));
    set_one(channels, 405 + TEAM_OF[observer]);
    if (state.previous_result) {
        set_one(channels, 407 + state.previous_result->winner_team);
        set_one(channels, 409 + outcome_index(state.previous_result->outcome_class));
        for (int seat = 0; seat < 4; ++seat) set_one(channels, 412 + seat, state.previous_result->ranking[seat]);
        set_one(channels, 486, state.previous_result->tribute_cancelled ? 1 : 0);
    }
    if (state.current_winning) {
        set_one(channels, 416 + static_cast<int>(state.current_winning->kind));
        set_one(channels, 426 + rank_order_index(state.current_winning->comparison_rank));
        const int size = state.current_winning->size();
        if (size < 0 || size > 10) throw CapacityError("current combination exceeds size channel capacity");
        set_one(channels, 441 + size);
    }
    for (int seat : state.tribute_donors) set_one(channels, 452 + seat);
    if (state.phase == Phase::PLAY || state.phase == Phase::TERMINAL) {
        for (std::size_t index = 0; index < std::min<std::size_t>(4, state.tribute_order.size()); ++index) set_one(channels, 456 + static_cast<int>(index), state.tribute_order[index] + 1);
    }
    if (observer == state.active_seat) set_bits(channels, 460, game.token_step, 13);
    set_bits(channels, 473, state.committed_step, 13);
    set_one(channels, 487, state.phase == Phase::PLAY || state.phase == Phase::TERMINAL ? 1 : 0);
    set_one(channels, 488, state.anti_tribute ? 1 : 0);
    set_one(channels, 489, state.current_winning ? 1 : 0);
    set_one(channels, 490, static_cast<int>(prefix.size()));
    set_one(channels, 491, observer == state.active_seat ? 1 : 0);
    set_one(channels, 492, contains(state.finished_ranks, observer) ? 1 : 0);
    set_one(channels, 493, contains(state.emptied_seats, observer) ? 1 : 0);
    set_one(channels, 494, !state.return_obligations.empty() ? 1 : 0);
    set_one(channels, 495, state.tribute_selections.contains(observer) ? 1 : 0);
    set_one(channels, 496, std::any_of(state.return_transfers.begin(), state.return_transfers.end(), [&](const auto& transfer) { return transfer.source == observer; }) ? 1 : 0);
    set_one(channels, 497, state.previous_result ? 1 : 0);
    set_one(channels, 498, state.current_winner_seat == observer ? 1 : 0);
    set_one(channels, 499, state.leader_seat == observer ? 1 : 0);
    set_one(channels, 500, TEAM_OF[state.active_seat] == TEAM_OF[observer] ? 1 : 0);
    set_one(channels, 511);
    return channels;
}

void pad_event(std::array<int, MAX_OBSERVATION_TOKENS>& output, const std::vector<int>& event, int capacity) {
    if (capacity <= 0 || capacity > MAX_OBSERVATION_TOKENS) throw std::invalid_argument("invalid observation capacity");
    if (event.size() > static_cast<std::size_t>(capacity)) throw CapacityError("event exceeds observation token capacity");
    for (int token : event) if (token <= 0 || token >= NUM_TOKENS) throw ProtocolError("event contains PAD or out-of-vocabulary token");
    std::copy(event.begin(), event.end(), output.begin());
}

}  // namespace

Observation encode_observation(const Game& game, const std::array<std::vector<int>, 4>& events, int observation_capacity) {
    Observation observation{};
    for (int observer = 0; observer < 4; ++observer) {
        pad_event(observation.tokens[observer], events[observer], observation_capacity);
        observation.channels[observer] = encode_channels(game, observer);
    }
    return observation;
}

std::array<std::vector<int>, 4> reset_events(const Game& game) {
    const std::vector<int> event{
        BOS_TOKEN, EVENT_RESET, phase_token(game), token_rank(game.state.level_rank), token_length(game.state.active_seat), EVENT_END,
    };
    return {event, event, event, event};
}

std::array<std::vector<int>, 4> step_events(const Game& before, const ProtocolResult& result, int token_id, const Game& after) {
    const int actor = before.state.active_seat;
    const int phase = phase_token(before);
    const int seat_token = token_length(actor);
    std::array<std::vector<int>, 4> events;
    if (!result.is_commit) {
        const std::vector<int> private_event{BOS_TOKEN, EVENT_PRIVATE_PREFIX, phase, seat_token, token_id, EVENT_END};
        events[actor] = private_event;
        return events;
    }
    std::vector<int> sequence;
    if (token_id == PASS_TOKEN) sequence = {PASS_TOKEN};
    else {
        sequence = before.prefix_tokens;
        sequence.push_back(token_id);
    }
    const bool before_exchange = before.state.phase == Phase::TRIBUTE || before.state.phase == Phase::RETURN;
    const bool after_exchange = after.state.phase == Phase::TRIBUTE || after.state.phase == Phase::RETURN;
    if (before_exchange && after_exchange) {
        std::vector<int> private_event{BOS_TOKEN, EVENT_PRIVATE_EXCHANGE_COMMIT, phase, seat_token};
        private_event.insert(private_event.end(), sequence.begin(), sequence.end());
        private_event.push_back(EVENT_END);
        events[actor] = std::move(private_event);
        return events;
    }
    if (before_exchange && after.state.phase == Phase::PLAY) {
        std::vector<int> event{BOS_TOKEN, EVENT_EXCHANGE_RESOLVED, phase, seat_token};
        for (const auto& transfer : after.state.tribute_transfers) {
            const std::array<int, 5> record{TRIBUTE_TOKEN, token_length(transfer.source), token_length(transfer.target), token_card(transfer.card_id), COMMIT_TOKEN};
            event.insert(event.end(), record.begin(), record.end());
        }
        for (const auto& transfer : after.state.return_transfers) {
            const std::array<int, 5> record{RETURN_TOKEN, token_length(transfer.source), token_length(transfer.target), token_card(transfer.card_id), COMMIT_TOKEN};
            event.insert(event.end(), record.begin(), record.end());
        }
        event.push_back(EVENT_END);
        events = {event, event, event, event};
        return events;
    }
    std::vector<int> event{BOS_TOKEN, EVENT_PUBLIC_COMMIT, phase, seat_token};
    event.insert(event.end(), sequence.begin(), sequence.end());
    event.push_back(EVENT_END);
    events = {event, event, event, event};
    return events;
}

}  // namespace nanopokerlib::guandan
