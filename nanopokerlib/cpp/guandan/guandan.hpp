#pragma once

// SPDX-License-Identifier: MIT
// Guandan-specific implementation. Rule semantics are reimplemented from the
// accepted Python oracle in this repository; no Doudizhu game code is copied.

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../core/types.hpp"

namespace nanopokerlib::guandan {

inline constexpr int NUM_PLAYERS = 4;
inline constexpr int NUM_PHYSICAL_CARDS = 108;
inline constexpr int NUM_TOKENS = 320;
inline constexpr int NUM_CHANNELS = 512;
inline constexpr int MAX_GAME_HORIZON = 1024;
inline constexpr int MAX_AVAILABLE_ACTIONS = 256;
inline constexpr int MAX_OBSERVATION_TOKENS = 48;
inline constexpr int MAX_ACTION_TOKENS = 8;
inline constexpr int MAX_PREFIX_TOKENS = 32;

inline constexpr const char* ACTION_VERSION = "GD-ACTION-0.2-MARVEL";
inline constexpr const char* ENV_VERSION = "GD-ENV-0.2-MARVEL";
inline constexpr const char* ENCODING_VERSION = "GD-ENCODING-0.2-MARVEL";
inline constexpr const char* SERIALIZATION_SCHEMA = "guandan-marvel-cpp-runtime-v1";
inline constexpr const char* BACKEND = "cpp";
inline constexpr const char* IDENTIFIER = "GuandanMARVELV2Cpp";

// V1 action vocabulary.
inline constexpr int PAD_TOKEN = 0;
inline constexpr int BOS_TOKEN = 1;
inline constexpr int LEAD_PLAY_TOKEN = 2;
inline constexpr int FOLLOW_PLAY_TOKEN = 3;
inline constexpr int TRIBUTE_TOKEN = 4;
inline constexpr int RETURN_TOKEN = 5;
inline constexpr int PASS_TOKEN = 6;
inline constexpr int COMMIT_TOKEN = 7;
inline constexpr int FAMILY_TOKEN_BASE = 8;
inline constexpr int RANK_TOKEN_BASE = 18;
inline constexpr int SUIT_TOKEN_BASE = 33;
inline constexpr int LENGTH_TOKEN_BASE = 38;
inline constexpr int CARD_TOKEN_BASE = 64;
inline constexpr int WILD_TOKEN_BASE = 172;

// V2 event/action framing vocabulary.
inline constexpr int EVENT_RESET = 256;
inline constexpr int EVENT_PRIVATE_PREFIX = 257;
inline constexpr int EVENT_PUBLIC_COMMIT = 258;
inline constexpr int EVENT_PRIVATE_EXCHANGE_COMMIT = 259;
inline constexpr int EVENT_EXCHANGE_RESOLVED = 260;
inline constexpr int EVENT_HIDDEN_PRIVATE_STEP = 261;
inline constexpr int ACTION_ROW_BEGIN = 262;
inline constexpr int ACTION_STAGE_BASE = 263;
inline constexpr int ACTION_KIND_BASE = 296;
inline constexpr int ACTION_PAYLOAD = 304;
inline constexpr int ACTION_NON_COMMIT = 305;
inline constexpr int ACTION_COMMIT = 306;
inline constexpr int ACTION_ROW_END = 307;
inline constexpr int EVENT_END = 308;

class GuandanError : public std::runtime_error { using std::runtime_error::runtime_error; };
class IllegalActionError : public GuandanError { using GuandanError::GuandanError; };
class EpisodeTerminatedError : public GuandanError { using GuandanError::GuandanError; };
class StateInvariantError : public GuandanError { using GuandanError::GuandanError; };
class ProtocolError : public GuandanError { using GuandanError::GuandanError; };
class CapacityError : public ProtocolError { using ProtocolError::ProtocolError; };
class HorizonError : public CapacityError { using CapacityError::CapacityError; };

// Numeric values intentionally match guandan.cards.Suit/Rank.
enum class Suit : int { CLUBS = 0, DIAMONDS = 1, HEARTS = 2, SPADES = 3, JOKER = 4 };
enum class Rank : int {
    THREE = 3, FOUR = 4, FIVE = 5, SIX = 6, SEVEN = 7, EIGHT = 8,
    NINE = 9, TEN = 10, JACK = 11, QUEEN = 12, KING = 13, ACE = 14,
    TWO = 15, SMALL_JOKER = 16, BIG_JOKER = 17,
};

enum class CombinationKind : int {
    SINGLE = 0,
    PAIR = 1,
    TRIPLE = 2,
    FULL_HOUSE = 3,
    STRAIGHT = 4,
    PAIR_SEQUENCE = 5,
    TRIPLE_SEQUENCE = 6,
    RANK_BOMB = 7,
    STRAIGHT_FLUSH = 8,
    FOUR_KINGS = 9,
};

enum class Phase : int { PLAY = 0, TRIBUTE = 1, RETURN = 2, TERMINAL = 3 };

struct Card {
    int id{};
    [[nodiscard]] int deck_index() const;
    [[nodiscard]] int deck_offset() const;
    [[nodiscard]] Rank rank() const;
    [[nodiscard]] Suit suit() const;
    [[nodiscard]] bool is_joker() const;
    [[nodiscard]] bool is_level_wild(Rank level) const;
    bool operator==(const Card&) const = default;
};

struct WildAssignment {
    int card_id{};
    Rank rank{Rank::THREE};
    Suit suit{Suit::CLUBS};
    bool natural{};
    bool operator==(const WildAssignment&) const = default;
};

struct Combination {
    CombinationKind kind{CombinationKind::SINGLE};
    std::vector<int> card_ids;
    Rank level_rank{Rank::TWO};
    Rank comparison_rank{Rank::THREE};
    std::vector<int> comparison_key;
    std::vector<WildAssignment> wild_assignments;
    std::vector<int> natural_wild_ids;

    [[nodiscard]] bool is_bomb() const;
    [[nodiscard]] int size() const { return static_cast<int>(card_ids.size()); }
    bool operator==(const Combination&) const = default;
};

struct Action {
    int player_id{};
    std::string kind;
    std::vector<int> card_ids;
    std::optional<Combination> declaration;

    [[nodiscard]] bool is_pass() const { return kind == "pass"; }
    bool operator==(const Action&) const = default;
};

struct PreviousResult {
    std::array<int, 4> ranking{};  // finishing rank by seat
    int winner_team{};
    std::string outcome_class;
    bool tribute_cancelled{};

    [[nodiscard]] int seat_with_rank(int rank) const;
    void validate() const;
    bool operator==(const PreviousResult&) const = default;
};

struct Transfer {
    int source{};
    int target{};
    int card_id{};
    bool operator==(const Transfer&) const = default;
};

struct RoundState {
    std::array<std::vector<int>, 4> hands;
    Rank level_rank{Rank::TWO};
    int active_seat{};
    int leader_seat{};
    Phase phase{Phase::PLAY};
    std::optional<Combination> current_winning;
    int current_winner_seat{-1};
    int consecutive_passes{};
    std::vector<int> finished_ranks;
    bool done{};
    std::array<float, 4> rewards{};
    int committed_step{};
    std::vector<Action> history;
    std::optional<PreviousResult> previous_result;
    std::vector<int> tribute_donors;
    std::vector<int> tribute_order;
    std::map<int, int> tribute_selections;
    std::vector<int> tribute_escrow;
    std::vector<Transfer> tribute_transfers;
    std::vector<std::pair<int, int>> return_obligations;
    std::vector<Transfer> return_transfers;
    bool anti_tribute{};
    std::vector<int> emptied_seats;
    std::string outcome_class;
    int winner_team{-1};
    std::array<float, 2> team_rewards{};
    int finish_committed_step{-1};
    int finish_token_step{-1};

    [[nodiscard]] std::array<int, 4> ranking() const;
    [[nodiscard]] std::vector<int> pending_tribute_donors() const;
    [[nodiscard]] std::optional<std::pair<int, int>> pending_return() const;
    void initialize_exchange();
    void check_invariants() const;
};

struct ProtocolResult {
    std::array<float, 4> rewards{};
    bool done{};
    bool truncated{};
    bool is_commit{};
    std::optional<Action> committed_action;
    int token_step{};
    int committed_step{};
    std::string termination_reason;
};

struct GameConfig {
    Rank level_rank{Rank::TWO};
    std::uint64_t seed{};
    int leader_seat{};
    std::optional<PreviousResult> previous_result;
    int max_token_steps{MAX_GAME_HORIZON};
};

struct Game {
    GameConfig config;
    RoundState state;
    std::vector<int> prefix_tokens;
    int token_step{};
    bool truncated{};
    std::mt19937_64 rng{};
    std::uint64_t reset_count{};
    mutable bool candidate_cache_valid{};
    mutable std::vector<std::pair<std::vector<int>, Action>> candidate_cache;

    explicit Game(GameConfig cfg = {});
    void reset(std::uint64_t seed);
    void reset_auto();
    void reset_from_hands(const std::array<std::vector<int>, 4>& hands);
    [[nodiscard]] std::vector<std::pair<std::vector<int>, Action>> candidate_sequences() const;
    [[nodiscard]] std::vector<int> legal_next_tokens() const;
    [[nodiscard]] std::vector<core::ActionRow> candidate_rows() const;
    ProtocolResult step_token(int token_id);
    void check_invariants() const;
};

using Observation = core::Observation<NUM_PLAYERS, MAX_OBSERVATION_TOKENS, NUM_CHANNELS>;
using StepResult = core::StepResult<NUM_PLAYERS, MAX_OBSERVATION_TOKENS, NUM_CHANNELS>;

[[nodiscard]] const std::array<Rank, 13>& printed_ranks();
[[nodiscard]] const std::array<Rank, 13>& window_ranks();
[[nodiscard]] const std::array<Rank, 15>& rank_order();
[[nodiscard]] int effective_rank_value(Rank rank, Rank level);
[[nodiscard]] int rank_order_index(Rank rank);
[[nodiscard]] int window_rank_value(Rank rank);
[[nodiscard]] std::vector<int> sort_card_ids(const std::vector<int>& ids);
[[nodiscard]] std::vector<Combination> recognize_combinations(const std::vector<int>& card_ids, Rank level);
[[nodiscard]] int compare_combinations(const Combination& left, const Combination& right);
[[nodiscard]] bool can_beat(const Combination& candidate, const Combination& current);
[[nodiscard]] std::vector<Combination> enumerate_plays(const RoundState& state, int seat);
[[nodiscard]] std::vector<Action> enumerate_legal_actions(const RoundState& state);
void apply_action(RoundState& state, const Action& action);
[[nodiscard]] std::vector<int> canonical_tokens(const Action& action);
[[nodiscard]] core::ActionRow action_row(const Game& game, int token_id);
[[nodiscard]] std::string combination_kind_name(CombinationKind kind);
[[nodiscard]] CombinationKind combination_kind_from_name(const std::string& name);
[[nodiscard]] std::string phase_name(Phase phase);
[[nodiscard]] int phase_token(const Game& game);
[[nodiscard]] std::string token_kind(int token_id);
[[nodiscard]] int token_rank(Rank rank);
[[nodiscard]] Rank rank_from_token(int token);
[[nodiscard]] int token_suit(Suit suit);
[[nodiscard]] int token_length(int length);
[[nodiscard]] int token_card(int card_id);
[[nodiscard]] int token_wild(Rank rank, Suit suit);

[[nodiscard]] Observation encode_observation(const Game& game, const std::array<std::vector<int>, 4>& events, int observation_capacity = MAX_OBSERVATION_TOKENS);
[[nodiscard]] std::array<std::vector<int>, 4> reset_events(const Game& game);
[[nodiscard]] std::array<std::vector<int>, 4> step_events(const Game& before, const ProtocolResult& result, int token_id, const Game& after);

}  // namespace nanopokerlib::guandan
