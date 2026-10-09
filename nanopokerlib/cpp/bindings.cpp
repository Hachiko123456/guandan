// SPDX-License-Identifier: MIT
// pybind batch interface built on the Guandan-specific C++ runtime.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "guandan/guandan.hpp"

namespace py = pybind11;
using namespace nanopokerlib::guandan;
namespace core = nanopokerlib::core;

namespace {

constexpr std::array<const char*, 8> TOKEN_KIND_NAMES{
    "PASS", "COMMIT", "FAMILY", "RANK", "SUIT", "LENGTH", "CARD", "WILD_ASSIGNMENT",
};

std::string local_vector_key(const std::vector<int>& values) {
    std::ostringstream out;
    for (int value : values) out << value << ",";
    return out.str();
}

struct PyObservationBatch {
    py::array_t<std::int64_t> tokens;
    py::array_t<std::uint8_t> channels;
};

struct PyAvailableActions {
    py::array_t<std::int64_t> tokens;
    py::array_t<float> weights;
    py::array_t<bool> is_commit;
};

struct PyCriticState {
    py::array_t<std::uint8_t> ownership;
    py::array_t<std::uint8_t> escrow;
    py::array_t<std::uint8_t> played;
};

struct PyEnvStep {
    py::array_t<float> rewards;
    py::array_t<bool> is_terminal;
    py::array_t<std::int64_t> player_ids;
    PyObservationBatch observations;
    PyAvailableActions available_actions;
    py::tuple infos;
};

PreviousResult previous_from_object(const py::handle& value) {
    if (value.is_none()) throw std::invalid_argument("previous result is None");
    py::dict data;
    if (py::isinstance<py::dict>(value)) data = py::reinterpret_borrow<py::dict>(value);
    else {
        data["ranking"] = value.attr("ranking");
        data["winner_team"] = value.attr("winner_team");
        data["outcome_class"] = value.attr("outcome_class");
        data["tribute_cancelled"] = py::hasattr(value, "tribute_cancelled") ? value.attr("tribute_cancelled") : py::bool_(false);
    }
    PreviousResult result;
    const auto ranking = data["ranking"].cast<std::vector<int>>();
    if (ranking.size() != 4) throw std::invalid_argument("previous_result ranking must have four entries");
    std::copy(ranking.begin(), ranking.end(), result.ranking.begin());
    result.winner_team = data["winner_team"].cast<int>();
    result.outcome_class = data["outcome_class"].cast<std::string>();
    result.tribute_cancelled = data.contains("tribute_cancelled") ? data["tribute_cancelled"].cast<bool>() : false;
    result.validate();
    return result;
}

py::dict previous_to_dict(const std::optional<PreviousResult>& value) {
    py::dict result;
    if (!value) return result;
    result["ranking"] = value->ranking;
    result["winner_team"] = value->winner_team;
    result["outcome_class"] = value->outcome_class;
    result["tribute_cancelled"] = value->tribute_cancelled;
    return result;
}

py::dict combination_to_dict(const Combination& combo) {
    py::dict result;
    result["kind"] = combination_kind_name(combo.kind);
    result["card_ids"] = combo.card_ids;
    result["level_rank"] = static_cast<int>(combo.level_rank);
    result["comparison_rank"] = static_cast<int>(combo.comparison_rank);
    result["comparison_key"] = combo.comparison_key;
    py::list assignments;
    for (const auto& value : combo.wild_assignments) assignments.append(py::make_tuple(value.card_id, static_cast<int>(value.rank), static_cast<int>(value.suit), value.natural));
    result["wild_assignments"] = assignments;
    result["natural_wild_ids"] = combo.natural_wild_ids;
    return result;
}

Combination combination_from_dict(const py::dict& data) {
    Combination combo;
    combo.kind = combination_kind_from_name(data["kind"].cast<std::string>());
    combo.card_ids = data["card_ids"].cast<std::vector<int>>();
    combo.level_rank = static_cast<Rank>(data["level_rank"].cast<int>());
    combo.comparison_rank = static_cast<Rank>(data["comparison_rank"].cast<int>());
    combo.comparison_key = data["comparison_key"].cast<std::vector<int>>();
    for (const py::handle item : data["wild_assignments"].cast<py::list>()) {
        const py::sequence values = py::reinterpret_borrow<py::sequence>(item);
        combo.wild_assignments.push_back({values[0].cast<int>(), static_cast<Rank>(values[1].cast<int>()), static_cast<Suit>(values[2].cast<int>()), values[3].cast<bool>()});
    }
    combo.natural_wild_ids = data["natural_wild_ids"].cast<std::vector<int>>();
    return combo;
}

py::dict action_to_dict(const Action& action) {
    py::dict result;
    result["player_id"] = action.player_id;
    result["kind"] = action.kind;
    result["card_ids"] = action.card_ids;
    if (action.declaration) result["declaration"] = combination_to_dict(*action.declaration);
    else result["declaration"] = py::none();
    return result;
}

Action action_from_dict(const py::dict& data) {
    Action action;
    action.player_id = data["player_id"].cast<int>();
    action.kind = data["kind"].cast<std::string>();
    action.card_ids = data["card_ids"].cast<std::vector<int>>();
    if (!data["declaration"].is_none()) action.declaration = combination_from_dict(data["declaration"].cast<py::dict>());
    return action;
}

py::dict transfer_to_dict(const Transfer& transfer) {
    py::dict result;
    result["source"] = transfer.source;
    result["target"] = transfer.target;
    result["card_id"] = transfer.card_id;
    return result;
}

Transfer transfer_from_dict(const py::dict& data) {
    return {data["source"].cast<int>(), data["target"].cast<int>(), data["card_id"].cast<int>()};
}

py::dict game_to_dict(const Game& game) {
    py::dict result;
    py::dict config;
    config["level_rank"] = static_cast<int>(game.config.level_rank);
    config["seed"] = game.config.seed;
    config["leader_seat"] = game.config.leader_seat;
    if (game.config.previous_result) config["previous_result"] = previous_to_dict(game.config.previous_result);
    else config["previous_result"] = py::none();
    config["max_token_steps"] = game.config.max_token_steps;
    result["config"] = config;
    const RoundState& state = game.state;
    py::dict raw;
    raw["hands"] = state.hands;
    raw["level_rank"] = static_cast<int>(state.level_rank);
    raw["active_seat"] = state.active_seat;
    raw["leader_seat"] = state.leader_seat;
    raw["phase"] = static_cast<int>(state.phase);
    if (state.current_winning) raw["current_winning"] = combination_to_dict(*state.current_winning);
    else raw["current_winning"] = py::none();
    raw["current_winner_seat"] = state.current_winner_seat;
    raw["consecutive_passes"] = state.consecutive_passes;
    raw["finished_ranks"] = state.finished_ranks;
    raw["done"] = state.done;
    raw["rewards"] = state.rewards;
    raw["committed_step"] = state.committed_step;
    py::list history;
    for (const auto& action : state.history) history.append(action_to_dict(action));
    raw["history"] = history;
    if (state.previous_result) raw["previous_result"] = previous_to_dict(state.previous_result);
    else raw["previous_result"] = py::none();
    raw["tribute_donors"] = state.tribute_donors;
    raw["tribute_order"] = state.tribute_order;
    py::list tribute_selections;
    for (const auto& [seat, card] : state.tribute_selections) tribute_selections.append(py::make_tuple(seat, card));
    raw["tribute_selections"] = tribute_selections;
    raw["tribute_escrow"] = state.tribute_escrow;
    py::list tribute_transfers;
    for (const auto& transfer : state.tribute_transfers) tribute_transfers.append(transfer_to_dict(transfer));
    raw["tribute_transfers"] = tribute_transfers;
    raw["return_obligations"] = state.return_obligations;
    py::list return_transfers;
    for (const auto& transfer : state.return_transfers) return_transfers.append(transfer_to_dict(transfer));
    raw["return_transfers"] = return_transfers;
    raw["anti_tribute"] = state.anti_tribute;
    raw["emptied_seats"] = state.emptied_seats;
    raw["outcome_class"] = state.outcome_class;
    raw["winner_team"] = state.winner_team;
    raw["team_rewards"] = state.team_rewards;
    raw["finish_committed_step"] = state.finish_committed_step;
    raw["finish_token_step"] = state.finish_token_step;
    result["state"] = raw;
    result["prefix_tokens"] = game.prefix_tokens;
    result["token_step"] = game.token_step;
    result["truncated"] = game.truncated;
    std::ostringstream rng_state;
    rng_state << game.rng;
    result["rng_state"] = rng_state.str();
    result["reset_count"] = game.reset_count;
    return result;
}

Game game_from_dict(const py::dict& data) {
    const py::dict config_data = data["config"].cast<py::dict>();
    GameConfig config;
    config.level_rank = static_cast<Rank>(config_data["level_rank"].cast<int>());
    config.seed = config_data["seed"].cast<std::uint64_t>();
    config.leader_seat = config_data["leader_seat"].cast<int>();
    if (!config_data["previous_result"].is_none()) config.previous_result = previous_from_object(config_data["previous_result"]);
    config.max_token_steps = config_data["max_token_steps"].cast<int>();
    Game game(config);
    const py::dict raw = data["state"].cast<py::dict>();
    RoundState state;
    state.hands = raw["hands"].cast<std::array<std::vector<int>, 4>>();
    state.level_rank = static_cast<Rank>(raw["level_rank"].cast<int>());
    state.active_seat = raw["active_seat"].cast<int>();
    state.leader_seat = raw["leader_seat"].cast<int>();
    state.phase = static_cast<Phase>(raw["phase"].cast<int>());
    if (!raw["current_winning"].is_none()) state.current_winning = combination_from_dict(raw["current_winning"].cast<py::dict>());
    state.current_winner_seat = raw["current_winner_seat"].cast<int>();
    state.consecutive_passes = raw["consecutive_passes"].cast<int>();
    state.finished_ranks = raw["finished_ranks"].cast<std::vector<int>>();
    state.done = raw["done"].cast<bool>();
    state.rewards = raw["rewards"].cast<std::array<float, 4>>();
    state.committed_step = raw["committed_step"].cast<int>();
    for (const py::handle item : raw["history"].cast<py::list>()) state.history.push_back(action_from_dict(py::reinterpret_borrow<py::dict>(item)));
    if (!raw["previous_result"].is_none()) state.previous_result = previous_from_object(raw["previous_result"]);
    state.tribute_donors = raw["tribute_donors"].cast<std::vector<int>>();
    state.tribute_order = raw["tribute_order"].cast<std::vector<int>>();
    for (const py::handle item : raw["tribute_selections"].cast<py::list>()) {
        const py::sequence values = py::reinterpret_borrow<py::sequence>(item);
        state.tribute_selections[values[0].cast<int>()] = values[1].cast<int>();
    }
    state.tribute_escrow = raw["tribute_escrow"].cast<std::vector<int>>();
    for (const py::handle item : raw["tribute_transfers"].cast<py::list>()) state.tribute_transfers.push_back(transfer_from_dict(py::reinterpret_borrow<py::dict>(item)));
    state.return_obligations = raw["return_obligations"].cast<std::vector<std::pair<int, int>>>();
    for (const py::handle item : raw["return_transfers"].cast<py::list>()) state.return_transfers.push_back(transfer_from_dict(py::reinterpret_borrow<py::dict>(item)));
    state.anti_tribute = raw["anti_tribute"].cast<bool>();
    state.emptied_seats = raw["emptied_seats"].cast<std::vector<int>>();
    state.outcome_class = raw["outcome_class"].cast<std::string>();
    state.winner_team = raw["winner_team"].cast<int>();
    state.team_rewards = raw["team_rewards"].cast<std::array<float, 2>>();
    state.finish_committed_step = raw["finish_committed_step"].cast<int>();
    state.finish_token_step = raw["finish_token_step"].cast<int>();
    game.state = std::move(state);
    game.prefix_tokens = data["prefix_tokens"].cast<std::vector<int>>();
    game.token_step = data["token_step"].cast<int>();
    game.truncated = data["truncated"].cast<bool>();
    std::istringstream rng_state(data["rng_state"].cast<std::string>());
    rng_state >> game.rng;
    game.reset_count = data["reset_count"].cast<std::uint64_t>();
    game.candidate_cache.clear();
    game.candidate_cache_valid = false;
    game.check_invariants();
    return game;
}

core::ActionRow parse_action_row(const py::handle& row) {
    py::array array = py::array::ensure(row);
    if (!array || array.ndim() != 1 || array.shape(0) != MAX_ACTION_TOKENS ||
        (array.dtype().kind() != 'i' && array.dtype().kind() != 'u')) {
        throw std::invalid_argument("action row must be an integer vector with shape (8,)");
    }
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> values(array);
    const auto view = values.unchecked<1>();
    core::ActionRow result;
    for (int index = 0; index < MAX_ACTION_TOKENS; ++index) result.tokens[index] = static_cast<int>(view(index));
    if (result.tokens[0] != ACTION_ROW_BEGIN || result.tokens[4] != ACTION_PAYLOAD || result.tokens[7] != ACTION_ROW_END) throw ProtocolError("action row framing is invalid");
    if (result.tokens[1] != LEAD_PLAY_TOKEN && result.tokens[1] != FOLLOW_PLAY_TOKEN && result.tokens[1] != TRIBUTE_TOKEN && result.tokens[1] != RETURN_TOKEN) throw ProtocolError("action row phase is invalid");
    result.prefix_stage = result.tokens[2] - ACTION_STAGE_BASE;
    const int kind_code = result.tokens[3] - ACTION_KIND_BASE;
    if (result.prefix_stage < 0 || result.prefix_stage > MAX_PREFIX_TOKENS || kind_code < 0 || kind_code >= 8) throw ProtocolError("action row stage or kind is invalid");
    result.token_id = result.tokens[5];
    result.kind = TOKEN_KIND_NAMES[kind_code];
    if (token_kind(result.token_id) != result.kind) throw ProtocolError("action row kind does not match V1 token");
    if (result.tokens[6] != ACTION_NON_COMMIT && result.tokens[6] != ACTION_COMMIT) throw ProtocolError("action row commit flag is invalid");
    result.is_commit = result.token_id == PASS_TOKEN || result.token_id == COMMIT_TOKEN;
    if ((result.tokens[6] == ACTION_COMMIT) != result.is_commit) throw ProtocolError("action row commit flag does not match V1 token");
    result.phase_token = result.tokens[1];
    for (int token : result.tokens) if (token < 0 || token >= NUM_TOKENS) throw ProtocolError("action row token outside V2 vocabulary");
    return result;
}

std::vector<core::ActionRow> actions_from_array(const py::array& input) {
    if (input.dtype().kind() != 'i' && input.dtype().kind() != 'u') throw std::invalid_argument("rows must use integer dtype");
    py::array array = input;
    if (array.ndim() == 1) {
        if (array.shape(0) != MAX_ACTION_TOKENS) throw std::invalid_argument("rows must have shape [B,8]");
        array = array.attr("reshape")(py::make_tuple(1, MAX_ACTION_TOKENS)).cast<py::array>();
    }
    if (array.ndim() != 2 || array.shape(1) != MAX_ACTION_TOKENS) throw std::invalid_argument("rows must have shape [B,8]");
    std::vector<core::ActionRow> result;
    for (py::ssize_t index = 0; index < array.shape(0); ++index) {
        py::object row = array.attr("__getitem__")(py::make_tuple(index, py::ellipsis()));
        result.push_back(parse_action_row(row));
    }
    return result;
}

std::array<std::vector<int>, 4> events_from_object(const py::handle& value) {
    const auto outer = value.cast<std::vector<std::vector<int>>>();
    if (outer.size() != 4) throw ProtocolError("serialized event must have four seat rows");
    return {outer[0], outer[1], outer[2], outer[3]};
}

class CppEnvBatch {
public:
    CppEnvBatch(
        int batch_size,
        const std::string& backend = BACKEND,
        py::object configs = py::none(),
        bool auto_reset = true,
        int max_horizon = MAX_GAME_HORIZON,
        int action_capacity = MAX_AVAILABLE_ACTIONS,
        int observation_capacity = MAX_OBSERVATION_TOKENS)
        : batch_size_(batch_size), auto_reset_(auto_reset), max_horizon_(max_horizon),
          action_capacity_(action_capacity), observation_capacity_(observation_capacity) {
        if (batch_size_ <= 0) throw std::invalid_argument("batch_size must be positive");
        if (backend != BACKEND) throw std::invalid_argument("unsupported backend; no fallback is available");
        if (max_horizon_ <= 0 || max_horizon_ > MAX_GAME_HORIZON) throw std::invalid_argument("invalid max_horizon");
        if (action_capacity_ <= 0 || action_capacity_ > MAX_AVAILABLE_ACTIONS || observation_capacity_ <= 0 || observation_capacity_ > MAX_OBSERVATION_TOKENS) {
            throw std::invalid_argument("configured capacity exceeds V2 schema");
        }
        std::vector<GameConfig> game_configs;
        if (configs.is_none()) {
            game_configs.assign(batch_size_, GameConfig{});
            for (auto& config : game_configs) config.max_token_steps = max_horizon_;
        } else {
            const py::sequence values = configs.cast<py::sequence>();
            if (py::len(values) != batch_size_) throw std::invalid_argument("configs must contain one entry per slot");
            for (const py::handle value : values) game_configs.push_back(config_from_object(value));
        }
        for (auto& config : game_configs) {
            config.max_token_steps = max_horizon_;
            games_.emplace_back(config);
        }
        events_.resize(batch_size_);
        trajectory_steps_.assign(batch_size_, 0);
        is_reset_.assign(batch_size_, false);
    }

    int batch_size() const { return batch_size_; }
    bool auto_reset() const { return auto_reset_; }
    int max_horizon() const { return max_horizon_; }
    int action_capacity() const { return action_capacity_; }
    int observation_capacity() const { return observation_capacity_; }
    std::string backend() const { return BACKEND; }

    PyEnvStep reset(std::uint64_t seed = 0) {
        auto trials = games_;
        auto trial_events = events_;
        for (int index = 0; index < batch_size_; ++index) {
            trials[index].reset(seed + static_cast<std::uint64_t>(index));
            trial_events[index] = reset_events(trials[index]);
        }
        auto output = build_output(trials, trial_events, {}, {}, {});
        games_ = std::move(trials);
        events_ = std::move(trial_events);
        std::fill(trajectory_steps_.begin(), trajectory_steps_.end(), 0);
        std::fill(is_reset_.begin(), is_reset_.end(), true);
        return output;
    }

    PyEnvStep reset_from_deals(
        const py::sequence& deals,
        py::object level_ranks = py::none(),
        py::object leader_seats = py::none(),
        py::object previous_results = py::none()) {
        if (py::len(deals) != batch_size_) throw std::invalid_argument("deals must contain one deal per slot");
        const auto levels = optional_int_vector(level_ranks, static_cast<int>(Rank::TWO), "level_ranks");
        const auto leaders = optional_int_vector(leader_seats, 0, "leader_seats");
        std::vector<std::optional<PreviousResult>> previous(batch_size_);
        if (!previous_results.is_none()) {
            const py::sequence values = previous_results.cast<py::sequence>();
            if (py::len(values) != batch_size_) throw std::invalid_argument("previous_results must contain one entry per slot");
            for (int index = 0; index < batch_size_; ++index) if (!values[index].is_none()) previous[index] = previous_from_object(values[index]);
        }
        auto trials = games_;
        auto trial_events = events_;
        for (int index = 0; index < batch_size_; ++index) {
            const py::sequence seat_hands = deals[index].cast<py::sequence>();
            if (py::len(seat_hands) != 4) throw std::invalid_argument("each deal must contain four hands");
            std::array<std::vector<int>, 4> hands;
            for (int seat = 0; seat < 4; ++seat) hands[seat] = seat_hands[seat].cast<std::vector<int>>();
            trials[index].config.level_rank = static_cast<Rank>(levels[index]);
            trials[index].config.leader_seat = leaders[index];
            trials[index].config.previous_result = previous[index];
            trials[index].config.max_token_steps = max_horizon_;
            trials[index].reset_from_hands(hands);
            trial_events[index] = reset_events(trials[index]);
        }
        auto output = build_output(trials, trial_events, {}, {}, {});
        games_ = std::move(trials);
        events_ = std::move(trial_events);
        std::fill(trajectory_steps_.begin(), trajectory_steps_.end(), 0);
        std::fill(is_reset_.begin(), is_reset_.end(), true);
        return output;
    }

    PyEnvStep observe() const {
        require_reset();
        return build_output(games_, events_, {}, {}, {});
    }

    PyEnvStep step(const py::array& rows) {
        require_reset();
        const auto actions = actions_from_array(rows);
        if (static_cast<int>(actions.size()) != batch_size_) throw std::invalid_argument("step requires one action row per slot");
        auto trials = games_;
        auto trial_events = events_;
        auto trial_steps = trajectory_steps_;
        std::vector<std::array<float, 4>> rewards(batch_size_);
        std::vector<bool> terminal(batch_size_, false);
        std::vector<py::dict> infos(batch_size_);
        for (int index = 0; index < batch_size_; ++index) {
            Game& game = trials[index];
            if (game.state.done || game.truncated) throw EpisodeTerminatedError("inactive V2 slot cannot be stepped");
            if (trial_steps[index] >= max_horizon_) throw HorizonError("trajectory horizon exceeded before terminal");
            const auto candidates = game.candidate_rows();
            const auto matches = std::count_if(candidates.begin(), candidates.end(), [&](const auto& candidate) { return candidate == actions[index]; });
            if (matches != 1) throw ProtocolError("action row is stale, forged, or not uniquely executable");
            const Game before = game;
            const ProtocolResult result = game.step_token(actions[index].token_id);
            if (result.truncated) throw HorizonError("token horizon reached before terminal");
            ++trial_steps[index];
            rewards[index] = result.rewards;
            terminal[index] = result.done;
            py::dict info;
            info["backend"] = BACKEND;
            info["action_version"] = ACTION_VERSION;
            info["environment_version"] = ENV_VERSION;
            info["encoding_version"] = ENCODING_VERSION;
            if (result.termination_reason.empty()) info["termination_reason"] = py::none();
            else info["termination_reason"] = result.termination_reason;
            if (result.done) {
                info["ranking"] = game.state.ranking();
                info["winner_team"] = game.state.winner_team;
                info["outcome_class"] = game.state.outcome_class;
                info["team_reward"] = game.state.team_rewards;
                info["finish_token_step"] = game.state.finish_token_step;
                info["finish_committed_step"] = game.state.finish_committed_step;
                info["terminal_state"] = game_to_dict(game);
                if (auto_reset_) {
                    game.reset_auto();
                    trial_events[index] = reset_events(game);
                    trial_steps[index] = 0;
                } else trial_events[index] = {};
            } else trial_events[index] = step_events(before, result, actions[index].token_id, game);
            infos[index] = std::move(info);
        }
        auto output = build_output(trials, trial_events, rewards, terminal, infos);
        games_ = std::move(trials);
        events_ = std::move(trial_events);
        trajectory_steps_ = std::move(trial_steps);
        return output;
    }

    PyCriticState critic_state() const {
        require_reset();
        PyCriticState result{
            py::array_t<std::uint8_t>({batch_size_, 4, NUM_PHYSICAL_CARDS}),
            py::array_t<std::uint8_t>({batch_size_, NUM_PHYSICAL_CARDS}),
            py::array_t<std::uint8_t>({batch_size_, NUM_PHYSICAL_CARDS}),
        };
        std::memset(result.ownership.mutable_data(), 0, static_cast<std::size_t>(batch_size_) * 4 * NUM_PHYSICAL_CARDS);
        std::memset(result.escrow.mutable_data(), 0, static_cast<std::size_t>(batch_size_) * NUM_PHYSICAL_CARDS);
        std::memset(result.played.mutable_data(), 0, static_cast<std::size_t>(batch_size_) * NUM_PHYSICAL_CARDS);
        auto ownership = result.ownership.mutable_unchecked<3>();
        auto escrow = result.escrow.mutable_unchecked<2>();
        auto played = result.played.mutable_unchecked<2>();
        for (int batch = 0; batch < batch_size_; ++batch) {
            const auto& state = games_[batch].state;
            for (int seat = 0; seat < 4; ++seat) for (int card : state.hands[seat]) ownership(batch, seat, card) = 1;
            for (int card : state.tribute_escrow) escrow(batch, card) = 1;
            for (const auto& action : state.history) if (action.kind != "pass" && action.kind != "tribute" && action.kind != "return") for (int card : action.card_ids) played(batch, card) = 1;
            for (int card = 0; card < NUM_PHYSICAL_CARDS; ++card) {
                int total = escrow(batch, card) + played(batch, card);
                for (int seat = 0; seat < 4; ++seat) total += ownership(batch, seat, card);
                if (total != 1) throw ProtocolError("privileged ownership labels do not conserve 108 physical cards");
            }
        }
        return result;
    }

    py::dict debug_state(int index) const {
        require_reset();
        if (index < 0 || index >= batch_size_) throw std::out_of_range("batch index out of range");
        py::dict result = game_to_dict(games_[index]);
        result["ranking"] = games_[index].state.ranking();
        result["trajectory_step"] = trajectory_steps_[index];
        py::list rows;
        if (!games_[index].state.done && !games_[index].truncated) for (const auto& row : games_[index].candidate_rows()) rows.append(row.tokens);
        result["available_action_rows"] = rows;
        return result;
    }

    py::bytes serialize() const {
        require_reset();
        py::dict payload;
        payload["schema"] = SERIALIZATION_SCHEMA;
        payload["backend"] = BACKEND;
        payload["action_version"] = ACTION_VERSION;
        payload["environment_version"] = ENV_VERSION;
        payload["encoding_version"] = ENCODING_VERSION;
        payload["auto_reset"] = auto_reset_;
        payload["max_horizon"] = max_horizon_;
        payload["action_capacity"] = action_capacity_;
        payload["observation_capacity"] = observation_capacity_;
        payload["trajectory_steps"] = trajectory_steps_;
        py::list raw_events;
        for (const auto& event : events_) raw_events.append(event);
        payload["events"] = raw_events;
        py::list raw_games;
        for (const auto& game : games_) raw_games.append(game_to_dict(game));
        payload["games"] = raw_games;
        const py::object json = py::module_::import("json");
        const std::string text = json.attr("dumps")(payload, py::arg("sort_keys") = true, py::arg("separators") = py::make_tuple(",", ":"), py::arg("allow_nan") = false).cast<std::string>();
        return py::bytes(text);
    }

    static std::unique_ptr<CppEnvBatch> deserialize(const py::bytes& payload) {
        const std::string text = payload;
        const py::dict data = py::module_::import("json").attr("loads")(text).cast<py::dict>();
        if (data["schema"].cast<std::string>() != SERIALIZATION_SCHEMA || data["backend"].cast<std::string>() != BACKEND ||
            data["action_version"].cast<std::string>() != ACTION_VERSION || data["environment_version"].cast<std::string>() != ENV_VERSION ||
            data["encoding_version"].cast<std::string>() != ENCODING_VERSION) throw ProtocolError("unsupported C++ serialization or protocol version");
        const py::list raw_games = data["games"].cast<py::list>();
        if (raw_games.empty()) throw ProtocolError("serialized batch must contain games");
        auto result = std::make_unique<CppEnvBatch>(
            static_cast<int>(raw_games.size()), BACKEND, py::none(), data["auto_reset"].cast<bool>(), data["max_horizon"].cast<int>(),
            data["action_capacity"].cast<int>(), data["observation_capacity"].cast<int>());
        result->games_.clear();
        for (const py::handle item : raw_games) result->games_.push_back(game_from_dict(py::reinterpret_borrow<py::dict>(item)));
        result->events_.clear();
        for (const py::handle item : data["events"].cast<py::list>()) result->events_.push_back(events_from_object(item));
        result->trajectory_steps_ = data["trajectory_steps"].cast<std::vector<int>>();
        if (result->events_.size() != result->games_.size() || result->trajectory_steps_.size() != result->games_.size()) throw ProtocolError("serialized batch shape mismatch");
        result->is_reset_.assign(result->batch_size_, true);
        result->observe();
        return result;
    }

private:
    int batch_size_;
    bool auto_reset_;
    int max_horizon_;
    int action_capacity_;
    int observation_capacity_;
    std::vector<Game> games_;
    std::vector<std::array<std::vector<int>, 4>> events_;
    std::vector<int> trajectory_steps_;
    std::vector<bool> is_reset_;

    GameConfig config_from_object(const py::handle& value) const {
        GameConfig config;
        config.max_token_steps = max_horizon_;
        if (py::isinstance<py::dict>(value)) {
            const py::dict data = py::reinterpret_borrow<py::dict>(value);
            if (data.contains("level_rank")) config.level_rank = static_cast<Rank>(data["level_rank"].cast<int>());
            if (data.contains("seed") && !data["seed"].is_none()) config.seed = data["seed"].cast<std::uint64_t>();
            if (data.contains("leader_seat")) config.leader_seat = data["leader_seat"].cast<int>();
            if (data.contains("previous_result") && !data["previous_result"].is_none()) config.previous_result = previous_from_object(data["previous_result"]);
        } else {
            if (py::hasattr(value, "level_rank")) config.level_rank = static_cast<Rank>(value.attr("level_rank").cast<int>());
            if (py::hasattr(value, "seed") && !value.attr("seed").is_none()) config.seed = value.attr("seed").cast<std::uint64_t>();
            if (py::hasattr(value, "leader_seat")) config.leader_seat = value.attr("leader_seat").cast<int>();
            if (py::hasattr(value, "previous_result") && !value.attr("previous_result").is_none()) config.previous_result = previous_from_object(value.attr("previous_result"));
        }
        return config;
    }

    std::vector<int> optional_int_vector(const py::object& value, int fallback, const char* name) const {
        if (value.is_none()) return std::vector<int>(batch_size_, fallback);
        const auto result = value.cast<std::vector<int>>();
        if (static_cast<int>(result.size()) != batch_size_) throw std::invalid_argument(std::string(name) + " must contain one entry per slot");
        return result;
    }

    void require_reset() const {
        if (std::any_of(is_reset_.begin(), is_reset_.end(), [](bool value) { return !value; })) throw std::runtime_error("every slot must be reset before use");
    }

    PyEnvStep build_output(
        const std::vector<Game>& games,
        const std::vector<std::array<std::vector<int>, 4>>& events,
        const std::vector<std::array<float, 4>>& rewards,
        const std::vector<bool>& terminal,
        const std::vector<py::dict>& infos) const {
        PyEnvStep result{
            py::array_t<float>({batch_size_, 4}),
            py::array_t<bool>({batch_size_}),
            py::array_t<std::int64_t>({batch_size_}),
            {py::array_t<std::int64_t>({batch_size_, 4, observation_capacity_}), py::array_t<std::uint8_t>({batch_size_, 4, NUM_CHANNELS})},
            {py::array_t<std::int64_t>({batch_size_, action_capacity_, MAX_ACTION_TOKENS}), py::array_t<float>({batch_size_, action_capacity_}), py::array_t<bool>({batch_size_, action_capacity_})},
            py::tuple(batch_size_),
        };
        std::fill(result.rewards.mutable_data(), result.rewards.mutable_data() + static_cast<std::size_t>(batch_size_) * 4, 0.0f);
        std::fill(result.is_terminal.mutable_data(), result.is_terminal.mutable_data() + batch_size_, false);
        std::fill(result.player_ids.mutable_data(), result.player_ids.mutable_data() + batch_size_, -1);
        std::fill(result.observations.tokens.mutable_data(), result.observations.tokens.mutable_data() + static_cast<std::size_t>(batch_size_) * 4 * observation_capacity_, 0);
        std::fill(result.observations.channels.mutable_data(), result.observations.channels.mutable_data() + static_cast<std::size_t>(batch_size_) * 4 * NUM_CHANNELS, 0);
        std::fill(result.available_actions.tokens.mutable_data(), result.available_actions.tokens.mutable_data() + static_cast<std::size_t>(batch_size_) * action_capacity_ * MAX_ACTION_TOKENS, 0);
        std::fill(result.available_actions.weights.mutable_data(), result.available_actions.weights.mutable_data() + static_cast<std::size_t>(batch_size_) * action_capacity_, 0.0f);
        std::fill(result.available_actions.is_commit.mutable_data(), result.available_actions.is_commit.mutable_data() + static_cast<std::size_t>(batch_size_) * action_capacity_, false);
        auto reward_view = result.rewards.mutable_unchecked<2>();
        auto terminal_view = result.is_terminal.mutable_unchecked<1>();
        auto player_view = result.player_ids.mutable_unchecked<1>();
        auto obs_tokens = result.observations.tokens.mutable_unchecked<3>();
        auto obs_channels = result.observations.channels.mutable_unchecked<3>();
        auto action_tokens = result.available_actions.tokens.mutable_unchecked<3>();
        auto action_weights = result.available_actions.weights.mutable_unchecked<2>();
        auto action_commit = result.available_actions.is_commit.mutable_unchecked<2>();
        for (int batch = 0; batch < batch_size_; ++batch) {
            if (!rewards.empty()) for (int seat = 0; seat < 4; ++seat) reward_view(batch, seat) = rewards[batch][seat];
            if (!terminal.empty()) terminal_view(batch) = terminal[batch];
            result.infos[batch] = infos.empty() ? py::dict() : infos[batch];
            if (games[batch].state.done || games[batch].truncated) continue;
            player_view(batch) = games[batch].state.active_seat;
            const Observation observation = encode_observation(games[batch], events[batch], observation_capacity_);
            for (int observer = 0; observer < 4; ++observer) {
                for (int token = 0; token < observation_capacity_; ++token) obs_tokens(batch, observer, token) = observation.tokens[observer][token];
                for (int channel = 0; channel < NUM_CHANNELS; ++channel) obs_channels(batch, observer, channel) = observation.channels[observer][channel];
            }
            const auto rows = games[batch].candidate_rows();
            if (static_cast<int>(rows.size()) > action_capacity_) throw CapacityError("legal action count exceeds configured capacity");
            for (int action = 0; action < static_cast<int>(rows.size()); ++action) {
                for (int token = 0; token < MAX_ACTION_TOKENS; ++token) action_tokens(batch, action, token) = rows[action].tokens[token];
                action_weights(batch, action) = 1.0f;
                action_commit(batch, action) = rows[action].is_commit;
            }
        }
        return result;
    }
};

py::dict card_info(int card_id, int level_rank_value) {
    const Card card{card_id};
    const Rank level = static_cast<Rank>(level_rank_value);
    py::dict result;
    result["card_id"] = card.id;
    result["deck_index"] = card.deck_index();
    result["deck_offset"] = card.deck_offset();
    result["rank"] = static_cast<int>(card.rank());
    result["suit"] = static_cast<int>(card.suit());
    result["is_joker"] = card.is_joker();
    result["is_level_wild"] = card.is_level_wild(level);
    result["effective_rank_value"] = effective_rank_value(card.rank(), level);
    return result;
}

}  // namespace

PYBIND11_MODULE(_guandan_cpp, module) {
    module.doc() = "C++/pybind Guandan MARVEL V2 runtime";

    py::register_exception<GuandanError>(module, "GuandanError");
    py::register_exception<IllegalActionError>(module, "IllegalActionError", PyExc_RuntimeError);
    py::register_exception<EpisodeTerminatedError>(module, "EpisodeTerminatedError", PyExc_RuntimeError);
    py::register_exception<StateInvariantError>(module, "StateInvariantError", PyExc_RuntimeError);
    py::register_exception<ProtocolError>(module, "ProtocolError", PyExc_RuntimeError);
    py::register_exception<CapacityError>(module, "AdapterCapacityError", PyExc_RuntimeError);
    py::register_exception<HorizonError>(module, "TrajectoryHorizonError", PyExc_RuntimeError);

    py::enum_<Suit>(module, "Suit")
        .value("CLUBS", Suit::CLUBS).value("DIAMONDS", Suit::DIAMONDS).value("HEARTS", Suit::HEARTS)
        .value("SPADES", Suit::SPADES).value("JOKER", Suit::JOKER);
    py::enum_<Rank>(module, "Rank")
        .value("THREE", Rank::THREE).value("FOUR", Rank::FOUR).value("FIVE", Rank::FIVE)
        .value("SIX", Rank::SIX).value("SEVEN", Rank::SEVEN).value("EIGHT", Rank::EIGHT)
        .value("NINE", Rank::NINE).value("TEN", Rank::TEN).value("JACK", Rank::JACK)
        .value("QUEEN", Rank::QUEEN).value("KING", Rank::KING).value("ACE", Rank::ACE)
        .value("TWO", Rank::TWO).value("SMALL_JOKER", Rank::SMALL_JOKER).value("BIG_JOKER", Rank::BIG_JOKER);
    py::enum_<CombinationKind>(module, "CombinationKind")
        .value("SINGLE", CombinationKind::SINGLE).value("PAIR", CombinationKind::PAIR)
        .value("TRIPLE", CombinationKind::TRIPLE).value("FULL_HOUSE", CombinationKind::FULL_HOUSE)
        .value("STRAIGHT", CombinationKind::STRAIGHT).value("PAIR_SEQUENCE", CombinationKind::PAIR_SEQUENCE)
        .value("TRIPLE_SEQUENCE", CombinationKind::TRIPLE_SEQUENCE).value("RANK_BOMB", CombinationKind::RANK_BOMB)
        .value("STRAIGHT_FLUSH", CombinationKind::STRAIGHT_FLUSH).value("FOUR_KINGS", CombinationKind::FOUR_KINGS);

    py::class_<WildAssignment>(module, "WildAssignment")
        .def_readonly("card_id", &WildAssignment::card_id)
        .def_readonly("rank", &WildAssignment::rank)
        .def_readonly("suit", &WildAssignment::suit)
        .def_readonly("natural", &WildAssignment::natural);

    py::class_<Combination>(module, "Combination")
        .def_property_readonly("kind", [](const Combination& value) { return combination_kind_name(value.kind); })
        .def_readonly("card_ids", &Combination::card_ids)
        .def_property_readonly("level_rank", [](const Combination& value) { return static_cast<int>(value.level_rank); })
        .def_property_readonly("comparison_rank", [](const Combination& value) { return static_cast<int>(value.comparison_rank); })
        .def_readonly("comparison_key", &Combination::comparison_key)
        .def_readonly("wild_assignments", &Combination::wild_assignments)
        .def_readonly("natural_wild_ids", &Combination::natural_wild_ids)
        .def_property_readonly("is_bomb", &Combination::is_bomb)
        .def_property_readonly("size", &Combination::size)
        .def("to_dict", &combination_to_dict)
        .def("__repr__", [](const Combination& value) {
            return "Combination(kind='" + combination_kind_name(value.kind) + "', cards=" + local_vector_key(value.card_ids) + ")";
        });

    py::class_<core::ActionRow>(module, "V2Action")
        .def_property_readonly("row", [](const core::ActionRow& value) { return value.tokens; })
        .def_readonly("token_id", &core::ActionRow::token_id)
        .def_readonly("phase_token", &core::ActionRow::phase_token)
        .def_readonly("prefix_stage", &core::ActionRow::prefix_stage)
        .def_readonly("kind", &core::ActionRow::kind)
        .def_readonly("is_commit", &core::ActionRow::is_commit);

    py::class_<PyObservationBatch>(module, "ObservationBatch")
        .def_readonly("tokens", &PyObservationBatch::tokens)
        .def_readonly("channels", &PyObservationBatch::channels);
    py::class_<PyAvailableActions>(module, "AvailableActionBatch")
        .def_readonly("tokens", &PyAvailableActions::tokens)
        .def_readonly("weights", &PyAvailableActions::weights)
        .def_readonly("is_commit", &PyAvailableActions::is_commit);
    py::class_<PyCriticState>(module, "PrivilegedCriticBatch")
        .def_readonly("ownership", &PyCriticState::ownership)
        .def_readonly("escrow", &PyCriticState::escrow)
        .def_readonly("played", &PyCriticState::played);
    py::class_<PyEnvStep>(module, "EnvStep")
        .def_readonly("rewards", &PyEnvStep::rewards)
        .def_readonly("is_terminal", &PyEnvStep::is_terminal)
        .def_readonly("player_ids", &PyEnvStep::player_ids)
        .def_readonly("observations", &PyEnvStep::observations)
        .def_readonly("available_actions", &PyEnvStep::available_actions)
        .def_readonly("infos", &PyEnvStep::infos);

    py::class_<CppEnvBatch>(module, "EnvBatch")
        .def(py::init<int, const std::string&, py::object, bool, int, int, int>(),
             py::arg("batch_size"), py::arg("backend") = BACKEND, py::arg("configs") = py::none(),
             py::arg("auto_reset") = true, py::arg("max_horizon") = MAX_GAME_HORIZON,
             py::arg("action_capacity") = MAX_AVAILABLE_ACTIONS, py::arg("observation_capacity") = MAX_OBSERVATION_TOKENS)
        .def_property_readonly("batch_size", &CppEnvBatch::batch_size)
        .def_property_readonly("backend", &CppEnvBatch::backend)
        .def_property_readonly("auto_reset", &CppEnvBatch::auto_reset)
        .def_property_readonly("max_horizon", &CppEnvBatch::max_horizon)
        .def_property_readonly("action_capacity", &CppEnvBatch::action_capacity)
        .def_property_readonly("observation_capacity", &CppEnvBatch::observation_capacity)
        .def("reset", &CppEnvBatch::reset, py::arg("seed") = 0)
        .def("reset_from_deals", &CppEnvBatch::reset_from_deals, py::arg("deals"), py::arg("level_ranks") = py::none(), py::arg("leader_seats") = py::none(), py::arg("previous_results") = py::none())
        .def("observe", &CppEnvBatch::observe)
        .def("step", &CppEnvBatch::step)
        .def("critic_state", &CppEnvBatch::critic_state)
        .def("debug_state", &CppEnvBatch::debug_state)
        .def("serialize", &CppEnvBatch::serialize)
        .def_static("deserialize", &CppEnvBatch::deserialize);

    module.def("card_info", &card_info, py::arg("card_id"), py::arg("level_rank") = static_cast<int>(Rank::TWO));
    module.def("effective_rank_value", [](int rank, int level) { return effective_rank_value(static_cast<Rank>(rank), static_cast<Rank>(level)); });
    module.def("recognize_combinations", [](const std::vector<int>& cards, int level) { return recognize_combinations(cards, static_cast<Rank>(level)); }, py::arg("card_ids"), py::arg("level_rank") = static_cast<int>(Rank::TWO));
    module.def("compare_combinations", &compare_combinations);
    module.def("canonical_tokens", [](const Combination& combo) { return canonical_tokens(Action{0, combination_kind_name(combo.kind), combo.card_ids, combo}); });
    module.def("actions_from_np", [](const py::array& rows) { return actions_from_array(rows); });

    module.attr("num_players") = NUM_PLAYERS;
    module.attr("num_tokens") = NUM_TOKENS;
    module.attr("num_channels") = NUM_CHANNELS;
    module.attr("pad_token") = PAD_TOKEN;
    module.attr("max_game_horizon") = MAX_GAME_HORIZON;
    module.attr("max_available_actions") = MAX_AVAILABLE_ACTIONS;
    module.attr("max_observation_tokens") = MAX_OBSERVATION_TOKENS;
    module.attr("max_action_tokens") = MAX_ACTION_TOKENS;
    module.attr("NUM_PLAYERS") = NUM_PLAYERS;
    module.attr("NUM_TOKENS") = NUM_TOKENS;
    module.attr("NUM_CHANNELS") = NUM_CHANNELS;
    module.attr("PAD_TOKEN") = PAD_TOKEN;
    module.attr("MAX_GAME_HORIZON") = MAX_GAME_HORIZON;
    module.attr("MAX_AVAILABLE_ACTIONS") = MAX_AVAILABLE_ACTIONS;
    module.attr("MAX_OBSERVATION_TOKENS") = MAX_OBSERVATION_TOKENS;
    module.attr("MAX_ACTION_TOKENS") = MAX_ACTION_TOKENS;
    module.attr("ACTION_VERSION") = ACTION_VERSION;
    module.attr("ENV_VERSION") = ENV_VERSION;
    module.attr("ENCODING_VERSION") = ENCODING_VERSION;
    module.attr("SERIALIZATION_SCHEMA") = SERIALIZATION_SCHEMA;
    module.attr("BACKEND") = BACKEND;
    module.attr("GAME_IDENTIFIER") = IDENTIFIER;
    module.attr("identifier") = IDENTIFIER;
    module.attr("protocol_version") = std::string(ACTION_VERSION) + "|" + ENV_VERSION + "|" + ENCODING_VERSION;

    for (const auto& [name, value] : std::array<std::pair<const char*, int>, 25>{{
        {"BOS_TOKEN", BOS_TOKEN}, {"LEAD_PLAY_TOKEN", LEAD_PLAY_TOKEN}, {"FOLLOW_PLAY_TOKEN", FOLLOW_PLAY_TOKEN},
        {"TRIBUTE_TOKEN", TRIBUTE_TOKEN}, {"RETURN_TOKEN", RETURN_TOKEN}, {"PASS_TOKEN", PASS_TOKEN}, {"COMMIT_TOKEN", COMMIT_TOKEN},
        {"FAMILY_TOKEN_BASE", FAMILY_TOKEN_BASE}, {"RANK_TOKEN_BASE", RANK_TOKEN_BASE}, {"SUIT_TOKEN_BASE", SUIT_TOKEN_BASE},
        {"LENGTH_TOKEN_BASE", LENGTH_TOKEN_BASE}, {"CARD_TOKEN_BASE", CARD_TOKEN_BASE}, {"WILD_TOKEN_BASE", WILD_TOKEN_BASE},
        {"EVENT_RESET", EVENT_RESET}, {"EVENT_PRIVATE_PREFIX", EVENT_PRIVATE_PREFIX}, {"EVENT_PUBLIC_COMMIT", EVENT_PUBLIC_COMMIT},
        {"EVENT_PRIVATE_EXCHANGE_COMMIT", EVENT_PRIVATE_EXCHANGE_COMMIT}, {"EVENT_EXCHANGE_RESOLVED", EVENT_EXCHANGE_RESOLVED},
        {"ACTION_ROW_BEGIN", ACTION_ROW_BEGIN}, {"ACTION_STAGE_BASE", ACTION_STAGE_BASE}, {"ACTION_KIND_BASE", ACTION_KIND_BASE},
        {"ACTION_PAYLOAD", ACTION_PAYLOAD}, {"ACTION_NON_COMMIT", ACTION_NON_COMMIT}, {"ACTION_COMMIT", ACTION_COMMIT},
        {"ACTION_ROW_END", ACTION_ROW_END},
    }}) module.attr(name) = value;

    py::dict action_schema;
    action_schema["version"] = ACTION_VERSION;
    action_schema["row_length"] = MAX_ACTION_TOKENS;
    action_schema["row"] = py::make_tuple("ROW_BEGIN", "PHASE", "PREFIX_STAGE", "KIND", "PAYLOAD_MARKER", "V1_TOKEN_ID", "IS_COMMIT", "ROW_END");
    action_schema["execution"] = "The complete row must exactly match one current C++ legal-next row.";
    module.attr("ACTION_SCHEMA") = action_schema;
    py::dict env_schema;
    env_schema["version"] = ENV_VERSION;
    env_schema["backend"] = BACKEND;
    env_schema["capacity_policy"] = "hard_fail_no_truncation";
    env_schema["critic_privileged_api"] = "EnvBatch.critic_state (never attached to EnvStep)";
    module.attr("ENV_SCHEMA") = env_schema;
    py::dict encoding_schema;
    encoding_schema["version"] = ENCODING_VERSION;
    encoding_schema["history"] = "one incremental event row per environment token step";
    encoding_schema["privacy"] = "no other-seat hand identity and no unresolved other-seat exchange card";
    encoding_schema["channel_dtype"] = "uint8";
    module.attr("ENCODING_SCHEMA") = encoding_schema;
}
