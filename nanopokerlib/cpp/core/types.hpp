#pragma once

// SPDX-License-Identifier: MIT
// Derived from nanopokerlib generic type concepts at
// D:/project/doudizhu/nanopokerlib/cpp/core/types/*
// Reference commit: acdcf199cf422e2b683cb22142b8e7cb33b14460.

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace nanopokerlib::core {

using ChannelValue = std::uint8_t;

inline constexpr int PAD_TOKEN = 0;
inline constexpr int PLAYER_CHANCE_TRAINER = -1;
inline constexpr int PLAYER_CHANCE_ENVIRONMENT = -2;

struct ActionRow {
    std::array<int, 8> tokens{};
    int token_id{};
    int phase_token{};
    int prefix_stage{};
    std::string kind;
    bool is_commit{};

    bool operator==(const ActionRow&) const = default;
};

template<std::size_t NumPlayers, std::size_t ObservationTokens, std::size_t Channels>
struct Observation {
    std::array<std::array<int, ObservationTokens>, NumPlayers> tokens{};
    std::array<std::array<ChannelValue, Channels>, NumPlayers> channels{};
};

template<std::size_t NumPlayers, std::size_t ObservationTokens, std::size_t Channels>
struct StepResult {
    std::array<float, NumPlayers> rewards{};
    bool is_terminal{};
    int player_id{-1};
    Observation<NumPlayers, ObservationTokens, Channels> observation{};
    std::vector<ActionRow> available_actions;
};

}  // namespace nanopokerlib::core
