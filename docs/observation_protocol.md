# Observation protocol

This document describes the intended adapter shape. Concrete dimensions are fixed only after A00 rules decisions are accepted.

A batch step exposes NumPy-compatible arrays:

- `observations.tokens`: `[B, P, L_obs]`, integer token IDs per viewer.
- `observations.channels`: `[B, P, C]`, public/self state channels.
- `player_ids`: `[B]`, active player or negative chance/administrative ID.
- `available_actions.tokens`: `[B, A, L_act]`.
- `available_actions.weights`: `[B, A]`, legal-action prior/weight.
- `available_actions.is_commit`: `[B, A]`, whether selecting this action commits a complete move.
- `rewards`: `[B, P]`, team-aligned reward vector.
- `is_terminal`: `[B]`.

The policy view may include the active player's hand and all public information, but never opponent or teammate private cards. The central critic and training-only belief labels are separate from the policy observation.
