# A05 training algorithms: fixed-seat IPPO and VRPO

## Scope and source

This implementation provides full backward GAE and a distinct Expected-SARSA
Q-boost estimator. It does **not** implement the full MARVEL architecture:
the existing actor is a mean-pooled MLP. Rollout collection, trainer scheduling,
and checkpointing are separate components owned by their respective workers.

Primary reference: Zhiyuan Fan and Gabriele Farina, *GAE Falls Short in
Imperfect-Information Self-Play Reinforcement Learning*, arXiv:2605.19235v1,
section 3.2, equations (3.2)-(3.3), submitted May 19, 2026.
Official source: https://arxiv.org/html/2605.19235v1#S3.SS2
The boundary masks below adapt those equations to batched, auto-reset,
finite-length token rollouts. No external implementation was reused.

## Public APIs

```python
compute_gae(rewards, values, next_values, terminated, truncated,
            *, gamma=1.0, gae_lambda=0.95) -> (advantages, returns)
compute_q_boost(rewards, selected_q, values, next_values, terminated, truncated,
                *, gamma=1.0, gae_lambda=0.95) -> (advantages, q_targets)
masked_expected_values(q_values, probabilities, legal_mask) -> values
acting_player_values(values, player_ids) -> selected_values
expand_team_values(actor_values, player_ids) -> fixed_seat_values

CentralQCritic(hidden_dim=32)
CentralQCritic.forward(critic_state, legal_tokens) -> q_values

ippo_loss(logits, new_values, actions, old_log_probs, advantages, returns,
          legal_tokens, *, clip=0.2, entropy_coef=0.01, value_coef=0.5)
vrpo_loss(logits, q_selected, actions, old_log_probs, advantages, q_targets,
          legal_tokens, *, clip=0.2, entropy_coef=0.01, value_coef=0.5)
# Both losses return (total, metrics: dict[str, Tensor]).
```

Import these directly from `guandan.training.estimators`,
`guandan.training.critic`, and `guandan.training.objectives`.

## Fixed-player coordinates and IPPO baseline

Rollouts use `[T,B,4]` floating tensors for rewards, values, explicit successor
values, advantages, and targets. The seat axis is always `[0,1,2,3]`, not
whichever player happens to be acting on that row. `terminated` and `truncated`
are boolean `[T,B]`. Float inputs to an estimator share dtype and device.
Only after finishing the four fixed-seat traces should the learner gather
`acting_player_values(advantages, player_ids)` and the corresponding targets.
Gathering before the recurrence mixes opponents' values when the actor changes.

`acting_player_values` accepts `values[...,4]` and integer `player_ids[...]`;
`expand_team_values` accepts scalar `actor_values[...]` and matching IDs.
IDs are 0 through 3. Both helpers retain autograd when used on live predictions.

The shared actor observes only the current player's policy projection and
predicts that player's scalar V. `expand_team_values` converts this into fixed
coordinates: an actor from team 0 (seat 0/2) with prediction v produces
`[v,-v,v,-v]`; an actor from team 1 (seat 1/3) produces `[-v,v,-v,v]`.
Perform this conversion separately for the current and successor actor IDs.

This is an **approximate decentralized baseline**, not four independently
observed values. Different actors' observations may imply different baselines
on successive rows. Their numerical predictions become detached training
labels; opponents' raw observations or hidden hands are never concatenated
into the actor input. Future rewards/labels may inform learning without
changing the actor's permitted inference-time information.

## IPPO / full GAE

For every fixed seat i independently:

```
delta[t,i] = r[t,i] + gamma * (not terminated[t]) * next_V[t,i] - V[t,i]
trace[t,i] = delta[t,i] + gamma * lambda
             * (not (terminated[t] or truncated[t])) * trace[t+1,i]
advantages[t,i] = trace[t,i]
returns[t,i] = V[t,i] + trace[t,i]
```

The reverse recurrence initializes the uncollected carry to zero. Defaults are
`gamma=1.0` and `gae_lambda=0.95`; both must lie in `[0,1]`. Empty time axes
are supported. Results retain dtype/device but are detached training labels.
No value bootstraps are inferred by shifting the next row in the buffer.

| Transition | Bootstrap | Continue trace into next collected row? |
| --- | --- | --- |
| Ordinary live token transition | Actual successor V | Yes |
| True terminal | Zero | No |
| Truncation | Pre-reset final-state V | No |
| Ordinary rollout horizon, not terminal/truncated | Actual successor V | No uncollected carry |
| Both terminated and truncated | Zero; termination wins | No |

A trace *before* a truncation still includes the truncated transition's
residual; it must not include a new episode's residual. Token-prefix extension
is an ordinary transition, even if it has zero reward and the actor does not
change. A committed game step is not itself a terminal boundary.

**Collector contract:** capture final-state successor inputs before resetting.
For IPPO, evaluate the scalar baseline on that final policy observation and
expand using its actor ID. A reset state's value is not a valid substitute.
The estimator cannot reconstruct missing final-state inputs. True-terminal
next-values are masked before arithmetic, so unused NaN placeholders there
are harmless; every nonterminal next-value, including truncations, must be
finite. Invalid current predictions/rewards are rejected.

## VRPO / centralized action critic and Q-boost

`CentralQCritic` is a separate training-only module with its own parameters.
It accepts `critic_state[B,688]`: four absolute-seat, 108-card ownership
one-hot vectors (`432` channels) followed by `256` policy channels. It also
accepts `legal_tokens[B,A]`, integer token IDs in `[0,255]`. Its candidate-
conditioned team-zero scalar is expanded as `[q,-q,q,-q]`, returning `[B,A,4]`.
The head scores actions, not just states. Padded candidate token IDs must
remain in the vocabulary; their Q predictions are ignored using the separate
legal mask. No full-state tensor is accepted by a loss or sent to the actor.
The fixed zero-sum team structure is a GuanDan-specific model constraint.

For the current active player's policy, not an average of all four policies:

```
V_i(s) = sum_a pi(a | o_active) * Q_i(s,a)
```

`masked_expected_values` accepts Q `[B,A,4]`, probabilities `[B,A]`, and a
boolean legal mask `[B,A]`, returning `[B,4]`. Invalid positions are zeroed
before multiplication, including nonfinite padding, then legal probabilities
are renormalized. Already masked softmax probabilities therefore reproduce
the expectation up to roundoff. Every row must have positive finite legal
mass. Legal Q/probabilities must be finite, and probabilities nonnegative;
invalid slots have exactly zero mass and gradient. No all-invalid terminal
policy evaluation is needed: collectors can supply zero terminal successor V.

Q-boost keeps a fixed seat i through every subsequent player's action:

```
delta_plus[t,i] = r[t,i] + gamma * (not terminated[t]) * next_V[t,i]
                  - selected_Q[t,i]
trace[t,i] = delta_plus[t,i] + gamma * lambda
             * (not (terminated[t] or truncated[t])) * trace[t+1,i]
advantages[t,i] = selected_Q[t,i] - V[t,i] + trace[t,i]
q_targets[t,i] = selected_Q[t,i] + trace[t,i]
```

The same boundary table applies. For a truncation, both the successor policy
probabilities and successor Qs must come from the final state before reset.
Use a rollout-policy/critic snapshot consistently; build and detach labels
before optimization epochs, rather than recomputing old labels after updates.

### Exact-Q matching-pennies check

A deterministic two-stage matching-pennies game is enumerated over all four
H/T action pairs. The second actor cannot observe the first commitment; a
centralized Q critic can. Under uniform policies, first-stage Q and both
state values are zero. At the second stage, selected Q equals the terminal
payoff vector `[z,-z,z,-z]`, with `z` in `{-1,+1}`. Consequently:

* GAE carries the sampled future residual backward: its first-stage advantage
  is `lambda * [z,-z,z,-z]` (gamma=1), with per-seat variance `lambda**2`.
* Q-boost's Expected-SARSA residuals are zero, so its first-stage advantage
  and variance are zero, while its second-stage action advantage is preserved.
* Its critic targets equal the exact selected Qs.

This is an actual difference from GAE, not a renamed PPO target. The test
covers lambda 0.25, 0.95, and 1.0 and acting-seat/team switches. This pathwise
cancellation claim concerns the deterministic-transition toy example, not
arbitrary chance-transition games or a guarantee for an approximate critic.

## PPO objectives and regression targets

`logits[N,A]` are already masked: legal entries finite, invalid entries
`-inf`. Because these APIs take no separate legal mask, callers must apply it
before calling the loss; a finite negative sentinel is still considered legal.
`actions[N]` are actual token IDs, not candidate indices. `legal_tokens[N,A]`
maps IDs to candidates. Each selected token must identify exactly one legal
position. Duplicate padding IDs at invalid positions do not match actions.

Predictions, old log probabilities, advantages, and targets are floating `[N]`
acting-seat scalars on the logits' dtype/device. Both objectives implement:

```
ratio = exp(new_log_prob - stop_gradient(old_log_prob))
policy_loss = -mean(min(ratio * stop_gradient(advantage),
                       clip(ratio, 1-clip, 1+clip) * stop_gradient(advantage)))
value_loss = mean((prediction - stop_gradient(target))**2)
total = policy_loss + value_coef * value_loss - entropy_coef * mean(entropy)
```

There is no implicit advantage normalization or value clipping. IPPO fits
`new_values` to GAE `returns`. VRPO fits the **selected action's Q** to
`q_targets`, not to the expected V. Q expectations are used to construct
rollout labels; detached labels do not create an actor-to-centralized-critic
gradient path. The unused actor value head is not trained by VRPO's Q loss.

Metrics are detached scalar tensors: `loss`, `policy_loss`, `value_loss`,
`entropy`, `approx_kl`, and `clip_fraction`. VRPO additionally reports `q_loss`
(equal to `value_loss`). Entropy masks log-probabilities before multiplication
so invalid slots never produce `0 * -inf`. Nonfinite losses/diagnostics are
rejected. Focused tests explicitly assert finite forward values and gradients
and zero invalid-slot gradients.

## Focused verification

The two new test modules cover independent Python forward-sum scalar
references for both estimators, hand-computed all-four-seat cases, asynchronous
batch resets, all four actor IDs, true terminal versus truncation versus
rollout horizon, no input mutation, label detachment, probability masking,
matching-pennies variance reduction, clipped PPO against a scalar reference,
selected-Q rather than V regression, critic action dependence and team
antisymmetry, invalid contracts, and actor/critic parameter isolation.
CPU and CUDA (when available) estimator/critic checks are included.

Use the specified interpreter and disable bytecode/pytest cache writes:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:CUDA_CACHE_DISABLE='1'
& 'C:\Users\yhx\.conda\envs\guandan_train\python.exe' -B -m pytest `
  -p no:cacheprovider tests/unit/test_estimators.py tests/unit/test_training_objectives.py -q
```

These tests validate algorithm components, not full A05 acceptance, resumed
training, performance against opponents, or full MARVEL reproduction.
