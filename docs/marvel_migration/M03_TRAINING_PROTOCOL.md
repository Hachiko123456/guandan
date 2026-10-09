# M03 MARVEL Local Training Protocol

- 日期：2026-10-09
- runtime schema：`guandan-marvel-m03-local-runtime-v2`
- checkpoint schema：`guandan-marvel-training-checkpoint-v3`
- semantic contract：`guandan-marvel-training-contract-v2`
- trajectory schema：`guandan-marvel-complete-trajectory-v1`
- sampler：`guandan-marvel-sampler-contract-v2`
- buffer：`guandan-marvel-buffer-v2`
- VRPO Q target：`guandan-marvel-vrpo-q-lambda-td1-v1`

## Scope

M03 connects the accepted Python reference adapter to small local IPPO and VRPO semantic smoke runs. It does not claim throughput, strength, GPU saturation, selection/champion behavior, C++ readiness, or Kaggle verification.

The accepted local profile uses an explicit smoke-only game identifier with `T=512`. Every consumed trajectory must terminate naturally before 512. The default V2 protocol remains `T=4096`.

## Pre-allocation gate

Before constructing a sampler or replay buffer the local trainer requires:

- `max_sampler_bytes_per_rank`;
- `trajectory_memory_budget()` accounting;
- `require_trajectory_memory_budget()` fail-closed enforcement.

The smoke profile is `rollout=1`, `collect=1`, `replay_buffer_scale=1`, `mini_batches=1`. Dense sampler estimate is about 32.681 MiB and one replay trajectory about 10.894 MiB.

## On-policy contract

Fresh IPPO/VRPO trajectories use:

```text
purpose=on_policy
sampling_temperature=1.0
forced_action_indices=None
```

PPO old action probability is `behavior_action_prob`; reference KL uses `policy_probs`; every actor step must have `on_policy_valid=true`.

## Complete trajectory validation

Before learning or replay insertion:

- required versioned fields must exist;
- critic-only ownership fields must not appear in actor tensors;
- `num_steps` must be within the configured horizon;
- exactly one terminal marker must occur at `num_steps-1`;
- the selected action must have positive availability weight;
- probability, reward, value, advantage, Q target, loss, and gradient tensors must be finite.

Invalid or interrupted batches do not increment `durable_updates`. A failed trainer is poisoned and cannot checkpoint or silently retry.

## IPPO

- `Agent(enable_value_head=True)`;
- multi-player value GAE;
- configurable multiple policy epochs and trajectory mini-batches;
- clipped policy objective;
- uniform KL regularization plus reference KL metric;
- value Huber loss;
- gradient finite check and clipping;
- iteration schedule and AdamW/Muon optimizer switch.

## VRPO

- actor without value head;
- four-information-set centralized Q critic;
- fresh-rollout Q-based GAE for actor updates;
- complete-trajectory replay only;
- first critic auxiliary epoch uses fresh rollout;
- later auxiliary epochs sample replay;
- lambda-return and TD(1) Q losses;
- independently scheduled actor and critic optimizers.

Actor rollout tensors never contain `ownership`, `escrow`, `played`, or `critic_state`.

## Resume and merged policy identity

Checkpoint identities are explicit:

- `agent_resume_state_dict`: optimizer continuation actor;
- `agent_state_dict`: merged evaluation copy;
- `critic_state_dict`: VRPO critic;
- optimizer, schedule, counters, sampler state, RNG, config/source/semantic hashes;
- VRPO replay state when exact resume is requested.

When `agent_merging_beta=0`, merged identity is `current_policy_copy`, not EMA smoothing.

Resume classes:

- `exact_resume`: semantic/config/source identity and all necessary state match;
- `compatible_resume`: semantic contract matches but source/config identity differs;
- `warm_restart`: optimizer/sampler/replay or other continuation state is missing;
- `migration`: semantic contract differs;
- `evaluation_only`: explicit inference load.

Legacy V1 checkpoints are rejected for V2 training resume.

## Initial smoke defaults

```text
CPU
history_dim=16
feature_dim=24
decoder_layers=1
decoder_heads=4
state_layers=1
action_layers=1
fuse_layers=1
agent_epoch=2
critic_aux_epoch=2
AdamW
Muon=false
compile=false
AMP=false
```

These values are correctness-only and are not the MARVEL 128/256 formal baseline.
## Interruption and lifecycle contract (repair)

Trainer lifecycle is explicit: `idle -> running -> completed`, or `running/loading -> failed`. A durable checkpoint is permitted only when:

- no iteration/load is active;
- trainer and sampler are healthy;
- `iteration == durable_updates`;
- `learner_updates == durable_learner_updates`;
- at least one durable iteration exists.

`RuntimeError`, `KeyboardInterrupt`, and other catchable `BaseException` values poison both trainer and sampler and are re-raised. Partial optimizer work remains diagnostic only and cannot be saved or resumed. Recovery requires a new instance loading the last complete checkpoint. Platform hard-kill/power-loss outside Python exception handling is not claimed recoverable.

## Semantic configuration v1

The v2 semantic contract embeds a hashed `guandan-marvel-semantic-config-v1` covering all consumed training semantics:

- complete model structure, including decoder heads/layers and state/action/fuse dimensions;
- complete game tensor/protocol specification;
- gamma, IPPO/VRPO GAE, epochs, clip/reg/value coefficients and learning rates;
- optimizer type/betas/weight decay/gradient clipping;
- schedule powers/stable iterations;
- merged policy beta/definition;
- seed, rollout/collect/replay/minibatch and replay persistence;
- deterministic/TF32, AMP, compile and DDP execution semantics;
- sampler/buffer/Q-target contracts.

Only explicit non-semantic fields may produce `compatible_resume`: project/run name, logging directory, debug/log intervals, total requested iterations and resume path. Gamma, model heads, optimizer, schedule, sampling, replay, source, device or dependency changes are migration/rejection, not compatible resume.

Declared hashes for resolved config, semantic config, semantic contract, source fingerprint and execution fingerprint are recomputed during classification. Errors identify concrete differing paths.

## Source and execution fingerprint boundary

`guandan-marvel-source-fingerprint-v1` stores component/file-level hashes for:

- A01/A02/A03/A04 rule and environment code;
- V2 game adapter and buffer;
- model/Transformer/cache/tensordict code;
- optimizer/Muon code;
- sampler/compile/logger/config utilities;
- GAE/loss/schedule/checkpoint/trainer/validation/metrics code.

`exact_resume` additionally requires the same Python, PyTorch, NumPy, device type and CUDA availability fingerprint. No unverified bitwise exact guarantee is made across devices or dependency versions. Source or execution changes conservatively classify as migration unless a future version adds a specifically verified compatibility rule.

## Pre-repair checkpoint boundary

`guandan-marvel-training-checkpoint-v2` remains readable for artifact inspection/evaluation classification but lacks semantic-config/source-fingerprint v1 and cannot be exact/compatible training resume under the repaired contract. New acceptance checkpoints use v3.
