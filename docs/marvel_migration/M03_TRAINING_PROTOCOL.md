# M03 MARVEL Local Training Protocol

- 日期：2026-10-08
- runtime schema：`guandan-marvel-m03-local-runtime-v1`
- checkpoint schema：`guandan-marvel-training-checkpoint-v2`
- semantic contract：`guandan-marvel-training-contract-v1`
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
