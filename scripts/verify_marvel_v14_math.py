"""Numerical verification of the MARVEL v1.4 plan's mathematical claims.

Checks:
  A. Pure-math claims (GAE decay table, Amdahl, KL identity, Huber scaling,
     entropy normalization bounds, product-ratio clip strictness).
  B. Real code (guandan_marvel.training.gae / losses) vs the plan's
     commit-boundary macro GAE and macro-sequence PPO on synthetic rollouts:
       B1. all-commit protocol env: macro GAE == token GAE exactly
       B2. all-commit protocol env: macro PPO loss == token PPO loss
       B3. mixed macro lengths: quantify token-vs-macro advantage divergence
       B4. macro logprob == sum of token logprobs (< 1e-6)
       B5. chosen_is_commit == gather(avail_is_commit, action_taken)
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, "D:/project/guandan")

import torch
import torch.nn as nn

from guandan_marvel.model.tensordict import TensorDict  # noqa: E402
from guandan_marvel.training.gae import compute_gae_from_value  # noqa: E402
from guandan_marvel.training.losses import compute_clipped_pg_loss  # noqa: E402

P = 4
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# A. pure math
# ---------------------------------------------------------------------------
print("== A. pure-math claims ==")
lam = 0.95
table = {20: 0.358, 50: 0.0769, 100: 0.00592, 200: 0.000035}
for k, claimed in table.items():
    actual = lam ** k
    check(f"0.95^{k} decay table", abs(actual - claimed) / claimed < 0.01,
          f"claimed={claimed} actual={actual:.3e}")

check("Amdahl S_max = 1/0.44", abs(1 / 0.44 - 2.27) < 0.01, f"{1/0.44:.4f}")
check("Amdahl S(learner 2x) = 1/(0.44+0.28)", abs(1 / (0.44 + 0.56 / 2) - 1.389) < 0.005,
      f"{1/(0.44+0.56/2):.4f}")
# robustness of the Amdahl conclusion to the actual measured split (51.7/48.3 smoke record)
s_max_actual = 1 / 0.517
s_actual = 1 / (0.517 + 0.483 / 2)
check("Amdahl conclusion robust to measured 51.7/48.3 split",
      1.3 < s_actual < 1.45 and 1.9 < s_max_actual < 2.3,
      f"S_max={s_max_actual:.3f} S={s_actual:.3f}")

rng = np.random.default_rng(7)
ok = True
for _ in range(200):
    n = int(rng.integers(2, 40))
    p = rng.dirichlet(np.ones(n))
    H = -(p * np.log(p)).sum()
    U = np.full(n, 1.0 / n)
    kl = (p * (np.log(p) - np.log(U))).sum()
    if abs(kl - (math.log(n) - H)) > 1e-10:
        ok = False
check("D_KL(pi||U) == log(n) - H(pi) over 200 random distributions", ok)

# Huber quadratic-regime scaling: targets +/-3 vs +/-1, predictions at 0
for delta in (10.0,):
    e1 = torch.tensor([1.0, -1.0])
    e3 = 3 * e1
    l1 = torch.nn.functional.huber_loss(torch.zeros_like(e1), e1, reduction="mean", delta=delta)
    l3 = torch.nn.functional.huber_loss(torch.zeros_like(e3), e3, reduction="mean", delta=delta)
    check(f"value loss ~9x when targets scale 3x (Huber delta={delta})",
          abs((l3 / l1).item() - 9.0) < 1e-6, f"ratio={(l3/l1).item():.6f}")

# entropy normalization bounds
n = 12
u = np.full(n, 1.0 / n)
check("normalized entropy of uniform == 1", abs((-(u * np.log(u)).sum()) / math.log(n) - 1.0) < 1e-12)
det = np.eye(1, n, 0)[0]
with np.errstate(divide="ignore", invalid="ignore"):
    det_entropy = -np.nansum(np.where(det > 0, det * np.log(det), 0.0))
check("normalized entropy of deterministic == 0", abs(det_entropy / math.log(n)) < 1e-12)
check("single legal action entropy term == 0 (log 1 = 0)", math.log(1) == 0.0)

check("max within-action dilution 0.95^8 (max_action_tokens=8)",
      abs(lam ** 8 - 0.663) < 0.001, f"{lam**8:.4f}")
check("per-player macro horizon ~36 -> 0.95^36",
      abs(lam ** 36 - 0.157) < 0.005, f"{lam**36:.4f}")
check("product-ratio clip strictness: 1.1^8 for L=8 macro",
      abs(1.1 ** 8 - 2.138) < 0.01, f"{1.1**8:.3f}")

# ---------------------------------------------------------------------------
# B. synthetic rollouts + real code
# ---------------------------------------------------------------------------
print("\n== B. synthetic rollout verification against real code ==")


def build_rollout(macro_specs, *, winner_team, gamma_rewards=True, seed=0):
    """macro_specs: list of (player, length); lengths>=1, last step of each is a commit.

    Returns dict of tensors shaped for the MARVEL rollout contract.
    """
    rng = np.random.default_rng(seed)
    A = 10  # action capacity
    steps = []  # (player, is_commit)
    for (p, L) in macro_specs:
        for i in range(L):
            steps.append((p, i == L - 1))
    T = len(steps)
    n = T
    player_ids = np.array([s[0] for s in steps], dtype=np.int64)
    chosen_is_commit = np.array([s[1] for s in steps], dtype=bool)

    avail_weights = np.zeros((T, A), dtype=np.float32)
    avail_is_commit = np.zeros((T, A), dtype=bool)
    policy_probs = np.zeros((T, A), dtype=np.float32)
    action_taken = np.zeros(T, dtype=np.int64)
    behavior_action_prob = np.zeros(T, dtype=np.float32)
    for t in range(T):
        n_legal = int(rng.integers(2, A))
        legal = rng.choice(A, size=n_legal, replace=False)
        avail_weights[t, legal] = 1.0
        # commit flags on random other actions to make the gather non-trivial
        commit_flags = rng.random(A) < 0.25
        commit_flags[~np.isin(np.arange(A), legal)] = False
        avail_is_commit[t] = commit_flags
        probs = rng.random(n_legal)
        probs = probs / probs.sum()
        policy_probs[t, legal] = probs
        take = int(rng.choice(legal))
        action_taken[t] = take
        behavior_action_prob[t] = policy_probs[t, take]
        avail_is_commit[t, take] = chosen_is_commit[t]  # taken action's flag is authoritative
    rewards = np.zeros((T, P), dtype=np.float32)
    if T > 0:
        rewards[T - 1] = np.array([1.0 if (q % 2) == winner_team else -1.0 for q in range(P)],
                                  dtype=np.float32)
    is_terminal = np.zeros(T, dtype=bool)
    is_terminal[T - 1] = True
    state_values = rng.random((T, P)).astype(np.float32) * 0.2 - 0.1
    # zero the values of non-acting players exactly like the real agent does
    for t in range(T):
        for q in range(P):
            if q != player_ids[t]:
                state_values[t, q] = 0.0
    return dict(
        num_steps=np.array([n], dtype=np.int64),
        player_ids=player_ids[None],
        action_taken=action_taken[None],
        avail_weights=avail_weights[None],
        avail_is_commit=avail_is_commit[None],
        policy_probs=policy_probs[None],
        behavior_action_prob=behavior_action_prob[None],
        on_policy_valid=np.ones((1, T), dtype=bool),
        rewards=rewards[None],
        is_terminal=is_terminal[None],
        state_values=state_values[None],
    )


def to_tensordict(roll):
    return TensorDict({
        "num_steps": torch.as_tensor(roll["num_steps"]),
        "player_ids": torch.as_tensor(roll["player_ids"]),
        "action_taken": torch.as_tensor(roll["action_taken"]),
        "avail_weights": torch.as_tensor(roll["avail_weights"]),
        "avail_is_commit": torch.as_tensor(roll["avail_is_commit"]),
        "policy_probs": torch.as_tensor(roll["policy_probs"]),
        "behavior_action_prob": torch.as_tensor(roll["behavior_action_prob"]),
        "on_policy_valid": torch.as_tensor(roll["on_policy_valid"]),
        "rewards": torch.as_tensor(roll["rewards"]),
        "is_terminal": torch.as_tensor(roll["is_terminal"]),
    }, batch_size=[1])


def segment_macros(player_ids, chosen_is_commit, n):
    """Return list of (start, end_inclusive, player); incomplete tails excluded."""
    segs = []
    start = 0
    for t in range(1, n):
        if chosen_is_commit[t - 1]:
            segs.append((start, t - 1, int(player_ids[start])))
            start = t
    if n > 0 and chosen_is_commit[n - 1]:
        segs.append((start, n - 1, int(player_ids[start])))
    return segs


def macro_gae_reference(roll, *, gamma=1.0, lam=0.95, lam_per_token=False):
    """The plan's commit-boundary GAE, per player over macro actions.

    lam_per_token=False -> plan's formula:      A_m = d_m + gamma^tau * lam * A_{m+1}
    lam_per_token=True  -> token-faithful form: A_m = d_m + (gamma*lam)^tau * A_{m+1}
    """
    player_ids = roll["player_ids"][0]
    rewards = roll["rewards"][0]
    values = roll["state_values"][0]
    n = int(roll["num_steps"][0])
    chosen_is_commit = roll["avail_is_commit"][0][np.arange(n), roll["action_taken"][0]]
    segs = segment_macros(player_ids, chosen_is_commit, n)
    adv = np.zeros((n, P), dtype=np.float64)
    per_player = {}
    for (s, e, p) in segs:
        per_player.setdefault(p, []).append((s, e))
    for p, macros in per_player.items():
        M = len(macros)
        A = np.zeros(M)
        for m in range(M - 1, -1, -1):
            s, e = macros[m]
            s_next = macros[m + 1][0] if m + 1 < M else n
            tau = s_next - s
            R = sum(gamma ** (k - s) * rewards[k, p] for k in range(s, s_next))
            decay = (gamma * lam) ** tau if lam_per_token else gamma ** tau * lam
            if m + 1 < M:
                delta = R + gamma ** tau * values[s_next, p] - values[s, p]
                A[m] = delta + decay * A[m + 1]
            else:
                delta = R - values[s, p]  # terminal: no bootstrap
                A[m] = delta
        for m, (s, e) in enumerate(macros):
            adv[s:e + 1, p] = A[m]
    return adv, segs, chosen_is_commit


def macro_pg_loss(roll, adv_macro, *, clip_coef):
    """Sequence-level PPO over macro actions (the plan's L_PPO^macro)."""
    player_ids = roll["player_ids"][0]
    n = int(roll["num_steps"][0])
    probs = roll["policy_probs"][0]
    taken = roll["action_taken"][0]
    behav = roll["behavior_action_prob"][0]
    chosen_is_commit = roll["avail_is_commit"][0][np.arange(n), taken]
    segs = segment_macros(player_ids, chosen_is_commit, n)
    losses = []
    for (s, e, p) in segs:
        logp_new = sum(math.log(probs[t, taken[t]]) for t in range(s, e + 1))
        logp_old = sum(math.log(behav[t]) for t in range(s, e + 1))
        r = math.exp(logp_new - logp_old)
        A = adv_macro[s, p]
        losses.append(max(-A * r, -A * min(max(r, 1 - clip_coef), 1 + clip_coef)))
    return float(np.mean(losses)), segs


# ---- B5: chosen_is_commit gather identity --------------------------------
roll = build_rollout([(0, 1), (1, 3), (2, 1), (3, 5), (0, 2), (1, 1)], winner_team=0, seed=1)
n = int(roll["num_steps"][0])
gathered = roll["avail_is_commit"][0][np.arange(n), roll["action_taken"][0]]
expected = np.array([i == L - 1 for (p, L) in [(0, 1), (1, 3), (2, 1), (3, 5), (0, 2), (1, 1)]
                     for i in range(L)])
check("B5 chosen_is_commit = gather(avail_is_commit, action_taken) marks exactly COMMIT/PASS steps",
      bool((gathered == expected).all()))

# ---- B1: all-commit env, macro GAE == token GAE ---------------------------
specs_all_commit = [(i % 4, 1) for i in range(60)]
roll = build_rollout(specs_all_commit, winner_team=1, seed=2)
n = int(roll["num_steps"][0])
td = to_tensordict(roll)
state_values = torch.as_tensor(roll["state_values"])
out = compute_gae_from_value(td, state_values, discount_gamma=1.0, gae_lambda=0.95)
token_adv = out.player_advantage[0].numpy()  # [T] acting player's advantage

macro_adv, segs, _ = macro_gae_reference(roll, gamma=1.0, lam=0.95)
macro_at_steps = np.array([macro_adv[t, int(roll["player_ids"][0][t])] for t in range(n)])
diff = np.abs(token_adv - macro_at_steps).max()
# KNOWN FINDING (not a code bug): the plan's written formula gamma^tau*lam is NOT
# equivalent to the token GAE even in a length-1 protocol env (4-player alternation
# gives tau~4, so per-gap decay is lam=0.95 vs lam^tau=0.8145). Recorded in
# docs/marvel_v14_plan_verification.md section 3.2. Equivalence requires (gamma*lam)^tau.
print(f"[FINDING] B1a plan formula gamma^tau*lam vs token GAE: max|diff|={diff:.3e} "
      f"(expected ~0.49; equivalence requires (gamma*lam)^tau, see B1b)")

macro_adv_tok, _, _ = macro_gae_reference(roll, gamma=1.0, lam=0.95, lam_per_token=True)
macro_tok_at_steps = np.array([macro_adv_tok[t, int(roll["player_ids"][0][t])] for t in range(n)])
diff2 = np.abs(token_adv - macro_tok_at_steps).max()
check("B1b all-commit env: token-faithful macro GAE ((g*lam)^tau per gap) == token GAE",
      diff2 < 1e-5, f"max|diff|={diff2:.3e}")
# quantify the plan-formula vs token-faithful divergence (lam vs lam^tau per gap, tau~4)
print(f"     plan formula vs token-faithful: max|diff|={np.abs(macro_at_steps - macro_tok_at_steps).max():.4f} "
      f"(per-gap factor 0.95 vs 0.95^4={0.95**4:.4f})")

# ---- B2: all-commit env, macro PPO == token PPO ---------------------------
# Isolate the PPO machinery: feed the SAME (token) advantages as macro advantages,
# which is what holds when the macro GAE reduces to the token GAE (length-1 env).
adv_mean = token_adv.mean()
adv_std = token_adv.std()
pg = compute_clipped_pg_loss(
    td, clip_coef=0.02,
    avail_probs_new=torch.as_tensor(roll["policy_probs"]),
    player_advantage=torch.as_tensor(token_adv[None].astype(np.float32)),
)
macro_adv_equiv = np.zeros_like(macro_adv)
for t in range(n):
    macro_adv_equiv[t, int(roll["player_ids"][0][t])] = token_adv[t]
macro_adv_norm = (macro_adv_equiv - adv_mean) / (adv_std + 1e-8)
macro_loss, _ = macro_pg_loss(roll, macro_adv_norm, clip_coef=0.02)
check("B2 all-commit env (L=1 macros): macro-sequence PPO loss == token PPO loss",
      abs(pg.pg_loss_mean.item() - macro_loss) < 2e-6,
      f"token={pg.pg_loss_mean.item():.8f} macro={macro_loss:.8f}")

# ---- C: degenerate equivalence tau_m == 1 (single-player consecutive actions) ----
# The ONLY condition under which the plan's gamma^tau*lam equals the token GAE:
# the same player acts at consecutive token steps, so tau_m = 1 for every macro.
specs_tau1 = [(0, 1) for _ in range(60)]
roll_c = build_rollout(specs_tau1, winner_team=1, seed=4)
n_c = int(roll_c["num_steps"][0])
td_c = to_tensordict(roll_c)
out_c = compute_gae_from_value(td_c, torch.as_tensor(roll_c["state_values"]),
                               discount_gamma=1.0, gae_lambda=0.95)
token_adv_c = out_c.player_advantage[0].numpy()
at_c = lambda adv: np.array([adv[t, int(roll_c["player_ids"][0][t])] for t in range(n_c)])
macro_plan_c, _, _ = macro_gae_reference(roll_c, gamma=1.0, lam=0.95)
macro_tok_c, _, _ = macro_gae_reference(roll_c, gamma=1.0, lam=0.95, lam_per_token=True)
check("C tau=1 env: plan formula gamma^tau*lam == token GAE (degenerate equivalence)",
      np.abs(token_adv_c - at_c(macro_plan_c)).max() < 1e-5,
      f"max|diff|={np.abs(token_adv_c - at_c(macro_plan_c)).max():.3e}")
check("C tau=1 env: token-faithful (g*lam)^tau == token GAE",
      np.abs(token_adv_c - at_c(macro_tok_c)).max() < 1e-5,
      f"max|diff|={np.abs(token_adv_c - at_c(macro_tok_c)).max():.3e}")

# ---- E: endpoint degeneracies lambda in {0, 1} --------------------------------
# At lambda=0 both clock formulas collapse to A=delta; at lambda=1 both collapse
# to A=delta+gamma^tau*A_next. Two checks:
#   E1 (all-commit env): both macro formulas == real token GAE at the endpoints
#       (length-1 macros make the two recursions coincide, per B1b).
#   E2 (mixed-length env): plan formula == token-faithful formula at the endpoints
#       (the two clocks degenerate to the same form regardless of tau).
# NOTE: on mixed-length envs NO macro formula equals the token GAE at any lambda,
# because the token timeline has extra intra-macro decision points (each token of
# a macro is a p-acting step that resets the GAE buffers). That is by design.
roll_e = build_rollout([(i % 4, int(1 + (i * 3) % 8)) for i in range(40)], winner_team=0, seed=5)
n_e = int(roll_e["num_steps"][0])
td_e = to_tensordict(roll_e)
for lam_ep in (0.0, 1.0):
    out_ep = compute_gae_from_value(td, torch.as_tensor(roll["state_values"]),
                                    discount_gamma=1.0, gae_lambda=lam_ep)
    ta_ep = out_ep.player_advantage[0].numpy()
    mp_ep, _, _ = macro_gae_reference(roll, gamma=1.0, lam=lam_ep)
    mt_ep, _, _ = macro_gae_reference(roll, gamma=1.0, lam=lam_ep, lam_per_token=True)
    at_p = np.array([mp_ep[t, int(roll["player_ids"][0][t])] for t in range(n)])
    at_t = np.array([mt_ep[t, int(roll["player_ids"][0][t])] for t in range(n)])
    check(f"E1 lambda={lam_ep} all-commit env: plan formula == token GAE",
          np.abs(ta_ep - at_p).max() < 1e-5, f"max|diff|={np.abs(ta_ep - at_p).max():.3e}")
    check(f"E1 lambda={lam_ep} all-commit env: token-faithful == token GAE",
          np.abs(ta_ep - at_t).max() < 1e-5, f"max|diff|={np.abs(ta_ep - at_t).max():.3e}")
    mp_e, _, _ = macro_gae_reference(roll_e, gamma=1.0, lam=lam_ep)
    mt_e, _, _ = macro_gae_reference(roll_e, gamma=1.0, lam=lam_ep, lam_per_token=True)
    d_pe = np.abs(mp_e - mt_e).max()
    check(f"E2 lambda={lam_ep} mixed env: plan formula == token-faithful (clocks collapse)",
          d_pe < 1e-9, f"max|diff|={d_pe:.3e}")

# ---- F: credit propagation with terminal-only reward and V=0 ------------------
# Isolates the trace-decay difference: player 0 acts at steps 0,4,...,56 in the
# all-commit env. With V=0 and a single terminal reward R, the advantage of the
# FIRST action relative to the LAST is:
#   token axis:  (gamma*lam)^56 = lam^56      (token-step distance)
#   macro axis:  lam^14                        (macro-decision distance)
roll_f = build_rollout(specs_all_commit, winner_team=1, seed=2)
roll_f["state_values"] = np.zeros_like(roll_f["state_values"])
n_f = int(roll_f["num_steps"][0])
td_f = to_tensordict(roll_f)
out_f = compute_gae_from_value(td_f, torch.zeros_like(torch.as_tensor(roll_f["state_values"])),
                               discount_gamma=1.0, gae_lambda=0.95)
ta_f = out_f.player_advantage[0].numpy()
mp_f, _, _ = macro_gae_reference(roll_f, gamma=1.0, lam=0.95)
at_f = np.array([mp_f[t, int(roll_f["player_ids"][0][t])] for t in range(n_f)])
first, last = 0, 56
ratio_token = ta_f[first] / ta_f[last]
ratio_macro = at_f[first] / at_f[last]
check("F credit propagation: token-axis end-to-end weight == lam^56",
      abs(ratio_token - lam ** 56) / (lam ** 56) < 1e-3,
      f"measured={ratio_token:.6f} expected={lam**56:.6f}")
check("F credit propagation: macro-axis end-to-end weight == lam^14",
      abs(ratio_macro - lam ** 14) / (lam ** 14) < 1e-3,
      f"measured={ratio_macro:.6f} expected={lam**14:.6f}")
print(f"     end-to-end credit weight, first vs last action of player 0: "
      f"token axis={ratio_token:.6f} vs macro axis={ratio_macro:.6f} "
      f"({ratio_macro / ratio_token:.0f}x more credit for early actions)")

# ---- B3: mixed macro lengths ----------------------------------------------
specs_mixed = []
rng2 = np.random.default_rng(11)
for i in range(80):
    specs_mixed.append((int(rng2.integers(0, 4)), int(rng2.integers(1, 9))))
roll = build_rollout(specs_mixed, winner_team=0, seed=3)
n = int(roll["num_steps"][0])
td = to_tensordict(roll)
out = compute_gae_from_value(td, torch.as_tensor(roll["state_values"]),
                             discount_gamma=1.0, gae_lambda=0.95)
token_adv = out.player_advantage[0].numpy()
macro_adv, segs, chosen = macro_gae_reference(roll)
macro_at_steps = np.array([macro_adv[t, int(roll["player_ids"][0][t])] for t in range(n)])
lengths = np.array([e - s + 1 for (s, e, p) in segs])
print(f"     mixed env: {len(segs)} macros, length mean={lengths.mean():.2f} max={lengths.max()}")
# within-macro dilution in the token timeline: advantage of first token vs commit token
dil = []
for (s, e, p) in segs:
    if e > s and abs(token_adv[e]) > 1e-9:
        dil.append(token_adv[s] / token_adv[e])
dil = np.array(dil)
print(f"     token-GAE within-action dilution ratio (first token / commit token): "
      f"mean={dil.mean():.3f} min={dil.min():.3f} max={dil.max():.3f}")
div = np.abs(token_adv - macro_at_steps)
rel = div / (np.abs(macro_at_steps) + 1e-9)
print(f"     |token_adv - macro_adv|: max={div.max():.4f} mean={div.mean():.4f}; "
      f"relative: p95={np.percentile(rel, 95):.2f} max={rel.max():.2f}")
check("B3 macro segmentation covers every commit and preserves players",
      all(roll["player_ids"][0][s] == p for (s, e, p) in segs) and
      all(chosen[e] for (s, e, p) in segs))

# ---- B4: macro logprob == sum of token logprobs ---------------------------
ok = True
maxerr = 0.0
for (s, e, p) in segs:
    lp_macro = sum(math.log(roll["policy_probs"][0][t, roll["action_taken"][0][t]]) for t in range(s, e + 1))
    # simulate float32 accumulation as torch would do
    acc = torch.zeros((), dtype=torch.float32)
    for t in range(s, e + 1):
        acc = acc + torch.log(torch.tensor(roll["policy_probs"][0][t, roll["action_taken"][0][t]], dtype=torch.float32))
    maxerr = max(maxerr, abs(float(acc) - lp_macro))
check("B4 macro logprob == sum of token logprobs (log domain, L<=8)", maxerr < 1e-5,
      f"max float32-vs-float64 accumulation error over L<=8 segments={maxerr:.2e} "
      f"(plan claims <1e-6; achievable in pure float32, marginal vs float64)")

# ---- B6: terminal reward reaches all four players' last macro ------------
winner = 0
roll = build_rollout(specs_mixed, winner_team=winner, seed=3)
n = int(roll["num_steps"][0])
macro_adv, segs, _ = macro_gae_reference(roll)
# last macro per player must carry nonzero advantage reflecting the terminal reward
last_macro_adv = {}
for (s, e, p) in segs:
    last_macro_adv[p] = macro_adv[s, p]
signs_ok = all(np.sign(last_macro_adv[q]) == (1.0 if q % 2 == winner else -1.0) for q in range(P))
check("B6 terminal reward sign reaches all 4 players' final macro action", signs_ok,
      str({q: round(float(v), 4) for q, v in sorted(last_macro_adv.items())}))

# ---- G: residual actor-fuse zero-init design (plan change #3) --------------
# Standalone replica of the adjudicated design (2026-10-17 second adjudication):
#   z' = z + W_out @ ResGatedMLP(W_in @ [s, a, z, |s-a|]),  W_out = 0, bias=False
# Verifies the exact claims the plan's acceptance criteria must be built on.
from guandan_marvel.model.base import ResGatedMLP  # noqa: E402


class _FuseDesign(nn.Module):
    def __init__(self, d: int = 16, layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(4 * d, d)
        self.trunk = ResGatedMLP(d, n_layers = layers, expansion_ratio = 1)
        self.output_proj = nn.Linear(d, d, bias = False)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z = s * a
        u = torch.cat([s, a, z, (s - a).abs()], dim = -1)
        return z + self.output_proj(self.trunk(self.input_proj(u)))


torch.manual_seed(0)
d_g = 16
s_g = torch.randn(6, d_g)
a_g = torch.randn(6, d_g)
head = nn.Linear(d_g, 1)
fuse = _FuseDesign(d_g)

with torch.no_grad():
    z_old = s_g * a_g
    z_new = fuse(s_g, a_g)
check("G1 zero-init W_out: fused features bitwise identical to old s*a path",
      torch.equal(z_old, z_new))
check("G1b zero-init W_out: pre-softmax logits bitwise identical (shared head)",
      torch.equal(head(z_old), head(z_new)) and
      torch.equal(head(z_old).argmax(-1), head(z_new).argmax(-1)))

# G2: first backward gates the trunk by construction (expected, not a bug)
fuse.zero_grad()
(head(fuse(s_g, a_g)) ** 2).sum().backward()
g_out = fuse.output_proj.weight.grad.norm().item()
g_in = fuse.input_proj.weight.grad.norm().item()
g_trunk = sum(p.grad.norm().item() for p in fuse.trunk.parameters() if p.grad is not None)
check("G2 first backward: zero-init output projection receives nonzero gradient",
      g_out > 0, f"grad_norm={g_out:.3e}")
check("G2b first backward: trunk/input_proj grads are exactly zero (expected gating)",
      g_in == 0.0 and g_trunk == 0.0, f"input={g_in} trunk={g_trunk}")

# G3: one optimizer step opens the gate; second backward reaches the trunk
opt = torch.optim.AdamW(fuse.parameters(), lr = 1e-2)
opt.step()
fuse.zero_grad()
(head(fuse(s_g, a_g)) ** 2).sum().backward()
g_in2 = fuse.input_proj.weight.grad.norm().item()
g_trunk2 = sum(p.grad.norm().item() for p in fuse.trunk.parameters() if p.grad is not None)
check("G3 after first optimizer step: gradients reach input_proj and trunk",
      g_in2 > 0 and g_trunk2 > 0, f"input={g_in2:.3e} trunk={g_trunk2:.3e}")
with torch.no_grad():
    differs = not torch.equal(s_g * a_g, fuse(s_g, a_g))
check("G3b after first optimizer step: fused features diverge from old path (learning active)",
      differs)

# G4: an output_proj bias would be non-identifiable (constant offset cancels in softmax)
bias_only = torch.randn(1, d_g)  # same additive residual for every candidate
cands = torch.randn(8, d_g)
logits = head(cands + bias_only)
check("G4 constant bias residual cancels in softmax (bias=False justified)",
      torch.allclose(torch.softmax(logits, -1), torch.softmax(head(cands), -1), atol = 1e-6))

# G5: alternative zeroing strategies exist but gate the trunk (adjudication nuance)
fuse_alt = _FuseDesign(d_g)
with torch.no_grad():
    fuse_alt.output_proj.weight.normal_()  # make W_out nonzero again
    fuse_alt.trunk.output_norm.weight.zero_()  # zero the final RMSNorm scale instead
check("G5 zeroing the final RMSNorm scale also yields exact zero branch output",
      torch.equal(s_g * a_g, fuse_alt(s_g, a_g)),
      "(valid alternative, but trunk stays gradient-dead on first backward; W_out preferred)")

print("\n" + ("ALL CHECKS PASSED (B1a is a recorded design finding, not a failure)"
               if not FAILURES else f"FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
