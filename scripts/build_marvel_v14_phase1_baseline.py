"""Freeze the MARVEL v1.4 phase-1 baseline manifest, report, and fixed probe.

Artifacts (defaults):
- docs/marvel_v14/phase1_baseline_manifest.json
- docs/marvel_v14/phase1_baseline_report.md
- runs/marvel_v14_phase1/probe_inputs.pt
- runs/marvel_v14_phase1/probe_outputs.pt
- runs/marvel_v14_phase1/macro_probe.json

The legacy M09 evaluation checkpoint was produced on a different native build
identity (Kaggle Linux/CPython 3.13).  The official strict loader is *not*
modified or weakened: its failure is recorded verbatim.  Model outputs are
produced through a separate, explicitly labelled diagnostic-only load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from guandan_marvel import cpp_game as cpp
from guandan_marvel.config import deserialize_train_config
from guandan_marvel.evaluation.checkpoints import load_v2_policy
from guandan_marvel.model import Agent
from guandan_marvel.model.tensordict import TensorDict
from guandan_marvel.training.checkpoints import (
    build_backend_identity, canonical_json_hash,
)
from guandan_marvel.training.macro_actions import macro_diagnostics, segment_macro_actions
from nanopokerlib.build_identity import compute_native_source_manifest

DEFAULT_CHECKPOINT = ROOT / "runs" / "comparison_delivery" / "dataset" / "m09_candidate_iter002000.pt"
FINAL_CANDIDATE_FREEZE = (
    ROOT / "docs" / "marvel_migration" / "evidence" / "M09_unified_segment2000"
    / "final_candidate_freeze.json"
)
DELIVERY_FILES = (
    "runs/bootcamp_week1_delivery/datasets/session1_start/guandan-marvel-source.zip",
    "runs/bootcamp_week1_delivery/datasets/session1_start/marvel_bootcamp_week1_plan.json",
    "runs/bootcamp_week1_delivery/datasets/session1_start/warm_start_policy.pt",
    "runs/bootcamp_week1_delivery/delivery_manifest.json",
    "runs/comparison_delivery/dataset/guandan-marvel-source.zip",
    "runs/comparison_delivery/dataset/comparison_control.json",
    "runs/comparison_delivery/dataset/m09_champion_iter000100.pt",
    "runs/comparison_delivery/delivery_manifest.json",
    "configs/marvel_m09_full_plan.json",
    "configs/marvel_m09_main_training_plan.json",
    "configs/marvel_m09_two_stage_continuation_plan.json",
)
SOURCE_SUFFIXES = (
    ".py", ".cpp", ".hpp", ".h", ".json", ".md", ".ipynb", ".toml", ".txt", ".cfg", ".ini",
)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)


def _is_excluded(relative: str, excludes: tuple[str, ...]) -> bool:
    normalized = relative.replace("\\", "/")
    for exclude in excludes:
        candidate = exclude.replace("\\", "/").rstrip("/")
        if normalized == candidate or normalized.startswith(candidate + "/"):
            return True
    return False


def collect_git(exclude_paths: tuple[str, ...] = ()) -> dict[str, object]:
    """Capture the git baseline; generated outputs must be excluded.

    The manifest/report/probe files produced by this very build are declared in
    ``generated_outputs`` instead of being frozen as inputs, so the snapshot
    never records a hash that this build is about to invalidate.
    """
    status = _run_git("status", "--porcelain")
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    untracked = [line for line in _run_git("ls-files", "--others", "--exclude-standard").splitlines() if line]
    untracked = [line for line in untracked if not _is_excluded(line, exclude_paths)]
    untracked_source = []
    for relative in sorted(untracked):
        path = ROOT / relative
        if path.suffix.lower() in SOURCE_SUFFIXES:
            untracked_source.append({
                "path": relative.replace("\\", "/"),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256_file(path),
            })
    return {
        "commit": _run_git("rev-parse", "HEAD").strip(),
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
        "tracked_diff_sha256": _sha256_bytes(diff),
        "tracked_diff_bytes": len(diff),
        "untracked_file_count": len(untracked),
        "untracked_source_files": untracked_source,
    }


def collect_environment() -> dict[str, object]:
    full_identity = dict(cpp.get_backend_build_identity())
    raw_identity = {key: value for key, value in full_identity.items() if key != "current_source_manifest"}
    try:
        strict_identity = build_backend_identity(cpp)
        strict_identity_error = None
    except Exception as error:  # pragma: no cover - reported honestly
        strict_identity = None
        strict_identity_error = f"{type(error).__name__}: {error}"
    extension_path = Path(str(raw_identity.get("extension_path", "")))
    source_manifest = compute_native_source_manifest(ROOT)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cpp_backend_identity_trimmed": raw_identity,
        "cpp_backend_identity_hash": canonical_json_hash(full_identity),
        "strict_build_backend_identity_hash": None if strict_identity is None else strict_identity["hash"],
        "strict_build_backend_identity_error": strict_identity_error,
        "extension_path": str(extension_path),
        "extension_bytes": extension_path.stat().st_size if extension_path.is_file() else None,
        "extension_sha256": _sha256_file(extension_path),
        "cpp_source_manifest": source_manifest,
        "cpp_source_manifest_hash": source_manifest["hash"],
        "source_matches_loaded_binary": bool(raw_identity.get("source_matches_loaded_binary")),
        "embedded_source_sha256": raw_identity.get("embedded_source_sha256"),
    }


def collect_checkpoint(checkpoint: Path) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resolved = dict(payload["resolved_config"])
    resolved_hash_recomputed = canonical_json_hash(resolved)
    identity = {
        "path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "resolved_path": str(checkpoint.resolve()),
        "bytes": checkpoint.stat().st_size,
        "mtime": datetime.fromtimestamp(checkpoint.stat().st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256_file(checkpoint),
        "schema": payload.get("schema"),
        "algorithm": payload.get("algorithm"),
        "architecture": payload.get("architecture"),
        "backend": payload.get("backend"),
        "protocol_version": payload.get("protocol_version"),
        "iteration": payload.get("iteration"),
        "merged_policy": payload.get("merged_policy"),
        "source_hash": payload.get("source_hash"),
        "source_parent_commit": payload.get("source_parent_commit"),
        "resolved_config_hash_declared": payload.get("resolved_config_hash"),
        "resolved_config_hash_recomputed": resolved_hash_recomputed,
        "resolved_config_matches": payload.get("resolved_config_hash") == resolved_hash_recomputed,
        "resolved_config": resolved,
        "training_data_provenance": {
            key: payload["training_data_provenance"].get(key)
            for key in ("schema", "domain", "hash", "complete", "deal_count")
        } if isinstance(payload.get("training_data_provenance"), dict) else None,
        "checkpoint_backend_identity_hash": payload["backend_identity"]["hash"],
        "checkpoint_backend_identity_implementation": {
            key: payload["backend_identity"]["implementation"].get(key)
            for key in (
                "extension_path", "extension_sha256", "embedded_source_sha256",
                "compiler", "cxx_standard", "python_version", "source_matches_loaded_binary",
            )
        },
        "agent_state_dict_key_count": len(payload["agent_state_dict"]),
    }
    bootcamp_warm_start = (
        ROOT / "runs" / "bootcamp_week1_delivery" / "datasets" / "session1_start" / "warm_start_policy.pt"
    )
    identity["warm_start_identity"] = {
        "bootcamp_warm_start_path": str(bootcamp_warm_start.relative_to(ROOT)).replace("\\", "/"),
        "bootcamp_warm_start_sha256": _sha256_file(bootcamp_warm_start),
        "identical_to_checkpoint": _sha256_file(bootcamp_warm_start) == identity["sha256"],
        "checkpoint_source_parent_commit": payload.get("source_parent_commit"),
        "checkpoint_source_hash": payload.get("source_hash"),
        "checkpoint_merged_policy": payload.get("merged_policy"),
        "resume_classification": payload.get("resume_classification"),
    }
    freeze = json.loads(FINAL_CANDIDATE_FREEZE.read_text(encoding="utf-8")) if FINAL_CANDIDATE_FREEZE.is_file() else None
    identity["matches_frozen_candidate_sha256"] = bool(
        freeze and freeze.get("candidate_checkpoint_sha256") == identity["sha256"]
    )
    identity["frozen_candidate_manifest"] = None if freeze is None else {
        "path": str(FINAL_CANDIDATE_FREEZE.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256_file(FINAL_CANDIDATE_FREEZE),
        "candidate_iteration": freeze.get("candidate_iteration"),
        "candidate_checkpoint_sha256": freeze.get("candidate_checkpoint_sha256"),
        "frozen_at": freeze.get("frozen_at"),
    }
    return identity


def try_official_load(checkpoint: Path) -> dict[str, object]:
    try:
        artifact = load_v2_policy(checkpoint, device="cpu", expected_algorithm="ippo")
        return {
            "status": "loaded",
            "detail": f"iteration={artifact.iteration}",
            "official_strict_load": True,
        }
    except Exception as error:
        return {
            "status": "rejected",
            "error_type": type(error).__name__,
            "error": str(error),
            "official_strict_load": False,
            "note": (
                "official strict loader is intentionally NOT weakened; legacy checkpoint was "
                "built against a different native backend identity"
            ),
        }


def diagnostic_model_load(checkpoint: Path):
    """Diagnostic-only model construction; never a training/evaluation path."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = deserialize_train_config(payload["resolved_config"])
    if cfg is None:
        raise ValueError("checkpoint resolved_config could not be decoded")
    algorithm = str(payload["algorithm"])
    enable_ntp = Agent.ntp_enabled_for_config(cfg) and algorithm == "ippo"
    model = Agent(cfg, enable_value_head=algorithm == "ippo", enable_ntp_head=enable_ntp)
    model.load_state_dict(payload["agent_state_dict"], strict=True)
    model.eval()
    return model, cfg, payload


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = 0
    modules: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        root = name.split(".", 1)[0]
        modules[root] = modules.get(root, 0) + count
    modules = dict(sorted(modules.items()))
    return {"total": total, "by_module": modules}


def run_probe(checkpoint: Path, device: str = "cpu") -> dict[str, object]:
    model, cfg, payload = diagnostic_model_load(checkpoint)
    state_dict = model.state_dict()
    state_hash = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        state_hash.update(key.encode("utf-8"))
        state_hash.update(str(tuple(tensor.shape)).encode("utf-8"))
        state_hash.update(str(tensor.dtype).encode("utf-8"))
        state_hash.update(tensor.numpy().tobytes())
    counts = parameter_counts(model)

    batch = cpp.EnvBatch(2, auto_reset=False)
    step = batch.reset(20261017)
    B = int(step.rewards.shape[0])
    inputs = {
        "obs_tokens": torch.as_tensor(step.observations.tokens, dtype=torch.long),
        "obs_channels": torch.as_tensor(step.observations.channels, dtype=torch.uint8),
        "player_ids": torch.as_tensor(step.player_ids, dtype=torch.long),
        "avail_tokens": torch.as_tensor(step.available_actions.tokens, dtype=torch.long),
        "avail_weights": torch.as_tensor(step.available_actions.weights, dtype=torch.float32),
        "avail_is_commit": torch.as_tensor(step.available_actions.is_commit, dtype=torch.bool),
        "rewards": torch.as_tensor(step.rewards, dtype=torch.float32),
        "is_terminal": torch.as_tensor(step.is_terminal, dtype=torch.bool),
    }
    rollout = TensorDict(
        {
            "num_steps": torch.ones(B, dtype=torch.long),
            "player_ids": inputs["player_ids"].unsqueeze(1),
            "obs_tokens": inputs["obs_tokens"].unsqueeze(1),
            "obs_channels": inputs["obs_channels"].unsqueeze(1),
            "avail_tokens": inputs["avail_tokens"].unsqueeze(1),
            "avail_weights": inputs["avail_weights"].unsqueeze(1),
        },
        batch_size=[B],
    )
    captured = []
    handle = model.logit_head.register_forward_hook(
        lambda module, args, output: captured.append(output.detach())
    )
    with torch.no_grad():
        output = model(rollout=rollout)
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("logit head hook did not capture exactly one output")
    raw_head_logits = captured[0].squeeze(-1).float()
    avail_probs = output.avail_probs.float()
    stepped_weights = inputs["avail_weights"].unsqueeze(1)
    mask = stepped_weights > 0
    reconstructed = raw_head_logits + mask.to(raw_head_logits.dtype) * stepped_weights.clamp(min=1e-9).log()
    reconstructed = reconstructed.masked_fill(
        ~mask, torch.finfo(reconstructed.dtype).min
    )
    argmax = avail_probs.argmax(dim=-1, keepdim=True)
    selected_prob = avail_probs.gather(-1, argmax)
    probability_match = bool(
        torch.allclose(
            torch.softmax(reconstructed.float(), dim=-1), avail_probs, rtol=1e-5, atol=1e-6,
        )
    )
    content_hash = hashlib.sha256()
    for name, tensor in (
        ("avail_probs", avail_probs),
        ("avail_logits_reconstructed", reconstructed),
        ("raw_head_logits", raw_head_logits),
        ("argmax_action", argmax),
        ("selected_prob", selected_prob),
    ):
        content_hash.update(name.encode("utf-8"))
        content_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    outputs = {
        "model_output_content_sha256": content_hash.hexdigest(),
        "diagnostic_only": True,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_state_dict_sha256": state_hash.hexdigest(),
        "parameter_counts": counts,
        "avail_probs": avail_probs,
        "avail_logits_reconstructed": reconstructed,
        "raw_head_logits": raw_head_logits,
        "argmax_action": argmax,
        "selected_prob": selected_prob,
        "state_values": None if output.state_values is None else output.state_values.float(),
        "probability_match_detail": {
            "reconstructed_softmax_matches_avail_probs": probability_match,
        },
    }
    return {"inputs": inputs, "outputs": outputs, "reset_seed": 20261017}


def run_macro_probe(games: int = 4) -> dict[str, object]:
    """Macro diagnostics over real random-play games (phase-1 baseline values)."""
    records = []
    for index in range(games):
        seed = 90001 + index * 7919
        batch = cpp.EnvBatch(1, auto_reset=False)
        step = batch.reset(seed)
        rng = np.random.default_rng(seed)
        player_ids, action_taken, avail_is_commit = [], [], []
        rewards, terminal = [], []
        while True:
            legal = np.flatnonzero(step.available_actions.weights[0] > 0)
            choice = int(rng.choice(legal))
            player_ids.append(int(step.player_ids[0]))
            action_taken.append(choice)
            avail_is_commit.append(step.available_actions.is_commit[0].copy())
            rows = step.available_actions.tokens[0, choice][None, :].astype(np.int64)
            next_step = batch.step(rows)
            rewards.append(next_step.rewards[0].copy())
            terminal.append(bool(next_step.is_terminal[0]))
            if next_step.is_terminal[0]:
                info = dict(next_step.infos[0])
                break
            step = next_step
        steps = len(player_ids)
        segmentation = segment_macro_actions(
            num_steps=torch.tensor([steps], dtype=torch.long),
            player_ids=torch.as_tensor(np.asarray(player_ids, dtype=np.int64)).unsqueeze(0),
            action_taken=torch.as_tensor(np.asarray(action_taken, dtype=np.int64)).unsqueeze(0),
            avail_is_commit=torch.as_tensor(np.stack(avail_is_commit, axis=0)).unsqueeze(0),
            is_terminal=torch.as_tensor(np.asarray(terminal, dtype=np.bool_)).unsqueeze(0),
        )
        diagnostics = macro_diagnostics(
            segmentation, engine_committed_steps=int(info["finish_committed_step"]),
        )
        records.append({
            "seed": seed,
            "token_steps": steps,
            "engine_committed_steps": int(info["finish_committed_step"]),
            "finish_token_step": int(info.get("finish_token_step", steps)),
            "outcome_class": info.get("outcome_class"),
            "diagnostics": diagnostics,
        })
    return {"schema": "guandan-marvel-macro-probe-v1", "games": records}


def collect_delivery() -> dict[str, object]:
    files = []
    for relative in DELIVERY_FILES:
        path = ROOT / relative
        files.append({
            "path": relative.replace("\\", "/"),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256_file(path),
        })
    manifests = {}
    for name in ("runs/bootcamp_week1_delivery/delivery_manifest.json",
                 "runs/comparison_delivery/delivery_manifest.json"):
        path = ROOT / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            declared = payload.get("hash")
            manifests[name.replace("\\", "/")] = {
                "declared_hash": declared,
                "recomputed_hash": canonical_json_hash({k: v for k, v in payload.items() if k != "hash"}),
                "hash_matches": declared == canonical_json_hash({k: v for k, v in payload.items() if k != "hash"}),
            }
    return {"files": files, "manifests": manifests}


def write_report(manifest: dict[str, object], report_path: Path) -> None:
    snapshot = manifest["snapshot_inputs"]
    git = snapshot["git"]
    environment = snapshot["environment"]
    checkpoint = snapshot["checkpoint"]
    official = manifest["official_strict_load"]
    probe = manifest["probe"]
    outputs = manifest["generated_outputs"]
    lines = [
        "# MARVEL v1.4 阶段 1 基线冻结报告（Baseline Report）",
        "",
        f"- 生成时间（Created）: {manifest['created_at']}",
        f"- Manifest: `{outputs.get('manifest', {}).get('path')}`（hash `{manifest['hash']}`）",
        f"- snapshot_inputs hash: `{manifest['snapshot_inputs_hash']}`",
        "",
        "## Git 基线",
        "",
        f"- commit: `{git['commit']}`（branch `{git['branch']}`，dirty={git['dirty']}）",
        f"- tracked diff sha256: `{git['tracked_diff_sha256']}`（{git['tracked_diff_bytes']} bytes）",
        f"- untracked source files: {len(git['untracked_source_files'])}",
        "",
        "## 环境（Environment）",
        "",
        f"- Python {environment['python_version']}（{environment['python_implementation']}），torch {environment['torch_version']}，numpy {environment['numpy_version']}",
        f"- C++ backend identity hash（checkpoint loader identity）: `{environment['strict_build_backend_identity_hash']}`",
        f"- C++ native identity hash（raw extension identity）: `{environment['cpp_backend_identity_hash']}`",
        f"- C++ source manifest hash: `{environment['cpp_source_manifest_hash']}`",
        f"- extension sha256: `{environment['extension_sha256']}`",
        f"- source matches loaded binary: {environment['source_matches_loaded_binary']}",
        "",
        "## 模型基线（Model baseline）",
        "",
        f"- M09 checkpoint: `{checkpoint['path']}`（{checkpoint['bytes']} bytes）",
        f"- sha256: `{checkpoint['sha256']}`（frozen candidate match: {checkpoint['matches_frozen_candidate_sha256']}）",
        f"- architecture/algorithm/iteration: {checkpoint['architecture']} / {checkpoint['algorithm']} / {checkpoint['iteration']}",
        f"- resolved config hash（declared/recomputed）: `{checkpoint['resolved_config_hash_declared']}` / `{checkpoint['resolved_config_hash_recomputed']}`",
        f"- checkpoint native backend identity: `{checkpoint['checkpoint_backend_identity_hash']}`",
        "",
        "## 正式 strict loader（未削弱）",
        "",
        f"- status: **{official['status']}**",
    ]
    if official["status"] != "loaded":
        lines.append(f"- error: `{official['error_type']}: {official['error']}`")
        lines.append(
            "- 旧 M09 checkpoint 在 Kaggle（Linux/CPython 3.13）构建，其 native backend identity "
            "与本地扩展不一致，正式 strict 校验如实拒绝且未被削弱。下方模型输出来自"
            "单独标注的 diagnostic-only 手工加载，不构成评估证据。"
        )
    lines.extend([
        "",
        "## 固定 probe（Diagnostic-only model probe）",
        "",
        f"- probe inputs: `{probe.get('inputs_path')}` sha256 `{probe.get('inputs_sha256')}`",
        f"- probe outputs: `{probe.get('outputs_path')}` sha256 `{probe.get('outputs_sha256')}`",
        f"- macro probe: `{probe.get('macro_probe_path')}` sha256 `{probe.get('macro_probe_sha256')}`",
        f"- model output content sha256（timestamp-free）: `{probe.get('model_output_content_sha256')}`",
        f"- state_dict sha256: `{probe.get('checkpoint_state_dict_sha256')}`",
        f"- parameter counts: total {probe.get('parameter_counts', {}).get('total')}",
        "",
        "## 真实环境宏分段 probe（Real-environment macro probe）",
        "",
    ])
    macro_probe = probe.get("macro_probe") or {}
    for game in macro_probe.get("games", []):
        diagnostics = game["diagnostics"]
        lines.append(
            f"- seed {game['seed']}: tokens={game['token_steps']}, engine commits={game['engine_committed_steps']}, "
            f"macro count={diagnostics.get('macro/count')}, commit match={diagnostics.get('macro/commit_step_match')}, "
            f"length_mean={diagnostics.get('macro/length_mean'):.3f}, tau_mean={diagnostics.get('macro/tau_mean'):.3f}, "
            f"pass={diagnostics.get('macro/pass_count')}, commit={diagnostics.get('macro/commit_count')}, "
            f"incomplete={diagnostics.get('macro/incomplete_count')}, violations={diagnostics.get('macro/player_switch_violation_count')}"
        )
    lines.extend([
        "",
        "## 交付物（Delivery artifacts）",
        "",
    ])
    for item in snapshot["delivery"]["files"]:
        lines.append(f"- `{item['path']}`: {item['bytes']} bytes, sha256 `{item['sha256']}`")
    lines.extend([
        "",
        "## 生成物（Generated outputs）",
        "",
        "以下文件由本次构建生成；其 sha256 记录在外部 build result 中，manifest 不写入自身哈希以避免自引用失效。",
        "",
    ])
    for name, entry in sorted(outputs.items()):
        if isinstance(entry, dict) and entry.get("path"):
            lines.append(f"- {name}: `{entry['path']}`")
    lines.extend([
        "",
        "## 范围声明（Scope statement）",
        "",
        "阶段 1 仅冻结本基线，并新增宏动作分段、legacy GAE 等价 Oracle、诊断指标与测试。"
        "训练语义未改变：token PPO、token GAE、flat ±1 reward、actor、context 与 AMP 状态"
        "均未改动，且未提前实施阶段 2 及以后内容。",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _display_path(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _manifest_relative_path(path: Path, manifest_dir: Path) -> str:
    """Output paths recorded inside the manifest, relative to the manifest dir.

    Keeping them layout-relative (instead of absolute) makes two builds with
    identical inputs and the same relative output layout byte-identical.
    """
    try:
        relative = os.path.relpath(path.resolve(), manifest_dir.resolve())
    except ValueError:
        return str(path.resolve()).replace("\\", "/")
    return Path(relative).as_posix()


def verify_manifest_hashes(
    manifest: dict[str, object],
    *,
    root: Path = ROOT,
    manifest_dir: Path | None = None,
) -> dict[str, object]:
    """Recursively verify every declared ``path`` + ``sha256`` pair.

    Snapshot-input paths are resolved against the repository root; generated
    probe paths are resolved against the manifest directory.  The manifest
    never embeds its own (or the report's/baseline result's) hash, so those are
    not verified here - they are recorded by the external build result instead.
    """
    checked: list[dict[str, object]] = []
    failures: list[str] = []
    base_dir = manifest_dir if manifest_dir is not None else root

    def check_entry(entry: dict[str, object], label: str, *, base: Path) -> None:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return
        declared = entry.get("sha256")
        path = Path(raw_path)
        if not path.is_absolute():
            path = base / raw_path
        if not path.is_file():
            failures.append(f"{label}: missing declared input {raw_path}")
            return
        actual = _sha256_file(path)
        if declared is None:
            failures.append(f"{label}: declared input {raw_path} has no sha256")
            return
        if actual != declared:
            failures.append(
                f"{label}: sha256 mismatch for {raw_path}: declared={declared} actual={actual}"
            )
            return
        expected_bytes = entry.get("bytes")
        if expected_bytes is not None and int(path.stat().st_size) != int(expected_bytes):
            failures.append(f"{label}: byte size mismatch for {raw_path}")
            return
        checked.append({"label": label, "path": raw_path, "sha256": actual})

    def walk(node: object, label: str, *, base: Path) -> None:
        if isinstance(node, dict):
            if "path" in node:
                check_entry(node, label, base=base)
            for key, value in node.items():
                if key in {"path", "sha256", "bytes", "exists"}:
                    continue
                walk(value, f"{label}.{key}", base=base)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{label}[{index}]", base=base)

    walk(manifest.get("snapshot_inputs", {}), "snapshot_inputs", base=root)
    environment = manifest.get("snapshot_inputs", {}).get("environment", {})
    if isinstance(environment, dict) and environment.get("extension_path"):
        check_entry(
            {
                "path": environment.get("extension_path"),
                "sha256": environment.get("extension_sha256"),
                "bytes": environment.get("extension_bytes"),
            },
            "snapshot_inputs.environment.extension",
            base=root,
        )
    probe = manifest.get("probe", {})
    if isinstance(probe, dict):
        for label, path_key, sha_key in (
            ("probe.inputs", "inputs_path", "inputs_sha256"),
            ("probe.outputs", "outputs_path", "outputs_sha256"),
            ("probe.macro_probe", "macro_probe_path", "macro_probe_sha256"),
        ):
            if probe.get(path_key):
                check_entry(
                    {"path": probe[path_key], "sha256": probe.get(sha_key)},
                    label,
                    base=base_dir,
                )
    return {
        "ok": not failures,
        "checked_count": len(checked),
        "failures": failures,
        "checked": checked,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs" / "marvel_v14" / "phase1_baseline_manifest.json")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "marvel_v14" / "phase1_baseline_report.md")
    parser.add_argument("--probe-dir", type=Path, default=ROOT / "runs" / "marvel_v14_phase1")
    parser.add_argument("--result", type=Path, default=None,
                        help="external build-result JSON (defaults to <probe-dir>/baseline_build_result.json)")
    parser.add_argument("--created-at", default=None,
                        help="fixed ISO8601 creation timestamp for reproducible builds")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else (ROOT / args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    report_path = args.report if args.report.is_absolute() else (ROOT / args.report)
    probe_dir = args.probe_dir if args.probe_dir.is_absolute() else (ROOT / args.probe_dir)
    result_path = args.result if args.result is not None else (probe_dir / "baseline_build_result.json")
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    probe_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    exclude_paths = (
        _display_path(manifest_path),
        _display_path(report_path),
        _display_path(probe_dir),
        _display_path(result_path),
    )
    snapshot_inputs: dict[str, object] = {
        "git": collect_git(exclude_paths=exclude_paths),
        "environment": collect_environment(),
        "checkpoint": collect_checkpoint(checkpoint),
        "delivery": collect_delivery(),
    }

    probe_inputs_path = probe_dir / "probe_inputs.pt"
    probe_outputs_path = probe_dir / "probe_outputs.pt"
    macro_probe_path = probe_dir / "macro_probe.json"
    probe_summary: dict[str, object] = {"status": "skipped"}
    if not args.skip_probe:
        probe = run_probe(checkpoint)
        environment = snapshot_inputs["environment"]
        torch.save(
            {
                "schema": "guandan-marvel-v14-phase1-probe-inputs-v1",
                "reset_seed": probe["reset_seed"],
                "backend_identity_hash": environment["strict_build_backend_identity_hash"],
                "extension_sha256": environment["extension_sha256"],
                **probe["inputs"],
            },
            probe_inputs_path,
        )
        torch.save(
            {
                "schema": "guandan-marvel-v14-phase1-probe-outputs-v1",
                "created_at": created_at,
                **probe["outputs"],
            },
            probe_outputs_path,
        )
        macro_probe = run_macro_probe()
        macro_probe_path.write_text(
            json.dumps(macro_probe, indent=2, sort_keys=True, default=str), encoding="utf-8",
        )
        probe_summary = {
            "status": "written",
            "inputs_path": _manifest_relative_path(probe_inputs_path, manifest_path.parent),
            "inputs_sha256": _sha256_file(probe_inputs_path),
            "outputs_path": _manifest_relative_path(probe_outputs_path, manifest_path.parent),
            "outputs_sha256": _sha256_file(probe_outputs_path),
            "model_output_content_sha256": probe["outputs"]["model_output_content_sha256"],
            "macro_probe_path": _manifest_relative_path(macro_probe_path, manifest_path.parent),
            "macro_probe_sha256": _sha256_file(macro_probe_path),
            "diagnostic_only": True,
            "parameter_counts": probe["outputs"]["parameter_counts"],
            "checkpoint_state_dict_sha256": probe["outputs"]["checkpoint_state_dict_sha256"],
            "macro_probe": macro_probe,
        }

    generated_outputs = {
        "manifest": {"path": manifest_path.name},
        "report": {"path": _manifest_relative_path(report_path, manifest_path.parent)},
        "probe_inputs": {"path": _manifest_relative_path(probe_inputs_path, manifest_path.parent)},
        "probe_outputs": {"path": _manifest_relative_path(probe_outputs_path, manifest_path.parent)},
        "macro_probe": {"path": _manifest_relative_path(macro_probe_path, manifest_path.parent)},
        "result": {"path": _manifest_relative_path(result_path, manifest_path.parent)},
        "note": (
            "all paths are relative to this manifest's directory; output sha256 values are "
            "recorded by the external build result; the manifest never embeds its own, the "
            "report's or the result's hash"
        ),
    }

    manifest: dict[str, object] = {
        "schema": "guandan-marvel-v14-phase1-baseline-manifest-v2",
        "created_at": created_at,
        "snapshot_inputs": snapshot_inputs,
        "snapshot_inputs_hash": canonical_json_hash(snapshot_inputs),
        "official_strict_load": try_official_load(checkpoint),
        "probe": probe_summary,
        "generated_outputs": generated_outputs,
    }
    manifest["hash"] = canonical_json_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8",
    )
    write_report(manifest, report_path)

    verification = verify_manifest_hashes(manifest, root=ROOT, manifest_dir=manifest_path.parent)
    build_result = {
        "schema": "guandan-marvel-v14-phase1-baseline-result-v1",
        "created_at": created_at,
        "manifest": {
            "path": _display_path(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
            "hash": manifest["hash"],
        },
        "report": {
            "path": _display_path(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": _sha256_file(report_path),
        },
        "probe_inputs": {"path": probe_summary.get("inputs_path"), "sha256": probe_summary.get("inputs_sha256")},
        "probe_outputs": {"path": probe_summary.get("outputs_path"), "sha256": probe_summary.get("outputs_sha256")},
        "macro_probe": {"path": probe_summary.get("macro_probe_path"), "sha256": probe_summary.get("macro_probe_sha256")},
        "model_output_content_sha256": probe_summary.get("model_output_content_sha256"),
        "snapshot_inputs_hash": manifest["snapshot_inputs_hash"],
        "verification": {
            "ok": verification["ok"],
            "checked_count": verification["checked_count"],
            "failures": verification["failures"],
        },
    }
    result_path.write_text(
        json.dumps(build_result, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_hash": manifest["hash"],
        "snapshot_inputs_hash": manifest["snapshot_inputs_hash"],
        "checkpoint_sha256": snapshot_inputs["checkpoint"]["sha256"],
        "official_strict_load": manifest["official_strict_load"]["status"],
        "strict_backend_identity_hash": snapshot_inputs["environment"]["strict_build_backend_identity_hash"],
        "probe": {k: v for k, v in probe_summary.items() if k != "macro_probe"},
        "verification": build_result["verification"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
