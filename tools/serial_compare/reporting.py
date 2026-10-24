# SPDX-License-Identifier: MIT
"""Result package: machine-readable result, Markdown/HTML report and ZIP."""
from __future__ import annotations

import hashlib
import html
import json
import zipfile
from pathlib import Path
from typing import Mapping

from .common import RESULT_SCHEMA, atomic_json, canonical, read_json, sha

REPORT_SCHEMA = "guandan-serial-compare-report-v1"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def render_markdown(result: Mapping[str, object]) -> str:
    confirmation = dict(result.get("confirmation") or {})
    verdict = dict(result.get("verdict") or {})
    development = dict(result.get("development") or {})
    arms = dict(result.get("arms") or {})
    lines = [
        "# 一次上传：双配方串行训练与统一评估",
        "",
        f"- 状态：`{result.get('status')}`",
        f"- plan hash：`{result.get('plan_hash')}`",
        f"- 失败阶段：`{result.get('failed_phase')}`",
        f"- FableDan：`{result.get('fabledan')}`；final holdout：`{result.get('final_holdout')}`",
        f"- formal production promotion：`{result.get('formal_production_promotion')}`（始终为 false）",
        "",
        "## 主确认（candidate_final vs control_final）",
        "",
        f"- 完成：`{confirmation.get('complete')}`，记录 {confirmation.get('completed_games')}/{confirmation.get('expected_games')} 场",
        f"- 独立牌局：{confirmation.get('independent_deals')}",
        f"- 团队胜率：{confirmation.get('win_rate')}",
        f"- 95% 按牌局聚类区间：{confirmation.get('clustered_ci95')}",
        f"- 正式判定：`{verdict.get('verdict')}` — {verdict.get('reason')}",
        "",
        "> 统计 INCONCLUSIVE 与技术失败不同；INCONCLUSIVE 表示证据不足，不是程序错误，也不是自动晋级。",
        "",
        "## 开发辅助（不并入主确认）",
        "",
    ]
    for arm, summary in (development.get("arms") or {}).items():
        lines.append(
            f"- `{arm}`：胜率 {summary.get('win_rate')}，CI {summary.get('clustered_ci95', {}).get('low')}"
            f"..{summary.get('clustered_ci95', {}).get('high')}，独立牌局 {summary.get('independent_deals')}"
        )
    lines += ["", "## 两臂", ""]
    for arm, info in arms.items():
        lines.append(
            f"- `{arm}`：迭代 {info.get('start_iteration')}→{info.get('final_iteration')}，"
            f"applied updates {info.get('applied_updates')}，target_kl {info.get('target_kl')}，"
            f"NTP {info.get('ntp_enabled')}，pool {info.get('opponent_pool_manifest_hash')}"
        )
    lines += [
        "",
        "## 预算",
        "",
        f"- 累计已用：{dict(result.get('budget') or {}).get('cumulative_spent')} 秒",
        f"- 冻结预算：{_json(dict(result.get('budget') or {}).get('resolved'))}",
        "",
        "## 阶段",
        "",
        "```json",
        _json(result.get("phases")),
        "```",
    ]
    if result.get("technical_failure"):
        lines += ["", "## 技术失败", "", "```json", _json(result["technical_failure"]), "```"]
    lines += [
        "",
        "## 边界",
        "",
        "- 未运行 Kaggle，未运行双 T4/NCCL，未运行 FableDan，未触碰 final holdout。",
        "- 本地验证范围见 `LOCAL_VALIDATION.json`，交付核对见 `ARTIFACT_QA.json`。",
        "- 单机单卡/Gloo 测试不等于目标双 T4 认证。",
    ]
    return "\n".join(lines) + "\n"


def render_html(result: Mapping[str, object]) -> str:
    body = "<h1>一次上传：双配方串行训练与统一评估</h1>"
    body += f"<p>状态：<code>{html.escape(str(result.get('status')))}</code></p>"
    body += "<h2>结果</h2><pre>" + html.escape(_json({
        "confirmation": result.get("confirmation"),
        "development": result.get("development"),
        "verdict": result.get("verdict"),
        "arms": result.get("arms"),
        "budget": result.get("budget"),
    })) + "</pre>"
    body += "<h2>阶段</h2><pre>" + html.escape(_json(result.get("phases"))) + "</pre>"
    body += "<p>formal_production_promotion 始终为 false；FableDan 未运行；final holdout 未触碰。</p>"
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:32px auto;padding:20px}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:16px}</style>"
        + body
    )


ARCHIVE_NAME = "guandan_serial_compare_results.zip"
RECOVERY_INDEX_SCHEMA = "guandan-serial-compare-recovery-index-v1"
RESULTS_MANIFEST_SCHEMA = "guandan-serial-compare-results-manifest-v1"
_STORED_SUFFIXES = {".zip", ".pt", ".pth", ".bin", ".so", ".pyd", ".dll"}


def _latest_recovery_paths(root_path: Path) -> tuple[set[Path], dict[str, object]]:
    """Select each arm's *latest* complete recovery, dropping older copies.

    Every training arm keeps its full run directory under
    ``phases/train_<arm>/run``.  Including every historical checkpoint would
    blow the Kaggle output budget, so the result ZIP keeps the latest recovery
    (directory or single-process file), its index and its per-file SHA256.
    """
    selected: set[Path] = set()
    arms: dict[str, object] = {}
    for arm in ("control", "candidate"):
        run_dir = root_path / "phases" / f"train_{arm}" / "run"
        index_path = run_dir / "checkpoint_manifest.json"
        latest = None
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            latest = dict(index.get("latest") or {})
        checkpoint_value = None if not latest else latest.get("checkpoint")
        if checkpoint_value:
            target = (run_dir / str(checkpoint_value)).resolve()
            if not target.is_relative_to(run_dir.resolve()):
                raise ValueError("checkpoint index points outside the arm run directory")
            if not target.exists():
                raise ValueError(f"latest recovery checkpoint is missing: {target}")
            if target.is_dir():
                files = sorted(path for path in target.rglob("*") if path.is_file())
            else:
                files = [target]
            selected.update(path.resolve() for path in files)
        if index_path.is_file():
            selected.add(index_path.resolve())
        try:
            run_dir_label = run_dir.resolve().relative_to(root_path.parent.resolve()).as_posix()
        except ValueError:
            run_dir_label = run_dir.name
        arms[arm] = {
            # Relative to the extracted run root: the result ZIP stays portable
            # after moving it to another absolute path.
            "run_dir": run_dir_label,
            "latest": latest or None,
            "files": [
                {
                    "path": path.resolve().relative_to(root_path.resolve()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
                for path in sorted(selected)
                if path.is_relative_to(run_dir.resolve()) and path.is_file()
            ],
        }
    return selected, arms


def _checkpoint_victim(relative: Path, selected: set[Path], file_path: Path) -> bool:
    """True when the file lives in a checkpoint tree but is not the kept copy."""
    if not any(part in ("checkpoints", "checkpoint") for part in relative.parts):
        return False
    return file_path.resolve() not in selected


def _bundle_members(root_path: Path, resume_inputs: Path | None) -> tuple[list[tuple[str, object]], dict[str, object]]:
    run_root = root_path.parent
    selected, arms = _latest_recovery_paths(root_path)
    recovery_index = {
        "schema": RECOVERY_INDEX_SCHEMA,
        "note": "each arm keeps only its latest complete recovery directory/index",
        "arms": arms,
    }
    recovery_index["hash"] = canonical(recovery_index)
    atomic_json(root_path / "recovery_index.json", recovery_index)

    members: list[tuple[str, object]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root_path)
        if path.suffix in (".zip", ".tmp"):
            continue
        if _checkpoint_victim(relative, selected, path):
            continue
        members.append((f"artifacts/{relative.as_posix()}", path))
    if resume_inputs is not None:
        resume_inputs = Path(resume_inputs)
        for path in sorted(resume_inputs.rglob("*")):
            if not path.is_file() or path.suffix == ".zip" and path.name == ARCHIVE_NAME:
                continue
            relative = path.relative_to(resume_inputs).as_posix()
            members.append((f"input/{relative}", path))
    assert members, "the result bundle cannot be empty"
    # A bundle extracted as a new RUN_ROOT must expose input/plan.json.
    if resume_inputs is not None:
        names = {name for name, _ in members}
        if "input/plan.json" not in names:
            raise ValueError("resume inputs do not contain input/plan.json")
    return members, {"recovery_index": recovery_index, "run_root": str(run_root)}


def _write_zip(members: list[tuple[str, object]], destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
        for name, source in members:
            if isinstance(source, (bytes, bytearray)):
                bundle.writestr(zipfile.ZipInfo(name, (2026, 10, 23, 0, 0, 0)), bytes(source),
                                compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
            else:
                path = Path(source)
                kind = zipfile.ZIP_STORED if path.suffix in _STORED_SUFFIXES else zipfile.ZIP_DEFLATED
                bundle.write(path, name, compress_type=kind)
    temporary.replace(destination)


def _member_records(members: list[tuple[str, object]]) -> list[dict[str, object]]:
    records = []
    for name, source in members:
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
            records.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        else:
            path = Path(source)
            records.append({"path": name, "bytes": path.stat().st_size, "sha256": sha(path)})
    return records


def _with_results_manifest(
    members: list[tuple[str, object]],
    context: Mapping[str, object],
    *,
    mode: str = "full",
    omissions: list[dict[str, object]] | None = None,
) -> list[tuple[str, object]]:
    """Sign the exact payload; the manifest itself is never self-declared.

    evidence_complete describes retention of the requested bundle evidence,
    not whether evaluation or training completed successfully.
    """
    manifest = {
        "schema": RESULTS_MANIFEST_SCHEMA,
        "run_root": context["run_root"],
        "mode": mode,
        "evidence_complete": not omissions,
        "omissions": [] if omissions is None else omissions,
        "files": _member_records(members),
    }
    manifest["hash"] = canonical(manifest)
    return [*members, ("RESULTS_MANIFEST.json", _json(manifest).encode("utf-8"))]


def write_reports(
    root: str | Path,
    result: Mapping[str, object],
    *,
    package: bool = True,
    resume_inputs: str | Path | None = None,
) -> dict[str, object]:
    root_path = Path(root)
    result_path = atomic_json(root_path / "RESULT.json", dict(result))
    markdown = render_markdown(result)
    (root_path / "REPORT.md").write_text(markdown, encoding="utf-8")
    (root_path / "report.html").write_text(render_html(result), encoding="utf-8")
    archive = None
    if package:
        archive = root_path / ARCHIVE_NAME
        members, context = _bundle_members(root_path, None if resume_inputs is None else Path(resume_inputs))
        try:
            _write_zip(_with_results_manifest(members, context), archive)
        except OSError:
            # Preserve every file named by the recovery index, including the
            # checkpoint index and checkpoints outside a "checkpoints" tree.
            recovery_names = {
                f"artifacts/{record['path']}"
                for arm in context["recovery_index"]["arms"].values()
                for record in arm["files"]
            }
            keep = [
                (name, source) for name, source in members
                if name.startswith("input/")
                or name in recovery_names
                or "/checkpoints/" in name
                or name.endswith(("RESULT.json", "REPORT.md", "report.html", "recovery_index.json"))
            ]
            if not keep:
                raise
            kept_names = {name for name, _ in keep}
            omissions = _member_records([(name, source) for name, source in members if name not in kept_names])
            # Build from raw retained members, never the failed full manifest:
            # the degraded archive must explicitly disclose missing evidence.
            _write_zip(_with_results_manifest(
                keep, context, mode="space_pressure_fallback", omissions=omissions,
            ), archive)
    return {
        "schema": REPORT_SCHEMA,
        "result": str(result_path),
        "result_sha256": sha(result_path),
        "report_markdown": str(root_path / "REPORT.md"),
        "report_html": str(root_path / "report.html"),
        "archive": None if archive is None else str(archive),
        "archive_sha256": None if archive is None else sha(archive),
        "hash": canonical({
            "result_sha256": sha(result_path),
            "archive_sha256": None if archive is None else sha(archive),
        }),
    }


def ensure_minimal_result(root: str | Path, *, plan_hash: str, package: bool = True) -> Path:
    """Create a valid minimal result *and result ZIP* before training.

    Spec 9.1 requires the result package to exist before the run, so a hard
    crash still ships a valid (if minimal) ZIP rather than nothing.
    """
    root_path = Path(root)
    target = root_path / "RESULT.json"
    if not target.is_file():
        minimal = {
            "schema": RESULT_SCHEMA,
            "plan_hash": plan_hash,
            "status": "in_progress",
            "failed_phase": None,
            "error": None,
            "phases": {},
            "budget": {},
            "arms": {},
            "confirmation": {},
            "development": {},
            "verdict": {"verdict": "NOT_EVALUATED", "formal_production_promotion": False},
            "technical_failure": None,
            "fabledan": "NOT_RUN",
            "final_holdout": "NOT_TOUCHED",
            "formal_production_promotion": False,
            "kaggle_execution": "NOT_RUN",
            "target_dual_t4_validation": "NOT_RUN",
        }
        minimal["hash"] = canonical(minimal)
        atomic_json(target, minimal)
    else:
        minimal = read_json(target)
    if package:
        write_reports(root_path, minimal, package=True)
    return target


__all__ = [
    "ARCHIVE_NAME", "RECOVERY_INDEX_SCHEMA", "REPORT_SCHEMA", "RESULTS_MANIFEST_SCHEMA",
    "ensure_minimal_result", "render_html", "render_markdown", "write_reports",
]
