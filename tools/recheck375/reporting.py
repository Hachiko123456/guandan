"""Honest actual/reused counts and atomic, resumable result delivery."""
from __future__ import annotations
import collections
import html
import json
import zipfile
from pathlib import Path
from .common import atomic_json, canonical, read_json, sha
from .protocol import decision_summary, read_records

RESULTS_NAME = "guandan_recheck375_results.zip"


def write_reports(root, plan, result, cached, new, *, inherited_count=0):
    root = Path(root)
    by_task = sorted(new, key=lambda r: r["task_id"])
    (root / "new_games.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in by_task), encoding="utf8")
    replay_count = 0
    attempts = 0
    worker_summary = []
    for path in sorted(root.glob("phases/*/worker[01].json")):
        status = read_json(path)
        attempts += int(status.get("games_attempted", status.get("games", 0)))
        if status["phase"] == "cache_replay":
            replay_count += status.get("games", 0)
        worker_summary.append(status)
    result.update({"cached_unique_records_reused": len(cached), "prior_cached_actual_records": 1536,
                   "new_verified_unique_games_total": len(new), "new_verified_unique_games_this_session": len(new) - inherited_count,
                   "validation_replay_games_this_session": replay_count, "actual_game_attempts_this_session": attempts,
                   "new_training_games": 0, "optimizer_updates": 0, "training_authorized": False,
                   "strength_unique_games_target": plan["new_unique_strength_games_cap"],
                   "resume_instruction": "Use this results ZIP OR its Kaggle-extracted directory with the SAME payload/Notebook in a fresh RUN_ROOT; only verified missing games are run.",
                   "worker_summary": worker_summary})
    if result["status"] == "complete":
        if len(new) != plan["new_unique_strength_games_cap"]:
            raise ValueError("cannot label incomplete recheck as complete")
        result["comparison"] = decision_summary(cached + new)
        atomic_json(root / "checkpoint_comparison.json", result["comparison"])
    telemetry = []
    if (root / "telemetry.jsonl").is_file():
        telemetry, ignored = read_records(root / "telemetry.jsonl", allow_partial_tail=True)
        if ignored:
            result["telemetry_partial_tail"] = ignored
    groups = collections.defaultdict(list)
    for sample in telemetry:
        for gpu in sample.get("gpus", []):
            if gpu.get("index") in (0, 1):
                groups[(sample["phase"], gpu["index"])].append(gpu)
    performance = []
    for (phase, rank), samples in sorted(groups.items()):
        values = [g["utilization_percent"] for g in samples if isinstance(g.get("utilization_percent"), (int, float))]
        performance.append({"phase": phase, "gpu_index": rank, "samples": len(samples),
                            "mean_utilization_percent": sum(values) / len(values) if values else None,
                            "max_sampled_memory_mib": max((g.get("memory_used_mib") or 0 for g in samples), default=0)})
    atomic_json(root / "performance.json", {"gpu_by_phase": performance, "stages": result["stages"],
                                           "note": "Independent eval workers, balanced arms. No claim of 100% GPU utilization or training acceleration."})
    result["results_zip"] = str(root.parent / RESULTS_NAME)
    atomic_json(root / "result.json", result)
    title = "iter250 / 375 / 500 同集复测（零训练）"
    text = ["# " + title, "", "本次只做开发诊断，不晋级模型，不运行FableDan，不消费final holdout。", "",
            f"技术状态：**{result['status']}**；本次新增训练局=0，optimizer更新=0。",
            f"已复用旧唯一记录{len(cached)}条；375唯一对局累计完成{len(new)} / {plan['new_unique_strength_games_cap']}，本次新增验证通过{len(new)-inherited_count}场。",
            f"本次另完成缓存一致性重放{replay_count}场；实际尝试执行游戏{attempts}场。缓存不是本次新跑，重放不增加独立牌局数。", ""]
    table = []
    if "comparison" in result:
        comparison = result["comparison"]
        table = ["| 对战（候选在左） | 唯一队伍分配记录 | 独立牌局 | 胜率 | 保守95%区间 | 队伍平均得分 |",
                 "|---|---:|---:|---:|---|---:|"]
        for arm, row in comparison["arms"].items():
            ci = row["clustered_ci95"]
            table.append(f"| {arm} | {row['records']} | {row['independent_deals']} | {100*row['win_rate']:.2f}% | [{100*ci['low']:.2f}%, {100*ci['high']:.2f}%] | {row['raw_team_reward_mean']:+.4f} |")
        text += table + ["", "判读：**" + comparison["verdict"] + "**。", "",
                         "这次把375放到了与250/500相同的旧开发集，并增加375对500的直接对战。它是事后诊断；筛出候选仍需新的确认牌局，不能当最终证明。",
                         "独立统计单位是牌局；两个队伍分配不是两个独立牌局。interval_available只表示区间可计算，不是胜率已显著提升。", ""]
    else:
        text += ["尚未完成全套同集比较，不公布部分对局胜率作为最终结论。已验证的完整逐局结果可以恢复继续。", ""]
    text += ["## 状态与恢复", "", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", "",
             "结果ZIP或其被Kaggle解压后的目录可作为RESUME_RESULTS_ZIP；目录应包含resume_manifest.json，不可填入旧训练结果ZIP，也不是训练恢复包。已经完成的375对局不重复执行；若已全量完成，只重建报告，零新增对局。"]
    (root / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf8")
    body = "<h1>" + title + "</h1><p class=notice>零训练；复用同一开发集；不自动晋级。训练局数=0，optimizer更新=0。</p>"
    body += f"<p>技术状态：<b>{html.escape(result['status'])}</b>。375已验证对局：<b>{len(new)}/{plan['new_unique_strength_games_cap']}</b>；本次新增{len(new)-inherited_count}场，另有{replay_count}场旧结果重放。</p>"
    if "comparison" in result:
        body += "<h2>同一128副牌上的比较</h2><table><tr><th>对战，候选在左</th><th>胜率</th><th>保守95%区间</th><th>队伍平均得分</th></tr>"
        for arm, row in result["comparison"]["arms"].items():
            ci = row["clustered_ci95"]
            body += f"<tr><td>{arm}</td><td>{100*row['win_rate']:.2f}%</td><td>[{100*ci['low']:.2f}%, {100*ci['high']:.2f}%]</td><td>{row['raw_team_reward_mean']:+.4f}</td></tr>"
        body += "</table><p><b>" + result["comparison"]["verdict"] + "</b></p>"
        body += "<h2>同牌局配对rule胜率差</h2><table><tr><th>差异</th><th>均值</th><th>95%区间，百分点</th></tr>"
        for name, row in result["comparison"]["paired_rule_differences"].items():
            body += f"<tr><td>{name}</td><td>{100*row['mean']:+.2f}pp</td><td>[{100*row['low']:+.2f}pp, {100*row['high']:+.2f}pp]</td></tr>"
        body += "</table><p>旧开发集的事后诊断，不替代新确认集。独立统计单位是牌局，不是重复轮换记录。</p>"
    else:
        body += "<p>结果未完成，未把部分胜率作为正式结论；已验证对局保留，可恢复。</p>"
    body += "<details><summary>展开原始状态、逐卡计数和恢复说明</summary><pre>" + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + "</pre></details>"
    (root / "report.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:16px}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:10px;border:1px solid #ddd;text-align:left}.notice{background:#fff4d6;padding:14px}details{margin-top:25px}</style>' + body, encoding="utf8")
    files = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "resume_manifest.json"]
    resume = {"schema": "guandan-recheck375-resume-v1", "plan_hash": plan["plan_hash"], "files": files,
              "verified_new_unique_games": len(new), "training_authorized": False}
    resume["manifest_hash"] = canonical(resume)
    atomic_json(root / "resume_manifest.json", resume)
    target = root.parent / RESULTS_NAME
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
    temporary.replace(target)
    return target
