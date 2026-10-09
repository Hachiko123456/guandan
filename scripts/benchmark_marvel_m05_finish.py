"""Frozen-manifest benchmark for the bounded M05 performance finish."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import torch

from guandan_marvel.config import ModelConfig, TrainConfig
from guandan_marvel.model import Agent
from guandan_marvel.testing.differential import build_case
from scripts.benchmark_marvel_m05_repair import run_agent_lane

SCHEMA = "guandan-marvel-m05-performance-finish-benchmark-v1"
PRIMARY_INDICES = tuple(range(50_000, 50_004))
GENERALIZATION_INDICES = tuple(range(51_000, 51_008))


class PeakRss:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set(); self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def build_config() -> TrainConfig:
    from guandan_marvel import python_game
    cfg = TrainConfig(
        game=python_game.GAME_SPEC, device="cpu", algo_name="ippo",
        logging_dir="runs/M05_performance_finish/benchmark",
        run_name="frozen-transformer", seed=91_001,
        rollout_batch_size=1, collect_batch_size=0, replay_buffer_scale=1,
        mini_batches=1, total_iterations=1,
        max_sampler_bytes_per_rank=256 * 2**20,
        checkpoint_replay_buffer=False, amp_enabled=False,
        model=ModelConfig(
            architecture="v1.3", history_dim=16, feature_dim=24,
            decoder_layers=1, decoder_heads=4,
            state_layers=1, action_layers=1, fuse_layers=1,
        ),
    )
    cfg.optimizer.use_muon = False
    cfg.compile.enabled = False
    return cfg


def load_frozen_agent(weights: Path) -> tuple[Agent, str, bool]:
    random.seed(91_001); np.random.seed(91_001); torch.manual_seed(91_001)
    cfg = build_config()
    agent = Agent(cfg, enable_value_head=False).eval()
    created = False
    if weights.exists():
        state = torch.load(weights, map_location="cpu", weights_only=True)
        agent.load_state_dict(state, strict=True)
    else:
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.state_dict(), weights)
        created = True
    return agent, sha256(weights), created


def measured(backend: str, cases, agent: Agent) -> dict[str, object]:
    gc.collect(); process=psutil.Process(); cpu0=process.cpu_times(); rss0=process.memory_info().rss
    started=time.perf_counter()
    with PeakRss() as peak:
        payload=run_agent_lane(backend,cases=cases,agent=agent)
    wall=time.perf_counter()-started;cpu1=process.cpu_times()
    payload.update({
        "measured_wall_seconds":wall,
        "process_cpu_seconds":cpu1.user-cpu0.user+cpu1.system-cpu0.system,
        "rss_before_bytes":rss0,"peak_rss_bytes":peak.peak,
        "rss_delta_peak_bytes":max(0,peak.peak-rss0),
    })
    return payload


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    walls=[float(x["measured_wall_seconds"]) for x in samples];median=statistics.median(walls)
    rep=min(samples,key=lambda x:abs(float(x["measured_wall_seconds"])-median))
    return {
        "samples":samples,"median_wall_seconds":median,
        "median_actions_per_second":statistics.median(int(x["actions"])/float(x["measured_wall_seconds"]) for x in samples),
        "wall_min_seconds":min(walls),"wall_max_seconds":max(walls),"wall_stdev_seconds":statistics.pstdev(walls),
        "actions":int(rep["actions"]),"completed_games":int(rep["completed_games"]),
        "trajectory_digest":rep["trajectory_digest"],
        "representative_segments_seconds":rep["segments_seconds"],
        "peak_rss_bytes_max":max(int(x["peak_rss_bytes"]) for x in samples),
    }


def run_pair(cases, agent: Agent, repetitions: int) -> dict[str, object]:
    # Untimed warmup for both backends.
    run_agent_lane("python_reference",cases=cases[:1],agent=agent)
    run_agent_lane("cpp",cases=cases[:1],agent=agent)
    collected={"python_reference":[],"cpp":[]};orders=[]
    for repetition in range(repetitions):
        order=["python_reference","cpp"] if repetition%2==0 else ["cpp","python_reference"]
        orders.append(order)
        for backend in order:collected[backend].append(measured(backend,cases,agent))
    py=summarize(collected["python_reference"]);cpp=summarize(collected["cpp"])
    digests={x["trajectory_digest"] for xs in collected.values() for x in xs};counts={int(x["actions"]) for xs in collected.values() for x in xs}
    if len(digests)!=1 or len(counts)!=1:raise AssertionError("paired workload digest/count mismatch")
    return {"python_reference":py,"cpp":cpp,"speedup":py["median_wall_seconds"]/cpp["median_wall_seconds"],"matched_digest":next(iter(digests)),"matched_actions":next(iter(counts)),"orders":orders}


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);parser.add_argument("--weights",required=True);parser.add_argument("--label",required=True);parser.add_argument("--repetitions",type=int,default=5);args=parser.parse_args()
    if args.repetitions<5:parser.error("formal benchmark requires at least five repetitions")
    torch.set_num_threads(1)
    weights=Path(args.weights).resolve();agent,weight_hash,created=load_frozen_agent(weights)
    primary=[build_case(i) for i in PRIMARY_INDICES];generalization=[build_case(i) for i in GENERALIZATION_INDICES]
    model_cfg=build_config().model
    manifest={
        "schema":"guandan-marvel-m05-frozen-workload-manifest-v1","model":asdict(model_cfg),
        "weights_path":str(weights),"weights_sha256":weight_hash,"weights_created_this_run":created,
        "primary_cases":[c.to_json() for c in primary],"generalization_cases":[c.to_json() for c in generalization],
        "primary_deals_sha256":canonical_hash([c.to_json() for c in primary]),"generalization_deals_sha256":canonical_hash([c.to_json() for c in generalization]),
        "sampling":"greedy temperature=0","mode":"eval+inference_mode","horizon":512,"batch_size":1,"device":"cpu","dtype":"float32","torch_num_threads":torch.get_num_threads(),
        "timing_scope":"stable rollout including per-game environment/sampler setup and cache preparation; imports, weight creation/loading and warmup excluded",
    }
    result={
        "schema":SCHEMA,"label":args.label,"timestamp_unix":time.time(),"source_commit":git_head(),"manifest":manifest,"manifest_sha256":canonical_hash(manifest),
        "environment":{"platform":platform.platform(),"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"cpu":platform.processor(),"physical_cpus":psutil.cpu_count(logical=False),"logical_cpus":psutil.cpu_count()},
        "primary":run_pair(primary,agent,args.repetitions),"generalization":run_pair(generalization,agent,args.repetitions),
    }
    result["primary"]["passes_2x"]=result["primary"]["speedup"]>=2.0
    result["generalization"]["cpp_faster"]=result["generalization"]["speedup"]>1.0
    result["valid_workload"]=bool(result["primary"]["matched_actions"]==1027 and result["primary"]["matched_digest"] and result["generalization"]["cpp_faster"])
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n",encoding='utf-8')
    print(json.dumps({"label":args.label,"primary_speedup":result["primary"]["speedup"],"generalization_speedup":result["generalization"]["speedup"],"primary_actions":result["primary"]["matched_actions"],"weights_sha256":weight_hash,"output":str(output)},sort_keys=True))
    return 0 if result["valid_workload"] else 2

if __name__=="__main__":raise SystemExit(main())
