# SPDX-License-Identifier: MIT
"""Versioned 250-iteration three-parameter experiment; never launches training."""
from __future__ import annotations
import ast
import base64
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import zipfile
from .common import atomic_json, canonical_hash, relative_file_manifest, sha256_file
from .package import SOURCE_ARCHIVE_NAME, build_source_archive, validate_source_archive
from .phase4_delivery import _training_block
from ..context_schedule import (
    CONTEXT_SCHEDULE_SCHEMA_V2,
    DEFAULT_LEADER_SEATS,
    DEFAULT_LEVEL_RANKS,
    _previous_payload,
    validate_context_schedule,
)

SCHEMA = "guandan-marvel-t4x2-combo-pilot-v1"
EXPERIMENT_ID = "guandan-marvel-v14-t4x2-combo-20261021-v1"
CAP_ITERATIONS = 250
COLLECT_GAMES = 40
EVALUATION_DEALS = 128
EVALUATION_SEED_BASE = 1_810_210_000
WALL_SECONDS = 12_600
EVALUATION_RESERVE_SECONDS = 5_400
PARENT_SHA256 = "1f4de9ad180398c3041e6737588a4f92f59e2e97d07d5868e9066efbee725eda"
PAYLOAD_NAME = "guandan_t4x2_combo_v1.bin"
NOTEBOOK_NAME = "guandan_t4x2_combo_v1.ipynb"
RESULTS_NAME = "guandan_t4x2_combo_results.zip"
TRIBUTE_PREVIOUS_RESULTS = [
    None,
    {"ranking": [1, 3, 2, 4], "winner_team": 0, "outcome_class": "DOUBLE_DOWN", "tribute_cancelled": False},
    {"ranking": [1, 2, 3, 4], "winner_team": 0, "outcome_class": "HEAD_THIRD", "tribute_cancelled": False},
    {"ranking": [1, 2, 4, 3], "winner_team": 0, "outcome_class": "HEAD_LAST", "tribute_cancelled": True},
]
RECIPE = {
    "clip": 0.2, "reg": 0.01, "lr": 0.0001,
    "schedules": "constant within this experiment",
    "rollout_global": 48, "collect_global": 40, "learner_batch_per_rank": 20,
    "tribute": True, "ntp": True, "belief": False, "symmetry": False, "pool": False, "search": False,
    "comparison": "single new-recipe branch vs frozen parent; NOT a matched old-recipe training A/B",
}


def _build_full_tribute_schedule() -> dict[str, object]:
    """Full Cartesian product: 13 levels x 4 leaders x 4 tribute modes = 208 contexts."""
    contexts = []
    for leader in DEFAULT_LEADER_SEATS:
        for level in DEFAULT_LEVEL_RANKS:
            for prev in TRIBUTE_PREVIOUS_RESULTS:
                contexts.append({
                    "level_rank": int(level),
                    "leader_seat": int(leader),
                    "previous_result": _previous_payload(prev),
                    "weight": 1,
                })
    schedule = {"schema": CONTEXT_SCHEDULE_SCHEMA_V2, "contexts": contexts}
    validate_context_schedule(schedule, allow_tribute=True)
    return schedule


def make_combo_profiles():
    training = deepcopy(_training_block())
    training.update({"seed": 580700, "rollout_batch_size": 48, "collect_batch_size": COLLECT_GAMES,
                     "base_lr": 0.0001, "base_clip_coef": 0.2, "base_reg_coef": 0.01,
                     "ntp_weight": 0.02,
                     "schedule": {"agent_lr_decay_power": 0.0, "critic_lr_decay_power": 0.0,
                                  "clip_coef_decay_power": 0.0, "reg_coef_decay_power": 0.0},
                     "trim_completed_padding": True, "target_iterations": 2,
                     "checkpoint_interval": 50, "evaluation_interval": 0,
                     "max_wall_seconds": 1800, "max_wall_seconds_per_session": 1800,
                     "checkpoint_keep_last": 1, "evaluation_keep_last": 1})
    training["training_context_schedule"] = _build_full_tribute_schedule()
    # Parent had already completed 50 iterations of the original 20+80 curriculum.
    # New optimizer/counters reset; curriculum progression must not reset silently.
    training["training_context_curriculum"]["iteration_offset"] = 50
    common = {"name": EXPERIMENT_ID, "algorithm": "ippo", "mode": "full",
              "ddp": {"world_size": 2, "backend": "nccl", "require_cuda": True,
                      "find_unused_parameters": False},
              "object_collectives_backend": "gloo", "checkpoint_io": {"compression_level": 1},
              "heartbeat_interval_seconds": 5, "telemetry_interval_seconds": 5,
              "pilot_observability": {"enabled": True, "cpu_threads": 2,
                  "maximum_iteration": CAP_ITERATIONS, "profile_rollout_stages": True}}
    profiles = {"precheck": {**deepcopy(common), "training": training}}
    for name, target in (("resume_check", 3), ("train", CAP_ITERATIONS)):
        profiles[name] = {**deepcopy(common), "mode": "resume", "expected_algorithm": "ippo",
            "resume": {"target_iterations": target, "checkpoint_interval": 50,
                       "evaluation_interval": 0, "checkpoint_keep_last": 1,
                       "evaluation_keep_last": 1, "max_wall_seconds": 10800}}
    return profiles


def validate_parent_policy(payload, *, sha256: str):
    from guandan_marvel.config import deserialize_train_config
    if sha256 != PARENT_SHA256:
        raise ValueError("parent must be the verified uploaded pilot candidate_iter000050.pt")
    if not isinstance(payload, dict) or payload.get("schema") != "guandan-marvel-evaluation-checkpoint-v1":
        raise ValueError("parent must be a policy-only evaluation checkpoint")
    if payload.get("algorithm") != "ippo" or payload.get("architecture") != "v1.4-macro-fuse" or payload.get("iteration") != 50:
        raise ValueError("parent algorithm/architecture/iteration mismatch")
    cfg = deserialize_train_config(payload.get("resolved_config"))
    state = payload.get("agent_state_dict")
    if (cfg is None or cfg.model.history_dim != 128 or cfg.model.feature_dim != 256
            or not isinstance(state, dict) or len(state) != 90
            or sum(t.numel() for t in state.values()) != 2_783_234):
        raise ValueError("parent is not the production-scale trained v1.4 policy")
    distributed = payload.get("distributed_training") or {}
    if distributed.get("global_counters", {}).get("completed_games_total") != 1000:
        raise ValueError("parent must carry verified 1000-game global counters")
    provenance = payload.get("training_data_provenance") or {}
    if not provenance.get("complete") or not provenance.get("deal_hashes"):
        raise ValueError("parent must carry complete actual training provenance")
    return {"schema": "guandan-marvel-combo-parent-v1", "checkpoint_sha256": sha256,
            "architecture": payload["architecture"], "source_iteration": 50,
            "source_completed_games": 1000, "parameter_count": 2_783_234, "state_dict_keys": 90,
            "optimizer_reset": True, "sampler_reset": True, "counters_reset": True,
            "curriculum_iteration_offset": 50, "source_hash": payload.get("source_hash"),
            "training_deal_hashes": sorted(set(provenance["deal_hashes"])),
            "initial_env_seeds": sorted(set(provenance.get("initial_env_seeds", [])))}


def validate_combo_manifest(root):
    from guandan_marvel.buffer import trajectory_memory_budget
    from guandan_marvel.game import GAME_SPEC
    from guandan_marvel.evaluation.manifests import validate_deals_manifest, deals_for
    root = Path(root).resolve()
    d = json.loads((root / "pilot_manifest.json").read_text(encoding="utf-8"))
    if d.get("schema") != SCHEMA or d.get("pilot_id") != EXPERIMENT_ID:
        raise ValueError("unsupported combo experiment identity")
    if d.get("manifest_hash") != canonical_hash({k:v for k,v in d.items() if k!="manifest_hash"}):
        raise ValueError("combo manifest hash mismatch")
    names=[]
    for row in d["files"]:
        name=str(row["path"])
        if not name or name in (".","..") or any(c in name for c in ("/","\\",":","\0")):
            raise ValueError("combo inputs must be flat regular files")
        raw=root/name; path=raw.resolve();names.append(name.casefold())
        if (raw.is_symlink() or not path.is_relative_to(root) or not path.is_file()
                or path.stat().st_size!=row["bytes"] or sha256_file(path)!=row["sha256"]):
            raise ValueError(f"combo input content mismatch: {name}")
    if len(names)!=len(set(names)):
        raise ValueError("duplicate/case-colliding combo inputs")
    required={SOURCE_ARCHIVE_NAME,"warm_start_policy.pt","parent_run.json","development_deals.json",*(f"{k}_profile.json"for k in make_combo_profiles())}
    if not required.issubset({row['path'] for row in d['files']}):
        raise ValueError("combo manifest omits required frozen inputs")
    if d["profiles"] != {k:f"{k}_profile.json" for k in make_combo_profiles()}:
        raise ValueError("combo profile bindings changed")
    profiles={k:json.loads((root/v).read_text(encoding="utf-8"))for k,v in d['profiles'].items()}
    if profiles != make_combo_profiles() or d.get("recipe") != RECIPE:
        raise ValueError("combo profiles differ from frozen three-parameter/performance recipe")
    if d.get("training_cap") != {"absolute_iterations":250,"completed_games":10000,"global_collect_batch_size":40}:
        raise ValueError(f"combo cap must remain {CAP_ITERATIONS} iterations / {CAP_ITERATIONS*COLLECT_GAMES} games")
    if d.get("wall_seconds_cap")!=WALL_SECONDS or d.get("evaluation_reserve_seconds")!=EVALUATION_RESERVE_SECONDS:
        raise ValueError("combo wall/evaluation budget mismatch")
    if d.get("fabledan")!="NOT_RUN" or d.get("final_holdout")!="NOT_TOUCHED" or d.get("results_archive_name")!=RESULTS_NAME:
        raise ValueError("combo evaluation/output boundary mismatch")
    parent=json.loads((root/"parent_run.json").read_text(encoding="utf-8"))
    if (sha256_file(root/"warm_start_policy.pt")!=PARENT_SHA256 or d['warm_start']['sha256']!=PARENT_SHA256
            or parent['checkpoint_sha256']!=PARENT_SHA256 or d.get('parent_run')!={k:v for k,v in parent.items()if k not in ('training_deal_hashes','initial_env_seeds')}):
        raise ValueError("combo parent identity mismatch")
    evaluation=json.loads((root/"development_deals.json").read_text(encoding="utf-8"));validate_deals_manifest(evaluation)
    selection=deals_for(evaluation,'selection')
    if (len(selection)!=EVALUATION_DEALS or deals_for(evaluation,'training') or deals_for(evaluation,'final')
            or evaluation['hash']!=d['evaluation_manifest_hash'] or d['expected_evaluation_games']!=1536
            or {r['seed']for r in selection}!=set(range(EVALUATION_SEED_BASE,EVALUATION_SEED_BASE+EVALUATION_DEALS))
            or any(r.get('previous_result')is not None for r in selection)):
        raise ValueError("combo development/final dataset contract mismatch")
    if sha256_file(root/SOURCE_ARCHIVE_NAME)!=d['source']['archive_sha256']:
        raise ValueError("combo source identity mismatch")
    budget=trajectory_memory_budget(GAME_SPEC,rollout_batch_size=48,collect_batch_size=40,world_size=2)
    if d.get('sampler_budget')!=budget or budget['sampler_bytes_per_rank']>4_294_967_296:
        raise ValueError("combo sampler budget mismatch/overflow")
    return d


def render_combo_notebook(repository: Path, *, payload_hash: str, manifest_hash: str):
    from .pilot_delivery import render_notebook
    script = repository / "scripts/repackage_t4x2_pilot_transport.py"
    spec = importlib.util.spec_from_file_location("_combo_transport_builder", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    notebook = module.render_transport_notebook(render_notebook(manifest_hash), payload_hash, manifest_hash)
    cell=next(c for c in notebook['cells']if c.get('cell_type')=='code')
    code=''.join(cell['source']).replace("TRANSPORT_FILENAME = 'pilot_payload_v3.bin'",f"TRANSPORT_FILENAME = {PAYLOAD_NAME!r}")
    code=code.replace('guandan_t4x2_pilot_v3','guandan_t4x2_combo_v1').replace('guandan_t4x2_pilot_results.zip',RESULTS_NAME)
    ast.parse(code);cell['source']=code.splitlines(keepends=True)
    notebook['cells'][0]['source']=[
        '# 掼蛋 T4×2 三参数组合实验 v1\n',
        f'**Dataset 只上传 `{PAYLOAD_NAME}`；本 Notebook 单独导入。**\n',
        '选择 GPU T4×2；无需联网。从已核验 iter50 权重 warm start，不继承旧优化器。\n',
        'clip=0.2 / reg=0.01 / lr=1e-4，均固定；含贡还上下文、NTP=0.02。\n',
        f'2 iter → 恢复到3 → 本轮绝对{CAP_ITERATIONS} iter，最多新增{CAP_ITERATIONS*COLLECT_GAMES}局；开发评估1536局。\n',
        '总安全预算3.5小时，预留90分钟评估；预算不足先保存，绝不自动超额。\n',
        '梯度使用双T4/NCCL；Gloo只传元数据，不是CPU训练回退。\n',
        '如失败，下载bootstrap诊断、日志和最近recovery；不要修改哈希或绕过预检。\n']
    return notebook


def build_combo_delivery(repository, output, warm_start, *, validation_record=None):
    import torch
    from guandan_marvel.buffer import trajectory_memory_budget
    from guandan_marvel.game import GAME_SPEC
    from guandan_marvel.evaluation.manifests import build_deals_manifest
    repo,out,warm=Path(repository).resolve(),Path(output).resolve(),Path(warm_start).resolve()
    build=out.parent/(out.name+'_build')
    if out.exists() or build.exists():raise FileExistsError('choose new delivery/build paths; never overwrite history')
    digest=sha256_file(warm)
    if digest!=PARENT_SHA256:raise ValueError('unexpected parent policy SHA256')
    parent=validate_parent_policy(torch.load(warm,map_location='cpu',weights_only=True),sha256=digest)
    dataset=build/'dataset';dataset.mkdir(parents=True)
    source=build_source_archive(repo,dataset/SOURCE_ARCHIVE_NAME,allow_dirty=True)
    validate_source_archive(dataset/SOURCE_ARCHIVE_NAME)
    shutil.copy2(warm,dataset/'warm_start_policy.pt');atomic_json(dataset/'parent_run.json',parent)
    for k,p in make_combo_profiles().items():atomic_json(dataset/f'{k}_profile.json',p)
    evaluation=build_deals_manifest(counts={'training':0,'selection':128,'final':0},
        bases={'training':0,'selection':EVALUATION_SEED_BASE,'final':2_000_000_000},
        level_ranks=tuple(range(3,16)),leader_seats=(0,1,2,3),previous_results=(None,))
    atomic_json(dataset/'development_deals.json',evaluation)
    if validation_record is not None:
        record=json.loads(Path(validation_record).read_text(encoding='utf-8'))
        for row in record.get('source_files',[]):
            p=(repo/row['path']).resolve()
            if not p.is_relative_to(repo) or sha256_file(p)!=row['sha256']:raise ValueError('stale local validation: '+row['path'])
        record['delivery_source_archive_sha256']=source['archive_sha256'];record['kaggle_t4x2_executed']=False
        atomic_json(dataset/'LOCAL_VALIDATION.json',record)
    body={'schema':SCHEMA,'pilot_id':EXPERIMENT_ID,'recipe':RECIPE,
        'training_cap':{'absolute_iterations':250,'completed_games':10000,'global_collect_batch_size':40},
        'wall_seconds_cap':WALL_SECONDS,'evaluation_reserve_seconds':EVALUATION_RESERVE_SECONDS,
        'source':{'archive_sha256':source['archive_sha256'],'content_manifest_hash':source['source_manifest']['manifest_hash'],
                  'git_commit':source['source_manifest']['git_commit'],'workspace_snapshot':source['source_manifest']['workspace_snapshot']},
        'warm_start':{'file':'warm_start_policy.pt','sha256':digest},
        'parent_run':{k:v for k,v in parent.items()if k not in ('training_deal_hashes','initial_env_seeds')},
        'profiles':{k:f'{k}_profile.json'for k in make_combo_profiles()},
        'sampler_budget':trajectory_memory_budget(GAME_SPEC,rollout_batch_size=48,collect_batch_size=40,world_size=2),
        'evaluation_manifest_hash':evaluation['hash'],'expected_evaluation_games':1536,'evaluation_role':'development_only',
        'results_archive_name':RESULTS_NAME,'fabledan':'NOT_RUN','final_holdout':'NOT_TOUCHED',
        'files':relative_file_manifest(dataset,[p for p in dataset.iterdir()if p.is_file()]),'remote_verified':False}
    manifest={**body,'manifest_hash':canonical_hash(body)};atomic_json(dataset/'pilot_manifest.json',manifest)
    validate_combo_manifest(dataset)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True)as z:
        for p in sorted(dataset.iterdir()):z.write(p,p.name)
    # Reuse the already-tested V3 *wire protocol*, not the old experiment contract.
    payload=b'GUANDAN-PILOT-TRANSPORT-V3\n'+base64.b64encode(stream.getvalue())+b'\n'
    if len(payload)>64*1024*1024 or zipfile.is_zipfile(io.BytesIO(payload)):raise ValueError('invalid/oversize opaque transport')
    out.mkdir();(out/PAYLOAD_NAME).write_bytes(payload)
    payload_hash=sha256_file(out/PAYLOAD_NAME)
    atomic_json(out/NOTEBOOK_NAME,render_combo_notebook(repo,payload_hash=payload_hash,manifest_hash=manifest['manifest_hash']))
    shutil.copy2(repo/'docs/t4x2_combo_pilot.md',out/'README_KAGGLE.md')
    if validation_record is not None:shutil.copy2(dataset/'LOCAL_VALIDATION.json',out/'LOCAL_VALIDATION.json')
    verification={'schema':'guandan-marvel-combo-delivery-verification-v1','status':'BUILT_NOT_RUN_ON_KAGGLE',
        'payload':PAYLOAD_NAME,'payload_bytes':len(payload),'payload_sha256':payload_hash,
        'notebook':NOTEBOOK_NAME,'notebook_sha256':sha256_file(out/NOTEBOOK_NAME),
        'inner_manifest_hash':manifest['manifest_hash'],'source_archive_sha256':source['archive_sha256'],
        'parent_policy_sha256':digest,'training_launched':False,'remote_verified':False}
    atomic_json(out/'DELIVERY_VERIFICATION.json',verification)
    return verification
