"""Small token-level PPO-style smoke trainer for A05."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from ..env_batch import GuandanEnvBatch
from ..environment_types import GameConfig
from ..model import PolicyValueNet
from .checkpoint import CURRENT_PROTOCOL_VERSIONS, load_checkpoint, save_checkpoint
from .config import load_profile

@dataclass(frozen=True, slots=True)
class TrainingResult:
    algorithm: str; profile: str; updates: int; total_token_steps: int; last_loss: float; checkpoint: str | None

def _inputs(obs, device):
    return (torch.as_tensor(obs.observation_tokens, dtype=torch.long, device=device), torch.as_tensor(obs.state_channels, dtype=torch.float32, device=device), torch.as_tensor(obs.legal_next_tokens, dtype=torch.long, device=device), torch.as_tensor(obs.legal_next_mask, dtype=torch.bool, device=device))

def train(*, algorithm="ippo", profile="local_fast", device=None, updates=None, checkpoint_dir="runs", resume=None, hidden_dim=32, seed=0, rollout_steps=None, rollout_envs=None):
    algorithm=str(algorithm).lower()
    if algorithm not in {"ippo","vrpo"}: raise ValueError("algorithm must be ippo or vrpo")
    cfg=load_profile(profile); requested=device or cfg.device; requested=("cuda" if requested=="auto" and torch.cuda.is_available() else "cpu" if requested=="auto" else requested)
    device_obj=torch.device(requested)
    if device_obj.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(seed); np.random.seed(seed)
    env_count=int(rollout_envs or cfg.rollout_envs); steps=int(rollout_steps or (cfg.token_steps_per_update // env_count)); total=int(updates or cfg.updates_per_algorithm)
    model=PolicyValueNet(hidden_dim=int(hidden_dim)).to(device_obj); optimizer=torch.optim.Adam(model.parameters(),lr=3e-4); start=0
    if resume:
        loaded=load_checkpoint(resume,model=model,optimizer=optimizer,expected_profile=profile,expected_algorithm=algorithm,map_location=device_obj); start=loaded.metadata.update_count
    batch=GuandanEnvBatch([GameConfig(seed=seed+i) for i in range(env_count)],auto_reset=True); obs=batch.reset(seeds=[seed+i for i in range(env_count)])
    out=Path(checkpoint_dir)/profile/algorithm; out.mkdir(parents=True,exist_ok=True); last_loss=0.0; total_steps=0
    for update in range(start+1,start+total+1):
        rows=[]
        for _ in range(steps):
            x=_inputs(obs,device_obj); logits,values=model(*x); actions=model.sample_actions(*x,deterministic=False); logp=model.log_prob(logits,actions,legal_next_tokens=x[2]); step=batch.step(actions.detach().cpu().numpy().astype(np.int32)); player=np.clip(obs.player_id.astype(np.int64),0,3); reward=torch.as_tensor(step.rewards[np.arange(env_count),player],dtype=torch.float32,device=device_obj); done=torch.as_tensor(step.done|step.truncated,dtype=torch.float32,device=device_obj); rows.append((x,actions.detach(),logp.detach(),values.detach(),reward,done)); obs=step.observation; total_steps+=env_count
        returns=[]; running=torch.zeros(env_count,device=device_obj)
        for x,actions,oldlogp,oldvalues,reward,done in reversed(rows): running=reward+0.99*running*(1-done); returns.append((x,actions,oldlogp,oldvalues,running.detach()))
        optimizer.zero_grad(set_to_none=True); losses=[]
        for x,actions,oldlogp,oldvalues,target in reversed(returns):
            logits,values=model(*x); newlogp=model.log_prob(logits,actions,legal_next_tokens=x[2]); adv=(target-oldvalues).detach(); ratio=torch.exp(newlogp-oldlogp); policy=-torch.minimum(ratio*adv,torch.clamp(ratio,.8,1.2)*adv).mean(); value=torch.nn.functional.mse_loss(values,target); entropy=model.entropy(logits).mean(); losses.append(policy+.5*value-.01*entropy)
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss): raise FloatingPointError("non-finite A05 loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); last_loss=float(loss.detach().cpu())
        if update%cfg.checkpoint_interval==0 or update==start+total:
            save_checkpoint(out/f"{algorithm}_update_{update:06d}.pt",model=model,optimizer=optimizer,profile=profile,algorithm=algorithm,update=update,seed=seed,protocol_versions=CURRENT_PROTOCOL_VERSIONS)
    return TrainingResult(algorithm,profile,total,total_steps,last_loss,str(out/f"{algorithm}_update_{start+total:06d}.pt"))

