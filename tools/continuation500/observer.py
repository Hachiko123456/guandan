"""Only observation changes. The frozen trainer, sampler and optimizer are untouched."""
from dataclasses import replace
import time
import torch
import torch.distributed as dist
from guandan_marvel.kaggle.pilot_observability import PilotObserver as LegacyObserver
from .common import atomic_json

class ContinuationObserver(LegacyObserver):
    authorized_cap=500
    plan_hash=None
    def __init__(self,profile,trainer,root,rank_root):
        self.enabled=bool((profile.get('pilot_observability')or{}).get('enabled',False))
        self.trainer,self.root,self.rank_root=trainer,root,rank_root
        self.start=0.;self.updates=self.skips=0;self.probe=None;self.stage_seconds={};self.stage_counts={}
        if not self.enabled:raise ValueError('continuation observability cannot be disabled')
        if not 1<=trainer.cfg.total_iterations<=self.authorized_cap:raise ValueError('observer cap exceeds signed continuation plan')
        if trainer.algorithm!='ippo':raise ValueError('continuation supports the frozen IPPO recipe only')
        previous=trainer.sampler.config.timing_callback
        def record(name,seconds):
            self.stage_seconds[name]=self.stage_seconds.get(name,0.)+float(seconds)
            self.stage_counts[name]=self.stage_counts.get(name,0)+1
            if previous is not None:previous(name,seconds)
        trainer.sampler.config=replace(trainer.sampler.config,timing_callback=record)
        d=trainer.device
        atomic_json(rank_root/'pilot_runtime.json',{'schema':'guandan-continuation-observability-v1','rank':trainer.rank,
            'local_rank':trainer.local_rank,'world_size':trainer.world_size,'device':str(d),
            'communication_backend':str(dist.get_backend())if dist.is_initialized()else None,
            'object_collective_backend':trainer.distributed.object_collective_backend,
            'gpu_name':torch.cuda.get_device_name(d)if d.type=='cuda'else None,
            'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
            'amp':trainer.amp_enabled,'trim_completed_padding':trainer.cfg.trim_completed_padding,
            'training_semantics':trainer.training_semantics_profile,'source_hash':trainer.source_hash,
            'plan_hash':self.plan_hash,'authorized_iteration_cap':self.authorized_cap})
