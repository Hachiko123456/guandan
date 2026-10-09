"""Non-comparative baseline profiler for M05 performance finish."""
from __future__ import annotations
import argparse,cProfile,io,json,pstats,sys,time
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))
import torch
from guandan_marvel.model import Agent
from guandan_marvel.testing.differential import build_case
from scripts.benchmark_marvel_m05_finish import build_config
from scripts.benchmark_marvel_m05_repair import run_agent_lane

def profile_backend(backend,case,agent):
 pr=cProfile.Profile();pr.enable();started=time.perf_counter()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as tp:
  result=run_agent_lane(backend,cases=[case],agent=agent)
 wall=time.perf_counter()-started;pr.disable();stream=io.StringIO();pstats.Stats(pr,stream=stream).sort_stats('cumtime').print_stats(40)
 ops=[]
 for item in tp.key_averages().table(sort_by='self_cpu_time_total',row_limit=40).splitlines():ops.append(item)
 return {'backend':backend,'wall_seconds_profiled':wall,'segments_seconds':result['segments_seconds'],'actions':result['actions'],'trajectory_digest':result['trajectory_digest'],'cprofile_top':stream.getvalue(),'torch_operator_table':ops}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weights',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(1);agent=Agent(build_config(),enable_value_head=False).eval();agent.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True),strict=True);case=build_case(50000)
 result={'schema':'guandan-marvel-m05-performance-profile-v1','note':'Profiling overhead; not used for speedup ratios.','python_reference':profile_backend('python_reference',case,agent),'cpp':profile_backend('cpp',case,agent)}
 p=Path(a.output).resolve();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8');print(p);return 0
if __name__=='__main__':raise SystemExit(main())
