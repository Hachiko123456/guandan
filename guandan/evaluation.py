"""A06 local fixed-deal evaluation with an actual A05 snapshot agent."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import json,time,hashlib
from pathlib import Path
from .action_state import StepwiseActionState
from .cards import Rank
from .environment import GameConfig,GuandanEnv
from .evaluation_agents import make_agent
from .execution_profiles import load_execution_profile
from .round import RoundState
from .state import TEAM_OF
from .training.runtime import EVALUATION_SEED_BASE
from .evaluation_stats import summarize_games
class EvaluationIncomplete(RuntimeError):
 def __init__(self,report): self.report=report;super().__init__(str(report))
@dataclass(frozen=True)
class GameRecord:
 opponent:str;deal_group:int;seat_rotation:int;policy_seat:int;candidate_team:int;seed:int;done:bool;truncated:bool;ranking:list;winner_team:int;outcome_class:str;rewards:list;team_rewards:list;token_steps:int;committed_steps:int;base_deal_sha256:str;terminal:bool
@dataclass(frozen=True)
class EvaluationResult:
 profile:str;status:str;target_games_per_opponent:int;completed_games_per_opponent:dict;total_games:int;records:tuple;runtime_seconds:float;seed_namespace:int;summary:dict;errors:tuple=()
 def as_dict(self):
  x=asdict(self);x['records']=[asdict(r) for r in self.records];return x
def _snapshot_path():
 files=sorted(Path('project_status/history').glob('*/A05/execution/local_fast/ippo/*update_000005*.pt'))
 if not files:raise FileNotFoundError('A05 update5 IPPO checkpoint is required')
 return files[-1]
def _rotate(base,rotation):return RoundState.from_hands([list(base.hands[(i-rotation)%4]) for i in range(4)],level_rank=Rank.FIVE,leader_seat=rotation)
def play_game(*,opponent,deal_group,seat_rotation,seed,snapshot_path):
 base=RoundState.deal(seed=seed,level_rank=Rank.FIVE);state=_rotate(base,seat_rotation);env=GuandanEnv(GameConfig(level_rank=int(Rank.FIVE),leader_seat=seat_rotation));obs=env.reset(initial_state=StepwiseActionState(state).serialize());agents={seat_rotation:make_agent('snapshot',checkpoint=snapshot_path,seed=seed+seat_rotation)}
 for seat in range(4):
  if seat==seat_rotation:continue
  agents[seat]=make_agent('rule') if TEAM_OF[seat]==TEAM_OF[seat_rotation] else make_agent(opponent,checkpoint=snapshot_path,seed=seed+seat*1009+deal_group*10007)
 result=None
 for _ in range(4096):
  if env.done:break
  result=env.step(agents[int(obs.player_id)].select_token(obs));obs=result.observation
 if not env.done:raise RuntimeError(f'evaluation game did not terminate: {opponent}/{deal_group}/{seat_rotation}')
 fs=env.full_state()['round_state'];digest=hashlib.sha256(json.dumps([sorted(card.card_id for card in hand) for hand in base.hands],separators=(',',':')).encode()).hexdigest();return GameRecord(opponent,deal_group,seat_rotation,seat_rotation,seat_rotation%2,seed,True,False,list(fs['ranking']),int(fs['winner_team']),fs['outcome_class'],list(result.rewards.tolist()),list(fs['team_rewards']),int(result.token_step),int(result.committed_step),digest,True)
def evaluate_profile(*,profile='local_fast',seed=EVALUATION_SEED_BASE,max_hours=None,evidence_path=None):
 r=load_execution_profile(Path(__file__).resolve().parents[1],profile);c=r['config']['evaluation'];g=int(c['deal_groups_per_pairing']);rots=tuple(c['seat_rotations']);ops=tuple(c['opponents']);target=g*len(rots);budget=float(c['max_hours'] if max_hours is None else max_hours);start=time.monotonic();snap=_snapshot_path();records=[]
 for op in ops:
  for group in range(g):
   for rot in rots:
    if time.monotonic()-start>=budget*3600:raise EvaluationIncomplete({'status':'incomplete','completed':len(records),'target':target*len(ops)})
    records.append(play_game(opponent=op,deal_group=group,seat_rotation=rot,seed=int(seed+group*100),snapshot_path=snap))
 counts={op:sum(x.opponent==op for x in records) for op in ops};summary=summarize_games([asdict(x) for x in records]);res=EvaluationResult(profile,'complete',target,counts,len(records),tuple(records),time.monotonic()-start,int(seed),summary)
 if evidence_path:Path(evidence_path).write_text(json.dumps(res.as_dict(),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 return res
__all__=['EvaluationIncomplete','EvaluationResult','GameRecord','evaluate_profile','play_game','EVALUATION_SEED_BASE']




