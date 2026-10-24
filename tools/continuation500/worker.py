"""Resume original V4 training math; versioned observational adapter only."""
from pathlib import Path
import argparse,sys
from .common import read_json,validate_plan,run_contract

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--continuation-plan',type=Path,required=True)
    parser.add_argument('--local-test',action='store_true')
    known,remaining=parser.parse_known_args(argv)
    plan=validate_plan(read_json(known.continuation_plan),testing=known.local_test)
    if known.local_test!=bool(plan.get('test_only')):raise ValueError('test/production worker mismatch')
    if not known.local_test:
        from .pipeline import verify_driver_files
        verify_driver_files(plan)
    def value(flag):return remaining[remaining.index(flag)+1]
    profile=read_json(value('--profile'))
    if profile.get('mode')!='resume'and not known.local_test:raise ValueError('production continuation may never warm-start/retrain')
    controls=profile.get('resume',profile.get('training'))
    target=int(controls['target_iterations'])
    if not 0<target<=plan['target_iteration']:raise ValueError('training target exceeds signed cap')
    if profile.get('ddp',{}).get('world_size')!=2:raise ValueError('requires both DDP ranks')
    if not known.local_test and ('--warm-start-checkpoint'in remaining or '--local-cpu-test'in remaining):raise ValueError('forbidden fallback/reinitialization')
    if profile.get('object_collectives_backend')!='gloo':raise ValueError('metadata side group must stay explicit')
    from .observer import ContinuationObserver
    import guandan_marvel.kaggle.worker as original
    ContinuationObserver.authorized_cap=plan['target_iteration'];ContinuationObserver.plan_hash=plan['plan_hash']
    original.PilotObserver=ContinuationObserver
    return original.main(remaining)
if __name__=='__main__':raise SystemExit(main())
