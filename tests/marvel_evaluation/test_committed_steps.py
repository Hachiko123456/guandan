from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from guandan.state import PreviousHandResult
from guandan.environment_types import GameConfig
from guandan_marvel import cpp_game
from guandan_marvel.evaluation.host import validated_commit_flag


def test_commit_marker_matches_available_action_flag_for_prefix_commit_pass_and_exchange():
    cfg=GameConfig(level_rank=5,seed=77,leader_seat=0,previous_result=PreviousHandResult((1,4,3,2),0,'HEAD_THIRD'))
    env=cpp_game.EnvBatch(1,configs=[cfg],auto_reset=False);step=env.reset(77)
    seen={'noncommit':False,'commit':False,'pass':False,'exchange_commit':False}
    committed=0;tokens=0
    for _ in range(4096):
        debug=env.debug_state(0);phase=int(debug['state']['phase'])
        legal=np.flatnonzero(step.available_actions.weights[0]>0)
        for index in legal:
            value=validated_commit_flag(step,int(index));row=int(step.available_actions.tokens[0,index,6])
            assert value==(row==int(cpp_game.ACTION_COMMIT))
        choice=None
        for index in legal:
            token=int(step.available_actions.tokens[0,index,5])
            if token==int(cpp_game.PASS_TOKEN): choice=int(index);seen['pass']=True;break
        if choice is None:
            for index in legal:
                token=int(step.available_actions.tokens[0,index,5])
                if token==int(cpp_game.COMMIT_TOKEN): choice=int(index);break
        if choice is None:choice=int(legal[0])
        token=int(step.available_actions.tokens[0,choice,5]);is_commit=validated_commit_flag(step,choice)
        seen['noncommit']|=not is_commit;seen['commit']|=is_commit
        if phase in {1,2} and is_commit:seen['exchange_commit']=True
        committed+=int(is_commit);tokens+=1
        row=np.asarray(step.available_actions.tokens[0,choice],dtype=np.int64)
        step=env.step(row.reshape(1,-1))
        if bool(step.is_terminal[0]):break
    else:raise AssertionError('game did not terminate')
    terminal=env.debug_state(0)['state']
    assert committed==int(terminal['finish_committed_step'])
    assert tokens==int(terminal['finish_token_step'])
    assert committed<tokens
    assert all(seen.values()),seen


def test_commit_marker_mismatch_fails_explicitly():
    tokens=np.zeros((1,1,8),dtype=np.int64);tokens[0,0,6]=int(cpp_game.ACTION_COMMIT)
    step=SimpleNamespace(available_actions=SimpleNamespace(tokens=tokens,is_commit=np.asarray([[False]])))
    with pytest.raises(RuntimeError,match='disagrees'):
        validated_commit_flag(step,0)
