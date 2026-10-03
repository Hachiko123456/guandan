from __future__ import annotations

from pathlib import Path

from guandan.training.trainer import train


def test_a05_local_fast_smoke(tmp_path: Path) -> None:
    result = train(algorithm="vrpo", profile="local_fast", device="cpu", updates=1, rollout_envs=1, rollout_steps=1, hidden_dim=16, checkpoint_dir=tmp_path, seed=17)
    assert result.total_token_steps == 1
    assert Path(result.checkpoint).is_file()
