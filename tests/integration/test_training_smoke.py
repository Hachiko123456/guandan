from __future__ import annotations

from pathlib import Path

from guandan.training.trainer import train


def test_a05_one_update_and_resume(tmp_path: Path) -> None:
    first = train(algorithm="ippo", profile="local_fast", device="cpu", updates=1, rollout_envs=1, rollout_steps=1, hidden_dim=16, checkpoint_dir=tmp_path, seed=11)
    assert first.last_loss == first.last_loss
    assert Path(first.checkpoint).is_file()
    second = train(algorithm="ippo", profile="local_fast", device="cpu", updates=1, rollout_envs=1, rollout_steps=1, hidden_dim=16, checkpoint_dir=tmp_path, seed=11, resume=first.checkpoint)
    assert second.updates == 1
    assert Path(second.checkpoint).is_file()
