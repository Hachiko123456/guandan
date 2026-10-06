from __future__ import annotations

import torch

from guandan.model import PolicyValueNet
from guandan.training.checkpoint import save_checkpoint
from guandan.training.snapshot import export_policy_snapshot, load_policy_snapshot


def _checkpoint(path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = torch.nn.ModuleDict({'actor': PolicyValueNet(hidden_dim=8), 'critic': PolicyValueNet(hidden_dim=8)})
    save_checkpoint(
        path, model, optimizer=None, update=4, seed=0, git_commit=None,
        model_config={
            'architecture': 'mean-pooled-token-MLP', 'hidden_dim': 8,
            'critic': 'decentralized-V-shared-teams', 'vocabulary': 256, 'channels': 256,
        },
    )
    return model


def test_export_policy_snapshot_extracts_actor_without_optimizer(tmp_path):
    checkpoint = tmp_path / 'training.pt'
    model = _checkpoint(checkpoint)
    snapshot = tmp_path / 'policy.snapshot.pt'
    result = export_policy_snapshot(checkpoint, snapshot)
    payload = torch.load(snapshot, weights_only=True)
    assert result['policy_snapshot_version'] == 1
    assert payload['policy_snapshot_version'] == 1
    assert 'optimizer_state' not in payload
    loaded = load_policy_snapshot(snapshot)
    assert set(loaded.model_state) == set(model['actor'].state_dict())
    for key, value in model['actor'].state_dict().items():
        assert torch.equal(value, loaded.model_state[key])


def test_export_is_atomic_and_refuses_legacy_fixture(tmp_path):
    legacy = tmp_path / 'legacy.pt'
    torch.save({'checkpoint_version': 2, 'model_state': {}}, legacy)
    try:
        export_policy_snapshot(legacy, tmp_path / 'bad.snapshot.pt')
    except Exception as exc:
        assert 'migration' in str(exc).lower() or 'contract' in str(exc).lower()
    else:
        raise AssertionError('legacy checkpoint unexpectedly exported')
    assert not (tmp_path / 'bad.snapshot.pt').exists()
