"""Pre-phase-3 behaviour shim (evidence only, never part of the normal suite).

Reinstates the exact behaviour the phase-3 tests are meant to be counterexamples
for:

* the actor has **no** residual fuse branch (``Agent.actor_fuse is None``),
* ``amp_enabled=True`` is rejected outright
  (``ValueError: AMP is explicitly disabled for the first M03 smoke profile``),
* the strict evaluation loader only accepts the ``v1.3`` architecture.

Run it explicitly, e.g.::

    python -m pytest tests/marvel_training/test_macro_fuse.py \
        tests/marvel_training/test_macro_fuse_checkpoints.py \
        tests/marvel_training/test_macro_fuse_amp.py \
        -p tests.marvel_training._phase3_old_impl_plugin -q

The phase-3 tests must fail under this shim; under the real (phase-3) code they
pass.
"""
from __future__ import annotations

from guandan_marvel.model import agent as agent_module
from guandan_marvel.evaluation import checkpoints as evaluation_module
from guandan_marvel.training import local as local_module

ACTIVE = False


def pytest_configure(config) -> None:
    global ACTIVE
    ACTIVE = True

    original_agent_init = agent_module.Agent.__init__

    def _agent_without_fuse(self, *args, **kwargs):
        original_agent_init(self, *args, **kwargs)
        # Pre-phase-3 actor: the residual fuse branch does not exist.
        self.actor_fuse = None

    agent_module.Agent.__init__ = _agent_without_fuse

    original_trainer_init = local_module.LocalTrainerBase.__init__

    def _trainer_without_amp(self, cfg, **kwargs):
        if getattr(cfg, "amp_enabled", False):
            raise ValueError("AMP is explicitly disabled for the first M03 smoke profile")
        original_trainer_init(self, cfg, **kwargs)

    local_module.LocalTrainerBase.__init__ = _trainer_without_amp
    evaluation_module.SUPPORTED_POLICY_ARCHITECTURES = ("v1.3",)


def pytest_unconfigure(config) -> None:
    del config
