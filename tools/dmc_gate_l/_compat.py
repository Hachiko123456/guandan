# SPDX-License-Identifier: MIT
"""Import compatibility for the python_reference-only Gate L runtime.

``guandan_marvel.evaluation`` (the M06 evaluation/pool/promotion package)
eagerly imports the C++ backend through ``checkpoints`` — ``from
``guandan_marvel import cpp_game`` at module level — and the DMC sampler
imports ``guandan_marvel.evaluation.agents`` for the row-agent classes and the
action-token column.  On a machine with the native extension this is harmless;
on the Gate L Kaggle runtime (pure Python, no native) the plain import chain
crashes before any training starts.

Gate L uses exactly one piece of that package: :mod:`...evaluation.agents`
(the shipped rule/random row heuristics — never a reimplementation).  When the
native backend is unavailable this module registers a minimal stand-in for the
package that:

* loads the **real** ``agents.py`` from the overlay verbatim;
* exposes loud placeholder checkpoint-artifact classes (the driver never
  constructs one; any attempt fails immediately instead of faking);
* never provides pool, promotion, manifest or evaluator machinery — those
  boundaries are disabled by the pre-declared plan.

When the native backend IS available (local development/test machines) the
shim is not installed at all: the real package imports normally and no test
process is ever polluted by stand-ins.  The whole chain is checked by
``tests/dmc/test_gate_l_driver.py::test_driver_import_chain_never_touches_cpp``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def install_python_reference_evaluation_shim() -> bool:
    """Install the evaluation-package stand-in when (and only when) needed.

    Returns ``True`` if the shim was installed.  ``False`` means the real
    package is (or will be) importable as-is.
    """
    if "guandan_marvel.evaluation" in sys.modules:
        return False
    try:
        import guandan_marvel.cpp_game  # noqa: F401  # capability probe only
    except Exception:
        pass  # no native backend (Gate L Kaggle runtime): install the shim
    else:
        return False  # native backend available: the real package imports fine

    import guandan_marvel.config as _config  # pure module; locates the overlay tree

    evaluation_dir = Path(_config.__file__).resolve().parent / "evaluation"

    package = types.ModuleType("guandan_marvel.evaluation")
    package.__path__ = [str(evaluation_dir)]
    package.__doc__ = "Gate L python_reference shim: lane agents only."
    sys.modules["guandan_marvel.evaluation"] = package

    checkpoints = types.ModuleType("guandan_marvel.evaluation.checkpoints")
    checkpoints.__doc__ = (
        "Gate L shim: policy-artifact loading is unavailable on the "
        "python_reference runtime; the driver never constructs artifacts."
    )

    class _UnavailableArtifact:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise NotImplementedError(
                "evaluation checkpoint artifacts are unavailable on the Gate L "
                "python_reference runtime; the driver never constructs them"
            )

    checkpoints.V2PolicyArtifact = _UnavailableArtifact
    checkpoints.LegacyV1OpponentArtifact = _UnavailableArtifact
    checkpoints.EVALUATION_CHECKPOINT_SCHEMA = "guandan-marvel-evaluation-checkpoint-v1"
    sys.modules["guandan_marvel.evaluation.checkpoints"] = checkpoints

    spec = importlib.util.spec_from_file_location(
        "guandan_marvel.evaluation.agents", evaluation_dir / "agents.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the shipped lane agents from {evaluation_dir / 'agents.py'}")
    agents = importlib.util.module_from_spec(spec)
    sys.modules["guandan_marvel.evaluation.agents"] = agents
    spec.loader.exec_module(agents)
    return True


__all__ = ["install_python_reference_evaluation_shim"]
