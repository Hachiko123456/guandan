# SPDX-License-Identifier: MIT
"""DMC Gate L notebook driver (stage 5).

Shipped inside the Gate L delivery payload (``driver.zip``).  The package is
self-contained on the Kaggle runtime: it may import the overlay sources
(``guandan``, ``guandan_marvel``) and numpy/torch only — never the workspace
``tools`` tree, the serial-compare driver or any native extension (Gate L runs
the explicit ``python_reference`` backend; there is no C++ fallback).

:mod:`dmc_gate_l.runner` is the notebook entry point (``run_notebook``);
:mod:`dmc_gate_l.evaluate` implements the pre-declared ε=0 greedy evaluation
lanes against the shipped rule/random row agents.  The evaluation-package
compatibility shim must be installed before anything imports
``guandan_marvel.dmc`` (whose sampler reaches into
``guandan_marvel.evaluation.agents``).
"""

from ._compat import install_python_reference_evaluation_shim

install_python_reference_evaluation_shim()

from .runner import run_notebook

__all__ = ["run_notebook"]
