"""Explicit MARVEL C++ game backend.

Importing this module requires the built pybind extension and never falls back
to :mod:`guandan_marvel.game`, which remains the Python reference oracle and
the default backend until M05 acceptance.
"""

from nanopokerlib.guandan import *  # noqa: F401,F403
