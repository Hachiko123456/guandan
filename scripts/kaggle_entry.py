"""Standard-library-only bootstrap for the main-owned deployment session CLI."""
from __future__ import annotations

from pathlib import Path
import sys


def bootstrap_root() -> Path:
    """Use this script's repository, never the caller's cwd or an input guess."""
    root = Path(__file__).resolve().parents[1]
    required = (
        root / "pyproject.toml",
        root / "configs" / "acceptance_profiles.json",
        root / "guandan" / "__init__.py",
        root / "guandan" / "deployment" / "session.py",
    )
    if any(not path.is_file() or not path.resolve().is_relative_to(root) for path in required):
        raise RuntimeError("incomplete source package: expected the deployment session and profile beside this script")
    # A notebook may already have imported a different mounted source tree.
    # Fail instead of combining two package versions in one interpreter.
    for name, module in tuple(sys.modules.items()):
        if name == "guandan" or name.startswith("guandan."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root / "guandan"):
                raise RuntimeError("another guandan package is loaded; restart the kernel or use a fresh Python process")
    sys.path[:] = [str(root), *(item for item in sys.path if item != str(root))]
    sys.dont_write_bytecode = True  # also safe when invoked from read-only source
    return root


def main(argv=None) -> int:
    bootstrap_root()
    from guandan.deployment.session import main as session_main

    return session_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
