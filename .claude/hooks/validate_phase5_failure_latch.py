"""PreToolUse: deny implementer Bash while a managed phase-5 failure is latched.

This is a workflow guard, not an operating-system sandbox. The coordinator may
acknowledge only after explicit human confirmation; the implementer cannot use
Bash to clear its own failure and retry behind the user's back.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    if payload.get("tool_name") not in {"Bash", "Edit", "Write", "NotebookEdit"}:
        return 0
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
    latch = root / "runs" / "phase5" / "execution_guard" / "failure.json"
    if not latch.exists():
        return 0
    try:
        record = json.loads(latch.read_text(encoding="utf-8"))
        identifier = record["failure_id"]
    except Exception:
        identifier = "unreadable failure record (fail closed)"
    print(f"Phase5 failed and awaits direct user confirmation. Failure: {identifier}. "
          f"Read {latch}, report the failure, and STOP. Do not clear the latch, "
          "edit guard code/state, change commands or switch agents to retry.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
