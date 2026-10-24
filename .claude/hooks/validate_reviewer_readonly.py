#!/usr/bin/env python3
"""Claude Code PreToolUse guard for the phase-reviewer Bash tool.

Exit 0 allows the command. Exit 2 blocks it and sends the reason to Claude.
This is a deterministic guard against common source-writing commands; it is
not an operating-system sandbox.
"""
from __future__ import annotations

import json
import re
import sys


def block(reason: str) -> "NoReturn":
    print(f"Reviewer guard blocked a potentially mutating command: {reason}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        block(f"invalid hook JSON ({type(exc).__name__})")

    if payload.get("tool_name") != "Bash":
        return 0

    command = str(payload.get("tool_input", {}).get("command", ""))
    if not command.strip():
        block("empty Bash command")

    patterns: tuple[tuple[str, str], ...] = (
        (r"(?im)(^|[;&|]\s*)git\s+(reset|clean|checkout|restore|commit|push|add|apply|am|merge|rebase|cherry-pick|revert|switch)\b", "Git mutation"),
        (r"(?im)(^|[;&|]\s*)git\s+branch\s+.*\s-[dD]\b", "Git branch deletion"),
        (r"(?im)(^|[;&|]\s*)git\s+tag\s+-d\b", "Git tag deletion"),
        (r"(?im)\b(Set-Content|Add-Content|Out-File|Clear-Content|Remove-Item|Move-Item|Copy-Item|Rename-Item|New-Item|Start-Transcript)\b", "PowerShell filesystem mutation"),
        (r"(?im)(^|[;&|]\s*)(rm|rmdir|del|erase|mv|cp|mkdir|touch|truncate|install)\b", "shell filesystem mutation"),
        (r"(?im)(^|[;&|]\s*)(chmod|chown|icacls|takeown|mklink)\b", "permission or link mutation"),
        (r"(?im)\bsed\s+[^\r\n]*\s-i(?:\s|$)", "in-place sed"),
        (r"(?im)\bperl\s+[^\r\n]*\s-pi(?:\s|$)", "in-place perl"),
        (r"(?im)(^|[^<])>>?\s*(?!\$null\b|/dev/null\b)", "file redirection"),
        (r"(?im)\|\s*(tee|Set-Content|Add-Content|Out-File)\b", "write pipeline"),
        (r"(?im)\b(pip|pip3|uv|conda)\s+install\b", "dependency installation"),
        (r"(?im)\b(npm|pnpm|yarn)\s+(install|add|remove|update|run\s+build)\b", "package or build mutation"),
        (r"(?im)\bpython(?:\.exe)?\s+[^\r\n]*scripts[/\\]build[^\s]*\.py\b", "official build script"),
        (r"(?im)\b(write_text|write_bytes|WriteAllText|WriteAllBytes|torch\.save|shutil\.(copy|copy2|copytree|move|rmtree)|os\.(remove|unlink|rename|replace|mkdir|makedirs)|Path\([^\)]*\)\.mkdir)\b", "programmatic filesystem mutation"),
        (r"(?im)\bopen\s*\([^\r\n]*,[^\r\n]*[\"'][wax+][^\"']*[\"']", "write-mode open"),
    )

    for pattern, reason in patterns:
        if re.search(pattern, command):
            block(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())