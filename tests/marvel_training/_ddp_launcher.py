"""Two-rank gloo launcher helper for the phase-2 macro DDP tests."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def launch_two_ranks(
    script: str | Path,
    *args: str,
    cwd: str | Path,
    timeout: float = 900.0,
) -> list[subprocess.CompletedProcess]:
    """Run ``script`` twice with rank env vars for torch.distributed env:// init."""
    port = free_port()
    environment = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": "2",
        "PYTHONPATH": str(cwd),
    }
    processes = []
    for rank in range(2):
        env = dict(os.environ)
        env.update(environment)
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        processes.append(subprocess.Popen(
            [sys.executable, str(script), *args, "--rank", str(rank)],
            cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        ))
    results = []
    deadline = time.time() + timeout
    for process in processes:
        remaining = max(deadline - time.time(), 1.0)
        try:
            output, _ = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            raise TimeoutError(f"DDP rank timed out; partial output:\n{output}")
        results.append(subprocess.CompletedProcess(process.args, process.returncode, output))
    return results
