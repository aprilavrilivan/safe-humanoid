"""Run Isaac Sim smoke checks with a pseudo-terminal on headless Linux.

Kit can exit successfully during startup without completing a test when its
standard streams are redirected to pipes. ``script`` supplies a PTY while
still allowing unittest to capture and verify the smoke test's completion.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path


def run_simulator(command: list[str], *, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    script = shutil.which("script")
    if script is None:
        raise RuntimeError("util-linux 'script' is required for GPU simulator tests")
    return subprocess.run(
        [script, "-q", "-e", "-c", shlex.join(command), "/dev/null"],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
