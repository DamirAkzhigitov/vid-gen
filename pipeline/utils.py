"""Small shared helpers."""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path


def get_logger(name: str = "pipeline", log_file: Path | None = None) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def run(cmd: list[str] | str, *, check: bool = True, log: logging.Logger | None = None,
        cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess, stream output, raise on non-zero by default."""
    if isinstance(cmd, list):
        printable = " ".join(shlex.quote(c) for c in cmd)
    else:
        printable = cmd
    if log:
        log.info("$ %s", printable)
    proc = subprocess.run(cmd, shell=isinstance(cmd, str), check=check, cwd=cwd)
    return proc


def frame_path(folder: Path, idx: int, ext: str = "png", pad: int = 6) -> Path:
    return Path(folder) / f"{idx:0{pad}d}.{ext}"
