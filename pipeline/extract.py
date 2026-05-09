"""Stage 1: cut optional 5s test clip and extract numbered PNG frames.

Locks resolution + FPS for the entire chain.
"""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .utils import get_logger, run


def cut_test_clip(cfg: PipelineConfig) -> Path:
    """Make a short test excerpt that matches production format.

    We deliberately *re-encode* the clip rather than ``-c copy`` so the
    extracted frames start exactly at the requested time and match the
    locked fps/size (``-c copy`` snaps to source keyframes).
    """
    log = get_logger("extract")
    out = Path(cfg.paths["test_clip"])
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{cfg.test_start_sec}",
        "-i", str(cfg.input_video),
        "-t", f"{cfg.test_duration_sec}",
        "-r", str(cfg.fps),
        "-vf", f"scale={cfg.width}:{cfg.height}:flags=lanczos",
        "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
        "-an",
        str(out),
    ]
    run(cmd, log=log)
    return out


def extract_frames(cfg: PipelineConfig, source: Path) -> int:
    """Decode source -> frames/000001.png ... at locked fps + size.

    Returns the number of frames written.
    """
    log = get_logger("extract")
    frames_dir = Path(cfg.paths["frames"])
    # Wipe any stale frames so indices stay deterministic.
    for old in frames_dir.glob("*.png"):
        old.unlink()
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-r", str(cfg.fps),
        "-vf", f"scale={cfg.width}:{cfg.height}:flags=lanczos",
        "-start_number", "1",
        str(frames_dir / "%06d.png"),
    ]
    run(cmd, log=log)
    n = sum(1 for _ in frames_dir.glob("*.png"))
    log.info("extracted %d frames to %s", n, frames_dir)
    return n


def run_stage(cfg: PipelineConfig) -> tuple[Path, int]:
    """Top-level entry: returns (source_used, frame_count)."""
    src = cut_test_clip(cfg) if cfg.test_mode else Path(cfg.input_video)
    n = extract_frames(cfg, src)
    return src, n
