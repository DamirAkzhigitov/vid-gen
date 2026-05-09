"""Stage 4: fill the gaps between stylized keyframes with RIFE.

We invoke Practical-RIFE's ``inference_img.py`` once per (k/S) timestep
between every consecutive *stylized* keyframe pair (A, B) and assemble
a tight 1..N PNG sequence in ``interp/``.

This avoids the ncnn-vulkan "directory mode + large -n" duplication
footgun mentioned in the plan: each intermediate slot is requested
explicitly with ``--ratio k/S``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tqdm import tqdm

from .config import PipelineConfig
from .keyframes import load_manifest
from .utils import get_logger, run, frame_path


def _interp_one(cfg: PipelineConfig, img0: Path, img1: Path,
                ratio: float, dst: Path) -> None:
    """Run one RIFE call: produce a single intermediate frame at ``ratio``."""
    cmd = [
        cfg.python_bin, "inference_img.py",
        "--img", str(img0), str(img1),
        "--ratio", f"{ratio:.6f}",
        "--output", str(dst),
        "--model", str(cfg.rife_model),
    ]
    run(cmd, cwd=str(cfg.rife_repo))


def run_stage(cfg: PipelineConfig) -> int:
    """Assemble the full timeline in ``interp/``. Returns frames written."""
    log = get_logger("rife")
    manifest = load_manifest(cfg)
    S = manifest["stride"]
    total = manifest["total_frames"]
    keyframes = manifest["keyframes"]

    interp_dir = Path(cfg.paths["interp"])
    stylized_dir = Path(cfg.paths["stylized"])
    for old in interp_dir.glob("*.png"):
        old.unlink()

    # 1. Drop every stylized keyframe at its true timeline index.
    for entry in keyframes:
        ord_ = entry["keyframe_ordinal"]
        ti = entry["timeline_index"]
        shutil.copy2(frame_path(stylized_dir, ord_),
                     frame_path(interp_dir, ti))

    # 2. Walk consecutive keyframe pairs and synthesize the gap.
    pairs = list(zip(keyframes[:-1], keyframes[1:]))
    for a, b in tqdm(pairs, desc="rife"):
        ord_a, ord_b = a["keyframe_ordinal"], b["keyframe_ordinal"]
        ti_a, ti_b = a["timeline_index"], b["timeline_index"]
        gap = ti_b - ti_a
        if gap <= 1:
            continue
        img0 = frame_path(stylized_dir, ord_a)
        img1 = frame_path(stylized_dir, ord_b)
        for k in range(1, gap):
            ratio = k / gap          # works correctly even on the trailing
                                     # short pair (gap may be < S)
            dst = frame_path(interp_dir, ti_a + k)
            _interp_one(cfg, img0, img1, ratio, dst)

    written = sum(1 for _ in interp_dir.glob("*.png"))
    if written != total:
        log.warning("expected %d frames in interp/, got %d", total, written)
    else:
        log.info("interpolated full timeline: %d frames", written)
    return written
