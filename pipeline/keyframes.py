"""Stage 2: build the sparse keyframe index map.

Frames are 1-indexed PNGs (``000001.png``...). For stride S we keep
indices ``1, 1+S, 1+2S, ...`` and *always* add the final frame so the
last RIFE gap is well-defined.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import PipelineConfig
from .utils import get_logger, frame_path


def build_manifest(cfg: PipelineConfig, total_frames: int) -> dict:
    log = get_logger("keyframes")
    S = cfg.stride
    indices = list(range(1, total_frames + 1, S))
    if indices[-1] != total_frames:
        indices.append(total_frames)

    frames_dir = Path(cfg.paths["frames"])
    kf_dir = Path(cfg.paths["keyframes"])
    for old in kf_dir.glob("*.png"):
        old.unlink()

    pairs = []  # list of (timeline_index, keyframe_ordinal)
    for ord_, idx in enumerate(indices):
        src = frame_path(frames_dir, idx)
        dst = frame_path(kf_dir, ord_)  # ordinal so ComfyUI sees a tight 0..N-1 range
        if not src.exists():
            raise FileNotFoundError(f"missing extracted frame: {src}")
        shutil.copy2(src, dst)
        pairs.append({"timeline_index": idx, "keyframe_ordinal": ord_})

    manifest = {
        "stride": S,
        "total_frames": total_frames,
        "fps": cfg.fps,
        "width": cfg.width,
        "height": cfg.height,
        "keyframes": pairs,
    }
    Path(cfg.paths["manifest"]).write_text(json.dumps(manifest, indent=2))
    log.info("manifest: %d keyframes (stride %d, %d total frames)",
             len(pairs), S, total_frames)
    return manifest


def load_manifest(cfg: PipelineConfig) -> dict:
    return json.loads(Path(cfg.paths["manifest"]).read_text())
