"""Stage 5: mux the final PNG sequence (and optional source audio) to mp4."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .utils import get_logger, run


def run_stage(cfg: PipelineConfig, source_for_audio: Path) -> Path:
    log = get_logger("mux")
    interp_dir = Path(cfg.paths["interp"])
    out = Path(cfg.output_video)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(cfg.fps),
        "-i", str(interp_dir / "%06d.png"),
    ]
    if cfg.copy_audio:
        cmd += ["-i", str(source_for_audio),
                "-map", "0:v", "-map", "1:a?",
                "-c:a", "copy", "-shortest"]
    cmd += [
        "-c:v", cfg.video_codec,
        "-crf", str(cfg.crf),
        "-pix_fmt", cfg.pix_fmt,
        str(out),
    ]
    run(cmd, log=log)
    log.info("wrote %s", out)
    return out
