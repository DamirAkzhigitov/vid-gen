"""End-to-end CLI orchestrator.

Usage:

    python -m pipeline.run \\
        --input /path/to/source.mp4 \\
        --work  /path/to/work_dir \\
        --output /path/to/final.mp4 \\
        --stride 4 --fps 30 --width 1280 --height 720 \\
        --denoise 0.25 --test         # <- start with --test (5s gate)

Drop ``--test`` only after the test render looks acceptable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import extract, keyframes, mux, rife, stylize
from .config import PipelineConfig
from .utils import get_logger


def _default_workflow(model_family: str) -> Path:
    if model_family == "flux":
        return Path("workflows/flux_img2img_multi_ref.json")
    return Path("workflows/sdxl_img2img_canny.json")


def _default_ckpt(model_family: str) -> str:
    if model_family == "flux":
        return "flux1-dev.safetensors"
    return "sd_xl_base_1.0.safetensors"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)

    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--segment-anchor-interval", type=int, default=120,
                   help="Long-term keyframe anchor cadence in timeline frames")

    p.add_argument("--test", action="store_true",
                   help="Run the gate: only process a 5s excerpt")
    p.add_argument("--test-start", type=float, default=30.0)
    p.add_argument("--test-duration", type=float, default=5.0)

    p.add_argument("--denoise", type=float, default=0.25)
    p.add_argument("--steps", type=int, default=22)
    p.add_argument("--cfg", type=float, default=5.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--neg-prompt", type=str, default=None)
    p.add_argument("--prev-blend", type=float, default=0.35)
    p.add_argument("--model-family", choices=["sdxl", "flux"], default="sdxl")
    p.add_argument("--no-prev-reference", action="store_true",
                   help="Disable t-1 reference conditioning in stylize stage")
    p.add_argument("--no-anchor-reference", action="store_true",
                   help="Disable long-term scene anchor reference conditioning")

    p.add_argument("--comfy-host", default="127.0.0.1:8188")
    p.add_argument("--comfy-ckpt", default=None,
                   help="Checkpoint filename in ComfyUI's models dir "
                        "(defaults by model family)")
    p.add_argument("--comfy-controlnet",
                   default="controlnet-canny-sdxl-1.0.safetensors")
    p.add_argument("--workflow", type=Path, default=None,
                   help="ComfyUI workflow template (defaults by model family)")

    p.add_argument("--rife-repo", type=Path,
                   default=Path("third_party/Practical-RIFE"))
    p.add_argument("--rife-model", type=Path,
                   default=Path("third_party/Practical-RIFE/train_log"))

    p.add_argument("--no-audio", action="store_true")

    p.add_argument("--skip", nargs="*", default=[],
                   choices=["extract", "keyframes", "stylize", "rife", "mux"],
                   help="Skip stages (useful when iterating on later stages)")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    workflow = args.workflow or _default_workflow(args.model_family)
    ckpt = args.comfy_ckpt or _default_ckpt(args.model_family)
    cfg = PipelineConfig(
        input_video=args.input,
        work_dir=args.work,
        output_video=args.output,
        fps=args.fps, width=args.width, height=args.height,
        stride=args.stride,
        segment_anchor_interval=args.segment_anchor_interval,
        test_mode=args.test,
        test_start_sec=args.test_start,
        test_duration_sec=args.test_duration,
        denoise=args.denoise, steps=args.steps, cfg=args.cfg, seed=args.seed,
        prev_frame_blend=args.prev_blend,
        model_family=args.model_family,
        use_prev_reference=not args.no_prev_reference,
        use_anchor_reference=not args.no_anchor_reference,
        comfy_host=args.comfy_host,
        comfy_workflow=workflow,
        comfy_ckpt=ckpt,
        comfy_controlnet=args.comfy_controlnet,
        rife_repo=args.rife_repo,
        rife_model=args.rife_model,
        copy_audio=not args.no_audio,
    )
    if args.prompt:
        cfg.prompt = args.prompt
    if args.neg_prompt:
        cfg.negative_prompt = args.neg_prompt
    cfg.ensure_layout()
    cfg.save()
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = build_config(args)
    log = get_logger("run", log_file=Path(cfg.paths["log"]))
    log.info("mode = %s", "TEST (5s)" if cfg.test_mode else "FULL")

    skip = set(args.skip)
    source_used = cfg.input_video

    if "extract" not in skip:
        source_used, n = extract.run_stage(cfg)
    else:
        n = sum(1 for _ in Path(cfg.paths["frames"]).glob("*.png"))
        source_used = Path(cfg.paths["test_clip"]) if cfg.test_mode else cfg.input_video

    if "keyframes" not in skip:
        keyframes.build_manifest(cfg, n)

    if "stylize" not in skip:
        stylize.run_stage(cfg)

    if "rife" not in skip:
        rife.run_stage(cfg)

    if "mux" not in skip:
        mux.run_stage(cfg, source_for_audio=source_used)

    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
