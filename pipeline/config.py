"""Pipeline configuration.

A single dataclass that fixes resolution/FPS/stride/denoise across the chain.
Keep these locked between the 5s test render and the full render so the test
is honest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # ---- I/O ----
    input_video: Path
    work_dir: Path                 # all intermediate artifacts live here
    output_video: Path

    # ---- Locked working format (must match across test + full) ----
    fps: int = 30
    width: int = 1280
    height: int = 720

    # ---- Sparse keyframes ----
    stride: int = 4                # S in the plan: AI runs on frames 0, S, 2S...
    segment_anchor_interval: int = 120  # long-term scene anchor cadence in timeline frames

    # ---- Test gate ----
    test_mode: bool = False        # if True, only run on a 5s excerpt
    test_start_sec: float = 30.0
    test_duration_sec: float = 5.0

    # ---- Stylization ----
    model_family: str = "sdxl"     # "sdxl" (ControlNet path) or "flux" (multi-ref path)
    comfy_host: str = "127.0.0.1:8188"
    comfy_workflow: Path = Path("workflows/sdxl_img2img_canny.json")
    comfy_ckpt: str = "sd_xl_base_1.0.safetensors"
    comfy_controlnet: str = "controlnet-canny-sdxl-1.0.safetensors"
    comfy_input_dir: Path = Path("")  # ComfyUI's "input" folder; if empty we POST uploads
    comfy_output_dir: Path = Path("")  # ComfyUI's "output" folder; if empty we pull via /history
    prompt: str = (
        "cinematic dashcam highway shot, golden hour, photorealistic, "
        "sharp lane markings, accurate vehicles, film grade"
    )
    negative_prompt: str = (
        "blurry, warped, melting, extra wheels, deformed cars, text, watermark"
    )
    denoise: float = 0.25          # 0.15-0.35 per plan
    cfg: float = 5.5
    steps: int = 22
    sampler: str = "dpmpp_2m_sde"
    scheduler: str = "karras"
    seed: int = 1234               # held fixed for temporal stability

    # ---- ControlNet (Canny) ----
    canny_low: float = 0.31
    canny_high: float = 0.71
    controlnet_strength: float = 0.85

    # ---- Temporal conditioning ----
    prev_frame_blend: float = 0.35  # how much of previous stylized to mix into init latent
    use_prev_reference: bool = True
    use_anchor_reference: bool = True

    # ---- RIFE ----
    rife_repo: Path = Path("third_party/Practical-RIFE")
    rife_model: Path = Path("third_party/Practical-RIFE/train_log")
    python_bin: str = "python"

    # ---- Mux ----
    copy_audio: bool = True
    crf: int = 17
    pix_fmt: str = "yuv420p"
    video_codec: str = "libx264"

    # ---- Derived paths (filled by ensure_layout) ----
    paths: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def ensure_layout(self) -> None:
        """Create the working subfolders and remember their paths."""
        wd = Path(self.work_dir)
        layout = {
            "test_clip":   wd / "test_5s.mp4",
            "frames":      wd / "frames",        # all extracted real frames
            "keyframes":   wd / "keyframes",     # input copies for stylization
            "stylized":    wd / "stylized",      # AI output for keyframes
            "interp":      wd / "interp",        # final timeline (stylized + RIFE)
            "manifest":    wd / "keyframes.json",
            "log":         wd / "pipeline.log",
        }
        for key, p in layout.items():
            if key in {"test_clip", "manifest", "log"}:
                p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
        self.paths = {k: str(v) for k, v in layout.items()}

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else Path(self.work_dir) / "config.json"
        data = asdict(self)
        for k, v in list(data.items()):
            if isinstance(v, Path):
                data[k] = str(v)
        path.write_text(json.dumps(data, indent=2))
        return path
