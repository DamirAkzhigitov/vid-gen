# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Sparse-keyframe RIFE video stylization pipeline. A single Python package (`pipeline/`) that extracts frames from video, selects sparse keyframes, stylizes them via ComfyUI (SDXL + Canny ControlNet), fills gaps with Practical-RIFE interpolation, and muxes back to mp4. See `README.md` for the full layout and CLI flags.

### Known issue: `config.py` location

`config.py` lives at the repo root but `pipeline/` imports it as `from .config import PipelineConfig`. A symlink `pipeline/config.py -> ../config.py` is required. The update script creates this automatically.

### Running the pipeline

```bash
# CLI help
python -m pipeline.run --help

# Run only extract + keyframes (no GPU/ComfyUI needed)
python -m pipeline.run \
  --input /path/to/video.mp4 \
  --work ./work --output ./out/test.mp4 \
  --fps 30 --width 1280 --height 720 --stride 4 \
  --skip stylize rife mux --no-audio
```

The `stylize`, `rife`, and `mux` stages require external services not available in Cloud Agent VMs:
- **ComfyUI** on `127.0.0.1:8188` with SDXL checkpoint + Canny ControlNet
- **Practical-RIFE** cloned to `third_party/Practical-RIFE` with model weights
- **GPU/CUDA** for both ComfyUI and RIFE inference

### Linting

```bash
ruff check .
```

There are 2 pre-existing lint warnings (unused variable in `rife.py`, unused import in `stylize.py`).

### Testing

No automated test suite exists. The `--test` CLI flag renders a 5-second excerpt for human visual inspection; it is not an automated test.
