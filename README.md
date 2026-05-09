# Sparse-keyframe stylization + RIFE interpolation pipeline

This repo extracts real video frames, stylizes sparse keyframes through ComfyUI,
interpolates gaps with Practical-RIFE, and muxes frames back to mp4.

It supports two generation modes:

- **SDXL mode**: `sdxl_img2img_canny` workflow (legacy baseline).
- **FLUX mode**: multi-reference workflow template with `current + prev + anchor`
  frame conditioning for stronger temporal consistency.

## Layout

```
pipeline/
  config.py     # pipeline dataclass + defaults
  extract.py    # 1) ffmpeg extraction
  keyframes.py  # 2) sparse keyframe manifest + anchor metadata
  stylize.py    # 3) ComfyUI generation over keyframes
  rife.py       # 4) gap interpolation
  mux.py        # 5) mp4 assembly
  run.py        # CLI orchestrator
workflows/
  sdxl_img2img_canny.json      # SDXL + ControlNet workflow template
  flux_img2img_multi_ref.json  # FLUX-style multi-reference template scaffold
docs/
  flux_frame_consistency_migration.md
scripts/
  run_pipeline.sh
```

## Prerequisites

1. `ffmpeg` available in `PATH`.
2. ComfyUI running on `127.0.0.1:8188` with your target workflow/model assets.
3. Practical-RIFE at `third_party/Practical-RIFE` (or override via flags).
4. Python deps: `pip install -r requirements.txt`.

## 5s gate, then full render

```bash
INPUT=/data/source.mp4 STRIDE=20 MODEL_FAMILY=flux \
WORKFLOW=workflows/flux_img2img_multi_ref.json \
./scripts/run_pipeline.sh
```

Direct CLI (test gate):

```bash
python -m pipeline.run \
  --input /data/source.mp4 \
  --work ./work \
  --output ./out/test_flux.mp4 \
  --model-family flux \
  --workflow workflows/flux_img2img_multi_ref.json \
  --stride 20 \
  --segment-anchor-interval 120 \
  --denoise 0.22 \
  --test
```

## Key consistency knobs

| Flag | Default | Why it matters |
|------|---------|----------------|
| `--model-family` | `sdxl` | Selects SDXL or FLUX-oriented workflow defaults. |
| `--stride` | `4` | Keyframe spacing; higher lowers cost, increases interpolation load. |
| `--segment-anchor-interval` | `120` | Long-term anchor cadence for scene consistency. |
| `--denoise` | `0.25` | Keep low (0.15-0.35) to reduce flicker/drift. |
| `--seed` | `1234` | Keep fixed per segment/job for stability. |
| `--no-prev-reference` | off | Disable short-term (`t-1`) reference conditioning. |
| `--no-anchor-reference` | off | Disable long-term anchor reference conditioning. |

## FLUX migration notes

See `docs/flux_frame_consistency_migration.md` for:

- SDXL -> FLUX migration strategy.
- Reference-image policy (`current + prev + anchor`).
- Scale strategy for 10k-20k frame sequences.
- Workflow simplification guidelines.
