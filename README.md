# Sparse-keyframe RIFE pipeline

Production pipeline that takes **real footage**, stylizes only a sparse
set of keyframes through **ComfyUI** (SDXL img2img + Canny ControlNet),
then fills the gaps with **Practical-RIFE** and muxes back to mp4.

Design principle is non-negotiable: motion, perspective, and lane
geometry come from the **real pixels**. Diffusion only changes style at
controlled strength (`denoise 0.15-0.35`). RIFE turns expensive
diffusion frames into cheap interpolated frames. See
[`Sparse keyframes  RIFE-6e993a25.plan.md`](./Sparse%20keyframes%20%20RIFE-6e993a25.plan.md)
for the full rationale.

## Layout

```
pipeline/
  config.py     # locked fps/size/stride/denoise + working-dir layout
  extract.py    # 1. ffmpeg: optional 5s test clip + numbered PNG extract
  keyframes.py  # 2. stride S keyframe selection + manifest.json
  stylize.py    # 3. ComfyUI HTTP/WS client (sequential, prev-frame blend)
  rife.py       # 4. Practical-RIFE per-timestep gap fill
  mux.py        # 5. ffmpeg mux frames + optional source audio
  run.py        # CLI orchestrator
workflows/
  sdxl_img2img_canny.json   # ComfyUI API workflow template
scripts/
  run_pipeline.sh           # test-gate then full-render convenience wrapper
```

## Cloud GPU prerequisites

Target host is a rented GPU (e.g. vast.ai RTX 4090). On the instance:

1. **ffmpeg** in `$PATH`.
2. **ComfyUI** running on `127.0.0.1:8188` with an SDXL checkpoint and a
   Canny ControlNet for SDXL. Adjust `--comfy-ckpt` /
   `--comfy-controlnet` to match exact filenames in your
   `models/checkpoints` and `models/controlnet` folders.
3. **Practical-RIFE** cloned into `third_party/Practical-RIFE` with its
   `train_log/` model directory in place. Override locations with
   `--rife-repo` / `--rife-model`.
4. `pip install -r requirements.txt` for this orchestrator.

## Test-gate then full render

Always rehearse on a 5s excerpt before paying for the full render:

```bash
INPUT=/data/source.mp4 STRIDE=4 DENOISE=0.25 \
  ./scripts/run_pipeline.sh
```

Or invoke the orchestrator directly:

```bash
# 1. 5s test (gate)
python -m pipeline.run \
  --input /data/source.mp4 \
  --work  ./work \
  --output ./out/test_5s_out.mp4 \
  --stride 4 --fps 30 --width 1280 --height 720 \
  --denoise 0.25 --test

# 2. After visually approving test_5s_out.mp4, drop --test:
python -m pipeline.run \
  --input /data/source.mp4 \
  --work  ./work \
  --output ./out/final.mp4 \
  --stride 4 --fps 30 --width 1280 --height 720 \
  --denoise 0.25
```

## Key knobs (held constant between test and full)

| Flag | Default | Notes |
|------|---------|-------|
| `--stride` | `4` | `S`. Larger = cheaper but assumes smooth motion. Try 2/4/8. |
| `--denoise` | `0.25` | Plan target band 0.15-0.35. >0.5 = morph/flicker risk. |
| `--prev-blend` | `0.35` | Latent blend of previous stylized frame into init. Higher = stickier (less flicker, more smearing). |
| `--seed` | `1234` | Held fixed for temporal stability. |
| `--fps`/`--width`/`--height` | `30/1280/720` | Locked across the chain. |

## Re-running individual stages

`--skip extract keyframes` etc. lets you iterate on the stylize / RIFE /
mux stages without re-decoding.

## What this avoids

- No text-to-video for the main timeline (motion is from real pixels).
- No frame-independent diffusion: stylize.py runs sequentially and the
  previous stylized frame is latent-blended into the next init.
- No `rife-ncnn-vulkan` directory mode with large `-n` (which duplicates
  end frames). Each in-between is requested at an explicit `k/S` ratio.
