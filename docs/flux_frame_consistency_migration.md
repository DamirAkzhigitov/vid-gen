# FLUX Frame Consistency Pipeline & Migration Guide

This guide describes how to migrate this repository from an SDXL + ControlNet
workflow to a FLUX-oriented workflow focused on temporal consistency and
scalability for long frame sequences.

## 1) Goal

Target outcomes for long sequences (10k-20k frames):

- Stable scene consistency across segments.
- Minimal flicker and identity drift.
- Practical runtime via sparse generation + interpolation.

## 2) Key Concept Shift: SDXL vs FLUX

### Legacy SDXL approach

- Base model: SDXL/SD1.5.
- Extra modules for consistency: ControlNet, IP-Adapter, LoRA.
- More nodes and tuning per frame.

### FLUX approach

- Treat FLUX as the primary consistency engine.
- Use native multi-image conditioning references:
  - `current frame` (structure anchor)
  - `previous generated frame`
  - `segment anchor keyframe`
- Keep prompts stable by scene; avoid per-frame prompt churn.

In this repo that maps to:

- `pipeline/stylize.py` workflow token injection for:
  - `__INPUT_IMAGE__`
  - `__PREV_STYLIZED__`
  - `__KEYFRAME_ANCHOR__`
- `pipeline/keyframes.py` manifest metadata for:
  - `prev_keyframe_ordinal`
  - `anchor_keyframe_ordinal`

## 3) Recommended FLUX Frame Strategy

For each processed keyframe `t`, condition on:

- frame `t` (current keyframe image)
- frame `t-1` generated keyframe output (short-term continuity)
- segment anchor keyframe (long-term scene continuity)

Practical defaults:

- `--segment-anchor-interval 120` (tune to 50-200 based on scene volatility)
- fixed prompt template per segment
- low denoise (`0.15-0.35`)
- fixed seed per segment/job unless variation is explicitly desired

## 4) Processing Pipeline

1. **Extract** frames from video.
2. **Build keyframe manifest** with stride and anchor references.
3. **Generate keyframes** through FLUX workflow template with multi-reference
   conditioning.
4. **Interpolate** non-keyframes with Practical-RIFE.
5. **Mux** final frames into output video.

This repo already supports this chain via:

- `pipeline/extract.py`
- `pipeline/keyframes.py`
- `pipeline/stylize.py`
- `pipeline/rife.py`
- `pipeline/mux.py`

## 5) Migration from SDXL Pipeline

### 5.1 IP-Adapter migration

- Remove IP-Adapter nodes from old graphs.
- Route references directly as workflow image inputs.
- In this repo, bind those inputs to:
  - `__PREV_STYLIZED__`
  - `__KEYFRAME_ANCHOR__`

### 5.2 ControlNet stack migration

- Start by removing stacked ControlNet conditions.
- Re-introduce only essential structure control if geometry must be constrained.
- Keep SDXL workflow (`workflows/sdxl_img2img_canny.json`) as fallback.

### 5.3 Prompt/seed policy migration

- Use stable prompt structure per scene.
- Keep style descriptors fixed for each segment.
- Use one seed per segment (or per full job) for temporal stability.

### 5.4 Graph simplification

- Prefer a minimal FLUX graph:
  - load image references
  - generate
  - decode/save
- Fewer moving parts means fewer drift vectors over long runs.

## 6) Consistency Rules (Critical)

Keep constant per scene:

- prompt structure
- style descriptors
- reference selection strategy

Control tightly:

- denoise strength
- inter-frame variation
- seed policy

Avoid:

- per-frame prompt rewrites
- unrelated references
- style switching inside one segment

## 7) Scaling to Large Sequences (18k+ frames)

Full per-frame generation can be too expensive. Prefer hybrid:

1. Generate keyframes every `N` frames (`stride`).
2. Interpolate in-betweens with RIFE.
3. Run corrective stylization passes only where needed.

Recommended stride candidates: `10`, `20`, `30` (scene-dependent).

## 8) Recommended Stack

- FLUX generation for keyframes
- Practical-RIFE for interpolation
- ffmpeg for extraction/mux

Optional stabilizers:

- depth guidance for geometry-sensitive scenes
- segmentation masks for selective edits

## 9) Repo-specific CLI examples

### FLUX-style run (5s gate)

`python -m pipeline.run --input /data/source.mp4 --work ./work --output ./out/test_flux.mp4 --model-family flux --workflow workflows/flux_img2img_multi_ref.json --stride 20 --segment-anchor-interval 120 --denoise 0.22 --test`

### FLUX full run

`python -m pipeline.run --input /data/source.mp4 --work ./work --output ./out/final_flux.mp4 --model-family flux --workflow workflows/flux_img2img_multi_ref.json --stride 20 --segment-anchor-interval 120 --denoise 0.22`

### SDXL fallback run

`python -m pipeline.run --input /data/source.mp4 --work ./work --output ./out/final_sdxl.mp4 --model-family sdxl --workflow workflows/sdxl_img2img_canny.json --stride 4 --denoise 0.25`

## 10) Notes on FLUX workflow template

`workflows/flux_img2img_multi_ref.json` is a template scaffold. Export a
known-good FLUX graph from your ComfyUI installation and map this repository's
tokens to your node IDs and class types.
