#!/usr/bin/env bash
# Convenience wrapper: run the test render first, gate on user approval,
# then run the full render with identical settings.
set -euo pipefail

INPUT="${INPUT:?set INPUT=/path/to/source.mp4}"
WORK="${WORK:-./work}"
OUT_DIR="${OUT_DIR:-./out}"
mkdir -p "$OUT_DIR"

COMMON=(
  --input  "$INPUT"
  --work   "$WORK"
  --fps    "${FPS:-30}"
  --width  "${WIDTH:-1280}"
  --height "${HEIGHT:-720}"
  --stride "${STRIDE:-4}"
  --denoise "${DENOISE:-0.25}"
  --comfy-host    "${COMFY_HOST:-127.0.0.1:8188}"
  --comfy-ckpt    "${COMFY_CKPT:-sd_xl_base_1.0.safetensors}"
  --comfy-controlnet "${COMFY_CONTROLNET:-controlnet-canny-sdxl-1.0.safetensors}"
)

echo "=== TEST RENDER (5s gate) ==="
python -m pipeline.run "${COMMON[@]}" --test \
  --output "$OUT_DIR/test_5s_out.mp4"

read -r -p "Test render at $OUT_DIR/test_5s_out.mp4 acceptable? [y/N] " ans
if [[ "${ans,,}" != "y" ]]; then
  echo "Aborting full render. Adjust settings and re-run."
  exit 1
fi

echo "=== FULL RENDER ==="
python -m pipeline.run "${COMMON[@]}" \
  --output "$OUT_DIR/final.mp4"
