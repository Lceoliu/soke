#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-all}
SPLITS=${SPLITS:-train,val,test}
HOW2SIGN_ROOT=${HOW2SIGN_ROOT:-data/How2Sign}
CSL_ROOT=${CSL_ROOT:-data/CSL-Daily}
PHOENIX_ROOT=${PHOENIX_ROOT:-data/Phoenix_2014T}
OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME:-contact_labels}
DEVICE=${DEVICE:-}
THRESHOLD=${THRESHOLD:-0.02}
CHUNK_A=${CHUNK_A:-512}
CHUNK_B=${CHUNK_B:-2048}
FK_CHUNK_FRAMES=${FK_CHUNK_FRAMES:-256}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:--1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
OVERWRITE=${OVERWRITE:-0}

CMD=(
  python scripts/pipeline/precompute_contact_labels.py
  --dataset "$DATASET"
  --splits "$SPLITS"
  --how2sign_root "$HOW2SIGN_ROOT"
  --csl_root "$CSL_ROOT"
  --phoenix_root "$PHOENIX_ROOT"
  --output_dir_name "$OUTPUT_DIR_NAME"
  --threshold "$THRESHOLD"
  --chunk_a "$CHUNK_A"
  --chunk_b "$CHUNK_B"
  --fk_chunk_frames "$FK_CHUNK_FRAMES"
  --start_idx "$START_IDX"
  --end_idx "$END_IDX"
  --max_samples "$MAX_SAMPLES"
)

if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "[RUN] ${CMD[*]}"
"${CMD[@]}"
