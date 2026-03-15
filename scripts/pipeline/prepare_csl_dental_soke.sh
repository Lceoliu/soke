#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT=${INPUT_ROOT:-"/nas/DDDataLang/raw_data/csl_dental/smplx"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/nas/DDDataLang/raw_data/csl_dental/soke"}
WORKERS=${WORKERS:-32}
FPS=${FPS:-25}
MIN_FRAMES=${MIN_FRAMES:-4}
MAX_CLIPS=${MAX_CLIPS:-0}

TRAIN_RATIO=${TRAIN_RATIO:-0.98}
VAL_RATIO=${VAL_RATIO:-0.01}
DEFAULT_TEXT=${DEFAULT_TEXT:-""}

USE_ONE_EURO=${USE_ONE_EURO:-1}
ONE_EURO_MIN_CUTOFF=${ONE_EURO_MIN_CUTOFF:-1.0}
ONE_EURO_BETA=${ONE_EURO_BETA:-0.005}
ONE_EURO_D_CUTOFF=${ONE_EURO_D_CUTOFF:-1.0}

CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:--1.0}
TRANSL_ABS_MAX=${TRANSL_ABS_MAX:-50.0}
JUMP_POSE_SCALE=${JUMP_POSE_SCALE:-0.02}
JUMP_MAD_K=${JUMP_MAD_K:-10.0}
JUMP_MIN_SCORE=${JUMP_MIN_SCORE:-0.8}
SPIKE_MAD_K=${SPIKE_MAD_K:-6.0}
SPIKE_BRIDGE_RATIO=${SPIKE_BRIDGE_RATIO:-0.35}

OVERWRITE=${OVERWRITE:-0}
STATS_FROM_EXISTING=${STATS_FROM_EXISTING:-1}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-0}

CMD=(python scripts/pipeline/prepare_csl_dental_soke.py
  --input-root "$INPUT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --workers "$WORKERS"
  --fps "$FPS"
  --min-frames "$MIN_FRAMES"
  --max-clips "$MAX_CLIPS"
  --train-ratio "$TRAIN_RATIO"
  --val-ratio "$VAL_RATIO"
  --default-text "$DEFAULT_TEXT"
  --one-euro-min-cutoff "$ONE_EURO_MIN_CUTOFF"
  --one-euro-beta "$ONE_EURO_BETA"
  --one-euro-d-cutoff "$ONE_EURO_D_CUTOFF"
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
  --transl-abs-max "$TRANSL_ABS_MAX"
  --jump-pose-scale "$JUMP_POSE_SCALE"
  --jump-mad-k "$JUMP_MAD_K"
  --jump-min-score "$JUMP_MIN_SCORE"
  --spike-mad-k "$SPIKE_MAD_K"
  --spike-bridge-ratio "$SPIKE_BRIDGE_RATIO"
)

if [[ "$USE_ONE_EURO" != "1" ]]; then
  CMD+=(--disable-one-euro)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  CMD+=(--clean-output)
fi
if [[ "$STATS_FROM_EXISTING" != "1" ]]; then
  CMD+=(--no-stats-from-existing)
fi

echo "Running:"
echo "  ${CMD[*]}"
"${CMD[@]}"
