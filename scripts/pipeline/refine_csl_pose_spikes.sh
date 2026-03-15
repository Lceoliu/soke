#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-"data/csl_dental"}
SPLITS=${SPLITS:-"train,val,test"}
WORKERS=${WORKERS:-32}
MAX_CLIPS=${MAX_CLIPS:-0}
NAMES_FILE=${NAMES_FILE:-""}

JUMP_POSE_SCALE=${JUMP_POSE_SCALE:-0.02}
JUMP_MAD_K=${JUMP_MAD_K:-8.0}
JUMP_MIN_SCORE=${JUMP_MIN_SCORE:-0.6}
SPIKE_MAD_K=${SPIKE_MAD_K:-5.0}
SPIKE_BRIDGE_RATIO=${SPIKE_BRIDGE_RATIO:-0.45}
TILT_MAD_K=${TILT_MAD_K:-6.0}
TILT_MIN_DEG=${TILT_MIN_DEG:-45.0}
TILT_ABS_DEG=${TILT_ABS_DEG:-70.0}
ROOT_VEL_MAD_K=${ROOT_VEL_MAD_K:-7.0}
ROOT_VEL_MIN_DEG=${ROOT_VEL_MIN_DEG:-45.0}
MAX_BAD_RATIO=${MAX_BAD_RATIO:-0.35}

APPLY_ONE_EURO=${APPLY_ONE_EURO:-0}
FPS=${FPS:-25.0}
ONE_EURO_MIN_CUTOFF=${ONE_EURO_MIN_CUTOFF:-1.0}
ONE_EURO_BETA=${ONE_EURO_BETA:-0.005}
ONE_EURO_D_CUTOFF=${ONE_EURO_D_CUTOFF:-1.0}
RECOMPUTE_MEAN_STD=${RECOMPUTE_MEAN_STD:-0}
DRY_RUN=${DRY_RUN:-0}

CMD=(python scripts/pipeline/refine_csl_pose_spikes.py
  --dataset-root "$DATASET_ROOT"
  --splits "$SPLITS"
  --workers "$WORKERS"
  --max-clips "$MAX_CLIPS"
  --jump-pose-scale "$JUMP_POSE_SCALE"
  --jump-mad-k "$JUMP_MAD_K"
  --jump-min-score "$JUMP_MIN_SCORE"
  --spike-mad-k "$SPIKE_MAD_K"
  --spike-bridge-ratio "$SPIKE_BRIDGE_RATIO"
  --tilt-mad-k "$TILT_MAD_K"
  --tilt-min-deg "$TILT_MIN_DEG"
  --tilt-abs-deg "$TILT_ABS_DEG"
  --root-vel-mad-k "$ROOT_VEL_MAD_K"
  --root-vel-min-deg "$ROOT_VEL_MIN_DEG"
  --max-bad-ratio "$MAX_BAD_RATIO"
  --fps "$FPS"
  --one-euro-min-cutoff "$ONE_EURO_MIN_CUTOFF"
  --one-euro-beta "$ONE_EURO_BETA"
  --one-euro-d-cutoff "$ONE_EURO_D_CUTOFF"
)

if [[ -n "$NAMES_FILE" ]]; then
  CMD+=(--names-file "$NAMES_FILE")
fi
if [[ "$APPLY_ONE_EURO" == "1" ]]; then
  CMD+=(--apply-one-euro)
fi
if [[ "$RECOMPUTE_MEAN_STD" == "1" ]]; then
  CMD+=(--recompute-mean-std)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "Running:"
echo "  ${CMD[*]}"
"${CMD[@]}"
