#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
END_EPOCH=${END_EPOCH:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
VAL_EVERY_EPOCHS=${VAL_EVERY_EPOCHS:-5}
EVAL_SPLIT=${EVAL_SPLIT:-val}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}

EXP_KIND=full \
GPU_IDS="$GPU_IDS" \
END_EPOCH="$END_EPOCH" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
VAL_EVERY_EPOCHS="$VAL_EVERY_EPOCHS" \
EVAL_SPLIT="$EVAL_SPLIT" \
EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
bash scripts/pipeline/run_mt5_m2t_quick_ablation.sh

EXP_KIND=no_motion \
GPU_IDS="$GPU_IDS" \
END_EPOCH="$END_EPOCH" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
VAL_EVERY_EPOCHS="$VAL_EVERY_EPOCHS" \
EVAL_SPLIT="$EVAL_SPLIT" \
EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
bash scripts/pipeline/run_mt5_m2t_quick_ablation.sh

EXP_KIND=generic_prompt \
GPU_IDS="$GPU_IDS" \
END_EPOCH="$END_EPOCH" \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
VAL_EVERY_EPOCHS="$VAL_EVERY_EPOCHS" \
EVAL_SPLIT="$EVAL_SPLIT" \
EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
bash scripts/pipeline/run_mt5_m2t_quick_ablation.sh

FULL_EXP_NAME=${FULL_EXP_NAME:-SOKE_MT5_M2T_FULL_E20}
FULL_CFG=$(ls -1t "experiments/mgpt/${FULL_EXP_NAME}"/config_*_train.yaml 2>/dev/null | head -n 1 || true)
FULL_CKPT=$(ls -1t "experiments/mgpt/${FULL_EXP_NAME}"/checkpoints/max-BLEU_4*.ckpt 2>/dev/null | head -n 1 || true)
if [[ -z "$FULL_CKPT" && -f "experiments/mgpt/${FULL_EXP_NAME}/checkpoints/last.ckpt" ]]; then
  FULL_CKPT="experiments/mgpt/${FULL_EXP_NAME}/checkpoints/last.ckpt"
fi
if [[ -z "$FULL_CFG" || -z "$FULL_CKPT" ]]; then
  echo "[FATAL] missing full-experiment config or checkpoint for shuffle eval"
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
FIRST_GPU="${GPUS[0]:-0}"
python scripts/analysis/eval_m2t_predictions.py \
  --cfg "$FULL_CFG" \
  --checkpoint "$FULL_CKPT" \
  --split "$EVAL_SPLIT" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --use_gpus "$FIRST_GPU" \
  --device 0 \
  --motion_mode shuffle_by_src \
  --output_json "experiments/mgpt/${FULL_EXP_NAME}/auto_reports/downstream/m2t_eval_${EVAL_SPLIT}_shuffle_by_src.json" \
  --output_jsonl "experiments/mgpt/${FULL_EXP_NAME}/auto_reports/downstream/m2t_examples_${EVAL_SPLIT}_shuffle_by_src.jsonl"

echo "[done] suite completed"
