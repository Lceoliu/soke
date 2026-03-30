#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "soke" && -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source /opt/conda/etc/profile.d/conda.sh || true
  conda activate soke >/dev/null 2>&1 || true
fi

GPU_IDS=${GPU_IDS:-0}
CFG=${CFG:-configs/soke_mt5_csl_overfit_m2t.yaml}
EXP_NAME=${EXP_NAME:-SOKE_MT5_CSL_OVERFIT12_M2T}
BATCH_SIZE=${BATCH_SIZE:-4}
END_EPOCH=${END_EPOCH:-200}
NUM_WORKERS=${NUM_WORKERS:-2}
PRETRAINED_VAE=${PRETRAINED_VAE:-experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt}
AUTO_EVAL_BLEU=${AUTO_EVAL_BLEU:-0}
AUTO_VIS=${AUTO_VIS:-0}
EVAL_SPLIT=${EVAL_SPLIT:-train}
AUTO_GENERATE=${AUTO_GENERATE:-1}
GENERATE_SPLIT=${GENERATE_SPLIT:-train}

GPU_IDS="$GPU_IDS" \
CFG="$CFG" \
EXP_NAME="$EXP_NAME" \
PREPARE_TOKENS=0 \
TRAIN_LM=1 \
AUTO_EVAL_BLEU="$AUTO_EVAL_BLEU" \
AUTO_VIS="$AUTO_VIS" \
EVAL_SPLIT="$EVAL_SPLIT" \
PRETRAINED_VAE="$PRETRAINED_VAE" \
BATCH_SIZE="$BATCH_SIZE" \
END_EPOCH="$END_EPOCH" \
NUM_WORKERS="$NUM_WORKERS" \
bash scripts/pipeline/train_lm_downstream_auto.sh

if [[ "$AUTO_GENERATE" == "1" ]]; then
  EXP_DIR="experiments/mgpt/${EXP_NAME}"
  RESULT_DIR="results/mgpt/${EXP_NAME}"
  CKPT_PATH="${EXP_DIR}/checkpoints/last.ckpt"
  if [[ ! -f "$CKPT_PATH" ]]; then
    CKPT_PATH=$(ls -1t "${EXP_DIR}"/checkpoints/*.ckpt 2>/dev/null | head -n 1 || true)
  fi
  if [[ -z "$CKPT_PATH" || ! -f "$CKPT_PATH" ]]; then
    echo "[FATAL] no checkpoint found under ${EXP_DIR}/checkpoints"
    exit 2
  fi

  mkdir -p "$RESULT_DIR"
  OUT_PATH="${RESULT_DIR}/${GENERATE_SPLIT}_m2t_generations.jsonl"
  FIRST_GPU="${GPU_IDS%%,*}"
  echo "[generate] exporting m2t generations from ${CKPT_PATH}"
  python scripts/pipeline/export_m2t_generations.py \
    --cfg "$CFG" \
    --checkpoint "$CKPT_PATH" \
    --split "$GENERATE_SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --exp_name "$EXP_NAME" \
    --pretrained_vae "$PRETRAINED_VAE" \
    --device "cuda:${FIRST_GPU}" \
    --output "$OUT_PATH"

  echo "[generate] results saved to ${OUT_PATH}"
fi
