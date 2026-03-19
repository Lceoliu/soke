#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=${NUM_SAMPLES:-12}
SIGNER=${SIGNER:-P0000}
GPU_IDS=${GPU_IDS:-0}
SUBSET_ROOT=${SUBSET_ROOT:-data/CSL-Daily-overfit${NUM_SAMPLES}}
EXP_NAME=${EXP_NAME:-SOKE_QWEN_CSL_OVERFIT${NUM_SAMPLES}_T2M}
CFG=${CFG:-configs/soke_csl_overfit_t2m.yaml}
PRETRAINED_VAE=${PRETRAINED_VAE:-experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt}
END_EPOCH=${END_EPOCH:-200}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-2}
VIS_NUM_SAMPLES=${VIS_NUM_SAMPLES:-12}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

source /opt/conda/etc/profile.d/conda.sh
conda activate soke

python scripts/pipeline/prepare_csl_overfit_subset.py \
  --num_samples "$NUM_SAMPLES" \
  --signer "$SIGNER" \
  --dst_root "$SUBSET_ROOT"

GPU_IDS="$GPU_IDS" \
CFG="$CFG" \
EXP_NAME="$EXP_NAME" \
PREPARE_TOKENS=1 \
TRAIN_LM=1 \
AUTO_EVAL_BLEU=0 \
AUTO_SHOW_M2T=0 \
AUTO_VIS=1 \
AUTO_VIS_MC=0 \
BATCH_SIZE="$BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
END_EPOCH="$END_EPOCH" \
VIS_NUM_SAMPLES="$VIS_NUM_SAMPLES" \
VIS_SPLIT=test \
EVAL_SPLIT=test \
CSL_ROOT="$SUBSET_ROOT" \
MEAN_PATH="$SUBSET_ROOT/mean.pt" \
STD_PATH="$SUBSET_ROOT/std.pt" \
PRETRAINED_VAE="$PRETRAINED_VAE" \
bash scripts/pipeline/train_qwen_downstream_auto.sh
