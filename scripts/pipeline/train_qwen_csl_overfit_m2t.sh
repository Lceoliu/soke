#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "soke" && -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source /opt/conda/etc/profile.d/conda.sh || true
  conda activate soke >/dev/null 2>&1 || true
fi

NUM_SAMPLES=${NUM_SAMPLES:-12}
SIGNER=${SIGNER:-P0000}
SUBSET_ROOT=${SUBSET_ROOT:-data/CSL-Daily-overfit${NUM_SAMPLES}}
GPU_IDS=${GPU_IDS:-0}
EXP_NAME=${EXP_NAME:-SOKE_QWEN_CSL_OVERFIT${NUM_SAMPLES}_M2T}
PRETRAINED_VAE=${PRETRAINED_VAE:-experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt}
SIGN_STREAMS=${SIGN_STREAMS:-""}
BATCH_SIZE=${BATCH_SIZE:-""}
END_EPOCH=${END_EPOCH:-""}
NUM_WORKERS=${NUM_WORKERS:-""}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-""}
PRECISION=${PRECISION:-""}
TORCH_DTYPE=${TORCH_DTYPE:-""}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-""}
MAX_LENGTH=${MAX_LENGTH:-""}
LORA_RANK=${LORA_RANK:-""}
LORA_ALPHA=${LORA_ALPHA:-""}

python scripts/pipeline/prepare_csl_overfit_subset.py \
  --src_root data/CSL-Daily \
  --dst_root "$SUBSET_ROOT" \
  --num_samples "$NUM_SAMPLES" \
  --signer "$SIGNER"

GPU_IDS="$GPU_IDS" \
CFG=configs/soke_csl_overfit_m2t.yaml \
EXP_NAME="$EXP_NAME" \
PREPARE_TOKENS=1 \
TRAIN_LM=1 \
AUTO_EVAL_BLEU=0 \
AUTO_SHOW_M2T=0 \
AUTO_VIS=0 \
AUTO_VIS_MC=0 \
PRETRAINED_VAE="$PRETRAINED_VAE" \
SIGN_STREAMS="$SIGN_STREAMS" \
BATCH_SIZE="$BATCH_SIZE" \
END_EPOCH="$END_EPOCH" \
NUM_WORKERS="$NUM_WORKERS" \
ACCUMULATE_GRAD_BATCHES="$ACCUMULATE_GRAD_BATCHES" \
PRECISION="$PRECISION" \
TORCH_DTYPE="$TORCH_DTYPE" \
GRADIENT_CHECKPOINTING="$GRADIENT_CHECKPOINTING" \
MAX_LENGTH="$MAX_LENGTH" \
LORA_RANK="$LORA_RANK" \
LORA_ALPHA="$LORA_ALPHA" \
CSL_ROOT="$SUBSET_ROOT" \
MEAN_PATH="$SUBSET_ROOT/mean.pt" \
STD_PATH="$SUBSET_ROOT/std.pt" \
bash scripts/pipeline/train_qwen_downstream_auto.sh
