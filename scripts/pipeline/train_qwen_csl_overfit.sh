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
SUBSET_ROOT=${SUBSET_ROOT:-data/CSL-Daily-overfit12}
GPU_IDS=${GPU_IDS:-0}
EXP_NAME=${EXP_NAME:-SOKE_QWEN_CSL_OVERFIT12}
PRETRAINED_VAE=${PRETRAINED_VAE:-experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt}

python scripts/pipeline/prepare_csl_overfit_subset.py \
  --src_root data/CSL-Daily \
  --dst_root "$SUBSET_ROOT" \
  --num_samples "$NUM_SAMPLES" \
  --signer "$SIGNER"

GPU_IDS="$GPU_IDS" \
CFG=configs/soke_csl_overfit.yaml \
EXP_NAME="$EXP_NAME" \
PREPARE_TOKENS=0 \
TRAIN_LM=1 \
AUTO_EVAL_BLEU=0 \
AUTO_VIS=0 \
PRETRAINED_VAE="$PRETRAINED_VAE" \
CSL_ROOT="$SUBSET_ROOT" \
MEAN_PATH="$SUBSET_ROOT/mean.pt" \
STD_PATH="$SUBSET_ROOT/std.pt" \
VIS_INPUT_FPS=50 \
MESH_RY_DEG=-180 \
bash scripts/pipeline/train_qwen_downstream_auto.sh
