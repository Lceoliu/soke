#!/usr/bin/env bash
set -euo pipefail

# LFQ5 scratch training on sign datasets (How2Sign + CSL + Phoenix).
CFG=${CFG:-"configs/vae/vae_sign_scratch_lfq5_allparts.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}

# Optional dataset path overrides
DATASET_NAME=${DATASET_NAME:-"how2sign_csl_phoenix"}
H2S_ROOT=${H2S_ROOT:-"data/How2Sign"}
CSL_ROOT=${CSL_ROOT:-"data/CSL-Daily"}
PHOENIX_ROOT=${PHOENIX_ROOT:-"data/Phoenix_2014T"}
MEAN_PATH=${MEAN_PATH:-"data/CSL-Daily/mean.pt"}
STD_PATH=${STD_PATH:-"data/CSL-Daily/std.pt"}

# OOM fallback and post-process options are inherited by the base script.
RETRY_ON_OOM=${RETRY_ON_OOM:-1}
MIN_BATCH_SIZE=${MIN_BATCH_SIZE:-1}
AUTO_POST=${AUTO_POST:-1}
AUTO_POST_STRICT=${AUTO_POST_STRICT:-0}
AUTO_POST_DEVICE=${AUTO_POST_DEVICE:-"cuda"}
AUTO_REPORT_MAX_SAMPLES=${AUTO_REPORT_MAX_SAMPLES:-3000}
AUTO_VIS_TRAIN=${AUTO_VIS_TRAIN:-2}
AUTO_VIS_TEST=${AUTO_VIS_TEST:-2}

# Explicitly disable pretrained loading for scratch training.
unset PRETRAINED_CKPT || true

export CFG GPU_IDS NUM_NODES BATCH_SIZE
export DATASET_NAME H2S_ROOT CSL_ROOT PHOENIX_ROOT MEAN_PATH STD_PATH
export RETRY_ON_OOM MIN_BATCH_SIZE
export AUTO_POST AUTO_POST_STRICT AUTO_POST_DEVICE AUTO_REPORT_MAX_SAMPLES AUTO_VIS_TRAIN AUTO_VIS_TEST

echo "Launching LFQ5 scratch training (all parts):"
echo "  CFG=$CFG"
echo "  GPU_IDS=$GPU_IDS"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  PRETRAINED_CKPT=<disabled>"

bash scripts/pipeline/train_vae_finetune_sign_ddp.sh
