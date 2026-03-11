#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-"configs/vae/vae_finetune_sign_lfq4.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-"experiments/mgpt/VAE_MOTIONX_PRETRAIN_LFQ4_C128H256/checkpoints/last.ckpt"}

DATASET_NAME=${DATASET_NAME:-"how2sign_csl_phoenix"}
H2S_ROOT=${H2S_ROOT:-"data/How2Sign"}
CSL_ROOT=${CSL_ROOT:-"data/CSL-Daily"}
PHOENIX_ROOT=${PHOENIX_ROOT:-"data/Phoenix_2014T"}
MEAN_PATH=${MEAN_PATH:-"data/CSL-Daily/mean.pt"}
STD_PATH=${STD_PATH:-"data/CSL-Daily/std.pt"}

CMD=(bash scripts/pipeline/train_vae_finetune_sign_ddp.sh)

export CFG GPU_IDS NUM_NODES BATCH_SIZE PRETRAINED_CKPT
export DATASET_NAME H2S_ROOT CSL_ROOT PHOENIX_ROOT MEAN_PATH STD_PATH

echo "Launching LFQ4 sign finetune with:"
echo "  CFG=$CFG"
echo "  PRETRAINED_CKPT=$PRETRAINED_CKPT"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  GPU_IDS=$GPU_IDS"

"${CMD[@]}"
