#!/usr/bin/env bash
set -euo pipefail

# Wrapper for How2Sign + CSL-Daily joint VAE finetuning.
export CFG=${CFG:-"configs/vae/vae_finetune_h2s_csl.yaml"}
export DATASET_NAME=${DATASET_NAME:-"how2sign_csl"}

bash scripts/pipeline/train_vae_finetune_sign_ddp.sh
