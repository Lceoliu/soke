#!/usr/bin/env bash
set -euo pipefail

# Default: LFQ4 sign finetune with wrist-relative FK hand loss.
CFG=${CFG:-"configs/vae/vae_finetune_sign_lfq4_fkhand_body_pretrained.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}

# Body-only pretrained loading (hands remain random init).
PRETRAINED_BODY_CKPT=${PRETRAINED_BODY_CKPT:-"/home/SOKE/experiments/mgpt/VAE_MOTIONX_PRETRAIN_LFQ4_C128H256/checkpoints/last.ckpt"}
# Backward-compatible alias (if user passes PRETRAINED_CKPT, use it as body ckpt).
PRETRAINED_CKPT=${PRETRAINED_CKPT:-""}
if [[ -n "$PRETRAINED_CKPT" ]]; then
  PRETRAINED_BODY_CKPT="$PRETRAINED_CKPT"
fi

# Optional dataset path overrides
DATASET_NAME=${DATASET_NAME:-"how2sign_csl_phoenix"}
H2S_ROOT=${H2S_ROOT:-"data/How2Sign"}
CSL_ROOT=${CSL_ROOT:-"data/CSL-Daily"}
PHOENIX_ROOT=${PHOENIX_ROOT:-"data/Phoenix_2014T"}
MEAN_PATH=${MEAN_PATH:-"data/CSL-Daily/mean.pt"}
STD_PATH=${STD_PATH:-"data/CSL-Daily/std.pt"}

# Auto post steps: report + visualization
AUTO_POST=${AUTO_POST:-1}
AUTO_POST_STRICT=${AUTO_POST_STRICT:-0}
AUTO_POST_DEVICE=${AUTO_POST_DEVICE:-"cuda"}
AUTO_REPORT_MAX_SAMPLES=${AUTO_REPORT_MAX_SAMPLES:-3000}
AUTO_VIS_TRAIN=${AUTO_VIS_TRAIN:-2}
AUTO_VIS_TEST=${AUTO_VIS_TEST:-2}
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ "${CONDA_DEFAULT_ENV:-}" != "soke" && -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # Best effort only; keep script usable in non-conda environments.
  source /opt/conda/etc/profile.d/conda.sh || true
  conda activate soke >/dev/null 2>&1 || true
fi

RUN_CFG="$CFG"
TMP_CFG=""
if [[ -n "$PRETRAINED_BODY_CKPT" ]]; then
  TMP_CFG="/tmp/vae_finetune_sign_lfq4_fkhand_body_pretrained_$(date +%s).yaml"
  "$PYTHON_BIN" - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$CFG")
cfg.TRAIN.PRETRAINED_VAE_BODY = "$PRETRAINED_BODY_CKPT"
cfg.TRAIN.PRETRAINED_VAE_LOAD_BODY = True
cfg.TRAIN.PRETRAINED_VAE_LOAD_HAND = False
cfg.TRAIN.PRETRAINED_VAE_LOAD_RHAND = False
OmegaConf.save(cfg, "$TMP_CFG")
print("saved", "$TMP_CFG")
PY
  RUN_CFG="$TMP_CFG"
fi

cleanup() {
  if [[ -n "$TMP_CFG" && -f "$TMP_CFG" ]]; then
    rm -f "$TMP_CFG"
  fi
}
trap cleanup EXIT

export CFG="$RUN_CFG"
export GPU_IDS NUM_NODES BATCH_SIZE
export DATASET_NAME H2S_ROOT CSL_ROOT PHOENIX_ROOT MEAN_PATH STD_PATH
export AUTO_POST AUTO_POST_STRICT AUTO_POST_DEVICE AUTO_REPORT_MAX_SAMPLES AUTO_VIS_TRAIN AUTO_VIS_TEST
export PYTHON_BIN

echo "Launching FK-hand finetune:"
echo "  CFG=$CFG"
echo "  PRETRAINED_BODY_CKPT=$PRETRAINED_BODY_CKPT"
echo "  GPU_IDS=$GPU_IDS"
echo "  AUTO_POST=$AUTO_POST (report + vis)"

bash scripts/pipeline/train_vae_finetune_sign_ddp.sh
