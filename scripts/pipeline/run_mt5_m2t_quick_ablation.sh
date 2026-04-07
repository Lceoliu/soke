#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "soke" && -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source /opt/conda/etc/profile.d/conda.sh || true
  conda activate soke >/dev/null 2>&1 || true
fi

EXP_KIND=${EXP_KIND:-full}
BASE_CFG=${BASE_CFG:-configs/soke_mt5_m2t.yaml}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
END_EPOCH=${END_EPOCH:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
VAL_EVERY_EPOCHS=${VAL_EVERY_EPOCHS:-5}
EVAL_SPLIT=${EVAL_SPLIT:-val}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
PRETRAINED_VAE=${PRETRAINED_VAE:-experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt}
CKPT_SELECT=${CKPT_SELECT:-best_bleu}
SKIP_TRAIN=${SKIP_TRAIN:-0}

case "$EXP_KIND" in
  full)
    MOTION_ABLATION="none"
    PROMPT_MODE="src_lang"
    DEFAULT_EXP_NAME="SOKE_MT5_M2T_FULL_E20"
    ;;
  no_motion)
    MOTION_ABLATION="zero"
    PROMPT_MODE="src_lang"
    DEFAULT_EXP_NAME="SOKE_MT5_M2T_NO_MOTION_E20"
    ;;
  generic_prompt)
    MOTION_ABLATION="none"
    PROMPT_MODE="generic"
    DEFAULT_EXP_NAME="SOKE_MT5_M2T_GENERIC_PROMPT_E20"
    ;;
  *)
    echo "[FATAL] Unsupported EXP_KIND=$EXP_KIND"
    exit 2
    ;;
esac

EXP_NAME=${EXP_NAME:-$DEFAULT_EXP_NAME}
TS=$(date +%Y%m%d_%H%M%S)
RUN_CFG="/tmp/${EXP_NAME}_${TS}.yaml"

python - <<PY
from omegaconf import OmegaConf
from mGPT.config import get_module_config

cfg_assets = OmegaConf.load("./configs/assets.yaml")
cfg_base = OmegaConf.load(f"{cfg_assets.CONFIG_FOLDER}/default.yaml")
cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load("$BASE_CFG"))
if not cfg_exp.FULL_CONFIG:
    cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
cfg = OmegaConf.merge(cfg_exp, cfg_assets)
cfg.NAME = "$EXP_NAME"
cfg.TRAIN.END_EPOCH = int("$END_EPOCH")
cfg.TRAIN.BATCH_SIZE = int("$BATCH_SIZE")
cfg.TRAIN.NUM_WORKERS = int("$NUM_WORKERS")
cfg.TRAIN.PRETRAINED_VAE = "$PRETRAINED_VAE"
cfg.TRAIN.RESUME = ""
cfg.TRAIN.PRETRAINED = ""
cfg.TRAIN.LR_SCHEDULER.params.T_max = int("$END_EPOCH")
cfg.EVAL.BATCH_SIZE = int("$EVAL_BATCH_SIZE")
cfg.EVAL.NUM_WORKERS = int("$NUM_WORKERS")
cfg.EVAL.VAL_SUBSET_RATIO = 1.0
cfg.EVAL.VAL_SUBSET_MAX_SAMPLES = 0
cfg.LOGGER.VAL_EVERY_STEPS = int("$VAL_EVERY_EPOCHS")
cfg.model.params.task = "m2t"
cfg.model.params.lm.params.motion_ablation = "$MOTION_ABLATION"
cfg.model.params.lm.params.prompt_mode = "$PROMPT_MODE"
cfg.FULL_CONFIG = True
OmegaConf.save(cfg, "$RUN_CFG")
print("saved", "$RUN_CFG")
PY

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
DEVICE_ARGS=()
for i in "${!GPUS[@]}"; do
  DEVICE_ARGS+=("$i")
done

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python train.py \
    --cfg "$RUN_CFG" \
    --nodebug \
    --use_gpus "$GPU_IDS" \
    --device "${DEVICE_ARGS[@]}"
fi

EXP_DIR="experiments/mgpt/${EXP_NAME}"
if [[ "$CKPT_SELECT" == "best_bleu" ]]; then
  EVAL_CKPT=$(ls -1t "$EXP_DIR"/checkpoints/max-BLEU_4*.ckpt 2>/dev/null | head -n 1 || true)
else
  EVAL_CKPT=""
fi
if [[ -z "$EVAL_CKPT" && -f "$EXP_DIR/checkpoints/last.ckpt" ]]; then
  EVAL_CKPT="$EXP_DIR/checkpoints/last.ckpt"
fi
if [[ -z "$EVAL_CKPT" ]]; then
  EVAL_CKPT=$(ls -1t "$EXP_DIR"/checkpoints/*.ckpt 2>/dev/null | head -n 1 || true)
fi
if [[ -z "$EVAL_CKPT" || ! -f "$EVAL_CKPT" ]]; then
  echo "[FATAL] no checkpoint found under $EXP_DIR/checkpoints"
  exit 2
fi

mkdir -p "$EXP_DIR/auto_reports/downstream"
FIRST_GPU="${GPUS[0]:-0}"
python scripts/analysis/eval_m2t_predictions.py \
  --cfg "$RUN_CFG" \
  --checkpoint "$EVAL_CKPT" \
  --split "$EVAL_SPLIT" \
  --batch_size "$EVAL_BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --use_gpus "$FIRST_GPU" \
  --device 0 \
  --motion_mode normal \
  --output_json "$EXP_DIR/auto_reports/downstream/m2t_eval_${EVAL_SPLIT}.json" \
  --output_jsonl "$EXP_DIR/auto_reports/downstream/m2t_examples_${EVAL_SPLIT}.jsonl"

echo "[done] EXP_KIND=$EXP_KIND EXP_NAME=$EXP_NAME"
echo "  cfg:   $RUN_CFG"
echo "  ckpt:  $EVAL_CKPT"
echo "  eval:  $EXP_DIR/auto_reports/downstream/m2t_eval_${EVAL_SPLIT}.json"
echo "  jsonl: $EXP_DIR/auto_reports/downstream/m2t_examples_${EVAL_SPLIT}.jsonl"
