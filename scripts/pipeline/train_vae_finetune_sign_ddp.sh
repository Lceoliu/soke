#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-"configs/vae/vae_finetune_sign.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-""}

# Optional dataset-path overrides
DATASET_NAME=${DATASET_NAME:-""}
H2S_ROOT=${H2S_ROOT:-""}
CSL_ROOT=${CSL_ROOT:-""}
PHOENIX_ROOT=${PHOENIX_ROOT:-""}
MEAN_PATH=${MEAN_PATH:-""}
STD_PATH=${STD_PATH:-""}

LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/vae_finetune_sign_$(date +%Y%m%d_%H%M%S).log"}
AUTO_POST=${AUTO_POST:-1}
AUTO_POST_STRICT=${AUTO_POST_STRICT:-0}
AUTO_POST_DEVICE=${AUTO_POST_DEVICE:-"cuda"}
AUTO_REPORT_MAX_SAMPLES=${AUTO_REPORT_MAX_SAMPLES:-3000}
AUTO_VIS_TRAIN=${AUTO_VIS_TRAIN:-2}
AUTO_VIS_TEST=${AUTO_VIS_TEST:-2}

export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

RUN_CFG="$CFG"
if [[ -n "$PRETRAINED_CKPT" || -n "$DATASET_NAME" || -n "$H2S_ROOT" || -n "$CSL_ROOT" || -n "$PHOENIX_ROOT" || -n "$MEAN_PATH" || -n "$STD_PATH" ]]; then
  TMP_CFG="/tmp/vae_finetune_sign_$(date +%s).yaml"
  python3 - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$CFG")
if "$PRETRAINED_CKPT":
    cfg.TRAIN.PRETRAINED = "$PRETRAINED_CKPT"
if "$DATASET_NAME":
    cfg.DATASET.H2S.DATASET_NAME = "$DATASET_NAME"
if "$H2S_ROOT":
    cfg.DATASET.H2S.ROOT = "$H2S_ROOT"
if "$CSL_ROOT":
    cfg.DATASET.H2S.CSL_ROOT = "$CSL_ROOT"
if "$PHOENIX_ROOT":
    cfg.DATASET.H2S.PHOENIX_ROOT = "$PHOENIX_ROOT"
if "$MEAN_PATH":
    cfg.DATASET.H2S.MEAN_PATH = "$MEAN_PATH"
if "$STD_PATH":
    cfg.DATASET.H2S.STD_PATH = "$STD_PATH"
OmegaConf.save(cfg, "$TMP_CFG")
print("saved", "$TMP_CFG")
PY
  RUN_CFG="$TMP_CFG"
fi

# Basic data-file checks (fast fail)
python3 - <<PY
import os, gzip, pickle, pandas as pd
from omegaconf import OmegaConf
cfg = OmegaConf.load("$RUN_CFG")
h2s = cfg.DATASET.H2S
dn = str(h2s.DATASET_NAME)
errs = []
if "how2sign" in dn:
    for sp in ["train", "val", "test"]:
        p = os.path.join(h2s.ROOT, sp, "re_aligned", f"how2sign_realigned_{sp}_preprocessed_fps.csv")
        if not os.path.exists(p):
            errs.append(f"missing {p}")
        else:
            try:
                pd.read_csv(p, nrows=2)
            except Exception as e:
                errs.append(f"bad csv {p}: {e}")
if "csl" in dn:
    for sp in ["train", "val", "test"]:
        key = "train" if sp == "train" else sp
        p = os.path.join(h2s.CSL_ROOT, f"csl_clean.{key}")
        if not os.path.exists(p):
            errs.append(f"missing {p}")
        else:
            try:
                with gzip.open(p, "rb") as f:
                    _ = pickle.load(f)
            except Exception as e:
                errs.append(f"bad ann {p}: {e}")
if "phoenix" in dn:
    split_map = {"train": "train", "val": "dev", "test": "test"}
    for sp in ["train", "val", "test"]:
        p = os.path.join(h2s.PHOENIX_ROOT, f"phoenix14t.{split_map[sp]}")
        if not os.path.exists(p):
            errs.append(f"missing {p}")
        else:
            try:
                with gzip.open(p, "rb") as f:
                    _ = pickle.load(f)
            except Exception as e:
                errs.append(f"bad ann {p}: {e}")
for p in [h2s.MEAN_PATH, h2s.STD_PATH]:
    if not os.path.exists(p):
        errs.append(f"missing {p}")
if errs:
    print("[FATAL] dataset check failed:")
    for e in errs:
        print(" -", e)
    raise SystemExit(2)
print("[OK] dataset check passed for", dn)
PY

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
DEVICE_ARGS=()
for i in "${!GPUS[@]}"; do
  DEVICE_ARGS+=("$i")
done

CMD=(python3 train.py
  --cfg "$RUN_CFG"
  --nodebug
  --use_gpus "$GPU_IDS"
  --num_nodes "$NUM_NODES"
  --device "${DEVICE_ARGS[@]}"
)

if [[ -n "$BATCH_SIZE" ]]; then
  CMD+=(--batch_size "$BATCH_SIZE")
fi

echo "Running command: ${CMD[*]}"
echo "Logs: $LOG_FILE"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

if [[ "$AUTO_POST" == "1" ]]; then
  echo "[post] Running automatic report + visualization ..."
  POST_CMD=(python3 scripts/analysis/auto_post_train_eval_vis.py
    --cfg "$RUN_CFG"
    --log_path "$LOG_FILE"
    --device "$AUTO_POST_DEVICE"
    --report_max_samples "$AUTO_REPORT_MAX_SAMPLES"
    --vis_train_num "$AUTO_VIS_TRAIN"
    --vis_test_num "$AUTO_VIS_TEST"
  )
  if [[ "$AUTO_POST_STRICT" == "1" ]]; then
    POST_CMD+=(--strict)
  fi
  "${POST_CMD[@]}" || {
    if [[ "$AUTO_POST_STRICT" == "1" ]]; then
      exit 1
    fi
    echo "[post][WARN] automatic post-processing failed. continue."
  }
fi
