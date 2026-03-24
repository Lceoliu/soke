#!/usr/bin/env bash
set -euo pipefail

# One-click downstream LM pipeline:
# 1) (optional) regenerate motion tokens
# 2) train LM
# 3) evaluate BLEU (m2t)
# 4) generate t2m predictions + render sample mesh videos

CFG=${CFG:-"configs/soke.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}
END_EPOCH=${END_EPOCH:-""}
EXP_NAME=${EXP_NAME:-""}
PRETRAINED_VAE=${PRETRAINED_VAE:-""}
RESUME_CKPT=${RESUME_CKPT:-""}
NUM_WORKERS=${NUM_WORKERS:-""}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-""}

# Optional dataset overrides
DATASET_NAME=${DATASET_NAME:-""}
H2S_ROOT=${H2S_ROOT:-""}
CSL_ROOT=${CSL_ROOT:-""}
PHOENIX_ROOT=${PHOENIX_ROOT:-""}
MEAN_PATH=${MEAN_PATH:-""}
STD_PATH=${STD_PATH:-""}
CODE_PATH=${CODE_PATH:-""}

# Stage switches
PREPARE_TOKENS=${PREPARE_TOKENS:-0}
FORCE_RETOKENIZE=${FORCE_RETOKENIZE:-0}
TRAIN_LM=${TRAIN_LM:-1}
AUTO_EVAL_BLEU=${AUTO_EVAL_BLEU:-1}
AUTO_VIS=${AUTO_VIS:-1}

# Eval/vis controls
EVAL_GPU=${EVAL_GPU:-0}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_SPLIT=${EVAL_SPLIT:-test}
VIS_SPLIT=${VIS_SPLIT:-test}
VIS_NUM_SAMPLES=${VIS_NUM_SAMPLES:-6}
VIS_CAM_Y=${VIS_CAM_Y:--0.5}
VIS_INPUT_FPS=${VIS_INPUT_FPS:-20}
EVAL_CKPT=${EVAL_CKPT:-""}

LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG=${TRAIN_LOG:-"$LOG_DIR/lm_train_${TS}.log"}
BLEU_LOG=${BLEU_LOG:-"$LOG_DIR/lm_bleu_eval_${TS}.log"}
T2M_TEST_LOG=${T2M_TEST_LOG:-"$LOG_DIR/lm_t2m_test_${TS}.log"}
PYTHON_BIN=${PYTHON_BIN:-python3}
if [[ -z "${PYTHON_BIN//[[:space:]]/}" ]]; then
  PYTHON_BIN=python3
fi

# Always run from repository root so local imports (mGPT, scripts, configs) work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "soke" && -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source /opt/conda/etc/profile.d/conda.sh || true
  conda activate soke >/dev/null 2>&1 || true
fi

export NCCL_TIMEOUT=${NCCL_TIMEOUT:-600}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-0}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

RUN_CFG="$CFG"
if [[ -n "$BATCH_SIZE" || -n "$END_EPOCH" || -n "$EXP_NAME" || -n "$PRETRAINED_VAE" || -n "$RESUME_CKPT" || -n "$NUM_WORKERS" || -n "$ACCUMULATE_GRAD_BATCHES" || -n "$DATASET_NAME" || -n "$H2S_ROOT" || -n "$CSL_ROOT" || -n "$PHOENIX_ROOT" || -n "$MEAN_PATH" || -n "$STD_PATH" || -n "$CODE_PATH" ]]; then
  TMP_CFG="/tmp/soke_lm_train_${TS}.yaml"
  "$PYTHON_BIN" - <<PY
from omegaconf import OmegaConf
from mGPT.config import get_module_config

cfg_assets = OmegaConf.load("./configs/assets.yaml")
cfg_base = OmegaConf.load(f"{cfg_assets.CONFIG_FOLDER}/default.yaml")
cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load("$CFG"))
if not cfg_exp.FULL_CONFIG:
    cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
cfg = OmegaConf.merge(cfg_exp, cfg_assets)
if "$BATCH_SIZE":
    cfg.TRAIN.BATCH_SIZE = int("$BATCH_SIZE")
if "$END_EPOCH":
    cfg.TRAIN.END_EPOCH = int("$END_EPOCH")
    cfg.TRAIN.LR_SCHEDULER.params.T_max = int("$END_EPOCH")
if "$EXP_NAME":
    cfg.NAME = "$EXP_NAME"
if "$PRETRAINED_VAE":
    cfg.TRAIN.PRETRAINED_VAE = "$PRETRAINED_VAE"
if "$RESUME_CKPT":
    cfg.TRAIN.PRETRAINED = "$RESUME_CKPT"
    cfg.TRAIN.RESUME = ""
else:
    # Avoid inheriting default RESUME from base config (e.g. configs/soke.yaml),
    # which may accidentally load an incompatible old LM checkpoint.
    cfg.TRAIN.RESUME = ""
    cfg.TRAIN.PRETRAINED = ""
if "$NUM_WORKERS":
    cfg.TRAIN.NUM_WORKERS = int("$NUM_WORKERS")
if "$ACCUMULATE_GRAD_BATCHES":
    cfg.TRAIN.ACCUMULATE_GRAD_BATCHES = int("$ACCUMULATE_GRAD_BATCHES")
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
if "$CODE_PATH":
    cfg.DATASET.CODE_PATH = "$CODE_PATH"
cfg.FULL_CONFIG = True
OmegaConf.save(cfg, "$TMP_CFG")
print("saved", "$TMP_CFG")
PY
  RUN_CFG="$TMP_CFG"
fi

EXP_DIR=$($PYTHON_BIN - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$RUN_CFG")
model = str(cfg.model.target).split('.')[-2].lower()
folder = cfg.get("FOLDER", "experiments")
name = cfg.get("NAME", "SOKE")
print(f"{folder}/{model}/{name}")
PY
)

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
DEVICE_ARGS=()
for i in "${!GPUS[@]}"; do
  DEVICE_ARGS+=("$i")
done

cleanup_train() {
  if [[ -n "${TRAIN_PID:-}" ]]; then
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "[cleanup] terminating train pid=$TRAIN_PID"
      kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
      pkill -TERM -P "$TRAIN_PID" 2>/dev/null || true
      sleep 2
      kill -KILL -- "-$TRAIN_PID" 2>/dev/null || true
      pkill -KILL -P "$TRAIN_PID" 2>/dev/null || true
    fi
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
}

should_skip_tokenize() {
  local meta_path="$1"
  [[ "$FORCE_RETOKENIZE" == "1" ]] && return 1
  [[ ! -f "$meta_path" ]] && return 1
  RUN_CFG_ENV="$RUN_CFG" META_PATH_ENV="$meta_path" "$PYTHON_BIN" - <<'PY'
import json
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["RUN_CFG_ENV"])
meta_path = os.environ["META_PATH_ENV"]
with open(meta_path, "r", encoding="utf-8") as f:
    old = json.load(f)
cur = {
    "pretrained_vae": str(cfg.TRAIN.get("PRETRAINED_VAE", "") or ""),
    "pretrained_vae_body": str(cfg.TRAIN.get("PRETRAINED_VAE_BODY", "") or ""),
    "pretrained_vae_hand": str(cfg.TRAIN.get("PRETRAINED_VAE_HAND", "") or ""),
    "pretrained_vae_rhand": str(cfg.TRAIN.get("PRETRAINED_VAE_RHAND", "") or ""),
    "dataset_name": str(cfg.DATASET.H2S.get("DATASET_NAME", "") or ""),
    "code_path": str(cfg.DATASET.get("CODE_PATH", "") or ""),
    "motion_vae": str(cfg.model.params.motion_vae),
    "hand_vae_cfg": str(cfg.model.params.get("hand_vae_cfg", None)),
    "rhand_vae_cfg": str(cfg.model.params.get("rhand_vae_cfg", None)),
    "body_num_quantizers": int(cfg.model.params.motion_vae.params.get("num_quantizers", 1)),
    "hand_num_quantizers": int(cfg.model.params.get("hand_vae_cfg", {}).get("params", {}).get("num_quantizers", cfg.model.params.motion_vae.params.get("num_quantizers", 1)) if cfg.model.params.get("hand_vae_cfg", None) is not None else cfg.model.params.motion_vae.params.get("num_quantizers", 1)),
    "rhand_num_quantizers": int(cfg.model.params.get("rhand_vae_cfg", {}).get("params", {}).get("num_quantizers", cfg.model.params.motion_vae.params.get("num_quantizers", 1)) if cfg.model.params.get("rhand_vae_cfg", None) is not None else cfg.model.params.motion_vae.params.get("num_quantizers", 1)),
    "shared_num_quantizers": int(min([
        int(cfg.model.params.motion_vae.params.get("num_quantizers", 1)),
        int(cfg.model.params.get("hand_vae_cfg", {}).get("params", {}).get("num_quantizers", cfg.model.params.motion_vae.params.get("num_quantizers", 1)) if cfg.model.params.get("hand_vae_cfg", None) is not None else cfg.model.params.motion_vae.params.get("num_quantizers", 1)),
        int(cfg.model.params.get("rhand_vae_cfg", {}).get("params", {}).get("num_quantizers", cfg.model.params.motion_vae.params.get("num_quantizers", 1)) if cfg.model.params.get("rhand_vae_cfg", None) is not None else cfg.model.params.motion_vae.params.get("num_quantizers", 1)),
    ])),
    "body_codebook_size": int(cfg.model.params.motion_vae.params.get("code_num", 0)),
    "hand_codebook_size": int(cfg.model.params.get("hand_vae_cfg", {}).get("params", {}).get("code_num", cfg.model.params.motion_vae.params.get("code_num", 0)) if cfg.model.params.get("hand_vae_cfg", None) is not None else cfg.model.params.motion_vae.params.get("code_num", 0)),
    "rhand_codebook_size": int(cfg.model.params.get("rhand_vae_cfg", {}).get("params", {}).get("code_num", cfg.model.params.motion_vae.params.get("code_num", 0)) if cfg.model.params.get("rhand_vae_cfg", None) is not None else cfg.model.params.motion_vae.params.get("code_num", 0)),
    "q_offset_mode": "per_q_offset_v1",
}
raise SystemExit(0 if old == cur else 1)
PY
}

if [[ "$PREPARE_TOKENS" == "1" ]]; then
  CODE_ROOT=$($PYTHON_BIN - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$RUN_CFG")
print(f"{cfg.DATASET.H2S.ROOT}/{cfg.DATASET.CODE_PATH}")
PY
)
  META_PATH="${CODE_ROOT}/_tokenizer_meta.json"
  TOKEN_SKIP_EXISTING=1
  if [[ "$FORCE_RETOKENIZE" == "1" ]]; then
    TOKEN_SKIP_EXISTING=0
  elif [[ -f "$META_PATH" ]] && ! should_skip_tokenize "$META_PATH"; then
    # Cache metadata exists but no longer matches current tokenizer setup.
    # Rebuild all tokens in-place to avoid mixing stale/new tokenizations.
    TOKEN_SKIP_EXISTING=0
  fi
  if should_skip_tokenize "$META_PATH"; then
    echo "[1/4] Reusing cached motion tokens at $CODE_ROOT"
  else
    echo "[1/4] Preparing motion tokens ..."
    SKIP_EXISTING_TOKENS="$TOKEN_SKIP_EXISTING" "$PYTHON_BIN" scripts/get_motion_code.py \
      --cfg "$RUN_CFG" \
      --nodebug \
      --use_gpus "$EVAL_GPU" \
      --device 0
  fi
fi

if [[ "$TRAIN_LM" == "1" ]]; then
  echo "[2/4] Training LM ..."
  CMD=("$PYTHON_BIN" train.py
    --cfg "$RUN_CFG"
    --nodebug
    --use_gpus "$GPU_IDS"
    --num_nodes "$NUM_NODES"
    --device "${DEVICE_ARGS[@]}"
  )
  trap cleanup_train INT TERM EXIT
  set +e
  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" > >(tee "$TRAIN_LOG") 2>&1 &
  else
    "${CMD[@]}" > >(tee "$TRAIN_LOG") 2>&1 &
  fi
  TRAIN_PID=$!
  wait "$TRAIN_PID"
  RC=$?
  TRAIN_PID=""
  set -e
  trap - INT TERM EXIT
  if [[ $RC -ne 0 ]]; then
    echo "[FATAL] training failed, exit=$RC"
    exit "$RC"
  fi
fi

mkdir -p "$EXP_DIR/auto_reports/downstream"

if [[ -z "$EVAL_CKPT" ]]; then
  if compgen -G "$EXP_DIR/checkpoints/max-BLEU_4*.ckpt" > /dev/null; then
    EVAL_CKPT=$(ls -1t "$EXP_DIR"/checkpoints/max-BLEU_4*.ckpt | head -n 1)
  elif [[ -f "$EXP_DIR/checkpoints/last.ckpt" ]]; then
    EVAL_CKPT="$EXP_DIR/checkpoints/last.ckpt"
  else
    EVAL_CKPT=$(ls -1t "$EXP_DIR"/checkpoints/*.ckpt 2>/dev/null | head -n 1 || true)
  fi
fi

if [[ -z "$EVAL_CKPT" || ! -f "$EVAL_CKPT" ]]; then
  echo "[FATAL] no checkpoint found for evaluation in $EXP_DIR/checkpoints"
  exit 2
fi

echo "[info] experiment dir: $EXP_DIR"
echo "[info] eval ckpt: $EVAL_CKPT"

if [[ "$AUTO_EVAL_BLEU" == "1" ]]; then
  echo "[3/4] Running BLEU evaluation (m2t) ..."
  BLEU_CFG="/tmp/soke_lm_bleu_eval_${TS}.yaml"
  "$PYTHON_BIN" - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$RUN_CFG")
cfg.model.params.task = "m2t"
cfg.METRIC.TYPE = ["M2TMetrics"]
cfg.TEST.CHECKPOINTS = "$EVAL_CKPT"
cfg.TEST.SPLIT = "$EVAL_SPLIT"
cfg.TEST.BATCH_SIZE = int("$EVAL_BATCH_SIZE")
cfg.TEST.REPLICATION_TIMES = 1
cfg.TEST.SAVE_PREDICTIONS = False
cfg.EVAL.BATCH_SIZE = int("$EVAL_BATCH_SIZE")
OmegaConf.save(cfg, "$BLEU_CFG")
print("saved", "$BLEU_CFG")
PY

  "$PYTHON_BIN" test.py \
    --cfg "$BLEU_CFG" \
    --nodebug \
    --task m2t \
    --use_gpus "$EVAL_GPU" \
    --device 0 \
    --batch_size "$EVAL_BATCH_SIZE" 2>&1 | tee "$BLEU_LOG"

  {
    echo "checkpoint: $EVAL_CKPT"
    echo "eval_split: $EVAL_SPLIT"
    echo "log: $BLEU_LOG"
    echo "--- metric lines ---"
    grep -E "Bleu_|ROUGE|Metrics/" "$BLEU_LOG" || true
  } > "$EXP_DIR/auto_reports/downstream/bleu_eval_summary.txt"
fi

if [[ "$AUTO_VIS" == "1" ]]; then
  echo "[4/4] Running t2m generation + mesh visualization ..."
  VIS_TEST_CFG="/tmp/soke_lm_t2m_vis_${TS}.yaml"
  "$PYTHON_BIN" - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$RUN_CFG")
cfg.model.params.task = "t2m"
cfg.METRIC.TYPE = ["TM2TMetrics"]
cfg.TEST.CHECKPOINTS = "$EVAL_CKPT"
cfg.TEST.SPLIT = "$VIS_SPLIT"
cfg.TEST.BATCH_SIZE = int("$EVAL_BATCH_SIZE")
cfg.TEST.REPLICATION_TIMES = 1
cfg.TEST.SAVE_PREDICTIONS = True
cfg.EVAL.BATCH_SIZE = int("$EVAL_BATCH_SIZE")
OmegaConf.save(cfg, "$VIS_TEST_CFG")
print("saved", "$VIS_TEST_CFG")
PY

  "$PYTHON_BIN" test.py \
    --cfg "$VIS_TEST_CFG" \
    --nodebug \
    --task t2m \
    --use_gpus "$EVAL_GPU" \
    --device 0 \
    --batch_size "$EVAL_BATCH_SIZE" 2>&1 | tee "$T2M_TEST_LOG"

  VIS_ROOT="$EXP_DIR/auto_vis/${TS}"
  VIS_NPY_DIR="$VIS_ROOT/npy"
  VIS_VIDEO_DIR="$VIS_ROOT/videos"
  mkdir -p "$VIS_NPY_DIR" "$VIS_VIDEO_DIR"

  PRED_ROOT="results/mgpt/$(basename "$EXP_DIR")"
  "$PYTHON_BIN" - <<PY
import glob, os, pickle, numpy as np
pred_root = "$PRED_ROOT"
split = "$VIS_SPLIT"
out_dir = "$VIS_NPY_DIR"
k = int("$VIS_NUM_SAMPLES")
rank_dirs = sorted(glob.glob(os.path.join(pred_root, f"{split}_rank_*")))
pkls = []
for d in rank_dirs:
    pkls.extend(sorted(glob.glob(os.path.join(d, "*.pkl"))))
if not pkls:
    raise SystemExit(f"No prediction pkl found under {pred_root}/{split}_rank_*")
selected = pkls[:k]
os.makedirs(out_dir, exist_ok=True)
manifest = []
for p in selected:
    name = os.path.splitext(os.path.basename(p))[0]
    with open(p, "rb") as f:
        item = pickle.load(f)
    for tag, key in [("pred", "feats_rst"), ("gt", "feats_ref")]:
        arr = np.asarray(item[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        save_p = os.path.join(out_dir, f"{name}_{tag}.npy")
        np.save(save_p, arr)
        manifest.append(save_p)
mf = os.path.join(out_dir, "manifest.txt")
with open(mf, "w") as f:
    for p in manifest:
        f.write(p + "\n")
print("manifest", mf)
PY

  MANIFEST="$VIS_NPY_DIR/manifest.txt"
  VIS_MEAN_PATH=$($PYTHON_BIN - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$RUN_CFG")
print(cfg.DATASET.H2S.MEAN_PATH)
PY
)
  VIS_STD_PATH=$($PYTHON_BIN - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("$RUN_CFG")
print(cfg.DATASET.H2S.STD_PATH)
PY
)
  if [[ -f "$MANIFEST" ]]; then
    while IFS= read -r npy_path; do
      [[ -z "$npy_path" ]] && continue
      sample_name=$(basename "$npy_path" .npy)
      "$PYTHON_BIN" scripts/visualize_smplx_raw_mesh.py \
        --pose_npy "$npy_path" \
        --input_type feat133_norm \
        --mean_path "$VIS_MEAN_PATH" \
        --std_path "$VIS_STD_PATH" \
        --cam_y "$VIS_CAM_Y" \
        --input_fps "$VIS_INPUT_FPS" \
        --output_dir "$VIS_VIDEO_DIR" \
        --sample_name "$sample_name" || true
    done < "$MANIFEST"
  fi

  {
    echo "checkpoint: $EVAL_CKPT"
    echo "vis_split: $VIS_SPLIT"
    echo "pred_root: $PRED_ROOT"
    echo "npy_dir: $VIS_NPY_DIR"
    echo "video_dir: $VIS_VIDEO_DIR"
    echo "test_log: $T2M_TEST_LOG"
  } > "$EXP_DIR/auto_reports/downstream/visualization_summary.txt"
fi

echo "[done] downstream pipeline finished"
echo "  exp_dir: $EXP_DIR"
echo "  ckpt:    $EVAL_CKPT"
echo "  bleu:    $EXP_DIR/auto_reports/downstream/bleu_eval_summary.txt"
echo "  vis:     $EXP_DIR/auto_reports/downstream/visualization_summary.txt"
