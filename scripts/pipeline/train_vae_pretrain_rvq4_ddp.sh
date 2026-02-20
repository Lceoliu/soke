#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-"configs/vae/motionx_vae_pretrain_rvq4.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}

LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/vae_pretrain_rvq4_$(date +%Y%m%d_%H%M%S).log"}
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

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
DEVICE_ARGS=()
for i in "${!GPUS[@]}"; do
  DEVICE_ARGS+=("$i")
done

CMD=(python3 train.py
  --cfg "$CFG"
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
    --cfg "$CFG"
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
