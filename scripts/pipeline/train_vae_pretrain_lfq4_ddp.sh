#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-"configs/vae/motionx_vae_pretrain_lfq4.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}

LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/vae_pretrain_lfq4_$(date +%Y%m%d_%H%M%S).log"}
AUTO_POST=${AUTO_POST:-1}
AUTO_POST_STRICT=${AUTO_POST_STRICT:-0}
AUTO_POST_DEVICE=${AUTO_POST_DEVICE:-"cuda"}
AUTO_REPORT_MAX_SAMPLES=${AUTO_REPORT_MAX_SAMPLES:-3000}
AUTO_VIS_TRAIN=${AUTO_VIS_TRAIN:-2}
AUTO_VIS_TEST=${AUTO_VIS_TEST:-2}

export NCCL_TIMEOUT=${NCCL_TIMEOUT:-600}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-0}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
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
TRAIN_PID=""
cleanup_train() {
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "[cleanup] terminating train pid=$TRAIN_PID"
    pkill -TERM -P "$TRAIN_PID" 2>/dev/null || true
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    sleep 2
    pkill -KILL -P "$TRAIN_PID" 2>/dev/null || true
    kill -KILL "$TRAIN_PID" 2>/dev/null || true
  fi
}
trap cleanup_train INT TERM
set +e
"${CMD[@]}" > >(tee "$LOG_FILE") 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID"
RC=$?
set -e
TRAIN_PID=""
trap - INT TERM

if [[ $RC -ne 0 ]]; then
  echo "[FATAL] training failed with exit code $RC"
  exit "$RC"
fi

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
