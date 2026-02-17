#!/usr/bin/env bash
set -euo pipefail

CFG=${CFG:-"configs/vae/large_vae_pretrain.yaml"}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}

LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-"$LOG_DIR/vae_pretrain_$(date +%Y%m%d_%H%M%S).log"}

# NCCL / distributed stability knobs for long-running large-scale jobs.
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
