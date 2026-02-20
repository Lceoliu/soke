#!/usr/bin/env bash
set -euo pipefail

# Run VAE sign ablation experiments:
#   A: body+hand pretrained -> sign finetune
#   B: body pretrained, hand scratch -> sign finetune
#   C: all scratch on sign
#
# Usage examples:
#   conda activate soke
#   bash scripts/pipeline/run_vae_sign_ablation_abc.sh
#
#   RUN=B GPU_IDS=0,1,2,3 BATCH_SIZE=32 \
#   bash scripts/pipeline/run_vae_sign_ablation_abc.sh

RUN=${RUN:-"all"}                 # all | A | B | C
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
NUM_NODES=${NUM_NODES:-1}
BATCH_SIZE=${BATCH_SIZE:-""}

LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "$LOG_DIR"

# Optional auto post-train report + visualization.
AUTO_POST=${AUTO_POST:-0}
AUTO_POST_STRICT=${AUTO_POST_STRICT:-0}
AUTO_POST_DEVICE=${AUTO_POST_DEVICE:-"cuda"}
AUTO_REPORT_MAX_SAMPLES=${AUTO_REPORT_MAX_SAMPLES:-3000}
AUTO_VIS_TRAIN=${AUTO_VIS_TRAIN:-2}
AUTO_VIS_TEST=${AUTO_VIS_TEST:-2}

CFG_A="configs/vae/ablation_vae_sign_A_baseline_pretrained_all.yaml"
CFG_B="configs/vae/ablation_vae_sign_B_body_pretrained_hand_scratch.yaml"
CFG_C="configs/vae/ablation_vae_sign_C_scratch_all.yaml"

for p in "$CFG_A" "$CFG_B" "$CFG_C"; do
  if [[ ! -f "$p" ]]; then
    echo "[FATAL] Missing config: $p"
    exit 2
  fi
done

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

run_one() {
  local tag="$1"
  local cfg="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="$LOG_DIR/ablation_${tag}_${ts}.log"

  local cmd=(python3 train.py
    --cfg "$cfg"
    --nodebug
    --use_gpus "$GPU_IDS"
    --num_nodes "$NUM_NODES"
    --device "${DEVICE_ARGS[@]}"
  )
  if [[ -n "$BATCH_SIZE" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
  fi

  local train_pid=""

  _cleanup_train() {
    if [[ -n "$train_pid" ]] && kill -0 "$train_pid" 2>/dev/null; then
      echo "[Ablation-$tag][cleanup] terminating train pid=$train_pid"
      pkill -TERM -P "$train_pid" 2>/dev/null || true
      kill -TERM "$train_pid" 2>/dev/null || true
      sleep 2
      pkill -KILL -P "$train_pid" 2>/dev/null || true
      kill -KILL "$train_pid" 2>/dev/null || true
    fi
  }
  trap _cleanup_train INT TERM

  echo "============================================================"
  echo "[Ablation-$tag] CFG=$cfg"
  echo "[Ablation-$tag] CMD: ${cmd[*]}"
  echo "[Ablation-$tag] LOG: $log_file"
  echo "============================================================"
  set +e
  "${cmd[@]}" > >(tee "$log_file") 2>&1 &
  train_pid=$!
  wait "$train_pid"
  local rc=$?
  set -e
  train_pid=""
  trap - INT TERM

  if [[ $rc -ne 0 ]]; then
    echo "[Ablation-$tag][FATAL] training failed with exit code $rc"
    return "$rc"
  fi

  if [[ "$AUTO_POST" == "1" ]]; then
    echo "[Ablation-$tag][post] Running automatic report + visualization ..."
    local post_cmd=(python3 scripts/analysis/auto_post_train_eval_vis.py
      --cfg "$cfg"
      --log_path "$log_file"
      --device "$AUTO_POST_DEVICE"
      --report_max_samples "$AUTO_REPORT_MAX_SAMPLES"
      --vis_train_num "$AUTO_VIS_TRAIN"
      --vis_test_num "$AUTO_VIS_TEST"
    )
    if [[ "$AUTO_POST_STRICT" == "1" ]]; then
      post_cmd+=(--strict)
    fi
    "${post_cmd[@]}" || {
      if [[ "$AUTO_POST_STRICT" == "1" ]]; then
        exit 1
      fi
      echo "[Ablation-$tag][post][WARN] post-processing failed, continue."
    }
  fi
}

case "${RUN^^}" in
  A)
    run_one "A" "$CFG_A"
    ;;
  B)
    run_one "B" "$CFG_B"
    ;;
  C)
    run_one "C" "$CFG_C"
    ;;
  ALL)
    run_one "A" "$CFG_A"
    run_one "B" "$CFG_B"
    run_one "C" "$CFG_C"
    ;;
  *)
    echo "[FATAL] Invalid RUN=$RUN. Use one of: all, A, B, C"
    exit 2
    ;;
esac

echo "[DONE] run_vae_sign_ablation_abc.sh finished (RUN=$RUN)"
