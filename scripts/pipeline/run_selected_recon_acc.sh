#!/usr/bin/env bash
set -euo pipefail

CFG="/home/SOKE/experiments/VAE_SIGN_FINETUNE_LFQ4_ACC/config_2026-03-06-01-16-27_train.yaml"
CKPT="/home/SOKE/experiments/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt"
SAMPLES_TXT="/home/SOKE/data/CSL-Daily/selected_samples.txt"
POSE_ROOT="/home/SOKE/data/CSL-Daily/poses"
OUT_ROOT="/home/SOKE/visualize/0307show/lfq4_finetune_recon_acc"

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${SAMPLES_TXT}" ]]; then
  echo "[ERR] missing samples file: ${SAMPLES_TXT}"
  exit 1
fi

mapfile -t SAMPLE_IDS < <(awk 'NF>0 && $1!="train" && $1!="test" {print $1}' "${SAMPLES_TXT}")
TOTAL="${#SAMPLE_IDS[@]}"
echo "[INFO] total selected samples: ${TOTAL}"

ok=0
fail=0

for sid in "${SAMPLE_IDS[@]}"; do
  pose_dir="${POSE_ROOT}/${sid}"
  if [[ ! -d "${pose_dir}" ]]; then
    echo "[WARN] skip missing pose dir: ${pose_dir}"
    fail=$((fail + 1))
    continue
  fi

  echo "[RUN] ${sid}"
  if conda run -n soke python /home/SOKE/scripts/tokenize_reconstruct_mesh_one.py \
      --cfg "${CFG}" \
      --tokenizer_ckpt "${CKPT}" \
      --pose_dir "${pose_dir}" \
      --output_dir "${OUT_ROOT}"; then
    ok=$((ok + 1))
  else
    echo "[ERR] failed: ${sid}"
    fail=$((fail + 1))
  fi
done

echo "[DONE] success=${ok}, failed=${fail}, total=${TOTAL}"
