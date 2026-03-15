#!/usr/bin/env bash
set -euo pipefail

SAMPLES_TXT="${SAMPLES_TXT:-/home/SOKE/data/CSL-Daily/selected_samples.txt}"
POSE_ROOT="${POSE_ROOT:-/home/SOKE/data/CSL-Daily/poses}"
OUT_ROOT="${OUT_ROOT:-/home/SOKE/visualize/0307show/mesh_origin}"
CONDA_ENV="${CONDA_ENV:-soke}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${SAMPLES_TXT}" ]]; then
  echo "[ERR] missing samples file: ${SAMPLES_TXT}"
  exit 1
fi

mapfile -t SAMPLE_IDS < <(awk 'NF>0 && $1!="train" && $1!="test" {print $1}' "${SAMPLES_TXT}")
TOTAL="${#SAMPLE_IDS[@]}"
if [[ "${TOTAL}" -eq 0 ]]; then
  echo "[ERR] no valid sample ids found in ${SAMPLES_TXT}"
  exit 1
fi

echo "[INFO] total selected samples: ${TOTAL}"
echo "[INFO] output dir: ${OUT_ROOT}"

ok=0
fail=0

for sid in "${SAMPLE_IDS[@]}"; do
  pose_dir="${POSE_ROOT}/${sid}"
  if [[ ! -d "${pose_dir}" ]]; then
    echo "[WARN] missing pose dir, skip: ${pose_dir}"
    fail=$((fail + 1))
    continue
  fi

  echo "[RUN] ${sid}"
  if conda run -n "${CONDA_ENV}" python /home/SOKE/scripts/visualize_smplx_raw_mesh.py \
      --pose_dir "${pose_dir}" \
      --output_dir "${OUT_ROOT}" \
      --cam_y -0.5 \
      --input_fps 50 \
      --device "${DEVICE}"; then
    ok=$((ok + 1))
  else
    echo "[ERR] failed: ${sid}"
    fail=$((fail + 1))
  fi
done

echo "[DONE] success=${ok}, failed=${fail}, total=${TOTAL}"
