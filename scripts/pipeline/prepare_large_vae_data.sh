#!/usr/bin/env bash
set -euo pipefail

# Prepare processed corpus + mean/std for large-scale VAE pretraining.
#
# Required:
#   RAW_MANIFEST: raw-manifest jsonl path
#
# Optional env vars:
#   OUTPUT_ROOT, PROCESSED_MANIFEST, ERROR_LOG
#   TARGET_NFEATS, TARGET_FPS, MIN_FRAMES, MAX_FRAMES
#   NUM_WORKERS, STATS_SPLIT
#   MEAN_PATH, STD_PATH

RAW_MANIFEST=${RAW_MANIFEST:-""}
OUTPUT_ROOT=${OUTPUT_ROOT:-"data/large_vae/processed"}
PROCESSED_MANIFEST=${PROCESSED_MANIFEST:-"data/large_vae/processed_manifest.jsonl"}
ERROR_LOG=${ERROR_LOG:-"data/large_vae/preprocess_errors.log"}

TARGET_NFEATS=${TARGET_NFEATS:-133}
TARGET_FPS=${TARGET_FPS:-24}
MIN_FRAMES=${MIN_FRAMES:-40}
MAX_FRAMES=${MAX_FRAMES:-0}
NUM_WORKERS=${NUM_WORKERS:-8}

STATS_SPLIT=${STATS_SPLIT:-train}
MEAN_PATH=${MEAN_PATH:-"data/large_vae/mean.pt"}
STD_PATH=${STD_PATH:-"data/large_vae/std.pt"}

if [[ -z "${RAW_MANIFEST}" ]]; then
  echo "[error] RAW_MANIFEST is empty."
  echo "Usage example:"
  echo "  RAW_MANIFEST=data/large_vae/raw_manifest.jsonl bash scripts/pipeline/prepare_large_vae_data.sh"
  exit 1
fi

mkdir -p "$(dirname "$PROCESSED_MANIFEST")"
mkdir -p "$(dirname "$MEAN_PATH")"

echo "[1/2] Preprocess raw corpus -> unified npy"
python3 scripts/pipeline/preprocess_vae_corpus.py \
  --raw-manifest "$RAW_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --output-manifest "$PROCESSED_MANIFEST" \
  --error-log "$ERROR_LOG" \
  --target-nfeats "$TARGET_NFEATS" \
  --target-fps "$TARGET_FPS" \
  --min-frames "$MIN_FRAMES" \
  --max-frames "$MAX_FRAMES" \
  --num-workers "$NUM_WORKERS" \
  --skip-existing

echo "[2/2] Compute streaming mean/std"
python3 scripts/pipeline/compute_mean_std_stream.py \
  --manifest "$PROCESSED_MANIFEST" \
  --split "$STATS_SPLIT" \
  --mean-out "$MEAN_PATH" \
  --std-out "$STD_PATH"

echo "Done."
echo "  Processed manifest: $PROCESSED_MANIFEST"
echo "  Mean path:          $MEAN_PATH"
echo "  Std path:           $STD_PATH"
