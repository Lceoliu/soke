#!/usr/bin/env bash
set -euo pipefail

# Batch contact statistics for CSL-Daily SMPLerX pose dirs.
# Default behavior:
# - randomly sample 15 clips from data/CSL-Daily/poses
# - run contact detection for each clip
# - aggregate per-sample + overall summaries

usage() {
  cat <<'EOF'
Usage:
  bash scripts/analysis/batch_contact_stats_csl_daily.sh [options]

Options:
  --data_root PATH         Root directory of pose clips (default: data/CSL-Daily/poses)
  --num_samples N          Number of sampled clips (default: 15)
  --seed N                 Random seed for sampling (default: 1234)
  --pairs STR              Comma-separated pairs (default: lhand-face,rhand-face,lhand-rhand)
  --bool_only 0|1          Only detect contact bool (default: 1)
  --thr_face_hand FLOAT    Threshold for hand-face (default: 0.02)
  --thr_hand_hand FLOAT    Threshold for left-right hand (default: 0.02)
  --device STR             cuda/cpu (default: cuda)
  --chunk_a N              Chunk size A for pairwise cdist (default: 256)
  --chunk_b N              Chunk size B for pairwise cdist (default: 1024)
  --output_dir PATH        Output directory (default: reports/contact_csl_daily_<time>)
  -h, --help               Show this help
EOF
}

DATA_ROOT=${DATA_ROOT:-"data/CSL-Daily/poses"}
NUM_SAMPLES=${NUM_SAMPLES:-15}
SEED=${SEED:-1234}
PAIRS=${PAIRS:-"lhand-face,rhand-face,lhand-rhand"}
BOOL_ONLY=${BOOL_ONLY:-1}
THR_FACE_HAND=${THR_FACE_HAND:-0.02}
THR_HAND_HAND=${THR_HAND_HAND:-0.02}
DEVICE=${DEVICE:-cuda}
CHUNK_A=${CHUNK_A:-256}
CHUNK_B=${CHUNK_B:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-"reports/contact_csl_daily_$(date +%Y%m%d_%H%M%S)"}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --pairs) PAIRS="$2"; shift 2 ;;
    --bool_only) BOOL_ONLY="$2"; shift 2 ;;
    --thr_face_hand) THR_FACE_HAND="$2"; shift 2 ;;
    --thr_hand_hand) THR_HAND_HAND="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --chunk_a) CHUNK_A="$2"; shift 2 ;;
    --chunk_b) CHUNK_B="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[FATAL] data_root does not exist: $DATA_ROOT"
  exit 2
fi

if [[ "$BOOL_ONLY" != "0" && "$BOOL_ONLY" != "1" ]]; then
  echo "[FATAL] --bool_only must be 0 or 1"
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
SAMPLES_TXT="$OUTPUT_DIR/selected_samples.txt"
RUN_LOG="$OUTPUT_DIR/run.log"
SUMMARY_JSON="$OUTPUT_DIR/summary_all.json"
SUMMARY_CSV="$OUTPUT_DIR/summary_all.csv"
PER_SAMPLE_DIR="$OUTPUT_DIR/per_sample"
mkdir -p "$PER_SAMPLE_DIR"

exec > >(tee "$RUN_LOG") 2>&1

echo "[INFO] output_dir: $OUTPUT_DIR"
echo "[INFO] data_root: $DATA_ROOT"
echo "[INFO] num_samples: $NUM_SAMPLES, seed: $SEED"
echo "[INFO] pairs: $PAIRS, bool_only: $BOOL_ONLY"
echo "[INFO] thresholds: face_hand=$THR_FACE_HAND, hand_hand=$THR_HAND_HAND"
echo "[INFO] device=$DEVICE, chunk_a=$CHUNK_A, chunk_b=$CHUNK_B"

# Ensure python runs in soke env.
if [[ "${CONDA_DEFAULT_ENV:-}" != "soke" ]]; then
  if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate soke
  else
    echo "[WARN] conda.sh not found; continuing with current python env."
  fi
fi

python - <<PY
import random
from pathlib import Path

data_root = Path("$DATA_ROOT")
num_samples = int("$NUM_SAMPLES")
seed = int("$SEED")
out_path = Path("$SAMPLES_TXT")

clips = sorted([str(p) for p in data_root.iterdir() if p.is_dir()])
if len(clips) == 0:
    raise SystemExit(f"[FATAL] no clip directories found under {data_root}")

rng = random.Random(seed)
if num_samples >= len(clips):
    selected = clips
else:
    selected = rng.sample(clips, num_samples)
    selected = sorted(selected)

out_path.write_text("\\n".join(selected) + "\\n", encoding="utf-8")
print(f"[INFO] selected {len(selected)} / {len(clips)} clips")
PY

IDX=0
while IFS= read -r clip_dir; do
  [[ -z "$clip_dir" ]] && continue
  clip_name="$(basename "$clip_dir")"
  out_json="$PER_SAMPLE_DIR/${IDX}_${clip_name}.json"

  cmd=(
    python scripts/analysis/detect_smplx_contacts.py
    --pose_dir "$clip_dir"
    --pairs "$PAIRS"
    --thr_face_hand "$THR_FACE_HAND"
    --thr_hand_hand "$THR_HAND_HAND"
    --device "$DEVICE"
    --chunk_a "$CHUNK_A"
    --chunk_b "$CHUNK_B"
    --output_json "$out_json"
  )
  if [[ "$BOOL_ONLY" == "1" ]]; then
    cmd+=(--bool_only)
  fi

  echo "[RUN][$IDX] $clip_name"
  "${cmd[@]}"
  IDX=$((IDX + 1))
done < "$SAMPLES_TXT"

python - <<PY
import csv
import glob
import json
from pathlib import Path

per_sample = sorted(glob.glob("$PER_SAMPLE_DIR/*.json"))
if not per_sample:
    raise SystemExit("[FATAL] no per-sample result json found")

records = []
all_pairs = set()
for jp in per_sample:
    with open(jp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    inp = obj.get("input", "")
    name = Path(inp).name
    num_frames = int(obj.get("num_frames", 0))
    summary = obj.get("summary", {})
    row = {
        "clip_name": name,
        "num_frames": num_frames,
    }
    for pair, vals in summary.items():
        all_pairs.add(pair)
        row[f"{pair}_contact_ratio"] = float(vals.get("contact_ratio", 0.0))
        if "min_of_min_dist" in vals:
            row[f"{pair}_min_of_min_dist"] = float(vals["min_of_min_dist"])
        if "mean_min_dist" in vals:
            row[f"{pair}_mean_min_dist"] = float(vals["mean_min_dist"])
    records.append(row)

all_pairs = sorted(all_pairs)
agg = {
    "num_samples": len(records),
    "pairs": all_pairs,
    "overall": {},
}

for pair in all_pairs:
    ratios = [float(r.get(f"{pair}_contact_ratio", 0.0)) for r in records]
    agg["overall"][pair] = {
        "mean_contact_ratio": float(sum(ratios) / max(1, len(ratios))),
        "samples_with_contact": int(sum(1 for x in ratios if x > 0)),
        "sample_contact_fraction": float(sum(1 for x in ratios if x > 0) / max(1, len(ratios))),
    }

out_json = Path("$SUMMARY_JSON")
out_json.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

fields = ["clip_name", "num_frames"]
for pair in all_pairs:
    fields.append(f"{pair}_contact_ratio")
    # min-dist fields may be absent in bool_only mode; still keep compatible schema.
    fields.append(f"{pair}_min_of_min_dist")
    fields.append(f"{pair}_mean_min_dist")

out_csv = Path("$SUMMARY_CSV")
with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in records:
        writer.writerow(r)

print(f"[DONE] per-sample dir: $PER_SAMPLE_DIR")
print(f"[DONE] summary json: {out_json}")
print(f"[DONE] summary csv: {out_csv}")
PY

echo "[DONE] All finished."
