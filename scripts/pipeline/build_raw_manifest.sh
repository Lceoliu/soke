#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Build raw manifest by scanning local folders.

Example:
  bash scripts/pipeline/build_raw_manifest.sh \
    --scan "source=motionx,split=train,format=npy,path=/data/MOTION-X/train,glob=**/*.npy,layout=smplx179" \
    --scan "source=motionx,split=val,format=npy,path=/data/MOTION-X/val,glob=**/*.npy,layout=smplx179" \
    --scan "source=amass,split=train,format=npy,path=/data/AMASS,glob=**/*.npy,layout=soke133" \
    --output data/large_vae/raw_manifest.jsonl \
    --sort
USAGE
  exit 1
fi

python3 scripts/pipeline/build_raw_manifest_from_scan.py "$@"
