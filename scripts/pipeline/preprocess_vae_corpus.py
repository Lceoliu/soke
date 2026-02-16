#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(x, **kwargs):  # type: ignore
        return x

SMPLX_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]


def _sample_uniform(lst: List[str], count: int) -> List[str]:
    if count <= 0:
        return []
    if count >= len(lst):
        return lst
    idx = np.linspace(0, len(lst) - 1, num=count, dtype=int)
    return [lst[i] for i in idx]


def _uniform_sample_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if target_len >= arr.shape[0]:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, num=target_len, dtype=int)
    return arr[idx]


def _sort_key_by_number(path: Path):
    m = re.findall(r"\d+", path.name)
    if m:
        return int(m[-1])
    return path.name


def smplx179_to_soke133(poses: np.ndarray) -> np.ndarray:
    """Convert SMPL-X 179-dim pose to repository-standard 133-dim features."""
    if poses.ndim != 2 or poses.shape[1] != 179:
        raise ValueError(f"Expected [T,179], got {poses.shape}")

    poses = poses[:, (3 + 3 * 11):]
    poses = np.concatenate([poses[:, :-20], poses[:, -10:]], axis=1)
    return poses.astype(np.float32, copy=False)


def motionx322_to_soke133(poses: np.ndarray) -> np.ndarray:
    """
    Convert Motion-X SMPL-X 322-dim format to SOKE 133-dim feature layout.

    Motion-X layout (from load_data.md):
      root_orient [0:3]
      pose_body   [3:66]
      pose_hand   [66:156] (lhand 45 + rhand 45)
      pose_jaw    [156:159]
      face_expr   [159:209] (50 dims)
      face_shape  [209:309] (100 dims)
      trans       [309:312]
      betas       [312:]

    SOKE 133 layout (used in this repo):
      upper_body(30) + lhand(45) + rhand(45) + jaw(3) + expr(10)
    """
    if poses.ndim != 2 or poses.shape[1] != 322:
        raise ValueError(f"Expected [T,322], got {poses.shape}")

    body = poses[:, 3:66]           # 63
    lhand = poses[:, 66:111]        # 45
    rhand = poses[:, 111:156]       # 45
    jaw = poses[:, 156:159]         # 3
    expr = poses[:, 159:169]        # first 10 dims of face expression

    upper_body = body[:, 33:63]     # drop lower-body related 11 joints => keep 30 dims
    out = np.concatenate([upper_body, lhand, rhand, jaw, expr], axis=1)
    if out.shape[1] != 133:
        raise RuntimeError(f"Internal conversion error, got {out.shape}")
    return out.astype(np.float32, copy=False)


def load_smplx_dir(path: str, target_fps: float = 24.0, src_fps: Optional[float] = None) -> np.ndarray:
    base = Path(path)
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"SMPL-X dir not found: {path}")

    files = [p for p in base.iterdir() if p.suffix in {".pkl", ".pt"}]
    if len(files) == 0:
        raise RuntimeError(f"No .pkl/.pt files found under {path}")

    files = sorted(files, key=_sort_key_by_number)

    if src_fps is not None and target_fps is not None and src_fps > target_fps:
        tgt_count = max(1, int(target_fps * len(files) / src_fps))
        files = _sample_uniform([str(x) for x in files], tgt_count)
        files = [Path(x) for x in files]

    clip = np.zeros((len(files), 179), dtype=np.float32)
    for i, f in enumerate(files):
        if f.suffix == ".pt":
            import torch
            data = torch.load(f, map_location="cpu", weights_only=False)
        else:
            with f.open("rb") as fp:
                data = pickle.load(fp)
        try:
            clip[i] = np.concatenate([data[k] for k in SMPLX_KEYS], axis=0).astype(np.float32)
        except KeyError as e:
            raise KeyError(f"Missing key in {f}: {e}")
    return clip


def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,C], got {arr.shape} from {path}")
    return arr.astype(np.float32, copy=False)


def _safe_name(entry: Dict) -> str:
    if entry.get("name"):
        base_name = str(entry["name"])
    else:
        p = Path(entry["path"])
        base_name = p.stem if p.is_file() else p.name

    digest = hashlib.md5(entry["path"].encode("utf-8")).hexdigest()[:8]
    return f"{base_name}_{digest}"


def process_one(
    entry: Dict,
    output_root: str,
    target_nfeats: int,
    min_frames: int,
    max_frames: int,
    target_fps: float,
    skip_existing: bool,
) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        fmt = entry["format"]
        src = entry.get("source", "generic")
        split = entry.get("split", "train")
        layout = entry.get("layout", "soke133")

        if fmt == "smplx_pkl_dir":
            arr = load_smplx_dir(
                entry["path"],
                target_fps=target_fps,
                src_fps=entry.get("src_fps"),
            )
            arr = smplx179_to_soke133(arr)
        elif fmt == "npy":
            arr = load_npy(entry["path"])
            if layout == "smplx179":
                arr = smplx179_to_soke133(arr)
            elif layout == "motionx322":
                arr = motionx322_to_soke133(arr)
            elif layout == "soke133":
                pass
            else:
                raise ValueError(
                    f"Unsupported layout={layout}, expected one of: smplx179, motionx322, soke133"
                )
        else:
            raise ValueError(f"Unsupported format={fmt}")

        if arr.shape[0] < min_frames:
            return None, f"too_short({arr.shape[0]})::{entry['path']}"

        if max_frames > 0 and arr.shape[0] > max_frames:
            arr = _uniform_sample_array(arr, max_frames)

        if arr.shape[1] != target_nfeats:
            raise ValueError(
                f"Feature dim mismatch: got {arr.shape[1]}, expected {target_nfeats}, path={entry['path']}"
            )

        name = _safe_name(entry)
        rel = Path(src) / split / f"{name}.npy"
        out_path = Path(output_root) / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not (skip_existing and out_path.exists()):
            np.save(out_path, arr)

        out_record = {
            "path": str(out_path.resolve()),
            "relpath": str(rel),
            "source": src,
            "split": split,
            "name": name,
            "length": int(arr.shape[0]),
            "nfeats": int(arr.shape[1]),
            "raw_path": entry["path"],
            "raw_format": fmt,
        }
        return out_record, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            items.append(json.loads(ln))
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess mixed raw motion corpora into unified 133-dim npy files for large-scale VAE pretraining."
    )
    parser.add_argument("--raw-manifest", required=True, help="Input raw-manifest jsonl")
    parser.add_argument("--output-root", required=True, help="Output folder for processed npy")
    parser.add_argument("--output-manifest", required=True, help="Output processed-manifest jsonl")
    parser.add_argument("--error-log", default="", help="Optional error log path")

    parser.add_argument("--target-nfeats", type=int, default=133)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no max clipping")

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Raise on first error")
    args = parser.parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)

    entries = read_jsonl(args.raw_manifest)
    print(f"Loaded raw entries: {len(entries)}")

    ok_records: List[Dict] = []
    errors: List[str] = []

    if args.num_workers <= 1:
        iterator = entries
        for entry in tqdm(iterator, desc="preprocess"):
            rec, err = process_one(
                entry=entry,
                output_root=args.output_root,
                target_nfeats=args.target_nfeats,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                target_fps=args.target_fps,
                skip_existing=args.skip_existing,
            )
            if rec is not None:
                ok_records.append(rec)
            if err is not None:
                if args.strict:
                    raise RuntimeError(err)
                errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [
                ex.submit(
                    process_one,
                    entry,
                    args.output_root,
                    args.target_nfeats,
                    args.min_frames,
                    args.max_frames,
                    args.target_fps,
                    args.skip_existing,
                )
                for entry in entries
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
                rec, err = fut.result()
                if rec is not None:
                    ok_records.append(rec)
                if err is not None:
                    if args.strict:
                        raise RuntimeError(err)
                    errors.append(err)

    # Keep stable manifest order by path.
    ok_records.sort(key=lambda x: x["path"])

    with open(args.output_manifest, "w", encoding="utf-8") as f:
        for r in ok_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.error_log:
        Path(args.error_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.error_log, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(e + "\n")

    print(f"Processed OK: {len(ok_records)}")
    print(f"Errors: {len(errors)}")
    print(f"Processed manifest: {args.output_manifest}")
    if args.error_log:
        print(f"Error log: {args.error_log}")


if __name__ == "__main__":
    main()
