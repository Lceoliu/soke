#!/usr/bin/env python3
import argparse
import gzip
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.utils.smplx_contact import detect_contacts_from_smplx, load_pose179_from_smplx_dir


CHANNELS = ["lhand_rhand", "lhand_face", "rhand_face"]
PAIR_ORDER = (("lhand", "rhand"), ("lhand", "face"), ("rhand", "face"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute per-frame binary contact labels [T,3] for sign datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "how2sign", "csl", "phoenix"],
        help="Which dataset(s) to process.",
    )
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits.")

    parser.add_argument("--how2sign_root", type=str, default="data/How2Sign")
    parser.add_argument("--csl_root", type=str, default="data/CSL-Daily")
    parser.add_argument("--phoenix_root", type=str, default="data/Phoenix_2014T")
    parser.add_argument("--output_dir_name", type=str, default="contact_labels")

    parser.add_argument("--threshold", type=float, default=0.02, help="Contact threshold in meter.")
    parser.add_argument("--device", type=str, default="", help="cuda/cpu. default: auto")
    parser.add_argument("--chunk_a", type=int, default=512)
    parser.add_argument("--chunk_b", type=int, default=2048)
    parser.add_argument(
        "--fk_chunk_frames",
        type=int,
        default=256,
        help="Temporal chunk size for FK+contact. Use <=0 for full-clip.",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def sample_indices(input_len: int, count: int) -> np.ndarray:
    if count <= 0:
        count = 1
    ss = float(input_len) / float(count)
    return np.asarray([int(np.floor(i * ss)) for i in range(count)], dtype=np.int64)


def gather_how2sign_items(root: Path, splits: List[str], output_dir_name: str) -> List[Dict]:
    items = []
    for split in splits:
        csv_path = root / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"
        if not csv_path.exists():
            continue
        ann = pd.read_csv(csv_path)
        ann["DURATION"] = ann["END_REALIGNED"] - ann["START_REALIGNED"]
        ann = ann[ann["DURATION"] < 30].reset_index(drop=True)
        for _, row in ann.iterrows():
            name = str(row["SENTENCE_NAME"])
            pose_dir = root / split / "poses" / name
            out_path = root / split / output_dir_name / f"{name}.pkl"
            items.append(
                {
                    "src": "how2sign",
                    "name": name,
                    "split": split,
                    "fps": float(row["fps"]),
                    "pose_dir": pose_dir,
                    "out_path": out_path,
                }
            )
    return items


def gather_csl_items(root: Path, splits: List[str], output_dir_name: str) -> List[Dict]:
    items = []
    seen = set()
    for split in splits:
        ann_path = root / f"csl_clean.{split}"
        if not ann_path.exists():
            continue
        with gzip.open(ann_path, "rb") as f:
            ann = pickle.load(f)
        for row in ann:
            name = str(row["name"])
            if name in seen:
                continue
            seen.add(name)
            pose_dir = root / "poses" / name
            out_path = root / output_dir_name / f"{name}.pkl"
            items.append(
                {
                    "src": "csl",
                    "name": name,
                    "split": split,
                    "fps": None,
                    "pose_dir": pose_dir,
                    "out_path": out_path,
                }
            )
    return items


def gather_phoenix_items(root: Path, splits: List[str], output_dir_name: str) -> List[Dict]:
    items = []
    seen = set()
    for split in splits:
        split_key = "dev" if split == "val" else split
        ann_path = root / f"phoenix14t.{split_key}"
        if not ann_path.exists():
            continue
        with gzip.open(ann_path, "rb") as f:
            ann = pickle.load(f)
        for row in ann:
            name = str(row["name"])
            if name in seen:
                continue
            seen.add(name)
            pose_dir = root / name
            out_path = root / output_dir_name / f"{name}.pkl"
            items.append(
                {
                    "src": "phoenix",
                    "name": name,
                    "split": split,
                    "fps": None,
                    "pose_dir": pose_dir,
                    "out_path": out_path,
                }
            )
    return items


def compute_contact_labels_for_item(item: Dict, args, device: str) -> Optional[np.ndarray]:
    pose179 = load_pose179_from_smplx_dir(item["pose_dir"])
    if pose179.shape[0] < 4:
        return None

    if item["src"] == "how2sign" and item["fps"] is not None and item["fps"] > 24:
        count = int(24 * len(pose179) / float(item["fps"]))
        idx = sample_indices(len(pose179), count)
        pose179 = pose179[idx]

    if pose179.shape[0] < 4:
        return None

    thr_map = {
        "lhand-rhand": float(args.threshold),
        "lhand-face": float(args.threshold),
        "rhand-face": float(args.threshold),
    }

    if int(args.fk_chunk_frames) > 0:
        labels_all = []
        chunk = int(args.fk_chunk_frames)
        for st in range(0, pose179.shape[0], chunk):
            ed = min(st + chunk, pose179.shape[0])
            results = detect_contacts_from_smplx(
                data=pose179[st:ed],
                data_type="pose179",
                pairs=PAIR_ORDER,
                threshold=thr_map,
                device=device,
                chunk_a=int(args.chunk_a),
                chunk_b=int(args.chunk_b),
                bool_only=True,
            )
            labels = np.stack(
                [
                    results["lhand-rhand"]["is_contact"].detach().cpu().numpy().astype(np.uint8),
                    results["lhand-face"]["is_contact"].detach().cpu().numpy().astype(np.uint8),
                    results["rhand-face"]["is_contact"].detach().cpu().numpy().astype(np.uint8),
                ],
                axis=-1,
            )
            labels_all.append(labels)
        return np.concatenate(labels_all, axis=0)

    results = detect_contacts_from_smplx(
        data=pose179,
        data_type="pose179",
        pairs=PAIR_ORDER,
        threshold=thr_map,
        device=device,
        chunk_a=int(args.chunk_a),
        chunk_b=int(args.chunk_b),
        bool_only=True,
    )
    labels = np.stack(
        [
            results["lhand-rhand"]["is_contact"].detach().cpu().numpy().astype(np.uint8),
            results["lhand-face"]["is_contact"].detach().cpu().numpy().astype(np.uint8),
            results["rhand-face"]["is_contact"].detach().cpu().numpy().astype(np.uint8),
        ],
        axis=-1,
    )
    return labels


def main():
    args = parse_args()
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    items: List[Dict] = []
    if args.dataset in ["all", "how2sign"]:
        items.extend(gather_how2sign_items(Path(args.how2sign_root), splits, args.output_dir_name))
    if args.dataset in ["all", "csl"]:
        items.extend(gather_csl_items(Path(args.csl_root), splits, args.output_dir_name))
    if args.dataset in ["all", "phoenix"]:
        items.extend(gather_phoenix_items(Path(args.phoenix_root), splits, args.output_dir_name))

    items = [x for x in items if x["pose_dir"].exists()]
    items.sort(key=lambda x: (x["src"], x["split"], x["name"]))

    start_idx = max(int(args.start_idx), 0)
    end_idx = len(items) if int(args.end_idx) < 0 else min(int(args.end_idx), len(items))
    items = items[start_idx:end_idx]
    if int(args.max_samples) > 0:
        items = items[: int(args.max_samples)]

    print(f"[INFO] device={device}, dataset={args.dataset}, splits={splits}, items={len(items)}")
    if args.dry_run:
        for item in items[:20]:
            print(f"[DRY] {item['src']} {item['name']} -> {item['out_path']}")
        return

    ok = 0
    skip_exist = 0
    skip_short = 0
    failed = 0

    for item in tqdm(items, desc="precompute_contact", dynamic_ncols=True):
        out_path = item["out_path"]
        if out_path.exists() and (not args.overwrite):
            skip_exist += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            labels = compute_contact_labels_for_item(item, args, device=device)
            if labels is None or labels.shape[0] < 4:
                skip_short += 1
                continue
            payload = {
                "labels": labels.astype(np.uint8, copy=False),
                "channels": CHANNELS,
                "threshold": float(args.threshold),
                "src": item["src"],
                "name": item["name"],
                "split": item["split"],
            }
            with open(out_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] failed: {item['src']} {item['name']} ({item['pose_dir']}): {e}")

    print("[DONE] summary:")
    print(f"  success      : {ok}")
    print(f"  skip_exists  : {skip_exist}")
    print(f"  skip_short   : {skip_short}")
    print(f"  failed       : {failed}")


if __name__ == "__main__":
    main()
