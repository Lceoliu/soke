#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.utils.smplx_contact import (
    DEFAULT_CONTACT_PAIRS,
    canonical_part_name,
    detect_contacts_from_smplx,
    load_pose179_from_smplx_dir,
    summarize_contact_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect SMPL-X part contacts from vertices (face/lhand/rhand)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pose_dir", type=str, default="", help="Directory of per-frame SMPLerX pkl/pt.")
    src.add_argument("--pose_npy", type=str, default="", help="Clip npy file [T,C].")

    parser.add_argument(
        "--input_type",
        type=str,
        default="pose179",
        choices=["pose179", "feat133_raw", "feat133_norm"],
        help="Input interpretation for --pose_npy. Ignored for --pose_dir.",
    )
    parser.add_argument("--mean_path", type=str, default="", help="Required if input_type=feat133_norm.")
    parser.add_argument("--std_path", type=str, default="", help="Required if input_type=feat133_norm.")

    parser.add_argument("--thr_face_hand", type=float, default=0.02, help="Threshold (meter) for hand-face.")
    parser.add_argument("--thr_hand_hand", type=float, default=0.02, help="Threshold (meter) for left-right hand.")
    parser.add_argument("--device", type=str, default="", help="cuda/cpu. default: auto")
    parser.add_argument("--chunk_a", type=int, default=512)
    parser.add_argument("--chunk_b", type=int, default=2048)
    parser.add_argument(
        "--pairs",
        type=str,
        default="lhand-face,rhand-face,lhand-rhand",
        help="Comma-separated pairs. Example: lhand-face,rhand-face",
    )
    parser.add_argument(
        "--bool_only",
        action="store_true",
        help="Only compute contact bool (distance<threshold), skip exact min-distance for speed.",
    )

    parser.add_argument("--output_json", type=str, default="", help="Output JSON path.")
    parser.add_argument("--output_npz", type=str, default="", help="Optional NPZ path for per-frame arrays.")
    return parser.parse_args()


def _load_mean_std(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        x = np.load(path)
    else:
        x = torch.load(path, map_location="cpu")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D stats, got {x.shape} from {path}")
    if x.shape[0] == 179:
        x = x[(3 + 3 * 11):]
        x = np.concatenate([x[:-20], x[-10:]], axis=0)
    if x.shape[0] != 133:
        raise ValueError(f"Stats must be 133 or 179 dim, got {x.shape[0]} from {path}")
    return x.astype(np.float32, copy=False)


def _load_input(args) -> Tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    mean133 = std133 = None
    if args.pose_dir:
        arr = load_pose179_from_smplx_dir(args.pose_dir)
        data_type = "pose179"
        return arr, data_type, mean133, std133

    arr = np.asarray(np.load(args.pose_npy), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,C], got {arr.shape} from {args.pose_npy}")
    data_type = args.input_type

    if data_type == "feat133_norm":
        if not args.mean_path or not args.std_path:
            raise ValueError("--mean_path and --std_path are required for feat133_norm")
        mean133 = _load_mean_std(args.mean_path)
        std133 = _load_mean_std(args.std_path)
    return arr, data_type, mean133, std133


def _parse_pairs(pairs_str: str):
    if not pairs_str.strip():
        return DEFAULT_CONTACT_PAIRS
    pairs = []
    for token in pairs_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Invalid pair token '{token}', expected format a-b")
        a, b = token.split("-", 1)
        pairs.append((canonical_part_name(a), canonical_part_name(b)))
    if not pairs:
        raise ValueError("No valid pairs parsed from --pairs")
    return tuple(pairs)


def main():
    args = parse_args()
    data, data_type, mean133, std133 = _load_input(args)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    thresholds_default: Dict[str, float] = {
        "lhand-face": float(args.thr_face_hand),
        "rhand-face": float(args.thr_face_hand),
        "lhand-rhand": float(args.thr_hand_hand),
    }

    pairs = _parse_pairs(args.pairs)
    thresholds: Dict[str, float] = {}
    for a, b in pairs:
        k = f"{a}-{b}"
        rk = f"{b}-{a}"
        if k in thresholds_default:
            thresholds[k] = thresholds_default[k]
        elif rk in thresholds_default:
            thresholds[k] = thresholds_default[rk]
        elif {a, b} == {"lhand", "rhand"}:
            thresholds[k] = float(args.thr_hand_hand)
        else:
            thresholds[k] = float(args.thr_face_hand)
    results = detect_contacts_from_smplx(
        data=data,
        data_type=data_type,
        pairs=pairs,
        threshold=thresholds,
        mean133=mean133,
        std133=std133,
        device=device,
        chunk_a=int(args.chunk_a),
        chunk_b=int(args.chunk_b),
        bool_only=bool(args.bool_only),
    )
    summary = summarize_contact_results(results)

    payload = {
        "input": args.pose_dir if args.pose_dir else args.pose_npy,
        "input_type": data_type,
        "num_frames": int(data.shape[0]),
        "device": device,
        "bool_only": bool(args.bool_only),
        "pairs": [f"{a}-{b}" for a, b in pairs],
        "summary": summary,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(text, encoding="utf-8")
        print(f"[saved] json => {out_json}")

    if args.output_npz:
        out_npz = Path(args.output_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np_payload = {}
        for pair, vals in results.items():
            is_contact = vals["is_contact"].detach().cpu().numpy().astype(np.uint8)
            np_payload[f"{pair}_is_contact"] = is_contact
            min_dist = vals.get("min_dist", None)
            if isinstance(min_dist, torch.Tensor):
                np_payload[f"{pair}_min_dist"] = min_dist.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(str(out_npz), **np_payload)
        print(f"[saved] npz => {out_npz}")


if __name__ == "__main__":
    main()
