#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def merge_stats(
    n_a: int,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    n_b: int,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
):
    if n_a == 0:
        return n_b, mean_b, m2_b
    if n_b == 0:
        return n_a, mean_a, m2_a

    delta = mean_b - mean_a
    n = n_a + n_b
    mean = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + delta * delta * (n_a * n_b / n)
    return n, mean, m2


def file_stats(path: str):
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,C], got {arr.shape} at {path}")

    n = int(arr.shape[0])
    mean = arr.mean(axis=0)
    centered = arr - mean
    m2 = (centered * centered).sum(axis=0)
    return n, mean, m2


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-feature mean/std for large motion corpus with streaming parallel-merge statistics."
    )
    parser.add_argument("--manifest", required=True, help="Processed manifest jsonl")
    parser.add_argument("--split", default="train", help="Only use this split")
    parser.add_argument("--mean-out", required=True)
    parser.add_argument("--std-out", required=True)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    entries = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            item = json.loads(ln)
            if item.get("split", "train") != args.split:
                continue
            entries.append(item)

    if args.max_samples > 0:
        entries = entries[: args.max_samples]

    if len(entries) == 0:
        raise RuntimeError(f"No entries found for split={args.split} in {args.manifest}")

    n_total = 0
    mean_total = None
    m2_total = None

    for item in tqdm(entries, desc="stats"):
        n_i, mean_i, m2_i = file_stats(item["path"])

        if mean_total is None:
            mean_total = mean_i
            m2_total = m2_i
            n_total = n_i
        else:
            n_total, mean_total, m2_total = merge_stats(
                n_total,
                mean_total,
                m2_total,
                n_i,
                mean_i,
                m2_i,
            )

    if n_total <= 1:
        raise RuntimeError(f"Not enough frames to compute std, n_total={n_total}")

    var = m2_total / (n_total - 1)
    var = np.maximum(var, args.eps)
    std = np.sqrt(var)

    Path(args.mean_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.std_out).parent.mkdir(parents=True, exist_ok=True)

    # Save with torch when available (for backward compatibility with existing configs),
    # otherwise fall back to numpy format.
    try:
        import torch

        mean_t = torch.tensor(mean_total, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32)
        torch.save(mean_t, args.mean_out)
        torch.save(std_t, args.std_out)
    except Exception:
        with open(args.mean_out, "wb") as f:
            np.save(f, mean_total.astype(np.float32))
        with open(args.std_out, "wb") as f:
            np.save(f, std.astype(np.float32))

    print(f"Saved mean -> {args.mean_out}")
    print(f"Saved std  -> {args.std_out}")
    print(f"Frames used: {n_total}")


if __name__ == "__main__":
    main()
