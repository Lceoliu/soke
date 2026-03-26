#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_gzip_pickle(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _save_gzip_pickle(path: Path, data):
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _symlink_force(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"Refuse to overwrite real directory: {dst}")
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def build_single_sample_subset(src_root: Path, dst_root: Path, sample_name: str, split: str):
    ann_path = src_root / f"csl_clean.{split}"
    ann = _load_gzip_pickle(ann_path)
    selected = [x for x in ann if str(x["name"]) == sample_name]
    if not selected:
        raise FileNotFoundError(f"Sample {sample_name!r} not found in {ann_path}")

    dst_root.mkdir(parents=True, exist_ok=True)
    _symlink_force(src_root / "poses", dst_root / "poses")
    for fname in ["mean.pt", "std.pt"]:
        _symlink_force(src_root / fname, dst_root / fname)

    for s in ["train", "val", "test"]:
        _save_gzip_pickle(dst_root / f"csl_clean.{s}", selected)

    with open(dst_root / "selected_samples.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(selected), f, ensure_ascii=False, indent=2)


def run(cmd: list[str], env: dict[str, str] | None = None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR), env=env)


def run_render_with_fallback(cmd: list[str], env: dict[str, str]):
    explicit = env.get("PYOPENGL_PLATFORM", "").strip().lower()
    candidates = [explicit] if explicit else ["egl", "osmesa"]
    last_error = None
    for platform in candidates:
        cur_env = dict(env)
        cur_env["PYOPENGL_PLATFORM"] = platform
        try:
            run(cmd, env=cur_env)
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            print(f"[warn] render failed with PYOPENGL_PLATFORM={platform}, trying next backend...", flush=True)
    if last_error is not None:
        raise last_error


def locate_prediction_pkl(sample_name: str, run_name: str, preferred_root: Path) -> Path:
    patterns = [
        str(preferred_root / "**" / f"{sample_name}.pkl"),
        str((ROOT_DIR / "results") / "**" / run_name / "**" / f"{sample_name}.pkl"),
        str((ROOT_DIR / "results") / "**" / f"{sample_name}.pkl"),
    ]
    matches = []
    import glob
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    uniq = sorted({str(Path(m).resolve()) for m in matches})
    if not uniq:
        raise FileNotFoundError(
            f"No prediction pkl found for sample {sample_name!r}. "
            f"Searched under {preferred_root} and {ROOT_DIR / 'results'}"
        )
    uniq.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(uniq[0])


def main():
    ap = argparse.ArgumentParser(description="Visualize one CSL sample for LM t2m/mc checkpoint.")
    ap.add_argument("--cfg", required=True, help="Training config yaml")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--task", required=True, choices=["t2m", "mc"])
    ap.add_argument("--sample_name", required=True, help="CSL sample name, e.g. S000000_P0000_T00")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--src_root", default="data/CSL-Daily")
    ap.add_argument("--subset_root", default="/tmp/soke_csl_onevis")
    ap.add_argument("--results_root", default="/tmp/soke_onevis_results")
    ap.add_argument("--output_dir", default="", help="Final render root. Default: <exp_dir>/onevis_<task>_<sample>")
    ap.add_argument("--use_gpus", default="0")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--cam_y", type=float, default=-0.2)
    ap.add_argument("--input_fps", type=float, default=20.0)
    ap.add_argument("--mesh_rx_deg", type=float, default=180.0)
    ap.add_argument("--mesh_ry_deg", type=float, default=0.0)
    args = ap.parse_args()

    cfg_path = Path(args.cfg).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    src_root = (ROOT_DIR / args.src_root).resolve()
    subset_root = Path(args.subset_root).resolve() / f"{args.sample_name}_{args.task}"
    results_root = Path(args.results_root).resolve()

    build_single_sample_subset(src_root, subset_root, args.sample_name, args.split)

    cfg = OmegaConf.load(str(cfg_path))
    cfg.NAME = f"ONEVIS_{args.task}_{args.sample_name}"
    cfg.FOLDER = str(results_root)
    cfg.model.params.task = args.task
    cfg.METRIC.TYPE = []
    cfg.DATASET.H2S.DATASET_NAME = "csl"
    cfg.DATASET.H2S.CSL_ROOT = str(subset_root)
    cfg.DATASET.H2S.MEAN_PATH = str(subset_root / "mean.pt")
    cfg.DATASET.H2S.STD_PATH = str(subset_root / "std.pt")
    cfg.TEST.CHECKPOINTS = str(ckpt_path)
    cfg.TEST.SPLIT = args.split
    cfg.TEST.BATCH_SIZE = int(args.batch_size)
    cfg.TEST.REPLICATION_TIMES = 1
    cfg.TEST.SAVE_PREDICTIONS = True
    cfg.EVAL.BATCH_SIZE = int(args.batch_size)
    cfg.TEST.FOLDER = str(results_root)
    cfg.FULL_CONFIG = True

    tmp_cfg = Path("/tmp") / f"soke_onevis_{args.task}_{args.sample_name}.yaml"
    OmegaConf.save(cfg, str(tmp_cfg))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.use_gpus)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env["PYTHONUNBUFFERED"] = "1"

    run(
        [
            sys.executable,
            "test.py",
            "--cfg",
            str(tmp_cfg),
            "--nodebug",
            "--task",
            args.task,
            "--use_gpus",
            str(args.use_gpus),
            "--device",
            str(args.device),
            "--batch_size",
            str(args.batch_size),
        ],
        env=env,
    )

    preferred_pred_root = results_root / "mgpt" / cfg.NAME
    pkl_path = locate_prediction_pkl(args.sample_name, cfg.NAME, preferred_pred_root)
    pred_root = pkl_path.parent.parent

    with open(pkl_path, "rb") as f:
        item = pickle.load(f)

    out_root = Path(args.output_dir).resolve() if args.output_dir else (pred_root / f"onevis_{args.task}_{args.sample_name}")
    npy_dir = out_root / "npy"
    video_dir = out_root / "videos"
    npy_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for tag, key in [("pred", "feats_rst"), ("gt", "feats_ref")]:
        arr = item[key]
        save_p = npy_dir / f"{args.sample_name}_{tag}.npy"
        np.save(save_p, np.asarray(arr, dtype=np.float32))
        written.append(save_p)

    for npy_path in written:
        sample_tag = npy_path.stem
        run_render_with_fallback(
            [
                sys.executable,
                "scripts/visualize_smplx_raw_mesh.py",
                "--pose_npy",
                str(npy_path),
                "--input_type",
                "feat133_norm",
                "--mean_path",
                str(subset_root / "mean.pt"),
                "--std_path",
                str(subset_root / "std.pt"),
                "--cam_y",
                str(args.cam_y),
                "--input_fps",
                str(args.input_fps),
                "--mesh_rx_deg",
                str(args.mesh_rx_deg),
                "--mesh_ry_deg",
                str(args.mesh_ry_deg),
                "--output_dir",
                str(video_dir),
                "--sample_name",
                sample_tag,
            ],
            env=env,
        )

    note = {
        "task": args.task,
        "sample_name": args.sample_name,
        "split": args.split,
        "subset_root": str(subset_root),
        "pred_root": str(pred_root),
        "pkl_path": str(pkl_path),
        "npy_dir": str(npy_dir),
        "video_dir": str(video_dir),
        "note": "For mc, current visualization renders GT suffix vs predicted suffix only; prefix is not included in video.",
    }
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(note, f, ensure_ascii=False, indent=2)

    print(json.dumps(note, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
