#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import get_module_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-train automation: generate RVQ report + sample reconstruction videos."
    )
    parser.add_argument("--cfg", required=True, help="Training cfg path used in current run")
    parser.add_argument("--cfg_assets", default="configs/assets.yaml")
    parser.add_argument("--exp_dir", default="", help="Optional experiment directory override")
    parser.add_argument("--log_path", default="", help="Optional train log path override")
    parser.add_argument("--ckpt_path", default="", help="Optional checkpoint path override")
    parser.add_argument("--python_bin", default="python3")

    parser.add_argument("--skip_report", action="store_true")
    parser.add_argument("--skip_vis", action="store_true")
    parser.add_argument("--report_max_samples", type=int, default=3000)
    parser.add_argument("--report_curve_every", type=int, default=50)
    parser.add_argument("--report_batch_size", type=int, default=32)
    parser.add_argument("--report_num_workers", type=int, default=4)

    parser.add_argument("--vis_train_num", type=int, default=2)
    parser.add_argument("--vis_test_num", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for report sampling and mesh visualization",
    )
    parser.add_argument("--strict", action="store_true", help="Fail fast on post-step errors")
    return parser.parse_args()


def load_cfg(cfg_path: str, cfg_assets_path: str):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    return OmegaConf.merge(cfg_exp, cfg_assets)


def infer_exp_dir(cfg) -> Path:
    model = str(cfg.model.target).split(".")[-2].lower()
    return Path(cfg.FOLDER) / model / str(cfg.NAME)


def newest_file(base: Path, pattern: str) -> Optional[Path]:
    candidates = list(base.glob(pattern))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def choose_ckpt(exp_dir: Path, ckpt_override: str) -> Optional[Path]:
    if ckpt_override:
        p = Path(ckpt_override)
        return p if p.exists() else None
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None

    best = newest_file(ckpt_dir, "min-val_loss-*.ckpt")
    if best is not None:
        return best
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    return newest_file(ckpt_dir, "*.ckpt")


def choose_log(exp_dir: Path, log_override: str) -> Optional[Path]:
    if log_override:
        p = Path(log_override)
        return p if p.exists() else None
    return newest_file(exp_dir, "log_*_train.log")


def choose_runtime_cfg(exp_dir: Path, cfg_fallback: Path) -> Path:
    cfg_saved = newest_file(exp_dir, "config_*_train.yaml")
    return cfg_saved if cfg_saved is not None else cfg_fallback


def run_cmd(cmd: List[str], strict: bool):
    print("[post] run:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[post][WARN] command failed with code={e.returncode}")
        if strict:
            raise
        return False


def load_pickle_gz(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def has_pose_frames(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    for x in p.iterdir():
        if x.suffix in {".pkl", ".pt"}:
            return True
    return False


def sample_how2sign(root: Path, split: str, limit: int) -> List[Dict]:
    out: List[Dict] = []
    csv_path = root / split / "re_aligned" / f"how2sign_realigned_{split}_preprocessed_fps.csv"
    if not csv_path.exists():
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("SENTENCE_NAME", "")).strip()
            if not name:
                continue
            pose_dir = root / split / "poses" / name
            if not has_pose_frames(pose_dir):
                continue
            fps = row.get("fps", "")
            try:
                fps_val = float(fps) if fps not in ["", None] else None
            except Exception:
                fps_val = None
            out.append(
                {
                    "source": "how2sign",
                    "split": split,
                    "name": name,
                    "pose_dir": str(pose_dir),
                    "input_fps": fps_val,
                }
            )
            if len(out) >= limit:
                break
    return out


def sample_csl(root: Path, split: str, limit: int) -> List[Dict]:
    out: List[Dict] = []
    ann_path = root / f"csl_clean.{split if split != 'train' else 'train'}"
    if not ann_path.exists():
        return out
    ann = load_pickle_gz(ann_path)
    for item in ann:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        pose_dir = root / "poses" / name
        if not has_pose_frames(pose_dir):
            continue
        out.append(
            {
                "source": "csl",
                "split": split,
                "name": name,
                "pose_dir": str(pose_dir),
                "input_fps": None,
            }
        )
        if len(out) >= limit:
            break
    return out


def sample_phoenix(root: Path, split: str, limit: int) -> List[Dict]:
    out: List[Dict] = []
    split_map = {"train": "train", "val": "dev", "test": "test"}
    ann_path = root / f"phoenix14t.{split_map[split]}"
    if not ann_path.exists():
        return out
    ann = load_pickle_gz(ann_path)
    for item in ann:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        pose_dir = root / name
        if not has_pose_frames(pose_dir):
            continue
        out.append(
            {
                "source": "phoenix",
                "split": split,
                "name": name,
                "pose_dir": str(pose_dir),
                "input_fps": None,
            }
        )
        if len(out) >= limit:
            break
    return out


def round_robin_take(groups: List[List[Dict]], k: int) -> List[Dict]:
    if k <= 0:
        return []
    groups = [list(g) for g in groups if g]
    out: List[Dict] = []
    idx = 0
    while groups and len(out) < k:
        g = groups[idx % len(groups)]
        if g:
            out.append(g.pop(0))
            if len(out) >= k:
                break
        groups = [x for x in groups if x]
        if not groups:
            break
        idx += 1
    return out


def select_vis_samples(cfg, split: str, k: int) -> List[Dict]:
    if k <= 0:
        return []
    if str(cfg.DATASET.target) != "mGPT.data.H2S.H2SDataModule":
        return []

    h2s_cfg = cfg.DATASET.H2S
    dataset_name = str(h2s_cfg.DATASET_NAME).lower()
    groups: List[List[Dict]] = []

    if "how2sign" in dataset_name:
        groups.append(sample_how2sign(Path(h2s_cfg.ROOT), split, max(2 * k, 8)))
    if "csl" in dataset_name:
        groups.append(sample_csl(Path(h2s_cfg.CSL_ROOT), split, max(2 * k, 8)))
    if "phoenix" in dataset_name:
        groups.append(sample_phoenix(Path(h2s_cfg.PHOENIX_ROOT), split, max(2 * k, 8)))

    return round_robin_take(groups, k)


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg, args.cfg_assets)

    exp_dir = Path(args.exp_dir) if args.exp_dir else infer_exp_dir(cfg)
    if not exp_dir.exists():
        msg = f"[post][WARN] experiment dir not found: {exp_dir}"
        print(msg)
        if args.strict:
            raise RuntimeError(msg)
        return

    ckpt = choose_ckpt(exp_dir, args.ckpt_path)
    log_path = choose_log(exp_dir, args.log_path)
    runtime_cfg = choose_runtime_cfg(exp_dir, Path(args.cfg))
    try:
        cfg_runtime = load_cfg(str(runtime_cfg), args.cfg_assets)
    except Exception:
        cfg_runtime = cfg

    print(f"[post] exp_dir={exp_dir}")
    print(f"[post] runtime_cfg={runtime_cfg}")
    print(f"[post] ckpt={ckpt}")
    print(f"[post] log={log_path}")

    if ckpt is None:
        msg = "[post][WARN] no checkpoint found, skip report/visualization"
        print(msg)
        if args.strict:
            raise RuntimeError(msg)
        return

    # 1) RVQ report
    if not args.skip_report:
        if log_path is None:
            print("[post][WARN] no train log found, skip RVQ report")
        else:
            out_dir = exp_dir / "auto_reports" / "rvq_stage1"
            cmd = [
                args.python_bin,
                "scripts/analysis/generate_rvq_stage1_report.py",
                "--cfg",
                str(runtime_cfg),
                "--cfg_assets",
                args.cfg_assets,
                "--log_path",
                str(log_path),
                "--ckpt_path",
                str(ckpt),
                "--output_dir",
                str(out_dir),
                "--batch_size",
                str(args.report_batch_size),
                "--num_workers",
                str(args.report_num_workers),
                "--max_samples",
                str(args.report_max_samples),
                "--curve_every",
                str(args.report_curve_every),
                "--device",
                str(args.device),
            ]
            run_cmd(cmd, strict=args.strict)

    # 2) Reconstruction visualizations
    if not args.skip_vis:
        samples_train = select_vis_samples(cfg_runtime, split="train", k=args.vis_train_num)
        samples_test = select_vis_samples(cfg_runtime, split="test", k=args.vis_test_num)
        samples = samples_train + samples_test

        if len(samples) == 0:
            print("[post][WARN] no visualization samples found for current dataset config")
        else:
            vis_root = exp_dir / "auto_vis"
            vis_root.mkdir(parents=True, exist_ok=True)

            meta = {
                "checkpoint": str(ckpt),
                "runtime_cfg": str(runtime_cfg),
                "samples": samples,
            }
            with open(vis_root / "selected_samples.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            for sample in samples:
                split = sample["split"]
                out_dir = vis_root / split
                out_dir.mkdir(parents=True, exist_ok=True)
                sample_name = f"{sample['source']}__{sample['name']}"
                cmd = [
                    args.python_bin,
                    "scripts/tokenize_reconstruct_mesh_one.py",
                    "--cfg",
                    str(runtime_cfg),
                    "--tokenizer_ckpt",
                    str(ckpt),
                    "--pose_dir",
                    sample["pose_dir"],
                    "--sample_name",
                    sample_name,
                    "--output_dir",
                    str(out_dir),
                    "--device",
                    str(args.device),
                ]
                if sample.get("input_fps") is not None:
                    cmd.extend(["--input_fps", str(sample["input_fps"])])
                run_cmd(cmd, strict=args.strict)

    print("[post] done")


if __name__ == "__main__":
    main()
