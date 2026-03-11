#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


FRAME_RE = re.compile(r"(\d+)(?:_(\d+))?\.npz$")

OUT_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]

CONF_KEY_HINTS = (
    "score",
    "conf",
    "confidence",
    "bbox_score",
    "person_score",
    "track_score",
    "det_score",
)


class OneEuroFilter:
    # Reference: Casiez et al., 1€ filter.
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.005, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(dt: float, cutoff):
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, t: float, x: np.ndarray):
        if self.t_prev is None:
            self.t_prev = float(t)
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x, dtype=np.float32)
            return x

        dt = max(float(t) - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.t_prev = float(t)
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat


def _flat(arr, target_dim: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    if x.shape[0] >= target_dim:
        return x[:target_dim].astype(np.float32, copy=False)
    pad = np.zeros((target_dim - x.shape[0],), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)


def _extract_confidence(npz_obj) -> Optional[float]:
    vals = []
    for k in npz_obj.keys():
        k_l = str(k).lower()
        if any(h in k_l for h in CONF_KEY_HINTS):
            v = np.asarray(npz_obj[k]).reshape(-1)
            if v.size == 1 and np.isfinite(v[0]):
                vals.append(float(v[0]))
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def load_smplerx_npz(path: str):
    data = np.load(path, allow_pickle=True)

    frame = {
        "smplx_root_pose": _flat(data.get("global_orient", np.zeros((1, 3), dtype=np.float32)), 3),
        "smplx_body_pose": _flat(data.get("body_pose", np.zeros((21, 3), dtype=np.float32)), 63),
        "smplx_lhand_pose": _flat(data.get("left_hand_pose", np.zeros((15, 3), dtype=np.float32)), 45),
        "smplx_rhand_pose": _flat(data.get("right_hand_pose", np.zeros((15, 3), dtype=np.float32)), 45),
        "smplx_jaw_pose": _flat(data.get("jaw_pose", np.zeros((1, 3), dtype=np.float32)), 3),
        "smplx_shape": _flat(data.get("betas", np.zeros((1, 10), dtype=np.float32)), 10),
        "smplx_expr": _flat(data.get("expression", np.zeros((1, 10), dtype=np.float32)), 10),
        "transl": _flat(data.get("transl", np.zeros((1, 3), dtype=np.float32)), 3),
    }
    conf = _extract_confidence(data)
    return frame, conf


def select_frame_files(smplx_dir: Path) -> List[Tuple[int, str]]:
    frame_groups: Dict[int, List[Tuple[int, str]]] = {}
    for p in smplx_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".npz":
            continue
        m = FRAME_RE.match(p.name)
        if m is None:
            continue
        fid = int(m.group(1))
        pid = int(m.group(2)) if m.group(2) is not None else 0
        frame_groups.setdefault(fid, []).append((pid, str(p)))

    out = []
    for fid in sorted(frame_groups.keys()):
        cands = sorted(frame_groups[fid], key=lambda x: x[0])  # prefer pid=0
        out.append((fid, cands[0][1]))
    return out


def deterministic_split(name: str, train_ratio: float, val_ratio: float) -> str:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    v = int(h, 16) / float(0xFFFFFFFF)
    if v < train_ratio:
        return "train"
    if v < train_ratio + val_ratio:
        return "val"
    return "test"


def apply_one_euro(arr: np.ndarray, t: np.ndarray, min_cutoff: float, beta: float, d_cutoff: float):
    if arr.shape[0] <= 1:
        return arr
    filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = filt(float(t[i]), arr[i].astype(np.float32, copy=False))
    return out


@dataclass
class WorkerCfg:
    output_root: str
    default_text: str
    fps: float
    min_frames: int
    use_one_euro: bool
    one_euro_min_cutoff: float
    one_euro_beta: float
    one_euro_d_cutoff: float
    confidence_threshold: float
    transl_abs_max: float
    jump_pose_scale: float
    jump_mad_k: float
    jump_min_score: float
    spike_mad_k: float
    spike_bridge_ratio: float
    train_ratio: float
    val_ratio: float
    overwrite: bool
    stats_from_existing: bool


def _read_feature179_from_pkl(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        d = pickle.load(f)
    feat = np.concatenate([np.asarray(d[k], dtype=np.float32).reshape(-1) for k in OUT_KEYS], axis=0)
    if feat.shape[0] != 179:
        raise ValueError(f"Expected feature dim 179, got {feat.shape[0]} in {p}")
    return feat.astype(np.float64, copy=False)


def _sum_saved_clip(out_dir: Path):
    pkl_files = sorted([p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pkl"])
    if len(pkl_files) == 0:
        return None, None, 0
    s = np.zeros((179,), dtype=np.float64)
    ss = np.zeros((179,), dtype=np.float64)
    n = 0
    for p in pkl_files:
        x = _read_feature179_from_pkl(p)
        s += x
        ss += x * x
        n += 1
    return s, ss, n


def process_one_clip(name: str, smplx_dir: str, cfg: WorkerCfg):
    split = deterministic_split(name, cfg.train_ratio, cfg.val_ratio)
    out_dir = Path(cfg.output_root) / "poses" / name
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    if out_dir.exists() and (not cfg.overwrite):
        pkl_files = sorted([p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pkl"])
        if len(pkl_files) >= cfg.min_frames:
            sum_vec = sq_sum_vec = None
            count = 0
            if split == "train" and cfg.stats_from_existing:
                sum_vec, sq_sum_vec, count = _sum_saved_clip(out_dir)
            return {
                "ok": True,
                "name": name,
                "split": split,
                "num_frames": len(pkl_files),
                "dropped_invalid": 0,
                "dropped_low_conf": 0,
                "dropped_jump": 0,
                "sum_vec": sum_vec,
                "sq_sum_vec": sq_sum_vec,
                "count": count,
                "from_existing": True,
            }

    frame_files = select_frame_files(Path(smplx_dir))
    if len(frame_files) < cfg.min_frames:
        return {"ok": False, "name": name, "reason": f"too_few_raw_frames:{len(frame_files)}"}

    frame_ids = []
    rows = []
    confs = []
    dropped_invalid = 0
    for fid, path in frame_files:
        try:
            d, conf = load_smplerx_npz(path)
        except Exception:
            dropped_invalid += 1
            continue

        feat = np.concatenate([d[k] for k in OUT_KEYS], axis=0)
        if feat.shape[0] != 179 or not np.all(np.isfinite(feat)):
            dropped_invalid += 1
            continue
        if not np.all(np.isfinite(d["transl"])) or np.linalg.norm(d["transl"]) > cfg.transl_abs_max:
            dropped_invalid += 1
            continue

        frame_ids.append(fid)
        rows.append(d)
        confs.append(np.nan if conf is None else float(conf))

    if len(rows) < cfg.min_frames:
        return {"ok": False, "name": name, "reason": f"too_few_valid_frames:{len(rows)}"}

    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    confs = np.asarray(confs, dtype=np.float32)
    keep = np.ones((len(rows),), dtype=bool)

    dropped_low_conf = 0
    if cfg.confidence_threshold >= 0:
        # Apply only where explicit confidence is available.
        has_conf = np.isfinite(confs)
        low_conf = has_conf & (confs < cfg.confidence_threshold)
        keep &= ~low_conf
        dropped_low_conf = int(low_conf.sum())

    # Transition outlier filtering (for unstable pose fits / identity switches).
    dropped_jump = 0
    dropped_spike = 0
    idxs = np.where(keep)[0]
    if len(idxs) >= 3:
        feat_pose = np.stack(
            [
                np.concatenate(
                    [
                        rows[i]["smplx_root_pose"],
                        rows[i]["smplx_body_pose"],
                        rows[i]["smplx_lhand_pose"],
                        rows[i]["smplx_rhand_pose"],
                        rows[i]["smplx_jaw_pose"],
                    ],
                    axis=0,
                )
                for i in idxs
            ],
            axis=0,
        )
        transl = np.stack([rows[i]["transl"] for i in idxs], axis=0)
        fid = frame_ids[idxs]

        d_pose = np.linalg.norm(feat_pose[1:] - feat_pose[:-1], axis=1)
        d_trans = np.linalg.norm(transl[1:] - transl[:-1], axis=1)
        gap = np.maximum(fid[1:] - fid[:-1], 1)
        jump_score = d_trans / gap + cfg.jump_pose_scale * (d_pose / gap)

        med = float(np.median(jump_score))
        mad = float(np.median(np.abs(jump_score - med)) + 1e-6)
        thr = max(med + cfg.jump_mad_k * mad, cfg.jump_min_score)
        bad_edges = np.where(jump_score > thr)[0]
        if bad_edges.size > 0:
            bad_local = bad_edges + 1
            bad_global = idxs[bad_local]
            keep[bad_global] = False
            dropped_jump = int(len(bad_global))

    # Two-sided isolated spike filtering:
    # Detect frame i where both i-1->i and i->i+1 changes are huge,
    # but i-1->i+1 remains small (single-frame failure / flip).
    idxs = np.where(keep)[0]
    if len(idxs) >= 5:
        feat_pose = np.stack(
            [
                np.concatenate(
                    [
                        rows[i]["smplx_root_pose"],
                        rows[i]["smplx_body_pose"],
                        rows[i]["smplx_lhand_pose"],
                        rows[i]["smplx_rhand_pose"],
                        rows[i]["smplx_jaw_pose"],
                    ],
                    axis=0,
                )
                for i in idxs
            ],
            axis=0,
        )
        d_step = np.linalg.norm(feat_pose[1:] - feat_pose[:-1], axis=1)
        d_med = float(np.median(d_step))
        d_mad = float(np.median(np.abs(d_step - d_med)) + 1e-6)
        d_thr = max(d_med + cfg.spike_mad_k * d_mad, d_med * 2.0)

        bad_local = []
        for i in range(1, len(feat_pose) - 1):
            d_prev = float(np.linalg.norm(feat_pose[i] - feat_pose[i - 1]))
            d_next = float(np.linalg.norm(feat_pose[i + 1] - feat_pose[i]))
            d_bridge = float(np.linalg.norm(feat_pose[i + 1] - feat_pose[i - 1]))
            if d_prev > d_thr and d_next > d_thr:
                if d_bridge < cfg.spike_bridge_ratio * min(d_prev, d_next):
                    bad_local.append(i)

        if len(bad_local) > 0:
            bad_global = idxs[np.asarray(bad_local, dtype=np.int32)]
            keep[bad_global] = False
            dropped_spike = int(len(bad_global))

    kept_idx = np.where(keep)[0]
    if len(kept_idx) < cfg.min_frames:
        return {"ok": False, "name": name, "reason": f"too_few_after_filter:{len(kept_idx)}"}

    kept_rows = [rows[i] for i in kept_idx]
    kept_fids = frame_ids[kept_idx].astype(np.float32)
    t = kept_fids / max(cfg.fps, 1e-6)

    root = np.stack([x["smplx_root_pose"] for x in kept_rows], axis=0)
    body = np.stack([x["smplx_body_pose"] for x in kept_rows], axis=0)
    lhand = np.stack([x["smplx_lhand_pose"] for x in kept_rows], axis=0)
    rhand = np.stack([x["smplx_rhand_pose"] for x in kept_rows], axis=0)
    jaw = np.stack([x["smplx_jaw_pose"] for x in kept_rows], axis=0)
    shape = np.stack([x["smplx_shape"] for x in kept_rows], axis=0)
    expr = np.stack([x["smplx_expr"] for x in kept_rows], axis=0)

    if cfg.use_one_euro:
        root = apply_one_euro(root, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        body = apply_one_euro(body, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        lhand = apply_one_euro(lhand, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        rhand = apply_one_euro(rhand, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        jaw = apply_one_euro(jaw, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        expr = apply_one_euro(expr, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)

    # Shape is identity-like and usually static; keep median to suppress frame-wise noise.
    shape = np.median(shape, axis=0, keepdims=True).repeat(len(kept_rows), axis=0).astype(np.float32, copy=False)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_vec = np.zeros((179,), dtype=np.float64)
    sq_sum_vec = np.zeros((179,), dtype=np.float64)
    count = 0

    for i in range(len(kept_rows)):
        d_out = {
            "smplx_root_pose": root[i].astype(np.float32, copy=False),
            "smplx_body_pose": body[i].astype(np.float32, copy=False),
            "smplx_lhand_pose": lhand[i].astype(np.float32, copy=False),
            "smplx_rhand_pose": rhand[i].astype(np.float32, copy=False),
            "smplx_jaw_pose": jaw[i].astype(np.float32, copy=False),
            "smplx_shape": shape[i].astype(np.float32, copy=False),
            "smplx_expr": expr[i].astype(np.float32, copy=False),
        }
        with open(out_dir / f"{i:06d}.pkl", "wb") as f:
            pickle.dump(d_out, f, protocol=pickle.HIGHEST_PROTOCOL)

        if split == "train":
            feat179 = np.concatenate([d_out[k] for k in OUT_KEYS], axis=0).astype(np.float64, copy=False)
            sum_vec += feat179
            sq_sum_vec += feat179 * feat179
            count += 1

    return {
        "ok": True,
        "name": name,
        "split": split,
        "num_frames": len(kept_rows),
        "dropped_invalid": int(dropped_invalid),
        "dropped_low_conf": int(dropped_low_conf),
        "dropped_jump": int(dropped_jump),
        "dropped_spike": int(dropped_spike),
        "sum_vec": sum_vec if split == "train" else None,
        "sq_sum_vec": sq_sum_vec if split == "train" else None,
        "count": int(count),
        "from_existing": False,
    }


def scan_clip_dirs(input_root: Path):
    clips: List[Tuple[str, str]] = []
    # Fast path for hash buckets: root/xx/yy/<clip_id>/smplx
    for lv1 in input_root.iterdir():
        if not lv1.is_dir():
            continue
        for lv2 in lv1.iterdir():
            if not lv2.is_dir():
                continue
            for lv3 in lv2.iterdir():
                if not lv3.is_dir():
                    continue
                smplx_dir = lv3 / "smplx"
                if smplx_dir.is_dir():
                    clips.append((lv3.name, str(smplx_dir)))

    if len(clips) > 0:
        return clips

    # Fallback for arbitrary folder structures.
    for p in input_root.rglob("smplx"):
        if p.is_dir():
            clips.append((p.parent.name, str(p)))
    return clips


def save_pickle_gz(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CSL-Dental SMPLerX outputs into SOKE CSL-style dataset with filtering + OneEuro smoothing."
    )
    parser.add_argument("--input-root", type=str, default="/nas/DDDataLang/raw_data/csl_dental/smplx")
    parser.add_argument("--output-root", type=str, default="/nas/DDDataLang/raw_data/csl_dental/soke")
    parser.add_argument("--fps", type=float, default=25.0, help="Assumed source fps for OneEuro timestamps.")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() // 2, 1))
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=0, help="0 means all clips.")

    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--default-text", type=str, default="")

    parser.add_argument("--disable-one-euro", action="store_true")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.0)
    parser.add_argument("--one-euro-beta", type=float, default=0.005)
    parser.add_argument("--one-euro-d-cutoff", type=float, default=1.0)

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=-1.0,
        help=">=0 to drop frames with explicit confidence below threshold; ignored if confidence not present.",
    )
    parser.add_argument("--transl-abs-max", type=float, default=50.0)
    parser.add_argument("--jump-pose-scale", type=float, default=0.02)
    parser.add_argument("--jump-mad-k", type=float, default=10.0)
    parser.add_argument("--jump-min-score", type=float, default=0.8)
    parser.add_argument(
        "--spike-mad-k",
        type=float,
        default=6.0,
        help="MAD multiplier for isolated-spike detection on pose-step magnitude.",
    )
    parser.add_argument(
        "--spike-bridge-ratio",
        type=float,
        default=0.35,
        help="Require ||i-1 - i+1|| < ratio * min(||i-i-1||, ||i+1-i||) to mark frame i as isolated spike.",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing output poses/ann/mean/std before processing.",
    )
    parser.add_argument("--stats-from-existing", dest="stats_from_existing", action="store_true")
    parser.add_argument("--no-stats-from-existing", dest="stats_from_existing", action="store_false")
    parser.set_defaults(stats_from_existing=True)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1.0:
        raise ValueError("Require train_ratio>0, val_ratio>=0 and train_ratio+val_ratio<1.")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    poses_root = output_root / "poses"
    if args.clean_output:
        if poses_root.exists():
            shutil.rmtree(poses_root)
        for p in [
            output_root / "csl_clean.train",
            output_root / "csl_clean.val",
            output_root / "csl_clean.test",
            output_root / "mean.pt",
            output_root / "std.pt",
            output_root / "preprocess_report.json",
        ]:
            if p.exists():
                p.unlink()
    poses_root.mkdir(parents=True, exist_ok=True)

    print(f"[scan] input_root={input_root}")
    clips = scan_clip_dirs(input_root)
    print(f"[scan] discovered clips: {len(clips)}")
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
        print(f"[scan] max_clips enabled: using first {len(clips)} clips")
    if len(clips) == 0:
        raise RuntimeError("No valid clip/smplx directories found.")

    wcfg = WorkerCfg(
        output_root=str(output_root),
        default_text=args.default_text,
        fps=args.fps,
        min_frames=args.min_frames,
        use_one_euro=not args.disable_one_euro,
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_d_cutoff=args.one_euro_d_cutoff,
        confidence_threshold=args.confidence_threshold,
        transl_abs_max=args.transl_abs_max,
        jump_pose_scale=args.jump_pose_scale,
        jump_mad_k=args.jump_mad_k,
        jump_min_score=args.jump_min_score,
        spike_mad_k=args.spike_mad_k,
        spike_bridge_ratio=args.spike_bridge_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        overwrite=args.overwrite,
        stats_from_existing=args.stats_from_existing,
    )

    records = {"train": [], "val": [], "test": []}
    failures = []
    total_drop_invalid = 0
    total_drop_low_conf = 0
    total_drop_jump = 0
    total_drop_spike = 0
    total_frames_kept = 0

    sum_vec = np.zeros((179,), dtype=np.float64)
    sq_sum_vec = np.zeros((179,), dtype=np.float64)
    count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_clip, name, smplx_dir, wcfg) for name, smplx_dir in clips]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="process_clips"):
            r = fut.result()
            if not r.get("ok", False):
                failures.append({"name": r.get("name", ""), "reason": r.get("reason", "unknown")})
                continue

            split = r["split"]
            name = r["name"]
            num_frames = int(r["num_frames"])
            records[split].append(
                {
                    "name": name,
                    "text": args.default_text,
                    "gloss": "",
                    "sign": "",
                    "signer": "unknown",
                    "num_frames": num_frames,
                    "src": "csl",
                }
            )
            total_frames_kept += num_frames
            total_drop_invalid += int(r["dropped_invalid"])
            total_drop_low_conf += int(r["dropped_low_conf"])
            total_drop_jump += int(r["dropped_jump"])
            total_drop_spike += int(r.get("dropped_spike", 0))

            if split == "train" and r["sum_vec"] is not None and r["count"] > 0:
                sum_vec += r["sum_vec"]
                sq_sum_vec += r["sq_sum_vec"]
                count += int(r["count"])

    for s in ["train", "val", "test"]:
        records[s].sort(key=lambda x: x["name"])
        save_pickle_gz(output_root / f"csl_clean.{s}", records[s])

    if count <= 0:
        raise RuntimeError("No train frames available for mean/std computation.")
    mean = sum_vec / float(count)
    var = np.maximum(sq_sum_vec / float(count) - mean * mean, 1e-8)
    std = np.sqrt(var)
    torch.save(torch.from_numpy(mean.astype(np.float32)), output_root / "mean.pt")
    torch.save(torch.from_numpy(std.astype(np.float32)), output_root / "std.pt")

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_clips_discovered": len(clips),
        "num_clips_success": sum(len(records[s]) for s in ["train", "val", "test"]),
        "num_clips_failed": len(failures),
        "num_split": {s: len(records[s]) for s in ["train", "val", "test"]},
        "total_frames_kept": int(total_frames_kept),
        "dropped_invalid": int(total_drop_invalid),
        "dropped_low_conf": int(total_drop_low_conf),
        "dropped_jump": int(total_drop_jump),
        "dropped_spike": int(total_drop_spike),
        "mean_std_count_frames_train": int(count),
        "one_euro": {
            "enabled": bool(not args.disable_one_euro),
            "min_cutoff": float(args.one_euro_min_cutoff),
            "beta": float(args.one_euro_beta),
            "d_cutoff": float(args.one_euro_d_cutoff),
        },
        "filters": {
            "confidence_threshold": float(args.confidence_threshold),
            "transl_abs_max": float(args.transl_abs_max),
            "jump_pose_scale": float(args.jump_pose_scale),
            "jump_mad_k": float(args.jump_mad_k),
            "jump_min_score": float(args.jump_min_score),
            "spike_mad_k": float(args.spike_mad_k),
            "spike_bridge_ratio": float(args.spike_bridge_ratio),
        },
        "failed_examples_head": failures[:100],
    }
    with open(output_root / "preprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[done] Output generated:")
    print(f"  poses root: {output_root / 'poses'}")
    print(f"  ann: {output_root / 'csl_clean.train'}")
    print(f"  ann: {output_root / 'csl_clean.val'}")
    print(f"  ann: {output_root / 'csl_clean.test'}")
    print(f"  mean/std: {output_root / 'mean.pt'}, {output_root / 'std.pt'}")
    print(f"  report: {output_root / 'preprocess_report.json'}")
    print(f"  split stats: train={len(records['train'])}, val={len(records['val'])}, test={len(records['test'])}")
    print(f"  failed clips: {len(failures)}")


if __name__ == "__main__":
    main()
