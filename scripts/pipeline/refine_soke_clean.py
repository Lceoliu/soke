#!/usr/bin/env python3
import argparse
import gzip
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


POSE_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]


def _axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = r / theta
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _estimate_tilt_degrees(root_pose: np.ndarray):
    if root_pose.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), -1

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_axes = np.eye(3, dtype=np.float64)
    Rs = [_axis_angle_to_matrix(root_pose[i]) for i in range(root_pose.shape[0])]

    best_axis = 1
    best_med = 1e9
    best_tilt = None
    for a in range(3):
        axis = local_axes[a]
        dots = []
        for R in Rs:
            v = R @ axis
            dots.append(abs(float(np.dot(v, world_up))))
        dots = np.clip(np.asarray(dots, dtype=np.float64), 0.0, 1.0)
        tilt = np.degrees(np.arccos(dots))
        med = float(np.median(tilt))
        if med < best_med:
            best_med = med
            best_axis = a
            best_tilt = tilt
    return best_tilt.astype(np.float32, copy=False), best_axis


def _find_true_segments(mask: np.ndarray):
    segs = []
    i = 0
    n = int(mask.shape[0])
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        s = i
        while i + 1 < n and bool(mask[i + 1]):
            i += 1
        e = i
        segs.append((s, e))
        i += 1
    return segs


def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros((3,), dtype=np.float32)

    if np.pi - theta < 1e-5:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / n
        return (axis * theta).astype(np.float32, copy=False)

    v = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    axis = v / (2.0 * np.sin(theta))
    return (axis * theta).astype(np.float32, copy=False)


def _rotation_geodesic_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    d = R1.T @ R2
    c = np.clip((np.trace(d) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(c))


def _single_tilt_deg(R: np.ndarray) -> float:
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up = R @ local_up
    c = np.clip(abs(float(np.dot(up, world_up))), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _axis_angle_to_quat(rotvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = r / theta
    h = 0.5 * theta
    s = np.sin(h)
    return np.array([np.cos(h), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.zeros((3,), dtype=np.float32)
    q = q / n
    w = np.clip(float(q[0]), -1.0, 1.0)
    v = q[1:]
    sin_h = float(np.linalg.norm(v))
    if sin_h < 1e-8:
        return np.zeros((3,), dtype=np.float32)
    h = np.arctan2(sin_h, w)
    theta = 2.0 * h
    axis = v / sin_h
    return (axis * theta).astype(np.float32, copy=False)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return _quat_normalize(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return _quat_normalize(s0 * q0 + s1 * q1)


def _interp_root_rotations(root_pose: np.ndarray, bad_mask: np.ndarray):
    out = root_pose.astype(np.float32, copy=True)
    n = int(out.shape[0])
    if n == 0:
        return out
    bad = np.asarray(bad_mask, dtype=bool)
    if not np.any(bad):
        return out

    good_idx = np.where(~bad)[0]
    if good_idx.size == 0:
        med = np.median(out, axis=0, keepdims=True)
        return med.repeat(n, axis=0).astype(np.float32, copy=False)

    quats = np.stack([_axis_angle_to_quat(out[i]) for i in range(n)], axis=0)
    for i in range(1, n):
        if np.dot(quats[i - 1], quats[i]) < 0.0:
            quats[i] = -quats[i]

    for i in range(n):
        if not bad[i]:
            continue
        l = i - 1
        while l >= 0 and bad[l]:
            l -= 1
        r = i + 1
        while r < n and bad[r]:
            r += 1
        if l >= 0 and r < n:
            t = float(i - l) / float(r - l)
            q = _quat_slerp(quats[l], quats[r], t)
            out[i] = _quat_to_axis_angle(q)
        elif l >= 0:
            out[i] = out[l]
        elif r < n:
            out[i] = out[r]
    return out


def _interp_bad_rows(arr: np.ndarray, bad_mask: np.ndarray):
    out = arr.astype(np.float32, copy=True)
    if out.shape[0] == 0:
        return out
    bad = np.asarray(bad_mask, dtype=bool)
    if not np.any(bad):
        return out
    good = ~bad
    if int(np.sum(good)) == 0:
        med = np.median(out, axis=0, keepdims=True)
        return med.repeat(out.shape[0], axis=0).astype(np.float32, copy=False)

    x = np.arange(out.shape[0], dtype=np.float32)
    xg = x[good]
    for d in range(out.shape[1]):
        y = out[:, d]
        yg = y[good]
        y[bad] = np.interp(x[bad], xg, yg).astype(np.float32, copy=False)
        out[:, d] = y
    return out


class OneEuroFilter:
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


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _sorted_pkl_files(clip_dir: Path) -> List[Path]:
    files = [x for x in clip_dir.iterdir() if x.is_file() and x.suffix.lower() == ".pkl"]
    files.sort(key=lambda p: p.name)
    return files


def _apply_one_euro(arr: np.ndarray, t: np.ndarray, min_cutoff: float, beta: float, d_cutoff: float):
    if arr.shape[0] <= 1:
        return arr
    filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = filt(float(t[i]), arr[i].astype(np.float32, copy=False))
    return out


@dataclass
class RefineCfg:
    fps: float
    min_frames: int
    jump_pose_scale: float
    jump_mad_k: float
    jump_min_score: float
    spike_mad_k: float
    spike_bridge_ratio: float
    tilt_mad_k: float
    tilt_min_deg: float
    tilt_abs_deg: float
    root_vel_mad_k: float
    root_vel_min_deg: float
    max_bad_ratio: float
    apply_one_euro: bool
    one_euro_min_cutoff: float
    one_euro_beta: float
    one_euro_d_cutoff: float
    dry_run: bool


def process_clip(clip_dir: str, cfg: RefineCfg):
    cdir = Path(clip_dir)
    files = _sorted_pkl_files(cdir)
    if len(files) < cfg.min_frames:
        return {"ok": False, "clip": cdir.name, "reason": f"too_short:{len(files)}"}

    frames = []
    for p in files:
        try:
            d = _load_pickle(p)
        except Exception:
            continue
        if any(k not in d for k in POSE_KEYS):
            continue
        if not all(np.all(np.isfinite(np.asarray(d[k]))) for k in POSE_KEYS):
            continue
        frames.append(d)

    if len(frames) < cfg.min_frames:
        return {"ok": False, "clip": cdir.name, "reason": f"too_few_valid:{len(frames)}"}

    root = np.stack([np.asarray(x["smplx_root_pose"], dtype=np.float32).reshape(-1)[:3] for x in frames], axis=0)
    body = np.stack([np.asarray(x["smplx_body_pose"], dtype=np.float32).reshape(-1)[:63] for x in frames], axis=0)
    lhand = np.stack([np.asarray(x["smplx_lhand_pose"], dtype=np.float32).reshape(-1)[:45] for x in frames], axis=0)
    rhand = np.stack([np.asarray(x["smplx_rhand_pose"], dtype=np.float32).reshape(-1)[:45] for x in frames], axis=0)
    jaw = np.stack([np.asarray(x["smplx_jaw_pose"], dtype=np.float32).reshape(-1)[:3] for x in frames], axis=0)
    shape = np.stack([np.asarray(x["smplx_shape"], dtype=np.float32).reshape(-1)[:10] for x in frames], axis=0)
    expr = np.stack([np.asarray(x["smplx_expr"], dtype=np.float32).reshape(-1)[:10] for x in frames], axis=0)

    pose = np.concatenate([root, body, lhand, rhand, jaw], axis=1)
    bad = np.zeros((pose.shape[0],), dtype=bool)
    jump_detected = 0
    spike_detected = 0

    # Step 1: detect high global pose jumps.
    if pose.shape[0] >= 3:
        d_pose = np.linalg.norm(pose[1:] - pose[:-1], axis=1)
        jump_score = cfg.jump_pose_scale * d_pose
        med = float(np.median(jump_score))
        mad = float(np.median(np.abs(jump_score - med)) + 1e-6)
        thr = max(med + cfg.jump_mad_k * mad, cfg.jump_min_score)
        bad_edges = np.where(jump_score > thr)[0]
        if bad_edges.size > 0:
            bad[bad_edges + 1] = True
            jump_detected = int(bad_edges.size)

    # Step 2: detect isolated spikes (two-sided large jump but bridge is short).
    if pose.shape[0] >= 5:
        d_step = np.linalg.norm(pose[1:] - pose[:-1], axis=1)
        d_med = float(np.median(d_step))
        d_mad = float(np.median(np.abs(d_step - d_med)) + 1e-6)
        d_thr = max(d_med + cfg.spike_mad_k * d_mad, d_med * 2.0)
        for i in range(1, pose.shape[0] - 1):
            d_prev = float(np.linalg.norm(pose[i] - pose[i - 1]))
            d_next = float(np.linalg.norm(pose[i + 1] - pose[i]))
            d_bridge = float(np.linalg.norm(pose[i + 1] - pose[i - 1]))
            if d_prev > d_thr and d_next > d_thr and d_bridge < cfg.spike_bridge_ratio * min(d_prev, d_next):
                bad[i] = True
                spike_detected += 1

    # Step 3: detect lying-like tilt anomalies and root angular-velocity anomalies.
    tilt_axis = -1
    tilt_deg_median = 0.0
    tilt_thr = 0.0
    dropped_tilt = 0
    if root.shape[0] >= 3:
        tilt_deg, tilt_axis = _estimate_tilt_degrees(root)
        tilt_deg_median = float(np.median(tilt_deg))
        tilt_mad = float(np.median(np.abs(tilt_deg - tilt_deg_median)) + 1e-6)
        tilt_thr = max(cfg.tilt_min_deg, tilt_deg_median + cfg.tilt_mad_k * tilt_mad)
        tilt_bad = (tilt_deg > tilt_thr) | (tilt_deg > cfg.tilt_abs_deg)
        bad = bad | tilt_bad
        dropped_tilt = int(np.sum(tilt_bad))

        R_seq = [_axis_angle_to_matrix(root[i]) for i in range(root.shape[0])]
        vel = np.array([_rotation_geodesic_angle(R_seq[i - 1], R_seq[i]) for i in range(1, len(R_seq))], dtype=np.float32)
        if vel.shape[0] > 0:
            vel_med = float(np.median(vel))
            vel_mad = float(np.median(np.abs(vel - vel_med)) + 1e-6)
            vel_thr = max(np.radians(cfg.root_vel_min_deg), vel_med + cfg.root_vel_mad_k * vel_mad)
            bad_vel = np.where(vel > vel_thr)[0] + 1
            if bad_vel.size > 0:
                bad[bad_vel] = True

    bad_ratio = float(np.mean(bad.astype(np.float32)))
    if bad_ratio > cfg.max_bad_ratio:
        return {"ok": False, "clip": cdir.name, "reason": f"too_many_bad:{bad_ratio:.3f}"}

    # Step 4: repair in-place by interpolation (no frame dropping, avoids new temporal jumps).
    repaired_cnt = int(np.sum(bad))
    root = _interp_root_rotations(root, bad)
    body = _interp_bad_rows(body, bad)
    lhand = _interp_bad_rows(lhand, bad)
    rhand = _interp_bad_rows(rhand, bad)
    jaw = _interp_bad_rows(jaw, bad)
    shape = _interp_bad_rows(shape, bad)
    expr = _interp_bad_rows(expr, bad)

    # USER DIRECTIVE: fix root rotation to prevent large rotations.
    # We lock it to the first frame's orientation.
    if root.shape[0] > 0:
        root = root[0:1, :].repeat(root.shape[0], axis=0)

    if cfg.apply_one_euro:
        t = np.arange(root.shape[0], dtype=np.float32) / max(cfg.fps, 1e-6)
        root = _apply_one_euro(root, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        body = _apply_one_euro(body, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        lhand = _apply_one_euro(lhand, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        rhand = _apply_one_euro(rhand, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        jaw = _apply_one_euro(jaw, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        expr = _apply_one_euro(expr, t, cfg.one_euro_min_cutoff, cfg.one_euro_beta, cfg.one_euro_d_cutoff)
        shape = np.median(shape, axis=0, keepdims=True).repeat(root.shape[0], axis=0).astype(np.float32, copy=False)

    if not cfg.dry_run:
        tmp = cdir.parent / (cdir.name + ".tmp_refine")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        for i in range(root.shape[0]):
            d = {
                "smplx_root_pose": root[i].astype(np.float32),
                "smplx_body_pose": body[i].astype(np.float32),
                "smplx_lhand_pose": lhand[i].astype(np.float32),
                "smplx_rhand_pose": rhand[i].astype(np.float32),
                "smplx_jaw_pose": jaw[i].astype(np.float32),
                "smplx_shape": shape[i].astype(np.float32),
                "smplx_expr": expr[i].astype(np.float32),
            }
            _save_pickle(tmp / f"{i:06d}.pkl", d)
        backup = cdir.parent / (cdir.name + ".bak_before_refine")
        if backup.exists():
            shutil.rmtree(backup)
        cdir.rename(backup)
        tmp.rename(cdir)
        shutil.rmtree(backup)

    return {
        "ok": True,
        "clip": cdir.name,
        "raw_frames": len(frames),
        "new_frames": int(root.shape[0]),
        "dropped_jump": int(jump_detected),
        "dropped_spike": int(spike_detected),
        "dropped_tilt": dropped_tilt,
        "repaired_tilt": int(repaired_cnt),
        "tilt_axis": int(tilt_axis),
        "tilt_deg_median": float(tilt_deg_median),
        "tilt_thr": float(tilt_thr),
        "bad_ratio": bad_ratio,
    }


def load_ann(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_ann(path: Path, ann):
    with gzip.open(path, "wb") as f:
        pickle.dump(ann, f, protocol=pickle.HIGHEST_PROTOCOL)


def recompute_train_mean_std(dataset_root: Path):
    ann_path = dataset_root / "csl_clean.train"
    poses_root = dataset_root / "poses"
    ann = load_ann(ann_path)

    sum_vec = np.zeros((179,), dtype=np.float64)
    sq_sum_vec = np.zeros((179,), dtype=np.float64)
    count = 0

    for a in ann:
        cdir = poses_root / a["name"]
        if not cdir.is_dir():
            continue
        for p in _sorted_pkl_files(cdir):
            try:
                d = _load_pickle(p)
            except Exception:
                continue
            try:
                feat = np.concatenate([np.asarray(d[k], dtype=np.float32).reshape(-1) for k in POSE_KEYS], axis=0)
            except Exception:
                continue
            if feat.shape[0] != 179 or not np.all(np.isfinite(feat)):
                continue
            x = feat.astype(np.float64, copy=False)
            sum_vec += x
            sq_sum_vec += x * x
            count += 1

    if count <= 0:
        raise RuntimeError("No valid train frames for mean/std recomputation.")

    mean = sum_vec / float(count)
    var = np.maximum(sq_sum_vec / float(count) - mean * mean, 1e-8)
    std = np.sqrt(var)

    import torch

    torch.save(torch.from_numpy(mean.astype(np.float32)), dataset_root / "mean.pt")
    torch.save(torch.from_numpy(std.astype(np.float32)), dataset_root / "std.pt")
    return count


def main():
    ap = argparse.ArgumentParser(description="Fast in-place refinement for occasional bad SMPL-X frames on existing csl poses.")
    ap.add_argument("--dataset-root", default="data/csl_dental")
    ap.add_argument("--splits", default="train,val,test", help="comma separated")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() // 2, 1))
    ap.add_argument("--max-clips", type=int, default=0, help="0 means all selected clips")
    ap.add_argument("--names-file", default="", help="optional text file with clip names (one per line)")
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--jump-pose-scale", type=float, default=0.02)
    ap.add_argument("--jump-mad-k", type=float, default=8.0)
    ap.add_argument("--jump-min-score", type=float, default=0.6)
    ap.add_argument("--spike-mad-k", type=float, default=5.0)
    ap.add_argument("--spike-bridge-ratio", type=float, default=0.45)
    ap.add_argument("--tilt-mad-k", type=float, default=6.0)
    ap.add_argument("--tilt-min-deg", type=float, default=45.0)
    ap.add_argument("--tilt-abs-deg", type=float, default=70.0)
    ap.add_argument("--root-vel-mad-k", type=float, default=7.0)
    ap.add_argument("--root-vel-min-deg", type=float, default=45.0)
    ap.add_argument("--max-bad-ratio", type=float, default=0.35)
    ap.add_argument("--apply-one-euro", action="store_true")
    ap.add_argument("--one-euro-min-cutoff", type=float, default=1.0)
    ap.add_argument("--one-euro-beta", type=float, default=0.005)
    ap.add_argument("--one-euro-d-cutoff", type=float, default=1.0)
    ap.add_argument("--recompute-mean-std", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    poses_root = root / "poses"
    if not poses_root.is_dir():
        raise FileNotFoundError(f"poses dir not found: {poses_root}")

    selected_names = None
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as f:
            selected_names = {x.strip() for x in f if x.strip()}

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    cfg = RefineCfg(
        fps=args.fps,
        min_frames=args.min_frames,
        jump_pose_scale=args.jump_pose_scale,
        jump_mad_k=args.jump_mad_k,
        jump_min_score=args.jump_min_score,
        spike_mad_k=args.spike_mad_k,
        spike_bridge_ratio=args.spike_bridge_ratio,
        tilt_mad_k=args.tilt_mad_k,
        tilt_min_deg=args.tilt_min_deg,
        tilt_abs_deg=args.tilt_abs_deg,
        root_vel_mad_k=args.root_vel_mad_k,
        root_vel_min_deg=args.root_vel_min_deg,
        max_bad_ratio=args.max_bad_ratio,
        apply_one_euro=bool(args.apply_one_euro),
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_d_cutoff=args.one_euro_d_cutoff,
        dry_run=bool(args.dry_run),
    )

    total_ok = 0
    total_fail = 0
    total_drop_jump = 0
    total_drop_spike = 0
    total_drop_tilt = 0
    total_repair_tilt = 0
    failed = []
    split_changed: Dict[str, int] = {}

    for sp in splits:
        ann_path = root / f"csl_clean.{sp}"
        if not ann_path.exists():
            print(f"[WARN] missing ann for split={sp}: {ann_path}")
            continue

        ann = load_ann(ann_path)
        tasks = []
        for a in ann:
            name = a["name"]
            if selected_names is not None and name not in selected_names:
                continue
            cdir = poses_root / name
            if not cdir.is_dir():
                failed.append((name, "missing_pose_dir"))
                continue
            tasks.append((name, str(cdir)))
        if args.max_clips > 0:
            tasks = tasks[: args.max_clips]

        print(f"[{sp}] selected clips: {len(tasks)}")
        name2new_frames: Dict[str, int] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_clip, cdir, cfg) for _, cdir in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"refine_{sp}"):
                r = fut.result()
                if not r.get("ok", False):
                    total_fail += 1
                    failed.append((r.get("clip", ""), r.get("reason", "unknown")))
                    continue
                total_ok += 1
                total_drop_jump += int(r.get("dropped_jump", 0))
                total_drop_spike += int(r.get("dropped_spike", 0))
                total_drop_tilt += int(r.get("dropped_tilt", 0))
                total_repair_tilt += int(r.get("repaired_tilt", 0))
                name2new_frames[r["clip"]] = int(r["new_frames"])

        # update ann num_frames
        changed = 0
        for a in ann:
            n = a["name"]
            if n in name2new_frames:
                old = int(a.get("num_frames", -1))
                new = int(name2new_frames[n])
                if old != new:
                    a["num_frames"] = new
                    changed += 1
        if not args.dry_run:
            bak = ann_path.with_suffix(ann_path.suffix + ".bak_refine")
            if not bak.exists():
                shutil.copy2(ann_path, bak)
            save_ann(ann_path, ann)
        split_changed[sp] = changed
        print(f"[{sp}] updated ann num_frames: {changed}")

    if args.recompute_mean_std and not args.dry_run:
        c = recompute_train_mean_std(root)
        print(f"[stats] mean/std recomputed from train frames: {c}")
    elif (not args.dry_run) and split_changed.get("train", 0) > 0:
        print("[stats] train num_frames changed; recommend adding --recompute-mean-std.")

    print("[done]")
    print(
        f"ok={total_ok}, fail={total_fail}, dropped_jump={total_drop_jump}, "
        f"dropped_spike={total_drop_spike}, dropped_tilt={total_drop_tilt}, repaired_tilt={total_repair_tilt}"
    )
    if failed:
        print("failed examples (head):", failed[:20])


if __name__ == "__main__":
    main()
