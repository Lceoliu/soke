#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.utils.smplx_contact import detect_contacts_from_smplx, load_pose179_from_smplx_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-frame SMPL-X contact distances and overlay on raw mesh video."
    )
    parser.add_argument("--pose_dir", type=str, required=True, help="CSL-Daily pose dir with frame pkl/pt.")
    parser.add_argument("--video_in", type=str, required=True, help="Input raw mesh video.")
    parser.add_argument("--video_out", type=str, required=True, help="Output annotated video.")
    parser.add_argument("--csv_out", type=str, default="", help="Optional CSV output for per-pose-frame distances.")
    parser.add_argument("--device", type=str, default="", help="cuda/cpu, default: auto.")
    parser.add_argument("--chunk_a", type=int, default=256)
    parser.add_argument("--chunk_b", type=int, default=1024)
    return parser.parse_args()


def _draw_text_block(frame, lines, x=12, y=24, line_h=24):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.62
    th = 2

    max_w = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, fs, th)
        max_w = max(max_w, w)
    box_h = line_h * len(lines) + 10
    box_w = max_w + 16
    cv2.rectangle(frame, (x - 8, y - 20), (x - 8 + box_w, y - 20 + box_h), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (x - 8, y - 20), (x - 8 + box_w, y - 20 + box_h), (255, 255, 255), thickness=1)

    yy = y
    for line in lines:
        cv2.putText(frame, line, (x, yy), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        yy += line_h


def main():
    args = parse_args()
    pose_dir = Path(args.pose_dir)
    video_in = Path(args.video_in)
    video_out = Path(args.video_out)
    csv_out = Path(args.csv_out) if args.csv_out else video_out.with_suffix("").with_name(video_out.stem + "_distances.csv")

    if not pose_dir.exists():
        raise FileNotFoundError(f"pose_dir not found: {pose_dir}")
    if not video_in.exists():
        raise FileNotFoundError(f"video_in not found: {video_in}")

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    pose179 = load_pose179_from_smplx_dir(pose_dir)
    pairs = (("lhand", "rhand"), ("lhand", "face"), ("rhand", "face"))
    results = detect_contacts_from_smplx(
        data=pose179,
        data_type="pose179",
        pairs=pairs,
        threshold=0.02,  # threshold is irrelevant for min_dist values; kept for API consistency
        device=device,
        chunk_a=int(args.chunk_a),
        chunk_b=int(args.chunk_b),
        bool_only=False,
    )

    d_lh_rh = results["lhand-rhand"]["min_dist"].detach().cpu().numpy().astype(np.float32)
    d_lh_face = results["lhand-face"]["min_dist"].detach().cpu().numpy().astype(np.float32)
    d_rh_face = results["rhand-face"]["min_dist"].detach().cpu().numpy().astype(np.float32)
    n_pose = pose179.shape[0]

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pose_frame_idx", "lhand_rhand_m", "lhand_face_m", "rhand_face_m"])
        for i in range(n_pose):
            writer.writerow([i, float(d_lh_rh[i]), float(d_lh_face[i]), float(d_rh_face[i])])

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video <= 0:
        # fallback for some codecs
        n_video = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n_video += 1
        cap.release()
        cap = cv2.VideoCapture(str(video_in))

    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 18.0,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {video_out}")

    if n_video <= 1:
        map_idx = np.zeros((max(n_video, 1),), dtype=np.int64)
    else:
        map_idx = np.rint(np.linspace(0, n_pose - 1, num=n_video)).astype(np.int64)
        map_idx = np.clip(map_idx, 0, n_pose - 1)

    vi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pi = int(map_idx[min(vi, len(map_idx) - 1)])
        lines = [
            f"pose_frame: {pi+1}/{n_pose}  video_frame: {vi+1}/{max(n_video, 1)}",
            f"lhand-rhand: {d_lh_rh[pi]:.4f} m",
            f"lhand-face : {d_lh_face[pi]:.4f} m",
            f"rhand-face : {d_rh_face[pi]:.4f} m",
        ]
        _draw_text_block(frame, lines, x=14, y=28, line_h=24)
        writer.write(frame)
        vi += 1

    cap.release()
    writer.release()
    print(f"[DONE] pose_frames={n_pose}, video_frames={n_video}, device={device}")
    print(f"[DONE] video_out={video_out}")
    print(f"[DONE] csv_out={csv_out}")


if __name__ == "__main__":
    main()
