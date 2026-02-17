#!/usr/bin/env python3
import argparse
import os
import pickle
import re
import sys
from pathlib import Path

# Headless servers in this project are usually more stable with OSMesa.
# You can override before running: `PYOPENGL_PLATFORM=egl ...`.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
import pyrender
from PIL import Image
from tqdm import tqdm

from mGPT.utils.human_models import get_coord, smpl_x


POSE_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]

SHAPE_DEFAULT = np.array(
    [-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
     0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize raw SMPL-X mesh from extracted data (without VQ/VAE)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pose_dir",
        type=str,
        help="Directory with per-frame .pkl/.pt containing SMPL-X keys.",
    )
    src.add_argument(
        "--pose_npy",
        type=str,
        help="Path to .npy clip. Use with --input_type {pose179,feat133_raw,feat133_norm}.",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="pose179",
        choices=["pose179", "feat133_raw", "feat133_norm"],
        help="Interpretation of --pose_npy data. Ignored when --pose_dir is used.",
    )
    parser.add_argument(
        "--mean_path",
        type=str,
        default=None,
        help="Required when input_type=feat133_norm. Mean file (.pt/.npy), original 179-dim stats.",
    )
    parser.add_argument(
        "--std_path",
        type=str,
        default=None,
        help="Required when input_type=feat133_norm. Std file (.pt/.npy), original 179-dim stats.",
    )
    parser.add_argument(
        "--input_fps",
        type=float,
        default=None,
        help="If provided and >24, clip is uniformly resampled to 24fps.",
    )
    parser.add_argument("--max_frames", type=int, default=0, help="Uniformly keep at most this many frames. 0 = keep all.")
    parser.add_argument("--fps", type=int, default=18, help="Output video FPS.")
    parser.add_argument("--width", type=int, default=512, help="Output width.")
    parser.add_argument("--height", type=int, default=512, help="Output height.")
    parser.add_argument("--focal", type=float, default=5000.0, help="Camera focal length.")
    parser.add_argument("--cam_x", type=float, default=-0.0026177440)
    parser.add_argument("--cam_y", type=float, default=0.1)
    parser.add_argument("--cam_z", type=float, default=-13.0)
    parser.add_argument("--mesh_rx_deg", type=float, default=0.0, help="Extra mesh rotation around X in degrees.")
    parser.add_argument("--mesh_ry_deg", type=float, default=0.0, help="Extra mesh rotation around Y in degrees.")
    parser.add_argument("--mesh_rz_deg", type=float, default=0.0, help="Extra mesh rotation around Z in degrees.")
    parser.add_argument("--raw_video", type=str, default=None, help="Optional raw video path for side-by-side comparison.")
    parser.add_argument("--output_dir", type=str, default="visualize/raw_smplx_mesh")
    parser.add_argument("--sample_name", type=str, default=None)
    parser.add_argument("--save_vertices", action="store_true", help="Also save vertices/joints/pose arrays.")
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default cuda if available.")
    return parser.parse_args()


def _load_pickle_or_pt(path: str):
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu", weights_only=False)
    with open(path, "rb") as f:
        return pickle.load(f)


def _sort_key(name: str):
    m = re.findall(r"\d+", name)
    if m:
        return (0, int(m[-1]), name)
    return (1, 0, name)


def collect_frame_files(pose_dir: str):
    files = [f for f in os.listdir(pose_dir) if f.endswith(".pkl") or f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No .pkl/.pt found in {pose_dir}")
    files = sorted(files, key=_sort_key)
    return [os.path.join(pose_dir, f) for f in files]


def sample_uniform_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0 or len(arr) <= target_len:
        return arr
    idx = np.linspace(0, len(arr) - 1, num=target_len, dtype=int)
    return arr[idx]


def maybe_resample(arr: np.ndarray, input_fps: float = None, max_frames: int = 0) -> np.ndarray:
    out = arr
    if input_fps is not None and input_fps > 24:
        tgt = max(1, int(24 * len(out) / input_fps))
        out = sample_uniform_array(out, tgt)
    if max_frames and max_frames > 0:
        out = sample_uniform_array(out, max_frames)
    return out


def load_pose179_from_dir(pose_dir: str) -> np.ndarray:
    frame_files = collect_frame_files(pose_dir)
    clip = np.zeros((len(frame_files), 179), dtype=np.float32)
    for i, p in enumerate(frame_files):
        frame = _load_pickle_or_pt(p)
        missing = [k for k in POSE_KEYS if k not in frame]
        if missing:
            raise KeyError(f"{p} missing keys: {missing}")
        clip[i] = np.concatenate([np.asarray(frame[k]).reshape(-1) for k in POSE_KEYS], axis=0).astype(np.float32)
    return clip


def load_any_array(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [T,C], got shape {arr.shape} from {path}")
    return arr.astype(np.float32)


def load_stats_133(mean_path: str, std_path: str):
    if mean_path is None or std_path is None:
        raise ValueError("mean/std paths are required for feat133_norm input.")
    if mean_path.endswith(".npy"):
        mean = np.load(mean_path)
    else:
        mean = torch.load(mean_path, map_location="cpu")
    if std_path.endswith(".npy"):
        std = np.load(std_path)
    else:
        std = torch.load(std_path, map_location="cpu")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape[0] == 179 and std.shape[0] == 179:
        mean = mean[(3 + 3 * 11):]
        std = std[(3 + 3 * 11):]
        mean = np.concatenate([mean[:-20], mean[-10:]], axis=0)
        std = np.concatenate([std[:-20], std[-10:]], axis=0)
    if mean.shape[0] != 133 or std.shape[0] != 133:
        raise ValueError(f"Expect 133-dim stats after processing, got mean {mean.shape}, std {std.shape}")
    return mean, std


def split_to_smplx_params(arr: np.ndarray, mode: str, mean133=None, std133=None):
    # Output each is [T, D]
    if mode == "pose179":
        if arr.shape[1] != 179:
            raise ValueError(f"pose179 expects C=179, got {arr.shape[1]}")
        root = arr[:, 0:3]
        body = arr[:, 3:66]
        lhand = arr[:, 66:111]
        rhand = arr[:, 111:156]
        jaw = arr[:, 156:159]
        shape = arr[:, 159:169]
        expr = arr[:, 169:179]
        return root, body, lhand, rhand, jaw, shape, expr

    if arr.shape[1] != 133:
        raise ValueError(f"{mode} expects C=133, got {arr.shape[1]}")
    feat = arr
    if mode == "feat133_norm":
        if mean133 is None or std133 is None:
            raise ValueError("feat133_norm requires mean133/std133")
        feat = feat * std133[None, :] + mean133[None, :]

    zero_pose = np.zeros((feat.shape[0], 36), dtype=np.float32)
    full169 = np.concatenate([zero_pose, feat], axis=1)
    root = full169[:, 0:3]
    body = full169[:, 3:66]
    lhand = full169[:, 66:111]
    rhand = full169[:, 111:156]
    jaw = full169[:, 156:159]
    expr = full169[:, 159:169]
    shape = np.repeat(SHAPE_DEFAULT[None, :], feat.shape[0], axis=0)
    return root, body, lhand, rhand, jaw, shape, expr


def smplx_to_vertices_and_joints(root, body, lhand, rhand, jaw, shape, expr, device):
    t = root.shape[0]
    root_t = torch.from_numpy(root).to(device=device, dtype=torch.float32)
    body_t = torch.from_numpy(body).to(device=device, dtype=torch.float32)
    lhand_t = torch.from_numpy(lhand).to(device=device, dtype=torch.float32)
    rhand_t = torch.from_numpy(rhand).to(device=device, dtype=torch.float32)
    jaw_t = torch.from_numpy(jaw).to(device=device, dtype=torch.float32)
    shape_t = torch.from_numpy(shape).to(device=device, dtype=torch.float32)
    expr_t = torch.from_numpy(expr).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        vertices, joints = get_coord(
            root_pose=root_t,
            body_pose=body_t,
            lhand_pose=lhand_t,
            rhand_pose=rhand_t,
            jaw_pose=jaw_t,
            shape=shape_t,
            expr=expr_t,
        )
    vertices = vertices.detach().cpu().numpy().reshape(t, -1, 3)
    joints = joints.detach().cpu().numpy().reshape(t, -1, 3)
    return vertices, joints


def render_one_frame(renderer, vertices, faces, width, height, focal, cam_trans, mesh_rot_deg):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rx, ry, rz = mesh_rot_deg
    if abs(rx) > 1e-8:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0]))
    if abs(ry) > 1e-8:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0]))
    if abs(rz) > 1e-8:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1]))

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(1.0, 1.0, 0.9, 1.0),
    )
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)

    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3), bg_color=[255, 255, 255, 255])
    scene.add(mesh_node, "mesh")

    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, 3] = cam_trans
    camera_pose[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    camera = pyrender.camera.IntrinsicsCamera(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
    light_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
    scene.add(light, pose=light_pose)

    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    bg = np.ones((height, width, 3), dtype=np.float32) * 255.0
    valid = (depth > 0)[:, :, None]
    frame = rgb * valid + bg * (1.0 - valid)
    return np.asarray(frame, dtype=np.uint8)


def render_mesh_video(vertices_seq, faces, out_mp4, fps, width, height, focal, cam_trans, mesh_rot_deg):
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    except Exception as e:
        backend = os.environ.get("PYOPENGL_PLATFORM", "<unset>")
        raise RuntimeError(
            f"Failed to create OffscreenRenderer with PYOPENGL_PLATFORM={backend}. "
            f"Try `PYOPENGL_PLATFORM=egl` or `PYOPENGL_PLATFORM=osmesa` explicitly. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    frames = []
    try:
        for i in tqdm(range(vertices_seq.shape[0]), desc="render mesh"):
            frame = render_one_frame(renderer, vertices_seq[i], faces, width, height, focal, cam_trans, mesh_rot_deg)
            frames.append(frame)
        imageio.mimsave(out_mp4, frames, fps=fps)
    finally:
        renderer.delete()
    return frames


def read_video_frames(video_path: str, target_len: int, width: int, height: int):
    frames = []
    reader = imageio.get_reader(video_path)
    for f in reader:
        img = Image.fromarray(f).convert("RGB").resize((width, height))
        frames.append(np.asarray(img))
    reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    idx = np.linspace(0, len(frames) - 1, num=target_len, dtype=int)
    return [frames[i] for i in idx]


def save_side_by_side(raw_frames, mesh_frames, out_path, fps):
    assert len(raw_frames) == len(mesh_frames)
    merged = [np.concatenate([r, m], axis=1) for r, m in zip(raw_frames, mesh_frames)]
    imageio.mimsave(out_path, merged, fps=fps)


def main():
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise RuntimeError("This script requires CUDA because get_coord uses a CUDA SMPL-X layer in current repo.")

    if args.pose_dir:
        arr = load_pose179_from_dir(args.pose_dir)
        mode = "pose179"
        sample_name = args.sample_name or Path(args.pose_dir).name
    else:
        arr = load_any_array(args.pose_npy)
        mode = args.input_type
        sample_name = args.sample_name or Path(args.pose_npy).stem

    arr = maybe_resample(arr, input_fps=args.input_fps, max_frames=args.max_frames)
    if len(arr) < 2:
        raise ValueError(f"Too few frames after sampling: {len(arr)}")

    mean133 = std133 = None
    if mode == "feat133_norm":
        mean133, std133 = load_stats_133(args.mean_path, args.std_path)

    root, body, lhand, rhand, jaw, shape, expr = split_to_smplx_params(arr, mode, mean133, std133)
    vertices, joints = smplx_to_vertices_and_joints(root, body, lhand, rhand, jaw, shape, expr, device)

    out_dir = Path(args.output_dir) / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_mp4 = out_dir / f"{sample_name}_raw_mesh.mp4"
    cam_trans = np.array([args.cam_x, args.cam_y, args.cam_z], dtype=np.float32)
    mesh_rot_deg = (args.mesh_rx_deg, args.mesh_ry_deg, args.mesh_rz_deg)
    mesh_frames = render_mesh_video(
        vertices_seq=vertices,
        faces=smpl_x.face,
        out_mp4=str(mesh_mp4),
        fps=args.fps,
        width=args.width,
        height=args.height,
        focal=args.focal,
        cam_trans=cam_trans,
        mesh_rot_deg=mesh_rot_deg,
    )

    if args.raw_video:
        raw_frames = read_video_frames(args.raw_video, target_len=len(mesh_frames), width=args.width, height=args.height)
        compare_mp4 = out_dir / f"{sample_name}_raw_vs_mesh.mp4"
        save_side_by_side(raw_frames, mesh_frames, str(compare_mp4), args.fps)
        print(f"[OK] side-by-side: {compare_mp4}")

    if args.save_vertices:
        np.save(out_dir / f"{sample_name}_vertices.npy", vertices)
        np.save(out_dir / f"{sample_name}_joints.npy", joints)
        np.save(out_dir / f"{sample_name}_pose_input.npy", arr)

    print(f"[OK] mesh video: {mesh_mp4}")
    print(f"Frames: {len(arr)}, Input mode: {mode}, Output dir: {out_dir}")


if __name__ == "__main__":
    main()
