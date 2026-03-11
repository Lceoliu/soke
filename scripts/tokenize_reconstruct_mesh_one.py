#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import imageio
import numpy as np
import torch
import trimesh
import pyrender
from tqdm import tqdm

from mGPT.utils.human_models import get_coord, smpl_x
from tokenize_reconstruct_one import (
    load_cfg,
    load_pose_clip,
    load_mean_std,
    build_tokenizers,
    encode_tokens,
    decode_tokens,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize one sign sample, reconstruct it, and render mesh MP4."
    )
    parser.add_argument("--cfg", type=str, default="configs/soke.yaml")
    parser.add_argument("--cfg_assets", type=str, default="configs/assets.yaml")
    parser.add_argument("--pose_dir", type=str, required=True, help="Directory of per-frame *_3D.pkl files.")
    parser.add_argument("--sample_name", type=str, default=None, help="Optional output name. Default: pose_dir basename.")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None, help="Tokenizer ckpt. Default from cfg.TRAIN.PRETRAINED_VAE.")
    parser.add_argument("--mean_path", type=str, default=None, help="Mean path. Default from cfg.DATASET.H2S.MEAN_PATH.")
    parser.add_argument("--std_path", type=str, default=None, help="Std path. Default from cfg.DATASET.H2S.STD_PATH.")
    parser.add_argument("--input_fps", type=float, default=None, help="Raw FPS. If >24, sequence is resampled to 24.")
    parser.add_argument("--fps", type=int, default=18, help="Output MP4 FPS.")
    parser.add_argument("--width", type=int, default=512, help="Output frame width.")
    parser.add_argument("--height", type=int, default=512, help="Output frame height.")
    parser.add_argument("--focal", type=float, default=5000.0, help="Camera focal length.")
    parser.add_argument("--cam_x", type=float, default=-0.0026177440)
    parser.add_argument("--cam_y", type=float, default=0.1)
    parser.add_argument("--cam_z", type=float, default=13.0)
    parser.add_argument("--output_dir", type=str, default="visualize/token_recon_mesh_one")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: cuda if available.")
    parser.add_argument("--save_ply_dir", action="store_true", help="Also export per-frame reconstructed mesh ply files.")
    return parser.parse_args()


def feats_to_vertices_and_joints(feats_denorm):
    # get_coord is CUDA-only in this repo because it builds SMPL-X layer with .cuda()
    if feats_denorm.device.type != "cuda":
        raise RuntimeError("Mesh rendering requires CUDA in current repo implementation.")

    bsz, tlen, _ = feats_denorm.shape
    zero_pose = torch.zeros((bsz, tlen, 36), device=feats_denorm.device, dtype=feats_denorm.dtype)
    shape_param = torch.tensor(
        [[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
           0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]],
        device=feats_denorm.device,
        dtype=feats_denorm.dtype,
    )
    shape_param = shape_param.repeat(bsz, tlen, 1).view(bsz * tlen, -1)
    x = torch.cat([zero_pose, feats_denorm], dim=-1).view(bsz * tlen, -1)

    vertices, joints = get_coord(
        root_pose=x[:, 0:3],
        body_pose=x[:, 3:66],
        lhand_pose=x[:, 66:111],
        rhand_pose=x[:, 111:156],
        jaw_pose=x[:, 156:159],
        shape=shape_param,
        expr=x[:, 159:169],
    )
    vertices = vertices.view(bsz, tlen, vertices.shape[1], 3)
    joints = joints.view(bsz, tlen, joints.shape[1], 3)
    return vertices, joints


def render_one_frame(
    renderer, vertices, faces, width=512, height=512, 
    focal=None, cam_trans=None, mesh_rot_deg=(0, 0, 0), auto_cam=False
):
    import trimesh.transformations as tf
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # --- 1. 旋转模型 ---
    rx, ry, rz = mesh_rot_deg
    if abs(rx) > 1e-8:
        mesh.apply_transform(tf.rotation_matrix(np.radians(rx),[1, 0, 0]))
    if abs(ry) > 1e-8:
        mesh.apply_transform(tf.rotation_matrix(np.radians(ry), [0, 1, 0]))
    if abs(rz) > 1e-8:
        mesh.apply_transform(tf.rotation_matrix(np.radians(rz), [0, 0, 1]))

    # --- 2. 优化材质 (增强细节辨识度) ---
    # 稍微降低粗糙度、增加一点金属感，打造类似塑料/黏土的质感
    # 这样手掌弯曲、手指交错时会有明显的高光和阴影过度
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,     # 引入微小金属感，收紧高光
        roughnessFactor=0.4,    # 降低粗糙度，使其具有漫反射光泽
        alphaMode="OPAQUE",
        baseColorFactor=(0.85, 0.85, 0.9, 1.0), # 微微偏冷的灰白色，显高级且不刺眼
    )
    
    # 开启 smooth=True，计算平滑法线，避免手部出现块状多边形干扰视觉
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    # --- 3. 设置深色背景 ---
    bg_color = np.array([40, 42, 45], dtype=np.float32) # 深灰黑色
    scene = pyrender.Scene(
        ambient_light=(0.15, 0.15, 0.15), 
        bg_color=list(bg_color) + [255]
    )
    scene.add(mesh_node, "mesh")

    # --- 4. 自动计算相机参数 (包围盒适配) ---
    if auto_cam or cam_trans is None or focal is None:
        min_v = mesh.vertices.min(axis=0)
        max_v = mesh.vertices.max(axis=0)
        center = (min_v + max_v) / 2.0
        max_extent = np.max(max_v - min_v)
        
        if focal is None:
            focal = max(width, height)
            
        distance = (max_extent * focal / min(width, height)) * 1.2
        
        cam_trans = np.array([center[0], center[1], center[2] + distance])

    # --- 5. 设置相机 ---
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, 3] = cam_trans
    camera_pose[:3, :3] = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    camera = pyrender.camera.IntrinsicsCamera(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0)
    scene.add(camera, pose=camera_pose)

    # --- 6. 三点布光系统 (立体感核心) ---
    # 主光 (Key Light): 相对相机向右偏、向上偏，照亮正面主体
    key_local = tf.euler_matrix(-np.pi/5, -np.pi/4, 0, 'rxyz')
    scene.add(pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=5.0), pose=camera_pose @ key_local)

    # 辅光 (Fill Light): 相对相机向左偏，用于打亮主光留下的死黑阴影
    fill_local = tf.euler_matrix(-np.pi/6, np.pi/4, 0, 'rxyz')
    scene.add(pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=2.0), pose=camera_pose @ fill_local)

    # 轮廓光 (Rim Light): 放在相机正对面的主体后侧，打出漂亮的高光边缘（能清楚看见每一根手指的边缘）
    rim_local = tf.euler_matrix(0, np.pi * 0.85, 0, 'rxyz')
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=camera_pose @ rim_local)


    # --- 7. 渲染与背景合成 ---
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    
    # 使用设定好的深色背景矩阵
    bg_mat = np.ones((height, width, 3), dtype=np.float32) * bg_color
    valid = (depth > 0)[:, :, None]
    frame = rgb * valid + bg_mat * (1.0 - valid)
    
    # 防止溢出并返回
    return np.clip(frame, 0, 255).astype(np.uint8)



def render_mesh_video(vertices_seq, faces, out_mp4, fps, width, height, focal, cam_trans, save_ply_dir=None):
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
    frames = []
    try:
        for i in tqdm(range(vertices_seq.shape[0]), desc="render mesh"):
            v = vertices_seq[i]
            frame = render_one_frame(renderer, v, faces, width, height, focal, cam_trans)
            frames.append(frame)
            if save_ply_dir is not None:
                trimesh.Trimesh(vertices=v, faces=faces, process=False).export(
                    os.path.join(save_ply_dir, f"{i:05d}.ply")
                )
        imageio.mimsave(out_mp4, frames, fps=fps)
    finally:
        renderer.delete()


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg, args.cfg_assets)

    pose_dir = os.path.abspath(args.pose_dir)
    sample_name = args.sample_name or os.path.basename(os.path.normpath(pose_dir))
    out_dir = os.path.join(args.output_dir, sample_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ckpt_path = args.tokenizer_ckpt or cfg.TRAIN.PRETRAINED_VAE
    if not ckpt_path:
        raise ValueError("Tokenizer checkpoint is empty. Provide --tokenizer_ckpt or set TRAIN.PRETRAINED_VAE in cfg.")
    mean_path = args.mean_path or cfg.DATASET.H2S.MEAN_PATH
    std_path = args.std_path or cfg.DATASET.H2S.STD_PATH

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Please run with CUDA (e.g., --device cuda).")

    print(f"[1/6] Loading clip from {pose_dir}")
    feats_raw = load_pose_clip(pose_dir, input_fps=args.input_fps)
    print(f"      Frames: {feats_raw.shape[0]}, Feature dim: {feats_raw.shape[1]}")

    print(f"[2/6] Loading mean/std from {mean_path} and {std_path}")
    mean, std = load_mean_std(mean_path, std_path)
    if feats_raw.shape[1] != mean.numel():
        raise ValueError(f"Feature dim {feats_raw.shape[1]} mismatches mean/std dim {mean.numel()}.")
    feats_norm = (torch.from_numpy(feats_raw).float() - mean[None, :]) / (std[None, :] + 1e-10)
    feats_norm = feats_norm.unsqueeze(0).to(device)

    print(f"[3/6] Loading tokenizer from {ckpt_path}")
    vae, hand_vae, rhand_vae = build_tokenizers(cfg, ckpt_path, device)

    print("[4/6] Tokenizing and reconstructing")
    with torch.no_grad():
        token_dict, tokens_np = encode_tokens(feats_norm, vae, hand_vae, rhand_vae)
        feats_recon_norm = decode_tokens(token_dict, vae, hand_vae, rhand_vae)

    mean_d = mean.to(device)
    std_d = std.to(device)
    if feats_recon_norm.shape[-1] != mean_d.numel():
        raise ValueError(
            f"Decoded feature dim {feats_recon_norm.shape[-1]} mismatches mean/std dim {mean_d.numel()}."
        )
    feats_recon = feats_recon_norm * std_d[None, None, :] + mean_d[None, None, :]

    print("[5/6] Converting reconstructed features to SMPL-X vertices")
    with torch.no_grad():
        vertices, joints = feats_to_vertices_and_joints(feats_recon)
    vertices_np = vertices[0].detach().cpu().numpy()
    joints_np = joints[0].detach().cpu().numpy()

    tokens_path = os.path.join(out_dir, f"{sample_name}_tokens.npy")
    recon_norm_path = os.path.join(out_dir, f"{sample_name}_recon_norm.npy")
    recon_path = os.path.join(out_dir, f"{sample_name}_recon.npy")
    vertices_path = os.path.join(out_dir, f"{sample_name}_vertices.npy")
    joints_path = os.path.join(out_dir, f"{sample_name}_joints.npy")
    mp4_path = os.path.join(out_dir, f"{sample_name}_mesh.mp4")

    np.save(tokens_path, tokens_np)
    np.save(recon_norm_path, feats_recon_norm[0].detach().cpu().numpy())
    np.save(recon_path, feats_recon[0].detach().cpu().numpy())
    np.save(vertices_path, vertices_np)
    np.save(joints_path, joints_np)

    save_ply_dir = None
    if args.save_ply_dir:
        save_ply_dir = os.path.join(out_dir, "mesh_ply")
        Path(save_ply_dir).mkdir(parents=True, exist_ok=True)

    print("[6/6] Rendering mesh video")
    cam_trans = np.array([args.cam_x, args.cam_y, args.cam_z], dtype=np.float32)
    render_mesh_video(
        vertices_seq=vertices_np,
        faces=smpl_x.face,
        out_mp4=mp4_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        focal=args.focal,
        cam_trans=cam_trans,
        save_ply_dir=save_ply_dir,
    )

    print(f"      token:      {tokens_path}")
    print(f"      recon norm: {recon_norm_path}")
    print(f"      recon:      {recon_path}")
    print(f"      vertices:   {vertices_path}")
    print(f"      joints:     {joints_path}")
    print(f"      mesh mp4:   {mp4_path}")
    if save_ply_dir is not None:
        print(f"      mesh ply:   {save_ply_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
