import json
import os, pickle; os.environ["PYOPENGL_PLATFORM"] = "egl"
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.utils.logger import create_logger
from mGPT.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
import mGPT.render.matplot.plot_3d_global as plot_3d
import pyrender, trimesh
from mGPT.utils.human_models import smpl_x
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips, clips_array
from moviepy.video.fx.all import crop
import matplotlib.pyplot as plt
from mGPT.utils.human_models import get_coord
import pandas as pd
import random; random.seed(0)
from PIL import Image
import seaborn as sns; sns.set_style('darkgrid')
import math
import random; random.seed(0)
from multiprocessing import Pool, get_context
import traceback


keys = ['smplx_root_pose', 
        'smplx_body_pose', 
        'smplx_lhand_pose', 
        'smplx_rhand_pose', 
        'smplx_jaw_pose', 
        'smplx_shape', 
        'smplx_expr'
    ]

h2s_csl_mean = torch.load('../data/rzuo/CSL-Daily/mean.pt').cuda()
h2s_csl_std = torch.load('../data/rzuo/CSL-Daily/std.pt').cuda()
h2s_csl_mean = h2s_csl_mean[(3+3*11):]
h2s_csl_mean = torch.cat([h2s_csl_mean[:-20], h2s_csl_mean[-10:]], dim=0)
h2s_csl_std = h2s_csl_std[(3+3*11):]
h2s_csl_std = torch.cat([h2s_csl_std[:-20], h2s_csl_std[-10:]], dim=0)


def sanitize_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width, x + w))
    y2 = np.min((img_height, y + h))

    x, y, w, h = x1, y1, x2 - x1, y2 - y1

    return x, y, w, h


def process_bbox(bbox, img_height, img_width):
    x, y, w, h = bbox
    x -= (w * 0.2) / 2
    y -= (h * 0.2) / 2
    w *= 1.2
    h *= 1.2

    x, y, w, h = sanitize_bbox([x, y, w, h], img_width, img_height)

    # aspect ratio preserving bbox
    w_h_ratio = w / h
    if w_h_ratio > 1:
        h_n = w / 1
        y -= (h_n - h) / 2
        h = h_n

    else:
        w_n = h * 1
        x -= (w_n - w) / 2
        w = w_n

    x, y, w, h = sanitize_bbox([x, y, w, h], img_width, img_height)

    return x, y, w, h


def feats2joints(features, mean, std, rot6d=False):
    #smpl2joints and drop lowerbody
    features = features * std + mean
    # return recover_from_ric(features, self.njoints)

    zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
    shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172, 
                            0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).to(features)
    B, T = features.shape[:2]
    shape_param = shape_param.repeat(B, T, 1).view(B*T, -1)
    # print(features.shape, shape_param.shape)

    if rot6d:
        # 6d rotation to axis angle
        expr = features[..., -10:] #B,T,10
        features = features[..., :-10].view(B, T, -1, 6)
        features = matrix_to_axis_angle(rotation_6d_to_matrix(features))  #B,T,N,3
        features = features.view(B, T, -1)
        features = torch.cat([features, expr], dim=-1)

    features = torch.cat([zero_pose, features], dim=-1).view(B*T, -1)  #133+36=169
    vertices, joints = get_coord(root_pose=features[..., 0:3], body_pose=features[..., 3:66], 
                                    lhand_pose=features[..., 66:111], rhand_pose=features[..., 111:156], 
                                    jaw_pose=features[..., 156:159], shape=shape_param, 
                                    expr=features[..., 159:169])
    return vertices, joints


def sample(lst, count):
    indices = np.linspace(0, len(lst)-1, count, dtype=int)
    return [lst[i] for i in indices]


def render_mesh(img, mesh, face, cam_trans):
    mesh = trimesh.Trimesh(vertices=mesh, faces=face, process=False)
    Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
    mesh.apply_transform(Rx)
    
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    # Camera setup
    camera_center = [img.shape[1]/2, img.shape[0]/2]
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_trans
    camera_pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    
    focal_length = 5000
    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)
    
    # Light
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
    light_pose = np.eye(4)
    light_pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    scene.add(light, pose=light_pose)
    
    # Render
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
        rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        rgb = rgb[:,:,:3].astype(np.float32)
        valid_mask = (depth > 0)[:,:,None]
        output_img = rgb * valid_mask + img * (1-valid_mask)
        output_img = np.array(output_img, dtype=np.uint8)
    except Exception as e:
        print(f"[WARN] OpenGL rendering failed ({type(e).__name__}): {str(e)[:100]}")
        output_img = np.array(img, dtype=np.uint8)
    
    return output_img, mesh


def process_sample(args):
    """Process a single sample - can run in parallel on different GPUs"""
    idx, sample_name, config_dict = args
    gpu_id = idx % 8  # Assign GPU based on process index
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        # Reconstruct config from dict
        dataset = config_dict['dataset']
        baseline = config_dict['baseline']
        ours = config_dict['ours']
        split = config_dict['split']
        fps = config_dict['fps']
        h = config_dict['h']
        w = config_dict['w']
        cam_trans = config_dict['cam_trans']
        save_dir = config_dict['save_dir']
        raw_vid_dir = config_dict['raw_vid_dir']
        rot6d = config_dict['rot6d']
        save_mesh = config_dict['save_mesh']
        save_mesh_dir = config_dict['save_mesh_dir']
        
        # Reload stats on this GPU
        mean = torch.load('../data/rzuo/CSL-Daily/mean.pt').cuda()
        std = torch.load('../data/rzuo/CSL-Daily/std.pt').cuda()
        mean = mean[(3+3*11):]
        mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
        std = std[(3+3*11):]
        std = torch.cat([std[:-20], std[-10:]], dim=0)
        
        n = sample_name
        print(f"[GPU {gpu_id}] Processing {idx}: {n}")
        
        # Load baseline result
        for r in range(8):
            dir = os.path.join(baseline, f'{split}_rank_{r}')
            if os.path.exists(dir) and f"{n.split('/')[-1]}.pkl" in os.listdir(dir):
                with open(os.path.join(dir, f"{n.split('/')[-1]}.pkl"), 'rb') as f:
                    res_base = pickle.load(f)
                break
        
        # Load SOKE result
        for r in range(8):
            dir = os.path.join(ours, f'{split}_rank_{r}')
            if os.path.exists(dir) and f"{n.split('/')[-1]}.pkl" in os.listdir(dir):
                with open(os.path.join(dir, f"{n.split('/')[-1]}.pkl"), 'rb') as f:
                    res_ours = pickle.load(f)
                break
        
        feats_ref = res_ours['feats_ref']
        feats_rst_ours = res_ours['feats_rst']
        feats_rst_base = res_base['feats_rst']
        text = res_ours['text']
        print(f"[GPU {gpu_id}] {text}")

        vertices_ref = feats2joints(torch.from_numpy(feats_ref).cuda().unsqueeze(0), mean=mean, std=std, rot6d=rot6d)[0].cpu().numpy()
        vertices_rst_ours = feats2joints(torch.from_numpy(feats_rst_ours).cuda().unsqueeze(0), mean=mean, std=std, rot6d=rot6d)[0].cpu().numpy()
        if dataset == 'how2sign':
            vertices_rst_base = feats2joints(torch.from_numpy(feats_rst_base).cuda().unsqueeze(0), mean, std)[0].cpu().numpy()
        elif dataset == 'csl':
            vertices_rst_base = feats2joints(torch.from_numpy(feats_rst_base).cuda().unsqueeze(0), mean, std)[0].cpu().numpy()
        elif dataset == 'phoenix':
            vertices_rst_base = feats2joints(torch.from_numpy(feats_rst_base).cuda().unsqueeze(0), mean, std)[0].cpu().numpy()

        frames = []
        rst_len = feats_rst_ours.shape[0]
        rst_len_base = feats_rst_base.shape[0]
        
        # read raw video
        gt_frames = []
        if dataset == 'how2sign':
            gt_vid_path = os.path.join(raw_vid_dir, n+'.mp4')
            clip = VideoFileClip(gt_vid_path)
            raw_w, raw_h = clip.size
            clip = clip.crop(width=720, height=720, x_center=raw_w//2, y_center=raw_h//2)
            clip = clip.resize(width=512, height=512)
            for frame in clip.iter_frames():
                frame = Image.fromarray(frame, 'RGB')
                gt_frames.append(frame)
            csv = pd.read_csv('../data/How2Sign/test/re_aligned/how2sign_realigned_test_preprocessed_fps.csv')
            raw_fps = csv[csv['SENTENCE_NAME']==n]['fps'].item()
            if raw_fps > 25:
                gt_frames = sample(gt_frames, count=int(25*len(gt_frames)/raw_fps))
        elif dataset == 'csl':
            # CSL uses MP4 files
            gt_vid_path = os.path.join(raw_vid_dir, n+'.mp4')
            try:
                clip = VideoFileClip(gt_vid_path)
                for frame in clip.iter_frames():
                    frame = Image.fromarray(frame, 'RGB').resize((w, h))
                    gt_frames.append(frame)
                clip.close()
            except Exception as e:
                print(f"[WARN] Could not load video {gt_vid_path}: {e}, using black frames")
                for _ in range(96):
                    gt_frames.append(Image.new('RGB', (w, h), color=(0, 0, 0)))
        else:
            # Phoenix and other datasets use frame directories
            gt_vid_path = os.path.join(raw_vid_dir, n)
            try:
                frame_lst = os.listdir(gt_vid_path)
                frame_lst = sorted(frame_lst)
                for fname in frame_lst:
                    img = Image.open(os.path.join(gt_vid_path, fname)).resize((w, h)).convert('RGB')
                    gt_frames.append(img)
            except Exception as e:
                print(f"[WARN] Could not load frames from {gt_vid_path}: {e}, using black frames")
                for _ in range(96):
                    gt_frames.append(Image.new('RGB', (w, h), color=(0, 0, 0)))
        ref_len = len(gt_frames)

        print(f"[GPU {gpu_id}] Frames: ref={ref_len}, ours={rst_len}, baseline={rst_len_base}")
        
        if save_mesh and save_mesh_dir:
            cur_mesh_dir = os.path.join(save_mesh_dir, n.split('/')[-1], 'mesh')
            os.makedirs(cur_mesh_dir, exist_ok=True)
        
        for f_idx in tqdm(range(0, max(ref_len, rst_len)), disable=True):  # disable tqdm for parallel
            img_gt = gt_frames[min(f_idx, ref_len-1)]
            if save_mesh:
                img_gt.save(os.path.join(cur_mesh_dir, f'{f_idx:03d}.png'))
            img_gt_mesh = np.zeros((h, w, 3), dtype=np.int8)
            img_gt_mesh, mesh_gt = render_mesh(img=img_gt_mesh, mesh=vertices_ref[min(f_idx, vertices_ref.shape[0]-1)], face=smpl_x.face, cam_trans=cam_trans)

            img_pred_base = np.zeros((h, w, 3), dtype=np.int8)
            img_pred_base, mesh_base = render_mesh(img=img_pred_base, mesh=vertices_rst_base[min(f_idx, rst_len_base-1)], face=smpl_x.face, cam_trans=cam_trans)
            
            img_pred = np.zeros((h, w, 3), dtype=np.int8)
            img_pred, mesh_ours = render_mesh(img=img_pred, mesh=vertices_rst_ours[min(f_idx, rst_len-1)], face=smpl_x.face, cam_trans=cam_trans)
            
            frames.append(np.concatenate([img_gt, img_pred_base, img_pred], axis=1))

            if save_mesh:
                mesh_gt.export(os.path.join(cur_mesh_dir, f'{f_idx:03d}_gt.ply'))
                mesh_base.export(os.path.join(cur_mesh_dir, f'{f_idx:03d}_base.ply'))
                mesh_ours.export(os.path.join(cur_mesh_dir, f'{f_idx:03d}_ours.ply'))
        
        save_path = os.path.join(save_dir, f"{idx:04d}_{n.split('/')[-1]}.mp4")
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(save_path, fps=fps, verbose=False, logger=None)
        print(f"[GPU {gpu_id}] ✓ Saved {save_path}")
        return idx
        
    except Exception as e:
        print(f"[ERROR] Failed to process {sample_name}: {str(e)}")
        traceback.print_exc()
        return None


def main(save_mesh=False):
    # parse options
    cfg = parse_args(phase="demo")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all 8 GPUs
    
    dataset = cfg.DEMO_DATASET
    # visualize parameters
    fps = 18
    h, w = 512, 512
    focal = [5000, 5000]
    princpt = [h/2, w/2]
    bbox = process_bbox([0,0,h,w], h, w)
    focal = [focal[0] / w * bbox[2], focal[1] / h * bbox[3]]
    princpt = [princpt[0] / w * bbox[2] + bbox[0], princpt[1] / h * bbox[3] + bbox[1]]
    cam_trans = np.array([-2.6177440e-03, 0.1, -13], dtype=np.float32)
    save_dir = f'visualize/compare_{dataset}'
    save_mesh_dir = f'visualize/compare_{dataset}'
    rot6d = OmegaConf.select(cfg, "DATASET.H2S.rot6d", default=False)
    os.makedirs(save_dir, exist_ok=True)
    if save_mesh:
        os.makedirs(save_mesh_dir, exist_ok=True)

    if dataset == 'csl' or dataset is None:
        baseline = 'results/mgpt/baseline'
        raw_vid_dir = 'data/CSL-Daily/csl-daily'
        dataset = 'csl'  # Set default if None
    elif dataset == 'how2sign':
        baseline = 'results/mgpt/baseline'
        raw_vid_dir = 'data/How2Sign/test/raw_videos'
    elif dataset == 'phoenix':
        baseline = 'results/mgpt/baseline'
        raw_vid_dir = 'data/Phoenix_2014T/fullFrame-210x260px'
    ours = 'results/mgpt/SOKE'
    split = 'test'

    scores_ours = {}
    for rank in range(8):
        score_path = os.path.join(ours, f'{split}_rank_{rank}/test_scores.json')
        if os.path.exists(score_path):
            with open(score_path, 'r') as f:
                scores = json.load(f)
                scores_ours.update(scores)
    
    names = []
    for k,v in scores_ours.items():
        if dataset in list(v.keys())[0]:
            names.append(k)
    random.shuffle(names)
    print('tot num: ', len(names))

    start, end = 0, 20
    sample_list = []
    for i in range(len(names)):
        if i < start:
            continue
        if i > end:
            break
        
        sample_list.append(i)

    # Prepare config dict for multiprocessing
    config_dict = {
        'dataset': dataset,
        'baseline': baseline,
        'ours': ours,
        'split': split,
        'fps': fps,
        'h': h,
        'w': w,
        'cam_trans': cam_trans,
        'save_dir': save_dir,
        'raw_vid_dir': raw_vid_dir,
        'rot6d': rot6d,
        'save_mesh': save_mesh,
        'save_mesh_dir': save_mesh_dir,
    }

    # Create task list: (index, sample_name, config)
    tasks = [(i, names[i], config_dict) for i in sample_list]
    
    print(f"\n{'='*60}")
    print(f"Processing {len(tasks)} samples in parallel using 8 GPUs")
    print(f"{'='*60}\n")
    
    # Process samples in parallel using 8 workers (one per GPU)
    # Use 'spawn' context to avoid CUDA fork issues
    ctx = get_context('spawn')
    with ctx.Pool(processes=8) as pool:
        results = pool.map(process_sample, tasks)
    
    successful = [r for r in results if r is not None]
    print(f"\n{'='*60}")
    print(f"✓ Successfully processed {len(successful)}/{len(tasks)} samples")
    print(f"Videos saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main(save_mesh=False)
