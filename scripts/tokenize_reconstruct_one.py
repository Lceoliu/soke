#!/usr/bin/env python3
import argparse
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from omegaconf import OmegaConf

from mGPT.config import get_module_config, instantiate_from_config


POSE_KEYS = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize one sign sample and reconstruct it for visualization."
    )
    parser.add_argument("--cfg", type=str, default="configs/soke.yaml")
    parser.add_argument("--cfg_assets", type=str, default="configs/assets.yaml")
    parser.add_argument("--pose_dir", type=str, required=True, help="Directory of per-frame *_3D.pkl files.")
    parser.add_argument("--sample_name", type=str, default=None, help="Optional output name. Default: pose_dir basename.")
    parser.add_argument("--tokenizer_ckpt", type=str, default=None, help="Tokenizer ckpt. Default from cfg.TRAIN.PRETRAINED_VAE.")
    parser.add_argument("--mean_path", type=str, default=None, help="Mean path. Default from cfg.DATASET.H2S.MEAN_PATH.")
    parser.add_argument("--std_path", type=str, default=None, help="Std path. Default from cfg.DATASET.H2S.STD_PATH.")
    parser.add_argument("--input_fps", type=float, default=None, help="Raw FPS. If >24, sequence is resampled to 24.")
    parser.add_argument("--output_dir", type=str, default="visualize/token_recon_one")
    parser.add_argument("--title", type=str, default="Reconstruction")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: cuda if available.")
    parser.add_argument("--no_gif", action="store_true", help="Disable skeleton gif output.")
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


def sample_uniform(lst, count):
    if count <= 0:
        return []
    idx = np.linspace(0, len(lst) - 1, num=count, dtype=int)
    return [lst[i] for i in idx]


def collect_pose_files(pose_dir: str):
    # files = [f for f in os.listdir(pose_dir) if f.endswith("_3D.pkl")]
    # if not files:
    #     raise FileNotFoundError(f"No '*_3D.pkl' files found in {pose_dir}")

    # def frame_id(name):
    #     m = re.search(r"_(\d+)_3D\.pkl$", name)
    #     return int(m.group(1)) if m else 10**9

    # files = sorted(files, key=lambda x: (frame_id(x), x))
    # return [os.path.join(pose_dir, f) for f in files]
    files = [f for f in os.listdir(pose_dir) if f.endswith((".pt", ".pkl"))]
    if not files:
        raise FileNotFoundError(f"No '*.pt' or '*.pkl' files found in {pose_dir}")

    def frame_id(name: str) -> int:
        # Prefer explicit how2sign frame suffix: *_<frame>_3D.pkl
        m = re.search(r"_(\d+)_3D\.(?:pkl|pt)$", name)
        if m:
            return int(m.group(1))
        # Common CSL style: 000123.pkl
        m = re.search(r"(\d+)\.(?:pkl|pt)$", name)
        if m:
            return int(m.group(1))
        # Fallback to last numeric group if naming is unconventional.
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else 10**9

    files = sorted(files, key=lambda x: (frame_id(x), x))
    return [os.path.join(pose_dir, f) for f in files]


def load_pose_clip(pose_dir: str, input_fps=None):
    frame_list = collect_pose_files(pose_dir)
    if input_fps is not None and input_fps > 24:
        new_len = int(24 * len(frame_list) / input_fps)
        frame_list = sample_uniform(frame_list, max(1, new_len))

    if len(frame_list) < 4:
        raise ValueError(f"Too few frames ({len(frame_list)}). Need at least 4.")

    poses = np.zeros((len(frame_list), 179), dtype=np.float32)
    for i, p in enumerate(frame_list):
        data = torch.load(p, map_location="cpu", weights_only=False) if p.endswith(".pt") else None
        if data is None:
            import pickle
            with open(p, "rb") as f:
                data = pickle.load(f)
        poses[i] = np.concatenate([data[k] for k in POSE_KEYS], axis=0).astype(np.float32)

    # keep upper-body + hands + jaw + expr; remove lower-body and shape
    poses = poses[:, (3 + 3 * 11):]
    poses = np.concatenate([poses[:, :-20], poses[:, -10:]], axis=1)  # 133 dim
    return poses


def load_mean_std(mean_path: str, std_path: str):
    mean = torch.load(mean_path, map_location="cpu")
    std = torch.load(std_path, map_location="cpu")
    mean = mean[(3 + 3 * 11):]
    std = std[(3 + 3 * 11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0).float()
    std = torch.cat([std[:-20], std[-10:]], dim=0).float()
    return mean, std


def extract_module_state(state_dict, prefix):
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def remap_legacy_decoder_keys(module, module_state):
    """
    Backward compatibility:
    old checkpoints used decoder.model.<idx>.* (single Sequential),
    while current decoder uses named modules:
      input_proj / upsample_blocks / pre_out / out_proj.
    """
    if not any(k.startswith("decoder.model.") for k in module_state.keys()):
        return module_state
    if not hasattr(module, "decoder") or not hasattr(module.decoder, "upsample_blocks"):
        return module_state

    n_blocks = len(module.decoder.upsample_blocks)
    # Start from non-legacy keys, then append converted decoder keys.
    remapped = OrderedDict(
        (k, v) for k, v in module_state.items() if not k.startswith("decoder.model.")
    )
    target_state = module.state_dict()

    for key, value in module_state.items():
        if not key.startswith("decoder.model."):
            continue
        parts = key.split(".")
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[2])
        except ValueError:
            continue
        suffix = ".".join(parts[3:])
        new_key = None
        if idx == 0:
            new_key = f"decoder.input_proj.{suffix}"
        elif 2 <= idx < 2 + n_blocks:
            new_key = f"decoder.upsample_blocks.{idx - 2}.{suffix}"
        elif idx == 2 + n_blocks:
            new_key = f"decoder.pre_out.{suffix}"
        elif idx == 4 + n_blocks:
            new_key = f"decoder.out_proj.{suffix}"

        if new_key is None or new_key not in target_state:
            continue
        if target_state[new_key].shape != value.shape:
            continue
        # Only fill if the new-format key is absent in source ckpt.
        remapped.setdefault(new_key, value)

    return remapped


def load_module_with_compat(module, module_state, module_name):
    if not module_state:
        raise RuntimeError(f"Empty state_dict for {module_name}.")

    module_state = remap_legacy_decoder_keys(module, module_state)
    load_res = module.load_state_dict(module_state, strict=False)
    missing = list(load_res.missing_keys)
    unexpected = list(load_res.unexpected_keys)

    # Ignore decoder activation placeholders that never have weights.
    benign_missing_prefixes = (
        "decoder.input_act",
        "decoder.pre_out_act",
        "contact_head.",
    )
    missing = [k for k in missing if not any(k.startswith(p) for p in benign_missing_prefixes)]
    benign_unexpected_prefixes = (
        "decoder.model.",
        "contact_head.",
    )
    unexpected = [k for k in unexpected if not any(k.startswith(p) for p in benign_unexpected_prefixes)]

    if missing or unexpected:
        preview_m = ", ".join(missing[:8])
        preview_u = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"Failed to fully load {module_name}. "
            f"missing={len(missing)} [{preview_m}] "
            f"unexpected={len(unexpected)} [{preview_u}]. "
            "This usually means cfg and ckpt architectures do not match."
        )


def build_tokenizers(cfg, ckpt_path, device):
    motion_cfg = OmegaConf.to_container(cfg.model.params.motion_vae, resolve=True)
    vae = instantiate_from_config(motion_cfg).to(device).eval()

    hand_cfg = OmegaConf.select(cfg, "model.params.hand_vae_cfg")
    rhand_cfg = OmegaConf.select(cfg, "model.params.rhand_vae_cfg")
    hand_vae = instantiate_from_config(OmegaConf.to_container(hand_cfg, resolve=True)).to(device).eval() if hand_cfg else None
    rhand_vae = instantiate_from_config(OmegaConf.to_container(rhand_cfg, resolve=True)).to(device).eval() if rhand_cfg else None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    vae_sd = extract_module_state(state_dict, "motion_vae.")
    if not vae_sd:
        vae_sd = extract_module_state(state_dict, "vae.")
    load_module_with_compat(vae, vae_sd, "body_vae")

    if hand_vae is not None:
        hand_sd = extract_module_state(state_dict, "hand_vae.")
        load_module_with_compat(hand_vae, hand_sd, "hand_vae")
    if rhand_vae is not None:
        rhand_sd = extract_module_state(state_dict, "rhand_vae.")
        load_module_with_compat(rhand_vae, rhand_sd, "rhand_vae")

    return vae, hand_vae, rhand_vae


def encode_tokens(feats_norm, vae, hand_vae=None, rhand_vae=None):
    feats = feats_norm
    out = {}
    if hand_vae is None:
        t_body, _ = vae.encode(feats)
        out["body"] = t_body[0]
        tokens_np = t_body.detach().cpu().numpy()
        return out, tokens_np

    if rhand_vae is None:
        feats_hand = feats[..., 30:120]
        feats_re = torch.cat([feats[..., :30], feats[..., 120:]], dim=-1)
        t_body, _ = vae.encode(feats_re)
        t_hand, _ = hand_vae.encode(feats_hand)
        out["body"] = t_body[0]
        out["hand"] = t_hand[0]
        tokens_np = np.stack([t_body.detach().cpu().numpy(), t_hand.detach().cpu().numpy()], axis=-1)
        return out, tokens_np

    feats_lhand = feats[..., 30:75]
    feats_rhand = feats[..., 75:120]
    feats_re = torch.cat([feats[..., :30], feats[..., 120:]], dim=-1)
    t_body, _ = vae.encode(feats_re)
    t_lhand, _ = hand_vae.encode(feats_lhand)
    t_rhand, _ = rhand_vae.encode(feats_rhand)
    out["body"] = t_body[0]
    out["hand"] = t_lhand[0]
    out["rhand"] = t_rhand[0]
    tokens_np = np.stack(
        [
            t_body.detach().cpu().numpy(),
            t_lhand.detach().cpu().numpy(),
            t_rhand.detach().cpu().numpy(),
        ],
        axis=-1,
    )
    return out, tokens_np


def decode_tokens(tokens, vae, hand_vae=None, rhand_vae=None):
    body = vae.decode(tokens["body"]).detach()
    if hand_vae is None:
        return body

    hand = hand_vae.decode(tokens["hand"]).detach()
    tail_dim = body.shape[-1] - 30

    if rhand_vae is None:
        max_len = max(body.shape[1], hand.shape[1])
        final_dim = 30 + hand.shape[-1] + tail_dim
        feats = torch.zeros((1, max_len, final_dim), device=body.device, dtype=body.dtype)
        feats[:, :body.shape[1], :30] = body[:, :, :30]
        feats[:, :hand.shape[1], 30:30 + hand.shape[-1]] = hand
        feats[:, :body.shape[1], -tail_dim:] = body[:, :, 30:]
        return feats

    rhand = rhand_vae.decode(tokens["rhand"]).detach()
    max_len = max(body.shape[1], hand.shape[1], rhand.shape[1])
    final_dim = 30 + hand.shape[-1] + rhand.shape[-1] + tail_dim
    feats = torch.zeros((1, max_len, final_dim), device=body.device, dtype=body.dtype)
    feats[:, :body.shape[1], :30] = body[:, :, :30]
    feats[:, :hand.shape[1], 30:30 + hand.shape[-1]] = hand
    feats[:, :rhand.shape[1], 30 + hand.shape[-1]:30 + hand.shape[-1] + rhand.shape[-1]] = rhand
    feats[:, :body.shape[1], -tail_dim:] = body[:, :, 30:]
    return feats


def feats_to_joints(feats_denorm):
    from mGPT.utils.human_models import get_coord

    # This path requires CUDA because get_coord internally uses a CUDA SMPL-X layer.
    if feats_denorm.device.type != "cuda":
        raise RuntimeError("Visualization requires CUDA device because SMPL-X layer is CUDA-only in current repo.")

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

    _, joints = get_coord(
        root_pose=x[:, 0:3],
        body_pose=x[:, 3:66],
        lhand_pose=x[:, 66:111],
        rhand_pose=x[:, 111:156],
        jaw_pose=x[:, 156:159],
        shape=shape_param,
        expr=x[:, 159:169],
    )
    joints = joints.view(bsz, tlen, joints.shape[1], 3)
    return joints


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

    print(f"[1/5] Loading clip from {pose_dir}")
    feats_raw = load_pose_clip(pose_dir, input_fps=args.input_fps)
    print(f"      Frames: {feats_raw.shape[0]}, Feature dim: {feats_raw.shape[1]}")

    print(f"[2/5] Loading mean/std from {mean_path} and {std_path}")
    mean, std = load_mean_std(mean_path, std_path)
    if feats_raw.shape[1] != mean.numel():
        raise ValueError(f"Feature dim {feats_raw.shape[1]} mismatches mean/std dim {mean.numel()}.")
    feats_norm = (torch.from_numpy(feats_raw).float() - mean[None, :]) / (std[None, :] + 1e-10)
    feats_norm = feats_norm.unsqueeze(0).to(device)

    print(f"[3/5] Loading tokenizer from {ckpt_path}")
    vae, hand_vae, rhand_vae = build_tokenizers(cfg, ckpt_path, device)

    print("[4/5] Tokenizing and reconstructing")
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

    tokens_path = os.path.join(out_dir, f"{sample_name}_tokens.npy")
    recon_norm_path = os.path.join(out_dir, f"{sample_name}_recon_norm.npy")
    recon_path = os.path.join(out_dir, f"{sample_name}_recon.npy")
    np.save(tokens_path, tokens_np)
    np.save(recon_norm_path, feats_recon_norm[0].detach().cpu().numpy())
    np.save(recon_path, feats_recon[0].detach().cpu().numpy())

    print(f"[5/5] Saving outputs to {out_dir}")
    print(f"      token:      {tokens_path}")
    print(f"      recon norm: {recon_norm_path}")
    print(f"      recon:      {recon_path}")

    if not args.no_gif:
        try:
            from mGPT.render.matplot.plot_3d_global import draw_to_batch

            joints = feats_to_joints(feats_recon).detach().cpu().numpy()[0]
            joints_22 = joints[:, :22, :]
            gif_path = os.path.join(out_dir, f"{sample_name}_recon.gif")
            draw_to_batch([joints_22], title_batch=[args.title], outname=[gif_path])
            np.save(os.path.join(out_dir, f"{sample_name}_joints.npy"), joints)
            print(f"      gif:        {gif_path}")
        except Exception as e:
            print(f"[WARN] GIF generation failed: {type(e).__name__}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
