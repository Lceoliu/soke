import os
import sys
import json
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained_vae


def flatten_token_levels(token_tensor, q_keep=None):
    # token_tensor: [B,T] or [B,T,Q]
    if token_tensor.dim() == 3:
        q = token_tensor.shape[-1]
        if q_keep is not None:
            q = min(int(q_keep), int(token_tensor.shape[-1]))
            token_tensor = token_tensor[..., :q]
        token_tensor = token_tensor.reshape(token_tensor.shape[0], -1)
        return token_tensor, int(q)
    if token_tensor.dim() == 2:
        return token_tensor, 1
    token_tensor = token_tensor.reshape(token_tensor.shape[0], -1)
    return token_tensor, 1


def build_token_cache_meta(cfg):
    train_cfg = cfg.TRAIN
    model_params = cfg.model.params
    return {
        "pretrained_vae": str(train_cfg.get("PRETRAINED_VAE", "") or ""),
        "pretrained_vae_body": str(train_cfg.get("PRETRAINED_VAE_BODY", "") or ""),
        "pretrained_vae_hand": str(train_cfg.get("PRETRAINED_VAE_HAND", "") or ""),
        "pretrained_vae_rhand": str(train_cfg.get("PRETRAINED_VAE_RHAND", "") or ""),
        "dataset_name": str(cfg.DATASET.H2S.get("DATASET_NAME", "") or ""),
        "code_path": str(cfg.DATASET.get("CODE_PATH", "") or ""),
        "motion_vae": str(model_params.motion_vae),
        "hand_vae_cfg": str(model_params.get("hand_vae_cfg", None)),
        "rhand_vae_cfg": str(model_params.get("rhand_vae_cfg", None)),
    }


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.USE_GPUS
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = build_data(cfg, phase='token')
    print("datasets module initialized")
    output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "_tokenizer_meta.json")
    cache_meta = build_token_cache_meta(cfg)

    # create model
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model loaded")

    # Strict load vae model
    assert cfg.TRAIN.PRETRAINED_VAE is not None
    load_pretrained_vae(cfg, model)

    if cfg.ACCELERATOR == "gpu":
        model = model.to('cuda')

    def encode_pose_batch(pose: torch.Tensor):
        if hasattr(model, 'hand_vae') and hasattr(model, 'rhand_vae'):
            pose_lhand = pose[..., 30:75]
            pose_rhand = pose[..., 75:120]
            pose_re = torch.cat([pose[..., :30], pose[..., 120:]], dim=-1)
            target_re, _ = model.vae.encode(pose_re)
            target_lhand, _ = model.hand_vae.encode(pose_lhand)
            target_rhand, _ = model.rhand_vae.encode(pose_rhand)
            q_re = target_re.shape[-1] if target_re.dim() == 3 else 1
            q_lhand = target_lhand.shape[-1] if target_lhand.dim() == 3 else 1
            q_rhand = target_rhand.shape[-1] if target_rhand.dim() == 3 else 1
            q_shared = min(q_re, q_lhand, q_rhand)
            target_re, _ = flatten_token_levels(target_re, q_keep=q_shared)
            target_lhand, _ = flatten_token_levels(target_lhand, q_keep=q_shared)
            target_rhand, _ = flatten_token_levels(target_rhand, q_keep=q_shared)
            min_len = min(target_re.shape[1], target_lhand.shape[1], target_rhand.shape[1])
            return np.stack(
                [
                    target_re[:, :min_len].to('cpu').numpy(),
                    target_lhand[:, :min_len].to('cpu').numpy(),
                    target_rhand[:, :min_len].to('cpu').numpy(),
                ],
                axis=-1,
            )
        else:
            if hasattr(model, 'hand_vae'):
                pose_hand = pose[..., 30:120]
                pose_re = torch.cat([pose[..., :30], pose[..., 120:]], dim=-1)
                target_hand, _ = model.hand_vae.encode(pose_hand)
                target_re, _ = model.vae.encode(pose_re)
                q_re = target_re.shape[-1] if target_re.dim() == 3 else 1
                q_hand = target_hand.shape[-1] if target_hand.dim() == 3 else 1
                q_shared = min(q_re, q_hand)
                target_re, _ = flatten_token_levels(target_re, q_keep=q_shared)
                target_hand, _ = flatten_token_levels(target_hand, q_keep=q_shared)
                min_len = min(target_re.shape[1], target_hand.shape[1])
                return np.stack([target_re[:, :min_len].to('cpu').numpy(), target_hand[:, :min_len].to('cpu').numpy()], axis=-1)
            else:
                target, _ = model.vae.encode(pose)
                target, _ = flatten_token_levels(target)
                return target.to('cpu').numpy()

    skip_existing = os.environ.get("SKIP_EXISTING_TOKENS", "1") == "1"
    overwrite_meta = os.environ.get("OVERWRITE_TOKEN_META", "1") == "1"
    num_done = 0
    num_skipped_existing = 0
    datasets.setup(None)
    split_loaders = [
        ("train", datasets.train_dataloader()),
        ("val", datasets.val_dataloader()),
        ("test", datasets.test_dataloader()),
    ]
    for split_name, loader in split_loaders:
        for batch in tqdm(loader, desc=f'motion tokenize ({split_name})'):
            name = batch['text']
            src = batch['src'][0]
            output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH, src)
            target_path = os.path.join(output_dir, name[0] + '.npy')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            if skip_existing and os.path.exists(target_path):
                num_skipped_existing += 1
                continue

            pose = batch['motion']
            pose = pose.cuda().float()
            if pose.shape[1] == 0:
                continue

            target = encode_pose_batch(pose)
            np.save(target_path, target)
            num_done += 1

    if overwrite_meta:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(cache_meta, f, indent=2, ensure_ascii=True)

    print(
        f"Motion tokenization done. saved={num_done}, skipped_existing={num_skipped_existing}, "
        f"tokens are under {output_dir}"
    )


if __name__ == "__main__":
    main()
