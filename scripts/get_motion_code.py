import os
import sys
import numpy as np
import pytorch_lightning as pl
import torch, json
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

    save_data = {}
    for batch in tqdm(datasets.train_dataloader(),
                      desc=f'motion tokenize'):
        name = batch['text']
        
        pose = batch['motion']
        src = batch['src'][0]
        pose = pose.cuda().float()

        if pose.shape[1] == 0:
            continue
        
        output_dir = os.path.join(datasets.hparams.data_root, cfg.DATASET.CODE_PATH, src)
        target_path = os.path.join(output_dir, name[0] + '.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
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
            target = np.stack(
                [
                    target_re[:, :min_len].to('cpu').numpy(),
                    target_lhand[:, :min_len].to('cpu').numpy(),
                    target_rhand[:, :min_len].to('cpu').numpy(),
                ],
                axis=-1,
            )
            # save_data[name[0]] = {'body': target_re.to('cpu').numpy()[0].tolist(), 
            #                       'lhand': target_lhand.to('cpu').numpy()[0].tolist(), 
            #                       'rhand': target_rhand.to('cpu').numpy()[0].tolist()}
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
                target = np.stack([target_re[:, :min_len].to('cpu').numpy(), target_hand[:, :min_len].to('cpu').numpy()], axis=-1)
            else:
                target, _ = model.vae.encode(pose)
                target, _ = flatten_token_levels(target)
                target = target.to('cpu').numpy()

        np.save(target_path, target)

    # with open('./iso_motion_code.json', 'w') as f:
    #     json.dump(save_data, f)

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
