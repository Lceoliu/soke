"""
G1-a: Codebook Utilization Analysis
====================================
Compare Conv1d VAE vs ST-GCN VAE codebook utilization on CSL-Daily train set.

Metrics per VAE (body / lhand / rhand):
  - unique codes used / codebook_size  (utilization rate)
  - token entropy H = -sum(p log p)    (theoretical max = log2(codebook_size))
  - dead codes (never used)
  - top-k concentration (fraction of tokens covered by top-k codes)

Output: experiments/analysis/g1_codebook_utilization/report.txt + summary.json

Usage:
  conda run -n soke python scripts/analysis/g1_codebook_utilization.py \
      --n_samples 500 --device 0
"""

import argparse, json, os, sys, pickle
import numpy as np
import torch
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mGPT.config import instantiate_from_config
from omegaconf import OmegaConf


# ---------- helpers ----------

def load_vae_triple(ckpt_path, body_cfg_name, hand_cfg_name, device):
    """Load body + hand VAE from a checkpoint."""
    body_cfg = OmegaConf.to_container(OmegaConf.load(f'configs/vq/{body_cfg_name}.yaml'), resolve=False)
    hand_cfg = OmegaConf.to_container(OmegaConf.load(f'configs/vq/{hand_cfg_name}.yaml'), resolve=False)
    body_vae = instantiate_from_config(body_cfg)
    lhand_vae = instantiate_from_config(hand_cfg)
    rhand_vae = instantiate_from_config(hand_cfg)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    def load_part(vae, prefix):
        vae_keys = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        missing, unexpected = vae.load_state_dict(vae_keys, strict=False)
        return len(missing), len(unexpected)

    m, u = load_part(body_vae, 'vae.')
    print(f'  body_vae: missing={m}, unexpected={u}')
    m, u = load_part(lhand_vae, 'hand_vae.')
    print(f'  lhand_vae: missing={m}, unexpected={u}')
    m, u = load_part(rhand_vae, 'rhand_vae.')
    print(f'  rhand_vae: missing={m}, unexpected={u}')

    body_vae.eval().to(device)
    lhand_vae.eval().to(device)
    rhand_vae.eval().to(device)
    return body_vae, lhand_vae, rhand_vae


KEYS = ['smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose',
        'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr']


def load_pose_sequence(sample, mean, std):
    """Load CSL-Daily sample → [T, 133] normalized tensor.
    Mirrors load_data.load_csl_sample exactly:
      - Concat all keys → [T, 179]
      - Strip first 3+3*11=36 dims → [T, 143]
      - Remove shape (last 20) + keep last expr 10 → [T, 133]
    """
    name = sample['name']
    pose_dir = Path('data/CSL-Daily/poses') / name
    frames = sorted(pose_dir.glob('*.pkl'))
    if len(frames) < 4:
        return None

    rows = []
    for fp in frames:
        try:
            with open(fp, 'rb') as f:
                d = pickle.load(f)
        except Exception:
            return None
        pose = np.concatenate([np.array(d[k]).flatten() for k in KEYS])  # [179]
        rows.append(pose)

    arr = np.stack(rows, axis=0).astype(np.float32)  # [T, 179]
    arr = arr[:, (3 + 3*11):]                          # [T, 143]
    arr = np.concatenate([arr[:, :-20], arr[:, -10:]], axis=1)  # [T, 133]

    t = torch.from_numpy(arr)
    t = (t - mean) / (std + 1e-8)
    return t


def get_mean_std():
    """Load and slice mean/std to [133] matching the dataset pipeline."""
    mean = torch.load('data/CSL-Daily/mean.pt', map_location='cpu')  # [179]
    std = torch.load('data/CSL-Daily/std.pt', map_location='cpu')
    mean = mean[(3 + 3*11):]                                          # [143]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)                 # [133]
    std = std[(3 + 3*11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean, std


def collect_tokens(vae, sequences, device, part_slice):
    """Encode a list of sequences through vae, return flat token array."""
    all_tokens = []
    with torch.no_grad():
        for seq in sequences:
            part = seq[:, part_slice].unsqueeze(0).to(device)  # [1, T, C]
            tokens, _ = vae.encode(part)   # tokens: list of [1, T', num_q]
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[0]
            tokens = tokens.cpu().numpy().flatten().astype(np.int32)
            all_tokens.append(tokens)
    return np.concatenate(all_tokens)


def analyze_codebook(tokens, codebook_size, name):
    counts = Counter(tokens.tolist())
    n_unique = len(counts)
    total = len(tokens)
    utilization = n_unique / codebook_size

    # Entropy
    probs = np.array(list(counts.values()), dtype=float) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(codebook_size)

    # Dead codes
    dead = codebook_size - n_unique

    # Top-k concentration
    sorted_counts = sorted(counts.values(), reverse=True)
    top10 = sum(sorted_counts[:10]) / total
    top50 = sum(sorted_counts[:50]) / total
    top100 = sum(sorted_counts[:100]) / total

    return {
        'name': name,
        'codebook_size': codebook_size,
        'total_tokens': total,
        'unique_codes': n_unique,
        'dead_codes': dead,
        'utilization': round(utilization, 4),
        'entropy_bits': round(float(entropy), 4),
        'max_entropy_bits': round(float(max_entropy), 4),
        'entropy_ratio': round(float(entropy / max_entropy), 4),
        'top10_concentration': round(float(top10), 4),
        'top50_concentration': round(float(top50), 4),
        'top100_concentration': round(float(top100), 4),
    }


def print_table(results):
    print(f"\n{'Name':<30} {'Util%':>6} {'Entropy':>8} {'MaxEnt':>8} {'Ent%':>6} {'Dead':>6} {'Top10%':>8} {'Top50%':>8}")
    print('-' * 90)
    for r in results:
        print(f"{r['name']:<30} {r['utilization']*100:>5.1f}%"
              f" {r['entropy_bits']:>8.2f} {r['max_entropy_bits']:>8.2f}"
              f" {r['entropy_ratio']*100:>5.1f}%"
              f" {r['dead_codes']:>6}"
              f" {r['top10_concentration']*100:>7.1f}%"
              f" {r['top50_concentration']*100:>7.1f}%")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', default='experiments/analysis/g1_codebook_utilization')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading mean/std...')
    mean, std = get_mean_std()

    print('Loading CSL-Daily train samples...')
    import json as _json
    with open('data/CSL-Daily/csl_clean.train.json') as f:
        train_data = _json.load(f)

    # subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_data), size=min(args.n_samples, len(train_data)), replace=False)
    samples = [train_data[i] for i in idx]

    print(f'Loading pose sequences for {len(samples)} samples...')
    sequences = []
    for s in samples:
        t = load_pose_sequence(s, mean, std)
        if t is not None and t.shape[0] >= 10:
            sequences.append(t)
    print(f'  Loaded: {len(sequences)} valid sequences')

    all_results = []

    # --- Conv1d VAE ---
    CONV1D_CKPT = 'experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt'
    print(f'\n=== Conv1d VAE ===')
    print(f'  ckpt: {CONV1D_CKPT}')
    body_c, lhand_c, rhand_c = load_vae_triple(
        CONV1D_CKPT, 're128_lfq4', 'hand256_lfq4', device)

    body_cb_size = body_c.code_num
    hand_cb_size = lhand_c.code_num
    print(f'  body codebook_size={body_cb_size}, hand codebook_size={hand_cb_size}')

    # Split sequences into per-part inputs (mirrors mgpt._encode_sign_tokens_from_motion)
    body_seqs  = [torch.cat([s[:, :30], s[:, 120:]], dim=-1) for s in sequences]  # [T,43]
    lhand_seqs = [s[:, 30:75]  for s in sequences]   # [T,45]
    rhand_seqs = [s[:, 75:120] for s in sequences]   # [T,45]

    print('  Encoding body...')
    body_tokens_c = collect_tokens(body_c, body_seqs, device, slice(None))
    print('  Encoding lhand...')
    lhand_tokens_c = collect_tokens(lhand_c, lhand_seqs, device, slice(None))
    print('  Encoding rhand...')
    rhand_tokens_c = collect_tokens(rhand_c, rhand_seqs, device, slice(None))

    all_results += [
        analyze_codebook(body_tokens_c, body_cb_size, 'conv1d/body'),
        analyze_codebook(lhand_tokens_c, hand_cb_size, 'conv1d/lhand'),
        analyze_codebook(rhand_tokens_c, hand_cb_size, 'conv1d/rhand'),
    ]

    # --- ST-GCN VAE ---
    STGCN_CKPT = 'experiments/mgpt/debug--VAE_SIGN_FINETUNE_STGCN/checkpoints/last.ckpt'
    print(f'\n=== ST-GCN VAE ===')
    print(f'  ckpt: {STGCN_CKPT}')
    body_s, lhand_s, rhand_s = load_vae_triple(
        STGCN_CKPT, 're128_stgcn', 'hand256_stgcn', device)

    body_cb_size_s = body_s.code_num
    hand_cb_size_s = lhand_s.code_num
    print(f'  body codebook_size={body_cb_size_s}, hand codebook_size={hand_cb_size_s}')

    print('  Encoding body...')
    body_tokens_s = collect_tokens(body_s, body_seqs, device, slice(None))
    print('  Encoding lhand...')
    lhand_tokens_s = collect_tokens(lhand_s, lhand_seqs, device, slice(None))
    print('  Encoding rhand...')
    rhand_tokens_s = collect_tokens(rhand_s, rhand_seqs, device, slice(None))

    all_results += [
        analyze_codebook(body_tokens_s, body_cb_size_s, 'stgcn/body'),
        analyze_codebook(lhand_tokens_s, hand_cb_size_s, 'stgcn/lhand'),
        analyze_codebook(rhand_tokens_s, hand_cb_size_s, 'stgcn/rhand'),
    ]

    # --- Print and Save ---
    print('\n' + '='*90)
    print('CODEBOOK UTILIZATION REPORT')
    print('='*90)
    print_table(all_results)

    # Interpretation
    print('\n--- Key Comparisons ---')
    for part in ['body', 'lhand', 'rhand']:
        c = next(r for r in all_results if r['name'] == f'conv1d/{part}')
        s = next(r for r in all_results if r['name'] == f'stgcn/{part}')
        print(f'\n{part}:')
        print(f'  Conv1d: util={c["utilization"]*100:.1f}%, entropy={c["entropy_bits"]:.2f}b '
              f'({c["entropy_ratio"]*100:.1f}% of max), dead={c["dead_codes"]}')
        print(f'  ST-GCN: util={s["utilization"]*100:.1f}%, entropy={s["entropy_bits"]:.2f}b '
              f'({s["entropy_ratio"]*100:.1f}% of max), dead={s["dead_codes"]}')
        ent_delta = s['entropy_bits'] - c['entropy_bits']
        util_delta = s['utilization'] - c['utilization']
        print(f'  Δentropy={ent_delta:+.2f}b, Δutil={util_delta*100:+.1f}%')

    report_path = os.path.join(args.output_dir, 'report.txt')
    json_path = os.path.join(args.output_dir, 'summary.json')

    import io
    buf = io.StringIO()
    print('='*90, file=buf)
    print('G1-a: Codebook Utilization Analysis', file=buf)
    print(f'n_samples={len(sequences)}, device={device}', file=buf)
    print('='*90, file=buf)
    for r in all_results:
        for k, v in r.items():
            print(f'  {k}: {v}', file=buf)
        print('', file=buf)

    with open(report_path, 'w') as f:
        f.write(buf.getvalue())
    with open(json_path, 'w') as f:
        json.dump({'results': all_results, 'n_samples': len(sequences)}, f, indent=2)

    print(f'\nSaved: {report_path}')
    print(f'Saved: {json_path}')


if __name__ == '__main__':
    main()
