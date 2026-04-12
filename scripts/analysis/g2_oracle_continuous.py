"""
R025 — G11 Oracle Decoder (Continuous Embeddings)
===================================================
Upper-bound estimate of mT5's actual input space for sign-to-text translation.

mT5 uses VAE.encode_continuous() (pre-quantization continuous latents),
NOT discrete LFQ tokens. This script measures the correct oracle:

For each test sample:
  1. Encode sign → encode_continuous() → [T', 512] per part → concat [T', 1536]
  2. Mean-pool over time → [1536] fixed-length vector
  3. L2-normalize → unit sphere
  4. Find cosine nearest-neighbor in TRAIN set
  5. Use that train text as prediction

Compute BLEU4 over test set.

Interpretation:
  oracle BLEU4 << mT5 val BLEU4 (~1.7): embeddings can't discriminate sentences
  oracle BLEU4 >> mT5 val BLEU4: embedding space is fine, LM training is the limit
  oracle BLEU4 ≈ mT5 val BLEU4: marginally useful signal

Output: experiments/analysis/g2_oracle_continuous/report.txt + summary.json

Usage:
  conda run -n soke python scripts/analysis/g2_oracle_continuous.py \\
      --n_train 2000 --n_test 500 --device 0
"""

import argparse, json, os, sys, pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sacrebleu.metrics import BLEU

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mGPT.config import instantiate_from_config
from omegaconf import OmegaConf


# ─────────────────────────────────────────────────────────────
# VAE loading
# ─────────────────────────────────────────────────────────────

def load_vae_triple(ckpt_path, body_cfg_name, hand_cfg_name, device):
    body_cfg = OmegaConf.to_container(OmegaConf.load(f'configs/vq/{body_cfg_name}.yaml'), resolve=False)
    hand_cfg = OmegaConf.to_container(OmegaConf.load(f'configs/vq/{hand_cfg_name}.yaml'), resolve=False)
    body_vae = instantiate_from_config(body_cfg)
    lhand_vae = instantiate_from_config(hand_cfg)
    rhand_vae = instantiate_from_config(hand_cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    def load_part(vae, prefix):
        vae_keys = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        vae.load_state_dict(vae_keys, strict=False)
    load_part(body_vae, 'vae.')
    load_part(lhand_vae, 'hand_vae.')
    load_part(rhand_vae, 'rhand_vae.')
    body_vae.eval().to(device)
    lhand_vae.eval().to(device)
    rhand_vae.eval().to(device)
    return body_vae, lhand_vae, rhand_vae


# ─────────────────────────────────────────────────────────────
# Pose loading (same slice as mT5 pipeline)
# ─────────────────────────────────────────────────────────────

KEYS = ['smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose',
        'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr']


def get_mean_std():
    mean = torch.load('data/CSL-Daily/mean.pt', map_location='cpu')
    std  = torch.load('data/CSL-Daily/std.pt',  map_location='cpu')
    mean = mean[(3 + 3*11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std  = std[(3 + 3*11):]
    std  = torch.cat([std[:-20], std[-10:]], dim=0)
    return mean, std


def load_pose_sequence(sample, mean, std):
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
        pose = np.concatenate([np.array(d[k]).flatten() for k in KEYS])
        rows.append(pose)
    arr = np.stack(rows, axis=0).astype(np.float32)
    arr = arr[:, (3 + 3*11):]
    arr = np.concatenate([arr[:, :-20], arr[:, -10:]], axis=1)
    t = torch.from_numpy(arr)
    t = (t - mean) / (std + 1e-8)
    return t


# ─────────────────────────────────────────────────────────────
# Continuous embedding → mean-pooled unit vector
# ─────────────────────────────────────────────────────────────

def encode_to_continuous_vec(seq, body_vae, lhand_vae, rhand_vae, device):
    """
    Returns: L2-normalised mean-pooled continuous embedding [1536].
    Matches the mT5 pipeline exactly (encode_continuous, concat body+lhand+rhand).
    """
    body_in  = torch.cat([seq[:, :30], seq[:, 120:]], dim=-1).unsqueeze(0).to(device)
    lhand_in = seq[:, 30:75].unsqueeze(0).to(device)
    rhand_in = seq[:, 75:120].unsqueeze(0).to(device)

    with torch.no_grad():
        eb = body_vae.encode_continuous(body_in)    # [1, T', 512]
        el = lhand_vae.encode_continuous(lhand_in)  # [1, T', 512]
        er = rhand_vae.encode_continuous(rhand_in)  # [1, T', 512]

    mt = min(eb.shape[1], el.shape[1], er.shape[1])
    emb = torch.cat([eb[:, :mt], el[:, :mt], er[:, :mt]], dim=-1)  # [1, T', 1536]
    vec = emb[0].mean(dim=0)                                         # [1536]
    vec = F.normalize(vec, dim=0)
    return vec.cpu().numpy()


# ─────────────────────────────────────────────────────────────
# Oracle evaluation
# ─────────────────────────────────────────────────────────────

def run_oracle(label, body_vae, lhand_vae, rhand_vae,
               train_samples, test_samples, mean, std, device):

    print(f'\n[{label}] Building train continuous embeddings ({len(train_samples)} samples)...')
    train_vecs  = []
    train_texts = []
    skipped = 0
    for s in train_samples:
        seq = load_pose_sequence(s, mean, std)
        if seq is None or seq.shape[0] < 10:
            skipped += 1
            continue
        vec = encode_to_continuous_vec(seq, body_vae, lhand_vae, rhand_vae, device)
        train_vecs.append(vec)
        train_texts.append(s['text'])
    print(f'  Built {len(train_vecs)} embeddings (skipped {skipped})')

    train_mat = np.stack(train_vecs, axis=0)  # [N_train, 1536], already L2-normalised

    print(f'[{label}] Evaluating test set ({len(test_samples)} samples)...')
    predictions  = []
    references   = []
    nn_sims      = []
    skipped_test = 0
    for s in test_samples:
        seq = load_pose_sequence(s, mean, std)
        if seq is None or seq.shape[0] < 10:
            skipped_test += 1
            continue
        vec = encode_to_continuous_vec(seq, body_vae, lhand_vae, rhand_vae, device)

        # Cosine similarity (both sides L2-normalised → dot product)
        sims   = train_mat @ vec          # [N_train]
        nn_idx = int(np.argmax(sims))
        nn_sims.append(float(sims[nn_idx]))
        predictions.append(train_texts[nn_idx])
        references.append(s['text'])

    print(f'  Evaluated {len(predictions)} samples (skipped {skipped_test})')

    # BLEU
    bleu  = BLEU(tokenize='char')
    bleu1 = BLEU(tokenize='char', max_ngram_order=1)
    res4 = bleu.corpus_score(predictions, [references])
    res1 = bleu1.corpus_score(predictions, [references])
    oracle_b4 = float(res4.score)
    oracle_b1 = float(res1.score)

    exact      = sum(p == r for p, r in zip(predictions, references))
    exact_rate = exact / max(len(predictions), 1)
    mean_sim   = float(np.mean(nn_sims))
    p90_sim    = float(np.percentile(nn_sims, 90))

    print(f'  [{label}] Oracle BLEU4={oracle_b4:.3f}, BLEU1={oracle_b1:.3f}, '
          f'exact={exact_rate*100:.1f}%, mean_nn_cosine={mean_sim:.4f}')

    return {
        'label':                label,
        'n_train':              len(train_vecs),
        'n_test':               len(predictions),
        'oracle_BLEU4':         round(oracle_b4, 3),
        'oracle_BLEU1':         round(oracle_b1, 3),
        'exact_retrieval_rate': round(exact_rate, 4),
        'mean_nn_cosine_sim':   round(mean_sim, 4),
        'p90_nn_cosine_sim':    round(p90_sim, 4),
        'example_preds':        list(zip(predictions[:5], references[:5])),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=2000,
                        help='Max train samples to index')
    parser.add_argument('--n_test', type=int, default=500,
                        help='Max test samples to evaluate')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', default='experiments/analysis/g2_oracle_continuous')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading mean/std...')
    mean, std = get_mean_std()

    print('Loading CSL-Daily data...')
    with open('data/CSL-Daily/csl_clean.train.json') as f:
        train_data = json.load(f)
    with open('data/CSL-Daily/csl_clean.test.json') as f:
        test_data = json.load(f)

    rng = np.random.default_rng(42)
    train_samples = [train_data[i] for i in rng.choice(
        len(train_data), size=min(args.n_train, len(train_data)), replace=False)]
    test_samples = [test_data[i] for i in rng.choice(
        len(test_data), size=min(args.n_test, len(test_data)), replace=False)]

    all_results = []

    # ── Conv1d VAE (the one mT5 actually uses in R003/R006) ──
    CONV1D_CKPT = 'experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt'
    print('\n=== Loading Conv1d VAE ===')
    body_c, lhand_c, rhand_c = load_vae_triple(
        CONV1D_CKPT, 're128_lfq4', 'hand256_lfq4', device)
    r = run_oracle('conv1d_continuous', body_c, lhand_c, rhand_c,
                   train_samples, test_samples, mean, std, device)
    all_results.append(r)
    del body_c, lhand_c, rhand_c
    torch.cuda.empty_cache()

    # ── ST-GCN VAE (for comparison) ──
    STGCN_CKPT = 'experiments/mgpt/debug--VAE_SIGN_FINETUNE_STGCN/checkpoints/last.ckpt'
    print('\n=== Loading ST-GCN VAE ===')
    body_s, lhand_s, rhand_s = load_vae_triple(
        STGCN_CKPT, 're128_stgcn', 'hand256_stgcn', device)
    r = run_oracle('stgcn_continuous', body_s, lhand_s, rhand_s,
                   train_samples, test_samples, mean, std, device)
    all_results.append(r)
    del body_s, lhand_s, rhand_s
    torch.cuda.empty_cache()

    # ─── Summary ───────────────────────────────────────────────
    print('\n' + '='*70)
    print('R025 — CONTINUOUS EMBEDDING ORACLE SUMMARY')
    print('Reference: mT5 baseline (R006) achieved BLEU4=1.742')
    print('='*70)
    for r in all_results:
        oracle = r['oracle_BLEU4']
        mt5    = 1.742
        ratio  = oracle / mt5 if mt5 > 0 else float('inf')
        if oracle < mt5 * 0.5:
            verdict = f'EMBEDDING CEILING BELOW mT5 ({oracle:.3f} < {mt5:.3f}) — retrieval fails worse than LM'
        elif oracle < mt5 * 2.0:
            verdict = f'EMBEDDING CEILING ≈ mT5 ({oracle:.3f} ≈ {mt5:.3f}) — embeddings marginally better than random'
        else:
            verdict = f'EMBEDDING CEILING > mT5 ({oracle:.3f} >> {mt5:.3f}) — rich information; LM training is the bottleneck'
        print(f'  [{r["label"]}] oracle_BLEU4={oracle:.3f}  → {verdict}')
        print(f'    mean NN cosine={r["mean_nn_cosine_sim"]:.4f}, p90={r["p90_nn_cosine_sim"]:.4f}')

    print('\nExample predictions (pred | ref):')
    for r in all_results:
        print(f"  [{r['label']}]")
        for pred, ref in r.get('example_preds', [])[:3]:
            print(f"    pred: {pred}")
            print(f"    ref:  {ref}")
            print()

    # Save
    json_path   = os.path.join(args.output_dir, 'summary.json')
    report_path = os.path.join(args.output_dir, 'report.txt')

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    lines = [
        'R025 — Continuous Embedding Oracle Report\n',
        '(mT5 baseline BLEU4 reference: 1.742)\n',
        '='*70 + '\n'
    ]
    for r in all_results:
        lines.append(f"[{r['label']}]\n")
        for k, v in r.items():
            if k != 'example_preds':
                lines.append(f"  {k}: {v}\n")
        lines.append('\n')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f'\nSaved: {json_path}')
    print(f'Saved: {report_path}')


if __name__ == '__main__':
    main()
