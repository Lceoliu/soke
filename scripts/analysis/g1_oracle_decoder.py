"""
G1-c: Oracle (Nearest-Neighbor) Decoder
=========================================
Upper-bound estimate of what the tokenizer can support for translation.

For each test sample:
  1. Encode sign sequence → token sequence (body+lhand+rhand)
  2. Find nearest-neighbor in the TRAIN set by L1/cosine distance on token sequence
  3. Use that train sample's ground-truth text as the prediction

Compute BLEU4 over the test set.

If oracle BLEU4 < 5 → tokenizer space is too sparse / non-discriminative → tokenizer is the bottleneck.
If oracle BLEU4 ≥ 10 → tokenizer has the information; LM is not using it → training is the bottleneck.

Output: experiments/analysis/g1_oracle_decoder/report.txt + summary.json

Usage:
  conda run -n soke python scripts/analysis/g1_oracle_decoder.py \
      --n_train 2000 --n_test 500 --device 0
"""

import argparse, json, os, sys, pickle
import numpy as np
import torch
from pathlib import Path
from sacrebleu.metrics import BLEU

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from mGPT.config import instantiate_from_config
from omegaconf import OmegaConf


# -------- helpers (shared with g1_codebook_utilization) --------

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


KEYS = ['smplx_root_pose', 'smplx_body_pose', 'smplx_lhand_pose',
        'smplx_rhand_pose', 'smplx_jaw_pose', 'smplx_shape', 'smplx_expr']


def get_mean_std():
    mean = torch.load('data/CSL-Daily/mean.pt', map_location='cpu')
    std = torch.load('data/CSL-Daily/std.pt', map_location='cpu')
    mean = mean[(3 + 3*11):]
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    std = std[(3 + 3*11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
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


def encode_to_token_vec(seq, body_vae, lhand_vae, rhand_vae, device):
    """Encode a sequence to a fixed-length token histogram vector for NN search."""
    body_seq = torch.cat([seq[:, :30], seq[:, 120:]], dim=-1).unsqueeze(0).to(device)
    lhand_seq = seq[:, 30:75].unsqueeze(0).to(device)
    rhand_seq = seq[:, 75:120].unsqueeze(0).to(device)
    with torch.no_grad():
        body_tok, _ = body_vae.encode(body_seq)
        lhand_tok, _ = lhand_vae.encode(lhand_seq)
        rhand_tok, _ = rhand_vae.encode(rhand_seq)
    body_tok = body_tok[0].cpu().numpy().flatten()
    lhand_tok = lhand_tok[0].cpu().numpy().flatten()
    rhand_tok = rhand_tok[0].cpu().numpy().flatten()
    # Return concatenated token sequence (variable length → we'll pad/truncate for distance)
    return np.concatenate([body_tok, lhand_tok, rhand_tok]).astype(np.int32)


def token_seq_to_histogram(tokens, vocab_size):
    """Convert a token sequence to a normalized histogram."""
    hist = np.bincount(tokens, minlength=vocab_size).astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def run_oracle(label, body_vae, lhand_vae, rhand_vae,
               train_samples, test_samples, mean, std, device,
               body_cb_size, hand_cb_size):

    total_vocab = body_cb_size + hand_cb_size * 2

    print(f'\n[{label}] Building train token histograms ({len(train_samples)} samples)...')
    train_hists = []
    train_texts = []
    skipped = 0
    for s in train_samples:
        seq = load_pose_sequence(s, mean, std)
        if seq is None or seq.shape[0] < 10:
            skipped += 1
            continue
        tokens = encode_to_token_vec(seq, body_vae, lhand_vae, rhand_vae, device)
        # offset hand tokens to avoid collision
        body_len = len(tokens) // 3
        lhand_tokens = tokens[body_len:2*body_len] + body_cb_size
        rhand_tokens = tokens[2*body_len:] + body_cb_size + hand_cb_size
        combined = np.concatenate([tokens[:body_len], lhand_tokens, rhand_tokens])
        hist = token_seq_to_histogram(combined, total_vocab)
        train_hists.append(hist)
        train_texts.append(s['text'])
    print(f'  Built {len(train_hists)} histograms (skipped {skipped})')

    train_mat = np.stack(train_hists, axis=0)  # [N_train, vocab]

    print(f'[{label}] Evaluating test set ({len(test_samples)} samples)...')
    predictions = []
    references = []
    skipped_test = 0
    for s in test_samples:
        seq = load_pose_sequence(s, mean, std)
        if seq is None or seq.shape[0] < 10:
            skipped_test += 1
            continue
        tokens = encode_to_token_vec(seq, body_vae, lhand_vae, rhand_vae, device)
        body_len = len(tokens) // 3
        lhand_tokens = tokens[body_len:2*body_len] + body_cb_size
        rhand_tokens = tokens[2*body_len:] + body_cb_size + hand_cb_size
        combined = np.concatenate([tokens[:body_len], lhand_tokens, rhand_tokens])
        test_hist = token_seq_to_histogram(combined, total_vocab)

        # L1 distance
        dists = np.abs(train_mat - test_hist).sum(axis=1)
        nn_idx = np.argmin(dists)
        predictions.append(train_texts[nn_idx])
        references.append(s['text'])

    print(f'  Evaluated {len(predictions)} samples (skipped {skipped_test})')

    # BLEU
    bleu = BLEU(tokenize='char')
    result = bleu.corpus_score(predictions, [references])
    bleu4 = float(result.score)
    bleu1 = float(BLEU(tokenize='char', max_ngram_order=1).corpus_score(predictions, [references]).score)

    # Self-retrieval rate (exact match with original)
    exact = sum(p == r for p, r in zip(predictions, references))
    exact_rate = exact / len(predictions)

    print(f'  [{label}] Oracle BLEU4={bleu4:.3f}, BLEU1={bleu1:.3f}, exact={exact_rate*100:.1f}%')

    return {
        'label': label,
        'n_train': len(train_hists),
        'n_test': len(predictions),
        'oracle_BLEU4': round(bleu4, 3),
        'oracle_BLEU1': round(bleu1, 3),
        'exact_retrieval_rate': round(exact_rate, 4),
        'example_preds': list(zip(predictions[:5], references[:5])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=2000,
                        help='Max train samples to index')
    parser.add_argument('--n_test', type=int, default=500,
                        help='Max test samples to evaluate')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', default='experiments/analysis/g1_oracle_decoder')
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

    # Conv1d
    CONV1D_CKPT = 'experiments/mgpt/VAE_SIGN_FINETUNE_LFQ4_ACC/checkpoints/last.ckpt'
    print('\n=== Loading Conv1d VAE ===')
    body_c, lhand_c, rhand_c = load_vae_triple(CONV1D_CKPT, 're128_lfq4', 'hand256_lfq4', device)
    r = run_oracle('conv1d', body_c, lhand_c, rhand_c,
                   train_samples, test_samples, mean, std, device,
                   body_c.code_num, lhand_c.code_num)
    all_results.append(r)
    del body_c, lhand_c, rhand_c
    torch.cuda.empty_cache()

    # ST-GCN
    STGCN_CKPT = 'experiments/mgpt/debug--VAE_SIGN_FINETUNE_STGCN/checkpoints/last.ckpt'
    print('\n=== Loading ST-GCN VAE ===')
    body_s, lhand_s, rhand_s = load_vae_triple(STGCN_CKPT, 're128_stgcn', 'hand256_stgcn', device)
    r = run_oracle('stgcn', body_s, lhand_s, rhand_s,
                   train_samples, test_samples, mean, std, device,
                   body_s.code_num, lhand_s.code_num)
    all_results.append(r)

    # Report
    print('\n' + '='*70)
    print('G1-c ORACLE DECODER SUMMARY')
    print('='*70)
    for r in all_results:
        verdict = (
            'TOKENIZER IS BOTTLENECK (oracle < 5)'
            if r['oracle_BLEU4'] < 5.0 else
            'TOKENIZER OK — LM is bottleneck (oracle >= 10)'
            if r['oracle_BLEU4'] >= 10.0 else
            'TOKENIZER BORDERLINE (5 ≤ oracle < 10)'
        )
        print(f"  [{r['label']}] oracle_BLEU4={r['oracle_BLEU4']:.3f}  → {verdict}")

    print('\nExample predictions (pred | ref):')
    for r in all_results:
        print(f"  [{r['label']}]")
        for pred, ref in r.get('example_preds', [])[:3]:
            print(f"    pred: {pred}")
            print(f"    ref:  {ref}")
            print()

    # Save
    json_path = os.path.join(args.output_dir, 'summary.json')
    report_path = os.path.join(args.output_dir, 'report.txt')

    with open(json_path, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    lines = ['G1-c Oracle Decoder Report\n', '='*70 + '\n']
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
