from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def load_ann(path: Path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def save_ann(path: Path, data):
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def symlink_or_replace(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f'{dst} exists and is a real directory; refuse to overwrite')
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_root', default='data/CSL-Daily')
    ap.add_argument('--dst_root', default='data/CSL-Daily-overfit12')
    ap.add_argument('--num_samples', type=int, default=12)
    ap.add_argument('--split', default='train')
    ap.add_argument('--signer', default='P0000')
    ap.add_argument('--unique_text', action='store_true')
    ap.add_argument('--same_for_all_splits', action='store_true', default=True)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    ann = load_ann(src_root / f'csl_clean.{args.split}')

    selected: List[Dict] = []
    seen_text = set()
    for item in ann:
        name = str(item['name'])
        text = str(item.get('text', ''))
        if args.signer and f'_{args.signer}_' not in name:
            continue
        pose_dir = src_root / 'poses' / name
        if not pose_dir.exists():
            continue
        if args.unique_text and text in seen_text:
            continue
        selected.append(item)
        seen_text.add(text)
        if len(selected) >= args.num_samples:
            break

    if len(selected) < args.num_samples:
        raise RuntimeError(f'Only found {len(selected)} samples, fewer than requested {args.num_samples}.')

    dst_root.mkdir(parents=True, exist_ok=True)
    symlink_or_replace(src_root / 'poses', dst_root / 'poses')
    for fname in ['mean.pt', 'std.pt']:
        src = src_root / fname
        if src.exists():
            symlink_or_replace(src, dst_root / fname)

    if args.same_for_all_splits:
        for split in ['train', 'val', 'test']:
            save_ann(dst_root / f'csl_clean.{split}', selected)
    else:
        save_ann(dst_root / 'csl_clean.train', selected)
        save_ann(dst_root / 'csl_clean.val', selected)
        save_ann(dst_root / 'csl_clean.test', selected)

    names = [x['name'] for x in selected]
    texts = [x.get('text', '') for x in selected]
    with open(dst_root / 'selected_samples.txt', 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')
    with open(dst_root / 'selected_samples.json', 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(selected), f, ensure_ascii=False, indent=2)

    print(f'[done] wrote {len(selected)} samples to {dst_root}')
    for name, text in zip(names, texts):
        print(f'{name}\t{text}')


if __name__ == '__main__':
    main()
