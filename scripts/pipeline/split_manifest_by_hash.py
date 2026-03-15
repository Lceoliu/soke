#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def stable_bucket(key: str, mod: int = 1000) -> int:
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % mod


def assign_split(name: str, train_ratio: float, val_ratio: float) -> str:
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1
    x = stable_bucket(name) / 1000.0
    if x < train_ratio:
        return 'train'
    if x < train_ratio + val_ratio:
        return 'val'
    return 'test'


def main():
    p = argparse.ArgumentParser(description='Deterministically split manifest into train/val/test by hash.')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-ratio', type=float, default=0.98)
    p.add_argument('--val-ratio', type=float, default=0.01)
    p.add_argument('--key-field', default='name', help='field used for hashing; fallback to path')
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n = 0
    c = {'train': 0, 'val': 0, 'test': 0}
    with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            key = str(obj.get(args.key_field) or obj.get('path'))
            obj['split'] = assign_split(key, args.train_ratio, args.val_ratio)
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
            c[obj['split']] += 1
            n += 1

    print(f'total={n}, train={c["train"]}, val={c["val"]}, test={c["test"]}')


if __name__ == '__main__':
    main()
