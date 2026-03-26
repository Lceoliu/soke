#!/usr/bin/env python3
"""Sign token statistical analysis.

Produces:
1. Per-part token frequency distribution (zipf-like analysis)
2. Adjacent-frame same-token repetition rate per part
3. N-gram frequency and BPE compression feasibility study

Usage:
    python scripts/analyze_sign_tokens.py [--token_dir data/TOKENS_h2s_csl_phoenix_backup_20260211]
"""
import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import islice

import numpy as np

PART_NAMES = ["body", "lhand", "rhand"]


# ─── Data Loading ────────────────────────────────────────────────────

def load_all_tokens(token_dir: str, max_files: int = 0):
    """Load all token .npy files, return per-part token sequences.

    Returns:
        sequences: dict[part_name] -> list of 1-D numpy arrays (per-sample token sequence)
        all_tokens: dict[part_name] -> 1-D numpy array (concatenated)
    """
    sequences = {p: [] for p in PART_NAMES}

    subdirs = sorted(d for d in os.listdir(token_dir) if os.path.isdir(os.path.join(token_dir, d)))
    file_count = 0
    for subdir in subdirs:
        dirpath = os.path.join(token_dir, subdir)
        files = sorted(f for f in os.listdir(dirpath) if f.endswith(".npy"))
        for fname in files:
            if 0 < max_files <= file_count:
                break
            fpath = os.path.join(dirpath, fname)
            try:
                arr = np.load(fpath)
            except Exception:
                continue
            # Expected shape: [1, T, P] or [T, P] or [1, T]
            if arr.ndim == 3:
                arr = arr[0]  # [T, P]
            if arr.ndim == 1:
                arr = arr[:, None]  # [T, 1]

            num_parts = min(arr.shape[1], len(PART_NAMES))
            for p_idx in range(num_parts):
                sequences[PART_NAMES[p_idx]].append(arr[:, p_idx].astype(np.int32))
            file_count += 1
        if 0 < max_files <= file_count:
            break

    all_tokens = {}
    for p in PART_NAMES:
        if sequences[p]:
            all_tokens[p] = np.concatenate(sequences[p])
        else:
            all_tokens[p] = np.array([], dtype=np.int32)

    print(f"Loaded {file_count} files from {token_dir}")
    for p in PART_NAMES:
        print(f"  {p}: {len(sequences[p])} sequences, {len(all_tokens[p]):,} tokens total")
    return sequences, all_tokens


# ─── 1. Frequency Analysis ──────────────────────────────────────────

def frequency_analysis(all_tokens: dict):
    print("\n" + "=" * 60)
    print("1. TOKEN FREQUENCY ANALYSIS")
    print("=" * 60)

    for part in PART_NAMES:
        tokens = all_tokens[part]
        if len(tokens) == 0:
            continue
        counter = Counter(tokens.tolist())
        total = len(tokens)
        vocab_size = len(counter)

        # Sort by frequency
        sorted_counts = counter.most_common()

        print(f"\n--- {part.upper()} ---")
        print(f"  Vocab size (used): {vocab_size}")
        print(f"  Total tokens: {total:,}")

        # Top-10
        print(f"  Top-10 tokens:")
        for tok, cnt in sorted_counts[:10]:
            print(f"    token {tok:>4d}: {cnt:>8,} ({100*cnt/total:.2f}%)")

        # Bottom-10
        print(f"  Bottom-10 tokens:")
        for tok, cnt in sorted_counts[-10:]:
            print(f"    token {tok:>4d}: {cnt:>8,} ({100*cnt/total:.4f}%)")

        # Entropy
        probs = np.array([c for _, c in sorted_counts], dtype=np.float64) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-30))
        max_entropy = math.log2(vocab_size) if vocab_size > 0 else 0
        print(f"  Entropy: {entropy:.3f} bits (max = {max_entropy:.3f} bits, ratio = {entropy/max_entropy:.3f})")

        # Zipf analysis: log-rank vs log-freq
        ranks = np.arange(1, len(sorted_counts) + 1, dtype=np.float64)
        freqs = np.array([c for _, c in sorted_counts], dtype=np.float64)
        log_ranks = np.log(ranks)
        log_freqs = np.log(freqs + 1e-30)
        # Linear fit: log(freq) = a * log(rank) + b  →  Zipf exponent = -a
        coeffs = np.polyfit(log_ranks, log_freqs, 1)
        zipf_exp = -coeffs[0]
        print(f"  Zipf exponent: {zipf_exp:.3f} (1.0 = perfect Zipf, >1 = more skewed)")

        # Coverage: top-K% tokens cover X% of data
        cumulative = np.cumsum(freqs) / total
        for pct in [0.5, 0.8, 0.9, 0.95]:
            idx = np.searchsorted(cumulative, pct)
            print(f"  Top {idx+1} tokens ({100*(idx+1)/vocab_size:.1f}% of vocab) cover {100*pct:.0f}% of data")


# ─── 2. Adjacent-Frame Repetition ───────────────────────────────────

def repetition_analysis(sequences: dict):
    print("\n" + "=" * 60)
    print("2. ADJACENT-FRAME SAME-TOKEN REPETITION")
    print("=" * 60)

    for part in PART_NAMES:
        seqs = sequences[part]
        if not seqs:
            continue

        total_pairs = 0
        same_pairs = 0
        run_lengths = []  # length of consecutive same-token runs

        for seq in seqs:
            if len(seq) < 2:
                continue
            # Count adjacent same tokens
            same = (seq[1:] == seq[:-1])
            total_pairs += len(same)
            same_pairs += same.sum()

            # Run-length encoding
            run_len = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i - 1]:
                    run_len += 1
                else:
                    run_lengths.append(run_len)
                    run_len = 1
            run_lengths.append(run_len)

        rep_rate = same_pairs / total_pairs if total_pairs > 0 else 0
        run_arr = np.array(run_lengths)

        print(f"\n--- {part.upper()} ---")
        print(f"  Adjacent-frame repetition rate: {100*rep_rate:.2f}%")
        print(f"  Run-length stats: mean={run_arr.mean():.2f}, median={np.median(run_arr):.1f}, "
              f"max={run_arr.max()}, std={run_arr.std():.2f}")

        # Run-length distribution
        run_counter = Counter(run_arr.tolist())
        print(f"  Run-length distribution (top-10):")
        for rl, cnt in sorted(run_counter.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * cnt / len(run_arr)
            print(f"    length {int(rl):>3d}: {cnt:>8,} ({pct:.1f}%)")

        # Cross-part same-frame analysis
    print(f"\n--- CROSS-PART SAME-FRAME CORRELATION ---")
    parts_with_data = [p for p in PART_NAMES if sequences[p]]
    if len(parts_with_data) >= 2:
        for i, p1 in enumerate(parts_with_data):
            for p2 in parts_with_data[i + 1:]:
                same = 0
                total = 0
                for s1, s2 in zip(sequences[p1], sequences[p2]):
                    min_len = min(len(s1), len(s2))
                    same += (s1[:min_len] == s2[:min_len]).sum()
                    total += min_len
                rate = same / total if total > 0 else 0
                print(f"  {p1} vs {p2}: same-token rate = {100*rate:.2f}% (chance ≈ {100/256:.2f}%)")


# ─── 3. N-gram & BPE Feasibility ────────────────────────────────────

def ngram_analysis(all_tokens: dict, sequences: dict, max_n: int = 6):
    print("\n" + "=" * 60)
    print("3. N-GRAM FREQUENCY & BPE FEASIBILITY")
    print("=" * 60)

    for part in PART_NAMES:
        tokens = all_tokens[part]
        seqs = sequences[part]
        if len(tokens) == 0:
            continue

        print(f"\n--- {part.upper()} ---")

        # N-gram analysis per sequence (don't span across samples)
        for n in [2, 3, 4, 5, 6]:
            ngram_counter = Counter()
            total_ngrams = 0
            for seq in seqs:
                if len(seq) < n:
                    continue
                for i in range(len(seq) - n + 1):
                    ngram_counter[tuple(seq[i:i + n].tolist())] += 1
                    total_ngrams += 1

            if total_ngrams == 0:
                continue

            unique = len(ngram_counter)
            top5 = ngram_counter.most_common(5)
            top1_pct = 100 * top5[0][1] / total_ngrams if top5 else 0

            print(f"  {n}-gram: {unique:,} unique / {total_ngrams:,} total | "
                  f"top-1 covers {top1_pct:.2f}%")
            for ng, cnt in top5:
                print(f"    {ng}: {cnt:,} ({100*cnt/total_ngrams:.3f}%)")

        # BPE simulation: iteratively merge most frequent bigram
        print(f"\n  BPE compression simulation:")
        _bpe_simulation(seqs, part, num_merges=100)


def _bpe_simulation(sequences, part_name: str, num_merges: int = 100):
    """Simulate BPE merges and report compression ratio."""
    # Work with list-of-lists for mutability
    corpus = [seq.tolist() for seq in sequences if len(seq) >= 2]
    original_total = sum(len(s) for s in corpus)

    merge_log = []

    for step in range(num_merges):
        # Count bigram frequencies
        pair_counts = Counter()
        for seq in corpus:
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += 1

        if not pair_counts:
            break

        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            break

        # Merge best pair → new token
        new_token = 10000 + step  # synthetic merged token

        new_corpus = []
        for seq in corpus:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_corpus.append(new_seq)
        corpus = new_corpus

        current_total = sum(len(s) for s in corpus)
        ratio = current_total / original_total

        if (step + 1) in [1, 5, 10, 20, 50, 100] or step == num_merges - 1:
            merge_log.append((step + 1, best_pair, best_count, ratio))

    for step, pair, count, ratio in merge_log:
        print(f"    After {step:>3d} merges: ratio={ratio:.4f} "
              f"(last merge: {pair} x{count})")

    final_total = sum(len(s) for s in corpus)
    print(f"    Original: {original_total:,} tokens → After {len(merge_log) and merge_log[-1][0] or 0} merges: "
          f"{final_total:,} tokens ({100*final_total/original_total:.1f}%)")


# ─── 4. Temporal Autocorrelation ─────────────────────────────────────

def temporal_analysis(sequences: dict):
    print("\n" + "=" * 60)
    print("4. TEMPORAL AUTOCORRELATION (same-token at lag k)")
    print("=" * 60)

    for part in PART_NAMES:
        seqs = sequences[part]
        if not seqs:
            continue
        print(f"\n--- {part.upper()} ---")

        max_lag = 10
        for lag in range(1, max_lag + 1):
            same = 0
            total = 0
            for seq in seqs:
                if len(seq) <= lag:
                    continue
                same += (seq[lag:] == seq[:-lag]).sum()
                total += len(seq) - lag
            rate = same / total if total > 0 else 0
            bar = "█" * int(rate * 50)
            print(f"  lag={lag:>2d}: {100*rate:>6.2f}% {bar}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sign token statistical analysis")
    parser.add_argument("--token_dir", type=str,
                        default="data/TOKENS_h2s_csl_phoenix_backup_20260211",
                        help="Root directory of token .npy files")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max files to load (0=all)")
    args = parser.parse_args()

    sequences, all_tokens = load_all_tokens(args.token_dir, max_files=args.max_files)

    frequency_analysis(all_tokens)
    repetition_analysis(sequences)
    ngram_analysis(all_tokens, sequences)
    temporal_analysis(sequences)


if __name__ == "__main__":
    main()
