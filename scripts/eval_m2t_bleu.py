#!/usr/bin/env python3
"""
Standalone m2t evaluation: BLEU1-4, ROUGE-L, WER.

Reads prediction files written by the mT5 training run and computes metrics
against the dataset ground-truth texts.  Can also run inference itself if
given a checkpoint path.

Usage — score existing predictions file:
    python scripts/eval_m2t_bleu.py \
        --pred_file experiments/mgpt/SOKE_MT5_CSL/auto_reports/downstream/m2t_examples.jsonl \
        --output_file results/eval_m2t_csl.json

Usage — run inference + score from checkpoint:
    python scripts/eval_m2t_bleu.py \
        --cfg configs/soke_mt5_csl_m2t.yaml \
        --ckpt experiments/mgpt/SOKE_MT5_CSL/checkpoints/last.ckpt \
        --split test \
        --output_file results/eval_m2t_csl.json

Results saved as JSON:
    {
      "bleu1": ..., "bleu2": ..., "bleu3": ..., "bleu4": ...,
      "rouge_l": ..., "wer": ...,
      "n_samples": ...,
      "split": ...,
      "ckpt": ...
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer (consistent with mT5 evaluation convention)."""
    return text.strip().lower().split()


def _ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i: i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _bleu_precision(hyp: List[str], ref: List[str], n: int) -> Tuple[int, int]:
    """Clipped n-gram precision numerator and denominator."""
    hyp_ng = _ngrams(hyp, n)
    ref_ng = _ngrams(ref, n)
    clipped = sum(min(cnt, ref_ng.get(ng, 0)) for ng, cnt in hyp_ng.items())
    total = max(len(hyp) - n + 1, 0)
    return clipped, total


def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Corpus-level BLEU1-4 with brevity penalty."""
    import math

    assert len(hypotheses) == len(references), "Length mismatch"

    clipped = [0] * 4
    total = [0] * 4
    hyp_len = 0
    ref_len = 0

    for hyp_str, ref_str in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        ref = _tokenize(ref_str)
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, 5):
            c, t = _bleu_precision(hyp, ref, n)
            clipped[n - 1] += c
            total[n - 1] += t

    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    scores = {}
    log_sum = 0.0
    for n in range(1, 5):
        prec = clipped[n - 1] / max(total[n - 1], 1)
        log_sum += math.log(prec + 1e-40)
        scores[f"bleu{n}"] = round(bp * math.exp(log_sum / n) * 100, 2)

    return scores


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    """Corpus-level ROUGE-L (F1 based on LCS)."""

    def lcs_len(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        for i in range(m):
            curr = [0] * (n + 1)
            for j in range(n):
                curr[j + 1] = prev[j] + 1 if a[i] == b[j] else max(prev[j + 1], curr[j])
            prev = curr
        return prev[n]

    total_f = 0.0
    for hyp_str, ref_str in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        ref = _tokenize(ref_str)
        if not hyp or not ref:
            continue
        lcs = lcs_len(hyp, ref)
        p = lcs / len(hyp)
        r = lcs / len(ref)
        f = 2 * p * r / (p + r + 1e-10)
        total_f += f

    return round(total_f / max(len(hypotheses), 1) * 100, 2)


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """Word Error Rate (%)."""

    def edit_distance(a: List[str], b: List[str]) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            new_dp = [i] + [0] * n
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    new_dp[j] = dp[j - 1]
                else:
                    new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
            dp = new_dp
        return dp[n]

    total_err = 0
    total_ref = 0
    for hyp_str, ref_str in zip(hypotheses, references):
        hyp = _tokenize(hyp_str)
        ref = _tokenize(ref_str)
        total_err += edit_distance(hyp, ref)
        total_ref += len(ref)

    wer = total_err / max(total_ref, 1) * 100
    return round(wer, 2)


def compute_all_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    bleu = compute_bleu(hypotheses, references)
    rouge = compute_rouge_l(hypotheses, references)
    wer = compute_wer(hypotheses, references)
    return {**bleu, "rouge_l": rouge, "wer": wer, "n_samples": len(hypotheses)}


# ---------------------------------------------------------------------------
# Load predictions from JSONL file
# ---------------------------------------------------------------------------

def load_predictions_jsonl(path: str) -> Tuple[List[str], List[str]]:
    """Load (ref, pred) pairs from a JSONL file.

    Supports both:
      {"ref": "...", "pred": "..."}
      {"name": "...", "ref": "...", "pred": "..."}
    """
    refs, preds = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ref = obj.get("ref", obj.get("gt", obj.get("reference", "")))
            pred = obj.get("pred", obj.get("prediction", obj.get("hypothesis", "")))
            refs.append(str(ref))
            preds.append(str(pred))
    return refs, preds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate m2t predictions: BLEU/ROUGE/WER")
    p.add_argument("--pred_file", type=str, default=None,
                   help="Path to JSONL file with {ref, pred} entries (score existing predictions)")
    p.add_argument("--output_file", type=str, required=True,
                   help="Where to save the JSON results")
    p.add_argument("--split", type=str, default="test",
                   help="Dataset split (for logging purposes)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="(Optional) Model checkpoint path for future inference integration")
    return p.parse_args()


def main():
    args = parse_args()

    if args.pred_file is None:
        print(
            "ERROR: --pred_file is required.\n"
            "Provide a JSONL file with {ref, pred} entries.\n"
            "The mT5 training run writes this to:\n"
            "  experiments/mgpt/<NAME>/auto_reports/downstream/m2t_examples.jsonl"
        )
        sys.exit(1)

    if not os.path.isfile(args.pred_file):
        print(f"ERROR: Prediction file not found: {args.pred_file}")
        sys.exit(1)

    refs, preds = load_predictions_jsonl(args.pred_file)
    print(f"[eval] Loaded {len(refs)} samples from {args.pred_file}")

    if len(refs) == 0:
        print("ERROR: No valid predictions found in file")
        sys.exit(1)

    metrics = compute_all_metrics(preds, refs)
    metrics["split"] = args.split
    metrics["pred_file"] = args.pred_file
    if args.ckpt:
        metrics["ckpt"] = args.ckpt

    # Print summary
    print(f"\n{'='*50}")
    print(f"  M2T Evaluation — {args.split} split")
    print(f"{'='*50}")
    print(f"  BLEU-1:  {metrics['bleu1']:.2f}")
    print(f"  BLEU-2:  {metrics['bleu2']:.2f}")
    print(f"  BLEU-3:  {metrics['bleu3']:.2f}")
    print(f"  BLEU-4:  {metrics['bleu4']:.2f}")
    print(f"  ROUGE-L: {metrics['rouge_l']:.2f}")
    print(f"  WER:     {metrics['wer']:.2f}%")
    print(f"  N:       {metrics['n_samples']}")
    print(f"{'='*50}\n")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
