# MT5 M2T Quick Ablation Plan

All quick training runs use `20` epochs for faster comparison.

| ID | Purpose | Train Setting | Eval Setting | Expected Interpretation |
| --- | --- | --- | --- | --- |
| E1 | Full baseline | normal motion + source-aware prompt | normal motion | Reference point |
| E2 | No-motion baseline | zeroed sign embeddings + source-aware prompt | normal motion | If close to E1, motion is not carrying much content |
| E3 | Prompt ablation | normal motion + generic prompt | normal motion | If style/domain prior drops sharply, prompt was doing heavy lifting |
| E4 | Shuffle-motion eval | reuse E1 checkpoint | shuffle motion within each source | If close to E1, model is weakly sensitive to true motion-content alignment |

## One-click Scripts

- Run one experiment:
  - `scripts/pipeline/run_mt5_m2t_quick_ablation.sh`
- Run the whole suite:
  - `scripts/pipeline/run_mt5_m2t_quick_suite.sh`
- Standalone evaluation:
  - `scripts/analysis/eval_m2t_predictions.py`

## Quick Commands

### E1 Full

```bash
EXP_KIND=full bash scripts/pipeline/run_mt5_m2t_quick_ablation.sh
```

### E2 No Motion

```bash
EXP_KIND=no_motion bash scripts/pipeline/run_mt5_m2t_quick_ablation.sh
```

### E3 Generic Prompt

```bash
EXP_KIND=generic_prompt bash scripts/pipeline/run_mt5_m2t_quick_ablation.sh
```

### E4 Shuffle Motion

Run after E1 is finished:

```bash
bash scripts/pipeline/run_mt5_m2t_quick_suite.sh
```

The suite script runs E1-E3 training and then E4 shuffle evaluation on the E1 checkpoint.

## Default Outputs

Each experiment writes:

- metrics json:
  - `experiments/mgpt/<EXP_NAME>/auto_reports/downstream/m2t_eval_val.json`
- qualitative jsonl:
  - `experiments/mgpt/<EXP_NAME>/auto_reports/downstream/m2t_examples_val.jsonl`

Shuffle evaluation for E1 writes:

- `experiments/mgpt/SOKE_MT5_M2T_FULL_E20/auto_reports/downstream/m2t_eval_val_shuffle_by_src.json`
- `experiments/mgpt/SOKE_MT5_M2T_FULL_E20/auto_reports/downstream/m2t_examples_val_shuffle_by_src.jsonl`
