# Auto Review Log — scripts/train_semantic_vae_finetune.py

**Target**: R027 Semantic VAE Encoder Fine-tuning Script  
**Started**: 2026-04-12T10:27:28Z  
**Max rounds**: 4  
**Difficulty**: medium  

---

## Round 1 — 2026-04-12T10:27:28Z

### Assessment
- **Score**: 4/10
- **Verdict**: Not ready (suitable only for plumbing sanity check)
- **Top weaknesses**:
  1. mT5 encoder in train mode → stochastic text targets
  2. False negatives from duplicate sentences in SLT data
  3. Projection heads absorb contrastive loss, raw embeddings may not improve
  4. Padding contaminates mean-pool (lengths ignored in encode_continuous_batch)
  5. Reconstruction loss unnormalized (scales with feature dim)
  6. VAE auxiliary losses (commitment/entropy) discarded
  7. avg_pos_sim logs temperature-scaled logit, not cosine
  8. val_loader passed but never used
  9. Checkpoints don't save optimizer/scheduler state
  10. Mean pooling ignores temporal structure

### Reviewer Raw Response

<details>
<summary>Click to expand</summary>

Score: 4/10. FrozenTextEncoder.encode() uses @torch.no_grad() but model.train() puts mT5 encoder back into train mode → stochastic text anchors. InfoNCE false negatives for duplicate texts. Projection heads can absorb contrastive loss without improving raw embeddings. encode_continuous_batch ignores lengths → padding contamination. Reconstruction loss divides by frames not frame-elements. VAE auxiliary losses (loss_b/loss_l/loss_r) discarded. avg_pos_sim is temperature-scaled logit. val_loader unused. Checkpoints incomplete. Mean pooling ignores temporal structure.

Verdict: Almost / Not ready for decisive R027 run.

</details>

### Round 2 Score: 6.5/10 — Almost
Additional fixes: text NFKC normalization for pos_mask, top-k checkpoint saving, quantizer loss logging, downsampled length smoke-test.
Bug fixes: `unicodedata` moved to top-level import, `get_vae_downsample_params` uses `is not None` instead of `or`.
AST parse + 12 sanity assertions: all OK.

### Actions Taken
Fixing issues #1, #2, #3, #4, #5, #7, #8 (critical path):
- Force mT5 eval mode inside encode() and after construction
- Multi-positive InfoNCE for duplicate sentences
- Add direct raw-embedding contrastive loss (no projection) as auxiliary diagnostic
- Masked mean-pooling using downsampled lengths
- Normalize reconstruction loss by `mask.sum() * D`
- Log raw cosine similarity (pre-temperature)
- Use val_loader for validation recon/contrastive metrics

---
