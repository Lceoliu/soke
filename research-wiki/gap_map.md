# Gap Map

> Known field gaps with stable IDs. Updated as papers are ingested.

| ID | Gap | Status | Linked Papers | Linked Ideas |
|----|-----|--------|---------------|--------------|
| G1 | What makes a tokenizer/embedding suitable for downstream LM (not just reconstruction)? | unresolved | paper:zuo2024_soke, paper:jiang2023_motiongpt | idea:001, idea:004 |
| G2 | Why does full-train generalization fail for both decoder-only and seq2seq sign-LM despite functional pipeline and discriminative embeddings? | unresolved | paper:li2025_unisign | idea:002, idea:003 |
| G3 | How can unified multi-task sign language AR training (t2m + m2t + mc) be made to generalize at current data scales? | unresolved | paper:jiang2023_motiongpt, paper:li2025_unisign | idea:002 |
| G4 | Is there a principled way to align sign language embedding space with text LM space beyond next-token prediction? | unresolved | paper:li2025_unisign | — |

## Gap Details

### G1 — Tokenizer LM-Compatibility

**Description:** Reconstruction quality and LM-compatibility are not the same objective. The field lacks clear criteria for what makes a sign tokenizer "LM-friendly": is it codebook utilization, smoothness of the discrete trajectory, entropy of the token distribution, or something else?

**Why it matters:** Current work shows that improving reconstruction (claim:C1) did not improve LM generalization (claim:C2). Understanding what the LM actually needs from the tokenizer is prerequisite to fixing this.

**Unresolved because:** No existing paper directly studies this for sign language. Some work exists for image/video tokenizers (MAGVIT-v2, BSQ) but the sign language case has additional structure (temporal, multi-body-part, linguistic).

---

### G2 — Full-Train Generalization Failure Mechanism

**Description:** Both Qwen (decoder-only) and mT5 (seq2seq) fail to generalize on full-dataset training despite (a) correct pipeline (exp:002), (b) active use of motion embeddings (claim:C4), and (c) discriminative embeddings (claim:C5). The failure mechanism is unknown.

**Why it matters:** Without understanding *why* generalization fails, it is impossible to systematically fix it. Current experiments rule out several hypotheses but the root cause remains open.

**Candidate hypotheses:**
1. Data volume is simply insufficient for emergent sign-text binding in current model sizes
2. The training objective (pure next-token or seq2seq CE) is misaligned with sign-text grounding
3. Task interference in multi-task (t2m+m2t+mc) setup hurts individual task performance
4. Sign token distribution has structural properties that break LM assumptions (non-Markovian, high entropy)

---

### G3 — Unified Sign Language AR Training at Current Data Scale

**Description:** MotionGPT succeeded for general body motion with HumanML3D (~15k samples), but sign language datasets are smaller and the task is harder (multilingual, fine-grained hand articulation, semantic binding). The right training recipe for unified sign language AR models at current scale is unknown.

---

### G4 — Sign-Text Space Alignment Beyond Next-Token Prediction

**Description:** Uni-Sign achieves good SLT by aligning pre-training and fine-tuning objectives. But it doesn't generate motion. A model that must both generate signs and translate them needs a deeper form of cross-modal alignment that next-token prediction alone may not provide.
