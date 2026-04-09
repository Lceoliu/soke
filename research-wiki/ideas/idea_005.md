---
node_id: idea:005
title: "ST-GCN-Based VAE Encoder for Sign Language Pose Embedding"
status: active
confidence: medium-high
origin: failure_analysis_M2
linked_gaps: [G1, G5, G6]
linked_papers: [paper:li2025_unisign]
date: 2026-04-09
---

## Status Update (2026-04-09)

**Uni-Sign correction**: Uni-Sign DOES use vanilla ST-GCN (3 layers, [64→128→256] channels) on 69 2D keypoints (x,y from RTMPose-x), NOT axis-angle rotations. Our data is SMPL-X axis-angle — different format.

**Critical finding from 3D motion literature**: Axis-angle parameters should NOT be fed directly to GCN due to periodicity/discontinuity (axis-angle is non-unique at ±π). Standard practice: convert to **6D rotation representation** first.

**3D motion generation field (DartControl, MotionMillion, OmniMotion-X)**: Uses Transformer, not GCN, on flattened SMPL-X vectors. GCN is not the consensus for SMPL-X axis-angle. However, graph structure is still unexplored for this format.

---

## Core Hypothesis

The current VAE encoder (Conv1d + Resnet1D blocks on a flat 179-dim concatenated pose vector) is **architecturally naive**: it treats all joint dimensions uniformly and cannot model the inherent graph structure of the human body (joints = nodes, bones = edges). This produces VAE embeddings that are semantically weak and temporally dense, which is why the mT5 LM fails to decode them regardless of pretraining strategy.

Replacing the encoder with a **Spatial-Temporal Graph Convolutional Network (ST-GCN)** that explicitly models the skeleton graph topology should produce richer, more discriminative per-frame embeddings with lower temporal density.

## Why This Is High Priority

1. **Architecture mismatch**: Sign language motion is fundamentally relational — the meaning of a hand position depends on its relation to the body, wrist, and other hand. Conv1d on flattened pose treats all 179 dimensions as an independent 1D signal, destroying this relational structure.

2. **Evidence from Uni-Sign**: The strongest pose-based SLT method (ICLR 2025) uses separate encoders per body-part group (left hand / right hand / body / face), then fuses with attention — explicitly acknowledging that body parts have different semantic roles and temporal dynamics.

3. **Evidence from SignFormer-GCN (PLOS ONE 2025)**: ST-GCN + Transformer for SLT on PHOENIX achieves 19.75 B@4 with 12× fewer parameters than RGB-based methods. The ST-GCN backbone is the key contributor.

4. **Field consensus**: Every recent SLR/SLT paper that is purely pose-based uses some form of graph convolution (DSTA-SLR, HA-GCN, S2Net). Our Conv1d baseline is 5+ years behind the curve.

## What Changes

Current encoder (each VAE independently):
```
flat 133-dim (body) or 45-dim (hand) → Conv1d → ReLU → [Conv1d+Resnet1D]×3 → Conv1d(→512)
```

Proposed encoder:
```
axis-angle → 6D rotation conversion → reshape [T, N_joints, 6] → ST-GCN → [T', 512]
```

**Step 1 — Representation conversion (mandatory)**:
- Convert each joint's axis-angle (3-dim) to **6D rotation** (first two columns of rotation matrix, 6-dim) using Rodrigues formula
- Body: 10 upper joints × 6 = 60-dim; lhand: 15 × 6 = 90-dim; rhand: 15 × 6 = 90-dim
- Reshape to [T, N_joints, 6] — compatible with graph convolution

**Step 2 — ST-GCN encoder (Uni-Sign style)**:
- Uni-Sign architecture: 3-layer ST-GCN, channels [64 → 128 → 256], then temporal encoder [256 → 256 → 256]
- Adjacency: SMPL-X kinematic tree edges for upper body + MANO hand tree for each hand
- Partition: 3-part spatial labeling (root / close / distant) following vanilla ST-GCN
- Per-body-part: **separate ST-GCN for body / lhand / rhand** → matches our 3-VAE structure

**Architecture options**:

| Option | Input | Adjacency | Effort |
|---|---|---|---|
| A — Per-part ST-GCN (recommended) | Body [T,10,6]; Hand [T,15,6] | SMPL-X tree per part | Medium |
| B — Full-body shared ST-GCN | All joints [T,40,6] | Full SMPL-X tree | Medium |
| C — Transformer on 6D features | [T, N_joints, 6] → flatten | No graph | Low (baseline for comparison) |

**Start with Option A. Option C is a useful ablation (Transformer-on-6D vs. ST-GCN-on-6D).**

**Uni-Sign reference**: channels [64,128,256] for pose encoder, [256,256,256] for temporal encoder — copy this directly and tune to our joint counts.

## Key Implementation Details

- **Input format**: reshape 179-dim flat vector into `[N_joints, C_coords]` per frame; build adjacency matrix from skeleton topology
- **Temporal dimension**: ST-GCN handles both spatial (graph conv) and temporal (1D conv) — replaces both Conv1d and Resnet1D
- **Compatibility**: output per-frame embedding must match existing `code_dim=512`; can add a final linear projection
- **VAE structure**: encoder changes, decoder stays (Conv1d decoder → per-frame prediction); LFQ quantizer unchanged
- **Per-body-part VAEs**: we already have body_vae / hand_vae / rhand_vae separation — use Option B naturally

## Pose Format: SMPL-X Axis-Angle (NOT 2D Keypoints)

**Critical detail**: Our 179-dim input is SMPL-X axis-angle rotation parameters (3 values per joint):
```
smplx_root_pose  (3,)  = 1  joint
smplx_body_pose  (63,) = 21 joints  [after trim: remove 11 lower body → keeps 10]
smplx_lhand_pose (45,) = 15 joints
smplx_rhand_pose (45,) = 15 joints
smplx_jaw_pose   (3,)  = 1  joint
```
After preprocessing (remove lower body, remove shape): **133-dim** actual input to VAE.

This means we have:
- **Body**: 10 joints × 3 (upper body: spine, chest, neck, head, shoulders, elbows, wrists)
- **Left hand**: 15 joints × 3
- **Right hand**: 15 joints × 3
- **Jaw**: 1 joint × 3
- **Total**: ~41 joints × 3 = 123, plus remaining root = 133 dim

This is NOT 2D keypoints but can still use graph convolution:
- Reshape `[T, 133]` → `[T, N_joints, 3]` where each joint has a 3-dim axis-angle feature
- Define adjacency from SMPL-X kinematic tree (parent-child joint connections)
- Apply ST-GCN: spatial graph conv on joint graph + temporal conv across T frames

The graph convolution still exploits joint topology even with rotation features instead of positions.

## What We Need to Know First

- Map the 133-dim back to joint indices: which 10 body joints remain after lower-body removal?
- SMPL-X kinematic tree adjacency for those joints (standard — available from SMPL-X model files)
- ST-GCN implementations: `mmskeleton`, `pyskl`, or re-implement from scratch (~200 lines)
- Whether to use fixed (anatomical) or adaptive adjacency (learnable A matrix)

## Predicted Effect

If the encoder produces richer per-frame embeddings:
1. **Temporal density decreases**: adjacent frames are more distinguishable (H2 confirmed fixed)
2. **Classifier accuracy increases**: above the current 81.8% (claim:C5 extends)
3. **mT5 fine-tune BLEU improves**: because the LM input is no longer a stream of near-identical vectors

Expected: CSL-Daily csl_BLEU4 jump from ~1.7 → ≥5.0 after ST-GCN encoder + existing contrastive pretraining or direct fine-tune.

## Open Questions

- ST-GCN was designed for action recognition (short clips, coarse labels). Does it transfer to fine-grained sign-level semantics?
- Will the graph inductive bias help or hurt when the adjacency structure doesn't perfectly match our pose format?
- Should the adjacency matrix be fixed (anatomical) or learnable (adaptive graph convolution)?
