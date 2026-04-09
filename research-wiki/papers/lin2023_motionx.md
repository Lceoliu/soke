---
node_id: paper:lin2023_motionx
title: "Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset"
authors: ["Jing Lin", "Ailing Zeng", "Shunlin Lu", "Yuanhao Cai", "Ruimao Zhang", "Haoqian Wang", "Lei Zhang"]
year: 2023
venue: NeurIPS 2023
arxiv: "2307.00818"
tags: ["dataset", "smplx", "whole-body", "3d-motion", "expressive"]
---

## One-line thesis

Large-scale SMPL-X annotated whole-body motion dataset (15.6M frames) with body+hands+face+expression, enabling expressive motion generation research.

## SMPL-X representation format

```
θ_b : (22, 3) — body joint rotations (axis-angle), incl. root
θ_h : (30, 3) — hand joint rotations (15L + 15R × 3)
θ_f : (3,)    — jaw rotation
ψ   : (50,)   — facial expression parameters (FLAME blendshape)
r   : (3,)    — global translation
```

**No encoder architecture proposed** — dataset paper. Downstream methods (MDM, MLD, T2M-GPT) use standard sequential/Conv architectures on flattened parameters.

## Key design choice re: axis-angle

Motion-X does NOT apply graph networks to axis-angle params directly. Downstream papers convert to **6D rotation representation** (Lee & Wicke, 2019) to avoid periodicity/discontinuity issues before neural processing.

## Relevance

Confirms that SMPL-X body representation used in our CSL-Daily pose data (`smplx_body_pose` = 22×3 axis-angle) is the field standard. The 22-joint body kinematic tree is the standard SMPL-X tree. Reference for building our adjacency matrix.
