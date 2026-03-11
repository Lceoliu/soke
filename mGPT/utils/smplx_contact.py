from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mGPT.utils.human_models import get_coord, smpl_x


TensorLike = Union[torch.Tensor, np.ndarray]

SMPLX_KEYS_179 = (
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
)

DEFAULT_SHAPE = np.array(
    [
        -0.07284723,
        0.1795129,
        -0.27608207,
        0.135155,
        0.10748172,
        0.16037364,
        -0.01616933,
        -0.03450319,
        0.01369138,
        0.01108842,
    ],
    dtype=np.float32,
)

PART_ALIASES = {
    "face": "face",
    "lhand": "lhand",
    "left_hand": "lhand",
    "left-hand": "lhand",
    "left": "lhand",
    "rhand": "rhand",
    "right_hand": "rhand",
    "right-hand": "rhand",
    "right": "rhand",
}

DEFAULT_CONTACT_PAIRS = (("lhand", "face"), ("rhand", "face"), ("lhand", "rhand"))


def _to_tensor(x: TensorLike, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x
    else:
        out = torch.from_numpy(np.asarray(x))
    out = out.to(dtype=torch.float32)
    if device is not None:
        out = out.to(device=device)
    return out


def canonical_part_name(part: str) -> str:
    key = str(part).strip().lower()
    if key not in PART_ALIASES:
        raise ValueError(f"Unsupported part: {part}. Supported: face/lhand/rhand (and aliases).")
    return PART_ALIASES[key]


def get_part_vertex_indices(part: str, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    p = canonical_part_name(part)
    if p == "face":
        idx = np.asarray(smpl_x.face_vertex_idx, dtype=np.int64)
    elif p == "lhand":
        idx = np.asarray(smpl_x.hand_vertex_idx["left_hand"], dtype=np.int64)
    elif p == "rhand":
        idx = np.asarray(smpl_x.hand_vertex_idx["right_hand"], dtype=np.int64)
    else:
        raise RuntimeError(f"Internal error for part={part}")
    out = torch.from_numpy(idx).long()
    if device is not None:
        out = out.to(device=device)
    return out


def _sort_key_by_last_number(path: Path):
    hits = re.findall(r"\d+", path.name)
    if hits:
        return (0, int(hits[-1]), path.name)
    return (1, 0, path.name)


def _load_frame_dict(path: Path):
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    with path.open("rb") as f:
        return pickle.load(f)


def load_pose179_from_smplx_dir(pose_dir: Union[str, Path]) -> np.ndarray:
    """
    Load per-frame SMPLerX files (.pkl/.pt) into [T, 179].
    """
    d = Path(pose_dir)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"pose_dir not found: {pose_dir}")
    files = sorted(
        [p for p in d.iterdir() if p.suffix in {".pkl", ".pt"}],
        key=_sort_key_by_last_number,
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl/.pt files found in {pose_dir}")

    arr = np.zeros((len(files), 179), dtype=np.float32)
    for i, fp in enumerate(files):
        data = _load_frame_dict(fp)
        missing = [k for k in SMPLX_KEYS_179 if k not in data]
        if missing:
            raise KeyError(f"{fp} missing keys: {missing}")
        arr[i] = np.concatenate([np.asarray(data[k]).reshape(-1) for k in SMPLX_KEYS_179], axis=0).astype(np.float32)
    return arr


def _split_pose179(pose179: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    if pose179.ndim != 2 or pose179.shape[1] != 179:
        raise ValueError(f"pose179 must be [T,179], got {tuple(pose179.shape)}")
    root = pose179[:, 0:3]
    body = pose179[:, 3:66]
    lhand = pose179[:, 66:111]
    rhand = pose179[:, 111:156]
    jaw = pose179[:, 156:159]
    shape = pose179[:, 159:169]
    expr = pose179[:, 169:179]
    return root, body, lhand, rhand, jaw, shape, expr


def _split_feat133(feat133: torch.Tensor, shape_template: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
    if feat133.ndim != 2 or feat133.shape[1] != 133:
        raise ValueError(f"feat133 must be [T,133], got {tuple(feat133.shape)}")
    t = feat133.shape[0]
    zero_pose = torch.zeros((t, 36), dtype=feat133.dtype, device=feat133.device)
    full169 = torch.cat([zero_pose, feat133], dim=-1)
    root = full169[:, 0:3]
    body = full169[:, 3:66]
    lhand = full169[:, 66:111]
    rhand = full169[:, 111:156]
    jaw = full169[:, 156:159]
    expr = full169[:, 159:169]
    if shape_template is None:
        shape_template = torch.from_numpy(DEFAULT_SHAPE).to(device=feat133.device, dtype=feat133.dtype)
    shape = shape_template.view(1, -1).repeat(t, 1)
    return root, body, lhand, rhand, jaw, shape, expr


def smplx_to_vertices(
    data: TensorLike,
    data_type: str = "pose179",
    mean133: Optional[TensorLike] = None,
    std133: Optional[TensorLike] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_joints: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    FK to vertices for:
    - data_type='pose179'   : [T,179]
    - data_type='feat133_raw'  : [T,133]
    - data_type='feat133_norm' : [T,133] + mean/std (133)
    """
    x = _to_tensor(data, device=device)

    dtype = str(data_type).strip().lower()
    if dtype == "pose179":
        root, body, lhand, rhand, jaw, shape, expr = _split_pose179(x)
    elif dtype in {"feat133_raw", "feat133_norm"}:
        feat = x
        if dtype == "feat133_norm":
            if mean133 is None or std133 is None:
                raise ValueError("mean133/std133 are required for data_type='feat133_norm'")
            mean_t = _to_tensor(mean133, device=feat.device).view(1, -1)
            std_t = _to_tensor(std133, device=feat.device).view(1, -1)
            if mean_t.shape[1] != 133 or std_t.shape[1] != 133:
                raise ValueError(f"mean/std must be 133-dim, got mean={tuple(mean_t.shape)}, std={tuple(std_t.shape)}")
            feat = feat * std_t + mean_t
        root, body, lhand, rhand, jaw, shape, expr = _split_feat133(feat)
    else:
        raise ValueError(f"Unsupported data_type={data_type}. Use pose179/feat133_raw/feat133_norm.")

    with torch.no_grad():
        vertices, joints = get_coord(
            root_pose=root,
            body_pose=body,
            lhand_pose=lhand,
            rhand_pose=rhand,
            jaw_pose=jaw,
            shape=shape,
            expr=expr,
            return_verts=True,
        )
    # get_coord returns [T,V,3] in this repo path
    vertices = vertices.contiguous()
    joints = joints.contiguous()
    if return_joints:
        return vertices, joints
    return vertices


def _normalize_pair_name(a: str, b: str) -> Tuple[str, str]:
    ca, cb = canonical_part_name(a), canonical_part_name(b)
    return (ca, cb)


def _resolve_threshold(
    pair: Tuple[str, str],
    threshold: Union[float, Mapping[str, float]],
) -> float:
    if isinstance(threshold, Mapping):
        k1 = f"{pair[0]}-{pair[1]}"
        k2 = f"{pair[1]}-{pair[0]}"
        if k1 in threshold:
            return float(threshold[k1])
        if k2 in threshold:
            return float(threshold[k2])
        raise KeyError(
            f"Missing threshold for pair '{k1}'. Available keys: {list(threshold.keys())}"
        )
    return float(threshold)


def min_pairwise_distance_chunked(
    verts_a: torch.Tensor,
    verts_b: torch.Tensor,
    chunk_a: int = 512,
    chunk_b: int = 2048,
) -> torch.Tensor:
    """
    Compute min_{i,j} ||a_i - b_j|| per frame.
    Input:
      verts_a: [T, Na, 3]
      verts_b: [T, Nb, 3]
    Output:
      min_dist: [T]
    """
    if verts_a.ndim != 3 or verts_b.ndim != 3:
        raise ValueError(f"Expected [T,N,3], got {tuple(verts_a.shape)} and {tuple(verts_b.shape)}")
    if verts_a.shape[0] != verts_b.shape[0] or verts_a.shape[-1] != 3 or verts_b.shape[-1] != 3:
        raise ValueError(f"Shape mismatch: {tuple(verts_a.shape)} vs {tuple(verts_b.shape)}")
    if chunk_a <= 0 or chunk_b <= 0:
        raise ValueError("chunk_a/chunk_b must be positive")

    t, na, _ = verts_a.shape
    nb = verts_b.shape[1]
    inf = torch.tensor(float("inf"), device=verts_a.device, dtype=verts_a.dtype)
    frame_min = torch.full((t,), inf.item(), device=verts_a.device, dtype=verts_a.dtype)

    for i in range(0, na, chunk_a):
        a = verts_a[:, i : i + chunk_a, :]
        # [T, ca]
        min_for_a = torch.full((t, a.shape[1]), inf.item(), device=verts_a.device, dtype=verts_a.dtype)
        for j in range(0, nb, chunk_b):
            b = verts_b[:, j : j + chunk_b, :]
            # [T, ca, cb]
            d = torch.cdist(a, b, p=2)
            min_for_a = torch.minimum(min_for_a, d.amin(dim=2))
        frame_min = torch.minimum(frame_min, min_for_a.amin(dim=1))
    return frame_min


def any_contact_within_threshold_chunked(
    verts_a: torch.Tensor,
    verts_b: torch.Tensor,
    threshold: float,
    chunk_a: int = 512,
    chunk_b: int = 2048,
) -> torch.Tensor:
    """
    Faster bool-only contact detection:
      exists i,j s.t. ||a_i - b_j|| < threshold
    Input:
      verts_a: [T, Na, 3]
      verts_b: [T, Nb, 3]
    Output:
      is_contact: [T] bool
    """
    if verts_a.ndim != 3 or verts_b.ndim != 3:
        raise ValueError(f"Expected [T,N,3], got {tuple(verts_a.shape)} and {tuple(verts_b.shape)}")
    if verts_a.shape[0] != verts_b.shape[0] or verts_a.shape[-1] != 3 or verts_b.shape[-1] != 3:
        raise ValueError(f"Shape mismatch: {tuple(verts_a.shape)} vs {tuple(verts_b.shape)}")
    if chunk_a <= 0 or chunk_b <= 0:
        raise ValueError("chunk_a/chunk_b must be positive")

    t, na, _ = verts_a.shape
    nb = verts_b.shape[1]
    thr = float(threshold)
    is_contact = torch.zeros((t,), dtype=torch.bool, device=verts_a.device)

    for i in range(0, na, chunk_a):
        if bool(is_contact.all()):
            break
        a = verts_a[:, i : i + chunk_a, :]
        for j in range(0, nb, chunk_b):
            if bool(is_contact.all()):
                break
            b = verts_b[:, j : j + chunk_b, :]
            # [T, ca, cb]
            d = torch.cdist(a, b, p=2)
            # bool per frame
            hit = (d < thr).any(dim=2).any(dim=1)
            is_contact |= hit
    return is_contact


def detect_contacts_from_vertices(
    vertices: TensorLike,
    pairs: Sequence[Tuple[str, str]] = DEFAULT_CONTACT_PAIRS,
    threshold: Union[float, Mapping[str, float]] = 0.02,
    chunk_a: int = 512,
    chunk_b: int = 2048,
    bool_only: bool = False,
) -> Dict[str, Dict[str, Union[torch.Tensor, float]]]:
    """
    Args:
      vertices: [T, V, 3]
      pairs: part pairs, each in {face, lhand, rhand}
      threshold: scalar or pair-specific dict, unit is same as vertices (usually meters)
    Returns:
      bool_only=False:
        {
          "lhand-face": {"min_dist": Tensor[T], "is_contact": BoolTensor[T], "threshold": float},
          ...
        }
      bool_only=True:
        {
          "lhand-face": {"is_contact": BoolTensor[T], "threshold": float},
          ...
        }
    """
    verts = _to_tensor(vertices)
    if verts.ndim != 3 or verts.shape[-1] != 3:
        raise ValueError(f"vertices must be [T,V,3], got {tuple(verts.shape)}")

    idx_cache: Dict[str, torch.Tensor] = {}
    out: Dict[str, Dict[str, Union[torch.Tensor, float]]] = {}

    for p1, p2 in pairs:
        a, b = _normalize_pair_name(p1, p2)
        if a not in idx_cache:
            idx_cache[a] = get_part_vertex_indices(a, device=verts.device)
        if b not in idx_cache:
            idx_cache[b] = get_part_vertex_indices(b, device=verts.device)

        va = verts.index_select(1, idx_cache[a])
        vb = verts.index_select(1, idx_cache[b])
        thr = _resolve_threshold((a, b), threshold)
        key = f"{a}-{b}"
        if bool_only:
            out[key] = {
                "is_contact": any_contact_within_threshold_chunked(
                    va, vb, threshold=thr, chunk_a=chunk_a, chunk_b=chunk_b
                ),
                "threshold": thr,
            }
        else:
            min_dist = min_pairwise_distance_chunked(va, vb, chunk_a=chunk_a, chunk_b=chunk_b)
            out[key] = {
                "min_dist": min_dist,
                "is_contact": min_dist < thr,
                "threshold": thr,
            }
    return out


def detect_contacts_from_smplx(
    data: TensorLike,
    data_type: str = "pose179",
    pairs: Sequence[Tuple[str, str]] = DEFAULT_CONTACT_PAIRS,
    threshold: Union[float, Mapping[str, float]] = 0.02,
    mean133: Optional[TensorLike] = None,
    std133: Optional[TensorLike] = None,
    device: Optional[Union[str, torch.device]] = None,
    chunk_a: int = 512,
    chunk_b: int = 2048,
    bool_only: bool = False,
) -> Dict[str, Dict[str, Union[torch.Tensor, float]]]:
    """
    One-stop API:
      smplx data -> FK vertices -> contact distances/flags.
    """
    verts = smplx_to_vertices(
        data=data,
        data_type=data_type,
        mean133=mean133,
        std133=std133,
        device=device,
        return_joints=False,
    )
    return detect_contacts_from_vertices(
        vertices=verts,
        pairs=pairs,
        threshold=threshold,
        chunk_a=chunk_a,
        chunk_b=chunk_b,
        bool_only=bool_only,
    )


def summarize_contact_results(
    results: Mapping[str, Mapping[str, Union[torch.Tensor, float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Convert tensor outputs to scalar summary for logging/reporting.
    """
    summary: Dict[str, Dict[str, float]] = {}
    for pair, payload in results.items():
        min_dist = payload.get("min_dist", None)
        is_contact = payload["is_contact"]
        thr = float(payload["threshold"])
        if not isinstance(is_contact, torch.Tensor):
            raise TypeError("Invalid result payload types")
        is_contact_cpu = is_contact.detach().float().cpu()
        row: Dict[str, float] = {
            "threshold": thr,
            "contact_ratio": float(is_contact_cpu.mean().item()),
        }
        if isinstance(min_dist, torch.Tensor):
            min_dist_cpu = min_dist.detach().float().cpu()
            row["min_of_min_dist"] = float(min_dist_cpu.min().item())
            row["mean_min_dist"] = float(min_dist_cpu.mean().item())
        summary[pair] = row
    return summary
