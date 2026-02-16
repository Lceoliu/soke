import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import BASEDataModule
from .utils import humanml3d_collate
from mGPT.utils.human_models import get_coord


def _uniform_sample(array: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if len(array) == target_len:
        return array
    idx = np.linspace(0, len(array) - 1, num=target_len, dtype=int)
    return array[idx]


@dataclass
class MotionRecord:
    path: str
    split: str
    source: str
    name: str
    length: Optional[int] = None
    nfeats: Optional[int] = None


class LargeMotionDataset(Dataset):
    """
    Generic large-scale motion dataset for VAE pretraining.

    Supported manifest formats:
    - .jsonl: one JSON object per line with keys path/split/source/name/length/nfeats
    - .tsv:  header required, with columns at least: path, split, source, name
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        mean,
        std,
        max_motion_length: int,
        min_motion_length: int,
        unit_length: int,
        nfeats: int,
        normalize: bool = True,
        sample_stride: int = 1,
        fallback_to_train: bool = True,
        cache_in_memory: bool = False,
        debug: bool = False,
        source_default: str = "generic",
        **kwargs,
    ):
        self.manifest_path = manifest_path
        self.split = split
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.nfeats = nfeats
        self.normalize = normalize
        self.sample_stride = max(int(sample_stride), 1)
        self.cache_in_memory = cache_in_memory
        self.debug = debug
        self.source_default = source_default
        self.metric_source_allowlist = {"how2sign", "csl", "phoenix"}
        if self.source_default not in self.metric_source_allowlist:
            self.source_default = "how2sign"

        if max_motion_length % unit_length != 0 or min_motion_length % unit_length != 0:
            raise ValueError(
                f"max/min length must be divisible by unit_length. "
                f"Got max={max_motion_length}, min={min_motion_length}, unit={unit_length}"
            )

        records = self._load_manifest(manifest_path)
        split_records = [x for x in records if x.split == split]

        if not split_records and fallback_to_train and split != "train":
            split_records = [x for x in records if x.split == "train"]

        if not split_records:
            split_records = records

        if self.sample_stride > 1:
            split_records = split_records[:: self.sample_stride]

        if debug:
            split_records = split_records[: min(len(split_records), 2048)]

        self.records: List[MotionRecord] = split_records
        self._cache: Dict[int, np.ndarray] = {}

        if len(self.records) == 0:
            raise RuntimeError(
                f"No valid records found in manifest={manifest_path} for split={split}."
            )

    @staticmethod
    def _load_manifest(manifest_path: str) -> List[MotionRecord]:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        records: List[MotionRecord] = []

        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if "path" not in item:
                        continue
                    records.append(
                        MotionRecord(
                            path=item["path"],
                            split=item.get("split", "train"),
                            source=item.get("source", "generic"),
                            name=item.get("name", Path(item["path"]).stem),
                            length=item.get("length"),
                            nfeats=item.get("nfeats"),
                        )
                    )
        elif path.suffix == ".tsv":
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for item in reader:
                    if "path" not in item or not item["path"]:
                        continue
                    records.append(
                        MotionRecord(
                            path=item["path"],
                            split=item.get("split", "train"),
                            source=item.get("source", "generic"),
                            name=item.get("name", Path(item["path"]).stem),
                            length=int(item["length"]) if item.get("length") else None,
                            nfeats=int(item["nfeats"]) if item.get("nfeats") else None,
                        )
                    )
        else:
            raise ValueError(
                f"Unsupported manifest extension: {path.suffix}. Use .jsonl or .tsv"
            )

        return records

    def __len__(self):
        return len(self.records)

    def _load_array(self, idx: int) -> np.ndarray:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        rec = self.records[idx]
        arr = np.load(rec.path, mmap_mode="r")
        arr = np.asarray(arr)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array [T, C], got shape={arr.shape}, path={rec.path}")

        if arr.shape[-1] != self.nfeats:
            raise ValueError(
                f"Feature dim mismatch for {rec.path}. Got {arr.shape[-1]}, expected {self.nfeats}."
            )

        arr = arr.astype(np.float32, copy=False)

        if self.cache_in_memory:
            self._cache[idx] = arr

        return arr

    def __getitem__(self, idx):
        rec = self.records[idx]
        clip = self._load_array(idx)

        m_length = clip.shape[0]
        if m_length < self.min_motion_length:
            clip = _uniform_sample(clip, self.min_motion_length)
        elif m_length > self.max_motion_length:
            clip = _uniform_sample(clip, self.max_motion_length)
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            start = (clip.shape[0] - m_length) // 2
            clip = clip[start: start + m_length]

        m_length = clip.shape[0]

        if self.normalize:
            clip = (clip - self.mean) / (self.std + 1e-10)

        # Keep output tuple layout compatible with `humanml3d_collate`.
        # Existing metrics in this repo only register source keys:
        # how2sign / csl / phoenix.
        # For generic large-scale corpora (e.g., motionx_*), map unknown
        # sources into source_default (recommended: how2sign) to avoid
        # metric state-key errors during sanity-check/validation.
        src = rec.source if rec.source in self.metric_source_allowlist else self.source_default
        if src not in self.metric_source_allowlist:
            src = "how2sign"

        return (
            None,
            torch.from_numpy(clip).float(),
            int(m_length),
            rec.name,
            None,
            None,
            None,
            None,
            None,
            src,
        )


class LargeMotionDataModule(BASEDataModule):
    """
    DataModule for large-scale VAE pretraining on unified [T, C] npy features.
    """

    def __init__(self, cfg, phase="train", **kwargs):
        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)

        if "LARGE" not in cfg.DATASET:
            raise ValueError("Expected DATASET.LARGE config for LargeMotionDataModule")

        large_cfg = cfg.DATASET.LARGE

        # Use a dedicated dataset name so BaseMetrics does not instantiate
        # text-related metrics (which require w_vectorizer).
        self.name = "large_motion"
        self.njoints = 22
        cfg.DATASET.JOINT_TYPE = "humanml3d"

        self.nfeats = int(cfg.DATASET.NFEATS)
        cfg.DATASET.NFEATS = self.nfeats

        manifest_path = large_cfg.MANIFEST
        if not manifest_path:
            raise ValueError("DATASET.LARGE.MANIFEST must be set")

        mean_path = large_cfg.get("MEAN_PATH", "")
        std_path = large_cfg.get("STD_PATH", "")
        if mean_path and std_path and os.path.exists(mean_path) and os.path.exists(std_path):
            mean = self._load_stats_file(mean_path)
            std = self._load_stats_file(std_path)
        else:
            mean = np.zeros((self.nfeats,), dtype=np.float32)
            std = np.ones((self.nfeats,), dtype=np.float32)

        if mean.shape[0] != self.nfeats or std.shape[0] != self.nfeats:
            raise ValueError(
                f"mean/std dim mismatch with DATASET.NFEATS={self.nfeats}. "
                f"mean={mean.shape}, std={std.shape}"
            )

        self.hparams.manifest_path = manifest_path
        # Keep compatibility with places that expect this key on datamodule.hparams.
        self.hparams.w_vectorizer = None
        self.hparams.mean = mean
        self.hparams.std = std
        self.hparams.mean_eval = mean
        self.hparams.std_eval = std
        self.hparams.max_motion_length = int(large_cfg.MAX_MOTION_LEN)
        self.hparams.min_motion_length = int(large_cfg.MIN_MOTION_LEN)
        self.hparams.unit_length = int(large_cfg.UNIT_LEN)
        self.hparams.nfeats = self.nfeats
        self.hparams.normalize = bool(large_cfg.get("NORMALIZE", True))
        self.hparams.sample_stride = int(large_cfg.get("SAMPLE_STRIDE", 1))
        self.hparams.fallback_to_train = bool(large_cfg.get("FALLBACK_TO_TRAIN", True))
        self.hparams.cache_in_memory = bool(large_cfg.get("CACHE_IN_MEMORY", False))
        self.hparams.debug = bool(cfg.DEBUG)
        self.hparams.source_default = str(large_cfg.get("SOURCE_DEFAULT", "generic"))

        self.Dataset = LargeMotionDataset
        self.DatasetEval = LargeMotionDataset

    @staticmethod
    def _load_stats_file(path: str) -> np.ndarray:
        suffix = Path(path).suffix.lower()
        if suffix in [".npy", ".npz"]:
            arr = np.load(path)
        else:
            # Keep compatibility with previous torch.save(mean.pt/std.pt) files.
            arr = torch.load(path, map_location="cpu")
        return np.asarray(arr, dtype=np.float32)

    def _persistent_workers_flag(self, num_workers: int) -> bool:
        return num_workers > 0

    def train_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.TRAIN.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TRAIN.NUM_WORKERS
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            persistent_workers=self._persistent_workers_flag(self.cfg.TRAIN.NUM_WORKERS),
            **dataloader_options,
        )

    def val_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.EVAL.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.val_dataset,
            persistent_workers=self._persistent_workers_flag(self.cfg.EVAL.NUM_WORKERS),
            **dataloader_options,
        )

    def test_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=self._persistent_workers_flag(self.cfg.TEST.NUM_WORKERS),
            **dataloader_options,
        )

    def feats2joints(self, features):
        # Match H2S behavior so MR/TM2T metrics still work on 133-dim SMPL-X features.
        mean = torch.tensor(self.hparams.mean, device=features.device, dtype=features.dtype)
        std = torch.tensor(self.hparams.std, device=features.device, dtype=features.dtype)
        features = features * std + mean

        zero_pose = torch.zeros(*features.shape[:-1], 36, device=features.device, dtype=features.dtype)
        shape_param = torch.tensor(
            [[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172,
               0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]],
            device=features.device,
            dtype=features.dtype,
        )

        bsz, tlen = features.shape[:2]
        shape_param = shape_param.repeat(bsz, tlen, 1).view(bsz * tlen, -1)
        features = torch.cat([zero_pose, features], dim=-1).view(bsz * tlen, -1)

        vertices, joints = get_coord(
            root_pose=features[..., 0:3],
            body_pose=features[..., 3:66],
            lhand_pose=features[..., 66:111],
            rhand_pose=features[..., 111:156],
            jaw_pose=features[..., 156:159],
            shape=shape_param,
            expr=features[..., 159:169],
        )
        return vertices, joints

    def renorm4t2m(self, features):
        ori_mean = torch.tensor(self.hparams.mean, device=features.device, dtype=features.dtype)
        ori_std = torch.tensor(self.hparams.std, device=features.device, dtype=features.dtype)
        eval_mean = torch.tensor(self.hparams.mean_eval, device=features.device, dtype=features.dtype)
        eval_std = torch.tensor(self.hparams.std_eval, device=features.device, dtype=features.dtype)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / (eval_std + 1e-10)
        return features

    def mm_mode(self, mm_on=True):
        # Keep interface compatibility.
        self.is_mm = bool(mm_on)
