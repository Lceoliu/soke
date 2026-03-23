import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np


class BASEDataModule(pl.LightningDataModule):
    def __init__(self, collate_fn):
        super().__init__()

        self.dataloader_options = {"collate_fn": collate_fn}
        self.persistent_workers = True
        self.is_mm = False

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def _maybe_subset_eval_dataset(self, dataset, split="val"):
        if split != "val":
            return dataset
        ratio = float(self.cfg.EVAL.get("VAL_SUBSET_RATIO", 1.0))
        max_samples = int(self.cfg.EVAL.get("VAL_SUBSET_MAX_SAMPLES", 0) or 0)
        if ratio >= 1.0 and max_samples <= 0:
            return dataset

        total = len(dataset)
        if total <= 0:
            return dataset

        target = total
        if ratio < 1.0:
            target = max(1, int(total * max(ratio, 0.0)))
        if max_samples > 0:
            target = min(target, max_samples)
        target = min(target, total)
        if target >= total:
            return dataset

        # Deterministic evenly spaced subset to keep validation comparable
        # across epochs while still covering the full dataset distribution.
        indices = np.linspace(0, total - 1, num=target, dtype=int).tolist()
        return Subset(dataset, indices)

    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.DatasetEval(**sample_params)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.Dataset(split=self.cfg.TRAIN.SPLIT,
                                               **self.hparams)
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            params = self.hparams.copy()
            if str(self.cfg.TRAIN.STAGE).startswith("lm"):
                params['code_path'] = getattr(self.hparams, 'code_path', None)
            else:
                params['code_path'] = None
            params['split'] = self.cfg.EVAL.SPLIT
            self._val_dataset = self.DatasetEval(**params)
        return self._val_dataset

    @property
    def test_dataset(self):
        if self._test_dataset is None:
            # self._test_dataset = self.DatasetEval(split=self.cfg.TEST.SPLIT,
            #                                       **self.hparams)
            params = self.hparams.copy()
            if str(self.cfg.TRAIN.STAGE).startswith("lm"):
                params['code_path'] = getattr(self.hparams, 'code_path', None)
            else:
                params['code_path'] = None
            params['split'] = self.cfg.TEST.SPLIT
            self._test_dataset = self.DatasetEval( **params)
        return self._test_dataset

    def setup(self, stage=None):
        # Use the getter the first time to load the data
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset

    def train_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.TRAIN.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TRAIN.NUM_WORKERS
        num_workers = int(dataloader_options["num_workers"])
        return DataLoader(
            self.train_dataset,
            shuffle=False,
            persistent_workers=(num_workers > 0),
            **dataloader_options,
        )

    def predict_dataloader(self):
        dataloader_options = self.dataloader_options.copy()
        dataloader_options[
            "batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        num_workers = int(dataloader_options["num_workers"])
        return DataLoader(
            self.test_dataset,
            persistent_workers=(num_workers > 0),
            **dataloader_options,
        )

    def val_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.EVAL.NUM_WORKERS
        dataloader_options["shuffle"] = False
        num_workers = int(dataloader_options["num_workers"])
        return DataLoader(
            self._maybe_subset_eval_dataset(self.val_dataset, split="val"),
            persistent_workers=(num_workers > 0),
            **dataloader_options,
        )

    def test_dataloader(self):
        # overrides batch_size and num_workers
        dataloader_options = self.dataloader_options.copy()
        dataloader_options[
            "batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        dataloader_options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        dataloader_options["shuffle"] = False
        num_workers = int(dataloader_options["num_workers"])
        return DataLoader(
            self.test_dataset,
            persistent_workers=(num_workers > 0),
            **dataloader_options,
        )
