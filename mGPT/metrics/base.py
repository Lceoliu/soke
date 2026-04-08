from torch import Tensor, nn
from os.path import join as pjoin
from .mr import MRMetrics
from .t2m import TM2TMetrics
from .mm import MMMetrics
from .m2t import M2TMetrics
from .m2m import PredMetrics
from .mc_token import MCTokenMetrics


class BaseMetrics(nn.Module):
    def __init__(self, cfg, datamodule, debug, **kwargs) -> None:
        super().__init__()

        njoints = datamodule.njoints

        data_name = datamodule.name
        metric_types = list(cfg.METRIC.get("TYPE", []))
        # Only load T2M/MM evaluators when explicitly requested and the required
        # checkpoint (deps/.../finest.tar) is available.  For m2t-only runs on
        # H2S/CSL-Daily these evaluators are not needed.
        _need_tm2t = data_name in ["humanml3d", "kit"] and any(
            t in metric_types for t in ["TM2TMetrics", "MMMetrics", "MCMetrics"]
        )
        _need_m2t  = data_name in ["humanml3d", "kit"] and "M2TMetrics" in metric_types
        if data_name in ["humanml3d", "kit"]:
            if _need_tm2t:
                self.TM2TMetrics = TM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
                self.MCMetrics = TM2TMetrics(
                    cfg=cfg,
                    dataname=data_name,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                    metric_prefix="mc_",
                )
                self.MCTokenMetrics = MCTokenMetrics(
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
                self.MMMetrics = MMMetrics(
                    cfg=cfg,
                    mm_num_times=cfg.METRIC.MM_NUM_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
                )
            if _need_m2t:
                self.M2TMetrics = M2TMetrics(
                    cfg=cfg,
                    w_vectorizer=datamodule.hparams.w_vectorizer,
                    diversity_times=30 if debug else cfg.METRIC.DIVERSITY_TIMES,
                    dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP)

        self.MRMetrics = MRMetrics(
            njoints=njoints,
            jointstype=cfg.DATASET.JOINT_TYPE,
            dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
        )
        self.PredMetrics = PredMetrics(
            cfg=cfg,
            njoints=njoints,
            jointstype=cfg.DATASET.JOINT_TYPE,
            dist_sync_on_step=cfg.METRIC.DIST_SYNC_ON_STEP,
            task=cfg.model.params.task,
        )
