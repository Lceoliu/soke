# PyTorch 2.6+ defaults to weights_only=True in torch.load, but Lightning
# checkpoints contain OmegaConf objects (ListConfig, DictConfig,
# ContainerMetadata, …) that fail safe-unpickling. Since all checkpoints
# are produced by our own training pipeline, we patch Lightning's loader
# to always use weights_only=False.
#
# Import this module early (before any trainer.fit / trainer.test call)
# to apply the patch:
#
#     import mGPT.utils.compat  # noqa: F401

import lightning_fabric.utilities.cloud_io as _cloud_io

_orig_load = _cloud_io._load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


_cloud_io._load = _patched_load
