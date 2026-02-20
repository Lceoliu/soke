import torch
from collections import OrderedDict
from mGPT.utils.misc import neq_load_customized


def load_pretrained(cfg, model, logger=None, phase="train"):    
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    
    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
        
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model


def _read_state_dict(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def _extract_module_state(state_dict, prefixes):
    for prefix in prefixes:
        module_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                module_dict[k[len(prefix):]] = v
        if len(module_dict) > 0:
            return module_dict, prefix
    return OrderedDict(), None


def load_pretrained_vae(cfg, model, logger=None):
    train_cfg = cfg.TRAIN
    default_path = str(train_cfg.get("PRETRAINED_VAE", "") or "")

    body_path = str(train_cfg.get("PRETRAINED_VAE_BODY", default_path) or "")
    hand_path = str(train_cfg.get("PRETRAINED_VAE_HAND", default_path) or "")
    rhand_path = str(train_cfg.get("PRETRAINED_VAE_RHAND", default_path) or "")

    load_body = bool(train_cfg.get("PRETRAINED_VAE_LOAD_BODY", True))
    load_hand = bool(train_cfg.get("PRETRAINED_VAE_LOAD_HAND", True))
    load_rhand = bool(train_cfg.get("PRETRAINED_VAE_LOAD_RHAND", True))

    if logger is not None:
        logger.info(
            "Loading VAE modules with settings: "
            f"body(load={load_body}, path='{body_path}'), "
            f"hand(load={load_hand}, path='{hand_path}'), "
            f"rhand(load={load_rhand}, path='{rhand_path}')"
        )

    ckpt_cache = {}

    def get_state_dict(path):
        if not path:
            return None
        if path not in ckpt_cache:
            ckpt_cache[path] = _read_state_dict(path)
        return ckpt_cache[path]

    def maybe_load_module(module_name, module_attr, enabled, path, prefixes):
        if not enabled:
            if logger is not None:
                logger.info(f"Skip loading {module_name}: disabled by config")
            return
        if not hasattr(model, module_attr):
            if logger is not None:
                logger.info(f"Skip loading {module_name}: model has no '{module_attr}'")
            return
        if not path:
            if logger is not None:
                logger.info(f"Skip loading {module_name}: empty checkpoint path")
            return

        state_dict = get_state_dict(path)
        module_dict, prefix_used = _extract_module_state(state_dict, prefixes)
        if len(module_dict) == 0:
            raise RuntimeError(
                f"Failed to load {module_name} from {path}: no keys found for prefixes {prefixes}"
            )

        if logger is not None:
            logger.info(f"Loading {module_name} from {path} (prefix={prefix_used})")

        neq_load_customized(getattr(model, module_attr), module_dict, verbose=True)

    maybe_load_module("body_vae", "vae", load_body, body_path, ["motion_vae.", "vae."])
    maybe_load_module("hand_vae", "hand_vae", load_hand, hand_path, ["hand_vae."])
    maybe_load_module("rhand_vae", "rhand_vae", load_rhand, rhand_path, ["rhand_vae."])

    return model
