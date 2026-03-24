import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
from os.path import join as pjoin
import os
import glob

# Keys that may be safely overridden when resuming from a checkpoint.
# Everything else is restored from the original training config.
_RESUME_SAFE_OVERRIDES = frozenset({
    "USE_GPUS",
    "DEVICE",
    "NUM_NODES",
    "PRECISION",
    "ACCELERATOR",
    "DEBUG",
    "TRAIN.END_EPOCH",
    "TRAIN.BATCH_SIZE",
    "TRAIN.NUM_WORKERS",
    "TRAIN.ACCUMULATE_GRAD_BATCHES",
    "TRAIN.NUM_SANITY_VAL_STEPS",
    "TRAIN.DDP_FIND_UNUSED_PARAMETERS",
    "EVAL.BATCH_SIZE",
    "EVAL.NUM_WORKERS",
    "EVAL.DISABLE_VAL",
    "TEST.BATCH_SIZE",
    "TEST.NUM_WORKERS",
    "LOGGER.VAL_EVERY_STEPS",
})


def get_module_config(cfg, filepath="./configs"):
    """
    Load yaml config files from subfolders
    """

    yamls = glob.glob(pjoin(filepath, '*', '*.yaml'))
    yamls = [y.replace(filepath, '') for y in yamls]
    for yaml in yamls:
        nodes = yaml.replace('.yaml', '').replace(os.sep, '.')
        nodes = nodes[1:] if nodes[0] == '.' else nodes
        OmegaConf.update(cfg, nodes, OmegaConf.load('./configs' + yaml))

    return cfg


def get_obj_from_str(string, reload=False):
    """
    Get object from string
    """

    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Instantiate object from config
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def resume_config(cfg: OmegaConf):
    """
    On resume: load the original training config from the experiment directory
    inferred from the checkpoint path, then apply only safe runtime overrides
    from the current (CLI-supplied) config.

    Semantics:
      TRAIN.RESUME    – checkpoint FILE path for full resume (weights + optimizer + epoch)
      TRAIN.PRETRAINED – checkpoint FILE path for weight-only init
      These two are mutually exclusive.
    """
    resume_path = str(cfg.TRAIN.get("RESUME", "") or "")
    pretrained_path = str(cfg.TRAIN.get("PRETRAINED", "") or "")

    if resume_path and pretrained_path:
        raise ValueError(
            "TRAIN.RESUME and TRAIN.PRETRAINED are mutually exclusive. "
            f"Got RESUME={resume_path}, PRETRAINED={pretrained_path}"
        )

    if not resume_path:
        return cfg

    if not os.path.isfile(resume_path):
        raise ValueError(
            f"TRAIN.RESUME must point to a checkpoint file, got: {resume_path}"
        )

    # Infer experiment dir: .../checkpoints/last.ckpt → .../
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(resume_path)))

    # Load original training config
    config_yamls = sorted(glob.glob(pjoin(exp_dir, "config_*_train.yaml")))
    if not config_yamls:
        raise FileNotFoundError(
            f"No config_*_train.yaml found in {exp_dir}. "
            "Cannot resume without the original training config."
        )
    original_cfg = OmegaConf.load(config_yamls[-1])

    # Apply safe runtime overrides from current config
    for key in _RESUME_SAFE_OVERRIDES:
        try:
            val = OmegaConf.select(cfg, key)
            if val is not None:
                OmegaConf.update(original_cfg, key, val)
        except Exception:
            pass

    # Keep LR scheduler horizon in sync with the (possibly overridden) END_EPOCH
    try:
        new_end_epoch = OmegaConf.select(original_cfg, "TRAIN.END_EPOCH")
        if new_end_epoch is not None and OmegaConf.select(original_cfg, "TRAIN.LR_SCHEDULER.params.T_max") is not None:
            OmegaConf.update(original_cfg, "TRAIN.LR_SCHEDULER.params.T_max", new_end_epoch)
    except Exception:
        pass

    # Preserve dataset/path config from the current run (paths may differ across machines)
    if "DATASET" in cfg:
        original_cfg.DATASET = cfg.DATASET

    # Force resume-specific fields
    original_cfg.TRAIN.RESUME = resume_path
    original_cfg.TRAIN.PRETRAINED = ""
    OmegaConf.update(original_cfg, "FOLDER_EXP", exp_dir)

    # Recover wandb run ID for seamless logging continuation
    wandb_dir = pjoin(exp_dir, "wandb", "latest-run")
    if os.path.isdir(wandb_dir):
        try:
            wandb_files = os.listdir(wandb_dir)
            wandb_run = [f for f in wandb_files if "run-" in f][0]
            original_cfg.LOGGER.WANDB.params.id = (
                wandb_run.replace("run-", "").replace(".wandb", "")
            )
        except (IndexError, OSError):
            pass

    return original_cfg

def parse_args(phase="train"):
    """
    Parse arguments and load config files
    """

    parser = ArgumentParser()
    group = parser.add_argument_group("Training options")

    # Assets
    group.add_argument(
        "--cfg_assets",
        type=str,
        required=False,
        default="./configs/assets.yaml",
        help="config file for asset paths",
    )

    # Default config
    if phase in ["train", "test", "demo"]:
        cfg_defualt = "./configs/default.yaml"
    elif phase == "render":
        cfg_defualt = "./configs/render.yaml"
    elif phase == "webui":
        cfg_defualt = "./configs/webui.yaml"
        
    group.add_argument(
        "--cfg",
        type=str,
        required=False,
        default=cfg_defualt,
        help="config file",
    )
    group.add_argument("--use_gpus",
                           type=str,
                           required=False,
                           default='1,2,3,4,5,6,7',
                           help="cuda environ devices")
    
    # Parse for each phase
    if phase in ["train", "test"]:
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--num_nodes",
                           type=int,
                           required=False,
                           help="number of nodes")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--task",
                           type=str,
                           required=False,
                           help="evaluation task type")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")


    if phase == "demo":
        group.add_argument("--task",
            type=str,
            required=False,
            help="evaluation task type")
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--out_dir",
            type=str,
            required=False,
            help="output dir",
        )
        group.add_argument(
            "--demo_dataset",
            default=None,
            type=str,
            required=False,
            help="output dir",
        )

    if phase == "render":
        group.add_argument("--npy",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion files")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion folder")
        group.add_argument("--fps",
                    type=int,
                    required=False,
                    default=30,
                    help="render fps")
        group.add_argument(
            "--mode",
            type=str,
            required=False,
            default="sequence",
            help="render target: video, sequence, frame",
        )

    params = parser.parse_args()
    
    # Load yaml config files
    OmegaConf.register_new_resolver("eval", eval)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    cfg.USE_GPUS = params.use_gpus
    # Update config with arguments
    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
        cfg.DEVICE = params.device if params.device else (cfg.DEVICE if "DEVICE" in cfg else [0])
        cfg.NUM_NODES = params.num_nodes if params.num_nodes else (cfg.NUM_NODES if "NUM_NODES" in cfg else 1)
        cfg.model.params.task = params.task if params.task else cfg.model.params.task
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG

        # Force no debug in test
        if phase == "test":
            cfg.DEBUG = False
            if params.batch_size:
                cfg.EVAL.BATCH_SIZE = params.batch_size
                cfg.TEST.BATCH_SIZE = params.batch_size
            # cfg.DEVICE = [0]
            print("Force no debugging when testing")

    if phase == "demo":
        cfg.DEMO_DATASET = params.demo_dataset
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.TASK = params.task
        cfg.TEST.FOLDER = params.out_dir if params.out_dir else cfg.TEST.FOLDER
        os.makedirs(cfg.TEST.FOLDER, exist_ok=True)

    if phase == "render":
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        if params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        if params.fps:
            cfg.RENDER.FPS = float(params.fps)
        cfg.RENDER.MODE = params.mode

    # Debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.params.offline = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1
        
    # Resume config (only applies during training)
    if phase == "train":
        cfg = resume_config(cfg)

    return cfg
