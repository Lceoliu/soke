from pathlib import Path
import os
import time
import logging
from omegaconf import OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_only

def create_logger(cfg, phase='train'):
    # root dir set by cfg
    root_output_dir = Path(cfg.FOLDER)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True, exist_ok=True)

    # On resume, FOLDER_EXP is already set by resume_config; reuse it.
    is_resume = bool(cfg.TRAIN.get("RESUME", ""))
    if is_resume and "FOLDER_EXP" in cfg and cfg.FOLDER_EXP:
        final_output_dir = Path(cfg.FOLDER_EXP)
    else:
        cfg_name = cfg.NAME
        model = cfg.model.target.split('.')[-2]
        cfg_name = os.path.basename(cfg_name).split('.')[0]
        final_output_dir = root_output_dir / model / cfg_name
        cfg.FOLDER_EXP = str(final_output_dir)

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')

    new_dir(cfg, phase, time_str, final_output_dir)

    # Re-read FOLDER_EXP: new_dir may update it on directory collision
    final_output_dir = Path(cfg.FOLDER_EXP)

    head = '%(asctime)-15s %(message)s'
    logger = config_logger(final_output_dir, time_str, phase, head)
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)
        logging.basicConfig(format=head)
    return logger


@rank_zero_only
def config_logger(final_output_dir, time_str, phase, head):
    log_file = '{}_{}_{}.log'.format('log', time_str, phase)
    final_log_file = final_output_dir / log_file
    logging.basicConfig(filename=str(final_log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    file_handler = logging.FileHandler(final_log_file, 'w')
    file_handler.setFormatter(logging.Formatter(head))
    file_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(file_handler)
    return logger


@rank_zero_only
def new_dir(cfg, phase, time_str, final_output_dir):
    cfg.TIME = str(time_str)
    is_resume = bool(cfg.TRAIN.get("RESUME", ""))

    if (
        os.path.exists(final_output_dir)
        and not is_resume
        and not cfg.DEBUG
        and phase not in ['test', 'demo']
    ):
        # Directory collision: give the NEW experiment a timestamp suffix
        # so the old experiment directory stays untouched.
        has_log = any(
            item.endswith('.log')
            for item in os.listdir(final_output_dir)
        )
        if has_log:
            new_path = str(final_output_dir) + '_' + time_str
            final_output_dir = Path(new_path)
            cfg.FOLDER_EXP = str(final_output_dir)

    final_output_dir.mkdir(parents=True, exist_ok=True)
    # write config yaml
    # Use a distinct prefix for resume runs so that resume_config() always
    # picks the original training config via "config_*_train.yaml".
    prefix = "resumed" if is_resume else "config"
    config_file = '{}_{}_{}.yaml'.format(prefix, time_str, phase)
    final_config_file = final_output_dir / config_file
    OmegaConf.save(config=cfg, f=final_config_file)
