import numpy as np
import torch
import os 
from os.path import join as pjoin
from torch.utils.data import DataLoader, Dataset
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, H2SMotionDatasetVQ, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T
from .utils import humanml3d_collate
from mGPT.utils.human_models import get_coord


class FixedTaskDataset(Dataset):
    def __init__(self, base_dataset, task_name):
        self.base_dataset = base_dataset
        self.task_name = str(task_name).lower()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = list(self.base_dataset[idx])
        if len(item) < 10:
            raise ValueError(f"Unexpected evaluation item length: {len(item)}")
        item[8] = {"class": self.task_name}
        return tuple(item)


class H2SDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'humanml3d'
        self.name = "humanml3d"
        self.njoints = 22

        self.hparams.dataset_name = cfg.DATASET.H2S.DATASET_NAME
        self.hparams.csl_root = cfg.DATASET.H2S.CSL_ROOT
        self.hparams.phoenix_root = cfg.DATASET.H2S.get('PHOENIX_ROOT', None)
        self.hparams.use_contact_labels = bool(cfg.DATASET.H2S.get('USE_CONTACT_LABELS', False))
        self.hparams.contact_dir_name = str(cfg.DATASET.H2S.get('CONTACT_DIR_NAME', 'contact_labels'))
        self.hparams.pred_data_dir = cfg.DATASET.H2S.get('pred_data_dir', False)
        self.hparams.dynamic_task_sampling = bool(cfg.DATASET.H2S.get('DYNAMIC_TASK_SAMPLING', False))
        self.hparams.train_task_classes = list(cfg.DATASET.H2S.get('TRAIN_TASKS', ['t2m', 'm2t', 'mc']))
        self.hparams.task_sampling = dict(cfg.DATASET.H2S.get(
            'TASK_SAMPLING',
            {'t2m': 0.5, 'm2t': 0.2, 'mc': 0.3},
        ))
        self.hparams.enable_token_drop_aug = bool(cfg.DATASET.H2S.get('ENABLE_TOKEN_DROP_AUG', True))
        self.hparams.lm_val_tasks = list(cfg.EVAL.get('LM_VAL_TASKS', ['t2m', 'm2t', 'mc']))
        
        # Path to the dataset
        data_root = cfg.DATASET.H2S.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "re_aligned")
        self.hparams.motion_dir = pjoin(data_root, "poses")
        
        # Mean and std of the dataset
        # if self.hparams.dataset_name == 'how2sign':
        #     mean_path = pjoin(data_root, "h2s_mean.pt")
        #     std_path = pjoin(data_root, "h2s_std.pt")
        # elif self.hparams.dataset_name == 'csl':
        #     mean_path = pjoin(self.hparams.csl_root, "csl_mean.pt")
        #     std_path = pjoin(self.hparams.csl_root, "csl_std.pt")
        # elif self.hparams.dataset_name == 'how2sign_csl':
        #     mean_path = pjoin(self.hparams.csl_root, "h2s_csl_mean.pt")
        #     std_path = pjoin(self.hparams.csl_root, "h2s_csl_std.pt")
        mean_path = cfg.DATASET.H2S.MEAN_PATH
        std_path = cfg.DATASET.H2S.STD_PATH
        print('mean path', mean_path, 'std_path: ', std_path)

        self.hparams.mean = torch.load(mean_path)
        self.hparams.std = torch.load(std_path)
        # filter out unwanted joints
        self.hparams.mean = self.hparams.mean[(3+3*11):]
        self.hparams.mean = torch.cat([self.hparams.mean[:-20], self.hparams.mean[-10:]], dim=0)
        self.hparams.std = self.hparams.std[(3+3*11):]
        self.hparams.std = torch.cat([self.hparams.std[:-20], self.hparams.std[-10:]], dim=0)
        
        # Mean and std for fair evaluation
        # dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "Comp_v6_KLD01", "meta")
        # self.hparams.mean_eval = np.load(pjoin(dis_data_root_eval, "mean.npy"))
        # self.hparams.std_eval = np.load(pjoin(dis_data_root_eval, "std.npy"))
        self.hparams.mean_eval = self.hparams.mean
        self.hparams.std_eval = self.hparams.std
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.H2S.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.H2S.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.H2S.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.H2S.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        self.hparams.lm_token_num_quantizers = 1
        self.hparams.lm_token_num_parts = 1
        self.hparams.lm_body_codebook_size = 0
        self.hparams.lm_hand_codebook_size = 0
        self.hparams.lm_rhand_codebook_size = 0
        self.hparams.lm_q_offset_mode = "per_q_offset_v1"

        def _extract_num_quantizers(module_cfg, default_q=1):
            if module_cfg is None:
                return int(default_q)
            try:
                params = module_cfg.get("params", None)
            except Exception:
                params = None
            if params is None:
                return int(default_q)
            try:
                return int(params.get("num_quantizers", default_q))
            except Exception:
                return int(default_q)

        def _extract_codebook_size(module_cfg, default_code_num=0):
            if module_cfg is None:
                return int(default_code_num)
            try:
                params = module_cfg.get("params", None)
            except Exception:
                params = None
            if params is None:
                return int(default_code_num)
            try:
                return int(params.get("code_num", default_code_num))
            except Exception:
                return int(default_code_num)

        # Dataset switch
        self.DatasetEval = H2SMotionDatasetVQ if cfg.TRAIN.STAGE in ["vae"] else Text2MotionDatasetEval

        if cfg.TRAIN.STAGE in ["vae"]:
            # if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
            self.hparams.win_size = 64
            self.Dataset = H2SMotionDatasetVQ
            # else:
                # self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.H2S.STD_TEXT

            # LM token files may already be flattened from [T, Q, P] to [T*Q, P].
            # We pass shared Q and active part count so dataset length clipping stays correct.
            model_params = cfg.model.params
            body_q = _extract_num_quantizers(model_params.get("motion_vae", None), default_q=1)
            q_list = [body_q]
            num_parts = 1
            body_code_num = _extract_codebook_size(model_params.get("motion_vae", None), default_code_num=0)
            hand_code_num = body_code_num
            rhand_code_num = body_code_num
            if model_params.get("hand_vae_cfg", None) is not None:
                q_list.append(_extract_num_quantizers(model_params.get("hand_vae_cfg"), default_q=body_q))
                num_parts += 1
                hand_code_num = _extract_codebook_size(model_params.get("hand_vae_cfg"), default_code_num=body_code_num)
            if model_params.get("rhand_vae_cfg", None) is not None:
                q_list.append(_extract_num_quantizers(model_params.get("rhand_vae_cfg"), default_q=body_q))
                num_parts += 1
                rhand_code_num = _extract_codebook_size(model_params.get("rhand_vae_cfg"), default_code_num=body_code_num)
            self.hparams.lm_token_num_quantizers = int(min(q_list))
            self.hparams.lm_token_num_parts = int(num_parts)
            self.hparams.lm_body_codebook_size = int(body_code_num)
            self.hparams.lm_hand_codebook_size = int(hand_code_num)
            self.hparams.lm_rhand_codebook_size = int(rhand_code_num)
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset

        # Get additional info of the dataset
        # self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        self.nfeats = 133  #self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats
        self._val_task_datasets = None

    @property
    def val_task_datasets(self):
        if self._val_task_datasets is None:
            self._val_task_datasets = [
                FixedTaskDataset(self.val_dataset, task_name)
                for task_name in self.hparams.lm_val_tasks
            ]
        return self._val_task_datasets

    def get_val_task_name(self, dataloader_idx: int) -> str:
        task_names = list(self.hparams.lm_val_tasks)
        if len(task_names) == 0:
            return str(self.cfg.model.params.task)
        if dataloader_idx < 0 or dataloader_idx >= len(task_names):
            return task_names[0]
        return str(task_names[dataloader_idx])
        

    def feats2joints(self, features):
        #smpl2joints and drop lowerbody
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = features * std + mean
        # return recover_from_ric(features, self.njoints)

        zero_pose = torch.zeros(*features.shape[:-1], 36).to(features)
        shape_param = torch.tensor([[[-0.07284723, 0.1795129, -0.27608207, 0.135155, 0.10748172, 
                              0.16037364, -0.01616933, -0.03450319, 0.01369138, 0.01108842]]]).to(features)
        B, T = features.shape[:2]
        shape_param = shape_param.repeat(B, T, 1).view(B*T, -1)
        # print(features.shape, shape_param.shape)
        features = torch.cat([zero_pose, features], dim=-1).view(B*T, -1)  #133+36=169
        vertices, joints = get_coord(root_pose=features[..., 0:3], body_pose=features[..., 3:66], 
                                     lhand_pose=features[..., 66:111], rhand_pose=features[..., 111:156], 
                                     jaw_pose=features[..., 156:159], shape=shape_param, 
                                     expr=features[..., 159:169])
        return vertices, joints

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = self.hparams.mean.to(features)
        ori_std = self.hparams.std.to(features)
        eval_mean = self.hparams.mean_eval.to(features)
        eval_std = self.hparams.std_eval.to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list

    def val_dataloader(self):
        if self.cfg.TRAIN.STAGE in ['lm_pretrain', 'lm_instruct', 'lm_rl'] and len(self.hparams.lm_val_tasks) > 1:
            dataloader_options = self.dataloader_options.copy()
            dataloader_options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
            dataloader_options["num_workers"] = self.cfg.EVAL.NUM_WORKERS
            dataloader_options["shuffle"] = False
            num_workers = int(dataloader_options["num_workers"])
            return [
                DataLoader(
                    self._maybe_subset_eval_dataset(dataset, split="val"),
                    persistent_workers=(num_workers > 0),
                    **dataloader_options,
                )
                for dataset in self.val_task_datasets
            ]
        return super().val_dataloader()
