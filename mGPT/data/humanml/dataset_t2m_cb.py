import rich
import random
import pickle
import os, gzip
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
import torch
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from .load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample, load_token_cache_meta

# Some how2sign ids are broken, failing in pose fitting.
bad_how2sign_ids = ['0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 'g3Cc_1-V31U_12-3-rgb_front']

class Text2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        dataset_name='how2sign',
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length
        self.data_root = data_root
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)

        # Data mean and std
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        assert max_motion_length % unit_length == 0 and min_motion_length % unit_length == 0
        self.max_motion_length = max_motion_length // self.unit_length  
        self.min_motion_length = min_motion_length // self.unit_length  #4x downsampling in code

        # Data path
        split = 'train'
        self.code_path = code_path
        self.lm_token_num_quantizers = int(kwargs.get("lm_token_num_quantizers", 1))
        self.lm_token_num_parts = int(kwargs.get("lm_token_num_parts", 1))
        self.lm_body_codebook_size = int(kwargs.get("lm_body_codebook_size", 0))
        self.lm_hand_codebook_size = int(kwargs.get("lm_hand_codebook_size", self.lm_body_codebook_size))
        self.lm_rhand_codebook_size = int(kwargs.get("lm_rhand_codebook_size", self.lm_body_codebook_size))
        self.expected_q_offset_mode = str(kwargs.get("lm_q_offset_mode", "per_q_offset_v1"))
        self.cache_q_offset_mode = "legacy"
        if self.code_path:
            code_root = os.path.join(self.data_root, self.code_path)
            meta = load_token_cache_meta(code_root)
            self.cache_q_offset_mode = str(meta.get("q_offset_mode", "legacy"))
        self.dynamic_task_sampling = bool(kwargs.get("dynamic_task_sampling", False))
        self.fixed_task = kwargs.get("fixed_task", None)
        self.train_task_classes = list(kwargs.get("train_task_classes", []))
        self.task_sampling = kwargs.get("task_sampling", {}) or {}
        self.enable_token_drop_aug = bool(kwargs.get("enable_token_drop_aug", True))
        
        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin('prepare/instructions', 'template_pretrain.json')
        elif stage in ['lm_instruct', "lm_rl"]:
            instructions = pjoin('prepare/instructions', 'template_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")
        
        self.all_data = []
        self.h2s_len = self.csl_len = self.phoenix_len = 0
        if 'how2sign' in dataset_name:
            self.data_dir = os.path.join(data_root, split, 'poses')
            self.csv_path = os.path.join(data_root, split, 're_aligned', 'how2sign_realigned_'+split+'_preprocessed_fps.csv')
            self.csv = pd.read_csv(self.csv_path)
            self.fps = self.csv['fps']
            self.csv['DURATION'] = self.csv['END_REALIGNED'] - self.csv['START_REALIGNED']
            self.csv = self.csv[self.csv['DURATION']<30].reset_index(drop=True) # remove sequences longer than 30 seconds
            self.ids = self.csv['SENTENCE_NAME'] #[:200]

            print('loading how2sign data...', len(self.ids))
            for idx in tqdm(range(len(self.ids))):
                name = self.ids[idx]
                if name in bad_how2sign_ids:
                    continue
                self.all_data.append({'name': name, 'fps': self.csv[self.csv['SENTENCE_NAME']==name]['fps'].item(), 
                                        'text': self.csv[self.csv['SENTENCE_NAME']==name]['SENTENCE'].item(), 'src': 'how2sign'})
                # _, text, n, code = load_h2s_sample(idx, self.ids, self.csv, self.data_dir, need_pose=False, code_path=os.path.join(data_root, code_path), need_code=True)
                # if text is None and n is None:
                #     continue  #some samples are missing due to too short length
                # self.all_data.append({'name': n, 'code': code, 'text': text, 'src': 'how2sign'})
            self.h2s_len = len(self.all_data)
        
        if 'csl' in dataset_name:
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            else:
                ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print('loading csl data...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'csl'
                self.all_data.append(ann)
                # _, text, n, code = load_csl_sample(idx, self.ann, self.csl_root, need_pose=False, code_path=os.path.join(data_root, code_path), need_code=True)
                # if text is None and n is None:
                #     continue
                # self.all_data.append({'name': n, 'code': code, 'text': text, 'src': 'csl'})
            self.csl_len = len(self.ann)

        if 'phoenix' in dataset_name:
            if split == 'val':
                ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
            else:
                ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print('loading phoenix data...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'phoenix'
                self.all_data.append(ann)
                # _, text, n, code = load_phoenix_sample(idx, self.ann, self.phoenix_root, need_pose=False, code_path=os.path.join(data_root, code_path), need_code=True)
                # if text is None and n is None:
                #     continue
                # self.all_data.append({'name': n, 'code': code, 'text': text, 'src': 'phoenix'})
            self.phoenix_len = len(self.ann)

        print(f'Data loading done. All: {len(self.all_data)}, How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}')

        # self.nlp = spacy.load('en_core_web_sm')
        self.std_text = std_text
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

        if self.fixed_task is not None:
            self.dynamic_task_sampling = False
            self.fixed_task = str(self.fixed_task).lower()
        elif self.dynamic_task_sampling:
            if len(self.train_task_classes) == 0:
                self.train_task_classes = ["t2m", "m2t", "mc"]
            self.train_task_classes = [str(x).lower() for x in self.train_task_classes]
            weights = []
            for task_name in self.train_task_classes:
                weights.append(float(self.task_sampling.get(task_name, 0.0)))
            if sum(weights) <= 0.0:
                raise ValueError(
                    "dynamic_task_sampling is enabled, but task_sampling contains no positive weights."
                )
            total = float(sum(weights))
            self.train_task_weights = [w / total for w in weights]


    def __len__(self):
        if self.fixed_task is not None or self.dynamic_task_sampling:
            return len(self.all_data)
        return len(self.all_data) * len(self.tasks)

    def _sample_task(self):
        if self.fixed_task is not None:
            return {"class": self.fixed_task}
        if self.dynamic_task_sampling:
            task_name = random.choices(self.train_task_classes, weights=self.train_task_weights, k=1)[0]
            return {"class": task_name}
        return None

    def _apply_q_offsets_np(self, m_tokens):
        tokens = np.asarray(m_tokens)
        q_hint = max(int(self.lm_token_num_quantizers), 1)
        if self.cache_q_offset_mode == self.expected_q_offset_mode:
            return tokens
        if q_hint > 1:
            if tokens.ndim == 3:
                cb_sizes = [
                    int(self.lm_body_codebook_size),
                    int(self.lm_hand_codebook_size),
                    int(self.lm_rhand_codebook_size),
                ]
                for p in range(min(tokens.shape[2], len(cb_sizes))):
                    cb = cb_sizes[p]
                    if cb > 0 and int(np.max(tokens[:, :, p])) >= cb:
                        return tokens
            elif tokens.ndim == 2 and self.lm_token_num_parts > 1 and int(tokens.shape[1]) == int(self.lm_token_num_parts):
                cb_sizes = [
                    int(self.lm_body_codebook_size),
                    int(self.lm_hand_codebook_size),
                    int(self.lm_rhand_codebook_size),
                ]
                for p in range(min(tokens.shape[1], len(cb_sizes))):
                    cb = cb_sizes[p]
                    if cb > 0 and int(np.max(tokens[:, p])) >= cb:
                        return tokens
            elif self.lm_body_codebook_size > 0 and int(np.max(tokens)) >= int(self.lm_body_codebook_size):
                return tokens

        if tokens.ndim == 3:
            q_use = min(int(tokens.shape[1]), q_hint)
            tokens = np.array(tokens[:, :q_use, :], copy=True)
            cb_sizes = [
                int(self.lm_body_codebook_size),
                int(self.lm_hand_codebook_size),
                int(self.lm_rhand_codebook_size),
            ]
            for p in range(tokens.shape[2]):
                cb = cb_sizes[p] if p < len(cb_sizes) else cb_sizes[0]
                if cb > 0 and q_use > 1:
                    tokens[:, :, p] += (np.arange(q_use, dtype=tokens.dtype) * cb)[None, :]
            return tokens

        if tokens.ndim == 2 and self.lm_token_num_parts > 1 and int(tokens.shape[1]) == int(self.lm_token_num_parts):
            if q_hint <= 1:
                return tokens
            valid = (int(tokens.shape[0]) // q_hint) * q_hint
            if valid <= 0:
                return tokens
            tokens_valid = np.array(tokens[:valid], copy=True).reshape(-1, q_hint, tokens.shape[1])
            cb_sizes = [
                int(self.lm_body_codebook_size),
                int(self.lm_hand_codebook_size),
                int(self.lm_rhand_codebook_size),
            ]
            for p in range(tokens_valid.shape[2]):
                cb = cb_sizes[p] if p < len(cb_sizes) else cb_sizes[0]
                if cb > 0:
                    tokens_valid[:, :, p] += (np.arange(q_hint, dtype=tokens_valid.dtype) * cb)[None, :]
            tokens_valid = tokens_valid.reshape(valid, tokens.shape[1])
            if valid == tokens.shape[0]:
                return tokens_valid
            return np.concatenate([tokens_valid, tokens[valid:]], axis=0)

        if tokens.ndim == 2:
            q_use = min(int(tokens.shape[1]), q_hint)
            tokens = np.array(tokens[:, :q_use], copy=True)
            if self.lm_body_codebook_size > 0 and q_use > 1:
                tokens += (np.arange(q_use, dtype=tokens.dtype) * self.lm_body_codebook_size)[None, :]
            return tokens

        if tokens.ndim == 1 and q_hint > 1 and self.lm_body_codebook_size > 0:
            valid = (int(tokens.shape[0]) // q_hint) * q_hint
            if valid <= 0:
                return tokens
            tokens_valid = np.array(tokens[:valid], copy=True).reshape(-1, q_hint)
            tokens_valid += (np.arange(q_hint, dtype=tokens_valid.dtype) * self.lm_body_codebook_size)[None, :]
            tokens_valid = tokens_valid.reshape(valid)
            if valid == tokens.shape[0]:
                return tokens_valid
            return np.concatenate([tokens_valid, tokens[valid:]], axis=0)

        return tokens

    def _flatten_motion_tokens(self, m_tokens):
        tokens = self._apply_q_offsets_np(m_tokens)
        if tokens.ndim == 3:
            # [T, Q, P] -> [T*Q, P]
            t, q, p = tokens.shape
            return tokens.reshape(t * q, p), int(q)
        if tokens.ndim == 2:
            # Already flattened multi-head representation: [T', P]
            # where P equals number of active token streams (body/hand/rhand).
            if self.lm_token_num_parts > 1 and int(tokens.shape[1]) == int(self.lm_token_num_parts):
                q_hint = max(int(self.lm_token_num_quantizers), 1)
                return tokens, q_hint
            # Single-stream multi-level code: [T, Q] -> [T*Q]
            t, q = tokens.shape
            return tokens.reshape(t * q), int(q)
        # Already flattened representation keeps [T'].
        q_hint = max(int(self.lm_token_num_quantizers), 1)
        return tokens, q_hint


    def __getitem__(self, idx):
        if self.fixed_task is not None or self.dynamic_task_sampling:
            data_idx = idx
            task_idx = None
        else:
            data_idx = idx % len(self.all_data)
            task_idx = idx // len(self.all_data)
        sample = self.all_data[data_idx]
        src = sample['src']
        # caption = sample['text']
        # m_tokens = sample['code']

        if src == 'how2sign':
            _, caption, name, m_tokens = load_h2s_sample(sample, self.data_dir, need_pose=False, code_path=os.path.join(self.data_root, self.code_path), need_code=True)
        elif src == 'csl':
            _, caption, name, m_tokens = load_csl_sample(sample, self.csl_root, need_pose=False, code_path=os.path.join(self.data_root, self.code_path), need_code=True)
        elif src == 'phoenix':
            _, caption, name, m_tokens = load_phoenix_sample(sample, self.phoenix_root, need_pose=False, code_path=os.path.join(self.data_root, self.code_path), need_code=True)

        m_tokens, q_factor = self._flatten_motion_tokens(m_tokens)
        all_captions = [caption]
        # print(m_tokens.shape)
        m_length = m_tokens.shape[0]
        min_motion_len = self.min_motion_length * q_factor
        max_motion_len = self.max_motion_length * q_factor
        if m_length < min_motion_len:
            idx = np.linspace(0, m_length-1, num=min_motion_len, dtype=int)
            m_tokens = m_tokens[idx]
        elif m_length > max_motion_len:
            idx = np.linspace(0, m_length-1, num=max_motion_len, dtype=int)
            m_tokens = m_tokens[idx]
        else:
            crop_unit = int(np.lcm(int(self.unit_length), int(max(q_factor, 1))))
            m_length = (m_length // crop_unit) * crop_unit
            idx = (m_tokens.shape[0] - m_length) // 2
            m_tokens = m_tokens[idx:idx + m_length]

        coin = self.enable_token_drop_aug and np.random.choice([False, False, True])
        if coin:
            # Drop one frame-worth token group at head/tail to keep level alignment.
            drop_count = int(max(q_factor, 1))
            if m_tokens.shape[0] <= drop_count:
                drop_count = 1
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-drop_count]
            else:
                m_tokens = m_tokens[drop_count:]
        m_length = m_tokens.shape[0]

        if self.fixed_task is not None or self.dynamic_task_sampling:
            tasks = self._sample_task()
        else:
            tasks = self.tasks[task_idx]

        return caption, torch.from_numpy(m_tokens).long(), m_length, name, None, None, None, all_captions, tasks, src
