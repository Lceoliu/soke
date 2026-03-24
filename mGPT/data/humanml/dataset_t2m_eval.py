import random
import numpy as np
import torch, os, pickle, math
from .load_data import load_csl_sample, load_h2s_sample, load_phoenix_sample, load_token_cache_meta
from .dataset_t2m import Text2MotionDataset


class Text2MotionDatasetEval(Text2MotionDataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        dataset_name='how2sign',
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, dataset_name=dataset_name, **kwargs)

        self.data_root = data_root
        self.w_vectorizer = w_vectorizer
        self.code_path = kwargs.get('code_path', None)
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
            t, q, p = tokens.shape
            return tokens.reshape(t * q, p), int(q)
        if tokens.ndim == 2:
            if self.lm_token_num_parts > 1 and int(tokens.shape[1]) == int(self.lm_token_num_parts):
                q_hint = max(int(self.lm_token_num_quantizers), 1)
                return tokens, q_hint
            t, q = tokens.shape
            return tokens.reshape(t * q), int(q)
        q_hint = max(int(self.lm_token_num_quantizers), 1)
        return tokens, q_hint


    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']

        code = None
        if src == 'how2sign':
            clip_poses, text, name, code = load_h2s_sample(
                sample,
                self.data_dir,
                code_path=os.path.join(self.data_root, self.code_path) if self.code_path else None,
                need_code=bool(self.code_path),
            )
        elif src == 'csl':
            clip_poses, text, name, code = load_csl_sample(
                sample,
                self.csl_root,
                code_path=os.path.join(self.data_root, self.code_path) if self.code_path else None,
                need_code=bool(self.code_path),
            )
        elif src == 'phoenix':
            clip_poses, text, name, code = load_phoenix_sample(
                sample,
                self.phoenix_root,
                code_path=os.path.join(self.data_root, self.code_path) if self.code_path else None,
                need_code=bool(self.code_path),
            )
        
        all_captions = [text]
        all_captions = all_captions * 3  #?

        clip_poses = (clip_poses - self.mean.numpy())/(self.std.numpy()+1e-10)
        # return torch.from_numpy(clip_poses).float(), basename, clip_text
        m_length = clip_poses.shape[0]
        if m_length < self.min_motion_length:
            idx = np.linspace(0, m_length-1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        elif m_length > self.max_motion_length:
            idx = np.linspace(0, m_length-1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[idx:idx + m_length]
        m_length = clip_poses.shape[0]

        token_tensor = None
        token_length = 0
        if self.code_path and code is None:
            raise FileNotFoundError(
                f"Missing precomputed motion token cache for sample '{name}' under code_path='{self.code_path}'. "
                "Regenerate train/val/test token cache before LM evaluation."
            )
        if code is not None:
            code, q_factor = self._flatten_motion_tokens(code)
            # Match Text2MotionDatasetCB exactly so LM train/val/test consume
            # token sequences clipped with the same rules.
            min_motion_len = self.min_motion_length * q_factor
            max_motion_len = self.max_motion_length * q_factor
            code_length = code.shape[0]
            if code_length < min_motion_len:
                idx = np.linspace(0, code_length - 1, num=min_motion_len, dtype=int)
                code = code[idx]
            elif code_length > max_motion_len:
                idx = np.linspace(0, code_length - 1, num=max_motion_len, dtype=int)
                code = code[idx]
            else:
                crop_unit = int(np.lcm(int(self.unit_length), int(max(q_factor, 1))))
                code_length = (code_length // crop_unit) * crop_unit
                idx = (code.shape[0] - code_length) // 2
                code = code[idx:idx + code_length]
            token_length = int(code.shape[0])
            token_tensor = torch.from_numpy(code).long()

        # Text
        tokens = text.split(' ')
        max_text_len = 40
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        return text, torch.from_numpy(clip_poses).float(), m_length, name, None, None, "_".join(tokens), all_captions, None, src, token_tensor, token_length


def sample(input,count):
    ss=float(len(input))/count
    return [ input[int(math.floor(i*ss))] for i in range(count) ]
