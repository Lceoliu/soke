import torch
import torch.nn as nn
from .base import BaseLosses
import torch.nn.functional as F


def create_mask(length, device):
    b_size = len(length)
    mask = torch.zeros(b_size, max(length), 1).to(device)
    for i, l in enumerate(length):
        mask[i, :l, 0] = 1
    return mask  #[B,maxT,1]


class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class SmoothL1LossWithMask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, pred, target, length=None):
        if length is not None:
            # mask paddings
            mask = create_mask(length, pred.device)
            while mask.dim() < pred.dim():
                mask = mask.unsqueeze(-1)
            pred = pred*mask
            target = target*mask
        return F.smooth_l1_loss(pred, target)


class BCEWithLogitsLossWithMask(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, target, length=None, sample_valid=None):
        target = target.to(dtype=pred.dtype)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if length is not None:
            mask = create_mask(length, pred.device)
            if mask.shape[1] != loss.shape[1]:
                if mask.shape[1] > loss.shape[1]:
                    mask = mask[:, :loss.shape[1]]
                else:
                    pad = torch.zeros(
                        (mask.shape[0], loss.shape[1] - mask.shape[1], 1),
                        device=mask.device,
                        dtype=mask.dtype,
                    )
                    mask = torch.cat([mask, pad], dim=1)
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
        else:
            mask = torch.ones_like(loss[..., :1], device=pred.device, dtype=pred.dtype)
        if sample_valid is not None:
            sv = sample_valid.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1)
            mask = mask * sv
        loss = loss * mask
        denom = (mask.sum() * loss.shape[-1]).clamp_min(1.0)
        return loss.sum() / denom


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS
        self.part_weights_cfg = cfg.LOSS.get("PART_WEIGHTS", None)
        self.velocity_part_weights_cfg = cfg.LOSS.get("VELOCITY_PART_WEIGHTS", None)

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            losses.append("recons_fk_hand")
            params['recons_fk_hand'] = float(cfg.LOSS.get("LAMBDA_FK_HAND", 0.0))

            losses.append("recons_accel_hand")
            params['recons_accel_hand'] = float(cfg.LOSS.get("LAMBDA_ACCEL_HAND", 0.0))

            losses.append("recons_accel_wrist_rel")
            params['recons_accel_wrist_rel'] = float(cfg.LOSS.get("LAMBDA_ACCEL_WRIST_REL", 0.0))

            losses.append("recons_contact")
            params['recons_contact'] = float(cfg.LOSS.get("LAMBDA_CONTACT", 0.0))

            losses.append("vq_commit")
            params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT
        elif stage in ["lm_pretrain", "lm_instruct"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS
            losses.append("gpthand_loss")
            params['gpthand_loss'] = cfg.LOSS.LAMBDA_CLS
            losses.append("gptrhand_loss")
            params['gptrhand_loss'] = cfg.LOSS.LAMBDA_CLS

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss == "recons_contact":
                losses_func[loss] = BCEWithLogitsLossWithMask
            elif loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    # losses_func[loss] = nn.SmoothL1Loss
                    losses_func[loss] = SmoothL1LossWithMask
            elif loss.split('_')[1] in [
                    'commit', 'loss', 'gpt', 'm2t2m', 't2m2t'
            ]:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    def _build_part_weights(self, x: torch.Tensor):
        """
        Optional feature reweighting for SOKE 133-dim layout:
        [0:30]=upper body, [30:120]=hands, [120:133]=face.
        """
        if self.part_weights_cfg is None or x.shape[-1] != 133:
            return None
        w_upper = float(self.part_weights_cfg.get("UPPER", 1.0))
        w_hand = float(self.part_weights_cfg.get("HAND", 1.0))
        w_face = float(self.part_weights_cfg.get("FACE", 1.0))
        w = torch.ones((x.shape[-1],), device=x.device, dtype=x.dtype)
        w[:30] = w_upper
        w[30:120] = w_hand
        w[120:] = w_face
        return w.view(1, 1, -1)

    def _build_velocity_part_weights(self, x: torch.Tensor):
        """
        Optional velocity reweighting for SOKE 133-dim layout.
        Effective per-part lambda = LAMBDA_VELOCITY * VELOCITY_PART_WEIGHTS[part].
        """
        if self.velocity_part_weights_cfg is None or x.shape[-1] != 133:
            return None
        w_upper = float(self.velocity_part_weights_cfg.get("UPPER", 1.0))
        w_hand = float(self.velocity_part_weights_cfg.get("HAND", 1.0))
        w_face = float(self.velocity_part_weights_cfg.get("FACE", 1.0))
        w = torch.ones((x.shape[-1],), device=x.device, dtype=x.dtype)
        w[:30] = w_upper
        w[30:120] = w_hand
        w[120:] = w_face
        return w.view(1, 1, -1)

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            m_rst = rs_set['m_rst']
            m_ref = rs_set['m_ref']
            part_w = self._build_part_weights(m_rst)
            if part_w is not None:
                m_rst = m_rst * part_w
                m_ref = m_ref * part_w
            total += self._update_loss("recons_feature", m_rst, m_ref, rs_set['length'])
            # total += self._update_loss("recons_joints", rs_set['joints_rst'], rs_set['joints_ref'])
            if self._params['recons_velocity'] != 0.0:
                vel_rst = rs_set['m_rst'][:, 1:, :] - rs_set['m_rst'][:, :-1, :]
                vel_ref = rs_set['m_ref'][:, 1:, :] - rs_set['m_ref'][:, :-1, :]
                vel_part_w = self._build_velocity_part_weights(vel_rst)
                if vel_part_w is not None:
                    vel_rst = vel_rst * vel_part_w
                    vel_ref = vel_ref * vel_part_w
                vel_lengths = [max(int(x) - 1, 0) for x in rs_set['length']]
                if max(vel_lengths) > 0:
                    total += self._update_loss("recons_velocity", vel_rst, vel_ref, vel_lengths)
            if self._params['recons_fk_hand'] != 0.0:
                fk_lhand_rst = rs_set.get('fk_lhand_rst', None)
                fk_lhand_ref = rs_set.get('fk_lhand_ref', None)
                fk_rhand_rst = rs_set.get('fk_rhand_rst', None)
                fk_rhand_ref = rs_set.get('fk_rhand_ref', None)
                if (
                    fk_lhand_rst is not None and fk_lhand_ref is not None
                    and fk_rhand_rst is not None and fk_rhand_ref is not None
                ):
                    # Hand FK loss:
                    # - fk_* tensors are wrist-relative SMPL-X hand joints.
                    # - Concatenate left/right hand joints, then apply masked reconstruction
                    #   loss over valid sequence length (same mask rule as feature loss).
                    fk_rst = torch.cat([fk_lhand_rst, fk_rhand_rst], dim=2)
                    fk_ref = torch.cat([fk_lhand_ref, fk_rhand_ref], dim=2)
                    total += self._update_loss("recons_fk_hand", fk_rst, fk_ref, rs_set['length'])
            if self._params['recons_accel_hand'] != 0.0:
                fk_lhand_rst = rs_set.get('fk_lhand_rst', None)
                fk_lhand_ref = rs_set.get('fk_lhand_ref', None)
                fk_rhand_rst = rs_set.get('fk_rhand_rst', None)
                fk_rhand_ref = rs_set.get('fk_rhand_ref', None)
                if (
                    fk_lhand_rst is not None and fk_lhand_ref is not None
                    and fk_rhand_rst is not None and fk_rhand_ref is not None
                ):
                    # Hand acceleration loss (wrist-relative):
                    # - Compute 2nd-order temporal difference on wrist-relative hand joints.
                    # - Concatenate left/right hands to supervise explosive finger motion.
                    acc_lhand_rst = fk_lhand_rst[:, 2:, ...] - 2 * fk_lhand_rst[:, 1:-1, ...] + fk_lhand_rst[:, :-2, ...]
                    acc_lhand_ref = fk_lhand_ref[:, 2:, ...] - 2 * fk_lhand_ref[:, 1:-1, ...] + fk_lhand_ref[:, :-2, ...]
                    acc_rhand_rst = fk_rhand_rst[:, 2:, ...] - 2 * fk_rhand_rst[:, 1:-1, ...] + fk_rhand_rst[:, :-2, ...]
                    acc_rhand_ref = fk_rhand_ref[:, 2:, ...] - 2 * fk_rhand_ref[:, 1:-1, ...] + fk_rhand_ref[:, :-2, ...]
                    acc_rst = torch.cat([acc_lhand_rst, acc_rhand_rst], dim=2)
                    acc_ref = torch.cat([acc_lhand_ref, acc_rhand_ref], dim=2)
                    acc_lengths = [max(int(x) - 2, 0) for x in rs_set['length']]
                    if max(acc_lengths) > 0:
                        total += self._update_loss("recons_accel_hand", acc_rst, acc_ref, acc_lengths)
            if self._params['recons_accel_wrist_rel'] != 0.0:
                wrist_l_rst = rs_set.get('wrist_l_rst', None)
                wrist_r_rst = rs_set.get('wrist_r_rst', None)
                wrist_l_ref = rs_set.get('wrist_l_ref', None)
                wrist_r_ref = rs_set.get('wrist_r_ref', None)
                if (
                    wrist_l_rst is not None and wrist_r_rst is not None
                    and wrist_l_ref is not None and wrist_r_ref is not None
                ):
                    # Two-hand relative acceleration loss:
                    # - Build global left->right wrist vector.
                    # - Apply 2nd-order temporal difference on this relative vector.
                    rel_vec_rst = wrist_r_rst - wrist_l_rst
                    rel_vec_ref = wrist_r_ref - wrist_l_ref
                    rel_acc_rst = rel_vec_rst[:, 2:, ...] - 2 * rel_vec_rst[:, 1:-1, ...] + rel_vec_rst[:, :-2, ...]
                    rel_acc_ref = rel_vec_ref[:, 2:, ...] - 2 * rel_vec_ref[:, 1:-1, ...] + rel_vec_ref[:, :-2, ...]
                    rel_acc_lengths = [max(int(x) - 2, 0) for x in rs_set['length']]
                    if max(rel_acc_lengths) > 0:
                        total += self._update_loss("recons_accel_wrist_rel", rel_acc_rst, rel_acc_ref, rel_acc_lengths)
            if self._params['recons_contact'] != 0.0:
                pred_contact = rs_set.get('contact_logits', None)
                gt_contact = rs_set.get('gt_contact_labels', None)
                gt_contact_has_label = rs_set.get('gt_contact_has_label', None)
                if pred_contact is not None and gt_contact is not None:
                    total += self._update_loss(
                        "recons_contact",
                        pred_contact,
                        gt_contact,
                        rs_set['length'],
                        sample_valid=gt_contact_has_label,
                    )
            total += self._update_loss("vq_commit", rs_set['loss_commit'],
                                       rs_set['loss_commit'])

        if self.stage in ["lm_pretrain", "lm_instruct"]:
            if type(rs_set['outputs']) == dict:
                total += self._update_loss("gpt_loss", rs_set['outputs']['loss'],
                                       rs_set['outputs']['loss'])
                total += self._update_loss("gpthand_loss", rs_set['outputs']['loss_hand'],
                                       rs_set['outputs']['loss_hand'])
                if rs_set['outputs']['loss_rhand'] is not None:
                    total += self._update_loss("gptrhand_loss", rs_set['outputs']['loss_rhand'],
                                       rs_set['outputs']['loss_rhand'])
            else:
                total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                        rs_set['outputs'].loss)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total
