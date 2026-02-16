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
            pred = pred*mask
            target = target*mask
        return F.smooth_l1_loss(pred, target)


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS
        self.part_weights_cfg = cfg.LOSS.get("PART_WEIGHTS", None)

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

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
            if loss.split('_')[0] == 'recons':
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
                vel_lengths = [max(int(x) - 1, 0) for x in rs_set['length']]
                if max(vel_lengths) > 0:
                    total += self._update_loss("recons_velocity", vel_rst, vel_ref, vel_lengths)
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
