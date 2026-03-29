import torch
from torchmetrics import Metric


class MCTokenMetrics(Metric):
    def __init__(self, top_k=5, dist_sync_on_step=True, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.top_k = int(top_k)
        self.add_state("correct_topk", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, logits, labels, group_size=1):
        if logits is None or labels is None:
            return
        if logits.dim() != 3 or labels.dim() != 2:
            return

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        topk_ids = torch.topk(
            shift_logits,
            k=min(self.top_k, int(shift_logits.shape[-1])),
            dim=-1,
        ).indices
        group_size = max(int(group_size), 1)

        for b in range(shift_labels.shape[0]):
            valid = torch.nonzero(shift_labels[b] != -100, as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            take = valid[:group_size]
            targets = shift_labels[b, take].unsqueeze(-1)
            hits = (topk_ids[b, take] == targets).any(dim=-1).float()
            self.correct_topk += hits.sum()
            self.total += torch.tensor(float(hits.numel()), device=self.correct_topk.device)

    @torch.no_grad()
    def compute(self, sanity_flag):
        if sanity_flag:
            return {"mc_next_token_top5_acc": torch.tensor(0.0, device=self.correct_topk.device)}
        denom = torch.clamp(self.total, min=1.0)
        metrics = {"mc_next_token_top5_acc": self.correct_topk / denom}
        self.reset()
        return metrics
