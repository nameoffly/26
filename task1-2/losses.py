import torch
from torch import nn


class PercentageEliminationLoss(nn.Module):
    def __init__(self, margin=0.01, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, total_scores, eliminated_idx_list, mask):
        total_loss = torch.tensor(0.0, device=total_scores.device)
        total_pairs = 0

        for i in range(total_scores.shape[0]):
            valid_mask = mask[i]
            if not torch.any(valid_mask):
                continue

            eliminated_idx = eliminated_idx_list[i]
            if not eliminated_idx:
                continue

            eliminated = [
                idx for idx in eliminated_idx if idx < len(valid_mask) and valid_mask[idx]
            ]
            if not eliminated:
                continue

            survivors = [
                idx
                for idx in torch.nonzero(valid_mask, as_tuple=False).squeeze(1).tolist()
                if idx not in eliminated
            ]
            if not survivors:
                continue

            scores_elim = total_scores[i, eliminated]
            scores_surv = total_scores[i, survivors]

            pairwise = scores_elim[:, None] - scores_surv[None, :] + self.margin
            loss = torch.relu(pairwise)
            total_loss = total_loss + loss.sum()
            total_pairs += loss.numel()

        if self.reduction == "sum":
            return total_loss
        if total_pairs == 0:
            return total_loss
        return total_loss / total_pairs
