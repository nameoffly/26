import torch
from torch import nn


def masked_softmax(logits, mask, dim=1):
    mask = mask.to(dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(masked_logits, dim=dim)


class AudiencePreferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features, judge_raw_scores, mask):
        raw_popularity = self.mlp(features).squeeze(-1)
        fan_percent = masked_softmax(raw_popularity, mask, dim=1)

        masked_judge = judge_raw_scores * mask.float()
        denom = masked_judge.sum(dim=1, keepdim=True)
        judge_percent = torch.where(
            denom > 0, masked_judge / denom, torch.zeros_like(masked_judge)
        )

        total_score = judge_percent + fan_percent
        return total_score, fan_percent, judge_percent
