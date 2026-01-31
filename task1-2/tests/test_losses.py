import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from losses import PercentageEliminationLoss  # noqa: E402


def test_loss_zero_when_eliminated_lowest():
    loss_fn = PercentageEliminationLoss(margin=0.0)
    total_scores = torch.tensor([[0.1, 0.2, 0.3]])
    mask = torch.tensor([[True, True, True]])
    eliminated_idx = [[0]]
    loss = loss_fn(total_scores, eliminated_idx, mask)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_loss_positive_when_eliminated_not_lowest():
    loss_fn = PercentageEliminationLoss(margin=0.0)
    total_scores = torch.tensor([[0.3, 0.2, 0.1]])
    mask = torch.tensor([[True, True, True]])
    eliminated_idx = [[0]]
    loss = loss_fn(total_scores, eliminated_idx, mask)
    assert loss.item() > 0.0
