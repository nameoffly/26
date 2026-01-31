import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audience_model import AudiencePreferenceModel  # noqa: E402


def test_model_outputs_sum_to_one():
    model = AudiencePreferenceModel(input_dim=4)
    features = torch.zeros((2, 3, 4))
    judge_raw = torch.tensor([[10.0, 5.0, 0.0], [3.0, 2.0, 1.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    total_scores, fan_percent, judge_percent = model(features, judge_raw, mask)
    assert total_scores.shape == (2, 3)
    assert fan_percent.shape == (2, 3)
    assert judge_percent.shape == (2, 3)

    fan_sum = fan_percent.masked_fill(~mask, 0.0).sum(dim=1)
    judge_sum = judge_percent.masked_fill(~mask, 0.0).sum(dim=1)
    assert torch.allclose(fan_sum, torch.ones_like(fan_sum), atol=1e-6)
    assert torch.allclose(judge_sum, torch.ones_like(judge_sum), atol=1e-6)
