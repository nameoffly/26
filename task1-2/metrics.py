import torch


def predict_eliminated_indices(scores, mask, k):
    if k <= 0:
        return []
    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        return []
    valid_scores = scores[valid_idx]
    order = torch.argsort(valid_scores, descending=False)
    selected = valid_idx[order[:k]].tolist()
    return selected


def evaluate_consistency(predicted_idx, actual_idx):
    if not actual_idx:
        return "NA"
    return set(predicted_idx) == set(actual_idx)


def compute_certainty_gap(scores, mask, actual_idx):
    if not actual_idx:
        return None
    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
    eliminated = [idx for idx in actual_idx if idx in valid_idx]
    if not eliminated:
        return None
    survivors = [idx for idx in valid_idx if idx not in eliminated]
    if not survivors:
        return None
    max_elim = torch.max(scores[eliminated]).item()
    min_surv = torch.min(scores[survivors]).item()
    return min_surv - max_elim
