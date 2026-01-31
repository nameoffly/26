import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from audience_dataset import WeekDataset, collate_week_samples  # noqa: E402


def test_week_dataset_basic():
    df = pd.DataFrame(
        {
            "celebrity_name": ["A", "B", "C"],
            "season": [3, 3, 3],
            "week1_judge_score": [20, 18, 15],
            "week2_judge_score": [22, 0, 19],
            "celebrity_age_during_season": [30, 40, 35],
            "celebrity_industry": ["Actor", "Athlete", "Actor"],
        }
    )
    dataset = WeekDataset(data_frame=df, season_start=3, season_end=3)
    assert len(dataset) == 2

    sample = dataset[0]
    assert sample["season"] == 3
    assert sample["week"] == 1
    assert len(sample["names"]) == 3
    assert sample["features"].shape[0] == 3
    assert sample["features"].shape[1] >= 6


def test_collate_week_samples():
    df = pd.DataFrame(
        {
            "celebrity_name": ["A", "B"],
            "season": [3, 3],
            "week1_judge_score": [20, 18],
            "celebrity_age_during_season": [30, 40],
            "celebrity_industry": ["Actor", "Athlete"],
        }
    )
    dataset = WeekDataset(data_frame=df, season_start=3, season_end=3)
    batch = collate_week_samples([dataset[0]])
    assert batch["features"].shape[0] == 1
    assert batch["mask"].sum().item() == 2
