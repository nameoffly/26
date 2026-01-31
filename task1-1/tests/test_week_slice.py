import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import data_processing  # noqa: E402


def test_week_slice_elimination_detection():
    df = pd.DataFrame(
        {
            "celebrity_name": ["A", "B", "C"],
            "season": [3, 3, 3],
            "week1_judge_score": [20, 18, 15],
            "week2_judge_score": [22, 0, 19],
            "1_percent": [0.4, 0.3, 0.3],
            "2_percent": [0.5, 0.0, 0.5],
        }
    )
    last_week = data_processing.compute_last_week(df)
    names, judge_pct, eliminated_names, eliminated_idx = data_processing.week_slice(
        df, week=1, last_week=last_week
    )

    assert names == ["A", "B", "C"]
    assert eliminated_names == ["B"]
    assert eliminated_idx == [1]
    assert len(judge_pct) == 3
