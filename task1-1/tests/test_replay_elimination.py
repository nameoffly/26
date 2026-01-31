import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import replay_elimination  # noqa: E402


def test_predict_eliminated_names_bottom_k():
    week_df = pd.DataFrame(
        {
            "celebrity_name": ["A", "B", "C"],
            "fan_mean": [0.1, 0.2, 0.7],
            "judge_pct": [0.2, 0.1, 0.0],
        }
    )
    predicted = replay_elimination.predict_eliminated_names(week_df, k=1)
    assert predicted == ["A"]


def test_evaluate_week_no_elimination():
    week_df = pd.DataFrame(
        {
            "celebrity_name": ["A", "B"],
            "fan_mean": [0.4, 0.6],
            "judge_pct": [0.3, 0.2],
        }
    )
    predicted, match = replay_elimination.evaluate_week(
        week_df, actual_eliminated=[]
    )
    assert predicted == []
    assert match == "NA"
