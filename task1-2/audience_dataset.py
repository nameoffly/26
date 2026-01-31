import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import data_processing


def _normalize_by_sum(values):
    total = np.sum(values)
    if total <= 0:
        return np.zeros_like(values, dtype=float)
    return values / total


def _normalize_by_max(values):
    max_val = np.max(values) if len(values) else 0.0
    if max_val <= 0:
        return np.zeros_like(values, dtype=float)
    return values / max_val


class WeekDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        data_frame=None,
        season_start=3,
        season_end=27,
        include_industry=True,
        include_age=True,
        include_prev_week=True,
    ):
        if data_frame is None:
            if data_path is None:
                raise ValueError("data_path or data_frame must be provided")
            df = pd.read_excel(data_path)
        else:
            df = data_frame.copy()

        self.df = df.copy()
        self.include_industry = include_industry
        self.include_age = include_age
        self.include_prev_week = include_prev_week

        if "celebrity_industry" in df.columns:
            categories = sorted(
                [str(c) for c in df["celebrity_industry"].dropna().unique()]
            )
        else:
            categories = []
        self.industry_categories = categories
        self.industry_to_idx = {c: i for i, c in enumerate(categories)}

        if "celebrity_age_during_season" in df.columns:
            ages = pd.to_numeric(
                df["celebrity_age_during_season"], errors="coerce"
            )
            self.age_mean = float(ages.mean()) if ages.notna().any() else 0.0
            self.age_std = float(ages.std()) if ages.notna().any() else 1.0
        else:
            self.age_mean = 0.0
            self.age_std = 1.0
        if self.age_std <= 0:
            self.age_std = 1.0

        self.samples = []
        for season in range(season_start, season_end + 1):
            season_df = df[df["season"] == season].copy()
            if season_df.empty:
                continue
            last_week = data_processing.compute_last_week(season_df)
            if last_week == 0:
                continue

            for week in range(1, last_week + 1):
                (
                    names,
                    judge_raw_scores,
                    eliminated_names,
                    eliminated_idx,
                ) = data_processing.week_slice(
                    season_df, week=week, last_week=last_week
                )
                if not names:
                    continue
                features = self._build_features(
                    season_df, week, last_week, names, judge_raw_scores
                )
                self.samples.append(
                    {
                        "season": season,
                        "week": week,
                        "names": names,
                        "features": features,
                        "judge_raw_scores": judge_raw_scores,
                        "eliminated_idx": eliminated_idx,
                        "eliminated_names": eliminated_names,
                    }
                )

    def _build_features(self, season_df, week, last_week, names, judge_raw_scores):
        raw_scores = data_processing.compute_week_raw_scores(season_df, week)
        active_mask = raw_scores.fillna(0).gt(0)
        active_df = season_df.loc[active_mask].copy()

        judge_percent = _normalize_by_sum(judge_raw_scores)
        judge_raw_norm = _normalize_by_max(judge_raw_scores)

        prev_raw_norm = np.zeros_like(judge_raw_norm, dtype=float)
        prev_percent = np.zeros_like(judge_percent, dtype=float)
        if self.include_prev_week and week > 1:
            prev_raw = data_processing.compute_week_raw_scores(season_df, week - 1)
            prev_raw = prev_raw[active_mask].astype(float).to_numpy()
            prev_raw_norm = _normalize_by_max(prev_raw)
            prev_percent = _normalize_by_sum(prev_raw)

        week_norm = np.full_like(judge_percent, fill_value=week / last_week, dtype=float)

        age_norm = np.zeros_like(judge_percent, dtype=float)
        if self.include_age and "celebrity_age_during_season" in active_df.columns:
            ages = pd.to_numeric(
                active_df["celebrity_age_during_season"], errors="coerce"
            ).to_numpy()
            ages = np.where(np.isnan(ages), self.age_mean, ages)
            age_norm = (ages - self.age_mean) / self.age_std

        industry_features = []
        if self.include_industry and self.industry_categories:
            industries = active_df.get(
                "celebrity_industry", pd.Series([], dtype=str)
            ).astype(str)
            for industry in industries:
                one_hot = np.zeros(len(self.industry_categories), dtype=float)
                idx = self.industry_to_idx.get(industry)
                if idx is not None:
                    one_hot[idx] = 1.0
                industry_features.append(one_hot)
        if industry_features:
            industry_features = np.stack(industry_features, axis=0)

        parts = [
            judge_raw_norm,
            judge_percent,
            prev_raw_norm,
            prev_percent,
            week_norm,
            age_norm,
        ]
        feature_matrix = np.stack(parts, axis=1)
        if isinstance(industry_features, np.ndarray):
            feature_matrix = np.concatenate([feature_matrix, industry_features], axis=1)

        return feature_matrix.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_week_samples(batch):
    if not batch:
        raise ValueError("Empty batch")
    max_n = max(len(item["judge_raw_scores"]) for item in batch)
    feat_dim = batch[0]["features"].shape[1]
    batch_size = len(batch)

    features = torch.zeros((batch_size, max_n, feat_dim), dtype=torch.float32)
    judge_raw_scores = torch.zeros((batch_size, max_n), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    eliminated_idx = []
    eliminated_names = []
    seasons = []
    weeks = []
    names = []

    for i, item in enumerate(batch):
        n = len(item["judge_raw_scores"])
        features[i, :n] = torch.from_numpy(item["features"])
        judge_raw_scores[i, :n] = torch.tensor(item["judge_raw_scores"], dtype=torch.float32)
        mask[i, :n] = True
        eliminated_idx.append(item["eliminated_idx"])
        eliminated_names.append(item["eliminated_names"])
        seasons.append(item["season"])
        weeks.append(item["week"])
        names.append(item["names"])

    return {
        "features": features,
        "judge_raw_scores": judge_raw_scores,
        "mask": mask,
        "eliminated_idx": eliminated_idx,
        "eliminated_names": eliminated_names,
        "season": seasons,
        "week": weeks,
        "names": names,
    }
