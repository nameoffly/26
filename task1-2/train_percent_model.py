import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from audience_dataset import WeekDataset, collate_week_samples
from audience_model import AudiencePreferenceModel
from losses import PercentageEliminationLoss
from metrics import predict_eliminated_indices, evaluate_consistency, compute_certainty_gap


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_season(dataset, val_frac, seed):
    seasons = sorted({item["season"] for item in dataset.samples})
    if not seasons:
        return [], []
    rng = np.random.default_rng(seed)
    rng.shuffle(seasons)
    split = int(len(seasons) * (1 - val_frac))
    split = max(1, split)
    split = min(split, len(seasons))
    train_seasons = set(seasons[:split])
    val_seasons = set(seasons[split:])
    if not val_seasons:
        val_seasons = set(train_seasons)
    train_idx = [
        i for i, item in enumerate(dataset.samples) if item["season"] in train_seasons
    ]
    val_idx = [
        i for i, item in enumerate(dataset.samples) if item["season"] in val_seasons
    ]
    return train_idx, val_idx


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    matched = 0
    total = 0
    gaps = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            judge_raw = batch["judge_raw_scores"].to(device)
            mask = batch["mask"].to(device)
            total_scores, fan_percent, _ = model(features, judge_raw, mask)
            loss = loss_fn(total_scores, batch["eliminated_idx"], mask)
            total_loss += float(loss.item())
            total_batches += 1

            for i in range(total_scores.shape[0]):
                k = len(batch["eliminated_idx"][i])
                predicted = predict_eliminated_indices(
                    total_scores[i], mask[i], k
                )
                consistency = evaluate_consistency(
                    predicted, batch["eliminated_idx"][i]
                )
                if consistency != "NA":
                    total += 1
                    if consistency:
                        matched += 1
                gap = compute_certainty_gap(
                    total_scores[i], mask[i], batch["eliminated_idx"][i]
                )
                if gap is not None:
                    gaps.append(gap)

    avg_loss = total_loss / total_batches if total_batches else 0.0
    accuracy = matched / total if total else float("nan")
    gap_mean = float(np.mean(gaps)) if gaps else float("nan")
    return {"loss": avg_loss, "accuracy": accuracy, "gap_mean": gap_mean}


def build_output_tables(model, data_loader, device):
    contestant_rows = []
    weekly_rows = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            judge_raw = batch["judge_raw_scores"].to(device)
            mask = batch["mask"].to(device)
            total_scores, fan_percent, judge_percent = model(
                features, judge_raw, mask
            )

            for i in range(features.shape[0]):
                n_active = int(mask[i].sum().item())
                names = batch["names"][i]
                season = batch["season"][i]
                week = batch["week"][i]
                eliminated_idx = batch["eliminated_idx"][i]
                k = len(eliminated_idx)
                predicted_idx = predict_eliminated_indices(
                    total_scores[i], mask[i], k
                )
                match = evaluate_consistency(predicted_idx, eliminated_idx)
                gap = compute_certainty_gap(
                    total_scores[i], mask[i], eliminated_idx
                )

                eliminated_names = [names[j] for j in eliminated_idx]
                predicted_names = [names[j] for j in predicted_idx]

                weekly_rows.append(
                    {
                        "season": season,
                        "week": week,
                        "n_active": n_active,
                        "n_eliminated": k,
                        "eliminated_names": ";".join(eliminated_names),
                        "predicted_eliminated_names": ";".join(predicted_names),
                        "match": match,
                        "certainty_gap": gap,
                    }
                )

                for j in range(n_active):
                    contestant_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "celebrity_name": names[j],
                            "judge_pct": float(judge_percent[i, j].item()),
                            "fan_percent": float(fan_percent[i, j].item()),
                            "total_score": float(total_scores[i, j].item()),
                            "eliminated": j in eliminated_idx,
                            "predicted_eliminated": j in predicted_idx,
                            "n_active": n_active,
                        }
                    )

    return pd.DataFrame(weekly_rows), pd.DataFrame(contestant_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/hisheep/d/MCM/26/Data_4.xlsx")
    parser.add_argument("--season-start", type=int, default=3)
    parser.add_argument("--season-end", type=int, default=27)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="/home/hisheep/d/MCM/26/task1-2/outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = WeekDataset(
        data_path=args.input,
        season_start=args.season_start,
        season_end=args.season_end,
    )
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查输入和赛季范围")

    train_idx, val_idx = split_by_season(dataset, args.val_frac, args.seed)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_week_samples
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_week_samples
    )

    input_dim = dataset.samples[0]["features"].shape[1]
    model = AudiencePreferenceModel(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = PercentageEliminationLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            judge_raw = batch["judge_raw_scores"].to(device)
            mask = batch["mask"].to(device)
            total_scores, _, _ = model(features, judge_raw, mask)
            loss = loss_fn(total_scores, batch["eliminated_idx"], mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            train_stats = evaluate_model(model, train_loader, loss_fn, device)
            val_stats = evaluate_model(model, val_loader, loss_fn, device)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_acc={val_stats['accuracy']:.3f} "
                f"val_gap={val_stats['gap_mean']:.4f}"
            )

    full_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_week_samples
    )
    weekly_df, contestant_df = build_output_tables(model, full_loader, device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weekly_df.to_csv(out_dir / "weekly_summary.csv", index=False)
    contestant_df.to_csv(out_dir / "contestant_fan_vote_summary.csv", index=False)


if __name__ == "__main__":
    main()
