import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import from_smiles


PFAS_LIST = ["PFOA", "PFOS", "PFBA", "PFBS", "PFPrA", "HFPO-DA", "TFSI", "TFA"]


# Utilities
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def make_strat_bins(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    y_series = pd.Series(y)
    try:
        bins = pd.qcut(
            y_series,
            q=min(n_bins, y_series.nunique()),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        bins = pd.cut(
            y_series,
            bins=min(n_bins, max(2, y_series.nunique())),
            labels=False,
            duplicates="drop",
        )

    bins = bins.astype(int)

    while bins.value_counts().min() < 2 and bins.nunique() > 2:
        nb = bins.nunique() - 1
        try:
            bins = pd.qcut(
                y_series, q=nb, labels=False, duplicates="drop"
            ).astype(int)
        except ValueError:
            bins = pd.cut(
                y_series, bins=nb, labels=False, duplicates="drop"
            ).astype(int)

    return bins.to_numpy()


@dataclass
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def from_loader(cls, loader: DataLoader) -> "TargetScaler":
        ys: List[torch.Tensor] = []
        for batch in loader:
            if hasattr(batch, "y") and batch.y is not None:
                ys.append(batch.y.view(-1).float().cpu())
        y = torch.cat(ys)
        mean = float(y.mean().item())
        std = float(y.std(unbiased=False).item())
        if std < 1e-8:
            std = 1.0
        return cls(mean=mean, std=std)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean

# CSV loading / validation
def load_csv(path: Path, target_col: str = "y_placeholder") -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["SMILES", "pfas_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df[np.isfinite(df[target_col])].copy()

    df = df[df["SMILES"].notna()].copy()

    valid_idx = []
    bad_rows = []

    for i, row in df.iterrows():
        try:
            g = from_smiles(str(row["SMILES"]))
            if g.x is not None and g.x.size(0) > 0:
                valid_idx.append(i)
            else:
                bad_rows.append((i, row.get("CID", None), row["SMILES"]))
        except Exception:
            bad_rows.append((i, row.get("CID", None), row["SMILES"]))

    print("Bad rows found:", len(bad_rows))
    if bad_rows:
        print("First few bad rows:", bad_rows[:10])

    df = df.loc[valid_idx].reset_index(drop=True)
    return df


# Split helpers
def split_regression_df(
    df: pd.DataFrame,
    target_col: str = "y_placeholder",
    split_seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target_col not in df.columns:
        raise ValueError(f"target_col={target_col} missing from dataframe")

    if stratify and len(df) >= 200:
        y_bins = make_strat_bins(df[target_col].to_numpy(), n_bins=5)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=split_seed,
            stratify=y_bins,
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=split_seed,
        )

    if stratify and len(train_df) >= 200:
        y_bins_train = make_strat_bins(train_df[target_col].to_numpy(), n_bins=5)
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=split_seed,
            stratify=y_bins_train,
        )
    else:
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=split_seed,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_pfas_splits(
    df: pd.DataFrame,
    target_col: str = "y_placeholder",
    split_seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    splits = {}
    for pfas in PFAS_LIST:
        df_pfas = df[df["pfas_id"] == pfas].copy().reset_index(drop=True)
        if len(df_pfas) < 50:
            splits[pfas] = None
            continue
        splits[pfas] = split_regression_df(
            df_pfas,
            target_col=target_col,
            split_seed=split_seed,
            test_size=test_size,
            val_size=val_size,
            stratify=stratify,
        )
    return splits


# Dataset with graph caching
class PFASGraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = "y_placeholder"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.graphs: List[Data] = []

        for _, row in self.df.iterrows():
            g = from_smiles(str(row["SMILES"]))

            if target_col is not None and target_col in row:
                g.y = torch.tensor([float(row[target_col])], dtype=torch.float)
            else:
                # dummy target for inference-only pool scoring
                g.y = torch.tensor([0.0], dtype=torch.float)

            g.pfas_id = row["pfas_id"] if "pfas_id" in row else ""
            g.cid = int(row["CID"]) if "CID" in row and pd.notna(row["CID"]) else -1
            g.inchikey = str(row["InChIKey"]) if "InChIKey" in row else ""
            self.graphs.append(g)

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]


# Models
class GCNRegressor(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch, size=data.num_graphs)
        out = self.head(x)
        return out.view(-1)


class GATRegressor(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch, size=data.num_graphs)
        out = self.head(x)
        return out.view(-1)


def build_model(model_name: str, in_channels: int, hidden_dim: int = 128, dropout: float = 0.1) -> nn.Module:
    if model_name.upper() == "GCN":
        return GCNRegressor(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    if model_name.upper() == "GAT":
        return GATRegressor(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown model_name={model_name}. Use GCN or GAT.")


# Train and eval
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: TargetScaler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = scaler.normalize(batch.y.view(-1).float())
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        n_graphs = batch.num_graphs
        total_loss += float(loss.item()) * n_graphs
        total_graphs += n_graphs

    return total_loss / max(1, total_graphs)


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        pred = scaler.denormalize(pred_norm).cpu().numpy()
        target = batch.y.view(-1).float().cpu().numpy()
        preds.append(pred)
        targets.append(target)

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)

    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = safe_r2(y_true, y_pred)

    return {
        "mae": mae,
        "r2": r2,
        "n": int(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_pfas_gnn_from_splits(
    pfas_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "GAT",
    hidden_dim: int = 128,
    dropout: float = 0.1,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 30,
    model_seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, TargetScaler, Dict]:
    set_seed(model_seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PFASGraphDataset(train_df, target_col="y_placeholder")
    val_ds = PFASGraphDataset(val_df, target_col="y_placeholder")
    test_ds = PFASGraphDataset(test_df, target_col="y_placeholder")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    sample = train_ds.get(0)
    in_channels = int(sample.x.size(-1))

    scaler = TargetScaler.from_loader(train_loader)
    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_metrics = evaluate_regression(model, val_loader, scaler, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{pfas_id}/{model_name}] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_MAE={val_metrics['mae']:.4f} "
                f"val_R2={val_metrics['r2']:.4f}"
            )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[{pfas_id}/{model_name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint stored for {pfas_id}")

    model.load_state_dict(best_state)
    test_metrics = evaluate_regression(model, test_loader, scaler, device)

    metrics = {
        "pfas_id": pfas_id,
        "model_name": model_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "in_channels": in_channels,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "model_seed": model_seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "train_target_mean": scaler.mean,
        "train_target_std": scaler.std,
        "test_MAE": test_metrics["mae"],
        "test_R2": test_metrics["r2"],
        "history": history,
    }

    return model, scaler, metrics


def fit_gnn_ensemble_on_fixed_split(
    pfas_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "GAT",
    M: int = 3,
    base_seed: int = 42,
    device: Optional[torch.device] = None,
    **train_kwargs,
) -> Tuple[List[nn.Module], List[TargetScaler], List[Dict]]:
    models = []
    scalers = []
    metrics_list = []

    for k in range(M):
        model_seed = base_seed + k
        model, scaler, metrics = train_one_pfas_gnn_from_splits(
            pfas_id=pfas_id,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name=model_name,
            model_seed=model_seed,
            device=device,
            **train_kwargs,
        )
        models.append(model)
        scalers.append(scaler)
        metrics_list.append(metrics)

    return models, scalers, metrics_list


@torch.no_grad()
def predict_gnn_ensemble(
    models: List[nn.Module],
    scalers: List[TargetScaler],
    df_candidates: pd.DataFrame,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PFASGraphDataset(df_candidates, target_col=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    for model, scaler in zip(models, scalers):
        model.eval()
        model = model.to(device)

        preds = []
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            pred = scaler.denormalize(pred_norm).cpu()
            preds.append(pred)

        preds = torch.cat(preds).numpy()
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=0)  # [M, N]
    mu = all_preds.mean(axis=0)
    sigma = all_preds.std(axis=0)
    return mu, sigma

# Run all PFAS
def run_all_pfas(
    df: pd.DataFrame,
    model_name: str = "GAT",
    use_ensemble: bool = False,
    ensemble_M: int = 3,
    split_seed: int = 42,
    model_seed: int = 42,
    device: Optional[torch.device] = None,
    **train_kwargs,
) -> Tuple[List[Dict], float, float]:
    splits = make_pfas_splits(
        df,
        target_col="y_placeholder",
        split_seed=split_seed,
        test_size=train_kwargs.get("test_size", 0.2),
        val_size=train_kwargs.get("val_size", 0.1),
        stratify=True,
    )

    results = []
    overall_mae_num = 0.0
    overall_r2_num = 0.0
    overall_count = 0

    for pfas in PFAS_LIST:
        split = splits[pfas]
        if split is None:
            results.append(
                {
                    "pfas_id": pfas,
                    "test_MAE": None,
                    "test_R2": None,
                    "mean_sigma": None,
                }
            )
            continue

        train_df, val_df, test_df = split

        if not use_ensemble:
            _, _, metrics = train_one_pfas_gnn_from_splits(
                pfas_id=pfas,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_name=model_name,
                model_seed=model_seed,
                device=device,
                **train_kwargs,
            )
            results.append({**metrics, "mean_sigma": None})
            overall_mae_num += metrics["test_MAE"] * metrics["n_test"]
            overall_r2_num += metrics["test_R2"] * metrics["n_test"]
            overall_count += metrics["n_test"]

        else:
            models, scalers, metrics_list = fit_gnn_ensemble_on_fixed_split(
                pfas_id=pfas,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_name=model_name,
                M=ensemble_M,
                base_seed=model_seed,
                device=device,
                **train_kwargs,
            )

            mu, sigma = predict_gnn_ensemble(
                models=models,
                scalers=scalers,
                df_candidates=test_df,
                batch_size=train_kwargs.get("batch_size", 64),
                device=device,
            )

            y_true = test_df["y_placeholder"].to_numpy()
            mae = float(mean_absolute_error(y_true, mu))
            r2 = safe_r2(y_true, mu)
            mean_sigma = float(np.mean(sigma))

            out = {
                "pfas_id": pfas,
                "model_name": model_name,
                "ensemble_M": ensemble_M,
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "test_MAE": mae,
                "test_R2": r2,
                "mean_sigma": mean_sigma,
                "member_best_val_mae_mean": float(np.mean([m["best_val_mae"] for m in metrics_list])),
            }
            results.append(out)

            overall_mae_num += mae * len(test_df)
            overall_r2_num += r2 * len(test_df)
            overall_count += len(test_df)

    overall_mae = overall_mae_num / max(1, overall_count)
    overall_r2 = overall_r2_num / max(1, overall_count)
    return results, overall_mae, overall_r2

def main():
    parser = argparse.ArgumentParser(description="Per-PFAS GNN regression (single + ensemble).")
    parser.add_argument("--data", type=str, default="data/quantum_espress_placeholder.csv")
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--model", type=str, default="GAT", help="GCN or GAT")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--ensemble-M", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.model_seed)

    root = Path(__file__).resolve().parents[1]
    data_path = root / args.data
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(data_path, target_col="y_placeholder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    common_kwargs = dict(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    # Single model per PFAS
    results_base, mae_base, r2_base = run_all_pfas(
        df=df,
        model_name=args.model,
        use_ensemble=False,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        device=device,
        **common_kwargs,
    )

    out_base = {
        "mode": f"single_{args.model.lower()}_per_pfas",
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "overall_weighted_MAE": mae_base,
        "overall_weighted_R2": r2_base,
        "per_pfas": results_base,
    }

    base_path = outdir / f"per_pfas_{args.model.lower()}_metrics.json"
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(out_base, f, indent=2)

    # Ensemble per PFAS
    results_ens, mae_ens, r2_ens = run_all_pfas(
        df=df,
        model_name=args.model,
        use_ensemble=True,
        ensemble_M=args.ensemble_M,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        device=device,
        **common_kwargs,
    )

    out_ens = {
        "mode": f"ensemble_{args.model.lower()}_per_pfas",
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "ensemble_M": args.ensemble_M,
        "overall_weighted_MAE": mae_ens,
        "overall_weighted_R2": r2_ens,
        "per_pfas": results_ens,
    }

    ens_path = outdir / f"per_pfas_{args.model.lower()}_ensemble_metrics.json"
    with open(ens_path, "w", encoding="utf-8") as f:
        json.dump(out_ens, f, indent=2)

    print("Saved:")
    print(base_path)
    print(ens_path)


if __name__ == "__main__":
    main()