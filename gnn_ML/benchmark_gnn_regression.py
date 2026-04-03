import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Subset
from torch_geometric.datasets import AQSOL, MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

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


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("Require val_ratio > 0, test_ratio > 0, and val_ratio + test_ratio < 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough data left for training. Reduce val_ratio/test_ratio.")

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return train_idx, val_idx, test_idx


@dataclass
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def from_loader(cls, loader: DataLoader) -> "TargetScaler":
        ys: List[torch.Tensor] = []
        for batch in loader:
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

    def forward(self, data):
        x = data.x.float()
        # If the sample is 1D, force it to be 2D:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
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

    def forward(self, data):
        x = data.x.float()
        # If the sample is 1D, force it to be 2D:
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        out = self.head(x)
        return out.view(-1)


def build_model(model_name: str, in_channels: int, hidden_dim: int, dropout: float) -> nn.Module:
    if model_name.upper() == "GCN":
        return GCNRegressor(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    if model_name.upper() == "GAT":
        return GATRegressor(in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown model_name={model_name}. Use GCN or GAT.")

# Data loading
def load_dataset(name: str, root: Path, seed: int, val_ratio: float, test_ratio: float):
    name_upper = name.upper()

    if name_upper == "AQSOL":
        train_ds = AQSOL(root=str(root / "AQSOL"), split="train")
        val_ds = AQSOL(root=str(root / "AQSOL"), split="val")
        test_ds = AQSOL(root=str(root / "AQSOL"), split="test")
        return train_ds, val_ds, test_ds, {"split_type": "predefined"}

    valid = {"ESOL", "LIPO", "FREESOLV"}
    if name_upper not in valid:
        raise ValueError(f"Unknown dataset={name}. Choose from ESOL, Lipo, FreeSolv, AQSOL.")

    canonical_name = {"ESOL": "ESOL", "LIPO": "Lipo", "FREESOLV": "FreeSolv"}[name_upper]
    dataset = MoleculeNet(root=str(root / "MoleculeNet"), name=canonical_name)
    train_idx, val_idx, test_idx = split_indices(len(dataset), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    meta = {
        "split_type": "random",
        "n_total": len(dataset),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "seed": seed,
    }
    return train_ds, val_ds, test_ds, meta

# Train
def train_epoch(model, loader, optimizer, scaler: TargetScaler, device: torch.device) -> float:
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
def evaluate(model, loader, scaler: TargetScaler, device: torch.device) -> Dict[str, float]:
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
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = safe_r2(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n": int(len(y_true)),
    }

def main():
    parser = argparse.ArgumentParser(description="Clean GNN benchmark for ESOL/Lipo/AQSOL/FreeSolv.")
    parser.add_argument("--dataset", type=str, default="ESOL", help="ESOL, Lipo, FreeSolv, or AQSOL")
    parser.add_argument("--model", type=str, default="GAT", help="GCN or GAT")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--outdir", type=str, default="benchmark_outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, split_meta = load_dataset(
        name=args.dataset,
        root=data_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sample = next(iter(train_loader))
    sample_x = sample.x
    if sample_x.dim() == 1:
        in_channels = 1
    else:
        in_channels = int(sample.x.size(-1))

    scaler = TargetScaler.from_loader(train_loader)
    model = build_model(
        model_name=args.model,
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, scaler, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{args.dataset}/{args.model}] epoch={epoch:03d} "
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

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best checkpoint was stored.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, scaler, device)

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "device": str(device),
        "split_meta": split_meta,
        "train_target_mean": scaler.mean,
        "train_target_std": scaler.std,
        "in_channels": in_channels,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_metrics": test_metrics,
        "history": history,
    }

    outfile = outdir / f"{args.dataset.lower()}_{args.model.lower()}_metrics.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Saved metrics to:", outfile)
    print(json.dumps({
        "dataset": args.dataset,
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_metrics": test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
