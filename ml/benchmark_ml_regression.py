#!/usr/bin/env python3
"""
Benchmark tabular regressors on the same molecule datasets used by benchmark_gnn_regression.py.

Supported datasets:
- ESOL
- Lipo
- FreeSolv
- AQSOL

Supported models:
- rf
- extratrees
- hgb
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from torch.utils.data import Subset
from torch_geometric.datasets import AQSOL, MoleculeNet


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float('nan')
    return float(r2_score(y_true, y_pred))


def split_indices(
    n: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError('Require val_ratio > 0, test_ratio > 0, and val_ratio + test_ratio < 1.')

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError('Not enough data left for training. Reduce val_ratio/test_ratio.')

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return train_idx, val_idx, test_idx


def load_dataset(name: str, root: Path, seed: int, val_ratio: float, test_ratio: float):
    name_upper = name.upper()

    if name_upper == 'AQSOL':
        train_ds = AQSOL(root=str(root / 'AQSOL'), split='train')
        val_ds = AQSOL(root=str(root / 'AQSOL'), split='val')
        test_ds = AQSOL(root=str(root / 'AQSOL'), split='test')
        return train_ds, val_ds, test_ds, {'split_type': 'predefined'}

    valid = {'ESOL', 'LIPO', 'FREESOLV'}
    if name_upper not in valid:
        raise ValueError(f'Unknown dataset={name}. Choose from ESOL, Lipo, FreeSolv, or AQSOL.')

    canonical_name = {'ESOL': 'ESOL', 'LIPO': 'Lipo', 'FREESOLV': 'FreeSolv'}[name_upper]
    dataset = MoleculeNet(root=str(root / 'MoleculeNet'), name=canonical_name)
    train_idx, val_idx, test_idx = split_indices(
        len(dataset),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    meta = {
        'split_type': 'random',
        'n_total': len(dataset),
        'n_train': len(train_ds),
        'n_val': len(val_ds),
        'n_test': len(test_ds),
        'seed': seed,
    }
    return train_ds, val_ds, test_ds, meta


def _summary_stats(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    d = arr.shape[1]
    if arr.shape[0] == 0:
        return np.zeros(5 * d, dtype=np.float32)

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    amin = arr.min(axis=0)
    amax = arr.max(axis=0)
    asum = arr.sum(axis=0)
    return np.concatenate([mean, std, amin, amax, asum]).astype(np.float32)


def graph_to_feature_vector(data: Any) -> np.ndarray:
    x = data.x.cpu().numpy().astype(np.float32) if getattr(data, 'x', None) is not None else np.zeros((0, 1), dtype=np.float32)
    edge_index = data.edge_index.cpu().numpy() if getattr(data, 'edge_index', None) is not None else np.zeros((2, 0), dtype=np.int64)
    edge_attr = (
        data.edge_attr.cpu().numpy().astype(np.float32)
        if getattr(data, 'edge_attr', None) is not None
        else np.zeros((0, 1), dtype=np.float32)
    )

    num_nodes = int(x.shape[0])
    num_edges = int(edge_index.shape[1])

    if num_nodes > 0 and num_edges > 0:
        deg = np.bincount(edge_index[0], minlength=num_nodes).astype(np.float32)
    else:
        deg = np.zeros((num_nodes,), dtype=np.float32)

    density = 0.0
    if num_nodes > 1:
        density = num_edges / float(num_nodes * (num_nodes - 1))

    graph_scalars = np.array(
        [
            num_nodes,
            num_edges,
            float(density),
            float(x.shape[1]) if x.ndim == 2 else 0.0,
            float(edge_attr.shape[1]) if edge_attr.ndim == 2 else 0.0,
        ],
        dtype=np.float32,
    )

    node_stats = _summary_stats(x)
    edge_stats = _summary_stats(edge_attr)
    degree_stats = _summary_stats(deg.reshape(-1, 1))

    return np.concatenate([graph_scalars, node_stats, edge_stats, degree_stats]).astype(np.float32)


def dataset_to_xy(ds) -> Tuple[np.ndarray, np.ndarray]:
    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []

    for data in ds:
        X_rows.append(graph_to_feature_vector(data))
        y_rows.append(float(data.y.view(-1)[0].item()))

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    return X, y


def get_model_and_space(model_name: str, seed: int):
    model_name = model_name.lower()

    if model_name == 'hgb':
        base_model = HistGradientBoostingRegressor(
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        param_space = {
            'learning_rate': [0.03, 0.05, 0.08, 0.1],
            'max_depth': [None, 4, 6, 8, 10],
            'max_iter': [200, 300, 400, 600],
            'min_samples_leaf': [5, 10, 20, 30],
            'l2_regularization': [0.0, 0.01, 0.1, 1.0],
            'max_bins': [128, 255],
        }

    elif model_name == 'extratrees':
        base_model = ExtraTreesRegressor(
            random_state=seed,
            n_jobs=-1,
        )
        param_space = {
            'n_estimators': [200, 400, 600],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
            'bootstrap': [False, True],
        }

    elif model_name == 'rf':
        base_model = RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
        )
        param_space = {
            'n_estimators': [200, 400, 600],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5, 0.8],
            'bootstrap': [True],
        }

    else:
        raise ValueError(f'Unsupported model: {model_name}')

    return base_model, param_space


def make_cv_folds(n: int, n_folds: int = 3, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    folds = np.array_split(idx, n_folds)
    out: List[Tuple[np.ndarray, np.ndarray]] = []

    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        out.append((train_idx, val_idx))
    return out


def cv_score_regression(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 3,
    seed: int = 42,
) -> Tuple[float, float]:
    maes, r2s = [], []

    for train_idx, val_idx in make_cv_folds(len(X), n_folds=n_splits, seed=seed):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)

        maes.append(mean_absolute_error(y_val, pred))
        r2s.append(safe_r2(y_val, pred))

    return float(np.mean(maes)), float(np.mean(r2s))


def small_random_search(
    base_model,
    param_space: Dict[str, List[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int,
    seed: int,
):
    best_model = None
    best_params = None
    best_mae = float('inf')
    best_r2 = float('-inf')
    results = []

    sampler = list(ParameterSampler(param_space, n_iter=n_iter, random_state=seed))

    for params in sampler:
        model = clone(base_model)
        model.set_params(**params)

        cv_mae, cv_r2 = cv_score_regression(
            model=model,
            X=X_train,
            y=y_train,
            n_splits=3,
            seed=seed,
        )

        results.append({
            'params': params,
            'cv_mae': cv_mae,
            'cv_r2': cv_r2,
        })

        if cv_mae < best_mae:
            best_mae = cv_mae
            best_r2 = cv_r2
            best_params = params
            best_model = clone(base_model).set_params(**params)

    assert best_model is not None
    best_model.fit(X_train, y_train)
    return best_model, best_params, results, best_mae, best_r2


def main():
    parser = argparse.ArgumentParser(description='Tabular baselines for the same datasets used in benchmark_gnn_regression.py')
    parser.add_argument('--dataset', type=str, default='ESOL', help='ESOL, Lipo, FreeSolv, or AQSOL')
    parser.add_argument('--model', choices=['rf', 'extratrees', 'hgb'], default='hgb')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--outdir', type=str, default='benchmark_outputs')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-search', type=int, default=12)
    args = parser.parse_args()

    set_seed(args.seed)

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

    X_train, y_train = dataset_to_xy(train_ds)
    X_val, y_val = dataset_to_xy(val_ds)
    X_test, y_test = dataset_to_xy(test_ds)

    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    base_model, param_space = get_model_and_space(args.model, args.seed)

    best_model, best_params, search_results, best_cv_mae, best_cv_r2 = small_random_search(
        base_model=base_model,
        param_space=param_space,
        X_train=X_train_full,
        y_train=y_train_full,
        n_iter=args.n_search,
        seed=args.seed,
    )

    pred = best_model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(math.sqrt(mean_squared_error(y_test, pred)))
    r2 = safe_r2(y_test, pred)

    result = {
        'dataset': args.dataset,
        'model': args.model,
        'feature_type': 'graph_summary_tabular',
        'split_meta': split_meta,
        'n_features': int(X_train_full.shape[1]),
        'seed': args.seed,
        'n_search': args.n_search,
        'best_params': best_params,
        'cv_metrics_best': {
            'mae': best_cv_mae,
            'r2': best_cv_r2,
        },
        'test_metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n': int(len(y_test)),
        },
        'search_results': search_results,
    }

    outfile = outdir / f'{args.dataset.lower()}_{args.model.lower()}_tabular_metrics.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print('Saved metrics to:', outfile)
    print(json.dumps({
        'dataset': args.dataset,
        'model': args.model,
        'feature_type': 'graph_summary_tabular',
        'n_features': int(X_train_full.shape[1]),
        'best_params': best_params,
        'cv_metrics_best': result['cv_metrics_best'],
        'test_metrics': result['test_metrics'],
    }, indent=2))


if __name__ == '__main__':
    main()
