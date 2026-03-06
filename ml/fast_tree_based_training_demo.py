#!/usr/bin/env python3
"""
Run:
python3 ml/fast_tree_based_training_demo.py --in data/quantum_espress_placeholder.csv --model hgb
python3 ml/fast_tree_based_training_demo.py --in data/quantum_espress_placeholder.csv --model extratrees
python3 ml/fast_tree_based_training_demo.py --in data/quantum_espress_placeholder.csv --model rf

Models:
- hgb        : HistGradientBoostingRegressor
- extratrees : ExtraTreesRegressor
- rf         : RandomForestRegressor

Notes:
- Uses stratified train/test split for regression by binning the target.
- Does a small randomized hyperparameter search on the training set only.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)

MAX_ROWS = 200_000

BASE_NUMERIC = [
    "Charge", "XLogP", "TPSA",
    "HBondDonorCount", "HBondAcceptorCount",
    "RotatableBondCount", "MolecularWeight", "ExactMass",
]

DROP_COLS = {
    "bucket", "CID", "source_query",
    "SMILES", "ConnectivitySMILES", "ConnectivitySMILES.1",
    "InChIKey", "MolecularFormula",
    "IUPACName", "Title", "compound_name",
    "bucket_membership",
    "pfas_id", "pfas_class",
}

def _to_01_bool(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        else:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map({"True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0})
                .fillna(0)
                .astype(int)
            )

def infer_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    pfas_onehot = [c for c in df.columns if c.startswith("pfas_") and c not in ("pfas_id", "pfas_class")]
    pfas_class_cols = [c for c in ["pfas_is_long", "pfas_is_short", "pfas_is_ultrashort"] if c in df.columns]

    feat_cols = [c for c in BASE_NUMERIC if c in df.columns] + flag_cols + pfas_onehot + pfas_class_cols
    feat_cols = [c for c in feat_cols if c != target]
    feat_cols = [c for c in feat_cols if c not in DROP_COLS]

    # preserve order, remove duplicates
    seen = set()
    out = []
    for c in feat_cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def make_strat_bins(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Create bins for stratified regression split.
    Falls back gracefully if too many duplicate values exist.
    """
    y_series = pd.Series(y)
    try:
        bins = pd.qcut(y_series, q=min(n_bins, y_series.nunique()), labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y_series, bins=min(n_bins, max(2, y_series.nunique())), labels=False, duplicates="drop")

    bins = bins.astype(int)

    # If some bins are too small, reduce bin count until stratification is safe
    while bins.value_counts().min() < 2 and bins.nunique() > 2:
        nb = bins.nunique() - 1
        try:
            bins = pd.qcut(y_series, q=nb, labels=False, duplicates="drop").astype(int)
        except ValueError:
            bins = pd.cut(y_series, bins=nb, labels=False, duplicates="drop").astype(int)

    return bins.to_numpy()

def get_model_and_space(model_name: str, seed: int):
    if model_name == "hgb":
        base_model = HistGradientBoostingRegressor(
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        param_space = {
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "max_depth": [None, 4, 6, 8, 10],
            "max_iter": [200, 300, 400, 600],
            "min_samples_leaf": [10, 20, 30, 50],
            "l2_regularization": [0.0, 0.01, 0.1, 1.0],
            "max_bins": [128, 255],
        }

    elif model_name == "extratrees":
        base_model = ExtraTreesRegressor(
            random_state=seed,
            n_jobs=-1,
        )
        param_space = {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
            "bootstrap": [False, True],
        }

    elif model_name == "rf":
        base_model = RandomForestRegressor(
            random_state=seed,
            n_jobs=-1,
        )
        param_space = {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5, 0.8],
            "bootstrap": [True],
        }

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return base_model, param_space

def cv_score_regression_stratified(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    y_bins: np.ndarray,
    n_splits: int = 3,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Returns mean CV MAE and mean CV R2 using stratified folds over target bins.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes, r2s = [], []

    for train_idx, val_idx in skf.split(X, y_bins):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)

        maes.append(mean_absolute_error(y_val, pred))
        r2s.append(r2_score(y_val, pred))

    return float(np.mean(maes)), float(np.mean(r2s))

def small_random_search(
    base_model,
    param_space: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    y_train_bins: np.ndarray,
    n_iter: int,
    seed: int,
):
    best_model = None
    best_params = None
    best_mae = float("inf")
    best_r2 = float("-inf")
    results = []

    sampler = list(ParameterSampler(param_space, n_iter=n_iter, random_state=seed))

    for params in sampler:
        model = clone(base_model)
        model.set_params(**params)

        cv_mae, cv_r2 = cv_score_regression_stratified(
            model=model,
            X=X_train,
            y=y_train,
            y_bins=y_train_bins,
            n_splits=3,
            seed=seed,
        )

        results.append({
            "params": params,
            "cv_mae": cv_mae,
            "cv_r2": cv_r2,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/quantum_espress_placeholder.csv")
    ap.add_argument("--target", default="y_placeholder")
    ap.add_argument("--model", choices=["hgb", "extratrees", "rf"], default="hgb")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-search", type=int, default=12, help="number of random hyperparameter configs to try")
    ap.add_argument("--n-strat-bins", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.inp)

    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=args.seed)
        print(f"[info] Subsampled to {len(df)} rows")

    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found in {args.inp}")

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    pfas_cols = [c for c in df.columns if c.startswith("pfas_") and c not in ("pfas_id", "pfas_class")]
    class_cols = [c for c in ["pfas_is_long", "pfas_is_short", "pfas_is_ultrashort"] if c in df.columns]
    _to_01_bool(df, flag_cols + pfas_cols + class_cols)

    feat_cols = infer_feature_columns(df, args.target)
    if not feat_cols:
        raise SystemExit("No feature columns found. Check CSV columns / prefixes.")

    X = df[feat_cols].copy()
    y = pd.to_numeric(df[args.target], errors="coerce").to_numpy()

    valid_mask = np.isfinite(y)
    X = X.loc[valid_mask].copy()
    y = y[valid_mask]

    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()].copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0)

    y_bins = make_strat_bins(y, n_bins=args.n_strat_bins)

    X_train, X_test, y_train, y_test, y_train_bins, y_test_bins = train_test_split(
        X, y, y_bins,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_bins,
    )

    print("[info] Train target bin counts:")
    print(pd.Series(y_train_bins).value_counts().sort_index().to_dict())
    print("[info] Test target bin counts:")
    print(pd.Series(y_test_bins).value_counts().sort_index().to_dict())

    base_model, param_space = get_model_and_space(args.model, args.seed)

    best_model, best_params, search_results, best_cv_mae, best_cv_r2 = small_random_search(
        base_model=base_model,
        param_space=param_space,
        X_train=X_train,
        y_train=y_train,
        y_train_bins=y_train_bins,
        n_iter=args.n_search,
        seed=args.seed,
    )

    pred = best_model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    bundle: Dict[str, Any] = {
        "model": best_model,
        "feature_columns": feat_cols,
        "target": args.target,
        "metrics": {
            "MAE": mae,
            "R2": r2,
            "CV_MAE_best": best_cv_mae,
            "CV_R2_best": best_cv_r2,
        },
        "best_params": best_params,
        "train_config": {
            "input": args.inp,
            "model": args.model,
            "test_size": args.test_size,
            "seed": args.seed,
            "n_rows": int(len(X)),
            "n_features": int(len(feat_cols)),
            "n_search": args.n_search,
            "n_strat_bins": args.n_strat_bins,
        },
        "search_results": search_results,
    }

    model_path = os.path.join(args.outdir, f"fast_tree_{args.model}.joblib")
    metrics_path = os.path.join(args.outdir, f"fast_tree_{args.model}_metrics.json")

    joblib.dump(bundle, model_path)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                **bundle["train_config"],
                **bundle["metrics"],
                "best_params": best_params,
            },
            f,
            indent=2,
        )

    print(f"[done] saved model:   {model_path}")
    print(f"[done] saved metrics: {metrics_path}")
    print(f"[metrics] Test MAE={mae:.4f}  Test R2={r2:.4f}")
    print(f"[metrics] Best CV MAE={best_cv_mae:.4f}  Best CV R2={best_cv_r2:.4f}")
    print(f"[best params] {best_params}")

    if args.model in {"extratrees", "rf"}:
        importances = best_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:25]
        print("\n[top 25 features]")
        for i in top_idx:
            print(f"{feat_cols[i]:<30} {importances[i]:.6f}")

if __name__ == "__main__":
    main()