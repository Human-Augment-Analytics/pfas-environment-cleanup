#!/usr/bin/env python3
"""
Run:

python3 ml/fast_tree_based_training_demo.py --in data/quantum_espress_placeholder.csv --model <model_name>

model_name - can either be hgb or extratrees

Note: extratrees is just a collection of random trees; hgb is a gradient boosting model(sequential)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor

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
                .map({"True": 1, "False": 0, "1": 1, "0": 0})
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

    return feat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/quantum_espress_placeholder.csv")
    ap.add_argument("--target", default="y_placeholder")
    ap.add_argument("--model", choices=["hgb", "extratrees"], default="hgb")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.inp)

    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
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
    y = df[args.target].astype(float).to_numpy()

    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()].copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    if args.model == "hgb":
        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=8,
            max_iter=400,
            random_state=args.seed,
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=500,
            random_state=args.seed,
            n_jobs=-1,
            min_samples_leaf=2,
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    bundle: Dict[str, Any] = {
        "model": model,
        "feature_columns": feat_cols,
        "target": args.target,
        "metrics": {"MAE": mae, "R2": r2},
        "train_config": {
            "input": args.inp,
            "model": args.model,
            "test_size": args.test_size,
            "seed": args.seed,
            "n_rows": int(len(df)),
            "n_features": int(len(feat_cols)),
        },
    }

    model_path = os.path.join(args.outdir, f"fast_tree_{args.model}.joblib")
    metrics_path = os.path.join(args.outdir, f"fast_tree_{args.model}_metrics.json")

    joblib.dump(bundle, model_path)
    with open(metrics_path, "w") as f:
        json.dump(bundle["train_config"] | bundle["metrics"], f, indent=2)

    print(f"[done] saved model:   {model_path}")
    print(f"[done] saved metrics: {metrics_path}")
    print(f"[metrics] MAE={mae:.4f}  R2={r2:.4f}")

    if args.model == "extratrees":
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:25]
        print("\n[top 25 features]")
        for i in top_idx:
            print(f"{feat_cols[i]:<30} {importances[i]:.6f}")

if __name__ == "__main__":
    main()