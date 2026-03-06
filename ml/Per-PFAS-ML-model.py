import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample

BASE_NUMERIC = [
    "Charge","XLogP","TPSA","HBondDonorCount","HBondAcceptorCount",
    "RotatableBondCount","MolecularWeight","ExactMass"
]

PFAS_LIST = ["PFOA","PFOS","PFBA","PFBS","PFPrA","HFPO-DA","TFSI","TFA"]


def to_01_bool(df, cols):
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        else:
            df[c] = (
                df[c].astype(str).str.strip()
                .map({"True":1,"False":0,"true":1,"false":0,"1":1,"0":0})
                .fillna(0).astype(int)
            )

def make_strat_bins(y, n_bins=10):
    y_series = pd.Series(y)
    try:
        bins = pd.qcut(y_series, q=min(n_bins, y_series.nunique()), labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(y_series, bins=min(n_bins, max(2, y_series.nunique())), labels=False, duplicates="drop")
    bins = bins.astype(int)
    while bins.value_counts().min() < 2 and bins.nunique() > 2:
        nb = bins.nunique() - 1
        try:
            bins = pd.qcut(y_series, q=nb, labels=False, duplicates="drop").astype(int)
        except ValueError:
            bins = pd.cut(y_series, bins=nb, labels=False, duplicates="drop").astype(int)
    return bins.to_numpy()

def infer_feature_columns(df, target="y_placeholder"):
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    # IMPORTANT: per-PFAS model should NOT use PFAS one-hots; PFAS is fixed by filtering
    feat_cols = [c for c in BASE_NUMERIC if c in df.columns] + flag_cols
    feat_cols = [c for c in feat_cols if c != target]
    return feat_cols

def load_and_clean(path):
    df = pd.read_csv(path)

    # Bool columns to 0/1
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    to_01_bool(df, flag_cols)

    # Numeric coercion + median fill
    for c in BASE_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[BASE_NUMERIC] = df[BASE_NUMERIC].fillna(df[BASE_NUMERIC].median(numeric_only=True))

    # target
    df["y_placeholder"] = pd.to_numeric(df["y_placeholder"], errors="coerce")
    df = df[np.isfinite(df["y_placeholder"])].copy()
    return df

def fit_rf_ensemble(X_tr, y_tr, M=8, seed=42):
    rng = np.random.default_rng(seed)
    models = []

    n = len(X_tr)
    idx_all = np.arange(n)

    for m in range(M):
        # bootstrap sample
        boot_idx = resample(
            idx_all,
            replace=True,
            n_samples=n,
            random_state=int(rng.integers(1e9))
        )
        model = RandomForestRegressor(
            n_estimators=400,
            random_state=int(rng.integers(1e9)),
            n_jobs=6,
            min_samples_leaf=2
        )
        model.fit(X_tr.iloc[boot_idx], y_tr[boot_idx])
        models.append(model)

    return models

def predict_ensemble_mu_sigma(models, X_te):
    preds = np.stack([m.predict(X_te) for m in models], axis=0)  # [M, n_test]
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu, sigma

def train_eval_per_pfas(df, feat_cols, seed=42, test_size=0.2, use_ensemble=False, M=8):
    X_all = df[feat_cols].copy()
    y_all = df["y_placeholder"].to_numpy()

    y_bins = make_strat_bins(y_all, n_bins=10)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=seed,
        stratify=y_bins
    )

    d_tr = df.iloc[train_idx]
    d_te = df.iloc[test_idx]

    results = []
    overall_mae_num = 0.0
    overall_r2_num = 0.0
    overall_count = 0

    for pfas in PFAS_LIST:
        tr = d_tr[d_tr["pfas_id"] == pfas]
        te = d_te[d_te["pfas_id"] == pfas]

        if len(tr) < 20 or len(te) < 10:
            results.append({
                "pfas_id": pfas,
                "n_train": len(tr),
                "n_test": len(te),
                "MAE": None,
                "R2": None,
                "mean_sigma": None if use_ensemble else None,
            })
            continue

        X_tr = tr[feat_cols]
        y_tr = tr["y_placeholder"].to_numpy()
        X_te = te[feat_cols]
        y_te = te["y_placeholder"].to_numpy()

        if not use_ensemble:
            model = RandomForestRegressor(n_estimators=500, random_state=seed,n_jobs=6)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            mean_sigma = None
        else:
            models = fit_rf_ensemble(X_tr, y_tr, M=M, seed=seed)
            pred, sigma = predict_ensemble_mu_sigma(models, X_te)
            mean_sigma = float(np.mean(sigma))

        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred))

        results.append({
            "pfas_id": pfas,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "MAE": mae,
            "R2": r2,
            "mean_sigma": mean_sigma,
        })

        overall_mae_num += mae * len(te)
        overall_r2_num += r2 * len(te)
        overall_count += len(te)

    overall_mae = overall_mae_num / max(1, overall_count)
    overall_r2 = overall_r2_num / max(1, overall_count)
    return results, overall_mae, overall_r2

# per pfas model:
# def train_eval_per_pfas(df, feat_cols, seed=42, test_size=0.2):
#     X = df[feat_cols].copy()
#     y = df["y_placeholder"].to_numpy()

#     # same style split as Arjot’s (stratified by target bins)
#     y_bins = make_strat_bins(y, n_bins=10)
#     train_idx, test_idx = train_test_split(
#         np.arange(len(df)),
#         test_size=test_size,
#         random_state=seed,
#         stratify=y_bins
#     )

#     results = []
#     overall_mae_num = 0.0
#     overall_count = 0

#     for pfas in PFAS_LIST:
#         d_tr = df.iloc[train_idx]
#         d_te = df.iloc[test_idx]

#         tr = d_tr[d_tr["pfas_id"] == pfas]
#         te = d_te[d_te["pfas_id"] == pfas]

#         if len(tr) < 20 or len(te) < 10:
#             results.append({"pfas_id": pfas, "n_train": len(tr), "n_test": len(te), "MAE": None, "R2": None})
#             continue

#         X_tr = tr[feat_cols]
#         y_tr = tr["y_placeholder"].to_numpy()
#         X_te = te[feat_cols]
#         y_te = te["y_placeholder"].to_numpy()

#         model = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
#         model.fit(X_tr, y_tr)
#         pred = model.predict(X_te)

#         mae = float(mean_absolute_error(y_te, pred))
#         r2 = float(r2_score(y_te, pred))

#         results.append({"pfas_id": pfas, "n_train": len(tr), "n_test": len(te), "MAE": mae, "R2": r2})

#         overall_mae_num += mae * len(te)
#         overall_count += len(te)

#     overall_mae = overall_mae_num / max(1, overall_count)
#     return results, overall_mae

def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "quantum_espress_placeholder.csv"
    df = load_and_clean(path)
    feat_cols = infer_feature_columns(df)

    (root / "models").mkdir(exist_ok=True)

    # Single RF baseline
    per_pfas_base, mae_base, r2_base = train_eval_per_pfas(df, feat_cols, seed=42, use_ensemble=False)
    print("\n=== Per-PFAS SINGLE RF ===")
    for r in per_pfas_base:
        print(r)
    print("Weighted overall MAE:", mae_base)
    print("Weighted overall R2 :", r2_base)

    out_base = {
        "mode": "single_rf_per_pfas",
        "overall_weighted_MAE": mae_base,
        "overall_weighted_R2": r2_base,
        "per_pfas": per_pfas_base,
        "n_rows": int(len(df)),
        "n_features": len(feat_cols),
        "seed": 42,
    }
    with open(root / "models" / "per_pfas_rf_metrics.json", "w") as f:
        json.dump(out_base, f, indent=2)

    # Ensemble mean RF
    per_pfas_ens, mae_ens, r2_ens = train_eval_per_pfas(df, feat_cols, seed=42, use_ensemble=True, M=8)
    print("\n=== Per-PFAS RF ENSEMBLE (mean over M) ===")
    for r in per_pfas_ens:
        print(r)
    print("Weighted overall MAE:", mae_ens)
    print("Weighted overall R2 :", r2_ens)

    out_ens = {
        "mode": "rf_ensemble_mean_per_pfas",
        "ensemble_M": 8,
        "overall_weighted_MAE": mae_ens,
        "overall_weighted_R2": r2_ens,
        "per_pfas": per_pfas_ens,
        "n_rows": int(len(df)),
        "n_features": len(feat_cols),
        "seed": 42,
    }
    with open(root / "models" / "per_pfas_rf_ensemble_metrics.json", "w") as f:
        json.dump(out_ens, f, indent=2)

if __name__ == "__main__":
    main()