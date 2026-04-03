#!/usr/bin/env python3
"""
Offline BO/UCB benchmark for tabular molecular-property surrogates.

This mirrors the style of benchmark_ml_regression.py and bo_ucb_selector.py,
but treats each benchmark dataset as a closed candidate pool. Labels are hidden
until selected, letting you compare acquisition strategies and surrogate models
in a sample-efficient discovery setting.

Supported datasets:
- ESOL
- Lipo
- FreeSolv
- AQSOL

Supported surrogate models:
- rf
- extratrees
- hgb

Example:
python benchmark_ml_bo_ucb.py --dataset ESOL --model rf --acquisition ucb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from benchmark_ml_regression import dataset_to_xy, load_dataset, set_seed


# -----------------------------
# Dataset helpers
# -----------------------------

def load_full_xy(name: str, root: Path, seed: int, val_ratio: float, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    train_ds, val_ds, test_ds, split_meta = load_dataset(
        name=name,
        root=root,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    X_train, y_train = dataset_to_xy(train_ds)
    X_val, y_val = dataset_to_xy(val_ds)
    X_test, y_test = dataset_to_xy(test_ds)

    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)

    meta = dict(split_meta)
    meta["n_pool_total"] = int(len(y))
    meta["bo_mode"] = "offline_full_pool"
    return X, y, meta


# -----------------------------
# Surrogates
# -----------------------------

def make_surrogate(model_name: str, seed: int):
    model_name = model_name.lower()

    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
            bootstrap=True,
            max_features="sqrt",
        )

    if model_name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
            bootstrap=False,
            max_features="sqrt",
        )

    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            random_state=seed,
            learning_rate=0.05,
            max_depth=8,
            max_iter=300,
            min_samples_leaf=10,
            l2_regularization=0.01,
            early_stopping=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def fit_ensemble(model_name: str, X_labeled: np.ndarray, y_labeled: np.ndarray, ensemble_size: int, seed: int):
    rng = np.random.default_rng(seed)
    models = []
    n = len(X_labeled)
    all_idx = np.arange(n)

    for _ in range(ensemble_size):
        boot_idx = rng.choice(all_idx, size=n, replace=True)
        model = clone(make_surrogate(model_name, seed=int(rng.integers(1_000_000_000))))
        model.fit(X_labeled[boot_idx], y_labeled[boot_idx])
        models.append(model)

    return models


def predict_ensemble_mu_sigma(models, X_pool: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    preds = np.stack([m.predict(X_pool) for m in models], axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu, sigma


# -----------------------------
# BO helpers
# -----------------------------

def acquisition_score(
    mu: np.ndarray,
    sigma: np.ndarray,
    mode: str = "ucb",
    beta: float = 2.0,
    objective: str = "maximize",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    mode = mode.lower()
    objective = objective.lower()

    if rng is None:
        rng = np.random.default_rng(42)

    if mode == "random":
        return rng.random(len(mu))

    sign = 1.0 if objective == "maximize" else -1.0

    if mode == "greedy":
        return sign * mu

    if mode == "ucb":
        return sign * mu + beta * sigma

    raise ValueError(f"Unknown acquisition mode: {mode}")


def make_initial_seed_and_pool(n: int, init_frac: float, min_seed: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_seed = max(min_seed, int(round(n * init_frac)))
    n_seed = min(max(1, n_seed), n - 1)
    all_idx = np.arange(n)
    labeled = np.sort(rng.choice(all_idx, size=n_seed, replace=False))
    pool = np.setdiff1d(all_idx, labeled, assume_unique=False)
    return labeled, pool


def is_top_candidate(y: np.ndarray, threshold: float, objective: str) -> np.ndarray:
    if objective == "maximize":
        return y >= threshold
    return y <= threshold


def get_top_threshold(y: np.ndarray, top_frac: float, objective: str) -> float:
    if objective == "maximize":
        return float(np.quantile(y, 1.0 - top_frac))
    return float(np.quantile(y, top_frac))


def best_value(y: np.ndarray, objective: str) -> float:
    return float(np.max(y)) if objective == "maximize" else float(np.min(y))


def best_seen(y: np.ndarray, objective: str) -> float:
    return best_value(y, objective)


def oracle_gap(best_seen_value: float, oracle_value: float, objective: str) -> float:
    if objective == "maximize":
        return float(oracle_value - best_seen_value)
    return float(best_seen_value - oracle_value)


def build_ranked_pool_df(
    candidate_ids: np.ndarray,
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    score: np.ndarray,
    top_threshold: float,
    objective: str,
) -> pd.DataFrame:
    ranked = pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "y_true_hidden": y_true,
            "mu": mu,
            "sigma": sigma,
            "score": score,
        }
    )
    ranked["is_global_top_frac"] = is_top_candidate(ranked["y_true_hidden"].to_numpy(), top_threshold, objective)
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def summarize_round(
    round_idx: int,
    query_count: int,
    acquisition: str,
    objective: str,
    labeled_idx: np.ndarray,
    pool_idx: np.ndarray,
    selected_df: pd.DataFrame,
    y_all: np.ndarray,
    oracle_value: float,
    top_threshold: float,
) -> Dict[str, Any]:
    y_labeled = y_all[labeled_idx]
    best_true_seen = best_seen(y_labeled, objective)
    cumulative_top_hits = int(is_top_candidate(y_labeled, top_threshold, objective).sum())

    return {
        "round": int(round_idx),
        "queries_so_far": int(query_count),
        "acquisition": acquisition,
        "objective": objective,
        "n_labeled": int(len(labeled_idx)),
        "n_pool": int(len(pool_idx)),
        "n_selected": int(len(selected_df)),
        "selected_mean_mu": float(selected_df["mu"].mean()) if len(selected_df) else None,
        "selected_mean_sigma": float(selected_df["sigma"].mean()) if len(selected_df) else None,
        "selected_mean_true_y": float(selected_df["y_true_hidden"].mean()) if len(selected_df) else None,
        "selected_top_hits": int(selected_df["is_global_top_frac"].sum()) if len(selected_df) else 0,
        "cumulative_top_hits": cumulative_top_hits,
        "cumulative_top_hit_rate": float(cumulative_top_hits / max(1, len(labeled_idx))),
        "best_true_seen": float(best_true_seen),
        "oracle_value": float(oracle_value),
        "oracle_gap": float(oracle_gap(best_true_seen, oracle_value, objective)),
    }


def bo_loop(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    model_name: str,
    outdir: Path,
    acquisition: str,
    objective: str,
    beta: float,
    ensemble_size: int,
    init_frac: float,
    min_seed: int,
    batch_size: int,
    num_rounds: int,
    top_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    candidate_ids = np.arange(len(y))
    labeled_idx, pool_idx = make_initial_seed_and_pool(len(y), init_frac=init_frac, min_seed=min_seed, seed=seed)

    top_threshold = get_top_threshold(y, top_frac=top_frac, objective=objective)
    oracle_value = best_value(y, objective)

    history: List[Dict[str, Any]] = []
    selected_all: List[pd.DataFrame] = []

    for round_idx in range(num_rounds):
        if len(pool_idx) == 0:
            break

        if acquisition.lower() == "random":
            mu = np.full(shape=len(pool_idx), fill_value=np.nan, dtype=np.float32)
            sigma = np.full(shape=len(pool_idx), fill_value=np.nan, dtype=np.float32)
            score = acquisition_score(mu=mu, sigma=sigma, mode="random", beta=beta, objective=objective, rng=rng)
        else:
            models = fit_ensemble(
                model_name=model_name,
                X_labeled=X[labeled_idx],
                y_labeled=y[labeled_idx],
                ensemble_size=ensemble_size,
                seed=seed + round_idx,
            )
            mu, sigma = predict_ensemble_mu_sigma(models, X[pool_idx])
            score = acquisition_score(mu=mu, sigma=sigma, mode=acquisition, beta=beta, objective=objective, rng=rng)

        ranked = build_ranked_pool_df(
            candidate_ids=candidate_ids[pool_idx],
            y_true=y[pool_idx],
            mu=mu,
            sigma=sigma,
            score=score,
            top_threshold=top_threshold,
            objective=objective,
        )

        selected = ranked.head(min(batch_size, len(ranked))).copy()
        selected["round"] = round_idx
        selected["query_order_start"] = len(labeled_idx)
        selected_all.append(selected)

        round_ranked_path = outdir / f"{dataset_name.lower()}_{model_name.lower()}_{acquisition.lower()}_ranked_pool_round{round_idx}.csv"
        round_selected_path = outdir / f"{dataset_name.lower()}_{model_name.lower()}_{acquisition.lower()}_selected_batch_round{round_idx}.csv"
        ranked.to_csv(round_ranked_path, index=False)
        selected.to_csv(round_selected_path, index=False)

        selected_ids = selected["candidate_id"].to_numpy(dtype=int)
        labeled_idx = np.concatenate([labeled_idx, selected_ids])
        pool_idx = np.setdiff1d(pool_idx, selected_ids, assume_unique=False)

        history.append(
            summarize_round(
                round_idx=round_idx,
                query_count=len(labeled_idx),
                acquisition=acquisition,
                objective=objective,
                labeled_idx=labeled_idx,
                pool_idx=pool_idx,
                selected_df=selected,
                y_all=y,
                oracle_value=oracle_value,
                top_threshold=top_threshold,
            )
        )

    selected_df = pd.concat(selected_all, ignore_index=True) if selected_all else pd.DataFrame()
    return history, selected_df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline BO/UCB benchmark for tabular molecular-property surrogates.")
    parser.add_argument("--dataset", type=str, default="ESOL", help="ESOL, Lipo, FreeSolv, or AQSOL")
    parser.add_argument("--model", choices=["rf", "extratrees", "hgb"], default="rf")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--outdir", type=str, default="benchmark_bo_outputs")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--acquisition", choices=["ucb", "greedy", "random"], default="ucb")
    parser.add_argument("--objective", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--init-frac", type=float, default=0.1)
    parser.add_argument("--min-seed", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--top-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, split_meta = load_full_xy(
        name=args.dataset,
        root=data_root,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    history, selected_df = bo_loop(
        X=X,
        y=y,
        dataset_name=args.dataset,
        model_name=args.model,
        outdir=outdir,
        acquisition=args.acquisition,
        objective=args.objective,
        beta=args.beta,
        ensemble_size=args.ensemble_size,
        init_frac=args.init_frac,
        min_seed=args.min_seed,
        batch_size=args.batch_size,
        num_rounds=args.rounds,
        top_frac=args.top_frac,
        seed=args.seed,
    )

    history_df = pd.DataFrame(history)
    if not history_df.empty:
        auc_best_so_far = float(np.trapz(history_df["best_true_seen"].to_numpy(), history_df["queries_so_far"].to_numpy()))
        final_summary = history[-1]
    else:
        auc_best_so_far = None
        final_summary = None

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "mode": "offline_bo_pool_search",
        "surrogate_family": "tabular",
        "acquisition": args.acquisition,
        "objective": args.objective,
        "beta": args.beta,
        "ensemble_size": args.ensemble_size,
        "init_frac": args.init_frac,
        "min_seed": args.min_seed,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "top_frac": args.top_frac,
        "seed": args.seed,
        "n_candidates": int(len(y)),
        "n_features": int(X.shape[1]),
        "split_meta": split_meta,
        "history": history,
        "auc_best_so_far": auc_best_so_far,
        "final_summary": final_summary,
    }

    metrics_out = outdir / f"{args.dataset.lower()}_{args.model.lower()}_{args.acquisition.lower()}_bo_metrics.json"
    history_out = outdir / f"{args.dataset.lower()}_{args.model.lower()}_{args.acquisition.lower()}_bo_history.csv"
    selected_out = outdir / f"{args.dataset.lower()}_{args.model.lower()}_{args.acquisition.lower()}_bo_selected_all.csv"

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    history_df.to_csv(history_out, index=False)
    selected_df.to_csv(selected_out, index=False)

    print("Saved metrics to:", metrics_out)
    print("Saved history to:", history_out)
    print("Saved selections to:", selected_out)
    if final_summary is not None:
        print(json.dumps({
            "dataset": args.dataset,
            "model": args.model,
            "acquisition": args.acquisition,
            "objective": args.objective,
            "final_summary": final_summary,
            "auc_best_so_far": auc_best_so_far,
        }, indent=2))


if __name__ == "__main__":
    main()
