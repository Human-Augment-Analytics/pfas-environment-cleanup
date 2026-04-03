import json
import numpy as np
import pandas as pd
from pathlib import Path

from per_pfas_rf_model import (
    load_and_clean,
    infer_feature_columns,
    fit_rf_ensemble,
    predict_ensemble_mu_sigma,
    PFAS_LIST,
)

def acquisition_score(mu, sigma, mode="ucb", beta=2.0, minimize=True, rng=None):
    """
    mode:
      - 'ucb'    : exploration + exploitation
      - 'greedy' : mean only
      - 'random' : random ranking baseline
    """
    if mode == "ucb":
        if minimize:
            return -mu + beta * sigma
        return mu + beta * sigma

    if mode == "greedy":
        if minimize:
            return -mu
        return mu

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng(42)
        return rng.random(len(mu))

    raise ValueError(f"Unknown acquisition mode: {mode}")


# seed / pool split
def make_initial_seed_and_pool(df, seed_frac=0.10, min_seed_per_pfas=20, seed=42):
    rng = np.random.default_rng(seed)
    labeled_idx = []

    for pfas in PFAS_LIST:
        sub_idx = df.index[df["pfas_id"] == pfas].to_numpy()
        if len(sub_idx) == 0:
            continue
        n_seed = max(min_seed_per_pfas, int(len(sub_idx) * seed_frac))
        n_seed = min(n_seed, len(sub_idx))
        chosen = rng.choice(sub_idx, size=n_seed, replace=False)
        labeled_idx.extend(chosen.tolist())

    labeled_idx = set(labeled_idx)
    labeled = df.loc[df.index.isin(labeled_idx)].copy()
    pool = df.loc[~df.index.isin(labeled_idx)].copy()
    return labeled, pool


# rank pool with surrogate and acquisition
def rank_pool(
    labeled_df,
    pool_df,
    feat_cols,
    acquisition="ucb",
    beta=2.0,
    M=8,
    seed=42,
):
    rng = np.random.default_rng(seed)
    ranked_parts = []

    for pfas in PFAS_LIST:
        tr = labeled_df[labeled_df["pfas_id"] == pfas].copy()
        cand = pool_df[pool_df["pfas_id"] == pfas].copy()

        if len(tr) < 20 or len(cand) == 0:
            continue

        X_tr = tr[feat_cols]
        y_tr = tr["y_placeholder"].to_numpy()

        models = fit_rf_ensemble(X_tr, y_tr, M=M, seed=seed)

        X_pool = cand[feat_cols]
        mu, sigma = predict_ensemble_mu_sigma(models, X_pool)

        cand["mu"] = mu
        cand["sigma"] = sigma
        cand["score"] = acquisition_score(
            mu=mu,
            sigma=sigma,
            mode=acquisition,
            beta=beta,
            minimize=True,
            rng=rng,
        )

        # keep hidden truth for offline simulation only
        if "y_placeholder" in cand.columns:
            cand["true_y_hidden"] = cand["y_placeholder"]

        ranked_parts.append(cand)

    if not ranked_parts:
        return pd.DataFrame()

    ranked = pd.concat(ranked_parts, ignore_index=True)

    # sort descending because higher acquisition score is better
    ranked = ranked.sort_values(
        ["pfas_id", "score"], ascending=[True, False]
    ).reset_index(drop=True)

    ranked["rank_within_pfas"] = ranked.groupby("pfas_id").cumcount() + 1
    return ranked


# batch selection
def select_batch(ranked_df, batch_size_per_pfas=5):
    if ranked_df.empty:
        return ranked_df

    # dedupe by pfas + candidate_id if present
    dedupe_cols = [c for c in ["pfas_id", "candidate_id"] if c in ranked_df.columns]
    if len(dedupe_cols) == 2:
        ranked_df = ranked_df.drop_duplicates(subset=dedupe_cols)

    selected = (
        ranked_df.groupby("pfas_id", group_keys=False)
        .head(batch_size_per_pfas)
        .reset_index(drop=True)
    )
    return selected


# evaluate
def evaluate_selected_offline(selected_df):
    """
    For now, the 'true' result is already present in y_placeholder.
    Later this function gets replaced by QE result parsing.
    """
    observed = selected_df.copy()
    observed["observed_y"] = observed["y_placeholder"]
    return observed


# update labeled pool state
def update_state(labeled_df, pool_df, observed_df):
    labeled_next = pd.concat([labeled_df, observed_df], ignore_index=True)

    # remove selected rows from pool
    if "candidate_id" in pool_df.columns and "candidate_id" in observed_df.columns:
        remove_keys = set(zip(observed_df["pfas_id"], observed_df["candidate_id"]))
        keep_mask = [
            (pfas, cid) not in remove_keys
            for pfas, cid in zip(pool_df["pfas_id"], pool_df["candidate_id"])
        ]
        pool_next = pool_df.loc[keep_mask].copy()
    else:
        # fallback: use exact row matches by index-like merge on available columns
        remove_cols = [c for c in ["pfas_id", "smiles"] if c in pool_df.columns and c in observed_df.columns]
        if remove_cols:
            merged = pool_df.merge(
                observed_df[remove_cols].drop_duplicates(),
                on=remove_cols,
                how="left",
                indicator=True,
            )
            pool_next = pool_df.loc[merged["_merge"] == "left_only"].copy()
        else:
            # weakest fallback if no unique ID exists
            pool_next = pool_df.drop(index=observed_df.index, errors="ignore").copy()

    return labeled_next, pool_next


# summarize rounds
def summarize_round(round_idx, acquisition, beta, labeled_df, pool_df, selected_df):
    summary = {
        "round": round_idx,
        "acquisition": acquisition,
        "beta": beta,
        "n_labeled": int(len(labeled_df)),
        "n_pool": int(len(pool_df)),
        "n_selected": int(len(selected_df)),
        "selected_mean_mu": float(selected_df["mu"].mean()) if len(selected_df) else None,
        "selected_mean_sigma": float(selected_df["sigma"].mean()) if len(selected_df) else None,
        "best_true_seen": float(labeled_df["y_placeholder"].min()) if len(labeled_df) else None,
    }
    return summary


# BO loop
def bo_loop(
    df,
    feat_cols,
    outdir,
    num_rounds=3,
    acquisition="ucb",
    beta=2.0,
    M=8,
    batch_size_per_pfas=5,
    seed=42,
):
    labeled, pool = make_initial_seed_and_pool(
        df, seed_frac=0.10, min_seed_per_pfas=20, seed=seed
    )

    history = []

    for round_idx in range(num_rounds):
        if pool.empty:
            print(f"Stopping early: pool empty at round {round_idx}")
            break

        ranked = rank_pool(
            labeled_df=labeled,
            pool_df=pool,
            feat_cols=feat_cols,
            acquisition=acquisition,
            beta=beta,
            M=M,
            seed=seed + round_idx,
        )

        selected = select_batch(ranked, batch_size_per_pfas=batch_size_per_pfas)

        # offline now; later replace with QE call + parser
        observed = evaluate_selected_offline(selected)

        # save round artifacts
        ranked.to_csv(outdir / f"{acquisition}_ranked_pool_round{round_idx}.csv", index=False)
        selected.to_csv(outdir / f"{acquisition}_selected_batch_round{round_idx}.csv", index=False)

        # update labeled/pool for next round
        labeled, pool = update_state(labeled, pool, observed)

        round_summary = summarize_round(
            round_idx=round_idx,
            acquisition=acquisition,
            beta=beta,
            labeled_df=labeled,
            pool_df=pool,
            selected_df=selected,
        )
        history.append(round_summary)

        print(f"\n=== Round {round_idx} ===")
        print(round_summary)

    return labeled, pool, history


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "quantum_espress_placeholder.csv"
    outdir = root / "proposals"
    outdir.mkdir(exist_ok=True)

    df = load_and_clean(data_path)

    # create a unique candidate_id if you do not already have one
    if "candidate_id" not in df.columns:
        df = df.reset_index(drop=True).copy()
        df["candidate_id"] = df.index.astype(str)

    feat_cols = infer_feature_columns(df)

    # UCB run
    acquisition="greedy"     # later compare with "greedy" and "random"
    labeled_final, pool_final, history = bo_loop(
        df=df,
        feat_cols=feat_cols,
        outdir=outdir,
        num_rounds=3,
        acquisition=acquisition,
        beta=2.0,
        M=8,
        batch_size_per_pfas=5,
        seed=42,
    )

    # save history
    with open(outdir / f"bo_history_{acquisition}.json", "w") as f:
        json.dump(history, f, indent=2)

    pd.DataFrame(history).to_csv(outdir / f"bo_history_{acquisition}.csv", index=False)

    print("\nSaved BO history to:")
    print(outdir / f"bo_history_{acquisition}.json")
    print(outdir / f"bo_history_{acquisition}.csv")


if __name__ == "__main__":
    main()