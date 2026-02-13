#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

PFAS_LIST = ["PFOA", "PFOS", "PFBA", "PFBS", "PFPrA", "HFPO-DA", "TFSI", "TFA"]
PFAS_CLASS = {
    "PFOA": "Long",
    "PFOS": "Long",
    "PFBA": "Short",
    "PFBS": "Short",
    "PFPrA": "Ultrashort",
    "HFPO-DA": "Short",
    "TFSI": "Ultrashort",
    "TFA": "Ultrashort",
}

BASE_NUMERIC = [
    "Charge", "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount",
    "RotatableBondCount", "MolecularWeight", "ExactMass",
]

def make_placeholder_score(df: pd.DataFrame, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def g(col: str) -> np.ndarray:
        return df[col].to_numpy() if col in df.columns else np.zeros(len(df))

    # Core physchem
    charge = np.clip(g("Charge"), -3, 3)
    xlogp = np.clip(g("XLogP"), -2, 6)
    tpsa = np.clip(g("TPSA"), 0, 250)
    mw = g("MolecularWeight")

    cationic = (
        2.5 * g("flag_quat_ammonium")
        + 2.0 * g("flag_imidazolium")
        + 1.8 * g("flag_pyridinium")
        + 1.2 * g("flag_guanidine")
    )
    hbondish = (
        0.10 * g("HBondDonorCount")
        + 0.08 * g("HBondAcceptorCount")
        + 0.30 * g("flag_urea")
        + 0.35 * g("flag_thiourea")
        + 0.15 * g("flag_sulfonamide")
    )
    hydroph = 0.25 * (xlogp / 2.0) + 0.15 * g("flag_aromatic") + 0.25 * g("flag_fluorinated")

    penalty = 0.0008 * np.clip(mw - 450, 0, None) + 0.0020 * np.clip(tpsa - 120, 0, None)

    pfas_offset = np.zeros(len(df))
    for p in PFAS_LIST:
        col = f"pfas_{p}"
        if col in df.columns:
            if p in ("PFOS", "PFOA"):
                pfas_offset += 0.35 * df[col].to_numpy()       # long-chain stronger baseline
            elif p in ("PFBS", "PFBA", "HFPO-DA"):
                pfas_offset += 0.20 * df[col].to_numpy()       # short-chain mid
            elif p in ("PFPrA", "TFSI", "TFA"):
                pfas_offset += 0.10 * df[col].to_numpy()       # ultrashort lower

    electrostatics = 0.9 * np.clip(charge, 0, 3)

    score = electrostatics + cationic + hbondish + hydroph + pfas_offset - penalty
    score = score + rng.normal(0, 0.15, size=len(score))
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pfas_adsorption_candidates.csv")
    ap.add_argument("--out", dest="out", default="data/quantum_espress_placeholder.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dedup", action="store_true", help="Deduplicate by CID before pairing")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    for req in ["CID", "SMILES"]:
        if req not in df.columns:
            raise SystemExit(f"Missing required column '{req}' in {args.inp}")

    if args.dedup:
        df = df.drop_duplicates(subset=["CID"]).reset_index(drop=True)

    for c in BASE_NUMERIC:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    parts = []
    for p in PFAS_LIST:
        tmp = df.copy()
        tmp["pfas_id"] = p
        parts.append(tmp)
    pairs = pd.concat(parts, ignore_index=True)

    for p in PFAS_LIST:
        pairs[f"pfas_{p}"] = (pairs["pfas_id"] == p).astype(int)
        pairs["pfas_class"] = pairs["pfas_id"].map(PFAS_CLASS)

        pairs["pfas_is_long"] = (pairs["pfas_class"] == "Long").astype(int)
        pairs["pfas_is_short"] = (pairs["pfas_class"] == "Short").astype(int)
        pairs["pfas_is_ultrashort"] = (pairs["pfas_class"] == "Ultrashort").astype(int)

    pairs["y_placeholder"] = make_placeholder_score(pairs, seed=args.seed)

    pairs.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out} rows={len(pairs)} cols={len(pairs.columns)}")

if __name__ == "__main__":
    main()