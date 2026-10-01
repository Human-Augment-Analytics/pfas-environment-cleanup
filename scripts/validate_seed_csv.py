#!/usr/bin/env python3
"""Validate the PubChem seed-set CSV (contract guard for the campaign).

Checks, in order:
  1. schema   - exact 12-column header, in push order (schema drift = FAIL).
  2. integrity- row count, contiguous cluster ids, unique medoid CIDs/SMILES,
                n_points >= 1, balanced single-fragment SMILES.
  3. elements - atom set within the repo pseudopotential coverage.
  4. charge   - the centroid-vs-medoid semantics check (see below).
  5. size     - heavy atoms / centroid MW vs campaign caps (informational).

The charge check is the reason this script exists.  The CSV's ``Charge`` (and
the other descriptor) columns are CLUSTER-CENTROID AVERAGES, not medoid
properties.  The centroid ``Charge`` carries float noise (e.g. 1e-16), so an
exact ``Charge != 0`` filter wrongly defers ~800 of 1000 rows as "charged".
This validator recomputes each medoid's true net formal charge from its
SMILES bracket atoms and reports:

  * how many rows a naive exact filter would misclassify  (the trap's size),
  * how many clusters genuinely mix charges               (fractional averages),
  * and FAILS if centroid and medoid disagree in a way that indicates data
    corruption rather than averaging (non-finite values, unparseable charges).

Usage:
  python scripts/validate_seed_csv.py [path/to/cluster_centers.csv]

Exit code 0 = contract holds; 1 = it does not (suitable for CI / pre-submit
gates).  Stdlib-only.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seedset_utils import (  # noqa: E402
    SEED_COLUMNS,
    SUPPORTED_ELEMENTS,
    classify_centroid_charge,
    elements_of,
    heavy_atoms,
    medoid_charge,
    smiles_issues,
)

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "shivani_ml_models" / "cluster_centers.csv"

HEAVY_CAP = 80     # campaign tractability cap (see make_seed_campaign.py)
MW_CAP = 1000.0    # loose secondary cap on the CENTROID average MW


def fail(msg: str) -> None:
    print(f"FAIL  {msg}", flush=True)


def info(msg: str) -> None:
    print(f"info  {msg}", flush=True)


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_CSV
    if not path.exists():
        fail(f"seed CSV not found: {path}")
        return 1

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    errors = 0

    # 1. schema -------------------------------------------------------------
    if header != SEED_COLUMNS:
        fail(f"header drift: expected {SEED_COLUMNS}, got {header}")
        return 1  # everything below assumes the known schema

    # 2. integrity ----------------------------------------------------------
    n = len(rows)
    info(f"rows: {n}  (columns: {len(header)})")
    if n == 0:
        fail("no data rows")
        return 1

    clusters: list[int] = []
    for i, r in enumerate(rows, start=2):
        try:
            clusters.append(int(r["cluster"]))
        except (TypeError, ValueError):
            fail(f"line {i}: cluster id not an integer: {r['cluster']!r}")
            errors += 1
        try:
            npoints = int(r["n_points"])
            if npoints < 1:
                fail(f"line {i}: n_points < 1 ({npoints})")
                errors += 1
        except (TypeError, ValueError):
            fail(f"line {i}: n_points not an integer: {r['n_points']!r}")
            errors += 1
        if not str(r["medoid_CID"]).strip().isdigit():
            fail(f"line {i}: medoid_CID not a positive integer: {r['medoid_CID']!r}")
            errors += 1
        for issue in smiles_issues(r["medoid_SMILES"] or ""):
            fail(f"line {i} (cluster {r['cluster']}): SMILES issue: {issue}")
            errors += 1

    if clusters and clusters != list(range(clusters[0], clusters[0] + n)):
        gaps = sorted(set(range(clusters[0], clusters[-1] + 1)) - set(clusters))
        fail(f"cluster ids not contiguous; missing: {gaps[:10]}{'...' if len(gaps) > 10 else ''}")
        errors += 1

    cids = [str(r["medoid_CID"]).strip() for r in rows]
    if len(set(cids)) != len(cids):
        dup = [c for c, k in Counter(cids).items() if k > 1]
        fail(f"duplicate medoid_CIDs: {dup[:10]}")
        errors += 1
    smis = [(r["medoid_SMILES"] or "").strip() for r in rows]
    if len(set(smis)) != len(smis):
        dup = [s for s, k in Counter(smis).items() if k > 1]
        fail(f"duplicate medoid_SMILES: {dup[:3]}")
        errors += 1
    info(f"total clustered molecules (sum n_points): {sum(int(r['n_points']) for r in rows):,}")

    # 3. elements -----------------------------------------------------------
    unknown: Counter = Counter()
    for r in rows:
        for e in elements_of(r["medoid_SMILES"] or ""):
            if e not in SUPPORTED_ELEMENTS:
                unknown[e] += 1
    if unknown:
        fail(f"elements outside pseudopotential coverage: {dict(unknown)}")
        errors += 1
    else:
        info("elements: all within repo pseudopotential coverage "
             f"({', '.join(sorted(SUPPORTED_ELEMENTS))})")

    # 4. charge semantics (the reason this validator exists) ----------------
    trap = mixed = agree = 0
    charge_hist: Counter = Counter()
    bad_q: list[str] = []
    for r in rows:
        centroid_q = _safe_float(r["Charge"])
        if centroid_q is None:
            bad_q.append(f"cluster {r['cluster']}: unparseable/non-finite Charge {r['Charge']!r}")
            continue
        try:
            med_q, src = medoid_charge(r)
        except (KeyError, ValueError) as exc:
            bad_q.append(f"cluster {r['cluster']}: {exc}")
            continue
        charge_hist[med_q] += 1
        kind = classify_centroid_charge(centroid_q, med_q)
        if kind == "noise_trap":
            trap += 1
            agree += 1
        elif kind == "agree":
            agree += 1
        else:
            mixed += 1
    for b in bad_q:
        fail(b)
        errors += 1

    if rows:
        info(f"medoid formal charge (true, via {medoid_charge(rows[0])[1]}): "
             f"{dict(sorted(charge_hist.items()))}")
        info(f"centroid-vs-medoid: {agree} agree | {mixed} mixed-cluster averages | "
             f"{trap} rows whose centroid-average Charge is a tiny nonzero "
             f"remainder an exact 'Charge != 0' filter would misclassify")
    if trap:
        info("NOTE: never filter this CSV on exact 'Charge != 0' -- use "
             "seedset_utils.medoid_charge() / make_seed_campaign.py instead "
             "(docs/seed_set_data_dictionary.md)")
    if sum(charge_hist.values()) != n:
        fail("charge check did not cover every row")
        errors += 1

    # 5. size caps (informational - lane policy lives in make_seed_campaign) -
    over_heavy = sum(1 for r in rows if heavy_atoms(r["medoid_SMILES"] or "") > HEAVY_CAP)
    over_mw = sum(1 for r in rows
                  if (_safe_float(r["MolecularWeight"]) or 0.0) > MW_CAP)
    info(f"tractability: {over_heavy} medoids > {HEAVY_CAP} heavy atoms; "
         f"{over_mw} centroid MW > {MW_CAP:.0f} (caps are campaign lane policy)")

    print(("PASS " if errors == 0 else "FAILED") +
          f": {errors} hard error(s), {n} rows checked", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main(sys.argv))
