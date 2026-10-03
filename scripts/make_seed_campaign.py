#!/usr/bin/env python3
"""Generate the seed-campaign adsorbent manifest + lane audit from the seed CSV.

Campaign lane (approved 2026-10-01, "most stable, least likely to cause
errors, path of most guaranteed accuracy" -- staged neutral-first):

  v1 lane  = medoids with net formal charge 0 (parsed from medoid_SMILES
             bracket atoms), <= 80 heavy atoms, and centroid MW <= 1000
             (loose secondary cap -- the MW column is a centroid average).
  deferred = everything else, recorded with reasons in the audit table.
             Never silently dropped.

Why charge comes from the SMILES and not the ``Charge`` column: that column
is the cluster-centroid average, reconstructed as a tiny nonzero remainder
(1e-16-class values) instead of an exact ``0``, so an exact ``!= 0`` test
wrongly defers ~800 rows.  See
``scripts/seedset_utils.py`` and ``docs/seed_set_data_dictionary.md``.

Outputs (into --out-dir, default: alongside this script):
  seed_campaign_v1.csv   - ID,Name,SMILES,Category rows for
                           run_batch_screening.sh (CSV molecule = adsorbent,
                           PFAS probe = TFA).  Name = seed_c<cluster>_CID<cid>
                           so each case directory is self-documenting.
  seedset_lane_audit.csv - every seed row with lane assignment + defer reason
                           + medoid truth columns (formal charge, heavy atoms).

Submit with the driver's CSV_FILE override (see run_batch_screening.sh):
  sbatch --export=ALL,CSV_FILE=scripts/seed_campaign_v1.csv \
         --array=2-895 scripts/run_batch_screening.sh

Usage:
  python scripts/make_seed_campaign.py [--seed-csv PATH] [--out-dir PATH]
                                       [--heavy-cap 80] [--mw-cap 1000]

Stdlib-only; deterministic (output order = input order).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seedset_utils import heavy_atoms, medoid_charge  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SEED = REPO / "shivani_ml_models" / "cluster_centers.csv"

MANIFEST_NAME = "seed_campaign_v1.csv"
AUDIT_NAME = "seedset_lane_audit.csv"
MANIFEST_FIELDS = ["ID", "Name", "SMILES", "Category"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build seed-campaign manifest + lane audit."
    )
    ap.add_argument(
        "--seed-csv",
        type=Path,
        default=DEFAULT_SEED,
        help="path to cluster_centers.csv (the seed set)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="where to write the manifest + audit CSVs",
    )
    ap.add_argument(
        "--heavy-cap",
        type=int,
        default=80,
        help="max heavy atoms for the v1 lane (default 80)",
    )
    ap.add_argument(
        "--mw-cap",
        type=float,
        default=1000.0,
        help="max centroid-average MolecularWeight (default 1000)",
    )
    args = ap.parse_args(argv)

    if not args.seed_csv.exists():
        print(f"error: seed CSV not found: {args.seed_csv}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.seed_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("error: seed CSV has no data rows", file=sys.stderr)
        return 1

    manifest, audit = [], []
    for r in rows:
        cluster = int(r["cluster"])
        cid = str(r["medoid_CID"]).strip()
        smi = (r["medoid_SMILES"] or "").strip()
        mw = float(r["MolecularWeight"])
        centroid_q = float(r["Charge"])
        fcharge, _src = medoid_charge(r)
        hv = heavy_atoms(smi)

        reasons = []
        if fcharge != 0:
            reasons.append(f"charge={fcharge:+d}")
        if hv > args.heavy_cap:
            reasons.append(f"heavy={hv}")
        if mw > args.mw_cap:
            reasons.append(f"mw>{args.mw_cap:.0f}")

        lane = (
            "v1_included"
            if not reasons
            else "deferred_"
            + "&".join(
                "charged"
                if c.startswith("charge")
                else ("heavy" if c.startswith("heavy") else "bigmw")
                for c in reasons
            )
        )
        audit.append(
            {
                "cluster": cluster,
                "n_points": r["n_points"],
                "medoid_CID": cid,
                "centroid_MolecularWeight": mw,
                "centroid_Charge": centroid_q,
                "medoid_formal_charge": fcharge,
                "heavy_atoms": hv,
                "lane": lane,
                "defer_reason": ";".join(reasons),
                "medoid_SMILES": smi,
            }
        )
        if not reasons:
            manifest.append(
                {
                    "ID": f"s{len(manifest) + 1:04d}",
                    "Name": f"seed_c{cluster}_CID{cid}",
                    "SMILES": smi,
                    "Category": "seed_neutral",
                }
            )

    with open(args.out_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)
    with open(args.out_dir / AUDIT_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0].keys()))
        w.writeheader()
        w.writerows(audit)

    lanes = Counter(a["lane"] for a in audit)
    fch = Counter(a["medoid_formal_charge"] for a in audit)
    n = len(manifest)
    print(f"seed rows         : {len(rows)}  ({args.seed_csv})")
    print(f"v1 manifest rows  : {n}  ({args.out_dir / MANIFEST_NAME})")
    print(f"deferred rows     : {len(rows) - n}  ({args.out_dir / AUDIT_NAME})")
    for lane, cnt in sorted(lanes.items(), key=lambda t: -t[1]):
        print(f"  {lane:28s} {cnt}")
    print(
        f"medoid formal-charge histogram (SMILES-derived): {dict(sorted(fch.items()))}"
    )
    print(
        f"job math: ({n} x 2) + 20x3 + 1 = {n * 2 + 60 + 1} pw.x runs "
        f"(seed pairs + CTAB-relative trio + shared TFA reference)"
    )
    print(
        f"array range: --array=2-{n + 1}  (task id = 1-based CSV line; line 1 = header)"
    )
    print("staging: anchors first, then a pilot slice, then the remainder")
    return 0


if __name__ == "__main__":
    _reconfigure = getattr(sys.stdout, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
