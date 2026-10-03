# Seed Set Data Dictionary — `shivani_ml_models/cluster_centers.csv`

Provenance: Shivani's seed set (branch `shivani_branch`, commit `2c8ebb9`
"seed set"). Method (from her `clustering.py`): a ~79k-row PubChem candidate
table (8 descriptors, `fetch_data.py` lineage) → `StandardScaler` z-score →
`KMeans(n_clusters=1000, n_init=10, random_state=42)` → per cluster, the
**medoid** = the real molecule nearest the centroid in scaled space. 1000
clusters cover 79,124 molecules; the file is committed byte-identical to her
push and must stay so.

## The one thing to know before filtering this file

**Columns 3–10 are cluster-centroid AVERAGES, not properties of the medoid
molecule.** They come from `scaler.inverse_transform(kmeans.cluster_centers_)`.
Only `cluster`, `n_points`, `medoid_CID`, `medoid_SMILES` are exact,
per-molecule facts.

Consequence (the trap this dictionary exists for): because the `Charge`
average is reconstructed by the clustering math, it comes out as a tiny
nonzero remainder (e.g. `0.0000000000000001`) instead of an exact `0`, so an
exact `Charge != 0` test wrongly defers ~800 of the 1000 rows as "charged".
Clusters that genuinely mix charge states make it worse: their average is
fractional (e.g. `0.31`), matching neither member.

**Correct pattern — derive the medoid's truth from its SMILES:**

```python
from seedset_utils import medoid_charge, heavy_atoms  # scripts/

q = medoid_charge(row)  # (net formal charge, provenance) — SMILES bracket atoms
hv = heavy_atoms(row["medoid_SMILES"])
```

`medoid_charge()` prefers an explicit `medoid_Charge` column automatically if
a future regeneration of the CSV adds one, and falls back to SMILES parsing
otherwise. The numbers for the committed file (verified 2026-10-01):

| Quantity | Value |
|---|---|
| medoid formal charge (SMILES-derived) | 929 neutral, 71 charged (49 +1, 13 +2, 3 +3, 3 −1, 2 −2, 1 −4) |
| rows an exact `Charge != 0` filter misclassifies | ~800 |
| clusters with fractional (mixed) charge averages | small fraction — expected, not an error |

## Column reference

| # | Column | Semantics | Exact for the medoid? |
|---|--------|-----------|-----------------------|
| 1 | `cluster` | k-means cluster id, 0–999, contiguous | yes |
| 2 | `n_points` | molecules assigned to the cluster | yes (cluster fact) |
| 3 | `MolecularWeight` | **centroid average** over cluster members | no |
| 4 | `ExactMass` | **centroid average** | no |
| 5 | `Charge` | **centroid average** (tiny nonzero remainders + fractional mixes; never filter exactly) | no |
| 6 | `XLogP` | **centroid average** | no |
| 7 | `TPSA` | **centroid average** | no |
| 8 | `HBondDonorCount` | **centroid average** | no |
| 9 | `HBondAcceptorCount` | **centroid average** | no |
| 10 | `RotatableBondCount` | **centroid average** | no |
| 11 | `medoid_CID` | PubChem CID of the medoid molecule | yes |
| 12 | `medoid_SMILES` | canonical SMILES of the medoid molecule | yes |

Not carried: InChIKeys (they exist in the upstream PubChem table). Re-join on
`medoid_CID` when InChIKey-keyed caching (`data/dft_cache/`) is needed.

## Tooling around this file

| Tool | Purpose |
|---|---|
| `scripts/validate_seed_csv.py` | Contract guard: schema, integrity, element coverage, and the centroid-vs-medoid charge report (prints the trap's size every run). CI/pre-submit safe (`exit 1` on drift). |
| `scripts/make_seed_campaign.py` | Builds the batch-driver manifest (`seed_campaign_v1.csv`, neutral-first v1 lane) + `seedset_lane_audit.csv` (every medoid with lane + reason; nothing silently dropped). |
| `scripts/seedset_utils.py` | The shared derivation library: `formal_charge()`, `heavy_atoms()`, `medoid_charge()`, SMILES sanity checks. |

## Campaign lane policy (v1)

Gates (rationale: stability and guaranteed accuracy first):

* **Neutral only** — net formal charge 0. The DFT workflow never sets
  `tot_charge`; a charged medoid would silently compute the wrong cell.
  Zwitterions (net 0, e.g. `[O-]…[NH+]`) are correctly included.
* **≤ 80 heavy atoms** — the heavy tail (up to 210 atoms) risks wall-clock /
  memory failures mid-array.
* **centroid MW ≤ 1000** — loose secondary cap (the column is an average; the
  heavy-atom gate is the real bound).

v1 result: 894 of 1000 medoids included; 106 deferred with reasons
(70 charged, 30 heavy+bigmw, 5 bigmw, 1 charged+heavy+bigmw). The deferred
lane is a documented future decision, not a rejection of the seed data.
