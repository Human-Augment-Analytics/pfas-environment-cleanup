#!/usr/bin/env python3
"""Shared helpers for the PubChem seed-set tooling.

``shivani_ml_models/cluster_centers.csv`` (Shivani's KMeans seed set, k=1000
over 79,124 PubChem molecules) stores cluster descriptors as *centroid
averages*: ``save_cluster_centers()`` inverse-transforms the scaled KMeans
centers back to descriptor space, so columns ``MolecularWeight`` through
``RotatableBondCount`` describe the cluster, not any real molecule.

The practical trap is the ``Charge`` column. The inverse transform
reconstructs the average as a tiny nonzero remainder (values like ``1e-16``)
instead of an exact ``0``, so an exact ``float(row["Charge"]) != 0``
test misclassifies ~800 of the 1000 medoids as charged when their true formal
charge is 0. The truth for the medoid -- the actual molecule we simulate --
is encoded in the bracket atoms of ``medoid_SMILES`` (e.g. ``[NH4+]`` -> +1,
``[O-]`` -> -1). These helpers parse that.

Design constraints:

- stdlib only (``re``): the validator and campaign generator must run
  anywhere, including the PACE-ICE login node;
- read-only with respect to Shivani's CSV: nothing here writes, renames or
  "fixes" her file. Any enrichment lives in our derived outputs;
- forward-compatible: :func:`medoid_charge` prefers an explicit
  ``medoid_Charge`` column if a future regeneration of the CSV adds one.

See ``docs/seed_set_data_dictionary.md`` for the full column reference.
"""
from __future__ import annotations

import re

#: Exact expected header of the seed CSV.
SEED_COLUMNS = [
    "cluster", "n_points",
    "MolecularWeight", "ExactMass", "Charge", "XLogP", "TPSA",
    "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
    "medoid_CID", "medoid_SMILES",
]

#: Columns that are inverse-transformed centroid averages, not medoid values.
CENTROID_AVERAGED_COLUMNS = SEED_COLUMNS[2:10]

#: Elements covered by the repo's DFT pseudopotential workflow.
#: Informational in the validator -- unknown elements are reported, not fatal.
SUPPORTED_ELEMENTS = {"C", "N", "O", "S", "Cl", "F", "Br", "I", "P"}

_ORGANIC = set("BCNOPSFIbcnopsf")  # organic-subset symbols, both cases
_BRACKET_RE = re.compile(r"\[[^]]*\]")
_BRACKET_SYMBOL_RE = re.compile(r"([A-Z][a-z]?|[bcnopsfi])")


def strip_brackets(smiles: str) -> tuple[str, int]:
    """Return ``(smiles without bracket atoms, number of bracket atoms)``."""
    brackets = _BRACKET_RE.findall(smiles)
    return _BRACKET_RE.sub("", smiles), len(brackets)


def formal_charge(smiles: str) -> int:
    """Total formal charge from bracket-atom charge labels.

    Handles both notations: magnitude suffix (``[Fe+3]``, ``[O-1]``) and
    repeated signs (``[Fe+++]``, ``[Fe--]``). Atoms outside brackets belong
    to the organic subset and are treated as neutral.
    """
    total = 0
    for tok in _BRACKET_RE.findall(smiles):
        for signs, digits in re.findall(r"([+-]+)(\d*)", tok):
            mag = int(digits) if digits else len(signs)
            total += mag if signs[-1] == "+" else -mag
    return total


def heavy_atoms(smiles: str) -> int:
    """Count heavy (non-hydrogen) atoms in a SMILES string.

    Organic-subset atoms outside brackets plus every bracket atom (a bracket
    atom is heavy by definition; hydrogens appear only inside brackets, as in
    ``[nH]`` or ``[NH4+]``, and are not counted).
    """
    stripped, n_bracket = strip_brackets(smiles)
    i = n = 0
    while i < len(stripped):
        if stripped[i:i + 2] in ("Cl", "Br"):
            n += 1
            i += 2
            continue
        if stripped[i] in _ORGANIC:
            n += 1
        i += 1
    return n + n_bracket


def elements_of(smiles: str) -> set[str]:
    """Set of element symbols appearing in a SMILES string."""
    elems: set[str] = set()
    for tok in _BRACKET_RE.findall(smiles):
        m = _BRACKET_SYMBOL_RE.search(tok)
        if m:
            elems.add(m.group(1).capitalize())
    stripped, _ = strip_brackets(smiles)
    for i, ch in enumerate(stripped):
        if ch in "BCNOPSFIbcnopsf":
            two = stripped[i:i + 2]
            if two in ("Cl", "Br"):
                elems.add(two)
            else:
                elems.add(ch.upper())
    return elems


def smiles_issues(smiles: str) -> list[str]:
    """Cheap structural sanity flags for a SMILES string (best effort).

    Catches the failure modes that would break downstream obabel/pymatgen
    processing: multi-fragment strings, unbalanced parentheses/brackets and
    odd ring-bond label counts.  Ring labels are counted on the
    bracket-stripped string: digits inside bracket atoms are hydrogen
    counts, isotopes or charge magnitudes (``[NH2+]``, ``[13C]``, ``[Fe+3]``),
    not ring closures.  Two-digit ``%NN`` labels are counted as one label.
    """
    issues: list[str] = []
    if not smiles or not smiles.strip():
        issues.append("empty SMILES")
        return issues
    s = smiles.strip()
    if "." in s:
        issues.append("dot-disconnected (multi-fragment)")
    if s.count("(") != s.count(")"):
        issues.append("unbalanced parentheses")
    if s.count("[") != s.count("]"):
        issues.append("unbalanced brackets")
    stripped, _ = strip_brackets(s)
    label_counts: dict[str, int] = {}
    i = 0
    while i < len(stripped):
        if stripped[i] == "%":
            label = stripped[i:i + 3]
            i += 3
        elif stripped[i].isdigit():
            label = stripped[i]
            i += 1
        else:
            i += 1
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
    if any(v % 2 for v in label_counts.values()):
        issues.append("unbalanced ring-bond digits")
    return issues


def medoid_charge(row: dict, source: str = "auto") -> tuple[int, str]:
    """Medoid formal charge as ``(charge, provenance)``.

    ``source="auto"`` (default) uses a ``medoid_Charge`` column if a future
    regeneration of the seed CSV adds one, falling back to parsing bracket
    atoms in ``medoid_SMILES``. ``source="column"`` requires the column and
    raises ``KeyError`` otherwise. ``source="smiles"`` always parses.
    """
    if source in ("auto", "column") and row.get("medoid_Charge") not in (None, ""):
        return int(round(float(row["medoid_Charge"]))), "column:medoid_Charge"
    if source == "column":
        raise KeyError("medoid_Charge column requested but absent")
    return formal_charge(row["medoid_SMILES"]), "smiles:bracket-atoms"


def classify_centroid_charge(centroid_q: float, medoid_q: int) -> str:
    """Classify one row's centroid ``Charge`` against the medoid's charge.

    - ``"agree"``: centroid charge is exactly the medoid's (typically 0.0);
    - ``"noise_trap"``: centroid is nonzero float noise (e.g. 1e-16) that an
      exact ``!= 0`` test would misread, but rounds to the medoid's charge;
    - ``"mixed"``: centroid genuinely differs (the cluster averages mixed
      charges); trust the medoid value.
    """
    if round(centroid_q, 1) == medoid_q:
        return "noise_trap" if centroid_q != 0 else "agree"
    return "mixed"


def _self_test() -> None:
    """Offline smoke checks; run ``python scripts/seedset_utils.py``."""
    assert formal_charge("CC(=O)[O-]") == -1
    assert formal_charge("C[NH3+]") == 1
    assert formal_charge("[NH4+]") == 1
    assert formal_charge("[Fe+++]") == 3
    assert formal_charge("[Fe+3]") == 3
    assert formal_charge("[O-]C[Cu+2]") == 1
    assert formal_charge("CC(=O)O") == 0
    assert heavy_atoms("CC(=O)O") == 4
    assert heavy_atoms("FC(F)(F)C(=O)O") == 7  # TFA = C2F3O2
    assert heavy_atoms("C[NH3+]") == 2
    assert elements_of("FC(F)(F)C(=O)[O-]") == {"F", "C", "O"}
    assert elements_of("[nH]1cccc1") == {"N", "C"}
    assert smiles_issues("CC(=O)O") == []
    assert smiles_issues("CC(=O)O.CC") == ["dot-disconnected (multi-fragment)"]
    # digits inside brackets are H-counts/charges, never ring labels (CID 448944)
    assert smiles_issues("COC1=CC(=C(C=C1)OCC[NH+]=C(N)N)C[NH2+]CCCC[NH+]=C(N)N") == []
    assert smiles_issues("C1CC") == ["unbalanced ring-bond digits"]
    assert medoid_charge({"medoid_SMILES": "CC(=O)[O-]"}) == (-1, "smiles:bracket-atoms")
    assert classify_centroid_charge(1e-16, 0) == "noise_trap"
    assert classify_centroid_charge(0.0, 0) == "agree"
    assert classify_centroid_charge(0.5, 0) == "mixed"
    assert strip_brackets("C[Na]") == ("C", 1)
    print("seedset_utils self-test: OK")


if __name__ == "__main__":
    _self_test()
