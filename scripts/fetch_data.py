#!/usr/bin/env python3
from __future__ import annotations

import time
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

import requests
import pandas as pd
from io import StringIO

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
UA = "pfas-adsorption-step1/1.0 (requests)"

OUT_CSV = "data/pfas_adsorption_candidates.csv"
PROPS_CSV = "data/pubchem_properties.csv"

PUBCHEM_PROPS = [
    "IsomericSMILES", # These might be the same with the CanonicalSMILES
    "CanonicalSMILES",
    "InChIKey",
    "MolecularFormula",
    "MolecularWeight",
    "ExactMass",
    "Charge",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
]

ALLOWED_ELEMENTS = set(["C", "H", "O", "N", "F", "S", "P", "Cl", "Br", "I"])
MW_MAX = 900.0
ALLOW_CHARGED = True    # set False if you want only neutral candidates

# PubChem request batching
CID_BATCH_SIZE = 200
LISTKEY_PAGE_SIZE = 10000
SLEEP_BETWEEN_CALLS = 0.35  # ~3 req/sec, limit set by pubchem


@dataclass(frozen=True)
class QuerySpec:
    bucket: str
    qtype: str        # "keyword" or "substructure"
    term: str         # keyword string or SMILES

QUERIES: List[QuerySpec] = [
    # Bucket 1: Anion-exchange / cationic binders
    QuerySpec("anion_exchange", "keyword", "quaternary ammonium"),
    QuerySpec("anion_exchange", "keyword", "imidazolium"),
    QuerySpec("anion_exchange", "keyword", "pyridinium"),
    # QuerySpec("anion_exchange", "substructure", "[N+](C)(C)(C)C"),

    # Bucket 2: H-bond / polar binders
    QuerySpec("hbond_polar", "keyword", "urea"),
    QuerySpec("hbond_polar", "keyword", "thiourea"),
    QuerySpec("hbond_polar", "keyword", "sulfonamide"),
    QuerySpec("hbond_polar", "keyword", "amidine"),
    QuerySpec("hbond_polar", "keyword", "guanidine"),
    # QuerySpec("hbond_polar", "substructure", "NC(=O)N"),
    # QuerySpec("hbond_polar", "substructure", "NC(=S)N"),
    # QuerySpec("hbond_polar", "substructure", "S(=O)(=O)N"),

    # Bucket 3: Hydrophobic / fluorophilic binders
    # QuerySpec("hydrophobic_fluorophilic", "keyword", "fluorinated aromatic"),
    # QuerySpec("hydrophobic_fluorophilic", "keyword", "perfluoroaryl"),
    # QuerySpec("hydrophobic_fluorophilic", "substructure", "c1ccccc1"),
    # QuerySpec("hydrophobic_fluorophilic", "substructure", "C(F)(F)"),

    # Bucket 4: Host–guest
    QuerySpec("host_guest", "keyword", "cyclodextrin"),
    QuerySpec("host_guest", "keyword", "calixarene"),
    QuerySpec("host_guest", "keyword", "cucurbituril"),

    # Bucket 5: Graftable monomers / surface-functional monomers
    QuerySpec("graftable_monomers", "keyword", "methacrylate"),
    QuerySpec("graftable_monomers", "keyword", "acrylate"),
    QuerySpec("graftable_monomers", "keyword", "vinyl"),
    QuerySpec("graftable_monomers", "keyword", "styrene"),
    QuerySpec("graftable_monomers", "keyword", "glycidyl"),
    # QuerySpec("graftable_monomers", "substructure", "C=CC(=O)O"),
    # QuerySpec("graftable_monomers", "substructure", "C1OC1"),
]


session = requests.Session()
session.headers.update({"User-Agent": UA})


def _get(url: str, params: Optional[dict] = None, timeout: int = 90) -> dict:
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _post(url: str, data: dict, timeout: int = 90) -> dict:
    r = session.post(url, data=data, timeout=timeout)
    r.raise_for_status()
    return r.json()


def request_with_backoff(method: str, url: str, *, max_tries: int = 8, timeout: int = 120, **kwargs) -> requests.Response:
    """Retry transient PubChem errors (403/429/5xx) with exponential backoff + jitter."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_tries):
        try:
            r = session.request(method, url, timeout=timeout, **kwargs)
            if r.status_code < 400:
                return r

            retry_after = r.headers.get("Retry-After")
            wait: Optional[float] = None
            if retry_after is not None:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = None

            if r.status_code in (403, 429, 500, 502, 503, 504):
                base = 2 ** attempt
                jitter = random.uniform(0.0, 1.0)
                sleep_s = wait if wait is not None else min(120.0, float(base) + jitter)
                print(f"[warn] HTTP {r.status_code} for {url}. Backing off {sleep_s:.1f}s (attempt {attempt+1}/{max_tries})")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
        except Exception as e:
            last_exc = e
            base = 2 ** attempt
            jitter = random.uniform(0.0, 1.0)
            sleep_s = min(120.0, float(base) + jitter)
            print(f"[warn] Request error {type(e).__name__}: {e}. Backing off {sleep_s:.1f}s (attempt {attempt+1}/{max_tries})")
            time.sleep(sleep_s)

    if last_exc:
        raise last_exc
    raise RuntimeError("request_with_backoff failed without exception")

def load_done_cids_from_csv(path: str) -> Set[int]:
    """Return set of CIDs already written to a CSV (for resume)."""
    if not os.path.exists(path):
        return set()
    try:
        done = pd.read_csv(path, usecols=["CID"])
        return set(done["CID"].dropna().astype(int).tolist())
    except Exception:
        return set()

def name_word_cids(term: str):
    """
    Partial synonym match using name_type=word.
    Much more stable than fastsearch for 'keyword' terms.
    """
    url = f"{BASE}/compound/name/{requests.utils.quote(term)}/cids/JSON"
    data = _get(url, params={"name_type": "word"})
    if "IdentifierList" in data and "CID" in data["IdentifierList"]:
        return data["IdentifierList"]["CID"], None
    # If none found, PubChem often returns empty IdentifierList or a Fault
    return [], None


def substructure_cids(smiles: str) -> Tuple[List[int], Optional[str]]:
    """Substructure CID search via POST. Returns (cids, listkey)."""
    url = f"{BASE}/compound/substructure/smiles/cids/JSON"
    data = _post(url, data={"smiles": smiles})
    if "IdentifierList" in data and "CID" in data["IdentifierList"]:
        return data["IdentifierList"]["CID"], None
    if "Waiting" in data and "ListKey" in data["Waiting"]:
        return [], data["Waiting"]["ListKey"]
    if "Fault" in data:
        raise RuntimeError(f"PubChem Fault for substructure smiles='{smiles}': {data['Fault']}")
    raise RuntimeError(f"Unexpected substructure response for smiles='{smiles}': {data.keys()}")


def page_listkey_cids(listkey: str, page_size: int = LISTKEY_PAGE_SIZE, max_total: Optional[int] = None) -> List[int]:
    """
    Pages CIDs from a ListKey.
    """
    cids: List[int] = []
    start = 1
    while True:
        url = f"{BASE}/compound/listkey/{listkey}/cids/JSON"
        params = {"list_return": "flat", "list_start": start, "list_count": page_size}
        data = _get(url, params=params)
        batch = data.get("IdentifierList", {}).get("CID", [])
        if not batch:
            break
        cids.extend(batch)
        start += len(batch)
        if max_total and len(cids) >= max_total:
            return cids[:max_total]
        time.sleep(SLEEP_BETWEEN_CALLS)
    return cids


# ---- Property fetching ----

def fetch_properties_for_cids(cids: List[int], props: List[str], out_csv: str = PROPS_CSV, batch_size: int = CID_BATCH_SIZE) -> pd.DataFrame:
    """Fetch PubChem properties in CSV form for given CIDs (batched, resumable, incremental save).

    This avoids losing progress and reduces chance of PubChem throttling/temporary blocks.
    Returns a DataFrame of ALL properties currently in out_csv (including newly fetched rows).
    """
    done = load_done_cids_from_csv(out_csv)
    todo = [int(cid) for cid in cids if int(cid) not in done]

    print(f"[info] properties cache: {out_csv} (done={len(done)}; remaining={len(todo)})")

    first_write = not os.path.exists(out_csv)

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        cid_str = ",".join(map(str, batch))
        url = f"{BASE}/compound/cid/{cid_str}/property/{','.join(props)}/CSV"

        r = request_with_backoff("GET", url, timeout=180)
        df = pd.read_csv(StringIO(r.text))

        # Append incrementally so we can resume later
        df.to_csv(out_csv, mode="a", index=False, header=first_write)
        first_write = False

        # Progress log every ~10 batches
        if ((i // batch_size) % 10) == 0:
            print(f"[info] fetched {min(i + batch_size, len(todo))}/{len(todo)} remaining CIDs")

        time.sleep(SLEEP_BETWEEN_CALLS)

    # Read full cache back
    if not os.path.exists(out_csv):
        return pd.DataFrame()
    return pd.read_csv(out_csv)


def parse_elements(formula: str) -> Set[str]:
    """
    Very small formula parser to get element symbols from a molecular formula string.
    Example: 'C8H7F3O2' -> {'C','H','F','O'}
    """
    elems = set()
    i = 0
    while i < len(formula):
        if formula[i].isupper():
            elem = formula[i]
            i += 1
            if i < len(formula) and formula[i].islower():
                elem += formula[i]
                i += 1
            elems.add(elem)
            # skip digits
            while i < len(formula) and formula[i].isdigit():
                i += 1
        else:
            i += 1
    return elems


def apply_sanity_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "MolecularWeight" in df.columns:
        df = df[df["MolecularWeight"].fillna(0) <= MW_MAX]

    if not ALLOW_CHARGED and "Charge" in df.columns:
        df = df[df["Charge"].fillna(0) == 0]

    if "MolecularFormula" in df.columns:
        mask = []
        for f in df["MolecularFormula"].fillna(""):
            elems = parse_elements(f) if f else set()
            # keep if formula is present and all elements allowed and includes Carbon
            ok = bool(elems) and ("C" in elems) and elems.issubset(ALLOWED_ELEMENTS)
            mask.append(ok)
        df = df[pd.Series(mask, index=df.index)]

    return df

def try_add_bucket_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds boolean columns based on SMILES substructure (SMARTS) if RDKit is installed.
    Works with either 'IsomericSMILES' or 'SMILES' column names.
    """
    try:
        from rdkit import Chem
    except Exception:
        print("[info] RDKit not installed; skipping SMARTS bucket flags.")
        return df

    # Use whichever SMILES column exists
    smiles_col = None
    for c in ("IsomericSMILES", "SMILES", "CanonicalSMILES"):
        if c in df.columns:
            smiles_col = c
            break

    if smiles_col is None:
        print("[warn] No SMILES column found (expected one of IsomericSMILES/SMILES/CanonicalSMILES).")
        return df

    smarts_map = {
        "flag_quat_ammonium": "[N+](C)(C)(C)C",
        "flag_imidazolium": "c1ncc[n+]1",
        "flag_pyridinium": "c1cc[n+]cc1",
        "flag_sulfonamide": "S(=O)(=O)N",
        "flag_sulfonate": "S(=O)(=O)[O-]",
        "flag_urea": "NC(=O)N",
        "flag_thiourea": "NC(=S)N",
        "flag_guanidine": "NC(=N)N",
        "flag_aromatic": "a1aaaaa1",
        "flag_fluorinated": "C(F)(F)",
        "flag_acrylate_like": "C=CC(=O)O",
        "flag_epoxide": "C1OC1",
    }

    patt = {k: Chem.MolFromSmarts(v) for k, v in smarts_map.items()}

    def has(mol, ptn):
        return bool(mol and ptn and mol.HasSubstructMatch(ptn))

    flags = {k: [] for k in smarts_map.keys()}

    for smi in df[smiles_col].fillna(""):
        mol = Chem.MolFromSmiles(smi) if smi else None
        for k, ptn in patt.items():
            flags[k].append(has(mol, ptn))

    for k, arr in flags.items():
        df[k] = arr

    print(f"[info] Added {len(flags)} RDKit flag columns using '{smiles_col}'.")
    return df


def build_candidate_table(queries: List[QuerySpec]) -> pd.DataFrame:
    rows = []
    seen_pairs: Set[Tuple[str, int]] = set()

    for qs in queries:
        print(f"[query] bucket={qs.bucket} type={qs.qtype} term={qs.term}")

        try:
            if qs.qtype == "keyword":
                cids, lk = name_word_cids(qs.term)
            elif qs.qtype == "substructure":
                cids, lk = substructure_cids(qs.term)
            else:
                raise ValueError(f"Unknown qtype: {qs.qtype}")
        except Exception as e:
            print(f"[warn] query failed: {qs} -> {e}")
            continue

        if lk:
            print(f"  [info] large result set, paging ListKey={lk}")
            cids = page_listkey_cids(lk, page_size=LISTKEY_PAGE_SIZE)

        print(f"  [info] got {len(cids)} CIDs")

        for cid in cids:
            key = (qs.bucket, int(cid))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            rows.append({"bucket": qs.bucket, "CID": int(cid), "source_query": f"{qs.qtype}:{qs.term}"})

        time.sleep(SLEEP_BETWEEN_CALLS)

    df = pd.DataFrame(rows)
    print(f"[info] total bucketed (bucket,CID) pairs: {len(df)}")
    return df


def main():
    # 1) Collect candidates (CID lists) by bucket
    candidates = build_candidate_table(QUERIES)
    if candidates.empty:
        raise SystemExit("No candidates retrieved. Check queries / network.")

    # 2) Fetch properties for unique CIDs
    unique_cids = sorted(candidates["CID"].unique().tolist())
    print(f"[info] unique CIDs: {len(unique_cids)}")

    props_df = fetch_properties_for_cids(unique_cids, PUBCHEM_PROPS, out_csv=PROPS_CSV, batch_size=CID_BATCH_SIZE)
    if props_df.empty:
        raise SystemExit("Property fetch returned empty.")

    # 3) Merge bucket info back (many buckets per CID possible)
    merged = candidates.merge(props_df, on="CID", how="left")

    # 4) Deduplicate & filter
    merged = merged.dropna(subset=["InChIKey", "SMILES"], how="any")
    merged = apply_sanity_filters(merged)

    # 5) RDKit structure flags
    merged = try_add_bucket_flags(merged)

    # 6) remove exact duplicates of same CID+bucket
    merged = merged.drop_duplicates(subset=["CID", "bucket"])

    bucket_list = merged.groupby("CID")["bucket"].apply(lambda x: "|".join(sorted(set(x)))).reset_index()
    merged = merged.merge(bucket_list.rename(columns={"bucket": "bucket_membership"}), on="CID", how="left")

    # 7) Save
    merged.to_csv(OUT_CSV, index=False)
    print(f"[done] wrote {len(merged)} rows to {OUT_CSV}")

    print("\n[bucket counts]")
    print(merged["bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()