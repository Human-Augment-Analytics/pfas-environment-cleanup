#!/usr/bin/env python3

import argparse
import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pyscf
from pyscf import gto, scf, dft, lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

PFAS_DATABASE = {
    "TFA": {
        "smiles": "FC(F)(F)C(=O)O",
        "description": "Trifluoroacetic acid"
    },
    "PFOA": {
        "smiles": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)O",
        "description": "Perfluorooctanoic acid"
    }
}

HARTREE_TO_KCAL = 627.509469
HARTREE_TO_KJ = 2625.499639


def generate_3d_mol(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES string: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed

    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status != 0:
        embed_status = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if embed_status != 0:
            raise RuntimeError(f"Could not generate 3D embedding for SMILES: {smiles}")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    elif AllChem.UFFHasAllMoleculeParams(mol):
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    return mol


def build_complex(mol_adsorbent: Chem.Mol, mol_pfas: Chem.Mol, offset_dist: float = 3.5) -> Chem.Mol:
    combo = Chem.CombineMols(mol_adsorbent, mol_pfas)
    n_ads = mol_adsorbent.GetNumAtoms()
    n_pfas = mol_pfas.GetNumAtoms()

    conf_ads = mol_adsorbent.GetConformer()
    conf_pfas = mol_pfas.GetConformer()

    coords_ads = np.array([list(conf_ads.GetAtomPosition(i)) for i in range(n_ads)])
    coords_pfas = np.array([list(conf_pfas.GetAtomPosition(i)) for i in range(n_pfas)])

    centroid_ads = coords_ads.mean(axis=0)
    centroid_pfas = coords_pfas.mean(axis=0)

    target_pos = centroid_ads + np.array([0.0, 0.0, offset_dist])
    translation = target_pos - centroid_pfas

    combo_conf = Chem.Conformer(combo.GetNumAtoms())

    for i in range(n_ads):
        combo_conf.SetAtomPosition(i, conf_ads.GetAtomPosition(i))
    for j in range(n_pfas):
        p_pos = np.array(list(conf_pfas.GetAtomPosition(j))) + translation
        combo_conf.SetAtomPosition(n_ads + j, p_pos)
    combo.RemoveAllConformers()
    combo.AddConformer(combo_conf)
    try:
        if AllChem.MMFFHasAllMoleculeParams(combo):
            ff = AllChem.MMFFGetMoleculeForceField(
                combo, AllChem.MMFFGetMoleculeProperties(combo)
            )
            ff.Initialize()
            ff.Minimize(500)
        elif AllChem.UFFHasAllMoleculeParams(combo):
            ff = AllChem.UFFGetMoleculeForceField(combo)
            ff.Initialize()
            ff.Minimize(500)
    except Exception as e:
        logging.warning(f"Complex force field relaxation warning: {e}")

    return combo


def rdkit_to_pyscf_atoms(mol: Chem.Mol) -> list:
    conf = mol.GetConformer()
    pyscf_atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        pos = conf.GetAtomPosition(i)
        pyscf_atoms.append((symbol, (pos.x, pos.y, pos.z)))
    return pyscf_atoms


def get_formal_charge_and_spin(mol: Chem.Mol) -> tuple[int, int]:
    charge = Chem.GetFormalCharge(mol)
    total_electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge
    spin = total_electrons % 2  
    return charge, spin


def run_pyscf_scf(
    mol_rdkit: Chem.Mol,
    basis: str = "3-21g",              
    method: str = "dft",
    xc: str = "pbe",                   
    grid_level: int = 1,             
    conv_thr: float = 1e-5,            
    use_density_fitting: bool = True,  
    nthreads: int = 4,
    verbose: int = 0
) -> float:
    lib.num_threads(nthreads)

    pyscf_atoms = rdkit_to_pyscf_atoms(mol_rdkit)
    charge, spin = get_formal_charge_and_spin(mol_rdkit)
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = pyscf_atoms
    pyscf_mol.basis = basis
    pyscf_mol.charge = charge
    pyscf_mol.spin = spin
    pyscf_mol.unit = "Angstrom"
    pyscf_mol.verbose = verbose
    pyscf_mol.build()

    if method.lower() == "hf":
        mf = scf.RHF(pyscf_mol) if spin == 0 else scf.UHF(pyscf_mol)
    elif method.lower() == "dft":
        mf = dft.RKS(pyscf_mol) if spin == 0 else dft.UKS(pyscf_mol)
        mf.xc = xc
        mf.grids.level = grid_level
    else:
        raise ValueError(f"Unsupported method: {method}")

    if use_density_fitting:
        mf = mf.density_fit()

    mf.conv_thr = conv_thr
    energy = mf.kernel()

    if not mf.converged:
        logging.warning("PySCF SCF calculation did not fully converge!")

    return float(energy)


def screen_adsorption(
    df_adsorbents: pd.DataFrame,
    smiles_col: str = "medoid_SMILES",
    id_col: str = "cluster",
    pfas_targets: list[str] = None,
    basis: str = "def2-svp",
    method: str = "dft",
    xc: str = "wb97x",
    nthreads: int = 4,
    output_csv: str = "pyscf_adsorption_results.csv",
    limit: int = None
):
    if pfas_targets is None or "all" in pfas_targets:
        pfas_targets = list(PFAS_DATABASE.keys())

    # Pre-compute energies for requested PFAS targets
    pfas_mols = {}
    pfas_energies = {}
    logging.info("=== Pre-calculating PFAS energies ===")
    for pfas_key in pfas_targets:
        pfas_info = PFAS_DATABASE[pfas_key]
        logging.info(f"Generating 3D structure and running PySCF for PFAS: {pfas_key} ({pfas_info['description']})")
        mol_pfas = generate_3d_mol(pfas_info["smiles"])
        e_pfas = run_pyscf_scf(mol_pfas, basis=basis, method=method, xc=xc, nthreads=nthreads)
        pfas_mols[pfas_key] = mol_pfas
        pfas_energies[pfas_key] = e_pfas
        logging.info(f"PFAS [{pfas_key}] E = {e_pfas:.6f} Hartree")

    results = []
    if os.path.exists(output_csv):
        logging.info(f"Existing results file found at '{output_csv}'. Resuming...")
        existing_df = pd.read_csv(output_csv)
        processed_pairs = set(zip(existing_df[id_col].astype(str), existing_df["pfas_name"]))
    else:
        existing_df = None
        processed_pairs = set()

    data_rows = df_adsorbents.to_dict("records")
    if limit is not None:
        data_rows = data_rows[:limit]

    logging.info(f"=== Beginning Adsorption Screening ({len(data_rows)} adsorbents x {len(pfas_targets)} PFAS) ===")

    for idx, row in enumerate(data_rows, 1):
        ads_id = str(row.get(id_col, f"index_{idx}"))
        smiles = row.get(smiles_col)

        if pd.isna(smiles) or not str(smiles).strip():
            logging.warning(f"Row {idx} (ID: {ads_id}): Empty SMILES string. Skipping.")
            continue

        logging.info(f"Processing Adsorbent [{idx}/{len(data_rows)}] ID: {ads_id}")

        try:
            mol_ads = generate_3d_mol(smiles)
            e_ads = run_pyscf_scf(mol_ads, basis=basis, method=method, xc=xc, nthreads=nthreads)
        except Exception as err:
            logging.error(f"Failed PySCF calculation for Adsorbent ID {ads_id}: {err}")
            continue

        for pfas_key in pfas_targets:
            if (ads_id, pfas_key) in processed_pairs:
                logging.info(f"Skipping already calculated pair: ({ads_id}, {pfas_key})")
                continue

            try:
                mol_pfas = pfas_mols[pfas_key]
                e_pfas = pfas_energies[pfas_key]

                complex_mol = build_complex(mol_ads, mol_pfas)
                e_complex = run_pyscf_scf(complex_mol, basis=basis, method=method, xc=xc, nthreads=nthreads)
                delta_e_hartree = e_complex - (e_ads + e_pfas)
                delta_e_kcal = delta_e_hartree * HARTREE_TO_KCAL
                delta_e_kj = delta_e_hartree * HARTREE_TO_KJ

                res_entry = {
                    id_col: ads_id,
                    "adsorbent_smiles": smiles,
                    "pfas_name": pfas_key,
                    "e_adsorbent_hartree": e_ads,
                    "e_pfas_hartree": e_pfas,
                    "e_complex_hartree": e_complex,
                    "e_adsorption_hartree": delta_e_hartree,
                    "e_adsorption_kcal_mol": delta_e_kcal,
                    "e_adsorption_kj_mol": delta_e_kj,
                    "basis_set": basis,
                    "method": method,
                    "xc_functional": xc if method == "dft" else "N/A"
                }

                results.append(res_entry)
                logging.info(f" -> Adsorbent {ads_id} + {pfas_key}: E_ads = {delta_e_kcal:.2f} kcal/mol ({delta_e_kj:.2f} kJ/mol)")

                res_df = pd.DataFrame(results)
                if existing_df is not None:
                    res_df = pd.concat([existing_df, res_df], ignore_index=True).drop_duplicates(
                        subset=[id_col, "pfas_name"]
                    )
                res_df.to_csv(output_csv, index=False)

            except Exception as err:
                logging.error(f"Failed calculation for complex ({ads_id} + {pfas_key}): {err}")
                continue

    logging.info(f"=== Screening Complete. Results saved to '{output_csv}' ===")


def main():
    parser = argparse.ArgumentParser(description="PySCF PFAS Adsorption Energy Screening")
    parser.add_argument("--csv", type=str, default="cluster_centers(1).csv", help="Input CSV file path")
    parser.add_argument("--smiles-col", type=str, default="medoid_SMILES", help="SMILES column name in CSV")
    parser.add_argument("--id-col", type=str, default="cluster", help="ID/Cluster column name in CSV")
    parser.add_argument("--pfas", type=str, nargs="+", default=["TFA", "PFOA"], choices=["TFA", "PFOA", "all"], help="PFAS target(s)")
    parser.add_argument("--basis", type=str, default="def2-svp", help="PySCF GTO basis set (e.g. def2-svp, 6-31g*, def2-tzvp)")
    parser.add_argument("--method", type=str, default="dft", choices=["dft", "hf"], help="Quantum chemistry method")
    parser.add_argument("--xc", type=str, default="wb97x", help="DFT exchange-correlation functional (e.g. wb97x, b3lyp, pbe)")
    parser.add_argument("--nthreads", type=int, default=4, help="Number of CPU threads for PySCF")
    parser.add_argument("--output", type=str, default="pyscf_adsorption_results.csv", help="Output CSV results file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of adsorbents to process (for testing)")

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Input CSV file '{args.csv}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    screen_adsorption(
        df_adsorbents=df,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        pfas_targets=args.pfas,
        basis=args.basis,
        method=args.method,
        xc=args.xc,
        nthreads=args.nthreads,
        output_csv=args.output,
        limit=args.limit
    )


if __name__ == "__main__":
    main()