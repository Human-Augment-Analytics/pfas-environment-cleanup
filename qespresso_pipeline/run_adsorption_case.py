#!/usr/bin/env python3

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Molecule, Lattice, Structure
from pymatgen.io.cif import CifWriter

from smiles_to_qe import run_obabel, mol_to_cif_pymatgen, run_cif2cell, modify_qe_input

RY_TO_EV = 13.605693009


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_pwscf(input_path: Path, pw_command="pw.x") -> Path:
    output_path = input_path.with_suffix(".out")
    print(f"[run] {input_path.name} -> {output_path.name}")

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        result = subprocess.run([pw_command], stdin=f_in, stdout=f_out, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(
            f"pw.x failed for {input_path}\n{result.stderr.decode(errors='ignore')}"
        )

    return output_path


def extract_total_energy_ry(path: Path) -> float:
    energy = None
    pat = re.compile(r'!\s+total energy\s+=\s+([-0-9.]+)\s+Ry')
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                energy = float(m.group(1))

    if energy is None:
        raise ValueError(f"No total energy found in {path}")
    return energy


def build_complex_cif(adsorbent_mol: Path, pfas_mol: Path, output_cif: Path, shift=(4.0, 0.0, 0.0), padding=12.0):
    ads = Molecule.from_file(str(adsorbent_mol))
    pfas = Molecule.from_file(str(pfas_mol))

    ads_coords = np.array(ads.cart_coords)
    pfas_coords = np.array(pfas.cart_coords)

    ads_center = ads_coords.mean(axis=0)
    pfas_center = pfas_coords.mean(axis=0)

    shifted_pfas = pfas_coords - pfas_center + ads_center + np.array(shift)

    all_species = list(ads.species) + list(pfas.species)
    all_coords = np.vstack([ads_coords, shifted_pfas])

    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    lengths = maxs - mins
    cell_size = float(max(lengths) + padding)

    lattice = Lattice.cubic(cell_size)
    center = (mins + maxs) / 2
    centered = all_coords - center + cell_size / 2

    structure = Structure(
        lattice,
        all_species,
        centered,
        coords_are_cartesian=True,
    )

    CifWriter(structure, symprec=None).write_file(str(output_cif))
    print(f"[info] Complex CIF created: {output_cif}")


def prepare_single_case(smiles: str, outbase: Path, padding: float):
    mol_path = run_obabel(smiles, str(outbase))
    cif_path = mol_to_cif_pymatgen(mol_path, str(outbase), padding=padding)
    in_path = run_cif2cell(cif_path, str(outbase))
    modify_qe_input(in_path, job_type="molecule")
    return mol_path, cif_path, in_path


def main():
    parser = argparse.ArgumentParser(description="Run one adsorption-energy demo case")
    parser.add_argument("--adsorbent-smiles", required=True)
    parser.add_argument("--pfas-smiles", required=True)
    parser.add_argument("--case-name", default="adsorption_demo")
    parser.add_argument("--workdir", default="dft_cases")
    parser.add_argument("--pw-command", default="pw.x")
    parser.add_argument("--padding", type=float, default=12.0)
    parser.add_argument("--complex-shift-x", type=float, default=4.0)
    parser.add_argument("--complex-shift-y", type=float, default=0.0)
    parser.add_argument("--complex-shift-z", type=float, default=0.0)
    args = parser.parse_args()

    case_dir = Path(args.workdir) / args.case_name
    ads_dir = case_dir / "adsorbent"
    pfas_dir = case_dir / "pfas"
    complex_dir = case_dir / "complex"

    ensure_dir(ads_dir)
    ensure_dir(pfas_dir)
    ensure_dir(complex_dir)

    print("[step] Preparing adsorbent")
    ads_base = ads_dir / "adsorbent"
    ads_mol, ads_cif, ads_in = prepare_single_case(
        args.adsorbent_smiles, ads_base, padding=args.padding
    )

    print("[step] Preparing PFAS")
    pfas_base = pfas_dir / "pfas"
    pfas_mol, pfas_cif, pfas_in = prepare_single_case(
        args.pfas_smiles, pfas_base, padding=args.padding
    )

    print("[step] Building complex")
    complex_cif = complex_dir / "complex.cif"
    build_complex_cif(
        adsorbent_mol=ads_mol,
        pfas_mol=pfas_mol,
        output_cif=complex_cif,
        shift=(args.complex_shift_x, args.complex_shift_y, args.complex_shift_z),
        padding=args.padding,
    )

    complex_in = run_cif2cell(complex_cif, str(complex_dir / "complex"))
    modify_qe_input(complex_in, job_type="molecule")

    print("[step] Running adsorbent calculation")
    ads_out = run_pwscf(ads_in, pw_command=args.pw_command)

    print("[step] Running PFAS calculation")
    pfas_out = run_pwscf(pfas_in, pw_command=args.pw_command)

    print("[step] Running complex calculation")
    complex_out = run_pwscf(complex_in, pw_command=args.pw_command)

    print("[step] Parsing energies")
    e_adsorbent_ry = extract_total_energy_ry(ads_out)
    e_pfas_ry = extract_total_energy_ry(pfas_out)
    e_complex_ry = extract_total_energy_ry(complex_out)

    e_adsorption_ry = e_complex_ry - e_adsorbent_ry - e_pfas_ry
    e_adsorption_ev = e_adsorption_ry * RY_TO_EV

    results = {
        "case_name": args.case_name,
        "adsorbent_smiles": args.adsorbent_smiles,
        "pfas_smiles": args.pfas_smiles,
        "energies_ry": {
            "adsorbent": e_adsorbent_ry,
            "pfas": e_pfas_ry,
            "complex": e_complex_ry,
            "adsorption": e_adsorption_ry,
        },
        "energies_ev": {
            "adsorption": e_adsorption_ev,
        },
        "complex_shift": {
            "x": args.complex_shift_x,
            "y": args.complex_shift_y,
            "z": args.complex_shift_z,
        },
    }

    results_path = case_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[done] Adsorption workflow complete")
    print(json.dumps(results, indent=2))
    print(f"[done] Saved results to: {results_path}")


if __name__ == "__main__":
    main()