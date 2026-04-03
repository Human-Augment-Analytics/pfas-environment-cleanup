#!/usr/bin/env python3

import os
import shlex
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from pymatgen.core import Molecule, Lattice, Structure
from pymatgen.io.cif import CifWriter

from smiles_to_qe import run_obabel, mol_to_cif_pymatgen, run_cif2cell, modify_qe_input

RY_TO_EV = 13.605693009


def get_mode_settings(mode: str):
    if mode == "lowmem":
        return {
            "ecutwfc": 25,
            "ecutrho": 200,
            "padding": 6.0,
            "use_gamma": True,
            "mixing_beta": 0.2,
            "omp_threads": 1,
        }
    elif mode == "cluster":
        return {
            "ecutwfc": 30,
            "ecutrho": 240,
            "padding": 8.0,
            "use_gamma": True,
            "mixing_beta": 0.15,
            "omp_threads": 1,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_pwscf(input_path: Path, pw_command="pw.x", omp_threads=1):
    output_path = input_path.with_suffix(".out")
    stderr_path = input_path.with_suffix(".err")

    print(f"[run] {input_path.name} -> {output_path.name}")
    workdir = input_path.parent
    (workdir / "Outputs").mkdir(exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = shlex.split(pw_command) + ["-in", input_path.name]

    print(f"[run] command: {' '.join(cmd)}")
    print(f"[run] OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")

    with open(output_path, "w") as f_out, open(stderr_path, "w") as f_err:
        result = subprocess.run(
            cmd,
            cwd=workdir,
            stdout=f_out,
            stderr=f_err,
            env=env,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"pw.x failed for {input_path}\n"
            f"Return code: {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"See {output_path} and {stderr_path}"
        )

    return output_path


def extract_total_energy_ry(path: Path) -> float:
    energy = None

    with open(path, "r", errors="ignore") as f:
        for line in f:
            lower = line.lower()
            if "total energy" in lower and "ry" in lower:
                parts = line.replace("=", " = ").split()
                for i, token in enumerate(parts):
                    if token == "=" and i + 1 < len(parts):
                        try:
                            energy = float(parts[i + 1])
                        except ValueError:
                            pass

    if energy is None:
        raise ValueError(f"No total energy found in {path}")
    return energy


def build_complex_cif(
    adsorbent_mol: Path,
    pfas_mol: Path,
    output_cif: Path,
    shift=(4.0, 0.0, 0.0),
    padding=12.0,
):
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

    box_lengths = lengths + padding
    lattice = Lattice.orthorhombic(*box_lengths)
    centered = all_coords - mins + padding / 2

    structure = Structure(
        lattice,
        all_species,
        centered,
        coords_are_cartesian=True,
    )

    CifWriter(structure, symprec=None).write_file(str(output_cif))
    print(f"[info] Complex CIF created: {output_cif}")


def prepare_single_case(smiles: str, outbase: Path, settings: dict):
    mol_path = run_obabel(smiles, str(outbase))
    cif_path = mol_to_cif_pymatgen(mol_path, str(outbase), padding=settings["padding"])
    in_path = run_cif2cell(cif_path, str(outbase))
    modify_qe_input(
        in_path,
        job_type="molecule",
        ecutwfc=settings["ecutwfc"],
        ecutrho=settings["ecutrho"],
        use_gamma=settings["use_gamma"],
        mixing_beta=settings["mixing_beta"],
    )
    return mol_path, cif_path, in_path


def link_pseudos(case_subdir: Path, pseudo_source: Path):
    pseudos_dir = case_subdir / "Pseudopotentials"
    outputs_dir = case_subdir / "Outputs"
    outputs_dir.mkdir(exist_ok=True)

    if pseudos_dir.exists() or pseudos_dir.is_symlink():
        return

    pseudos_dir.symlink_to(pseudo_source.resolve(), target_is_directory=True)


def require_existing(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} was skipped, but required file does not exist: {path}"
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="Run one adsorption-energy demo case")
    parser.add_argument("--adsorbent-smiles", required=False)
    parser.add_argument("--pfas-smiles", required=False)
    parser.add_argument("--case-name", default="adsorption_demo")
    parser.add_argument("--workdir", default="dft_cases")
    parser.add_argument("--pw-command", default="pw.x")
    parser.add_argument("--padding", type=float, default=12.0)
    parser.add_argument("--complex-shift-x", type=float, default=4.0)
    parser.add_argument("--complex-shift-y", type=float, default=0.0)
    parser.add_argument("--complex-shift-z", type=float, default=0.0)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--mode", choices=["lowmem", "cluster"], default="cluster")
    parser.add_argument("--prepare-only", action="store_true")

    parser.add_argument("--skip-ads", action="store_true",
                        help="Skip adsorbent preparation and pw.x run; reuse existing files")
    parser.add_argument("--skip-pfas", action="store_true",
                        help="Skip PFAS preparation and pw.x run; reuse existing files")
    parser.add_argument("--skip-complex", action="store_true",
                        help="Skip complex build/preparation and pw.x run; reuse existing files")
    parser.add_argument("--adsorbent-name", required=True)
    parser.add_argument("--pfas-name", required=True)
    parser.add_argument("--compound-root", default="compounds")
    parser.add_argument(
        "--pfas-energy-ry",
        type=float,
        default=None,
        help="Provide PFAS energy directly (Ry); skips PFAS energy parsing"
    )

    args = parser.parse_args()
    settings = get_mode_settings(args.mode)

    if not args.skip_ads and not args.adsorbent_smiles:
        raise ValueError("--adsorbent-smiles is required unless --skip-ads is used")

    if args.pfas_energy_ry is None and not args.skip_pfas and not args.pfas_smiles:
        raise ValueError("--pfas-smiles is required unless --skip-pfas or --pfas-energy-ry is used")

    case_dir = Path(args.workdir) / args.case_name
    compound_root = Path(args.compound_root)

    ads_dir = compound_root / "adsorbents" / args.adsorbent_name
    pfas_dir = compound_root / "pfas" / args.pfas_name
    complex_dir = case_dir / "complex"

    ensure_dir(ads_dir)
    ensure_dir(pfas_dir)
    ensure_dir(complex_dir)

    pseudo_source = Path(args.pseudo_dir).resolve()
    link_pseudos(ads_dir, pseudo_source)
    link_pseudos(pfas_dir, pseudo_source)
    link_pseudos(complex_dir, pseudo_source)

    ads_base = ads_dir / "adsorbent"
    pfas_base = pfas_dir / "pfas"
    complex_base = complex_dir / "complex"

    ads_mol_expected = ads_base.with_suffix(".mol")
    ads_cif_expected = ads_base.with_suffix(".cif")
    ads_in_expected = ads_base.with_suffix(".in")
    ads_out_expected = ads_base.with_suffix(".out")

    pfas_mol_expected = pfas_base.with_suffix(".mol")
    pfas_cif_expected = pfas_base.with_suffix(".cif")
    pfas_in_expected = pfas_base.with_suffix(".in")
    pfas_out_expected = pfas_base.with_suffix(".out")

    complex_cif_expected = complex_base.with_suffix(".cif")
    complex_in_expected = complex_base.with_suffix(".in")
    complex_out_expected = complex_base.with_suffix(".out")

    # Adsorbent
    if args.skip_ads:
        print("[skip] Skipping adsorbent preparation and run")
        ads_mol = require_existing(ads_mol_expected, "Adsorbent MOL")
        ads_cif = require_existing(ads_cif_expected, "Adsorbent CIF")
        ads_in = require_existing(ads_in_expected, "Adsorbent input")
        ads_out = require_existing(ads_out_expected, "Adsorbent output")
    else:
        print("[step] Preparing adsorbent")
        ads_mol, ads_cif, ads_in = prepare_single_case(
            args.adsorbent_smiles, ads_base, settings=settings
        )
        ads_out = None

    # PFAS
    e_pfas_ry = None
    if args.pfas_energy_ry is not None:
        print("[info] Using provided PFAS energy")
        e_pfas_ry = args.pfas_energy_ry
        if args.skip_complex:
            pfas_mol = None
            pfas_cif = None
            pfas_in = None
            pfas_out = None
        else:
            # Need structure for complex build
            print("[step] Preparing PFAS structure for complex build")
            if args.skip_pfas:
                pfas_mol = require_existing(pfas_mol_expected, "PFAS MOL")
                pfas_cif = require_existing(pfas_cif_expected, "PFAS CIF")
                pfas_in = require_existing(pfas_in_expected, "PFAS input")
                pfas_out = require_existing(pfas_out_expected, "PFAS output")
            else:
                if not args.pfas_smiles:
                    raise ValueError("PFAS structure is needed to build complex; provide --pfas-smiles or use --skip-complex")
                pfas_mol, pfas_cif, pfas_in = prepare_single_case(
                    args.pfas_smiles, pfas_base, settings=settings
                )
                pfas_out = None
    elif args.skip_pfas:
        print("[skip] Skipping PFAS preparation and run")
        pfas_mol = require_existing(pfas_mol_expected, "PFAS MOL")
        pfas_cif = require_existing(pfas_cif_expected, "PFAS CIF")
        pfas_in = require_existing(pfas_in_expected, "PFAS input")
        pfas_out = require_existing(pfas_out_expected, "PFAS output")
    else:
        print("[step] Preparing PFAS")
        pfas_mol, pfas_cif, pfas_in = prepare_single_case(
            args.pfas_smiles, pfas_base, settings=settings
        )
        pfas_out = None

    # Complex
    if args.skip_complex:
        print("[skip] Skipping complex build, preparation, and run")
        complex_in = require_existing(complex_in_expected, "Complex input")
        complex_out = require_existing(complex_out_expected, "Complex output")
    else:
        if ads_mol is None:
            raise ValueError("Cannot build complex without adsorbent structure")
        if pfas_mol is None:
            raise ValueError("Cannot build complex without PFAS structure")

        print("[step] Building complex")
        build_complex_cif(
            adsorbent_mol=ads_mol,
            pfas_mol=pfas_mol,
            output_cif=complex_cif_expected,
            shift=(args.complex_shift_x, args.complex_shift_y, args.complex_shift_z),
            padding=settings["padding"],
        )

        complex_in = run_cif2cell(complex_cif_expected, str(complex_base))
        modify_qe_input(
            complex_in,
            job_type="molecule",
            ecutwfc=settings["ecutwfc"],
            ecutrho=settings["ecutrho"],
            use_gamma=settings["use_gamma"],
            mixing_beta=settings["mixing_beta"],
        )
        complex_out = None

    if args.prepare_only:
        print("[done] Preparation complete; QE inputs generated but not run.")
        return

    if not args.skip_ads:
        print("[step] Running adsorbent calculation")
        ads_out = run_pwscf(
            ads_in,
            pw_command=args.pw_command,
            omp_threads=settings["omp_threads"],
        )

    if e_pfas_ry is None and not args.skip_pfas:
        print("[step] Running PFAS calculation")
        pfas_out = run_pwscf(
            pfas_in,
            pw_command=args.pw_command,
            omp_threads=settings["omp_threads"],
        )

    if not args.skip_complex:
        print("[step] Running complex calculation")
        complex_out = run_pwscf(
            complex_in,
            pw_command=args.pw_command,
            omp_threads=settings["omp_threads"],
        )

    print("[step] Parsing energies")
    e_adsorbent_ry = extract_total_energy_ry(ads_out)

    if e_pfas_ry is None:
        e_pfas_ry = extract_total_energy_ry(pfas_out)

    e_complex_ry = extract_total_energy_ry(complex_out)

    e_adsorption_ry = e_complex_ry - e_adsorbent_ry - e_pfas_ry
    e_adsorption_ev = e_adsorption_ry * RY_TO_EV

    results = {
        "case_name": args.case_name,
        "adsorbent_name": args.adsorbent_name,
        "pfas_name": args.pfas_name,
        "adsorbent_smiles": args.adsorbent_smiles,
        "pfas_smiles": args.pfas_smiles,
        "pfas_energy_source": "provided" if args.pfas_energy_ry is not None else "parsed",
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