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
from pymatgen.io.ase import AseAtomsAdaptor

from smiles_to_qe import run_obabel, mol_to_cif_pymatgen, run_cif2cell, modify_qe_input

RY_TO_EV = 13.605693009


def get_mode_settings(mode: str, system_type: str):
    base = {
        "lowmem": {
            "ecutwfc": 25,
            "ecutrho": 200,
            "mixing_beta": 0.2,
            "omp_threads": 1,
            "input_dft": "pbe",
        },
        "cluster": {
            "ecutwfc": 30,
            "ecutrho": 240,
            "mixing_beta": 0.15,
            "omp_threads": 1,
            "input_dft": "pbe",
        },
        "production": {
            "ecutwfc": 40,
            "ecutrho": 200,
            "mixing_beta": 0.15,
            "omp_threads": 1,
            "input_dft": "vdW-DF-cx" if system_type == "periodic" else "pbe",
        },
    }

    if mode not in base:
        raise ValueError(f"Unknown mode: {mode}")

    s = dict(base[mode])

    if system_type == "molecule":
        s.update({
            "job_type": "molecule",
            "use_gamma": True,
            "kpts": (1, 1, 1),
            "padding": 8.0,
        })
    elif system_type == "periodic":
        s.update({
            "job_type": "periodic",
            "use_gamma": False,
            "kpts": (2, 2, 1),
            "padding": 0.0,
        })
    else:
        raise ValueError(f"Unknown system_type: {system_type}")

    return s


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def require_existing(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    return path


def link_pseudos(case_subdir: Path, pseudo_source: Path):
    pseudos_dir = case_subdir / "Pseudopotentials"
    outputs_dir = case_subdir / "Outputs"
    outputs_dir.mkdir(exist_ok=True)

    if pseudos_dir.exists() or pseudos_dir.is_symlink():
        return

    pseudos_dir.symlink_to(pseudo_source.resolve(), target_is_directory=True)


def run_pwscf(input_path: Path, pw_command="pw.x", omp_threads=1):
    output_path = input_path.with_suffix(".out")
    stderr_path = input_path.with_suffix(".err")

    workdir = input_path.parent
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)

    cmd = shlex.split(pw_command) + ["-in", input_path.name]

    print(f"[run] {' '.join(cmd)} (cwd={workdir})")

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


def clean_structure_from_cif(input_cif: Path) -> Structure:
    try:
        return Structure.from_file(str(input_cif))
    except Exception as e:
        print(f"[warn] Pymatgen failed to parse CIF directly: {e}")
        print("[warn] Falling back to ASE -> pymatgen conversion")
        from ase.io import read
        ase_atoms = read(str(input_cif))
        return AseAtomsAdaptor.get_structure(ase_atoms)


def prepare_from_smiles(smiles: str, outbase: Path, settings: dict):
    mol_path = run_obabel(smiles, str(outbase))
    cif_path = mol_to_cif_pymatgen(mol_path, str(outbase), padding=settings["padding"])
    in_path = run_cif2cell(cif_path, str(outbase))

    modify_qe_input(
        in_path,
        job_type=settings["job_type"],
        ecutwfc=settings["ecutwfc"],
        ecutrho=settings["ecutrho"],
        use_gamma=settings["use_gamma"],
        mixing_beta=settings["mixing_beta"],
        input_dft=settings["input_dft"],
        kpts=settings["kpts"],
    )

    return mol_path, cif_path, in_path


def prepare_from_cif(input_cif: Path, outbase: Path, settings: dict):
    cif_path = outbase.with_suffix(".cif")

    structure = clean_structure_from_cif(input_cif)
    CifWriter(structure, symprec=None).write_file(str(cif_path))

    in_path = run_cif2cell(cif_path, str(outbase))

    modify_qe_input(
        in_path,
        job_type=settings["job_type"],
        ecutwfc=settings["ecutwfc"],
        ecutrho=settings["ecutrho"],
        use_gamma=settings["use_gamma"],
        mixing_beta=settings["mixing_beta"],
        input_dft=settings["input_dft"],
        kpts=settings["kpts"],
    )

    return cif_path, in_path


def patch_qe_input(in_path: Path, settings: dict):
    modify_qe_input(
        in_path,
        job_type=settings["job_type"],
        ecutwfc=settings["ecutwfc"],
        ecutrho=settings["ecutrho"],
        use_gamma=settings["use_gamma"],
        mixing_beta=settings["mixing_beta"],
        input_dft=settings["input_dft"],
        kpts=settings["kpts"],
    )


def build_molecular_complex_cif(
    adsorbent_mol: Path,
    pfas_mol: Path,
    output_cif: Path,
    shift=(4.0, 0.0, 0.0),
    padding=8.0,
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


def build_periodic_pfas_reference_cif(
    adsorbent_cif: Path,
    pfas_mol: Path,
    output_cif: Path,
    shift=(0.0, 0.0, 3.2),
):
    ads_structure = clean_structure_from_cif(adsorbent_cif)
    pfas = Molecule.from_file(str(pfas_mol))

    ads_coords = np.array(ads_structure.cart_coords)
    pfas_coords = np.array(pfas.cart_coords)

    ads_xy_center = ads_coords[:, :2].mean(axis=0)
    pfas_xy_center = pfas_coords[:, :2].mean(axis=0)

    shifted = pfas_coords.copy()
    shifted[:, 0] += ads_xy_center[0] - pfas_xy_center[0] + shift[0]
    shifted[:, 1] += ads_xy_center[1] - pfas_xy_center[1] + shift[1]

    top_z = ads_coords[:, 2].max()
    pfas_min_z = shifted[:, 2].min()
    shifted[:, 2] += top_z - pfas_min_z + shift[2]

    structure = Structure(
        ads_structure.lattice,
        list(pfas.species),
        shifted,
        coords_are_cartesian=True,
    )

    CifWriter(structure, symprec=None).write_file(str(output_cif))


def build_periodic_complex_cif(
    adsorbent_cif: Path,
    pfas_mol: Path,
    output_cif: Path,
    shift=(0.0, 0.0, 3.2),
):
    ads_structure = clean_structure_from_cif(adsorbent_cif)
    pfas = Molecule.from_file(str(pfas_mol))

    ads_coords = np.array(ads_structure.cart_coords)
    pfas_coords = np.array(pfas.cart_coords)

    ads_xy_center = ads_coords[:, :2].mean(axis=0)
    pfas_xy_center = pfas_coords[:, :2].mean(axis=0)

    shifted = pfas_coords.copy()
    shifted[:, 0] += ads_xy_center[0] - pfas_xy_center[0] + shift[0]
    shifted[:, 1] += ads_xy_center[1] - pfas_xy_center[1] + shift[1]

    top_z = ads_coords[:, 2].max()
    pfas_min_z = shifted[:, 2].min()
    shifted[:, 2] += top_z - pfas_min_z + shift[2]

    all_species = list(ads_structure.species) + list(pfas.species)
    all_coords = np.vstack([ads_coords, shifted])

    structure = Structure(
        ads_structure.lattice,
        all_species,
        all_coords,
        coords_are_cartesian=True,
    )

    CifWriter(structure, symprec=None).write_file(str(output_cif))


def main():
    parser = argparse.ArgumentParser(description="Run adsorption-energy workflow")

    parser.add_argument("--case-name", required=True)
    parser.add_argument("--adsorbent-name", required=True)
    parser.add_argument("--pfas-name", required=True)

    parser.add_argument("--adsorbent-source", choices=["smiles", "cif"], required=True)
    parser.add_argument("--adsorbent-smiles", required=False)
    parser.add_argument("--adsorbent-cif", required=False)

    parser.add_argument("--pfas-smiles", required=False)
    parser.add_argument("--pfas-energy-ry", type=float, default=None)

    parser.add_argument("--system-type", choices=["molecule", "periodic"], default="molecule")
    parser.add_argument("--mode", choices=["lowmem", "cluster", "production"], default="cluster")

    parser.add_argument("--workdir", default="dft_cases")
    parser.add_argument("--compound-root", default="compounds")
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--pw-command", default="pw.x")

    parser.add_argument("--complex-shift-x", type=float, default=4.0)
    parser.add_argument("--complex-shift-y", type=float, default=0.0)
    parser.add_argument("--complex-shift-z", type=float, default=3.2)

    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-ads", action="store_true")
    parser.add_argument("--skip-pfas", action="store_true")
    parser.add_argument("--skip-complex", action="store_true")

    args = parser.parse_args()
    settings = get_mode_settings(args.mode, args.system_type)

    if args.adsorbent_source == "smiles" and not args.skip_ads and not args.adsorbent_smiles:
        raise ValueError("--adsorbent-smiles is required when --adsorbent-source smiles is used")

    if args.adsorbent_source == "cif" and not args.skip_ads and not args.adsorbent_cif:
        raise ValueError("--adsorbent-cif is required when --adsorbent-source cif is used")

    if args.pfas_energy_ry is None and not args.skip_pfas and not args.pfas_smiles:
        raise ValueError("--pfas-smiles is required unless --skip-pfas or --pfas-energy-ry is used")

    shift = (args.complex_shift_x, args.complex_shift_y, args.complex_shift_z)

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

    ads_mol = None
    ads_cif = None
    ads_in = None
    ads_out = None

    if args.skip_ads:
        print("[skip] Reusing adsorbent files")
        ads_cif = require_existing(ads_cif_expected, "Adsorbent CIF")
        ads_in = require_existing(ads_in_expected, "Adsorbent input")
        ads_out = require_existing(ads_out_expected, "Adsorbent output")
        if args.adsorbent_source == "smiles":
            ads_mol = require_existing(ads_mol_expected, "Adsorbent MOL")
    else:
        if args.adsorbent_source == "smiles":
            print("[step] Preparing adsorbent from SMILES")
            ads_mol, ads_cif, ads_in = prepare_from_smiles(
                args.adsorbent_smiles,
                ads_base,
                settings,
            )
        else:
            print("[step] Preparing adsorbent from CIF")
            ads_cif, ads_in = prepare_from_cif(
                Path(args.adsorbent_cif),
                ads_base,
                settings,
            )

    pfas_mol = None
    pfas_in = None
    pfas_out = None
    e_pfas_ry = None

    if args.pfas_energy_ry is not None:
        e_pfas_ry = args.pfas_energy_ry

        if not args.skip_complex:
            if args.skip_pfas:
                pfas_mol = require_existing(pfas_mol_expected, "PFAS MOL")
            else:
                print("[step] Preparing PFAS MOL from SMILES")
                pfas_mol = run_obabel(args.pfas_smiles, str(pfas_base))

    elif args.skip_pfas:
        print("[skip] Reusing PFAS files")
        pfas_mol = require_existing(pfas_mol_expected, "PFAS MOL")
        pfas_in = require_existing(pfas_in_expected, "PFAS input")
        pfas_out = require_existing(pfas_out_expected, "PFAS output")

    else:
        print("[step] Preparing PFAS from SMILES")

        if args.system_type == "periodic":
            pfas_mol = run_obabel(args.pfas_smiles, str(pfas_base))

            if ads_cif is None:
                raise ValueError("Periodic PFAS reference requires adsorbent CIF")

            build_periodic_pfas_reference_cif(
                adsorbent_cif=ads_cif,
                pfas_mol=pfas_mol,
                output_cif=pfas_cif_expected,
                shift=shift,
            )

            pfas_in = run_cif2cell(pfas_cif_expected, str(pfas_base))
            patch_qe_input(pfas_in, settings)

        else:
            pfas_settings = get_mode_settings(args.mode, "molecule")
            pfas_mol, _, pfas_in = prepare_from_smiles(
                args.pfas_smiles,
                pfas_base,
                pfas_settings,
            )

    complex_in = None
    complex_out = None

    if args.skip_complex:
        print("[skip] Reusing complex files")
        complex_in = require_existing(complex_in_expected, "Complex input")
        complex_out = require_existing(complex_out_expected, "Complex output")
    else:
        if pfas_mol is None:
            raise ValueError("Cannot build complex without PFAS structure")

        print("[step] Building complex")

        if args.system_type == "molecule":
            if ads_mol is None:
                raise ValueError("Cannot build molecular complex without adsorbent MOL")

            build_molecular_complex_cif(
                adsorbent_mol=ads_mol,
                pfas_mol=pfas_mol,
                output_cif=complex_cif_expected,
                shift=shift,
                padding=settings["padding"],
            )

        else:
            if ads_cif is None:
                raise ValueError("Cannot build periodic complex without adsorbent CIF")

            build_periodic_complex_cif(
                adsorbent_cif=ads_cif,
                pfas_mol=pfas_mol,
                output_cif=complex_cif_expected,
                shift=shift,
            )

        complex_in = run_cif2cell(complex_cif_expected, str(complex_base))
        patch_qe_input(complex_in, settings)

    if args.prepare_only:
        print("[done] Preparation complete")
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
            omp_threads=settings["omp_threads"] if args.system_type == "periodic" else 1,
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
        "adsorbent_source": args.adsorbent_source,
        "system_type": args.system_type,
        "mode": args.mode,
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
    }

    results_path = case_dir / "results.json"
    ensure_dir(case_dir)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()