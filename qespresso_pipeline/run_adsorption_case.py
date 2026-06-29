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
            "ecutwfc": 40,
            "ecutrho": 400,
            "mixing_beta": 0.2,
            "omp_threads": 1,
            "input_dft": "pbe",
        },
        "cluster": {
            "ecutwfc": 60,
            "ecutrho": 600,
            "mixing_beta": 0.15,
            "omp_threads": 1,
            "input_dft": "pbe",
        },
        "production": {
            "ecutwfc": 80,
            "ecutrho": 800,
            "mixing_beta": 0.1,
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
            "padding": 12.0,
        })
    elif system_type == "periodic":
        s.update({
            "job_type": "periodic",
            "use_gamma": False,
            "kpts": (6, 6, 1),
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

    if not pseudos_dir.exists() and not pseudos_dir.is_symlink():
        pseudos_dir.symlink_to(pseudo_source.resolve(), target_is_directory=True)


def run_pwscf(input_path: Path, pw_command="mpirun -np 1 pw.x"):
    output_path = input_path.with_suffix(".out")
    stderr_path = input_path.with_suffix(".err")
    workdir = input_path.parent
    cmd = shlex.split(pw_command) + ["-in", input_path.name]

    print(f"[run] {' '.join(cmd)} (cwd={workdir})")
    with open(output_path, "w") as f_out, open(stderr_path, "w") as f_err:
        result = subprocess.run(cmd, cwd=workdir, stdout=f_out, stderr=f_err)

    if result.returncode != 0:
        raise RuntimeError(f"pw.x failed. Return code: {result.returncode}\nSee {stderr_path}")
    return output_path


def extract_total_energy_ry(path: Path) -> float:
    energy = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "!" in line and "total energy" in line:
                try: energy = float(line.split()[4])
                except IndexError: pass
    if energy is None: raise ValueError(f"No total energy found in {path}")
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

def patch_qe_input(in_path: Path, settings: dict, nspin: int, tot_mag: float):
    modify_qe_input(
        in_path,
        job_type=settings["job_type"],
        ecutwfc=settings["ecutwfc"],
        ecutrho=settings["ecutrho"],
        use_gamma=settings["use_gamma"],
        mixing_beta=settings["mixing_beta"],
        input_dft=settings["input_dft"],
        kpts=settings["kpts"],
        calculation="relax",
        nspin=nspin,
        tot_magnetization=tot_mag
    )


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


def patch_qe_input(in_path: Path, settings: dict, nspin: int, tot_mag: float):
    modify_qe_input(
        in_path,
        job_type=settings["job_type"],
        ecutwfc=settings["ecutwfc"],
        ecutrho=settings["ecutrho"],
        use_gamma=settings["use_gamma"],
        mixing_beta=settings["mixing_beta"],
        input_dft=settings["input_dft"],
        kpts=settings["kpts"],
        calculation="relax",
        nspin=nspin,
        tot_magnetization=tot_mag
    )


def build_molecular_complex_cif(adsorbent_mol: Path, pfas_mol: Path, output_cif: Path, padding=12.0, vdw_gap=2.5):
    ads = Molecule.from_file(str(adsorbent_mol))
    pfas = Molecule.from_file(str(pfas_mol))

    ads_coords = np.array(ads.cart_coords)
    pfas_coords = np.array(pfas.cart_coords)

    ads_center = ads_coords.mean(axis=0)
    pfas_center = pfas_coords.mean(axis=0)

    ads_top_z = ads_coords[:, 2].max()
    pfas_bottom_z = pfas_coords[:, 2].min()
    
    z_shift = ads_top_z - pfas_bottom_z + vdw_gap
    
    shifted_pfas = pfas_coords - pfas_center
    shifted_pfas[:, 0] += ads_center[0]
    shifted_pfas[:, 1] += ads_center[1]
    shifted_pfas[:, 2] += pfas_center[2] + z_shift

    all_species = list(ads.species) + list(pfas.species)
    all_coords = np.vstack([ads_coords, shifted_pfas])

    mins, maxs = all_coords.min(axis=0), all_coords.max(axis=0)
    cell_size = float(max(maxs - mins) + padding)

    structure = Structure(
        Lattice.cubic(cell_size),
        all_species,
        all_coords - ((mins + maxs) / 2) + (cell_size / 2),
        coords_are_cartesian=True,
    )
    CifWriter(structure, symprec=None).write_file(str(output_cif))

def build_periodic_pfas_reference_cif(adsorbent_cif: Path, pfas_mol: Path, output_cif: Path, vdw_gap=2.5):
    # wrapped with above
    ads_structure = clean_structure_from_cif(adsorbent_cif)
    pfas = Molecule.from_file(str(pfas_mol))

    ads_coords = np.array(ads_structure.cart_coords)
    pfas_coords = np.array(pfas.cart_coords)

    cell_center_frac = np.array([0.5, 0.5, 0.5])
    cell_center_cart = ads_structure.lattice.get_cartesian_coords(cell_center_frac)
    
    pfas_xy_center = pfas_coords[:, :2].mean(axis=0)
    shifted = pfas_coords.copy()
    
    shifted[:, 0] += cell_center_cart[0] - pfas_xy_center[0]
    shifted[:, 1] += cell_center_cart[1] - pfas_xy_center[1]

    top_z = ads_coords[:, 2].max()
    pfas_min_z = shifted[:, 2].min()
    shifted[:, 2] += top_z - pfas_min_z + vdw_gap

    structure = Structure(
        ads_structure.lattice,
        list(pfas.species),
        shifted,
        coords_are_cartesian=True,
    )
    CifWriter(structure, symprec=None).write_file(str(output_cif))

def build_periodic_complex_cif_nopfas(adsorbent_cif: Path, pfas_mol: Path, output_cif: Path, vdw_gap=2.5):
    # wrapped with above
    ads_structure = clean_structure_from_cif(adsorbent_cif)
    pfas = Molecule.from_file(str(pfas_mol))

    ads_coords = np.array(ads_structure.cart_coords)
    pfas_coords = np.array(pfas.cart_coords)

    cell_center_frac = np.array([0.5, 0.5, 0.5])
    cell_center_cart = ads_structure.lattice.get_cartesian_coords(cell_center_frac)
    
    pfas_xy_center = pfas_coords[:, :2].mean(axis=0)
    shifted = pfas_coords.copy()
    
    shifted[:, 0] += cell_center_cart[0] - pfas_xy_center[0]
    shifted[:, 1] += cell_center_cart[1] - pfas_xy_center[1]

    top_z = ads_coords[:, 2].max()
    pfas_min_z = shifted[:, 2].min()
    shifted[:, 2] += top_z - pfas_min_z + vdw_gap

    structure = Structure(
        ads_structure.lattice,
        list(ads_structure.species),
        ads_coords,
        coords_are_cartesian=True,
    )
    CifWriter(structure, symprec=None).write_file(str(output_cif))

def build_periodic_complex_cif(adsorbent_cif: Path, pfas_mol: Path, output_cif: Path, vdw_gap=2.5):
    # wrapped with above
    ads_structure = clean_structure_from_cif(adsorbent_cif)
    pfas = Molecule.from_file(str(pfas_mol))

    ads_coords = np.array(ads_structure.cart_coords)
    pfas_coords = np.array(pfas.cart_coords)

    cell_center_frac = np.array([0.5, 0.5, 0.5])
    cell_center_cart = ads_structure.lattice.get_cartesian_coords(cell_center_frac)
    
    pfas_xy_center = pfas_coords[:, :2].mean(axis=0)
    shifted = pfas_coords.copy()
    
    shifted[:, 0] += cell_center_cart[0] - pfas_xy_center[0]
    shifted[:, 1] += cell_center_cart[1] - pfas_xy_center[1]

    top_z = ads_coords[:, 2].max()
    pfas_min_z = shifted[:, 2].min()
    shifted[:, 2] += top_z - pfas_min_z + vdw_gap

    structure = Structure(
        ads_structure.lattice,
        list(ads_structure.species) + list(pfas.species),
        np.vstack([ads_coords, shifted]),
        coords_are_cartesian=True,
    )
    CifWriter(structure, symprec=None).write_file(str(output_cif))


def main():
    parser = argparse.ArgumentParser()
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
    
    parser.add_argument("--vdw-gap", type=float, default=2.5, help="Initial Z-distance between adsorbate and slab")
    parser.add_argument("--nspin", type=int, default=1, help="Set to 2 for magnetic transition metals")
    parser.add_argument("--tot-magnetization", type=float, default=0.0, help="Initial magnetization")
    
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-ads", action="store_true")
    parser.add_argument("--skip-pfas", action="store_true")
    parser.add_argument("--skip-complex", action="store_true")

    args = parser.parse_args()
    settings = get_mode_settings(args.mode, args.system_type)

    case_dir = Path(args.workdir) / args.case_name
    compound_root = Path(args.compound_root)

    ads_dir, pfas_dir, complex_dir = compound_root / "adsorbents" / args.adsorbent_name, compound_root / "pfas" / args.pfas_name, case_dir / "complex"
    for d in [ads_dir, pfas_dir, complex_dir]: ensure_dir(d)

    pseudo_source = Path(args.pseudo_dir).resolve()
    for d in [ads_dir, pfas_dir, complex_dir]: link_pseudos(d, pseudo_source)

    ads_base, pfas_base, complex_base = ads_dir / "adsorbent", pfas_dir / "pfas", complex_dir / "complex"

    ads_mol, ads_cif, ads_in, ads_out = None, None, None, None
    if not args.skip_ads:
        if args.adsorbent_source == "smiles":
            ads_mol = run_obabel(args.adsorbent_smiles, str(ads_base))
            ads_cif = mol_to_cif_pymatgen(ads_mol, str(ads_base), padding=settings["padding"])
        else:
            ads_cif = Path(args.adsorbent_cif)
            CifWriter(clean_structure_from_cif(ads_cif), symprec=None).write_file(str(ads_base.with_suffix(".cif")))
        
        ads_in = run_cif2cell(ads_base.with_suffix(".cif"), str(ads_base))
        patch_qe_input(ads_in, settings, args.nspin, args.tot_magnetization)

    pfas_mol, pfas_in = None, None
    if args.pfas_energy_ry is None and not args.skip_pfas:
        pfas_mol = run_obabel(args.pfas_smiles, str(pfas_base))
        pfas_cif = mol_to_cif_pymatgen(pfas_mol, str(pfas_base), padding=settings.get("padding", 12.0))
        pfas_in = run_cif2cell(pfas_cif, str(pfas_base))
        patch_qe_input(pfas_in, get_mode_settings(args.mode, "molecule"), 1, 0.0)

    if not args.skip_complex:
        if args.system_type == "molecule":
            build_molecular_complex_cif(ads_base.with_suffix(".mol"), pfas_base.with_suffix(".mol"), complex_base.with_suffix(".cif"), padding=settings["padding"], vdw_gap=args.vdw_gap)
        else:
            build_periodic_complex_cif(ads_base.with_suffix(".cif"), pfas_base.with_suffix(".mol"), complex_base.with_suffix(".cif"), vdw_gap=args.vdw_gap)
            
        complex_in = run_cif2cell(complex_base.with_suffix(".cif"), str(complex_base))
        patch_qe_input(complex_in, settings, args.nspin, args.tot_magnetization)

    if args.prepare_only: return

    if not args.skip_ads: ads_out = run_pwscf(ads_in, args.pw_command)
    if args.pfas_energy_ry is None and not args.skip_pfas: pfas_out = run_pwscf(pfas_in, args.pw_command)
    if not args.skip_complex: complex_out = run_pwscf(complex_in, args.pw_command)

    e_ads = extract_total_energy_ry(ads_out) if not args.skip_ads else 0.0
    e_pfas = args.pfas_energy_ry if args.pfas_energy_ry else (extract_total_energy_ry(pfas_out) if not args.skip_pfas else 0.0)
    e_comp = extract_total_energy_ry(complex_out) if not args.skip_complex else 0.0

    if e_ads and e_pfas and e_comp:
        e_adsorption_ry = (e_comp - e_ads - e_pfas) 
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
                "adsorbent": e_ads,
                "pfas": e_pfas,
                "complex": e_comp,
                "adsorption": e_adsorption_ry,
            },
            "energies_ev": {
                "adsorption": e_adsorption_ev,
            }
        },
        with open(case_dir / "results.json", "w") as f: json.dump(results, f, indent=2)
        print(f"\n[SUCCESS] Adsorption Energy: {e_adsorption_ev:.4f} eV")

if __name__ == "__main__":
    main()