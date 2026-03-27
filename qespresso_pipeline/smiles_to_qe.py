#!/usr/bin/env python3

import argparse
import subprocess
import sys
import shutil
import re
import numpy as np
from pymatgen.core import Molecule, Lattice, Structure
from pymatgen.io.cif import CifWriter
from pathlib import Path

VESTA_PATH = "/home/max/Downloads/VESTA-gtk3/VESTA"


def check_executable(name: str):
    if shutil.which(name) is None and not Path(name).exists():
        print(f"Error: '{name}' not found.", file=sys.stderr)
        sys.exit(1)


def run_obabel(input_string: str, output_name: str) -> Path:
    output_mol = Path(f"{output_name}.mol")
    command = ["obabel", f"-:{input_string}", "-O", str(output_mol), "--gen3d"]
    print(f"[info] Creating MOL: {output_mol}")
    subprocess.run(command, check=True)

    if not output_mol.exists():
        raise FileNotFoundError(f"MOL file not created: {output_mol}")
    return output_mol


def mol_to_cif_pymatgen(input_mol: Path, output_name: str, padding: float = 10.0) -> Path:
    output_cif = Path(f"{output_name}.cif")
    print(f"[info] Creating CIF from MOL: {output_cif}")

    mol = Molecule.from_file(str(input_mol))
    coords = np.array(mol.cart_coords)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    lengths = maxs - mins

    # cell_size = float(max(lengths) + padding) # this uses a cubic boxes
    # lattice = Lattice.cubic(cell_size)

    # center = (mins + maxs) / 2
    # shifted_coords = coords - center + cell_size / 2
    box_lengths = lengths + padding
    lattice = Lattice.orthorhombic(*box_lengths)
    shifted_coords = coords - mins + padding / 2

    structure = Structure(
        lattice,
        mol.species,
        shifted_coords,
        coords_are_cartesian=True,
    )

    CifWriter(structure, symprec=None).write_file(str(output_cif))

    if not output_cif.exists():
        raise FileNotFoundError(f"CIF file not created: {output_cif}")

    print(f"[info] Complex CIF created: {output_cif} (box = {box_lengths})")
    return output_cif


def run_cif2cell(input_cif: Path, output_name: str) -> Path:
    output_in = Path(f"{output_name}.in")
    command = ["cif2cell", str(input_cif), "-p", "quantum-espresso", "-o", str(output_in)]
    print(f"[info] Creating QE input: {output_in}")
    subprocess.run(command, check=True)

    if not output_in.exists():
        raise FileNotFoundError(f"QE input not created: {output_in}")
    return output_in


def modify_qe_input(
    input_file,
    output_file=None,
    job_type="molecule",
    ecutwfc=35,
    ecutrho=280,
    use_gamma=True,
    mixing_beta=None,
):
    input_path = Path(input_file)
    prefix = input_path.stem

    if output_file is None:
        output_file = input_path

    with open(input_path, "r") as f:
        lines = f.readlines()

    header = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("&SYSTEM"):
            body_start = i
            break
        header.append(line)

    body = lines[body_start:]

    if job_type == "molecule":
        if mixing_beta is None:
            mixing_beta = 0.2
        system_extra = [
            f"  ecutwfc={ecutwfc},\n",
            f"  ecutrho={ecutrho},\n",
            "  input_dft='pbe',\n",
            "  occupations='fixed',\n",
        ]
        electrons_block = f"""&ELECTRONS
  conv_thr=1d-06,
  mixing_beta={mixing_beta}d0,
/
"""
        if use_gamma:
            kpoints_block = "K_POINTS gamma\n"
        else:
            kpoints_block = """K_POINTS {automatic}
  1 1 1 0 0 0
"""
    else:
        if mixing_beta is None:
            mixing_beta = 0.5
        system_extra = [
            f"  ecutwfc={ecutwfc},\n",
            f"  ecutrho={ecutrho},\n",
            "  input_dft='pbe',\n",
            "  occupations='smearing',\n",
            "  smearing='mv',\n",
            "  degauss=0.005d0,\n",
        ]
        electrons_block = f"""&ELECTRONS
  conv_thr=1d-06,
  mixing_beta={mixing_beta}d0,
/
"""
        kpoints_block = """K_POINTS {automatic}
  3 3 3 0 0 0
"""

    system_block = []
    i = 0
    while i < len(body):
        line = body[i]
        if line.strip().startswith("&SYSTEM"):
            system_block.append("&SYSTEM\n")
            i += 1
            while i < len(body) and not body[i].strip().startswith("/"):
                system_block.append(body[i])
                i += 1
            system_block.extend(system_extra)
            system_block.append("/\n")
            i += 1
        else:
            system_block.append(line)
            i += 1

    fixed_body = []
    for line in system_block:
        line = re.sub(r"\b([A-Za-z]+)_PSEUDO\b", r"\1.UPF", line)
        fixed_body.append(line)

    control_block = f"""&CONTROL
  calculation='scf',
  outdir='./Outputs',
  prefix='{prefix}',
  pseudo_dir='./Pseudopotentials',
  verbosity='low',
  tprnfor=.true.,
  tstress=.true.,
/
"""

    final_body = []
    inserted = False
    for line in fixed_body:
        final_body.append(line)
        if line.strip() == "/" and not inserted:
            final_body.append("\n")
            final_body.append(electrons_block)
            inserted = True

    with open(output_file, "w") as f:
        f.writelines(header)
        f.write(control_block)
        f.writelines(final_body)
        f.write("\n")
        f.write(kpoints_block)

    print(f"[info] Patched QE input: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="SMILES -> MOL -> CIF -> QE input")
    parser.add_argument("input", help="Input SMILES")
    parser.add_argument("output", help="Base output filename")
    parser.add_argument("--padding", type=float, default=10.0)
    parser.add_argument("--job-type", choices=["molecule", "periodic"], default="molecule")
    args = parser.parse_args()

    check_executable("obabel")
    check_executable("cif2cell")

    try:
        mol_file = run_obabel(args.input, args.output)
        cif_file = mol_to_cif_pymatgen(mol_file, args.output, padding=args.padding)
        qe_input = run_cif2cell(cif_file, args.output)
        modify_qe_input(qe_input, job_type=args.job_type)

        print("\n[done] Pipeline completed successfully.")
        print(f"  - {mol_file}")
        print(f"  - {cif_file}")
        print(f"  - {qe_input}")

    except subprocess.CalledProcessError as e:
        print(f"External command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()