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

# Set the appropriate path for VESTA
VESTA_PATH = "/home/max/Downloads/VESTA-gtk3/VESTA"

def check_executable(name: str):
    """Ensure required external executable exists."""
    if shutil.which(name) is None:
        print(f"Error: '{name}' not found in PATH.", file=sys.stderr)
        sys.exit(1)


def run_obabel(input_string: str, output_name: str) -> Path:
    """
    Run OpenBabel command in the structure obabel -:"[input]" -O output.mol --gen3d
    """
    output_mol = Path(f"{output_name}.mol")

    command = [
        "obabel",
        f"-:{input_string}",
        "-O",
        str(output_mol),
        "--gen3d",
    ]

    print("Creating .mol file.")
    subprocess.run(command, check=True)

    if not output_mol.exists():
        print("Error: MOL file not created.", file=sys.stderr)
        sys.exit(1)

    return output_mol


def run_vesta(input_mol: Path, output_name: str) -> Path:
    """
    Run VESTA command in the structure VESTA -nogui -i input.mol -save *.cif output.cif
    """
    output_cif = Path(f"{output_name}.cif")

    command = [
        VESTA_PATH,
        "-nogui",
        "-i",
        str(input_mol),
        "-save",
        "*.cif",
        str(output_cif),
    ]

    print("Creating .cif file.")
    subprocess.run(command, check=True)

    if not output_cif.exists():
        print("Warning: CIF file not detected after VESTA.", file=sys.stderr)

    return output_cif

def mol_to_cif_pymatgen(input_mol: Path, output_name: str, padding: float = 5.0) -> Path:
    """
    Convert a MOL file to a CIF file using pymatgen:
    Automatically detecting molecule size and placing in a cubic unit cell,
    adding optional padding to avoid overlap and have appropriate size.
    """

    output_cif = Path(f"{output_name}.cif")

    print("Creating .cif file from MOL using pymatgen.")

    # Load molecule
    mol = Molecule.from_file(str(input_mol))
    coords = np.array(mol.cart_coords)

    # Determine molecule bounding box
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    lengths = maxs - mins

    # Determine cubic cell size
    cell_size = float(max(lengths) + padding)

    # Create lattice
    lattice = Lattice.cubic(cell_size)

    # Center molecule
    center = (mins + maxs) / 2
    shifted_coords = coords - center + cell_size / 2

    # Create structure
    structure = Structure(
        lattice,
        mol.species,
        shifted_coords,
        coords_are_cartesian=True
    )

    # Write CIF
    writer = CifWriter(structure, symprec=None)
    writer.write_file(str(output_cif))

    if not output_cif.exists():
        print("Warning: CIF file not detected after conversion.", file=sys.stderr)
    else:
        print(f"CIF file created: {output_cif} (cell size ≈ {cell_size:.2f} Å)")

    return output_cif


def run_cif2cell(input_cif: Path, output_name: str) -> Path:
    """
    Run cif2cell in the structure cif2cell input.cif -p quantum-espresso -o output.in
    """
    output_in = Path(f"{output_name}.in")

    command = [
        "cif2cell",
        str(input_cif),
        "-p",
        "quantum-espresso",
        "-o",
        str(output_in),
    ]

    print("Creating QE .in file.")
    subprocess.run(command, check=True)

    if not output_in.exists():
        print("Warning: QE input file not detected after cif2cell.", file=sys.stderr)

    return output_in
    
def modify_qe_input(input_file, output_file=None):
    """
    Modify a Quantum Espresso input file produced by cif2cell by:
      - inserting CONTROL and ELECTRONS blocks
      - extending SYSTEM block
      - mapping *_PSEUDO to appropriate .upf file
      - appending K-mesh footer block
    Note: ensure Pseudopotential files are in proper directory.
    """

    input_path = Path(input_file)
    prefix = input_path.stem

    if output_file is None:
        output_file = input_path

    with open(input_path, "r") as f:
        lines = f.readlines()

    # Split SYSTEM header and body
    header = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("&SYSTEM"):
            body_start = i
            break
        header.append(line)

    body = lines[body_start:]

    # Replace with modified SYSTEM block
    system_block = []
    i = 0
    while i < len(body):
        line = body[i]

        if line.strip().startswith("&SYSTEM"):
            system_block.append("&SYSTEM\n")
            i += 1

            # copy existing parameters
            while not body[i].strip().startswith("/"):
                system_block.append(body[i])
                i += 1

            # add new parameters before closing
            # ecutwfc and ecutrho thresholds can be adjusted lower
            # adjustment depends on atomic species used
            # occupations metric depends on compound character (molecular, metallic, etc.)
            # TO-DO: automatic adjustment and occupations setting?
            system_block.append("  ecutwfc=60,\n")
            system_block.append("  ecutrho=480,\n")
            system_block.append("  input_dft='pbe',\n")
            system_block.append("  occupations='smearing',\n")
            system_block.append("  smearing='mv',\n")
            system_block.append("  degauss=0.005d0,\n")
            system_block.append("/\n")

            i += 1
        else:
            system_block.append(line)
            i += 1

    # Fix pseudopotential names with new files
    fixed_body = []
    for line in system_block:
        if "ATOMIC_SPECIES" in line:
            fixed_body.append(line)
            continue

        # replace *_PSEUDO
        line = re.sub(r"\b([A-Za-z]+)_PSEUDO\b", r"\1.UPF", line)
        fixed_body.append(line)

    # CONTROL and ELECTRONS blocks
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

    # conv_thr can be increase to hasten convergence 
    electrons_block = """&ELECTRONS
  conv_thr=1d-07,
  mixing_beta=0.7d0,
/
"""

    # Append K-mesh footer
    # Depending on compound, K-mesh can be modified for optimality 
    # Typically tested at different values for convergence
    # Can be standardized via Monkhorst-Pack method, or 0.15-0.3 1/A distance
    # Different for non-metals/metals, lattices, etc.
    # TO-DO: Standardize a method for choosing initial K-mesh, following below:
    # https://qeinputgenerator.materialscloud.io/
    kpoints_block = """K_POINTS {automatic}
  5 5 5 0 0 0
"""

    # Insert ELECTRONS block after SYSTEM block
    final_body = []
    inserted = False
    for line in fixed_body:
        final_body.append(line)
        if line.strip() == "/" and not inserted:
            final_body.append("\n")
            final_body.append(electrons_block)
            inserted = True

    # Write final file
    with open(output_file, "w") as f:
        f.writelines(header)
        f.write(control_block)
        f.writelines(final_body)
        f.write("\n")
        f.write(kpoints_block)


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: SMILES to MOL to CIF to Quantum ESPRESSO input"
    )
    parser.add_argument(
        "input",
        help="Input SMILES format:"
    )
    parser.add_argument(
        "output",
        help="Base output filename (without extension):"
    )

    args = parser.parse_args()

    # Check required executables
    check_executable("obabel")
    check_executable(VESTA_PATH)
    check_executable("cif2cell")

    try:
        mol_file = run_obabel(args.input, args.output)
        cif_file = mol_to_cif_pymatgen(mol_file, args.output)
        qe_input = run_cif2cell(cif_file, args.output)
        modify_qe_input(qe_input)

        print("\nPipeline completed successfully.")
        print("Generated files:")
        print(f"  - {mol_file}")
        print(f"  - {cif_file}")
        print(f"  - {qe_input}")

    except subprocess.CalledProcessError as e:
        print(f"External command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
