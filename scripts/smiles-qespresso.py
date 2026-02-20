#!/usr/bin/env python3

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


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
    Run VESTA command in the structure VESTA -nogui -open input.mol -save *.cif output.cif
    """
    output_cif = Path(f"{output_name}.cif")

    command = [
        "VESTA",
        "-nogui",
        "-open",
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
    check_executable("VESTA")
    check_executable("cif2cell")

    try:
        mol_file = run_obabel(args.input, args.output)
        cif_file = run_vesta(mol_file, args.output)
        qe_input = run_cif2cell(cif_file, args.output)

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