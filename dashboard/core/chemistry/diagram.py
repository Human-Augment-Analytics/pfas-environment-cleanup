"""Generate a representative diagram without importing RDKit in the web server."""

from pathlib import Path


def main(*, smiles: str, output: Path) -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid representative SMILES")
    Draw.MolToFile(molecule, str(output), size=(600, 400))
