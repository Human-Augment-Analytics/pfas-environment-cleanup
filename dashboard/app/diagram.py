"""Command-line adapter for diagram."""

import argparse
from pathlib import Path

from dashboard.core.chemistry import diagram as workflow


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", type=str)
    parser.add_argument("output", type=Path)
    return parser


def main() -> None:
    workflow.main(**vars(get_parser().parse_args()))


if __name__ == "__main__":
    main()
