"""Command-line adapter for legacy worker."""

import argparse
from pathlib import Path

from dashboard.core.slurm import legacy_worker as workflow


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    return parser


def main() -> None:
    workflow.main(**vars(get_parser().parse_args()))


if __name__ == "__main__":
    main()
