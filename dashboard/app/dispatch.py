"""One-shot compute-node queue advancement; never performs server recovery."""

import argparse
from pathlib import Path

from dashboard.core.slurm import chaining


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predecessor", required=True)
    return parser


def main() -> None:
    arguments = get_parser().parse_args()
    chaining.main(
        config=arguments.config,
        predecessor=arguments.predecessor,
    )


if __name__ == "__main__":
    main()
