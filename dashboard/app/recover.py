"""One-shot chain recovery for manual use or a periodic Slurm allocation."""

import argparse
from pathlib import Path

from dashboard.core.slurm import chaining


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    arguments = get_parser().parse_args()
    chaining.main(config=arguments.config)


if __name__ == "__main__":
    main()
