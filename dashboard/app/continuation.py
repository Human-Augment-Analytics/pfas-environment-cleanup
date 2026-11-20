"""Inspect downloaded QE output and optionally prepare a coordinate continuation."""

import argparse
from pathlib import Path

from dashboard.core.jobs import continuation


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Write a new input file; refuses overwrites"
    )
    return parser


def main() -> None:
    arguments = get_parser().parse_args()
    continuation.main(directory=arguments.directory, output=arguments.output)


if __name__ == "__main__":
    main()
