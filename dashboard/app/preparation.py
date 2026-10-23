"""Command-line adapter for preparation."""

import argparse
from pathlib import Path

from dashboard.core.chemistry import preparation as workflow


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("directory", type=Path)
    return parser


def main() -> None:
    workflow.main(**vars(get_parser().parse_args()))


if __name__ == "__main__":
    main()
