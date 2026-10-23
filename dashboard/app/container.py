"""Command-line adapter for workstation memory isolation."""

import argparse
from pathlib import Path

from dashboard.core.runtimes import container


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    raise SystemExit(container.main(**vars(get_parser().parse_args())))


if __name__ == "__main__":
    main()
