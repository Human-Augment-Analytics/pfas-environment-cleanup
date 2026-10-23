"""Command-line adapter for the local affinity helper."""

import argparse

from dashboard.core.runtimes import cpu


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpus")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    cpu.main(**vars(get_parser().parse_args()))


if __name__ == "__main__":
    main()
