"""Parse the dashboard CLI and pass explicit options to the core workflow."""

import argparse

from dashboard.core import main as workflow


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PFAS React/FastAPI dashboard")
    parser.add_argument(
        "action", nargs="?", choices=["serve", "export"], default="serve"
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output")
    return parser


def main() -> None:
    parser = get_parser()
    try:
        workflow.main(**vars(parser.parse_args()))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
