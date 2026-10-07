"""Forward the repository-root launcher to the isolated web package."""

from pathlib import Path

__path__ = [str(Path(__file__).parent / "web/dashboard")]
if __name__ == "__main__":
    from dashboard.__main__ import main  # ty: ignore[unresolved-import]

    main()
