"""Compatibility for already-submitted jobs; new jobs execute pw.x directly."""

from dashboard.app.legacy_worker import main

if __name__ == "__main__":
    main()
