"""
Entry point for `python -m pv_pdf_to_csv`.

This delegates to the converter module without introducing any CLI arguments.
Edit variables in run.py and then run:

    uv run --project <project_path> --module pv_pdf_to_csv

"""
from __future__ import annotations

from .run import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
