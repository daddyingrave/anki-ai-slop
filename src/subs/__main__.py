"""
Entry point for `python -m subs`.

This delegates to the runner module without introducing any CLI arguments.
Edit variables in run.py and then run:

    uv run --project <project_path> --module subs

"""
from __future__ import annotations

from .run import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
