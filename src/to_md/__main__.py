"""
Entry point for `python -m to_md`.

This delegates to the IDE-configured runner in `to_md.run` without
introducing any CLI arguments. Edit variables in run.py and then run:

    uv run --project <project_path> --module to_md

"""
from __future__ import annotations

from .run import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
