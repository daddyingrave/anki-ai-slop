"""
Entry point for `python -m docling_tools`.

This delegates to the IDE-configured runner in `docling_tools.run` without
introducing any CLI arguments. Edit variables in run.py and then run:

    uv run --project <project_path> --module docling_tools

"""
from __future__ import annotations

from .run import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
