"""
Run this from your IDE. Set the variables below and run the file/module.
No CLI arguments are used.
"""
from __future__ import annotations

from pathlib import Path

from .pdf_to_markdown import convert_pdf

# === Configure these variables in your IDE run configuration or by editing below ===
INPUT_FILE_PATH: str | Path = ".testdata/Computing Systems Elements 2nd Ed-part-3.pdf"  # e.g., "/Users/me/Docs/file.pdf"
OUTPUT_DIR: str | Path = "./out"                  # e.g., "/Users/me/Docs/converted"
OUTPUT_FORMAT: str = "markdown"                    # only "markdown" supported currently
# ==================================================================================


def main() -> None:
    out_path = convert_pdf(
        input_file_path=INPUT_FILE_PATH,
        output_dir=OUTPUT_DIR,
        output_format="markdown" if OUTPUT_FORMAT.lower() in {"md", "markdown"} else OUTPUT_FORMAT,  # type: ignore[arg-type]
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
