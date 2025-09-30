from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ConversionArgs:
    """Arguments for converting a document.

    This utility is intended to be run from an IDE. Prefer constructing this
    dataclass (or passing its fields to convert_pdf) rather than using CLI args.
    """

    input_file_path: Path
    output_dir: Path
    output_format: Literal["markdown"] = "markdown"


class PdfConversionError(Exception):
    pass


def convert_pdf(
    *,
    input_file_path: str | Path,
    output_dir: str | Path,
    output_format: Literal["markdown"] = "markdown",
    override_existing: bool = True,
) -> Path:
    """Convert a PDF to the requested format (currently only Markdown).

    - input_file_path: path to a single PDF file to process.
    - output_dir: directory where the output will be written.
      The output filename will reuse the original stem with a new extension.
    - output_format: only "markdown" is supported for now.
    - override_existing: if False and output exists, raises instead of overwriting.

    Returns the path to the produced file.
    """
    in_path = Path(input_file_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()

    if not in_path.exists() or not in_path.is_file():
        raise PdfConversionError(f"Input file does not exist: {in_path}")
    if in_path.suffix.lower() != ".pdf":
        raise PdfConversionError(f"Only PDF files are supported, got: {in_path.suffix}")

    if output_format != "markdown":
        raise PdfConversionError(
            f"Unsupported output format: {output_format}. Only 'markdown' is supported."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{in_path.stem}.md"
    if out_file.exists() and not override_existing:
        raise PdfConversionError(f"Output file already exists: {out_file}")

    # Import lazily so general imports don't hard-require docling unless used.
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except Exception as e:  # pragma: no cover - import-time issues
        raise PdfConversionError(
            "Failed to import 'docling'. Please install project dependencies."
        ) from e

    # Convert the document using Docling
    converter = DocumentConverter()
    try:
        result = converter.convert(str(in_path))
    except Exception as e:
        raise PdfConversionError(f"Docling conversion failed: {e}") from e

    # Export markdown
    try:
        dl_doc = result.document
        md_text: str = dl_doc.export_to_markdown()  # type: ignore[attr-defined]
    except Exception as e:
        raise PdfConversionError(f"Failed to export markdown: {e}") from e

    # Try to persist media assets, if docling exposes such methods. We try a few
    # common method names across versions and ignore if not available.
    saved_media = False
    for method_name in (
        "save_media_to_folder",
        "save_media",
        "save_images",
        "save_images_to_folder",
    ):
        try:
            method = getattr(dl_doc, method_name, None)
            if callable(method):
                method(str(media_dir))
                saved_media = True
                break
        except Exception:
            # Try next method name
            pass

    # Persist markdown to the desired location.
    out_file.write_text(md_text, encoding="utf-8")

    # Provide a simple hint inside the markdown header if media saving was not possible.
    if not saved_media:
        hint = (
            "\n\n<!-- NOTE: Media extraction could not be performed via Docling API in this environment. -->\n"
            f"<!-- If images are referenced, ensure they are exported to: {media_dir} -->\n"
        )
        with out_file.open("a", encoding="utf-8") as fp:
            fp.write(hint)

    return out_file


def _demo_run(
    input_file_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    output_format: Literal["markdown"] = "markdown",
):
    """A small demo helper to run from IDE while editing variables.

    Provide values for input_file_path and output_dir below or pass them as
    parameters to this function in your IDE run configuration.
    """
    # -- Set your variables here when running from IDE --
    input_file_path = input_file_path or "/path/to/your.pdf"
    output_dir = output_dir or "./out"

    out_path = convert_pdf(
        input_file_path=input_file_path,
        output_dir=output_dir,
        output_format=output_format,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Edit these variables in your IDE and run this module.
    _demo_run(
        input_file_path=None,  # e.g., "/Users/me/Documents/file.pdf"
        output_dir=None,       # e.g., "/Users/me/Documents/converted"
        output_format="markdown",
    )
