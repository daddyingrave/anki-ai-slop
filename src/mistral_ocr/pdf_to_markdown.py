from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from mistralai import Mistral
from PIL import Image

logger = logging.getLogger(__name__)


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
    """Exception raised when PDF conversion fails."""

    pass


def _pdf_to_base64(pdf_path: Path) -> str:
    """Convert PDF file to base64 encoded string.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Base64 encoded string of the PDF content.

    Raises:
        IOError: If the file cannot be read.
    """
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_base64_from_data_url(data_url: str) -> str:
    """Extract base64 data from a data URL.

    Args:
        data_url: Data URL in format 'data:image/jpeg;base64,<base64_data>'.

    Returns:
        The base64 encoded string.

    Raises:
        ValueError: If the data URL format is invalid.
    """
    if not data_url.startswith("data:"):
        return data_url

    parts = data_url.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid data URL format: {data_url[:50]}...")

    return parts[1]


def _save_image(
    img_id: str, img_base64: str, media_dir: Path
) -> tuple[bool, Optional[str]]:
    """Save an image from base64 data to the media directory.

    Args:
        img_id: Original image identifier from OCR response.
        img_base64: Base64 encoded image data (may be a data URL).
        media_dir: Directory to save the image.

    Returns:
        Tuple of (success: bool, filename: Optional[str]).
    """
    try:
        # Extract base64 from data URL if needed
        img_base64 = _extract_base64_from_data_url(img_base64)

        # Decode base64 image data
        img_data = base64.b64decode(img_base64)

        if len(img_data) == 0:
            logger.warning(f"Empty image data for {img_id}")
            return False, None

        # Open image with PIL and save as PNG
        pil_image = Image.open(io.BytesIO(img_data))

        # Convert image ID to PNG extension
        img_id_parts = img_id.rsplit(".", 1)
        if len(img_id_parts) == 2:
            img_filename = f"{img_id_parts[0]}.png"
        else:
            img_filename = f"{img_id}.png"

        img_path = media_dir / img_filename
        pil_image.save(str(img_path), format="PNG")

        return True, img_filename

    except Exception as e:
        logger.warning(f"Failed to save image {img_id}: {e}")
        return False, None


def _process_pdf_with_mistral_ocr(
    client: Mistral, pdf_b64: str, media_dir: Path
) -> str:
    """Process PDF with Mistral OCR API and save extracted images.

    Args:
        client: Initialized Mistral API client.
        pdf_b64: Base64 encoded PDF content.
        media_dir: Directory to save extracted images.

    Returns:
        Markdown formatted text with page separators and corrected image paths.

    Raises:
        PdfConversionError: If OCR processing fails.
    """
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
            include_image_base64=True,
        )

        # Extract markdown content from pages and save images
        markdown_parts: list[str] = []

        if not hasattr(ocr_response, "pages") or not ocr_response.pages:
            logger.warning("No pages found in OCR response")
            return "<!-- No pages found in OCR response -->\n"

        for page in ocr_response.pages:
            page_index = getattr(page, "index", len(markdown_parts) + 1)
            page_markdown = getattr(page, "markdown", "")

            # Process and save images from this page
            if hasattr(page, "images") and page.images:
                for img in page.images:
                    img_id = getattr(img, "id", None)
                    img_base64 = getattr(img, "image_base64", None)

                    if img_id and img_base64:
                        success, img_filename = _save_image(
                            img_id, img_base64, media_dir
                        )

                        if success and img_filename:
                            # Update markdown references to point to media folder
                            page_markdown = page_markdown.replace(
                                f"]({img_id})", f"](media/{img_filename})"
                            )

            if page_markdown:
                markdown_parts.append(
                    f"<!-- Page {page_index} -->\n\n{page_markdown}"
                )
            else:
                logger.warning(f"No content extracted for page {page_index}")
                markdown_parts.append(
                    f"<!-- Page {page_index}: No content extracted -->"
                )

        if not markdown_parts:
            logger.warning("No markdown content extracted from PDF")
            return "<!-- No content extracted from PDF -->\n"

        return "\n\n".join(markdown_parts)

    except Exception as e:
        raise PdfConversionError(f"Mistral OCR API call failed: {e}") from e


def convert_pdf(
    *,
    input_file_path: str | Path,
    output_dir: str | Path,
    output_format: Literal["markdown"] = "markdown",
    override_existing: bool = True,
) -> Path:
    """Convert a PDF to Markdown using Mistral OCR API.

    This function processes a PDF document using Mistral's OCR service, extracting
    text while preserving document structure, formatting, and embedded images.
    Images are saved to a media subdirectory and referenced in the markdown output.

    Args:
        input_file_path: Path to a single PDF file to process.
        output_dir: Directory where the output will be written. The output filename
            will reuse the original stem with a .md extension.
        output_format: Output format. Currently only "markdown" is supported.
        override_existing: If False and output file exists, raises an error instead
            of overwriting.

    Returns:
        Path to the produced markdown file.

    Raises:
        PdfConversionError: If input validation fails, API key is missing, or
            conversion process encounters an error.

    Environment Variables:
        MISTRAL_API_KEY: Required. Your Mistral API key for authentication.

    Example:
        >>> from pathlib import Path
        >>> output = convert_pdf(
        ...     input_file_path="document.pdf",
        ...     output_dir="./output"
        ... )
        >>> print(f"Converted to: {output}")
    """
    # Validate and resolve paths
    in_path = Path(input_file_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()

    if not in_path.exists() or not in_path.is_file():
        raise PdfConversionError(f"Input file does not exist: {in_path}")
    if in_path.suffix.lower() != ".pdf":
        raise PdfConversionError(
            f"Only PDF files are supported, got: {in_path.suffix}"
        )

    if output_format != "markdown":
        raise PdfConversionError(
            f"Unsupported output format: {output_format}. Only 'markdown' is supported."
        )

    # Validate API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise PdfConversionError(
            "MISTRAL_API_KEY environment variable is not set. "
            "Please set it with your Mistral API key."
        )

    # Prepare output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{in_path.stem}.md"
    if out_file.exists() and not override_existing:
        raise PdfConversionError(f"Output file already exists: {out_file}")

    # Initialize Mistral client
    try:
        client = Mistral(api_key=api_key)
    except Exception as e:
        raise PdfConversionError(f"Failed to initialize Mistral client: {e}") from e

    # Convert PDF to base64
    try:
        pdf_b64 = _pdf_to_base64(in_path)
    except Exception as e:
        raise PdfConversionError(f"Failed to read PDF file: {e}") from e

    # Process PDF with Mistral OCR API
    try:
        md_text = _process_pdf_with_mistral_ocr(client, pdf_b64, media_dir)
    except Exception as e:
        raise PdfConversionError(f"Failed to process PDF with Mistral OCR: {e}") from e

    # Write output file
    try:
        out_file.write_text(md_text, encoding="utf-8")
    except Exception as e:
        raise PdfConversionError(f"Failed to write output file: {e}") from e

    logger.info(f"Successfully converted {in_path.name} to {out_file}")
    return out_file


def _demo_run(
    input_file_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    output_format: Literal["markdown"] = "markdown",
) -> None:
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
        output_dir=None,  # e.g., "/Users/me/Documents/converted"
        output_format="markdown",
    )
