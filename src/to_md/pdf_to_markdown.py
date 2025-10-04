from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTableSerializer,
    MarkdownPictureSerializer,
)
from docling_core.types.doc.document import DoclingDocument, PictureItem


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


# Custom picture serializer to emit Markdown image links pointing to saved media files
class FileLinkPictureSerializer(MarkdownPictureSerializer):  # type: ignore[misc]
    def __init__(self, pic_relmap: dict[int, str]):
        # Mapping from picture identity (id) -> relative path string to use in Markdown
        self._pic_relmap = pic_relmap

    @staticmethod
    def _key(obj: PictureItem) -> int:
        # Use Python object identity as a stable, hashable key during one conversion run
        return id(obj)

    def serialize(  # type: ignore[override]
            self,
            *,
            item: PictureItem,
            doc_serializer: BaseDocSerializer,
            doc: DoclingDocument,
            separator: Optional[str] = None,
            **kwargs,
    ) -> SerializationResult:
        # Determine alt text from caption or page
        alt_text = "Image"
        try:
            page_no = item.prov[0].page_no if getattr(item, "prov", None) else None
            caption = getattr(item, "caption", None)
            if caption:
                # caption may be a DocItem; try text attr or str(caption)
                cap_txt = getattr(caption, "text", None) or str(caption)
                cap_txt = str(cap_txt).strip()
                if cap_txt:
                    alt_text = cap_txt
            elif page_no is not None:
                alt_text = f"Image (page {page_no})"
        except Exception:
            pass

        rel_key = self._key(item)
        rel_path = self._pic_relmap.get(rel_key)
        if not rel_path:
            # Fallback to placeholder if mapping missing
            text_res = getattr(doc_serializer, "params", None)
            placeholder = "<!-- image -->"
            if hasattr(text_res, "image_placeholder"):
                placeholder = str(text_res.image_placeholder)
            return create_ser_result(text=placeholder, span_source=item)

        md_line = f"![{alt_text}]({rel_path})"
        md_line = doc_serializer.post_process(text=md_line)
        return create_ser_result(text=md_line, span_source=item)


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

    # Configure DocumentConverter with optimized settings for quality and image extraction
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,  # Enable table structure detection
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=False,  # Let the table model define text cells to avoid merged-PDF-cell artifacts
        ),
        generate_picture_images=True,  # Keep picture images available for serializers
        generate_table_images=True
    )

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }

    converter = DocumentConverter(format_options=format_options)
    try:
        result = converter.convert(str(in_path))
    except Exception as e:
        raise PdfConversionError(f"Docling conversion failed: {e}") from e

    # Prepare document and build picture mapping + media files
    try:
        dl_doc = result.document
    except Exception as e:
        raise PdfConversionError(f"Failed to access converted document: {e}") from e

    # Save images and build a mapping used by the Markdown serializer
    pic_relmap: dict[int, str] = {}
    saved_media = False
    try:
        pictures = getattr(dl_doc, "pictures", []) or []
        for i, pic in enumerate(pictures):
            try:
                page_no = pic.prov[0].page_no if getattr(pic, "prov", None) else i
                image_name = f"image_page_{page_no}_{i}.png"
                rel_path = f"media/{image_name}"
                image_path = media_dir / image_name

                pil_image = pic.get_image(dl_doc)
                if pil_image is not None:
                    pil_image.save(str(image_path))
                    saved_media = True

                # map even if not saved (best effort) so that links appear
                pic_relmap[id(pic)] = rel_path
            except Exception:
                # Continue with next image if one fails
                continue
    except Exception:
        # If the API changes, fall back gracefully
        pass

    # Serialize Markdown with improved table rendering and embedded image links
    md_text: Optional[str] = None
    try:
        serializer = MarkdownDocSerializer(
            doc=dl_doc,  # type: ignore[arg-type]
            table_serializer=MarkdownTableSerializer(),
            picture_serializer=FileLinkPictureSerializer(pic_relmap),
            params=MarkdownParams(),
        )
        ser_out = serializer.serialize()
        md_text = ser_out.text
    except Exception as e:
        raise PdfConversionError(f"Markdown serialization failed: {e}") from e

    # Fallback to default export if advanced serializer not available
    if md_text is None:
        try:
            md_text = dl_doc.export_to_markdown()  # type: ignore[attr-defined]
        except Exception as e:
            raise PdfConversionError(f"Failed to export markdown: {e}") from e

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
        output_dir=None,  # e.g., "/Users/me/Documents/converted"
        output_format="markdown",
    )
