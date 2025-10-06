from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class MistralOcrConfig(BaseModel):
    """Configuration for the Mistral OCR pipeline (PDF → Markdown).

    - input_file_path: path to the PDF file to convert
    - output_dir: directory where markdown and media will be saved
    - output_format: output format (currently only "markdown" supported)
    - override_existing: whether to overwrite existing output files
    - page_start: optional start page (0-indexed, inclusive). If None, starts from first page.
    - page_end: optional end page (0-indexed, inclusive). If None, processes to last page.
    """

    input_file_path: Path = Field(..., description="Path to the PDF file to convert")
    output_dir: Path = Field(..., description="Directory where markdown and media will be saved")
    output_format: str = Field(default="markdown", description="Output format (currently only 'markdown' supported)")
    override_existing: bool = Field(default=True, description="Whether to overwrite existing output files")
    page_start: int | None = Field(default=None, description="Optional start page (0-indexed, inclusive). If None, starts from first page.")
    page_end: int | None = Field(default=None, description="Optional end page (0-indexed, inclusive). If None, processes to last page.")
