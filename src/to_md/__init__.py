"""
Docling-based document processing utilities.

This package provides IDE-run helpers to convert PDF documents to Markdown
using the `docling` library. It is not intended to be used as a CLI; instead,
open the `run.py` file in your IDE, set the variables, and run it.
"""

__all__ = [
    "convert_pdf",
]

from .pdf_to_markdown import convert_pdf
