"""
Subtitle processing module for parsing SRT files.

This package provides functionality to read and process subtitle files,
extracting text, metadata, and song lyrics.
"""

from .reader import Reader, Processed, new_reader

__all__ = [
    "Reader",
    "Processed",
    "new_reader",
]
