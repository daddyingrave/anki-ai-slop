"""
Lemmatizer package for text lemmatization and phrasal verb extraction.

This package provides functionality for:
- Extracting lemmas from text
- Processing phrasal verbs
- Text normalization and processing
"""

from .lemma_extractor import (
    LemmaExtractor,
    LanguageMnemonic,
    ModelType,
    LemmaEntry,
    PhrasalVerbEntry,
)

__all__ = [
    "LemmaExtractor",
    "LanguageMnemonic",
    "ModelType",
    "LemmaEntry",
    "PhrasalVerbEntry",
]
