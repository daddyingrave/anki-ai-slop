"""
Lemmatizer to Anki pipeline.

This package processes lemmatized text and generates Anki vocabulary cards
with translations to multiple languages.
"""

from .models import VocabularyCard, VocabularyDeck
from .chains import generate_vocabulary_card, process_lemma_batch

__all__ = [
    "VocabularyCard",
    "VocabularyDeck",
    "generate_vocabulary_card",
    "process_lemma_batch",
]
