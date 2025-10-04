"""
Lemmatizer to Anki pipeline.

This package processes lemmatized text and generates Anki vocabulary cards
with translations to multiple languages using batch translation for efficiency.
"""

from .models import VocabularyCard, VocabularyDeck
from .chains import build_vocabulary_pipeline
