"""
Data models for vocabulary cards generated from lemmatizer output.
"""
from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from anki.anki_sync.anki_connect import anki_id


class Translation(BaseModel):
    """Translation to a target language."""
    word_translation: str = Field(..., description="Translation of the word/phrase")
    context_translation: Optional[str] = Field(None, description="Translation of the full context sentence")
    common_translations: List[str] = Field(default_factory=list, description="List of common translations (up to 2)")


class VocabularyCard(BaseModel):
    """A vocabulary card generated from a lemma or phrasal verb."""

    # Identity field for Anki deduplication
    card_id: str = Field(..., description="Unique ID for the card (lemma/part_of_speech or phrasal_verb)")

    # English fields
    english_lemma: str = Field(..., description="The lemma (base form) of the word")
    english_original_word: str = Field(..., description="The original word as it appeared in text")
    english_context: str = Field(..., description="Context sentence with the word highlighted in <b> tags")
    part_of_speech: str = Field(..., description="Part of speech (e.g., 'noun', 'verb', 'phrasal verb')")

    # Russian translations
    russian_word_translation: str = Field(..., description="Russian translation of the word in context")
    russian_context_translation: Optional[str] = Field(None, description="Russian translation of the entire sentence")
    russian_common_translations: str = Field(default="", description="Common Russian translations, comma-separated")

    # Spanish translations
    spanish_word_translation: str = Field(..., description="Spanish translation of the word in context")
    spanish_common_translations: str = Field(default="", description="Common Spanish translations, comma-separated")


class VocabularyDeck(BaseModel):
    """A collection of vocabulary cards."""

    cards: List[VocabularyCard] = Field(..., description="List of vocabulary cards")


@dataclass
class VocabularyNote:
    """
    Dataclass representation of a vocabulary card for AnkiConnect sync.

    This matches the "Vocabulary Improved" note type structure from the Go project.
    """
    ID: str = anki_id()  # Identity field for deduplication
    PartOfSpeech: str = ""
    EnglishLemma: str = ""
    EnglishOriginalWord: str = ""
    EnglishContext: str = ""
    RussianLemmaUsageTranslation: str = ""
    SpanishLemmaUsageTranslation: str = ""
    RussianContextTranslation: str = ""
    RussianCommonTranslations: str = ""
    SpanishCommonTranslations: str = ""
    EnglishAudio: str = ""  # Will be populated with [sound:filename.mp3] format
    EnglishContextAudio: str = ""  # Will be populated with [sound:filename.mp3] format


def vocabulary_card_to_note(card: VocabularyCard) -> VocabularyNote:
    """Convert a VocabularyCard to a VocabularyNote for Anki sync."""
    return VocabularyNote(
        ID=card.card_id,
        PartOfSpeech=card.part_of_speech,
        EnglishLemma=card.english_lemma,
        EnglishOriginalWord=card.english_original_word,
        EnglishContext=card.english_context,
        RussianLemmaUsageTranslation=card.russian_word_translation,
        SpanishLemmaUsageTranslation=card.spanish_word_translation,
        RussianContextTranslation=card.russian_context_translation or "",
        RussianCommonTranslations=card.russian_common_translations,
        SpanishCommonTranslations=card.spanish_common_translations,
        EnglishAudio="",  # Audio will be added later if needed
        EnglishContextAudio="",
    )
