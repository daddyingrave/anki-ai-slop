"""
LangChain chains for generating vocabulary cards from lemmatizer output.
"""
from __future__ import annotations

import re
from typing import Dict, List, cast

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .models import VocabularyCard, VocabularyDeck, Translation
from .prompts import build_translation_prompts, build_word_translation_prompts
from ..common.llm import build_llm
from ..common.reliability import retry_invoke
from ..config_models import StepConfig


class ContextTranslationResponse(BaseModel):
    """Response model for word-in-context translation."""
    russian: Translation = Field(..., description="Russian translation")
    spanish: Translation = Field(..., description="Spanish translation")


class WordTranslationResponse(BaseModel):
    """Response model for general word translation."""
    russian: List[str] = Field(default_factory=list, description="Common Russian translations (up to 2)")
    spanish: List[str] = Field(default_factory=list, description="Common Spanish translations (up to 2)")


def highlight_word_in_context(word: str, context: str) -> str:
    """Highlight a word in context by wrapping it with <b> tags.

    Args:
        word: The word to highlight
        context: The sentence containing the word

    Returns:
        Context with the word wrapped in <b> tags
    """
    # Create a case-insensitive pattern that matches the word (with word boundaries)
    pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)

    # Replace the first occurrence with highlighted version
    highlighted = pattern.sub(lambda m: f"<b>{m.group()}</b>", context, count=1)

    return highlighted


def translate_word_in_context(
    word: str,
    original_word: str,
    context: str,
    part_of_speech: str,
    step: StepConfig,
) -> ContextTranslationResponse:
    """Translate a word considering its context.

    Args:
        word: The lemma (base form) to translate
        original_word: The word as it appears in the sentence
        context: The sentence containing the word
        part_of_speech: Part of speech tag
        step: Step configuration (model, temperature, retries, etc.)

    Returns:
        Translations for Russian and Spanish with context
    """
    llm = build_llm(model=step.model, temperature=step.temperature)

    prompts = build_translation_prompts()
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts["system"]),
        ("human", prompts["human"]),
    ])

    chain = prompt | llm.with_structured_output(ContextTranslationResponse)
    result = cast(ContextTranslationResponse, retry_invoke(
        chain,
        {
            "word": word,
            "original_word": original_word,
            "context": context,
            "part_of_speech": part_of_speech,
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))

    if result is None:
        raise RuntimeError(f"LLM did not return translation for word '{word}'")

    return result


def translate_word_general(
    word: str,
    part_of_speech: str,
    step: StepConfig,
) -> WordTranslationResponse:
    """Get general translations for a word (not context-specific).

    Args:
        word: The word to translate
        part_of_speech: Part of speech tag
        step: Step configuration

    Returns:
        Common translations for Russian and Spanish
    """
    # Skip translation for proper nouns
    if part_of_speech.lower() == "proper noun":
        return WordTranslationResponse(russian=[], spanish=[])

    llm = build_llm(model=step.model, temperature=step.temperature)

    prompts = build_word_translation_prompts()
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts["system"]),
        ("human", prompts["human"]),
    ])

    chain = prompt | llm.with_structured_output(WordTranslationResponse)
    result = cast(WordTranslationResponse, retry_invoke(
        chain,
        {
            "word": word,
            "part_of_speech": part_of_speech,
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))

    if result is None:
        raise RuntimeError(f"LLM did not return general translation for word '{word}'")

    return result


def generate_vocabulary_card(
    lemma: str,
    original_word: str,
    context: str,
    part_of_speech: str,
    step: StepConfig,
) -> VocabularyCard:
    """Generate a complete vocabulary card for a lemma.

    This performs two translation steps:
    1. Context-aware translation of the word in the given sentence
    2. General translations of the word (common meanings)

    Args:
        lemma: The base form of the word
        original_word: The word as it appears in text
        context: The sentence containing the word
        part_of_speech: Part of speech tag
        step: Step configuration

    Returns:
        A complete VocabularyCard with all translations
    """
    # Step 1: Get context-aware translation
    ctx_translation = translate_word_in_context(
        word=lemma,
        original_word=original_word,
        context=context,
        part_of_speech=part_of_speech,
        step=step,
    )

    # Step 2: Get general translations
    general_translation = translate_word_general(
        word=lemma,
        part_of_speech=part_of_speech,
        step=step,
    )

    # Create card ID
    card_id = f"{lemma}/{part_of_speech}"

    # Highlight the word in context
    highlighted_context = highlight_word_in_context(original_word, context)

    # Build the vocabulary card
    card = VocabularyCard(
        card_id=card_id,
        english_lemma=lemma,
        english_original_word=original_word,
        english_context=highlighted_context,
        part_of_speech=part_of_speech,
        russian_word_translation=ctx_translation.russian.word_translation,
        russian_context_translation=ctx_translation.russian.context_translation,
        russian_common_translations=", ".join(general_translation.russian),
        spanish_word_translation=ctx_translation.spanish.word_translation,
        spanish_common_translations=", ".join(general_translation.spanish),
    )

    return card


def process_lemma_batch(
    lemma_entries: Dict[str, List[Dict[str, str]]],
    step: StepConfig,
) -> VocabularyDeck:
    """Process a batch of lemma entries and generate vocabulary cards.

    Args:
        lemma_entries: Dictionary mapping lemmas to their entries
            Each entry should have: sentence, original_word, part_of_speech
        step: Step configuration

    Returns:
        A VocabularyDeck containing cards for all processed lemmas
    """
    cards: List[VocabularyCard] = []

    for lemma, entries in lemma_entries.items():
        if not entries:
            continue

        # Use the first entry for each lemma
        first_entry = entries[0]

        try:
            card = generate_vocabulary_card(
                lemma=lemma,
                original_word=first_entry["original_word"],
                context=first_entry["sentence"],
                part_of_speech=first_entry["part_of_speech"],
                step=step,
            )
            cards.append(card)
        except Exception as e:
            print(f"Error generating card for lemma '{lemma}': {e}")
            continue

    return VocabularyDeck(cards=cards)


__all__ = [
    "ContextTranslationResponse",
    "WordTranslationResponse",
    "highlight_word_in_context",
    "translate_word_in_context",
    "translate_word_general",
    "generate_vocabulary_card",
    "process_lemma_batch",
]
