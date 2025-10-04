"""
LangChain chains for generating vocabulary cards from lemmatizer output.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, cast

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from anki.pipelines.vocabulary.models import VocabularyCard, VocabularyDeck, Translation
from anki.pipelines.vocabulary.prompts import (
    build_batch_translation_prompts,
    build_batch_word_translation_prompts,
)
from anki.common.llm import build_llm
from anki.common.reliability import retry_invoke
from anki.config_models import StepConfig
from anki.lemmatizer import LemmaExtractor, LanguageMnemonic, ModelType


class SentenceContext(BaseModel):
    """Context information for a sentence."""
    sentence: str = Field(..., description="The current sentence")
    previous_sentence: str | None = Field(None, description="Previous sentence for context")
    next_sentence: str | None = Field(None, description="Next sentence for context")


class WordInSentence(BaseModel):
    """Information about a word occurrence in a specific sentence."""
    lemma: str = Field(..., description="The base form of the word")
    original_word: str = Field(..., description="Word as it appears in the sentence")
    part_of_speech: str = Field(..., description="Part of speech tag")
    is_phrasal_verb: bool = Field(default=False, description="Whether this is a phrasal verb")


class SentenceWithWords(BaseModel):
    """A sentence with all words that need translation."""
    sentence: str = Field(..., description="The sentence text")
    words: List[WordInSentence] = Field(..., description="Words to translate in this sentence")
    context: SentenceContext | None = Field(None, description="Context sentences")


def transform_to_sentence_map(
        lemma_map: Dict[str, List[Dict[str, str]]],
        phrasal_verb_map: Dict[str, List[Dict[str, str]]],
        text: str,
) -> List[SentenceWithWords]:
    """Transform lemma_map and phrasal_verb_map into a sentence-grouped structure.

    This enables batch processing: translate all words in a sentence together.

    Args:
        lemma_map: Maps lemma -> list of occurrences with {sentence, original_word, part_of_speech}
        phrasal_verb_map: Maps phrasal_verb -> list of occurrences with {sentence, original_text, ...}
        text: Full text to extract sentence context

    Returns:
        List of SentenceWithWords, each containing all words to translate in that sentence
    """
    # Build reverse mapping: sentence -> list of words
    sentence_to_words: Dict[str, List[WordInSentence]] = defaultdict(list)

    # Process regular lemmas
    for lemma, entries in lemma_map.items():
        for entry in entries:
            sentence = entry["sentence"]
            word_info = WordInSentence(
                lemma=lemma,
                original_word=entry["original_word"],
                part_of_speech=entry["part_of_speech"],
                is_phrasal_verb=False,
            )
            # Avoid duplicates
            if not any(w.lemma == word_info.lemma and w.original_word == word_info.original_word
                       for w in sentence_to_words[sentence]):
                sentence_to_words[sentence].append(word_info)

    # Process phrasal verbs
    for pv_key, entries in phrasal_verb_map.items():
        for entry in entries:
            sentence = entry["sentence"]
            word_info = WordInSentence(
                lemma=pv_key,
                original_word=entry["original_text"],
                part_of_speech="phrasal verb",
                is_phrasal_verb=True,
            )
            # Avoid duplicates
            if not any(w.lemma == word_info.lemma and w.original_word == word_info.original_word
                       for w in sentence_to_words[sentence]):
                sentence_to_words[sentence].append(word_info)

    # Split text into sentences for context extraction
    sentences = [s.strip() for s in text.split('\n') if s.strip()]

    # Build sentence index for fast lookup
    sentence_index = {sent: idx for idx, sent in enumerate(sentences)}

    # Build result with context
    result: List[SentenceWithWords] = []
    for sentence, words in sentence_to_words.items():
        # Find context sentences
        context = None
        if sentence in sentence_index:
            idx = sentence_index[sentence]
            prev_sent = sentences[idx - 1] if idx > 0 else None
            next_sent = sentences[idx + 1] if idx < len(sentences) - 1 else None
            context = SentenceContext(
                sentence=sentence,
                previous_sentence=prev_sent,
                next_sentence=next_sent,
            )

        result.append(SentenceWithWords(
            sentence=sentence,
            words=words,
            context=context,
        ))

    return result


class ContextTranslationResponse(BaseModel):
    """Response model for word-in-context translation."""
    russian: Translation = Field(..., description="Russian translation")
    spanish: Translation = Field(..., description="Spanish translation")


class WordTranslationResponse(BaseModel):
    """Response model for general word translation."""
    russian: List[str] = Field(default_factory=list, description="Common Russian translations (up to 2)")
    spanish: List[str] = Field(default_factory=list, description="Common Spanish translations (up to 2)")


class SingleWordGeneralTranslation(BaseModel):
    """Translation for a single word (general meanings, not context-specific)."""
    lemma: str = Field(..., description="The word being translated")
    russian: List[str] = Field(default_factory=list, description="Common Russian translations (up to 2)")
    spanish: List[str] = Field(default_factory=list, description="Common Spanish translations (up to 2)")


class BatchWordTranslationResponse(BaseModel):
    """Response model for batch general word translation.

    Contains a list of translations for multiple words.
    """
    translations: List[SingleWordGeneralTranslation] = Field(
        ...,
        description="List of translations for each word in the batch"
    )


# Batch translation models
class BatchWordContextTranslation(BaseModel):
    """Translation for a single word in a sentence (batch context)."""
    lemma: str = Field(..., description="The base form of the word being translated")
    is_phrasal_verb: bool = Field(..., description="Whether this is a phrasal verb")
    russian_word: str = Field(..., description="Russian translation of the word in this context")
    spanish_word: str = Field(..., description="Spanish translation of the word in this context")
    russian_sentence: str | None = Field(None, description="Russian translation of the full sentence (only provided once per sentence)")


class BatchSentenceTranslationResponse(BaseModel):
    """Batch translation response for all words in a single sentence."""
    words: List[BatchWordContextTranslation] = Field(..., description="Translations for each word in the sentence")


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


def batch_translate_words_general(
        words: Dict[str, WordInSentence],
        step: StepConfig,
) -> BatchWordTranslationResponse:
    """Translate general meanings of multiple words in a single LLM call.

    This is more efficient than calling translate_word_general for each word separately.

    Args:
        words: Dictionary mapping lemma -> WordInSentence info
        step: Step configuration

    Returns:
        Batch response with general translations for all words
    """
    llm = build_llm(model=step.model, temperature=step.temperature)

    prompts = build_batch_word_translation_prompts()

    # Format words list with part of speech info
    words_list = ""
    for lemma, word_info in words.items():
        words_list += f"- \"{lemma}\" ({word_info.part_of_speech})\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts["system"]),
        ("human", prompts["human"]),
    ])

    chain = prompt | llm.with_structured_output(BatchWordTranslationResponse)
    result = cast(BatchWordTranslationResponse, retry_invoke(
        chain,
        {
            "words_list": words_list,
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))

    if result is None:
        raise RuntimeError("LLM did not return batch general translations")

    return result


def batch_translate_sentence_words(
        sentence_with_words: SentenceWithWords,
        step: StepConfig,
) -> BatchSentenceTranslationResponse:
    """Translate multiple words from the same sentence in a single LLM call.

    This is more efficient than calling translate_word_in_context for each word separately.

    Args:
        sentence_with_words: Sentence with all words to translate
        step: Step configuration

    Returns:
        Batch translation response with translations for all words
    """
    llm = build_llm(model=step.model, temperature=step.temperature)

    prompts = build_batch_translation_prompts()

    # Format context info
    context_info = ""
    if sentence_with_words.context:
        ctx = sentence_with_words.context
        if ctx.previous_sentence:
            context_info += f"Previous sentence: {ctx.previous_sentence}\n"
        context_info += f"Current sentence: {ctx.sentence}\n"
        if ctx.next_sentence:
            context_info += f"Next sentence: {ctx.next_sentence}\n"
    else:
        context_info = f"Sentence: {sentence_with_words.sentence}"

    # Format words list
    words_list = ""
    for idx, word in enumerate(sentence_with_words.words, 1):
        words_list += f"{idx}. Lemma: {word.lemma}\n"
        words_list += f"   As it appears: {word.original_word}\n"
        words_list += f"   Part of speech: {word.part_of_speech}\n"
        if word.is_phrasal_verb:
            words_list += f"   NOTE: This is a phrasal verb\n"
        words_list += "\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts["system"]),
        ("human", prompts["human"]),
    ])

    chain = prompt | llm.with_structured_output(BatchSentenceTranslationResponse)
    result = cast(BatchSentenceTranslationResponse, retry_invoke(
        chain,
        {
            "sentence": sentence_with_words.sentence,
            "context_info": context_info,
            "words_list": words_list,
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))

    if result is None:
        raise RuntimeError(f"LLM did not return batch translation for sentence: {sentence_with_words.sentence[:50]}...")

    return result


def build_vocabulary_pipeline(
        input_file: str,
        language: str,
        model_type: str,
        phrasal_verbs_file: str | None,
        translate_step: StepConfig,
) -> List[VocabularyCard]:
    """Complete pipeline to process a text file and generate vocabulary cards.

    This pipeline:
    1. Extracts lemmas using the lemmatizer
    2. Generates vocabulary cards with translations
    3. Returns a list of VocabularyCard objects

    Args:
        input_file: Path to the input text file
        language: Language code (e.g., "EN")
        model_type: Model type ("EFFICIENT", "ACCURATE", "TRANSFORMER")
        phrasal_verbs_file: Optional path to phrasal verbs CSV file
        translate_step: Step configuration for translation

    Returns:
        List of VocabularyCard objects
    """
    # Step 1: Extract lemmas using the lemmatizer
    print(f"Processing file with lemmatizer: {input_file}")
    print(f"Language: {language}, Model: {model_type}")

    try:
        lang_enum = LanguageMnemonic(language)
    except ValueError:
        raise ValueError(f"Invalid language: {language}")

    try:
        model_type_map = {
            "EFFICIENT": ModelType.EFFICIENT,
            "ACCURATE": ModelType.ACCURATE,
            "TRANSFORMER": ModelType.TRANSFORMER,
        }
        model_enum = model_type_map[model_type.upper()]
    except KeyError:
        raise ValueError(f"Invalid model_type: {model_type}")

    extractor = LemmaExtractor(lang_enum, model_enum)
    lemma_map, phrasal_verb_map, text = extractor.process_file(
        input_file,
        phrasal_verbs_file
    )

    print(f"Extracted {len(lemma_map)} lemmas and {len(phrasal_verb_map)} phrasal verbs")

    # Step 2: Group words by sentence for batch translation
    print("Grouping words by sentence for batch translation...")
    sentence_groups = transform_to_sentence_map(lemma_map, phrasal_verb_map, text)
    print(f"Grouped into {len(sentence_groups)} unique sentences")

    # Step 3: Batch translate words by sentence (ONE LLM call per sentence for all words)
    print("Batch translating words in context...")
    sentence_translations: Dict[str, BatchSentenceTranslationResponse] = {}

    for idx, sentence_group in enumerate(sentence_groups, 1):
        try:
            # ONE LLM call translates ALL words in this sentence
            batch_result = batch_translate_sentence_words(sentence_group, translate_step)
            sentence_translations[sentence_group.sentence] = batch_result

            if idx % 5 == 0 or idx == len(sentence_groups):
                print(f"  Progress: {idx}/{len(sentence_groups)} sentences translated")
        except Exception as e:
            print(f"  Error translating sentence: {e}")
            continue

    # Step 4: Collect unique lemmas for general translation
    all_unique_lemmas: Dict[str, WordInSentence] = {}
    for sentence_group in sentence_groups:
        for word in sentence_group.words:
            if word.lemma not in all_unique_lemmas:
                all_unique_lemmas[word.lemma] = word

    # Step 5: Batch translate general meanings (ALL lemmas in ONE LLM call)
    print(f"Batch translating {len(all_unique_lemmas)} unique lemmas (general meanings)...")

    batch_result = batch_translate_words_general(all_unique_lemmas, translate_step)

    # Convert list of translations to dict: lemma -> WordTranslationResponse
    general_translations: Dict[str, WordTranslationResponse] = {}
    for trans in batch_result.translations:
        general_translations[trans.lemma] = WordTranslationResponse(
            russian=trans.russian,
            spanish=trans.spanish
        )

    print(f"  Success: Translated all {len(general_translations)} lemmas in 1 LLM call")

    # Debug: Show sample translations
    if general_translations:
        sample_lemma = list(general_translations.keys())[0]
        sample_trans = general_translations[sample_lemma]
        print(f"  Sample: '{sample_lemma}' -> RU: {sample_trans.russian}, ES: {sample_trans.spanish}")
    else:
        print(f"  WARNING: Batch translation returned empty list!")

    # Step 6: Build vocabulary cards from batch results
    print("Constructing vocabulary cards...")
    cards: List[VocabularyCard] = []

    # Build lookup: (sentence, lemma) -> context translation
    context_lookup: Dict[tuple[str, str], BatchWordContextTranslation] = {}
    # Build sentence translation lookup: sentence -> russian_sentence
    sentence_translation_lookup: Dict[str, str] = {}

    for sentence, batch_result in sentence_translations.items():
        # Extract the sentence translation from the first word that has it
        for word_trans in batch_result.words:
            context_lookup[(sentence, word_trans.lemma)] = word_trans
            # Cache the sentence translation (it's in the first word)
            if word_trans.russian_sentence and sentence not in sentence_translation_lookup:
                sentence_translation_lookup[sentence] = word_trans.russian_sentence

    # Create one card per unique lemma (use first occurrence)
    processed_lemmas = set()

    for sentence_group in sentence_groups:
        sentence = sentence_group.sentence
        # Get the cached sentence translation for this sentence
        russian_sentence = sentence_translation_lookup.get(sentence)

        for word in sentence_group.words:
            # Only create card for first occurrence of each lemma
            if word.lemma in processed_lemmas:
                continue

            processed_lemmas.add(word.lemma)

            # Get context translation from batch result
            context_trans = context_lookup.get((sentence, word.lemma))
            if not context_trans:
                print(f"  Warning: No context translation for '{word.lemma}'")
                continue

            # Get general translation
            general_trans = general_translations.get(word.lemma)
            if not general_trans:
                general_trans = WordTranslationResponse(russian=[], spanish=[])

            # Create card
            card_id = f"{word.lemma}/{word.part_of_speech}"
            highlighted_context = highlight_word_in_context(word.original_word, sentence)

            card = VocabularyCard(
                card_id=card_id,
                english_lemma=word.lemma,
                english_original_word=word.original_word,
                english_context=highlighted_context,
                part_of_speech=word.part_of_speech,
                russian_word_translation=context_trans.russian_word,
                russian_context_translation=russian_sentence,  # Use cached sentence translation
                russian_common_translations=", ".join(general_trans.russian),
                spanish_word_translation=context_trans.spanish_word,
                spanish_common_translations=", ".join(general_trans.spanish),
            )
            cards.append(card)

    print(f"Generated {len(cards)} vocabulary cards total")
    print(f"Context translations: {len(sentence_groups)} LLM calls (one per sentence)")
    print(f"General translations: 1 LLM call (batch)")
    total_llm_calls = len(sentence_groups) + 1  # +1 for batch general translation
    old_approach_calls = 2 * (len(lemma_map) + len(phrasal_verb_map))
    if old_approach_calls > 0:
        savings = 100 * (1 - total_llm_calls / old_approach_calls)
        print(f"Efficiency: {savings:.1f}% fewer LLM calls vs old approach")

    return cards


__all__ = [
    # Data models
    "SentenceContext",
    "WordInSentence",
    "SentenceWithWords",
    "ContextTranslationResponse",
    "WordTranslationResponse",
    "SingleWordGeneralTranslation",
    "BatchWordTranslationResponse",
    "BatchWordContextTranslation",
    "BatchSentenceTranslationResponse",
    # Helper functions
    "transform_to_sentence_map",
    "highlight_word_in_context",
    # Batch translation
    "batch_translate_sentence_words",
    "batch_translate_words_general",
    # Pipeline
    "build_vocabulary_pipeline",
]
