"""
LangChain chains for generating vocabulary cards from lemmatizer output.
"""
from __future__ import annotations

import re
from typing import Dict, List, cast

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from anki.common.llm import build_llm
from anki.common.reliability import retry_invoke
from anki.config_models import StepConfig
from anki.lemmatizer import (
    LemmaExtractor,
    LanguageMnemonic,
    ModelType,
    SentenceWithWords,
    WordInSentence,
)
from anki.pipelines.vocabulary.models import VocabularyCard, Translation
from anki.pipelines.vocabulary.prompts import (
    build_ctx_translation_prompts,
    build_general_translation_prompts,
    build_ctx_review_prompts,
    build_general_review_prompts,
)


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


class GeneralTranslationResponse(BaseModel):
    """Response model for general word translation.

    Contains a list of translations for multiple words.
    """
    translations: List[SingleWordGeneralTranslation] = Field(
        ...,
        description="List of translations for each word"
    )


class WordContextTranslation(BaseModel):
    """Translation for a single word in a sentence context."""
    lemma: str = Field(..., description="The base form of the word being translated")
    is_phrasal_verb: bool = Field(..., description="Whether this is a phrasal verb")
    russian_word: str = Field(..., description="Russian translation of the word in this context")
    spanish_word: str = Field(..., description="Spanish translation of the word in this context")
    russian_sentence: str = Field(None,
                                  description="Russian translation of the full sentence (only provided once per sentence)")


class CtxTranslationResponse(BaseModel):
    """Translation response for all words in a single sentence."""
    words: List[WordContextTranslation] = Field(..., description="Translations for each word in the sentence")


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


def translate_words_general(
        words: Dict[str, WordInSentence],
        step: StepConfig,
) -> GeneralTranslationResponse:
    """Translate general meanings of multiple words in a single LLM call.

    Args:
        words: Dictionary mapping lemma -> WordInSentence info
        step: Step configuration

    Returns:
        Response with general translations for all words
    """
    llm = build_llm(model=step.model, temperature=step.temperature)

    prompts = build_general_translation_prompts()

    # Format words list with part of speech info
    words_list = ""
    for lemma, word_info in words.items():
        words_list += f"- \"{lemma}\" ({word_info.part_of_speech})\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts["system"]),
        ("human", prompts["human"]),
    ])

    chain = prompt | llm.with_structured_output(GeneralTranslationResponse)
    result = cast(GeneralTranslationResponse, retry_invoke(
        chain,
        {
            "words_list": words_list,
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))

    if result is None:
        raise RuntimeError("LLM did not return general translations")

    return result


def translate_words_ctx(
        sentence_with_words: SentenceWithWords,
        step: StepConfig,
) -> CtxTranslationResponse:
    """Translate multiple words from a sentence with context in a single LLM call.

    Args:
        sentence_with_words: Sentence with all words to translate
        step: Step configuration

    Returns:
        Translation response with translations for all words in context
    """
    llm = build_llm(model=step.model, temperature=step.temperature)

    prompts = build_ctx_translation_prompts()

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

    chain = prompt | llm.with_structured_output(CtxTranslationResponse)
    result = cast(CtxTranslationResponse, retry_invoke(
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
        raise RuntimeError(f"LLM did not return translation for sentence: {sentence_with_words.sentence[:50]}...")

    russian_sentence = ""
    for word in result.words:
        if word.russian_sentence:
            russian_sentence = word.russian_sentence
            break

    if russian_sentence:
        for word in result.words:
            if not word.russian_sentence:
                word.russian_sentence = russian_sentence

    return result


def review_translation_ctx(
        sentence_with_words: SentenceWithWords,
        translations: CtxTranslationResponse,
        step: StepConfig,
) -> CtxTranslationResponse:
    """Review and fix context translations for a sentence.

    Args:
        sentence_with_words: Sentence with all words
        translations: Current translations to review
        step: Step configuration

    Returns:
        Reviewed and corrected translations
    """
    import json

    llm = build_llm(model=step.model, temperature=step.temperature)
    prompts = build_ctx_review_prompts()

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

    # Serialize current translations
    translations_json = json.dumps(translations.model_dump(), indent=2, ensure_ascii=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts["system"]),
        ("human", prompts["human"]),
    ])

    chain = prompt | llm.with_structured_output(CtxTranslationResponse)
    result = cast(CtxTranslationResponse, retry_invoke(
        chain,
        {
            "sentence": sentence_with_words.sentence,
            "context_info": context_info,
            "words_list": words_list,
            "translations_json": translations_json,
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))

    if result is None:
        raise RuntimeError(f"LLM did not return review for sentence: {sentence_with_words.sentence[:50]}...")

    russian_sentence = ""
    for word in result.words:
        if word.russian_sentence:
            russian_sentence = word.russian_sentence
            break

    # Copy to all words that don't have it set
    if russian_sentence:
        for word in result.words:
            if not word.russian_sentence:
                word.russian_sentence = russian_sentence

    return result


def review_general_translations(
        words: Dict[str, WordInSentence],
        translations: GeneralTranslationResponse,
        step: StepConfig,
) -> GeneralTranslationResponse:
    """Review and fix general translations.

    Args:
        words: Dictionary mapping lemma -> WordInSentence info
        translations: Current translations to review
        step: Step configuration

    Returns:
        Reviewed and corrected translations
    """
    import json

    llm = build_llm(model=step.model, temperature=step.temperature)
    prompts = build_general_review_prompts()

    # Format words list with part of speech info
    words_list = ""
    for lemma, word_info in words.items():
        words_list += f"- \"{lemma}\" ({word_info.part_of_speech})\n"

    # Serialize current translations
    translations_json = json.dumps(translations.model_dump(), indent=2, ensure_ascii=False)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts["system"]),
        ("human", prompts["human"]),
    ])

    chain = prompt | llm.with_structured_output(GeneralTranslationResponse)
    result = cast(GeneralTranslationResponse, retry_invoke(
        chain,
        {
            "words_list": words_list,
            "translations_json": translations_json,
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))

    if result is None:
        raise RuntimeError("LLM did not return review for general translations")

    return result


def build_vocabulary_pipeline(
        input_file: str,
        language: str,
        model_type: str,
        phrasal_verbs_file: str | None,
        translate_step: StepConfig,
        review_step: StepConfig,
) -> List[VocabularyCard]:
    """Complete pipeline to process a text file and generate vocabulary cards.

    This pipeline:
    1. Extracts lemmas using the lemmatizer
    2. Translates words in context and general meanings
    3. Reviews and fixes translations
    4. Returns a list of VocabularyCard objects

    Args:
        input_file: Path to the input text file
        language: Language code (e.g., "EN")
        model_type: Model type ("EFFICIENT", "ACCURATE", "TRANSFORMER")
        phrasal_verbs_file: Optional path to phrasal verbs CSV file
        translate_step: Step configuration for translation
        review_step: Step configuration for review

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

    # Step 2: Extract and group words by sentence
    print("Extracting lemmas and grouping by sentence...")
    sentences = extractor.process_file(input_file, phrasal_verbs_file)

    # Step 3: Translate and review words by sentence (context + general meanings)
    print("Translating and reviewing words in context and general meanings...")
    ctx_translations: Dict[str, CtxTranslationResponse] = {}
    general_translations: Dict[str, WordTranslationResponse] = {}

    for idx, sentence_group in enumerate(sentences, 1):
        try:
            # Translate all words in this sentence with context
            ctx_result = translate_words_ctx(sentence_group, translate_step)

            # Review context translation
            ctx_result = review_translation_ctx(sentence_group, ctx_result, review_step)
            ctx_translations[sentence_group.sentence] = ctx_result

            # Collect lemmas from current sentence that we haven't translated yet
            sentence_lemmas: Dict[str, WordInSentence] = {}
            for word in sentence_group.words:
                if word.lemma not in general_translations:
                    sentence_lemmas[word.lemma] = word

            # Translate general meanings for lemmas in this sentence
            if sentence_lemmas:
                general_result = translate_words_general(sentence_lemmas, translate_step)

                # Review general translation
                general_result = review_general_translations(sentence_lemmas, general_result, review_step)

                for trans in general_result.translations:
                    general_translations[trans.lemma] = WordTranslationResponse(
                        russian=trans.russian,
                        spanish=trans.spanish
                    )

            if idx % 5 == 0 or idx == len(sentences):
                print(f"  Progress: {idx}/{len(sentences)} sentences translated and reviewed")
        except Exception as e:
            print(f"  Error translating/reviewing sentence: {e}")
            continue

    # Step 4: Build vocabulary cards from translation results
    print("Constructing vocabulary cards...")
    cards: List[VocabularyCard] = []

    # Build lookup: (sentence, lemma) -> context translation
    context_lookup: Dict[tuple[str, str], WordContextTranslation] = {}
    for sentence_text, ctx_result in ctx_translations.items():
        # Extract the sentence translation from the first word that has it
        for word_trans in ctx_result.words:
            context_lookup[(sentence_text, word_trans.lemma)] = word_trans

    # Create one card per unique lemma (use first occurrence)
    processed_lemmas = set()

    for sentence in sentences:
        sentence_raw = sentence.sentence
        # Get the cached sentence translation for this sentence

        for word in sentence.words:
            # Only create card for first occurrence of each lemma
            if word.lemma in processed_lemmas:
                continue

            processed_lemmas.add(word.lemma)

            # Get context translation from batch result
            context_trans = context_lookup.get((sentence_raw, word.lemma))
            if not context_trans:
                print(f"  Warning: No context translation for '{word.lemma}'")
                continue

            # Get general translation
            general_trans = general_translations.get(word.lemma)
            if not general_trans:
                general_trans = WordTranslationResponse(russian=[], spanish=[])

            # Create card
            card_id = f"{word.lemma}/{word.part_of_speech}"

            card = VocabularyCard(
                card_id=card_id,
                english_lemma=word.lemma,
                english_original_word=word.original_word,
                english_context=highlight_word_in_context(word.original_word, sentence_raw),
                part_of_speech=word.part_of_speech,
                russian_word_translation=context_trans.russian_word,
                russian_context_translation=highlight_word_in_context(context_trans.russian_word,
                                                                      context_trans.russian_sentence),
                russian_common_translations=", ".join(general_trans.russian),
                spanish_word_translation=context_trans.spanish_word,
                spanish_common_translations=", ".join(general_trans.spanish),
            )
            cards.append(card)

    print(f"Generated {len(cards)} vocabulary cards total")

    return cards
