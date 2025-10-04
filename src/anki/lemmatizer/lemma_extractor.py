"""
Lemma and Phrasal Verb Extractor

This module provides a class for extracting lemmas and phrasal verbs from text files.
It supports different source languages and model types.
"""

import os.path
from collections import defaultdict
from enum import Enum, auto
from typing import Dict, List, TypedDict, Optional, Tuple

import spacy
from spacy.matcher import Matcher

from .phrasal_verb_processor import (
    read_phrasal_verbs,
    normalize_phrasal_verbs,
    split_phrasal_verbs,
    VerbParticlePair
)
from .text_processor import exit_with_error, process_text_file


class LanguageMnemonic(str, Enum):
    """Predefined constants for language mnemonics."""
    EN = "EN"
    DE = "DE"
    FR = "FR"
    ES = "ES"
    IT = "IT"
    RU = "RU"
    JA = "JA"
    ZH = "ZH"


class ModelType(Enum):
    """Model type selection for language processing."""
    EFFICIENT = auto()
    ACCURATE = auto()
    TRANSFORMER = auto()


class LemmaEntry(TypedDict):
    """Structure for storing information about a lemma occurrence in a sentence."""
    sentence: str
    original_word: str
    part_of_speech: str


class PhrasalVerbEntry(TypedDict):
    """Structure for storing information about a phrasal verb occurrence in a sentence."""
    sentence: str
    original_text: str
    verb: str
    particle: str


class LemmaExtractor:
    """Class for extracting lemmas and phrasal verbs from text files."""

    # Languages that have phrasal verbs as a concept
    _LANGUAGES_WITH_PHRASAL_VERBS = {LanguageMnemonic.EN, LanguageMnemonic.DE}

    # Mapping from language mnemonics to spaCy models
    _LANGUAGE_TO_MODEL = {
        LanguageMnemonic.EN: {
            ModelType.EFFICIENT: "en_core_web_sm",
            ModelType.ACCURATE: "en_core_web_md",
            ModelType.TRANSFORMER: "en_core_web_trf"
        },
        LanguageMnemonic.DE: {
            ModelType.EFFICIENT: "de_core_news_sm",
            ModelType.ACCURATE: "de_core_news_md",
            ModelType.TRANSFORMER: "de_dep_news_trf"
        },
        LanguageMnemonic.FR: {
            ModelType.EFFICIENT: "fr_core_news_sm",
            ModelType.ACCURATE: "fr_core_news_md",
            ModelType.TRANSFORMER: "fr_dep_news_trf"
        },
        LanguageMnemonic.ES: {
            ModelType.EFFICIENT: "es_core_news_sm",
            ModelType.ACCURATE: "es_core_news_md",
            ModelType.TRANSFORMER: "es_dep_news_trf"
        },
    }

    def __init__(self, lang: LanguageMnemonic, mod_type: ModelType = ModelType.ACCURATE):
        """
        Initialize the LemmaExtractor with a specific language and model type.

        Args:
            lang (LanguageMnemonic): The source language mnemonic
            mod_type (ModelType, optional): The model type to use

        Raises:
            ValueError: If the language is not supported
        """
        self.language = lang
        self.model_type = mod_type

        # Check if the language is supported
        if lang not in self._LANGUAGE_TO_MODEL:
            langs = ", ".join([l.value for l in self._LANGUAGE_TO_MODEL.keys()])
            exit_with_error(f"Error: Language '{lang}' is not supported. Supported languages: {langs}")

        # Load spaCy model
        model_name = self._LANGUAGE_TO_MODEL[lang][mod_type]

        print(f"Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name)
        print(f"Model loaded successfully")

    @staticmethod
    def get_human_readable_pos(pos_tag: str) -> str:
        """
        Convert spaCy's POS tags to human-readable forms.

        Args:
            pos_tag (str): The POS tag from spaCy

        Returns:
            str: A human-readable, lowercase version of the part of speech
        """
        explanation = spacy.explain(pos_tag)
        return explanation.lower() if explanation else pos_tag.lower()

    def process_lemmas(self, doc) -> Dict[str, List[LemmaEntry]]:
        """
        Process a spaCy document and extract lemmas.

        Args:
            doc: A spaCy document

        Returns:
            Dict[str, List[LemmaEntry]]: A mapping of lemmas to lists of sentences with original words
        """
        lemma_to_sentences: Dict[str, List[LemmaEntry]] = defaultdict(list)

        # Process each sentence
        for sent in doc.sents:
            sentence_text: str = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
                continue

            # Process each token in the sentence for lemmas
            for token in sent:
                # Skip punctuation and whitespace
                if token.is_punct or token.is_space:
                    continue

                # Get the lemma and original word
                # Don't lowercase proper nouns
                lemma: str = token.lemma_ if token.pos_ == "PROPN" else token.lemma_.lower()
                original_word: str = token.text

                # Skip if a lemma is empty
                if not lemma:
                    continue

                # Add the sentence to the mapping
                entry: LemmaEntry = {
                    "sentence": sentence_text,
                    "original_word": original_word,
                    "part_of_speech": self.get_human_readable_pos(token.pos_)
                }

                # Only add if this exact sentence isn't already in the list for this lemma
                if not any(el["sentence"] == sentence_text and el["original_word"] == original_word
                           for el in lemma_to_sentences[lemma]):
                    lemma_to_sentences[lemma].append(entry)

        return dict(lemma_to_sentences)

    def process_phrasal_verbs(self, doc, phrasal_verbs_path: Optional[str] = None) -> Dict[str, List[PhrasalVerbEntry]]:
        """
        Process a spaCy document and extract phrasal verbs.

        Args:
            doc: A spaCy document
            phrasal_verbs_path (Optional[str]): Path to the phrasal verbs file (optional)

        Returns:
            Dict[str, List[PhrasalVerbEntry]]: A mapping of phrasal verbs to lists of sentences
        """
        # Check if the language supports phrasal verbs
        if self.language not in self._LANGUAGES_WITH_PHRASAL_VERBS:
            print(f"Warning: Language '{self.language}' does not have phrasal verbs as a concept. Skipping.")
            return {}

        phrasal_verb_to_sentences: Dict[str, List[PhrasalVerbEntry]] = defaultdict(list)

        # Create a matcher for phrasal verbs if a phrasal verbs file is provided
        if not phrasal_verbs_path or not os.path.isfile(phrasal_verbs_path):
            if phrasal_verbs_path:
                print(f"Warning: Phrasal verbs file '{phrasal_verbs_path}' not found.")
            return {}

        # Load and process phrasal verbs
        raw_phrasal_verbs = read_phrasal_verbs(phrasal_verbs_path)
        normalized_phrasal_verbs = normalize_phrasal_verbs(raw_phrasal_verbs)
        verb_particle_pairs = split_phrasal_verbs(normalized_phrasal_verbs)

        # Create a matcher for phrasal verbs
        matcher = Matcher(self.nlp.vocab)

        # Add patterns for each phrasal verb
        for i, (verb, particle) in enumerate(verb_particle_pairs):
            particle_words = particle.split()

            if len(particle_words) == 1:
                # Single-word particle
                adjacent_pattern = [{"LEMMA": verb}, {"LOWER": particle}]
                separated_pattern = [
                    {"LEMMA": verb},
                    {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                    {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                    {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                    {"LOWER": particle, "POS": {"IN": ["ADP", "ADV", "PART"]}}
                ]
            else:
                # Multi-word particle
                adjacent_pattern = [{"LEMMA": verb}]
                for word in particle_words:
                    adjacent_pattern.append({"LOWER": word})

                separated_pattern = [
                    {"LEMMA": verb},
                    {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                    {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                    {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"}
                ]
                for j, word in enumerate(particle_words):
                    if j == 0:
                        separated_pattern.append({"LOWER": word, "POS": {"IN": ["ADP", "ADV", "PART"]}})
                    else:
                        separated_pattern.append({"LOWER": word})

            matcher.add(f"PV_{i}_ADJ", [adjacent_pattern])
            matcher.add(f"PV_{i}_SEP", [separated_pattern])

        # Process each sentence
        for sent in doc.sents:
            sentence_text: str = sent.text.strip()

            if not sentence_text:
                continue

            # Find phrasal verbs in the sentence
            sent_doc = self.nlp(sentence_text)
            matches = matcher(sent_doc)

            for match_id, start, end in matches:
                span = sent_doc[start:end]
                pattern_name = self.nlp.vocab.strings[match_id]
                pv_index = int(pattern_name.split('_')[1])

                if pv_index < len(verb_particle_pairs):
                    verb, particle = verb_particle_pairs[pv_index]
                    phrasal_verb_key = f"{verb} {particle}"

                    first_token = span[0]
                    last_token = span[-1]

                    if first_token.lemma_.lower() != verb:
                        continue

                    particle_words = particle.split()
                    if last_token.text.lower() != particle_words[-1]:
                        continue

                    if len(particle_words) > 1:
                        span_text = span.text.lower()
                        if not all(word in span_text for word in particle_words):
                            continue

                    entry: PhrasalVerbEntry = {
                        "sentence": sentence_text,
                        "original_text": span.text,
                        "verb": verb,
                        "particle": particle
                    }

                    if not any(el["sentence"] == sentence_text and el["original_text"] == span.text
                               for el in phrasal_verb_to_sentences[phrasal_verb_key]):
                        phrasal_verb_to_sentences[phrasal_verb_key].append(entry)

        return dict(phrasal_verb_to_sentences)

    def process_file(self, file_path: str, phrasal_verbs_path: Optional[str] = None) -> Tuple[
        Dict[str, List[LemmaEntry]], Dict[str, List[PhrasalVerbEntry]], str]:
        """
        Process a text file and create mappings of lemmas and phrasal verbs to sentences.

        Args:
            file_path (str): Path to the text file to process
            phrasal_verbs_path (Optional[str]): Path to the phrasal verbs file (optional)

        Returns:
            Tuple: (lemma_map, phrasal_verb_map, text)
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            exit_with_error(f"Error: File '{file_path}' does not exist.")

        # Read and normalize the text file
        file_content: str = process_text_file(file_path)

        print("Processing text with spaCy...")
        doc = self.nlp(file_content)

        # Process lemmas
        processed_lemmas = self.process_lemmas(doc)

        # Process phrasal verbs
        processed_phrasal_verbs = self.process_phrasal_verbs(doc, phrasal_verbs_path)

        return processed_lemmas, processed_phrasal_verbs, file_content


__all__ = [
    "LemmaExtractor",
    "LanguageMnemonic",
    "ModelType",
    "LemmaEntry",
    "PhrasalVerbEntry",
]
