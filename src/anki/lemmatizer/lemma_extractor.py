"""
Lemma and Phrasal Verb Extractor

This module provides a class for extracting lemmas and phrasal verbs from text files.
It supports different source languages and model types including transformer models.

Usage:
    1. As a module:
        from lemmatizer import LemmaExtractor, LanguageMnemonic, ModelType

        extractor = LemmaExtractor(LanguageMnemonic.EN, ModelType.TRANSFORMER)
        lemma_map, phrasal_verb_map, text = extractor.process_file(text_file_path, phrasal_verbs_file_path)

    2. As a script:
        python lemma_extractor.py --language EN --input text.txt --output results.txt [--phrasal-verbs phrasal_verbs.csv] [--model-type TRANSFORMER]
"""

import argparse
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
    EN = "EN"  # English
    DE = "DE"  # German
    FR = "FR"  # French
    ES = "ES"  # Spanish
    IT = "IT"  # Italian
    RU = "RU"  # Russian
    JA = "JA"  # Japanese
    ZH = "ZH"  # Chinese


class ModelType(Enum):
    """Model type selection for language processing."""
    EFFICIENT = auto()
    ACCURATE = auto()
    TRANSFORMER = auto()  # New transformer option


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
            ModelType.ACCURATE: "en_core_web_lg",
            ModelType.TRANSFORMER: "en_core_web_trf"
        },
        LanguageMnemonic.DE: {
            ModelType.EFFICIENT: "de_core_news_sm",
            ModelType.ACCURATE: "de_core_news_lg",
            ModelType.TRANSFORMER: "de_dep_news_trf"
        },
        LanguageMnemonic.FR: {
            ModelType.EFFICIENT: "fr_core_news_sm",
            ModelType.ACCURATE: "fr_core_news_lg",
            ModelType.TRANSFORMER: "fr_dep_news_trf"
        },
        LanguageMnemonic.ES: {
            ModelType.EFFICIENT: "es_core_news_sm",
            ModelType.ACCURATE: "es_core_news_lg",
            ModelType.TRANSFORMER: "es_dep_news_trf"
        },
        LanguageMnemonic.IT: {
            ModelType.EFFICIENT: "it_core_news_sm",
            ModelType.ACCURATE: "it_core_news_lg",
            ModelType.TRANSFORMER: "it_core_news_lg"  # No transformer model available yet
        },
        LanguageMnemonic.RU: {
            ModelType.EFFICIENT: "ru_core_news_sm",
            ModelType.ACCURATE: "ru_core_news_lg",
            ModelType.TRANSFORMER: "ru_core_news_lg"  # No transformer model available yet
        },
        LanguageMnemonic.JA: {
            ModelType.EFFICIENT: "ja_core_news_sm",
            ModelType.ACCURATE: "ja_core_news_lg",
            ModelType.TRANSFORMER: "ja_core_news_lg"  # No transformer model available yet
        },
        LanguageMnemonic.ZH: {
            ModelType.EFFICIENT: "zh_core_web_sm",
            ModelType.ACCURATE: "zh_core_web_lg",
            ModelType.TRANSFORMER: "zh_core_web_trf"
        }
    }

    def __init__(self, lang: LanguageMnemonic, mod_type: ModelType = ModelType.ACCURATE):
        """
        Initialize the LemmaExtractor with a specific language and model type.

        Args:
            lang (LanguageMnemonic): The source language mnemonic (e.g., "EN", "DE")
            mod_type (ModelType, optional): The model type to use. Defaults to ModelType.ACCURATE.

        Raises:
            ValueError: If the language is not supported
        """
        self.language = lang
        self.model_type = mod_type

        # Check if the language is supported
        if lang not in self._LANGUAGE_TO_MODEL:
            langs = ", ".join([lang.value for lang in self._LANGUAGE_TO_MODEL.keys()])
            exit_with_error(f"Error: Language '{lang}' is not supported. Supported languages: {langs}")

        # Load spaCy model
        model_name = self._LANGUAGE_TO_MODEL[lang][mod_type]

        # Check if transformer model is requested but not available
        if mod_type == ModelType.TRANSFORMER and model_name == self._LANGUAGE_TO_MODEL[lang][ModelType.ACCURATE]:
            print(f"Warning: Transformer model not available for {lang.value}. Using accurate model instead.")

        try:
            # For transformer models, ensure curated transformers are available
            if mod_type == ModelType.TRANSFORMER and "trf" in model_name:
                import spacy_curated_transformers
                print("Curated transformers components loaded.")

            self.nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")

            # Check if we need to install additional dependencies for transformer models
            if mod_type == ModelType.TRANSFORMER and "trf" in model_name:
                self._check_transformer_dependencies()

        except OSError:
            print(f"Downloading spaCy model {model_name}...")
            try:
                from spacy.cli import download
                download(model_name)

                # For transformer models, ensure curated transformers are available after download
                if mod_type == ModelType.TRANSFORMER and "trf" in model_name:
                    import spacy_curated_transformers
                    print("Curated transformers components loaded after download.")

                self.nlp = spacy.load(model_name)
                print(f"Successfully loaded spaCy model: {model_name}")

                # Check transformer dependencies after download
                if mod_type == ModelType.TRANSFORMER and "trf" in model_name:
                    self._check_transformer_dependencies()

            except Exception as e:
                exit_with_error(f"Error downloading or loading model {model_name}: {e}")

    def _check_transformer_dependencies(self):
        """Check if transformer dependencies are installed."""
        try:
            import torch
            import transformers
            print("Transformer dependencies are available.")
        except ImportError as e:
            print(f"Warning: Transformer dependencies not found: {e}")
            print("For optimal performance with transformer models, install:")
            print("pip install spacy-transformers")
            print("or")
            print("pip install torch transformers")

    @staticmethod
    def get_human_readable_pos(pos_tag: str) -> str:
        """
        Convert spaCy's POS tags to human-readable forms using spaCy's explain function.

        Args:
            pos_tag (str): The POS tag from spaCy (e.g., 'NOUN', 'VERB')

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
                # Skip punctuation and whitespace but include stop words that might be part of phrasal verbs
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
            print(
                f"Warning: Language '{self.language}' does not have phrasal verbs as a concept. Skipping phrasal verb processing.")
            return {}

        phrasal_verb_to_sentences: Dict[str, List[PhrasalVerbEntry]] = defaultdict(list)

        # Create a matcher for phrasal verbs if a phrasal verbs file is provided
        matcher = None
        verb_particle_pairs: List[VerbParticlePair] = []

        if phrasal_verbs_path:
            if os.path.isfile(phrasal_verbs_path):
                # Load and process phrasal verbs
                raw_phrasal_verbs = read_phrasal_verbs(phrasal_verbs_path)
                normalized_phrasal_verbs = normalize_phrasal_verbs(raw_phrasal_verbs)
                verb_particle_pairs = split_phrasal_verbs(normalized_phrasal_verbs)

                # Create a matcher for phrasal verbs
                matcher = Matcher(self.nlp.vocab)

                # Add patterns for each phrasal verb
                for i, (verb, particle) in enumerate(verb_particle_pairs):
                    # Split the particle into individual words for multi-word particles
                    particle_words = particle.split()

                    if len(particle_words) == 1:
                        # Single-word particle (e.g., "up", "off")
                        # Pattern for adjacent verb and particle
                        adjacent_pattern = [{"LEMMA": verb}, {"LOWER": particle}]

                        # Pattern for separated verb and particle (up to 3 tokens between)
                        # Use more specific patterns to avoid matching unrelated words
                        separated_pattern = [
                            {"LEMMA": verb},
                            {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                            {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                            {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                            {"LOWER": particle, "POS": {"IN": ["ADP", "ADV", "PART"]}}
                        ]
                    else:
                        # Multi-word particle (e.g., "forward to")
                        # Pattern for adjacent verb and multi-word particle
                        adjacent_pattern = [{"LEMMA": verb}]
                        for word in particle_words:
                            adjacent_pattern.append({"LOWER": word})

                        # Pattern for separated verb and multi-word particle
                        # Allow up to 3 tokens between verb and first particle word
                        # Use more specific patterns to avoid matching unrelated words
                        separated_pattern = [
                            {"LEMMA": verb},
                            {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                            {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"},
                            {"POS": {"IN": ["DET", "PRON", "ADJ", "ADV", "NOUN"]}, "OP": "?"}
                        ]
                        for j, word in enumerate(particle_words):
                            # For the first word of the particle, specify it should be an adposition or adverb
                            if j == 0:
                                separated_pattern.append({"LOWER": word, "POS": {"IN": ["ADP", "ADV", "PART"]}})
                            else:
                                separated_pattern.append({"LOWER": word})

                    matcher.add(f"PV_{i}_ADJ", [adjacent_pattern])
                    matcher.add(f"PV_{i}_SEP", [separated_pattern])
            else:
                print(
                    f"Warning: Phrasal verbs file '{phrasal_verbs_path}' not found. Proceeding without phrasal verb detection.")

        if not matcher:
            return {}

        # Process each sentence
        for sent in doc.sents:
            sentence_text: str = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
                continue

            # Find phrasal verbs in the sentence
            sent_doc = self.nlp(sentence_text)  # Process the sentence separately for matching
            matches = matcher(sent_doc)

            for match_id, start, end in matches:
                # Get the matched span
                span = sent_doc[start:end]

                # Get the pattern name to identify which phrasal verb was matched
                pattern_name = self.nlp.vocab.strings[match_id]
                pv_index = int(pattern_name.split('_')[1])  # Extract index from pattern name

                # Get the verb and particle from the original list
                if pv_index < len(verb_particle_pairs):
                    verb, particle = verb_particle_pairs[pv_index]

                    # Create the phrasal verb key (verb + particle)
                    phrasal_verb_key = f"{verb} {particle}"

                    # Skip invalid matches where the first token is not the verb
                    # or the last token is not part of the particle
                    first_token = span[0]
                    last_token = span[-1]

                    # Check if the first token's lemma matches the verb
                    if first_token.lemma_.lower() != verb:
                        continue

                    # Check if the last token's text matches the last word of the particle
                    particle_words = particle.split()
                    if last_token.text.lower() != particle_words[-1]:
                        continue

                    # For multi-word particles, check if all words are present in the span
                    if len(particle_words) > 1:
                        # Get the text of the span and check if all particle words are in it
                        span_text = span.text.lower()
                        if not all(word in span_text for word in particle_words):
                            continue

                    # Create an entry for this phrasal verb occurrence
                    entry: PhrasalVerbEntry = {
                        "sentence": sentence_text,
                        "original_text": span.text,
                        "verb": verb,
                        "particle": particle
                    }

                    # Only add if this exact sentence and original text isn't already in the list
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
            Tuple[Dict[str, List[LemmaEntry]], Dict[str, List[PhrasalVerbEntry]], str]:
                A tuple containing:
                - A mapping of lemmas to lists of sentences with original words
                - A mapping of phrasal verbs to lists of sentences with original text
                - The processed text from the file
        """
        # Check if file exists
        if not os.path.isfile(file_path):
            exit_with_error(f"Error: File '{file_path}' does not exist.")

        # Read and normalize the text file
        try:
            file_content: str = process_text_file(file_path)
        except Exception as e:
            exit_with_error(f"Error reading file: {e}")

        # Process the text with progress indication for transformer models
        if self.model_type == ModelType.TRANSFORMER:
            print("Processing text with transformer model (this may take longer)...")

        doc = self.nlp(file_content)

        # Process lemmas
        processed_lemmas = self.process_lemmas(doc)

        # Process phrasal verbs
        processed_phrasal_verbs = self.process_phrasal_verbs(doc, phrasal_verbs_path)

        return processed_lemmas, processed_phrasal_verbs, file_content


def display_results(lemmas: Dict[str, List[LemmaEntry]],
                    phrasal_verbs: Dict[str, List[PhrasalVerbEntry]]) -> None:
    """
    Display the mapping of lemmas and phrasal verbs to sentences.

    Args:
        lemmas (Dict[str, List[LemmaEntry]]): A mapping of lemmas to lists of sentences with original words
        phrasal_verbs (Dict[str, List[PhrasalVerbEntry]]): A mapping of phrasal verbs to lists of sentences
    """
    # Display lemmas
    print("\nLemma to Sentences Mapping:")
    print("===========================\n")

    for lemma, entries in sorted(lemmas.items()):
        print(f"Lemma: '{lemma}'")
        for i, entry in enumerate(entries, 1):
            print(f"  {i}. Original word: '{entry['original_word']}', Part of speech: '{entry['part_of_speech']}'")
            print(f"     Sentence: \"{entry['sentence']}\"")
        print()

    # Display phrasal verbs if any
    if phrasal_verbs:
        print("\nPhrasal Verb to Sentences Mapping:")
        print("=================================\n")

        for phrasal_verb, entries in sorted(phrasal_verbs.items()):
            print(f"Phrasal Verb: '{phrasal_verb}'")
            for i, entry in enumerate(entries, 1):
                print(f"  {i}. Original text: '{entry['original_text']}'")
                print(f"     Verb: '{entry['verb']}', Particle: '{entry['particle']}'")
                print(f"     Sentence: \"{entry['sentence']}\"")
            print()


def save_results_to_file(lemmas: Dict[str, List[LemmaEntry]],
                         phrasal_verbs: Dict[str, List[PhrasalVerbEntry]],
                         output_file: str) -> None:
    """
    Save the mapping of lemmas and phrasal verbs to sentences to a file.

    Args:
        lemmas (Dict[str, List[LemmaEntry]]): A mapping of lemmas to lists of sentences with original words
        phrasal_verbs (Dict[str, List[PhrasalVerbEntry]]): A mapping of phrasal verbs to lists of sentences
        output_file (str): Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write lemmas
        f.write("LEMMA TO SENTENCES MAPPING:\n")
        f.write("===========================\n\n")

        for lemma, entries in sorted(lemmas.items()):
            f.write(f"Lemma: '{lemma}'\n")
            for i, entry in enumerate(entries, 1):
                f.write(
                    f"  {i}. Original word: '{entry['original_word']}', Part of speech: '{entry['part_of_speech']}'\n")
                f.write(f"     Sentence: \"{entry['sentence']}\"\n")
            f.write("\n")

        # Write phrasal verbs if any
        if phrasal_verbs:
            f.write("\nPHRASAL VERB TO SENTENCES MAPPING:\n")
            f.write("=================================\n\n")

            for phrasal_verb, entries in sorted(phrasal_verbs.items()):
                f.write(f"Phrasal Verb: '{phrasal_verb}'\n")
                for i, entry in enumerate(entries, 1):
                    f.write(f"  {i}. Original text: '{entry['original_text']}'\n")
                    f.write(f"     Verb: '{entry['verb']}', Particle: '{entry['particle']}'\n")
                    f.write(f"     Sentence: \"{entry['sentence']}\"\n")
                f.write("\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract lemmas and phrasal verbs from text files.')
    parser.add_argument('--language', type=str, required=True,
                        help='Source language mnemonic (e.g., EN, DE, FR)')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input text file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output file')
    parser.add_argument('--phrasal-verbs', type=str,
                        help='Path to the phrasal verbs file (optional)')
    parser.add_argument('--model-type', type=str, choices=['EFFICIENT', 'ACCURATE', 'TRANSFORMER'], default='ACCURATE',
                        help='Model type to use (default: ACCURATE)')

    args = parser.parse_args()

    try:
        # Convert language string to LanguageMnemonic enum
        language = LanguageMnemonic(args.language)
    except ValueError:
        supported_languages = ", ".join([lang.value for lang in LanguageMnemonic])
        exit_with_error(
            f"Error: Language '{args.language}' is not supported. Supported languages: {supported_languages}")

    # Convert model_type string to ModelType enum
    model_type_map = {
        'EFFICIENT': ModelType.EFFICIENT,
        'ACCURATE': ModelType.ACCURATE,
        'TRANSFORMER': ModelType.TRANSFORMER
    }
    model_type = model_type_map[args.model_type]

    # Create LemmaExtractor instance
    extractor = LemmaExtractor(language, model_type)

    # Process the file
    try:
        lemma_map, phrasal_verb_map, text = extractor.process_file(args.input, args.phrasal_verbs)
    except Exception as e:
        exit_with_error(f"Error processing file: {e}")

    # Display results to console
    display_results(lemma_map, phrasal_verb_map)

    # Save results to file
    save_results_to_file(lemma_map, phrasal_verb_map, args.output)

    print(f"Processed {len(lemma_map)} lemmas and {len(phrasal_verb_map)} phrasal verbs.")
    print(f"Results displayed on console and saved to {args.output}")
