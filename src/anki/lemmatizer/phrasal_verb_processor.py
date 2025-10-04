"""
Phrasal Verb Processor

This script processes a CSV file containing phrasal verbs and prepares them for use with spaCy.

Usage:
    python phrasal_verb_processor.py <path_to_phrasal_verbs_csv_file>
"""

import csv
import os.path
from typing import List, Tuple, Dict

import sys

from .text_processor import exit_with_error

# Type aliases for clarity
PhrasalVerb = str
VerbParticlePair = Tuple[str, str]
SpaCyPattern = List[Dict[str, str]]


def read_phrasal_verbs(file_path: str) -> List[PhrasalVerb]:
    """
    Read phrasal verbs from a CSV or text file.

    Args:
        file_path (str): Path to the file containing phrasal verbs (CSV or text)

    Returns:
        List[PhrasalVerb]: A list of phrasal verbs read from the file
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        exit_with_error(f"Error: File '{file_path}' does not exist.")

    # Determine file type based on extension
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.csv':
            # Read CSV file
            phrasal_verbs = []
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                # Skip header row
                next(reader, None)
                # Extract phrasal verbs from the first column
                for row in reader:
                    if row and len(row) > 0:
                        phrasal_verbs.append(row[0])
            return phrasal_verbs
        else:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as file:
                lines: List[str] = file.readlines()
            return lines
    except Exception as e:
        exit_with_error(f"Error reading file: {e}")


def normalize_phrasal_verbs(phrasal_verbs: List[PhrasalVerb]) -> List[PhrasalVerb]:
    """
    Normalize phrasal verbs by stripping whitespace and converting to lowercase.
    Skip empty lines and lines starting with a comment character (#).

    Args:
        phrasal_verbs (List[PhrasalVerb]): List of raw phrasal verbs

    Returns:
        List[PhrasalVerb]: List of normalized phrasal verbs
    """
    normalized: List[PhrasalVerb] = []

    for pv in phrasal_verbs:
        # Strip whitespace and convert to lowercase
        pv = pv.strip().lower()

        # Skip empty lines and comments
        if not pv or pv.startswith('#'):
            continue

        normalized.append(pv)

    return normalized


def split_phrasal_verbs(phrasal_verbs: List[PhrasalVerb]) -> List[VerbParticlePair]:
    """
    Split each phrasal verb into its verb and particle components.

    Args:
        phrasal_verbs (List[PhrasalVerb]): List of normalized phrasal verbs

    Returns:
        List[VerbParticlePair]: List of (verb, particle) tuples
    """
    verb_particle_pairs: List[VerbParticlePair] = []

    for pv in phrasal_verbs:
        # Split the phrasal verb into components
        components = pv.split()

        # Skip if not a valid phrasal verb (needs at least two components)
        if len(components) < 2:
            print(f"Warning: '{pv}' does not appear to be a valid phrasal verb. Skipping.")
            continue

        # Extract verb and particle
        verb = components[0]
        particle = ' '.join(components[1:])  # Join all remaining components as the particle

        verb_particle_pairs.append((verb, particle))

    return verb_particle_pairs


def create_spacy_patterns(verb_particle_pairs: List[VerbParticlePair]) -> List[List[Dict[str, str]]]:
    """
    Convert the list of (verb, particle) pairs into spaCy matcher patterns.

    Args:
        verb_particle_pairs (List[VerbParticlePair]): List of (verb, particle) tuples

    Returns:
        List[List[Dict[str, str]]]: List of spaCy matcher patterns
    """
    patterns: List[List[Dict[str, str]]] = []

    for verb, particle in verb_particle_pairs:
        # Create a pattern for the verb followed immediately by the particle
        adjacent_pattern: List[Dict[str, str]] = [
            {"LOWER": verb},
            {"LOWER": particle}
        ]

        # Create a pattern for separable phrasal verbs
        # This allows for up to 3 tokens between the verb and particle
        separable_pattern: List[Dict[str, str]] = [
            {"LOWER": verb},
            {"OP": "?"},
            {"OP": "?"},
            {"OP": "?"},
            {"LOWER": particle}
        ]

        patterns.append(adjacent_pattern)
        patterns.append(separable_pattern)

    return patterns


def display_results(verb_particle_pairs: List[VerbParticlePair]) -> None:
    """
    Display the processed phrasal verbs.

    Args:
        verb_particle_pairs (List[VerbParticlePair]): List of (verb, particle) tuples
    """
    print("\nProcessed Phrasal Verbs:")
    print("=======================\n")

    for i, (verb, particle) in enumerate(verb_particle_pairs, 1):
        print(f"{i}. Verb: '{verb}', Particle: '{particle}'")

    print(f"\nTotal: {len(verb_particle_pairs)} phrasal verbs processed.")


def main() -> None:
    """Main function to process phrasal verbs from a file."""
    # Check command-line arguments
    if len(sys.argv) != 2:
        exit_with_error(f"Usage: {sys.argv[0]} <path_to_phrasal_verbs_file>")

    file_path: str = sys.argv[1]

    # Process the file
    raw_phrasal_verbs: List[PhrasalVerb] = read_phrasal_verbs(file_path)
    normalized_phrasal_verbs: List[PhrasalVerb] = normalize_phrasal_verbs(raw_phrasal_verbs)
    verb_particle_pairs: List[VerbParticlePair] = split_phrasal_verbs(normalized_phrasal_verbs)

    # Display the results
    display_results(verb_particle_pairs)

    # Example of creating spaCy patterns
    patterns: List[List[Dict[str, str]]] = create_spacy_patterns(verb_particle_pairs)
    print("\nExample spaCy pattern for the first phrasal verb:")
    if patterns:
        print(patterns[0])


if __name__ == "__main__":
    main()
