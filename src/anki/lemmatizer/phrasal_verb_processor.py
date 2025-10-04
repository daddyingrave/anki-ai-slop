"""
Phrasal Verb Processor

This script processes a CSV file containing phrasal verbs and prepares them for use with spaCy.
"""

import csv
import os.path
import sys
from typing import List, Tuple, Dict

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
