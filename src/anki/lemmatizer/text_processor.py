"""
Text Processor

This script provides functions for processing text files, including normalization
and utility functions for text handling.
"""

import os.path
import re
import sys
from typing import NoReturn


def normalize_text(text: str) -> str:
    """
    Normalize text by handling multiple linebreaks and detecting headings.

    This function:
    1. Removes excessive line breaks
    2. Detects headings (all caps or short lines followed by empty lines)
    3. Ensures each heading is treated as a separate sentence

    Args:
        text (str): The raw text to normalize

    Returns:
        str: Normalized text suitable for spaCy processing
    """
    # Split text into lines
    lines = text.split('\n')

    # Remove empty lines at the beginning and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    normalized_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if this line is a potential heading
        is_heading = False

        # Check if line is all uppercase (potential heading)
        if line.isupper():
            is_heading = True

        # Check if line is short (less than 50 chars) and followed by empty line(s)
        elif len(line) < 50 and i + 1 < len(lines) and not lines[i + 1].strip():
            is_heading = True

        # If it's a heading, ensure it ends with a period if it doesn't have ending punctuation
        if is_heading and not line[-1] in '.!?:;':
            line += '.'

        normalized_lines.append(line)
        i += 1

    # Join lines with a space between them to create continuous text
    normalized_text = ' '.join(normalized_lines)

    # Replace multiple spaces with a single space
    normalized_text = re.sub(r'\s+', ' ', normalized_text)

    return normalized_text


def exit_with_error(message: str, exit_code: int = 1) -> NoReturn:
    """
    Print an error message and exit the program.

    Args:
        message (str): The error message to display
        exit_code (int): The exit code to use (default: 1)

    Returns:
        NoReturn: This function never returns
    """
    print(message)
    sys.exit(exit_code)


def read_text_file(file_path: str) -> str:
    """
    Read and return the content of a text file.

    Args:
        file_path (str): Path to the text file to read

    Returns:
        str: The content of the text file

    Raises:
        SystemExit: If the file doesn't exist or can't be read
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        exit_with_error(f"Error: File '{file_path}' does not exist.")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text: str = file.read()
    return raw_text


def process_text_file(file_path: str) -> str:
    """
    Read a text file and normalize its content.

    Args:
        file_path (str): Path to the text file to process

    Returns:
        str: Normalized text content
    """
    raw_text = read_text_file(file_path)
    return normalize_text(raw_text)
