"""
Phrasal Verb PDF to CSV Converter

This script converts the Complete-PV-list.pdf file to CSV format for easier processing.
It extracts phrasal verbs, their meanings, and examples from the PDF and saves them to a CSV file.

Usage:
    python converter.py <path_to_pdf_file> <output_csv_file>
"""

import csv
import os.path
import re
from typing import List, Tuple, NoReturn, TypeAlias

import PyPDF2
import sys

# Type aliases for clarity
PhrasalVerb: TypeAlias = str
PhrasalVerbEntry: TypeAlias = Tuple[str, str, str]  # (phrasal_verb, meaning, example)


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


def read_pdf_file(file_path: str, max_pages: int = 0) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_path (str): Path to the PDF file
        max_pages (int, optional): Maximum number of pages to process.
            Set to 0 to process all pages.

    Returns:
        str: The extracted text from the PDF
    """
    try:
        # Open the PDF file
        with open(file_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)

            # Determine number of pages to process
            total_pages = len(pdf_reader.pages)
            pages_to_process = total_pages if max_pages == 0 else min(max_pages, total_pages)

            print(f"Processing PDF: {file_path}")
            print(f"Total pages: {total_pages}, processing first {pages_to_process} pages")

            # Extract text from each page
            text = ""
            for page_num in range(pages_to_process):
                print(f"Processing page {page_num + 1}/{pages_to_process}...", end="\r")
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            print("\nText extraction complete.")
            return text
    except Exception as e:
        exit_with_error(f"Error reading PDF file: {e}")


def extract_phrasal_verb_entries(text: str) -> List[PhrasalVerbEntry]:
    """
    Extract phrasal verbs, meanings, and examples from the Complete-PV-list.pdf file.

    The PDF has a tabular format with columns for:
    1. Phrasal Verb (e.g., "Abide by")
    2. Meaning (e.g., "Accept or follow a decision or rule.")
    3. Example (e.g., "We have to ABIDE BY what the court says.")

    When extracted as text, the columns run together, making it challenging to parse.
    This function attempts to extract all three components.

    Args:
        text (str): The extracted text from the PDF

    Returns:
        List[PhrasalVerbEntry]: A list of tuples containing (phrasal_verb, meaning, example)
    """
    # List to store the extracted phrasal verb entries
    entries = []

    # Split the text into lines
    lines = text.split('\n')

    # Skip the header lines
    start_processing = False
    for i, line in enumerate(lines):
        if "Complete Phrasal Verbs List" in line:
            start_processing = True
            continue

        if not start_processing:
            continue

        # Skip empty lines and header lines
        if not line.strip() or "Phrasal" in line and "Verb" in line and "Meaning" in line:
            continue

        # Pattern 1: Look for patterns like "Abide byAccept or follow a decision or rule.We have to ABIDE BY what the court says."
        # The pattern is: starts with capital letter, followed by lowercase letters and spaces,
        # then another capital letter (which starts the meaning), followed by text, then uppercase letters (which starts the example)
        match = re.match(r'^([A-Z][a-z]+(?:\s+[a-z]+)+)([A-Z][^A-Z]+)([A-Z].*)', line)
        if match:
            phrasal_verb = match.group(1).strip().lower()
            meaning = match.group(2).strip()
            example = match.group(3).strip()
            entries.append((phrasal_verb, meaning, example))
            continue

        # Pattern 2: Look for patterns where the example might be on the next line
        match = re.match(r'^([A-Z][a-z]+(?:\s+[a-z]+)+)([A-Z][^A-Z]+)$', line)
        if match and i + 1 < len(lines):
            phrasal_verb = match.group(1).strip().lower()
            meaning = match.group(2).strip()
            example = lines[i + 1].strip()
            entries.append((phrasal_verb, meaning, example))
            continue

        # Pattern 3: Look for lines that start with a capital letter followed by lowercase letters and a space
        # This catches phrasal verbs at the beginning of lines, but we might not be able to separate meaning and example
        match = re.match(r'^([A-Z][a-z]+\s+[a-z]+)(.*)', line)
        if match:
            phrasal_verb = match.group(1).strip().lower()
            rest_of_line = match.group(2).strip()
            # Try to split the rest of the line into meaning and example
            meaning_example_match = re.match(r'^([^A-Z]+)([A-Z].*)', rest_of_line)
            if meaning_example_match:
                meaning = meaning_example_match.group(1).strip()
                example = meaning_example_match.group(2).strip()
            else:
                # If we can't split, just use the rest as meaning and leave example empty
                meaning = rest_of_line
                example = ""
            entries.append((phrasal_verb, meaning, example))
            continue

    # Clean up the entries
    cleaned_entries = []
    for pv, meaning, example in entries:
        # Only keep phrasal verbs that have 2-3 words
        words = pv.split()
        if 2 <= len(words) <= 3:
            # Check if all words are alphabetic
            if all(word.isalpha() for word in words):
                # Check if the first word is a verb (this is a heuristic)
                if words[0] not in ["the", "a", "an", "in", "on", "at", "by", "for", "with", "to", "this", "that",
                                    "these", "those"]:
                    cleaned_entries.append((pv, meaning, example))

    return cleaned_entries


def save_to_csv(entries: List[PhrasalVerbEntry], output_file: str) -> None:
    """
    Save the phrasal verb entries to a CSV file.

    Args:
        entries (List[PhrasalVerbEntry]): List of tuples containing (phrasal_verb, meaning, example)
        output_file (str): Path to the output CSV file
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Phrasal Verb', 'Meaning', 'Example'])
            # Write data
            for pv, meaning, example in entries:
                writer.writerow([pv, meaning, example])

        print(f"Successfully saved {len(entries)} phrasal verb entries to {output_file}")
    except Exception as e:
        exit_with_error(f"Error saving to CSV file: {e}")


def main() -> None:
    """Main function to convert PDF to CSV."""
    # Check command-line arguments
    if len(sys.argv) != 3:
        exit_with_error(f"Usage: {sys.argv[0]} <path_to_pdf_file> <output_csv_file>")

    pdf_file_path: str = sys.argv[1]
    output_csv_file: str = sys.argv[2]

    # Check if PDF file exists
    if not os.path.isfile(pdf_file_path):
        exit_with_error(f"Error: File '{pdf_file_path}' does not exist.")

    # Check if the file is a PDF
    if not pdf_file_path.lower().endswith('.pdf'):
        exit_with_error(f"Error: File '{pdf_file_path}' is not a PDF file.")

    # Process the PDF file
    pdf_text = read_pdf_file(pdf_file_path)

    # Extract phrasal verb entries
    entries = extract_phrasal_verb_entries(pdf_text)

    # Display some statistics
    print(f"\nExtracted {len(entries)} phrasal verb entries.")
    if entries:
        print("\nExample entries:")
        for i, (pv, meaning, example) in enumerate(entries[:5], 1):
            print(f"{i}. Phrasal Verb: '{pv}', Meaning: '{meaning}', Example: '{example}'")

    # Save to CSV
    save_to_csv(entries, output_csv_file)


if __name__ == "__main__":
    main()
