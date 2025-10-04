"""
Phrasal Verb PDF to CSV Converter - Runner

Configure the variables below and run via:
    uv run --project <project_path> --module pv_pdf_to_csv
"""
from __future__ import annotations

from pathlib import Path

from .converter import read_pdf_file, extract_phrasal_verb_entries, save_to_csv


def main() -> None:
    """
    Main runner function for PDF to CSV conversion.

    Edit the variables below to configure the conversion:
    """
    # CONFIGURE THESE VARIABLES
    pdf_file_path = "path/to/your/phrasal_verbs.pdf"
    output_csv_file = "output/phrasal_verbs.csv"

    # Validate inputs
    if not Path(pdf_file_path).is_file():
        print(f"Error: PDF file '{pdf_file_path}' does not exist.")
        print("Please edit the pdf_file_path variable in src/pv_pdf_to_csv/run.py")
        return

    if not pdf_file_path.lower().endswith('.pdf'):
        print(f"Error: File '{pdf_file_path}' is not a PDF file.")
        return

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
