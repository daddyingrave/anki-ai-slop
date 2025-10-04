"""
Subtitle processor - Runner

Configure the variables below and run via:
    uv run --project <project_path> --module subs
"""
from __future__ import annotations

from pathlib import Path

from .reader import new_reader


def main() -> None:
    """
    Main runner function for subtitle processing.

    Edit the variables below to configure the processing:
    """
    # CONFIGURE THESE VARIABLES
    subtitle_file_path = "path/to/your/subtitle.srt"
    output_text_file = "output/extracted_text.txt"

    # Validate inputs
    if not Path(subtitle_file_path).is_file():
        print(f"Error: Subtitle file '{subtitle_file_path}' does not exist.")
        print("Please edit the subtitle_file_path variable in src/subs/run.py")
        return

    # Process the subtitle file
    reader = new_reader()
    try:
        processed = reader.read(subtitle_file_path)
    except Exception as e:
        print(f"Error processing subtitle file: {e}")
        return

    # Display statistics
    print(f"\nProcessed subtitle file: {subtitle_file_path}")
    print(f"Extracted text length: {len(processed.text)} characters")
    print(f"Metadata entries: {len(processed.meta)}")
    print(f"Song lyrics entries: {len(processed.songs)}")

    # Display samples
    if processed.text:
        print(f"\nText preview (first 200 chars):\n{processed.text[:200]}...")

    if processed.meta:
        print(f"\nMetadata (first 5 entries):")
        for i, meta_item in enumerate(processed.meta[:5], 1):
            print(f"  {i}. {meta_item}")

    if processed.songs:
        print(f"\nSong lyrics (first 5 entries):")
        for i, song in enumerate(processed.songs[:5], 1):
            print(f"  {i}. {song}")

    # Save to file
    try:
        output_path = Path(output_text_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("EXTRACTED TEXT:\n")
            f.write("=" * 80 + "\n\n")
            f.write(processed.text)
            f.write("\n\n")

            if processed.meta:
                f.write("\nMETADATA:\n")
                f.write("=" * 80 + "\n\n")
                for meta_item in processed.meta:
                    f.write(f"- {meta_item}\n")
                f.write("\n")

            if processed.songs:
                f.write("\nSONG LYRICS:\n")
                f.write("=" * 80 + "\n\n")
                for song in processed.songs:
                    f.write(f"- {song}\n")

        print(f"\nOutput saved to: {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")


if __name__ == "__main__":
    main()
