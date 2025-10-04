"""
Subtitle processing module for parsing SRT files.

This module provides functionality to read and process subtitle files,
extracting text, metadata, and song lyrics.
"""

import re

import pysrt


class Processed:
    """
    Represents processed subtitle data.

    Attributes:
        text (str): The main text content.
        meta (list): Metadata extracted from brackets like [information].
        songs (list): Lines that are enclosed in ♪ symbols.
    """

    def __init__(self, text="", meta=None, songs=None):
        self.text = text
        self.meta = meta or []
        self.songs = songs or []


class Reader:
    """
    Reader for subtitle files.

    Provides functionality to read and process subtitle files,
    extracting text, metadata, and song lyrics.
    """

    # Constants for thresholds
    SONG_LINE_THRESHOLD = 6
    META_LINE_THRESHOLD = 3
    TEXT_LINE_THRESHOLD = 5

    def read(self, path):
        """
        Read and process a subtitle file.

        Args:
            path (str): Path to the subtitle file.

        Returns:
            Processed: A Processed object containing the extracted text, metadata, and songs.

        Raises:
            Exception: If the file cannot be read.
        """
        try:
            subs = pysrt.open(path)
        except Exception as e:
            raise Exception(f"Cannot read file: {str(e)}")

        songs = []
        meta = []
        text_parts = []

        for sub in subs:
            # Get the text content without styling
            line_content = self._remove_styling(sub.text)

            # Process songs (lines with ♪ symbols)
            if line_content.startswith("♪") and line_content.endswith("♪"):
                line_content = line_content.replace("♪", "").strip()
                # Don't want to put short lines into songs, because it's probably a mistake
                if len(line_content) > self.SONG_LINE_THRESHOLD:
                    songs.append(line_content)
            else:
                remaining_text = ""
                meta_blocks = []

                # Process meta blocks (text in square brackets)
                while True:
                    start_idx = line_content.find("[")
                    if start_idx == -1:
                        remaining_text += line_content
                        break

                    end_idx = line_content.find("]", start_idx)
                    if end_idx == -1:
                        remaining_text += line_content
                        break

                    # Add text before the meta block to remaining_text
                    remaining_text += line_content[:start_idx]

                    # Extract the meta block content (without brackets)
                    meta_content = line_content[start_idx + 1:end_idx]
                    if len(meta_content) > self.META_LINE_THRESHOLD:
                        meta_blocks.append(meta_content)

                    # Continue with the text after the meta block
                    line_content = line_content[end_idx + 1:]

                # Add all extracted meta blocks to meta
                meta.extend(meta_blocks)

                # Add remaining text if it's long enough
                remaining_text = remaining_text.strip()
                if len(remaining_text) > self.TEXT_LINE_THRESHOLD:
                    text_parts.append(remaining_text)

        # Join all text parts with spaces
        full_text = " ".join(text_parts)

        return Processed(text=full_text, meta=meta, songs=songs)

    def _remove_styling(self, text):
        """
        Remove styling tags from the text.

        Args:
            text (str): Text with potential styling tags.

        Returns:
            str: Text without styling tags.
        """
        # Remove HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove other common styling markers
        text = text.replace('-', ' ').replace('  ', ' ')
        return text.strip()


def new_reader():
    """
    Create a new Reader instance.

    Returns:
        Reader: A new Reader instance.
    """
    return Reader()
