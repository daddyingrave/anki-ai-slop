"""
Text-to-speech audio generation using Google Cloud TTS.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_google_community import TextToSpeechTool


@dataclass
class AudioFile:
    """Represents generated audio file."""
    filename: str
    content: bytes


class TTSClient:
    """Text-to-speech client with caching support."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize TTS client.

        Args:
            cache_dir: Directory to cache audio files. If None, no caching.
        """
        self._client: TextToSpeechTool | None = None
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> TextToSpeechTool:
        """Get or create TTS client instance (lazy initialization)."""
        if self._client is None:
            self._client = TextToSpeechTool()
        return self._client

    def _generate_filename(self, text: str) -> str:
        """Generate filename from text content hash.

        Args:
            text: Text content

        Returns:
            Filename with .mp3 extension
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{text_hash}.mp3"

    def generate_audio(self, text: str) -> AudioFile:
        """Generate audio from text.

        Checks cache first if cache_dir is set. Returns audio bytes and filename.

        Args:
            text: Text to convert to speech

        Returns:
            AudioFile with filename and binary content

        Raises:
            RuntimeError: If audio generation fails
        """
        filename = self._generate_filename(text)

        # Check cache first
        if self._cache_dir is not None:
            cache_path = self._cache_dir / filename
            if cache_path.exists():
                return AudioFile(filename=filename, content=cache_path.read_bytes())

        # Generate audio
        client = self._get_client()
        try:
            temp_audio_path = client.run(text)
            if temp_audio_path is None or not Path(temp_audio_path).exists():
                raise RuntimeError(f"Failed to generate audio for text: {text[:50]}")

            audio_content = Path(temp_audio_path).read_bytes()

            # Cache if cache_dir is set
            if self._cache_dir is not None:
                cache_path = self._cache_dir / filename
                cache_path.write_bytes(audio_content)

            # Clean up temp file
            Path(temp_audio_path).unlink(missing_ok=True)

            return AudioFile(filename=filename, content=audio_content)

        except Exception as e:
            raise RuntimeError(f"Failed to generate audio for text '{text[:50]}...': {e}") from e
