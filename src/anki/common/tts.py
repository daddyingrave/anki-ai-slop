"""
Text-to-speech audio generation using Google Cloud TTS with Gemini voices.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from google.cloud import texttospeech


@dataclass
class AudioFile:
    """Represents generated audio file."""
    filename: str
    content: bytes


@dataclass
class TTSVoiceConfig:
    """Voice configuration for Gemini Flash TTS."""
    model_name: str = "gemini-2.5-flash-tts"
    voice_name: str = "Enceladus"
    language_code: str = "en-us"
    speaking_rate: float = 1.0
    pitch: float = 0.0


class TTSClient:
    """Text-to-speech client with caching support using Google Cloud TTS."""

    def __init__(self, cache_dir: Path | None = None, voice_config: TTSVoiceConfig | None = None):
        """Initialize TTS client.

        Args:
            cache_dir: Directory to cache audio files. If None, no caching.
            voice_config: Voice configuration. If None, uses defaults.
        """
        self._client: texttospeech.TextToSpeechClient | None = None
        self._cache_dir = cache_dir
        self._voice_config = voice_config or TTSVoiceConfig()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> texttospeech.TextToSpeechClient:
        """Get or create TTS client (lazy initialization with service account auth)."""
        if self._client is None:
            self._client = texttospeech.TextToSpeechClient()
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
        """Generate audio from text using Google Cloud TTS with Gemini voices.

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

        # Generate audio using Google Cloud TTS
        client = self._get_client()
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)

            voice = texttospeech.VoiceSelectionParams(
                name=self._voice_config.voice_name,
                language_code=self._voice_config.language_code,
                model_name=self._voice_config.model_name,
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=self._voice_config.speaking_rate,
                pitch=self._voice_config.pitch,
            )

            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )

            audio_content = response.audio_content

            # Cache if cache_dir is set
            if self._cache_dir is not None:
                cache_path = self._cache_dir / filename
                cache_path.write_bytes(audio_content)

            return AudioFile(filename=filename, content=audio_content)

        except Exception as e:
            raise RuntimeError(f"Failed to generate audio for text '{text[:50]}...': {e}") from e
