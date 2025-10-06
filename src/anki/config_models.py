from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field


class StepConfig(BaseModel):
    model: str = Field(..., description="Model name for this step")
    temperature: float = Field(..., description="Sampling temperature for this step")
    max_retries: int = Field(default=2, ge=0, le=5, description="Max retries for this step")
    backoff_initial_seconds: float = Field(default=0.5, gt=0, description="Initial backoff delay in seconds")
    backoff_multiplier: float = Field(default=2.0, gt=1.0, description="Exponential backoff multiplier")


class ObsidianPipelineConfig(BaseModel):
    """Configuration for the Obsidian → Anki pipeline.

    - vault_dir: absolute path to the Obsidian vault on the local filesystem
    - notes_path: relative path within the vault; may point to a single note file or a directory
    - generate/review: two-step pipeline configurations
    """

    # Discovery
    vault_dir: Path = Field(..., description="Absolute path to the Obsidian vault directory")
    notes_path: str = Field(..., description="Relative path within the vault to a note or a directory of notes")

    # Named step configurations
    generate: StepConfig
    review: StepConfig


class TTSConfig(BaseModel):
    """TTS voice configuration for Gemini Flash TTS."""
    model_name: str = Field(default="gemini-2.5-flash-tts", description="Gemini TTS model name")
    voice_name: str = Field(default="Enceladus", description="Prebuilt voice name (e.g. Enceladus, Puck, Charon, Kore)")
    language_code: str = Field(default="en-us", description="Language code")
    speaking_rate: float = Field(default=1.0, ge=0.25, le=4.0, description="Speaking rate (0.25-4.0)")
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0, description="Voice pitch (-20.0 to 20.0)")


class VocabularyPipelineConfig(BaseModel):
    """Configuration for the Vocabulary pipeline (formerly Lemmatizer → Anki).

    - input_file: path to the text file to process
    - language: language mnemonic for lemmatization (e.g., EN, DE, FR)
    - model_type: spaCy model type (EFFICIENT, ACCURATE, TRANSFORMER)
    - deck_name: name of the Anki deck to create/update
    - phrasal_verbs_file: optional path to phrasal verbs CSV file
    - translate: configuration for translation step
    """

    # Input configuration
    input_file: Path = Field(..., description="Path to the text file to process")
    language: str = Field(default="EN", description="Language mnemonic (EN, DE, FR, etc.)")
    model_type: str = Field(default="ACCURATE", description="spaCy model type (EFFICIENT, ACCURATE, TRANSFORMER)")
    deck_name: str = Field(default="Vocabulary", description="Anki deck name")
    phrasal_verbs_file: Path | None = Field(default=None, description="Optional path to phrasal verbs CSV file")
    audio_output_dir: str
    tts: TTSConfig = Field(default_factory=TTSConfig, description="TTS voice configuration")
    # Translation step configuration
    translate: StepConfig
    review: StepConfig


class RunConfig(BaseModel):
    """Top-level run configuration (new schema only).

    - pipelines: contains static keys for each pipeline type

    Note: Pipeline values are stored as Any to avoid Pydantic union validation issues.
    Cast to specific config types (ObsidianPipelineConfig, VocabularyPipelineConfig, MistralOcrConfig)
    after retrieval using .get()
    """

    # Structured pipelines (required to run)
    # Using Any instead of Union to avoid Pydantic trying to validate all union members for each value
    pipelines: Dict[str, Any] | None = Field(
        default=None, description="Pipelines configuration keyed by pipeline id"
    )
