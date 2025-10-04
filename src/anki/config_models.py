from __future__ import annotations

from pathlib import Path
from typing import Dict

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

    # Translation step configuration
    translate: StepConfig


# Backwards compatibility aliases
ObsidianToAnkiPipelineConfig = ObsidianPipelineConfig
LemmatizerToAnkiPipelineConfig = VocabularyPipelineConfig


class RunConfig(BaseModel):
    """Top-level run configuration (new schema only).

    - pipelines: contains static keys for each pipeline type
    """

    # Structured pipelines (required to run)
    pipelines: Dict[str, ObsidianPipelineConfig | VocabularyPipelineConfig] | None = Field(
        default=None, description="Pipelines configuration keyed by pipeline id"
    )
