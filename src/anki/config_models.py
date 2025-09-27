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


class ObsidianToAnkiPipelineConfig(BaseModel):
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


class RunConfig(BaseModel):
    """Top-level run configuration (new schema only).

    - pipelines: contains static keys for each pipeline type (currently only 'obsidian_to_anki')
    """

    # Structured pipelines (required to run)
    pipelines: Dict[str, ObsidianToAnkiPipelineConfig] | None = Field(
        default=None, description="Pipelines configuration keyed by pipeline id"
    )
