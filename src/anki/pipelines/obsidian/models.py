from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AnkiCard(BaseModel):
    """An Anki card representation."""

    Front: str = Field(..., description="Front of the card (question/prompt)")
    Back: str = Field(..., description="Back of the card (answer/explanation)")


class AnkiDeck(BaseModel):
    """A collection of Anki cards."""

    cards: List[AnkiCard] = Field(..., description="List of Anki cards.")
