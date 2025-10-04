"""
Prompts for vocabulary card generation and translation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def build_translation_prompts() -> Dict[str, str]:
    """Build system and human prompts for word-in-context translation."""
    return {
        "system": load_prompt("translation_system"),
        "human": load_prompt("translation_human"),
    }


def build_word_translation_prompts() -> Dict[str, str]:
    """Build system and human prompts for general word translation."""
    return {
        "system": load_prompt("word_translation_system"),
        "human": load_prompt("word_translation_human"),
    }


__all__ = [
    "build_translation_prompts",
    "build_word_translation_prompts",
]
