"""
Prompts for vocabulary card generation and translation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

PROMPTS_DIR = Path(__file__).parent / "prompts"
STEPS_DIR = PROMPTS_DIR / "steps"


def _load_prompt(file_path: Path) -> str:
    """Load a prompt file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def _load_shared_rules() -> str:
    """Load shared translation rules."""
    return _load_prompt(PROMPTS_DIR / "translation_rules.shared.txt")


def _inject_rules(template: str, rules: str) -> str:
    """Inject shared rules into template."""
    return template.replace("{{TRANSLATION_RULES}}", rules)


def build_step_prompts(step_name: str) -> Dict[str, str]:
    """Build system and human prompts for a step, with shared rules injected."""
    system_file = STEPS_DIR / f"{step_name}.system.txt"
    human_file = STEPS_DIR / f"{step_name}.human.txt"

    if not system_file.exists() or not human_file.exists():
        raise FileNotFoundError(f"Step templates not found for {step_name}")

    system_template = _load_prompt(system_file)
    human_template = _load_prompt(human_file)
    shared_rules = _load_shared_rules()

    return {
        "system": _inject_rules(system_template, shared_rules),
        "human": _inject_rules(human_template, shared_rules),
    }


# Convenience functions for specific steps
def build_ctx_translation_prompts() -> Dict[str, str]:
    """Build system and human prompts for context translation (words in sentences)."""
    return build_step_prompts("1_ctx_translation")


def build_general_translation_prompts() -> Dict[str, str]:
    """Build system and human prompts for general word translation."""
    return build_step_prompts("1_general_translation")


def build_ctx_review_prompts() -> Dict[str, str]:
    """Build system and human prompts for context translation review."""
    return build_step_prompts("2_review_ctx_translation")


def build_general_review_prompts() -> Dict[str, str]:
    """Build system and human prompts for general translation review."""
    return build_step_prompts("2_review_general_translation")
