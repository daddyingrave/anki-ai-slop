from __future__ import annotations

from pathlib import Path
from typing import Dict

PROMPTS_DIR = Path(__file__).with_name("prompts")
STEP_DIR = PROMPTS_DIR / "steps"


def load_prompt(filename: str) -> str:
    """Load a prompt file from the local prompts directory."""
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def _load_fragment(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _assemble(template_path: Path, tokens: Dict[str, str]) -> str:
    text = template_path.read_text(encoding="utf-8")
    for k, v in tokens.items():
        # Escape curly braces in fragment values so LangChain doesn't treat them as
        # template variables (e.g., examples like { Front: string, Back: string }).
        v_escaped = v.replace("{", "{{").replace("}", "}}")
        text = text.replace(f"{{{{{k}}}}}", v_escaped)
    return text


def build_step(step_name: str) -> Dict[str, str]:
    """Build system + human prompts for a pipeline step.

    Looks for templates under prompts/steps/{step_name}.system.txt and .human.txt, then replaces
    only the shared ANKI_RULES fragment.
    """
    tokens = {
        "ANKI_RULES": _load_fragment("anki_card_rules.shared.txt"),
    }
    system_tmpl = STEP_DIR / f"{step_name}.system.txt"
    human_tmpl = STEP_DIR / f"{step_name}.human.txt"
    if not system_tmpl.exists() or not human_tmpl.exists():
        raise FileNotFoundError(f"Step templates not found for {step_name}")
    system = _assemble(system_tmpl, tokens)
    human = _assemble(human_tmpl, tokens)
    return {"system": system, "human": human}


__all__ = ["load_prompt", "build_step"]
