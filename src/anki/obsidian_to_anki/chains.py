from __future__ import annotations

import html
import re
from typing import cast

from langchain_core.prompts import ChatPromptTemplate

from .models import AnkiDeck
from .prompts import build_step
from ..common.llm import build_llm
from ..common.reliability import retry_invoke
from ..config_models import StepConfig, ObsidianToAnkiPipelineConfig


def generate_anki_deck(
        article: str,
        step: StepConfig,
) -> AnkiDeck:
    """Generate a ready-to-use Anki deck (Front/Back only) directly from the article."""
    llm = build_llm(model=step.model, temperature=step.temperature)

    step_prompts = build_step("1_generate_deck")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", step_prompts["system"]),
            ("human", step_prompts["human"]),
        ]
    )

    chain = prompt | llm.with_structured_output(AnkiDeck)
    result = cast(AnkiDeck, retry_invoke(
        chain,
        {"article": article},
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))
    if result is None:
        raise RuntimeError(
            "LLM did not return structured AnkiDeck. Check the model, prompts, and inputs."
        )
    result = _normalize_math_delimiters(result)
    return result


def review_anki_deck(
        article: str,
        deck: AnkiDeck,
        step: StepConfig,
) -> AnkiDeck:
    """Review and fix the generated Anki deck. Returns the revised deck."""
    llm = build_llm(model=step.model, temperature=step.temperature)

    step_prompts = build_step("2_review_deck")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", step_prompts["system"]),
            ("human", step_prompts["human"]),
        ]
    )

    chain = prompt | llm.with_structured_output(AnkiDeck)
    result = cast(AnkiDeck, retry_invoke(
        chain,
        {
            "article": article,
            "deck_json": deck.model_dump_json(),
        },
        max_retries=step.max_retries,
        backoff_initial_seconds=step.backoff_initial_seconds,
        backoff_multiplier=step.backoff_multiplier,
    ))
    if result is None:
        raise RuntimeError(
            "LLM did not return structured AnkiDeck during review. Check the model, prompts, and inputs."
        )
    result = _normalize_math_delimiters(result)
    from .validators import validate_deck
    validate_deck(result)
    return result


def build_obsidian_pipeline(
        vault_dir: str,
        notes_path: str,
        pipeline_cfg: ObsidianToAnkiPipelineConfig,
) -> list[tuple[str, AnkiDeck]]:
    """Complete pipeline to process Obsidian notes and generate Anki decks.

    This pipeline:
    1. Discovers note files in the vault
    2. Processes each file through the deck generation pipeline
    3. Returns a list of (deck_name, deck) tuples

    Args:
        vault_dir: Path to the Obsidian vault directory
        notes_path: Path to notes (file or directory) relative to vault_dir
        pipeline_cfg: Pipeline configuration for obsidian_to_anki

    Returns:
        List of tuples containing (deck_name, AnkiDeck)
    """
    from pathlib import Path

    vault_path = Path(vault_dir)
    if not vault_path.exists() or not vault_path.is_dir():
        raise ValueError(f"vault_dir does not exist or is not a directory: {vault_dir}")

    # Discover note files
    def _discover_note_files(vault_dir: Path, notes_path: str) -> list[Path]:
        base = Path(vault_dir)
        candidate = base / notes_path
        if candidate.is_file():
            return [candidate]
        if candidate.is_dir():
            return sorted(p for p in candidate.rglob("*.md") if p.is_file())
        raise FileNotFoundError(f"notes_path not found under vault_dir: {candidate}")

    def _derive_deck_name(vault_dir: Path, note_file: Path) -> str:
        rel = note_file.relative_to(vault_dir)
        parts = list(rel.parts)
        if not parts:
            return note_file.stem
        parts[-1] = Path(parts[-1]).stem
        return "::".join(parts)

    note_files = _discover_note_files(vault_path, notes_path)

    results: list[tuple[str, AnkiDeck]] = []
    for nf in note_files:
        article = nf.read_text(encoding="utf-8")

        initial_deck = generate_anki_deck(
            article,
            step=pipeline.generate,
        )
        reviewed_deck = review_anki_deck(
            article,
            initial_deck,
            step=pipeline.review,
        )

        deck: AnkiDeck = reviewed_deck
        deck_name = _derive_deck_name(vault_path, nf)
        results.append((deck_name, deck))

    return results


def _normalize_math_delimiters(deck: AnkiDeck) -> AnkiDeck:
    """Normalize card text fields for MathJax and formatting cleanliness.

    - Collapse double-escaped MathJax delimiters to single escapes.
    - Convert fenced code blocks (``` ... ```) to <pre><code>...</code></pre>.
    - Convert inline code (`...`) to <code>...</code>.
    - Remove any stray backtick characters.

    Context: When passing JSON into prompts (e.g., during review), backslashes in strings
    appear escaped (\\). Some models mirror those escapes into their output, leading to
    double-escaped MathJax delimiters like "\\\\(" in the structured result. This breaks
    MathJax rendering in Anki. Some models also insist on Markdown backticks; we convert or remove
    them to keep output HTML/MathJax compliant.
    """
    # Precompile lightweight regexes
    fenced = re.compile(r"```[a-zA-Z0-9_\-]*\n(.*?)\n?```", re.DOTALL)
    inline = re.compile(r"`([^`]*)`")

    def _escape_html(s: str) -> str:
        # Escape minimal set to render within HTML code blocks
        return html.escape(s, quote=False)

    for c in deck.cards:
        for field in ("Front", "Back"):
            s = getattr(c, field)
            if not isinstance(s, str) or not s:
                continue
            # 1) Collapse for MathJax delimiters
            s = s.replace("\\\\(", "\\(").replace("\\\\)", "\\)")
            s = s.replace("\\\\[", "\\[").replace("\\\\]", "\\]")

            # 2) Fenced code blocks -> HTML
            def _fenced_repl(m: re.Match) -> str:
                code = m.group(1)
                return f"<pre><code>{_escape_html(code)}</code></pre>"

            s = fenced.sub(_fenced_repl, s)
            # 3) Inline code -> HTML
            s = inline.sub(lambda m: f"<code>{_escape_html(m.group(1))}</code>", s)
            # 4) Remove any stray backticks that remain
            if "`" in s:
                s = s.replace("`", "")
            setattr(c, field, s)
    return deck
