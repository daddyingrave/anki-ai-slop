from __future__ import annotations

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
    from .validators import validate_deck
    validate_deck(result)
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


def build_deck_pipeline(
        article: str,
        pipeline: ObsidianToAnkiPipelineConfig,
) -> AnkiDeck:
    """Two-stage pipeline: generate → review."""
    initial_deck = generate_anki_deck(
        article,
        step=pipeline.generate,
    )
    reviewed_deck = review_anki_deck(
        article,
        initial_deck,
        step=pipeline.review,
    )
    return reviewed_deck

def _normalize_math_delimiters(deck: AnkiDeck) -> AnkiDeck:
    """Collapse double-escaped MathJax delimiters to single escapes.

    Context: When passing JSON into prompts (e.g., during review), backslashes in strings
    appear escaped (\\). Some models mirror those escapes into their output, leading to
    double-escaped MathJax delimiters like "\\\\(" in the structured result. This breaks
    MathJax rendering in Anki. We normalize only the MathJax delimiters here.
    """
    for c in deck.cards:
        for field in ("Front", "Back"):
            s = getattr(c, field)
            if not isinstance(s, str) or not s:
                continue
            # Only collapse for MathJax delimiters
            s = s.replace("\\\\(", "\\(").replace("\\\\)", "\\)")
            s = s.replace("\\\\[", "\\[").replace("\\\\]", "\\]")
            setattr(c, field, s)
    return deck
