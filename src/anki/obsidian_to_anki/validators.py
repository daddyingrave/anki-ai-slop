from __future__ import annotations

import re

from .models import AnkiDeck


_MATH_FORBIDDEN = [
    re.compile(r"\$[^$]+\$"),  # inline $...$
    re.compile(r"`\\?\\?\(.*?\\?\\?\)`"),  # backticked math tokens
]

# Disallowed markdown (lightweight): headings, links, images, html tags
_MD_FORBIDDEN = [
    re.compile(r"^[ \t]*#", re.MULTILINE),
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),
    re.compile(r"\[[^\]]+\]\([^)]*\)"),
    re.compile(r"<[^>]+>"),
]

_CODE_FENCE = re.compile(r"```")


def _has_forbidden_math(text: str) -> bool:
    return any(p.search(text) for p in _MATH_FORBIDDEN)


def _has_forbidden_md(text: str) -> bool:
    return any(p.search(text) for p in _MD_FORBIDDEN)


def _non_empty(s: str) -> bool:
    return bool(s and s.strip())


class ValidationError(RuntimeError):
    pass


def validate_deck(deck: AnkiDeck) -> None:
    # Ensure cards only have Front/Back and are non-empty, and basic formatting compliance
    texts = []
    for c in deck.cards:
        if not _non_empty(c.Front) or not _non_empty(c.Back):
            raise ValidationError("Empty Front/Back in a card")
        texts.extend([c.Front, c.Back])

    # Ensure reviewer notes/tags/rationale are not leaked into final fields (heuristic)
    leak_patterns = [
        re.compile(r"\b(rationale|reviewer[_ ]?note|tags?)\b", re.IGNORECASE)
    ]
    for t in texts:
        if any(p.search(t) for p in leak_patterns):
            raise ValidationError("Reviewer metadata appears to leak into final deck")
