from __future__ import annotations

import re
from typing import Set

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
    re.compile(r"<[^>]+>"),  # NOTE: we allow simple HTML tags in our cards; do not enforce this.
]

_CODE_FENCE = re.compile(r"```")


def _has_forbidden_math(text: str) -> bool:
    return any(p.search(text) for p in _MATH_FORBIDDEN)


def _has_forbidden_md_light(text: str) -> bool:
    # Only enforce Markdown headings/images/links; HTML tags are allowed in cards.
    for p in _MD_FORBIDDEN[:3]:  # headings, images, links
        if p.search(text):
            return True
    return False


def _non_empty(s: str) -> bool:
    return bool(s and s.strip())


def _normalize_front(s: str) -> str:
    return " ".join(s.lower().split()) if s else ""


class ValidationError(RuntimeError):
    pass


def validate_deck(deck: AnkiDeck) -> None:
    # Ensure cards only have Front/Back and are non-empty, and basic formatting compliance
    texts = []
    seen_fronts: Set[str] = set()
    for c in deck.cards:
        if not _non_empty(c.Front) or not _non_empty(c.Back):
            raise ValidationError("Empty Front/Back in a card")
        nf = _normalize_front(c.Front)
        if nf in seen_fronts:
            raise ValidationError("Duplicate or near-duplicate Front detected")
        seen_fronts.add(nf)
        texts.extend([c.Front, c.Back])

    # Ensure basic formatting compliance
    for t in texts:
        if _CODE_FENCE.search(t):
            raise ValidationError("Markdown code fences ``` are not allowed; use <pre><code>...</code></pre>.")
        if _has_forbidden_math(t):
            raise ValidationError("Forbidden math delimiters detected ($...$ or backticked math). Use \\( ... \\) or \\[ ... \\].")
        if _has_forbidden_md_light(t):
            raise ValidationError("Markdown headings/links/images are not allowed in card fields.")

    # Ensure reviewer notes/tags/rationale are not leaked into final fields (heuristic)
    leak_patterns = [
        re.compile(r"\b(rationale|reviewer[_ ]?note|tags?)\b", re.IGNORECASE)
    ]
    for t in texts:
        if any(p.search(t) for p in leak_patterns):
            raise ValidationError("Reviewer metadata appears to leak into final deck")
