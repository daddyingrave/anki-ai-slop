"""
Validators for vocabulary cards.
"""
from __future__ import annotations

from .models import VocabularyCard, VocabularyDeck


def validate_card(card: VocabularyCard) -> None:
    """Validate a vocabulary card.

    Args:
        card: The card to validate

    Raises:
        ValueError: If validation fails
    """
    # Check required fields
    if not card.card_id or not card.card_id.strip():
        raise ValueError("Card ID cannot be empty")

    if not card.english_lemma or not card.english_lemma.strip():
        raise ValueError(f"Card {card.card_id}: English lemma cannot be empty")

    if not card.english_original_word or not card.english_original_word.strip():
        raise ValueError(f"Card {card.card_id}: English original word cannot be empty")

    if not card.english_context or not card.english_context.strip():
        raise ValueError(f"Card {card.card_id}: English context cannot be empty")

    if not card.part_of_speech or not card.part_of_speech.strip():
        raise ValueError(f"Card {card.card_id}: Part of speech cannot be empty")

    # Check that translations exist
    if not card.russian_word_translation or not card.russian_word_translation.strip():
        raise ValueError(f"Card {card.card_id}: Russian word translation cannot be empty")

    if not card.spanish_word_translation or not card.spanish_word_translation.strip():
        raise ValueError(f"Card {card.card_id}: Spanish word translation cannot be empty")

    # Check that context is highlighted
    if "<b>" not in card.english_context or "</b>" not in card.english_context:
        raise ValueError(f"Card {card.card_id}: English context must contain highlighted word in <b> tags")


def validate_deck(deck: VocabularyDeck) -> None:
    """Validate a vocabulary deck.

    Args:
        deck: The deck to validate

    Raises:
        ValueError: If validation fails
    """
    if not deck.cards:
        raise ValueError("Deck cannot be empty")

    # Validate each card
    for card in deck.cards:
        validate_card(card)

    # Check for duplicate card IDs
    card_ids = [card.card_id for card in deck.cards]
    if len(card_ids) != len(set(card_ids)):
        duplicates = [cid for cid in card_ids if card_ids.count(cid) > 1]
        raise ValueError(f"Deck contains duplicate card IDs: {set(duplicates)}")


__all__ = [
    "validate_card",
    "validate_deck",
]
