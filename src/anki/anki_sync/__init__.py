"""Generic AnkiConnect sync module exports.

This package is implementation-agnostic and can be used by any future module
that needs to sync notes to a local Anki via AnkiConnect.

Example usage:

from dataclasses import dataclass
from anki.anki_sync import ANKI_ID, sync_anki_cards

@dataclass
class Basic:
    Front: str = ANKI_ID()
    Back: str = ""

sync_anki_cards(deck_name="My Deck", note_type="Basic", cards=[Basic("Q", "A")])
"""
from .anki_connect import anki_id, AnkiConnectClient, SyncResult, sync_anki_cards
