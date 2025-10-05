"""
An implementation-agnostic AnkiConnect sync module.

Public API:
- dataclass note field models can mark ID fields via metadata={"anki_id": True}
- sync_anki_cards(deck_name: str, note_type: str, cards: Sequence[Any], anki_connect_url: str = "http://127.0.0.1:8765")

Usage example:

from dataclasses import dataclass, field
from anki.anki_sync import sync_anki_cards, ANKI_ID

@dataclass
class BasicNote:
    Front: str = ANKI_ID()  # mark as identity field
    Back: str = ""

cards = [
    BasicNote(Front="What is 2+2?", Back="4"),
    BasicNote(Front="Capital of France?", Back="Paris"),
]

sync_anki_cards(deck_name="My Deck", note_type="Basic", cards=cards)

Notes:
- This module uses only stdlib (urllib) to talk to AnkiConnect.
- It does NOT depend on existing pipeline models.
"""
from __future__ import annotations

from dataclasses import MISSING, Field, fields as dataclass_fields, is_dataclass, field, make_dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import json
import sys
import urllib.request

from anki.anki_sync.vocabulary_improved import create_vocabulary_improved_model_payload

ANKI_CONNECT_VERSION = 6


def anki_id(default: Any = ""):
    """Helper to mark a dataclass field as identity field for Anki lookups.

    Example:
        @dataclass
        class Basic:
            Front: str = ANKI_ID()
            Back: str = ""
    """
    return field(default=default, metadata={"anki_id": True})


def _escape_anki_query_value(value: str) -> str:
    """Escape value for Anki search query (wrap in quotes and escape quotes)."""
    if value is None:
        value = ""
    s = str(value)
    s = s.replace('"', '\\"')
    return f'"{s}"'


def _extract_fields_and_ids(card: Any) -> Tuple[Mapping[str, Any], List[Tuple[str, Any]]]:
    """Extract all field values and a list of (field_name, value) for ID-marked fields.

    Supports dataclass instances or generic mappings/objects with __dict__.
    """
    if is_dataclass(card):
        values: Dict[str, Any] = {}
        ids: List[Tuple[str, Any]] = []
        for f in dataclass_fields(card):
            val = getattr(card, f.name)
            values[f.name] = val
            if f.metadata.get("anki_id"):
                ids.append((f.name, val))
        return values, ids

    # If it's a mapping-like object
    if isinstance(card, Mapping):
        values = dict(card)
        ids = [(k, v) for k, v in values.items() if getattr(v, "anki_id", False)]  # unlikely path
        # No metadata in plain mapping; users must pass dataclasses to use ID fields.
        return values, ids

    # Fallback to object attributes
    if hasattr(card, "__dict__"):
        values = dict(vars(card))
        ids = []
        return values, ids

    raise TypeError("Unsupported card type: expected dataclass, mapping, or object with __dict__")


class AnkiConnectClient:
    def __init__(self, url: str = "http://127.0.0.1:8765") -> None:
        self.url = url

    def invoke(self, action: str, params: Mapping[str, Any] | None = None) -> Any:
        payload = {
            "action": action,
            "version": ANKI_CONNECT_VERSION,
        }
        if params:
            payload["params"] = params
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            parsed = json.loads(raw.decode("utf-8"))
        if parsed.get("error") is not None:
            raise RuntimeError(f"AnkiConnect error on action '{action}': {parsed['error']}")
        return parsed.get("result")

    def create_deck(self, name: str) -> None:
        self.invoke("createDeck", {"deck": name})

    def create_model(self, model_payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Create a new note type (model) in Anki.

        Args:
            model_payload: Dictionary with modelName, inOrderFields, css, and cardTemplates

        Returns:
            Result from AnkiConnect (usually the model object or None if it already exists)
        """
        return self.invoke("createModel", model_payload)

    def model_names(self) -> List[str]:
        """Get list of all note type names."""
        return list(self.invoke("modelNames") or [])

    def find_notes(self, query: str) -> List[int]:
        return list(self.invoke("findNotes", {"query": query}) or [])

    def add_notes(self, notes: Sequence[Mapping[str, Any]]) -> List[int | None]:
        return list(self.invoke("addNotes", {"notes": list(notes)}) or [])

    def store_media_file(self, filename: str, data: bytes) -> None:
        """Store media file in Anki's media collection.

        Args:
            filename: Name of the file (e.g., "hash.mp3")
            data: Binary file content

        Raises:
            RuntimeError: If storing media fails
        """
        import base64

        params = {
            "filename": filename,
            "data": base64.b64encode(data).decode("utf-8"),
        }
        self.invoke("storeMediaFile", params)


class SyncResult:
    def __init__(self) -> None:
        self.added: int = 0
        self.skipped_existing: int = 0
        self.failures: List[str] = []

    def __repr__(self) -> str:
        return f"SyncResult(added={self.added}, skipped_existing={self.skipped_existing}, failures={len(self.failures)})"


def ensure_vocabulary_improved_model(client: AnkiConnectClient) -> None:
    """Ensure the Vocabulary Improved note type exists in Anki.

    Args:
        client: AnkiConnect client instance

    Raises:
        RuntimeError: If model creation fails
    """
    # Check if model already exists
    existing_models = client.model_names()
    if "Vocabulary Improved" in existing_models:
        return  # Model already exists

    # Create the model
    model_payload = create_vocabulary_improved_model_payload()
    try:
        client.create_model(model_payload)
    except RuntimeError as e:
        # If error contains "Model name already exists", it's fine
        if "already exists" not in str(e).lower():
            raise


def sync_anki_cards(
    deck_name: str,
    note_type: str,
    cards: Sequence[Any],
    anki_connect_url: str = "http://127.0.0.1:8765",
) -> SyncResult:
    """Sync cards with local Anki via AnkiConnect.

    - Cards should be dataclass instances. Mark one or more fields as identity using ANKI_ID().
    - For each card we search by deck, note type and all ID fields; if found, we print a warning and continue.
      If not found, we add a new note to the specified deck with the provided fields.

    Returns SyncResult with counts and failures captured.
    """
    client = AnkiConnectClient(anki_connect_url)
    result = SyncResult()

    # Preflight: check AnkiConnect availability. If unreachable, warn and skip syncing gracefully.
    try:
        client.invoke("version")
    except Exception as e:
        print(
            f"[anki-sync] WARNING: Could not connect to AnkiConnect at {anki_connect_url}: {e}. Skipping sync.\n"
            f"Start Anki with the AnkiConnect add-on enabled, or set ANKI_CONNECT_URL.",
            file=sys.stderr,
        )
        result.failures.append(f"AnkiConnect unreachable at {anki_connect_url}")
        return result

    # Ensure Vocabulary Improved model exists if using that note type
    if note_type == "Vocabulary Improved":
        try:
            ensure_vocabulary_improved_model(client)
        except RuntimeError as e:
            print(f"[anki-sync] WARNING: Failed to create Vocabulary Improved model: {e}", file=sys.stderr)

    # Ensure deck exists (create if not)
    try:
        client.create_deck(deck_name)
    except RuntimeError as e:
        # If deck exists, AnkiConnect returns no error; if an error still happens, record and continue
        pass

    for idx, card in enumerate(cards):
        try:
            values, ids = _extract_fields_and_ids(card)
            if not values:
                print(f"[anki-sync] Warning: empty values for card #{idx+1}; skipping", file=sys.stderr)
                result.skipped_existing += 1
                continue
            if not ids:
                raise ValueError("No ID fields provided. Use ANKI_ID() in your dataclass to mark identity fields.")

            # Build search query
            parts = [
                f'deck:{_escape_anki_query_value(deck_name)}',
                f'note:{_escape_anki_query_value(note_type)}',
            ]
            for fname, fval in ids:
                parts.append(f'{fname}:{_escape_anki_query_value(str(fval))}')

            query = " ".join(parts)

            # Check if note already exists
            try:
                existing_notes = client.find_notes(query)
                if existing_notes:
                    # Note already exists, skip it
                    result.skipped_existing += 1
                    continue
            except RuntimeError as e:
                # If search fails, log but continue to try adding
                print(f"[anki-sync] Warning: search query failed for card #{idx+1}: {e}", file=sys.stderr)

            # Prepare note payload
            # Convert all values to strings as Anki fields are strings
            field_values: Dict[str, str] = {k: "" if v is None else str(v) for k, v in values.items()}
            note_payload = {
                "deckName": deck_name,
                "modelName": note_type,
                "fields": field_values,
                "options": {
                    # We do the dedup ourselves, so allow duplicates to avoid Anki's front-field-only rules.
                    "allowDuplicate": True
                },
            }

            try:
                add_result = client.add_notes([note_payload])
                if not add_result or add_result[0] is None:
                    raise RuntimeError("addNotes returned no note id")
                result.added += 1
            except RuntimeError as e:
                # Try to create deck and retry once in case of missing deck
                client.create_deck(deck_name)
                add_result = client.add_notes([note_payload])
                if not add_result or add_result[0] is None:
                    raise
                result.added += 1

        except Exception as e:
            msg = f"Card #{idx+1} failed: {e}"
            print(f"[anki-sync] ERROR: {msg}", file=sys.stderr)
            result.failures.append(msg)

    return result
