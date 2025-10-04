from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import yaml
from pydantic import ValidationError

from .anki_sync.anki_connect import anki_id, sync_anki_cards
from .common.observability import enable_cache
from .config_models import RunConfig, ObsidianToAnkiPipelineConfig, LemmatizerToAnkiPipelineConfig
from .obsidian_to_anki.chains import build_deck_pipeline
from .obsidian_to_anki.models import AnkiDeck
from .lemmatizer_to_anki.chains import generate_vocabulary_card
from .lemmatizer_to_anki.models import vocabulary_card_to_note, VocabularyCard


def _require_google_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "Please export it before running, e.g.:\n"
            "  export GOOGLE_API_KEY=your_key_here"
        )


def _maybe_write(path: Optional[Path], data: object) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _discover_note_files(vault_dir: Path, notes_path: str) -> List[Path]:
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


@dataclass
class BasicNote:
    Front: str = anki_id()
    Back: str = ""


def run_from_config(pipeline_name: str, config_path: Optional[Path] = None) -> None:
    """Run a selected pipeline based on a YAML configuration file and then sync to Anki.

    If config_path is None, defaults to ./config.yaml in the current working directory.
    """
    # Initialize caching and tracing before any LLM usage
    enable_cache(Path.cwd() / "cache.db")
    _require_google_key()

    path = config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    try:
        cfg = RunConfig.model_validate(raw)
    except ValidationError as ve:
        raise SystemExit(f"Invalid configuration in {path}:\n{ve}")

    if not cfg.pipelines:
        raise SystemExit("Missing pipelines configuration")

    if pipeline_name == "obsidian_to_anki":
        pipeline_cfg: ObsidianToAnkiPipelineConfig | None = cfg.pipelines.get("obsidian_to_anki")
        if pipeline_cfg is None:
            raise SystemExit("Missing pipelines.obsidian_to_anki configuration after normalization")

        vault_dir = Path(pipeline_cfg.vault_dir)
        if not vault_dir.exists() or not vault_dir.is_dir():
            raise SystemExit(f"vault_dir does not exist or is not a directory: {vault_dir}")

        note_files = _discover_note_files(vault_dir, pipeline_cfg.notes_path)
        multiple = len(note_files) > 1

        # After pipeline: sync to Anki via AnkiConnect
        note_type = os.getenv("ANKI_NOTE_TYPE", "Basic")
        anki_url = os.getenv("ANKI_CONNECT_URL", "http://127.0.0.1:8765")

        all_decks: List[AnkiDeck] = []
        for nf in note_files:
            article = nf.read_text(encoding="utf-8")
            deck: AnkiDeck = build_deck_pipeline(article, pipeline=pipeline_cfg)

            deck_name = _derive_deck_name(vault_dir, nf)

            notes = [BasicNote(Front=c.Front, Back=c.Back) for c in deck.cards]
            sync_anki_cards(
                deck_name=deck_name,
                note_type=note_type,
                cards=notes,
                anki_connect_url=anki_url,
            )
            all_decks.append(deck)

            Path(f"out/{deck_name}.json").write_text(
                json.dumps(deck.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8")

    elif pipeline_name == "lemmatizer_to_anki":
        from .lemmatizer import LemmaExtractor, LanguageMnemonic, ModelType

        pipeline_cfg: LemmatizerToAnkiPipelineConfig | None = cfg.pipelines.get("lemmatizer_to_anki")
        if pipeline_cfg is None:
            raise SystemExit("Missing pipelines.lemmatizer_to_anki configuration after normalization")

        input_file = Path(pipeline_cfg.input_file)
        if not input_file.exists() or not input_file.is_file():
            raise SystemExit(f"input_file does not exist or is not a file: {input_file}")

        # Get phrasal verbs file path if provided
        phrasal_verbs_path = None
        if pipeline_cfg.phrasal_verbs_file:
            pv_file = Path(pipeline_cfg.phrasal_verbs_file)
            if pv_file.exists() and pv_file.is_file():
                phrasal_verbs_path = str(pv_file)
            else:
                print(f"Warning: phrasal_verbs_file not found: {pv_file}")

        # Step 1: Extract lemmas using the lemmatizer
        print(f"Processing file with lemmatizer: {input_file}")
        print(f"Language: {pipeline_cfg.language}, Model: {pipeline_cfg.model_type}")

        try:
            lang_enum = LanguageMnemonic(pipeline_cfg.language)
        except ValueError:
            raise SystemExit(f"Invalid language: {pipeline_cfg.language}")

        try:
            model_type_map = {
                "EFFICIENT": ModelType.EFFICIENT,
                "ACCURATE": ModelType.ACCURATE,
                "TRANSFORMER": ModelType.TRANSFORMER,
            }
            model_enum = model_type_map[pipeline_cfg.model_type.upper()]
        except KeyError:
            raise SystemExit(f"Invalid model_type: {pipeline_cfg.model_type}")

        extractor = LemmaExtractor(lang_enum, model_enum)
        lemma_map, phrasal_verb_map, text = extractor.process_file(
            str(input_file),
            phrasal_verbs_path
        )

        print(f"Extracted {len(lemma_map)} lemmas and {len(phrasal_verb_map)} phrasal verbs")

        # Step 2: Generate vocabulary cards with translations
        cards: List[VocabularyCard] = []
        total_lemmas = len(lemma_map)

        print(f"Generating vocabulary cards with translations...")
        for idx, (lemma, entries) in enumerate(lemma_map.items(), 1):
            if not entries:
                continue

            first_entry = entries[0]

            try:
                card = generate_vocabulary_card(
                    lemma=lemma,
                    original_word=first_entry["original_word"],
                    context=first_entry["sentence"],
                    part_of_speech=first_entry["part_of_speech"],
                    step=pipeline_cfg.translate,
                )
                cards.append(card)
                if idx % 10 == 0 or idx == total_lemmas:
                    print(f"  Progress: {idx}/{total_lemmas} cards generated")
            except Exception as e:
                print(f"  Error generating card for '{lemma}': {e}")
                continue

        # Process phrasal verbs similarly
        for pv_key, entries in phrasal_verb_map.items():
            if not entries:
                continue

            first_entry = entries[0]

            try:
                card = generate_vocabulary_card(
                    lemma=pv_key,
                    original_word=first_entry["original_text"],
                    context=first_entry["sentence"],
                    part_of_speech="phrasal verb",
                    step=pipeline_cfg.translate,
                )
                cards.append(card)
            except Exception as e:
                print(f"  Error generating card for phrasal verb '{pv_key}': {e}")
                continue

        print(f"Generated {len(cards)} vocabulary cards total")

        # Step 3: Sync to Anki
        anki_url = os.getenv("ANKI_CONNECT_URL", "http://127.0.0.1:8765")
        note_type = "Vocabulary Improved"

        notes = [vocabulary_card_to_note(card) for card in cards]

        print(f"Syncing {len(notes)} cards to Anki deck: {pipeline_cfg.deck_name}")
        result = sync_anki_cards(
            deck_name=pipeline_cfg.deck_name,
            note_type=note_type,
            cards=notes,
            anki_connect_url=anki_url,
        )

        print(f"Sync complete: {result.added} added, {result.skipped_existing} skipped, {len(result.failures)} failures")

        # Save cards to JSON for reference
        output_dir = Path("out")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{pipeline_cfg.deck_name}_vocabulary.json"
        output_file.write_text(
            json.dumps([card.model_dump() for card in cards], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"Cards saved to: {output_file}")

    else:
        raise SystemExit(f"Unknown pipeline: {pipeline_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipelines and sync to Anki")
    parser.add_argument(
        "--pipeline-name",
        required=True,
        help="Pipeline to run (e.g., obsidian_to_anki, lemmatizer_to_anki)"
    )
    parser.add_argument(
        "--config",
        required=False,
        default="config.yaml",
        help="Path to YAML config (defaults to ./config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    run_from_config(pipeline_name=args.pipeline_name, config_path=config_path)
