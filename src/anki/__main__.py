from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from anki.anki_sync.anki_connect import anki_id, sync_anki_cards, AnkiConnectClient
from anki.common.logging_config import setup_logging
from anki.common.observability import enable_cache
from anki.common.tts import TTSClient, TTSVoiceConfig
from anki.config_models import RunConfig, ObsidianPipelineConfig, VocabularyPipelineConfig
from anki.pipelines.obsidian.chains import build_obsidian_pipeline
from anki.pipelines.vocabulary.chains import build_vocabulary_pipeline
from anki.pipelines.vocabulary.models import vocabulary_card_to_note

logger = logging.getLogger(__name__)


def _require_google_key() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "Please export it before running, e.g.:\n"
            "  export GOOGLE_API_KEY=your_key_here"
        )


@dataclass
class BasicNote:
    Front: str = anki_id()
    Back: str = ""


def run_from_config(pipeline_name: str, config_path: Optional[Path] = None, log_level: str = "INFO") -> None:
    """Run a selected pipeline based on a YAML configuration file and then sync to Anki.

    If config_path is None, defaults to ./config.yaml in the current working directory.

    Args:
        pipeline_name: Name of the pipeline to run
        config_path: Path to the configuration file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Initialize logging first
    setup_logging(log_level)
    logger.info(f"Starting pipeline: {pipeline_name}", extra={"pipeline": pipeline_name, "log_level": log_level})

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
        pipeline_cfg_raw = cfg.pipelines.get("obsidian_to_anki")
        if pipeline_cfg_raw is None:
            raise SystemExit("Missing pipelines.obsidian_to_anki configuration after normalization")

        # Parse raw dict to typed config
        pipeline_cfg = ObsidianPipelineConfig(**pipeline_cfg_raw)

        # Run the obsidian pipeline
        deck_results = build_obsidian_pipeline(
            vault_dir=pipeline_cfg.vault_dir,
            notes_path=pipeline_cfg.notes_path,
            pipeline_cfg=pipeline_cfg,
        )

        # Sync to Anki via AnkiConnect
        note_type = os.getenv("ANKI_NOTE_TYPE", "Basic")
        anki_url = os.getenv("ANKI_CONNECT_URL", "http://127.0.0.1:8765")

        for deck_name, deck in deck_results:
            notes = [BasicNote(Front=c.Front, Back=c.Back) for c in deck.cards]
            sync_anki_cards(
                deck_name=deck_name,
                note_type=note_type,
                cards=notes,
                anki_connect_url=anki_url,
            )

            Path(f"out/{deck_name}.json").write_text(
                json.dumps(deck.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8")

    elif pipeline_name == "vocabulary":
        pipeline_cfg_raw = cfg.pipelines.get("vocabulary")
        if pipeline_cfg_raw is None:
            raise SystemExit("Missing pipelines.vocabulary configuration after normalization")

        # Parse raw dict to typed config
        pipeline_cfg = VocabularyPipelineConfig(**pipeline_cfg_raw)

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
                logger.warning(f"phrasal_verbs_file not found: {pv_file}", extra={"file": str(pv_file)})

        # Setup clients
        anki_url = os.getenv("ANKI_CONNECT_URL", "http://127.0.0.1:8765")
        voice_config = TTSVoiceConfig(
            model_name=pipeline_cfg.tts.model_name,
            voice_name=pipeline_cfg.tts.voice_name,
            language_code=pipeline_cfg.tts.language_code,
            speaking_rate=pipeline_cfg.tts.speaking_rate,
            pitch=pipeline_cfg.tts.pitch,
        )
        tts_client = TTSClient(cache_dir=Path(pipeline_cfg.audio_output_dir), voice_config=voice_config)
        anki_client = AnkiConnectClient(anki_url)

        # Run the vocabulary pipeline
        cards = build_vocabulary_pipeline(
            input_file=str(input_file),
            language=pipeline_cfg.language,
            model_type=pipeline_cfg.model_type,
            phrasal_verbs_file=phrasal_verbs_path,
            translate_step=pipeline_cfg.translate,
            review_step=pipeline_cfg.review,
            tts_client=tts_client,
            anki_client=anki_client,
        )

        # Sync to Anki
        note_type = "vocabulary-improved"

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
        help="Pipeline to run (e.g., obsidian_to_anki, vocabulary)"
    )
    parser.add_argument(
        "--config",
        required=False,
        default="config.yaml",
        help="Path to YAML config (defaults to ./config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        required=False,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO). Use DEBUG to see LLM prompts/responses.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    run_from_config(
        pipeline_name=args.pipeline_name,
        config_path=config_path,
        log_level=args.log_level
    )
