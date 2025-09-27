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
from .config_models import RunConfig, ObsidianToAnkiPipelineConfig
from .obsidian_to_anki.chains import build_deck_pipeline
from .obsidian_to_anki.models import AnkiDeck


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

    else:
        raise SystemExit(f"Unknown pipeline: {pipeline_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipelines and sync to Anki")
    parser.add_argument("--pipeline-name", required=True, help="Pipeline to run (e.g., obsidian_to_anki)")
    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Path to YAML config (defaults to ./config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    run_from_config(pipeline_name=args.pipeline_name, config_path=config_path)
