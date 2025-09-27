# Anki AI slop

AI utilities to simplify some Anki interactions. Currently provides a simple Obsidian → Anki pipeline powered by Google Gemini.

Quick summary:
- Reads Markdown notes from your Obsidian vault
- Generates Front/Back cards with an LLM
- Can sync cards to Anki via AnkiConnect

Requirements:
- Python 3.13+
- uv (https://docs.astral.sh/uv/)
- GOOGLE_API_KEY set in your environment
- Anki desktop with the AnkiConnect add-on running

Quick start:
1) Install dependencies: `uv sync`
2) Copy `config.yaml` from this repo and adjust `vault_dir` and `notes_path` for your vault
3) Run: `uv run -m anki --pipeline-name obsidian_to_anki --config ./config.yaml`

That’s it — minimal setup to generate cards and (optionally) sync them to Anki.
