# Anki AI slop

AI utilities to simplify some Anki interactions. Provides multiple pipelines powered by Google Gemini.

## Pipelines

### 1. Obsidian → Anki
Converts Markdown notes from your Obsidian vault into Anki flashcards.

**Quick summary:**
- Reads Markdown notes from your Obsidian vault
- Generates Front/Back cards with an LLM
- Can sync cards to Anki via AnkiConnect

### 2. Lemmatizer → Anki
Extracts vocabulary from text and creates multilingual vocabulary cards with translations.

**Quick summary:**
- Processes text files and extracts lemmas (word base forms) and phrasal verbs
- Generates vocabulary cards with context-aware translations to Russian and Spanish
- Creates the "Vocabulary Improved" note type with multiple card templates
- Supports multiple languages (EN, DE, FR, ES, IT, RU, JA, ZH)

## Requirements

- Python 3.13+
- uv (https://docs.astral.sh/uv/)
- GOOGLE_API_KEY set in your environment
- Anki desktop with the AnkiConnect add-on running
- For lemmatizer: spaCy language models (auto-downloaded on first use)

## Quick Start

### Installation
```bash
uv sync
```

### Obsidian → Anki Pipeline
1) Copy `config.yaml` from this repo and adjust `vault_dir` and `notes_path` for your vault
2) Run:
```bash
uv run -m anki --pipeline-name obsidian_to_anki --config ./config.yaml
```

### Lemmatizer → Anki Pipeline
1) Create a text file with the content you want to learn (e.g., `text.txt`)
2) Configure in `config.yaml`:
```yaml
pipelines:
  lemmatizer_to_anki:
    input_file: text.txt
    language: EN
    model_type: ACCURATE
    deck_name: Vocabulary::My Deck
    phrasal_verbs_file: src/lemmatizer/phrasal_verbs.csv
    translate:
      model: models/gemini-2.0-flash-lite
      temperature: 0.0
      max_retries: 3
      backoff_initial_seconds: 0.5
      backoff_multiplier: 2.0
```
3) Run:
```bash
export GOOGLE_API_KEY=your_key_here
uv run -m anki --pipeline-name lemmatizer_to_anki --config ./config.yaml
```

The pipeline will:
- Extract all unique words and their lemmas from the text
- Generate translations to Russian and Spanish
- Create Anki cards with context sentences
- Sync to your specified deck

## Standalone Utilities

The project also includes standalone utilities:

- **PDF to CSV converter** (`src/pv_pdf_to_csv`): Converts phrasal verb PDFs to CSV
- **Subtitle processor** (`src/subs`): Extracts text from SRT subtitle files

That's it — minimal setup to generate cards and sync them to Anki.
