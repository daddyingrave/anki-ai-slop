"""
Helper functions for creating and managing the "vocabulary-improved" note type in Anki.

This note type is used for vocabulary cards with translations to multiple languages.
Templates are loaded from external files in the vocabulary_improved_templates directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any


# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent / "vocabulary_improved_templates"


def get_vocabulary_improved_fields() -> List[str]:
    """Get the list of fields for the vocabulary-improved note type.

    Fields are loaded from fields.txt file.
    """
    fields_file = TEMPLATES_DIR / "fields.txt"
    return [line.strip() for line in fields_file.read_text().strip().split("\n") if line.strip()]


def get_vocabulary_improved_css() -> str:
    """Get the CSS styles for vocabulary-improved cards.

    CSS is loaded from styles.css file.
    """
    css_file = TEMPLATES_DIR / "styles.css"
    return css_file.read_text()


def _load_template(filename: str) -> str:
    """Load a template file from the templates directory.

    Args:
        filename: Name of the template file to load

    Returns:
        Template content as string
    """
    template_file = TEMPLATES_DIR / filename
    return template_file.read_text()


def get_vocabulary_improved_templates() -> List[Dict[str, str]]:
    """Get the card templates for vocabulary-improved note type.

    Templates are loaded from HTML files in the templates directory.
    """
    return [
        {
            "Name": "english->russian",
            "Front": _load_template("english_russian_front.html"),
            "Back": _load_template("english_russian_back.html"),
        },
        {
            "Name": "russian->english",
            "Front": _load_template("russian_english_front.html"),
            "Back": _load_template("russian_english_back.html"),
        },
    ]


def create_vocabulary_improved_model_payload() -> Dict[str, Any]:
    """Create the payload for creating the vocabulary-improved note type via AnkiConnect.

    Returns:
        Dictionary payload for the createModel AnkiConnect action
    """
    return {
        "modelName": "vocabulary-improved",
        "inOrderFields": get_vocabulary_improved_fields(),
        "css": get_vocabulary_improved_css(),
        "cardTemplates": get_vocabulary_improved_templates(),
    }
