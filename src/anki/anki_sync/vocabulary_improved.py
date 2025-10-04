"""
Helper functions for creating and managing the "Vocabulary Improved" note type in Anki.

This note type is used for vocabulary cards with translations to multiple languages.
"""
from __future__ import annotations

from typing import Dict, List, Any


def get_vocabulary_improved_fields() -> List[str]:
    """Get the list of fields for the Vocabulary Improved note type."""
    return [
        "ID",
        "PartOfSpeech",
        "EnglishLemma",
        "EnglishOriginalWord",
        "EnglishContext",
        "RussianLemmaUsageTranslation",
        "SpanishLemmaUsageTranslation",
        "RussianContextTranslation",
        "RussianCommonTranslations",
        "SpanishCommonTranslations",
        "SpanishAudio",
        "EnglishAudio",
        "EnglishContextAudio",
        "Image",
    ]


def get_vocabulary_improved_css() -> str:
    """Get the CSS styles for Vocabulary Improved cards."""
    return """
.card {
  font-family: arial;
  font-size: 20px;
  text-align: center;
  color: black;
  background-color: white;
}

.highlight {
  font-weight: bold;
  color: #2196F3;
}

.context {
  font-style: italic;
  color: #555;
  margin-top: 10px;
}

.translation {
  margin-top: 15px;
  font-size: 18px;
}

.common {
  color: #666;
  font-size: 16px;
  margin-top: 10px;
}

.pos {
  color: #999;
  font-size: 14px;
  font-style: italic;
}
"""


def get_vocabulary_improved_templates() -> List[Dict[str, str]]:
    """Get the card templates for Vocabulary Improved note type."""
    return [
        {
            "Name": "English → Russian",
            "Front": """
<div class="card">
  <div class="pos">{{PartOfSpeech}}</div>
  <div>{{EnglishLemma}}</div>
  <div class="context">{{EnglishContext}}</div>
  <div>{{EnglishAudio}}</div>
  <div>{{EnglishContextAudio}}</div>
</div>
""",
            "Back": """
<div class="card">
  <div class="pos">{{PartOfSpeech}}</div>
  <div>{{EnglishLemma}}</div>
  <div class="context">{{EnglishContext}}</div>
  <div>{{EnglishAudio}}</div>
  <div>{{EnglishContextAudio}}</div>
  <hr>
  <div class="translation">{{RussianLemmaUsageTranslation}}</div>
  <div class="context translation">{{RussianContextTranslation}}</div>
  <div class="common">Common: {{RussianCommonTranslations}}</div>
</div>
"""
        },
        {
            "Name": "Russian → English",
            "Front": """
<div class="card">
  <div class="pos">{{PartOfSpeech}}</div>
  <div>{{RussianLemmaUsageTranslation}}</div>
  <div class="context">{{RussianContextTranslation}}</div>
</div>
""",
            "Back": """
<div class="card">
  <div class="pos">{{PartOfSpeech}}</div>
  <div>{{RussianLemmaUsageTranslation}}</div>
  <div class="context">{{RussianContextTranslation}}</div>
  <hr>
  <div class="translation">{{EnglishLemma}}</div>
  <div class="context translation">{{EnglishContext}}</div>
  <div>{{EnglishAudio}}</div>
  <div>{{EnglishContextAudio}}</div>
  <div class="common">Original: {{EnglishOriginalWord}}</div>
</div>
"""
        },
        {
            "Name": "Russian → English (Type)",
            "Front": """
<div class="card">
  <div class="pos">{{PartOfSpeech}}</div>
  <div>{{RussianLemmaUsageTranslation}}</div>
  <div class="context">{{RussianContextTranslation}}</div>
  <br>
  {{type:EnglishLemma}}
</div>
""",
            "Back": """
<div class="card">
  <div class="pos">{{PartOfSpeech}}</div>
  <div>{{RussianLemmaUsageTranslation}}</div>
  <div class="context">{{RussianContextTranslation}}</div>
  <hr>
  <div class="translation">{{EnglishLemma}}</div>
  <div class="context translation">{{EnglishContext}}</div>
  <div>{{EnglishAudio}}</div>
  <div>{{EnglishContextAudio}}</div>
</div>
"""
        },
    ]


def create_vocabulary_improved_model_payload() -> Dict[str, Any]:
    """Create the payload for creating the Vocabulary Improved note type via AnkiConnect.

    Returns:
        Dictionary payload for the createModel AnkiConnect action
    """
    return {
        "modelName": "Vocabulary Improved",
        "inOrderFields": get_vocabulary_improved_fields(),
        "css": get_vocabulary_improved_css(),
        "cardTemplates": get_vocabulary_improved_templates(),
    }
