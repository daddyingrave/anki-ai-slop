---
apply: always
---

# AI Project Guidelines

This document contains coding standards and best practices for AI assistants working on this Python project.

## 1. Import Guidelines

### ✅ DO: Use absolute imports
```python
# Good
from anki.pipelines.vocabulary.chains import build_vocabulary_pipeline
from anki.common.llm import build_llm
```

### ❌ DON'T: Use relative imports for cross-package imports
```python
# Bad
from ...common.llm import build_llm
from ..models import VocabularyCard
```

**Exception:** Relative imports are acceptable only within `__init__.py` files for re-exporting.

### ✅ DO: Place ALL imports at the top of the file
```python
# Good
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from anki.common.llm import build_llm
from anki.pipelines.vocabulary.models import VocabularyCard

def my_function():
    # Function code here
    pass
```

### ❌ DON'T: Place imports in the middle of code
```python
# Bad
def my_function():
    from anki.common.llm import build_llm  # NEVER DO THIS
    return build_llm()
```

### ❌ FORBIDDEN: Try-catch imports
```python
# ABSOLUTELY FORBIDDEN
try:
    from anki.common.llm import build_llm
except ImportError:
    from anki.common.llm_fallback import build_llm
```

**Rationale:**
- Makes dependencies unclear and hard to audit
- Breaks static analysis tools
- Creates runtime surprises
- Violates "explicit is better than implicit"

**Valid exceptions:** NONE. If you need optional dependencies, use proper dependency management.

### Import Order (follow PEP 8)
1. `__future__` imports
2. Standard library imports
3. Third-party library imports
4. Local application imports

Separate each group with a blank line.

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from anki.common.llm import build_llm
from anki.pipelines.vocabulary.models import VocabularyCard
```

## 2. Code Organization

### ✅ DO: Keep functions and classes focused
- Single Responsibility Principle
- Functions should do one thing well
- Maximum function length: ~50 lines (use judgment)

### ✅ DO: Use type hints
```python
# Good
def process_cards(cards: List[VocabularyCard], deck_name: str) -> SyncResult:
    pass
```

### ❌ DON'T: Use magic numbers or strings
```python
# Bad
if status == 200:
    pass

# Good
HTTP_OK = 200
if status == HTTP_OK:
    pass
```

## 3. Error Handling

### ✅ DO: Use specific exception types
```python
# Good
try:
    result = process_file(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
except PermissionError:
    logger.error(f"Permission denied: {path}")
```

### ❌ DON'T: Use bare except
```python
# Bad
try:
    result = process_file(path)
except:  # NEVER DO THIS
    pass
```

### ❌ DON'T: Silently swallow exceptions
```python
# Bad
try:
    result = process_file(path)
except Exception:
    pass  # Silent failure is evil
```

## 4. Naming Conventions

### Follow PEP 8:
- **Functions/variables:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private members:** `_leading_underscore`
- **Module names:** `lowercase` or `snake_case`

### ✅ DO: Use descriptive names
```python
# Good
def calculate_vocabulary_card_embeddings(cards: List[VocabularyCard]) -> np.ndarray:
    pass

# Bad
def calc(c):
    pass
```

## 5. Documentation

### ✅ DO: Write docstrings for public functions and classes
```python
def build_vocabulary_pipeline(
    input_file: str,
    language: str,
    model_type: str,
) -> List[VocabularyCard]:
    """Complete pipeline to process a text file and generate vocabulary cards.

    This pipeline:
    1. Extracts lemmas using the lemmatizer
    2. Generates vocabulary cards with translations
    3. Returns a list of VocabularyCard objects

    Args:
        input_file: Path to the input text file
        language: Language code (e.g., "EN")
        model_type: Model type ("EFFICIENT", "ACCURATE", "TRANSFORMER")

    Returns:
        List of VocabularyCard objects

    Raises:
        ValueError: If language or model_type is invalid
        FileNotFoundError: If input_file doesn't exist
    """
    pass
```

### ❌ DON'T: Write obvious or useless docstrings
```python
# Bad
def get_name():
    """Gets the name."""  # Adds no value
    pass
```

## 6. Project Structure

### Package organization:
```
src/anki/
├── pipelines/           # All pipeline implementations
│   ├── obsidian/       # Obsidian pipeline
│   └── vocabulary/     # Vocabulary pipeline
├── common/             # Shared utilities
├── anki_sync/          # Anki synchronization
└── config_models.py    # Configuration schemas
```

### ✅ DO: Keep packages cohesive
- Related functionality should be in the same package
- Avoid circular dependencies
- Use dependency injection when needed

## 7. Git Commit Guidelines

### ✅ DO: Write clear commit messages
```
Add vocabulary card deduplication in anki_connect

- Check for existing notes before adding new ones
- Use card ID field for duplicate detection
- Update SyncResult to track skipped cards
```

### ❌ DON'T: Write vague commit messages
```
fix bug
update code
wip
```

## 8. Testing Considerations

### ✅ DO: Design for testability
- Avoid global state
- Use dependency injection
- Keep side effects explicit
- Separate pure logic from I/O

### ✅ DO: Validate inputs early
```python
def process_file(path: str) -> Result:
    if not path:
        raise ValueError("path cannot be empty")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Proceed with processing
```

## 9. Performance Considerations

### ✅ DO: Be mindful of expensive operations
- Use generators for large datasets
- Avoid unnecessary copies
- Profile before optimizing

### ❌ DON'T: Premature optimization
- Write clear code first
- Optimize only when needed
- Measure before and after

## 10. Security

### ❌ DON'T: Commit secrets or credentials
- Use environment variables
- Use `.env` files (git-ignored)
- Never hardcode API keys

### ✅ DO: Validate external input
- Sanitize user input
- Validate file paths
- Check file sizes before processing

## 11. Python-Specific Best Practices

### ✅ DO: Use context managers
```python
# Good
with open(file_path) as f:
    content = f.read()
```

### ✅ DO: Use comprehensions for simple transformations
```python
# Good
squares = [x**2 for x in range(10)]

# Bad (but sometimes clearer for complex logic)
squares = []
for x in range(10):
    squares.append(x**2)
```

### ✅ DO: Use `pathlib.Path` instead of string paths
```python
# Good
from pathlib import Path
config_path = Path("config.yaml")
if config_path.exists():
    content = config_path.read_text()

# Avoid
import os
config_path = "config.yaml"
if os.path.exists(config_path):
    with open(config_path) as f:
        content = f.read()
```

### ✅ DO: Use f-strings for formatting
```python
# Good
message = f"Processing {count} cards from {deck_name}"

# Avoid
message = "Processing {} cards from {}".format(count, deck_name)
message = "Processing %s cards from %s" % (count, deck_name)
```

## 12. Common Anti-Patterns to Avoid

### ❌ Mutable default arguments
```python
# Bad
def add_card(cards=[]):  # DON'T DO THIS
    cards.append(new_card)
    return cards

# Good
def add_card(cards: List[Card] | None = None) -> List[Card]:
    if cards is None:
        cards = []
    cards.append(new_card)
    return cards
```

### ❌ Comparing to True/False/None with `==`
```python
# Bad
if flag == True:
    pass
if value == None:
    pass

# Good
if flag:
    pass
if value is None:
    pass
```

### ❌ Using `len()` to check if container is empty
```python
# Bad
if len(cards) == 0:
    pass

# Good
if not cards:
    pass
```

## 13. LLM Prompts and Schema Management

### ⚠️ IMPORTANT: Avoid hardcoded JSON examples in prompts

**Problem:** JSON schemas in prompts can become outdated when Pydantic models change.

**Bad Practice:**
```
# In prompt file:
Example output: {
  "russian": {"word_translation": "...", "context_translation": "..."},
  "spanish": {"word_translation": "..."}
}
```

**Issues:**
- Manual sync required when models change
- Silent failures (LLM learns wrong format)
- No compile-time validation

**Good Practice:** Use conceptual examples without exact JSON structure
```
# In prompt file:
Note: These examples show the translation logic, not exact JSON structure (which is provided by the system).

Example 1 - Regular verb:
Input: word="flexed", context="..."
Expected:
  - Russian word: "размял"
  - Russian sentence: "Отбросив свой черный плащ..."
  - Spanish word: "flexionó"
```

**Benefits:**
- ✅ Schema is enforced by `with_structured_output(Model)` - single source of truth
- ✅ Examples teach translation logic, not JSON structure
- ✅ Model changes don't break prompts
- ✅ More maintainable

**When schemas must be in prompts:**
If you absolutely need schema info in prompts, generate it dynamically:
```python
schema = MyModel.model_json_schema()
prompt_with_schema = f"{base_prompt}\n\nSchema:\n{json.dumps(schema, indent=2)}"
```

## 14. AI Assistant Specific Guidelines

### When refactoring:
1. **Always read files before editing** - understand context
2. **Make atomic changes** - one logical change per commit
3. **Verify after changes** - run imports/tests
4. **Explain breaking changes** - communicate impact

### When uncertain:
1. **Ask for clarification** - don't guess requirements
2. **Propose alternatives** - explain trade-offs
3. **Show examples** - make suggestions concrete

### When making architectural decisions:
1. **Follow existing patterns** - maintain consistency
2. **Document new patterns** - update guidelines
3. **Consider maintainability** - optimize for readability

---

## Summary: Core Principles

1. **Explicit over implicit** - make intentions clear
2. **Readability over cleverness** - code is read more than written
3. **Fail fast** - catch errors early
4. **Consistency** - follow project conventions
5. **Simplicity** - solve the problem, nothing more

---

*These guidelines are living documentation. Update them as the project evolves.*
