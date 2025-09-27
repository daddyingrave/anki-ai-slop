from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler


def enable_cache(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(db_path)))
    print(f"[observability] LLM cache enabled at {db_path}")


# ---- Minimal Gemini token usage logging via LangChain callback ----
class GeminiTokenUsageCallback(BaseCallbackHandler):
    """Log input/output/total token counts for Gemini.

    We only look at LangChain's usage_metadata with canonical keys.
    """

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs) -> None:  # type: ignore[override]
        try:
            in_tokens = out_tokens = total_tokens = None

            # Primary: llm_output.usage_metadata
            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("usage_metadata") if isinstance(llm_output, dict) else None
            if isinstance(usage, dict):
                in_tokens = usage.get("input_tokens")
                out_tokens = usage.get("output_tokens")
                total_tokens = usage.get("total_tokens")

            # Secondary: first generation message usage_metadata (or response_metadata.usage_metadata)
            if in_tokens is None and out_tokens is None and total_tokens is None:
                gens = getattr(response, "generations", None) or []
                if gens and gens[0]:
                    first = gens[0][0]
                    msg = getattr(first, "message", None)
                    if msg is not None:
                        usage = getattr(msg, "usage_metadata", None)
                        if not isinstance(usage, dict):
                            resp_meta = getattr(msg, "response_metadata", {}) or {}
                            usage = resp_meta.get("usage_metadata") if isinstance(resp_meta, dict) else None
                        if isinstance(usage, dict):
                            in_tokens = usage.get("input_tokens")
                            out_tokens = usage.get("output_tokens")
                            total_tokens = usage.get("total_tokens")

            # Print only one simple line; omit model name for simplicity
            print(
                f"[observability] Gemini token usage: input_tokens={in_tokens} output_tokens={out_tokens} total_tokens={total_tokens}"
            )
        except Exception as e:  # pragma: no cover
            print(f"[observability] Failed to log token usage: {e}")


def get_default_callbacks() -> List[BaseCallbackHandler]:
    return [GeminiTokenUsageCallback()]


__all__ = ["enable_cache", "GeminiTokenUsageCallback", "get_default_callbacks"]
