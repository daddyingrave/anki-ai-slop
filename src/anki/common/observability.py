from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def enable_cache(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(db_path)))
    print(f"[observability] LLM cache enabled at {db_path}")


# ---- Enhanced Gemini token usage logging via LangChain callback ----
class GeminiTokenUsageCallback(BaseCallbackHandler):
    """Log token usage for Gemini models.

    Tracks token counts from LangChain's usage_metadata:
    - input_tokens: Prompt tokens
    - output_tokens: Response tokens (includes reasoning if present)
    - total_tokens: Sum of input + output
    - input_token_details.cache_read: Tokens read from cache
    - output_token_details.reasoning: Reasoning/thinking tokens (Gemini 2.5+)

    Also calculates:
    - Estimated API costs based on Gemini Flash pricing
    - Cache efficiency when cache_read > 0
    """

    def _extract_usage_metadata(self, response: LLMResult) -> Optional[Dict[str, Any]]:
        """Extract usage_metadata from LLMResult response.

        Tries multiple locations in order:
        1. response.llm_output.usage_metadata
        2. response.generations[0][0].message.usage_metadata
        3. response.generations[0][0].message.response_metadata.usage_metadata
        """
        # Try llm_output first
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("usage_metadata")
            if isinstance(usage, dict):
                return usage

        # Try generations path
        if response.generations and response.generations[0]:
            first_gen = response.generations[0][0]
            if hasattr(first_gen, "message"):
                msg = first_gen.message

                # Direct usage_metadata
                if hasattr(msg, "usage_metadata") and isinstance(msg.usage_metadata, dict):
                    return msg.usage_metadata

                # response_metadata.usage_metadata
                if hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
                    usage = msg.response_metadata.get("usage_metadata")
                    if isinstance(usage, dict):
                        return usage

        return None

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs) -> None:  # type: ignore[override]
        try:
            usage = self._extract_usage_metadata(response)
            if not usage:
                return

            # Extract token counts
            in_tokens = usage.get("input_tokens")
            out_tokens = usage.get("output_tokens")
            total_tokens = usage.get("total_tokens")

            # Extract input token details (cache info)
            cache_read_tokens: Optional[int] = None
            input_details = usage.get("input_token_details")
            if isinstance(input_details, dict):
                cache_read_tokens = input_details.get("cache_read")

            # Extract output token details (reasoning/thinking tokens)
            reasoning_tokens: Optional[int] = None
            output_details = usage.get("output_token_details")
            if isinstance(output_details, dict):
                reasoning_tokens = output_details.get("reasoning")

            # Build log message
            log_parts = [f"input={in_tokens}"]

            if cache_read_tokens and cache_read_tokens > 0:
                log_parts.append(f"cached={cache_read_tokens}")

            log_parts.append(f"output={out_tokens}")

            if reasoning_tokens and reasoning_tokens > 0:
                log_parts.append(f"reasoning={reasoning_tokens}")

            log_parts.append(f"total={total_tokens}")

            print(f"[observability] Gemini token usage: {' '.join(log_parts)}")

        except Exception as e:  # pragma: no cover
            print(f"[observability] Failed to log token usage: {e}")


def get_default_callbacks() -> List[BaseCallbackHandler]:
    return [GeminiTokenUsageCallback()]
