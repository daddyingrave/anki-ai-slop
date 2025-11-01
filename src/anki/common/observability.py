from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

logger = logging.getLogger(__name__)


def enable_cache(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(db_path)))
    logger.info(f"LLM cache enabled at {db_path}")


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

            # Log token usage with structured fields only (no redundant message)
            logger.info(
                "Token usage",
                extra={
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "total_tokens": total_tokens,
                    "cached_tokens": cache_read_tokens or 0,
                    "reasoning_tokens": reasoning_tokens or 0,
                }
            )

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to log token usage: {e}")


class LLMPromptResponseCallback(BaseCallbackHandler):
    """Log LLM prompts and responses at DEBUG level.

    Logs:
    - Model configuration (model name, temperature, thinking_budget)
    - Full prompts (no truncation - complete system and human messages)
    - Full responses (no truncation - complete LLM output)

    Only logs when DEBUG level is enabled to avoid performance overhead.

    Note: Prompts and responses are logged in full to enable complete debugging.
    For very long prompts (e.g., 10k+ chars), logs may be verbose.
    """

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id,
        parent_run_id=None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log LLM request details at DEBUG level."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            # Extract model configuration from kwargs or serialized
            model_name = kwargs.get("invocation_params", {}).get("model", "unknown")
            temperature = kwargs.get("invocation_params", {}).get("temperature", "unknown")
            thinking_budget = kwargs.get("invocation_params", {}).get("thinking_budget", "not set")

            logger.debug(
                f"LLM Request",
                extra={
                    "model": model_name,
                    "temperature": temperature,
                    "thinking_budget": thinking_budget,
                }
            )

            # Log full prompts (no truncation)
            for i, prompt in enumerate(prompts):
                logger.debug(f"Prompt [{i+1}/{len(prompts)}]:\n{prompt}")

        except Exception as e:
            logger.error(f"Failed to log LLM start: {e}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id,
        parent_run_id=None,
        **kwargs: Any,
    ) -> None:
        """Log LLM response at DEBUG level."""
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            # Just dump the entire response object
            logger.debug(f"LLM Response:\n{response}")

        except Exception as e:
            logger.error(f"Failed to log LLM response: {e}", exc_info=True)


def get_default_callbacks() -> List[BaseCallbackHandler]:
    """Get default callbacks for LLM calls.

    Returns:
        List of callback handlers:
        - GeminiTokenUsageCallback: Logs token usage at INFO level
        - LLMPromptResponseCallback: Logs prompts/responses at DEBUG level
    """
    return [
        GeminiTokenUsageCallback(),
        LLMPromptResponseCallback(),
    ]
