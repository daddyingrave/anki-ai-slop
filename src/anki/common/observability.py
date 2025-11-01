from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

logger = logging.getLogger(__name__)


# ---- Token accumulation for cost tracking ----
@dataclass
class TokenUsage:
    """Thread-safe token usage accumulator."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int = 0, output_tokens: int = 0,
            cached_tokens: int = 0, reasoning_tokens: int = 0) -> None:
        """Thread-safe token accumulation."""
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cached_tokens += cached_tokens
            self.reasoning_tokens += reasoning_tokens

    def get_totals(self) -> Dict[str, int]:
        """Get current totals (thread-safe)."""
        with self._lock:
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            }

    def reset(self) -> None:
        """Reset all counters (thread-safe)."""
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0
            self.cached_tokens = 0
            self.reasoning_tokens = 0


class TokenAccumulator:
    """Global token accumulator for tracking usage across pipeline execution."""

    _instance: Optional['TokenAccumulator'] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.usage = TokenUsage()
        self.model_name: Optional[str] = None

    @classmethod
    def get_instance(cls) -> 'TokenAccumulator':
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_model(self, model_name: str) -> None:
        """Set the model name for pricing lookup."""
        self.model_name = model_name

    def add_usage(self, input_tokens: int = 0, output_tokens: int = 0,
                  cached_tokens: int = 0, reasoning_tokens: int = 0) -> None:
        """Add token usage (thread-safe)."""
        self.usage.add(input_tokens, output_tokens, cached_tokens, reasoning_tokens)

    def get_summary(self) -> Dict[str, int]:
        """Get usage summary."""
        return self.usage.get_totals()

    def reset(self) -> None:
        """Reset accumulator."""
        self.usage.reset()
        self.model_name = None

    def print_summary(self, pricing_config: Optional[Dict[str, Any]] = None) -> None:
        """Print token usage summary with cost estimation.

        Args:
            pricing_config: Model pricing configuration from config.yaml
                           Format: {model_name: {input_token_price: float, output_token_price: float}}
        """
        totals = self.get_summary()

        print("\n" + "=" * 70)
        print("TOKEN USAGE SUMMARY")
        print("=" * 70)
        print(f"Input tokens:      {totals['input_tokens']:>12,}")
        print(f"Output tokens:     {totals['output_tokens']:>12,}")
        print(f"{'─' * 70}")
        print(f"Total tokens:      {totals['total_tokens']:>12,}")

        # Calculate cost if pricing config is provided
        if pricing_config and self.model_name:
            model_pricing = pricing_config.get(self.model_name)
            if model_pricing:
                # Convert to float in case YAML loaded as string
                input_price = float(model_pricing.get('input_token_price', 0.0))
                output_price = float(model_pricing.get('output_token_price', 0.0))

                # Calculate costs (prices are per 1M tokens)
                # Note: All tokens are billed. LangChain cache hits don't reach the LLM at all.
                input_cost = (totals['input_tokens'] / 1_000_000) * input_price
                output_cost = (totals['output_tokens'] / 1_000_000) * output_price
                total_cost = input_cost + output_cost

                print(f"\n{'─' * 70}")
                print(f"ESTIMATED COST (Model: {self.model_name})")
                print(f"{'─' * 70}")
                print(f"Input cost:        ${input_cost:>12.6f}  (${input_price}/1M tokens)")
                print(f"Output cost:       ${output_cost:>12.6f}  (${output_price}/1M tokens)")
                print(f"{'─' * 70}")
                print(f"Total cost:        ${total_cost:>12.6f}")
            else:
                print(f"\n⚠️  No pricing configured for model: {self.model_name}")
        elif not pricing_config:
            print(f"\n⚠️  No pricing configuration provided")
        elif not self.model_name:
            print(f"\n⚠️  Model name not set")

        print("=" * 70 + "\n")


def enable_cache(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(db_path)))
    logger.info(f"LLM cache enabled at {db_path}")


# ---- Enhanced Gemini token usage logging via LangChain callback ----
class GeminiTokenUsageCallback(BaseCallbackHandler):
    """Log token usage for Gemini models and accumulate totals.

    Tracks token counts from LangChain's usage_metadata:
    - input_tokens: Prompt tokens
    - output_tokens: Response tokens (includes reasoning if present)
    - total_tokens: Sum of input + output
    - input_token_details.cache_read: Tokens read from cache
    - output_token_details.reasoning: Reasoning/thinking tokens (Gemini 2.5+)

    Also:
    - Accumulates tokens in global TokenAccumulator for cost tracking
    - Logs individual LLM call usage
    """

    def __init__(self, accumulate: bool = True) -> None:
        """Initialize callback.

        Args:
            accumulate: Whether to accumulate tokens in global TokenAccumulator (default: True)
        """
        super().__init__()
        self.accumulate = accumulate
        if accumulate:
            self.accumulator = TokenAccumulator.get_instance()

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
            # Check if this is a LangChain cache hit
            # When SQLiteCache returns a cached result, llm_output is None or empty
            # If cache hit, don't accumulate tokens (no actual LLM call was made)
            if not response.llm_output or len(response.llm_output) == 0:
                logger.debug("LangChain cache hit - skipping token accumulation")
                return

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

            # Accumulate tokens in global accumulator
            if self.accumulate and in_tokens is not None and out_tokens is not None:
                self.accumulator.add_usage(
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    cached_tokens=cache_read_tokens or 0,
                    reasoning_tokens=reasoning_tokens or 0,
                )

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
