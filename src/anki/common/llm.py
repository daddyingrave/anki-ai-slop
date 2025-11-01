from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI


def build_llm(
    model: str,
    temperature: float,
    thinking_budget: int | None = None,
) -> ChatGoogleGenerativeAI:
    """Return a configured Google Gemini chat LLM instance.

    Parameters
    - model: Gemini model name (e.g., "gemini-1.5-flash", "gemini-2.5-flash").
    - temperature: Sampling temperature.
    - thinking_budget: Token budget for internal reasoning (Gemini 2.5+ only).
        - 0: Disable thinking (fastest, cheapest - recommended for simple tasks)
        - -1: Dynamic thinking (model decides based on complexity)
        - N (e.g., 512, 1024): Specific token limit for thinking
        - None: Use model default (dynamic thinking for 2.5 models)

    Note: Thinking budget is only supported on Gemini 2.5+ models.
    For simple tasks like translation, thinking_budget=0 is recommended.
    """
    # Build kwargs for ChatGoogleGenerativeAI
    kwargs = {
        "model": model,
        "temperature": temperature,
    }

    # Add thinking_budget if specified (only for Gemini 2.5+ models)
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget

    # Prefer plain-text response to avoid tool-calling conflict and reduce prose around JSON
    try:
        return ChatGoogleGenerativeAI(
            **kwargs,
            response_mime_type="text/plain",
        )
    except TypeError:
        try:
            return ChatGoogleGenerativeAI(
                **kwargs,
                generation_config={"response_mime_type": "text/plain"},
            )
        except TypeError:
            # Fallback for older versions that don't support thinking_budget
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
            )
