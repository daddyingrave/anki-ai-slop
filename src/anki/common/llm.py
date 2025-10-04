from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI


def build_llm(model: str, temperature: float) -> ChatGoogleGenerativeAI:
    """Return a configured Google Gemini chat LLM instance.

    Parameters
    - model: Gemini model name (e.g., "gemini-1.5-flash").
    - temperature: Sampling temperature.
    """
    # Prefer plain-text response to avoid tool-calling conflict and reduce prose around JSON
    try:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            response_mime_type="text/plain",
        )
    except TypeError:
        try:
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                generation_config={"response_mime_type": "text/plain"},
            )
        except TypeError:
            return ChatGoogleGenerativeAI(model=model, temperature=temperature)
