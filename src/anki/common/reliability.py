from __future__ import annotations

import inspect
import time

from anki.common.observability import get_default_callbacks


def retry_invoke(chain, inputs: dict, *, max_retries: int, backoff_initial_seconds: float, backoff_multiplier: float):
    """Invoke a chain with retries and exponential backoff.

    max_retries is the number of retries after the initial attempt.
    """
    attempt = 0
    delay = backoff_initial_seconds
    callbacks = get_default_callbacks()

    # Feature-detect whether chain.invoke supports a 'config' kwarg (directly or via **kwargs)
    try:
        sig = inspect.signature(chain.invoke)
        supports_config = (
                'config' in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        )
    except Exception:
        # If inspection fails, be conservative and skip passing config
        supports_config = False

    while True:
        try:
            if supports_config:
                # LangChain 0.2+ Runnable.invoke accepts config with callbacks
                return chain.invoke(inputs, config={"callbacks": callbacks})
            # Fallback for older versions
            return chain.invoke(inputs)
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(delay)
            delay *= backoff_multiplier
            attempt += 1
