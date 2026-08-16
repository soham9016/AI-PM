"""
Retry/backoff decorator for LLM provider calls.

Groq's free tier has fairly tight rate limits; this wraps any callable
(typically an agent node's LLM invocation) so transient rate-limit /
connection errors are retried with exponential backoff + jitter instead of
crashing the run.

ALSO COVERS CEREBRAS' RATE LIMITS, NOT JUST GROQ'S: Cerebras runs on an
OpenAI-compatible client (langchain_cerebras wraps `openai`), so its
errors are openai.* exception classes, not groq.*'s -- a Cerebras rate
limit would silently fall through this tuple entirely without the openai
imports below. Non-rate-limit Cerebras errors (bad model name, payment
required, etc.) are NOT meant to land here at all -- config.py's
invoke_llm() intercepts those at the invocation site itself and falls
back to Groq before they ever reach this decorator; only a genuine rate
limit is deliberately left to propagate up to this backoff loop (see its
docstring).
"""

import functools
import logging
import random
import time

import openai
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

logger = logging.getLogger("business_copilot.retry")

RETRYABLE_EXCEPTIONS = (
    RateLimitError, APIConnectionError, APITimeoutError, APIStatusError,
    openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
)


def with_retry(max_attempts: int = 4, base_delay: float = 1.5, max_delay: float = 30.0):
    """Decorator factory. Retries the wrapped call on Groq rate-limit /
    connection errors with exponential backoff and random jitter.

    Usage:
        @with_retry()
        def call_llm(...):
            return llm.invoke(...)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, delay * 0.25)
                    logger.warning(
                        "%s: attempt %d/%d failed (%s), retrying in %.1fs",
                        func.__name__, attempt, max_attempts, type(exc).__name__, delay,
                    )
                    time.sleep(delay)
            raise last_exc

        return wrapper

    return decorator
