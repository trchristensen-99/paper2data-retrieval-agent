from __future__ import annotations

import asyncio
import random
import re
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


def _extract_wait_seconds(error_text: str) -> float:
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 30.0


async def run_with_rate_limit_retry(coro_factory: Any, max_retries: int = 4) -> Any:
    """Run an async callable and retry on transient OpenAI/API network failures."""
    attempt = 0
    while True:
        try:
            return await coro_factory()
        except RateLimitError as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            wait_seconds = _extract_wait_seconds(str(exc)) + 1.0
            await asyncio.sleep(wait_seconds)
        except (APIConnectionError, APITimeoutError, InternalServerError):
            attempt += 1
            if attempt > max_retries:
                raise
            # Exponential backoff with a bit of jitter to avoid retry stampedes.
            wait_seconds = min(60.0, (2**attempt) + random.uniform(0.0, 1.0))
            await asyncio.sleep(wait_seconds)
