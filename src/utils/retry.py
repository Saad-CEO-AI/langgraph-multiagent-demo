from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_transient(fn: Callable[[], T], attempts: int = 3, base_delay_seconds: float = 1.0) -> T:
    """Retry fn() on any exception, with linear backoff, re-raising the last error if all attempts fail.

    Deliberately broad rather than pinned to specific exception types: a
    transient network/TLS failure surfaces as different classes depending on
    where the handshake breaks (ssl.SSLError, httpx.ConnectError,
    openai.APIConnectionError, ...). Narrowing this to one type risks missing
    the next variant; the tradeoff is retrying a small number of times and
    then propagating the original error unchanged, never swallowing it.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - see docstring
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(base_delay_seconds * (attempt + 1))

    assert last_error is not None
    raise last_error
