from __future__ import annotations

import pytest

from src.utils.retry import retry_transient


def test_retry_transient_returns_the_result_on_first_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result = retry_transient(fn, attempts=3, base_delay_seconds=0)

    assert result == "ok"
    assert len(calls) == 1


def test_retry_transient_recovers_after_transient_failures():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("transient network blip")
        return "ok"

    result = retry_transient(fn, attempts=3, base_delay_seconds=0)

    assert result == "ok"
    assert len(calls) == 3


def test_retry_transient_reraises_the_original_error_once_attempts_are_exhausted():
    def fn():
        raise ConnectionError("still failing")

    with pytest.raises(ConnectionError, match="still failing"):
        retry_transient(fn, attempts=2, base_delay_seconds=0)
