from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import structlog


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


@contextmanager
def timed(logger: structlog.stdlib.BoundLogger, event: str, **context: object) -> Iterator[None]:
    """Logs event_finished with a duration_ms field once the wrapped block completes."""
    start = time.perf_counter()
    yield
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(f"{event}_finished", duration_ms=duration_ms, **context)
