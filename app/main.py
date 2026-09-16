from __future__ import annotations

from typing import Iterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.logging_config import configure_logging
from app.schemas import ChatRequestPayload
from app.service import stream_chat_response

load_dotenv()
configure_logging()

app = FastAPI(title="Research, Critique, Revise")


def _ndjson(query: str) -> Iterator[str]:
    for event in stream_chat_response(query):
        yield event.model_dump_json() + "\n"


@app.post("/chat/stream")
def chat_stream(payload: ChatRequestPayload) -> StreamingResponse:
    return StreamingResponse(_ndjson(payload.query), media_type="application/x-ndjson")
