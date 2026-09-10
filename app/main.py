from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequestPayload
from app.service import run_supervisor

load_dotenv()

app = FastAPI(title="Research, Critique, Revise")


@app.post("/chat/stream")
def chat_stream(payload: ChatRequestPayload) -> StreamingResponse:
    return StreamingResponse(run_supervisor(payload.query), media_type="text/plain")
