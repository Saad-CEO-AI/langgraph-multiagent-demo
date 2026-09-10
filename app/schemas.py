from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequestPayload(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class CritiqueVerdict(BaseModel):
    verdict: Literal["approved", "needs_revision"] = Field(
        description="approved if the draft fully and accurately answers the "
        "question; needs_revision otherwise."
    )
    critique: str = Field(
        description="Specific, actionable feedback the researcher can act on. "
        "Empty string if approved."
    )
