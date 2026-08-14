"""Typed I/O for every harness stage (PDR §6: structured I/O requirement)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Strategy(str, Enum):
    """Which chunker produced a chunk. Kept in the Qdrant payload so a single
    collection can serve all five strategies and still be ablated apart."""

    SEMANTIC = "C1"
    SLIDING = "C2"
    CHILD = "C3"
    PARENT = "C3P"
    QUESTION = "C5"


class Chunk(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    strategy: Strategy
    lang: str
    position: int = 0
    char_span: tuple[int, int] = (0, 0)
    # C3 children point at the parent that gets returned in their place.
    parent_id: str | None = None
    # C5 keys are questions; the answer text lives on the passage they point to.
    answers_passage: str | None = None


class Retrieved(BaseModel):
    chunk_id: str
    passage_id: str
    text: str
    strategy: Strategy
    lang: str
    # Fused RRF score — rank-based, good for ordering, useless as a threshold.
    score: float
    # Raw cosine to the query. This is the meaningful "does anything here
    # actually answer the question" signal, and what the no-grounding gate uses.
    dense_score: float = 0.0
    dense_rank: int | None = None
    lexical_rank: int | None = None


class Citation(BaseModel):
    n: int
    passage_id: str
    text: str
    char_span: tuple[int, int] | None = None


class StageTiming(BaseModel):
    stage: str
    ms: float


class Answer(BaseModel):
    text: str
    path: Literal["extractive", "generative", "refusal"]
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    lang: str = "en"


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)
    lang: str | None = None
    # Lets the bench harness force a path instead of trusting the escalator.
    force_path: Literal["extractive", "generative"] | None = None


class AskResponse(BaseModel):
    trace_id: str
    answer: Answer
    retrieved: list[Retrieved] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0
    # The spec'd number: chunking+retrieval through final output, excluding STT.
    budget_ms: float = 0.0
    # Set when the sync path wants the async layer to try Path 2 (generative).
    escalate: bool = False


class RefusalReason(str, Enum):
    LOW_STT_CONFIDENCE = "low_stt_confidence"
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"
    NO_GROUNDING = "no_grounding"
