"""Generator surface required by the eval suite.

This is the real extractive answerer (goarag/extractive.py), not a shim around
an LLM: candidate sentences from the supplied contexts are scored against the
query and the best span is returned verbatim. Copying text cannot hallucinate,
which is the whole reason the live system answers this way.

`grounded` is the signal the suite's reliability check reads, and it is the one
number worth getting right. It reports whether the extractive score cleared the
same confidence floor the live harness uses to decide between answering and
refusing — so a fabricated answer on an unanswerable query shows up as
`grounded=True` and is counted against us, exactly as intended. Hard-wiring it
to True would make the check unable to catch anything.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from . import _bootstrap  # noqa: F401  (sys.path side effect)
from goarag.embedder import get_embedder  # noqa: E402
from goarag.extractive import extract_answer  # noqa: E402
from goarag.harness import EXTRACTIVE_FLOOR, RETRIEVAL_FLOOR  # noqa: E402
from goarag.schemas import Retrieved, Strategy  # noqa: E402

MODEL_LABEL = "goarag-extractive/minilm-l12-v2-onnx"

REFUSAL = "I don't have enough grounded context to answer that."


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def _as_retrieved(results) -> list[Retrieved]:
    """The suite passes its own duck-typed objects carrying `.text`/`.source`;
    map them onto the schema the extractive stage expects."""
    out: list[Retrieved] = []
    for i, r in enumerate(results):
        source = str(getattr(r, "source", "") or f"ctx-{i}")
        out.append(
            Retrieved(
                chunk_id=f"eval-{i}",
                passage_id=source,
                text=str(getattr(r, "text", "") or ""),
                strategy=Strategy.PARENT,
                lang="en",
                score=1.0 / (i + 1),  # suite order is the ranking
            )
        )
    return out


def generate_answer(query: str, results) -> GeneratedAnswer:
    t0 = time.perf_counter()

    retrieved = [r for r in _as_retrieved(results) if r.text.strip()]
    if not retrieved:
        return GeneratedAnswer(
            text=REFUSAL,
            grounded=False,
            generation_ms=(time.perf_counter() - t0) * 1000,
            model=MODEL_LABEL,
        )

    embedder = get_embedder()
    qvec = embedder.encode_one(query)
    # No sentence_cache here: the suite builds its own index, so these passages
    # are not ones we precomputed vectors for.
    answer, confidence = extract_answer(query, qvec, retrieved, embedder)

    # Threshold calibration hook: set GOARAG_SCORE_LOG to record the raw
    # extractive confidence per query, so the grounded/refuse boundary can be
    # chosen from measured separation rather than guessed.
    log_path = os.getenv("GOARAG_SCORE_LOG")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"query": query, "confidence": round(float(confidence), 4),
                                 "text": answer.text[:80]}, ensure_ascii=False) + chr(10))

    # Both gates, exactly as the live harness applies them. The first version
    # of this adapter checked only the extractive score and dropped the
    # retrieval floor entirely, which is half the guardrail -- and it answered
    # 100% of unanswerable queries as a result. `.score` on the suite's context
    # objects is inner-product over L2-normalised vectors, i.e. the same cosine
    # the live retrieval floor thresholds.
    top_score = max((r.score for r in retrieved), default=0.0)
    grounded = (
        bool(answer.text)
        and top_score >= RETRIEVAL_FLOOR
        and confidence >= EXTRACTIVE_FLOOR
    )
    return GeneratedAnswer(
        text=answer.text if grounded else REFUSAL,
        grounded=grounded,
        generation_ms=(time.perf_counter() - t0) * 1000,
        model=MODEL_LABEL,
    )
