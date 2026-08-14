"""GoaRAG API — FastAPI, single process, index resident in RAM.

Deliberately NOT serverless (PDR §9): a cold start would reload a 220 MB model
and rebuild the index, which would blow the latency budget on the first query
a judge ever runs.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from goarag.build import Passage, build_index
from goarag.generate import GenerationUnavailable, generate
from goarag.guardrails import groundedness
from goarag.embedder import get_embedder
from goarag.harness import Pipeline, TraceStore
from goarag.schemas import Answer, AskRequest, AskResponse, Citation, StageTiming

CORPUS_PATH = Path(os.getenv("GOARAG_CORPUS", "data/corpus.json"))

# Used when no corpus file exists yet, so the API is runnable from a clean
# clone. scripts/build_corpus.py writes the real MSMARCO-XI subset.
DEMO_PASSAGES = [
    Passage(
        "demo-1",
        "The capital of India is New Delhi. It became the capital in 1911 when "
        "the British moved it from Calcutta. The metro area has about 33 million people.",
        "en",
        ("what is the capital of india", "when did delhi become the capital"),
    ),
    Passage(
        "demo-2",
        "भारत की राजधानी नई दिल्ली है। यह 1911 में राजधानी बनी जब अंग्रेजों ने "
        "इसे कलकत्ता से स्थानांतरित किया।",
        "hi",
        ("भारत की राजधानी क्या है",),
    ),
    Passage(
        "demo-3",
        "Mount Everest is the highest mountain on Earth at 8849 metres above sea "
        "level. It lies on the border between Nepal and Tibet.",
        "en",
        ("how tall is mount everest",),
    ),
]


def load_corpus() -> list[Passage]:
    if CORPUS_PATH.exists():
        raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        return [
            Passage(
                passage_id=str(r["passage_id"]),
                text=r["text"],
                lang=r.get("lang", "en"),
                questions=tuple(r.get("questions", ())),
            )
            for r in raw
        ]
    return DEMO_PASSAGES


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Everything expensive happens here, once, before the first request."""
    passages = load_corpus()
    print(f"[boot] corpus: {len(passages)} passages from "
          f"{CORPUS_PATH if CORPUS_PATH.exists() else 'built-in demo set'}")
    embedder = get_embedder()
    index, gate = build_index(passages, embedder)
    state["pipeline"] = Pipeline(index, embedder, domain_gate=gate, store=TraceStore())
    state["passages"] = len(passages)
    print(f"[boot] ready — {index.size} chunks resident")
    yield
    state.clear()


app = FastAPI(title="GoaRAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public read-only demo API
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    pipe: Pipeline | None = state.get("pipeline")
    return {
        "ok": pipe is not None,
        "chunks": pipe.index.size if pipe else 0,
        "passages": state.get("passages", 0),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    pipe: Pipeline | None = state.get("pipeline")
    if pipe is None:
        raise HTTPException(503, "Index still loading.")

    # Hot path is sync and CPU-bound; keep it off the event loop.
    resp = await run_in_threadpool(pipe.run, req)
    if not resp.escalate:
        return resp

    # ---- Path 2: generative escalation, outside the 200 ms budget (PDR §5) --
    t0 = time.perf_counter()
    try:
        gen = await generate(req.query, resp.retrieved)
    except GenerationUnavailable:
        return resp  # no provider — the honest refusal stands

    if "NOT_IN_CONTEXT" in gen.text:
        return resp

    # Guardrail 7: an uncited generative answer is retried stricter by the
    # provider prompt already; if still uncited here, downgrade to refusal.
    if not re.search(r"\[\d+\]", gen.text):
        return resp

    # Guardrail 6: strip unsupported sentences; >50% loss → refusal stands.
    contexts = [r.text for r in resp.retrieved]
    kept, dropped = await run_in_threadpool(
        groundedness, gen.text, contexts, pipe.embedder
    )
    if not kept or dropped > len(kept):
        return resp

    resp.answer = Answer(
        text=" ".join(kept),
        path="generative",
        citations=[
            Citation(n=i + 1, passage_id=r.passage_id, text=r.text[:400])
            for i, r in enumerate(resp.retrieved)
        ],
        confidence=0.5,
        lang=req.lang or resp.answer.lang,
    )
    resp.escalate = False
    resp.timings.append(
        StageTiming(stage=f"generate[{gen.provider}]",
                    ms=round((time.perf_counter() - t0) * 1000, 1))
    )
    resp.total_ms = round(resp.budget_ms + (time.perf_counter() - t0) * 1000, 1)
    return resp


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile) -> dict:
    """Voice front door: audio in, transcript + detected language out.

    Kept separate from /ask so the UI can show the transcript (and let the
    user correct it) before retrieval fires — and so the measured budget
    starts at query text, exactly as the PDR defines it.
    """
    from goarag.stt import STTError, transcribe

    audio = await file.read()
    if not audio or len(audio) > 10 * 1024 * 1024:
        raise HTTPException(413, "Audio empty or over 10 MB.")
    try:
        t = await transcribe(audio, file.content_type or "audio/webm")
    except STTError as e:
        raise HTTPException(502, f"All STT providers failed: {e}")
    return {
        "text": t.text,
        "lang": t.lang,
        "confidence": t.confidence,
        "provider": t.provider,
    }


@app.get("/stats")
def stats() -> dict:
    """Live percentiles over everything served since boot (PDR §8)."""
    pipe: Pipeline | None = state.get("pipeline")
    if pipe is None:
        raise HTTPException(503, "Index still loading.")
    traces = pipe.store.all()
    paths: dict[str, int] = {}
    for t in traces:
        paths[t.path] = paths.get(t.path, 0) + 1
    return {
        "budget_ms": pipe.store.percentiles("budget_ms"),
        "paths": paths,
        "refusals": sum(1 for t in traces if t.refused),
    }
