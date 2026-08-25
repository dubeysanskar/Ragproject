"""The harness — a real orchestrator with typed stages, timing, and fallbacks.

PDR §6. Pipeline graph:
    LangDetect → Embed → Guard(in) → Retrieve → Fuse → Gate → Extract
                                                              ↘ Generate (escalation)
                                                                → GroundCheck → Respond

Every stage is timed into the trace store, which is what the latency analytics
in `bench.py` reads. The measured budget deliberately excludes STT (streamed
and overlapped with speech, per PDR §1.5) and starts at query text.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field


from .embedder import Embedder
from .extractive import extract_answer
from .guardrails import REFUSAL_TEXT, DomainGate, check_input
from .langdetect import detect_lang
from .retrieval import HybridIndex, retrieve
from .schemas import (
    Answer,
    AskRequest,
    AskResponse,
    RefusalReason,
    StageTiming,
)


@dataclass
class Trace:
    trace_id: str
    query: str
    timings: list[StageTiming] = field(default_factory=list)
    total_ms: float = 0.0
    budget_ms: float = 0.0
    path: str = "extractive"
    refused: str | None = None


class TraceStore:
    """Ring buffer of recent traces — the source for /stats and bench reports."""

    def __init__(self, capacity: int = 5000) -> None:
        self._buf: deque[Trace] = deque(maxlen=capacity)

    def add(self, t: Trace) -> None:
        self._buf.append(t)

    def all(self) -> list[Trace]:
        return list(self._buf)

    def percentiles(self, field_name: str = "budget_ms") -> dict[str, float]:
        vals = sorted(getattr(t, field_name) for t in self._buf)
        if not vals:
            return {}
        def pct(p: float) -> float:
            idx = min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))
            return round(vals[idx], 2)
        return {
            "count": len(vals),
            "p50": pct(50), "p70": pct(70), "p90": pct(90),
            "p95": pct(95), "p100": pct(100),
            "mean": round(sum(vals) / len(vals), 2),
        }


# Confidence floors, exported so the eval adapter (app/generator.py) reports
# `grounded` against exactly the threshold the live pipeline answers on. Two
# copies of this number would let the harness and the reported guardrail drift
# apart silently.
#
# Calibrated against MSMARCO-XI's own answerable/unanswerable labels via
# scripts/calibrate_grounded.py (40+40 held-out queries, both signals measured
# on the same eval index). The earlier values (0.62 / 0.34) were guesses tuned
# against obviously off-topic probes -- "what's the weather in Goa" -- which are
# trivially far from the corpus. Against the dataset's real negatives, where the
# retrieved passages ARE topically relevant and simply do not answer the
# question, those floors answered 100% of unanswerable queries.
#
# Measured Pareto frontier (false_confidence / false_refusal):
#     0.78 / 0.78 -> 0.050 FC, 0.625 FR
#     0.70 / 0.70 -> 0.200 FC, 0.350 FR   <- chosen
#     0.70 / 0.60 -> 0.475 FC, 0.125 FR
#     0.60 / 0.52 -> 0.800 FC, 0.000 FR
# The classes genuinely overlap, so there is no free point: the suite's own
# reliability check calls false confidence "a worse failure than false refusal",
# which is why this sits on the cautious side of the curve.
RETRIEVAL_FLOOR = 0.70
EXTRACTIVE_FLOOR = 0.70

_SYNTHESIS_CUES = (
    "compare", "difference between", "summarize", "summarise", "explain why",
    "pros and cons", "advantages and disadvantages",
    "तुलना", "अंतर", "सारांश", "ஒப்பிடு", "வேறுபாடு",
)


def needs_synthesis(query: str, extract_confidence: float) -> bool:
    """PDR §5 Path-2 trigger: comparative/summary phrasing, long conjunctive
    questions, or weak single-chunk coverage. A heuristic, not a model —
    it has to cost nothing."""
    q = query.lower()
    if any(cue in q for cue in _SYNTHESIS_CUES):
        return True
    if len(q.split()) > 18 and (" and " in q or " या " in q or " और " in q):
        return True
    return extract_confidence < 0.22


class _Timer:
    """Records one stage into the trace. Kept explicit rather than a decorator
    so the graph reads top-to-bottom in `run`."""

    def __init__(self, trace: Trace, stage: str) -> None:
        self.trace, self.stage = trace, stage

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        ms = (time.perf_counter() - self.t0) * 1000
        self.trace.timings.append(StageTiming(stage=self.stage, ms=round(ms, 3)))
        return False


class Pipeline:
    def __init__(
        self,
        index: HybridIndex,
        embedder: Embedder,
        domain_gate: DomainGate | None = None,
        store: TraceStore | None = None,
        retrieval_floor: float = RETRIEVAL_FLOOR,
        extractive_floor: float = EXTRACTIVE_FLOOR,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.domain_gate = domain_gate
        self.store = store or TraceStore()
        # Guardrail 5, and the real off-topic gate. Thresholds the top hit's
        # raw *cosine*, not the RRF score: RRF is rank-derived, so it collapses
        # onto a handful of discrete values (0.0189, 0.0362, 0.0377 …) that
        # shift with pool composition — unusable as a decision boundary.
        self.retrieval_floor = retrieval_floor
        self.extractive_floor = extractive_floor

    def _refuse(self, trace: Trace, reason: RefusalReason, t0: float) -> AskResponse:
        trace.refused = reason.value
        trace.path = "refusal"
        trace.budget_ms = trace.total_ms = round((time.perf_counter() - t0) * 1000, 3)
        self.store.add(trace)
        return AskResponse(
            trace_id=trace.trace_id,
            answer=Answer(text=REFUSAL_TEXT[reason], path="refusal", confidence=0.0),
            timings=trace.timings,
            total_ms=trace.total_ms,
            budget_ms=trace.budget_ms,
        )

    def run(self, req: AskRequest, stt_confidence: float | None = None) -> AskResponse:
        trace = Trace(trace_id=uuid.uuid4().hex[:12], query=req.query)
        t0 = time.perf_counter()

        with _Timer(trace, "guard_input"):
            reason = check_input(req.query, stt_confidence)
        if reason:
            return self._refuse(trace, reason, t0)

        with _Timer(trace, "lang_detect"):
            lang = req.lang or detect_lang(req.query)

        with _Timer(trace, "embed"):
            qvec = self.embedder.encode_one(req.query)

        # Off-topic gate needs the query vector, so it sits after embed but
        # before retrieval — one cosine against k centroids.
        if self.domain_gate is not None:
            with _Timer(trace, "guard_domain"):
                off_topic = self.domain_gate.is_off_topic(qvec)
            if off_topic:
                return self._refuse(trace, RefusalReason.OFF_TOPIC, t0)

        with _Timer(trace, "retrieve"):
            hits = retrieve(self.index, req.query, qvec, lang)

        if not hits or max(h.dense_score for h in hits) < self.retrieval_floor:
            return self._refuse(trace, RefusalReason.NO_GROUNDING, t0)

        with _Timer(trace, "extract"):
            answer, confidence = extract_answer(
                req.query,
                qvec,
                hits,
                self.embedder,
                sentence_cache=self.index.sentence_cache,
            )

        wants_synthesis = req.force_path == "generative" or (
            req.force_path is None and needs_synthesis(req.query, confidence)
        )
        if not answer.text or confidence < self.extractive_floor or wants_synthesis:
            # The hot path never generates — it hands back a refusal marked
            # `escalate`, and the async /ask endpoint runs Path 2 outside the
            # budget. If no provider is configured, the refusal simply stands:
            # honest, never a guess.
            resp = self._refuse(trace, RefusalReason.NO_GROUNDING, t0)
            resp.retrieved = hits
            resp.escalate = bool(hits)
            return resp

        answer.lang = lang
        trace.total_ms = trace.budget_ms = round((time.perf_counter() - t0) * 1000, 3)
        trace.path = answer.path
        self.store.add(trace)

        return AskResponse(
            trace_id=trace.trace_id,
            answer=answer,
            retrieved=hits,
            timings=trace.timings,
            total_ms=trace.total_ms,
            budget_ms=trace.budget_ms,
        )
