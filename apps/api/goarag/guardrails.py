"""Guardrails — "knows when not to answer" (PDR §7).

Every check here is in-process and costs at most one cosine, because they run
inside the same 200 ms budget as the answer itself. Anything needing a model
call belongs on the generative path, not here.
"""

from __future__ import annotations

import re

import numpy as np

from .schemas import RefusalReason

# Multilingual blocklist. Deliberately narrow: this gate exists to refuse
# clearly abusive or self-harm input, not to police topic — that is the
# off-topic gate's job, and conflating the two produces smug refusals.
_UNSAFE = re.compile(
    r"\b(kill\s+(myself|yourself)|suicide|make\s+a\s+bomb|child\s+porn)\b", re.I
)

# Prompt-injection phrasings that sometimes ride in on retrieved web passages.
_INJECTION = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|disregard\s+the\s+above|"
    r"you\s+are\s+now\s+|system\s*:\s*)",
    re.I,
)


class DomainGate:
    """Off-topic detection by distance to the corpus centroid set (PDR §7.2).

    A single global centroid would reject narrow-but-valid queries, so we keep
    k centroids (cheap k-means over a sample) and score against the nearest.

    Calibrated on the real 1,200-passage corpus (120 eval + 10 out-of-scope):
    centroid similarity separates the two sets poorly, because a centroid is an
    average of ~460 unrelated web passages and ends up roughly equidistant from
    everything. Measured overlap at every threshold tried:

        tau   off-topic allowed   in-domain refused
        0.30       8/10                4/120
        0.40       2/10               15/120
        0.45       1/10               21/120

    There is no setting that refuses off-topic queries without refusing real
    ones, so the centroid check is now only a coarse pre-filter, and the real
    decision moved to `retrieval_floor` in the harness — top-1 *passage*
    similarity, which is what actually distinguishes "nothing here answers
    this" from "something does".
    """

    def __init__(self, centroids: np.ndarray, tau: float = 0.15) -> None:
        self.centroids = centroids
        self.tau = tau

    @classmethod
    def fit(cls, vectors: np.ndarray, k: int = 16, iters: int = 12, seed: int = 0) -> "DomainGate":
        rng = np.random.default_rng(seed)
        n = len(vectors)
        k = min(k, max(1, n))
        centroids = vectors[rng.choice(n, size=k, replace=False)].copy()
        for _ in range(iters):
            assign = np.argmax(vectors @ centroids.T, axis=1)
            for j in range(k):
                members = vectors[assign == j]
                if len(members):
                    c = members.mean(axis=0)
                    norm = np.linalg.norm(c)
                    centroids[j] = c / norm if norm else c
        return cls(centroids)

    def similarity(self, qvec: np.ndarray) -> float:
        return float(np.max(self.centroids @ qvec))

    def is_off_topic(self, qvec: np.ndarray) -> bool:
        return self.similarity(qvec) < self.tau


def check_input(text: str, stt_confidence: float | None, min_confidence: float = 0.55):
    """Input guardrails 1 and 3. Returns a RefusalReason or None."""
    if stt_confidence is not None and stt_confidence < min_confidence:
        return RefusalReason.LOW_STT_CONFIDENCE
    if _UNSAFE.search(text):
        return RefusalReason.UNSAFE
    return None


def sanitize_context(text: str) -> str:
    """Guardrail 4: retrieved passages are data, never instructions. Neutralise
    injection phrasings before they reach a prompt."""
    return _INJECTION.sub("[redacted-instruction] ", text)


def groundedness(answer: str, contexts: list[str], embedder, threshold: float = 0.55):
    """Output guardrail 6, for the generative path.

    Returns (kept_sentences, dropped_count). Each answer sentence must be
    supported by a retrieved chunk via max cosine; unsupported ones are cut.
    The extractive path skips this — it is grounded by construction.
    """
    from .chunking import sentences

    sents = sentences(answer)
    if not sents or not contexts:
        return [], len(sents)

    svecs = embedder.encode(sents)
    cvecs = embedder.encode(contexts)
    sims = svecs @ cvecs.T

    kept = [s for i, s in enumerate(sents) if float(np.max(sims[i])) >= threshold]
    return kept, len(sents) - len(kept)


REFUSAL_TEXT = {
    RefusalReason.LOW_STT_CONFIDENCE: "I didn't catch that — could you repeat it?",
    RefusalReason.OFF_TOPIC: (
        "That's outside my knowledge base, which covers MS MARCO web passages."
    ),
    RefusalReason.UNSAFE: "I can't help with that request.",
    RefusalReason.NO_GROUNDING: (
        "I don't have enough grounded context to answer that confidently."
    ),
}
