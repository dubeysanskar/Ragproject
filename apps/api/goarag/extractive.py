"""Extractive answering — the default, in-budget path (PDR §5 Path 1).

MS MARCO passages exist *because* they answer queries, so the fastest correct
answer is usually a span already sitting in a retrieved passage. Copying text
verbatim is also hallucination-proof by construction: there is no generation
step in which a model could invent something.
"""

from __future__ import annotations

import re

import numpy as np

from .chunking import sentences
from .embedder import Embedder
from .retrieval import tokenize_lexical
from .schemas import Answer, Citation, Retrieved

# Answer-type cues, per PDR §5. Keys are the wh-intent; values are regexes that
# suggest a sentence carries that kind of payload. Deliberately small and
# multilingual-tolerant rather than a full QA-type classifier.
_NUMERIC = re.compile(r"\d")
_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_MONEY = re.compile(r"[$₹€£]\s?\d|\b\d+(?:[.,]\d+)?\s?(?:crore|lakh|million|billion|percent|%)")

_INTENTS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    # (query trigger, evidence the sentence answers it)
    "when": (re.compile(r"\b(when|what year|कब|எப்போது)\b", re.I), _YEAR),
    "how_many": (
        re.compile(r"\b(how many|how much|कितन|எவ்வளவு)\b", re.I),
        re.compile(r"\d"),
    ),
    "cost": (re.compile(r"\b(cost|price|salary|कीमत|விலை)\b", re.I), _MONEY),
    "who": (
        re.compile(r"\b(who|किसने|कौन|யார்)\b", re.I),
        re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"),
    ),
}


def _intent_bonus(query: str, sentence: str) -> float:
    """Small nudge when the sentence carries the *kind* of fact asked for.
    Kept at 0.08 so it breaks ties without overriding semantic similarity."""
    bonus = 0.0
    for _name, (trigger, evidence) in _INTENTS.items():
        if trigger.search(query) and evidence.search(sentence):
            bonus += 0.08
    return min(bonus, 0.16)


def _lexical_overlap(q_tokens: set[str], sentence: str) -> float:
    s_tokens = set(tokenize_lexical(sentence))
    if not q_tokens or not s_tokens:
        return 0.0
    return len(q_tokens & s_tokens) / len(q_tokens)


def extract_answer(
    query: str,
    qvec: np.ndarray,
    retrieved: list[Retrieved],
    embedder: Embedder,
    max_candidates: int = 24,
    max_sentences: int = 2,
    sentence_cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[Answer, float]:
    """Pick the best answering sentence(s) across the retrieved passages.

    Returns (answer, confidence). Confidence is the winning blended score and
    is what the groundedness gate in the harness thresholds on.

    `sentence_cache` holds build-time sentence vectors per passage; with it this
    stage is a matrix multiply instead of an encode, which is the difference
    between ~53 ms and ~1 ms.
    """
    candidates: list[tuple[str, Retrieved]] = []
    cached_vecs: list[np.ndarray] = []
    misses: list[int] = []

    for r in retrieved:
        # Look up by sentence *text*, not position: retrieval often returns a
        # C1/C2 fragment of a passage rather than the full parent, so the i-th
        # sentence of the hit is rarely the i-th sentence of the cached passage.
        # Positional matching missed on those and re-encoded, costing up to 96 ms.
        by_text = sentence_cache.get(r.passage_id) if sentence_cache else None
        for s in sentences(r.text):
            # Single words are never answers; very long ones are whole paragraphs.
            if not (3 <= len(s.split()) <= 80):
                continue
            slot = len(candidates)
            candidates.append((s, r))
            vec = by_text.get(s) if by_text else None
            if vec is not None:
                cached_vecs.append(vec)
            else:
                cached_vecs.append(None)  # type: ignore[arg-type]
                misses.append(slot)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        return Answer(text="", path="extractive", confidence=0.0), 0.0

    # Only sentences the cache didn't cover get encoded.
    if misses:
        fresh = embedder.encode([candidates[i][0] for i in misses])
        for j, slot in enumerate(misses):
            cached_vecs[slot] = fresh[j]

    vecs = np.vstack(cached_vecs)
    sims = vecs @ qvec  # both L2-normalised → cosine

    q_tokens = set(tokenize_lexical(query))
    scored: list[tuple[float, int]] = []
    for i, (sent, _r) in enumerate(candidates):
        score = (
            0.70 * float(sims[i])
            + 0.30 * _lexical_overlap(q_tokens, sent)
            + _intent_bonus(query, sent)
        )
        scored.append((score, i))
    scored.sort(reverse=True)

    best_score, best_i = scored[0]
    best_sent, best_ret = candidates[best_i]

    # Pull in an adjacent sentence from the same passage when it also scores
    # well — "who won X" answers often span a claim plus its qualifier.
    picked = [best_i]
    for score, i in scored[1:]:
        if len(picked) >= max_sentences:
            break
        if candidates[i][1].passage_id == best_ret.passage_id and score > best_score * 0.82:
            picked.append(i)
    picked.sort()

    text = " ".join(candidates[i][0] for i in picked)
    citations = [
        Citation(n=1, passage_id=best_ret.passage_id, text=best_ret.text[:400])
    ]
    return (
        Answer(
            text=text,
            path="extractive",
            citations=citations,
            confidence=float(best_score),
            lang=best_ret.lang,
        ),
        float(best_score),
    )
