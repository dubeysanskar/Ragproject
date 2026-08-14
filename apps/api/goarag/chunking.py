"""The chunking matrix — C1..C5 from PDR §4.

All of this runs offline, in the index builder. Nothing here is on the hot
path, which is why it can afford to be thorough. Every chunk carries the
metadata (C4) that the retriever later filters and de-duplicates on.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

import numpy as np

from .schemas import Chunk, Strategy

# Indic scripts end sentences with danda / double danda as well as ASCII stops.
# Devanagari ।, ॥ plus the usual .?! and their fullwidth forms.
_SENT_END = re.compile(r"(?<=[।॥\.\?\!۔।])\s+")
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def sentences(text: str) -> list[str]:
    """Indic-aware sentence split (PDR C1). Falls back to the whole string when
    a passage has no terminator at all, which is common in MS MARCO."""
    parts = [s.strip() for s in _SENT_END.split(_clean(text)) if s.strip()]
    return parts or ([_clean(text)] if _clean(text) else [])


def _cid(passage_id: str, strategy: Strategy, index: int) -> str:
    raw = f"{passage_id}:{strategy.value}:{index}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()


def _spans(text: str, sents: Sequence[str]) -> list[tuple[int, int]]:
    """Character spans for each sentence, so the UI can highlight the exact
    answer inside the source passage."""
    out: list[tuple[int, int]] = []
    cursor = 0
    for s in sents:
        idx = text.find(s, cursor)
        if idx < 0:
            idx = cursor
        out.append((idx, idx + len(s)))
        cursor = idx + len(s)
    return out


# --------------------------------------------------------------------- C1
def semantic_chunks(
    passage_id: str,
    text: str,
    lang: str,
    embed: "callable[[list[str]], np.ndarray]",
    tau: float = 0.62,
    max_sents: int = 6,
) -> list[Chunk]:
    """Merge adjacent sentences while they stay semantically close; break at
    similarity drops. `embed` is injected so this module never imports a model.
    """
    sents = sentences(text)
    if len(sents) <= 1:
        return [
            Chunk(
                chunk_id=_cid(passage_id, Strategy.SEMANTIC, 0),
                passage_id=passage_id,
                text=t,
                strategy=Strategy.SEMANTIC,
                lang=lang,
                position=0,
                char_span=span,
            )
            for t, span in zip(sents, _spans(text, sents))
        ]

    vecs = embed(sents)
    # Cosine between neighbours; vectors from fastembed arrive L2-normalised,
    # but normalise defensively so a swapped model can't silently skew tau.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    sims = np.sum(unit[:-1] * unit[1:], axis=1)

    groups: list[list[int]] = [[0]]
    for i, sim in enumerate(sims, start=1):
        if sim >= tau and len(groups[-1]) < max_sents:
            groups[-1].append(i)
        else:
            groups.append([i])

    spans = _spans(text, sents)
    chunks: list[Chunk] = []
    for pos, group in enumerate(groups):
        body = " ".join(sents[i] for i in group)
        chunks.append(
            Chunk(
                chunk_id=_cid(passage_id, Strategy.SEMANTIC, pos),
                passage_id=passage_id,
                text=body,
                strategy=Strategy.SEMANTIC,
                lang=lang,
                position=pos,
                char_span=(spans[group[0]][0], spans[group[-1]][1]),
            )
        )
    return chunks


# --------------------------------------------------------------------- C2
def sliding_chunks(
    passage_id: str,
    text: str,
    lang: str,
    tokenize: "callable[[str], list[int]]",
    detokenize: "callable[[list[int]], str]",
    window: int = 256,
    overlap: int = 64,
) -> list[Chunk]:
    """Token windows, not character windows — for Indic scripts a character
    count is a meaningless proxy for model context (PDR C2)."""
    ids = tokenize(_clean(text))
    if not ids:
        return []
    step = max(1, window - overlap)
    chunks: list[Chunk] = []
    for pos, start in enumerate(range(0, len(ids), step)):
        piece = ids[start : start + window]
        if not piece:
            break
        body = detokenize(piece).strip()
        if body:
            chunks.append(
                Chunk(
                    chunk_id=_cid(passage_id, Strategy.SLIDING, pos),
                    passage_id=passage_id,
                    text=body,
                    strategy=Strategy.SLIDING,
                    lang=lang,
                    position=pos,
                )
            )
        if start + window >= len(ids):
            break
    return chunks


# --------------------------------------------------------------------- C3
def parent_child_chunks(
    passage_id: str, text: str, lang: str, child_sents: int = 2
) -> tuple[Chunk, list[Chunk]]:
    """Small children for precise matching, one parent returned for context.
    The retriever indexes only the children and resolves up (PDR C3)."""
    body = _clean(text)
    parent = Chunk(
        chunk_id=_cid(passage_id, Strategy.PARENT, 0),
        passage_id=passage_id,
        text=body,
        strategy=Strategy.PARENT,
        lang=lang,
        char_span=(0, len(body)),
    )
    sents = sentences(body)
    spans = _spans(body, sents)
    children: list[Chunk] = []
    for pos, start in enumerate(range(0, len(sents), child_sents)):
        group = sents[start : start + child_sents]
        if not group:
            continue
        children.append(
            Chunk(
                chunk_id=_cid(passage_id, Strategy.CHILD, pos),
                passage_id=passage_id,
                text=" ".join(group),
                strategy=Strategy.CHILD,
                lang=lang,
                position=pos,
                char_span=(spans[start][0], spans[min(start + child_sents, len(sents)) - 1][1]),
                parent_id=parent.chunk_id,
            )
        )
    return parent, children


# --------------------------------------------------------------------- C5
def question_chunks(
    passage_id: str, questions: Iterable[str], lang: str
) -> list[Chunk]:
    """Index the dataset's own questions as alternate keys pointing at their
    passage. A spoken question matches a stored question far better than it
    matches prose — this is the single biggest recall win on MS MARCO."""
    out: list[Chunk] = []
    for pos, q in enumerate(questions):
        q = _clean(q)
        if not q:
            continue
        out.append(
            Chunk(
                chunk_id=_cid(passage_id, Strategy.QUESTION, pos),
                passage_id=passage_id,
                text=q,
                strategy=Strategy.QUESTION,
                lang=lang,
                position=pos,
                answers_passage=passage_id,
            )
        )
    return out


def build_all(
    passage_id: str,
    text: str,
    lang: str,
    questions: Sequence[str],
    embed,
    tokenize,
    detokenize,
) -> tuple[list[Chunk], list[Chunk]]:
    """Run the whole matrix over one passage.

    Returns (indexed, parents): parents are stored for resolution but never
    embedded — indexing both parent and children would double-count the same
    text in the fused ranking.
    """
    parent, children = parent_child_chunks(passage_id, text, lang)
    indexed: list[Chunk] = []
    indexed += semantic_chunks(passage_id, text, lang, embed)
    indexed += sliding_chunks(passage_id, text, lang, tokenize, detokenize)
    indexed += children
    indexed += question_chunks(passage_id, questions, lang)
    return indexed, [parent]
