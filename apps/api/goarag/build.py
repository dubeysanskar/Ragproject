"""Offline index construction (PDR §3). Not on the hot path — this is allowed
to be slow, and is where the whole chunking matrix gets applied.

"Allowed to be slow" is not "allowed to be quadratic", though. The first
version called the embedder once per passage (semantic chunking) plus once per
parent (sentence cache) — ~2,400 calls for 1,200 passages. fastembed carries a
large fixed cost per `embed()` invocation regardless of batch size, so boot took
over 40 minutes of CPU and never finished within a usable window.

This version embeds every sentence in the corpus in one batched pass and reuses
those vectors for *both* consumers, because they need exactly the same thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .chunking import (
    parent_child_chunks,
    question_chunks,
    semantic_chunks,
    sentences as split_sentences,
    sliding_chunks,
)
from .embedder import Embedder
from .guardrails import DomainGate
from .retrieval import HybridIndex
from .schemas import Chunk


@dataclass
class Passage:
    passage_id: str
    text: str
    lang: str
    # MSMARCO gives us the queries a passage answers — C5 indexes them as keys.
    questions: tuple[str, ...] = ()


def build_index(
    passages: list[Passage],
    embedder: Embedder,
    batch_size: int = 512,
    verbose: bool = True,
    return_vectors: bool = False,
):
    """Returns (index, gate), or (index, gate, vectors) when `return_vectors` —
    the snapshot writer needs the raw chunk matrix."""
    t0 = time.perf_counter()
    index = HybridIndex(dim=len(embedder.encode_one("dim probe")))

    # ---- pass 1: split every passage into sentences (pure string work) ------
    per_passage: list[list[str]] = [split_sentences(p.text) for p in passages]

    # ---- pass 2: embed every sentence in the corpus, once, in big batches ---
    flat: list[str] = [s for sents in per_passage for s in sents]
    if verbose:
        print(f"  embedding {len(flat)} sentences from {len(passages)} passages...")
    flat_vecs = (
        np.vstack([embedder.encode(flat[i : i + batch_size])
                   for i in range(0, len(flat), batch_size)])
        if flat
        else np.zeros((0, index.dim), dtype=np.float32)
    )

    # ---- pass 3: chunk, reusing the vectors we already have -----------------
    all_chunks: list[Chunk] = []
    all_parents: list[Chunk] = []
    cursor = 0
    for p, sents in zip(passages, per_passage):
        vecs = flat_vecs[cursor : cursor + len(sents)]
        cursor += len(sents)

        # C1 gets a lookup instead of a model call — same vectors, no inference.
        def lookup(texts: list[str], _v=vecs, _s=sents) -> np.ndarray:
            if len(texts) == len(_s):
                return _v
            idx = {t: i for i, t in enumerate(_s)}
            return np.vstack([_v[idx[t]] if t in idx else embedder.encode_one(t)
                              for t in texts])

        parent, children = parent_child_chunks(p.passage_id, p.text, p.lang)
        all_chunks += semantic_chunks(p.passage_id, p.text, p.lang, lookup)
        all_chunks += sliding_chunks(
            p.passage_id, p.text, p.lang, embedder.tokenize, embedder.detokenize
        )
        all_chunks += children
        all_chunks += question_chunks(p.passage_id, p.questions, p.lang)
        all_parents.append(parent)

        # The answer stage needs these exact vectors; they are already computed.
        if sents:
            index.sentence_cache[p.passage_id] = {s: vecs[i] for i, s in enumerate(sents)}

    if verbose:
        by_strategy: dict[str, int] = {}
        for c in all_chunks:
            by_strategy[c.strategy.value] = by_strategy.get(c.strategy.value, 0) + 1
        print(f"  chunks: {len(all_chunks)} {by_strategy}")

    # ---- pass 4: embed the chunk texts, also batched ------------------------
    vectors = np.zeros((len(all_chunks), index.dim), dtype=np.float32)
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors[i : i + len(batch)] = embedder.encode([c.text for c in batch])

    index.add(all_chunks, vectors)
    index.add_parents(all_parents)
    index.finalize()

    # The domain gate is fit on the corpus vectors themselves, so "off-topic"
    # means "unlike anything indexed" rather than a hand-written topic list.
    gate = DomainGate.fit(vectors)

    if verbose:
        print(f"  index built in {time.perf_counter() - t0:.1f}s ({index.size} chunks)")
    return (index, gate, vectors) if return_vectors else (index, gate)
