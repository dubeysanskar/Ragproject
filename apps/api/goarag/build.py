"""Offline index construction (PDR §3). Not on the hot path — this is allowed
to be slow, and is where the whole chunking matrix gets applied."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .chunking import build_all
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
    batch_size: int = 256,
    verbose: bool = True,
) -> tuple[HybridIndex, DomainGate]:
    t0 = time.perf_counter()
    index = HybridIndex(dim=len(embedder.encode_one("dim probe")))

    all_chunks: list[Chunk] = []
    all_parents: list[Chunk] = []
    for p in passages:
        indexed, parents = build_all(
            p.passage_id,
            p.text,
            p.lang,
            p.questions,
            embed=embedder.encode,
            tokenize=embedder.tokenize,
            detokenize=embedder.detokenize,
        )
        all_chunks.extend(indexed)
        all_parents.extend(parents)

    if verbose:
        by_strategy: dict[str, int] = {}
        for c in all_chunks:
            by_strategy[c.strategy.value] = by_strategy.get(c.strategy.value, 0) + 1
        print(f"  chunks: {len(all_chunks)} from {len(passages)} passages {by_strategy}")

    vectors = np.zeros((len(all_chunks), index.dim), dtype=np.float32)
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors[i : i + len(batch)] = embedder.encode([c.text for c in batch])

    index.add(all_chunks, vectors)
    index.add_parents(all_parents)

    # Precompute sentence vectors for every passage the answer stage can reach.
    # Paid once here so the hot path never encodes candidate sentences.
    from .chunking import sentences as split_sentences

    for parent in all_parents:
        sents = split_sentences(parent.text)
        if sents:
            index.sentence_cache[parent.passage_id] = (sents, embedder.encode(sents))

    index.finalize()

    # The domain gate is fit on the corpus vectors themselves, so "off-topic"
    # means "unlike anything indexed" rather than a hand-written topic list.
    gate = DomainGate.fit(vectors)

    if verbose:
        print(f"  index built in {time.perf_counter() - t0:.1f}s ({index.size} chunks)")
    return index, gate
