"""Hybrid retrieval: dense ANN + BM25, fused with RRF (PDR §4 hot path).

Everything here is in-process and in-RAM. The only cost that scales with
corpus size is the ANN search itself; BM25 is scored over the same chunk set.
No network calls, no serialisation, no managed vector DB.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient, models

from .schemas import Chunk, Retrieved, Strategy

COLLECTION = "goarag"
_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize_lexical(text: str) -> list[str]:
    """Unicode-aware word split for BM25. `\\w+` with re.UNICODE keeps
    Devanagari/Tamil words intact, which a naive [a-z]+ would erase entirely."""
    return [t.lower() for t in _WORD.findall(text)]


class InvertedBM25:
    """BM25 over an inverted index.

    Replaces `rank_bm25.BM25Okapi`, which scores by walking *every* document
    for *every* query term in Python:

        q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])

    At 7.4k chunks that measured 76 ms mean / 178 ms max per query and was
    single-handedly blowing the 200 ms budget. Posting lists touch only the
    documents that actually contain a term, which is a tiny fraction of the
    corpus for the rare words that carry the ranking signal.
    """

    __slots__ = ("postings", "idf", "doc_len", "avgdl", "n", "k1", "b")

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0

        raw: dict[str, list[tuple[int, int]]] = {}
        for idx, doc in enumerate(corpus):
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            for term, tf in counts.items():
                raw.setdefault(term, []).append((idx, tf))

        # Freeze each posting list into arrays so scoring is vectorised.
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for term, plist in raw.items():
            docs = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            freqs = np.fromiter((f for _, f in plist), dtype=np.float32, count=len(plist))
            self.postings[term] = (docs, freqs)
            df = len(plist)
            self.idf[term] = np.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def top_n(self, query: list[str], n: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in query:
            entry = self.postings.get(term)
            if entry is None:
                continue
            docs, freqs = entry
            dl = self.doc_len[docs]
            denom = freqs + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            contrib = self.idf[term] * (freqs * (self.k1 + 1)) / denom
            for d, c in zip(docs.tolist(), contrib.tolist()):
                scores[d] = scores.get(d, 0.0) + c
        if not scores:
            return []
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


class HybridIndex:
    """Owns the vector collection, the BM25 table, and the parent lookup."""

    def __init__(self, dim: int) -> None:
        # ":memory:" keeps Qdrant embedded in this process — a function call,
        # not a network hop. This is the single biggest latency decision.
        self.client = QdrantClient(":memory:")
        self.dim = dim
        self.chunks: dict[str, Chunk] = {}
        self.parents: dict[str, Chunk] = {}
        self.parents_by_passage: dict[str, Chunk] = {}
        # Sentence vectors per passage, precomputed at build time. Passage text
        # never changes, so re-encoding candidates per request would be paying
        # ~50 ms for an answer we could look up. See extractive.extract_answer.
        self.sentence_cache: dict[str, dict[str, np.ndarray]] = {}
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=self.dim, distance=models.Distance.COSINE
            ),
        )

    # ------------------------------------------------------------- build time
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = []
        for i, ch in enumerate(chunks):
            self.chunks[ch.chunk_id] = ch
            points.append(
                models.PointStruct(
                    # Qdrant wants int/uuid ids; the real id lives in the payload.
                    id=len(self.chunks) - 1,
                    vector=vectors[i].tolist(),
                    payload={
                        "chunk_id": ch.chunk_id,
                        "passage_id": ch.passage_id,
                        "strategy": ch.strategy.value,
                        "lang": ch.lang,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=COLLECTION, points=points, wait=True)

    def add_parents(self, parents: list[Chunk]) -> None:
        for p in parents:
            self.parents[p.chunk_id] = p
            # C5 keys are questions and know only their passage_id, so parents
            # need to be reachable that way too.
            self.parents_by_passage[p.passage_id] = p

    def finalize(self) -> None:
        """Build the BM25 inverted index once, after all chunks are in."""
        self._bm25_ids = list(self.chunks.keys())
        corpus = [tokenize_lexical(self.chunks[c].text) for c in self._bm25_ids]
        self._bm25 = InvertedBM25(corpus) if corpus else None

    # --------------------------------------------------------------- hot path
    def dense(
        self, qvec: np.ndarray, k: int, strategies: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """`strategies` restricts the search to given chunkers. Unused on the
        hot path (all strategies compete); it exists so scripts/ablate.py can
        measure one strategy at a time against a single index build."""
        flt = (
            models.Filter(
                must=[models.FieldCondition(
                    key="strategy", match=models.MatchAny(any=sorted(strategies))
                )]
            )
            if strategies
            else None
        )
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=qvec.tolist(),
            limit=k,
            with_payload=True,
            query_filter=flt,
        ).points
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]

    def lexical(
        self, query: str, k: int, strategies: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        # Over-fetch then filter: the BM25 index is flat, so restricting has to
        # happen after scoring.
        n = k * 6 if strategies else k
        out: list[tuple[str, float]] = []
        for i, score in self._bm25.top_n(tokenize_lexical(query), n):
            cid = self._bm25_ids[i]
            if strategies and self.chunks[cid].strategy.value not in strategies:
                continue
            out.append((cid, score))
            if len(out) >= k:
                break
        return out

    def resolve(self, chunk_id: str) -> Chunk:
        """Map a matched key back to the text worth answering from.

        C3: a child match returns its parent, so the answer stage gets full
        context instead of the one sentence that happened to match.
        C5: a question match returns the passage that *answers* it — without
        this the pipeline cheerfully echoes the user's own question back, since
        the matched chunk's text is the question itself.
        """
        ch = self.chunks[chunk_id]
        if ch.strategy is Strategy.QUESTION and ch.answers_passage:
            parent = self.parents_by_passage.get(ch.answers_passage)
            if parent is not None:
                return parent
        if ch.parent_id and ch.parent_id in self.parents:
            return self.parents[ch.parent_id]
        return ch

    @property
    def size(self) -> int:
        return len(self.chunks)


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> dict[str, float]:
    """Reciprocal Rank Fusion. Rank-based, so dense cosines and BM25 scores —
    which live on completely different scales — combine without normalisation."""
    fused: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, (cid, _score) in enumerate(lst):
            fused[cid] += 1.0 / (k + rank + 1)
    return fused


def retrieve(
    index: HybridIndex,
    query: str,
    qvec: np.ndarray,
    lang: str | None,
    top_k: int = 4,
    pool: int = 40,
    lang_boost: float = 0.15,
    strategies: set[str] | None = None,
    use_lexical: bool = True,
) -> list[Retrieved]:
    """Dense + lexical → RRF → language boost (C4) → dedupe by passage → parents.

    `strategies` and `use_lexical` are ablation knobs; both default to the full
    system that actually runs in production.
    """
    dense_hits = index.dense(qvec, pool, strategies)
    lexical_hits = index.lexical(query, pool, strategies) if use_lexical else []

    fused = rrf_fuse([dense_hits, lexical_hits])
    if not fused:
        return []

    dense_rank = {cid: i for i, (cid, _) in enumerate(dense_hits)}
    dense_score = {cid: s for cid, s in dense_hits}
    lex_rank = {cid: i for i, (cid, _) in enumerate(lexical_hits)}

    # C4: same-language chunks get a relative lift, but other languages stay in
    # the pool — cross-lingual fallback is the point of a multilingual encoder.
    if lang:
        for cid in list(fused):
            if index.chunks[cid].lang == lang:
                fused[cid] *= 1.0 + lang_boost

    order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    seen_passages: set[str] = set()
    out: list[Retrieved] = []
    for cid, score in order:
        resolved = index.resolve(cid)
        if resolved.passage_id in seen_passages:
            continue
        seen_passages.add(resolved.passage_id)
        src = index.chunks[cid]
        out.append(
            Retrieved(
                chunk_id=resolved.chunk_id,
                passage_id=resolved.passage_id,
                text=resolved.text,
                strategy=src.strategy,
                lang=src.lang,
                score=float(score),
                dense_score=float(dense_score.get(cid, 0.0)),
                dense_rank=dense_rank.get(cid),
                lexical_rank=lex_rank.get(cid),
            )
        )
        if len(out) >= top_k:
            break
    return out
