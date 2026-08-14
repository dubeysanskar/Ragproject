"""Index snapshot — build once, load anywhere.

Boot cost is dominated by inference: ~11.5k embeddings (4.2k corpus sentences +
7.3k chunk texts). That is ~10 minutes on a 4-core laptop and far worse on a
1-vCPU VPS sharing its core with other services, which would make every deploy
a long outage and every container restart a cold demo.

Nothing about those vectors is machine-specific, so they are computed once and
shipped. Loading re-inserts them into Qdrant and rebuilds BM25 — both pure data
structure work, no model inference — turning a ~10 minute boot into seconds.

Format: one .npz for the float arrays, one .json for the metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .guardrails import DomainGate
from .retrieval import HybridIndex
from .schemas import Chunk


def save(index: HybridIndex, gate: DomainGate, vectors: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chunk order must match `vectors` row order — the loader re-pairs them.
    chunk_ids = list(index.chunks.keys())
    meta = {
        "dim": index.dim,
        "chunks": [index.chunks[c].model_dump(mode="json") for c in chunk_ids],
        "parents": [p.model_dump(mode="json") for p in index.parents.values()],
        # Flattened so the sentence vectors can live in one dense array.
        "sentence_index": [
            {"passage_id": pid, "sentences": list(cache.keys())}
            for pid, cache in index.sentence_cache.items()
        ],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )

    sentence_rows = [
        vec
        for cache in index.sentence_cache.values()
        for vec in cache.values()
    ]
    np.savez_compressed(
        out_dir / "vectors.npz",
        chunks=vectors.astype(np.float32),
        sentences=(
            np.vstack(sentence_rows).astype(np.float32)
            if sentence_rows
            else np.zeros((0, index.dim), dtype=np.float32)
        ),
        centroids=gate.centroids.astype(np.float32),
    )


def load(snapshot_dir: Path) -> tuple[HybridIndex, DomainGate]:
    meta = json.loads((snapshot_dir / "meta.json").read_text(encoding="utf-8"))
    arrays = np.load(snapshot_dir / "vectors.npz")

    index = HybridIndex(dim=meta["dim"])
    chunks = [Chunk.model_validate(c) for c in meta["chunks"]]
    index.add(chunks, arrays["chunks"])
    index.add_parents([Chunk.model_validate(p) for p in meta["parents"]])

    sentences = arrays["sentences"]
    cursor = 0
    for entry in meta["sentence_index"]:
        sents = entry["sentences"]
        block = sentences[cursor : cursor + len(sents)]
        cursor += len(sents)
        index.sentence_cache[entry["passage_id"]] = dict(zip(sents, block))

    index.finalize()  # BM25 is cheap to rebuild; not worth serialising
    return index, DomainGate(arrays["centroids"])
