"""Build a portable index snapshot (see goarag/snapshot.py).

Run this wherever CPU is cheap, ship data/snapshot/ to the server, and the API
boots in seconds instead of re-embedding 11.5k texts on a 1-vCPU box.

  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe scripts/build_snapshot.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from goarag.build import Passage, build_index  # noqa: E402
from goarag.embedder import get_embedder  # noqa: E402
from goarag.snapshot import load as snapshot_load, save as snapshot_save  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", type=Path, default=Path("data/corpus.json"))
ap.add_argument("--out", type=Path, default=Path("data/snapshot"))
args = ap.parse_args()

raw = json.loads(args.corpus.read_text(encoding="utf-8"))
passages = [
    Passage(str(r["passage_id"]), r["text"], r.get("lang", "en"), tuple(r.get("questions", ())))
    for r in raw
]
print(f"corpus: {len(passages)} passages")

embedder = get_embedder()
index, gate, vectors = build_index(passages, embedder, return_vectors=True)
snapshot_save(index, gate, vectors, args.out)

size = sum(f.stat().st_size for f in args.out.iterdir()) / 1e6
print(f"wrote snapshot -> {args.out} ({size:.1f} MB)")

# Verify the snapshot round-trips before anyone ships it.
t0 = time.perf_counter()
loaded, loaded_gate = snapshot_load(args.out)
print(f"reload check: {loaded.size} chunks in {time.perf_counter() - t0:.1f}s "
      f"(built: {index.size})")
assert loaded.size == index.size, "chunk count mismatch"
assert len(loaded.sentence_cache) == len(index.sentence_cache), "sentence cache mismatch"
print("snapshot verified")
