"""Measure whether any threshold separates answerable from unanswerable, using
the eval suite's own dataset, index and search path."""
import sys, json, statistics
sys.path.insert(0, ".")
from eval import dataset, index_build, target
from eval.pipeline import _search, _Context

N = 40
examples = dataset.load_examples(num_answerable=N, num_unanswerable=N)
print(f"examples: {sum(e.is_answerable for e in examples)} answerable / "
      f"{sum(not e.is_answerable for e in examples)} unanswerable")

index, records = index_build.build_index(examples)
embed_one = target.get_embedder().embed_one

sys.path.insert(0, "apps/api")
from goarag.embedder import get_embedder
from goarag.extractive import extract_answer
from goarag.schemas import Retrieved, Strategy

emb = get_embedder()
rows = []
for ex in examples:
    hits, _, _, recs = _search(ex.query_en, index, records, 5, embed_one)
    if not hits:
        continue
    ctxs = [_Context(text=r.text, source=f"q{r.query_id}", score=h.score) for r, h in zip(recs, hits)]
    retrieved = [Retrieved(chunk_id=f"e{i}", passage_id=c.source, text=c.text,
                           strategy=Strategy.PARENT, lang="en", score=c.score)
                 for i, c in enumerate(ctxs) if c.text.strip()]
    if not retrieved:
        continue
    qv = emb.encode_one(ex.query_en)
    _ans, conf = extract_answer(ex.query_en, qv, retrieved, emb)
    rows.append({"answerable": ex.is_answerable,
                 "top_score": float(max(h.score for h in hits)),
                 "conf": float(conf)})

json.dump(rows, open("calib.json", "w"), indent=1)
ans = [r for r in rows if r["answerable"]]
una = [r for r in rows if not r["answerable"]]

def stats(name, key, xs):
    v = sorted(r[key] for r in xs)
    print(f"  {name:12s} n={len(v):3d} min={v[0]:.3f} p10={v[len(v)//10]:.3f} "
          f"p50={v[len(v)//2]:.3f} p90={v[int(len(v)*0.9)]:.3f} max={v[-1]:.3f}")

print("\nTOP RETRIEVAL SCORE (cosine):")
stats("answerable", "top_score", ans); stats("unanswerable", "top_score", una)
print("\nEXTRACTIVE CONFIDENCE:")
stats("answerable", "conf", ans); stats("unanswerable", "conf", una)

print("\nSweep — refuse when signal < tau:")
for key, taus in (("top_score", [0.55,0.60,0.65,0.70,0.75,0.80]),
                  ("conf", [0.30,0.40,0.50,0.60,0.70])):
    print(f"  by {key}:")
    for t in taus:
        fc = sum(1 for r in una if r[key] >= t) / max(len(una),1)   # answered though unanswerable
        fr = sum(1 for r in ans if r[key] <  t) / max(len(ans),1)   # refused though answerable
        print(f"    tau={t:.2f}  false_confidence={fc:.3f}  false_refusal={fr:.3f}")
