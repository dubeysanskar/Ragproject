# GoaRAG — voice-enabled RAG · HH Goa 2026 Task 2

**`#RAGInGoa`** · speak a question in English, हिन्दी or தமிழ் → grounded answer
from `ai4bharat/MSMARCO-XI`, **query text → final output in well under 200 ms**.

```
Voice → Sarvam STT → LangDetect → Embed → Guards → Hybrid retrieve (RRF)
      → Extract answer  ─┬─→ grounded answer + citations
                         └─→ (escalate) Generate → GroundCheck → answer
```

---

## The thesis: why this is fast

200 ms through a normal cloud-LLM call is impossible — network alone eats
50–150 ms and generation eats 500 ms+. Anything that bolts an LLM onto a hosted
vector DB misses the target. So **nothing in the measured path leaves the
process**:

| Decision | Effect |
|---|---|
| Qdrant **embedded** (`:memory:`), not a hosted cluster | retrieval is a function call, ~2 ms |
| **Local int8 ONNX** query embedding | ~6 ms, no API, no rate limit |
| **Extractive-first** answering | MS MARCO passages *are* answers — no generation needed, and copied text cannot hallucinate |
| **Precomputed sentence vectors** | answer selection is a matrix multiply: 0.35 ms instead of 53 ms |
| Generation only on **escalation**, outside the budget | reported separately and honestly |

STT is deliberately outside the measured budget: it streams while the user
speaks, so the query text exists the moment they stop talking. The budget
starts at query text, which is the quantity the task specifies
(*chunking + retrieval + everything through to final output*).

---

## Measured results

From `bench/report.md` — 132 queries (120 MSMARCO-XI eval + 10 out-of-scope +
2 unsafe), against the live API over HTTP:

| | P50 | P70 | P90 | P100 |
|---|---|---|---|---|
| **All queries** | **68.4** | **80.3** | 94.8 | **144.9 ms** |
| English | 51.3 | 55.2 | 68.4 | 111.7 ms |
| हिन्दी | 79.4 | 87.8 | 108.2 | 144.9 ms |
| தமிழ் | 74.6 | 84.5 | 93.8 | 112.2 ms |

**P100 = 144.9 ms — under the 200 ms target on every query, in every language.**

Per-stage, warm (`/ask` waterfall, visible live in the UI):

```
guard_input    0.01 ms
lang_detect    0.01 ms     ← script ranges, not a model
embed          6.11 ms     ← local ONNX, the irreducible floor
guard_domain   0.02 ms
retrieve       2.06 ms     ← dense ANN + BM25 + RRF, in-process
extract        0.35 ms     ← precomputed sentence vectors
```

Quality: **114/120 answered**, and of those, the gold passage was retrieved
**114/114 (100%)**.

Reproduce: `python scripts/bench.py --api http://127.0.0.1:8099`

---

## Chunking — five strategies, indexed side by side

Not one splitter. Every passage goes through the whole matrix, all strategies
land in one collection tagged by `strategy_id`, and retrieval fuses across them.

| | Strategy | How | Why |
|---|---|---|---|
| **C1** | Semantic | Indic-aware sentence split (`।`, `॥`), merge neighbours while cosine > τ, break at drops | chunks follow meaning, not character counts |
| **C2** | Sliding window | 256-token windows, 64 overlap, using **the embedding model's own tokenizer** | in Indic scripts a character count is a meaningless proxy for tokens |
| **C3** | Parent–child | index 1–2 sentence children, return the parent passage | precise matching + full context to answer from |
| **C4** | Metadata-aware | every chunk carries `{lang, passage_id, position, char_span, strategy}`; same-language hits boosted, others still eligible | language filtering is a real relevance win on a multilingual corpus |
| **C5** | Question-indexed | embed MS MARCO's **own queries** as alternate keys pointing at their passage | a spoken question matches a stored question far better than it matches prose |

Retrieval: query embed → dense ANN over all strategies **+** BM25 lexical →
**Reciprocal Rank Fusion** → language boost (C4) → dedupe by `passage_id` →
resolve children→parents and questions→answering passage → top-4.

> **C5 subtlety that bites:** a question-chunk's *text is the question*. Return
> it directly and the system confidently echoes the user's own question back as
> the answer. `resolve()` maps a C5 hit to the passage that answers it — this
> was a real bug, caught because the demo answer read exactly like the query.

---

## Guardrails — knowing when not to answer

| # | Gate | Mechanism |
|---|---|---|
| 1 | STT confidence | below threshold → "I didn't catch that" — never retrieve on garbage |
| 2 | Unsafe input | multilingual regex blocklist → refusal, never reaches retrieval |
| 3 | **No-grounding floor** | top hit's **raw cosine** below τ → refuse instead of guessing |
| 4 | Injection resistance | retrieved chunks sanitised and wrapped as data, never instructions |
| 5 | Groundedness (generative) | every generated sentence must be supported by a retrieved chunk; unsupported sentences stripped; >50% loss → refusal stands |
| 6 | Citation enforcement | a generative answer with no `[n]` marker is rejected and downgraded |

**On the off-topic gate — a negative result worth stating.** The PDR proposed
comparing the query against corpus *centroids*. Calibrated on the real corpus,
that signal does not separate the classes:

| τ | off-topic allowed | in-domain refused |
|---|---|---|
| 0.30 | 8/10 | 4/120 |
| 0.40 | 2/10 | 15/120 |
| 0.45 | 1/10 | 21/120 |

A centroid is the average of ~460 unrelated web passages, so it sits roughly
equidistant from everything. There is no threshold that refuses off-topic
queries without refusing real ones. The centroid check was demoted to a coarse
pre-filter and the decision moved to **top-1 passage cosine**, which measures
the thing that actually matters: does anything in the corpus answer this?

---

## Harness

`goarag/harness.py` — a real orchestrator, not a prompt string.

- **Typed stages** with Pydantic models on every boundary (`schemas.py`).
- **Per-stage timing** into a ring-buffer trace store; `/stats` serves live
  percentiles, and the UI renders the waterfall per query.
- **Fallback edges:** Sarvam → ElevenLabs (STT); Groq → Gemini → extractive-only
  (generation); low extractive confidence → escalate; no provider → honest
  refusal, never a guess.
- **Hot path stays sync and off the event loop** (`run_in_threadpool`), so one
  slow request cannot inflate another's measured latency.

---

## Corpus

`ai4bharat/MSMARCO-XI` is **55.6 GB / 11.4 M rows** — far too large to pull, and
HF's rows/filter API is broken for it (`ArrowNotImplementedError` server-side).
`scripts/build_corpus.py` therefore streams parquet row batches over HTTP and
stops early, never downloading a full file.

Indexed subset (declared honestly — scale is a knob, not an architecture change):
**1,200 passages → 7,403 chunks**, 600 en / 400 ta / 200 hi, plus 120 held-out
eval queries with gold answers and gold passage ids.

Each query contributes its **gold** passage *and* a **distractor** from the same
result list — a corpus of only gold passages would make Recall@5 meaninglessly
easy.

```bash
python scripts/build_corpus.py --langs hin_Deva tam_Taml --rows-per-lang 400
```

---

## Run it

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r apps/api/requirements.txt
python scripts/build_corpus.py                      # writes data/corpus.json
cd apps/api && uvicorn main:app --port 8099          # boots with the index in RAM
cd apps/web && npm install && npm run dev            # http://localhost:3002
```

Runs at **$0 with no keys at all**: embeddings, index, and the extractive path
are entirely local. Keys only unlock voice (`SARVAM_API_KEY`) and the generative
escalation path (`GROQ_API_KEY` / `GEMINI_API_KEY`). See `apps/api/.env.example`.

> **Python 3.13 note:** pin *floors*, not exact versions — `fastembed==0.5.1`
> and `mmh3<5` have no cp313 wheel and fall back to a source build needing MSVC.
> The PDR's `intfloat/multilingual-e5-small` is no longer in fastembed's
> catalog; we use `paraphrase-multilingual-MiniLM-L12-v2` — same 384 dims,
> measured cross-lingual cosine **0.995** en↔hi paraphrase vs ~0.00 unrelated.
