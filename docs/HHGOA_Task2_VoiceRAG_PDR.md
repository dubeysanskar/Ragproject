# PDR — "GoaRAG" · HH Goa 2026 Task 2: Voice-Enabled RAG Pipeline

**Codename:** `goa-rag`
**Hashtag:** `#RAGInGoa`
**Deadline:** Aug 22, 2026, 11:59 PM IST (launched Aug 13 — 9 days)
**Dataset:** `ai4bharat/MSMARCO-XI` (MS MARCO translated into Indian languages by AI4Bharat)
**Pipeline shape:** Voice → STT → Retrieval (vector DB) → Grounded answer
**Hard constraints:** Sarvam **or** ElevenLabs STT · non-naive multi-strategy chunking · **<200 ms** chunking+retrieval→final output · P50/P70/P100 across many queries · proper harness · guardrails · no resubmissions.

---

## 1. Strategy: how to actually hit 200 ms (the thesis)

200 ms through *answer generation* is impossible with a normal cloud-LLM call (network alone eats 50–150 ms; generation eats 500 ms+). Everyone who bolts GPT onto Pinecone will miss the target. Our whole architecture is built backwards from the latency budget:

**Design principle: everything hot lives in one process, in RAM, on one box. Zero network hops inside the measured path.**

1. **In-process vector index** — no managed vector DB over the network. Qdrant *embedded/in-memory* (or FAISS HNSW) inside the FastAPI process. Retrieval = a function call, ~2–8 ms.
2. **Local ONNX embedding of the query** — `fastembed` with a quantized multilingual model (`intfloat/multilingual-e5-small`, int8 ONNX). Query embedding ≈ 5–12 ms on CPU. No embedding API calls, ever, at query time.
3. **Answer-first architecture (the unique bit):** MS MARCO is a QA dataset — passages exist *because* they answer queries. So the primary answer path is **extractive, not generative**:
   - Retrieve top-k chunks (dense + BM25 hybrid, fused with RRF).
   - A cross-encoder-free **lexical+embedding span scorer** picks the best answering sentence(s) from the top chunks (~5–15 ms).
   - If the groundedness/confidence gate passes → return the extracted, cited answer. **Total: comfortably under 200 ms, typically 30–80 ms.**
4. **Generative fallback (outside the strict budget, streamed):** when the query needs synthesis across chunks, the harness escalates to a fast-inference LLM (Groq `llama-3.1-8b-instant` or Cerebras — ~100 ms to first token) and streams. Latency analytics report both paths separately and honestly: `P50/P70/P100 (retrieval→extractive answer)` is the headline number that meets the spec; TTFT for the generative path is reported alongside.
5. **STT is streamed and overlapped:** transcription happens *while the user speaks* (streaming STT), so by the time they stop talking the query text is already there — perceived latency ≈ retrieval latency.

This "extractive-first, generative-fallback" design is defensible, honest about what 200 ms permits, and different from 95% of submissions.

---

## 2. Speech-to-text (pick: **Sarvam**)

**Choice: Sarvam `saarika` (STT) / `saaras` (STT-translate), streaming API.**
Rationale: the dataset is AI4Bharat's *Indic* MS MARCO — the demo should take questions in Hindi/Marathi/Tamil/etc. Sarvam is built for Indic ASR, returns language codes we reuse for metadata-aware retrieval, and has low-latency streaming from India (servers close to Goa judges 🌴). ElevenLabs Scribe is the fallback if Sarvam quota/keys fail — the STT layer is behind an interface (`transcribe(audio) -> {text, lang, confidence}`) so swapping is a config change, which is itself harness evidence.

Details:
- Browser captures mic via `MediaRecorder`/Web Audio → 16 kHz PCM chunks over WebSocket to our backend → proxied to Sarvam streaming endpoint.
- Partial transcripts shown live in the UI (great for the demo video).
- STT confidence < threshold → guardrail asks the user to repeat (never retrieve on garbage).
- Detected language tag flows downstream as a retrieval filter.

---

## 3. Dataset & indexing (offline, not in the latency path)

`ai4bharat/MSMARCO-XI`: MS MARCO passages/queries machine-translated into ~11 Indic languages (+ English). Plan:

- Scope the index to a manageable, honest subset for the live demo (e.g., 100k–300k passages across 2–4 languages: English + Hindi + one Dravidian + one more) — declared clearly in the README. Full-corpus indexing is a scale knob, not an architecture change.
- Offline pipeline (`scripts/build_index.py`): download via `datasets` → clean → chunk (below) → embed with the same e5-small ONNX model (batch, GPU optional) → write Qdrant snapshot + BM25 (tantivy/`rank-bm25`) index + a `chunks.parquet` with metadata. The API server loads the snapshot into RAM at boot.

---

## 4. Chunking — the multi-strategy system (explicit requirement)

Not one splitter. A **chunking matrix**, indexed side-by-side, with retrieval fused across strategies:

| # | Strategy | How | Why |
|---|---|---|---|
| C1 | **Semantic splitting** | Sentence-split (Indic-aware: `indic-nlp-library` sentence tokenizer, handles ।), then merge adjacent sentences while embedding cosine similarity > τ; break at semantic drops | Chunks follow meaning, not character counts |
| C2 | **Sliding window w/ overlap** | 256-token windows, 64-token overlap (token counts via the embedding model's tokenizer, not chars — critical for Indic scripts where chars ≠ tokens) | Guarantees no answer straddles a boundary |
| C3 | **Parent-child (small-to-big)** | Index child chunks of 1–2 sentences for precise matching; retrieval returns the *parent* passage for context | Best precision for matching + best recall for answering |
| C4 | **Metadata-aware** | Every chunk carries `{lang, passage_id, source_query_ids, position, char_span, strategy_id}`; retrieval filters by detected query language first, cross-lingual fallback second | Indic multilingual corpus makes language-filtering a real relevance win |
| C5 | **Question-indexed chunks** | MSMARCO pairs queries with passages — embed the *original questions* as alternate keys pointing at their passages (HyDE-in-reverse, using ground truth) | A spoken question matches a stored question far better than it matches prose |

**Retrieval flow (hot path):** query embedding → parallel: dense ANN over C1+C2+C3-children+C5 (single Qdrant collection, `strategy_id` in payload) + BM25 lexical → **Reciprocal Rank Fusion** → language-filter boost (C4) → dedupe by `passage_id` → resolve children→parents → top-4 chunks to the answer stage. Every stage timed individually.

Ablation notebook (`notebooks/chunking_ablation.ipynb`) reports Recall@5 / MRR@10 per strategy vs fused on MSMARCO-XI dev queries — this is the "real thought" evidence the judges asked for.

---

## 5. Answer generation

**Path 1 — Extractive (default, in-budget):**
- Candidate sentences from top chunks scored by: embedding similarity to query + lexical overlap + answer-type match (who/when/how-many heuristics per language).
- Return best sentence(s) verbatim **with citation** (passage_id + highlighted span in UI).
- This is also inherently hallucination-proof — you cannot hallucinate text you copied.

**Path 2 — Generative (escalation):**
- Trigger: multi-hop/comparative/summarize-style queries (classifier heuristic: query length, conjunctions, low single-chunk coverage).
- Groq/Cerebras 8B, strict prompt: answer *only* from the numbered context, cite `[1][2]`, reply "I don't have that in my knowledge base" if uncounted.
- `max_tokens=150`, streamed to UI token-by-token.
- Responds in the query's language (Sarvam lang tag → instruction).

---

## 5.5 Provider matrix — free-first stack, paid upgrades optional

GoaRAG is deliberately **free-first**: the hot path is local/in-process (that's *why* it's fast), so the free choices aren't a compromise — they're the design. Paid options exist only as upgrades for quality headroom or demo-day rate limits. Every provider sits behind an adapter interface with a `PROVIDER_*` env flag; the repo runs end-to-end at **$0 by default**.

| Layer | Free default (in repo & live) | Paid upgrade (optional) | Notes |
|---|---|---|---|
| **Speech-to-text** | **Sarvam free/starter API credits** (task mandates Sarvam or ElevenLabs — pick stays Sarvam) | Sarvam paid tier (higher rate limits, priority) | Fallback adapter: ElevenLabs free starter credits. Dev-only offline path: `faster-whisper` small — never in the submitted pipeline, clearly labeled |
| **Query embeddings** | **Local ONNX `multilingual-e5-small` int8** via `fastembed` — $0, ~5–12 ms, no rate limits | Cohere `embed-multilingual-v3` / OpenAI `text-embedding-3-small` | Local is *faster* than any API (no network) — the paid option is quality headroom for the offline index build only, never the hot path |
| **Corpus embedding (offline)** | Same local model, batch on CPU (or free Colab/Kaggle GPU for speed) | Managed embedding API batch job | Offline step — latency irrelevant, so free always suffices |
| **Vector store** | **Embedded Qdrant (in-process, RAM snapshot)** — $0 | Qdrant Cloud free tier → paid cluster (durability, multi-instance) | Embedded is the latency play; cloud only if we ever scale past one box |
| **Lexical index** | `rank-bm25` / tantivy in-process — $0 | — | No paid equivalent needed |
| **Fallback LLM (generative path)** | **Groq free tier `llama-3.1-8b-instant`** (~100 ms TTFT) or **Gemini 2.0 Flash free tier** | OpenAI `gpt-4o-mini` / Claude Haiku (rate-limit headroom, consistency) | Adapter fallback chain: Groq → Gemini → paid key if set — this chain doubles as harness error-recovery |
| **Safety classifier** | Keyword/regex blocklists + tiny ONNX toxicity model (local, $0); **Llama-Guard via Groq free tier** for a second opinion | Azure Content Safety / OpenAI Moderation (moderation is actually free too) | Effectively $0 either way |
| **Groundedness check** | In-process overlap + embedding-similarity stripper (sync, $0) | Async LLM-as-judge sampling (gpt-4o-mini) for the eval report | Free check guards every answer; paid judge only decorates the README metrics |
| **API hosting** | **Fly.io free allowance / Render free** + 5-min keep-alive ping (cron-job.org, free) to prevent spin-down | Railway/Fly paid always-on (no cold-start risk during judging) | The one place ~$5–10 of paid spend is genuinely worth it for the judging window |
| **Web hosting** | Vercel Hobby (free) | — | Sufficient |
| **Bench/analytics** | Self-hosted trace store + matplotlib report — $0 | — | No external APM needed |

**Bottom line:** fully free = $0 and meets the 200 ms target (local embeddings + embedded index are the fastest options available at any price). Recommended spend: ~$5–10 on always-on API hosting for the judging window; everything else stays free. State this table in the README — cost-aware engineering reads well to judges.

---

## 6. Harness (explicit requirement)

A real orchestrator, not a prompt string — `app/harness/`:

- **Pipeline graph:** `STT → LangDetect → Embed → Retrieve → Fuse → Gate → [Extract | Generate] → GroundCheck → Respond`, each stage a typed node with **Pydantic input/output models** (structured I/O requirement ✅).
- **Per-stage timeouts** (embed 50 ms, retrieve 50 ms, extract 60 ms; generative path 3 s) with **fallback edges**: Sarvam→ElevenLabs on STT failure; dense-only→hybrid on BM25 failure; generative→extractive on LLM timeout; total failure → honest error message, never a hang.
- **Retries:** exponential backoff (2 attempts) on network stages only — never on in-process stages (protects the latency budget).
- **Tracing:** every request gets a `trace_id`; per-stage `t_start/t_end` appended to a ring buffer + JSONL log. This trace store *is* the latency-analytics source.
- **Circuit breaker** on external APIs (open after 5 consecutive failures, half-open probes).
- **Structured errors:** every failure mode maps to a typed error the UI renders distinctly (mic denied / STT failed / low confidence / off-topic / not grounded).

---

## 7. Guardrails (explicit requirement) — "knows when not to answer"

**Input guardrails:**
1. **STT confidence gate** — below threshold → "I didn't catch that, could you repeat?"
2. **Off-topic gate** — query embedded and compared against a centroid set of the corpus embedding space; distance beyond τ_domain → "That's outside my knowledge base (MS MARCO web passages). Try asking about …" with 3 sampled example questions. Cheap (one cosine), in-budget.
3. **Unsafe-input filter** — multilingual keyword/regex blocklist + a tiny ONNX toxicity classifier; unsafe → firm refusal template, logged, never sent to retrieval.
4. **Injection resistance** — retrieved chunks are wrapped as data (`<context>` with numbered items); system prompt instructs the model to treat context as quotable material, never instructions.

**Output guardrails:**
5. **Retrieval-score floor** — top fused score < τ_retrieval → refuse with "not enough grounded context" instead of guessing.
6. **Groundedness check (generative path)** — every generated sentence must exceed an entailment-lite threshold: max cosine + n-gram overlap vs the retrieved chunks; failing sentences are stripped; if the answer loses >50% of its sentences → downgrade to extractive answer or refuse. (Extractive path is grounded by construction.)
7. **Citation enforcement** — generative answers missing `[n]` markers are rejected and retried once with a stricter instruction, then downgraded.
8. **No-answer honesty set in the eval:** the latency/quality harness includes deliberately out-of-scope and adversarial queries, and the README reports **refusal precision** (how often we correctly said "I don't know") — turns the guardrail requirement into a measured number.

---

## 8. Latency analytics (explicit requirement)

`scripts/bench.py`:
- Loads **300 queries**: 200 MSMARCO-XI dev queries (mixed languages) + 50 paraphrased + 50 out-of-scope (guardrail hits count as completed responses).
- Runs them against the live API (same box, localhost, to measure pipeline not internet — plus a second run over the public URL for transparency).
- Records per-stage and end-to-end timings from the trace store.
- Outputs `bench/report.md` + `bench/latencies.csv` + a histogram PNG:
  - **P50 / P70 / P90 / P95 / P100** for: (a) retrieval→extractive answer [the spec'd number], (b) each stage, (c) generative-path TTFT, (d) STT partial-final lag.
- Target/expected: extractive path **P50 ≈ 35–60 ms, P100 < 200 ms** on a 4-vCPU box with the index in RAM.
- The live UI shows a per-query latency waterfall (stage bars) — spectacular in the demo video and self-evidencing for judges.

---

## 9. System architecture & deployment

```
┌─ Frontend (Next.js, Vercel) ──────────────────────────────┐
│ Mic capture → WS audio stream → live transcript →         │
│ answer + citations + latency waterfall + history          │
└──────────────▲────────────────────────────────────────────┘
               │ WebSocket + REST
┌──────────────┴─ Backend (FastAPI, single Docker, Railway/Fly.io — │
│                 NOT serverless: index must stay warm in RAM)      │
│  /ws/audio  → Sarvam streaming proxy                              │
│  /ask       → harness pipeline (hot path, all in-process)         │
│  /bench,/stats → trace store, percentile endpoints                │
│  RAM: Qdrant embedded snapshot + BM25 + ONNX e5-small + scorer    │
└───────────────────────────────────────────────────────────────────┘
External (never in hot path): Sarvam STT (streamed, overlapped),
Groq/Cerebras (fallback path only)
```

- Backend box: 4 vCPU / 8 GB (index for 300k chunks × 384-dim int8 ≈ well under 1 GB).
- Vercel hosts the frontend + `/s/` share pages only — serverless cold starts would murder the latency SLO, so the RAG core deliberately lives on an always-on container. State this reasoning in the README; it reads as engineering maturity.
- Repo: monorepo `apps/web`, `apps/api`, `scripts/`, `notebooks/`, `bench/`, Dockerfile, `docker-compose up` reproduces everything; README with architecture diagram, chunking ablation table, latency report, guardrail demo GIFs.

---

## 10. Build plan (9 days)

| Day | Milestone |
|---|---|
| 1 | Repo scaffold; dataset pull; C2 baseline chunker; embed+index 50k; `/ask` extractive path returns answers locally |
| 2 | Full chunking matrix C1–C5; hybrid BM25+RRF; ablation notebook first pass |
| 3 | Harness: typed stages, timeouts, fallbacks, trace store; per-stage timing solid |
| 4 | Sarvam streaming STT via WS proxy; frontend mic UI + live transcript |
| 5 | Guardrails 1–7; generative fallback via Groq; groundedness stripper |
| 6 | Frontend polish: citations, highlighted spans, latency waterfall, Goa-brand styling (reuse Task-1 palette — nice cross-submission identity) |
| 7 | Deploy (Fly/Railway + Vercel); `bench.py` 300-query run on prod; tune thresholds; refusal-precision eval |
| 8 | Record Video 1 (90 s team/process) + Video 2 (end-to-end demo incl. Hindi query, off-topic refusal, latency waterfall); edit |
| 9 | Buffer. Everyone posts both videos on Instagram + X + LinkedIn with **#RAGInGoa** (≥1 public IG); verify every post has the hashtag; fill form once — **no resubmissions** |

## 11. Demo video shot list (Video 2)
1. English spoken question → instant extracted answer + citation + waterfall showing ~50 ms.
2. Hindi spoken question → answer in Hindi (language-aware retrieval on screen).
3. Complex "compare/summarize" question → harness escalates, streamed generative answer with [1][2] citations.
4. Off-topic question ("what's the weather in Goa?") → polite refusal + suggested questions (guardrails on camera).
5. `bench/report.md` scroll: P50/P70/P100 table + histogram.

## 12. Submission checklist
- [ ] GitHub repo public, README complete (architecture, ablations, latency report, guardrail evidence)
- [ ] Live link warm and tested from a phone on mobile data
- [ ] P50/P70/P100 from ≥300 queries in repo and in the form
- [ ] Video 1 (process, 90 s) + Video 2 (demo) final
- [ ] **Every team member** posted **both videos** on **Instagram + X + LinkedIn**, all with `#RAGInGoa`, ≥1 Instagram account public
- [ ] Form: https://forms.gle/MNvCjcv23Hn2Eeu58 — submitted once, before Aug 22, 11:59 PM
