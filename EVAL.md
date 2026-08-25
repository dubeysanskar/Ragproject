# Third-party evaluation — BeaconBandhu/rag-local-eval-loop

The judges' harness drives our real code in-process (its `TARGET_INTERFACE.md`
contract), scores it against MSMARCO-XI's own labels, and explicitly ignores
whatever a team's README claims. Our adapter lives in [`app/`](app/):

| module | what it exposes |
|---|---|
| `app/embedder.py` | `embed` / `embed_one` / `get_model` → the same local ONNX MiniLM the service uses |
| `app/generator.py` | `generate_answer(query, results)` → the same extractive answerer |
| `app/config.py` | optional `LATENCY_BUDGET_MS`, model label |

```bash
pip install -r eval-requirements.txt
RAG_PROJECT_ROOT=. python -m eval.runner --num-answerable 40 --num-unanswerable 40 --workers 1
```

Note the suite builds **its own throwaway index** and never touches ours, so
its retrieval numbers score our *embedding model*, not our chunking matrix.
The chunking ablation in [`bench/ablation.md`](bench/ablation.md) is what
measures that.

## Measured (40 answerable + 40 unanswerable, held-out seed 7)

| check | result |
|---|---|
| Recall@1 / @3 / @5 (cross-lingual) | 0.350 / 0.725 / 0.825 |
| MRR | 0.552 |
| False refusal rate | 0.300 |
| False confidence rate | 0.350 |
| Retrieval p95 | ~10 ms vs 50 ms budget — PASS |
| Faithfulness / correctness | SKIPPED — no judge credential configured |

## The finding that mattered

This harness caught a real hole that our own bench did not.

Our `bench.py` reported **9/10 off-topic queries refused**, and that number is
true — but the queries were things like *"what's the weather in Goa tomorrow"*,
which sit far from the entire corpus and are trivial to reject. This suite's
negatives are far harder: real MSMARCO-XI rows where `is_selected` is all zero,
so the retrieved passages **are** topically relevant and simply do not answer
the question.

Against those, the first run scored a **false confidence rate of 1.000** — we
fabricated an answer on every single unanswerable query. Two causes:

1. **The adapter implemented half the guardrail.** The live harness gates on
   *both* a retrieval floor and an extractive floor; the adapter checked only
   the extractive score and ignored the context `.score` entirely, so nothing
   could ever be refused.
2. **The floors were tuned against easy negatives.** 0.62 / 0.34 were set using
   obviously off-topic probes and do not survive contact with topically-near
   negatives.

`scripts/calibrate_grounded.py` measures both signals against the dataset's own
labels. They overlap substantially:

| signal | answerable | unanswerable |
|---|---|---|
| top retrieval cosine | min 0.613, p50 0.805 | max 0.882, p50 0.733 |
| extractive confidence | min 0.554, p50 0.764 | max 0.869, p50 0.659 |

so there is no threshold that is free. The measured Pareto frontier:

| false confidence | false refusal | thresholds |
|---|---|---|
| 0.050 | 0.625 | 0.78 / 0.78 |
| 0.200 | 0.350 | **0.70 / 0.70 — chosen** |
| 0.475 | 0.125 | 0.70 / 0.60 |
| 0.800 | 0.000 | 0.60 / 0.52 |

We chose the cautious side because the suite's own reliability check calls false
confidence "a worse failure than false refusal": a false refusal loses an answer,
a false confidence hands the user a fabrication. On the held-out seed that point
delivers **0.350 false confidence / 0.300 false refusal** — worse than the
calibration set's 0.200/0.350, which is ordinary calibration optimism and the
reason the held-out number is the one reported above.

**What would actually fix this** is answerability judgment rather than
similarity: asking whether the context *contains* the answer, not whether it
looks related. That is a job for the generative path — the suite allows 1500 ms
for generation and we currently use ~343 ms, so the headroom exists. It is not
wired in because it needs an LLM credential the default $0 configuration does
not assume.
