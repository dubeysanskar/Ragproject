# Chunking ablation

Corpus: **1200 passages / 7283 chunks** (MSMARCO-XI, en+hi+ta) · **120 eval queries** with `is_selected` gold passages.

One index build, every strategy present; each arm restricts retrieval to a subset. Identical vectors and corpus across arms, so the only variable is which chunkers compete.

| Arm | Recall@5 | MRR@10 | retrieve p50 |
|---|---|---|---|
| C2 sliding window only (naive baseline) | 0.533 | 0.427 | 117.1 ms |
| C1 semantic only | 0.508 | 0.416 | 118.5 ms |
| C3 parent-child only | 0.558 | 0.462 | 115.8 ms |
| C5 question-indexed only | 0.000 | 0.000 | 117.4 ms |
| C1+C2+C3 (no question keys) | 0.567 | 0.453 | 118.1 ms |
| All strategies, dense only | 0.508 | 0.414 | 21.2 ms |
| All strategies + BM25 hybrid (shipped) | **0.758** | 0.554 | 25.8 ms |

**+42% Recall@5 over a single fixed-size splitter** — the naive approach the task warns against.

## Reading these numbers honestly

**BM25 hybrid is where the win comes from, not the chunkers.** All strategies dense-only scores 0.508 — no better than C1 alone. Adding the lexical arm takes it to 0.758. Dense similarity is soft on the rare tokens (names, numbers, transliterations) that decide these queries, and RRF lets an exact lexical match outvote a fuzzy semantic one. The chunking matrix supplies candidates; fusion is what ranks them correctly.

**C5 scores 0.000 by construction, and that is the leakage fix working.** Held-out eval passages carry `questions: []`, so they have no question-key chunks at all — a C5-only arm cannot return them even in principle. The arm is kept because its zero is the clearest evidence that eval queries are no longer indexed against themselves. It also means this eval set cannot measure C5's true contribution: that needs held-out queries which are *paraphrases* of indexed ones, which the dataset does not provide.

**Single-strategy arms look ~5x slower (117 ms vs 21 ms) — an artifact, not a finding.** Restricting by strategy adds a Qdrant payload filter with no payload index behind it, forcing a scan. The shipped path filters nothing and runs at the 21–26 ms shown in the last two rows.