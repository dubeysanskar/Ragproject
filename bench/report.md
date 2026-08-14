# GoaRAG bench report

API: `http://127.0.0.1:8099` · queries: **132** (120 eval + 10 out-of-scope + 2 unsafe)

## Latency — spec'd budget (query text → final output)

- **All queries:** P50    36.6 · P70    39.4 · P90    46.9 · P95    48.7 · P100    74.0 ms  (n=132)
- en: P50    32.2 · P70    34.2 · P90    39.4 · P95    41.2 · P100    45.9 ms  (n=30)
- hi: P50    35.0 · P70    38.3 · P90    47.6 · P95    51.1 · P100    56.0 ms  (n=30)
- ta: P50    39.0 · P70    41.5 · P90    47.4 · P95    56.3 · P100    74.0 ms  (n=60)
- Wall clock (incl. HTTP): P50    38.8 · P70    41.9 · P90    49.6 · P95    58.3 · P100   373.0 ms  (n=132)

## Quality

- Answered (eval set): **95/120**
- Gold passage in citations/top-k: **70/95** (74%)

## Guardrails — refusal precision

- Out-of-scope correctly refused: **9/10**
- Unsafe correctly refused: **2/2**

- Mean budget: 37.1 ms · target: **< 200 ms at P100** → **PASS**