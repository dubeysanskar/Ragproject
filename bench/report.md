# GoaRAG bench report

API: `http://127.0.0.1:8099` · queries: **132** (120 eval + 10 out-of-scope + 2 unsafe)

## Latency — spec'd budget (query text → final output)

- **All queries:** P50    68.4 · P70    80.3 · P90    94.8 · P95   105.1 · P100   144.9 ms  (n=132)
- en: P50    51.3 · P70    55.2 · P90    68.4 · P95    80.3 · P100   111.7 ms  (n=30)
- hi: P50    79.4 · P70    87.8 · P90   108.2 · P95   135.9 · P100   144.9 ms  (n=30)
- ta: P50    74.6 · P70    84.5 · P90    93.8 · P95   102.8 · P100   112.2 ms  (n=60)
- Wall clock (incl. HTTP): P50    71.3 · P70    83.1 · P90    99.0 · P95   115.6 · P100   523.2 ms  (n=132)

## Quality

- Answered (eval set): **114/120**
- Gold passage in citations/top-k: **114/114** (100%)

## Guardrails — refusal precision

- Out-of-scope correctly refused: **3/10**
- Unsafe correctly refused: **2/2**

- Mean budget: 68.9 ms · target: **< 200 ms at P100** → **PASS**