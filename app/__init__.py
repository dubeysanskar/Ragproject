"""Adapter package for BeaconBandhu/rag-local-eval-loop (TARGET_INTERFACE.md).

The eval suite imports `app.embedder` and `app.generator` from the repo root
and drives them directly, using its own throwaway index — it never touches
ours. These modules are a thin shim onto the real pipeline in apps/api/goarag,
so the suite measures the same embedder and the same extractive answerer that
the live service runs.
"""
