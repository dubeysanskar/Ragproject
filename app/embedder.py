"""Embedder surface required by the eval suite.

Wraps the exact model the live service uses — local int8 ONNX
`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, L2-normalised. The suite
infers the dimension empirically from a real `embed_one` call, so nothing is
declared here.
"""
from __future__ import annotations

import numpy as np

from . import _bootstrap  # noqa: F401  (sys.path side effect)
from goarag.embedder import DIM, get_embedder  # noqa: E402


def get_model():
    """Called once by the suite; only the side effect (loading) matters."""
    return get_embedder()


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return get_embedder().encode(list(texts))


def embed_one(text: str) -> np.ndarray:
    return get_embedder().encode_one(text)
