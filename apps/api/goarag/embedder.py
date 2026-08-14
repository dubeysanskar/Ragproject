"""Local ONNX embeddings — the reason the hot path has no network in it.

Measured on this box (4-core laptop CPU, fastembed 0.8.0, ORT 1.28):
query embed p50 ≈ 6.7 ms across en/hi/ta, model load ≈ 4.6 s at boot.

Model note: the PDR specifies `intfloat/multilingual-e5-small`, which current
fastembed no longer carries. `paraphrase-multilingual-MiniLM-L12-v2` is the
substitute — same 384 dims the PDR's RAM budget assumes, 50+ languages
including Indic, and it needs no "query:"/"passage:" prefixes. Measured
cross-lingual cosine: 0.995 for en/hi paraphrases, ~0.00 for unrelated pairs.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIM = 384


class Embedder:
    """Thin wrapper that also exposes the model's own tokenizer, so C2 can cut
    token windows with the exact vocabulary the encoder will see."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        self._tok = self._model.model.tokenizer
        # fastembed's ONNX session is not documented as thread-safe; the API
        # serves concurrent requests, so guard it rather than find out.
        self._lock = threading.Lock()
        self.warm()

    def warm(self) -> None:
        """Force the first (slow) inference at boot, never inside a request."""
        with self._lock:
            list(self._model.embed(["warm up"]))

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        with self._lock:
            vecs = list(self._model.embed(texts))
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # -- tokenizer surface used by the C2 sliding-window chunker --------------
    def tokenize(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def detokenize(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Process-wide singleton: the model is ~220 MB and must be loaded once."""
    return Embedder()
