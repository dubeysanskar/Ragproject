# GoaRAG API — one process, index resident in RAM.
#
# NOT serverless, deliberately (PDR §9): the index lives in memory and takes
# minutes to build, so a cold start would blow the latency budget on the first
# query a judge runs. This image is meant for an always-on container.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    # fastembed caches the ONNX model here; baked into the image below so the
    # container never downloads it at boot.
    HF_HOME=/models

WORKDIR /app

# Dependencies first — this layer only rebuilds when requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Without this the first boot pulls
# ~220 MB from HuggingFace, which turns a redeploy into a multi-minute outage.
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY goarag/ ./goarag/
COPY main.py .

# Corpus is mounted or copied in; falls back to the built-in demo set if absent.
COPY data/ ./data/

ENV GOARAG_CORPUS=/app/data/corpus.json
EXPOSE 8000

# Single worker on purpose: each worker would hold its own copy of the index
# and model (~1.3 GB), and the hot path is CPU-bound anyway.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
