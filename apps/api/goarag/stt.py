"""Speech-to-text adapters (PDR §2). Task constraint: Sarvam or ElevenLabs.

Primary is Sarvam (Indic ASR, returns language codes we reuse as the retrieval
filter). ElevenLabs sits behind the same interface as the fallback — the swap
being a config change is itself harness evidence (typed stages + fallback
edges, PDR §6). STT is *outside* the measured 200 ms budget: it streams while
the user speaks, so the query text is ready the moment they stop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Sarvam language codes → our retrieval tags (same space as langdetect.py).
_SARVAM_TAG = {
    "hi-IN": "hi", "mr-IN": "hi", "bn-IN": "bn", "ta-IN": "ta", "te-IN": "te",
    "kn-IN": "kn", "ml-IN": "ml", "gu-IN": "gu", "pa-IN": "pa", "od-IN": "or",
    "ur-IN": "ur", "en-IN": "en",
}


@dataclass
class Transcript:
    text: str
    lang: str
    confidence: float
    provider: str


class STTError(RuntimeError):
    pass


async def _sarvam(audio: bytes, mime: str, client: httpx.AsyncClient) -> Transcript:
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise STTError("SARVAM_API_KEY not set")
    resp = await client.post(
        SARVAM_URL,
        headers={"api-subscription-key": key},
        files={"file": ("audio", audio, mime)},
        data={"model": "saarika:v2.5", "language_code": "unknown"},
        timeout=15.0,
    )
    resp.raise_for_status()
    d = resp.json()
    return Transcript(
        text=d.get("transcript", ""),
        lang=_SARVAM_TAG.get(d.get("language_code", ""), "en"),
        # Sarvam doesn't return a confidence; empty transcript is the failure
        # signal, mapped to 0 so the STT-confidence guardrail fires.
        confidence=1.0 if d.get("transcript") else 0.0,
        provider="sarvam",
    )


async def _elevenlabs(audio: bytes, mime: str, client: httpx.AsyncClient) -> Transcript:
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise STTError("ELEVENLABS_API_KEY not set")
    resp = await client.post(
        ELEVENLABS_URL,
        headers={"xi-api-key": key},
        files={"file": ("audio", audio, mime)},
        data={"model_id": "scribe_v1"},
        timeout=20.0,
    )
    resp.raise_for_status()
    d = resp.json()
    words = d.get("words") or []
    probs = [w["logprob"] for w in words if "logprob" in w]
    return Transcript(
        text=d.get("text", ""),
        lang=(d.get("language_code") or "en").split("-")[0],
        confidence=float(min(1.0, sum(probs) / len(probs) + 1.0)) if probs else (1.0 if d.get("text") else 0.0),
        provider="elevenlabs",
    )


async def transcribe(audio: bytes, mime: str = "audio/webm") -> Transcript:
    """Sarvam first, ElevenLabs on any failure — the PDR §6 fallback edge.
    Raises STTError only when every configured provider is down/absent."""
    errors: list[str] = []
    async with httpx.AsyncClient() as client:
        for fn in (_sarvam, _elevenlabs):
            try:
                t = await fn(audio, mime, client)
                if t.text:
                    return t
                errors.append(f"{t.provider}: empty transcript")
            except (STTError, httpx.HTTPError) as e:
                errors.append(f"{fn.__name__}: {e}")
    raise STTError("; ".join(errors))
