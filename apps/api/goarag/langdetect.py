"""Script-based language detection — microseconds, no model, no network.

The corpus is MSMARCO-XI, whose languages map almost 1:1 onto distinct Unicode
blocks, so counting script ranges identifies the language without the ~10 ms a
statistical detector would spend. Hindi/Marathi share Devanagari and are not
separable this way; both tag as `hi`, which is correct for retrieval filtering
because they share the script *and* most of the vocabulary that matters here.
"""

from __future__ import annotations

# (language tag, inclusive codepoint range)
_BLOCKS: list[tuple[str, int, int]] = [
    ("hi", 0x0900, 0x097F),  # Devanagari — Hindi, Marathi
    ("bn", 0x0980, 0x09FF),  # Bengali, Assamese
    ("pa", 0x0A00, 0x0A7F),  # Gurmukhi
    ("gu", 0x0A80, 0x0AFF),  # Gujarati
    ("or", 0x0B00, 0x0B7F),  # Odia
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("te", 0x0C00, 0x0C7F),  # Telugu
    ("kn", 0x0C80, 0x0CFF),  # Kannada
    ("ml", 0x0D00, 0x0D7F),  # Malayalam
    ("ur", 0x0600, 0x06FF),  # Arabic script — Urdu
]


def detect_lang(text: str, default: str = "en") -> str:
    """Returns the language whose script dominates the letters in `text`."""
    counts: dict[str, int] = {}
    latin = 0
    for ch in text:
        cp = ord(ch)
        if not ch.isalpha():
            continue
        if cp < 0x0250:
            latin += 1
            continue
        for tag, lo, hi in _BLOCKS:
            if lo <= cp <= hi:
                counts[tag] = counts.get(tag, 0) + 1
                break

    if not counts:
        return default
    tag, n = max(counts.items(), key=lambda kv: kv[1])
    # A stray Indic character in an otherwise Latin query shouldn't flip the tag.
    return tag if n >= latin else default
