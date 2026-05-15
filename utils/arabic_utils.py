"""
utils/arabic_utils.py
─────────────────────
Egyptian Arabic text normalization and utility functions.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

# ── Unicode ranges ────────────────────────────────────────────────────────────
_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
_TATWEEL = "\u0640"
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# Sentence-ending punctuation (Arabic + Latin)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?؟،])\s+")

# Characters to keep: Arabic script, Latin alphanum, digits, common punctuation
_ALLOWED_RE = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\w\s.,!?\-،؟؛:()\[\]]")


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel (diacritical marks) from text."""
    return _DIACRITICS_RE.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove Arabic tatweel (kashida elongation character)."""
    return text.replace(_TATWEEL, "")


def normalize_alef(text: str) -> str:
    """Normalize all alef variants (أ إ آ ٱ) to bare alef (ا)."""
    return re.sub(r"[أإآٱ]", "ا", text)


def normalize_yeh(text: str) -> str:
    """Normalize final yeh variants to dotless yeh (ى)."""
    # In Egyptian colloquial text, ي and ى are often used interchangeably
    # at word endings — we keep both to preserve spelling.
    return text


def normalize_text(text: str, strip_diacritics_flag: bool = True) -> str:
    """
    Full normalization pipeline for Egyptian Arabic text before TTS synthesis.

    Steps:
      1. Unicode NFC normalization
      2. Remove tatweel
      3. Optionally strip diacritics (tashkeel)
      4. Collapse whitespace
      5. Remove unsupported characters

    Args:
        text: Raw input text.
        strip_diacritics_flag: Whether to remove tashkeel (default True).

    Returns:
        Normalized text string.
    """
    text = unicodedata.normalize("NFC", text)
    text = remove_tatweel(text)
    if strip_diacritics_flag:
        text = strip_diacritics(text)
    # Collapse runs of whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Drop characters that TTS engines typically can't handle
    text = _ALLOWED_RE.sub("", text)
    # Final strip in case substitutions left leading/trailing spaces
    return text.strip()


def segment_text(text: str, max_chars: int = 200) -> List[str]:
    """
    Split long text into segments of at most *max_chars* characters,
    breaking preferably at sentence boundaries.

    Args:
        text: Input text.
        max_chars: Maximum characters per segment.

    Returns:
        List of text segments.
    """
    if len(text) <= max_chars:
        return [text]

    raw_segments = _SENT_SPLIT_RE.split(text)
    result: List[str] = []
    current = ""

    for seg in raw_segments:
        seg = seg.strip()
        if not seg:
            continue
        if len(current) + len(seg) + 1 <= max_chars:
            current = f"{current} {seg}".strip() if current else seg
        else:
            if current:
                result.append(current)
            # If a single sentence exceeds the limit, hard-truncate it
            if len(seg) > max_chars:
                result.append(seg[:max_chars])
                current = ""
            else:
                current = seg

    if current:
        result.append(current)

    return result


def is_arabic_text(text: str, threshold: float = 0.30) -> bool:
    """
    Return True if at least *threshold* fraction of characters are Arabic.

    Args:
        text: Input text.
        threshold: Minimum Arabic character ratio (default 0.30).
    """
    if not text:
        return False
    arabic_count = sum(1 for ch in text if _ARABIC_CHAR_RE.match(ch))
    return arabic_count / len(text) >= threshold


def contains_latin(text: str) -> bool:
    """Return True if the text contains any Latin (A–Z / a–z) characters."""
    return bool(re.search(r"[A-Za-z]", text))


def has_digits(text: str) -> bool:
    """Return True if the text contains Arabic-Indic or ASCII digits."""
    return bool(re.search(r"[\d٠-٩]", text))


def estimate_token_count(text: str) -> int:
    """
    Rough token-count estimate for Arabic text (one token ≈ 4 chars).
    Useful for splitting before sending to TTS engines with length limits.
    """
    return max(1, len(text) // 4)
