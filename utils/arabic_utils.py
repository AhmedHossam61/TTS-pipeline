"""
utils/arabic_utils.py
─────────────────────
Egyptian Arabic text normalization and utility functions.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import List

log = logging.getLogger(__name__)

# ── Unicode ranges ────────────────────────────────────────────────────────────
_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
_TATWEEL = "\u0640"
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# Sentence-ending punctuation (Arabic + Latin)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?؟،])\s+")

# Characters to keep: Arabic script, digits, common punctuation (Latin stripped later)
_ALLOWED_RE = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\w\s.,!?\-،؟؛:()\[\]]")

# ── Numeral normalization ─────────────────────────────────────────────────────
# Map Arabic-Indic digits to ASCII digits
_ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_ONES = {
    0: "صفر", 1: "واحد", 2: "اتنين", 3: "تلاتة", 4: "أربعة",
    5: "خمسة", 6: "ستة", 7: "سبعة", 8: "تمانية", 9: "تسعة",
    10: "عشرة", 11: "حداشر", 12: "اتناشر", 13: "تلتاشر",
    14: "أربعتاشر", 15: "خمستاشر", 16: "ستاشر", 17: "سبعتاشر",
    18: "تمنتاشر", 19: "تسعتاشر",
}
_TENS = {
    20: "عشرين", 30: "تلاتين", 40: "أربعين", 50: "خمسين",
    60: "ستين", 70: "سبعين", 80: "تمانين", 90: "تسعين",
}
_HUNDREDS = {
    100: "مية", 200: "ميتين", 300: "تلتمية", 400: "أربعمية",
    500: "خمسمية", 600: "ستمية", 700: "سبعمية", 800: "تمنمية", 900: "تسعمية",
}


def _int_to_arabic_words(n: int) -> str:
    """Convert a non-negative integer to Egyptian Arabic spoken words."""
    if n < 0:
        return "سالب " + _int_to_arabic_words(-n)
    if n in _ONES:
        return _ONES[n]
    if 21 <= n <= 99:
        tens, ones = (n // 10) * 10, n % 10
        return f"{_ONES[ones]} و{_TENS[tens]}" if ones else _TENS[tens]
    if 100 <= n <= 999:
        h = (n // 100) * 100
        rem = n % 100
        return (_HUNDREDS[h] + (" و" + _int_to_arabic_words(rem) if rem else ""))
    if 1000 <= n <= 9999:
        thousands = n // 1000
        rem = n % 1000
        if thousands == 1:
            prefix = "ألف"
        elif thousands == 2:
            prefix = "ألفين"
        elif 3 <= thousands <= 10:
            prefix = f"{_ONES[thousands]} آلاف"
        else:
            prefix = f"{_int_to_arabic_words(thousands)} ألف"
        return prefix + (" و" + _int_to_arabic_words(rem) if rem else "")
    # Very large numbers: fall back to digit-by-digit spelling
    return " ".join(_ONES[int(d)] for d in str(n))


def normalize_numerals(text: str) -> str:
    """
    Normalize numerals in Arabic text for consistent TTS rendering.

    Steps:
      1. Convert Arabic-Indic digits (٣, ٢٠٢٤) to Western digits.
      2. Expand standalone digit sequences to Egyptian Arabic words.
         e.g. "٣ كيلو" → "تلاتة كيلو", "2024" → "ألفين وأربعة وعشرين"

    Mixed alphanumeric tokens (e.g. "A4", "COVID19") are left as-is to avoid
    corrupting product codes or identifiers.
    """
    # Step 1: Arabic-Indic → ASCII digits
    text = text.translate(_ARABIC_INDIC_MAP)

    # Step 2: Replace standalone digit runs with Arabic words.
    # A digit run is "standalone" when it is not glued to Latin letters.
    def _replace_num(m: re.Match) -> str:
        full = m.group(0)
        # If the match is surrounded by Latin letters, leave it alone
        start, end = m.start(), m.end()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if re.match(r"[A-Za-z]", before) or re.match(r"[A-Za-z]", after):
            return full
        try:
            return _int_to_arabic_words(int(full))
        except (ValueError, OverflowError):
            return full

    return re.sub(r"\d+", _replace_num, text)


# ── Latin character handling ──────────────────────────────────────────────────

def strip_latin(text: str) -> str:
    """
    Remove stray Latin characters from Egyptian Arabic text.

    Strategy:
      - Whole tokens that consist entirely of Latin characters (e.g. "lol",
        "OK", "TTS") are dropped completely.
      - Latin characters embedded inside mixed Arabic-Latin tokens are stripped,
        leaving the Arabic portion.

    Purely Latin tokens that carry no Arabic content (common code-switching in
    informal Egyptian writing) add no phonetic value to TTS and can confuse
    voice models trained only on Arabic script.
    """
    tokens = text.split()
    cleaned: List[str] = []
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z]+", tok):
            # Purely alphabetic Latin token (e.g. "lol", "OK", "TTS") —
            # these are informal code-switched words with no Arabic rendering.
            # Drop them.
            continue
        if re.fullmatch(r"[A-Za-z0-9_\-\.]+", tok) and re.search(r"\d", tok):
            # Alphanumeric identifier (e.g. "A4", "MP3", "USB3", "COVID19") —
            # carries semantic meaning; keep as-is and let the TTS engine handle it.
            cleaned.append(tok)
            continue
        # Strip any remaining Latin letters from mixed Arabic-Latin tokens
        tok = re.sub(r"[A-Za-z]", "", tok).strip()
        if tok:
            cleaned.append(tok)
    return " ".join(cleaned)


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


def normalize_text(
    text: str,
    strip_diacritics_flag: bool = True,
    normalize_numerals_flag: bool = True,
    strip_latin_flag: bool = True,
    max_chars: int | None = 200,
) -> str:
    """
    Full normalization pipeline for Egyptian Arabic text before TTS synthesis.

    Steps:
      1. Unicode NFC normalization
      2. Remove tatweel
      3. Optionally strip diacritics (tashkeel)
      4. Normalize numerals — Arabic-Indic → Western, then expand to words
      5. Strip stray Latin tokens / characters
      6. Collapse whitespace
      7. Remove remaining unsupported characters
      8. Hard-cap at *max_chars* characters to prevent XTTS/engine failures

    Args:
        text: Raw input text.
        strip_diacritics_flag: Whether to remove tashkeel (default True).
        normalize_numerals_flag: Whether to expand digits to Arabic words (default True).
        strip_latin_flag: Whether to remove Latin character tokens (default True).
        max_chars: Maximum allowed character length; None disables the cap.

    Returns:
        Normalized text string.
    """
    text = unicodedata.normalize("NFC", text)
    text = remove_tatweel(text)
    if strip_diacritics_flag:
        text = strip_diacritics(text)
    if normalize_numerals_flag:
        text = normalize_numerals(text)
    if strip_latin_flag:
        text = strip_latin(text)
    # Collapse runs of whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Drop characters that TTS engines typically can't handle
    text = _ALLOWED_RE.sub("", text)
    text = text.strip()
    # Hard cap to prevent XTTS / edge-tts failures on very long prompts
    if max_chars is not None and len(text) > max_chars:
        log.warning(
            "Prompt truncated from %d to %d chars: %r…", len(text), max_chars, text[:40]
        )
        # Truncate at the last space within the limit to avoid mid-word cuts
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        text = truncated[:last_space] if last_space > max_chars // 2 else truncated
    return text


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
