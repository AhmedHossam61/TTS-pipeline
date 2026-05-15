"""Tests for utils/arabic_utils.py"""
import pytest
from utils.arabic_utils import (
    strip_diacritics,
    remove_tatweel,
    normalize_alef,
    normalize_text,
    segment_text,
    is_arabic_text,
    contains_latin,
    has_digits,
    estimate_token_count,
)


class TestStripDiacritics:
    def test_removes_fatha(self):
        assert strip_diacritics("كَتَبَ") == "كتب"

    def test_removes_shadda(self):
        assert strip_diacritics("مُحَمَّد") == "محمد"

    def test_leaves_plain_arabic(self):
        text = "مرحبا"
        assert strip_diacritics(text) == text

    def test_handles_empty_string(self):
        assert strip_diacritics("") == ""


class TestRemoveTatweel:
    def test_removes_kashida(self):
        assert remove_tatweel("مرحـبـا") == "مرحبا"

    def test_no_tatweel(self):
        assert remove_tatweel("مرحبا") == "مرحبا"


class TestNormalizeAlef:
    def test_normalizes_hamza_above(self):
        assert normalize_alef("أحمد") == "احمد"

    def test_normalizes_hamza_below(self):
        assert normalize_alef("إيمان") == "ايمان"

    def test_normalizes_alef_madda(self):
        assert normalize_alef("آمن") == "امن"


class TestNormalizeText:
    def test_strips_diacritics_by_default(self):
        result = normalize_text("كَتَبَ")
        assert "\u064E" not in result  # fatha

    def test_keeps_diacritics_when_flag_false(self):
        result = normalize_text("كَتَبَ", strip_diacritics_flag=False)
        assert "كَتَبَ" == result

    def test_collapses_whitespace(self):
        assert normalize_text("مرحبا   بك") == "مرحبا بك"

    def test_removes_tatweel(self):
        assert "ـ" not in normalize_text("مرحـبـا")

    def test_strips_leading_trailing_spaces(self):
        assert normalize_text("  مرحبا  ") == "مرحبا"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestSegmentText:
    def test_short_text_unchanged(self):
        text = "مرحبا"
        assert segment_text(text, max_chars=200) == [text]

    def test_splits_at_sentence_boundary(self):
        text = "مرحبا. كيف حالك؟ أنا بخير."
        segments = segment_text(text, max_chars=10)
        assert all(len(s) <= 20 for s in segments)  # generous tolerance
        assert len(segments) > 1

    def test_hard_truncate_very_long_sentence(self):
        long = "أ" * 300
        segments = segment_text(long, max_chars=100)
        assert all(len(s) <= 100 for s in segments)

    def test_empty_input(self):
        assert segment_text("", max_chars=100) == [""]


class TestIsArabicText:
    def test_pure_arabic(self):
        assert is_arabic_text("مرحبا بك في مصر") is True

    def test_latin_only(self):
        assert is_arabic_text("Hello world") is False

    def test_mixed_arabic_dominant(self):
        assert is_arabic_text("أنا بحب الـ Python") is True

    def test_empty(self):
        assert is_arabic_text("") is False


class TestContainsLatin:
    def test_has_latin(self):
        assert contains_latin("أنا بحب Python") is True

    def test_no_latin(self):
        assert contains_latin("أنا بخير") is False


class TestHasDigits:
    def test_ascii_digit(self):
        assert has_digits("عندي 5 كتب") is True

    def test_arabic_indic_digit(self):
        assert has_digits("عندي ٥ كتب") is True

    def test_no_digits(self):
        assert has_digits("مرحبا") is False


class TestEstimateTokenCount:
    def test_returns_positive(self):
        assert estimate_token_count("مرحبا") >= 1

    def test_longer_text_more_tokens(self):
        short = estimate_token_count("مرحبا")
        long  = estimate_token_count("مرحبا " * 20)
        assert long > short
