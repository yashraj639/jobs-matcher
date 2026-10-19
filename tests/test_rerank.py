import pytest

from pipeline.llm import LLMParseError
from pipeline.rerank import _parse_picks


# ── valid input ───────────────────────────────────────────────────────────────

def test_parse_picks_valid():
    raw = {"picks": [{"id": 1, "reason": "good fit"}]}
    picks, rejected = _parse_picks(raw, valid_ids={1})
    assert picks == [{"id": 1, "reason": "good fit"}]
    assert rejected == 0


# ── filtering ─────────────────────────────────────────────────────────────────

def test_parse_picks_filters_unknown_id():
    raw = {"picks": [{"id": 99, "reason": "good fit"}]}
    with pytest.raises(LLMParseError):
        # no valid picks remain → raises
        _parse_picks(raw, valid_ids={1})


def test_parse_picks_filters_unknown_id_count():
    """One valid, one unknown — unknown is rejected, count is 1."""
    raw = {"picks": [
        {"id": 1, "reason": "strong match"},
        {"id": 99, "reason": "unknown company"},
    ]}
    picks, rejected = _parse_picks(raw, valid_ids={1})
    assert len(picks) == 1
    assert picks[0]["id"] == 1
    assert rejected == 1


def test_parse_picks_filters_duplicate():
    raw = {"picks": [
        {"id": 1, "reason": "first occurrence"},
        {"id": 1, "reason": "duplicate"},
    ]}
    picks, rejected = _parse_picks(raw, valid_ids={1})
    assert len(picks) == 1
    assert rejected == 1


def test_parse_picks_filters_missing_reason():
    raw = {"picks": [
        {"id": 1, "reason": "   "},   # blank after strip
        {"id": 2, "reason": "valid reason"},
    ]}
    picks, rejected = _parse_picks(raw, valid_ids={1, 2})
    assert len(picks) == 1
    assert picks[0]["id"] == 2
    assert rejected == 1


# ── error cases ───────────────────────────────────────────────────────────────

def test_parse_picks_empty_raises():
    with pytest.raises(LLMParseError):
        _parse_picks({"picks": []}, valid_ids={1})


def test_parse_picks_no_picks_key():
    with pytest.raises(LLMParseError):
        _parse_picks({"foo": []}, valid_ids={1})
