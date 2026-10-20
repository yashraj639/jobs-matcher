import pytest

from pipeline import rerank as rerank_module
from pipeline.llm import LLMParseError
from pipeline.rerank import _build_payload, _parse_picks


# ── payload ───────────────────────────────────────────────────────────────────

def test_build_payload_preserves_unknown_hiring() -> None:
    payload = _build_payload(
        [{"id": 1, "name": "Acme", "is_hiring": None}], {}
    )
    assert payload[0]["is_hiring"] is None


# ── valid input ───────────────────────────────────────────────────────────────

def test_parse_picks_valid() -> None:
    raw = {"picks": [{"id": 1, "reason": "good fit"}]}
    picks, rejected = _parse_picks(raw, valid_ids={1})
    assert picks == [{"id": 1, "reason": "good fit"}]
    assert rejected == 0


# ── filtering ─────────────────────────────────────────────────────────────────

def test_parse_picks_filters_unknown_id() -> None:
    raw = {"picks": [{"id": 99, "reason": "good fit"}]}
    with pytest.raises(LLMParseError):
        # no valid picks remain → raises
        _parse_picks(raw, valid_ids={1})


def test_parse_picks_filters_unknown_id_count() -> None:
    """One valid, one unknown — unknown is rejected, count is 1."""
    raw = {"picks": [
        {"id": 1, "reason": "strong match"},
        {"id": 99, "reason": "unknown company"},
    ]}
    picks, rejected = _parse_picks(raw, valid_ids={1})
    assert len(picks) == 1
    assert picks[0]["id"] == 1
    assert rejected == 1


def test_parse_picks_filters_duplicate() -> None:
    raw = {"picks": [
        {"id": 1, "reason": "first occurrence"},
        {"id": 1, "reason": "duplicate"},
    ]}
    picks, rejected = _parse_picks(raw, valid_ids={1})
    assert len(picks) == 1
    assert rejected == 1


def test_parse_picks_filters_missing_reason() -> None:
    raw = {"picks": [
        {"id": 1, "reason": "   "},   # blank after strip
        {"id": 2, "reason": "valid reason"},
    ]}
    picks, rejected = _parse_picks(raw, valid_ids={1, 2})
    assert len(picks) == 1
    assert picks[0]["id"] == 2
    assert rejected == 1


# ── error cases ───────────────────────────────────────────────────────────────

def test_parse_picks_empty_raises() -> None:
    with pytest.raises(LLMParseError):
        _parse_picks({"picks": []}, valid_ids={1})


def test_parse_picks_no_picks_key() -> None:
    with pytest.raises(LLMParseError):
        _parse_picks({"foo": []}, valid_ids={1})


# ── retry / fallback ─────────────────────────────────────────────────────────

def test_rerank_raises_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_generate_json(prompt: str) -> dict:
        calls.append(prompt)
        raise LLMParseError("not JSON")

    monkeypatch.setattr(rerank_module, "generate_json", fake_generate_json)
    monkeypatch.setattr(rerank_module, "_load_descriptions", lambda ids: {})

    with pytest.raises(LLMParseError, match="after retry"):
        rerank_module.rerank_with_stats("profile", [{"id": 1}])
    assert len(calls) == 2


def test_rerank_raises_when_no_usable_picks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rerank_module, "generate_json", lambda prompt: {"picks": []})
    monkeypatch.setattr(
        rerank_module, "_parse_picks", lambda raw, valid_ids: ([], 0)
    )
    monkeypatch.setattr(rerank_module, "_load_descriptions", lambda ids: {})

    with pytest.raises(LLMParseError, match="no usable picks"):
        rerank_module.rerank_with_stats("profile", [{"id": 1}])
