import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline import founders


def _page(company_id: int, active_founders: list[dict]) -> str:
    payload = {
        "props": {
            "company": {"id": company_id, "founders": active_founders}
        }
    }
    encoded = html.escape(json.dumps(payload), quote=True)
    return f'<div data-page="{encoded}"></div>'


def _fresh() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stale() -> str:
    age = timedelta(days=founders.CACHE_TTL_DAYS + 1)
    return (datetime.now(timezone.utc) - age).isoformat()


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "yc.db"
    monkeypatch.setattr(founders, "DB_PATH", path)
    return path


def _write_lookup(slug: str, fetched_at: str) -> None:
    with founders._connect() as conn:
        conn.execute(
            "INSERT INTO founder_lookups (slug, fetched_at) VALUES (?, ?)",
            (slug, fetched_at),
        )


def _fail_fetch(slug: str) -> str:
    raise AssertionError(f"unexpected fetch for {slug}")


# ── empty cache ──────────────────────────────────────────────────────────────

def test_fresh_empty_cache_is_reused(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_lookup("acme", _fresh())
    monkeypatch.setattr(founders, "_fetch_page", _fail_fetch)

    assert founders.get_founders("acme") == []


def test_stale_empty_cache_refetches(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_lookup("acme", _stale())
    calls: list[str] = []

    def fake_fetch(slug: str) -> str:
        calls.append(slug)
        return _page(1, [])

    monkeypatch.setattr(founders, "_fetch_page", fake_fetch)

    assert founders.get_founders("acme") == []
    assert calls == ["acme"]

    monkeypatch.setattr(founders, "_fetch_page", _fail_fetch)
    assert founders.get_founders("acme") == []


def test_missing_cache_refetches(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_fetch(slug: str) -> str:
        calls.append(slug)
        return _page(1, [])

    monkeypatch.setattr(founders, "_fetch_page", fake_fetch)

    assert founders.get_founders("acme") == []
    assert calls == ["acme"]


def test_empty_cache_reports_cached_source(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_lookup("acme", _fresh())
    monkeypatch.setattr(founders, "_fetch_page", _fail_fetch)

    result, source = founders.get_founders_with_source("acme")

    assert result == []
    assert source == "cached"


# ── missing user_id ──────────────────────────────────────────────────────────

def test_extract_raises_when_active_founder_has_no_user_id() -> None:
    page = _page(1, [{"full_name": "Ada Lovelace", "is_active": True}])

    with pytest.raises(founders.FoundersError, match="acme.*Ada Lovelace"):
        founders._extract(page, "acme")
