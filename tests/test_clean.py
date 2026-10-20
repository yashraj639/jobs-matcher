import json
import sqlite3
from pathlib import Path

from pipeline.clean import (
    has_description,
    is_active,
    not_declined_hiring,
    save_sqlite,
    to_row,
)


# ── is_active ────────────────────────────────────────────────────────────────

def test_is_active_true() -> None:
    assert is_active({"status": "Active"}) is True


def test_is_active_false() -> None:
    assert is_active({"status": "Inactive"}) is False


def test_is_active_missing() -> None:
    assert is_active({}) is False


# ── has_description ──────────────────────────────────────────────────────────

def test_has_description_long() -> None:
    rec = {"long_description": "We build AI-powered widgets for enterprise.", "one_liner": ""}
    assert has_description(rec) is True


def test_has_description_one_liner_only() -> None:
    rec = {"long_description": "", "one_liner": "The best widget maker."}
    assert has_description(rec) is True


def test_has_description_empty() -> None:
    rec = {"long_description": "", "one_liner": ""}
    assert has_description(rec) is False


# ── not_declined_hiring ──────────────────────────────────────────────────────

def test_not_declined_hiring_true() -> None:
    assert not_declined_hiring({"isHiring": True}) is True


def test_not_declined_hiring_false() -> None:
    assert not_declined_hiring({"isHiring": False}) is False


def test_not_declined_hiring_missing() -> None:
    # no key → kept (returns True)
    assert not_declined_hiring({}) is True


# ── to_row ───────────────────────────────────────────────────────────────────

def test_to_row_serializes_tags() -> None:
    rec = {
        "id": 42,
        "name": "Acme",
        "slug": "acme",
        "one_liner": "We do stuff.",
        "long_description": "We do a lot of stuff.",
        "tags": ["AI", "B2B"],
        "industries": ["SaaS"],
        "team_size": 10,
        "batch": "W23",
        "website": "https://acme.com",
        "isHiring": True,
        "stage": "Series A",
        "regions": ["US"],
        "launched_at": 1700000000,
        "app_answers": None,
    }
    row = to_row(rec)
    assert json.loads(row["tags"]) == ["AI", "B2B"]
    assert json.loads(row["industries"]) == ["SaaS"]
    assert row["is_hiring"] == 1
    assert row["app_answers"] is None


def test_to_row_is_hiring_false() -> None:
    rec = {
        "id": 1,
        "name": "Foo",
        "isHiring": False,
        "tags": [],
        "industries": [],
        "regions": [],
    }
    row = to_row(rec)
    assert row["is_hiring"] == 0


def test_to_row_is_hiring_missing() -> None:
    rec = {
        "id": 1,
        "name": "Foo",
        "tags": [],
        "industries": [],
        "regions": [],
    }
    row = to_row(rec)
    assert row["is_hiring"] is None


# ── orphan founder cleanup ───────────────────────────────────────────────────

def test_save_sqlite_deletes_orphan_founders(tmp_path: Path) -> None:
    db = tmp_path / "yc.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE founders (company_id INTEGER, slug TEXT, user_id INTEGER)"
        )
        conn.execute(
            "INSERT INTO founders (company_id, slug, user_id) VALUES (1, 'keep', 10)"
        )
        conn.execute(
            "INSERT INTO founders (company_id, slug, user_id) VALUES (99, 'drop', 20)"
        )

    save_sqlite([to_row({"id": 1, "name": "Keep"})], db)

    with sqlite3.connect(db) as conn:
        remaining = conn.execute(
            "SELECT company_id FROM founders ORDER BY company_id"
        ).fetchall()
    assert remaining == [(1,)]


def test_save_sqlite_does_not_create_founders_table(tmp_path: Path) -> None:
    db = tmp_path / "yc.db"

    save_sqlite([to_row({"id": 1, "name": "Keep"})], db)

    with sqlite3.connect(db) as conn:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'founders'"
        ).fetchone()
    assert table is None
