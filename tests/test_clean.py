import json

from pipeline.clean import has_description, is_active, not_declined_hiring, to_row


# ── is_active ────────────────────────────────────────────────────────────────

def test_is_active_true():
    assert is_active({"status": "Active"}) is True


def test_is_active_false():
    assert is_active({"status": "Inactive"}) is False


def test_is_active_missing():
    assert is_active({}) is False


# ── has_description ──────────────────────────────────────────────────────────

def test_has_description_long():
    rec = {"long_description": "We build AI-powered widgets for enterprise.", "one_liner": ""}
    assert has_description(rec) is True


def test_has_description_one_liner_only():
    rec = {"long_description": "", "one_liner": "The best widget maker."}
    assert has_description(rec) is True


def test_has_description_empty():
    rec = {"long_description": "", "one_liner": ""}
    assert has_description(rec) is False


# ── not_declined_hiring ──────────────────────────────────────────────────────

def test_not_declined_hiring_true():
    assert not_declined_hiring({"isHiring": True}) is True


def test_not_declined_hiring_false():
    assert not_declined_hiring({"isHiring": False}) is False


def test_not_declined_hiring_missing():
    # no key → kept (returns True)
    assert not_declined_hiring({}) is True


# ── to_row ───────────────────────────────────────────────────────────────────

def test_to_row_serializes_tags():
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


def test_to_row_is_hiring_false():
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
