from pathlib import Path

import numpy as np
import pytest

from pipeline import rank as rank_module
from pipeline.rank import (
    MAX_PROFILE_CHARS,
    RankError,
    load_profile,
    validate_cache,
)


def test_load_profile_success(tmp_path: Path) -> None:
    p = tmp_path / "profile.txt"
    p.write_text("  Software engineer with 5 years in backend systems.  ")
    result = load_profile(p)
    assert result == "Software engineer with 5 years in backend systems."


def test_load_profile_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "nonexistent.txt")


def test_load_profile_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("   \n  \t  ")
    with pytest.raises(ValueError, match="empty"):
        load_profile(p)


def test_load_profile_too_long(tmp_path: Path) -> None:
    p = tmp_path / "long.txt"
    p.write_text("x" * (MAX_PROFILE_CHARS + 1))
    with pytest.raises(ValueError, match="too long"):
        load_profile(p)


# ── missing data files ───────────────────────────────────────────────────────

def test_load_embeddings_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rank_module, "EMB_PATH", tmp_path / "missing.npy")
    rank_module._load_embeddings.cache_clear()

    with pytest.raises(RankError, match="cli.py embed"):
        rank_module._load_embeddings()


def test_load_ids_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rank_module, "IDS_PATH", tmp_path / "missing.npy")
    rank_module._load_ids.cache_clear()

    with pytest.raises(RankError, match="cli.py embed"):
        rank_module._load_ids()


def test_load_company_index_missing_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "missing.db"
    monkeypatch.setattr(rank_module, "DB_PATH", db_path)
    rank_module._load_company_index.cache_clear()

    with pytest.raises(RankError, match="cli.py clean"):
        rank_module._load_company_index()
    assert not db_path.exists()


# ── validate_cache ───────────────────────────────────────────────────────────

def _index(*ids: int) -> dict[int, object]:
    return {company_id: object() for company_id in ids}


def test_validate_cache_ok() -> None:
    ids = np.array([1, 2], dtype=np.int64)
    embeddings = np.zeros((2, 4))
    validate_cache(ids, embeddings, _index(1, 2))


def test_validate_cache_shape_mismatch() -> None:
    ids = np.array([1, 2], dtype=np.int64)
    embeddings = np.zeros((3, 4))
    with pytest.raises(RankError, match="cli.py embed"):
        validate_cache(ids, embeddings, _index(1, 2))


def test_validate_cache_missing_company() -> None:
    ids = np.array([1, 2], dtype=np.int64)
    embeddings = np.zeros((2, 4))
    with pytest.raises(RankError, match="not in the companies table"):
        validate_cache(ids, embeddings, _index(1))


def test_validate_cache_extra_company() -> None:
    ids = np.array([1], dtype=np.int64)
    embeddings = np.zeros((1, 4))
    with pytest.raises(RankError, match="companies have no embedding"):
        validate_cache(ids, embeddings, _index(1, 2))


# ── hiring filter ────────────────────────────────────────────────────────────

def _company_row(company_id: int, is_hiring: int | None) -> dict:
    return {
        "id": company_id,
        "name": f"Company {company_id}",
        "one_liner": "Widgets",
        "team_size": 10,
        "batch": "W23",
        "is_hiring": is_hiring,
    }


def test_rank_hiring_only_filters_before_slicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = np.array([1, 2, 3], dtype=np.int64)
    embeddings = np.array([[0.3, 0.0], [0.2, 0.0], [0.1, 0.0]])
    index = {
        1: _company_row(1, None),
        2: _company_row(2, 0),
        3: _company_row(3, 1),
    }
    monkeypatch.setattr(rank_module, "_load_embeddings", lambda: embeddings)
    monkeypatch.setattr(rank_module, "_load_ids", lambda: ids)
    monkeypatch.setattr(rank_module, "_load_company_index", lambda: index)
    monkeypatch.setattr(
        rank_module, "embed_profile", lambda text: np.array([1.0, 0.0])
    )

    results = rank_module.rank("profile", top_n=1, hiring_only=True)

    assert [r["id"] for r in results] == [3]
