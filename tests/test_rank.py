from pathlib import Path

import numpy as np
import pytest

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
