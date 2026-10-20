from pathlib import Path

import numpy as np
import pytest

from pipeline import embed


def _use_tmp_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    embeddings_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "ids.npy"
    hash_path = tmp_path / "embedding_hash.txt"
    monkeypatch.setattr(embed, "EMB_PATH", embeddings_path)
    monkeypatch.setattr(embed, "IDS_PATH", ids_path)
    monkeypatch.setattr(embed, "EMBEDDING_HASH_PATH", hash_path)
    return embeddings_path, ids_path, hash_path


def _write_cache(
    embeddings_path: Path,
    ids_path: Path,
    hash_path: Path,
    ids: np.ndarray,
    texts: list[str],
    *,
    write_hash: bool = True,
) -> None:
    np.save(embeddings_path, np.zeros((len(ids), 4)))
    np.save(ids_path, ids)
    if write_hash:
        hash_path.write_text(embed.text_hash(texts), encoding="utf-8")


def test_is_cached_same_ids_and_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _use_tmp_paths(tmp_path, monkeypatch)
    ids = np.array([1, 2], dtype=np.int64)
    texts = ["Acme builds widgets", "Beta builds tools"]
    _write_cache(*paths, ids, texts)

    assert embed.is_cached(ids, texts) is True


def test_is_cached_changed_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _use_tmp_paths(tmp_path, monkeypatch)
    ids = np.array([1, 2], dtype=np.int64)
    _write_cache(*paths, ids, ["Acme builds widgets", "Beta builds tools"])

    assert embed.is_cached(ids, ["Acme builds gadgets", "Beta builds tools"]) is False


def test_is_cached_missing_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _use_tmp_paths(tmp_path, monkeypatch)
    ids = np.array([1, 2], dtype=np.int64)
    texts = ["Acme builds widgets", "Beta builds tools"]
    _write_cache(*paths, ids, texts, write_hash=False)

    assert embed.is_cached(ids, texts) is False
