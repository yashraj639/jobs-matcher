import sqlite3
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.paths import DB_PATH, EMB_PATH, IDS_PATH

MODEL_NAME = "all-MiniLM-L6-v2"

MAX_PROFILE_CHARS = 15_000  # ~3,000 words, well within LLM context

COMPANY_FIELDS = ("id", "name", "one_liner", "team_size", "batch", "is_hiring")

REBUILD_HELP = "Run `uv run python cli.py embed` to rebuild embeddings."


class RankError(RuntimeError):
    """The embedding cache is stale or does not match the companies table."""


def format_hiring(value: object) -> str:
    if value is None:
        return "unknown"
    return "1" if value == 1 else "0"


def load_profile(path: str | Path) -> str:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"profile file not found: {profile_path}")
    text = profile_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"profile file is empty: {profile_path}")
    if len(text) > MAX_PROFILE_CHARS:
        raise ValueError(
            f"profile is too long ({len(text)} chars, max {MAX_PROFILE_CHARS}). "
            "Trim it or use a shorter summary."
        )
    return text


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _load_embeddings() -> np.ndarray:
    embeddings: np.ndarray = np.load(EMB_PATH)
    return embeddings


@lru_cache(maxsize=1)
def _load_ids() -> np.ndarray:
    ids: np.ndarray = np.load(IDS_PATH)
    return ids


@lru_cache(maxsize=1)
def _load_company_index() -> dict[int, sqlite3.Row]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {', '.join(COMPANY_FIELDS)} FROM companies"
        ).fetchall()
    return {int(row["id"]): row for row in rows}


def embed_profile(text: str) -> np.ndarray:
    vectors: np.ndarray = _load_model().encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    vector: np.ndarray = vectors[0]
    return vector


def validate_cache(
    ids: np.ndarray,
    embeddings: np.ndarray,
    index: Mapping[int, object],
) -> None:
    """Fail before scoring if embeddings do not match the ids and companies."""
    if embeddings.shape[0] != len(ids):
        raise RankError(
            f"embedding cache is stale: {embeddings.shape[0]} embeddings "
            f"for {len(ids)} ids. {REBUILD_HELP}"
        )

    embedding_ids = {int(company_id) for company_id in ids}
    company_ids = set(index)
    missing = embedding_ids - company_ids
    extra = company_ids - embedding_ids
    if missing or extra:
        details = []
        if missing:
            details.append(
                f"{len(missing)} embedding ids are not in the companies table"
            )
        if extra:
            details.append(
                f"{len(extra)} companies have no embedding"
            )
        raise RankError(
            "embedding cache is stale: "
            + "; ".join(details)
            + f". {REBUILD_HELP}"
        )


def rank(
    profile_text: str, top_n: int = 50, hiring_only: bool = False
) -> list[dict]:
    embeddings = _load_embeddings()
    ids = _load_ids()
    index = _load_company_index()
    validate_cache(ids, embeddings, index)

    scores = embeddings @ embed_profile(profile_text)
    order = np.argsort(scores)[::-1]

    results = []
    for i in order:
        company_id = int(ids[i])
        row = index.get(company_id)
        if row is None:
            raise KeyError(f"embedding id {company_id} not found in companies table")
        if hiring_only and row["is_hiring"] != 1:
            continue
        results.append(
            {
                "id": company_id,
                "name": row["name"],
                "one_liner": row["one_liner"],
                "team_size": row["team_size"],
                "batch": row["batch"],
                "is_hiring": row["is_hiring"],
                "score": float(scores[i]),
            }
        )
        if len(results) >= top_n:
            break
    return results
