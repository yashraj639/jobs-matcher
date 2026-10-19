import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.paths import DB_PATH, EMB_PATH, IDS_PATH

MODEL_NAME = "all-MiniLM-L6-v2"

MAX_PROFILE_CHARS = 15_000  # ~3,000 words, well within LLM context

COMPANY_FIELDS = ("id", "name", "one_liner", "team_size", "batch", "is_hiring")


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
    return np.load(EMB_PATH)


@lru_cache(maxsize=1)
def _load_ids() -> np.ndarray:
    return np.load(IDS_PATH)


@lru_cache(maxsize=1)
def _load_company_index() -> dict[int, sqlite3.Row]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT {', '.join(COMPANY_FIELDS)} FROM companies"
        ).fetchall()
    return {int(row["id"]): row for row in rows}


def embed_profile(text: str) -> np.ndarray:
    return _load_model().encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


def rank(profile_text: str, top_n: int = 50) -> list[dict]:
    scores = _load_embeddings() @ embed_profile(profile_text)
    ids = _load_ids()
    order = np.argsort(scores)[::-1][:top_n]

    index = _load_company_index()
    results = []
    for i in order:
        company_id = int(ids[i])
        row = index.get(company_id)
        if row is None:
            raise KeyError(f"embedding id {company_id} not found in companies table")
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
    return results
