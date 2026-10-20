import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

from pipeline.paths import DB_PATH, EMB_PATH, EMBEDDING_HASH_PATH, IDS_PATH

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
HASH_SEPARATOR = "\x00"


def load_companies(db_path: Path = DB_PATH) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT id, name, one_liner, long_description, tags "
            "FROM companies ORDER BY id"
        ).fetchall()


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(t) for t in parsed]


def company_text(row: sqlite3.Row) -> str:
    parts = [
        row["name"] or "",
        row["one_liner"] or "",
        row["long_description"] or "",
        ", ".join(parse_tags(row["tags"])),
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())


def text_hash(texts: list[str]) -> str:
    """Hash company texts so changed text invalidates the embedding cache."""
    joined = HASH_SEPARATOR.join(texts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def is_cached(ids: np.ndarray, texts: list[str]) -> bool:
    if not (
        EMB_PATH.exists()
        and IDS_PATH.exists()
        and EMBEDDING_HASH_PATH.exists()
    ):
        return False
    cached_ids = np.load(IDS_PATH)
    if not np.array_equal(cached_ids, ids):
        return False
    stored_hash = EMBEDDING_HASH_PATH.read_text(encoding="utf-8").strip()
    return stored_hash == text_hash(texts)


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def run() -> None:
    rows = load_companies()
    ids = np.array([row["id"] for row in rows], dtype=np.int64)
    texts = [company_text(row) for row in rows]
    if is_cached(ids, texts):
        logger.info("cached")
        return

    start = time.time()
    embeddings = embed_texts(texts)
    elapsed = time.time() - start

    np.save(EMB_PATH, embeddings)
    np.save(IDS_PATH, ids)
    EMBEDDING_HASH_PATH.write_text(text_hash(texts), encoding="utf-8")

    logger.info(f"companies embedded: {len(rows)}")
    logger.info(f"embedding shape: {embeddings.shape}")
    logger.info(f"time taken: {elapsed:.2f}s")


if __name__ == "__main__":
    run()
