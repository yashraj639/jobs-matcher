import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "yc.db"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_ids.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


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


def is_cached(count: int) -> bool:
    if not (EMB_PATH.exists() and IDS_PATH.exists()):
        return False
    return len(np.load(IDS_PATH)) == count


def embed_texts(texts: list[str]) -> np.ndarray:
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
    if is_cached(len(rows)):
        print("cached")
        return

    ids = np.array([row["id"] for row in rows], dtype=np.int64)
    texts = [company_text(row) for row in rows]

    start = time.time()
    embeddings = embed_texts(texts)
    elapsed = time.time() - start

    np.save(EMB_PATH, embeddings)
    np.save(IDS_PATH, ids)

    print(f"companies embedded: {len(rows)}")
    print(f"embedding shape: {embeddings.shape}")
    print(f"time taken: {elapsed:.2f}s")


if __name__ == "__main__":
    run()
