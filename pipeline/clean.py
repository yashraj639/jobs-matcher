import json
import sqlite3
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw.json"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "yc.db"

COLUMNS = [
    "id",
    "name",
    "slug",
    "one_liner",
    "long_description",
    "tags",
    "industries",
    "team_size",
    "batch",
    "website",
    "is_hiring",
    "stage",
    "regions",
    "launched_at",
    "app_answers",
]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY,
    name TEXT,
    slug TEXT,
    one_liner TEXT,
    long_description TEXT,
    tags TEXT,
    industries TEXT,
    team_size INTEGER,
    batch TEXT,
    website TEXT,
    is_hiring INTEGER,
    stage TEXT,
    regions TEXT,
    launched_at INTEGER,
    app_answers TEXT
)
"""


def load_raw(path: Path = RAW_PATH) -> list[dict]:
    return json.loads(path.read_text())


def is_active(rec: dict) -> bool:
    return rec.get("status") == "Active"


def has_description(rec: dict) -> bool:
    long_desc = (rec.get("long_description") or "").strip()
    one_liner = (rec.get("one_liner") or "").strip()
    return bool(long_desc or one_liner)


def not_declined_hiring(rec: dict) -> bool:
    return rec.get("isHiring") is not False


def filter_records(records: list[dict]) -> list[dict]:
    active = [r for r in records if is_active(r)]
    described = [r for r in active if has_description(r)]
    return [r for r in described if not_declined_hiring(r)]


def to_row(rec: dict) -> dict:
    app_answers = rec.get("app_answers")
    return {
        "id": rec.get("id"),
        "name": rec.get("name"),
        "slug": rec.get("slug"),
        "one_liner": rec.get("one_liner"),
        "long_description": rec.get("long_description"),
        "tags": json.dumps(rec.get("tags") or []),
        "industries": json.dumps(rec.get("industries") or []),
        "team_size": rec.get("team_size"),
        "batch": rec.get("batch"),
        "website": rec.get("website"),
        "is_hiring": int(bool(rec.get("isHiring"))),
        "stage": rec.get("stage"),
        "regions": json.dumps(rec.get("regions") or []),
        "launched_at": rec.get("launched_at"),
        "app_answers": json.dumps(app_answers) if app_answers is not None else None,
    }


def save_sqlite(rows: list[dict], db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ", ".join(COLUMNS)
    placeholders = ", ".join("?" for _ in COLUMNS)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS companies")
        conn.execute(CREATE_SQL)
        conn.executemany(
            f"INSERT INTO companies ({cols}) VALUES ({placeholders})",
            [[row[c] for c in COLUMNS] for row in rows],
        )


def run() -> None:
    records = load_raw()
    active = [r for r in records if is_active(r)]
    described = [r for r in active if has_description(r)]
    kept = filter_records(records)
    save_sqlite([to_row(r) for r in kept])
    print(f"total: {len(records)}")
    print(f"after status filter: {len(active)}")
    print(f"after description filter: {len(described)}")
    print(f"after hiring filter: {len(kept)}")
    print(f"saved to {DB_PATH}")
