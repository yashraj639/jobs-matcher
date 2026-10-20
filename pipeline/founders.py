import html
import json
import re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from pipeline.paths import DB_PATH

BASE_URL = "https://www.ycombinator.com/companies/"
USER_AGENT = "personal-yc-job-matcher/0.1 (personal project; contact via GitHub)"
TIMEOUT = 15
FETCH_DELAY = 2.0
CACHE_TTL_DAYS = 7

DATA_PAGE_RE = re.compile(r'data-page="([^"]*)"')

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS founders (
    company_id INTEGER NOT NULL,
    slug TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    full_name TEXT,
    title TEXT,
    linkedin_url TEXT,
    twitter_url TEXT,
    is_active INTEGER,
    founder_bio TEXT,
    fetched_at TEXT,
    PRIMARY KEY (company_id, user_id)
)
"""

INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_founders_slug ON founders(slug)"

CREATE_LOOKUPS_SQL = """
CREATE TABLE IF NOT EXISTS founder_lookups (
    slug TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL
)
"""

FIELDS = (
    "company_id",
    "slug",
    "user_id",
    "full_name",
    "title",
    "linkedin_url",
    "twitter_url",
    "is_active",
    "founder_bio",
    "fetched_at",
)

_last_fetch_at: float | None = None


class FoundersError(RuntimeError):
    pass


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_SQL)
    conn.execute(INDEX_SQL)
    conn.execute(CREATE_LOOKUPS_SQL)
    return conn


def _is_fresh(fetched_at: str | None) -> bool:
    if not fetched_at:
        return False
    try:
        fetched = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched < timedelta(days=CACHE_TTL_DAYS)


def _load_cached(slug: str) -> list[dict] | None:
    with _connect() as conn:
        lookup = conn.execute(
            "SELECT fetched_at FROM founder_lookups WHERE slug = ?", (slug,)
        ).fetchone()
        if lookup is None or not _is_fresh(lookup["fetched_at"]):
            return None
        rows = conn.execute(
            f"SELECT {', '.join(FIELDS)} FROM founders WHERE slug = ? ORDER BY user_id",
            (slug,),
        ).fetchall()
    return [dict(row) for row in rows]


def _polite_wait() -> None:
    global _last_fetch_at
    if _last_fetch_at is not None:
        elapsed = time.monotonic() - _last_fetch_at
        if elapsed < FETCH_DELAY:
            time.sleep(FETCH_DELAY - elapsed)


def _fetch_page(slug: str) -> str:
    global _last_fetch_at
    url = BASE_URL + slug
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    _polite_wait()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            body: bytes = resp.read()
    except urllib.error.HTTPError as err:
        if err.code == 429:
            raise FoundersError(
                f"429 rate limited fetching {slug}, stopping (no retry)"
            ) from err
        raise FoundersError(
            f"HTTP {err.code} fetching {slug} ({url})"
        ) from err
    except urllib.error.URLError as err:
        raise FoundersError(f"network error fetching {slug}: {err.reason}") from err
    except TimeoutError as err:
        raise FoundersError(f"timeout fetching {slug} after {TIMEOUT}s") from err
    _last_fetch_at = time.monotonic()
    return body.decode("utf-8", errors="replace")


def _require_user_id(slug: str, founder: dict) -> int:
    user_id = founder.get("user_id")
    if user_id is not None:
        try:
            return int(user_id)
        except (TypeError, ValueError):
            pass
    name = founder.get("full_name") or "unknown founder"
    raise FoundersError(
        f"active founder on page for {slug} has no usable user_id: {name}"
    ) from None


def _extract(html_text: str, slug: str) -> tuple[int, list[dict]]:
    match = DATA_PAGE_RE.search(html_text)
    if not match:
        raise FoundersError(f"no data-page attribute on page for {slug}")

    try:
        data = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError as err:
        raise FoundersError(f"could not parse data-page JSON for {slug}: {err}") from err

    company = (data.get("props") or {}).get("company")
    if not isinstance(company, dict):
        raise FoundersError(f"page for {slug} has no props.company")

    company_id = company.get("id")
    if company_id is None:
        raise FoundersError(f"page for {slug} has no props.company.id")

    founders = company.get("founders")
    if founders is None:
        raise FoundersError(f"page for {slug} has no props.company.founders")
    if not isinstance(founders, list):
        raise FoundersError(f"props.company.founders for {slug} is not a list")

    active = [f for f in founders if isinstance(f, dict) and f.get("is_active")]
    for founder in active:
        _require_user_id(slug, founder)
    return int(company_id), active


def _to_row(company_id: int, slug: str, founder: dict, fetched_at: str) -> dict:
    return {
        "company_id": company_id,
        "slug": slug,
        "user_id": founder.get("user_id"),
        "full_name": founder.get("full_name"),
        "title": founder.get("title"),
        "linkedin_url": founder.get("linkedin_url"),
        "twitter_url": founder.get("twitter_url"),
        "is_active": int(bool(founder.get("is_active"))),
        "founder_bio": founder.get("founder_bio"),
        "fetched_at": fetched_at,
    }


def _store(company_id: int, slug: str, founders: list[dict]) -> list[dict]:
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [_to_row(company_id, slug, f, fetched_at) for f in founders]
    with _connect() as conn:
        conn.execute("DELETE FROM founders WHERE company_id = ?", (company_id,))
        placeholders = ", ".join("?" for _ in FIELDS)
        conn.executemany(
            f"INSERT INTO founders ({', '.join(FIELDS)}) VALUES ({placeholders})",
            [[row[c] for c in FIELDS] for row in rows],
        )
        conn.execute(
            "INSERT OR REPLACE INTO founder_lookups (slug, fetched_at) "
            "VALUES (?, ?)",
            (slug, fetched_at),
        )
    return rows


def get_founders_with_source(slug: str) -> tuple[list[dict], str]:
    cached = _load_cached(slug)
    if cached is not None:
        return [{**row, "source": "cached"} for row in cached], "cached"

    html_text = _fetch_page(slug)
    company_id, active = _extract(html_text, slug)
    rows = _store(company_id, slug, active)
    return [{**row, "source": "fetched"} for row in rows], "fetched"


def get_founders(slug: str) -> list[dict]:
    founders, _ = get_founders_with_source(slug)
    return founders
