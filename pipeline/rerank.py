import json
import sqlite3
from pathlib import Path

from pipeline.llm import LLMParseError, generate_json

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "yc.db"

DESC_LIMIT = 400
DEFAULT_TOP_N = 15
UNKNOWN_TEAM = "unknown"


def _load_descriptions(ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT id, long_description FROM companies WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    return {int(row[0]): (row[1] or "") for row in rows}


def _team_size(value: object) -> str:
    try:
        size = int(value)  # may be None or garbage
    except (TypeError, ValueError):
        return UNKNOWN_TEAM
    return str(size) if size > 0 else UNKNOWN_TEAM


def _build_payload(candidates: list[dict], descriptions: dict[int, str]) -> list[dict]:
    return [
        {
            "id": int(c["id"]),
            "name": c.get("name"),
            "one_liner": c.get("one_liner"),
            "long_description": descriptions.get(int(c["id"]), "")[:DESC_LIMIT],
            "team_size": _team_size(c.get("team_size")),
            "batch": c.get("batch"),
            "is_hiring": bool(c.get("is_hiring")),
        }
        for c in candidates
    ]


def _build_prompt(profile_text: str, payload: list[dict], top_n: int) -> str:
    return (
        "Pick the YC startups that best fit the candidate below.\n\n"
        f"## Candidate profile\n{profile_text}\n\n"
        f"## Startups\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "## Rules\n"
        f"1. Only use ids from the list above, choose at most {top_n}.\n"
        "2. Base every reason only on the text above; never invent facts "
        "about a company.\n"
        "3. Prefer concrete fit (stack, domain, problem space) over generic "
        "'AI' overlap.\n"
        f"4. A team_size of \"{UNKNOWN_TEAM}\" means the size is not known, "
        "not that the team is small.\n"
        "5. Give one specific sentence per startup, no flattery.\n\n"
        "## Output\n"
        "Return JSON only, exactly of this shape:\n"
        '{"picks": [{"id": <id from the list>, "reason": "<one sentence>"}]}'
    )


def _parse_picks(raw: dict, valid_ids: set[int]) -> tuple[list[dict], int]:
    """Return clean picks plus the count of entries dropped in validation."""
    picks = raw.get("picks")
    if not isinstance(picks, list) or not picks:
        raise LLMParseError("response has no non-empty 'picks' list")

    cleaned: list[dict] = []
    seen: set[int] = set()
    rejected = 0
    for item in picks:
        if not isinstance(item, dict):
            rejected += 1
            continue
        try:
            company_id = int(item.get("id"))
        except (TypeError, ValueError):
            rejected += 1
            continue
        reason = item.get("reason")
        if company_id not in valid_ids or company_id in seen:
            rejected += 1
            continue
        if not isinstance(reason, str) or not reason.strip():
            rejected += 1
            continue
        seen.add(company_id)
        cleaned.append({"id": company_id, "reason": reason.strip()})

    if not cleaned:
        raise LLMParseError("no usable picks (unknown ids or missing reasons)")
    return cleaned, rejected


def rerank_with_stats(
    profile_text: str, candidates: list[dict], top_n: int = DEFAULT_TOP_N
) -> tuple[list[dict], int]:
    """Re-rank and also report how many picks were dropped in validation."""
    if not candidates:
        return [], 0

    valid_ids = {int(c["id"]) for c in candidates}
    payload = _build_payload(candidates, _load_descriptions(list(valid_ids)))
    prompt = _build_prompt(profile_text, payload, top_n)

    picks: list[dict] | None = None
    rejected = 0
    for attempt in range(2):
        try:
            picks, rejected = _parse_picks(generate_json(prompt), valid_ids)
            break
        except LLMParseError as err:
            if attempt == 1:
                raise LLMParseError(f"re-rank failed after retry: {err}") from err

    by_id = {int(c["id"]): c for c in candidates}
    results = []
    for pick in picks[:top_n]:
        candidate = dict(by_id[pick["id"]])
        candidate["reason"] = pick["reason"]
        results.append(candidate)
    return results, rejected


def rerank(
    profile_text: str, candidates: list[dict], top_n: int = DEFAULT_TOP_N
) -> list[dict]:
    results, _ = rerank_with_stats(profile_text, candidates, top_n)
    return results
