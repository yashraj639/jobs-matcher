import json
import logging
import urllib.request
from pathlib import Path

from pipeline.paths import RAW_PATH

logger = logging.getLogger(__name__)

ALL_JSON_URL = "https://yc-oss.github.io/api/companies/all.json"


def download(url: str = ALL_JSON_URL) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        records: list[dict] = json.load(resp)
        return records


def save_raw(records: list[dict], path: Path = RAW_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))
    return path


def fetch() -> list[dict]:
    records = download()
    save_raw(records)
    return records


def print_samples(records: list[dict], n: int = 5) -> None:
    logger.info(f"Total companies: {len(records)}")
    for rec in records[:n]:
        logger.info("-" * 60)
        for key, value in rec.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    print_samples(fetch())
