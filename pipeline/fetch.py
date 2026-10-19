import json
import urllib.request
from pathlib import Path

ALL_JSON_URL = "https://yc-oss.github.io/api/companies/all.json"
RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw.json"


def download(url: str = ALL_JSON_URL) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def save_raw(records: list[dict], path: Path = RAW_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))
    return path


def fetch() -> list[dict]:
    records = download()
    save_raw(records)
    return records


def print_samples(records: list[dict], n: int = 5) -> None:
    print(f"Total companies: {len(records)}")
    for rec in records[:n]:
        print("-" * 60)
        for key, value in rec.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    print_samples(fetch())
