from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "yc.db"
RAW_PATH = DATA_DIR / "raw.json"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_ids.npy"

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
