import json
from functools import lru_cache
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "therapy_model.pkl"
META_PATH = MODELS_DIR / "model_metadata.json"


@lru_cache(maxsize=1)
def get_model():
    """Load the trained pipeline once and reuse it for every request."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_metadata() -> dict:
    return json.loads(META_PATH.read_text())