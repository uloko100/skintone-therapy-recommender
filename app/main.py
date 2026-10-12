from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model_loader import get_metadata, get_model
from app.schemas import PredictionResponse, SkinProfile

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="SkinTone Therapy Recommender",
    description="Recommends facial therapies based on skin tone and skin concern. "
                "Educational demo for CSC 303, Veritas University Abuja.",
    version="0.2.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    """Serve the web page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.get("/metadata", tags=["model"])
def metadata():
    """Model details: version, features, classes and scores."""
    try:
        return get_metadata()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model metadata not found")


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(profile: SkinProfile):
    """Return the top 3 recommended therapies with confidences."""
    try:
        model = get_model()
        meta = get_metadata()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model file not found")

    row = pd.DataFrame([profile.model_dump()])[meta["features"]]
    proba = model.predict_proba(row)[0]
    top = np.argsort(proba)[::-1][:3]

    return {
        "recommendations": [
            {"therapy": str(model.classes_[i]), "confidence": round(float(proba[i]), 3)}
            for i in top
        ],
        "disclaimer": meta["disclaimer"],
    }