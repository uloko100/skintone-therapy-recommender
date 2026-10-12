from typing import Literal

from pydantic import BaseModel, Field


class SkinProfile(BaseModel):
    monk_tone: int = Field(..., ge=1, le=10, description="Monk skin tone, 1 (lightest) to 10 (darkest)")
    age: int = Field(..., ge=16, le=65)
    skin_type: Literal["dry", "oily", "combination", "normal"]
    skin_concern: Literal["acne", "hyperpigmentation", "dryness", "fine_lines", "redness", "dullness"]
    sensitive: int = Field(..., ge=0, le=1, description="1 if the skin reacts easily, otherwise 0")

    model_config = {
        "json_schema_extra": {
            "example": {
                "monk_tone": 8, "age": 24, "skin_type": "combination",
                "skin_concern": "hyperpigmentation", "sensitive": 0,
            }
        }
    }


class Recommendation(BaseModel):
    therapy: str
    confidence: float


class PredictionResponse(BaseModel):
    recommendations: list[Recommendation]
    disclaimer: str