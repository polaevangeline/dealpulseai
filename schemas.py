from pydantic import BaseModel, Field
from typing import List, Dict, Any


class PredictRequest(BaseModel):
    deal_id:   str = Field(..., example="D001")
    crm_notes: str = Field(..., example="Sent the proposal and pricing document to the client today.")


class StageConfidence(BaseModel):
    stage:      str
    confidence: float


class TopWord(BaseModel):          # ← ADDED: proper schema for top words
    word:  str
    score: float


class PredictResponse(BaseModel):
    deal_id:          str
    predicted_stage:  str
    confidence:       float
    all_scores:       List[StageConfidence]
    top_words:        List[TopWord]  # ← CHANGED: was List[Dict[str, float]]


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str