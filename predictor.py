import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from src.models.bert_classifier import load_model
from src.evaluation.explainability import predict_proba, explain_prediction

LABELS_PATH  = Path("data/labels.json")
MODEL_PATH   = Path("data/models/best_model.pt")
MODEL_NAME   = "distilbert-base-uncased"


def load_stage_names():
    with open(LABELS_PATH) as f:
        return list(json.load(f)["stages"].values())


class Predictor:
    """
    Singleton predictor — loads model once,
    reuses for every API request.
    """
    def __init__(self):
        self.stage_names = load_stage_names()
        self.model, self.device = load_model(str(MODEL_PATH))
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("[predictor] Ready!")

    def predict(self, deal_id: str, crm_notes: str) -> dict:
        # Get probabilities
        probs        = predict_proba([crm_notes], self.model, self.tokenizer, self.device)[0]
        predicted_id = int(np.argmax(probs))

        # Get SHAP explanation
        background = [
            "Follow up call scheduled",
            "Contract signed today",
            "Lost to competitor"
        ]
        explanation = explain_prediction(
            crm_notes, self.model, self.tokenizer, self.device, background
        )

        return {
            "deal_id":         deal_id,
            "predicted_stage": self.stage_names[predicted_id],
            "confidence":      round(float(probs[predicted_id]), 4),
            "all_scores": [
                {"stage": s, "confidence": round(float(p), 4)}
                for s, p in zip(self.stage_names, probs)
            ],
            "top_words": explanation["top_words"]
        }


# Single instance reused across all requests
predictor = Predictor()