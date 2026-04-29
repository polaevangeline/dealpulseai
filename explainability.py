import sys
sys.path.insert(0, "src")          # ← ADDED: fixes ModuleNotFoundError

import json
import torch
import shap
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

LABELS_PATH = Path("data/labels.json")
MODEL_NAME  = "distilbert-base-uncased"


def load_stage_names():
    with open(LABELS_PATH) as f:
        return list(json.load(f)["stages"].values())


def predict_proba(texts: list, model, tokenizer, device) -> np.ndarray:
    """Returns softmax probabilities for a list of texts"""
    model.eval()
    encodings = tokenizer(
        texts,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def explain_prediction(
    text: str,
    model,
    tokenizer,
    device,
    background_texts: list
) -> dict:
    """
    Use SHAP to explain which words drove the prediction.
    Returns top words and their importance scores.
    """
    stage_names = load_stage_names()

    # Wrap predict for SHAP
    def predictor(texts):
        return predict_proba(list(texts), model, tokenizer, device)

    # SHAP explainer
    explainer   = shap.Explainer(predictor, shap.maskers.Text(tokenizer))
    shap_values = explainer([text])

    # Get predicted class
    probs           = predict_proba([text], model, tokenizer, device)[0]
    predicted_id    = int(np.argmax(probs))
    predicted_stage = stage_names[predicted_id]

    # Top words for predicted class
    values = shap_values[0, :, predicted_id].values
    tokens = shap_values[0, :, predicted_id].data

    word_scores = sorted(
        zip(tokens, values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    result = {
        "text":            text,
        "predicted_stage": predicted_stage,
        "confidence":      float(probs[predicted_id]),
        "top_words":       [{"word": w, "score": round(float(s), 4)}
                            for w, s in word_scores]
    }

    print(f"\n[explainability] Predicted: {predicted_stage} ({result['confidence']:.2%})")
    print(f"Top influential words:")
    for item in result["top_words"]:
        print(f"  {item['word']:20s} → {item['score']:.4f}")

    return result


if __name__ == "__main__":
    from models.bert_classifier import load_model      # ← CHANGED: removed 'src.'
    from transformers import AutoTokenizer

    model, device = load_model("data/models/best_model.pt")
    tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)

    sample_text = "Sent the pricing proposal. Client asked for a discount on annual plan."
    background  = ["Follow up call scheduled", "Contract signed today", "Lost to competitor"]

    result = explain_prediction(sample_text, model, tokenizer, device, background)