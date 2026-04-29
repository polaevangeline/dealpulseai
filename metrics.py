import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

LABELS_PATH = Path("data/labels.json")


def load_stage_names():
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    return list(labels["stages"].values())


def compute_metrics(y_true: list, y_pred: list) -> dict:
    """Compute all evaluation metrics"""
    stage_names = load_stage_names()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    cm  = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=stage_names)

    print(f"\n{'='*40}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"{'='*40}\n")

    return {
        "accuracy":  acc,
        "f1_score":  f1,
        "confusion_matrix": cm.tolist(),
        "report":    report
    }


def save_metrics(metrics: dict, path: str = "data/processed/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metrics_to_save = {k: v for k, v in metrics.items() if k != "report"}
    with open(path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"[metrics] Saved to {path}")