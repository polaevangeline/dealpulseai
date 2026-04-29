import sys
sys.path.insert(0, "src")

import torch
import mlflow
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import pandas as pd

from models.bert_classifier import load_model
from features.bert_embedder import build_all_dataloaders

PROCESSED_DIR = Path("data/processed")
MODEL_SAVE    = Path("data/models/best_model.pt")
MODEL_NAME    = "distilbert-base-uncased"
EPOCHS        = 3
LR            = 2e-5
BATCH_SIZE    = 16


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            logits      = model(input_ids, attention_mask)
            loss        = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    f1  = f1_score(all_labels, all_preds, average="weighted")
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return total_loss / len(loader), acc, f1


def train():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader = build_all_dataloaders(
        train_df, val_df, test_df, tokenizer, BATCH_SIZE
    )

    model, device = load_model()
    criterion     = nn.CrossEntropyLoss()
    optimizer     = AdamW(model.parameters(), lr=LR)
    total_steps   = len(train_loader) * EPOCHS
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    MODEL_SAVE.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    mlflow.set_experiment("DealPulse-AI")

    with mlflow.start_run():
        mlflow.log_params({"epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE})

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, device
            )
            val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

            print(f"Epoch {epoch}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "val_f1":     val_f1
            }, step=epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), MODEL_SAVE)
                print(f"[train] Best model saved (F1: {best_f1:.4f})")

        _, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, device)
        mlflow.log_metrics({"test_acc": test_acc, "test_f1": test_f1})
        mlflow.pytorch.log_model(model, "deal_stage_model")
        print(f"\n[train] Done! Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    train()