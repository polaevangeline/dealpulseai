import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


PROCESSED_DIR  = Path("data/processed")
FEATURES_DIR   = Path("data/processed/features")
MODEL_NAME     = "distilbert-base-uncased"
MAX_LENGTH     = 256
BATCH_SIZE     = 16


# ─── Custom Dataset ───────────────────────────────────────────────────────────

class CRMDataset(Dataset):
    """
    PyTorch Dataset for CRM notes.
    Tokenizes text and returns input tensors for BERT.
    """
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = MAX_LENGTH):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str = MODEL_NAME):
    """Load HuggingFace tokenizer"""
    print(f"[bert] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[bert] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    return tokenizer


def tokenize_texts(
    texts: list,
    tokenizer,
    max_length: int = MAX_LENGTH
) -> dict:
    """
    Tokenize a list of texts.
    Returns dict with input_ids and attention_mask tensors.
    """
    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    print(f"[bert] Tokenized {len(texts)} texts → shape: {encodings['input_ids'].shape}")
    return encodings


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_dataset(
    df: pd.DataFrame,
    tokenizer,
    text_col:  str = "clean_notes",
    label_col: str = "label"
) -> CRMDataset:
    """Build a CRMDataset from a DataFrame"""
    texts  = df[text_col].tolist()
    labels = df[label_col].tolist()
    return CRMDataset(texts, labels, tokenizer)


def build_dataloader(
    dataset: CRMDataset,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True
) -> DataLoader:
    """Wrap dataset in a DataLoader for batched training"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ─── Embedding Extractor ──────────────────────────────────────────────────────

def extract_embeddings(
    texts: list,
    tokenizer,
    model,
    device: str = None,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Extract [CLS] token embeddings from DistilBERT.
    These are 768-dimensional vectors representing each CRM note.
    Used for similarity search or lightweight downstream classifiers.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]

        encodings = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids      = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token = first token = sentence-level representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

        print(f"[bert] Embedded batch {i // batch_size + 1} / {(len(texts) - 1) // batch_size + 1}")

    embeddings = np.vstack(all_embeddings)
    print(f"[bert] Final embedding shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, split: str = "train"):
    """Save embeddings as .npy files"""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / f"{split}_embeddings.npy"
    np.save(path, embeddings)
    print(f"[bert] Embeddings saved to {path}")


def load_embeddings(split: str = "train") -> np.ndarray:
    """Load saved embeddings"""
    path = FEATURES_DIR / f"{split}_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found at {path}. Run bert_embedder.py first.")
    embeddings = np.load(path)
    print(f"[bert] Loaded {split} embeddings: {embeddings.shape}")
    return embeddings


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def build_all_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    tokenizer,
    batch_size: int = BATCH_SIZE
):
    """
    Build DataLoaders for all 3 splits.
    Used directly in bert_classifier.py training loop.
    """
    train_dataset = build_dataset(train_df, tokenizer)
    val_dataset   = build_dataset(val_df,   tokenizer)
    test_dataset  = build_dataset(test_df,  tokenizer)

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = build_dataloader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = build_dataloader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"[bert] DataLoaders ready → "
          f"Train: {len(train_loader)} batches | "
          f"Val: {len(val_loader)} batches | "
          f"Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Load processed data
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df   = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "test.csv")

    # Load tokenizer and model
    tokenizer = load_tokenizer()
    model     = AutoModel.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bert] Using device: {device}")

    # Extract and save embeddings for all splits
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        embeddings = extract_embeddings(
            df["clean_notes"].tolist(),
            tokenizer,
            model,
            device=device
        )
        save_embeddings(embeddings, split=split)

    print("\n[bert] All embeddings extracted and saved successfully!")