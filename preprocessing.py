import re
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

LABELS_PATH   = Path("data/labels.json")
PROCESSED_DIR = Path("data/processed")


def load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    labels      = load_labels()
    stage_to_id = labels["stage_to_id"]
    unknown     = set(df["stage"].unique()) - set(stage_to_id.keys())
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}")
    df["label"] = df["stage"].map(stage_to_id)
    print(f"[preprocessing] Labels encoded:\n{df['stage'].value_counts()}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[preprocessing] Starting on {len(df)} records...")
    df["clean_notes"] = df["crm_notes"].apply(clean_text)
    df = encode_labels(df)
    before = len(df)
    df = df.drop_duplicates(subset=["clean_notes"])
    print(f"[preprocessing] Removed {before - len(df)} duplicates. Remaining: {len(df)}")
    df = df[["deal_id", "crm_notes", "clean_notes", "stage", "label"]].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame,
               train_size: float = 0.70,
               val_size:   float = 0.15,
               test_size:  float = 0.15,
               random_state: int = 42):

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=random_state
    )

    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=random_state
    )

    print(f"[preprocessing] Split → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True),   \
           test_df.reset_index(drop=True)


def save_splits(train_df, val_df, test_df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
    test_df.to_csv(PROCESSED_DIR  / "test.csv",  index=False)
    print(f"[preprocessing] Saved to {PROCESSED_DIR}/")


def run_preprocessing(df: pd.DataFrame, save: bool = True):
    df = preprocess(df)
    train_df, val_df, test_df = split_data(df)
    if save:
        save_splits(train_df, val_df, test_df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    from ingestion import load_crm_data
    df = load_crm_data("sales_pipeline.csv")
    train_df, val_df, test_df = run_preprocessing(df, save=True)