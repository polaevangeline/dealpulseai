import json
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
LABELS_PATH  = Path("data/labels.json")


def load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


def load_crm_data(filename: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print(f"[ingestion] Loaded {len(df)} records from {filename}")

    # Rename columns
    df = df.rename(columns={
        "opportunity_id": "deal_id",
        "deal_stage":     "stage"
    })

    # Build crm_notes from existing columns
    df["crm_notes"] = (
        "Deal stage " + df["stage"].astype(str) + ". " +
        "Sales agent " + df["sales_agent"].astype(str) +
        " is working on deal with " + df["account"].astype(str) +
        " for product " + df["product"].astype(str) +
        ". Close value is " + df["close_value"].astype(str) +
        ". Engaged on " + df["engage_date"].astype(str) +
        ". Expected close on " + df["close_date"].astype(str)
    )

    # Keep only needed columns
    df = df[["deal_id", "crm_notes", "stage"]]
    df = df.dropna().reset_index(drop=True)

    print(f"[ingestion] Final records: {len(df)}")
    print(f"[ingestion] Stage distribution:\n{df['stage'].value_counts()}")
    return df


if __name__ == "__main__":
    df = load_crm_data("sales_pipeline.csv")
    print(df.head(3))