import pandas as pd
import json
import argparse
from pathlib import Path
import re

REQUIRED_COLUMNS = {"Zip", "Overall Score"}

def load_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\xa0", " ", regex=False)   # non-breaking spaces
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    return df
def validate_schema(df: pd.DataFrame):
    print("\n=== COLUMN DEBUG ===")
    for i, c in enumerate(df.columns):
        print(f"{i}: {repr(c)}")
    print("====================\n")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
# def validate_schema(df: pd.DataFrame):
#     missing = REQUIRED_COLUMNS - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing required columns: {missing}")

def convert_to_json(input_path, output_path):
    df = load_file(input_path)
    df = normalize_columns(df)
    validate_schema(df)

    df["Zip"] = df["Zip"].astype(str).str.zfill(5)

    records = df.to_dict(orient="records")

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"✔ Converted {input_path.name} → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV/Excel to canonical JSON")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    convert_to_json(
        Path(args.input),
        Path(args.output)
    )