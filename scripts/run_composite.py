import sys
from pathlib import Path
import json
import pandas as pd

# Path fix
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.composite_score import compute_composite_score

# Load data
with open("data/processed/town_scores.json") as f:
    df = pd.DataFrame(json.load(f))

# Example: Overall Score + two high-variance columns
columns = [
    "Overall Score",
    "Home Price",
    "Median Household Income ($1,000s)"
]

weights = {
    "Overall Score": 0.5,
    "Home Price": 0.25,
    "Median Household Income ($1,000s)": 0.25
}

df["Composite Score"] = compute_composite_score(df, columns, weights)

print(df[["Town", "Zip", "Composite Score"]].head())