import json
import pandas as pd
from src.variance_analysis import compute_relative_variance_cv

# Load canonical JSON
with open("data/processed/town_scores.json") as f:
    records = json.load(f)

df = pd.DataFrame(records)

rel_var = compute_relative_variance_cv(df)

print("\n=== VARIANCE RANKING ===")
for col, var in rel_var.items():
    print(f"{col:35s} {var:,.2f}")

print("\nTOP 2 VARIANCE COLUMNS:")
print(list(rel_var.index[:2]))