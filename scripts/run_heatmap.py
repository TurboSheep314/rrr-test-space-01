import sys
from pathlib import Path
import json
import pandas as pd

# Path fix
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.geo_utils import load_zip_shapes
from src.heatmap import create_zip_heatmap
from src.composite_score import compute_composite_score

# Load town data
with open("data/processed/town_scores.json") as f:
    df = pd.DataFrame(json.load(f))

# Columns + weights (static for now)
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

# Load ZIP shapes
zip_gdf = load_zip_shapes(
    "data/zcta/tl_2024_us_zcta520/tl_2024_us_zcta520.shp",
    zip_col="ZCTA5CE20"
)

# Merge
merged = zip_gdf.merge(df, on="Zip", how="inner")

# Create map
m = create_zip_heatmap(
    merged,
    value_col="Composite Score"
)

# Save output
output_path = "output/zip_heatmap.html"
Path("output").mkdir(exist_ok=True)
m.save(output_path)

print(f"✔ Heatmap saved to {output_path}")