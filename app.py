import json
from pathlib import Path
import os

import pandas as pd
import streamlit as st

from src.geo_utils import load_zip_shapes
from src.variance_analysis import compute_relative_variance_cv
from src.composite_score import compute_composite_score
from src.heatmap import create_zip_heatmap

st.set_page_config(layout="wide")
st.title("ZIP Heatmap (Overall + Top-2 Variance Sliders)")

DATA_JSON = "data/processed/town_scores.json"
ZCTA_SHP = "data/zcta/tl_2024_us_zcta520/tl_2024_us_zcta520.shp"

@st.cache_data
# def load_scores(json_path: str) -> pd.DataFrame:
#     with open(json_path) as f:
#         records = json.load(f)
#     df = pd.DataFrame(records)
#     # Ensure Zip is 5-digit string
#     df["Zip"] = df["Zip"].astype(str).str.zfill(5)
#     return df
def load_scores():
    # Prefer JSON if present (local dev)
    if os.path.exists("data/processed/town_scores.json"):
        with open("data/processed/town_scores.json") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["Zip"] = df["Zip"].astype(str).str.zfill(5)
        return df

    # Cloud fallback: load from raw file committed to repo
    if os.path.exists("data/raw/town_scores.xlsx"):
        df = pd.read_excel("data/raw/town_scores.xlsx", header=1)
    elif os.path.exists("data/raw/town_scores.csv"):
        df = pd.read_csv("data/raw/town_scores.csv", header=1)
    else:
        raise FileNotFoundError("No data found. Expected data/processed/town_scores.json or data/raw/town_scores.(xlsx|csv)")

    # Normalize the two required fields
    df.columns = df.columns.map(lambda c: str(c).strip())
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    
    def canon(s: str) -> str:
    s = str(s).strip().lower()
    for ch in ["(", ")", "%", "$", ",", "/", "-", "_"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

    canon_map = {canon(c): c for c in df.columns}

    zip_candidates = [
        "zip", "zipcode", "zip code", "zcta", "zcta5", "zcta5ce", "zcta5ce20", "zcta5ce10"
    ]
    overall_candidates = [
        "overall score", "overall", "score", "total score"
    ]

    zip_col = next((canon_map[k] for k in zip_candidates if k in canon_map), None)
    overall_col = next((canon_map[k] for k in overall_candidates if k in canon_map), None)

    if zip_col is None:
        raise ValueError(f"Missing ZIP column. Found: {list(df.columns)}")
    if overall_col is None:
        raise ValueError(f"Missing Overall Score column. Found: {list(df.columns)}")

    df = df.rename(columns={zip_col: "Zip", overall_col: "Overall Score"})

    df["Zip"] = (
        df["Zip"]
        .astype(str)
        .str.extract(r"(\d{5})", expand=False)
        .str.zfill(5)
    )

    return df

@st.cache_data
def load_shapes(shp_path: str):
    return load_zip_shapes(shp_path)

# df = load_scores(DATA_JSON)
df = load_scores()
zip_gdf = load_shapes(ZCTA_SHP)

# Optional speed filter (recommended for first run)
st.sidebar.header("Map Scope")
scope = st.sidebar.selectbox("Scope", ["All US (slow)", "Massachusetts (fast)"], index=1)
if scope == "Massachusetts (fast)":
    zip_gdf = zip_gdf[zip_gdf["Zip"].str.startswith("0")]

# Find top-2 “highest variance” columns using CV
cv = compute_relative_variance_cv(df)
top2 = cv.index[:2].tolist()

st.sidebar.header("Weights (auto-normalized)")
w_overall = st.sidebar.slider("Overall Score", 0.0, 1.0, 0.50, 0.01)
w_1 = st.sidebar.slider(top2[0], 0.0, 1.0, 0.25, 0.01)
w_2 = st.sidebar.slider(top2[1], 0.0, 1.0, 0.25, 0.01)

columns = ["Overall Score", top2[0], top2[1]]
weights = {"Overall Score": w_overall, top2[0]: w_1, top2[1]: w_2}

# Compute composite
df["Composite Score"] = compute_composite_score(df, columns, weights)

# Merge shapes with scores
merged = zip_gdf.merge(df, on="Zip", how="inner")

# Display info + map
st.write("Using top-2 CV columns:", top2)
st.write("Merged ZIPs:", len(merged))

# Rough center: use merged centroid average (keeps it general)
center_lat = float(merged.geometry.centroid.y.mean())
center_lon = float(merged.geometry.centroid.x.mean())

m = create_zip_heatmap(merged, "Composite Score", center=(center_lat, center_lon), zoom=9 if scope.endswith("(fast)") else 4)
st.components.v1.html(m._repr_html_(), height=720)