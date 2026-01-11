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
def read_table_with_best_header(path: str) -> pd.DataFrame:
    # Try a few header rows and choose the one that produces sensible column names
    candidates = []
    if path.endswith(".xlsx"):
        for h in [0, 1, 2, 3, 4, 5]:
            try:
                tmp = pd.read_excel(path, header=h)
                candidates.append((h, tmp))
            except Exception:
                pass
    else:
        for h in [0, 1, 2, 3, 4, 5]:
            try:
                tmp = pd.read_csv(path, header=h)
                candidates.append((h, tmp))
            except Exception:
                pass

    def score_cols(cols):
        cols = [str(c).strip().lower() for c in cols]
        score = 0
        for c in cols:
            if "zip" in c or "zcta" in c or "postal" in c:
                score += 5
            if "overall" in c and "score" in c:
                score += 5
            if c == "town":
                score += 3
        return score

    if not candidates:
        raise ValueError("Could not read file with any tested header rows (0-5).")

    best_h, best_df = max(candidates, key=lambda t: score_cols(t[1].columns))
    st.write("DEBUG: selected header row =", best_h)
    st.write("DEBUG: columns =", list(best_df.columns))
    return best_df
#     return df
def load_scores():
    # ----------------------------------------
    # 1) Prefer processed JSON (local dev)
    # ----------------------------------------
    if os.path.exists("data/processed/town_scores.json"):
        with open("data/processed/town_scores.json") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["Zip"] = df["Zip"].astype(str).str.zfill(5)
        return df

    # ----------------------------------------
    # 2) Cloud fallback: load raw data
    # ----------------------------------------
    if os.path.exists("data/raw/town_scores.xlsx"):
        df = read_table_with_best_header("data/raw/town_scores.xlsx")
    elif os.path.exists("data/raw/town_scores.csv"):
        df = read_table_with_best_header("data/raw/town_scores.csv")
    else:
        raise FileNotFoundError(
            "No data found. Expected data/processed/town_scores.json "
            "or data/raw/town_scores.(xlsx|csv)"
        )

    # ----------------------------------------
    # 3) Clean column headers
    # ----------------------------------------
    df.columns = df.columns.map(lambda c: str(c).strip())
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

    # TEMP DEBUG — remove after first successful deploy
    st.write("DEBUG columns:", list(df.columns))

    # ----------------------------------------
    # 4) Auto-detect Zip + Overall Score columns
    # ----------------------------------------
    def canon(s: str) -> str:
        s = str(s).strip().lower()
        for ch in ["(", ")", "%", "$", ",", "/", "-", "_"]:
            s = s.replace(ch, " ")
        s = " ".join(s.split())
        return s

    zip_col = None
    overall_col = None

    for c in df.columns:
        cc = canon(c)

        if zip_col is None and (
            "zip" in cc
            or "postal" in cc
            or "zcta" in cc
            or cc in ["geoid", "geoid20"]
        ):
            zip_col = c

        if overall_col is None and (
            cc == "overall score"
            or ("overall" in cc and "score" in cc)
            or cc in ["overall", "score", "total score"]
        ):
            overall_col = c

    if zip_col is None:
        raise ValueError(f"Missing ZIP column. Found: {list(df.columns)}")

    if overall_col is None:
        raise ValueError(f"Missing Overall Score column. Found: {list(df.columns)}")

    # ----------------------------------------
    # 5) Normalize + return
    # ----------------------------------------
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