import os
import urllib.request
import zipfile

import pandas as pd
import streamlit as st

from src.geo_utils import load_zip_shapes
from src.variance_analysis import compute_relative_variance_cv
from src.composite_score import compute_composite_score
from src.heatmap import create_zip_heatmap

st.set_page_config(layout="wide")
st.title("ZIP Heatmap (Overall + Top-2 Variance Sliders)")

SHEET_ID = "138F3qdX_VAHuC6eI6z_AfFqTj3xtJMFk"
GID = 1097485755  # tab gid from your URL


# ----------------------------
# Shapes loader (single definition)
# ----------------------------
@st.cache_resource
def load_shapes():
    # IMPORTANT: ZCTA520 (not ZCTA5)
    url = "https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip"
    extract_dir = "data/zcta_cache"
    shp_path = os.path.join(extract_dir, "tl_2024_us_zcta520.shp")

    if not os.path.exists(shp_path):
        os.makedirs(extract_dir, exist_ok=True)
        zip_path = os.path.join(extract_dir, "tl_2024_us_zcta520.zip")

        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    return load_zip_shapes(shp_path)


# ----------------------------
# Scores loader (Google Sheets)
# ----------------------------
@st.cache_data
def load_scores(sheet_id: str, gid: int = 0) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    # Quick connectivity check (optional)
    with urllib.request.urlopen(url) as resp:
        st.write("DEBUG sheets status:", resp.status)

    # CRITICAL: dtype=str preserves leading zeros in ZIPs
    df = pd.read_csv(url, dtype=str)

    # Clean headers
    df.columns = df.columns.map(lambda c: str(c).strip())
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

    def canon(s: str) -> str:
        s = str(s).strip().lower()
        for ch in ["(", ")", "%", "$", ",", "/", "-", "_"]:
            s = s.replace(ch, " ")
        return " ".join(s.split())

    zip_col = None
    overall_col = None

    for c in df.columns:
        cc = canon(c)

        if zip_col is None and (
            cc == "zip"
            or "zip code" in cc
            or "zipcode" in cc
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

    df = df.rename(columns={zip_col: "Zip", overall_col: "Overall Score"})

    # Bulletproof ZIP normalization
    df["Zip"] = (
        df["Zip"].astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(5)
        .str[-5:]
    )
    df = df[df["Zip"].str.match(r"^\d{5}$", na=False)]

    return df


# ----------------------------
# Load data + shapes
# ----------------------------
df = load_scores(SHEET_ID, gid=GID)
zip_gdf = load_shapes()

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

# Debug + stop if empty
if merged.empty:
    st.error("No ZIP geometries matched your score data (merged is empty).")
    st.write("Sample ZIPs in data:", df["Zip"].head(20).tolist())
    st.write("Sample ZIPs in shapes:", zip_gdf["Zip"].head(20).tolist())
    st.write("Unique ZIPs in data:", int(df["Zip"].nunique()))
    st.write("Unique ZIPs in shapes:", int(zip_gdf["Zip"].nunique()))
    st.stop()

st.write("Using top-2 CV columns:", top2)
st.write("Merged ZIPs:", len(merged))

# Safe center using bounds
minx, miny, maxx, maxy = merged.total_bounds
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

m = create_zip_heatmap(
    merged,
    "Composite Score",
    center=(center_lat, center_lon),
    zoom=9 if scope.endswith("(fast)") else 4,
)

st.components.v1.html(m._repr_html_(), height=720)