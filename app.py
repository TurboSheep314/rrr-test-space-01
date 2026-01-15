# app.py

import os
import urllib.request
import zipfile

import pandas as pd
import streamlit as st

from src.geo_utils import load_zip_shapes
from src.variance_analysis import compute_relative_variance_cv
from src.composite_score import compute_composite_score
from src.heatmap import create_zip_heatmap


# ----------------------------
# Streamlit page
# ----------------------------
st.set_page_config(layout="wide")
st.title("ZIP Heatmap (Overall + Top-2 Variance Sliders)")

SHEET_ID = "138F3qdX_VAHuC6eI6z_AfFqTj3xtJMFk"
GID = 1097485755  # tab gid from your URL


# ----------------------------
# Shapes loader
# ----------------------------
@st.cache_resource
def load_shapes():
    """
    Download TIGER/Line ZCTA shapefile (if needed), unzip, and load into GeoDataFrame.
    Cached across reruns.
    """
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
    """
    Load town scores from a public Google Sheet (CSV export),
    normalize headers, auto-detect Zip and Overall Score columns,
    and return a cleaned DataFrame.
    """
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    # Read as strings to preserve leading zeros in ZIPs
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

    # Bulletproof ZIP normalization (avoid turning blanks into '00000')
    z = (
        df["Zip"].astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)  # digits only
    )
    z = z.replace("", pd.NA)           # blanks -> NA
    z = z.str.zfill(5).str[-5:]        # pad, then keep last 5 (ZIP+4 safe)

    df["Zip"] = z

    # Keep only real 5-digit ZIPs, and drop 00000
    df = df[df["Zip"].notna()]
    df = df[df["Zip"].str.match(r"^\d{5}$", na=False)]
    df = df[df["Zip"] != "00000"]

    return df


# ----------------------------
# Main execution (keep debug)
# ----------------------------
try:
    df = load_scores(SHEET_ID, gid=GID)
    zip_gdf = load_shapes()
except Exception as e:
    st.exception(e)
    st.stop()

# Optional speed filter (recommended for first run)
st.sidebar.header("Map Scope")
scope = st.sidebar.selectbox("Scope", ["All US (slow)", "Massachusetts (fast)"], index=1)
if scope == "Massachusetts (fast)":
    zip_gdf = zip_gdf[zip_gdf["Zip"].str.startswith("0")]

# Debug AFTER scope filter (so it matches what you'll merge)
# st.write("DEBUG df type:", type(df))
# st.write("DEBUG df columns:", list(df.columns))
# st.write("DEBUG df rows:", len(df))
# st.write("DEBUG df Zip sample:", df["Zip"].head(20).tolist())
# st.write("DEBUG unique df zips:", int(df["Zip"].nunique()))

# st.write("DEBUG zip_gdf type:", type(zip_gdf))
# st.write("DEBUG zip_gdf columns:", list(zip_gdf.columns))
# st.write("DEBUG zip_gdf rows:", len(zip_gdf))
# st.write("DEBUG zip_gdf Zip sample:", zip_gdf["Zip"].head(20).tolist())
# st.write("DEBUG unique shape zips:", int(zip_gdf["Zip"].nunique()))

overlap = set(df["Zip"].dropna()) & set(zip_gdf["Zip"].dropna())
st.write("DEBUG overlap ZIPs:", len(overlap))
if overlap:
    st.write("DEBUG overlap examples:", list(sorted(overlap))[:20])

# Convert likely numeric columns to numeric (keeps Town/Zip as strings)
for c in df.columns:
    if c in ["Town", "Zip"]:
        continue
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Columns that should NEVER be considered for variance sliders
exclude_cols = {"Town", "Zip", "Overall Score"}

# cv = compute_relative_variance_cv(df)

# # Drop excluded columns if present
# cv = cv.drop(labels=[c for c in cv.index if c in exclude_cols], errors="ignore")

# # Clean infinities / NaNs
# cv = cv.replace([float("inf"), -float("inf")], pd.NA).dropna()

# # Take top 2 remaining
# top2 = cv.index[:2].tolist()

# st.write("DEBUG CV top entries:", cv.head(10))
# st.write("DEBUG top2:", top2)
cv = compute_relative_variance_cv(df)
top2 = cv.index[:2].tolist()

st.write("DEBUG CV top entries:", cv.head(10))
st.write("DEBUG top2:", top2)
st.write("DEBUG non-null counts:", df.notna().sum().sort_values(ascending=False).head(30))

# Sliders
st.sidebar.header("Weights (top-2 variance only)")

columns = []
weights = {}

if len(top2) >= 1:
    w_1 = st.sidebar.slider(top2[0], 0.0, 1.0, 0.50, 0.01)
    columns.append(top2[0])
    weights[top2[0]] = w_1

if len(top2) >= 2:
    w_2 = st.sidebar.slider(top2[1], 0.0, 1.0, 0.50, 0.01)
    columns.append(top2[1])
    weights[top2[1]] = w_2

if len(top2) < 2:
    st.warning(
        "Not enough numeric columns to create two variance sliders. "
        "Composite score will use available columns only."
    )

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