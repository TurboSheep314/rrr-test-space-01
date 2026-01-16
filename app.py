# app.py

import os
import urllib.request
import zipfile

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.geo_utils import load_zip_shapes
from src.variance_analysis import compute_relative_variance_cv
from src.composite_score import compute_composite_score
from src.heatmap import create_zip_heatmap

import json
from pathlib import Path
#----------------------------
# Helper Functions
#----------------------------

def to_numeric_loose(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()

    # treat common missing tokens as NA
    x = x.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA, "NA": pd.NA})

    # strip common formatting
    x = x.str.replace(r"[\$,]", "", regex=True)  # "$1,234" -> "1234"
    x = x.str.replace("%", "", regex=False)      # "12.4%"  -> "12.4"

    # keep only numeric characters (and decimal/minus)
    x = x.str.replace(r"[^0-9\.\-]", "", regex=True)

    return pd.to_numeric(x, errors="coerce")

# ----------------------------
# Streamlit page
# ----------------------------


CONFIG_PATH = Path(__file__).parent / "sheets.json"

if not CONFIG_PATH.exists():
    st.error("Missing sheets.json (put it in the same folder as app.py)")
    st.stop()

try:
    SHEETS_CFG = json.loads(CONFIG_PATH.read_text())
except Exception as e:
    st.error("Could not parse sheets.json (check JSON formatting)")
    st.exception(e)
    st.stop()

# Build dropdown labels from comparison_city (and keep a stable mapping back to dataset keys)
options = []
for dataset_key, cfg in SHEETS_CFG.items():
    city = cfg.get("comparison_city", dataset_key)
    options.append((city, dataset_key))

# Sort by city label (nice UX)
options.sort(key=lambda x: x[0].lower())

st.sidebar.header("Dataset")
selected_city = st.sidebar.selectbox(
    "Comparison city",
    [city for city, _ in options],
    index=0,
    key="comparison_city"
)

# Map back to the dataset config
selected_dataset_key = dict(options)[selected_city]
cfg = SHEETS_CFG[selected_dataset_key]

comparison_city = cfg.get("comparison_city", selected_city)
st.title(f"ZIP Heatmap — From {comparison_city}")

# Keep these ready for later wiring (doesn't change your loading yet)
#SHEET_ID = cfg.get("spreadsheet_id")
#GID = int(cfg.get("gid", 0))

st.set_page_config(layout="wide")
st.title("ZIP Heatmap (Overall + Top-2 Variance Sliders)")

SHEET_ID = "1aZmL78kZZgcKa4anOHkKYC-TRQyaHrJm"
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

overlap = set(df["Zip"].dropna()) & set(zip_gdf["Zip"].dropna())
# ***** DEBUG *****
#st.write("DEBUG overlap ZIPs:", len(overlap))

#if overlap:
#    st.write("DEBUG overlap examples:", list(sorted(overlap))[:20])


# Columns that should NEVER be considered for variance sliders
exclude_cols = {"Town", "Zip", "Overall Score"}

# Convert likely numeric columns to numeric (keeps Town/Zip as strings)
for c in df.columns:
    if c in exclude_cols:
        continue
    df[c] = to_numeric_loose(df[c])

# DEBUG: show which columns actually became numeric
candidates = [c for c in df.columns if c not in exclude_cols]
nonnull = df[candidates].notna().sum().sort_values(ascending=False)

# ***** DEBUG *****
#st.write("DEBUG numeric non-null counts (top 25):", nonnull.head(25))

# Only keep columns with enough numeric values (you have ~17 overlapping zips)
keep = nonnull[nonnull >= 5].index.tolist()
#st.write("DEBUG numeric columns kept:", keep)

#st.write("DEBUG CV top entries:", cv.head(10))
#st.write("DEBUG top2:", top2)
#st.write("DEBUG non-null counts:", df.notna().sum().sort_values(ascending=False).head(30))
if len(keep) == 0:
    cv = pd.Series(dtype=float)
    top2 = []
else:
    cv = compute_relative_variance_cv(df)
    top2 = cv.index[:2].tolist()

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

# --- Filter out ZIPs that have no geometry before merging ---

# Convert both to strings in a clean, consistent format
df["Zip"] = df["Zip"].astype(str).str.zfill(5)
zip_gdf["Zip"] = zip_gdf["Zip"].astype(str).str.zfill(5)

# Find only the ZIP codes that exist in both
valid_zips = set(df["Zip"]).intersection(zip_gdf["Zip"])

# Filter your score dataframe to only keep those valid ZIPs
df = df[df["Zip"].isin(valid_zips)].copy()

# Optional: show how many were dropped for transparency
st.write(f"Dropped {len(set(df['Zip']) ^ valid_zips)} ZIPs that have no matching ZCTA geometry.")

# Merge shapes with scores
merged = zip_gdf.merge(df, on="Zip", how="inner")
# Which ZIPs matched?
matched_zips = set(merged["Zip"].astype(str))

# Which ZIPs *did not* match?
all_data_zips = set(df["Zip"].astype(str))
missing_zips = sorted(list(all_data_zips - matched_zips))

st.write("🔍 Unmatched ZIP codes:", missing_zips)
st.write("Total missing ZIPs:", len(missing_zips))

# ***** DEBUG *****
# # Debug + stop if empty
# if merged.empty:
#     st.error("No ZIP geometries matched your score data (merged is empty).")
#     st.write("Sample ZIPs in data:", df["Zip"].head(20).tolist())
#     st.write("Sample ZIPs in shapes:", zip_gdf["Zip"].head(20).tolist())
#     st.write("Unique ZIPs in data:", int(df["Zip"].nunique()))
#     st.write("Unique ZIPs in shapes:", int(zip_gdf["Zip"].nunique()))
#     st.stop()

# st.write("Using top-2 CV columns:", top2)
# st.write("Merged ZIPs:", len(merged))

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

#st.components.v1.html(m._repr_html_(), height=720)
# --- Persist user’s map view between reruns ---

# Initialize session state map center + zoom (only on first load)
if "map_center" not in st.session_state:
    st.session_state["map_center"] = (center_lat, center_lon)

if "map_zoom" not in st.session_state:
    st.session_state["map_zoom"] = 9 if scope.endswith("(fast)") else 4

# Create the folium map using session state
m = create_zip_heatmap(
    merged,
    "Composite Score",
    center=st.session_state["map_center"],
    zoom=st.session_state["map_zoom"],
)

# Render map and capture user interactions
map_state = st_folium(
    m,
    height=720,
    returned_objects=["center", "zoom"],
    key="zip_map",
)

# If user moved the map (pan/zoom), update session state
if map_state is not None:
    if map_state.get("center"):
        st.session_state["map_center"] = (
            map_state["center"]["lat"],
            map_state["center"]["lng"],
        )
    if map_state.get("zoom") is not None:
        st.session_state["map_zoom"] = map_state["zoom"]

# Reset button
if st.sidebar.button("Reset map view"):
    st.session_state["map_center"] = (center_lat, center_lon)
    st.session_state["map_zoom"] = 9 if scope.endswith("(fast)") else 4

