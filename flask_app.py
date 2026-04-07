import json
import os
import urllib.request
import zipfile
import math
import statistics
import time
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, render_template_string, redirect, url_for, session, Response

from src.geo_utils import load_zip_shapes
from src.composite_score import compute_composite_score
from src.heatmap import create_zip_heatmap


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "sheets.json"
INTAKE_PATH = APP_DIR / "out" / "intake_profile.json"
MOVING_FROM_PATH = APP_DIR / "data" / "moving_from.csv"
MOVING_TO_PATH = APP_DIR / "data" / "moving_to.csv"
HOME_SALES_PATH = APP_DIR / "data" / "Home Sales Data - previous year.csv"
PREBUILT_ZCTA_GEOJSON_PATH = APP_DIR / "data" / "processed" / "ma_zcta_simplified.geojson"

ALPHA_BASE = 0.3
SCALE = 100.0
SIGMA_EPSILON = 1e-6
TARGET_SPREAD = 10.0

FEATURES = [
    "Education",
    "Healthcare & Fitness",
    "Commute/Transit Score",
    "Accessibility",
    "Culture/Entertainment",
]
RATING_FIELDS = [
    ("education_rating", "Education", "How would you rate the local schools in your current town?"),
    ("healthcare_fitness_rating", "Healthcare & Fitness", "How would you rate healthcare and fitness options in your current town?"),
    ("commute_transit_rating", "Commute/Transit Score", "How would you rate the commute/transit in your current town?"),
    ("accessibility_rating", "Accessibility", "How would you rate accessibility in your current town?"),
    ("culture_entertainment_rating", "Culture/Entertainment", "How would you rate culture and entertainment in your current town?"),
]
RATING_OPTIONS = [
    (1, "Very Low"),
    (2, "Low"),
    (3, "Medium"),
    (4, "High"),
    (5, "Very High"),
]
FEATURE_FIELDS = {
    "Education": "education",
    "Healthcare & Fitness": "healthcare_fitness",
    "Commute/Transit Score": "commute_transit",
    "Accessibility": "accessibility",
    "Culture/Entertainment": "culture_entertainment",
}

SYSTEM_PROMPT = """
You are a friendly housing search assistant. Your job is to ask short, one-at-a-time questions
that help build a structured profile for ranking neighborhoods and ZIP codes.

Rules:
- Ask only ONE question per turn.
- Keep it short and conversational.
- The FIRST assistant message should be a short introduction and then ask the first question.
- The first question must ask where the user lives now (current town/city).
- If the user asks an unrelated question, answer it briefly (1 sentence max), then continue the intake by asking the next question.
- When you can infer an answer from the user's last message, update the profile.
- If a field is already filled, don't ask about it again.
- Return your result as strict JSON ONLY with keys: assistant_message, updated_profile, is_complete.
- updated_profile must be an object with the current filled fields.
- is_complete should be true when all required fields are filled.

Required fields:
- current_town (string)
- education_rating (integer, 1-10)
- healthcare_fitness_rating (integer, 1-10)
- commute_transit_rating (integer, 1-10)
- accessibility_rating (integer, 1-10)
- culture_entertainment_rating (integer, 1-10)
""".strip()

REQUIRED_FIELDS = [
    "current_town",
    "education_rating",
    "healthcare_fitness_rating",
    "commute_transit_rating",
    "accessibility_rating",
    "culture_entertainment_rating",
]

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
PERSIST_INTAKE_TO_DISK = os.getenv("PERSIST_INTAKE_TO_DISK", "false").lower() == "true"


def get_llm_provider() -> str:
    if LLM_PROVIDER in {"openai", "ollama"}:
        return LLM_PROVIDER
    if OPENAI_API_KEY:
        return "openai"
    return "ollama"

def to_numeric_loose(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA, "NA": pd.NA})
    x = x.str.replace(r"[\$,]", "", regex=True)
    x = x.str.replace("%", "", regex=False)
    x = x.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def canon(col: str) -> str:
    s = str(col).strip().lower()
    for ch in ["(", ")", "%", "$", ",", "/", "-", "_"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def normalize_zip_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    raw = raw.mask(raw.str.upper().eq("NOT RECOGNIZED"), pd.NA)
    z = (
        raw
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
    )
    z = z.replace("", pd.NA)
    return z.str.zfill(5).str[-5:]


def standardize_score_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

    rename_map = {}
    for c in df.columns:
        cc = canon(c)
        if cc in {"town city", "town"}:
            rename_map[c] = "Town"
        elif cc == "zip":
            rename_map[c] = "Zip"
        elif cc == "overall score" or ("overall" in cc and "score" in cc):
            rename_map[c] = "Overall Score"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Zip" not in df.columns:
        raise ValueError(f"Missing ZIP column. Found: {list(df.columns)}")

    df["Zip"] = normalize_zip_series(df["Zip"])
    df = df[df["Zip"].notna()]
    df = df[df["Zip"].str.match(r"^\d{5}$", na=False)]
    df = df[df["Zip"] != "00000"]

    for c in df.columns:
        if c in {"Town", "Zip", "State", "County", "Overall Score"}:
            continue
        df[c] = to_numeric_loose(df[c])

    return df


@lru_cache(maxsize=1)
def load_moving_to_scores() -> pd.DataFrame:
    if not MOVING_TO_PATH.exists():
        raise FileNotFoundError(f"Missing {MOVING_TO_PATH}")
    df = pd.read_csv(MOVING_TO_PATH)
    return standardize_score_df(df)


@lru_cache(maxsize=1)
def load_moving_from_truth() -> Dict[str, Dict[str, Any]]:
    if not MOVING_FROM_PATH.exists():
        raise FileNotFoundError(f"Missing {MOVING_FROM_PATH}")
    df = standardize_score_df(pd.read_csv(MOVING_FROM_PATH))
    truth: Dict[str, Dict[str, Any]] = {}
    if df.empty:
        return truth

    for _, row in df.iterrows():
        row_truth = {
            feature: float(row[feature])
            for feature in FEATURES
            if feature in df.columns and pd.notna(row.get(feature))
        }
        if not row_truth:
            continue

        town_name = str(row.get("Town", "")).strip()
        state_name = str(row.get("State", "")).strip()
        if town_name:
            truth[town_name] = row_truth
            if state_name:
                truth[f"{town_name}, {state_name}"] = row_truth
    return truth


@lru_cache(maxsize=1)
def load_moving_from_town_options() -> list[str]:
    if not MOVING_FROM_PATH.exists():
        return []
    df = standardize_score_df(pd.read_csv(MOVING_FROM_PATH))
    if "Town" not in df.columns:
        return []
    towns = (
        df["Town"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted([town for town in towns.unique().tolist() if town])


@lru_cache(maxsize=1)
def load_synthetic_homes() -> pd.DataFrame:
    if not SYNTHETIC_HOMES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SYNTHETIC_HOMES_PATH}. Run scripts/generate_synthetic_homes.py first."
        )
    df = pd.read_csv(SYNTHETIC_HOMES_PATH)
    if "zip" in df.columns:
        df["zip"] = normalize_zip_series(df["zip"])
    if "town" in df.columns:
        df["town_norm"] = df["town"].astype(str).map(normalize_town)
    return df


def extract_zip_from_text(text: str) -> Optional[str]:
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if len(digits) >= 5:
        return digits[:5]
    return None


def select_synthetic_home(address: str, current_town: str) -> Optional[Dict[str, Any]]:
    df = load_synthetic_homes()
    if df.empty:
        return None

    subset = pd.DataFrame()
    address_zip = extract_zip_from_text(address)
    if address_zip and "zip" in df.columns:
        subset = df[df["zip"] == address_zip]

    if subset.empty and current_town:
        town_norm = normalize_town(current_town)
        if "town_norm" in df.columns:
            subset = df[df["town_norm"] == town_norm]

    if subset.empty:
        return None

    digest = hashlib.sha256(f"{current_town}|{address}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(subset)
    row = subset.iloc[idx]
    return {
        "town": row.get("town"),
        "zip": row.get("zip"),
        "home_price": float(row.get("home_price")),
        "price_per_sqft": float(row.get("price_per_sqft")),
        "interior_sqft": float(row.get("interior_sqft")),
        "lot_size_sqft": float(row.get("lot_size_sqft")),
        "bedrooms": int(row.get("bedrooms")),
        "bathrooms": int(row.get("bathrooms")),
    }


@lru_cache(maxsize=1)
def load_home_sales() -> pd.DataFrame:
    if not HOME_SALES_PATH.exists():
        raise FileNotFoundError(f"Missing {HOME_SALES_PATH}")

    df = pd.read_csv(HOME_SALES_PATH)
    df["zip"] = normalize_zip_series(df["ZIP OR POSTAL CODE"])
    df["city_norm"] = df["CITY"].astype(str).map(normalize_town)
    numeric_cols = {
        "PRICE": "price",
        "BEDS": "beds",
        "BATHS": "baths",
        "SQUARE FEET": "square_feet",
        "LOT SIZE": "lot_size",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "$/SQUARE FEET": "price_per_sqft",
    }
    for source_col, target_col in numeric_cols.items():
        df[target_col] = pd.to_numeric(df[source_col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude", "price"])
    return df


@lru_cache(maxsize=1)
def compute_zip_pricing_metrics() -> pd.DataFrame:
    eps = 1e-6
    df = load_home_sales().copy()
    zip_col = "zip"
    price_col = "price"

    def compute_group(g: pd.DataFrame) -> pd.Series:
        prices = g[price_col].dropna()

        if len(prices) < 5:
            return pd.Series(
                {
                    "sale_count": len(prices),
                    "price_q1": np.nan,
                    "price_median": np.nan,
                    "price_mean": np.nan,
                    "price_skew_direction": np.nan,
                    "affordability_skew_index": np.nan,
                }
            )

        q1 = prices.quantile(0.25)
        median = prices.median()
        mean = prices.mean()
        skew_direction = mean - median
        denom = mean - median
        if abs(denom) < eps:
            asi = np.nan
        else:
            asi = (median - q1) / denom

        return pd.Series(
            {
                "sale_count": len(prices),
                "price_q1": q1,
                "price_median": median,
                "price_mean": mean,
                "price_skew_direction": skew_direction,
                "affordability_skew_index": asi,
            }
        )

    rows = []
    for zip_value, group in df.groupby(zip_col):
        metrics = compute_group(group)
        row = {"zip": zip_value}
        row.update(metrics.to_dict())
        rows.append(row)
    result = pd.DataFrame(rows)

    valid_q1 = result["price_q1"].dropna()
    q1_split = float(valid_q1.median()) if not valid_q1.empty else np.nan

    def classify(row: pd.Series) -> pd.Series:
        if pd.isna(row["price_skew_direction"]) or pd.isna(row["price_q1"]) or pd.isna(row["price_median"]) or pd.isna(row["price_mean"]):
            return pd.Series(
                {
                    "market_type": "insufficient_data",
                    "market_signal": "Insufficient sales data",
                    "market_interpretation": "There are not enough recent sales here to estimate a reliable entry point into the local housing market.",
                }
            )

        q1_is_high = row["price_q1"] >= q1_split if not pd.isna(q1_split) else False
        median = max(abs(float(row["price_median"])), eps)
        skew_ratio = float(row["price_mean"] - row["price_median"]) / median

        if skew_ratio > 0.05:
            if q1_is_high:
                return pd.Series(
                    {
                        "market_type": "high_q1_right_skew",
                        "market_signal": "High entry point, luxury pressure",
                        "market_interpretation": "Even the lower end of this ZIP is expensive, and higher-end sales are pushing the average above the typical price.",
                    }
                )
            return pd.Series(
                {
                    "market_type": "low_q1_right_skew",
                    "market_signal": "Lower entry point, wide price spread",
                    "market_interpretation": "This ZIP still has lower-cost entry points, but higher-end sales pull the average above the typical price, suggesting a mixed market.",
                }
            )

        if skew_ratio < -0.05:
            return pd.Series(
                {
                    "market_type": "left_skew_distressed",
                    "market_signal": "Low-end stress in the market",
                    "market_interpretation": "The average price falls below the typical price, which can happen when distressed lower-end sales pull the market downward.",
                }
            )

        return pd.Series(
            {
                "market_type": "stable_symmetric",
                "market_signal": "Stable entry point band",
                "market_interpretation": "Entry-level pricing and typical pricing are relatively consistent here, suggesting a more uniform and predictable market.",
            }
        )

    classified = result.apply(classify, axis=1)
    result = pd.concat([result, classified], axis=1)
    result["zip"] = normalize_zip_series(result["zip"])
    for col in ["price_q1", "price_median", "price_mean", "price_skew_direction", "affordability_skew_index"]:
        result[col] = result[col].round(2)
    return result


def score_home_match(home_row: pd.Series, synthetic_home: Dict[str, Any]) -> float:
    # Relative-distance score; lower is better.
    components = []
    comparisons = [
        ("price", "home_price"),
        ("beds", "bedrooms"),
        ("baths", "bathrooms"),
        ("square_feet", "interior_sqft"),
        ("lot_size", "lot_size_sqft"),
    ]
    for sale_key, synth_key in comparisons:
        sale_val = home_row.get(sale_key)
        synth_val = synthetic_home.get(synth_key)
        if pd.isna(sale_val) or synth_val in (None, 0):
            continue
        denom = max(abs(float(synth_val)), 1.0)
        components.append(abs(float(sale_val) - float(synth_val)) / denom)

    if not components:
        return float("inf")
    return float(sum(components) / len(components))


def find_best_home_matches(synthetic_home: Optional[Dict[str, Any]], limit: int = 5) -> list[Dict[str, Any]]:
    if not synthetic_home:
        return []

    df = load_home_sales()
    subset = pd.DataFrame()

    synth_zip = synthetic_home.get("zip")
    if synth_zip:
        subset = df[df["zip"] == synth_zip]

    if subset.empty and synthetic_home.get("town"):
        subset = df[df["city_norm"] == normalize_town(str(synthetic_home["town"]))]

    if subset.empty:
        subset = df

    ranked = subset.copy()
    ranked["match_score"] = ranked.apply(score_home_match, axis=1, synthetic_home=synthetic_home)
    ranked = ranked.replace([np.inf, -np.inf], np.nan).dropna(subset=["match_score"])
    ranked = ranked.sort_values("match_score").head(limit)

    results = []
    for _, row in ranked.iterrows():
        results.append(
            {
                "address": row.get("ADDRESS"),
                "city": row.get("CITY"),
                "zip": row.get("zip"),
                "price": float(row.get("price")) if pd.notna(row.get("price")) else None,
                "beds": float(row.get("beds")) if pd.notna(row.get("beds")) else None,
                "baths": float(row.get("baths")) if pd.notna(row.get("baths")) else None,
                "square_feet": float(row.get("square_feet")) if pd.notna(row.get("square_feet")) else None,
                "latitude": float(row.get("latitude")),
                "longitude": float(row.get("longitude")),
                "match_score": float(row.get("match_score")),
                "url": row.get("URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)"),
            }
        )
    return results


def ensure_zcta_shapes() -> Path:
    url = "https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/tl_2024_us_zcta520.zip"
    cache_root = Path(os.getenv("ZCTA_CACHE_DIR", APP_DIR / "data" / "zcta_cache"))
    extract_dir = cache_root
    shp_path = extract_dir / "tl_2024_us_zcta520.shp"

    if not shp_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_path = extract_dir / "tl_2024_us_zcta520.zip"
        urllib.request.urlretrieve(url, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    return shp_path


def ensure_zip_geometry() -> Path:
    prebuilt_path = Path(os.getenv("PREBUILT_ZCTA_GEOJSON", PREBUILT_ZCTA_GEOJSON_PATH))
    if prebuilt_path.exists():
        return prebuilt_path
    return ensure_zcta_shapes()


@lru_cache(maxsize=2)
def load_cached_zip_geometry(path_str: str):
    return load_zip_shapes(path_str)


@lru_cache(maxsize=1)
def load_base_map_geometry() -> tuple[Any, tuple[float, float]]:
    geometry_path = ensure_zip_geometry()
    zip_gdf = load_cached_zip_geometry(str(geometry_path)).copy()
    df = load_moving_to_scores().copy()

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns in moving_to.csv: {', '.join(missing_features)}")

    df["Zip"] = df["Zip"].astype(str).str.zfill(5)
    zip_gdf["Zip"] = zip_gdf["Zip"].astype(str).str.zfill(5)

    base = zip_gdf.merge(df, on="Zip", how="inner")
    if base.empty:
        raise ValueError(
            f"No overlapping ZIP geometries found between {geometry_path.name} and moving_to.csv"
        )

    keep_cols = ["Zip", "geometry", *FEATURES]
    base = base[[col for col in keep_cols if col in base.columns]].copy()

    minx, miny, maxx, maxy = base.total_bounds
    center = ((miny + maxy) / 2, (minx + maxx) / 2)
    return base, center


@lru_cache(maxsize=1)
def load_zip_touching_map() -> Dict[str, list[str]]:
    base_gdf, _ = load_base_map_geometry()
    zip_map: Dict[str, list[str]] = {}
    if base_gdf.empty:
        return zip_map

    geometries = base_gdf.set_index("Zip")["geometry"]
    buffered = {zip_code: geom.buffer(1e-6) for zip_code, geom in geometries.items()}
    for zip_code, geom in geometries.items():
        neighbors = [zip_code]
        for other_zip, other_geom in geometries.items():
            if other_zip == zip_code:
                continue
            if geom.touches(other_geom) or buffered[zip_code].intersects(buffered[other_zip]):
                neighbors.append(other_zip)
        zip_map[zip_code] = sorted(set(neighbors))
    return zip_map


def expand_selected_zips(seed_zip: str) -> list[str]:
    if not seed_zip:
        return []
    return load_zip_touching_map().get(str(seed_zip).zfill(5), [])


def expand_multiple_selected_zips(seed_zips: list[str]) -> list[str]:
    expanded: set[str] = set()
    for seed_zip in seed_zips:
        expanded.update(expand_selected_zips(seed_zip))
    return sorted(expanded)


def filter_gdf_to_selected_zips(gdf, selected_zips: Optional[list[str]]):
    if not selected_zips:
        return gdf
    normalized = {str(zip_code).zfill(5) for zip_code in selected_zips}
    filtered = gdf[gdf["Zip"].astype(str).str.zfill(5).isin(normalized)].copy()
    return filtered if not filtered.empty else gdf


def normalize_town(s: str) -> str:
    return " ".join(s.strip().lower().replace(",", " ").split())


def match_town(current_town: str) -> Optional[str]:
    if not current_town:
        return None
    ground_truth_towns = load_moving_from_truth()
    norm = normalize_town(current_town)
    for town in ground_truth_towns.keys():
        if normalize_town(town) == norm:
            return town
    for town in ground_truth_towns.keys():
        if normalize_town(town) in norm or norm in normalize_town(town):
            return town
    return None


def load_intake() -> Dict[str, Any]:
    if not INTAKE_PATH.exists():
        return {}
    try:
        return json.loads(INTAKE_PATH.read_text())
    except Exception:
        return {}


def bucket_slider(value: float) -> str:
    if value < 0.20:
        return "xlow"
    if value < 0.40:
        return "low"
    if value < 0.60:
        return "medium"
    if value < 0.80:
        return "high"
    return "xhigh"


def bucket_delta(value: float) -> str:
    if value <= -15:
        return "xlow"
    if value <= -5:
        return "low"
    if value < 5:
        return "medium"
    if value < 15:
        return "high"
    return "xhigh"


def bucket_relative_weight(value: float, values: list[float]) -> str:
    if not values:
        return "medium"
    ordered = sorted(values)
    if len(ordered) == 1:
        return "medium"
    q1 = np.quantile(ordered, 0.2)
    q2 = np.quantile(ordered, 0.4)
    q3 = np.quantile(ordered, 0.6)
    q4 = np.quantile(ordered, 0.8)
    if value <= q1:
        return "xlow"
    if value <= q2:
        return "low"
    if value <= q3:
        return "medium"
    if value <= q4:
        return "high"
    return "xhigh"


def explain_weight_change(
    feature: str,
    slider: Optional[float],
    delta: Optional[float],
    final_weight: Optional[float],
    all_final_weights: list[float],
) -> str:
    if slider is None or delta is None or final_weight is None:
        return "We used your slider value directly because there was not enough hometown data to adjust this feature."

    slider_bucket = bucket_slider(float(slider))
    delta_bucket = bucket_delta(float(delta))
    final_bucket = bucket_relative_weight(float(final_weight), all_final_weights)

    slider_text = {
        "xlow": "very low importance",
        "low": "low importance",
        "medium": "medium importance",
        "high": "high importance",
        "xhigh": "very high importance",
    }[slider_bucket]

    delta_text = {
        "xlow": "much lower than the stored hometown score",
        "low": "a bit lower than the stored hometown score",
        "medium": "about the same as the stored hometown score",
        "high": "a bit higher than the stored hometown score",
        "xhigh": "much higher than the stored hometown score",
    }[delta_bucket]

    final_text = {
        "xlow": "one of the weakest final factors",
        "low": "a weaker-than-average final factor",
        "medium": "a middle-of-the-pack final factor",
        "high": "a stronger-than-average final factor",
        "xhigh": "one of the strongest final factors",
    }[final_bucket]

    return (
        f"You started this at {slider_text}, rated your hometown {delta_text}, "
        f"and it finished as {final_text}."
    )


def compute_weights_from_intake(
    intake: Dict[str, Any],
    sliders: Dict[str, float],
    candidate_df_override: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    ground_truth_towns = load_moving_from_truth()
    candidate_df = candidate_df_override.copy() if candidate_df_override is not None else load_moving_to_scores()
    matched = match_town(str(intake.get("current_town", "")))
    if not matched:
        # Fallback: normalize sliders to sum to 1
        total = sum(sliders.values()) or 1.0
        w_var = {k: round(v / total, 6) for k, v in sliders.items()}
        trace_rows = []
        for feature in FEATURES:
            trace_rows.append(
                {
                    "feature": feature,
                    "slider": float(sliders.get(feature, 0.0)),
                    "user_rating": None,
                    "stored_score": None,
                    "difference": None,
                    "after_calibration": round(w_var.get(feature, 0.0), 4),
                    "variance_effect": 1.0,
                    "final_weight": round(w_var.get(feature, 0.0), 4),
                    "explanation": "We used your slider value directly because the app could not match your hometown to the reference data.",
                }
            )
        return {
            "matched_town": None,
            "w_var": w_var,
            "gamma": 1.0,
            "trace_rows": trace_rows,
            "candidate_zip_count": int(len(candidate_df)),
        }

    gt = ground_truth_towns[matched]

    def to_100(v: Any) -> Optional[float]:
        try:
            return float(v) * 20.0
        except Exception:
            return None

    user_scores = {
        "Education": to_100(intake.get("education_rating")),
        "Healthcare & Fitness": to_100(intake.get("healthcare_fitness_rating")),
        "Commute/Transit Score": to_100(intake.get("commute_transit_rating")),
        "Accessibility": to_100(intake.get("accessibility_rating")),
        "Culture/Entertainment": to_100(intake.get("culture_entertainment_rating")),
    }

    deltas = {}
    for k, gt_val in gt.items():
        u = user_scores.get(k)
        deltas[k] = None if u is None else float(u) - float(gt_val)

    w_cal = {}
    w_cal_raw = {}
    for k in FEATURES:
        delta = deltas.get(k)
        if delta is None:
            w_cal[k] = None
            w_cal_raw[k] = None
            continue
        w = sliders[k] * (1 - ALPHA_BASE * (delta / SCALE))
        w_cal_raw[k] = w
        if w < 0:
            w = 0.0
        w_cal[k] = w

    total = sum(v for v in w_cal.values() if v is not None)
    if total > 0:
        for k, v in w_cal.items():
            if v is None:
                continue
            w_cal[k] = v / total

    candidate_rows = candidate_df[FEATURES].dropna()
    sigma = {}
    for feature in FEATURES:
        values = candidate_rows[feature].tolist() if feature in candidate_rows.columns else []
        if len(values) >= 2:
            sigma[feature] = statistics.pstdev(values) + SIGMA_EPSILON
        elif len(values) == 1:
            sigma[feature] = SIGMA_EPSILON
        else:
            sigma[feature] = None

    w_var = {}
    w_var_raw = {}
    for feature, w in w_cal.items():
        s = sigma.get(feature)
        if w is None or s is None:
            w_var[feature] = None
            w_var_raw[feature] = None
        else:
            w_var_raw[feature] = w / (s + SIGMA_EPSILON)
            w_var[feature] = w_var_raw[feature]

    total_var = sum(v for v in w_var.values() if v is not None)
    if total_var > 0:
        for k, v in w_var.items():
            if v is None:
                continue
            w_var[k] = v / total_var

    rel_values = []
    for _, candidate in candidate_rows.iterrows():
        score = 0.0
        for feature, w in w_var.items():
            if w is None:
                continue
            s_z = candidate.get(feature)
            s_home = gt.get(feature)
            if s_z is None or s_home is None:
                continue
            score += w * (float(s_z) - float(s_home))
        rel_values.append(score)

    if rel_values:
        med = statistics.median(rel_values)
        mad = statistics.median([abs(v - med) for v in rel_values])
    else:
        mad = None

    if mad is None:
        gamma = 1.0
    else:
        gamma = TARGET_SPREAD / (mad + SIGMA_EPSILON)
        gamma = max(0.5, min(gamma, 5.0))

    trace_rows = []
    final_weight_values = [float(v) for v in w_var.values() if v is not None]
    for feature in FEATURES:
        user_rating_raw = intake.get(FEATURE_FIELDS[feature])
        if feature == "Education":
            user_rating_raw = intake.get("education_rating")
        elif feature == "Healthcare & Fitness":
            user_rating_raw = intake.get("healthcare_fitness_rating")
        elif feature == "Commute/Transit Score":
            user_rating_raw = intake.get("commute_transit_rating")
        elif feature == "Accessibility":
            user_rating_raw = intake.get("accessibility_rating")
        elif feature == "Culture/Entertainment":
            user_rating_raw = intake.get("culture_entertainment_rating")

        trace_rows.append(
            {
                "feature": feature,
                "slider": round(float(sliders.get(feature, 0.0)), 4),
                "user_rating": user_rating_raw,
                "stored_score": round(float(gt.get(feature)), 2) if gt.get(feature) is not None else None,
                "difference": round(float(deltas.get(feature)), 2) if deltas.get(feature) is not None else None,
                "after_calibration": round(float(w_cal.get(feature)), 4) if w_cal.get(feature) is not None else None,
                "variance_effect": (
                    round(float(w_var.get(feature)) / float(w_cal.get(feature)), 3)
                    if w_var.get(feature) is not None and w_cal.get(feature) not in (None, 0)
                    else None
                ),
                "final_weight": round(float(w_var.get(feature)), 4) if w_var.get(feature) is not None else None,
                "explanation": explain_weight_change(
                    feature,
                    sliders.get(feature),
                    deltas.get(feature),
                    w_var.get(feature),
                    final_weight_values,
                ),
            }
        )

    return {
        "matched_town": matched,
        "w_var": w_var,
        "gamma": gamma,
        "alpha_base": ALPHA_BASE,
        "scale": SCALE,
        "sigma_epsilon": SIGMA_EPSILON,
        "target_spread": TARGET_SPREAD,
        "trace_rows": trace_rows,
        "candidate_zip_count": int(len(candidate_df)),
    }

def missing_fields(profile: Dict[str, Any]) -> list[str]:
    return [f for f in REQUIRED_FIELDS if profile.get(f) in (None, "", [])]

def next_question(missing: list[str]) -> str:
    if not missing:
        return ""
    field = missing[0]
    questions = {
        "current_town": "Where do you live now (current town or city)?",
        "current_address": "What address do you live at right now?",
        "education_rating": "On a scale of 1 to 10, how would you rate the local schools in your current town?",
        "healthcare_fitness_rating": "On a scale of 1 to 10, how would you rate healthcare and fitness options in your current town?",
        "commute_transit_rating": "On a scale of 1 to 10, how would you rate the commute/transit in your current town?",
        "accessibility_rating": "On a scale of 1 to 10, how would you rate accessibility in your current town?",
        "culture_entertainment_rating": "On a scale of 1 to 10, how would you rate culture and entertainment in your current town?",
    }
    return questions.get(field, "Could you tell me a bit more?")

def call_llm(messages: list[Dict[str, str]]) -> str:
    provider = get_llm_provider()
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        payload["model"] = OPENAI_MODEL
    else:
        url = f"{LLM_BASE_URL}/chat/completions"
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        payload["model"] = LLM_MODEL

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        return {
            "assistant_message": text.strip(),
            "updated_profile": {},
            "is_complete": False,
        }

def llm_turn(state: Dict[str, Any]) -> None:
    profile = state.get("profile", {})
    messages = state.get("messages", [])

    payload = {
        "profile": profile,
        "missing_fields": missing_fields(profile),
    }
    messages.append(
        {
            "role": "user",
            "content": "Here is the current profile state:\n" + json.dumps(payload, indent=2),
        }
    )

    raw = call_llm(messages)
    parsed = parse_llm_json(raw)

    messages.append({"role": "assistant", "content": raw})

    updated_profile = parsed.get("updated_profile", {})
    if isinstance(updated_profile, dict):
        profile.update(updated_profile)

    state["profile"] = profile
    state["messages"] = messages
    is_complete = bool(parsed.get("is_complete", False))
    parsed_answer = parsed.get("assistant_message", "")

    # Always drive the intake with our fixed questions to avoid drift.
    missing = missing_fields(profile)
    if not is_complete and missing:
        if missing[0] == "current_town":
            if not state.get("intro_shown", False):
                assistant_message = (
                    "Hi there! I’ll ask a few quick questions to personalize the map. "
                    + next_question(missing)
                )
                state["intro_shown"] = True
            else:
                assistant_message = next_question(missing)
        elif missing[0] == "current_address":
            assistant_message = next_question(missing)
        else:
            # Hybrid mode: allow a brief answer, then continue intake.
            assistant_message = parsed_answer.strip()
            if not assistant_message:
                assistant_message = next_question(missing)
            elif "?" not in assistant_message:
                assistant_message = f"{assistant_message} {next_question(missing)}"
    else:
        assistant_message = parsed_answer

    state["assistant_message"] = assistant_message
    state["is_complete"] = is_complete

def get_chat_state() -> Dict[str, Any]:
    chat_id = session.get("chat_id")
    if chat_id is None:
        chat_id = os.urandom(8).hex()
        session["chat_id"] = chat_id

    if "chat_state" not in session:
        session["chat_state"] = {
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "profile": {},
            "assistant_message": "",
            "is_complete": False,
            "chat_log": [],
            "intro_shown": False,
        }

    return session["chat_state"]


def parse_intake_form(form, existing_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = dict(existing_profile or {})
    profile["current_town"] = form.get("current_town", "").strip()

    for field_name, _, _ in RATING_FIELDS:
        raw = form.get(field_name, "").strip()
        try:
            value = int(raw)
        except Exception:
            value = None
        if value is not None and 1 <= value <= 5:
            profile[field_name] = value
        else:
            profile[field_name] = None

    return profile


def build_map(sliders: Dict[str, float], intake: Optional[Dict[str, Any]] = None) -> str:
    base_gdf, center = load_base_map_geometry()
    merged = base_gdf.copy()
    selected_zips = session.get("selected_zips") or []
    candidate_scope = filter_gdf_to_selected_zips(base_gdf, selected_zips)
    pricing_metrics = compute_zip_pricing_metrics().copy()
    pricing_metrics = pricing_metrics.rename(columns={"zip": "Zip"})
    merged = merged.merge(pricing_metrics, on="Zip", how="left")

    # Compute personalized weights
    intake = intake or {}
    weights_info = compute_weights_from_intake(
        intake,
        sliders,
        candidate_df_override=candidate_scope[FEATURES] if not candidate_scope.empty else base_gdf[FEATURES],
    )
    w_var = weights_info["w_var"]
    gamma = weights_info["gamma"]

    columns = FEATURES
    weights = {k: w_var[k] for k in columns if w_var.get(k) is not None}
    w_sum = sum(weights.values()) if weights else 0.0
    if w_sum <= 0:
        # Fallback to equal weights if calibration zeroed everything.
        weights = {k: 1.0 for k in columns}

    merged["Composite Score"] = compute_composite_score(merged, columns, weights)

    # Apply gamma to map scale
    if gamma is not None:
        merged["Composite Score (Calibrated)"] = merged["Composite Score"] * float(gamma)
        value_col = "Composite Score (Calibrated)"
    else:
        value_col = "Composite Score"

    m = create_zip_heatmap(
        merged,
        value_col,
        center=center,
        zoom=11,
        featured_homes=None,
        selected_zips=selected_zips,
    )
    return m.get_root().render()


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev")


TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>ZIP Heatmap (Personalized)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <style>
    :root {
      --panel-width: 340px;
      --border: #ddd;
      --text: #1f2937;
      --muted: #555;
      --panel-bg: #ffffff;
      --chat-bg: #fafafa;
      --accent-bg: #eef3ff;
      --accent-border: #cbd8ff;
    }
    * { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; margin: 0; color: var(--text); background: #f7f7f7; }
    .layout { display: grid; grid-template-columns: minmax(300px, var(--panel-width)) 1fr; min-height: 100vh; }
    .panel { padding: 16px; border-right: 1px solid var(--border); overflow: auto; background: var(--panel-bg); }
    .panel h2 { margin-top: 0; }
    .intake-card {
      border: 1px solid var(--border);
      padding: 12px;
      background: var(--chat-bg);
      border-radius: 12px;
      margin-bottom: 14px;
    }
    .field {
      margin-bottom: 14px;
    }
    .field label {
      display: block;
      font-weight: bold;
      margin-bottom: 6px;
    }
    .field input[type="text"] {
      width: 100%;
      padding: 12px;
      font-size: 16px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: white;
    }
    .field select {
      width: 100%;
      padding: 12px;
      font-size: 16px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: white;
    }
    .radio-group {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
    }
    .radio-option {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: white;
      padding: 8px;
      text-align: center;
      font-size: 12px;
    }
    .radio-option input {
      margin-bottom: 6px;
    }
    .pill {
      display: inline-block;
      padding: 6px 10px;
      background: var(--accent-bg);
      border: 1px solid var(--accent-border);
      border-radius: 999px;
      margin-right: 6px;
      margin-bottom: 6px;
      font-size: 12px;
    }
    details {
      margin-top: 14px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fcfcfc;
      overflow: hidden;
    }
    summary {
      cursor: pointer;
      padding: 12px 14px;
      font-weight: bold;
      background: #f3f4f6;
    }
    .trace-wrap {
      overflow-x: auto;
      padding: 10px 12px 14px 12px;
    }
    table.trace {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    table.trace th,
    table.trace td {
      text-align: left;
      vertical-align: top;
      padding: 8px;
      border-bottom: 1px solid var(--border);
    }
    table.trace th {
      background: #f9fafb;
      position: sticky;
      top: 0;
    }
    .slider { margin-bottom: 18px; }
    .slider label { display: block; font-weight: bold; margin-bottom: 8px; font-size: 15px; }
    .slider input { width: 100%; min-height: 40px; }
    .map {
      min-height: 100vh;
      background: #e5e7eb;
    }
    .map iframe {
      display: block;
      border: 0;
      width: 100%;
      height: 100%;
      min-height: 100vh;
    }
    .help { font-size: 12px; color: var(--muted); }
    button {
      min-height: 44px;
      padding: 10px 14px;
      font-size: 16px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
    }
    hr { border: 0; border-top: 1px solid var(--border); margin: 20px 0; }
    @media (max-width: 900px) {
      .layout {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto;
      }
      .panel {
        border-right: 0;
        border-bottom: 1px solid var(--border);
        max-height: none;
      }
      .map {
        min-height: 60vh;
      }
      .map iframe {
        min-height: 60vh;
      }
    }
    @media (max-width: 640px) {
      .panel {
        padding: 14px;
      }
      .radio-group {
        grid-template-columns: 1fr;
      }
      .map {
        min-height: 52vh;
      }
      .map iframe {
        min-height: 52vh;
      }
      button {
        width: 100%;
      }
    }
  </style>
  <script>
    function updateValue(id, val) {
      document.getElementById(id).innerText = Number(val).toFixed(2);
    }
    function submitSliders() {
      const form = document.getElementById("slider-form");
      if (form) form.submit();
    }
    window.addEventListener("message", (event) => {
      if (!event.data) return;
      if (event.data.type === "mapZipSelected" && event.data.zip) {
        const field = document.getElementById("selected-zip-input");
        const form = document.getElementById("select-zip-form");
        if (field && form) {
          field.value = event.data.zip;
          form.submit();
        }
      }
    });
  </script>
</head>
<body>
  <div class="layout">
    <div class="panel">
      <h2>Home Intake</h2>
      <div class="intake-card">
        <p class="help">Tell us about where you live now, then we’ll personalize the map from there.</p>
        <form method="POST" action="/intake">
          <div class="field">
            <label for="current_town">Current town or city</label>
            <select id="current_town" name="current_town">
              <option value="">Select a hometown</option>
              {% for town in hometown_options %}
                <option value="{{ town }}" {% if profile.current_town | default('') == town %}selected{% endif %}>{{ town }}</option>
              {% endfor %}
            </select>
          </div>
          {% for field_name, feature_name, question in rating_fields %}
            <div class="field">
              <label>{{ question }}</label>
              <div class="radio-group">
                {% for option_value, option_label in rating_options %}
                  <label class="radio-option">
                    <input
                      type="radio"
                      name="{{ field_name }}"
                      value="{{ option_value }}"
                      {% if profile.get(field_name) == option_value %}checked{% endif %}
                    >
                    <div>{{ option_value }}</div>
                    <div>{{ option_label }}</div>
                  </label>
                {% endfor %}
              </div>
            </div>
          {% endfor %}
          <button type="submit">Apply Intake</button>
        </form>
      </div>
      {% if is_complete %}
        <p class="help">Intake complete. Sliders now apply personalized calibration.</p>
        <div>
          {% if personalization.matched_town %}
            <span class="pill">Matched: {{ personalization.matched_town }}</span>
          {% endif %}
          <span class="pill">Gamma: {{ (personalization.gamma | default(1.0)) | round(3) }}</span>
          <span class="pill">Alpha: {{ (personalization.alpha_base | default(0.3)) | round(3) }}</span>
          <span class="pill">Scale: {{ (personalization.scale | default(100.0)) | round(3) }}</span>
          <span class="pill">Sigma ε: {{ (personalization.sigma_epsilon | default(0.000001)) | round(6) }}</span>
          <span class="pill">Target: {{ (personalization.target_spread | default(10.0)) | round(3) }}</span>
        </div>
        {% if personalization.trace_rows %}
          <details>
            <summary>Why these weights?</summary>
            <div class="trace-wrap">
              <table class="trace">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Slider</th>
                    <th>Your Rating</th>
                    <th>Stored Score</th>
                    <th>Difference</th>
                    <th>After Calibration</th>
                    <th>Variance Effect</th>
                    <th>Final Weight</th>
                    <th>What Happened</th>
                  </tr>
                </thead>
                <tbody>
                  {% for row in personalization.trace_rows %}
                    <tr>
                      <td>{{ row.feature }}</td>
                      <td>{{ "%.2f"|format(row.slider) if row.slider is not none else "—" }}</td>
                      <td>{% if row.user_rating is not none %}{{ row.user_rating }}/5{% else %}—{% endif %}</td>
                      <td>{% if row.stored_score is not none %}{{ "%.0f"|format(row.stored_score) }}/100{% else %}—{% endif %}</td>
                      <td>
                        {% if row.difference is not none %}
                          {% if row.difference > 0 %}+{% endif %}{{ "%.0f"|format(row.difference) }}
                        {% else %}
                          —
                        {% endif %}
                      </td>
                      <td>{{ "%.2f"|format(row.after_calibration) if row.after_calibration is not none else "—" }}</td>
                      <td>{% if row.variance_effect is not none %}{{ "%.2f"|format(row.variance_effect) }}x{% else %}—{% endif %}</td>
                      <td>{{ "%.2f"|format(row.final_weight) if row.final_weight is not none else "—" }}</td>
                      <td>{{ row.explanation }}</td>
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          </details>
        {% endif %}
      {% else %}
        <p class="help">Fill in the town, address, and five ratings to personalize the map.</p>
      {% endif %}

      <hr>
      <h2>Weights</h2>
      <form id="slider-form" method="POST">
        {% for f in features %}
        <div class="slider">
          <label>{{f}}: <span id="v{{loop.index}}">{{sliders[f]|round(2)}}</span></label>
          <input type="range" min="0" max="1" step="0.01" name="{{feature_fields[f]}}" value="{{sliders[f]}}" oninput="updateValue('v{{loop.index}}', this.value)" onchange="submitSliders()">
        </div>
        {% endfor %}
        <button type="submit">Update Map</button>
        <p class="help">Scope: Massachusetts (ZIPs starting with 0)</p>
      </form>
      <form method="POST" action="/reset">
        <button type="submit">Reset Form</button>
      </form>
      <hr>
      <h2>ZIP Selection</h2>
      <p class="help">
        Click a ZIP on the map to select it. The app will automatically include that ZIP and every ZIP that touches it.
      </p>
      {% if selected_seed_zips %}
        <p class="help">
          Selected ZIPs: {{ selected_seed_zips|join(", ") }}<br>
          Included ZIP count: {{ personalization.candidate_zip_count | default(selected_zips|length) }}
        </p>
        <p class="help">
          Included ZIPs: {{ selected_zips|join(", ") }}
        </p>
      {% else %}
        <p class="help">No ZIPs selected yet. Click one on the map to start a local comparison set.</p>
      {% endif %}
      <form id="select-zip-form" method="POST" action="/select-zip">
        <input id="selected-zip-input" type="hidden" name="zip">
        <button type="button" disabled>Select ZIPs</button>
      </form>
      <form method="POST" action="/clear-selection">
        <button type="submit">Clear Selected ZIPs</button>
      </form>
      {% if map_error %}
        <p class="help">Map error: {{ map_error }}</p>
      {% endif %}
    </div>
    <div class="map">
      <iframe src="/map?ts={{ map_ts }}"></iframe>
    </div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    reset_ts = request.args.get("reset_ts")
    sliders = {f: 0.5 for f in FEATURES}
    if request.method == "POST":
        for f in FEATURES:
            try:
                field = FEATURE_FIELDS[f]
                sliders[f] = float(request.form.get(field, sliders[f]))
            except Exception:
                sliders[f] = sliders[f]

    state = get_chat_state()

    if state.get("is_complete"):
        profile = state.get("profile", {})
        selected_scope = filter_gdf_to_selected_zips(load_base_map_geometry()[0], session.get("selected_zips"))
        weights_info = compute_weights_from_intake(
            profile,
            sliders,
            candidate_df_override=selected_scope[FEATURES] if not selected_scope.empty else load_base_map_geometry()[0][FEATURES],
        )
        state["personalization"] = weights_info

    map_error = None
    session["sliders"] = sliders
    # Map is served via /map to avoid srcdoc escaping issues.
    session["chat_state"] = state
    try:
        return render_template_string(
            TEMPLATE,
            features=FEATURES,
            feature_fields=FEATURE_FIELDS,
            rating_fields=RATING_FIELDS,
            rating_options=RATING_OPTIONS,
            hometown_options=load_moving_from_town_options(),
            sliders=sliders,
            profile=state.get("profile", {}),
            is_complete=state.get("is_complete", False),
            personalization=state.get("personalization", {}),
            selected_seed_zips=session.get("selected_seed_zips") or [],
            selected_zips=session.get("selected_zips") or [],
            map_error=map_error,
            map_ts=int(reset_ts) if reset_ts and reset_ts.isdigit() else int(time.time() * 1000),
        )
    except Exception as e:
        return f"<pre>Template render failed: {e}</pre>"


@app.route("/intake", methods=["POST"])
def intake():
    state = get_chat_state()
    profile = parse_intake_form(request.form, state.get("profile"))
    state["profile"] = profile
    state["is_complete"] = len(missing_fields(profile)) == 0

    if state.get("is_complete"):
        state["profile"] = profile
        if PERSIST_INTAKE_TO_DISK:
            INTAKE_PATH.parent.mkdir(parents=True, exist_ok=True)
            INTAKE_PATH.write_text(json.dumps(profile, indent=2))
        selected_scope = filter_gdf_to_selected_zips(load_base_map_geometry()[0], session.get("selected_zips"))
        weights_info = compute_weights_from_intake(
            profile,
            {f: 1.0 for f in FEATURES},
            candidate_df_override=selected_scope[FEATURES] if not selected_scope.empty else load_base_map_geometry()[0][FEATURES],
        )
        state["personalization"] = weights_info
    else:
        state["personalization"] = {}

    session["chat_state"] = state
    return redirect(url_for("index"))


@app.route("/select-zip", methods=["POST"])
def select_zip():
    zip_code = str(request.form.get("zip", "")).strip()
    zip_code = str(zip_code).zfill(5) if zip_code else ""
    existing_seed_zips = [str(z).zfill(5) for z in (session.get("selected_seed_zips") or [])]
    if zip_code and zip_code not in existing_seed_zips:
        existing_seed_zips.append(zip_code)

    selected = expand_multiple_selected_zips(existing_seed_zips)
    if selected:
        session["selected_seed_zips"] = existing_seed_zips
        session["selected_zips"] = selected
    return redirect(url_for("index", reset_ts=int(time.time() * 1000000)))


@app.route("/clear-selection", methods=["POST"])
def clear_selection():
    session.pop("selected_seed_zips", None)
    session.pop("selected_zips", None)
    return redirect(url_for("index", reset_ts=int(time.time() * 1000000)))


@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    if PERSIST_INTAKE_TO_DISK:
        try:
            if INTAKE_PATH.exists():
                INTAKE_PATH.unlink()
        except Exception:
            pass
    return redirect(url_for("index", reset_ts=int(time.time() * 1000000)))


@app.route("/map", methods=["GET"])
def map_view():
    sliders = session.get("sliders") or {f: 0.5 for f in FEATURES}
    chat_state = session.get("chat_state") or {}
    intake = chat_state.get("profile") if chat_state.get("is_complete") else None
    try:
        map_html = build_map(sliders, intake=intake)
        resp = Response(map_html, mimetype="text/html")
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception as e:
        html = f"<p style='padding:12px;'>Map failed to render: {e}</p>"
        resp = Response(html, mimetype="text/html")
        resp.headers["Cache-Control"] = "no-store"
        return resp


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
