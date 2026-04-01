#!/usr/bin/env python3
"""Build a lightweight Massachusetts-only ZCTA GeoJSON for deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MOVING_TO = APP_DIR / "data" / "moving_to.csv"
DEFAULT_OUTPUT = APP_DIR / "data" / "processed" / "ma_zcta_simplified.geojson"
SOURCE_CANDIDATES = [
    APP_DIR / "data" / "zcta_cache" / "tl_2024_us_zcta520.shp",
    APP_DIR / "data" / "zcta" / "tl_2024_us_zcta520" / "tl_2024_us_zcta520.shp",
]


def normalize_zip_series(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    raw = raw.mask(raw.str.upper().eq("NOT RECOGNIZED"), pd.NA)
    cleaned = (
        raw
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
    )
    cleaned = cleaned.replace("", pd.NA)
    return cleaned.str.zfill(5).str[-5:]


def detect_zip_column(gdf: gpd.GeoDataFrame) -> str:
    for candidate in ["ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID", "Zip"]:
        if candidate in gdf.columns:
            return candidate
    raise ValueError(f"No ZIP/ZCTA column found. Columns: {list(gdf.columns)}")


def resolve_source_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Source geometry not found: {path}")
        return path

    for candidate in SOURCE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a source ZCTA shapefile. Looked in: "
        + ", ".join(str(path) for path in SOURCE_CANDIDATES)
    )


def load_target_zips(moving_to_path: Path) -> list[str]:
    if not moving_to_path.exists():
        raise FileNotFoundError(f"Missing moving_to dataset: {moving_to_path}")

    df = pd.read_csv(moving_to_path)
    if "Zip" not in df.columns:
        raise ValueError(f"moving_to.csv is missing Zip column. Found: {list(df.columns)}")

    zips = normalize_zip_series(df["Zip"]).dropna()
    zips = zips[zips != "00000"]
    return sorted(zips.unique().tolist())


def build_geojson(source_path: Path, moving_to_path: Path, output_path: Path, simplify_tolerance_m: float) -> Path:
    target_zips = load_target_zips(moving_to_path)
    if not target_zips:
        raise ValueError("No valid ZIP codes found in moving_to.csv")

    gdf = gpd.read_file(source_path)
    zip_col = detect_zip_column(gdf)
    gdf["Zip"] = gdf[zip_col].astype(str).str.zfill(5)

    ma_gdf = gdf[gdf["Zip"].isin(target_zips)].copy()
    if ma_gdf.empty:
        raise ValueError("No matching ZIP geometries found for the ZIPs in moving_to.csv")

    ma_gdf = ma_gdf[["Zip", "geometry"]].drop_duplicates(subset=["Zip"]).to_crs(epsg=3857)

    # Simplify in meters for more predictable visual results than degree-based simplification.
    ma_gdf["geometry"] = ma_gdf.geometry.simplify(simplify_tolerance_m, preserve_topology=True)
    ma_gdf = ma_gdf.to_crs(epsg=4326)
    ma_gdf["geometry"] = ma_gdf.geometry.make_valid()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ma_gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Massachusetts-only simplified ZCTA GeoJSON from the full Census shapefile."
    )
    parser.add_argument(
        "--source",
        help="Path to the source ZCTA shapefile. If omitted, common local cache locations are checked.",
    )
    parser.add_argument(
        "--moving-to",
        default=str(DEFAULT_MOVING_TO),
        help="Path to moving_to.csv used to determine which Massachusetts ZIPs to keep.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output GeoJSON path.",
    )
    parser.add_argument(
        "--simplify-tolerance-m",
        type=float,
        default=120.0,
        help="Geometry simplification tolerance in meters. Higher is smaller/faster but less detailed.",
    )
    args = parser.parse_args()

    source_path = resolve_source_path(args.source)
    moving_to_path = Path(args.moving_to)
    output_path = Path(args.output)

    result = build_geojson(
        source_path=source_path,
        moving_to_path=moving_to_path,
        output_path=output_path,
        simplify_tolerance_m=args.simplify_tolerance_m,
    )

    size_kb = result.stat().st_size / 1024
    print(f"Wrote {result} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
