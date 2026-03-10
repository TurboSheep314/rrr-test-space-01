import json
import os
from typing import Dict, Any, Optional

import pandas as pd
import requests

# ---------------------------
# Config
# ---------------------------
ZIP = "02465"
ACS_YEAR = 2023  # adjust if needed
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

# Zillow ZHVI ZIP dataset (set to the direct CSV download URL from Zillow Research)
ZILLOW_ZHVI_URL = os.getenv("ZILLOW_ZHVI_URL", "")

# EJScreen CSV (set to a direct CSV from the EJSCREEN download site/FTP)
EJSCREEN_CSV_URL = os.getenv("EJSCREEN_CSV_URL", "")

# BLS API key (optional)
BLS_API_KEY = os.getenv("BLS_API_KEY", "")


# ---------------------------
# Helpers
# ---------------------------

def census_get_zcta(zip_code: str, year: int, variables: list[str], api_key: str) -> Dict[str, Any]:
    if not api_key:
        raise ValueError("Missing CENSUS_API_KEY env var")

    base = f"https://api.census.gov/data/{year}/acs/acs5"
    get_vars = ",".join(variables)
    params = {
        "get": get_vars,
        "for": f"zip code tabulation area:{zip_code}",
        "key": api_key,
    }
    resp = requests.get(base, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    header = data[0]
    row = data[1]
    return dict(zip(header, row))


def fetch_acs_metrics(zip_code: str, year: int, api_key: str) -> Dict[str, Any]:
    # Property tax: B25103_001E (median real estate taxes paid)
    # Income: B19013_001E (median household income)
    # Education: B15003 (bachelor's+ as share of total 25+)
    # Commute: B08301 (work-from-home share of workers)
    # Unemployment: B23025 (unemployed / labor force)
    vars_needed = [
        "B25103_001E",
        "B19013_001E",
        "B15003_001E",
        "B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E",
        "B08301_001E", "B08301_021E",
        "B23025_002E", "B23025_003E",
    ]
    row = census_get_zcta(zip_code, year, vars_needed, api_key)

    # Parse numbers (the API returns strings)
    def to_num(x: Any) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    total_edu = to_num(row.get("B15003_001E"))
    bachelors_plus = sum(filter(None, [
        to_num(row.get("B15003_022E")),
        to_num(row.get("B15003_023E")),
        to_num(row.get("B15003_024E")),
        to_num(row.get("B15003_025E")),
    ]))

    workers_total = to_num(row.get("B08301_001E"))
    wfh = to_num(row.get("B08301_021E"))

    labor_force = to_num(row.get("B23025_002E"))
    unemployed = to_num(row.get("B23025_003E"))

    return {
        "property_tax_median": to_num(row.get("B25103_001E")),
        "median_household_income": to_num(row.get("B19013_001E")),
        "education_bachelors_plus_share": (bachelors_plus / total_edu) if total_edu else None,
        "commute_wfh_share": (wfh / workers_total) if workers_total else None,
        "unemployment_rate_acs": (unemployed / labor_force) if labor_force else None,
        "raw": row,
    }


def fetch_zillow_zhvi(zip_code: str, csv_url: str) -> Dict[str, Any]:
    if not csv_url:
        return {"zillow_zhvi": None, "note": "Set ZILLOW_ZHVI_URL to a Zillow ZHVI ZIP CSV"}

    df = pd.read_csv(csv_url)
    # Zillow ZIP datasets typically use RegionName as ZIP
    zip_rows = df[df.get("RegionName").astype(str) == str(zip_code)]
    if zip_rows.empty:
        return {"zillow_zhvi": None, "note": "ZIP not found in Zillow CSV"}

    # Last column is usually the most recent month
    last_col = df.columns[-1]
    value = zip_rows.iloc[0][last_col]
    return {"zillow_zhvi": float(value) if pd.notna(value) else None, "zillow_zhvi_date": last_col}


def fetch_ejscreen(zip_code: str, csv_url: str) -> Dict[str, Any]:
    if not csv_url:
        return {"ejscreen": None, "note": "Set EJSCREEN_CSV_URL to a CSV from EJSCREEN downloads"}

    df = pd.read_csv(csv_url, dtype=str)
    # Try common ZIP/ZCTA column names
    zip_cols = [c for c in df.columns if c.lower() in {"zip", "zipcode", "zcta", "zcta5", "zcta5ce"}]
    if not zip_cols:
        return {"ejscreen": None, "note": "Could not detect ZIP/ZCTA column in EJSCREEN CSV"}

    col = zip_cols[0]
    row = df[df[col].astype(str).str.zfill(5) == str(zip_code)].head(1)
    if row.empty:
        return {"ejscreen": None, "note": "ZIP not found in EJSCREEN CSV"}

    # Return a small subset of air quality indicators if present
    candidate_cols = [
        "PM25", "O3", "NATA_RESP", "NATA_CANCER", "RSEI_AIR",
        "AIRPT", "AIRPT_S", "AIRPTI",
    ]
    found = {c: row.iloc[0][c] for c in candidate_cols if c in row.columns}
    return {"ejscreen": found or row.iloc[0].to_dict()}


def fetch_bls_unemployment_by_county(zip_code: str, bls_key: str) -> Dict[str, Any]:
    # BLS LAUS is county-level. You need ZIP -> county FIPS first.
    # This is a placeholder to wire in once you decide a crosswalk source.
    return {
        "bls_unemployment": None,
        "note": "BLS LAUS is county-level; add a ZIP->county crosswalk and then query BLS API",
    }


def main() -> None:
    result: Dict[str, Any] = {"zip": ZIP}

    # ACS metrics (property tax, education, income, commute, unemployment)
    try:
        result.update(fetch_acs_metrics(ZIP, ACS_YEAR, CENSUS_API_KEY))
    except Exception as e:
        result["acs_error"] = str(e)

    # Zillow ZHVI (home price)
    result.update(fetch_zillow_zhvi(ZIP, ZILLOW_ZHVI_URL))

    # EJScreen (air quality)
    result.update(fetch_ejscreen(ZIP, EJSCREEN_CSV_URL))

    # BLS (unemployment, county-level)
    result.update(fetch_bls_unemployment_by_county(ZIP, BLS_API_KEY))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
