from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "moving_from.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_homes_from.csv"


def clean_currency(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).replace(r"[\$,]", "", regex=True),
        errors="coerce",
    )


def clean_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).replace("%", "", regex=False),
        errors="coerce",
    )


def load_source_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["home_price"] = clean_currency(df["Home Price"])
    df["price_per_sqft"] = clean_currency(df["Price Per Sqft"])
    df["property_tax_rate_pct"] = clean_percent(df["Property Tax"])

    df["interior_sqft"] = df["home_price"] / df["price_per_sqft"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Town/City", "home_price", "price_per_sqft", "interior_sqft"])
    df = df[df["price_per_sqft"] > 0]
    df = df[df["interior_sqft"] > 0]

    return df


def generate_synthetic_homes(row: pd.Series, n: int = 500) -> pd.DataFrame:
    price_sqft_mean = float(row["price_per_sqft"])
    sqft_mean = float(row["interior_sqft"])

    price_sqft = np.random.normal(price_sqft_mean, price_sqft_mean * 0.15, n)
    price_sqft = np.clip(price_sqft, 1, None)

    interior_sqft = np.random.lognormal(np.log(sqft_mean), 0.25, n)
    lot_size = np.random.lognormal(np.log(8000), 0.6, n)

    bedrooms = np.random.poisson(3, n) + 1
    bathrooms = np.random.poisson(2, n) + 1

    home_price = interior_sqft * price_sqft

    return pd.DataFrame(
        {
            "town": row["Town/City"],
            "zip": str(row["Zip"]).strip(),
            "home_price": home_price,
            "price_per_sqft": price_sqft,
            "interior_sqft": interior_sqft,
            "lot_size_sqft": lot_size,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
        }
    )


def main() -> None:
    np.random.seed(42)

    df = load_source_data(INPUT_PATH)
    synthetic_frames = [generate_synthetic_homes(row) for _, row in df.iterrows()]
    synthetic_data = pd.concat(synthetic_frames, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(synthetic_data):,} synthetic homes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
