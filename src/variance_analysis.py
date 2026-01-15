import numpy as np
import pandas as pd

def compute_relative_variance_cv(
    df: pd.DataFrame,
    exclude_cols=("Town", "Zip", "Overall Score"),
    min_nonnull: int = 5,
    drop_constant: bool = True,
) -> pd.Series:
    """
    Compute coefficient of variation (CV = std / |mean|) for numeric-like columns.

    Robust behavior:
    - does NOT slice based on 'Overall Score' position
    - does NOT drop rows (uses skipna=True)
    - excludes non-feature columns (Town/Zip/Overall Score by default)
    - filters columns with too few numeric values
    - optionally drops constant columns (std==0)

    Returns:
        pd.Series sorted descending by CV, indexed by column name.
    """
    # Pick candidate feature columns
    candidates = [c for c in df.columns if c not in set(exclude_cols)]
    if not candidates:
        return pd.Series(dtype=float)

    # Keep only numeric columns (assumes caller already coerced with to_numeric_loose)
    num = df[candidates].select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        return pd.Series(dtype=float)

    # Drop columns with too few valid values
    nonnull = num.notna().sum()
    num = num.loc[:, nonnull >= min_nonnull]
    if num.shape[1] == 0:
        return pd.Series(dtype=float)

    means = num.mean(skipna=True).abs()
    stds = num.std(skipna=True)

    # Avoid divide-by-zero / nonsense CV when mean is 0
    means = means.replace(0, np.nan)

    cv = stds / means
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()

    if drop_constant:
        cv = cv[cv > 0]

    return cv.sort_values(ascending=False)