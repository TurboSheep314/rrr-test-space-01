import pandas as pd
import numpy as np

def compute_composite_score(
    df: pd.DataFrame,
    columns: list[str],
    weights: dict[str, float]
) -> pd.Series:
    """
    Z-normalize selected columns and compute weighted composite score.
    """

    z = (df[columns] - df[columns].mean()) / df[columns].std()

    # Normalize weights to sum to 1
    w_sum = sum(weights.values())
    norm_weights = {k: v / w_sum for k, v in weights.items()}

    composite = sum(
        norm_weights[col] * z[col]
        for col in columns
    )

    return composite