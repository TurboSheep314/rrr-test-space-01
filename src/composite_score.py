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

    # Normalize weights to sum to 1 (guard against zeros)
    w_sum = sum(weights.values())
    if w_sum == 0:
        raise ValueError("Sum of weights is zero; cannot compute composite score.")
    norm_weights = {k: v / w_sum for k, v in weights.items()}

    composite = sum(
        norm_weights[col] * z[col]
        for col in columns
    )

    return composite


def compute_home_relative_score(
    df: pd.DataFrame,
    columns: list[str],
    weights: dict[str, float],
    home_vector: dict[str, float],
) -> pd.Series:
    """
    Compute a weighted relative score for each row against a home-town baseline.
    A score of 0 means the row matches the home baseline on the weighted features.
    Positive values mean above-home on balance; negative values mean below-home.
    """

    w_sum = sum(weights.values())
    if w_sum == 0:
        raise ValueError("Sum of weights is zero; cannot compute relative score.")
    norm_weights = {k: v / w_sum for k, v in weights.items()}

    score = pd.Series(0.0, index=df.index)
    for col in columns:
        home_val = home_vector.get(col)
        if home_val is None:
            continue
        score = score + norm_weights[col] * (df[col] - float(home_val))

    return score


def _trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (c - x) / (c - b) if c != b else 0.0


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a <= x < b:
        return (x - a) / (b - a) if b != a else 0.0
    return (d - x) / (d - c) if d != c else 0.0


def fuzzy_feature_favorability(diff: float, slider: float) -> float:
    """
    Fuzzy preference-aware favorability for a single feature.
    diff: candidate feature minus home feature, in raw 0-100 score space.
    slider: user preference intensity on [0, 1].

    Returns a value in roughly [-1, 1]:
      0   -> about as good as home for a neutral preference
      > 0 -> favorable improvement over home
      < 0 -> worse than desired, including "same as home" when preference is high
    """

    slider = float(np.clip(slider, 0.0, 1.0))
    diff = float(np.clip(diff, -100.0, 100.0))

    low = _trapmf(slider, 0.0, 0.0, 0.25, 0.50)
    medium = _trimf(slider, 0.25, 0.50, 0.75)
    high = _trapmf(slider, 0.50, 0.75, 1.0, 1.0)

    much_worse = _trapmf(diff, -100.0, -100.0, -20.0, -8.0)
    slightly_worse = _trimf(diff, -15.0, -5.0, 0.0)
    same = _trimf(diff, -6.0, 0.0, 6.0)
    slightly_better = _trimf(diff, 0.0, 5.0, 15.0)
    much_better = _trapmf(diff, 8.0, 20.0, 100.0, 100.0)

    rules = [
        (much_worse, -1.0),
        (min(slightly_worse, low), -0.35),
        (min(slightly_worse, medium), -0.60),
        (min(slightly_worse, high), -0.90),
        (min(same, low), 0.15),
        (min(same, medium), 0.00),
        (min(same, high), -0.45),
        (min(slightly_better, low), 0.35),
        (min(slightly_better, medium), 0.60),
        (min(slightly_better, high), 0.90),
        (much_better, 1.0),
    ]

    numerator = sum(weight * output for weight, output in rules if weight > 0)
    denominator = sum(weight for weight, _ in rules if weight > 0)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_fuzzy_home_relative_score(
    df: pd.DataFrame,
    columns: list[str],
    weights: dict[str, float],
    sliders: dict[str, float],
    home_vector: dict[str, float],
) -> pd.Series:
    """
    Compute a weighted fuzzy favorability score relative to home.
    Unlike a pure difference model, matching home can become less favorable
    when the user raises preference for improvement on a feature.
    """
    w_sum = sum(weights.values())
    if w_sum == 0:
        raise ValueError("Sum of weights is zero; cannot compute fuzzy relative score.")
    norm_weights = {k: v / w_sum for k, v in weights.items()}

    score = pd.Series(0.0, index=df.index, dtype=float)
    for col in columns:
        home_val = home_vector.get(col)
        if home_val is None:
            continue
        slider_val = float(sliders.get(col, norm_weights.get(col, 0.5)))
        fuzzy_vals = df[col].apply(
            lambda value: fuzzy_feature_favorability(float(value) - float(home_val), slider_val)
            if pd.notna(value)
            else 0.0
        )
        score = score + norm_weights[col] * fuzzy_vals

    # Expand to a more legible map range while keeping 0 as the neutral center.
    return score * 100.0
