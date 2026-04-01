import json
from typing import Dict


FEATURES = [
    "Education",
    "Healthcare & Fitness",
    "Commute/Transit Score",
    "Accessibility",
    "Culture/Entertainment",
]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def low_membership(w: float) -> float:
    if w <= 0.15:
        return 1.0
    if w >= 0.30:
        return 0.0
    return (0.30 - w) / 0.15


def medium_membership(w: float) -> float:
    if w <= 0.15 or w >= 0.45:
        return 0.0
    if w <= 0.30:
        return (w - 0.15) / 0.15
    return (0.45 - w) / 0.15


def high_membership(w: float) -> float:
    if w <= 0.25:
        return 0.0
    if w >= 0.45:
        return 1.0
    return (w - 0.25) / 0.20


def fuzzify_weights(weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    fuzzy = {}
    for feature in FEATURES:
        w = float(weights.get(feature, 0.0))
        fuzzy[feature] = {
            "low": low_membership(w),
            "medium": medium_membership(w),
            "high": high_membership(w),
        }
    return fuzzy


def infer_archetypes(weights: Dict[str, float]) -> Dict[str, float]:
    f = fuzzify_weights(weights)

    family_oriented = max(
        min(f["Education"]["high"], f["Accessibility"]["high"]),
        min(f["Education"]["high"], f["Healthcare & Fitness"]["medium"]),
    )

    urban_lifestyle = max(
        min(f["Culture/Entertainment"]["high"], f["Commute/Transit Score"]["medium"]),
        min(f["Culture/Entertainment"]["high"], f["Accessibility"]["medium"]),
    )

    mobility_sensitive = max(
        min(f["Commute/Transit Score"]["high"], f["Accessibility"]["high"]),
        min(f["Commute/Transit Score"]["high"], f["Culture/Entertainment"]["medium"]),
    )

    wellness_focused = max(
        min(f["Healthcare & Fitness"]["high"], f["Accessibility"]["medium"]),
        min(f["Healthcare & Fitness"]["high"], f["Education"]["medium"]),
    )

    culture_seeking = max(
        f["Culture/Entertainment"]["high"],
        min(f["Culture/Entertainment"]["high"], f["Accessibility"]["medium"]),
    )

    practical_balancer = min(
        f["Education"]["medium"],
        f["Healthcare & Fitness"]["medium"],
        f["Commute/Transit Score"]["medium"],
        f["Accessibility"]["medium"],
    )

    return {
        "family_oriented": clamp(family_oriented),
        "urban_lifestyle": clamp(urban_lifestyle),
        "mobility_sensitive": clamp(mobility_sensitive),
        "wellness_focused": clamp(wellness_focused),
        "culture_seeking": clamp(culture_seeking),
        "practical_balancer": clamp(practical_balancer),
    }


def main() -> None:
    # Replace this with the app's final normalized weights when you want to test.
    example_weights = {
        "Education": 0.28,
        "Healthcare & Fitness": 0.18,
        "Commute/Transit Score": 0.24,
        "Accessibility": 0.16,
        "Culture/Entertainment": 0.14,
    }

    fuzzy = fuzzify_weights(example_weights)
    archetypes = infer_archetypes(example_weights)

    print("Input weights:")
    print(json.dumps(example_weights, indent=2))
    print("\nFuzzy memberships:")
    print(json.dumps(fuzzy, indent=2))
    print("\nPreference archetypes:")
    print(json.dumps(archetypes, indent=2))


if __name__ == "__main__":
    main()
