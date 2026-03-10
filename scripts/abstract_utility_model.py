import json
import argparse
from collections import defaultdict


def load_weights(json_path):
    """Load weight profiles from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["profiles"]


def compute_abstract_weights(profiles, priors=None):
    """
    Compute abstract expected utility weights:
    w_bar_i = sum_k (pi_k * w_ik)
    """

    profile_names = list(profiles.keys())

    # Default: equal priors
    if priors is None:
        priors = {name: 1.0 / len(profile_names) for name in profile_names}

    # Normalize priors just in case
    total_prior = sum(priors.values())
    priors = {k: v / total_prior for k, v in priors.items()}

    aggregated = defaultdict(float)

    for profile_name, weights in profiles.items():
        pi_k = priors.get(profile_name, 0.0)

        for feature, weight in weights.items():
            aggregated[feature] += pi_k * weight

    return dict(aggregated)


def print_top_features(weights, top_n=3):
    """Print top N features by abstract importance."""
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    print("\nAbstract Utility Weights:")
    print("-" * 40)
    for feature, value in sorted_items:
        print(f"{feature:35s} {value:.4f}")

    print("\nTop Predictors:")
    print("-" * 40)
    for feature, value in sorted_items[:top_n]:
        print(f"{feature:35s} {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Abstract Utility Model")
    parser.add_argument(
        "--json",
        type=str,
        default="weights.json",
        help="Path to JSON weights file"
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="Prior probability for baseline profile"
    )
    parser.add_argument(
        "--work",
        type=float,
        default=None,
        help="Prior probability for work-move profile"
    )
    parser.add_argument(
        "--intangibles",
        type=float,
        default=None,
        help="Prior probability for intangibles profile"
    )

    args = parser.parse_args()

    profiles = load_weights(args.json)

    # If no custom priors provided → equal weighting
    if args.baseline is None and args.work is None and args.intangibles is None:
        priors = None
    else:
        priors = {
            "naive_ordering": args.baseline or 0.0,
            "naive_ordering_work_move_price_fixed": args.work or 0.0,
            "naive_ordering_intangibles": args.intangibles or 0.0,
        }

    abstract_weights = compute_abstract_weights(profiles, priors)
    print_top_features(abstract_weights)


if __name__ == "__main__":
    main()