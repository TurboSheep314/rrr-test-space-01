import json
import os
import sys
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# ---------------------------
# Minimal LLM chat prototype
# ---------------------------
# This script expects an OpenAI-compatible chat endpoint.
# Defaults are set for Ollama's OpenAI-compatible server:
#   ollama serve
#   ollama pull llama3.1
# ---------------------------

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3.1")
API_KEY = os.getenv("LLM_API_KEY", "")  # optional for local servers
OUTPUT_PATH = Path(os.getenv("LLM_OUTPUT_PATH", "out/intake_profile.json"))
ALPHA_BASE = 0.3
SCALE = 100.0
SIGMA_EPSILON = 1e-6
TARGET_SPREAD = 10.0

SYSTEM_PROMPT = """
You are a friendly housing search assistant. Your job is to ask short, one-at-a-time questions
that help build a structured profile for ranking neighborhoods and ZIP codes.

Rules:
- Ask only ONE question per turn.
- Keep it short and conversational.
- DO NOT repeat the scores to the person.
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

If the user says they are unsure, ask a clarifying question or suggest a reasonable default and confirm.
""".strip()

REQUIRED_FIELDS = [
    "current_town",
    "education_rating",
    "healthcare_fitness_rating",
    "commute_transit_rating",
    "accessibility_rating",
    "culture_entertainment_rating",
]

# Ground truth ratings for comparison (1-100 scale as provided in source).
GROUND_TRUTH_TOWNS = {
    "Memphis": {
        "Education": 43,
        "Healthcare & Fitness": 24,
        "Commute/Transit Score": 66,
        "Accessibility": 37,
        "Culture/Entertainment": 49,
    },
    "Clarence Center, NY": {
        "Education": 75,
        "Healthcare & Fitness": 65,
        "Commute/Transit Score": 57,
        "Accessibility": 24,
        "Culture/Entertainment": 14,
    },
}


def call_llm(messages: List[Dict[str, str]]) -> str:
    url = f"{DEFAULT_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.3,
        # Some OpenAI-compatible servers support forcing JSON output.
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to recover if the model wrapped JSON in text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        # Fallback: wrap raw text into the expected JSON shape
        return {
            "assistant_message": text.strip(),
            "updated_profile": {},
            "is_complete": False,
        }


def missing_fields(profile: Dict[str, Any]) -> List[str]:
    return [f for f in REQUIRED_FIELDS if profile.get(f) in (None, "", [])]

def build_summary(profile: Dict[str, Any]) -> str:
    return (
        "Thanks! Here is what I captured:\n"
        f"- Current town: {profile.get('current_town')}\n"
        f"- Education rating (1-10): {profile.get('education_rating')}\n"
        f"- Healthcare + Fitness rating (1-10): {profile.get('healthcare_fitness_rating')}\n"
        f"- Commute/Transit rating (1-10): {profile.get('commute_transit_rating')}\n"
        f"- Accessibility rating (1-10): {profile.get('accessibility_rating')}\n"
        f"- Culture/Entertainment rating (1-10): {profile.get('culture_entertainment_rating')}\n"
        "If anything looks off, just tell me and I can correct it."
    )

def validate_profile(profile: Dict[str, Any]) -> Optional[str]:
    def in_range(v: Any) -> bool:
        try:
            iv = int(v)
            return 1 <= iv <= 10
        except Exception:
            return False

    required_ratings = [
        "education_rating",
        "healthcare_fitness_rating",
        "commute_transit_rating",
        "accessibility_rating",
        "culture_entertainment_rating",
    ]

    for key in required_ratings:
        if key in profile and profile[key] is not None and not in_range(profile[key]):
            return f"Please give a whole number from 1 to 10 for {key.replace('_', ' ')}."

    return None

def normalize_town(s: str) -> str:
    return " ".join(s.strip().lower().replace(",", " ").split())

def match_town(current_town: str) -> Optional[str]:
    if not current_town:
        return None
    norm = normalize_town(current_town)
    for town in GROUND_TRUTH_TOWNS.keys():
        if normalize_town(town) == norm:
            return town
    # Fuzzy contains match
    for town in GROUND_TRUTH_TOWNS.keys():
        if normalize_town(town) in norm or norm in normalize_town(town):
            return town
    return None

def compute_personalization(profile: Dict[str, Any]) -> Dict[str, Any]:
    matched = match_town(str(profile.get("current_town", "")))
    if not matched:
        return {
            "matched_town": None,
            "ground_truth": None,
            "deltas": None,
            "note": "Current town did not match known ground-truth towns.",
        }

    gt = GROUND_TRUTH_TOWNS[matched]

    # Convert user 1-10 ratings to 0-100 scale for comparison
    def to_100(v: Any) -> Optional[float]:
        try:
            return float(v) * 10.0
        except Exception:
            return None

    user_scores = {
        "Education": to_100(profile.get("education_rating")),
        "Healthcare & Fitness": to_100(profile.get("healthcare_fitness_rating")),
        "Commute/Transit Score": to_100(profile.get("commute_transit_rating")),
        "Accessibility": to_100(profile.get("accessibility_rating")),
        "Culture/Entertainment": to_100(profile.get("culture_entertainment_rating")),
    }

    deltas = {}
    for k, gt_val in gt.items():
        u = user_scores.get(k)
        deltas[k] = None if u is None else round(u - float(gt_val), 2)

    # Calibration bias per feature is delta[i] = R_home[i] - S_home[i]
    # where R_home is the user's rating (0-100) and S_home is ground truth (0-100).
    calibration_bias = deltas

    # Calibration rule per feature:
    # w_cal[i] = slider[i] * (1 - alpha_base * (delta[i] / scale))
    # If no slider is provided, default to 1.0.
    slider_defaults = {
        "Education": float(profile.get("slider_education", 1.0)),
        "Healthcare & Fitness": float(profile.get("slider_healthcare_fitness", 1.0)),
        "Commute/Transit Score": float(profile.get("slider_commute_transit", 1.0)),
        "Accessibility": float(profile.get("slider_accessibility", 1.0)),
        "Culture/Entertainment": float(profile.get("slider_culture_entertainment", 1.0)),
    }

    w_cal = {}
    for k, delta in calibration_bias.items():
        if delta is None:
            w_cal[k] = None
            continue
        w = slider_defaults[k] * (1 - ALPHA_BASE * (delta / SCALE))
        if w < 0:
            w = 0.0
        w_cal[k] = round(w, 4)

    # Renormalize calibrated weights to sum to 1 (ignoring None values).
    total = sum(v for v in w_cal.values() if v is not None)
    if total > 0:
        for k, v in w_cal.items():
            if v is None:
                continue
            w_cal[k] = round(v / total, 4)

    # Feature spread across candidate towns
    candidate_towns = list(GROUND_TRUTH_TOWNS.values())
    sigma = {}
    if candidate_towns:
        for feature in gt.keys():
            values = [t[feature] for t in candidate_towns if feature in t]
            if len(values) >= 2:
                sigma[feature] = statistics.pstdev(values) + SIGMA_EPSILON
            elif len(values) == 1:
                sigma[feature] = SIGMA_EPSILON
            else:
                sigma[feature] = None

    # Boost low-variance features
    w_var = {}
    for feature, w in w_cal.items():
        s = sigma.get(feature)
        if w is None or s is None:
            w_var[feature] = None
        else:
            w_var[feature] = round(w / (s + SIGMA_EPSILON), 6)

    # Renormalize variance-boosted weights to sum to 1 (ignoring None values).
    total_var = sum(v for v in w_var.values() if v is not None)
    if total_var > 0:
        for k, v in w_var.items():
            if v is None:
                continue
            w_var[k] = round(v / total_var, 6)

    # Contrast score vs home for each candidate town:
    # Rel[z] = sum_i w_var[i] * (S[z][i] - S_home[i])
    rel_contrast = {}
    for town_name, town_scores in GROUND_TRUTH_TOWNS.items():
        score = 0.0
        for feature, w in w_var.items():
            if w is None:
                continue
            s_z = town_scores.get(feature)
            s_home = gt.get(feature)
            if s_z is None or s_home is None:
                continue
            score += w * (float(s_z) - float(s_home))
        rel_contrast[town_name] = round(score, 6)

    # Spread of relative scores using median absolute deviation (MAD)
    rel_values = list(rel_contrast.values())
    if rel_values:
        med = statistics.median(rel_values)
        mad = statistics.median([abs(v - med) for v in rel_values])
    else:
        mad = None

    # Adaptive gain
    if mad is None:
        gamma = None
    else:
        gamma = TARGET_SPREAD / (mad + SIGMA_EPSILON)
        gamma = max(0.5, min(gamma, 5.0))

    return {
        "matched_town": matched,
        "ground_truth": gt,
        "user_scores_0_100": user_scores,
        "deltas_user_minus_truth": deltas,
        "calibration_bias": calibration_bias,
        "w_cal": w_cal,
        "alpha_base": ALPHA_BASE,
        "scale": SCALE,
        "sigma": sigma,
        "sigma_epsilon": SIGMA_EPSILON,
        "w_var": w_var,
        "rel_contrast": rel_contrast,
        "rel_mad": mad,
        "target_spread": TARGET_SPREAD,
        "gamma": gamma,
    }


def main() -> None:
    print("Housing Chat Prototype (LLM)\n")
    print("Type 'exit' to quit.\n")

    profile: Dict[str, Any] = {}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        # Ask the LLM for the next question
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

        try:
            raw = call_llm(messages)
        except Exception as e:
            print("LLM call failed:", e)
            print("Make sure a local OpenAI-compatible server is running, e.g. Ollama.")
            sys.exit(1)

        try:
            parsed = parse_llm_json(raw)
        except Exception as e:
            print("Failed to parse LLM JSON:", e)
            print("Raw response:", raw)
            sys.exit(1)

        assistant_message = parsed.get("assistant_message", "")
        updated_profile = parsed.get("updated_profile", {})
        is_complete = parsed.get("is_complete", False)

        # Update messages and profile
        messages.append({"role": "assistant", "content": raw})
        if isinstance(updated_profile, dict):
            profile.update(updated_profile)

        # Enforce numeric validation (1-10) when ratings appear
        validation_error = validate_profile(profile)
        if validation_error:
            assistant_message = validation_error
            is_complete = False

        print("Assistant:", assistant_message)
        if is_complete:
            personalization = compute_personalization(profile)
            profile["personalization"] = personalization
            print("\n" + build_summary(profile))
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(profile, indent=2))
            print(f"\nSaved: {OUTPUT_PATH}")
            print("\nProfile JSON:")
            print(json.dumps(profile, indent=2))
            break

        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user_input})


if __name__ == "__main__":
    main()
