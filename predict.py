"""
predict.py
──────────
Inference module: load a trained model artefact and predict on one transaction.
Also provides a CLI for batch or single-row prediction from the terminal.

Usage:
    python predict.py --model "Random Forest"
    python predict.py --model "Logistic Regression" --amount 85000
"""

import os
import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

from preprocessing import prepare_single_input

log = logging.getLogger("predict")

MODELS_DIR = "models"

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest"      : "random_forest.pkl",
    "Gradient Boosting"  : "gradient_boosting.pkl",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

_artefact_cache: dict = {}   # module-level cache — avoids re-loading on every call


def load_artefact(model_name: str) -> dict:
    """
    Load a model artefact from disk. Results are cached in memory so
    repeated calls do not hit the filesystem.

    Parameters
    ──────────
    model_name : one of "Logistic Regression", "Random Forest", "Gradient Boosting"

    Returns
    ───────
    dict with keys: model, scaler, encoders, feature_cols, threshold, uses_scaling
    """
    if model_name in _artefact_cache:
        return _artefact_cache[model_name]

    filename = MODEL_FILENAMES.get(model_name)
    if filename is None:
        raise ValueError(f"Unknown model: '{model_name}'. "
                         f"Choose from: {list(MODEL_FILENAMES.keys())}")

    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: '{path}'\n"
            "Run  python train.py  first."
        )

    artefact = joblib.load(path)
    _artefact_cache[model_name] = artefact
    log.info("Loaded '%s' from %s", model_name, path)
    return artefact


def get_available_models() -> list[str]:
    """Return the list of model names that have been trained (pkl files exist)."""
    available = []
    for name, fname in MODEL_FILENAMES.items():
        if os.path.exists(os.path.join(MODELS_DIR, fname)):
            available.append(name)
    return available


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(inputs: dict, model_name: str = "Random Forest") -> dict:
    """
    Run fraud inference on a single transaction.

    Parameters
    ──────────
    inputs     : dict with raw feature values (same keys as the Streamlit form)
    model_name : which trained model to use

    Returns
    ───────
    {
      prediction        : int   — 0 = Normal, 1 = Fraud
      fraud_probability : float — probability of fraud (0.0 – 1.0)
      safe_probability  : float — probability of normal (0.0 – 1.0)
      result_label      : str   — "FRAUD" or "SAFE"
      confidence        : float — probability of the predicted class
      model_name        : str
      threshold         : float
    }
    """
    artefact = load_artefact(model_name)

    model        = artefact["model"]
    scaler       = artefact["scaler"]
    encoders     = artefact["encoders"]
    feature_cols = artefact["feature_cols"]
    threshold    = artefact.get("threshold", 0.35)
    uses_scaling = artefact.get("uses_scaling", False)

    # Prepare input array
    X = prepare_single_input(inputs, encoders, scaler, feature_cols)

    # If this model was trained on unscaled data, invert the scaling
    # (prepare_single_input always scales; for tree models we need raw values)
    if not uses_scaling:
        X = scaler.inverse_transform(X)

    # Predict
    proba       = model.predict_proba(X)[0]   # [prob_normal, prob_fraud]
    fraud_prob  = float(proba[1])
    prediction  = int(fraud_prob >= threshold)
    result      = "FRAUD" if prediction == 1 else "SAFE"
    confidence  = fraud_prob if prediction == 1 else float(proba[0])

    return {
        "prediction"        : prediction,
        "fraud_probability" : round(fraud_prob, 6),
        "safe_probability"  : round(float(proba[0]), 6),
        "result_label"      : result,
        "confidence"        : round(confidence, 4),
        "model_name"        : model_name,
        "threshold"         : threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RISK RATING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_risk_rating(fraud_probability: float) -> tuple[str, str]:
    """
    Map a fraud probability to a human-readable risk level and colour.

    Returns (label, hex_colour)
    """
    p = fraud_probability
    if p < 0.20:
        return "LOW",    "#16a34a"   # green
    elif p < 0.40:
        return "MEDIUM", "#ca8a04"   # amber
    elif p < 0.65:
        return "HIGH",   "#dc2626"   # red
    else:
        return "CRITICAL", "#7f1d1d" # dark red


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD TRAINING REPORT
# ══════════════════════════════════════════════════════════════════════════════

def load_training_report(report_path: str = "training_report.json") -> dict:
    """Load the JSON metrics report produced by train.py."""
    if not os.path.exists(report_path):
        return {}
    with open(report_path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI  (for terminal testing)
# ══════════════════════════════════════════════════════════════════════════════

def _demo_inputs(amount: float = 500.0) -> dict:
    """Return a sample transaction dict for CLI testing."""
    return {
        "transaction_hour"              : 14,
        "day_of_week"                   : 1,
        "is_weekend"                    : 0,
        "transaction_amount"            : amount,
        "sender_balance_before"         : 10000.0,
        "receiver_balance_before"       : 5000.0,
        "transaction_type"              : "P2P",
        "device_type"                   : "ANDROID",
        "location_cluster"              : "NORTH_INDIA",
        "upi_app"                       : "GPay",
        "is_new_device"                 : 0,
        "is_night_transaction"          : 0,
        "transaction_velocity_last_1hr" : 2,
        "transaction_velocity_last_24hr": 5,
        "account_age_days"              : 365,
        "failed_login_attempts"         : 0,
    }


def _fraud_inputs(amount: float = 85000.0) -> dict:
    """Return a fraud-pattern transaction dict for CLI testing."""
    return {
        "transaction_hour"              : 2,
        "day_of_week"                   : 6,
        "is_weekend"                    : 1,
        "transaction_amount"            : amount,
        "sender_balance_before"         : 90000.0,
        "receiver_balance_before"       : 1000.0,
        "transaction_type"              : "P2P",
        "device_type"                   : "ANDROID",
        "location_cluster"              : "SUSPICIOUS_ORIGIN",
        "upi_app"                       : "PhonePe",
        "is_new_device"                 : 1,
        "is_night_transaction"          : 1,
        "transaction_velocity_last_1hr" : 12,
        "transaction_velocity_last_24hr": 35,
        "account_age_days"              : 3,
        "failed_login_attempts"         : 7,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="UPI Fraud Predictor — CLI")
    parser.add_argument("--model",  default="Random Forest",
                        choices=list(MODEL_FILENAMES.keys()))
    parser.add_argument("--amount", type=float, default=500.0)
    parser.add_argument("--fraud-test", action="store_true",
                        help="Use a known fraud-pattern input for testing")
    args = parser.parse_args()

    inputs = _fraud_inputs(args.amount) if args.fraud_test else _demo_inputs(args.amount)

    print(f"\n── Input Transaction ({'FRAUD PATTERN' if args.fraud_test else 'NORMAL PATTERN'})")
    for k, v in inputs.items():
        print(f"  {k:<40} = {v}")

    result = predict(inputs, model_name=args.model)

    rating, color = get_risk_rating(result["fraud_probability"])
    print(f"\n── Prediction  [{args.model}]")
    print(f"  Result        : {result['result_label']}")
    print(f"  Fraud prob    : {result['fraud_probability']*100:.2f}%")
    print(f"  Risk rating   : {rating}")
    print(f"  Threshold     : {result['threshold']}")
    print()


if __name__ == "__main__":
    main()
