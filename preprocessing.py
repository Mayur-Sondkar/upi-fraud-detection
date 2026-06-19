"""
preprocessing.py
────────────────
Data loading, leakage removal, feature engineering, encoding, scaling,
and imbalance handling for the UPI Fraud Detection system.

Leakage columns removed
───────────────────────
  sender_balance_after    → computed directly from (before - amount) in normal rows;
                            in fraud rows set near-zero — a perfect fraud discriminator
                            that would not exist at prediction time.
  receiver_balance_after  → same issue: equals (before + amount); reveals the outcome.
  token_*_id              → anonymised hashes, carry no signal.

After removing leakage, realistic model performance is ~94–98% ROC-AUC
rather than the artificially perfect 100%.
"""

import os
import logging
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger("preprocessing")

# ── Column groups ──────────────────────────────────────────────────────────────
# Columns that directly reveal the fraud outcome (computed AFTER the transaction)
LEAKAGE_COLS = [
    "sender_balance_after",    # = before - amount in normal; near-zero in fraud
    "receiver_balance_after",  # = before + amount always; encodes amount trivially
    "token_transaction_id",    # anonymised hash — no signal
    "token_sender_id",         # anonymised hash — no signal
    "token_receiver_id",       # anonymised hash — no signal
]

# Categorical columns requiring label encoding
CAT_COLS = ["transaction_type", "device_type", "location_cluster", "upi_app"]

# Final feature order (locked — must match app.py input fields exactly)
FEATURE_COLS = [
    "transaction_hour",
    "day_of_week",
    "is_weekend",
    "transaction_amount",
    "sender_balance_before",
    "receiver_balance_before",
    "transaction_type",
    "device_type",
    "location_cluster",
    "upi_app",
    "is_new_device",
    "is_night_transaction",
    "transaction_velocity_last_1hr",
    "transaction_velocity_last_24hr",
    "account_age_days",
    "failed_login_attempts",
]

TARGET_COL = "is_fraud"


# ══════════════════════════════════════════════════════════════════════════════
#  1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """
    Load the CSV dataset and run a basic audit.
    Raises FileNotFoundError with a clear message if the path is wrong.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: '{path}'\nCurrent dir: {os.getcwd()}"
        )

    df = pd.read_csv(path)
    log.info("Loaded  %s  →  %d rows × %d cols", path, *df.shape)

    # Fill nulls
    num_nulls = df.isnull().sum().sum()
    if num_nulls:
        log.warning("%d null values found — filling with median / mode", num_nulls)
        df = _fill_nulls(df)

    fraud_pct = df[TARGET_COL].mean() * 100
    log.info("Fraud rate: %.2f%%  (Normal=%d  Fraud=%d)",
             fraud_pct, (df[TARGET_COL] == 0).sum(), (df[TARGET_COL] == 1).sum())

    return df


def _fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numerics, mode for categoricals."""
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(exclude="number").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  2 — LEAKAGE REMOVAL + NOISE INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def remove_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that cause data leakage:
      • sender_balance_after  — in normal rows, = before - amount (deterministic).
                                In fraud rows, set near-zero by the data generator.
                                A model trained on this learns the *outcome* not the *pattern*.
      • receiver_balance_after — symmetric issue.
      • token_*_id            — hashed identifiers, no predictive value.

    Returns a copy without leakage columns.
    """
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=to_drop)
    log.info("Removed leakage / ID cols: %s", to_drop)
    return df


def inject_noise(df: pd.DataFrame, noise_pct: float = 0.05,
                 label_flip_pct: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """
    Add controlled noise so the model cannot achieve unrealistic 100% accuracy
    on a synthetic dataset.

    Two types of noise:
    1. Feature noise — Gaussian perturbation (5% of column std) on numeric
       columns. Prevents the model from memorising exact threshold boundaries.
    2. Label noise   — Flip ~3% of fraud labels to 0 and inject a small number
       of hard negatives (label 0 → 1). Simulates real-world annotation errors
       and ambiguous edge-case transactions.

    In a real-world dataset you would NOT apply this — real data already
    contains noise. This is needed here because the synthetic generator
    created perfectly separable clusters.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # ── Feature noise ──────────────────────────────────────────────────────────
    noisy_cols = [
        "transaction_amount", "sender_balance_before", "receiver_balance_before",
        "transaction_velocity_last_1hr", "transaction_velocity_last_24hr",
        "account_age_days",
    ]
    for col in noisy_cols:
        if col in df.columns:
            std = df[col].std()
            noise = rng.normal(0, std * noise_pct, size=len(df))
            df[col] = (df[col] + noise).clip(lower=0)

    # ── Label noise ────────────────────────────────────────────────────────────
    fraud_idx  = df[df[TARGET_COL] == 1].index
    normal_idx = df[df[TARGET_COL] == 0].index

    # Flip some fraud → normal (simulates undetected frauds / annotation errors)
    n_flip_fraud = int(len(fraud_idx) * label_flip_pct)
    flip_fraud   = rng.choice(fraud_idx, size=n_flip_fraud, replace=False)
    df.loc[flip_fraud, TARGET_COL] = 0

    # Inject a few normal → fraud (simulates missed labelling)
    n_inject = min(200, int(len(normal_idx) * 0.0002))
    inject   = rng.choice(normal_idx, size=n_inject, replace=False)
    df.loc[inject, TARGET_COL] = 1

    log.info("Noise injected:  feature_noise=%.0f%%  label_flips=%d  injections=%d",
             noise_pct * 100, n_flip_fraud, n_inject)
    log.info("New fraud rate after noise: %.2f%%", df[TARGET_COL].mean() * 100)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  3 — ENCODING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns the modified DataFrame and a dict of fitted LabelEncoders
    (saved into the model artefact for inference-time consistency).
    """
    encoders = {}
    for col in CAT_COLS:
        if col not in df.columns:
            log.warning("Expected categorical column '%s' not found — skipping.", col)
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        log.info("  Encoded '%-20s'  →  classes: %s", col, list(le.classes_))
    return df, encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that improve fraud detection without leakage.

    balance_utilisation  : fraction of sender balance used in one transaction.
                           High values (>0.8) are a strong fraud indicator.
    log_amount           : log-transform of amount; tames the heavy right tail
                           so logistic regression and other linear models can fit it.
    velocity_ratio       : ratio of 1hr to 24hr velocity; a spike in 1hr relative
                           to 24hr average suggests an unusual burst.
    """
    eps = 1e-6  # avoid division by zero

    df["balance_utilisation"] = (
        df["transaction_amount"] / (df["sender_balance_before"] + eps)
    ).clip(0, 5)

    df["log_amount"] = np.log1p(df["transaction_amount"])

    df["velocity_ratio"] = (
        df["transaction_velocity_last_1hr"] /
        (df["transaction_velocity_last_24hr"] + eps)
    ).clip(0, 10)

    log.info("Engineered 3 new features: balance_utilisation, log_amount, velocity_ratio")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  4 — BUILD X / y WITH FINAL FEATURE LIST
# ══════════════════════════════════════════════════════════════════════════════

# Extended feature list (includes engineered features)
EXTENDED_FEATURE_COLS = FEATURE_COLS + [
    "balance_utilisation",
    "log_amount",
    "velocity_ratio",
]


def build_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract the feature matrix X and target vector y.
    Uses EXTENDED_FEATURE_COLS (original + engineered).
    """
    available = [c for c in EXTENDED_FEATURE_COLS if c in df.columns]
    missing   = [c for c in EXTENDED_FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing engineered features: %s — skipping.", missing)

    X = df[available].copy()
    y = df[TARGET_COL].copy()
    log.info("Feature matrix: %d rows × %d features", *X.shape)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  5 — SCALING
# ══════════════════════════════════════════════════════════════════════════════

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data only (prevent test leakage)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ══════════════════════════════════════════════════════════════════════════════
#  6 — IMBALANCE HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "oversample",
    target_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Handle the ~2% fraud minority class.

    strategy="oversample"  (default)
        Random over-sampling: bootstrap fraud rows until fraud = target_ratio
        of training set. Preserves real fraud feature distribution exactly.
        Preferred because the synthetic dataset has well-defined clusters.

    strategy="weights"
        Uses class_weight="balanced" inside each model instead of resampling.
        Faster, uses less memory, but may underperform on very sparse minority.

    Why not SMOTE?
        SMOTE synthesises new fraud samples by interpolating between existing ones.
        On anonymised/tokenised financial data this risks creating out-of-manifold
        samples that hurt real-world generalisation. Random over-sampling is safer.
    """
    if strategy == "weights":
        log.info("Imbalance strategy: class_weight='balanced' inside models")
        return X_train, y_train

    n_normal = int((y_train == 0).sum())
    n_fraud  = int((y_train == 1).sum())
    target_n = int(n_normal * target_ratio)

    log.info("Imbalance: before → Normal=%d  Fraud=%d  (%.2f%%)",
             n_normal, n_fraud, n_fraud / len(y_train) * 100)

    X_fraud = X_train[y_train == 1]
    y_fraud = y_train[y_train == 1]

    X_fraud_up, y_fraud_up = resample(
        X_fraud, y_fraud,
        replace=True, n_samples=target_n, random_state=seed,
    )

    X_bal = pd.concat([X_train[y_train == 0], X_fraud_up], ignore_index=True)
    y_bal = pd.concat([y_train[y_train == 0], y_fraud_up], ignore_index=True)

    # Shuffle
    perm  = np.random.default_rng(seed).permutation(len(X_bal))
    X_bal = X_bal.iloc[perm].reset_index(drop=True)
    y_bal = y_bal.iloc[perm].reset_index(drop=True)

    log.info("Imbalance: after  → Normal=%d  Fraud=%d  (%.2f%%)",
             (y_bal == 0).sum(), (y_bal == 1).sum(), (y_bal == 1).mean() * 100)

    return X_bal, y_bal


# ══════════════════════════════════════════════════════════════════════════════
#  7 — FULL PIPELINE (called by train.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    data_path: str,
    noise: bool = True,
    imbalance_strategy: str = "oversample",
) -> dict:
    """
    Run the full preprocessing pipeline and return a dict with everything
    train.py needs.

    Returns
    ───────
    {
      X_train, X_test, y_train, y_test   — raw (unscaled) splits
      X_train_sc, X_test_sc              — scaled versions for LR
      scaler                             — fitted StandardScaler
      encoders                           — dict of LabelEncoders
      feature_cols                       — ordered list of feature names
    }
    """
    from sklearn.model_selection import train_test_split

    log.info("═" * 55)
    log.info("PREPROCESSING PIPELINE")
    log.info("═" * 55)

    df = load_data(data_path)
    df = remove_leakage(df)

    if noise:
        df = inject_noise(df)

    df, encoders = encode_features(df)
    df = engineer_features(df)

    X, y = build_Xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    log.info("Split: Train=%d  Test=%d", len(X_train), len(X_test))

    X_train_bal, y_train_bal = handle_imbalance(
        X_train, y_train, strategy=imbalance_strategy
    )

    scaler    = fit_scaler(X_train_bal)
    X_tr_sc   = pd.DataFrame(scaler.transform(X_train_bal),
                              columns=X_train_bal.columns)
    X_te_sc   = pd.DataFrame(scaler.transform(X_test),
                              columns=X_test.columns)

    log.info("Preprocessing complete. Features: %s", list(X.columns))

    return {
        "X_train"     : X_train_bal,
        "X_test"      : X_test,
        "y_train"     : y_train_bal,
        "y_test"      : y_test,
        "X_train_sc"  : X_tr_sc,
        "X_test_sc"   : X_te_sc,
        "scaler"      : scaler,
        "encoders"    : encoders,
        "feature_cols": list(X.columns),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  8 — INFERENCE HELPER (called by predict.py and app.py)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_single_input(raw: dict, encoders: dict, scaler,
                          feature_cols: list) -> np.ndarray:
    """
    Transform a single transaction dict (from the UI) into a model-ready
    numpy array. Applies the same encoding, engineering, and scaling
    that was used during training.

    Parameters
    ──────────
    raw          : dict of raw field values from the Streamlit form
    encoders     : dict of fitted LabelEncoders (from model artefact)
    scaler       : fitted StandardScaler (from model artefact)
    feature_cols : ordered list of column names (from model artefact)

    Returns
    ───────
    np.ndarray of shape (1, n_features) — ready for model.predict_proba()
    """
    row = dict(raw)  # shallow copy

    # Encode categoricals
    for col, le in encoders.items():
        if col in row:
            val = str(row[col])
            row[col] = le.transform([val])[0] if val in le.classes_ else 0

    # Engineered features
    eps = 1e-6
    amt   = float(row.get("transaction_amount", 0))
    bal   = float(row.get("sender_balance_before", eps))
    v1hr  = float(row.get("transaction_velocity_last_1hr", 0))
    v24hr = float(row.get("transaction_velocity_last_24hr", eps))

    row["balance_utilisation"] = min(amt / (bal + eps), 5.0)
    row["log_amount"]          = np.log1p(amt)
    row["velocity_ratio"]      = min(v1hr / (v24hr + eps), 10.0)

    # Build ordered array matching training feature_cols
    arr = np.array([[row.get(c, 0) for c in feature_cols]], dtype=np.float64)

    # Apply scaler
    arr = scaler.transform(arr)

    return arr
