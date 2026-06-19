"""
database.py
───────────
SQLite integration for storing transaction predictions.

Provides:
  init_db()          — create tables on first run
  insert_prediction() — save one prediction record
  fetch_history()    — return recent predictions as a DataFrame
  fetch_stats()      — aggregate KPIs for the dashboard
"""

import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager

import pandas as pd

log = logging.getLogger("database")

# Default database file (created automatically in the project root)
DEFAULT_DB = "fraud_predictions.db"


# ══════════════════════════════════════════════════════════════════════════════
#  CONNECTION CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def get_conn(db_path: str = DEFAULT_DB):
    """
    Yield a SQLite connection with WAL mode (safe for concurrent Streamlit access)
    and auto-commit / rollback.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE CREATION
# ══════════════════════════════════════════════════════════════════════════════

CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                       TEXT    NOT NULL,
    model_used                      TEXT    NOT NULL,

    -- Transaction inputs
    transaction_hour                INTEGER,
    day_of_week                     INTEGER,
    is_weekend                      INTEGER,
    transaction_amount              REAL,
    sender_balance_before           REAL,
    receiver_balance_before         REAL,
    transaction_type                TEXT,
    device_type                     TEXT,
    location_cluster                TEXT,
    upi_app                         TEXT,
    is_new_device                   INTEGER,
    is_night_transaction            INTEGER,
    transaction_velocity_last_1hr   INTEGER,
    transaction_velocity_last_24hr  INTEGER,
    account_age_days                INTEGER,
    failed_login_attempts           INTEGER,

    -- Prediction output
    prediction                      INTEGER NOT NULL,   -- 0=Normal 1=Fraud
    fraud_probability               REAL    NOT NULL,   -- 0.0 – 1.0
    result_label                    TEXT    NOT NULL    -- 'FRAUD' or 'SAFE'
)
"""

CREATE_IDX = """
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
ON predictions (timestamp DESC)
"""


def init_db(db_path: str = DEFAULT_DB) -> None:
    """
    Initialise the database and create tables if they don't exist.
    Safe to call on every startup.
    """
    with get_conn(db_path) as conn:
        conn.execute(CREATE_PREDICTIONS_TABLE)
        conn.execute(CREATE_IDX)
    log.info("Database ready: %s", db_path)


# ══════════════════════════════════════════════════════════════════════════════
#  INSERT PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def insert_prediction(
    inputs: dict,
    model_used: str,
    prediction: int,
    fraud_probability: float,
    db_path: str = DEFAULT_DB,
) -> int:
    """
    Insert one prediction record into the database.

    Parameters
    ──────────
    inputs           : dict of raw feature values from the UI form
    model_used       : name of the model that made the prediction
    prediction       : 0 = Normal, 1 = Fraud
    fraud_probability: float in [0, 1]

    Returns the auto-generated row id.
    """
    result_label = "FRAUD" if prediction == 1 else "SAFE"
    timestamp    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sql = """
    INSERT INTO predictions (
        timestamp, model_used,
        transaction_hour, day_of_week, is_weekend,
        transaction_amount, sender_balance_before, receiver_balance_before,
        transaction_type, device_type, location_cluster, upi_app,
        is_new_device, is_night_transaction,
        transaction_velocity_last_1hr, transaction_velocity_last_24hr,
        account_age_days, failed_login_attempts,
        prediction, fraud_probability, result_label
    ) VALUES (
        :timestamp, :model_used,
        :transaction_hour, :day_of_week, :is_weekend,
        :transaction_amount, :sender_balance_before, :receiver_balance_before,
        :transaction_type, :device_type, :location_cluster, :upi_app,
        :is_new_device, :is_night_transaction,
        :transaction_velocity_last_1hr, :transaction_velocity_last_24hr,
        :account_age_days, :failed_login_attempts,
        :prediction, :fraud_probability, :result_label
    )
    """
    record = {
        "timestamp"                     : timestamp,
        "model_used"                    : model_used,
        "transaction_hour"              : inputs.get("transaction_hour", 0),
        "day_of_week"                   : inputs.get("day_of_week", 0),
        "is_weekend"                    : inputs.get("is_weekend", 0),
        "transaction_amount"            : inputs.get("transaction_amount", 0.0),
        "sender_balance_before"         : inputs.get("sender_balance_before", 0.0),
        "receiver_balance_before"       : inputs.get("receiver_balance_before", 0.0),
        "transaction_type"              : inputs.get("transaction_type", ""),
        "device_type"                   : inputs.get("device_type", ""),
        "location_cluster"              : inputs.get("location_cluster", ""),
        "upi_app"                       : inputs.get("upi_app", ""),
        "is_new_device"                 : inputs.get("is_new_device", 0),
        "is_night_transaction"          : inputs.get("is_night_transaction", 0),
        "transaction_velocity_last_1hr" : inputs.get("transaction_velocity_last_1hr", 0),
        "transaction_velocity_last_24hr": inputs.get("transaction_velocity_last_24hr", 0),
        "account_age_days"              : inputs.get("account_age_days", 0),
        "failed_login_attempts"         : inputs.get("failed_login_attempts", 0),
        "prediction"                    : prediction,
        "fraud_probability"             : round(float(fraud_probability), 6),
        "result_label"                  : result_label,
    }

    with get_conn(db_path) as conn:
        cursor = conn.execute(sql, record)
        row_id = cursor.lastrowid

    log.info("Saved prediction #%d  →  %s  (prob=%.4f)", row_id,
             result_label, fraud_probability)
    return row_id


# ══════════════════════════════════════════════════════════════════════════════
#  FETCH HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def fetch_history(limit: int = 50, db_path: str = DEFAULT_DB) -> pd.DataFrame:
    """
    Return the most recent `limit` predictions as a DataFrame.
    Returns an empty DataFrame if the table is empty.
    """
    sql = """
    SELECT
        id,
        timestamp,
        model_used,
        transaction_amount,
        transaction_type,
        location_cluster,
        fraud_probability,
        result_label
    FROM predictions
    ORDER BY id DESC
    LIMIT ?
    """
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()

    if not rows:
        return pd.DataFrame(columns=[
            "id", "timestamp", "model_used", "transaction_amount",
            "transaction_type", "location_cluster",
            "fraud_probability", "result_label",
        ])

    df = pd.DataFrame([dict(r) for r in rows])
    df["fraud_probability"] = (df["fraud_probability"] * 100).round(2)
    df.rename(columns={
        "id"              : "#",
        "timestamp"       : "Time",
        "model_used"      : "Model",
        "transaction_amount": "Amount (₹)",
        "transaction_type": "Type",
        "location_cluster": "Location",
        "fraud_probability": "Fraud Prob %",
        "result_label"    : "Result",
    }, inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATE STATS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stats(db_path: str = DEFAULT_DB) -> dict:
    """
    Return aggregate KPIs for the Streamlit dashboard header cards.

    Returns
    ───────
    {
      total          : int
      fraud_count    : int
      safe_count     : int
      fraud_rate_pct : float
      avg_fraud_prob : float
    }
    """
    sql = """
    SELECT
        COUNT(*)                                  AS total,
        SUM(prediction)                           AS fraud_count,
        COUNT(*) - SUM(prediction)                AS safe_count,
        ROUND(AVG(prediction) * 100, 2)           AS fraud_rate_pct,
        ROUND(AVG(fraud_probability) * 100, 2)    AS avg_fraud_prob
    FROM predictions
    """
    with get_conn(db_path) as conn:
        row = conn.execute(sql).fetchone()

    if not row or row["total"] == 0:
        return {"total": 0, "fraud_count": 0, "safe_count": 0,
                "fraud_rate_pct": 0.0, "avg_fraud_prob": 0.0}

    return {
        "total"         : int(row["total"]          or 0),
        "fraud_count"   : int(row["fraud_count"]    or 0),
        "safe_count"    : int(row["safe_count"]     or 0),
        "fraud_rate_pct": float(row["fraud_rate_pct"] or 0),
        "avg_fraud_prob": float(row["avg_fraud_prob"]  or 0),
    }


def clear_history(db_path: str = DEFAULT_DB) -> None:
    """Delete all prediction records (used by the 'Clear' button in the UI)."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM predictions")
    log.info("Prediction history cleared.")
