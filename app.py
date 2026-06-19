"""
app.py
──────
Streamlit UI for the UPI Fraud Detection system.

Features
────────
  • Model selection dropdown (LR, RF, Gradient Boosting)
  • Input form grouped into 4 sections
  • Fraud / Safe alert with probability
  • Risk gauge + confidence bar chart
  • Model comparison metrics table (from training_report.json)
  • SQLite-backed transaction history log
  • Session KPI cards

Run:
    streamlit run app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from predict  import predict, get_available_models, load_training_report, get_risk_rating
from database import init_db, insert_prediction, fetch_history, fetch_stats, clear_history

# ─────────────────────────────────────────────────────────────────────────────
# Page config — MUST be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "UPI Fraud Detection",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — professional dark-accent fintech theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Header */
  .app-header { background: linear-gradient(135deg,#0f3460,#16213e,#1a1a2e);
    padding:2rem 2.5rem; border-radius:14px; margin-bottom:1.2rem; }
  .app-header h1 { color:#e2e8f0; margin:0; font-size:1.9rem; }
  .app-header p  { color:#94a3b8; margin:.3rem 0 0; font-size:.95rem; }

  /* Section titles */
  .sec-title { color:#94a3b8; font-size:.75rem; font-weight:600;
    text-transform:uppercase; letter-spacing:.08em;
    border-bottom:1px solid #334155; padding-bottom:.3rem; margin-bottom:.8rem; }

  /* Result boxes */
  .fraud-box { background:#450a0a; border:2px solid #dc2626; border-radius:12px;
    padding:1.6rem; text-align:center; }
  .safe-box  { background:#052e16; border:2px solid #16a34a; border-radius:12px;
    padding:1.6rem; text-align:center; }
  .box-title { font-size:1.8rem; font-weight:800; margin:0; }
  .box-sub   { font-size:.95rem; margin:.4rem 0 0; opacity:.85; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background:#0f172a !important; }
  section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] div { color:#e2e8f0 !important; }

  /* Buttons */
  .stButton>button { background:#2563eb!important; color:#fff!important;
    border:none!important; border-radius:8px!important;
    padding:.65rem 1.4rem!important; font-weight:600!important;
    font-size:.95rem!important; width:100%!important; }
  .stButton>button:hover { background:#1d4ed8!important; }
</style>
""", unsafe_allow_html=True)

DB_PATH = "fraud_predictions.db"


# ─────────────────────────────────────────────────────────────────────────────
# Initialise database once per session
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_db():
    init_db(DB_PATH)

setup_db()


# ─────────────────────────────────────────────────────────────────────────────
# Gauge chart
# ─────────────────────────────────────────────────────────────────────────────
def draw_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw={"aspect": "equal"})
    for patch in (fig, ax):
        patch.set_facecolor("#0f172a")

    theta  = np.linspace(np.pi, 0, 300)
    r_out, r_in = 1.0, 0.62
    ax.fill_between(np.cos(theta), np.sin(theta),
                    r_in * np.sin(theta), color="#1e293b", zorder=1)
    ax.fill_between(np.cos(theta), r_in * np.sin(theta), 0,
                    color="#0f172a", zorder=2)

    for lo, hi, c in [(0,.3,"#16a34a"),(0.3,.6,"#d97706"),(0.6,1.0,"#dc2626")]:
        th = np.linspace(np.pi - lo*np.pi, np.pi - hi*np.pi, 120)
        ax.fill_between(np.cos(th), np.sin(th), r_in*np.sin(th),
                        color=c, alpha=0.88, zorder=3)

    angle = np.pi - prob * np.pi
    nx, ny = 0.78 * np.cos(angle), 0.78 * np.sin(angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=14))
    ax.plot(0, 0, "o", color="white", ms=8, zorder=6)

    for x_, txt, c in [(-1.1, "Safe", "#16a34a"), (1.1, "Fraud", "#dc2626")]:
        ax.text(x_, 0, txt, color=c, ha="center", va="center",
                fontsize=9, fontweight="bold")
    ax.text(0, -0.22, f"{prob*100:.1f}%", color="white",
            ha="center", va="center", fontsize=19, fontweight="bold")
    ax.text(0, -0.42, "Fraud Probability", color="#94a3b8",
            ha="center", va="center", fontsize=8)

    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.2); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Confidence bar chart
# ─────────────────────────────────────────────────────────────────────────────
def draw_confidence_bar(fraud_prob: float) -> plt.Figure:
    safe_prob = 1 - fraud_prob
    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    bars = ax.barh(["Fraud", "Safe"], [fraud_prob, safe_prob],
                   color=["#ef4444", "#22c55e"], edgecolor="#0f172a", height=0.42)
    for bar, v in zip(bars, [fraud_prob, safe_prob]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v*100:.1f}%", va="center", color="white",
                fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.25); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#334155")
    for lbl in ax.get_yticklabels(): lbl.set_color("white")
    plt.tight_layout(pad=0.4)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
      <h1>🛡️ UPI Fraud Detection System  <span style="font-size:.9rem;color:#64748b;">v2.0</span></h1>
      <p>Production-grade AI fraud analysis  ·  3-model ensemble  ·  Real-time prediction  ·  SQLite audit log</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load available models ──────────────────────────────────────────────────
    available_models = get_available_models()
    if not available_models:
        st.error("""
        **No trained models found.**

        Please run:
        ```
        python train.py --data upi_anonymized_dataset.csv
        ```
        Then restart the app.
        """)
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose prediction model",
            options=available_models,
            help="All models are trained on identical data. Switch to compare real-time outputs.",
        )

        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        report = load_training_report()
        if report and "models" in report:
            best = report.get("best_model", "")
            for name, metrics in report["models"].items():
                tag = "⭐ Best" if name == best else ""
                with st.expander(f"{name}  {tag}"):
                    m_cols = st.columns(2)
                    m_cols[0].metric("Accuracy",     f"{metrics.get('Accuracy',0):.3f}")
                    m_cols[1].metric("ROC-AUC",      f"{metrics.get('ROC_AUC',0):.3f}")
                    m_cols[0].metric("Fraud Recall",  f"{metrics.get('Fraud_Recall',0):.3f}")
                    m_cols[1].metric("Fraud F1",      f"{metrics.get('Fraud_F1',0):.3f}")

        st.markdown("---")
        st.markdown("### 📘 Risk Legend")
        st.success("🟢  < 20%  — Low Risk")
        st.warning("🟡  20–40% — Medium Risk")
        st.error("🔴  40–65% — High Risk")
        st.error("⚫  > 65%  — Critical")

        st.markdown("---")
        if st.button("🗑️ Clear History"):
            clear_history(DB_PATH)
            st.rerun()

    # ── KPI cards from DB ──────────────────────────────────────────────────────
    stats = fetch_stats(DB_PATH)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Checked",     stats["total"])
    k2.metric("Fraud Detected",    stats["fraud_count"])
    k3.metric("Safe Transactions", stats["safe_count"])
    k4.metric("Session Fraud Rate",f"{stats['fraud_rate_pct']:.1f}%")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  INPUT FORM
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<p class="sec-title">📝 Transaction Input</p>', unsafe_allow_html=True)

    with st.form("tx_form"):

        # ── Group 1: Transaction Details ─────────────────────────────────────
        st.markdown("**Transaction Details**")
        g1 = st.columns(4)
        transaction_amount   = g1[0].number_input("Amount (₹)", min_value=1.0,
                                                    value=500.0, step=100.0)
        transaction_type     = g1[1].selectbox("Type",
                                                ["P2P","P2M","Recharge","BillPayment"])
        upi_app              = g1[2].selectbox("UPI App", ["GPay","PhonePe","Paytm"])
        transaction_hour     = g1[3].number_input("Hour (0-23)", min_value=0,
                                                   max_value=23, value=14)

        g2 = st.columns(4)
        day_of_week    = g2[0].selectbox("Day of Week",
                                          [0,1,2,3,4,5,6],
                                          format_func=lambda x:
                                          ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        is_weekend     = g2[1].selectbox("Weekend?", [0,1],
                                          format_func=lambda x: "Yes" if x else "No")
        is_night_transaction = g2[2].selectbox("Night Transaction?", [0,1],
                                                format_func=lambda x: "Yes" if x else "No")
        st.markdown("")

        # ── Group 2: Balance Information ─────────────────────────────────────
        st.markdown("**Balance Information**  *(balance_after columns excluded — leakage prevention)*")
        b1, b2 = st.columns(2)
        sender_balance_before   = b1.number_input("Sender Balance Before (₹)",
                                                   min_value=0.0, value=10000.0, step=500.0)
        receiver_balance_before = b2.number_input("Receiver Balance Before (₹)",
                                                   min_value=0.0, value=5000.0, step=500.0)

        # ── Group 3: Device & Location ────────────────────────────────────────
        st.markdown("**Device & Location**")
        d1, d2, d3 = st.columns(3)
        device_type      = d1.selectbox("Device Type",  ["ANDROID","IOS","OTHER"])
        location_cluster = d2.selectbox("Location Cluster",
                                         ["NORTH_INDIA","WEST_INDIA","SOUTH_INDIA",
                                          "EAST_INDIA","SUSPICIOUS_ORIGIN"])
        is_new_device    = d3.selectbox("New Device?", [0,1],
                                         format_func=lambda x: "Yes" if x else "No")

        # ── Group 4: Behavioural Features ────────────────────────────────────
        st.markdown("**Behavioural Risk Signals**")
        bh = st.columns(4)
        transaction_velocity_last_1hr   = bh[0].number_input("Velocity (1 hr)",
                                                              min_value=0, max_value=100, value=2)
        transaction_velocity_last_24hr  = bh[1].number_input("Velocity (24 hr)",
                                                              min_value=0, max_value=200, value=5)
        account_age_days                = bh[2].number_input("Account Age (days)",
                                                              min_value=0, max_value=5000, value=365)
        failed_login_attempts           = bh[3].number_input("Failed Logins",
                                                              min_value=0, max_value=20, value=0)

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  VALIDATION + PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    if submitted:
        # Validation
        errors = []
        if transaction_amount <= 0:
            errors.append("Transaction amount must be > 0")
        if sender_balance_before < transaction_amount:
            errors.append("Sender balance before should be ≥ transaction amount "
                          "(for a legitimate transaction)")
        if account_age_days < 0:
            errors.append("Account age cannot be negative")

        for err in errors:
            st.warning(f"⚠️  {err}")

        inputs = {
            "transaction_hour"              : int(transaction_hour),
            "day_of_week"                   : int(day_of_week),
            "is_weekend"                    : int(is_weekend),
            "transaction_amount"            : float(transaction_amount),
            "sender_balance_before"         : float(sender_balance_before),
            "receiver_balance_before"       : float(receiver_balance_before),
            "transaction_type"              : str(transaction_type),
            "device_type"                   : str(device_type),
            "location_cluster"              : str(location_cluster),
            "upi_app"                       : str(upi_app),
            "is_new_device"                 : int(is_new_device),
            "is_night_transaction"          : int(is_night_transaction),
            "transaction_velocity_last_1hr" : int(transaction_velocity_last_1hr),
            "transaction_velocity_last_24hr": int(transaction_velocity_last_24hr),
            "account_age_days"              : int(account_age_days),
            "failed_login_attempts"         : int(failed_login_attempts),
        }

        with st.spinner(f"Running {selected_model} ..."):
            result = predict(inputs, model_name=selected_model)

        # Save to DB
        insert_prediction(
            inputs        = inputs,
            model_used    = selected_model,
            prediction    = result["prediction"],
            fraud_probability = result["fraud_probability"],
            db_path       = DB_PATH,
        )

        # ── Result display ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="sec-title">📊 Prediction Result</p>',
                    unsafe_allow_html=True)

        alert_col, gauge_col, bar_col = st.columns([1.5, 1, 1])

        prob    = result["fraud_probability"]
        rating, rating_color = get_risk_rating(prob)

        with alert_col:
            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="fraud-box">
                  <p class="box-title" style="color:#ff4444;">🚨 FRAUD ALERT</p>
                  <p class="box-sub"   style="color:#fca5a5;">
                    Fraud probability: <strong>{prob*100:.2f}%</strong><br>
                    Risk level: <strong>{rating}</strong><br>
                    Model: {selected_model}
                  </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-box">
                  <p class="box-title" style="color:#4ade80;">✅ SAFE</p>
                  <p class="box-sub"   style="color:#86efac;">
                    Fraud probability: <strong>{prob*100:.2f}%</strong><br>
                    Risk level: <strong>{rating}</strong><br>
                    Model: {selected_model}
                  </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(float(prob), text=f"Fraud probability: {prob*100:.1f}%")

            # Signal summary
            with st.expander("🔎 Risk Signal Summary"):
                signals = [
                    ("New device",         is_new_device == 1,                    "High-risk if new"),
                    ("Night transaction",   is_night_transaction == 1,             "00:00–05:59 window"),
                    ("Suspicious location", location_cluster == "SUSPICIOUS_ORIGIN","VPN/Proxy/Offshore"),
                    ("High 1hr velocity",   transaction_velocity_last_1hr > 5,     f"{transaction_velocity_last_1hr} txns"),
                    ("High 24hr velocity",  transaction_velocity_last_24hr > 15,    f"{transaction_velocity_last_24hr} txns"),
                    ("Failed logins",       failed_login_attempts > 2,              f"{failed_login_attempts} attempts"),
                    ("New account",         account_age_days < 30,                  f"{account_age_days} days old"),
                    ("Large amount",        transaction_amount > 50_000,            f"₹{transaction_amount:,.0f}"),
                ]
                for name, is_risky, detail in signals:
                    icon = "⚠️" if is_risky else "✅"
                    col_a, col_b = st.columns([2, 1])
                    col_a.markdown(f"{icon} **{name}** — {detail}")
                    col_b.markdown("Risk" if is_risky else "OK")

        with gauge_col:
            st.pyplot(draw_gauge(prob), use_container_width=True)
            plt.close("all")

        with bar_col:
            st.markdown("#### Confidence")
            st.pyplot(draw_confidence_bar(prob), use_container_width=True)
            plt.close("all")

            # Probability breakdown
            st.metric("Fraud Probability",  f"{prob*100:.2f}%")
            st.metric("Safe  Probability",  f"{result['safe_probability']*100:.2f}%")
            st.metric("Threshold Used",     f"{result['threshold']}")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<p class="sec-title">📈 Model Comparison  (from last training run)</p>',
                unsafe_allow_html=True)

    report = load_training_report()
    if report and "models" in report:
        best = report.get("best_model", "")
        rows = []
        for name, m in report["models"].items():
            rows.append({
                "Model"         : f"⭐ {name}" if name == best else name,
                "Accuracy"      : m.get("Accuracy",    0),
                "Precision"     : m.get("Precision",   0),
                "Recall"        : m.get("Recall",      0),
                "F1"            : m.get("F1",          0),
                "ROC-AUC"       : m.get("ROC_AUC",     0),
                "Fraud Recall"  : m.get("Fraud_Recall",0),
                "Fraud F1"      : m.get("Fraud_F1",    0),
            })
        comp_df = pd.DataFrame(rows).set_index("Model")

        def colour_recall(val):
            if isinstance(val, float):
                if val >= 0.95: return "color: #4ade80; font-weight:bold"
                if val >= 0.85: return "color: #fbbf24"
                return "color: #f87171"
            return ""

        styled = comp_df.style.format("{:.4f}").applymap(
            colour_recall, subset=["Recall", "Fraud Recall", "ROC-AUC"]
        )
        st.dataframe(styled, use_container_width=True)

        threshold = report.get("threshold", 0.35)
        st.caption(
            f"Decision threshold = {threshold}  ·  "
            "Fraud Recall is the primary metric  ·  "
            f"Best model = {best}"
        )
    else:
        st.info("No training report found. Run  `python train.py`  first.")

    # ══════════════════════════════════════════════════════════════════════════
    #  TRANSACTION HISTORY (from SQLite)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown('<p class="sec-title">📋 Transaction History  (SQLite log)</p>',
                unsafe_allow_html=True)

    history_df = fetch_history(limit=50, db_path=DB_PATH)

    if history_df.empty:
        st.info("No transactions yet. Submit a prediction above.")
    else:
        def style_result(val):
            if "FRAUD" in str(val):
                return "color:#ef4444; font-weight:bold"
            return "color:#4ade80; font-weight:bold"

        styled_hist = history_df.style.applymap(style_result, subset=["Result"])
        st.dataframe(styled_hist, use_container_width=True, height=320)

        h1, h2, h3 = st.columns(3)
        total_rows  = len(history_df)
        fraud_rows  = history_df["Result"].str.contains("FRAUD").sum()
        h1.metric("Transactions in log",    total_rows)
        h2.metric("Fraud in log",           int(fraud_rows))
        h3.metric("Fraud rate in log",      f"{fraud_rows/max(total_rows,1)*100:.1f}%")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🛡️ UPI Fraud Detection System v2.0  ·  "
        "ML: Logistic Regression | Random Forest | Gradient Boosting  ·  "
        "Leakage-free features  ·  SQLite audit log  ·  Demo only"
    )


if __name__ == "__main__":
    main()
