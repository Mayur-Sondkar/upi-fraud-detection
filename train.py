"""
train.py
────────
Trains Logistic Regression, Random Forest, and Gradient Boosting (XGBoost-style)
on the UPI fraud dataset. Compares all three, saves every model,
and produces evaluation charts.

Run:
    python train.py
    python train.py --data upi_anonymized_dataset.csv
    python train.py --data upi_anonymized_dataset.csv --no-noise
"""

import os
import time
import logging
import argparse
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  |  %(levelname)-8s  |  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score,
)

# Local module
from preprocessing import run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_DATA   = "upi_anonymized_dataset.csv"
MODELS_DIR     = "models"
PLOTS_DIR      = "plots"
REPORT_FILE    = "training_report.json"
RANDOM_STATE   = 42
THRESHOLD      = 0.35       # Lower → higher recall, fewer missed frauds

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 130


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_models() -> dict:
    """
    Define three models with fraud-detection–optimised hyperparameters.

    Logistic Regression
        Linear baseline. C=0.1 = strong regularisation to prevent overfit.
        class_weight='balanced' upweights fraud in the loss function.
        Requires scaled features (uses X_train_sc / X_test_sc).

    Random Forest
        Bagging ensemble of 200 trees. max_depth=12 limits tree complexity
        (prevents memorising the training set). min_samples_leaf=15 ensures
        each leaf has enough support. class_weight='balanced_subsample'
        re-weights per bootstrap sample — better for imbalanced data than
        plain 'balanced'.

    Gradient Boosting  (XGBoost-equivalent in sklearn)
        Sequential boosting. learning_rate=0.08 with subsample=0.8 and
        max_features=0.8 provide stochastic regularisation. n_iter_no_change=20
        enables early stopping when validation loss plateaus.
        This gives near-XGBoost accuracy without an external dependency.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                C=0.1,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "uses_scaling": True,   # LR must use scaled features
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=15,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            "uses_scaling": False,  # Trees don't need scaling
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.08,
                max_depth=5,
                subsample=0.80,
                max_features=0.80,
                min_samples_leaf=20,
                validation_fraction=0.10,
                n_iter_no_change=20,
                tol=1e-4,
                random_state=RANDOM_STATE,
            ),
            "uses_scaling": False,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             threshold: float = THRESHOLD) -> dict:
    """
    Compute all metrics at the given probability threshold.

    Why threshold=0.35 not 0.5?
    In fraud detection, the cost of a False Negative (missed fraud) is far
    higher than a False Positive (legitimate flagged for review). Lowering
    the threshold shifts the operating point toward higher recall.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "Accuracy"      : round(accuracy_score(y_test, y_pred),                    4),
        "Precision"     : round(precision_score(y_test, y_pred, zero_division=0),  4),
        "Recall"        : round(recall_score(y_test, y_pred, zero_division=0),     4),
        "F1"            : round(f1_score(y_test, y_pred, zero_division=0),         4),
        "ROC_AUC"       : round(roc_auc_score(y_test, y_proba),                   4),
        "Avg_Precision" : round(average_precision_score(y_test, y_proba),          4),
        "Fraud_Recall"  : round(report.get("1", {}).get("recall", 0),              4),
        "Fraud_Prec"    : round(report.get("1", {}).get("precision", 0),           4),
        "Fraud_F1"      : round(report.get("1", {}).get("f1-score", 0),            4),
        "y_pred"        : y_pred,
        "y_proba"       : y_proba,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(data: dict) -> tuple[dict, dict]:
    """
    Train all models, evaluate on the held-out test set, return results.

    Uses scaled data for Logistic Regression and unscaled for tree models.
    Prints a formatted comparison table.
    """
    log.info("═" * 60)
    log.info("MODEL TRAINING  (threshold=%.2f)", THRESHOLD)
    log.info("═" * 60)

    models_config = get_models()
    trained   = {}
    results   = {}

    print("\n" + "═" * 85)
    print(f"  {'Model':<22} {'Acc':>7} {'Prec':>7} {'Recall':>8} "
          f"{'F1':>7} {'ROC-AUC':>9} {'FraudRec':>10} {'FraudF1':>9}")
    print("═" * 85)

    for name, cfg in models_config.items():
        model      = cfg["model"]
        use_scale  = cfg["uses_scaling"]

        X_tr = data["X_train_sc"] if use_scale else data["X_train"]
        X_te = data["X_test_sc"]  if use_scale else data["X_test"]
        y_tr = data["y_train"]
        y_te = data["y_test"]

        log.info("Training '%s' ...", name)
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        log.info("  Done in %.1fs", elapsed)

        scores = evaluate(model, X_te, y_te)
        trained[name] = model
        results[name] = scores

        print(f"  {name:<22} {scores['Accuracy']:>7.4f} {scores['Precision']:>7.4f} "
              f"{scores['Recall']:>8.4f} {scores['F1']:>7.4f} "
              f"{scores['ROC_AUC']:>9.4f} {scores['Fraud_Recall']:>10.4f} "
              f"{scores['Fraud_F1']:>9.4f}")

    print("═" * 85)
    print(f"  Decision threshold = {THRESHOLD}  |  Primary metric = Fraud Recall\n")

    # Full classification reports
    for name, scores in results.items():
        print(f"\n── {name} " + "─" * 60)
        cfg = models_config[name]
        use_scale = cfg["uses_scaling"]
        X_te = data["X_test_sc"] if use_scale else data["X_test"]
        print(classification_report(
            data["y_test"], scores["y_pred"],
            target_names=["Normal", "Fraud"], digits=4
        ))

    return trained, results


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(trained: dict, results: dict,
                             y_test: pd.Series) -> None:
    """Side-by-side confusion matrices for all three models."""
    n   = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained.items()):
        cm  = confusion_matrix(y_test, results[name]["y_pred"])
        pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        labels = np.array([[f"{v:,}\n({p:.1f}%)" for v, p in zip(rv, rp)]
                            for rv, rp in zip(cm, pct)])
        sns.heatmap(cm, annot=labels, fmt="", ax=ax, cmap="Blues",
                    linewidths=0.8, xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"], annot_kws={"size": 11})
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        tn, fp, fn, tp = cm.ravel()
        ax.text(0.5, -0.20,
                f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}",
                transform=ax.transAxes, ha="center", fontsize=9, color="dimgray")

    fig.suptitle(f"Confusion Matrices  (threshold={THRESHOLD})",
                 fontsize=14, fontweight="bold", y=1.03)
    out = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_metrics_comparison(results: dict) -> None:
    """Grouped bar chart: Accuracy, Precision, Recall, F1, ROC-AUC."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    models  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.25
    colors  = ["#4C9BE8", "#E87B4C", "#4CE87B"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (name, color) in enumerate(zip(models, colors)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=color, edgecolor="white", alpha=0.90)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison  (threshold=0.35  |  primary metric = Recall)",
                 fontweight="bold")
    ax.legend(loc="lower right")

    out = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


def plot_feature_importance(trained: dict, feature_cols: list,
                             top_n: int = 15) -> None:
    """Feature importance chart for tree-based models."""
    tree_models = {k: v for k, v in trained.items()
                   if hasattr(v, "feature_importances_")}
    if not tree_models:
        return

    n   = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        feat_df = (
            pd.DataFrame({"Feature": feature_cols,
                          "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=True)
            .tail(top_n)
        )
        ax.barh(feat_df["Feature"], feat_df["Importance"],
                color="#4C9BE8", edgecolor="white", height=0.7)
        ax.set_title(f"{name} — Top {top_n} Features", fontweight="bold")
        ax.set_xlabel("Importance Score")

    out = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest"      : "random_forest.pkl",
    "Gradient Boosting"  : "gradient_boosting.pkl",
}


def save_models(trained: dict, results: dict, data: dict) -> str:
    """
    Save every trained model as a self-contained artefact dict.
    Returns the name of the best model (highest Fraud_Recall).

    Each .pkl file contains:
      model        → fitted estimator
      scaler       → fitted StandardScaler (used for LR)
      encoders     → dict of LabelEncoders
      feature_cols → ordered feature list
      threshold    → decision threshold
      uses_scaling → bool (True = must scale input before predict)
      metrics      → evaluation scores
      trained_at   → ISO timestamp
    """
    best_name   = max(results, key=lambda n: (results[n]["Fraud_Recall"],
                                              results[n]["ROC_AUC"]))
    models_cfg  = get_models()

    for name, model in trained.items():
        artefact = {
            "model"       : model,
            "scaler"      : data["scaler"],
            "encoders"    : data["encoders"],
            "feature_cols": data["feature_cols"],
            "threshold"   : THRESHOLD,
            "uses_scaling": models_cfg[name]["uses_scaling"],
            "metrics"     : {k: v for k, v in results[name].items()
                             if k not in ("y_pred", "y_proba")},
            "trained_at"  : datetime.now().isoformat(),
            "model_name"  : name,
        }
        path = os.path.join(MODELS_DIR, MODEL_FILENAMES[name])
        joblib.dump(artefact, path, compress=3)
        size_kb = os.path.getsize(path) / 1024
        log.info("Saved %-25s → %s  (%.1f KB)", name, path, size_kb)

    log.info("Best model: %s  (FraudRecall=%.4f  ROC-AUC=%.4f)",
             best_name,
             results[best_name]["Fraud_Recall"],
             results[best_name]["ROC_AUC"])
    return best_name


# ══════════════════════════════════════════════════════════════════════════════
#  JSON REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_report(results: dict, best_name: str, feature_cols: list) -> None:
    """Save a JSON report consumed by the Streamlit UI for the comparison table."""
    report = {
        "generated_at" : datetime.now().isoformat(),
        "threshold"    : THRESHOLD,
        "best_model"   : best_name,
        "feature_cols" : feature_cols,
        "models"       : {
            name: {k: v for k, v in scores.items()
                   if k not in ("y_pred", "y_proba")}
            for name, scores in results.items()
        },
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report saved → %s", REPORT_FILE)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="UPI Fraud Detection — Trainer")
    p.add_argument("--data",     default=DEFAULT_DATA, help="CSV dataset path")
    p.add_argument("--no-noise", action="store_true",  help="Skip noise injection")
    return p.parse_args()


def main():
    args      = parse_args()
    t_total   = time.time()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  UPI Fraud Detection  —  Training Pipeline v2.0.0   ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # 1 Preprocess
    data = run_pipeline(
        data_path = args.data,
        noise     = not args.no_noise,
    )

    # 2 Train & compare
    trained, results = train_all_models(data)

    # 3 Visualise
    plot_confusion_matrices(trained, results, data["y_test"])
    plot_metrics_comparison(results)
    plot_feature_importance(trained, data["feature_cols"])

    # 4 Save
    best_name = save_models(trained, results, data)
    save_report(results, best_name, data["feature_cols"])

    # 5 Final summary
    elapsed = time.time() - t_total
    s = results[best_name]
    print("\n" + "═" * 58)
    print("  TRAINING COMPLETE")
    print("═" * 58)
    print(f"  Best model        : {best_name}")
    print(f"  Accuracy          : {s['Accuracy']:.4f}")
    print(f"  Precision (fraud) : {s['Fraud_Prec']:.4f}")
    print(f"  Recall    (fraud) : {s['Fraud_Recall']:.4f}  ← primary metric")
    print(f"  F1        (fraud) : {s['Fraud_F1']:.4f}")
    print(f"  ROC-AUC           : {s['ROC_AUC']:.4f}")
    print(f"  All models saved  : {MODELS_DIR}/")
    print(f"  Training time     : {elapsed:.1f}s")
    print("═" * 58 + "\n")


if __name__ == "__main__":
    main()
