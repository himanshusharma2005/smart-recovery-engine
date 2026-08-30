"""
Day 4: Train the retry-success classifier.

Trains two models on features.csv - Logistic Regression (interpretable
baseline) and Random Forest (usually stronger, catches non-linear patterns)
- and picks the better one based on ROC-AUC, not just accuracy (accuracy
alone is a weak metric even on a near-balanced dataset like this one).

Saves the winning model to data/generated/model.pkl and a feature
importance chart to docs/images/, so Day 6's simulation engine can load
the model instead of retraining it every time.

Usage:
    python src/train_model.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path(__file__).parent.parent / "data" / "generated" / "features.csv"
MODEL_PATH = Path(__file__).parent.parent / "data" / "generated" / "model.pkl"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"

RANDOM_SEED = 42

# These columns live on very different numeric ranges than the 0/1 binary
# columns (e.g. subscription_amount up to 1499 vs. is_upi at 0 or 1).
# Logistic Regression's solver struggles to converge cleanly without
# scaling these - Random Forest doesn't care, but we scale for both so
# the comparison uses identical input data.
NUMERIC_COLS = ["charge_day_of_month", "retry_day_of_month",
                 "days_since_failure", "subscription_amount"]


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in ("transaction_id", "retry_success")]
    X = df[feature_cols]
    y = df["retry_success"]
    return X, y, feature_cols


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC:  {auc:.3f}  (0.5 = random guessing, 1.0 = perfect)")
    print(f"Confusion matrix (rows=actual, cols=predicted):")
    print(f"                 pred_fail  pred_success")
    print(f"  actual_fail    {cm[0][0]:>9}  {cm[0][1]:>12}")
    print(f"  actual_success {cm[1][0]:>9}  {cm[1][1]:>12}")
    print()
    print(classification_report(y_test, preds, target_names=["fail", "success"]))
    print()

    return {"name": name, "model": model, "accuracy": acc, "auc": auc}


def plot_feature_importance(model, feature_cols, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    order = np.argsort(importances)[-10:]  # top 10
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([feature_cols[i] for i in order], importances[order], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 10 features - {model_name}")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def main():
    if not FEATURES_PATH.exists():
        print(f"ERROR: {FEATURES_PATH.name} not found.")
        print(f"Run this first: python src/build_features.py")
        raise SystemExit(1)

    X, y, feature_cols = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows\n")

    # Fit the scaler on TRAIN ONLY, then apply to both - fitting on the full
    # dataset (including test) would leak test-set information into training,
    # which quietly inflates reported performance.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    log_reg.fit(X_train, y_train)
    result_lr = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    result_rf = evaluate_model("Random Forest", rf, X_test, y_test)

    winner = result_rf if result_rf["auc"] >= result_lr["auc"] else result_lr

    print("=" * 60)
    loser = result_lr if winner["name"] != "Logistic Regression" else result_rf
    print(f"WINNER: {winner['name']} (ROC-AUC {winner['auc']:.3f} vs "
          f"{loser['name']}'s {loser['auc']:.3f})")
    print("=" * 60)

    joblib.dump({"model": winner["model"], "feature_cols": feature_cols,
                 "scaler": scaler, "numeric_cols": NUMERIC_COLS}, MODEL_PATH)
    print(f"Saved winning model to {MODEL_PATH}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(winner["model"], feature_cols, winner["name"])
    print(f"Saved feature importance chart to {IMAGES_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
