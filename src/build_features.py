"""
Day 3, part 2: Feature engineering.

Takes the raw transactions.csv and produces a clean, model-ready table
containing ONLY the rows that matter for the ML classifier we build
tomorrow (Day 4): failed transactions that were actually retried
(excludes successes - nothing to predict - and voluntary churn -
we don't retry those by design).

Usage:
    python src/build_features.py
"""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).parent.parent / "data" / "generated" / "transactions.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "generated" / "features.csv"


def load_retryable_transactions() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    for col in ["is_voluntary_churn", "is_salary_window", "retry_success"]:
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    failed = df[df["initial_status"] == "failed"]
    retryable = failed[failed["is_voluntary_churn"] == False].copy()
    return retryable


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()

    features["transaction_id"] = df["transaction_id"]

    # Payment method as binary flag - simpler for a linear model than a string category
    features["is_upi"] = (df["payment_method"] == "upi").astype(int)

    # One-hot encode decline_code - this is the single most predictive feature,
    # since hard declines (CARD_EXPIRED etc.) have near-zero retry success
    # almost regardless of anything else.
    decline_dummies = pd.get_dummies(df["decline_code"], prefix="decline").astype(int)
    features = pd.concat([features, decline_dummies], axis=1)

    # One-hot encode category too - gives the model a coarser signal as a backup
    # in case a decline_code appears too rarely to learn a stable pattern from.
    category_dummies = pd.get_dummies(df["decline_category"], prefix="category").astype(int)
    features = pd.concat([features, category_dummies], axis=1)

    # Timing features
    features["charge_day_of_month"] = df["charge_day_of_month"]
    features["retry_day_of_month"] = df["retry_day_of_month"]
    features["is_salary_window"] = df["is_salary_window"].astype(int)

    # Interaction feature: the salary-window boost was only ever designed to
    # apply to funds-related failures (Day 2's generator only boosts
    # INSUFFICIENT_FUNDS and DO_NOT_HONOR_TEMP). Without this interaction term,
    # a linear model can only learn ONE global salary-window effect across all
    # 8 decline codes, which dilutes the real signal down to near-zero because
    # it's genuinely irrelevant for 6 of them. This lets the model learn the
    # effect specifically where it actually exists.
    funds_related = df["decline_code"].isin(["INSUFFICIENT_FUNDS", "DO_NOT_HONOR_TEMP"]).astype(int)
    features["salary_window_x_funds_related"] = features["is_salary_window"] * funds_related

    # Days between original failure and retry attempt (handles month wraparound
    # the same way generate_data.py did, so it's consistent with how the data was made)
    days_gap = (df["retry_day_of_month"] - df["charge_day_of_month"]) % 28
    features["days_since_failure"] = days_gap

    # Subscription amount - kept raw rather than bucketed, since tree-based and
    # linear models both handle a single numeric column fine, and bucketing
    # would throw away information without a clear reason to.
    features["subscription_amount"] = df["subscription_amount"]

    # Target variable - what Day 4's model will predict
    features["retry_success"] = df["retry_success"].astype(int)

    return features


def print_feature_summary(features: pd.DataFrame):
    print(f"Feature table shape: {features.shape[0]} rows, {features.shape[1]} columns")
    print()
    print("Columns:")
    for col in features.columns:
        print(f"  - {col}")
    print()
    print("Target balance (retry_success):")
    print(features["retry_success"].value_counts(normalize=True).round(3))
    print()
    print("Sample rows:")
    print(features.head(3).to_string())


def main():
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH.name} not found.")
        print(f"Run this first: python src/generate_data.py")
        raise SystemExit(1)

    retryable = load_retryable_transactions()
    features = engineer_features(retryable)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved feature table to {OUTPUT_PATH}\n")
    print_feature_summary(features)


if __name__ == "__main__":
    main()
