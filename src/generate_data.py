"""
Day 2: Synthetic transaction data generator.

Generates ~10,000 subscription renewal attempts, using data/decline_codes.json
as the source of truth for how often each failure happens and how likely a
retry is to succeed. Bakes in one deliberate real-world pattern: retries near
salary-credit windows (1st-3rd, 7th-8th of the month) succeed more often for
insufficient-funds-type failures - this is what will let the ML model in
Day 4 actually learn something meaningful instead of just noise.

Usage:
    python src/generate_data.py
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
NUM_TRANSACTIONS = 10_000
OVERALL_FAILURE_RATE = 0.13  # sits inside the real-world 10-15% benchmark range

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "decline_codes.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "generated" / "transactions.csv"

# Salary-credit windows in India: 1st of month and 7th (common corporate payroll dates)
SALARY_WINDOW_DAYS = {1, 2, 3, 7, 8}

CARD_CODES = {"INSUFFICIENT_FUNDS", "ISSUER_TIMEOUT", "DO_NOT_HONOR_TEMP",
              "CARD_EXPIRED", "CARD_LOST_STOLEN", "INVALID_CARD_DETAILS",
              "BANK_SERVER_DOWN"}
UPI_CODES = {"UPI_MANDATE_NOT_CONFIRMED", "UPI_PSP_APP_ERROR", "UPI_MANDATE_REVOKED"}


def load_taxonomy():
    with open(TAXONOMY_PATH, "r") as f:
        return json.load(f)["decline_codes"]


def pick_decline_code(taxonomy, payment_method, rng):
    """Sample a decline code, restricted to codes valid for this payment method,
    weighted by occurrence_weight (renormalized within the subset)."""
    valid_codes = [c for c in taxonomy if
                   (payment_method == "card" and c["code"] in CARD_CODES) or
                   (payment_method == "upi" and c["code"] in UPI_CODES)]
    weights = np.array([c["occurrence_weight"] for c in valid_codes])
    weights = weights / weights.sum()
    return rng.choice(valid_codes, p=weights)


def simulate_retry_success(decline_entry, retry_day, rng):
    """Simulate whether a single retry attempt succeeds, given the decline
    reason's base probability, a salary-window boost for funds-related
    failures, and random noise so the data isn't artificially clean."""
    base_prob = decline_entry["base_retry_success_prob"]

    # Salary-window boost only makes sense for failures actually related to
    # the customer not having money available yet.
    funds_related = decline_entry["code"] in {"INSUFFICIENT_FUNDS", "DO_NOT_HONOR_TEMP"}
    boost = 0.25 if (funds_related and retry_day in SALARY_WINDOW_DAYS) else 0.0

    # Small random noise (+/- 5%) so outcomes aren't a deterministic function
    # of decline_code alone - a model trained on this should still have to
    # learn real signal, not just memorize a lookup table.
    noise = rng.uniform(-0.05, 0.05)

    final_prob = np.clip(base_prob + boost + noise, 0.0, 1.0)
    # bool(...) matters here: numpy bool_ does saturating addition (True+True=True),
    # which silently breaks .mean()/.sum() once this column mixes with None values.
    return bool(rng.random() < final_prob)


def generate_dataset(seed: int = RANDOM_SEED, n: int = NUM_TRANSACTIONS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    taxonomy = load_taxonomy()
    taxonomy_by_code = {c["code"]: c for c in taxonomy}

    rows = []
    for i in range(n):
        transaction_id = f"txn_{i:06d}"
        customer_id = f"cust_{rng.integers(1, n // 3):06d}"  # some repeat customers
        payment_method = "upi" if rng.random() < 0.55 else "card"  # UPI AutoPay edges out cards in India
        subscription_amount = int(rng.choice([199, 299, 499, 799, 999, 1499]))
        charge_day = int(rng.integers(1, 29))  # keep it simple, avoid month-length edge cases

        failed = rng.random() < OVERALL_FAILURE_RATE

        if not failed:
            rows.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "subscription_amount": subscription_amount,
                "payment_method": payment_method,
                "charge_day_of_month": charge_day,
                "initial_status": "success",
                "decline_code": None,
                "decline_category": None,
                "is_voluntary_churn": False,
                "retry_day_of_month": None,
                "is_salary_window": None,
                "retry_success": None,
            })
            continue

        decline_entry = pick_decline_code(taxonomy, payment_method, rng)
        is_voluntary = decline_entry["code"] == "UPI_MANDATE_REVOKED"

        if is_voluntary:
            # Voluntary churn: no retry makes sense, excluded from the recovery funnel entirely.
            rows.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "subscription_amount": subscription_amount,
                "payment_method": payment_method,
                "charge_day_of_month": charge_day,
                "initial_status": "failed",
                "decline_code": decline_entry["code"],
                "decline_category": decline_entry["category"],
                "is_voluntary_churn": True,
                "retry_day_of_month": None,
                "is_salary_window": None,
                "retry_success": None,
            })
            continue

        # Retry happens 1-5 days after the original failed charge (typical dunning window)
        retry_offset = py_rng.randint(1, 5)
        retry_day = ((charge_day - 1 + retry_offset) % 28) + 1
        is_salary_window = bool(retry_day in SALARY_WINDOW_DAYS)
        retry_success = simulate_retry_success(decline_entry, retry_day, rng)

        rows.append({
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "subscription_amount": subscription_amount,
            "payment_method": payment_method,
            "charge_day_of_month": charge_day,
            "initial_status": "failed",
            "decline_code": decline_entry["code"],
            "decline_category": decline_entry["category"],
            "is_voluntary_churn": False,
            "retry_day_of_month": retry_day,
            "is_salary_window": is_salary_window,
            "retry_success": retry_success,
        })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    total = len(df)
    failed = df[df["initial_status"] == "failed"]
    failure_rate = len(failed) / total

    print(f"Generated {total:,} transactions")
    print(f"Overall first-attempt failure rate: {failure_rate:.1%} "
          f"(target range: 10-15%)")
    print()

    print("Decline code breakdown (failed transactions only):")
    print(failed["decline_code"].value_counts())
    print()

    retried = failed[~failed["is_voluntary_churn"]]
    print(f"Retryable failures (excludes voluntary churn): {len(retried):,}")
    print(f"Naive retry success rate: {retried['retry_success'].mean():.1%}")
    print()

    salary_window = retried[retried["is_salary_window"] == True]
    non_salary = retried[retried["is_salary_window"] == False]
    print(f"Retry success rate IN salary window: {salary_window['retry_success'].mean():.1%} "
          f"(n={len(salary_window)})")
    print(f"Retry success rate OUTSIDE salary window: {non_salary['retry_success'].mean():.1%} "
          f"(n={len(non_salary)})")


def main():
    df = generate_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved dataset to {OUTPUT_PATH}\n")
    print_summary(df)


if __name__ == "__main__":
    main()
