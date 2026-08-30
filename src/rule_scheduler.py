"""
Day 5: Rule-based scheduler.

Combines two things:
1. Hard domain rules that override the model when we already KNOW the
   answer (e.g. CARD_EXPIRED will not magically start working - no amount
   of clever timing fixes an actually-expired card)
2. The Day 4 model's predictions, used to search across several candidate
   retry days and pick whichever one has the highest predicted success
   probability - this is the actual "smart" part of Smart Recovery Engine

Produces one final action per failed, retryable transaction:
    - smart_retry_scheduled  (retry on the model's best predicted day)
    - whatsapp_nudge         (UPI mandate confirmation issues - nudge, don't blind-retry)
    - escalate_update_payment_method  (retrying would waste the attempt)
    - exclude_route_to_winback  (voluntary churn - not a recovery case at all)

Usage:
    python src/rule_scheduler.py
"""

from pathlib import Path

import joblib
import pandas as pd

TRANSACTIONS_PATH = Path(__file__).parent.parent / "data" / "generated" / "transactions.csv"
MODEL_PATH = Path(__file__).parent.parent / "data" / "generated" / "model.pkl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "generated" / "scheduled_actions.csv"

# Card network rule of thumb: issuers cap retries at a handful of attempts
# within a rolling window before treating further attempts as abusive.
# This dataset only models a single retry per transaction, so this constant
# doesn't bind yet - it's here as a documented real-world constraint the
# scheduler is designed to respect once multi-retry sequences are modeled.
MAX_RETRIES_PER_CYCLE = 4

# How many days ahead the scheduler is willing to search for a better retry
# day. Capped at 5 to match the 1-5 day retry-offset range used when the
# training data was generated (Day 2) - searching further than that would
# ask the model to extrapolate beyond what it actually learned from.
SEARCH_WINDOW_DAYS = 5

SALARY_WINDOW_DAYS = {1, 2, 3, 7, 8}

# Codes where retrying is pointless regardless of what the model predicts -
# these override the model entirely. A model trained on noisy synthetic data
# could theoretically learn a spurious "20% chance" for CARD_LOST_STOLEN;
# domain knowledge says that's not real signal worth trusting.
HARD_OVERRIDE_CODES = {"CARD_EXPIRED", "CARD_LOST_STOLEN", "INVALID_CARD_DETAILS"}

# This one gets its own action even though the model would say it's "worth
# retrying" - the real fix isn't a blind retry, it's getting the customer to
# open their UPI app and confirm the mandate. Retrying without that just
# fails again for the same reason.
NUDGE_INSTEAD_OF_RETRY = {"UPI_MANDATE_NOT_CONFIRMED"}

RETRY_THRESHOLD = 0.50


def load_model_bundle():
    return joblib.load(MODEL_PATH)


def build_feature_row(decline_code, decline_category, is_upi, charge_day,
                       candidate_day, subscription_amount, feature_cols):
    """Builds one feature row matching the exact column layout the model
    was trained on (Day 4), for a hypothetical retry on candidate_day."""
    row = {col: 0 for col in feature_cols}

    decline_col = f"decline_{decline_code}"
    category_col = f"category_{decline_category}"
    if decline_col in row:
        row[decline_col] = 1
    if category_col in row:
        row[category_col] = 1

    row["is_upi"] = int(is_upi)
    row["charge_day_of_month"] = charge_day
    row["retry_day_of_month"] = candidate_day
    is_salary = int(candidate_day in SALARY_WINDOW_DAYS)
    row["is_salary_window"] = is_salary
    funds_related = int(decline_code in {"INSUFFICIENT_FUNDS", "DO_NOT_HONOR_TEMP"})
    row["salary_window_x_funds_related"] = is_salary * funds_related
    row["days_since_failure"] = (candidate_day - charge_day) % 28
    row["subscription_amount"] = subscription_amount

    return row


def find_best_retry_day(model_bundle, decline_code, decline_category,
                         is_upi, charge_day, subscription_amount):
    """Searches candidate retry days 1..SEARCH_WINDOW_DAYS ahead, scores
    each with the trained model, returns the best day and its predicted
    probability."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    scaler = model_bundle["scaler"]
    numeric_cols = model_bundle["numeric_cols"]

    candidates = []
    for offset in range(1, SEARCH_WINDOW_DAYS + 1):
        candidate_day = ((charge_day - 1 + offset) % 28) + 1
        row = build_feature_row(decline_code, decline_category, is_upi,
                                  charge_day, candidate_day, subscription_amount,
                                  feature_cols)
        candidates.append((candidate_day, row))

    X_candidates = pd.DataFrame([c[1] for c in candidates])[feature_cols]
    X_candidates_scaled = X_candidates.copy()
    X_candidates_scaled[numeric_cols] = scaler.transform(X_candidates[numeric_cols])

    probs = model.predict_proba(X_candidates_scaled)[:, 1]
    best_idx = probs.argmax()

    return candidates[best_idx][0], probs[best_idx]


def decide_action(row, model_bundle):
    if row["initial_status"] == "success":
        return pd.Series({"action": "none", "scheduled_retry_day": None,
                           "predicted_success_prob": None})

    if row["is_voluntary_churn"]:
        return pd.Series({"action": "exclude_route_to_winback",
                           "scheduled_retry_day": None, "predicted_success_prob": None})

    decline_code = row["decline_code"]

    if decline_code in HARD_OVERRIDE_CODES:
        return pd.Series({"action": "escalate_update_payment_method",
                           "scheduled_retry_day": None, "predicted_success_prob": None})

    if decline_code in NUDGE_INSTEAD_OF_RETRY:
        return pd.Series({"action": "whatsapp_nudge",
                           "scheduled_retry_day": None, "predicted_success_prob": None})

    best_day, best_prob = find_best_retry_day(
        model_bundle, decline_code, row["decline_category"],
        row["payment_method"] == "upi", row["charge_day_of_month"],
        row["subscription_amount"]
    )

    if best_prob >= RETRY_THRESHOLD:
        return pd.Series({"action": "smart_retry_scheduled",
                           "scheduled_retry_day": best_day,
                           "predicted_success_prob": round(best_prob, 3)})
    else:
        return pd.Series({"action": "escalate_update_payment_method",
                           "scheduled_retry_day": None,
                           "predicted_success_prob": round(best_prob, 3)})


def main():
    df = pd.read_csv(TRANSACTIONS_PATH)
    for col in ["is_voluntary_churn", "is_salary_window", "retry_success"]:
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    model_bundle = load_model_bundle()

    print(f"Scheduling actions for {len(df):,} transactions "
          f"(searching up to {SEARCH_WINDOW_DAYS} days ahead per failure)...")

    decisions = df.apply(lambda row: decide_action(row, model_bundle), axis=1)
    result = pd.concat([df[["transaction_id", "decline_code", "decline_category",
                             "payment_method"]], decisions], axis=1)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}\n")

    print("Action breakdown:")
    print(result["action"].value_counts())
    print()

    print("Sample smart_retry_scheduled decisions:")
    smart = result[result["action"] == "smart_retry_scheduled"]
    print(smart.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
