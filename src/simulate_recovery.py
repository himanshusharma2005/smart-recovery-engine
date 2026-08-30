"""
Day 6: Simulation engine - the headline result of the whole project.

Compares two strategies on the exact same 10,000 transactions:

NAIVE strategy: blind retry, same way, regardless of decline reason. This
is literally what got generated in Day 2 - a retry attempt 1-5 days later,
outcome already stored in transactions.csv. No decline-code awareness at all.

SMART strategy: whatever the Day 5 scheduler decided - retry on the model's
best predicted day, WhatsApp nudge, escalate, or exclude - each with its
own, independently re-simulated outcome.

Runs as a Monte Carlo simulation (many repeated trials) rather than a
single run, because the smart strategy's outcomes involve randomness
(the retry re-simulation, and the nudge/escalation assumptions below).
A single trial could get lucky or unlucky; averaging over many trials
gives a stable, defensible number.

Usage:
    python src/simulate_recovery.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate_data import load_taxonomy, simulate_retry_success  # noqa: E402

TRANSACTIONS_PATH = Path(__file__).parent.parent / "data" / "generated" / "transactions.csv"
SCHEDULED_PATH = Path(__file__).parent.parent / "data" / "generated" / "scheduled_actions.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "generated" / "simulation_results.csv"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"

N_TRIALS = 200  # Monte Carlo repetitions

# These two numbers are NOT measured from any dataset - they're documented
# assumptions, since no public data exists for "conversion rate of a
# WhatsApp nudge" or "conversion rate of an update-payment-method prompt"
# at the level of specificity this project needs. Kept deliberately
# conservative rather than picking numbers that flatter the result.
NUDGE_SUCCESS_ASSUMPTION = 0.55       # UPI_MANDATE_NOT_CONFIRMED -> WhatsApp nudge
ESCALATION_SUCCESS_ASSUMPTION = 0.25  # hard declines -> update payment method prompt


def load_and_merge():
    txns = pd.read_csv(TRANSACTIONS_PATH)
    for col in ["is_voluntary_churn", "is_salary_window", "retry_success"]:
        txns[col] = txns[col].map({"True": True, "False": False, True: True, False: False})

    scheduled = pd.read_csv(SCHEDULED_PATH)
    merged = txns.merge(
        scheduled[["transaction_id", "action", "scheduled_retry_day", "predicted_success_prob"]],
        on="transaction_id", how="left"
    )
    return merged


def run_one_trial(df, taxonomy_by_code, seed):
    """Runs one Monte Carlo trial. Returns naive and smart recovered revenue
    for the retryable failure pool (matches Day 4's training scope: excludes
    successes and voluntary churn from the comparison, since neither strategy
    can or should touch those). Also tracks recovery broken down by action
    type, so the final lift can be explained, not just reported."""
    rng = np.random.default_rng(seed)

    retryable = df[df["action"].isin(
        ["smart_retry_scheduled", "whatsapp_nudge", "escalate_update_payment_method"]
    )].copy()

    naive_recovered = retryable.loc[retryable["retry_success"] == True, "subscription_amount"].sum()
    naive_recovered_count = int((retryable["retry_success"] == True).sum())

    smart_recovered = 0.0
    smart_recovered_count = 0
    by_action = {"smart_retry_scheduled": [0, 0], "whatsapp_nudge": [0, 0],
                 "escalate_update_payment_method": [0, 0]}  # [recovered_count, total_count]

    for _, row in retryable.iterrows():
        action = row["action"]
        by_action[action][1] += 1

        if action == "smart_retry_scheduled":
            decline_entry = taxonomy_by_code[row["decline_code"]]
            success = simulate_retry_success(decline_entry, row["scheduled_retry_day"], rng)
        elif action == "whatsapp_nudge":
            success = rng.random() < NUDGE_SUCCESS_ASSUMPTION
        elif action == "escalate_update_payment_method":
            success = rng.random() < ESCALATION_SUCCESS_ASSUMPTION
        else:
            success = False

        if success:
            smart_recovered += row["subscription_amount"]
            smart_recovered_count += 1
            by_action[action][0] += 1

    return {
        "n_retryable": len(retryable),
        "naive_recovered_revenue": naive_recovered,
        "naive_recovered_count": naive_recovered_count,
        "smart_recovered_revenue": smart_recovered,
        "smart_recovered_count": smart_recovered_count,
        "retry_recovered": by_action["smart_retry_scheduled"][0],
        "retry_total": by_action["smart_retry_scheduled"][1],
        "nudge_recovered": by_action["whatsapp_nudge"][0],
        "nudge_total": by_action["whatsapp_nudge"][1],
        "escalate_recovered": by_action["escalate_update_payment_method"][0],
        "escalate_total": by_action["escalate_update_payment_method"][1],
    }


def main():
    if not SCHEDULED_PATH.exists():
        print(f"ERROR: {SCHEDULED_PATH.name} not found.")
        print(f"Run this first: python src/rule_scheduler.py")
        raise SystemExit(1)

    df = load_and_merge()
    taxonomy = load_taxonomy()
    taxonomy_by_code = {c["code"]: c for c in taxonomy}

    print(f"Running {N_TRIALS} Monte Carlo trials...\n")

    trials = [run_one_trial(df, taxonomy_by_code, seed=1000 + i) for i in range(N_TRIALS)]
    results = pd.DataFrame(trials)
    results.to_csv(OUTPUT_PATH, index=False)

    n_retryable = results["n_retryable"].iloc[0]

    naive_rate_mean = (results["naive_recovered_count"] / n_retryable).mean()
    smart_rate_mean = (results["smart_recovered_count"] / n_retryable).mean()
    smart_rate_std = (results["smart_recovered_count"] / n_retryable).std()

    naive_rev_mean = results["naive_recovered_revenue"].mean()
    smart_rev_mean = results["smart_recovered_revenue"].mean()
    smart_rev_std = results["smart_recovered_revenue"].std()

    print("=" * 60)
    print(f"RETRYABLE FAILED TRANSACTIONS: {n_retryable:,}")
    print("=" * 60)
    print()
    print(f"NAIVE strategy (blind retry, same for every decline reason):")
    print(f"  Recovery rate:     {naive_rate_mean:.1%}")
    print(f"  Revenue recovered: Rs {naive_rev_mean:,.0f}")
    print()
    print(f"SMART strategy (decline-aware retry + rules + ML timing):")
    print(f"  Recovery rate:     {smart_rate_mean:.1%}  (std across {N_TRIALS} trials: {smart_rate_std:.1%})")
    print(f"  Revenue recovered: Rs {smart_rev_mean:,.0f}  (std: Rs {smart_rev_std:,.0f})")
    print()
    print("=" * 60)
    lift_pp = (smart_rate_mean - naive_rate_mean) * 100
    lift_rev = smart_rev_mean - naive_rev_mean
    lift_pct = (lift_rev / naive_rev_mean) * 100
    print(f"RESULT: Smart Recovery Engine recovers {lift_pp:+.1f} percentage points")
    print(f"        more successfully than naive blind retry.")
    print(f"        That's Rs {lift_rev:,.0f} more revenue recovered on this dataset")
    print(f"        ({lift_pct:+.1f}% more than naive) out of {n_retryable:,} retryable failures.")
    print("=" * 60)
    print()
    print("Where the lift comes from (avg. success rate per action, this run):")
    for action, label in [("retry", "smart_retry_scheduled (timed retry)"),
                            ("nudge", "whatsapp_nudge"),
                            ("escalate", "escalate_update_payment_method")]:
        total = results[f"{action}_total"].iloc[0]
        rate = (results[f"{action}_recovered"] / total).mean()
        print(f"  {label:45s} n={total:>4}  success rate={rate:.1%}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(["Naive", "Smart"], [naive_rate_mean * 100, smart_rate_mean * 100],
                color=["#C44E52", "#55A868"])
    axes[0].set_ylabel("Recovery rate (%)")
    axes[0].set_title("Recovery rate: naive vs smart")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate([naive_rate_mean * 100, smart_rate_mean * 100]):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center")

    axes[1].bar(["Naive", "Smart"], [naive_rev_mean, smart_rev_mean],
                color=["#C44E52", "#55A868"])
    axes[1].set_ylabel("Revenue recovered (Rs)")
    axes[1].set_title("Revenue recovered: naive vs smart")
    for i, v in enumerate([naive_rev_mean, smart_rev_mean]):
        axes[1].text(i, v + 5000, f"Rs {v:,.0f}", ha="center")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "naive_vs_smart_comparison.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved comparison chart to {IMAGES_DIR / 'naive_vs_smart_comparison.png'}")


if __name__ == "__main__":
    main()
