"""
Day 3, part 1: Exploratory data analysis on the synthetic transaction dataset.

Answers three questions before any modeling starts:
1. Does the decline-code distribution roughly match what we designed in the taxonomy?
2. Which decline reasons are actually worth retrying (vs. wasting a retry on)?
3. Is the salary-window timing signal visible enough to be learnable?

Usage:
    python src/eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNG files
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "generated" / "transactions.csv"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # CSV round-trip turns True/False into strings in some pandas versions - normalize explicitly
    for col in ["is_voluntary_churn", "is_salary_window", "retry_success"]:
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    return df


def chart_decline_distribution(failed: pd.DataFrame):
    counts = failed["decline_code"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Number of failed transactions")
    ax.set_title("Decline code distribution (failed transactions only)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "decline_code_distribution.png", dpi=120)
    plt.close(fig)


def chart_retry_success_by_code(retried: pd.DataFrame):
    success_rate = retried.groupby("decline_code")["retry_success"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if v < 0.2 else "#DD8452" if v < 0.5 else "#55A868"
              for v in success_rate.values]
    success_rate.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Retry success rate")
    ax.set_title("Retry success rate by decline code (red = don't bother retrying)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "retry_success_by_code.png", dpi=120)
    plt.close(fig)


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    failed = df[df["initial_status"] == "failed"]
    retried = failed[failed["is_voluntary_churn"] == False]

    print("=" * 60)
    print("DECLINE CODE DISTRIBUTION")
    print("=" * 60)
    print(failed["decline_code"].value_counts())
    print()

    print("=" * 60)
    print("RETRY SUCCESS RATE BY DECLINE CODE (sorted, worst to best)")
    print("=" * 60)
    print(retried.groupby("decline_code")["retry_success"].mean().sort_values())
    print()

    print("=" * 60)
    print("KEY FINDING: which codes are worth retrying at all?")
    print("=" * 60)
    rates = retried.groupby("decline_code")["retry_success"].mean().sort_values()
    not_worth_it = rates[rates < 0.15]
    worth_it = rates[rates >= 0.15]
    print(f"NOT worth blind retry (success < 15%): {list(not_worth_it.index)}")
    print(f"  -> these should go straight to escalation, wasting a retry costs")
    print(f"     network fees and burns a merchant's limited retry attempts")
    print(f"Worth retrying: {list(worth_it.index)}")
    print()

    print("=" * 60)
    print("SALARY WINDOW EFFECT (funds-related codes only)")
    print("=" * 60)
    funds_related = retried[retried["decline_code"].isin(["INSUFFICIENT_FUNDS", "DO_NOT_HONOR_TEMP"])]
    by_window = funds_related.groupby("is_salary_window")["retry_success"].agg(["mean", "count"])
    print(by_window)
    print()

    print(f"Saving charts to {IMAGES_DIR}/ ...")
    chart_decline_distribution(failed)
    chart_retry_success_by_code(retried)
    print("Done: decline_code_distribution.png, retry_success_by_code.png")


if __name__ == "__main__":
    main()
