# Day 3 - EDA and feature engineering

## What was built

- `src/eda.py` - explores `transactions.csv`, prints key stats, saves two
  charts to `docs/images/`
- `src/build_features.py` - transforms the raw retryable-failure rows into
  a clean, model-ready table saved to `data/generated/features.csv`

## Key findings from EDA

**Decline codes split cleanly into three retry-worthiness tiers:**
- Near-zero success (don't bother retrying): `CARD_LOST_STOLEN` (0%),
  `CARD_EXPIRED` (1.1%), `INVALID_CARD_DETAILS` (3.1%)
- Medium success: `UPI_MANDATE_NOT_CONFIRMED` (41%), `DO_NOT_HONOR_TEMP` (49%)
- High success: `INSUFFICIENT_FUNDS` (59%), `UPI_PSP_APP_ERROR` (61%),
  `BANK_SERVER_DOWN` (75%), `ISSUER_TIMEOUT` (76%)

This is the single most important finding in the whole project: **a naive
system retries all of these the same way, wasting retry attempts on
transactions that had essentially no chance of succeeding.** This is
exactly the gap the Smart Recovery Engine is meant to close, and now it's
backed by data, not just a claim from the taxonomy design.

**Salary-window effect confirmed on funds-related codes:** 54.4% success
outside the window vs. 66.1% inside it, on `INSUFFICIENT_FUNDS` and
`DO_NOT_HONOR_TEMP` transactions specifically. This gap is bigger than the
overall aggregate salary-window gap reported on Day 2 (which was diluted by
non-funds-related codes) - this is a clearer, more honest way to report the
finding since it isolates the mechanism the effect is actually based on.

See `docs/images/decline_code_distribution.png` and
`docs/images/retry_success_by_code.png` for the visuals.

## Feature table design

Deliberately scoped to only the 1,202 retryable failed transactions -
successful transactions have nothing to predict, and voluntary churn
(`UPI_MANDATE_REVOKED`) isn't retried by design (Day 1 decision), so
including either would just add noise the model has to learn to ignore.

**Features included:**
- One-hot encoded `decline_code` (most predictive - the EDA chart makes
  clear why) and `decline_category` as a coarser backup signal
- `is_upi` binary flag
- `is_salary_window` binary flag
- `days_since_failure` (retry timing gap)
- `subscription_amount` (kept raw, not bucketed)

**Deliberately excluded:** a "customer's past failure count" feature. This
would be a genuinely useful real-world signal, but this synthetic dataset
only models a single billing cycle per customer with day-of-month values,
not real calendar dates across multiple months - so there's no honest way
to compute "past retry history" without fabricating a pattern that isn't
actually grounded in anything. Documented here rather than silently
skipped, in case this comes up in an interview - the honest answer is "the
dataset's scope didn't support it credibly, so I left it out rather than
fake it."

**Target balance:** 51.5% failure / 48.5% success on retry - close to
50/50, so Day 4's classifier can be trained without needing class-imbalance
techniques (SMOTE, class weighting, etc.) - one less thing to get wrong
under time pressure.

## What's next (Day 4)

Train the ML retry-success classifier on `features.csv` - start with
Logistic Regression as a baseline (easy to explain, coefficients are
interpretable), then compare against a Random Forest to see if the
non-linear model meaningfully outperforms it.
