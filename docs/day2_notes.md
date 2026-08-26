# Day 2 - Synthetic data generator

## What was built

`src/generate_data.py` - generates 10,000 synthetic subscription renewal
transactions using `data/decline_codes.json` as the probability source.
Output saved to `data/generated/transactions.csv` (gitignored - only the
generator script is version-controlled, not its output, since it's
regeneratable and shouldn't bloat repo history).

## Key design choices

**Decline codes are restricted by payment method.** A UPI transaction can't
fail with `CARD_EXPIRED`, so the generator splits the taxonomy into
card-valid and UPI-valid subsets and renormalizes weights within each,
rather than sampling from all 10 codes regardless of method.

**Voluntary churn is generated but excluded from retry simulation.**
`UPI_MANDATE_REVOKED` transactions get a decline_code and category, but no
retry is simulated for them (`retry_success` stays null) - consistent with
the Day 1 decision that this isn't involuntary churn and shouldn't inflate
recovery-rate metrics.

**Salary-window boost only applies to funds-related failures.** The +25%
success boost for retries landing on the 1st-3rd or 7th-8th of the month
only applies to `INSUFFICIENT_FUNDS` and `DO_NOT_HONOR_TEMP` - it would be
unrealistic for e.g. `CARD_EXPIRED` to suddenly succeed just because it's
payday, since an expired card doesn't care about account balance.

## Bug found and fixed

The first run showed a 0.1% retry success rate, obviously wrong. Root cause:
`rng.random() < final_prob` returns a **NumPy bool** (`np.True_`/`np.False_`),
not a Python `bool`. Because the `retry_success` column also contains `None`
for non-retried rows, pandas stored the whole column as dtype `object`
instead of a clean boolean/float column. NumPy bools use *saturating*
addition (`True + True` evaluates to `True`, not `2`), so `.mean()` and
`.sum()` on that object column silently produced meaningless results
instead of raising an error.

Fix: explicitly cast with `bool(...)` before storing, so the column holds
native Python booleans that behave normally under arithmetic even when
mixed with `None`.

This is worth remembering for Day 4 (model training) - always sanity-check
a `.mean()` on a small sample manually before trusting it at scale,
especially on any column with mixed types or nulls.

## Validation against real-world benchmarks

| Metric | Generated | Real-world benchmark |
|---|---|---|
| Overall first-attempt failure rate | 13.2% | 10-15% (Slicker) |
| Naive single-retry success rate | 48.5% | ~50-60% for manual/naive dunning (Recurly/industry) |
| Salary-window lift for funds-related failures | +5.9pp (53.4% vs 47.5%) | Directional only - real magnitude not publicly benchmarked, flagged as an estimate |

Numbers land close enough to published ranges to trust as a foundation.
The salary-window effect is intentionally modest rather than dramatic, so
Day 4's ML model has to actually learn the pattern rather than trivially
memorize an obvious signal.

## What's next (Day 3)

Exploratory data analysis notebook + feature engineering: turning raw
columns (`decline_code`, `charge_day_of_month`, `payment_method`) into
model-ready features (`is_salary_window`, `days_since_last_failure`,
`customer_retry_history`, one-hot encoded categories).
