# Day 1 - Project setup and decline-code taxonomy

## What was done today

1. Set up project structure (`data/`, `src/`, `notebooks/`, `dashboard/`, `tests/`, `docs/`)
2. Designed the decline-code taxonomy (`data/decline_codes.json`) - 10 decline
   reasons across three categories: soft decline, hard decline, UPI-specific
3. Wrote and ran a validation script (`src/validate_taxonomy.py`) to catch
   structural mistakes in the taxonomy before building anything on top of it

## Design decisions and why

**Why three categories instead of just soft/hard?**
Most Western dunning tools (Stripe, Chargebee) only distinguish soft vs. hard
declines because they're card-first. Since UPI AutoPay is the dominant recurring
payment method in India and behaves fundamentally differently (mandate-based, not
card-based), it needed its own category with its own recommended actions
(WhatsApp nudge instead of blind retry). This is the deliberate India-specific
angle of the project, matching what Razorpay is actually building.

**Why does `UPI_MANDATE_REVOKED` have `base_retry_success_prob: 0.00` and get
excluded from the recovery funnel?**
This is a case where the customer *actively* cancelled - that's voluntary churn,
not involuntary. Including it in the "recoverable revenue" pool would be
misleading and would inflate my recovery numbers artificially. Excluding it and
routing it to a hypothetical win-back flow instead keeps the project's core
metric (involuntary churn recovery rate) honest.

**Where did the occurrence weights and base retry-success probabilities come from?**
Calibrated against published, cited figures:
- Involuntary churn is 20-40% of total churn (ProfitWell, via ChurnWard)
- ~10-15% of recurring payments fail on first attempt (Slicker)
- Soft declines are highly recoverable via retry alone; hard declines require
  customer action (industry-standard card network classification)
- UPI-specific failure modes are the least standardized publicly, so those
  numbers are a reasoned estimate rather than a cited figure - flagged
  explicitly as such in the taxonomy notes field, not hidden

This will be re-validated against the synthetic dataset's actual output on
Day 2 - if the generated failure rate doesn't land close to the 10-15%
industry range, the weights get revisited before building anything else on
top of them.

## What's next (Day 2)

Build `src/generate_data.py` to produce ~10,000 synthetic subscription
transactions using this taxonomy as the underlying probability distribution.
