# Day 5 - Rule-based scheduler

## What was built

`src/rule_scheduler.py` - combines hard domain rules with the Day 4 model's
predictions to produce one final action per transaction. For retryable
failures, it searches candidate retry days and asks the model to score
each one, then picks the best day rather than using a fixed offset.

## The decision logic, in order

1. **Success transactions** -> no action needed
2. **Voluntary churn** (`UPI_MANDATE_REVOKED`) -> excluded, routed to a
   hypothetical win-back flow instead of retry
3. **Hard override codes** (`CARD_EXPIRED`, `CARD_LOST_STOLEN`,
   `INVALID_CARD_DETAILS`) -> always escalate, **regardless of what the
   model predicts**. This is a deliberate rule that overrides the ML
   model - if a card is genuinely expired, no clever retry timing fixes
   that, and trusting a model's output here would just waste a retry
   attempt on a lost cause. Domain knowledge beats a statistical model
   when the answer is already certain.
4. **`UPI_MANDATE_NOT_CONFIRMED`** -> WhatsApp nudge instead of a blind
   retry, since the real fix is getting the customer to open their UPI app,
   not retrying a charge against an unconfirmed mandate
5. **Everything else** (soft declines, other UPI-specific failures) -> the
   model searches candidate retry days and schedules the one with the
   highest predicted success probability, or escalates if even the best
   day falls below a 50% threshold

## A real bug I found and fixed today (not a design choice - an actual mistake)

The first version scheduled `txn_000017` (an `INSUFFICIENT_FUNDS` failure)
for day 9, skipping day 8 - even though day 8 is inside the salary window
and day 9 isn't. Investigated by printing the model's raw coefficients:
`is_salary_window` had learned a near-zero (even slightly negative) effect
overall.

Root cause: the salary-window boost was only ever designed (Day 2) to help
funds-related failures specifically. But the model only had a single global
`is_salary_window` column, applied across all 8 decline codes. Since salary
timing genuinely doesn't matter for 6 of them, the real effect got diluted
into near-nothing when the model tried to learn one flat coefficient across
the whole dataset.

**Fix:** added an interaction feature,
`salary_window_x_funds_related = is_salary_window * is_funds_related_code`,
to `build_features.py`, and retrained. Result: `is_salary_window` alone
dropped to -0.357 (no real effect on its own), while the new interaction
term picked up +0.453 - the model now correctly learned that salary timing
only matters for the codes it should matter for. Re-ran the scheduler:
`txn_000017` now correctly picks day 8. ROC-AUC also nudged up slightly
(0.736 -> 0.739) as a side benefit.

This is worth explaining in an interview as a genuine debugging story: the
scheduler's own output caught a modeling gap that the training metrics
alone didn't surface - accuracy and ROC-AUC looked fine, but a specific,
checkable prediction was wrong for an explainable reason.

## Also capped the search window at 5 days, not 7

The first draft searched 7 days ahead for the best retry day. But Day 2's
generator only ever simulated retry offsets of 1-5 days - asking the model
to score day 6 or 7 means asking it to extrapolate beyond what it actually
saw in training, which isn't reliable. Capped `SEARCH_WINDOW_DAYS` at 5 to
match the training distribution exactly.

## Results

| Action | Count | What it means |
|---|---|---|
| none (success) | 8,675 | No failure, nothing to do |
| smart_retry_scheduled | 705 | Model found a day worth retrying on |
| whatsapp_nudge | 324 | UPI mandate needs manual confirmation |
| escalate_update_payment_method | 173 | Retrying would be pointless |
| exclude_route_to_winback | 123 | Voluntary churn, not a recovery case |

Cross-checked every decline code against its assigned action - every hard
decline routes to escalation with zero exceptions, every
`UPI_MANDATE_NOT_CONFIRMED` routes to a nudge, every `UPI_MANDATE_REVOKED`
is excluded. `DO_NOT_HONOR_TEMP` (an intentionally ambiguous code) is the
only one that splits between retry and escalate based on the model's
day-by-day prediction - which is exactly the kind of case-by-case judgment
a naive one-size-fits-all retry policy can't make.

## What's next (Day 6)

Build the simulation engine: run a naive strategy (retry everything once,
same way, no rules) against this smart scheduler's decisions on the same
10,000 transactions, and measure the actual recovery-rate lift - this is
the headline number the whole project has been building toward.
