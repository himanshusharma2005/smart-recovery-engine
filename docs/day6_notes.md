# Day 6 - Simulation engine (the headline result)

## What was built

`src/simulate_recovery.py` - runs a 200-trial Monte Carlo simulation
comparing the naive strategy (blind retry, same for every decline reason -
literally what Day 2 generated) against the smart strategy (Day 5's
scheduler: decline-aware retry timing, WhatsApp nudges, and escalation).

## The headline number

| Metric | Naive | Smart | Lift |
|---|---|---|---|
| Recovery rate | 48.5% | 56.4% | **+7.9 percentage points** |
| Revenue recovered (this dataset) | Rs 4,12,517 | Rs 4,79,578 | **+Rs 67,061 (+16.3%)** |

Standard deviation across 200 trials: 1.5 percentage points on recovery
rate. The lift is stable, not a lucky single run.

## Why Monte Carlo instead of one run

The smart strategy's outcome involves randomness in three places: the
re-simulated retry outcome on the model's chosen day, the WhatsApp nudge
assumption, and the escalation assumption. A single run could land
favorably or unfavorably by chance. Running 200 independent trials and
reporting the mean and standard deviation is the honest way to report a
result that includes randomness - the alternative (reporting one lucky run)
would not hold up if someone asked "is this reproducible?"

## Where the lift actually comes from - broken down by action

| Action | Count | Success rate |
|---|---|---|
| Timed retry (smart_retry_scheduled) | 705 | 64.9% |
| WhatsApp nudge | 324 | 54.8% |
| Escalate to update payment method | 173 | 24.9% |

This breakdown matters more than the headline number for explaining the
project honestly. The escalation and nudge rates (24.9%, 54.8%) land almost
exactly on the assumptions coded into the simulation (25%, 55%) - which is
a sanity check that the simulation logic works correctly, but it also means
**a meaningful share of the total lift depends on two assumptions this
project could not validate against real data**, not purely on the ML
model's timing predictions.

## Being honest about what's assumption vs. what's measured

This is the most important thing to say clearly if asked about this
project's results:

- **Measured, not assumed:** the timed-retry success rate (64.9%) comes
  directly from the taxonomy's base probabilities plus the ML model's
  day-selection logic - this part is grounded in the project's own data
  and modeling work.
- **Assumed, clearly documented, not measured:** the 55% WhatsApp nudge
  conversion rate and 25% escalation conversion rate. No public dataset
  gives an exact number for "how often does a customer respond to a
  payment-update prompt", so these were set as clearly-labeled, deliberately
  conservative assumptions rather than optimistic ones chosen to inflate
  the result. Real-world A/B testing against Razorpay's actual traffic
  would be needed to validate or correct these two numbers specifically.

Stating this proactively, rather than waiting to be asked, is the honest
and technically credible way to present this project's numbers.

## What's next (Day 7)

Build the Streamlit dashboard - this is where all of Days 1-6 finally
becomes something a judge or interviewer can interact with directly,
instead of reading terminal output and static images.
