# Day 7 - The dashboard

## What was built

`dashboard/app.py` - a 5-tab Streamlit dashboard, the live demo interface
for the hackathon. Built with a fintech-style theme (Razorpay-blue accents,
clean metric cards) and Plotly for interactive charts instead of static
images.

## The 5 tabs

1. **Overview** - headline metrics (naive vs. smart recovery rate and
   revenue), a comparison chart, and a pie chart of what the engine decided
   across all 10,000 transactions
2. **Decline Insights** - the retry-worthiness chart from Day 3, plus the
   salary-window timing effect, made interactive
3. **Live Simulator** - the centerpiece. Pick a payment method, decline
   reason, charge day, and amount, and watch the actual scheduler logic
   (imported directly from `rule_scheduler.py`, not reimplemented) run live
   and explain its own decision
4. **Transaction Explorer** - filter/search across the real 10,000-row
   dataset by action or transaction ID
5. **Methodology** - states plainly which numbers are measured vs. assumed,
   rather than burying that in a docs file nobody reads during a demo

## Why the Live Simulator matters most

Anyone can show static charts. The simulator is the only part of the
dashboard that proves the system's logic actually runs and reasons about
inputs it hasn't seen before - not just replaying pre-computed results.
This is the single highest-value thing to demo live: pick an unusual
combination (e.g. `INSUFFICIENT_FUNDS`, charge day 28, upi) and show the
judges the engine finding the best retry day in real time, or pick
`CARD_LOST_STOLEN` and show the hard-rule override kicking in instead of
trusting the model.

## Testing approach

Rather than eyeballing the dashboard once and assuming it works, tested it
properly using Streamlit's `AppTest` framework, which runs the actual
script and reports real exceptions:

- Full script run: no exceptions
- Clicked the simulator's main button programmatically: no exceptions
- Cycled through **all 10 decline codes across both payment methods**
  (20 combinations total) and clicked "run" for each: all 20 passed cleanly

This matters because the simulator has several different code paths (hard
override, WhatsApp nudge, voluntary churn exclusion, and the ML search
path) - a single manual click-through could easily miss one. Testing every
combination programmatically means every button click a judge could
possibly make during a live demo has already been verified to work.

## One real fix made today

Streamlit's `use_container_width` parameter is deprecated (removal
deadline already passed as of this project's build date) in favor of
`width='stretch'`. Caught this via a deprecation warning during testing
and fixed all 5 occurrences before it could break on a newer Streamlit
version - small, but exactly the kind of detail that matters for a
dashboard meant to run reliably during a live demo in front of judges.

## How to run it

```bash
streamlit run dashboard/app.py
```

Opens in the browser automatically. Requires all previous pipeline scripts
to have been run at least once (`generate_data.py` through
`simulate_recovery.py`), since the dashboard reads their output files
rather than regenerating anything itself - this keeps the dashboard fast
and avoids re-running the Monte Carlo simulation on every page load.

## What's next (Day 8)

Integration and polish - make sure the whole pipeline runs end-to-end with
a single command, fix any rough edges, and do a final pass on error
handling for anyone running this for the first time on their own machine.
