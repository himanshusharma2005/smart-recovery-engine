# Smart recovery engine

An AI-assisted revenue recovery system for failed recurring payments, built for
[Razorpay's hackathon](#) under Track 3: AI Revenue Recovery.

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run dashboard/app.py
```

That's it - `run_pipeline.py` runs all 5 pipeline steps in order (data
generation, feature engineering, model training, scheduling, simulation)
and reports progress as it goes. Takes about 20-30 seconds. The dashboard
then opens in your browser automatically.

## The problem

Involuntary churn — a paying customer being silently dropped because a recurring
payment failed for a boring, fixable reason (expired card, bank timeout,
insufficient funds) — accounts for 20-40% of total subscription churn industry-wide.
Most systems retry every failure the same way: one generic retry, one generic email.
That leaves a large, recoverable share of revenue on the table.

Razorpay has publicly identified this as a priority: their Subscription Recovery
Agent (built with ElevenLabs) is named as their highest-impact Agent Studio product,
and their Sprint 2026 release included a WhatsApp-based nudge system specifically
for UPI AutoPay mandate failures - a failure mode global tools like Stripe Billing
don't natively handle, since UPI AutoPay is India-specific.

## The result

Simulated on 1,202 retryable failed transactions (10,000-transaction synthetic dataset):

| | Naive (blind retry) | Smart Recovery Engine |
|---|---|---|
| Recovery rate | 48.5% | **56.4%** |
| Revenue recovered | Rs 4,12,517 | **Rs 4,79,578** |

**+7.9 percentage points, +16.3% more revenue recovered** - averaged over
200 Monte Carlo trials (std dev: 1.5pp), not a single lucky run. Full
breakdown of where the lift comes from, including which numbers are
measured vs. assumed, in `docs/day6_notes.md`.

## What this project does

Instead of retrying every failed payment the same way, this engine:
1. Classifies *why* a payment failed (decline-code taxonomy, see `data/decline_codes.json`)
2. Predicts the probability a retry will succeed, using a trained classifier
3. Applies India-aware scheduling rules (salary-credit windows, card network retry
   caps, UPI-specific escalation via WhatsApp-style nudges instead of blind retries)
4. Simulates naive vs. smart recovery strategies on the same synthetic dataset and
   reports the actual recovery-rate lift

## Why synthetic data

Real transaction-level retry logs (decline code -> retry attempt -> outcome) are
not publicly available for good reason - they're commercially sensitive. This
project generates a synthetic dataset whose decline-code distribution and base
recovery rates are calibrated against published industry benchmarks (Slicker,
ChurnWard, Chargebee, Razorpay's own blog posts), documented in
`data/decline_codes.json`. This mirrors the real constraint Razorpay's own Vulcan
model faces internally - no public dataset replicates transaction-level retry data,
so any credible system in this space has to be built and validated against
first-party or synthetic data.

## Project status

Built day by day for the hackathon. Full build log and design reasoning
for each day in `docs/`.

- [x] Day 1: Project scaffolding + decline-code taxonomy
- [x] Day 2: Synthetic data generator
- [x] Day 3: EDA + feature engineering
- [x] Day 4: ML retry-success classifier
- [x] Day 5: Rule-based scheduler
- [x] Day 6: Simulation engine (naive vs. smart)
- [x] Day 7: Dashboard
- [x] Day 8: Integration + polish

## Detailed setup

If you'd rather run each pipeline step individually instead of
`run_pipeline.py` (useful when iterating on one step):

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/validate_taxonomy.py
python src/generate_data.py
python src/eda.py
python src/build_features.py
python src/train_model.py
python src/rule_scheduler.py
python src/simulate_recovery.py
streamlit run dashboard/app.py
```

Each script checks that its required input exists and tells you exactly
which earlier script to run if something's missing, rather than crashing
with an unclear error.

## Project structure

```
smart-recovery-engine/
├── run_pipeline.py       one-command runner for the full pipeline
├── data/                 decline code taxonomy + generated synthetic data
├── src/                  core logic: data gen, features, model, rules, simulation
├── notebooks/            EDA and model training notebooks
├── dashboard/            Streamlit app
├── tests/                unit tests
└── docs/                 day-by-day build notes and design decisions
```

## Tech stack

Python, pandas, scikit-learn, Streamlit. No external services or paid APIs required
to run the demo end-to-end.
