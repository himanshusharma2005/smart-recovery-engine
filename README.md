# Smart recovery engine

An AI-assisted revenue recovery system for failed recurring payments, built for
[Razorpay's hackathon](#) under Track 3: AI Revenue Recovery.

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

Being built day by day for the hackathon (deadline Sept 3, 2026). Progress log in
`docs/`.

- [x] Day 1: Project scaffolding + decline-code taxonomy
- [x] Day 2: Synthetic data generator
- [x] Day 3: EDA + feature engineering
- [x] Day 4: ML retry-success classifier
- [x] Day 5: Rule-based scheduler
- [ ] Day 6: Simulation engine (naive vs. smart)
- [ ] Day 7: Dashboard
- [ ] Day 8: Integration
- [ ] Day 9: Docs + demo video
- [ ] Day 10: Final test + submission

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/validate_taxonomy.py
```

## Project structure

```
smart-recovery-engine/
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
