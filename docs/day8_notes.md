# Day 8 - Integration and polish

## What was built

`run_pipeline.py` - a single script that runs all 5 pipeline steps in
order (data generation, feature engineering, model training, scheduling,
simulation), with progress messages and timing per step. Reduces "how do I
run this project" from 5 remembered commands in the right order down to
one.

## Friendly error handling added to every pipeline script

Before today, running a script out of order (e.g. `build_features.py`
before `generate_data.py`) produced a raw `FileNotFoundError` traceback -
technically correct, but not the kind of thing a judge trying this for the
first time should have to decode. Added an explicit check at the top of
`build_features.py`, `train_model.py`, `rule_scheduler.py`, and
`simulate_recovery.py`: if the required input file is missing, print
exactly which script to run first, and exit cleanly instead of crashing.

## The real test: clean-state end-to-end run

Deleted `data/generated/` entirely (every CSV, the trained model, the
chart images - everything regenerable) and ran `python run_pipeline.py`
from scratch. Result: identical output to every previous run - 13.2%
failure rate, ROC-AUC 0.739, the same 705/324/173/123 action breakdown,
same 48.5% -> 56.4% recovery lift - confirming the whole pipeline is
genuinely reproducible from nothing, not dependent on leftover state from
earlier days. Total runtime: under 30 seconds.

Also re-ran the dashboard's automated test suite (Streamlit's `AppTest`)
against this freshly rebuilt data to confirm it still loads without
errors - the dashboard and the pipeline are properly decoupled (dashboard
reads finished output files, doesn't regenerate anything itself), but
worth confirming they still agree with each other after a clean rebuild.

## README polish

Added a "Quickstart" section immediately after the intro - three commands,
nothing else, for someone who just wants to see it run. Moved the
step-by-step manual instructions (still useful if someone wants to run and
inspect one step at a time) into a "Detailed setup" section further down,
rather than making it the first thing a reader sees.

## Result

All 8 days of the pipeline - taxonomy, data generation, EDA, feature
engineering, model training, rule-based scheduling, simulation, and the
dashboard - now run end-to-end from a single command, verified from a
completely clean state.
