"""
Day 8: Run the entire pipeline with one command.

Runs, in order: generate_data -> build_features -> train_model ->
rule_scheduler -> simulate_recovery. Each step's output feeds the next,
so order matters - this script exists so nobody (including a judge trying
this for the first time) has to remember that order or run five separate
commands.

Usage:
    python run_pipeline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

STEPS = [
    ("Generating synthetic transaction data", "generate_data"),
    ("Engineering features", "build_features"),
    ("Training the ML retry-success classifier", "train_model"),
    ("Running the rule-based scheduler", "rule_scheduler"),
    ("Simulating naive vs. smart recovery (200 trials)", "simulate_recovery"),
]


def run_step(description, module_name):
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print("=" * 60)

    start = time.time()
    try:
        module = __import__(module_name)
        module.main()
    except FileNotFoundError as e:
        print(f"\nPIPELINE STOPPED: a required input file is missing.")
        print(f"  {e}")
        print(f"  This usually means an earlier step didn't complete. "
              f"Try running the pipeline from the start again.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPIPELINE STOPPED: '{module_name}' raised an unexpected error.")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n-- done in {elapsed:.1f}s --")


def main():
    print("Smart Recovery Engine - full pipeline")
    print(f"Running {len(STEPS)} steps in order...\n")

    overall_start = time.time()
    for description, module_name in STEPS:
        run_step(description, module_name)

    overall_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {overall_elapsed:.1f}s")
    print("=" * 60)
    print("\nAll data, model, and simulation files are ready.")
    print("Launch the dashboard with:")
    print("\n    streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
