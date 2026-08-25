"""
Day 1 sanity check: validates data/decline_codes.json before anything else
gets built on top of it. Run this first, every time you edit the taxonomy.

Usage:
    python src/validate_taxonomy.py
"""

import json
import sys
from pathlib import Path

TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "decline_codes.json"

REQUIRED_FIELDS = {
    "code",
    "category",
    "description",
    "occurrence_weight",
    "base_retry_success_prob",
    "notes",
    "recommended_action",
}

VALID_CATEGORIES = {"soft_decline", "hard_decline", "upi_specific"}


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def validate(taxonomy: dict) -> list[str]:
    errors = []
    codes = taxonomy.get("decline_codes", [])

    if not codes:
        errors.append("No decline_codes found in taxonomy file.")
        return errors

    seen_codes = set()
    total_weight = 0.0

    for entry in codes:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            errors.append(f"{entry.get('code', '???')}: missing fields {missing}")

        if entry.get("category") not in VALID_CATEGORIES:
            errors.append(f"{entry.get('code')}: invalid category '{entry.get('category')}'")

        code = entry.get("code")
        if code in seen_codes:
            errors.append(f"Duplicate decline code: {code}")
        seen_codes.add(code)

        prob = entry.get("base_retry_success_prob")
        if prob is not None and not (0.0 <= prob <= 1.0):
            errors.append(f"{code}: base_retry_success_prob {prob} out of [0,1] range")

        weight = entry.get("occurrence_weight", 0)
        total_weight += weight

    if abs(total_weight - 1.0) > 1e-6:
        errors.append(
            f"occurrence_weight values sum to {total_weight:.4f}, expected 1.0 "
            "(this will silently skew your synthetic dataset's realism)"
        )

    return errors


def main():
    taxonomy = load_taxonomy()
    errors = validate(taxonomy)

    print(f"Loaded {len(taxonomy['decline_codes'])} decline codes from {TAXONOMY_PATH.name}\n")

    by_category = {}
    for entry in taxonomy["decline_codes"]:
        by_category.setdefault(entry["category"], []).append(entry["code"])

    for cat, codes in by_category.items():
        print(f"  {cat}: {len(codes)} codes -> {codes}")

    print()
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Validation passed. Taxonomy is ready to feed the data generator (Day 2).")


if __name__ == "__main__":
    main()
