"""Analyze MOS subjective listening-test responses.

Reads:
    survey_order.csv      — trial_id -> system / class mapping
    mos_responses.json    — one or more response files, each a list of
                            {trial_id, mos} dicts

Reports per-system MOS mean with 95% confidence intervals,
plus a per-class breakdown.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

SYSTEM_ORDER = ["gt", "audiogen", "llama", "rf"]


def load_order(path):
    trials = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            trials[int(row["trial_id"])] = row
    return trials


def load_responses(paths):
    all_responses = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        all_responses.append(data)
    return all_responses


def ci95(values):
    n = len(values)
    if n < 2:
        return 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    return 1.96 * se


def analyze(order, all_responses):
    by_system = defaultdict(list)
    by_system_class = defaultdict(lambda: defaultdict(list))

    for responses in all_responses:
        for r in responses:
            tid = r["trial_id"]
            if tid not in order:
                continue
            trial = order[tid]
            system = trial["system"]
            cls = trial["ebird_code"]
            by_system[system].append(r["mos"])
            by_system_class[system][cls].append(r["mos"])

    total_ratings = sum(len(v) for v in by_system.values())
    print(
        f"Total ratings: {total_ratings}  "
        f"({len(all_responses)} response file(s), {len(order)} trials)\n"
    )

    print("=" * 55)
    print(f"{'System':<12} {'MOS Mean':>10} {'± 95% CI':>10} {'N':>6}")
    print("-" * 55)
    for system in SYSTEM_ORDER:
        if system not in by_system:
            continue
        vals = np.array(by_system[system])
        print(f"{system:<12} {vals.mean():>10.2f} {ci95(vals):>10.2f} {len(vals):>6d}")
    print("=" * 55)

    print("\n\nPer-class breakdown:")
    for system in SYSTEM_ORDER:
        if system not in by_system_class:
            continue
        print(f"\n--- {system} ---")
        print(f"  {'Class':<12} {'MOS':>8} {'± CI':>8} {'N':>5}")
        for cls in sorted(by_system_class[system]):
            vals = np.array(by_system_class[system][cls])
            print(f"  {cls:<12} {vals.mean():>8.2f} {ci95(vals):>8.2f} {len(vals):>5d}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MOS listening-test responses",
    )
    parser.add_argument(
        "response_files",
        nargs="*",
        type=Path,
        default=[SCRIPT_DIR / "mos_responses.json"],
        help="One or more response JSON files",
    )
    parser.add_argument("--order", type=Path, default=SCRIPT_DIR / "survey_order.csv")
    args = parser.parse_args()

    order = load_order(args.order)
    responses = load_responses(args.response_files)
    analyze(order, responses)


if __name__ == "__main__":
    main()
