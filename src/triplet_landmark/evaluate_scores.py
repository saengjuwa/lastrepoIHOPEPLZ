from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triplet similarity accuracy.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--positive-score-col", default="sim_anchor_positive")
    parser.add_argument("--negative-score-col", default="sim_anchor_negative")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total = 0
    correct = 0
    with args.scores.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {args.positive_score_col, args.negative_score_col}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Missing score columns in {args.scores}: {sorted(missing)}")

        for row in reader:
            positive_score = float(row[args.positive_score_col])
            negative_score = float(row[args.negative_score_col])
            if not math.isfinite(positive_score) or not math.isfinite(negative_score):
                raise ValueError("Score CSV contains NaN or infinity.")
            total += 1
            if positive_score > negative_score:
                correct += 1

    if total == 0:
        raise ValueError(f"No score rows found in {args.scores}")

    accuracy = correct / total
    wrong = total - correct
    print(f"accuracy={accuracy:.6f}")
    print(f"correct={correct}")
    print(f"wrong={wrong}")
    print(f"total={total}")


if __name__ == "__main__":
    main()
