"""
Stratified train/val/test split for the processed ER-REASON dataset.

Reads a .jsonl file where each line is:
    {"id": int, "text": str, "label": int}

Produces three splits with the same label distribution:
    data/processed/er_reason/original_processed/er_reason_processed_train.jsonl
    data/processed/er_reason/original_processed/er_reason_processed_val.jsonl
    data/processed/er_reason/original_processed/er_reason_processed_test.jsonl

Usage
-----
python split_er_reason.py \\
    --input-path data/processed/er_reason/er_reason_processed.jsonl \\
    [--train-ratio 0.7] \\
    [--val-ratio 0.15] \\
    [--test-ratio 0.15] \\
    [--seed 42]
"""

import argparse
import json
import os
import sys
from collections import Counter

from sklearn.model_selection import train_test_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split for processed ER-REASON data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the processed ER-REASON .jsonl file.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {i + 1}: {e}", file=sys.stderr)
    return records


def _save_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _label_distribution(records: list[dict]) -> dict:
    counts = Counter(r["label"] for r in records)
    total = len(records)
    return {label: (count, count / total) for label, count in sorted(counts.items())}


def main() -> None:
    args = _parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        print(
            f"Train/val/test ratios must sum to 1.0, got {total_ratio:.4f}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load data
    print(f"Loading data from: {args.input_path}")
    records = _load_jsonl(args.input_path)
    print(f"Total records loaded: {len(records)}")

    # Extract labels for stratification
    labels = [r["label"] for r in records]

    # Print overall label distribution
    print("\nOverall label distribution:")
    for label, (count, frac) in _label_distribution(records).items():
        print(f"  Label {label}: {count} ({frac:.1%})")

    # First split: train vs (val + test)
    val_test_ratio = args.val_ratio + args.test_ratio
    train_records, val_test_records = train_test_split(
        records,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=args.seed,
    )

    # Second split: val vs test (from the val+test pool)
    val_test_labels = [r["label"] for r in val_test_records]
    relative_test_ratio = args.test_ratio / val_test_ratio
    val_records, test_records = train_test_split(
        val_test_records,
        test_size=relative_test_ratio,
        stratify=val_test_labels,
        random_state=args.seed,
    )

    # Output paths
    out_dir = "data/processed/er_reason/original_processed"
    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    # Save and report
    print()
    for split_name, split_records in splits.items():
        out_path = os.path.join(out_dir, f"er_reason_processed_{split_name}.jsonl")
        _save_jsonl(split_records, out_path)

        dist = _label_distribution(split_records)
        print(
            f"{split_name.upper()} split → {len(split_records)} records saved to: {out_path}"
        )
        for label, (count, frac) in dist.items():
            print(f"  Label {label}: {count} ({frac:.1%})")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
