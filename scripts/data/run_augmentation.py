"""
CLI entrypoint for the dataset augmentation pipeline.

Combines original CoT profiles with selected counterfactual profiles to
produce class-balanced augmented datasets for student model training.

Normal run (LD1 val):
    python scripts/data/run_augmentation.py \\
        --dataset ld1 \\
        --split val \\
        --teacher-model-suff Qwen3-4b \\
        --run-name aug_ld1_val

Dry run (print manifest only, no files written):
    python scripts/data/run_augmentation.py \\
        --dataset ld1 \\
        --split val \\
        --teacher-model-suff Qwen3-4b \\
        --run-name aug_ld1_val_dry \\
        --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.augment_dataset import run_augmentation
from utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a dataset split with selected counterfactual profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["er_reason", "ld1"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--teacher-model-suff",
        type=str,
        required=True,
        help="Model suffix for file naming (e.g. Qwen3-4b).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip output file writing; print manifest to console only.",
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=0.05,
        help="Tolerance for ratio matching (±).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle reproducibility.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Base data directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Logger file name prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"augment_{args.dataset}_{args.split}"
    setup_logging(run_name)

    run_augmentation(
        dataset=args.dataset,
        split=args.split,
        model_suff=args.teacher_model_suff,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        soft_threshold=args.soft_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
