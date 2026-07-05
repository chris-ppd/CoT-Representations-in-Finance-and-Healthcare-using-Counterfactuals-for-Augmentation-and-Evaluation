"""
Entry point for CF profile reconstruction.

Normal run (one split):
    python scripts/generation/run_cf_reconstruction.py \\
        --config-path configs/counterfactual/cf_ld1.yaml \\
        --input-path data/processed/finbench/ld1/counterfactual_cot/ld1_counterfactual_cot_train.jsonl \\
        --model-name Qwen/Qwen3-4B \\
        --path-suff train \\
        --run-name cf_reconstruction_ld1_train

Local test (limit to 2 batches × batch_size profiles):
    python scripts/generation/run_cf_reconstruction.py \\
        --config-path configs/counterfactual/cf_ld1.yaml \\
        --input-path data/processed/finbench/ld1/counterfactual_cot/ld1_counterfactual_cot_train.jsonl \\
        --model-name Qwen/Qwen3-4B \\
        --num-batches 2 \\
        --batch-size 4 \\
        --path-suff train \\
        --run-name cf_reconstruction_ld1_pilot
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.counterfactual.reconstruct_cf_profiles import run_cf_reconstruction
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct plain CF profiles from counterfactual CoT reasoning text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to CF config YAML (must contain a reconstruction block).",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input CF JSONL file (counterfactual_cot split file).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to output JSONL file. Overrides config output_dir if provided.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Qwen model name or local path.",
    )
    parser.add_argument(
        "--teacher-model-suff",
        type=str,
        default=None,
        help=(
            "Suffix appended to the output filename. "
            "Does NOT affect model loading (model is always read from config). "
            "Usually omitted — the suffix is inherited from the input filename automatically."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Profiles per Qwen batch call. Overrides config if provided.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help=(
            "Limit processing to first N batches (N × batch_size profiles). "
            "Overrides config if provided. Use for local testing."
        ),
    )
    parser.add_argument(
        "--path-suff",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split identifier used for logging and output filename.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Logger file name prefix.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(run_name=args.run_name)

    run_cf_reconstruction(
        config_path=args.config_path,
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
        teacher_model_suff=args.teacher_model_suff,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        path_suff=args.path_suff,
    )
