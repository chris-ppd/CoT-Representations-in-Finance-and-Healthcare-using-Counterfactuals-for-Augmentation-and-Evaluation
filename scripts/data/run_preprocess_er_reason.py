"""
CLI entry point for the ER-REASON preprocessing pipeline.

    python scripts/run_preprocess_er_reason.py \\
        --input-path data/raw/er_reason/er_reason.csv \\
        --output-path data/processed/er_reason/er_reason_processed.jsonl \\
        --model-name Qwen/Qwen3-4B \\
        --quantization 8bit \\
        --batch-size 1 --column-batch-size 4 \\
        --run-name er_reason_preprocessing_run1_full_pipeline

    For local debugging (minimal memory footprint):
        --batch-size 1 --column-batch-size 2

Resume support: if --output-path already exists, profiles whose encounterkey is
already present are skipped automatically.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_preprocessing.preprocess_er_reason import run_pipeline
from utils.logger import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ER-REASON preprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the raw ER-REASON CSV file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to the output JSONL for preprocessed profiles.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Experiment name used for logging.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name for local LLM summarization.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="8bit",
        choices=["8bit", "none"],
        help="BitsAndBytes quantization mode. Use 'none' to disable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of profiles per outer batch.",
    )
    parser.add_argument(
        "--column-batch-size",
        type=int,
        default=7,
        help=(
            "Number of historical note columns per memory chunk. "
            "Use 7 to process all columns without chunking; 2 for minimal memory."
        ),
    )
    parser.add_argument(
        "--max-note-chars",
        type=int,
        default=3000,
        help=(
            "Truncate each raw note to this many characters before sending to the LLM. "
            "Set 0 to disable truncation."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per LLM generation call.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Limit the main pass to this many batches. Omit to process the full dataset.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()

    setup_logging(run_name=_args.run_name)

    _quantization = None if _args.quantization == "none" else _args.quantization

    run_pipeline(
        input_path=_args.input_path,
        output_path=_args.output_path,
        model_name=_args.model_name,
        quantization=_quantization,
        batch_size=_args.batch_size,
        column_batch_size=_args.column_batch_size,
        run_name=_args.run_name,
        max_note_chars=_args.max_note_chars,
        max_new_tokens=_args.max_new_tokens,
        num_batches=_args.num_batches,
    )
