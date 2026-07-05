"""
CLI entry point for all BERT fine-tuning across the thesis.

Selects a YAML config per model/dataset, then delegates to finetune_bert.finetune().

Typical usage
-------------
# Train finbert as a validator on FinBench LD1
python scripts/train_bert.py \\
    --config configs/bert/finbert_ld1.yaml \\
    --role validator \\
    --run-name finbert_val_ld1_run1

# Train bert-base as a student (original text) on FinBench LD1
python scripts/train_bert.py \\
    --config configs/bert/bert_base_ld1.yaml \\
    --role student --mode original \\
    --run-name bert_student_orig_ld1_run1

# Train finbert as a student (CoT text), overriding lr and seed
python scripts/train_bert.py \\
    --config configs/bert/finbert_ld1.yaml \\
    --role student --mode cot \\
    --lr 3e-5 --seed 1 \\
    --run-name finbert_student_cot_ld1_seed1
"""

# PROBABLY REDUNDANAAAAAAAAAAAAAAAAAAAAAT (THIS IMPORTANT!)

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.finetune_bert import finetune
from utils.logger import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune any BERT-family model for the thesis (validator or student).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ required
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file (e.g. configs/bert/finbert_ld1.yaml).",
    )
    parser.add_argument(
        "--role",
        type=str,
        required=True,
        choices=["validator", "student"],
        help=(
            "'validator': concatenate all splits + stratified 80/20 split. "
            "'student': train on CoT split files (train/val/test)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name used in WandB and as the checkpoint subdirectory.",
    )

    # ------------------------------------------------------------------ optional
    parser.add_argument(
        "--mode",
        type=str,
        default="original",
        choices=["original", "cot"],
        help=(
            "Input text mode (student role only). "
            "'original': use the text field. 'cot': use the cot_text field. "
            "Both modes use the exact same set of data-point IDs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output.base_dir from the YAML config.",
    )

    # -------------------------------------------------------- hyperparameter overrides
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override per-device batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override peak learning rate."
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=None, help="Override LR warmup ratio."
    )
    parser.add_argument(
        "--max-len", type=int, default=None, help="Override max tokenizer length."
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=None,
        help="Override eval split fraction (validator only).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--class-weight",
        type=json.loads,
        default=None,
        help=(
            'Override class weights as a JSON dict, e.g. \'{"0": 0.5, "1": 2.0}\'. '
            "Pass 'null' to use the sklearn balanced strategy."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parents[1]
    _args = _parse_args()

    _hparam_overrides = {
        "epochs": _args.epochs,
        "batch_size": _args.batch_size,
        "lr": _args.lr,
        "warmup_ratio": _args.warmup_ratio,
        "max_len": _args.max_len,
        "eval_split": _args.eval_split,
        "seed": _args.seed,
        "class_weight": _args.class_weight,
    }

    setup_logging(run_name=_args.run_name)

    finetune(
        config_path=_args.config,
        role=_args.role,
        mode=_args.mode,
        run_name=_args.run_name,
        project_root=_project_root,
        output_dir_override=_args.output_dir,
        hparam_overrides=_hparam_overrides,
    )
