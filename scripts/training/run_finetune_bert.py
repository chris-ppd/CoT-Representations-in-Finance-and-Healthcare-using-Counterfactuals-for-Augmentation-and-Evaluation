"""
Sweep-aware CLI entry point for BERT fine-tuning.

Extends train_bert.py with --learning-rate, --class-weights, and an optional
--run-name (auto-filled from wandb.run.name when launched by a sweep agent).
Hyperparameters injected via wandb.config (sweep context) override CLI values,
which in turn override YAML values.

Typical standalone usage
------------------------
python /scripts/training/run_finetune_bert.py \
    --config configs/bert_based_models/medbert_er_reason.yaml \
    --role validator \
    --run-name medbert-validator-with-cot

WandB sweep usage (agent fills in hyperparameters via wandb.config)
-------------------------------------------------------------------
wandb agent <entity>/<project>/<sweep_id> \\
    -- python scripts/finetune_bert.py \\
       --config configs/bert/finbert_ld1.yaml \\
       --role validator
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.finetune_bert import finetune
from utils.logger import setup_logging


def _parse_class_weights(raw: str):
    """Accept a JSON dict string or the literal 'balanced'."""
    if raw.strip().lower() == "balanced":
        return "balanced"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--class-weights must be a JSON dict or 'balanced', got: {raw!r}"
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune any BERT-family model (sweep-aware). "
            "Hyperparameter CLI args are overridden by wandb.config when "
            "running inside a WandB sweep."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------------------- required
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

    # ---------------------------------------------------------------- optional
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Name used in WandB and as the checkpoint subdirectory. "
            "Defaults to the auto-generated wandb.run.name when inside a sweep."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="original",
        choices=["original", "cot", "injection"],
        help="Input text mode (student role only).",
    )
    parser.add_argument(
        "--injection-position",
        type=str,
        default=None,
        choices=["pre_group", "post_group"],
        help="Required when --mode injection. Where the teacher residual is added relative to each layer group.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate config/data (and, for --mode injection, hidden-state "
            "alignment) and print a report, then exit before trainer.train()."
        ),
    )
    parser.add_argument(
        "--results-experiment",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which experiment's result CSVs to append to (1 or 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output.base_dir from the YAML config.",
    )

    # ------------------------------------------------ sweep hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Override peak learning rate (maps to training.lr in the YAML). "
            "Overridden by wandb.config.learning_rate when inside a sweep."
        ),
    )
    parser.add_argument(
        "--class-weights",
        type=_parse_class_weights,
        default=None,
        metavar='\'{"0": 0.5, "1": 2.0}\' | balanced',
        help=(
            "Override class weights. Pass a JSON dict or the string 'balanced' "
            "for sklearn inverse-frequency weighting. "
            "Overridden by wandb.config.class_weights when inside a sweep."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override random seed. Overridden by wandb.config.seed when inside a sweep."
        ),
    )

    # ------------------------------------------------ other hparam overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument(
        "--eval-split", type=float, default=None, help="Validator only."
    )

    # ------------------------------------------------ experiment 1 flags
    parser.add_argument(
        "--no-use-weighted-loss",
        action="store_true",
        default=False,
        help=(
            "Disable weighted loss — overrides use_weighted_loss in the YAML to False. "
            "Use for augmented LD1 runs where the dataset is approximately balanced."
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help=(
            "Load augmented training/val/test splits (aug_{mode}_* YAML keys) "
            "instead of the standard CoT splits."
        ),
    )
    parser.add_argument(
        "--cf-test-path",
        type=str,
        default=None,
        help=(
            "Path to a CF test JSONL file. When provided (non-augmented runs "
            "only), a second post-training evaluation is run under test_cf/."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset identifier written to Experiment 1 result CSVs (e.g. ld1, er_reason).",
    )
    parser.add_argument(
        "--student-model-name",
        type=str,
        default=None,
        help="Model identifier written to Experiment 1 result CSVs (e.g. BERT_original, BERT_cot).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save the results table",
    )
    parser.add_argument(
        "--drop-last-step",
        action="store_true",
        default=False,
        help=(
            "When mode=cot, strip the 4th [STEP]...[EORS] block from every CoT "
            "text before training.  Useful for removing label-leaking synthesis "
            "steps from the input."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parents[2]
    _args = _parse_args()

    _hparam_overrides = {
        "epochs": _args.epochs,
        "batch_size": _args.batch_size,
        # --learning-rate maps to the 'lr' key used throughout finetune_bert.py
        "lr": _args.learning_rate,
        "warmup_ratio": _args.warmup_ratio,
        "max_len": _args.max_len,
        "eval_split": _args.eval_split,
        "seed": _args.seed,
        # --class-weights maps to the 'class_weight' key
        "class_weight": _args.class_weights,
        # --no-use-weighted-loss maps to the 'use_weighted_loss' key
        "use_weighted_loss": False if _args.no_use_weighted_loss else None,
    }

    setup_logging(run_name=_args.run_name or "sweep")

    finetune(
        config_path=_args.config,
        role=_args.role,
        mode=_args.mode,
        run_name=_args.run_name,
        project_root=_project_root,
        output_dir_override=_args.output_dir,
        hparam_overrides=_hparam_overrides,
        augment=_args.augment,
        cf_test_path=_args.cf_test_path,
        dataset_name=_args.dataset,
        student_model_name=_args.student_model_name,
        results_dir=_args.results_dir,
        drop_last_step=_args.drop_last_step,
        injection_position=_args.injection_position,
        dry_run=_args.dry_run,
        results_experiment=_args.results_experiment,
    )
