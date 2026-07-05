"""
Experiment 1 orchestrator.

Fires 8 sequential fine-tuning runs that together populate a 4-row results
table comparing BERT_original vs BERT_cot student models — with and without
counterfactual (CF) augmentation — across two datasets: LD1 and ER-REASON.

Run-to-results mapping
----------------------
Each (dataset, student_model) pair gets two runs:

  Run 1  LD1  BERT_original  non-aug  → writes Standard F1, CF F1, CF Drop,
  Run 2  LD1  BERT_original  aug      →         Augmented F1  (same row)

  Run 3  LD1  BERT_cot       non-aug  → writes Standard F1, CF F1, CF Drop,
  Run 4  LD1  BERT_cot       aug      →         Augmented F1  (same row)

  Run 5  ER-REASON  BERT_original  non-aug  → ...
  Run 6  ER-REASON  BERT_original  aug      → ...
  Run 7  ER-REASON  BERT_cot       non-aug  → ...
  Run 8  ER-REASON  BERT_cot       aug      → ...

Non-augmented runs additionally write CF F1 / CF Drop when --cf-test-path-*
files are provided (they always are in this orchestrator).  Augmented runs
skip CF evaluation entirely.

The two CSVs at --results-dir are pre-initialized before any training starts.
Each individual training run calls _append_csv() inside finetune_bert.py to
update only its own columns in the matching row.

Typical usage
-------------
python scripts/experiments/run_experiment1.py \\
    --config-ld1 configs/bert_based_models/finbert_ld1.yaml \\
    --config-er-reason configs/bert_based_models/medbert_er_reason.yaml \\
    --cf-test-path-ld1-original data/processed/finbench/ld1/counterfactual/<file>.jsonl \\
    --cf-test-path-ld1-cot data/processed/finbench/ld1/counterfactual_cot/<file>.jsonl \\
    --cf-test-path-er-reason-original data/processed/er_reason/counterfactual/<file>.jsonl \\
    --cf-test-path-er-reason-cot data/processed/er_reason/counterfactual_cot/<file>.jsonl \\
    --run-name-prefix exp1 \\
    --run-name exp1_orchestrator
"""

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.logger import setup_logging

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FINETUNE_SCRIPT = str(_PROJECT_ROOT / "scripts" / "training" / "run_finetune_bert.py")

PRIMARY_COLS = [
    "dataset",
    "student_model",
    "Standard F1",
    "CF F1",
    "CF Drop",
    "Augmented F1",
]
AP_COLS = [
    "dataset",
    "student_model",
    "Standard AP0",
    "Standard AP1",
    "CF AP0",
    "CF AP1",
    "Aug AP0",
    "Aug AP1",
]
_INIT_ROWS = [
    {"dataset": "ld1", "student_model": "BERT_original"},
    {"dataset": "ld1", "student_model": "BERT_cot"},
    {"dataset": "er_reason", "student_model": "BERT_original"},
    {"dataset": "er_reason", "student_model": "BERT_cot"},
]


# ---------------------------------------------------------------------------
# CSV pre-initialization


def _init_csv(path: Path, columns: list[str]) -> None:
    """Write header + 4 empty-metric rows; skip without touching if file exists."""
    if path.exists():
        logger.warning(
            "CSV already exists — skipping initialization to preserve existing results: %s",
            path,
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in _INIT_ROWS:
            writer.writerow({col: row.get(col, "") for col in columns})
    logger.info("Initialized %s with %d rows", path, len(_INIT_ROWS))


# ---------------------------------------------------------------------------
# Run descriptors


def _build_runs(args: argparse.Namespace) -> list[dict]:
    """Return the ordered list of 8 run descriptors for Experiment 1."""
    p = args.run_name_prefix
    return [
        dict(
            label="Run 1 — LD1 BERT_original (non-augmented)",
            config=args.config_ld1,
            mode="original",
            augment=False,
            dataset="ld1",
            student_model_name="BERT_original",
            cf_test_path=args.cf_test_path_ld1_original,
            run_name=f"{p}_ld1_original",
        ),
        dict(
            label="Run 2 — LD1 BERT_original (augmented)",
            config=args.config_ld1,
            mode="original",
            augment=True,
            dataset="ld1",
            student_model_name="BERT_original",
            cf_test_path=None,
            run_name=f"{p}_ld1_original_augmented",
        ),
        dict(
            label="Run 3 — LD1 BERT_cot (non-augmented)",
            config=args.config_ld1,
            mode="cot",
            augment=False,
            dataset="ld1",
            student_model_name="BERT_cot",
            cf_test_path=args.cf_test_path_ld1_cot,
            run_name=f"{p}_ld1_cot",
        ),
        dict(
            label="Run 4 — LD1 BERT_cot (augmented)",
            config=args.config_ld1,
            mode="cot",
            augment=True,
            dataset="ld1",
            student_model_name="BERT_cot",
            cf_test_path=None,
            run_name=f"{p}_ld1_cot_augmented",
        ),
        dict(
            label="Run 5 — ER-REASON BERT_original (non-augmented)",
            config=args.config_er_reason,
            mode="original",
            augment=False,
            dataset="er_reason",
            student_model_name="BERT_original",
            cf_test_path=args.cf_test_path_er_reason_original,
            run_name=f"{p}_er_reason_original",
        ),
        dict(
            label="Run 6 — ER-REASON BERT_original (augmented)",
            config=args.config_er_reason,
            mode="original",
            augment=True,
            dataset="er_reason",
            student_model_name="BERT_original",
            cf_test_path=None,
            run_name=f"{p}_er_reason_original_augmented",
        ),
        dict(
            label="Run 7 — ER-REASON BERT_cot (non-augmented)",
            config=args.config_er_reason,
            mode="cot",
            augment=False,
            dataset="er_reason",
            student_model_name="BERT_cot",
            cf_test_path=args.cf_test_path_er_reason_cot,
            run_name=f"{p}_er_reason_cot",
        ),
        dict(
            label="Run 8 — ER-REASON BERT_cot (augmented)",
            config=args.config_er_reason,
            mode="cot",
            augment=True,
            dataset="er_reason",
            student_model_name="BERT_cot",
            cf_test_path=None,
            run_name=f"{p}_er_reason_cot_augmented",
        ),
    ]


def _build_cmd(run: dict, results_dir: str, drop_last_step: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        _FINETUNE_SCRIPT,
        "--config",
        run["config"],
        "--role",
        "student",
        "--mode",
        run["mode"],
        "--dataset",
        run["dataset"],
        "--student-model-name",
        run["student_model_name"],
        "--run-name",
        run["run_name"],
    ]
    cmd.extend(["--results-dir", str(results_dir)])
    if run["augment"]:
        cmd.append("--augment")
    if run["augment"] and run["dataset"] == "ld1":
        cmd.append("--no-use-weighted-loss")
    if run["cf_test_path"]:
        cmd.extend(["--cf-test-path", run["cf_test_path"]])
    if drop_last_step and run["mode"] == "cot":
        cmd.append("--drop-last-step")
    return cmd


# ---------------------------------------------------------------------------
# Argument parsing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1 orchestrator: 8 sequential BERT fine-tuning runs "
            "comparing BERT_original vs BERT_cot with/without CF augmentation "
            "across LD1 and ER-REASON datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------------------- required
    parser.add_argument(
        "--config-ld1",
        required=True,
        help="Path to the LD1 YAML config (e.g. configs/bert_based_models/finbert_ld1.yaml).",
    )
    parser.add_argument(
        "--config-er-reason",
        required=True,
        help="Path to the ER-REASON YAML config (e.g. configs/bert_based_models/medbert_er_reason.yaml).",
    )
    parser.add_argument(
        "--cf-test-path-ld1-original",
        required=True,
        help="LD1 CF test JSONL for mode=original (from the counterfactual/ directory).",
    )
    parser.add_argument(
        "--cf-test-path-ld1-cot",
        required=True,
        help="LD1 CF test JSONL for mode=cot (from the counterfactual_cot/ directory).",
    )
    parser.add_argument(
        "--cf-test-path-er-reason-original",
        required=True,
        help="ER-REASON CF test JSONL for mode=original (from the counterfactual/ directory).",
    )
    parser.add_argument(
        "--cf-test-path-er-reason-cot",
        required=True,
        help="ER-REASON CF test JSONL for mode=cot (from the counterfactual_cot/ directory).",
    )
    parser.add_argument(
        "--run-name-prefix",
        required=True,
        help="Prefix appended with the run identifier to form each WandB run name.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run name used for orchestrator-level logging setup.",
    )

    # ---------------------------------------------------------------- optional
    parser.add_argument(
        "--results-dir",
        default="experiment1_default",
        help="Directory where the two result CSVs are pre-initialized and written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the 8 commands without executing them.",
    )
    parser.add_argument(
        "--start-from-run",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="{1..8}",
        help="Resume from this run number, skipping all earlier runs (1 = run all).",
    )
    parser.add_argument(
        "--drop-last-step",
        action="store_true",
        default=False,
        help=(
            "Strip the 4th [STEP]...[EORS] block before training, for CoT runs only "
            "(runs 3, 4, 7, 8).  Has no effect on original-mode runs (1, 2, 5, 6)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point


def main() -> None:
    args = _parse_args()
    setup_logging(run_name=args.run_name)

    results_dir = _PROJECT_ROOT / "results" / args.results_dir
    logger.info("Experiment 1 results directory: %s", results_dir)

    _init_csv(results_dir / "primary_results.csv", PRIMARY_COLS)
    _init_csv(results_dir / "ap_per_class_results.csv", AP_COLS)

    runs = _build_runs(args)

    start = args.start_from_run
    if start > 1:
        skipped = [r["label"] for r in runs[: start - 1]]
        logger.info(
            "--start-from-run=%d: skipping %d run(s): %s",
            start,
            len(skipped),
            "; ".join(skipped),
        )
    runs_to_fire = runs[start - 1 :]

    if args.dry_run:
        logger.info(
            "--dry-run active: printing %d commands without executing",
            len(runs_to_fire),
        )
        for run in runs_to_fire:
            cmd = _build_cmd(run, args.results_dir, drop_last_step=args.drop_last_step)
            print(f"\n# {run['label']}")
            print(" ".join(cmd))
        return

    for i, run in enumerate(runs_to_fire, start):
        logger.info(
            "[%d/8] Starting %s  (dataset=%s  mode=%s  augment=%s)",
            i,
            run["label"],
            run["dataset"],
            run["mode"],
            run["augment"],
        )
        result = subprocess.run(
            _build_cmd(run, args.results_dir, drop_last_step=args.drop_last_step)
        )
        if result.returncode != 0:
            logger.error(
                "[%d/8] %s failed with exit code %d — stopping pipeline",
                i,
                run["label"],
                result.returncode,
            )
            sys.exit(result.returncode)
        logger.info("[%d/8] %s completed successfully (exit code 0)", i, run["label"])

    logger.info(
        "Experiment 1 complete: all 8 runs finished. Results at %s",
        results_dir,
    )


if __name__ == "__main__":
    main()
