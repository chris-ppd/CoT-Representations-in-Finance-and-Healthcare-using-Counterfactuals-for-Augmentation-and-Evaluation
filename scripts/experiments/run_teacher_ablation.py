"""
Qwen3-8B teacher ablation orchestrator.

Fires 2 sequential fine-tuning runs comparing BERT_original vs BERT_cot on
ER-REASON only, using CoT/CF data generated with the Qwen3-8B teacher instead
of Qwen3-4B. No augmentation (out of scope for this ablation, consistent with
the earlier BERT-base ablation study).

Run-to-results mapping
----------------------
  Run 1  ER-REASON  BERT_original  → writes Standard F1, CF F1, CF Drop
                                      (evaluated against the plain reconstructed
                                      CF test set, Qwen3-8B-derived)

  Run 2  ER-REASON  BERT_cot (8B)  → writes Standard F1, CF F1, CF Drop
                                      (trained on Qwen3-8B CoT text with Step 4
                                      dropped; evaluated against the CoT-formatted
                                      CF test set, Qwen3-8B-derived, Step 4 also
                                      dropped)

No Augmented F1 column is populated — this ablation has no augmented runs.

3 reruns are done by invoking this orchestrator 3 separate times with distinct
--results-dir / --run-name-prefix values (no rerun loop inside this script,
consistent with how Experiment 1/2 reruns were done).

Typical usage
-------------
python scripts/experiments/run_ablation_8b.py \\
    --config configs/bert_based_models/medbert_er_reason_Qwen3-8b.yaml \\
    --cf-test-path-original data/processed/er_reason/counterfactual/er_reason_counterfactual_test_Qwen3-8b.jsonl \\
    --cf-test-path-cot data/processed/er_reason/counterfactual_cot/er_reason_counterfactual_cot_test_Qwen3-8b.jsonl \\
    --run-name-prefix ablation8b_run1 \\
    --results-dir ablation_8b_run1 \\
    --run-name ablation8b_orchestrator_run1
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
    {"dataset": "er_reason_8b", "student_model": "BERT_original"},
    {"dataset": "er_reason_8b", "student_model": "BERT_cot_8b"},
]


# ---------------------------------------------------------------------------
# CSV pre-initialization


def _init_csv(path: Path, columns: list[str]) -> None:
    """Write header + 2 empty-metric rows; skip without touching if file exists."""
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
    """Return the ordered list of 2 run descriptors for the 8B ablation."""
    p = args.run_name_prefix
    return [
        dict(
            label="Run 1 — ER-REASON BERT_original (Qwen3-8B CF test)",
            config=args.config,
            mode="original",
            dataset="er_reason_8b",
            student_model_name="BERT_original",
            cf_test_path=args.cf_test_path_original,
            run_name=f"{p}_er_reason_original_8b",
        ),
        dict(
            label="Run 2 — ER-REASON BERT_cot (Qwen3-8B teacher)",
            config=args.config,
            mode="cot",
            dataset="er_reason_8b",
            student_model_name="BERT_cot_8b",
            cf_test_path=args.cf_test_path_cot,
            run_name=f"{p}_er_reason_cot_8b",
        ),
    ]


def _build_cmd(run: dict, results_dir: str, drop_last_step: bool) -> list[str]:
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
        "--results-dir",
        str(results_dir),
    ]
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
            "Qwen3-8B teacher ablation orchestrator: 2 sequential BERT fine-tuning "
            "runs comparing BERT_original vs BERT_cot on ER-REASON, using Qwen3-8B "
            "generated CoT/CF data. No augmentation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------------------- required
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the ER-REASON Qwen3-8B YAML config "
        "(e.g. configs/bert_based_models/medbert_er_reason_Qwen3-8b.yaml).",
    )
    parser.add_argument(
        "--cf-test-path-original",
        required=True,
        help="Plain-text reconstructed CF test JSONL (Qwen3-8B-derived), "
        "for mode=original.",
    )
    parser.add_argument(
        "--cf-test-path-cot",
        required=True,
        help="CoT-formatted CF test JSONL (Qwen3-8B-derived), for mode=cot.",
    )
    parser.add_argument(
        "--run-name-prefix",
        required=True,
        help="Prefix appended with the run identifier to form each WandB run name. "
        "Use a distinct prefix per rerun (e.g. ablation8b_run1, _run2, _run3).",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run name used for orchestrator-level logging setup.",
    )

    # ---------------------------------------------------------------- optional
    parser.add_argument(
        "--results-dir",
        default="ablation_8b_default",
        help="Directory where the two result CSVs are pre-initialized and written. "
        "Use a distinct value per rerun (e.g. ablation_8b_run1, _run2, _run3) so "
        "reruns don't overwrite each other.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the 2 commands without executing them.",
    )
    parser.add_argument(
        "--start-from-run",
        type=int,
        default=1,
        choices=range(1, 3),
        metavar="{1..2}",
        help="Resume from this run number, skipping earlier runs (1 = run all).",
    )
    parser.add_argument(
        "--drop-last-step",
        action="store_true",
        default=False,
        help=(
            "Strip the 4th [STEP]...[EORS] block before training, for the CoT run "
            "only (run 2). Has no effect on the original-mode run (run 1). "
            "Matches the Step-4 label-leakage fix applied in Experiment 1's final "
            "results — should be passed for a fair comparison against Table 5.4."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point


def main() -> None:
    args = _parse_args()
    setup_logging(run_name=args.run_name)

    results_dir = _PROJECT_ROOT / "results" / args.results_dir
    logger.info("8B ablation results directory: %s", results_dir)

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
            "[%d/2] Starting %s  (dataset=%s  mode=%s)",
            i,
            run["label"],
            run["dataset"],
            run["mode"],
        )
        result = subprocess.run(
            _build_cmd(run, args.results_dir, drop_last_step=args.drop_last_step)
        )
        if result.returncode != 0:
            logger.error(
                "[%d/2] %s failed with exit code %d — stopping pipeline",
                i,
                run["label"],
                result.returncode,
            )
            sys.exit(result.returncode)
        logger.info("[%d/2] %s completed successfully (exit code 0)", i, run["label"])

    logger.info(
        "8B ablation complete: both runs finished. Results at %s",
        results_dir,
    )


if __name__ == "__main__":
    main()
