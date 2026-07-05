"""
Experiment 2 orchestrator.

Fires 12 sequential fine-tuning runs that together populate a 6-row results
table comparing BERT_original vs BERT_injected_pre_group vs
BERT_injected_post_group student models — with and without counterfactual
(CF) augmentation — across two datasets: LD1 and ER-REASON. Injected variants
additionally populate a 4-row gate-values table (BERT_original has no gates).

Run-to-results mapping
----------------------
Each (dataset, student_model) pair gets two runs:

  Run 1   LD1  BERT_original          non-aug
  Run 2   LD1  BERT_original          aug
  Run 3   LD1  BERT_injected_pre_group   non-aug
  Run 4   LD1  BERT_injected_pre_group   aug
  Run 5   LD1  BERT_injected_post_group  non-aug
  Run 6   LD1  BERT_injected_post_group  aug
  Run 7   ER-REASON  BERT_original          non-aug
  Run 8   ER-REASON  BERT_original          aug
  Run 9   ER-REASON  BERT_injected_pre_group   non-aug
  Run 10  ER-REASON  BERT_injected_pre_group   aug
  Run 11  ER-REASON  BERT_injected_post_group  non-aug
  Run 12  ER-REASON  BERT_injected_post_group  aug

Non-augmented runs additionally write CF F1 / CF Drop when --cf-test-path-*
is provided (it always is in this orchestrator). All student models in this
experiment train on the dataset's plain `text` field (BERT_original via
--mode original; injected variants via --mode injection, which internally
loads text the same way as --mode original and injects teacher hidden
states instead), so a single CF test path per dataset covers all three
student models.

--dry-run here passes --dry-run through to every one of the 12
`run_finetune_bert.py` subprocess calls (which validates config/data and
exits before training) — unlike Experiment 1's orchestrator-level --dry-run,
this one still actually launches all 12 subprocesses.

Typical usage
-------------
python scripts/experiments/run_experiment2.py \\
    --config-ld1 configs/bert_based_models/finbert_ld1.yaml \\
    --config-er-reason configs/bert_based_models/medbert_er_reason.yaml \\
    --cf-test-path-ld1 data/processed/finbench/ld1/counterfactual/<file>.jsonl \\
    --cf-test-path-er-reason data/processed/er_reason/counterfactual/<file>.jsonl \\
    --run-name-prefix exp2 \\
    --run-name exp2_orchestrator
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
GATE_COLS = [
    "dataset",
    "student_model",
    "gate_1_init",
    "gate_2_init",
    "gate_3_init",
    "gate_1_final",
    "gate_2_final",
    "gate_3_final",
]

_STUDENT_MODELS = [
    "BERT_original",
    "BERT_injected_pre_group",
    "BERT_injected_post_group",
]
_INJECTED_STUDENT_MODELS = ["BERT_injected_pre_group", "BERT_injected_post_group"]

_INIT_ROWS_PRIMARY = [
    {"dataset": dataset, "student_model": student_model}
    for dataset in ("ld1", "er_reason")
    for student_model in _STUDENT_MODELS
]
_INIT_ROWS_GATE = [
    {"dataset": dataset, "student_model": student_model}
    for dataset in ("ld1", "er_reason")
    for student_model in _INJECTED_STUDENT_MODELS
]


# ---------------------------------------------------------------------------
# CSV pre-initialization


def _init_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write header + empty-metric rows; skip without touching if file exists."""
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
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    logger.info("Initialized %s with %d rows", path, len(rows))


# ---------------------------------------------------------------------------
# Run descriptors

_MODEL_SPECS = {
    "BERT_original": {"mode": "original", "injection_position": None},
    "BERT_injected_pre_group": {"mode": "injection", "injection_position": "pre_group"},
    "BERT_injected_post_group": {
        "mode": "injection",
        "injection_position": "post_group",
    },
}


def _build_runs(args: argparse.Namespace) -> list[dict]:
    """Return the ordered list of 12 run descriptors for Experiment 2."""
    p = args.run_name_prefix
    runs = []
    run_no = 0
    for dataset, config, cf_test_path in (
        ("ld1", args.config_ld1, args.cf_test_path_ld1),
        ("er_reason", args.config_er_reason, args.cf_test_path_er_reason),
    ):
        for student_model in _STUDENT_MODELS:
            spec = _MODEL_SPECS[student_model]
            for augment in (False, True):
                run_no += 1
                runs.append(
                    dict(
                        label=(
                            f"Run {run_no} — {dataset.upper()} {student_model} "
                            f"({'augmented' if augment else 'non-augmented'})"
                        ),
                        config=config,
                        mode=spec["mode"],
                        injection_position=spec["injection_position"],
                        augment=augment,
                        dataset=dataset,
                        student_model_name=student_model,
                        cf_test_path=None if augment else cf_test_path,
                        run_name=f"{p}_{dataset}_{student_model.lower()}"
                        + ("_augmented" if augment else ""),
                    )
                )
    return runs


def _build_cmd(run: dict, results_dir: str, dry_run: bool = False) -> list[str]:
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
        "--results-experiment",
        "2",
    ]
    cmd.extend(["--results-dir", str(results_dir)])
    if run["injection_position"]:
        cmd.extend(["--injection-position", run["injection_position"]])
    if run["augment"]:
        cmd.append("--augment")
    if run["augment"] and run["dataset"] == "ld1":
        cmd.append("--no-use-weighted-loss")
    if run["cf_test_path"]:
        cmd.extend(["--cf-test-path", run["cf_test_path"]])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


# ---------------------------------------------------------------------------
# Argument parsing


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 2 orchestrator: 12 sequential BERT fine-tuning runs "
            "comparing BERT_original vs BERT_injected_pre_group vs "
            "BERT_injected_post_group with/without CF augmentation across "
            "LD1 and ER-REASON datasets."
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
        "--cf-test-path-ld1",
        required=True,
        help="LD1 CF test JSONL (original text) shared by all three student models.",
    )
    parser.add_argument(
        "--cf-test-path-er-reason",
        required=True,
        help="ER-REASON CF test JSONL (original text) shared by all three student models.",
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
        default="experiment2_default",
        help="Directory where the three result CSVs are pre-initialized and written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Pass --dry-run through to every one of the 12 run_finetune_bert.py "
            "subprocess calls (each validates config/data and exits before "
            "trainer.train()). Subprocesses are still launched."
        ),
    )
    parser.add_argument(
        "--start-from-run",
        type=int,
        default=1,
        choices=range(1, 13),
        metavar="{1..12}",
        help="Resume from this run number, skipping all earlier runs (1 = run all).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point


def main() -> None:
    args = _parse_args()
    setup_logging(run_name=args.run_name)

    results_dir = _PROJECT_ROOT / "results" / args.results_dir
    logger.info("Experiment 2 results directory: %s", results_dir)

    _init_csv(
        results_dir / "experiment2_primary_results.csv",
        PRIMARY_COLS,
        _INIT_ROWS_PRIMARY,
    )
    _init_csv(
        results_dir / "experiment2_ap_per_class_results.csv",
        AP_COLS,
        _INIT_ROWS_PRIMARY,
    )
    _init_csv(results_dir / "experiment2_gate_values.csv", GATE_COLS, _INIT_ROWS_GATE)

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

    for i, run in enumerate(runs_to_fire, start):
        logger.info(
            "[%d/12] Starting %s  (dataset=%s  mode=%s  injection_position=%s  augment=%s)",
            i,
            run["label"],
            run["dataset"],
            run["mode"],
            run["injection_position"],
            run["augment"],
        )
        result = subprocess.run(_build_cmd(run, args.results_dir, dry_run=args.dry_run))
        if result.returncode != 0:
            logger.error(
                "[%d/12] %s failed with exit code %d — stopping pipeline",
                i,
                run["label"],
                result.returncode,
            )
            sys.exit(result.returncode)
        logger.info("[%d/12] %s completed successfully (exit code 0)", i, run["label"])

    logger.info(
        "Experiment 2 complete: all 12 runs finished. Results at:\n  %s\n  %s\n  %s",
        results_dir / "experiment2_primary_results.csv",
        results_dir / "experiment2_ap_per_class_results.csv",
        results_dir / "experiment2_gate_values.csv",
    )


if __name__ == "__main__":
    main()
