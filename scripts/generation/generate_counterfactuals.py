"""
Entry point for counterfactual generation (Script 2 of 2).

Normal run (one split):
python scripts/generation/generate_counterfactuals.py \
        --config configs/counterfactual/cf_ld1.yaml \
        --split val \
        --run-name cf_ld1_val_check_cf


Pilot experiment (choose k before full run):
    python scripts/generation/generate_counterfactuals.py \\
        --config configs/counterfactual/cf_ld1.yaml \\
        --split train \\
        --run-name cf_ld1_pilot \\
        --pilot

The pilot runs k=3, k=5, k=10 on 100 profiles each and prints a comparison table.
Lock the winning k in the config's generation.top_k field before the full run.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.counterfactual.cf_generator import run_cf_generation
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate counterfactual profiles for a dataset split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to counterfactual YAML config (e.g. configs/counterfactual/cf_ld1.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Logger file name prefix.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help=(
            "Run pilot experiment: evaluate k=3, k=5, k=7 on 50 profiles each "
            "and print a comparison table.  Does not write to the main output file."
        ),
    )
    parser.add_argument(
        "--pilot-n",
        type=int,
        default=50,
        help="Number of profiles per k value in pilot mode.",
    )
    parser.add_argument(
        "--teacher-model-suff",
        type=str,
        default=None,
        help=(
            "Suffix appended to the output file names produced by this run "
            "(e.g. 'Qwen3-4b' → ld1_counterfactual_cot_train_Qwen3-4b.jsonl). "
            "Keeps outputs from different model variants distinguishable."
        ),
    )
    parser.add_argument(
        "--extract-hidden-states",
        action="store_true",
        default=False,
        help=(
            "After each approved CF profile passes the BERT flip gate, run a single "
            "forward pass through Qwen3-4B (output_hidden_states=True) and extract "
            "last-layer hidden states at [EORS] and [EOA] positions.  Saved under "
            "hidden_states/{dataset}/cf/{split}/.  Ignored in --pilot mode."
        ),
    )
    return parser.parse_args()


def _print_pilot_table(results: list[dict]) -> None:
    header = (
        f"{'k':>4}  {'SLFR':>8}  {'Avg CosSim':>12}  {'Avg PPL':>10}  {'Approved':>10}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("Pilot experiment results")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['top_k']:>4}  {r['slfr']:>8.2%}  "
            f"{r['avg_cosine_sim']:>12.4f}  {r['avg_ppl']:>10.2f}  "
            f"{r['total_approved']:>10}/{r['total_attempted']}"
        )
    print(sep)
    print(
        "Pick k with the best tradeoff between SLFR and cosine similarity, "
        "then set generation.top_k in the config.\n"
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(run_name=args.run_name)

    if args.pilot:
        pilot_results = []
        for k in [3, 5, 7]:
            print(f"\n--- Pilot: k={k} ---")
            summary = run_cf_generation(
                config_path=args.config,
                split=args.split,
                top_k=k,
                pilot=True,
                pilot_n=args.pilot_n,
                teacher_model_suff=args.teacher_model_suff,
            )
            pilot_results.append(summary)
        _print_pilot_table(pilot_results)
    else:
        run_cf_generation(
            config_path=args.config,
            split=args.split,
            extract_hidden_states=args.extract_hidden_states,
            teacher_model_suff=args.teacher_model_suff,
        )
