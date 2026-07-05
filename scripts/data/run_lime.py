"""
Run the script:
    python scripts/data/run_lime.py --config configs/counterfactual/cf_er_reason.yaml --split val --run-name lime-er-reason-val-final
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.counterfactual.lime_runner import run_lime
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LIME attribution for one dataset split and save top features.",
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
        help="Logger file name prefix (e.g. lime_ld1_train).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(run_name=args.run_name)
    run_lime(config_path=args.config, split=args.split)
