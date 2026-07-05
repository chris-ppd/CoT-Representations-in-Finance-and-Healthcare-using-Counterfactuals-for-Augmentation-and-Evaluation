import argparse
import sys
from pathlib import Path

# Add project root to Python path so src package is importable (maybe we can use a toml file later)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.cot.cot_generator import run_cot_generation
from utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CoT texts for FinBench profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cot/er_reason_cot.yaml",
        help="Path to CoT YAML config.",
    )
    parser.add_argument(
        "--path-suff",
        type=str,
        help="Suffix inserted into the output file name (e.g. train, val, test).",
    )
    parser.add_argument(
        "--teacher-model-suff",
        type=str,
        help="Suffix appended after the teacher model name in the output file name.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name used in to name the logger file for this cot generation run",
    )
    parser.add_argument(
        "--extract-hidden-states",
        action="store_true",
        default=False,  # consistent with action="store_true"
        help="enables extraction of the last hidden state of the teacher model during cot generation",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="skip the CoT validation pipeline and save all generated profiles regardless of quality",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(run_name=args.run_name)

    run_cot_generation(
        # "configs/cot/finbench_cot.yaml",
        args.config,
        path_suff=args.path_suff,
        teacher_model_suff=args.teacher_model_suff,
        extract_hidden_states=args.extract_hidden_states
        if args.extract_hidden_states
        else None,
        skip_validation=args.skip_validation,
    )
