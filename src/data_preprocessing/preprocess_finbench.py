"""
Merge FinBench profiles with their labels into a single .jsonl file.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def add_arguments():
    # Parse CLI paths: profiles (.jsonl), labels (.npy), and output (.jsonl)
    parser = argparse.ArgumentParser(
        description="Append FinBench labels to profile records and write to .jsonl."
    )
    parser.add_argument("profiles_path", type=str, help="path to the original profiles")
    parser.add_argument(
        "labels_path", type=str, help="path to the labels of the original profiles"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="path to save the processed profiles providing the name of the file as well",
    )

    args = parser.parse_args()

    return args


def append_labels():
    # Read profiles and labels line-by-line, merge, and write to output .jsonl
    args = add_arguments()

    labels = np.load(args.labels_path)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.profiles_path, "r") as f_in, open(args.output_path, "w") as f_out:
        for idx, (line, label) in enumerate(zip(f_in, labels)):
            profile_text = json.loads(line.strip())
            # Each record: sequential id, original profile text, binary label
            data_point = {"id": idx, "text": profile_text, "label": int(label)}
            f_out.write(json.dumps(data_point) + "\n")

    print(f"Saved {idx + 1} data points to {args.output_path}")


if __name__ == "__main__":
    append_labels()
