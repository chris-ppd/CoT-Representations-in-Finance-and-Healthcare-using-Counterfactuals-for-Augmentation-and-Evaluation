"""
Parse CoT generation logs to analyze the full distribution of faithfulness
and BERT confidence scores across all profiles (passed and failed).

This avoids rerunning generation with different thresholds — the logs already
contain every profile's scores regardless of whether it passed or failed.

Usage
-----
    python scripts/utils/analyze_cot_scores.py \
    --log-path logs/Qwen3-4b-ereason-bert-score-075_20260524_193754.log \
    --profiles-path data/processed/er_reason/cot/er_reason_cot_test.jsonl \
    --output threshold_analysis/er_reason \
    --dataset er_reason \
    --faith-threshold 0.8 \
    --bert-threshold 0.75 \
    --id2label "0=Discharge,1=Admit"

Outputs
-------
    results/score_analysis/score_report.txt   <- full stats + interpretation
    results/score_analysis/score_plots.png    <- distributions + scatter plot
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_profile_labels(profiles_path: Path) -> dict[str, int]:
    """Return {str(id): label} from a JSONL profiles file."""
    labels: dict[str, int] = {}
    with open(profiles_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels[str(record["id"])] = int(record["label"])
    return labels


def parse_log(
    log_path: Path, profile_labels: dict[str, int] | None = None
) -> list[dict]:
    """Parse validator summary lines from a CoT generation log, one record per profile.

    When profile_labels is provided (loaded from the original profiles JSONL
    via load_profile_labels), it is used as the primary label source for all
    IDs — this is the recommended path since check_bert_label warnings only
    fire for a subset of attempts.

    Without profile_labels, falls back to a two-pass approach:
      Pass 1 — extract gt label from check_bert_label warning lines keyed by
               profile ID.
      Pass 2 — extract faith/bert/quality/status from summary lines and join
               with gt label by profile ID.

    Deduplication: each profile ID is stored in a dict and overwritten on each
    new summary line encountered. Since retries appear in log order, the last
    entry for each ID represents the final outcome after all retries are
    exhausted — and that is the one returned.

    Structural failures (faith=0, bert=0) are kept; they represent a genuine
    final outcome when all retries failed a binary gate.
    """

    # Pattern for check_bert_label warning line
    # e.g. check_bert_label [medbert_er_reason, ID 123abc]: predicted 1 (gt=0), ...
    label_pattern = re.compile(
        r"check_bert_label\s+\[.*?,\s*ID\s+(\S+)\].*?\(gt=(\d+)\)"
    )

    # Pattern for validator summary line
    # e.g. [medbert_er_reason] ID 123abc — ... | faith=0.87 bert=0.65 quality=0.76 | FAILED ✗
    summary_pattern = re.compile(
        r"\]\s+ID\s+(\S+)\s+—.*?faith=([\d.]+)\s+bert=([\d.]+)\s+quality=([\d.]+)\s+\|\s+(APPROVED|FAILED)"
    )

    # Pass 1 — build {profile_id: gt_label} from warning lines
    # If a profile has multiple retries, all retries have the same gt label
    # so last-seen is fine
    gt_labels: dict[str, int] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = label_pattern.search(line)
            if m:
                pid = m.group(1)
                gt = int(m.group(2))
                gt_labels[pid] = gt

    # Pass 2 — parse summary lines, keeping only the last entry per profile ID
    records_by_id: dict[str, dict] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = summary_pattern.search(line)
            if m:
                pid = m.group(1)
                faith = float(m.group(2))
                bert = float(m.group(3))
                quality = float(m.group(4))
                status = m.group(5)

                if profile_labels is not None:
                    label = profile_labels.get(pid)
                else:
                    label = gt_labels.get(pid)

                records_by_id[pid] = {
                    "id": pid,
                    "faith": faith,
                    "bert": bert,
                    "quality": quality,
                    "status": status,
                    "label": label,
                }

    records = list(records_by_id.values())

    n_missing = sum(1 for r in records if r["label"] is None)
    if n_missing > 0:
        if profile_labels is not None:
            print(
                f"Warning: {n_missing} profiles have no label in the profiles file "
                f"— IDs in the log may not match profile IDs."
            )
        else:
            print(
                f"Warning: {n_missing} profiles have no gt label — "
                f"provide --profiles-path for complete label coverage."
            )

    return records


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def desc_stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def threshold_sensitivity(values: list[float], thresholds: list[float]) -> list[dict]:
    arr = np.array(values)
    results = []
    for t in thresholds:
        passing = arr[arr >= t]
        results.append(
            {
                "threshold": t,
                "pass_rate": float(len(passing) / len(arr)),
                "n_passing": len(passing),
                "n_failing": len(arr) - len(passing),
                "mean_passing": float(np.mean(passing)) if len(passing) > 0 else 0.0,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    records: list[dict],
    faith_threshold: float,
    bert_threshold: float,
    dataset: str,
    checkpoint: str,
    log_file: str,
    output_dir: Path,
) -> None:
    faith_all = [r["faith"] for r in records]
    bert_all = [r["bert"] for r in records]
    quality_all = [r["quality"] for r in records]

    approved = [r for r in records if r["status"] == "APPROVED"]
    failed = [r for r in records if r["status"] == "FAILED"]

    lines = []
    lines.append("=" * 70)
    lines.append(f"COT SCORE DISTRIBUTION ANALYSIS — {dataset.upper()}")
    lines.append(rf"Checkpoint : {checkpoint}")
    lines.append(f"Log File : {log_file}")
    lines.append(
        f"Log contains {len(records)} profiles (final outcome per ID) "
        f"({len(approved)} APPROVED, {len(failed)} FAILED)"
    )
    lines.append(f"Thresholds : faith >= {faith_threshold}, bert >= {bert_threshold}")
    lines.append("=" * 70)

    # Overall distributions
    for label, values in [
        ("Faithfulness (all)", faith_all),
        ("BERT confidence (all)", bert_all),
        ("Quality score (all)", quality_all),
    ]:
        s = desc_stats(values)
        lines.append(f"\n{label}:")
        lines.append(
            f"  mean={s['mean']:.4f}  median={s['median']:.4f}  std={s['std']:.4f}"
        )
        lines.append(
            f"  min={s['min']:.4f}  p25={s['p25']:.4f}  "
            f"p75={s['p75']:.4f}  max={s['max']:.4f}"
        )

    # APPROVED vs FAILED breakdown
    lines.append(f"\n{'─' * 70}")
    lines.append("APPROVED vs FAILED breakdown (final outcome per profile):")
    for group_label, group in [("APPROVED", approved), ("FAILED", failed)]:
        if not group:
            continue
        gf = [r["faith"] for r in group]
        gb = [r["bert"] for r in group]
        lines.append(f"\n  {group_label} (n={len(group)}):")
        lines.append(
            f"    faith — mean={np.mean(gf):.4f}  "
            f"median={np.median(gf):.4f}  std={np.std(gf):.4f}"
        )
        lines.append(
            f"    bert  — mean={np.mean(gb):.4f}  "
            f"median={np.median(gb):.4f}  std={np.std(gb):.4f}"
        )

    # Failed profiles broken down by label
    failed_with_label = [r for r in failed if r["label"] is not None]
    if failed_with_label:
        lines.append("\n  FAILED profiles by ground truth label:")
        for lbl in sorted(set(r["label"] for r in failed_with_label)):
            group = [r for r in failed_with_label if r["label"] == lbl]
            gf = [r["faith"] for r in group]
            gb = [r["bert"] for r in group]
            lines.append(
                f"    Label {lbl} (n={len(group)}): "
                f"faith mean={np.mean(gf):.4f}  "
                f"bert mean={np.mean(gb):.4f}"
            )

    # Profiles failing faith gate vs bert gate
    faith_ok_bert_not = [
        r
        for r in failed
        if r["faith"] >= faith_threshold and r["bert"] < bert_threshold
    ]
    lines.append(
        f"\n  Profiles passing faith >= {faith_threshold} "
        f"but failing bert < {bert_threshold}: "
        f"{len(faith_ok_bert_not)} "
        f"({100 * len(faith_ok_bert_not) / len(records):.1f}% of all profiles)"
    )
    if faith_ok_bert_not:
        bv = [r["bert"] for r in faith_ok_bert_not]
        lines.append(
            f"    bert scores — mean={np.mean(bv):.4f}  median={np.median(bv):.4f}"
        )

    # BERT threshold sensitivity
    lines.append(f"\n{'─' * 70}")
    lines.append(
        f"BERT threshold sensitivity (among profiles with faith >= {faith_threshold}):"
    )
    lines.append(
        f"  {'Threshold':<12} {'Pass rate':<12} {'N passing':<12} "
        f"{'N failing':<12} {'Mean (passing)'}"
    )
    faith_ok_bert = [r["bert"] for r in records if r["faith"] >= faith_threshold]
    for res in threshold_sensitivity(
        faith_ok_bert, [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    ):
        lines.append(
            f"  {res['threshold']:<12.2f} "
            f"{res['pass_rate']:<12.1%} "
            f"{res['n_passing']:<12} "
            f"{res['n_failing']:<12} "
            f"{res['mean_passing']:.4f}"
        )

    # Faithfulness threshold sensitivity
    lines.append(f"\n{'─' * 70}")
    lines.append("Faithfulness threshold sensitivity (all profiles):")
    lines.append(
        f"  {'Threshold':<12} {'Pass rate':<12} {'N passing':<12} "
        f"{'N failing':<12} {'Mean (passing)'}"
    )
    for res in threshold_sensitivity(
        faith_all, [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    ):
        lines.append(
            f"  {res['threshold']:<12.2f} "
            f"{res['pass_rate']:<12.1%} "
            f"{res['n_passing']:<12} "
            f"{res['n_failing']:<12} "
            f"{res['mean_passing']:.4f}"
        )

    lines.append(f"\n{'=' * 70}")
    report_text = "\n".join(lines)
    print(report_text)

    report_path = output_dir / "score_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distributions(
    records: list[dict],
    faith_threshold: float,
    bert_threshold: float,
    dataset: str,
    id2label: dict[int, str],
    output_dir: Path,
) -> None:
    faith_all = np.array([r["faith"] for r in records])
    bert_all = np.array([r["bert"] for r in records])
    approved = [r for r in records if r["status"] == "APPROVED"]
    failed = [r for r in records if r["status"] == "FAILED"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"CoT Score Distributions — {dataset.upper()}", fontsize=14, fontweight="bold"
    )

    colors = {"APPROVED": "#2196F3", "FAILED": "#FF5722", "all": "#9C27B0"}

    # --- Plot 1: Faithfulness histogram ---
    ax = axes[0]
    ax.hist(
        faith_all,
        bins=30,
        alpha=0.7,
        color=colors["all"],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        faith_threshold,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"threshold={faith_threshold}",
    )
    ax.axvline(
        np.mean(faith_all),
        color="orange",
        linestyle="dashed",
        linewidth=1.5,
        label=f"mean={np.mean(faith_all):.3f}",
    )
    ax.axvline(
        np.median(faith_all),
        color="green",
        linestyle="dashed",
        linewidth=1.5,
        label=f"median={np.median(faith_all):.3f}",
    )
    ax.set_title("Faithfulness Score — All Profiles (final outcome)")
    ax.set_xlabel("Faithfulness Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # --- Plot 2: BERT confidence histogram ---
    # ax = axes[0][1]
    # ax.hist(
    #     bert_all,
    #     bins=30,
    #     alpha=0.7,
    #     color=colors["all"],
    #     edgecolor="white",
    #     linewidth=0.5,
    # )
    # ax.axvline(
    #     bert_threshold,
    #     color="red",
    #     linestyle="dashed",
    #     linewidth=2,
    #     label=f"threshold={bert_threshold}",
    # )
    # ax.axvline(
    #     np.mean(bert_all),
    #     color="orange",
    #     linestyle="dashed",
    #     linewidth=1.5,
    #     label=f"mean={np.mean(bert_all):.3f}",
    # )
    # ax.axvline(
    #     np.median(bert_all),
    #     color="green",
    #     linestyle="dashed",
    #     linewidth=1.5,
    #     label=f"median={np.median(bert_all):.3f}",
    # )
    # ax.set_title("BERT Confidence — All Profiles (final outcome)")
    # ax.set_xlabel("BERT Confidence Score")
    # ax.set_ylabel("Count")
    # ax.legend(fontsize=8)

    # --- Plot 3: Faithfulness APPROVED vs FAILED ---
    # ax = axes[1][0]
    # if approved:
    #     ax.hist(
    #         [r["faith"] for r in approved],
    #         bins=20,
    #         alpha=0.6,
    #         color=colors["APPROVED"],
    #         label=f"APPROVED (n={len(approved)})",
    #         edgecolor="white",
    #         linewidth=0.5,
    #     )
    # if failed:
    #     ax.hist(
    #         [r["faith"] for r in failed],
    #         bins=20,
    #         alpha=0.6,
    #         color=colors["FAILED"],
    #         label=f"FAILED (n={len(failed)})",
    #         edgecolor="white",
    #         linewidth=0.5,
    #     )
    # ax.axvline(
    #     faith_threshold,
    #     color="red",
    #     linestyle="dashed",
    #     linewidth=2,
    #     label=f"threshold={faith_threshold}",
    # )
    # ax.set_title("Faithfulness — APPROVED vs FAILED Profiles (final outcome)")
    # ax.set_xlabel("Faithfulness Score")
    # ax.set_ylabel("Count")
    # ax.legend(fontsize=8)

    # --- Plot 4: Scatter — color=threshold pass, marker=label ---
    # One point per attempt, no double plotting
    ax = axes[1]

    # color encodes whether faith >= faith_threshold AND bert >= bert_threshold
    # marker encodes ground truth label
    scatter_groups = [
        (True, 0, "s", colors["APPROVED"], 0.8, 40),
        (True, 1, "^", colors["APPROVED"], 0.8, 50),
        (False, 0, "s", colors["FAILED"], 0.4, 25),
        (False, 1, "^", colors["FAILED"], 0.4, 35),
    ]

    for passes_both, lbl, marker, color, alpha, size in scatter_groups:
        group = [
            r
            for r in records
            if (r["faith"] >= faith_threshold and r["bert"] >= bert_threshold)
            == passes_both
            and r["label"] == lbl
        ]
        if not group:
            continue
        lbl_name = id2label.get(lbl, str(lbl))
        threshold_label = "passes both" if passes_both else "fails ≥1"
        ax.scatter(
            [r["faith"] for r in group],
            [r["bert"] for r in group],
            alpha=alpha,
            color=color,
            marker=marker,
            label=f"{threshold_label} | {lbl_name} (n={len(group)})",
            s=size,
            zorder=3 if passes_both else 2,
            edgecolors="black",
        )

    # Plot any attempts with missing labels separately
    no_label = [r for r in records if r["label"] is None]
    if no_label:
        ax.scatter(
            [r["faith"] for r in no_label],
            [r["bert"] for r in no_label],
            alpha=0.3,
            color="grey",
            marker="s",
            label=f"no label (n={len(no_label)})",
            s=20,
            zorder=1,
        )

    ax.axvline(
        faith_threshold, color="red", linestyle="dashed", linewidth=1.5, alpha=0.7
    )
    ax.axhline(
        bert_threshold, color="darkred", linestyle="dashed", linewidth=1.5, alpha=0.7
    )
    ax.annotate(
        f"faith={faith_threshold}",
        xy=(faith_threshold + 0.01, ax.get_ylim()[0] + 0.02),
        fontsize=7,
        color="red",
    )
    ax.annotate(
        f"bert={bert_threshold}",
        xy=(ax.get_xlim()[0] + 0.01, bert_threshold + 0.02),
        fontsize=7,
        color="darkred",
    )
    ax.set_title(
        "Faithfulness vs BERT Confidence — Final Outcome per Profile\n"
        "color=both thresholds satisfied (blue) | shape=ground truth label  "
        "(square=label 0, triangle=label 1)"
    )
    ax.set_xlabel("Faithfulness Score")
    ax.set_ylabel("BERT Confidence Score")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plot_path = output_dir / "score_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CoT score distributions from generation logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
        help="Path to the CoT generation .log file.",
    )
    parser.add_argument(
        "--profiles-path",
        type=str,
        default=None,
        help=(
            "Path to the profiles JSONL file (e.g. data/processed/finbench/ld1/cot/ld1_cot_val_*.jsonl). "
            "Used to look up gt labels by ID for all attempts. "
            "Recommended — without this, only IDs that appear in check_bert_label "
            "warning lines will have labels."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/score_analysis",
        help="Directory to save report and plots.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset",
        help="Dataset name for plot titles (e.g. er_reason, cc3, ld1).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="unknown",
        help="Validator checkpoint name to include in the report.",
    )
    parser.add_argument(
        "--faith-threshold",
        type=float,
        default=0.8,
        help="Faithfulness threshold to visualize.",
    )
    parser.add_argument(
        "--bert-threshold",
        type=float,
        default=0.75,
        help="BERT confidence threshold to visualize.",
    )
    parser.add_argument(
        "--id2label",
        type=str,
        default="0=label0,1=label1",
        help=(
            "Label mapping for scatter plot legend, e.g. "
            "'0=Discharge,1=Admit' or '0=repaid,1=defaulted'."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    id2label = {}
    for pair in args.id2label.split(","):
        k, v = pair.strip().split("=")
        id2label[int(k)] = v

    profile_labels = None
    if args.profiles_path:
        print(f"Loading profile labels from: {args.profiles_path}")
        profile_labels = load_profile_labels(Path(args.profiles_path))
        print(f"Loaded {len(profile_labels)} profile labels.")

    print(f"Parsing log: {args.log_path}")
    records = parse_log(Path(args.log_path), profile_labels=profile_labels)
    print(f"Found {len(records)} profiles (deduplicated to final outcome per ID)")

    if not records:
        print("No scored attempts found in log. Check log format.")
        exit(1)

    print_report(
        records,
        faith_threshold=args.faith_threshold,
        bert_threshold=args.bert_threshold,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        log_file=args.log_path,
        output_dir=output_dir,
    )
    plot_distributions(
        records,
        faith_threshold=args.faith_threshold,
        bert_threshold=args.bert_threshold,
        dataset=args.dataset,
        id2label=id2label,
        output_dir=output_dir,
    )
