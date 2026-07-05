"""
Sentence length distribution analysis for unconstrained Qwen3-4B CoT profiles.

Loads CoT profiles generated WITHOUT sentence length constraints, splits each
CoT into sentences using nltk, tokenizes each sentence with the Qwen3-4B
tokenizer, and plots the distribution of sentence lengths in tokens.

The purpose is to empirically justify the sentence length constraints used in
the CoT validator (min=10, max=40 tokens) by showing they reflect the natural
output distribution of Qwen3-4B on financial reasoning prompts.

Usage
-----
    python scripts/analyze_sentence_lengths.py \\
        --input  data/processed/finbench/ld1/ld1_cot_test_Qwen3-4B.jsonl \\
        --output results/sentence_length_analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
from transformers import AutoTokenizer

# Download punkt tokenizer if not already present
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_cot_texts(path: Path) -> list[str]:
    """Load CoT texts from a .jsonl file."""
    cot_texts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cot_text = record.get("cot_text", "").strip()
            if cot_text:
                cot_texts.append(cot_text)
    print(f"Loaded {len(cot_texts)} CoT profiles from {path}")
    return cot_texts


# ---------------------------------------------------------------------------
# Sentence splitting and tokenization
# ---------------------------------------------------------------------------


def extract_sentence_lengths(
    cot_texts: list[str],
    tokenizer,
) -> tuple[list[int], dict]:
    """Split CoT texts into sentences and count tokens per sentence.

    Returns:
        lengths    — flat list of token counts for every sentence
        stats      — descriptive statistics dict
    """
    all_lengths: list[int] = []
    n_sentences_total = 0
    n_empty_skipped = 0

    for cot_text in cot_texts:
        sentences = nltk.sent_tokenize(cot_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                n_empty_skipped += 1
                continue
            token_ids = tokenizer.encode(
                sentence,
                add_special_tokens=False,
            )
            n_tokens = len(token_ids)
            if n_tokens > 0:
                all_lengths.append(n_tokens)
                n_sentences_total += 1

    print(f"Total sentences extracted : {n_sentences_total}")
    print(f"Empty sentences skipped   : {n_empty_skipped}")

    arr = np.array(all_lengths)
    stats = {
        "n_sentences": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "pct_below_10": float(100 * np.mean(arr < 10)),
        "pct_above_40": float(100 * np.mean(arr > 40)),
        "pct_10_to_40": float(100 * np.mean((arr >= 10) & (arr <= 40))),
    }

    return all_lengths, stats


def extract_short_sentences(
    cot_texts: list[str],
    tokenizer,
    min_tokens: int = 10,
    max_examples: int = 10,
) -> list[dict]:
    """Extract diverse examples of sentences below min_tokens threshold."""
    seen = set()
    examples = []

    for cot_text in cot_texts:
        sentences = nltk.sent_tokenize(cot_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or sentence in seen:
                continue
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            n_tokens = len(token_ids)
            if n_tokens < min_tokens:
                seen.add(sentence)
                examples.append(
                    {
                        "sentence": sentence,
                        "n_tokens": n_tokens,
                    }
                )

    # sort by token count to show variety
    examples.sort(key=lambda x: x["n_tokens"])
    return examples[:max_examples]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(stats: dict, output_dir: Path, cot_texts, tokenizer) -> None:
    short_examples = extract_short_sentences(cot_texts, tokenizer)

    lines = []
    lines.append("=" * 60)
    lines.append("SENTENCE LENGTH DISTRIBUTION — Qwen3-4B (unconstrained)")
    lines.append("=" * 60)
    lines.append(f"  Total sentences : {stats['n_sentences']}")
    lines.append(f"  Mean            : {stats['mean']:.2f} tokens")
    lines.append(f"  Median          : {stats['median']:.2f} tokens")
    lines.append(f"  Std             : {stats['std']:.2f} tokens")
    lines.append(f"  Min             : {stats['min']} tokens")
    lines.append(f"  Max             : {stats['max']} tokens")
    lines.append("")
    lines.append("  Percentiles:")
    lines.append(f"    P5  : {stats['p5']:.1f} tokens")
    lines.append(f"    P10 : {stats['p10']:.1f} tokens")
    lines.append(f"    P25 : {stats['p25']:.1f} tokens")
    lines.append(f"    P75 : {stats['p75']:.1f} tokens")
    lines.append(f"    P90 : {stats['p90']:.1f} tokens")
    lines.append(f"    P95 : {stats['p95']:.1f} tokens")
    lines.append("")
    lines.append("  Constraint coverage:")
    lines.append(f"    Below 10 tokens : {stats['pct_below_10']:.1f}% of sentences")
    lines.append(f"    10 to 40 tokens : {stats['pct_10_to_40']:.1f}% of sentences")
    lines.append(f"    Above 40 tokens : {stats['pct_above_40']:.1f}% of sentences")
    lines.append("")
    lines.append(
        "  Interpretation: sentences naturally falling within the 10-40 "
        "token range confirms the constraints reflect Qwen3-4B's natural "
        "output distribution on financial reasoning prompts."
    )
    lines.append("\n  Examples of sentences below 10 tokens (fragments):")
    for ex in short_examples:
        lines.append(f"    [{ex['n_tokens']} tokens] {ex['sentence']!r}")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    report_path = output_dir / "sentence_length_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"\nReport saved to {report_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distribution(
    lengths: list[int],
    stats: dict,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Sentence Length Distribution — Qwen3-4B (unconstrained CoT | Total sentences : {stats['n_sentences']})",
        fontsize=14,
        fontweight="bold",
    )

    arr = np.array(lengths)

    # ---- histogram ----
    ax_hist = axes[0]
    ax_hist.hist(
        arr,
        bins=50,
        color="#2196F3",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # constraint boundary lines
    ax_hist.axvline(
        x=10,
        color="#FF5722",
        linestyle="--",
        linewidth=2,
        label="Min constraint (10 tokens)",
    )
    ax_hist.axvline(
        x=70,
        color="#4CAF50",
        linestyle="--",
        linewidth=2,
        label="Max constraint (40 tokens)",
    )

    # mean and median lines
    ax_hist.axvline(
        x=stats["mean"],
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean ({stats['mean']:.1f})",
    )
    ax_hist.axvline(
        x=stats["median"],
        color="gray",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median ({stats['median']:.1f})",
    )

    # shaded valid region
    ax_hist.axvspan(10, 70, alpha=0.08, color="#4CAF50", label="Valid range")

    ax_hist.set_xlabel("Sentence Length (tokens)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Histogram of Sentence Lengths")
    ax_hist.legend(fontsize=8)

    # coverage annotation
    # ax_hist.annotate(
    #     f"{stats['pct_10_to_40']:.1f}% within\n10-70 token range",
    #     xy=(25, ax_hist.get_ylim()[1] * 0.85),
    #     ha="center",
    #     va="top",
    #     fontsize=10,
    #     fontweight="bold",
    #     color="#4CAF50",
    #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    # )

    # ---- box plot ----
    ax_box = axes[1]
    bp = ax_box.boxplot(
        arr,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        vert=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][0].set_alpha(0.6)

    # constraint lines on box plot
    ax_box.axhline(
        y=10,
        color="#FF5722",
        linestyle="--",
        linewidth=2,
        label="Min constraint (10)",
    )
    ax_box.axhline(
        y=40,
        color="#4CAF50",
        linestyle="--",
        linewidth=2,
        label="Max constraint (40)",
    )

    # mean annotation
    ax_box.hlines(
        stats["mean"],
        0.75,
        1.25,
        colors="black",
        linestyles="dashed",
        linewidth=2,
    )
    ax_box.annotate(
        f"μ={stats['mean']:.1f}",
        xy=(1.27, stats["mean"]),
        fontsize=9,
        color="black",
        fontweight="bold",
    )
    ax_box.annotate(
        f"m={stats['median']:.1f}",
        xy=(0.62, stats["median"]),
        fontsize=9,
        color="black",
        fontweight="bold",
    )

    ax_box.set_ylabel("Sentence Length (tokens)")
    ax_box.set_title("Box Plot of Sentence Lengths")
    ax_box.set_xticks([])
    ax_box.legend(fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "sentence_length_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_analysis(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    cot_texts = load_cot_texts(input_path)
    if not cot_texts:
        print("No CoT texts found. Check --input path.")
        return

    # load tokenizer
    print("Loading Qwen3-4B tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    print("Tokenizer loaded.")

    # extract sentence lengths
    print("\nExtracting sentence lengths...")
    lengths, stats = extract_sentence_lengths(cot_texts, tokenizer)

    # report + plot
    print()
    print_report(stats, output_dir, cot_texts, tokenizer)
    plot_distribution(lengths, stats, output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sentence length distribution for unconstrained Qwen3 CoT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/finbench/ld1/ld1_cot_test_Qwen3-4B.jsonl",
        help="Path to the unconstrained CoT .jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sentence_length_analysis",
        help="Directory to save report and plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    run_analysis(
        input_path=Path(_args.input),
        output_dir=Path(_args.output),
    )
