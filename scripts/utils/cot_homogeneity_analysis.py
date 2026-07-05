"""
CoT step-text homogeneity analysis.

Tests the hypothesis that ER-REASON CoT profiles are more textually homogeneous
(same-class profiles are more similar) than LD1 profiles, which would explain
near-perfect standard accuracy on ER-REASON alongside catastrophic counterfactual
collapse (BERT_cot learning surface patterns instead of transferable reasoning).

Measure: mean pairwise TF-IDF cosine similarity among same-class, same-step texts.
Groups:  2 datasets × 2 classes × 4 steps = 16 groups.

Usage:
    python scripts/utils/cot_homogeneity_analysis.py
"""

import csv
import json
import logging
import re
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.logger import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

SEED = 42
MAX_SAMPLES_PER_CLASS = 300
N_STEPS = 4
STEP_PATTERN = re.compile(r"\[STEP\](.*?)\[EORS\]", re.DOTALL)
SPECIAL_TOKENS = re.compile(r"\[ES\]|\[STEP\]|\[EORS\]|\[EOA\]|\[ANSWER\]")

DATASET_CONFIGS = {
    "LD1": "configs/bert_based_models/finbert_ld1.yaml",
    "ER-REASON": "configs/bert_based_models/medbert_er_reason.yaml",
}


# ─── NLTK ──────────────────────────────────────────────────────────────────────

def _ensure_nltk_resources() -> set[str]:
    """Return English stopwords, downloading the corpus if absent."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        logger.info("NLTK stopwords not found — downloading...")
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


# ─── Config / data loading ─────────────────────────────────────────────────────

def _load_cot_train_path(config_rel: str) -> Path:
    config_path = PROJECT_ROOT / config_rel
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return PROJECT_ROOT / cfg["dataset"]["cot_train"]


def _load_records(path: Path, dataset_name: str) -> list[dict]:
    if not path.exists():
        logger.warning(f"[{dataset_name}] CoT train file not found: {path} — skipping dataset.")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cot_text = rec.get("cot_text", "")
            label = rec.get("label")
            if cot_text and cot_text.strip() and label is not None:
                records.append({"cot_text": cot_text.strip(), "label": int(label)})
    logger.info(f"[{dataset_name}] Loaded {len(records)} valid records from {path}")
    return records


# ─── Sampling ──────────────────────────────────────────────────────────────────

def _sample_by_class(
    records: list[dict],
    dataset_name: str,
    rng: np.random.Generator,
) -> dict[int, list[dict]]:
    by_class: dict[int, list[dict]] = {}
    for rec in records:
        by_class.setdefault(rec["label"], []).append(rec)

    sampled: dict[int, list[dict]] = {}
    for cls, group in sorted(by_class.items()):
        if len(group) <= MAX_SAMPLES_PER_CLASS:
            logger.warning(
                f"[{dataset_name}] class {cls}: only {len(group)} records available "
                f"(fewer than {MAX_SAMPLES_PER_CLASS}), using all."
            )
            sampled[cls] = list(group)
        else:
            indices = rng.choice(len(group), size=MAX_SAMPLES_PER_CLASS, replace=False)
            sampled[cls] = [group[i] for i in indices]
            logger.info(f"[{dataset_name}] class {cls}: sampled {MAX_SAMPLES_PER_CLASS}/{len(group)} records.")
    return sampled


# ─── Step extraction ───────────────────────────────────────────────────────────

def _extract_steps(cot_text: str) -> list[str] | None:
    """Extract exactly N_STEPS step texts; return None if count doesn't match."""
    matches = STEP_PATTERN.findall(cot_text)
    if len(matches) != N_STEPS:
        return None
    return [SPECIAL_TOKENS.sub(" ", m).strip() for m in matches]


# ─── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str, stop_words: set[str]) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in stop_words]
    return " ".join(tokens)


# ─── Similarity ────────────────────────────────────────────────────────────────

def _mean_pairwise_similarity(texts: list[str]) -> float:
    """Mean TF-IDF cosine similarity over the upper triangle (no diagonal)."""
    if len(texts) < 2:
        return float("nan")
    tfidf = TfidfVectorizer().fit_transform(texts)
    sim = cosine_similarity(tfidf)
    n = sim.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return float(np.mean(sim[rows, cols]))


# ─── Core analysis ─────────────────────────────────────────────────────────────

def run_analysis(stop_words: set[str]) -> list[dict]:
    rng = np.random.default_rng(SEED)
    results: list[dict] = []

    for dataset_name, config_rel in DATASET_CONFIGS.items():
        logger.info(f"=== Dataset: {dataset_name} ===")
        cot_train_path = _load_cot_train_path(config_rel)
        records = _load_records(cot_train_path, dataset_name)
        if not records:
            continue

        sampled_by_class = _sample_by_class(records, dataset_name, rng)

        # Collect cleaned step texts: step_texts[cls][step_idx] → list[str]
        step_texts: dict[int, dict[int, list[str]]] = {
            cls: {s: [] for s in range(N_STEPS)}
            for cls in sampled_by_class
        }
        skipped = 0

        for cls, group in sampled_by_class.items():
            for rec in group:
                steps = _extract_steps(rec["cot_text"])
                if steps is None:
                    actual = len(STEP_PATTERN.findall(rec["cot_text"]))
                    logger.warning(
                        f"[{dataset_name}] class={cls}: expected {N_STEPS} steps, "
                        f"got {actual} — skipping record."
                    )
                    skipped += 1
                    continue
                for step_idx, step_text in enumerate(steps):
                    step_texts[cls][step_idx].append(_clean_text(step_text, stop_words))

        if skipped:
            logger.warning(f"[{dataset_name}] Total skipped records: {skipped}")

        for cls in sorted(step_texts):
            for step_idx in range(N_STEPS):
                texts = step_texts[cls][step_idx]
                n = len(texts)
                logger.info(
                    f"[{dataset_name}] class={cls} step={step_idx + 1}: "
                    f"{n} texts — computing similarity..."
                )
                mean_sim = _mean_pairwise_similarity(texts)
                logger.info(
                    f"[{dataset_name}] class={cls} step={step_idx + 1}: "
                    f"mean_similarity={mean_sim:.4f}"
                )
                results.append({
                    "dataset": dataset_name,
                    "step": step_idx + 1,
                    "class": cls,
                    "mean_similarity": round(mean_sim, 4),
                    "n_samples": n,
                })

    return results


# ─── CSV output ────────────────────────────────────────────────────────────────

def _save_csv(results: list[dict], out_dir: Path) -> None:
    out_path = out_dir / "step_similarity.csv"
    fieldnames = ["dataset", "step", "class", "mean_similarity", "n_samples"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV saved to {out_path}")


# ─── Plot ──────────────────────────────────────────────────────────────────────

def _plot(results: list[dict], out_dir: Path) -> None:
    # Blue shades for LD1, orange shades for ER-REASON; lighter = class 0
    bar_configs = [
        ("LD1",       0, "#90CAF9", "LD1 class 0"),
        ("LD1",       1, "#1565C0", "LD1 class 1"),
        ("ER-REASON", 0, "#FFCC80", "ER-REASON class 0"),
        ("ER-REASON", 1, "#E65100", "ER-REASON class 1"),
    ]

    by_key: dict[tuple, float] = {
        (r["dataset"], r["step"], r["class"]): r["mean_similarity"]
        for r in results
    }

    steps = [1, 2, 3, 4]
    n_bars = len(bar_configs)
    bar_height = 0.18
    group_pitch = n_bars * bar_height + 0.15  # vertical space per step group

    fig, ax = plt.subplots(figsize=(10, 7))

    for gi, step in enumerate(steps):
        base_y = gi * group_pitch
        for bi, (ds, cls, color, _) in enumerate(bar_configs):
            val = by_key.get((ds, step, cls), float("nan"))
            y = base_y + bi * bar_height
            bar_label = bar_configs[bi][3] if gi == 0 else "_nolegend_"
            if not np.isnan(val):
                ax.barh(y, val, height=bar_height * 0.85, color=color, label=bar_label)
                ax.text(val + 0.005, y, f"{val:.3f}", va="center", fontsize=7.5)
            else:
                ax.barh(y, 0, height=bar_height * 0.85, color=color, label=bar_label)
                ax.text(0.01, y, "N/A", va="center", fontsize=7.5, color="gray")

    group_centers = [
        gi * group_pitch + (n_bars - 1) * bar_height / 2
        for gi in range(len(steps))
    ]
    ax.set_yticks(group_centers)
    ax.set_yticklabels([f"Step {s}" for s in steps], fontsize=11)
    ax.set_xlabel("Mean TF-IDF Cosine Similarity", fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_title("CoT Step Text Homogeneity by Dataset and Class", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "step_similarity_chart.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved to {out_path}")


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(run_name="cot_homogeneity_analysis")
    out_dir = PROJECT_ROOT / "results" / "cot_homogeneity"
    out_dir.mkdir(parents=True, exist_ok=True)

    stop_words = _ensure_nltk_resources()
    results = run_analysis(stop_words)

    if not results:
        logger.error("No results produced — check that CoT train files exist.")
        sys.exit(1)

    _save_csv(results, out_dir)
    _plot(results, out_dir)
    logger.info("Analysis complete.")
