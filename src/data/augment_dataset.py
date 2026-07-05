"""
Dataset augmentation pipeline for CoT-based knowledge distillation.

Combines original CoT profiles with selected counterfactual (CF) CoT profiles
to produce balanced augmented datasets for student model training.

Label conventions:
  LD1       — 0=repaid, 1=defaulted
  ER-REASON — 0=discharge, 1=admit
"""

import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Dataset routing config
# ---------------------------------------------------------------------------

DATASET_CONFIG: dict[str, dict] = {
    "ld1": {
        "path_prefix": "finbench/ld1",
        "file_prefix": "ld1",
    },
    "er_reason": {
        "path_prefix": "er_reason",
        "file_prefix": "er_reason",
    },
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _save_jsonl(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _label_counts(profiles: list[dict], label_field: str = "label") -> tuple[int, int]:
    """Return (N_class0, N_class1)."""
    n0 = sum(1 for p in profiles if p[label_field] == 0)
    n1 = sum(1 for p in profiles if p[label_field] == 1)
    return n0, n1


# ---------------------------------------------------------------------------
# Step 2 — Target ratio
# ---------------------------------------------------------------------------


def get_target_ratio(dataset: str, cot_profiles: list[dict]) -> float:
    """Return target ratio for class 0 after augmentation."""
    if dataset == "ld1":
        return 0.50
    if dataset == "er_reason":
        n0, n1 = _label_counts(cot_profiles)
        return n0 / (n0 + n1)
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Step 3 — Pool A
# ---------------------------------------------------------------------------


def get_pool_a(dataset: str, cf_cot_profiles: list[dict]) -> list[dict]:
    """Return Pool A entries (CF CoT profiles to add unconditionally).

    LD1:       empty — no admit-equivalent class to boost.
    ER-REASON: all admit→discharge flips (counterfactual_label=0).
    """
    if dataset == "ld1":
        return []
    if dataset == "er_reason":
        return [p for p in cf_cot_profiles if p["counterfactual_label"] == 0]
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Step 4 — Pool B
# ---------------------------------------------------------------------------


def get_pool_b(dataset: str, cf_cot_profiles: list[dict]) -> tuple[list[dict], int]:
    """Return (pool_b sorted by confidence desc, n_discarded).

    LD1:       repaid→defaulted flips (counterfactual_label=1); logs discarded defaulted→repaid.
    ER-REASON: discharge→admit flips (counterfactual_label=1).
    """
    if dataset == "ld1":
        pool_b = [p for p in cf_cot_profiles if p["counterfactual_label"] == 1]
        n_discarded = sum(1 for p in cf_cot_profiles if p["counterfactual_label"] == 0)
        if n_discarded:
            logger.info(
                f"Pool B (LD1): discarding {n_discarded} defaulted→repaid flip(s)"
            )
        pool_b_sorted = sorted(
            pool_b,
            key=lambda x: x.get("target_label_confidence", 0.0),
            reverse=True,
        )
        return pool_b_sorted, n_discarded

    if dataset == "er_reason":
        pool_b = [p for p in cf_cot_profiles if p["counterfactual_label"] == 1]
        pool_b_sorted = sorted(
            pool_b,
            key=lambda x: x.get("target_label_confidence", 0.0),
            reverse=True,
        )
        return pool_b_sorted, 0

    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Step 5 — Compute how many Pool B entries to take
# ---------------------------------------------------------------------------


def compute_pool_b_count(
    cot_profiles: list[dict],
    pool_a: list[dict],
    pool_b: list[dict],
    target_ratio: float,
    soft_threshold: float,
) -> tuple[int, bool, float]:
    """Compute how many top-confidence Pool B CFs are needed to hit target_ratio ± soft_threshold.

    Pool B always adds class-1 entries.  Pool A may add class-0 or class-1 entries
    depending on its counterfactual_label.

    Returns:
        n_to_take       — number of Pool B entries to select
        pool_b_exhausted — True if Pool B ran out before reaching the target
        achieved_ratio  — actual ratio_class0 after augmentation
    """
    n0_cot, n1_cot = _label_counts(cot_profiles)

    n0_a = sum(1 for p in pool_a if p["counterfactual_label"] == 0)
    n1_a = sum(1 for p in pool_a if p["counterfactual_label"] == 1)

    n0_after_a = n0_cot + n0_a
    n1_after_a = n1_cot + n1_a
    total_after_a = n0_after_a + n1_after_a

    # Solve for N_B:  target_ratio = n0_after_a / (total_after_a + N_B)
    if 0 < target_ratio < 1:
        n_needed = max(0, round(n0_after_a / target_ratio - total_after_a))
    else:
        n_needed = 0

    n_available = len(pool_b)

    if n_needed <= n_available:
        n_to_take = n_needed
        pool_b_exhausted = False
    else:
        n_to_take = n_available
        pool_b_exhausted = True
        logger.warning(
            f"Pool B exhausted: needed {n_needed} entries, only {n_available} available."
        )

    total_final = total_after_a + n_to_take
    achieved_ratio = n0_after_a / total_final if total_final > 0 else 0.0

    if pool_b_exhausted:
        logger.warning(
            f"Achieved ratio_class0={achieved_ratio:.4f} "
            f"(target={target_ratio:.4f}, delta={abs(achieved_ratio - target_ratio):.4f})"
        )

    return n_to_take, pool_b_exhausted, achieved_ratio


# ---------------------------------------------------------------------------
# Step 6 — Collect selected CF IDs
# ---------------------------------------------------------------------------


def collect_selected_ids(pool_a: list[dict], pool_b_selected: list[dict]) -> list[str]:
    """Return flat list of cf_-prefixed string IDs from Pool A then Pool B."""
    return [str(p["id"]) for p in pool_a] + [str(p["id"]) for p in pool_b_selected]


# ---------------------------------------------------------------------------
# Step 7 — Filter reconstructed plain CF profiles
# ---------------------------------------------------------------------------


def filter_plain_cfs(
    plain_cf_profiles: list[dict], selected_ids: list[str]
) -> list[dict]:
    """Keep only plain CF entries whose id appears in selected_ids."""
    selected_set = set(selected_ids)
    return [p for p in plain_cf_profiles if str(p["id"]) in selected_set]


# ---------------------------------------------------------------------------
# Step 8 — Build augmented CoT file
# ---------------------------------------------------------------------------


def build_augmented_cot(
    cot_profiles: list[dict],
    selected_cf_cot: list[dict],
    seed: int,
) -> list[dict]:
    """Combine original CoT entries with selected CF CoT entries and shuffle."""
    records: list[dict] = []

    for p in cot_profiles:
        records.append(
            {
                "id": str(p["id"]),
                "cot_text": p["cot_text"],
                "label": p["label"],
                "is_counterfactual": False,
                "original_id": None,
                "target_label_confidence": None,
            }
        )

    for cf in selected_cf_cot:
        records.append(
            {
                "id": str(cf["id"]),
                "cot_text": cf["counterfactual_cot_text"],
                "label": cf["counterfactual_label"],
                "is_counterfactual": True,
                "original_id": str(cf["original_id"]),
                "target_label_confidence": cf.get("target_label_confidence"),
            }
        )

    random.Random(seed).shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Step 9 — Build augmented original file
# ---------------------------------------------------------------------------


def build_augmented_original(
    cot_profiles: list[dict],
    filtered_plain_cfs: list[dict],
    selected_cf_cot: list[dict],
    seed: int,
) -> list[dict]:
    """Combine original plain entries with filtered plain CF entries and shuffle.

    Original text is read from the CoT profiles (which carry a "text" field).
    selected_cf_cot is used only to look up target_label_confidence per CF id.
    """
    cf_cot_map = {str(p["id"]): p for p in selected_cf_cot}

    records: list[dict] = []

    for p in cot_profiles:
        records.append(
            {
                "id": str(p["id"]),
                "text": p["text"],
                "label": p["label"],
                "is_counterfactual": False,
                "original_id": None,
                "target_label_confidence": None,
            }
        )

    for cf in filtered_plain_cfs:
        cf_id = str(cf["id"])
        confidence = cf_cot_map.get(cf_id, {}).get("target_label_confidence")
        records.append(
            {
                "id": cf_id,
                "text": cf["counterfactual_text"],
                "label": cf["counterfactual_label"],
                "is_counterfactual": True,
                "original_id": str(cf["original_id"]),
                "target_label_confidence": confidence,
            }
        )

    random.Random(seed).shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Step 10 — Build run manifest
# ---------------------------------------------------------------------------


def build_manifest(
    dataset: str,
    split: str,
    model_suff: str,
    dry_run: bool,
    data_paths: dict,
    cot_profiles: list[dict],
    cf_cot_profiles: list[dict],
    pool_a: list[dict],
    pool_b: list[dict],
    pool_b_discarded: int,
    n_pool_b_selected: int,
    target_ratio: float,
    achieved_ratio: float,
    soft_threshold: float,
) -> dict:
    cot_n0, cot_n1 = _label_counts(cot_profiles)

    cf_n0 = sum(1 for p in cf_cot_profiles if p["counterfactual_label"] == 0)
    cf_n1 = sum(1 for p in cf_cot_profiles if p["counterfactual_label"] == 1)
    cf_total = cf_n0 + cf_n1

    pool_a_n0 = sum(1 for p in pool_a if p["counterfactual_label"] == 0)
    pool_a_n1 = sum(1 for p in pool_a if p["counterfactual_label"] == 1)
    aug_n0 = cot_n0 + pool_a_n0
    aug_n1 = cot_n1 + pool_a_n1 + n_pool_b_selected
    aug_total = aug_n0 + aug_n1

    ratio_delta = abs(achieved_ratio - target_ratio)

    return {
        "dataset": dataset,
        "split": split,
        "model_suff": model_suff,
        "dry_run": dry_run,
        "data_paths": data_paths,
        "cot_split": {
            "N_class0": cot_n0,
            "N_class1": cot_n1,
            "ratio_class0": round(cot_n0 / (cot_n0 + cot_n1), 4)
            if (cot_n0 + cot_n1) > 0
            else 0.0,
        },
        "counterfactual_split": {
            "N_class0": cf_n0,
            "N_class1": cf_n1,
            "ratio_class0": round(cf_n0 / cf_total, 4) if cf_total > 0 else 0.0,
            "pool_a_available": len(pool_a),
            "pool_b_available": len(pool_b),
            "pool_b_discarded": pool_b_discarded,
        },
        "augmented_split": {
            "target_ratio": target_ratio,
            "N_class0": aug_n0,
            "N_class1": aug_n1,
            "ratio_class0": round(aug_n0 / aug_total, 4) if aug_total > 0 else 0.0,
            "pool_a_added": len(pool_a),
            "pool_b_added": n_pool_b_selected,
            "ratio_delta": round(ratio_delta, 4),
            "within_threshold": ratio_delta <= soft_threshold,
        },
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_augmentation(
    dataset: str,
    split: str,
    model_suff: str,
    data_dir: str = "data/processed",
    dry_run: bool = False,
    soft_threshold: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run the full augmentation pipeline for one split.

    Args:
        dataset:        One of "ld1", "er_reason".
        split:          "train", "val", or "test".
        model_suff:     File-naming suffix, e.g. "Qwen3-4b".
        data_dir:       Base data directory (default "data/processed").
        dry_run:        If True, skip Steps 8–9 and print manifest only.
        soft_threshold: Acceptable ± deviation from target_ratio.
        seed:           Shuffle seed for reproducibility.

    Returns:
        The manifest dict.
    """
    cfg = DATASET_CONFIG[dataset]
    prefix = cfg["file_prefix"]
    base_dir = Path(data_dir) / cfg["path_prefix"]

    cot_path = base_dir / "cot" / f"{prefix}_cot_{split}_{model_suff}.jsonl"
    cf_cot_path = (
        base_dir
        / "counterfactual_cot"
        / f"{prefix}_counterfactual_cot_{split}_{model_suff}.jsonl"
    )
    plain_cf_path = (
        base_dir
        / "counterfactual"
        / f"{prefix}_counterfactual_{split}_{model_suff}.jsonl"
    )
    aug_cot_path = (
        base_dir
        / "augmented"
        / "cot"
        / f"{prefix}_augmented_cot_{split}_{model_suff}.jsonl"
    )
    aug_orig_path = (
        base_dir
        / "augmented"
        / "original"
        / f"{prefix}_augmented_original_{split}_{model_suff}.jsonl"
    )
    manifest_path = (
        base_dir
        / "augmented"
        / "manifests"
        / f"{prefix}_augmentation_manifest_{split}_{model_suff}.json"
    )

    data_paths = {
        "cot": str(cot_path),
        "counterfactual_cot": str(cf_cot_path),
        "counterfactual": str(plain_cf_path),
    }

    logger.info("=" * 65)
    logger.info(
        f"Augmentation  |  dataset={dataset}, split={split}, model={model_suff}"
    )
    logger.info(f"  dry_run        : {dry_run}")
    logger.info(f"  soft_threshold : {soft_threshold}")
    logger.info(f"  seed           : {seed}")
    logger.info("=" * 65)

    # Step 1 — Load input files
    logger.info("Step 1: loading input files")
    cot_profiles = _load_jsonl(cot_path)
    cf_cot_profiles = _load_jsonl(cf_cot_path)
    plain_cf_profiles = _load_jsonl(plain_cf_path)
    logger.info(
        f"  CoT={len(cot_profiles)}  CF-CoT={len(cf_cot_profiles)}  "
        f"Plain-CF={len(plain_cf_profiles)}"
    )

    # Step 2 — Target ratio
    target_ratio = get_target_ratio(dataset, cot_profiles)
    logger.info(f"Step 2: target_ratio={target_ratio:.4f}")

    # Step 3 — Pool A
    pool_a = get_pool_a(dataset, cf_cot_profiles)
    logger.info(f"Step 3: Pool A — {len(pool_a)} entries")

    # Step 4 — Pool B
    pool_b, pool_b_discarded = get_pool_b(dataset, cf_cot_profiles)
    logger.info(
        f"Step 4: Pool B — {len(pool_b)} entries (discarded={pool_b_discarded})"
    )

    # Step 5 — Compute Pool B count
    n_to_take, pool_b_exhausted, achieved_ratio = compute_pool_b_count(
        cot_profiles, pool_a, pool_b, target_ratio, soft_threshold
    )
    pool_b_selected = pool_b[:n_to_take]
    logger.info(
        f"Step 5: taking {n_to_take}/{len(pool_b)} from Pool B "
        f"(exhausted={pool_b_exhausted}, achieved_ratio={achieved_ratio:.4f})"
    )

    # Step 6 — Collect selected CF IDs
    selected_ids = collect_selected_ids(pool_a, pool_b_selected)
    logger.info(f"Step 6: {len(selected_ids)} CF IDs selected")

    # Build manifest (always, even on dry run)
    manifest = build_manifest(
        dataset=dataset,
        split=split,
        model_suff=model_suff,
        dry_run=dry_run,
        data_paths=data_paths,
        cot_profiles=cot_profiles,
        cf_cot_profiles=cf_cot_profiles,
        pool_a=pool_a,
        pool_b=pool_b,
        pool_b_discarded=pool_b_discarded,
        n_pool_b_selected=n_to_take,
        target_ratio=target_ratio,
        achieved_ratio=achieved_ratio,
        soft_threshold=soft_threshold,
    )

    if dry_run:
        logger.info("Dry run — skipping Steps 7–9, printing manifest")
        print(json.dumps(manifest, indent=2))
        return manifest

    # Step 7 — Filter plain CFs
    filtered_plain_cfs = filter_plain_cfs(plain_cf_profiles, selected_ids)
    logger.info(f"Step 7: {len(filtered_plain_cfs)} plain CF profiles retained")

    selected_cf_cot = pool_a + pool_b_selected

    # Step 8 — Augmented CoT file
    logger.info("Step 8: building augmented CoT file")
    aug_cot = build_augmented_cot(cot_profiles, selected_cf_cot, seed)
    _save_jsonl(aug_cot, aug_cot_path)
    logger.info(f"  saved {len(aug_cot)} entries → {aug_cot_path}")

    # Step 9 — Augmented original file
    logger.info("Step 9: building augmented original file")
    aug_orig = build_augmented_original(
        cot_profiles, filtered_plain_cfs, selected_cf_cot, seed
    )
    _save_jsonl(aug_orig, aug_orig_path)
    logger.info(f"  saved {len(aug_orig)} entries → {aug_orig_path}")

    # Step 10 — Save manifest
    logger.info("Step 10: saving manifest")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  manifest → {manifest_path}")

    return manifest
