"""
Unit tests for src/data/augment_dataset.py.

All tests operate on in-memory synthetic data — no file I/O required.

Invariants tested
-----------------
1. Original and CF IDs never collide in augmented output
2. is_counterfactual is True for CF entries and False for originals
3. Achieved ratio is within soft_threshold when pool B is sufficient
4. Pool B entries are selected in descending confidence order
5. Shuffle is deterministic — same seed, same order every time
6. LD1 Pool A is always empty; ER-REASON Pool A contains only counterfactual_label=0 entries
7. Additional: pool B exhaustion, discarded-flip counting, filter/ID helpers, manifest fields
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.augment_dataset import (
    build_augmented_cot,
    build_augmented_original,
    build_manifest,
    collect_selected_ids,
    compute_pool_b_count,
    filter_plain_cfs,
    get_pool_a,
    get_pool_b,
    get_target_ratio,
)

# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def _cot(id_, label):
    return {"id": id_, "text": f"text_{id_}", "cot_text": f"cot_{id_}", "label": label}


def _cf_cot(num, original_id, original_label, cf_label, confidence):
    return {
        "id": f"cf_{num}",
        "original_id": original_id,
        "original_label": original_label,
        "counterfactual_label": cf_label,
        "counterfactual_cot_text": f"cf_cot_{num}",
        "target_label_confidence": confidence,
    }


def _plain_cf(num, original_id, original_label, cf_label):
    return {
        "id": f"cf_{num}",
        "original_id": original_id,
        "original_label": original_label,
        "counterfactual_label": cf_label,
        "counterfactual_text": f"cf_text_{num}",
    }


# ---------------------------------------------------------------------------
# Shared synthetic datasets
# ---------------------------------------------------------------------------

# LD1: 6 repaid (class 0), 2 defaulted (class 1)
LD1_COT = [_cot(i, 0) for i in range(1, 7)] + [_cot(7, 1), _cot(8, 1)]

# Pool B candidates (repaid→defaulted, cf_label=1) sorted by decreasing confidence
# Plus one defaulted→repaid flip (cf_label=0) that should be discarded
LD1_CF_COT = [
    _cf_cot(1, 1, 0, 1, 0.90),
    _cf_cot(2, 2, 0, 1, 0.70),
    _cf_cot(3, 3, 0, 1, 0.60),
    _cf_cot(4, 4, 0, 1, 0.40),
    _cf_cot(5, 5, 0, 1, 0.20),
    _cf_cot(6, 7, 1, 0, 0.80),  # defaulted→repaid — must be discarded for LD1
]

LD1_PLAIN_CF = [_plain_cf(i, i, 0, 1) for i in range(1, 6)] + [_plain_cf(6, 7, 1, 0)]

# ER-REASON: 6 discharge (class 0), 4 admit (class 1)  →  target_ratio = 0.6
ER_COT = [_cot(i, 0) for i in range(1, 7)] + [_cot(i, 1) for i in range(7, 11)]

ER_CF_COT = [
    _cf_cot(10, 7, 1, 0, 0.85),  # admit→discharge  → Pool A
    _cf_cot(11, 8, 1, 0, 0.75),  # admit→discharge  → Pool A
    _cf_cot(12, 1, 0, 1, 0.90),  # discharge→admit  → Pool B
    _cf_cot(13, 2, 0, 1, 0.60),  # discharge→admit  → Pool B
    _cf_cot(14, 3, 0, 1, 0.30),  # discharge→admit  → Pool B
]

# ---------------------------------------------------------------------------
# Helpers used across multiple tests
# ---------------------------------------------------------------------------


def _ld1_pool_b():
    pool_b, _ = get_pool_b("ld1", LD1_CF_COT)
    return pool_b


def _augmented_cot_ld1(n_from_pool_b=4, seed=42):
    pool_b = _ld1_pool_b()
    selected = pool_b[:n_from_pool_b]
    return build_augmented_cot(LD1_COT, selected, seed)


def _augmented_orig_ld1(n_from_pool_b=4, seed=42):
    pool_b = _ld1_pool_b()
    selected = pool_b[:n_from_pool_b]
    selected_ids = [str(p["id"]) for p in selected]
    filtered = filter_plain_cfs(LD1_PLAIN_CF, selected_ids)
    return build_augmented_original(LD1_COT, filtered, selected, seed)


# ===========================================================================
# 1. No ID collision
# ===========================================================================


def test_no_id_collision_cot():
    records = _augmented_cot_ld1()
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate IDs found in augmented CoT output"


def test_no_id_collision_original():
    records = _augmented_orig_ld1()
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate IDs found in augmented original output"


# ===========================================================================
# 2. is_counterfactual flag
# ===========================================================================


def test_is_counterfactual_flag_cot():
    pool_b = _ld1_pool_b()[:4]
    cf_ids = {str(p["id"]) for p in pool_b}
    for r in build_augmented_cot(LD1_COT, pool_b, seed=42):
        if r["id"] in cf_ids:
            assert r["is_counterfactual"] is True
            assert r["original_id"] is not None
        else:
            assert r["is_counterfactual"] is False
            assert r["original_id"] is None
            assert r["target_label_confidence"] is None


def test_is_counterfactual_flag_original():
    pool_b = _ld1_pool_b()[:4]
    cf_ids = {str(p["id"]) for p in pool_b}
    filtered = filter_plain_cfs(LD1_PLAIN_CF, list(cf_ids))
    for r in build_augmented_original(LD1_COT, filtered, pool_b, seed=42):
        if r["id"] in cf_ids:
            assert r["is_counterfactual"] is True
            assert r["original_id"] is not None
        else:
            assert r["is_counterfactual"] is False
            assert r["original_id"] is None
            assert r["target_label_confidence"] is None


# ===========================================================================
# 3. Achieved ratio within soft_threshold
# ===========================================================================


def test_ratio_within_threshold_when_pool_sufficient():
    # LD1: 6 class0, 2 class1, target=0.5  →  n_needed=4, pool has 5
    n_to_take, exhausted, achieved = compute_pool_b_count(
        LD1_COT, [], _ld1_pool_b(), target_ratio=0.5, soft_threshold=0.05
    )
    assert not exhausted
    assert abs(achieved - 0.5) <= 0.05


def test_ratio_exact_when_math_is_clean():
    # Exact 50/50: 6 class0, 2 class1 + 4 class1 = 6/6 → ratio=0.5
    _, _, achieved = compute_pool_b_count(
        LD1_COT, [], _ld1_pool_b(), target_ratio=0.5, soft_threshold=0.05
    )
    assert achieved == pytest.approx(0.5)


# ===========================================================================
# 4. Pool B selected in descending confidence order
# ===========================================================================


def test_pool_b_sorted_descending_confidence_ld1():
    pool_b, _ = get_pool_b("ld1", LD1_CF_COT)
    confidences = [p["target_label_confidence"] for p in pool_b]
    assert confidences == sorted(confidences, reverse=True)


def test_pool_b_sorted_descending_confidence_er_reason():
    pool_b, _ = get_pool_b("er_reason", ER_CF_COT)
    confidences = [p["target_label_confidence"] for p in pool_b]
    assert confidences == sorted(confidences, reverse=True)


def test_pool_b_top_n_have_highest_confidence():
    pool_b, _ = get_pool_b("ld1", LD1_CF_COT)
    # take top 3 — their confidences must beat the 4th entry
    n_to_take, _, _ = compute_pool_b_count(
        LD1_COT, [], pool_b, target_ratio=0.5, soft_threshold=0.05
    )
    # n_to_take=4; verify by slicing manually
    top_n = pool_b[:n_to_take]
    rest = pool_b[n_to_take:]
    if rest:
        assert min(p["target_label_confidence"] for p in top_n) >= max(
            p["target_label_confidence"] for p in rest
        )


# ===========================================================================
# 5. Shuffle determinism
# ===========================================================================


def test_shuffle_same_seed_same_order_cot():
    result_a = _augmented_cot_ld1(seed=42)
    result_b = _augmented_cot_ld1(seed=42)
    assert [r["id"] for r in result_a] == [r["id"] for r in result_b]


# non-zero but astronomically small probability of false failure
def test_shuffle_different_seeds_different_order_cot():
    result_a = _augmented_cot_ld1(seed=0)
    result_b = _augmented_cot_ld1(seed=99)
    # With 12 entries two independent shuffles are astronomically unlikely to match
    assert [r["id"] for r in result_a] != [r["id"] for r in result_b]


def test_shuffle_same_seed_same_order_original():
    result_a = _augmented_orig_ld1(seed=7)
    result_b = _augmented_orig_ld1(seed=7)
    assert [r["id"] for r in result_a] == [r["id"] for r in result_b]


# ===========================================================================
# 6. Pool A constraints
# ===========================================================================


def test_ld1_pool_a_always_empty():
    assert get_pool_a("ld1", LD1_CF_COT) == []


def test_ld1_pool_a_empty_regardless_of_input():
    mixed = LD1_CF_COT + [_cf_cot(99, 10, 1, 0, 0.5)]
    assert get_pool_a("ld1", mixed) == []


def test_er_reason_pool_a_only_counterfactual_label_0():
    pool_a = get_pool_a("er_reason", ER_CF_COT)
    assert all(p["counterfactual_label"] == 0 for p in pool_a)


def test_er_reason_pool_a_correct_count():
    pool_a = get_pool_a("er_reason", ER_CF_COT)
    expected = sum(1 for p in ER_CF_COT if p["counterfactual_label"] == 0)
    assert len(pool_a) == expected


# ===========================================================================
# 7. Additional invariants
# ===========================================================================


def test_pool_b_exhausted_flag():
    # Only 1 pool B entry but need 4 → exhausted
    tiny_pool_b = [_cf_cot(1, 1, 0, 1, 0.9)]
    n_to_take, exhausted, _ = compute_pool_b_count(
        LD1_COT, [], tiny_pool_b, target_ratio=0.5, soft_threshold=0.05
    )
    assert exhausted is True
    assert n_to_take == len(tiny_pool_b)


def test_pool_b_discarded_count_ld1():
    _, n_discarded = get_pool_b("ld1", LD1_CF_COT)
    expected = sum(1 for p in LD1_CF_COT if p["counterfactual_label"] == 0)
    assert n_discarded == expected


def test_pool_b_no_discards_er_reason():
    _, n_discarded = get_pool_b("er_reason", ER_CF_COT)
    assert n_discarded == 0


def test_pool_b_excludes_label_0_entries_ld1():
    pool_b, _ = get_pool_b("ld1", LD1_CF_COT)
    assert all(p["counterfactual_label"] == 1 for p in pool_b)


def test_filter_plain_cfs_subset():
    selected = ["cf_1", "cf_3"]
    filtered = filter_plain_cfs(LD1_PLAIN_CF, selected)
    returned_ids = {r["id"] for r in filtered}
    assert returned_ids == set(selected)


def test_filter_plain_cfs_unknown_id_ignored():
    filtered = filter_plain_cfs(LD1_PLAIN_CF, ["cf_1", "cf_999"])
    assert len(filtered) == 1
    assert filtered[0]["id"] == "cf_1"


def test_collect_selected_ids_pool_a_before_pool_b():
    pool_a = [_cf_cot(10, 7, 1, 0, 0.8), _cf_cot(11, 8, 1, 0, 0.7)]
    pool_b_sel = [_cf_cot(12, 1, 0, 1, 0.9)]
    ids = collect_selected_ids(pool_a, pool_b_sel)
    assert ids == ["cf_10", "cf_11", "cf_12"]


def test_get_target_ratio_ld1_fixed():
    assert get_target_ratio("ld1", LD1_COT) == 0.5
    assert get_target_ratio("ld1", ER_COT) == 0.5  # independent of input


def test_get_target_ratio_er_reason_from_cot():
    # 6 class0, 4 class1 → 0.6
    assert get_target_ratio("er_reason", ER_COT) == pytest.approx(0.6)


def test_get_target_ratio_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_target_ratio("unknown_ds", LD1_COT)


def test_augmented_cot_total_count():
    pool_b = _ld1_pool_b()[:4]
    result = build_augmented_cot(LD1_COT, pool_b, seed=42)
    assert len(result) == len(LD1_COT) + len(pool_b)


def test_augmented_original_total_count():
    pool_b = _ld1_pool_b()[:4]
    selected_ids = [str(p["id"]) for p in pool_b]
    filtered = filter_plain_cfs(LD1_PLAIN_CF, selected_ids)
    result = build_augmented_original(LD1_COT, filtered, pool_b, seed=42)
    assert len(result) == len(LD1_COT) + len(filtered)


def test_augmented_original_confidence_from_cf_cot_map():
    pool_b = _ld1_pool_b()[:1]  # cf_1 with confidence 0.9
    filtered = filter_plain_cfs(LD1_PLAIN_CF, ["cf_1"])
    records = build_augmented_original(LD1_COT, filtered, pool_b, seed=42)
    cf_entry = next(r for r in records if r["id"] == "cf_1")
    assert cf_entry["target_label_confidence"] == pytest.approx(0.9)


def test_augmented_cot_text_field():
    pool_b = _ld1_pool_b()[:1]  # cf_1
    records = build_augmented_cot(LD1_COT, pool_b, seed=42)
    cf_entry = next(r for r in records if r["id"] == "cf_1")
    assert cf_entry["cot_text"] == "cf_cot_1"


def test_augmented_original_text_from_cot_profiles():
    records = build_augmented_original(LD1_COT, [], [], seed=42)
    orig_entry = next(r for r in records if r["id"] == "1")
    assert orig_entry["text"] == "text_1"


def test_build_manifest_cot_split_counts():
    pool_b, n_disc = get_pool_b("ld1", LD1_CF_COT)
    n_take, _, achieved = compute_pool_b_count(
        LD1_COT, [], pool_b, target_ratio=0.5, soft_threshold=0.05
    )
    manifest = build_manifest(
        dataset="ld1",
        split="val",
        model_suff="Qwen3-4b",
        dry_run=False,
        data_paths={},
        cot_profiles=LD1_COT,
        cf_cot_profiles=LD1_CF_COT,
        pool_a=[],
        pool_b=pool_b,
        pool_b_discarded=n_disc,
        n_pool_b_selected=n_take,
        target_ratio=0.5,
        achieved_ratio=achieved,
        soft_threshold=0.05,
    )
    assert manifest["cot_split"]["N_class0"] == 6
    assert manifest["cot_split"]["N_class1"] == 2
    assert manifest["cot_split"]["ratio_class0"] == pytest.approx(0.75)


def test_build_manifest_augmented_split_within_threshold():
    pool_b, n_disc = get_pool_b("ld1", LD1_CF_COT)
    n_take, _, achieved = compute_pool_b_count(
        LD1_COT, [], pool_b, target_ratio=0.5, soft_threshold=0.05
    )
    manifest = build_manifest(
        dataset="ld1",
        split="val",
        model_suff="Qwen3-4b",
        dry_run=False,
        data_paths={},
        cot_profiles=LD1_COT,
        cf_cot_profiles=LD1_CF_COT,
        pool_a=[],
        pool_b=pool_b,
        pool_b_discarded=n_disc,
        n_pool_b_selected=n_take,
        target_ratio=0.5,
        achieved_ratio=achieved,
        soft_threshold=0.05,
    )
    aug = manifest["augmented_split"]
    assert aug["within_threshold"] is True
    assert aug["pool_b_added"] == n_take
    assert aug["ratio_delta"] == pytest.approx(abs(achieved - 0.5))


def test_build_manifest_counterfactual_split_discarded():
    pool_b, n_disc = get_pool_b("ld1", LD1_CF_COT)
    manifest = build_manifest(
        dataset="ld1",
        split="val",
        model_suff="Qwen3-4b",
        dry_run=False,
        data_paths={},
        cot_profiles=LD1_COT,
        cf_cot_profiles=LD1_CF_COT,
        pool_a=[],
        pool_b=pool_b,
        pool_b_discarded=n_disc,
        n_pool_b_selected=4,
        target_ratio=0.5,
        achieved_ratio=0.5,
        soft_threshold=0.05,
    )
    assert manifest["counterfactual_split"]["pool_b_discarded"] == n_disc
