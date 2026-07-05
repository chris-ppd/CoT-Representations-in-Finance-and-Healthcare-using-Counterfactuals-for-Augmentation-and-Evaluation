"""
Tests for src/data/injection_data_utils.py using fully synthetic in-memory data.

No dependency on real cluster-generated files — all tensors and jsonl files are
created in pytest's tmp_path fixture.

Invariants tested
-----------------
1. load_hidden_states: excludes extraction_ok=False entries; tensor shape preserved.
2. get_step_states (real profile): dispatches via cot_index.
3. get_step_states (CF with is_counterfactual=True): dispatches via cf_index.
4. get_step_states (CF WITHOUT is_counterfactual field, cf_-prefix only):
   CRITICAL — if only the explicit field were checked, this would fall through
   to the real-profile branch and raise a misleading KeyError on "cf_1".
5. get_step_states: missing id raises KeyError with a descriptive message.
6. get_step_states: CF entry with cf_index=None raises ValueError.
7. validate_dataset: mixed file — correct n_resolved / n_unresolved counts;
   CF entries without is_counterfactual field are resolved correctly.
8. validate_dataset: attrition reporting when original_jsonl_path is provided.
9. validate_dataset: CF entries fail cleanly (ValueError) when cf_index=None.
10. check_gate_init: all-pass and one-fail cases.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.injection_data_utils import (
    check_gate_init,
    get_step_states,
    load_hidden_states,
    validate_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data(tmp_path):
    """
    Write synthetic .pt tensors and .jsonl indices to tmp_path.

    COT (real profiles):
      tensor shape (5, 4, 2560); entries "0".."4"; "4" has extraction_ok=False.

    CF profiles:
      tensor shape (3, 4, 2560); profile_ids "0", "1", "2" (matching originals),
      all extraction_ok=True.

    Returns (tmp_path, cot_tensor, cf_tensor) so tests can do exact-value assertions.
    """
    # --- COT ---
    torch.manual_seed(42)
    cot_tensor = torch.randn(5, 4, 2560)
    torch.save(cot_tensor, tmp_path / "cot_states.pt")
    _write_jsonl(
        tmp_path / "cot_index.jsonl",
        [
            {"profile_id": "0", "profile_row_idx": 0, "extraction_ok": True},
            {"profile_id": "1", "profile_row_idx": 1, "extraction_ok": True},
            {"profile_id": "2", "profile_row_idx": 2, "extraction_ok": True},
            {"profile_id": "3", "profile_row_idx": 3, "extraction_ok": True},
            {"profile_id": "4", "profile_row_idx": 4, "extraction_ok": False},
        ],
    )

    # --- CF ---
    torch.manual_seed(99)
    cf_tensor = torch.randn(3, 4, 2560)
    torch.save(cf_tensor, tmp_path / "cf_states.pt")
    _write_jsonl(
        tmp_path / "cf_index.jsonl",
        [
            {"profile_id": "0", "profile_row_idx": 0, "extraction_ok": True},
            {"profile_id": "1", "profile_row_idx": 1, "extraction_ok": True},
            {"profile_id": "2", "profile_row_idx": 2, "extraction_ok": True},
        ],
    )

    return tmp_path, cot_tensor, cf_tensor


# ---------------------------------------------------------------------------
# Convenience loader (avoids repeating the same two calls in every test)
# ---------------------------------------------------------------------------


def _load_all(tmp_path):
    cot_index, cot_states = load_hidden_states(
        str(tmp_path / "cot_index.jsonl"), str(tmp_path / "cot_states.pt")
    )
    cf_index, cf_states = load_hidden_states(
        str(tmp_path / "cf_index.jsonl"), str(tmp_path / "cf_states.pt")
    )
    return cot_index, cot_states, cf_index, cf_states


# ===========================================================================
# TestLoadHiddenStates
# ===========================================================================


class TestLoadHiddenStates:
    def test_excluded_extraction_not_ok(self, synthetic_data):
        tmp_path, _, _ = synthetic_data
        cot_index, _ = load_hidden_states(
            str(tmp_path / "cot_index.jsonl"), str(tmp_path / "cot_states.pt")
        )
        assert len(cot_index) == 4, "extraction_ok=False entry must be excluded"
        assert "4" not in cot_index

    def test_all_ok_entries_present(self, synthetic_data):
        tmp_path, _, _ = synthetic_data
        cot_index, _ = load_hidden_states(
            str(tmp_path / "cot_index.jsonl"), str(tmp_path / "cot_states.pt")
        )
        for pid in ("0", "1", "2", "3"):
            assert pid in cot_index

    def test_tensor_shape(self, synthetic_data):
        tmp_path, cot_tensor, _ = synthetic_data
        _, tensor = load_hidden_states(
            str(tmp_path / "cot_index.jsonl"), str(tmp_path / "cot_states.pt")
        )
        assert tensor.shape == (5, 4, 2560)

    def test_tensor_values_match(self, synthetic_data):
        tmp_path, cot_tensor, _ = synthetic_data
        _, tensor = load_hidden_states(
            str(tmp_path / "cot_index.jsonl"), str(tmp_path / "cot_states.pt")
        )
        assert torch.equal(tensor, cot_tensor)


# ===========================================================================
# TestGetStepStates
# ===========================================================================


class TestGetStepStates:
    def test_real_profile_shape_and_values(self, synthetic_data):
        """cot_*.jsonl shape: {"id": "0", "label": 0}"""
        tmp_path, _, _ = synthetic_data
        cot_index, cot_states, cf_index, cf_states = _load_all(tmp_path)

        entry = {"id": "0", "is_counterfactual": False, "label": 0}
        result = get_step_states(entry, cot_index, cf_index, cot_states, cf_states)

        assert result.shape == (3, 2560)
        expected = cot_states[cot_index["0"], :3, :]
        assert torch.equal(result, expected)

    def test_cf_with_explicit_flag(self, synthetic_data):
        """augmented_*.jsonl shape: explicit is_counterfactual=True + original_id."""
        tmp_path, _, _ = synthetic_data
        cot_index, cot_states, cf_index, cf_states = _load_all(tmp_path)

        entry = {"id": "cf_0", "original_id": 0, "is_counterfactual": True, "label": 1}
        result = get_step_states(entry, cot_index, cf_index, cot_states, cf_states)

        assert result.shape == (3, 2560)
        expected = cf_states[cf_index["0"], :3, :]
        assert torch.equal(result, expected)

    def test_cf_without_is_counterfactual_field(self, synthetic_data):
        """
        CRITICAL: counterfactual_*.jsonl shape — no is_counterfactual field at all.
        Dispatch must rely on the cf_-prefix of the id.  If only the explicit flag
        were checked, this entry would fall through to the real-profile branch and
        raise KeyError on cot_index["cf_1"].
        """
        tmp_path, _, _ = synthetic_data
        cot_index, cot_states, cf_index, cf_states = _load_all(tmp_path)

        entry = {
            "id": "cf_1",
            "original_id": 1,
            "counterfactual_text": "some text",
            "original_label": 0,
            "counterfactual_label": 1,
            # No "is_counterfactual" key
        }
        result = get_step_states(entry, cot_index, cf_index, cot_states, cf_states)

        assert result.shape == (3, 2560)
        expected = cf_states[cf_index["1"], :3, :]
        assert torch.equal(result, expected)

    def test_missing_original_id_raises_key_error(self, synthetic_data):
        """original_id=99 not in cf_index → KeyError with descriptive message."""
        tmp_path, _, _ = synthetic_data
        cot_index, cot_states, cf_index, cf_states = _load_all(tmp_path)

        entry = {
            "id": "cf_99",
            "original_id": 99,
            "counterfactual_text": "x",
            "original_label": 0,
            "counterfactual_label": 1,
        }
        with pytest.raises(KeyError, match="99"):
            get_step_states(entry, cot_index, cf_index, cot_states, cf_states)

    def test_cf_with_explicit_flag_and_no_cf_index_raises_value_error(self, synthetic_data):
        """CF entry (is_counterfactual=True) + cf_index=None → ValueError, not AttributeError."""
        tmp_path, _, _ = synthetic_data
        cot_index, cot_states, _, _ = _load_all(tmp_path)

        entry = {"id": "cf_0", "original_id": 0, "is_counterfactual": True}
        with pytest.raises(ValueError, match="counterfactual"):
            get_step_states(entry, cot_index, None, cot_states, None)

    def test_cf_prefix_and_no_cf_index_raises_value_error(self, synthetic_data):
        """CF via prefix only (no is_counterfactual) + cf_index=None → ValueError."""
        tmp_path, _, _ = synthetic_data
        cot_index, cot_states, _, _ = _load_all(tmp_path)

        entry = {
            "id": "cf_1",
            "original_id": 1,
            "counterfactual_text": "x",
            "original_label": 0,
            "counterfactual_label": 1,
        }
        with pytest.raises(ValueError, match="counterfactual"):
            get_step_states(entry, cot_index, None, cot_states, None)

    def test_returns_only_first_3_steps(self, synthetic_data):
        """Step 4 (index 3) must be excluded — slice must be [:3, :]."""
        tmp_path, cot_tensor, _ = synthetic_data
        cot_index, cot_states, cf_index, cf_states = _load_all(tmp_path)

        entry = {"id": "0"}
        result = get_step_states(entry, cot_index, cf_index, cot_states, cf_states)

        assert result.shape[0] == 3, "Must return exactly 3 step states, not 4"
        row = cot_index["0"]
        assert not torch.equal(result, cot_states[row, :, :]), (
            "Result must not equal all 4 steps — step 4 should be excluded"
        )


# ===========================================================================
# TestValidateDataset
# ===========================================================================


class TestValidateDataset:
    def test_mixed_file_resolved_and_unresolved_counts(self, synthetic_data, tmp_path):
        """
        File contains:
          2 real profiles   → resolved via cot_index
          2 CF entries (counterfactual_*.jsonl shape — no is_counterfactual field)
                           → resolved via cf_prefix dispatch
          1 CF entry with original_id=99 not in cf_index → unresolved

        This test fails if only is_counterfactual were checked for CF dispatch,
        because the CF entries here have no such field.
        """
        cot_index, cot_states, cf_index, cf_states = _load_all(tmp_path)

        entries = [
            {"id": "0", "label": 0},
            {"id": "1", "label": 1},
            # counterfactual_*.jsonl shaped — no is_counterfactual
            {
                "id": "cf_0",
                "original_id": 0,
                "counterfactual_text": "x",
                "original_label": 0,
                "counterfactual_label": 1,
            },
            {
                "id": "cf_1",
                "original_id": 1,
                "counterfactual_text": "y",
                "original_label": 0,
                "counterfactual_label": 1,
            },
            # Deliberately broken
            {
                "id": "cf_99",
                "original_id": 99,
                "counterfactual_text": "z",
                "original_label": 0,
                "counterfactual_label": 1,
            },
        ]
        jsonl_path = tmp_path / "validate_mixed.jsonl"
        _write_jsonl(jsonl_path, entries)

        summary = validate_dataset(
            str(jsonl_path), cot_index, cf_index, cot_states, cf_states
        )

        assert summary["n_entries"] == 5
        assert summary["n_resolved"] == 4
        assert summary["n_unresolved"] == 1
        assert summary["shape_check_passed"] is True
        assert len(summary["unresolved"]) == 1
        assert summary["unresolved"][0]["id"] == "cf_99"
        assert summary["attrition"] is None

    def test_attrition_reporting(self, synthetic_data, tmp_path):
        cot_index, cot_states, cf_index, cf_states = _load_all(tmp_path)

        original_path = tmp_path / "original.jsonl"
        this_path = tmp_path / "this.jsonl"
        _write_jsonl(original_path, [
            {"id": "0", "label": 0},
            {"id": "1", "label": 1},
            {"id": "2", "label": 0},
        ])
        _write_jsonl(this_path, [
            {"id": "0", "label": 0},
            {"id": "1", "label": 1},
        ])

        summary = validate_dataset(
            str(this_path),
            cot_index, cf_index, cot_states, cf_states,
            original_jsonl_path=str(original_path),
        )

        assert summary["attrition"] is not None
        assert summary["attrition"]["original_count"] == 3
        assert summary["attrition"]["this_count"] == 2

    def test_cf_entries_fail_cleanly_without_cf_index(self, synthetic_data, tmp_path):
        """CF entries → ValueError caught per entry; does not crash the whole run."""
        cot_index, cot_states, _, _ = _load_all(tmp_path)

        entries = [
            {"id": "0", "label": 0},   # real → resolves
            {
                "id": "cf_0",
                "original_id": 0,
                "counterfactual_text": "x",
                "original_label": 0,
                "counterfactual_label": 1,
            },  # CF → ValueError, caught
        ]
        jsonl_path = tmp_path / "cf_no_index.jsonl"
        _write_jsonl(jsonl_path, entries)

        summary = validate_dataset(
            str(jsonl_path), cot_index, None, cot_states, None
        )

        assert summary["n_entries"] == 2
        assert summary["n_resolved"] == 1
        assert summary["n_unresolved"] == 1


# ===========================================================================
# TestCheckGateInit
# ===========================================================================


class TestCheckGateInit:
    class _MockModel:
        def __init__(self, values: dict[str, float]):
            self._values = values

        def get_gate_values(self) -> dict[str, float]:
            return dict(self._values)

    def test_all_gates_at_init(self):
        model = self._MockModel({"gate_1": 0.1, "gate_2": 0.1, "gate_3": 0.1})
        results = check_gate_init(model, expected_init=0.1)
        assert results == {"gate_1": True, "gate_2": True, "gate_3": True}

    def test_one_gate_deviates(self):
        model = self._MockModel({"gate_1": 0.1, "gate_2": 0.5, "gate_3": 0.1})
        results = check_gate_init(model, expected_init=0.1)
        assert results["gate_1"] is True
        assert results["gate_2"] is False
        assert results["gate_3"] is True

    def test_custom_tolerance(self):
        model = self._MockModel({"gate_1": 0.10005, "gate_2": 0.1, "gate_3": 0.1})
        assert check_gate_init(model, expected_init=0.1, tol=1e-6)["gate_1"] is False
        assert check_gate_init(model, expected_init=0.1, tol=1e-3)["gate_1"] is True
