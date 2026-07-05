"""
Unit tests for the retry and checkpoint logic in run_cot_generation
(src/cot/cot_generator.py).

All I/O-heavy dependencies (model, tokenizer, JSONL loading, validator) are
mocked so tests run fast with no GPU or network access required.

Test cases
----------
TestAllPointsPassValidation  – zero failed points, no retry triggered
TestNonePassFirstValidation  – all fail on main pass, all recover on retry round 1
TestFailedLessThanBatchSize  – total failures < batch_size → single retry batch
TestFailedMoreThanBatchSize  – total failures > batch_size → multiple retry batches
TestMaxRetriesRespected      – retry loop stops after max_retries even with remaining failures
TestNumBatchesLimit          – num_batches config caps how many main-pass batches run
TestCheckpointResumption     – already-processed IDs are excluded; fully-done datasets skipped
TestPartialRetrySuccess      – subset passes on each retry round; remainder carries forward

Run with:
    pytest tests/test_cot_generator.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cot.cot_generator import run_cot_generation

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(
    output_path: Path, batch_size: int = 2, num_batches=None, max_retries: int = 3
) -> dict:
    """Build a minimal config dict pointing the output at a tmp_path file."""
    return {
        "teacher_model": {
            "name": "mock-model",
            "quantization": None,
            "batch_size": batch_size,
            "max_new_tokens": 64,
            "num_batches": num_batches,
            "max_retries": max_retries,
        },
        "dataset": {
            "active_sub_dataset": "ld1",
            "sub_datasets": {
                "ld1": {
                    # Absolute path: resolve_path will keep it as-is (pathlib override)
                    "processed_profiles_path": "dummy_input.jsonl",
                    "prompt_path": "dummy_prompt.txt",
                    "cot_output_path": str(output_path),
                }
            },
        },
    }


def _dp(id_: str, label: int = 0) -> dict:
    """Minimal data point dict."""
    return {"id": id_, "text": f"profile text for {id_}", "label": label}


def _approved(n: int = 1) -> list[dict]:
    """Return n approved validation result dicts."""
    return [
        {
            "approved": True,
            "cleaned_cot": f"valid cot {i}",
            "gates": {
                k: True
                for k in (
                    "eors_tokens",
                    "step_tokens",
                    "token_count",
                    "forbidden_phrases",
                )
            },
            "faithfulness_score": 0.90,
            "finbert_confidence": 0.80,
            "quality_score": 0.86,
        }
        for i in range(n)
    ]


def _failed(n: int = 1) -> list[dict]:
    """Return n failed validation result dicts."""
    return [
        {
            "approved": False,
            "cleaned_cot": f"bad cot {i}",
            "gates": {
                "eors_tokens": False,
                "step_tokens": True,
                "token_count": True,
                "forbidden_phrases": True,
            },
            "faithfulness_score": 0.0,
            "finbert_confidence": 0.0,
            "quality_score": 0.0,
        }
        for i in range(n)
    ]


def _mixed(n_approved: int, n_failed: int) -> list[dict]:
    """Return approved results followed by failed results (order matches batch order)."""
    return _approved(n_approved) + _failed(n_failed)


def _read_output(output_path: Path) -> list[dict]:
    """Parse the output JSONL file and return all records."""
    if not output_path.exists():
        return []
    return [
        json.loads(line)
        for line in output_path.read_text().splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Shared autouse fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    """
    Set up mocks for all heavy dependencies and expose them on a
    simple namespace so individual tests can configure side_effects cleanly.

    Attributes
    ----------
    output_path : Path     – writable tmp file used as cot_output_path
    config      : dict     – mutable config dict (tests may override keys)
    mock_batches            – controls what load_jsonl_batches yields
    mock_gen                – controls what generate_cot_batch returns
    mock_validate           – controls what validate_cot_batch returns
    """

    class Ctx:
        pass

    c = Ctx()
    c.output_path = tmp_path / "cot_output.jsonl"
    c.config = _make_config(c.output_path)

    with (
        patch("src.cot.cot_generator.load_config", return_value=c.config),
        patch(
            "src.cot.cot_generator.load_model", return_value=(MagicMock(), MagicMock())
        ),
        patch("src.cot.cot_generator.load_prompt", return_value="mock prompt"),
        patch("src.cot.cot_generator.load_jsonl_batches") as mock_batches,
        patch(
            "src.cot.cot_generator.generate_cot_batch",
            side_effect=lambda texts, *a, **kw: ["mock cot"] * len(texts),
        ) as mock_gen,
        patch("src.cot.cot_generator.validate_cot_batch") as mock_validate,
        patch("src.cot.cot_generator.print_memory_usage"),
    ):
        c.mock_batches = mock_batches
        c.mock_gen = mock_gen
        c.mock_validate = mock_validate
        yield c


def _run():
    """Invoke run_cot_generation with a dummy config path and no path suffix."""
    run_cot_generation("dummy_config.yaml", path_suff=None)


# ---------------------------------------------------------------------------
# TestAllPointsPassValidation
# ---------------------------------------------------------------------------


class TestAllPointsPassValidation:
    """All data points are approved on the first validation pass — no retry triggered."""

    def test_no_retry_rounds_executed(self, ctx):
        """generate_cot_batch is called exactly once per main batch and never for retries."""
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
                [_dp("id_2"), _dp("id_3")],
            ]
        )
        ctx.mock_validate.side_effect = [_approved(2), _approved(2)]

        _run()

        assert ctx.mock_gen.call_count == 2
        assert ctx.mock_validate.call_count == 2

    def test_all_points_written_to_output(self, ctx):
        """Every approved data point is persisted to the output JSONL file."""
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
                [_dp("id_2"), _dp("id_3")],
            ]
        )
        ctx.mock_validate.side_effect = [_approved(2), _approved(2)]

        _run()

        written = _read_output(ctx.output_path)
        assert len(written) == 4
        assert {dp["id"] for dp in written} == {"id_0", "id_1", "id_2", "id_3"}
        assert all("cot_text" in dp for dp in written)

    def test_failed_points_list_is_empty(self, ctx):
        """With zero failures there should be no warning about unprocessed points."""
        ctx.mock_batches.return_value = iter([[_dp("id_0"), _dp("id_1")]])
        ctx.mock_validate.side_effect = [_approved(2)]

        # Simply verify it completes without error and produces output
        _run()
        assert len(_read_output(ctx.output_path)) == 2


# ---------------------------------------------------------------------------
# TestNonePassFirstValidation
# ---------------------------------------------------------------------------


class TestNonePassFirstValidation:
    """No data points pass the first validation pass; all recover on retry round 1."""

    def test_retry_round_triggered(self, ctx):
        """A second generation+validation pass is made for all failed points."""
        ctx.mock_batches.return_value = iter([[_dp("id_0"), _dp("id_1")]])
        ctx.mock_validate.side_effect = [
            _failed(2),  # main pass: both fail
            _approved(2),  # retry round 1: both pass
        ]

        _run()

        assert ctx.mock_gen.call_count == 2
        assert ctx.mock_validate.call_count == 2

    def test_all_points_written_after_retry(self, ctx):
        """Points approved on retry are still persisted to the output file."""
        ctx.mock_batches.return_value = iter([[_dp("id_0"), _dp("id_1")]])
        ctx.mock_validate.side_effect = [_failed(2), _approved(2)]

        _run()

        written = _read_output(ctx.output_path)
        assert len(written) == 2
        assert {dp["id"] for dp in written} == {"id_0", "id_1"}

    def test_failed_points_not_written_before_retry(self, ctx):
        """Failed points must NOT appear in the output file before retrying."""
        ctx.mock_batches.return_value = iter([[_dp("id_0"), _dp("id_1")]])
        # All fail every round so max_retries is exhausted
        ctx.config["teacher_model"]["max_retries"] = 1
        ctx.mock_validate.side_effect = [_failed(2), _failed(2)]

        _run()

        written = _read_output(ctx.output_path)
        assert len(written) == 0


# ---------------------------------------------------------------------------
# TestFailedLessThanBatchSize
# ---------------------------------------------------------------------------


class TestFailedLessThanBatchSize:
    """Total number of failed points is less than batch_size → single retry batch."""

    def test_exactly_one_retry_batch_generated(self, ctx):
        """Only one retry batch is dispatched when failures < batch_size."""
        # batch_size=3, 2 main batches, 1 failure total (< 3)
        ctx.config["teacher_model"]["batch_size"] = 3
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2")],
                [_dp("id_3"), _dp("id_4"), _dp("id_5")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(2, 1),  # batch 0: id_0 & id_1 pass, id_2 fails
            _approved(3),  # batch 1: all pass
            _approved(1),  # retry round 1: the 1 failed point passes
        ]

        _run()

        # 2 main batches + 1 retry batch = 3 total generate calls
        assert ctx.mock_gen.call_count == 3
        assert ctx.mock_validate.call_count == 3

    def test_all_points_written(self, ctx):
        """All points (including the single retried one) appear in the output."""
        ctx.config["teacher_model"]["batch_size"] = 3
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2")],
                [_dp("id_3"), _dp("id_4"), _dp("id_5")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(2, 1),
            _approved(3),
            _approved(1),
        ]

        _run()

        assert len(_read_output(ctx.output_path)) == 6

    def test_retry_batch_size_matches_failure_count(self, ctx):
        """The retry batch passed to generate_cot_batch contains exactly the failed texts."""
        ctx.config["teacher_model"]["batch_size"] = 4
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2"), _dp("id_3")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(3, 1),  # 1 fails
            _approved(1),  # retry
        ]

        _run()

        retry_call_texts = ctx.mock_gen.call_args_list[1][0][0]
        assert len(retry_call_texts) == 1


# ---------------------------------------------------------------------------
# TestFailedMoreThanBatchSize
# ---------------------------------------------------------------------------


class TestFailedMoreThanBatchSize:
    """Total failed points exceed batch_size → retry spans multiple batches."""

    def test_multiple_retry_batches_generated(self, ctx):
        """generate_cot_batch is called more than once during the retry round."""
        # batch_size=2, 3 main batches × 1 failure each = 3 failed > 2
        ctx.config["teacher_model"]["batch_size"] = 2
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
                [_dp("id_2"), _dp("id_3")],
                [_dp("id_4"), _dp("id_5")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(1, 1),  # main batch 0: id_1 fails
            _mixed(1, 1),  # main batch 1: id_3 fails
            _mixed(1, 1),  # main batch 2: id_5 fails
            _approved(2),  # retry batch 1: id_1, id_3 pass
            _approved(1),  # retry batch 2: id_5 passes
        ]

        _run()

        # 3 main + 2 retry = 5 total generate calls
        assert ctx.mock_gen.call_count == 5
        assert ctx.mock_validate.call_count == 5

    def test_all_points_eventually_written(self, ctx):
        """Every originally failed point is written once it passes a retry."""
        ctx.config["teacher_model"]["batch_size"] = 2
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
                [_dp("id_2"), _dp("id_3")],
                [_dp("id_4"), _dp("id_5")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(1, 1),
            _mixed(1, 1),
            _mixed(1, 1),
            _approved(2),
            _approved(1),
        ]

        _run()

        written = _read_output(ctx.output_path)
        assert len(written) == 6

    def test_retry_batches_respect_batch_size(self, ctx):
        """Each retry batch sent to generate_cot_batch is at most batch_size."""
        ctx.config["teacher_model"]["batch_size"] = 2
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2"), _dp("id_3")],
            ]
        )
        # All 4 fail in main pass; batch_size=2 → 2 retry batches of ≤2 each
        ctx.mock_validate.side_effect = [
            _failed(4),
            _approved(2),  # retry batch 1
            _approved(2),  # retry batch 2
        ]

        _run()

        retry_calls = ctx.mock_gen.call_args_list[1:]  # skip main pass call
        assert len(retry_calls) == 2
        for call_ in retry_calls:
            assert len(call_[0][0]) <= 2


# ---------------------------------------------------------------------------
# TestMaxRetriesRespected
# ---------------------------------------------------------------------------


class TestMaxRetriesRespected:
    """Retry loop stops after max_retries rounds even if points are still failing."""

    def test_generation_stops_at_max_retries(self, ctx):
        """generate_cot_batch is called exactly 1 + max_retries times when all always fail."""
        ctx.config["teacher_model"]["max_retries"] = 2
        ctx.mock_batches.return_value = iter([[_dp("id_0"), _dp("id_1")]])
        ctx.mock_validate.side_effect = [
            _failed(2),  # main pass
            _failed(2),  # retry round 1
            _failed(2),  # retry round 2
        ]

        _run()

        assert ctx.mock_gen.call_count == 3  # 1 main + 2 retries

    def test_nothing_written_when_always_failing(self, ctx):
        """Output file stays empty when no point ever passes validation."""
        ctx.config["teacher_model"]["max_retries"] = 2
        ctx.mock_batches.return_value = iter([[_dp("id_0"), _dp("id_1")]])
        ctx.mock_validate.side_effect = [_failed(2), _failed(2), _failed(2)]

        _run()

        assert len(_read_output(ctx.output_path)) == 0

    def test_zero_retries_skips_retry_loop(self, ctx):
        """With max_retries=0 the retry loop is never entered."""
        ctx.config["teacher_model"]["max_retries"] = 0
        ctx.mock_batches.return_value = iter([[_dp("id_0"), _dp("id_1")]])
        ctx.mock_validate.side_effect = [_failed(2)]

        _run()

        assert ctx.mock_gen.call_count == 1
        assert len(_read_output(ctx.output_path)) == 0


# ---------------------------------------------------------------------------
# TestNumBatchesLimit
# ---------------------------------------------------------------------------


class TestNumBatchesLimit:
    """num_batches config key caps how many batches the main loop processes."""

    def test_only_num_batches_processed(self, ctx):
        """Batches beyond num_batches are never read or generated."""
        ctx.config["teacher_model"]["num_batches"] = 2
        ctx.config["teacher_model"]["max_retries"] = 0
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],  # batch 0 — processed
                [_dp("id_2"), _dp("id_3")],  # batch 1 — processed
                [_dp("id_4"), _dp("id_5")],  # batch 2 — must NOT be processed
            ]
        )
        ctx.mock_validate.side_effect = [_approved(2), _approved(2)]

        _run()

        assert ctx.mock_gen.call_count == 2

    def test_points_beyond_limit_not_in_output(self, ctx):
        """Data points from batches past num_batches do not appear in the output."""
        ctx.config["teacher_model"]["num_batches"] = 2
        ctx.config["teacher_model"]["max_retries"] = 0
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
                [_dp("id_2"), _dp("id_3")],
                [_dp("id_4"), _dp("id_5")],
            ]
        )
        ctx.mock_validate.side_effect = [_approved(2), _approved(2)]

        _run()

        written_ids = {dp["id"] for dp in _read_output(ctx.output_path)}
        assert written_ids == {"id_0", "id_1", "id_2", "id_3"}
        assert "id_4" not in written_ids
        assert "id_5" not in written_ids

    def test_none_num_batches_processes_full_dataset(self, ctx):
        """num_batches=None (default) processes all available batches."""
        ctx.config["teacher_model"]["num_batches"] = None
        ctx.config["teacher_model"]["max_retries"] = 0
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
                [_dp("id_2"), _dp("id_3")],
                [_dp("id_4"), _dp("id_5")],
            ]
        )
        ctx.mock_validate.side_effect = [_approved(2), _approved(2), _approved(2)]

        _run()

        assert ctx.mock_gen.call_count == 3
        assert len(_read_output(ctx.output_path)) == 6


# ---------------------------------------------------------------------------
# TestCheckpointResumption
# ---------------------------------------------------------------------------


class TestCheckpointResumption:
    """IDs already present in the output file are excluded from generation."""

    def test_processed_ids_excluded_from_generation(self, ctx):
        """Batch contains 4 IDs but 2 are already in the output — only 2 are generated."""
        existing = [
            {"id": "id_0", "text": "t", "label": 0, "cot_text": "old cot"},
            {"id": "id_1", "text": "t", "label": 0, "cot_text": "old cot"},
        ]
        ctx.output_path.write_text("\n".join(json.dumps(dp) for dp in existing) + "\n")

        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2"), _dp("id_3")],
            ]
        )
        ctx.mock_validate.side_effect = [_approved(2)]

        _run()

        assert ctx.mock_gen.call_count == 1
        generated_texts = ctx.mock_gen.call_args[0][0]
        assert len(generated_texts) == 2

    def test_pre_existing_records_not_duplicated(self, ctx):
        """Pre-existing records are not re-appended; only new approvals are added."""
        existing = [
            {"id": "id_0", "text": "t", "label": 0, "cot_text": "old cot"},
        ]
        ctx.output_path.write_text(json.dumps(existing[0]) + "\n")

        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
            ]
        )
        ctx.mock_validate.side_effect = [_approved(1)]

        _run()

        written = _read_output(ctx.output_path)
        # id_0 was pre-existing (not re-written), id_1 is newly added
        assert len(written) == 2
        ids = [dp["id"] for dp in written]
        assert ids.count("id_0") == 1
        assert ids.count("id_1") == 1

    def test_fully_processed_dataset_skips_generation(self, ctx):
        """If all IDs in every batch are already processed, generate is never called."""
        existing = [
            {"id": "id_0", "text": "t", "label": 0, "cot_text": "old cot"},
            {"id": "id_1", "text": "t", "label": 0, "cot_text": "old cot"},
        ]
        ctx.output_path.write_text("\n".join(json.dumps(dp) for dp in existing) + "\n")

        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1")],
            ]
        )

        _run()

        ctx.mock_gen.assert_not_called()
        ctx.mock_validate.assert_not_called()


# ---------------------------------------------------------------------------
# TestPartialRetrySuccess
# ---------------------------------------------------------------------------


class TestPartialRetrySuccess:
    """Some points recover on each retry round; the rest carry forward."""

    def test_failing_points_carry_forward_correctly(self, ctx):
        """Points that still fail after retry round 1 are retried again in round 2."""
        ctx.config["teacher_model"]["max_retries"] = 3
        ctx.config["teacher_model"]["batch_size"] = 4
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2"), _dp("id_3")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(2, 2),  # main pass: id_0, id_1 pass; id_2, id_3 fail
            _mixed(1, 1),  # retry round 1: id_2 passes; id_3 still fails
            _approved(1),  # retry round 2: id_3 finally passes
        ]

        _run()

        assert ctx.mock_gen.call_count == 3
        assert ctx.mock_validate.call_count == 3
        assert len(_read_output(ctx.output_path)) == 4

    def test_final_retry_batch_size_shrinks_with_recoveries(self, ctx):
        """Each retry round only dispatches exactly the still-failing points."""
        ctx.config["teacher_model"]["max_retries"] = 3
        ctx.config["teacher_model"]["batch_size"] = 4
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2"), _dp("id_3")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(2, 2),  # 2 fail → retry round 1 gets 2 points
            _mixed(1, 1),  # 1 fail → retry round 2 gets 1 point
            _approved(1),
        ]

        _run()

        retry1_texts = ctx.mock_gen.call_args_list[1][0][0]
        retry2_texts = ctx.mock_gen.call_args_list[2][0][0]
        assert len(retry1_texts) == 2
        assert len(retry2_texts) == 1

    def test_partial_failure_does_not_exceed_max_retries(self, ctx):
        """Even with partial recoveries per round, max_retries is still the upper bound."""
        ctx.config["teacher_model"]["max_retries"] = 2
        ctx.config["teacher_model"]["batch_size"] = 4
        ctx.mock_batches.return_value = iter(
            [
                [_dp("id_0"), _dp("id_1"), _dp("id_2"), _dp("id_3")],
            ]
        )
        ctx.mock_validate.side_effect = [
            _mixed(2, 2),  # main: 2 fail
            _mixed(1, 1),  # retry 1: 1 recovers, 1 still failing
            _failed(1),  # retry 2: still failing — max_retries hit here
        ]

        _run()

        # 1 main + 2 retry rounds = 3 generate calls
        assert ctx.mock_gen.call_count == 3
        # 3 points were approved across the rounds
        assert len(_read_output(ctx.output_path)) == 3
