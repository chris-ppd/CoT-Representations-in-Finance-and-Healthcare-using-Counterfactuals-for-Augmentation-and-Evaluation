"""
Pytest unit tests for the CoT validation pipeline defined in
src/validation/cot_validator.py.

Each test corresponds to exactly one layer of the pipeline and is
self-contained: it constructs a minimal CoT fixture, calls the relevant
validator function, and asserts True or False.

Test structure
--------------
- TestCleanCotText          – preprocessing / cleaning
- TestCheckEorsTokens       – structural gate: [EORS] count
- TestCheckStepTokens       – structural gate: [STEP] count
- TestCheckTokenCount       – BERT token length gate
- TestCheckForbiddenPhrases – label-leakage regex gate
- TestCheckFaithfulness     – DeBERTa NLI quality score (uses mocks)
- TestCheckFinbertLabel     – FinBERT classification check (uses mocks)
- TestValidateCot           – end-to-end master function

Run with:
    pytest tests/test_cot.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to Python path so src package is importable (maybe we can use a toml file later)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cot.cot_validator import (
    EORS_TOKEN,
    STEP_TOKEN,
    check_eors_tokens,
    check_faithfulness,
    check_finbert_label,
    check_forbidden_phrases,
    check_step_tokens,
    check_token_count,
    clean_cot_text,
    validate_cot,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Minimal well-formed CoT used as a baseline across multiple tests.
# Each step is intentionally short to stay well under 512 tokens.
VALID_COT = (
    "[STEP] The applicant requested a loan of $12,000 for home improvement. "
    "They have been employed as a technician for 6 years. [EORS] "
    "[STEP] The property is valued at $95,000 with a mortgage due of $40,000, "
    "giving an equity position of $55,000. [EORS] "
    "[STEP] Credit history shows 0 derogatory marks, 0 delinquencies, and "
    "a credit line age of 180 months with 2 recent inquiries. [EORS] "
    "[STEP] The debt-to-income ratio is 28%, which is moderate. "
    "Employment stability and equity position are reasonable. [EORS]"
)

ORIGINAL_PROFILE = (
    "Loan amount: 12000. Purpose: home improvement. "
    "Occupation: technician. Years employed: 6. "
    "Property value: 95000. Mortgage due: 40000. "
    "Derogatory marks: 0. Delinquencies: 0. "
    "Credit line age: 180 months. Inquiries: 2. "
    "Debt-to-income ratio: 28%."
)

GROUND_TRUTH_LABEL = 0  # repaid


# ---------------------------------------------------------------------------
# TestCleanCotText
# ---------------------------------------------------------------------------


class TestCleanCotText:
    """Tests for the clean_cot_text preprocessing function."""

    def test_removes_preamble(self):
        """Text before the first [STEP] token should be stripped."""
        raw = "Sure! Here is the reasoning:\n" + VALID_COT
        cleaned = clean_cot_text(raw)
        assert cleaned.startswith(STEP_TOKEN)

    def test_removes_trailing_text(self):
        """Text after the final [EORS] token should be stripped."""
        raw = VALID_COT + "\n\nHope this helps! Let me know if you need more."
        cleaned = clean_cot_text(raw)
        assert cleaned.endswith(EORS_TOKEN)

    def test_removes_both_ends(self):
        """Both preamble and trailing text should be stripped simultaneously."""
        raw = "Preamble text. " + VALID_COT + " Trailing text."
        cleaned = clean_cot_text(raw)
        assert cleaned.startswith(STEP_TOKEN)
        assert cleaned.endswith(EORS_TOKEN)

    def test_returns_original_if_no_tokens(self):
        """If no [STEP] or [EORS] tokens exist, return the raw text unchanged."""
        raw = "This has no special tokens at all."
        cleaned = clean_cot_text(raw)
        assert cleaned == raw

    def test_already_clean_unchanged(self):
        """A CoT that starts with [STEP] and ends with [EORS] should be unchanged."""
        cleaned = clean_cot_text(VALID_COT)
        assert cleaned == VALID_COT.strip()


# ---------------------------------------------------------------------------
# TestCheckEorsTokens
# ---------------------------------------------------------------------------


class TestCheckEorsTokens:
    """Tests for the check_eors_tokens structural gate."""

    def test_passes_with_four_eors(self):
        """A valid CoT with exactly 4 [EORS] tokens should return True."""
        assert check_eors_tokens(VALID_COT) is True

    def test_fails_with_three_eors(self):
        """A CoT missing one [EORS] token should return False."""
        broken = VALID_COT.replace(EORS_TOKEN, "", 1)  # remove first occurrence
        assert check_eors_tokens(broken) is False

    def test_fails_with_five_eors(self):
        """A CoT with an extra [EORS] token should return False."""
        extra = VALID_COT + " [EORS]"
        assert check_eors_tokens(extra) is False

    def test_fails_with_zero_eors(self):
        """A CoT with no [EORS] tokens should return False."""
        no_tokens = VALID_COT.replace(EORS_TOKEN, "")
        assert check_eors_tokens(no_tokens) is False


# ---------------------------------------------------------------------------
# TestCheckStepTokens
# ---------------------------------------------------------------------------


class TestCheckStepTokens:
    """Tests for the check_step_tokens structural gate."""

    def test_passes_with_four_step(self):
        """A valid CoT with exactly 4 [STEP] tokens should return True."""
        assert check_step_tokens(VALID_COT) is True

    def test_fails_with_three_step(self):
        """A CoT missing one [STEP] token should return False."""
        broken = VALID_COT.replace(STEP_TOKEN, "", 1)
        assert check_step_tokens(broken) is False

    def test_fails_with_five_step(self):
        """A CoT with an extra [STEP] token should return False."""
        extra = "[STEP] Extra step. " + VALID_COT
        assert check_step_tokens(extra) is False

    def test_fails_with_zero_step(self):
        """A CoT with no [STEP] tokens should return False."""
        no_tokens = VALID_COT.replace(STEP_TOKEN, "")
        assert check_step_tokens(no_tokens) is False


# ---------------------------------------------------------------------------
# TestCheckTokenCount
# ---------------------------------------------------------------------------


class TestCheckTokenCount:
    """Tests for the BERT token count gate."""

    def test_passes_for_short_text(self):
        """A short CoT well under 512 tokens should return True."""
        assert check_token_count(VALID_COT) is True

    def test_fails_for_long_text(self):
        """A CoT exceeding 512 BERT tokens should return False."""
        # Repeat the valid CoT enough times to exceed 512 tokens
        long_text = (VALID_COT + " ") * 10
        assert check_token_count(long_text) is False

    def test_boundary_511_tokens(self):
        """A text that sits just under the limit should pass."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # Build a string of exactly 509 tokens + 2 special tokens = 511
        word = "loan "
        text = word * 509
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)
        assert len(tokens["input_ids"]) < 512
        assert check_token_count(text) is True


# ---------------------------------------------------------------------------
# TestCheckForbiddenPhrases
# ---------------------------------------------------------------------------


class TestCheckForbiddenPhrases:
    """Tests for the forbidden phrase / label-leakage regex gate."""

    def test_passes_for_clean_cot(self):
        """A CoT with no forbidden phrases should return True."""
        assert check_forbidden_phrases(VALID_COT) is True

    @pytest.mark.parametrize(
        "phrase",
        [
            "The loan should be approved.",
            "This applicant is creditworthy.",
            "The client is likely to default on this loan.",
            "The application should be rejected.",
            "The applicant is eligible for the loan.",
            "The loan should be denied.",
            "They will default.",
            "They will repay the loan.",
            "This is a high risk profile.",
        ],
    )
    def test_fails_for_forbidden_phrase(self, phrase):
        """Any CoT containing a forbidden phrase should return False."""
        cot_with_leakage = VALID_COT + " " + phrase
        assert check_forbidden_phrases(cot_with_leakage) is False

    def test_case_insensitive(self):
        """Forbidden phrase detection should be case-insensitive."""
        assert check_forbidden_phrases(VALID_COT + " APPROVED.") is False
        assert check_forbidden_phrases(VALID_COT + " Creditworthy applicant.") is False


# ---------------------------------------------------------------------------
# TestCheckFaithfulness (DeBERTa — mocked to avoid loading model in CI)
# ---------------------------------------------------------------------------


class TestCheckFaithfulness:
    """Tests for the DeBERTa NLI faithfulness check.

    The model pipeline is mocked so these tests run without a GPU or
    network access. Integration tests that call the real model should be
    placed in a separate tests/integration/ directory.
    """

    def _make_mock_pipe(self, contradiction_score: float):
        """Helper that returns a mock DeBERTa pipeline with a fixed output."""
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "CONTRADICTION", "score": contradiction_score},
            {"label": "NEUTRAL", "score": (1 - contradiction_score) / 2},
            {"label": "ENTAILMENT", "score": (1 - contradiction_score) / 2},
        ]
        return mock_pipe

    def test_passes_when_contradiction_is_low(self):
        """A low contradiction probability should produce a passing quality score."""
        with patch(
            "src.cot.cot_validator._get_deberta_pipe",
            return_value=self._make_mock_pipe(0.05),
        ):
            passed, score = check_faithfulness(
                VALID_COT, ORIGINAL_PROFILE, threshold=0.5
            )
        assert passed is True
        assert score == pytest.approx(0.95, abs=1e-4)

    def test_fails_when_contradiction_is_high(self):
        """A high contradiction probability should produce a failing quality score."""
        with patch(
            "src.cot.cot_validator._get_deberta_pipe",
            return_value=self._make_mock_pipe(0.80),
        ):
            passed, score = check_faithfulness(
                VALID_COT, ORIGINAL_PROFILE, threshold=0.5
            )
        assert passed is False
        assert score == pytest.approx(0.20, abs=1e-4)

    def test_quality_score_is_one_minus_contradiction(self):
        """Quality score must equal 1 - contradiction_probability exactly."""
        contradiction = 0.33
        with patch(
            "src.cot.cot_validator._get_deberta_pipe",
            return_value=self._make_mock_pipe(contradiction),
        ):
            _, score = check_faithfulness(VALID_COT, ORIGINAL_PROFILE)
        assert score == pytest.approx(1.0 - contradiction, abs=1e-4)

    def test_boundary_at_threshold(self):
        """A quality score exactly at the threshold should pass."""
        threshold = 0.5
        contradiction = 1.0 - threshold  # quality score == threshold exactly
        with patch(
            "src.cot.cot_validator._get_deberta_pipe",
            return_value=self._make_mock_pipe(contradiction),
        ):
            passed, score = check_faithfulness(
                VALID_COT, ORIGINAL_PROFILE, threshold=threshold
            )
        assert passed is True


# ---------------------------------------------------------------------------
# TestCheckFinbertLabel (FinBERT — mocked)
# ---------------------------------------------------------------------------


class TestCheckFinbertLabel:
    """Tests for the FinBERT classification check.

    The model pipeline is mocked to avoid requiring a fine-tuned
    FinBERT checkpoint during unit testing.
    """

    def _make_mock_finbert(self, positive: float, negative: float, neutral: float):
        """Helper that returns a mock FinBERT pipeline with fixed scores."""
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "positive", "score": positive},
            {"label": "negative", "score": negative},
            {"label": "neutral", "score": neutral},
        ]
        return mock_pipe

    def test_passes_for_correct_label_repaid(self):
        """FinBERT predicting 'positive' with high confidence for label=0 should pass."""
        with patch(
            "src.cot.cot_validator._get_finbert_pipe",
            return_value=self._make_mock_finbert(0.85, 0.10, 0.05),
        ):
            passed, confidence = check_finbert_label(VALID_COT, ground_truth_label=0)
        assert passed is True
        assert confidence == pytest.approx(0.85, abs=1e-4)

    def test_passes_for_correct_label_defaulted(self):
        """FinBERT predicting 'negative' with high confidence for label=1 should pass."""
        with patch(
            "src.cot.cot_validator._get_finbert_pipe",
            return_value=self._make_mock_finbert(0.05, 0.90, 0.05),
        ):
            passed, confidence = check_finbert_label(VALID_COT, ground_truth_label=1)
        assert passed is True
        assert confidence == pytest.approx(0.90, abs=1e-4)

    def test_fails_for_wrong_predicted_label(self):
        """FinBERT predicting the wrong class should return False."""
        # Ground truth is repaid (0 = positive) but FinBERT says negative
        with patch(
            "src.cot.cot_validator._get_finbert_pipe",
            return_value=self._make_mock_finbert(0.10, 0.85, 0.05),
        ):
            passed, confidence = check_finbert_label(VALID_COT, ground_truth_label=0)
        assert passed is False

    def test_fails_when_confidence_below_threshold(self):
        """Correct predicted label but low confidence should still fail."""
        # Correct label=0 (positive) predicted but confidence is below threshold
        with patch(
            "src.cot.cot_validator._get_finbert_pipe",
            return_value=self._make_mock_finbert(0.40, 0.35, 0.25),
        ):
            passed, confidence = check_finbert_label(
                VALID_COT, ground_truth_label=0, threshold=0.5
            )
        assert passed is False
        assert confidence == pytest.approx(0.40, abs=1e-4)


# ---------------------------------------------------------------------------
# TestValidateCot — Test end-to-end the master validate_cot pipeline
# ---------------------------------------------------------------------------


class TestValidateCot:
    """End-to-end tests for the validate_cot master function.

    Both model pipelines are mocked so no GPU or model downloads are
    required.
    """

    def _patch_models(
        self, contradiction=0.05, positive=0.85, negative=0.10, neutral=0.05
    ):
        """Context manager helper that mocks both DeBERTa and FinBERT."""
        deberta_mock = MagicMock()
        deberta_mock.return_value = [
            {"label": "CONTRADICTION", "score": contradiction},
            {"label": "NEUTRAL", "score": (1 - contradiction) / 2},
            {"label": "ENTAILMENT", "score": (1 - contradiction) / 2},
        ]
        finbert_mock = MagicMock()
        finbert_mock.return_value = [
            {"label": "positive", "score": positive},
            {"label": "negative", "score": negative},
            {"label": "neutral", "score": neutral},
        ]
        return deberta_mock, finbert_mock

    def test_approves_valid_cot(self):
        """A fully valid CoT should be approved with approved=True."""
        deberta_mock, finbert_mock = self._patch_models()
        with (
            patch(
                "src.cot.cot_validator._get_deberta_pipe",
                return_value=deberta_mock,
            ),
            patch(
                "src.cot.cot_validator._get_finbert_pipe",
                return_value=finbert_mock,
            ),
        ):
            result = validate_cot(VALID_COT, ORIGINAL_PROFILE, GROUND_TRUTH_LABEL)

        assert result["approved"] is True
        assert all(result["gates"].values())
        assert result["quality_score"] > 0.5

    def test_rejects_cot_with_missing_eors(self):
        """A CoT missing an [EORS] token should be rejected at the gate stage."""
        broken = VALID_COT.replace(EORS_TOKEN, "", 1)
        deberta_mock, finbert_mock = self._patch_models()
        with (
            patch(
                "src.cot.cot_validator._get_deberta_pipe",
                return_value=deberta_mock,
            ),
            patch(
                "src.cot.cot_validator._get_finbert_pipe",
                return_value=finbert_mock,
            ),
        ):
            result = validate_cot(broken, ORIGINAL_PROFILE, GROUND_TRUTH_LABEL)

        assert result["approved"] is False
        assert result["gates"]["eors_tokens"] is False

    def test_rejects_cot_with_forbidden_phrase(self):
        """A CoT with a forbidden phrase should be rejected at the gate stage."""
        leaky = VALID_COT.replace(
            EORS_TOKEN, " The applicant is creditworthy.[EORS]", 1
        )
        print(leaky)
        deberta_mock, finbert_mock = self._patch_models()
        with (
            patch(
                "src.cot.cot_validator._get_deberta_pipe",
                return_value=deberta_mock,
            ),
            patch(
                "src.cot.cot_validator._get_finbert_pipe",
                return_value=finbert_mock,
            ),
        ):
            result = validate_cot(leaky, ORIGINAL_PROFILE, GROUND_TRUTH_LABEL)

        assert result["approved"] is False
        assert result["gates"]["forbidden_phrases"] is False

    def test_rejects_cot_with_low_faithfulness(self):
        """A CoT that contradicts the profile should be rejected on faithfulness."""
        deberta_mock, finbert_mock = self._patch_models(contradiction=0.90)
        with (
            patch(
                "src.cot.cot_validator._get_deberta_pipe",
                return_value=deberta_mock,
            ),
            patch(
                "src.cot.cot_validator._get_finbert_pipe",
                return_value=finbert_mock,
            ),
        ):
            result = validate_cot(VALID_COT, ORIGINAL_PROFILE, GROUND_TRUTH_LABEL)

        assert result["approved"] is False
        assert result["faithfulness_score"] < 0.5

    def test_quality_score_weighted_combination(self):
        """Quality score should equal 0.6 * faithfulness + 0.4 * finbert_confidence."""
        deberta_mock, finbert_mock = self._patch_models(
            contradiction=0.20,  # faithfulness = 0.80
            positive=0.70,  # finbert confidence on label=0 = 0.70
        )
        with (
            patch(
                "src.cot.cot_validator._get_deberta_pipe",
                return_value=deberta_mock,
            ),
            patch(
                "src.cot.cot_validator._get_finbert_pipe",
                return_value=finbert_mock,
            ),
        ):
            result = validate_cot(VALID_COT, ORIGINAL_PROFILE, GROUND_TRUTH_LABEL)

        expected = round(0.6 * 0.80 + 0.4 * 0.70, 4)
        assert result["quality_score"] == pytest.approx(expected, abs=1e-3)

    def test_output_contains_cleaned_cot(self):
        """The result dict should contain the cleaned version of the CoT."""
        raw_with_preamble = "Here is my reasoning:\n" + VALID_COT
        deberta_mock, finbert_mock = self._patch_models()
        with (
            patch(
                "src.cot.cot_validator._get_deberta_pipe",
                return_value=deberta_mock,
            ),
            patch(
                "src.cot.cot_validator._get_finbert_pipe",
                return_value=finbert_mock,
            ),
        ):
            result = validate_cot(
                raw_with_preamble, ORIGINAL_PROFILE, GROUND_TRUTH_LABEL
            )

        assert result["cleaned_cot"].startswith(STEP_TOKEN)
        assert result["cleaned_cot"].endswith(EORS_TOKEN)
