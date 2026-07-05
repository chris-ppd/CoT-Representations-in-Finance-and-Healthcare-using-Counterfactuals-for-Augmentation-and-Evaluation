"""
Unit tests for src/counterfactual/lime_runner.py.

All heavy dependencies (LIME explainer, BERT validator, NLTK corpus) are
mocked so tests run without GPU, network access, or corpus downloads.
File I/O uses pytest's tmp_path fixture.

Test classes
------------
TestMakePredictProba  – predict_proba wrapper: output shape, label-to-column
                        mapping, numpy input acceptance, unknown label handling
TestRunLime           – output field structure, per-sentence splitting, structural
                        token stripping, stopword filtering, whitespace token
                        exclusion, checkpoint resume

Run with:
    pytest test/test_lime_runner.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.counterfactual.lime_runner import _make_predict_proba, run_lime

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LABEL_MAP = {"repaid": 0, "defaulted": 1}

# Two minimal CoT profiles — cot_text has no [ANSWER] block so the strip
# is a no-op, matching the finbert-scored file schema.
_PROFILES = [
    {
        "id": "id_0",
        "text": "the customer has high debt ratio income stable",
        "cot_text": "the customer has high debt ratio income stable",
        "label": 0,
    },
    {
        "id": "id_1",
        "text": "poor credit history delinquency accounts overdue",
        "cot_text": "poor credit history delinquency accounts overdue",
        "label": 1,
    },
]


def _pipe_output(texts, label="repaid", score=0.9):
    """Return HuggingFace-style all-scores output for n input texts."""
    other = "defaulted" if label == "repaid" else "repaid"
    return [
        [{"label": label, "score": score}, {"label": other, "score": round(1 - score, 2)}]
        for _ in texts
    ]


def _fake_explanation(as_list: list[str], local_exp_items: list[tuple], top_label: int = 1):
    """Build a minimal mock LIME Explanation whose key attributes are controllable."""
    exp = MagicMock()
    exp.top_labels = [top_label]
    exp.local_exp = {top_label: local_exp_items}
    # TextDomainMapper wraps IndexedString; as_list lives one level deeper
    exp.domain_mapper.indexed_string.as_list = as_list
    return exp


def _interleave(text: str) -> list[str]:
    """Reproduce LIME's capturing-group split interleaving (word, space, word, …)."""
    words = text.split()
    result: list[str] = []
    for i, w in enumerate(words):
        result.append(w)
        if i < len(words) - 1:
            result.append(" ")
    return result


# ---------------------------------------------------------------------------
# TestMakePredictProba
# ---------------------------------------------------------------------------


class TestMakePredictProba:
    """Tests for the predict_proba factory that wraps a HuggingFace pipeline."""

    def test_output_shape_is_n_by_2(self):
        """Wrapper should return an ndarray of shape (n, 2) for n input texts."""
        mock_pipe = MagicMock(side_effect=lambda texts, **kw: _pipe_output(texts))
        fn = _make_predict_proba(mock_pipe, LABEL_MAP, 64)

        result = fn(["text a", "text b", "text c"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

    def test_label_indices_map_to_correct_columns(self):
        """'repaid' probability must land in column 0, 'defaulted' in column 1."""
        mock_pipe = MagicMock(return_value=[
            [{"label": "repaid", "score": 0.8}, {"label": "defaulted", "score": 0.2}],
        ])
        fn = _make_predict_proba(mock_pipe, LABEL_MAP, 64)

        result = fn(["any text"])

        assert result[0, 0] == pytest.approx(0.8)
        assert result[0, 1] == pytest.approx(0.2)

    def test_accepts_numpy_array_input(self):
        """Wrapper must convert np.ndarray to list before calling the pipeline."""
        mock_pipe = MagicMock(side_effect=lambda texts, **kw: _pipe_output(texts))
        fn = _make_predict_proba(mock_pipe, LABEL_MAP, 64)

        # LIME internally passes perturbed texts as a numpy array of strings
        result = fn(np.array(["text x", "text y"]))

        assert result.shape == (2, 2)

    def test_unknown_label_leaves_column_at_zero(self):
        """A label string not in label_map must not raise and must leave the column 0."""
        mock_pipe = MagicMock(return_value=[
            [{"label": "LABEL_UNKNOWN", "score": 1.0}],
        ])
        fn = _make_predict_proba(mock_pipe, LABEL_MAP, 64)

        result = fn(["any text"])

        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestRunLime — shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def lime_ctx(tmp_path):
    """
    Wire temp JSONL files and mock all heavy dependencies for run_lime.

    Attributes
    ----------
    input_path            : Path       – temp CoT JSONL with two profiles
    output_path           : Path       – destination for LIME output JSONL
    config                : dict       – mutable config pointing to temp files
    mock_explain_instance : MagicMock  – the LIME explainer instance returned
                                         by LimeTextExplainer(); set
                                         .explain_instance.return_value to
                                         control what each sentence gets
    """

    class Ctx:
        pass

    c = Ctx()
    c.input_path = tmp_path / "cot_input.jsonl"
    c.output_path = tmp_path / "lime_output.jsonl"
    c.input_path.write_text(
        "\n".join(json.dumps(p) for p in _PROFILES) + "\n"
    )

    c.config = {
        "dataset": {
            "sub_dataset": "ld1",
            "splits": {
                "train": {
                    "cot_input": str(c.input_path),
                    "lime_output": str(c.output_path),
                    "cf_output": str(tmp_path / "cf.jsonl"),
                }
            },
        },
        "lime": {"num_features": 10, "num_samples": 32, "batch_size": 16},
    }

    # Plain-text profiles (no [EORS]/[ES]) produce one sentence per profile,
    # so one explain_instance call each.
    # Profile 0: "the customer has high debt ratio income stable"
    # Word positions (interleaved): the=0, customer=2, has=4, high=6, debt=8,
    #   ratio=10, income=12, stable=14
    text0 = _PROFILES[0]["cot_text"]
    c.as_list_0 = _interleave(text0)
    # debt(8)=0.45, ratio(10)=0.32, high(6)=-0.28, the(0)=0.10 (stopword)
    c.local_exp_0 = [(8, 0.45), (10, 0.32), (6, -0.28), (0, 0.10)]

    text1 = _PROFILES[1]["cot_text"]
    c.as_list_1 = _interleave(text1)
    c.local_exp_1 = [(0, 0.55), (2, 0.30)]  # 'poor'=0, 'credit'=2

    with (
        patch("src.counterfactual.lime_runner.load_config", return_value=c.config),
        patch("src.counterfactual.lime_runner._load_validator_pipe", return_value=MagicMock()),
        patch(
            "src.counterfactual.lime_runner._make_predict_proba",
            return_value=MagicMock(),
        ),
        patch(
            "src.counterfactual.lime_runner._get_stop_words",
            return_value={"the", "has", "a", "an", "of"},
        ),
        patch("src.counterfactual.lime_runner.torch") as mock_torch,
        patch("src.counterfactual.lime_runner.LimeTextExplainer") as mock_cls,
    ):
        mock_torch.cuda.is_available.return_value = False
        c.mock_explain_instance = mock_cls.return_value
        # Default: one call per profile (plain-text profiles produce one sentence each)
        c.mock_explain_instance.explain_instance.side_effect = [
            _fake_explanation(c.as_list_0, c.local_exp_0),
            _fake_explanation(c.as_list_1, c.local_exp_1),
        ]
        yield c


class TestRunLime:
    """End-to-end tests for run_lime — LIME explainer and validator are mocked."""

    def test_output_records_have_required_fields(self, lime_ctx):
        """Each JSONL record must have 'id' and 'sentence_features' with the expected structure.

        Each entry in sentence_features must carry step, sentence, and top_features.
        Each feature dict must have token, score, and position_idx — no context field.
        """
        run_lime("dummy.yaml", split="train")

        records = [
            json.loads(l)
            for l in lime_ctx.output_path.read_text().splitlines()
            if l.strip()
        ]
        assert len(records) == 2
        for rec in records:
            assert "id" in rec
            assert "sentence_features" in rec
            for sf in rec["sentence_features"]:
                assert "step" in sf
                assert "sentence" in sf
                assert "top_features" in sf
                for feat in sf["top_features"]:
                    assert all(k in feat for k in ("token", "score", "position_idx"))
                    assert "context" not in feat

    def test_stopwords_excluded_from_top_features(self, lime_ctx):
        """Tokens in the stopword set ('the', 'has', ...) must not appear in any top_features."""
        run_lime("dummy.yaml", split="train")

        records = [
            json.loads(l)
            for l in lime_ctx.output_path.read_text().splitlines()
            if l.strip()
        ]
        all_tokens = [
            f["token"]
            for rec in records
            for sf in rec["sentence_features"]
            for f in sf["top_features"]
        ]
        assert "the" not in all_tokens, "'the' is a stopword and must be filtered"
        assert "has" not in all_tokens, "'has' is a stopword and must be filtered"

    def test_whitespace_tokens_excluded_from_top_features(self, lime_ctx):
        """Whitespace-only entries in LIME's interleaved as_list must not appear in output."""
        # Inject a whitespace token with the highest weight to confirm it still gets filtered
        ws_injected_exp = [(1, 0.99)] + lime_ctx.local_exp_0  # pos 1 = ' '
        lime_ctx.mock_explain_instance.explain_instance.side_effect = [
            _fake_explanation(lime_ctx.as_list_0, ws_injected_exp),
            _fake_explanation(lime_ctx.as_list_1, lime_ctx.local_exp_1),
        ]

        run_lime("dummy.yaml", split="train")

        records = [
            json.loads(l)
            for l in lime_ctx.output_path.read_text().splitlines()
            if l.strip()
        ]
        all_tokens = [
            f["token"]
            for rec in records
            for sf in rec["sentence_features"]
            for f in sf["top_features"]
        ]
        assert " " not in all_tokens
        assert "" not in all_tokens

    def test_punctuation_tokens_excluded_from_top_features(self, lime_ctx):
        """Punctuation-only tokens must not appear in top_features even with the highest LIME weight."""
        # Replace the space at position 1 with a punctuation-only token ','
        punct_as_list = list(lime_ctx.as_list_0)
        punct_as_list[1] = ","
        # Inject ',' at position 1 with the highest weight so it would appear first without filtering
        punct_injected_exp = [(1, 0.99)] + lime_ctx.local_exp_0
        lime_ctx.mock_explain_instance.explain_instance.side_effect = [
            _fake_explanation(punct_as_list, punct_injected_exp),
            _fake_explanation(lime_ctx.as_list_1, lime_ctx.local_exp_1),
        ]

        run_lime("dummy.yaml", split="train")

        records = [
            json.loads(l)
            for l in lime_ctx.output_path.read_text().splitlines()
            if l.strip()
        ]
        all_tokens = [
            f["token"]
            for rec in records
            for sf in rec["sentence_features"]
            for f in sf["top_features"]
        ]
        assert "," not in all_tokens, "Punctuation-only ',' must be filtered even at highest weight"

    def test_per_sentence_splitting_generates_one_call_per_sentence(self, lime_ctx):
        """Each non-empty sentence in the CoT produces exactly one explain_instance call,
        keyed by (step, sentence) in the output."""
        structured_profile = {
            "id": "id_struct",
            "text": "Structured profile",
            "cot_text": "[STEP] Debt is high. [ES] Income is low. [EORS]",
            "label": 0,
        }
        lime_ctx.input_path.write_text(json.dumps(structured_profile) + "\n")
        # 1 step × 2 sentences = 2 LIME calls
        lime_ctx.mock_explain_instance.explain_instance.side_effect = [
            _fake_explanation(["Debt", " ", "is", " ", "high", "."], [(0, 0.5), (4, 0.3)]),
            _fake_explanation(["Income", " ", "is", " ", "low", "."], [(0, 0.4)]),
        ]

        run_lime("dummy.yaml", split="train")

        assert lime_ctx.mock_explain_instance.explain_instance.call_count == 2
        records = [
            json.loads(l)
            for l in lime_ctx.output_path.read_text().splitlines()
            if l.strip()
        ]
        assert len(records) == 1
        slot_keys = {(sf["step"], sf["sentence"]) for sf in records[0]["sentence_features"]}
        assert (0, 0) in slot_keys
        assert (0, 1) in slot_keys

    def test_structural_tokens_stripped_from_explain_instance_input(self, lime_ctx):
        """[STEP], [ES], and [EORS] must not appear in the text passed to explain_instance."""
        structured_profile = {
            "id": "id_struct",
            "text": "Structured profile",
            "cot_text": "[STEP] Token A. [ES] Token B. [EORS]",
            "label": 0,
        }
        lime_ctx.input_path.write_text(json.dumps(structured_profile) + "\n")
        lime_ctx.mock_explain_instance.explain_instance.side_effect = [
            _fake_explanation(["Token", " ", "A", "."], [(0, 0.5)]),
            _fake_explanation(["Token", " ", "B", "."], [(0, 0.4)]),
        ]

        run_lime("dummy.yaml", split="train")

        call_texts = [
            call[0][0]
            for call in lime_ctx.mock_explain_instance.explain_instance.call_args_list
        ]
        for text in call_texts:
            assert "[STEP]" not in text
            assert "[ES]" not in text
            assert "[EORS]" not in text

    def test_lime_receives_cot_text_with_answer_stripped(self, lime_ctx):
        """Sentences passed to explain_instance must not contain anything from the [ANSWER] block."""
        # Inject [ANSWER] block into one profile's cot_text
        profile_with_answer = dict(_PROFILES[0])
        profile_with_answer["cot_text"] = (
            "the customer has high debt ratio income stable"
            " [ANSWER] Assessed as repaid. [EOA]"
        )
        lime_ctx.input_path.write_text(
            json.dumps(profile_with_answer) + "\n" + json.dumps(_PROFILES[1]) + "\n"
        )

        run_lime("dummy.yaml", split="train")

        # The first explain_instance call must NOT include [ANSWER] in its text argument.
        # Plain-text profile (no delimiters) → one sentence = the full cot_text_only.
        first_call_text = lime_ctx.mock_explain_instance.explain_instance.call_args_list[0][0][0]
        assert "[ANSWER]" not in first_call_text
        assert "Assessed as repaid" not in first_call_text
        assert "the customer has high debt ratio income stable" == first_call_text

    def test_checkpoint_resume_skips_already_processed_ids(self, lime_ctx):
        """Profiles whose IDs already exist in the output file must not be re-explained."""
        # Seed the output file with id_0 already processed (new sentence_features format)
        lime_ctx.output_path.write_text(
            json.dumps({"id": "id_0", "sentence_features": []}) + "\n"
        )
        # Reset side_effect so only one explanation is needed (for id_1)
        lime_ctx.mock_explain_instance.explain_instance.side_effect = [
            _fake_explanation(lime_ctx.as_list_1, lime_ctx.local_exp_1),
        ]

        run_lime("dummy.yaml", split="train")

        # explain_instance called only once — id_0 was already in output
        assert lime_ctx.mock_explain_instance.explain_instance.call_count == 1
        records = [
            json.loads(l)
            for l in lime_ctx.output_path.read_text().splitlines()
            if l.strip()
        ]
        ids = [r["id"] for r in records]
        assert ids.count("id_0") == 1  # existing entry preserved, not duplicated
        assert "id_1" in ids
