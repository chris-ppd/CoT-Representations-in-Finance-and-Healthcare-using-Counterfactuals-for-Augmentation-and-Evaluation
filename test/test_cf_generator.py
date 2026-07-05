"""
Unit tests for src/counterfactual/cf_generator.py.

All model-loading and heavy-compute functions are mocked so tests run without
GPU, network access, or model downloads.  File I/O uses pytest's tmp_path
fixture.

Test classes
------------
TestCheckLabelFlip             – flip gate: True/False return, label mapping,
                                 predicted label integer
TestReconstructCotSteps        – split/join reconstruction: substitution,
                                 preservation, special-token structure
TestComputeCosineSimilarities  – batch cosine sim: output length, known values,
                                 valid range, symmetric identity
TestComputePerplexity          – per-sequence GPT-2 PPL: output length, uniform-
                                 logit expectation, padding masking, positivity
TestRunCfGeneration            – end-to-end control flow: output fields, flip-gate
                                 rejections queued for retry, max_retries cap,
                                 checkpoint resume, pilot profile count,
                                 counterfactual_label correctness

Run with:
    pytest test/test_cf_generator.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.counterfactual.cf_generator import (
    _check_label_flip_batch,
    _compute_cosine_similarities,
    _compute_perplexity,
    _reconstruct_cot_steps,
    run_cf_generation,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LD1_LABEL_MAP = {"repaid": 0, "defaulted": 1}

# CoT text with clean 2-step × 2-sentence structure
_COT_TEXT = (
    "[STEP] Employment is uncertain. [ES] Risk is elevated. [ES] [EORS]\n"
    "[STEP] Debt level is moderate. [ES] Income is stable. [ES] [EORS]"
)

_PROFILES = [
    {"id": "id_0", "text": "Profile text A.", "cot_text": _COT_TEXT, "label": 0},
    {"id": "id_1", "text": "Profile text B.", "cot_text": _COT_TEXT, "label": 1},
    {"id": "id_2", "text": "Profile text C.", "cot_text": _COT_TEXT, "label": 0},
]

# LIME features in the per-sentence JSONL format.
# Step 0, sentence 0 is the only active slot for all profiles so that
# _process_batch routes each profile's rewrite to slot (0, 0).
_LIME_FEATURES_JSONL = [
    {
        "id": "id_0",
        "sentence_features": [
            {"step": 0, "sentence": 0, "top_features": [{"token": "debt", "score": 0.5, "position_idx": 2}]},
        ],
    },
    {
        "id": "id_1",
        "sentence_features": [
            {"step": 0, "sentence": 0, "top_features": [{"token": "risk", "score": 0.4, "position_idx": 0}]},
        ],
    },
    {
        "id": "id_2",
        "sentence_features": [
            {"step": 0, "sentence": 0, "top_features": [{"token": "income", "score": 0.3, "position_idx": 4}]},
        ],
    },
]


def _read_output(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# TestCheckLabelFlip
# ---------------------------------------------------------------------------


def _make_validator_pipe(predicted_label_str: str, score: float = 0.9):
    """Return a mock validator pipeline that always predicts predicted_label_str."""
    other = "defaulted" if predicted_label_str == "repaid" else "repaid"
    mock_pipe = MagicMock(
        return_value=[
            [
                {"label": predicted_label_str, "score": score},
                {"label": other, "score": round(1 - score, 2)},
            ],
        ]
    )
    return mock_pipe


class TestCheckLabelFlip:
    """Tests for the hard label flip gate."""

    def test_returns_true_when_label_flips(self):
        """flipped=True when the validator predicts 1 - original_label."""
        pipe = _make_validator_pipe("defaulted")
        flipped, _, _ = _check_label_flip_batch(["some cf text"], [0], pipe, LD1_LABEL_MAP)[0]
        assert flipped is True

    def test_returns_false_when_label_unchanged(self):
        """flipped=False when the validator predicts the same class as original."""
        pipe = _make_validator_pipe("repaid")
        flipped, _, _ = _check_label_flip_batch(["some cf text"], [0], pipe, LD1_LABEL_MAP)[0]
        assert flipped is False

    def test_second_return_value_is_predicted_integer_label(self):
        """The second return value must be the integer predicted label, not a string."""
        pipe = _make_validator_pipe("defaulted")
        _, pred, _ = _check_label_flip_batch(["cf text"], [0], pipe, LD1_LABEL_MAP)[0]
        assert pred == 1
        assert isinstance(pred, int)

    def test_uses_label_map_to_resolve_predicted_label(self):
        """The gate resolves the pipeline's string label to an int via label_map."""
        pipe = _make_validator_pipe("repaid")
        flipped, pred, _ = _check_label_flip_batch(["cf text"], [1], pipe, LD1_LABEL_MAP)[0]
        assert flipped is True
        assert pred == 0

    def test_third_return_value_is_confidence_float(self):
        """The third return value must be the best score as a float in (0, 1]."""
        pipe = _make_validator_pipe("defaulted", score=0.9)
        _, _, confidence = _check_label_flip_batch(["cf text"], [0], pipe, LD1_LABEL_MAP)[0]
        assert isinstance(confidence, float)
        assert 0.0 < confidence <= 1.0


# ---------------------------------------------------------------------------
# TestReconstructCotSteps
# ---------------------------------------------------------------------------


class TestReconstructCotSteps:
    """Tests for the [ES]/[EORS] split-join reconstruction."""

    _COT = "[STEP] Original A. [ES] Original B. [ES] [EORS]"

    def test_modified_sentence_replaces_original(self):
        """A slot entry in modified_sentences must replace the sentence at that position."""
        result = _reconstruct_cot_steps(self._COT, {(0, 0): "Rewritten A."})
        assert "Rewritten A." in result
        assert "Original A." not in result

    def test_unmodified_sentences_are_preserved(self):
        """Sentences not in modified_sentences must remain unchanged."""
        result = _reconstruct_cot_steps(self._COT, {(0, 0): "Rewritten A."})
        assert "Original B." in result

    def test_empty_modified_dict_returns_original(self):
        """With no modifications the output must equal the input exactly."""
        result = _reconstruct_cot_steps(self._COT, {})
        assert result == self._COT

    def test_special_tokens_preserved_in_structure(self):
        """[ES] and [EORS] delimiters must survive reconstruction unchanged."""
        result = _reconstruct_cot_steps(self._COT, {(0, 1): "New B."})
        assert "[ES]" in result
        assert "[EORS]" in result

    def test_multiple_slots_all_substituted(self):
        """Both sentence slots in one step can be rewritten independently."""
        result = _reconstruct_cot_steps(
            self._COT, {(0, 0): "New A.", (0, 1): "New B."}
        )
        assert "New A." in result
        assert "New B." in result
        assert "Original A." not in result
        assert "Original B." not in result


# ---------------------------------------------------------------------------
# TestComputeCosineSimilarities
# ---------------------------------------------------------------------------


class TestComputeCosineSimilarities:
    """Tests for the batched cosine similarity computation."""

    def _mock_sbert(self, embeddings: np.ndarray):
        sbert = MagicMock()
        sbert.encode.return_value = embeddings
        return sbert

    def test_output_length_matches_number_of_pairs(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        sbert = self._mock_sbert(embs)
        result = _compute_cosine_similarities(["a", "b"], ["c", "d"], sbert)
        assert len(result) == 2

    def test_identical_embeddings_give_similarity_of_one(self):
        v = np.array([3.0, 4.0])
        embs = np.stack([v, v])
        sbert = self._mock_sbert(embs)
        result = _compute_cosine_similarities(["text"], ["text"], sbert)
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_embeddings_give_similarity_near_zero(self):
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sbert = self._mock_sbert(embs)
        result = _compute_cosine_similarities(["text"], ["other"], sbert)
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_pairs_each_original_with_its_own_counterfactual(self):
        orig_0 = np.array([1.0, 0.0])
        orig_1 = np.array([0.0, 1.0])
        cf_0 = np.array([1.0, 0.0])
        cf_1 = np.array([0.0, 1.0])
        embs = np.stack([orig_0, orig_1, cf_0, cf_1])
        sbert = self._mock_sbert(embs)

        originals = ["orig_text_0", "orig_text_1"]
        counterfactuals = ["cf_text_0", "cf_text_1"]
        result = _compute_cosine_similarities(originals, counterfactuals, sbert)

        assert result[0] == pytest.approx(1.0, abs=1e-5)
        assert result[1] == pytest.approx(1.0, abs=1e-5)

        cross_0 = float(np.dot(orig_0, cf_1))
        cross_1 = float(np.dot(orig_1, cf_0))
        assert result[0] > cross_0
        assert result[1] > cross_1

        sbert.encode.assert_called_once_with(
            originals + counterfactuals,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def test_all_values_within_valid_range(self):
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((6, 64))
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        sbert = self._mock_sbert(embs)
        result = _compute_cosine_similarities(["a", "b", "c"], ["d", "e", "f"], sbert)
        for sim in result:
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# TestComputePerplexity
# ---------------------------------------------------------------------------


class TestComputePerplexity:
    """Tests for the per-sequence GPT-2 perplexity computation."""

    _VOCAB = 10

    def _make_mocks(self, n_texts: int, seq_len: int = 4, all_valid: bool = True):
        input_ids = (
            torch.arange(n_texts * seq_len).reshape(n_texts, seq_len) % self._VOCAB
        )
        attention_mask = torch.ones(n_texts, seq_len, dtype=torch.long)
        if not all_valid:
            attention_mask[-1, -1] = 0

        mock_tok = MagicMock(
            return_value={"input_ids": input_ids, "attention_mask": attention_mask}
        )
        logits = torch.zeros(n_texts, seq_len, self._VOCAB)
        mock_outputs = MagicMock()
        mock_outputs.logits = logits
        mock_gpt2 = MagicMock(return_value=mock_outputs)
        return mock_gpt2, mock_tok

    def test_output_length_matches_number_of_texts(self):
        gpt2, tok = self._make_mocks(n_texts=3)
        result = _compute_perplexity(["a", "b", "c"], gpt2, tok)
        assert len(result) == 3

    def test_uniform_logits_give_vocab_size_as_ppl(self):
        gpt2, tok = self._make_mocks(n_texts=1, seq_len=5)
        result = _compute_perplexity(["text"], gpt2, tok)
        assert result[0] == pytest.approx(self._VOCAB, rel=1e-4)

    def test_padding_tokens_excluded_from_loss(self):
        gpt2, tok = self._make_mocks(n_texts=2, seq_len=4, all_valid=False)
        result = _compute_perplexity(["text1", "text2"], gpt2, tok)
        assert result[0] == pytest.approx(result[1], rel=1e-4)

    def test_all_ppl_values_are_positive(self):
        gpt2, tok = self._make_mocks(n_texts=4, seq_len=6)
        result = _compute_perplexity(["a", "b", "c", "d"], gpt2, tok)
        assert all(ppl > 0 for ppl in result)


# ---------------------------------------------------------------------------
# TestRunCfGeneration — shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def cf_ctx(tmp_path):
    """
    Wire temp files and mock all heavy dependencies for run_cf_generation.

    Attributes
    ----------
    output_path  : Path       – destination CF JSONL
    config       : dict       – mutable config pointing to temp files
    mock_gen     : MagicMock  – controls _generate_qwen_batch return value
    mock_flip    : MagicMock  – controls _check_label_flip_batch return value
    mock_cos     : MagicMock  – controls _compute_cosine_similarities return value
    mock_ppl     : MagicMock  – controls _compute_perplexity return value
    """

    class Ctx:
        pass

    c = Ctx()
    c.output_path = tmp_path / "cf_output.jsonl"

    # Write temp CoT profiles JSONL
    cot_input_path = tmp_path / "cot_input.jsonl"
    cot_input_path.write_text("\n".join(json.dumps(p) for p in _PROFILES) + "\n")

    # Write temp LIME features JSONL in per-sentence format
    lime_path = tmp_path / "lime.jsonl"
    lime_path.write_text(
        "\n".join(json.dumps(entry) for entry in _LIME_FEATURES_JSONL) + "\n"
    )

    c.config = {
        "dataset": {
            "sub_dataset": "ld1",
            "id2label": {0: "repaid", 1: "defaulted"},
            "splits": {
                "train": {
                    "cot_input": str(cot_input_path),
                    "lime_output": str(lime_path),
                    "cf_output": str(c.output_path),
                }
            },
        },
        "generation": {
            "teacher_model": {
                "name": "mock-qwen",
                "quantization": None,
                "max_new_tokens": 64,
                "batch_size": 10,
            },
            "top_k": 3,
            "max_retries": 3,
        },
    }

    with (
        patch("src.counterfactual.cf_generator.load_config", return_value=c.config),
        patch(
            "src.counterfactual.cf_generator._load_qwen",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "src.counterfactual.cf_generator._load_validator", return_value=MagicMock()
        ),
        patch("src.counterfactual.cf_generator._load_sbert", return_value=MagicMock()),
        patch(
            "src.counterfactual.cf_generator._load_gpt2",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "src.counterfactual.cf_generator._load_prompt",
            return_value="mock system prompt",
        ),
        patch("src.counterfactual.cf_generator._check_token_budget", return_value=True),
        patch("src.counterfactual.cf_generator._generate_qwen_batch") as mock_gen,
        patch("src.counterfactual.cf_generator._check_label_flip_batch") as mock_flip,
        patch(
            "src.counterfactual.cf_generator._compute_cosine_similarities"
        ) as mock_cos,
        patch("src.counterfactual.cf_generator._compute_perplexity") as mock_ppl,
        patch("src.counterfactual.cf_generator.print_memory_usage"),
    ):
        # _generate_qwen_batch: return one string per prompt in the batch
        mock_gen.side_effect = lambda msgs, *a, **kw: [
            f"rewritten_{i}" for i in range(len(msgs))
        ]

        # Flip gate: always passes, predicts the target label
        mock_flip.side_effect = lambda cf_texts, labels, *a, **kw: [
            (True, 1 - lbl, 0.9) for lbl in labels
        ]

        mock_cos.side_effect = lambda orig, cfs, *a, **kw: [0.9] * len(cfs)
        mock_ppl.side_effect = lambda texts, *a, **kw: [42.0] * len(texts)

        c.mock_gen = mock_gen
        c.mock_flip = mock_flip
        c.mock_cos = mock_cos
        c.mock_ppl = mock_ppl
        yield c


def _run(pilot=False, pilot_n=100, top_k=None):
    run_cf_generation(
        "dummy.yaml", split="train", top_k=top_k, pilot=pilot, pilot_n=pilot_n
    )


class TestRunCfGeneration:
    """End-to-end control flow tests for run_cf_generation."""

    def test_approved_records_have_all_required_fields(self, cf_ctx):
        """Every saved record must contain all fields defined in the output schema."""
        _run()

        records = _read_output(cf_ctx.output_path)
        assert len(records) == 3
        required = {
            "id",
            "original_id",
            "original_cot_text",
            "original_answer_text",
            "counterfactual_cot_text",
            "counterfactual_answer_text",
            "original_label",
            "counterfactual_label",
            "modified_sentences",
            "top_k_words",
            "cosine_similarity",
            "ppl",
            "retries",
        }
        for rec in records:
            assert required.issubset(rec.keys()), (
                f"Missing fields: {required - rec.keys()}"
            )

    def test_cf_id_is_prefixed_with_cf_and_original_id_matches(self, cf_ctx):
        """CF record id must be 'cf_{original_id}' and original_id must equal source id."""
        _run()

        for rec in _read_output(cf_ctx.output_path):
            assert rec["id"] == f"cf_{rec['original_id']}"

    def test_counterfactual_label_equals_one_minus_original_label(self, cf_ctx):
        """counterfactual_label must equal 1 - original_label for every saved record."""
        _run()

        for rec in _read_output(cf_ctx.output_path):
            assert rec["counterfactual_label"] == 1 - rec["original_label"]

    def test_original_cot_text_preserved_in_output(self, cf_ctx):
        """original_cot_text in each record must equal the input cot_text field."""
        _run()

        for rec in _read_output(cf_ctx.output_path):
            assert rec["original_cot_text"] == _COT_TEXT

    def test_modified_sentences_is_list_of_slot_pairs(self, cf_ctx):
        """modified_sentences must be a list of [step_idx, sent_idx] pairs."""
        _run()

        for rec in _read_output(cf_ctx.output_path):
            assert isinstance(rec["modified_sentences"], list)
            for pair in rec["modified_sentences"]:
                assert len(pair) == 2
                assert all(isinstance(x, int) for x in pair)

    def test_flip_gate_failures_are_queued_and_retried(self, cf_ctx):
        """Profiles where the gate fails are not saved on the main pass and get retried."""
        # Main pass: id_0 fails, id_1 and id_2 pass
        # Retry round 1: id_0 passes
        cf_ctx.mock_flip.side_effect = [
            [(False, 0, 0.3), (True, 1, 0.9), (True, 0, 0.9)],
            [(True, 1, 0.9)],
        ]
        cf_ctx.mock_cos.side_effect = lambda orig, cfs, *a, **kw: [0.88] * len(cfs)
        cf_ctx.mock_ppl.side_effect = lambda texts, *a, **kw: [50.0] * len(texts)

        _run()

        records = _read_output(cf_ctx.output_path)
        assert len(records) == 3
        assert {r["original_id"] for r in records} == {"id_0", "id_1", "id_2"}

    def test_retry_loop_stops_at_max_retries(self, cf_ctx):
        """Flip gate is called exactly 1 + max_retries times when all profiles always fail."""
        cf_ctx.config["generation"]["max_retries"] = 2
        cf_ctx.mock_flip.side_effect = lambda cf_texts, labels, *a, **kw: [
            (False, 0, 0.3)
        ] * len(cf_texts)

        _run()

        # 1 main pass + 2 retry rounds = 3 flip-gate calls
        assert cf_ctx.mock_flip.call_count == 3
        assert len(_read_output(cf_ctx.output_path)) == 0

    def test_checkpoint_resume_skips_already_saved_original_ids(self, cf_ctx):
        """Profiles whose original_id already appears in the output file are skipped."""
        existing = {
            "id": "cf_id_0",
            "original_id": "id_0",
            "original_text": "old",
            "original_cot_text": "old cot",
            "counterfactual_cot_text": "old cf cot",
            "original_label": 0,
            "counterfactual_label": 1,
            "modified_sentences": [[0, 0]],
            "top_k_words": [],
            "cosine_similarity": 0.8,
            "ppl": 35.0,
            "retries": 0,
        }
        cf_ctx.output_path.write_text(json.dumps(existing) + "\n")

        _run()

        records = _read_output(cf_ctx.output_path)
        original_ids = [r["original_id"] for r in records]
        # id_0 was pre-existing; must not be duplicated
        assert original_ids.count("id_0") == 1
        # id_1 and id_2 must be newly generated
        assert "id_1" in original_ids
        assert "id_2" in original_ids
        # Flip gate was called only once with 2 profiles (id_1, id_2)
        flip_batch_sizes = [len(args[0]) for args, _ in cf_ctx.mock_flip.call_args_list]
        assert all(n <= 2 for n in flip_batch_sizes)

    def test_pilot_mode_processes_exactly_pilot_n_profiles(self, cf_ctx):
        """In pilot mode, at most pilot_n profiles are processed and written to the
        k-specific pilot file (not the main output file)."""
        top_k = cf_ctx.config["generation"]["top_k"]
        pilot_path = Path(
            str(cf_ctx.output_path).replace(".jsonl", f"_pilot_k{top_k}.jsonl")
        )

        _run(pilot=True, pilot_n=2)

        records = _read_output(pilot_path)
        assert len(records) == 2
