"""
Validation pipeline for generated Chain-of-Thought (CoT) profiles.

Each validator function acts as a unit test: it takes the CoT text (and
optionally the original profile text / ground-truth label) and returns
True if the check passes, False otherwise.

The final function `validate_cot` runs all checks in order and returns
True only if every gate passes and the continuous quality scores meet
the minimum thresholds.

Pipeline order
--------------
1. clean_cot_text          – strip preamble / trailing text (preprocessing, not a gate)
2. check_eors_tokens       – exactly 4 [EORS] tokens present
3. check_step_tokens       – exactly 4 [STEP] tokens present
4. check_es_tokens         – exactly 8 [ES] tokens (2 per step) in correct alternating order
5. check_answer_block      – exactly 1 [ANSWER] and 1 [EOA], ordered after 4th [EORS], 30-35 tokens
6. check_sentence_structure – per-step sentence length and balance constraints (2 sentences)
7. check_token_count       – BERT token count of CoT steps ≤ 512 tokens
8. check_forbidden_phrases – no explicit label-leakage keywords in CoT steps (not answer block)
9. check_faithfulness      – DeBERTa contradiction probability on (profile, answer) (quality signal)
10. check_bert_label       – BERT validator predicted label matches ground truth

Usage
-----
    from cot_validator import validate_cot, FINANCE_VALIDATOR_CONFIG

    result = validate_cot(
        cot_text=my_cot,
        original_profile=raw_profile_text,
        ground_truth_label=1,
        config=FINANCE_VALIDATOR_CONFIG,
        verbose=True,
    )
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    pipeline,
)

from utils.memory_printer import print_memory_usage

# suppress HuggingFace and transformers noisy warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


STEP_TOKEN = "[STEP]"
EORS_TOKEN = "[EORS]"  # end of reasoning step special token
ES_TOKEN = "[ES]"  # end sentence special token
ANSWER_TOKEN = "[ANSWER]"  # start of answer block
EOA_TOKEN = "[EOA]"  # end of answer block
BERT_MAX_TOKENS = 512

# Sentence structure constraints (token counts measured with BERT tokenizer)
SENTENCE_MIN_TOKENS = 10
SENTENCE_MAX_TOKENS = 70
SENTENCE_MAX_DIFF = 50  # max token-count difference between any two sentences in a step (redundant but ok)

# Answer block constraints (token counts measured with DeBERTa tokenizer)
ANSWER_MIN_TOKENS = 10
ANSWER_MAX_TOKENS = 70

# Domain-agnostic model identifiers
DEBERTA_MODEL_ID = "microsoft/deberta-base-mnli"
BERT_TOKENIZER_ID = "bert-base-uncased"

# ---------------------------------------------------------------------------
# Domain-specific forbidden phrases
# ---------------------------------------------------------------------------

# Finance: phrases that directly reveal the credit/loan decision outcome for ld1.
FINANCE_FORBIDDEN_PHRASES: list[str] = [
    r"\bapprove[sd]?\b",
    r"\breject(?:ed|ion)?\b",
    r"\bcreditworthy\b",
    r"\blikely\s+to\s+default\b",
    r"\beligible\b",
    r"\bloan\s+should\s+be\s+granted\b",
    r"\bloan\s+should\s+be\s+denied\b",
    r"\bwill\s+default\b",
    r"\bwill\s+repay\b",
    r"\bhigh\s+risk\b",
    r"\blow\s+risk\b",
    r"\bdefault\s+risk\b",
    r"\bshould\s+not\s+be\s+approved\b",
    r"\bshould\s+be\s+approved\b",
    r"\bground\s+truth\b",
    r"\blabel\s+of\s+default\b",
    r"\blabel\s+of\s+repaid\b",
    r"\bultimately\s+defaulted\b",
    r"\bultimately\s+repaid\b",
    r"\beventually\s+defaulted\b",
    r"\beventually\s+repaid\b",
    r"\bresulted\s+in\s+default\b",
    r"\bresulted\s+in\s+repayment\b",
    r"\bthis\s+client\s+defaulted\b",
    r"\bthis\s+client\s+repaid\b",
]

# Healthcare: phrases that directly reveal the clinical outcome prediction for er-reason.
HEALTHCARE_FORBIDDEN_PHRASES: list[str] = [
    r"\bwill\s+(?:not\s+)?survive\b",
    r"\bwill\s+(?:not\s+)?recover\b",
    r"\bwill\s+die\b",
    r"\bfatal(?:\s+outcome)?\b",
    r"\bpoor\s+prognosis\b",
    r"\bgood\s+prognosis\b",
    r"\bhigh\s+mortality\b",
    r"\blow\s+mortality\b",
    r"\blikely\s+to\s+(?:survive|die|recover)\b",
    r"\bunlikely\s+to\s+(?:survive|die|recover)\b",
    r"\bpredicted\s+(?:to\s+)?(?:die|survive|recover)\b",
    r"\breadmission\s+(?:is\s+)?(?:likely|unlikely)\b",
    r"\bwill\s+be\s+readmitted\b",
    r"\bshould\s+be\s+discharged\b",
    r"\bhigh\s+risk\s+of\s+(?:death|mortality|readmission)\b",
    r"\blow\s+risk\s+of\s+(?:death|mortality|readmission)\b",
    r"\bshould\s+be\s+admitted\b",
    r"\bwill\s+be\s+admitted\b",
    r"\badmit\s+the\s+patient\b",
    r"\bdischarge\s+the\s+patient\b",
    r"\brequires?\s+admission\b",
    r"\bdoes\s+not\s+require\s+admission\b",
    r"\bappropriate\s+for\s+discharge\b",
    r"\bnot\s+appropriate\s+for\s+discharge\b",
    r"\bsafe\s+for\s+discharge\b",
    r"\bunsafe\s+for\s+discharge\b",
    r"\bneeds?\s+inpatient\s+care\b",
    r"\bdoes\s+not\s+need\s+inpatient\s+care\b",
    # Admission verdict via supporting verbs
    r"\bsupports?\s+(?:a\s+disposition\s+of\s+)?admission\b",
    r"\bsuggests?\s+(?:a\s+)?(?:need\s+for\s+)?admission\b",
    r"\bnecessitates?\s+(?:inpatient\s+)?admission\b",
    r"\bwarrants?\s+(?:inpatient\s+)?admission\b",
    r"\bindicates?\s+(?:the\s+need\s+for\s+)?admission\b",
    # Need for admission variants
    r"\bneed\s+for\s+(?:hospital\s+)?admission\b",
    r"\bdisposition\s+of\s+admission\b",
]

# Finance: phrases that directly reveal the churn or retain decision outcome of the customer for cc3.
CHURN_FORBIDDEN_PHRASES: list[str] = [
    r"\bwill\s+(?:not\s+)?churn\b",
    r"\blikely\s+to\s+churn\b",
    r"\bunlikely\s+to\s+churn\b",
    r"\bat\s+risk\s+of\s+churning\b",
    r"\bnot\s+at\s+risk\s+of\s+churning\b",
    r"\bwill\s+(?:not\s+)?leave\b",
    r"\bwill\s+stay\b",
    r"\bwill\s+cancel\b",
    r"\bpredicted\s+to\s+(?:churn|stay)\b",
    r"\bhigh\s+churn\s+risk\b",
    r"\blow\s+churn\s+risk\b",
    r"\bretention\s+is\s+(?:unlikely|likely)\b",
    r"\bshould\s+be\s+retained\b",
    r"\bcannot\s+be\s+retained\b",
    r"\bcustomer\s+will\s+(?:terminate|continue)\b",
]

# ---------------------------------------------------------------------------
# ValidatorConfig — one instance per domain
# ---------------------------------------------------------------------------


@dataclass
class ValidatorConfig:
    """All domain-specific settings for a single BERT validator run."""

    name: str
    bert_model_path: str  # relative path to fine-tuned model weights
    label_map: dict[str, int]  # model output label string → integer label
    forbidden_phrases: list[str]  # regex patterns that signal label leakage
    faithfulness_threshold: float = 0.8  # min DeBERTa quality score (1 − contradiction)
    confidence_threshold: float = 0.6  # min BERT validator confidence on correct class
    faithfulness_weight: float = 0.5  # weight in final quality score
    confidence_weight: float = 0.5  # weight in final quality score


# Finance (FinBench) ld1: 0 = repaid (positive sentiment), 1 = defaulted (negative sentiment).
# Verify against the fine-tuned model's config.json id2label before use.
FINANCE_VALIDATOR_CONFIG = ValidatorConfig(
    name="finbert",
    # bert_model_path=("model_weights/finbert_validator_ld1/lr1e-05_cw0_5-3_0_seed10"),
    bert_model_path=("model_weights/finbert_validator_ld1/lr5e-06_cw0_5-2_0_seed1"),
    label_map={"repaid": 0, "defaulted": 1},
    forbidden_phrases=FINANCE_FORBIDDEN_PHRASES,
    faithfulness_threshold=0.8,
    confidence_threshold=0.6,
)

# Finance (FinBench) cc3: 0 = retained (positive sentiment), 1 = churned (negative sentiment).
# Verify against the fine-tuned model's config.json id2label before use.
CC3_VALIDATOR_CONFIG = ValidatorConfig(
    name="finbert_cc3",
    bert_model_path="model_weights/finbert_validator_cc3/lr5e-06_cw1.0-1.0_seed1",
    label_map={
        "retained": 0,
        "churned": 1,
    },
    forbidden_phrases=CHURN_FORBIDDEN_PHRASES,
)

# Healthcare (ER-REASON): 0 = Discharge (positive sentiment), 1 = Admit (negative sentiment).
# Verify against the fine-tuned model's config.json id2label before use.
HEALTHCARE_VALIDATOR_CONFIG = ValidatorConfig(
    name="medbert_er_reason",
    bert_model_path="model_weights/medbert_validator_er_reason/lr1e-05_cw1_0-1_0_seed42",  # populate once the er-reason fine-tuned model is available
    label_map={
        "discharge": 0,
        "admit": 1,
    },  # populate once the er-reason fine-tuned model is available
    forbidden_phrases=HEALTHCARE_FORBIDDEN_PHRASES,
    faithfulness_threshold=0.6,
    confidence_threshold=0.0,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_step_blocks(cot_text: str) -> list[str]:
    """Return the text content between each [STEP] … [EORS] pair (4 expected)."""
    return re.findall(r"\[STEP\](.*?)\[EORS\]", cot_text, re.DOTALL)


def _extract_cot_steps(cleaned_text: str) -> str:
    """Return only the 4 reasoning steps, up to and including the last [EORS]."""
    last_eors = cleaned_text.rfind(EORS_TOKEN)
    if last_eors == -1:
        return cleaned_text
    return cleaned_text[: last_eors + len(EORS_TOKEN)]


def _extract_answer_text(cleaned_text: str) -> str:
    """Return the text between [ANSWER] and [EOA], stripped of whitespace."""
    start = cleaned_text.find(ANSWER_TOKEN)
    end = cleaned_text.find(EOA_TOKEN)
    if start == -1 or end == -1:
        return ""
    return cleaned_text[start : end + len(EOA_TOKEN)].strip()


# def _strip_cot_special_tokens(text: str) -> str:
#     for token in ["[STEP]", "[EORS]", "[ES]"]:
#         text = text.replace(token, "")
#     return " ".join(text.split())  # collapse extra whitespace


# ---------------------------------------------------------------------------
# Lazy model loading (models are loaded once and reused across calls)
# ---------------------------------------------------------------------------

_bert_tokenizer = None
_deberta_tokenizer = None
_deberta_pipe = None
_bert_validator_pipes: dict = {}  # keyed by resolved model path string


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def _get_bert_tokenizer():
    global _bert_tokenizer
    if _bert_tokenizer is None:
        logger.info("Loading BERT tokenizer (%s)...", BERT_TOKENIZER_ID)
        _bert_tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_ID)
    return _bert_tokenizer


def _get_deberta_tokenizer():
    global _deberta_tokenizer
    if _deberta_tokenizer is None:
        logger.info("Loading DeBERTa tokenizer (%s)...", DEBERTA_MODEL_ID)
        _deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_ID)
    return _deberta_tokenizer


def _get_deberta_pipe():
    global _deberta_pipe
    if _deberta_pipe is None:
        logger.info("Loading DeBERTa NLI model (%s)...", DEBERTA_MODEL_ID)
        _deberta_pipe = pipeline(
            "text-classification",
            model=DEBERTA_MODEL_ID,
            return_all_scores=True,
            device="cpu",
        )
    return _deberta_pipe


def _get_bert_validator_pipe(model_path: str):
    resolved = str(resolve_path(model_path))
    if resolved not in _bert_validator_pipes:
        logger.info("Loading BERT validator model (%s)...", resolved)

        tokenizer = AutoTokenizer.from_pretrained(resolved)
        tokenizer.truncation_side = "left"

        _bert_validator_pipes[resolved] = pipeline(
            "text-classification",
            model=resolved,
            tokenizer=tokenizer,
            return_all_scores=True,
            device="cpu",
        )
    return _bert_validator_pipes[resolved]


# ---------------------------------------------------------------------------
# Step 1 — Text cleaning (preprocessing, not a binary gate)
# ---------------------------------------------------------------------------


def clean_cot_text(cot_text: str) -> str:
    """Remove any text appearing before the first [STEP] token and after
    the [EOA] token (or the final [EORS] if [EOA] is absent).

    This strips LLM preamble (e.g. "Sure! Here is the reasoning:") and
    trailing commentary that the model occasionally appends despite
    instructions.

    Args:
        cot_text: Raw CoT string as returned by the teacher LLM.

    Returns:
        Cleaned CoT string starting at the first [STEP] and ending
        immediately after [EOA] (preferred) or the last [EORS] (fallback),
        or the original string if no anchor tokens are found.
    """
    first_step = cot_text.find(STEP_TOKEN)
    if first_step == -1:
        logger.warning(
            "clean_cot_text: could not find [STEP] token — returning raw text."
        )
        return cot_text

    last_eoa = cot_text.rfind(EOA_TOKEN)
    if last_eoa != -1:
        end = last_eoa + len(EOA_TOKEN)
    else:
        last_eors = cot_text.rfind(EORS_TOKEN)
        if last_eors == -1:
            logger.warning(
                "clean_cot_text: could not find [EOA] or [EORS] tokens — returning raw text."
            )
            return cot_text
        end = last_eors + len(EORS_TOKEN)

    return cot_text[first_step:end].strip()


# ---------------------------------------------------------------------------
# Step 2 — Structural check: exactly 4 [EORS] tokens
# ---------------------------------------------------------------------------


def check_eors_tokens(cot_text: str) -> bool:
    """Verify that exactly 4 [EORS] tokens appear in the CoT text.

    Each of the 4 reasoning steps must be terminated by exactly one
    [EORS] token. Missing or extra tokens indicate a malformed generation.

    Args:
        cot_text: Cleaned CoT string.

    Returns:
        True if exactly 4 [EORS] tokens are found, False otherwise.
    """
    count = cot_text.count(EORS_TOKEN)
    passed = count == 4
    if not passed:
        logger.warning("check_eors_tokens: found %d [EORS] tokens, expected 4.", count)
    return passed


# ---------------------------------------------------------------------------
# Step 3 — Structural check: exactly 4 [STEP] tokens
# ---------------------------------------------------------------------------


def check_step_tokens(cot_text: str) -> bool:
    """Verify that exactly 4 [STEP] tokens appear in the CoT text.

    Each reasoning step must begin with a [STEP] token. This check
    confirms the structural integrity of the generated CoT.

    Args:
        cot_text: Cleaned CoT string.

    Returns:
        True if exactly 4 [STEP] tokens are found, False otherwise.
    """
    count = cot_text.count(STEP_TOKEN)
    passed = count == 4
    if not passed:
        logger.warning("check_step_tokens: found %d [STEP] tokens, expected 4.", count)
    return passed


# ---------------------------------------------------------------------------
# Step 4 — Structural check: exactly 12 [ES] tokens in correct alternating order
# ---------------------------------------------------------------------------


def check_es_tokens(cot_text: str) -> bool:
    """Verify that exactly 8 [ES] tokens appear with 2 per step in correct order.

    Each reasoning step must contain exactly 2 [ES] sentence-boundary markers,
    producing the pattern: [STEP] s1 [ES] s2 [ES] [EORS] × 4.
    The [ANSWER]...[EOA] block must not contain any [ES] tokens.

    Args:
        cot_text: Cleaned CoT string.

    Returns:
        True if exactly 8 [ES] tokens are found and distributed 2-per-step,
        False otherwise.
    """
    total_es = cot_text.count(ES_TOKEN)
    if total_es != 8:
        logger.warning("check_es_tokens: found %d [ES] tokens, expected 8.", total_es)
        return False

    step_blocks = _extract_step_blocks(cot_text)
    if len(step_blocks) != 4:
        logger.warning(
            "check_es_tokens: expected 4 step blocks, found %d.", len(step_blocks)
        )
        return False

    for i, block in enumerate(step_blocks):
        es_in_block = block.count(ES_TOKEN)
        if es_in_block != 2:
            logger.warning(
                "check_es_tokens: step %d has %d [ES] tokens, expected 2.",
                i + 1,
                es_in_block,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Step 5 — Structural check: [ANSWER]...[EOA] block
# ---------------------------------------------------------------------------


def check_answer_block(cot_text: str) -> bool:
    """Verify the [ANSWER]...[EOA] block is present, correctly ordered, and
    within the 10-70 token length budget.

    Rules enforced:
      - Exactly 1 [ANSWER] token and 1 [EOA] token.
      - [ANSWER] appears after the last [EORS] (i.e. after the 4th step).
      - [EOA] appears after [ANSWER].
      - No non-whitespace text follows [EOA].
      - Answer content is between ANSWER_MIN_TOKENS and ANSWER_MAX_TOKENS
        (measured with the DeBERTa tokenizer, without special tokens).

    Args:
        cot_text: Cleaned CoT string.

    Returns:
        True if all answer-block constraints are satisfied, False otherwise.
    """
    answer_count = cot_text.count(ANSWER_TOKEN)
    eoa_count = cot_text.count(EOA_TOKEN)

    if answer_count != 1:
        logger.warning(
            "check_answer_block: found %d [ANSWER] tokens, expected 1.", answer_count
        )
        return False

    if eoa_count != 1:
        logger.warning(
            "check_answer_block: found %d [EOA] tokens, expected 1.", eoa_count
        )
        return False

    last_eors_pos = cot_text.rfind(EORS_TOKEN)
    answer_pos = cot_text.find(ANSWER_TOKEN)
    eoa_pos = cot_text.find(EOA_TOKEN)

    if answer_pos <= last_eors_pos:
        logger.warning(
            "check_answer_block: [ANSWER] does not appear after the last [EORS]."
        )
        return False

    if eoa_pos <= answer_pos:
        logger.warning("check_answer_block: [EOA] does not appear after [ANSWER].")
        return False

    after_eoa = cot_text[eoa_pos + len(EOA_TOKEN) :].strip()
    if after_eoa:
        logger.warning(
            "check_answer_block: unexpected text after [EOA]: %r", after_eoa[:60]
        )
        return False

    answer_text = cot_text[answer_pos + len(ANSWER_TOKEN) : eoa_pos].strip()
    tokenizer = _get_deberta_tokenizer()
    token_count = len(
        tokenizer(answer_text, truncation=False, add_special_tokens=False)["input_ids"]
    )
    if token_count < ANSWER_MIN_TOKENS or token_count > ANSWER_MAX_TOKENS:
        logger.warning(
            "check_answer_block: answer has %d tokens, expected %d–%d.",
            token_count,
            ANSWER_MIN_TOKENS,
            ANSWER_MAX_TOKENS,
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Step 6 — Sentence structure check
# ---------------------------------------------------------------------------


def check_sentence_structure(cot_text: str) -> bool:
    """Verify sentence length and balance constraints for each reasoning step.

    For each of the 4 steps the [ES] markers delimit 2 sentences. Each
    sentence is tokenised with the BERT tokenizer and must satisfy:
      - length ≥ SENTENCE_MIN_TOKENS (10)
      - length ≤ SENTENCE_MAX_TOKENS (70)
      - max token-count difference across the 2 sentences < SENTENCE_MAX_DIFF (50)

    Args:
        cot_text: Cleaned CoT string.

    Returns:
        True if all sentences in all steps satisfy the constraints.
    """
    tokenizer = _get_bert_tokenizer()
    step_blocks = _extract_step_blocks(cot_text)

    if len(step_blocks) != 4:
        logger.warning(
            "check_sentence_structure: expected 4 step blocks, found %d.",
            len(step_blocks),
        )
        return False

    for step_idx, block in enumerate(step_blocks):
        sentences = [s.strip() for s in block.split(ES_TOKEN) if s.strip()]

        if len(sentences) != 2:
            logger.warning(
                "check_sentence_structure: step %d yielded %d sentences, expected 2.",
                step_idx + 1,
                len(sentences),
            )
            return False

        token_counts: list[int] = []
        for sent_idx, sentence in enumerate(sentences):
            count = len(
                tokenizer(sentence, truncation=False, add_special_tokens=False)[
                    "input_ids"
                ]
            )
            token_counts.append(count)

            if count < SENTENCE_MIN_TOKENS:
                logger.warning(
                    "check_sentence_structure: step %d sentence %d has %d tokens (min %d).",
                    step_idx + 1,
                    sent_idx + 1,
                    count,
                    SENTENCE_MIN_TOKENS,
                )
                return False

            if count > SENTENCE_MAX_TOKENS:
                logger.warning(
                    "check_sentence_structure: step %d sentence %d has %d tokens (max %d).",
                    step_idx + 1,
                    sent_idx + 1,
                    count,
                    SENTENCE_MAX_TOKENS,
                )
                return False

        max_diff = max(token_counts) - min(token_counts)
        if max_diff >= SENTENCE_MAX_DIFF:
            logger.warning(
                "check_sentence_structure: step %d token-count spread %d >= %d.",
                step_idx + 1,
                max_diff,
                SENTENCE_MAX_DIFF,
            )
            return False

    return True


# ---------------------------------------------------------------------------
# Step 7 — Token count check for each cot profile
# ---------------------------------------------------------------------------


def check_token_count(cot_steps: str) -> bool:
    """Verify that the CoT reasoning steps fit within the BERT 512-token limit.

    Tokenizes only the reasoning steps (up to and including the last [EORS]),
    mirroring what the student model receives during training and inference.

    Args:
        cot_steps: Reasoning steps text (up to and including the last [EORS]).

    Returns:
        True if len(input_ids) ≤ 512, False otherwise.
    """
    tokenizer = _get_bert_tokenizer()
    tokens = tokenizer(
        cot_steps,
        truncation=False,
        add_special_tokens=True,
    )
    count = len(tokens["input_ids"])
    passed = count <= BERT_MAX_TOKENS
    if not passed:
        logger.warning(
            "check_token_count: CoT steps have %d tokens, maximum is %d.",
            count,
            BERT_MAX_TOKENS,
        )
    return passed


# ---------------------------------------------------------------------------
# Step 8 — Forbidden phrase / label leakage check
# ---------------------------------------------------------------------------


def check_forbidden_phrases(cot_text: str, forbidden_phrases: list[str]) -> bool:
    """Scan the CoT reasoning steps for explicit verdict or label-leakage language.

    Only the reasoning steps (up to and including the last [EORS]) are scanned.
    The [ANSWER]...[EOA] block is intentionally excluded — verdict language is
    permitted there by design.

    Args:
        cot_text: CoT reasoning steps only (caller must exclude the answer block).
        forbidden_phrases: Domain-specific list of regex patterns to reject.

    Returns:
        True if no forbidden phrases are found, False otherwise.
    """
    text_lower = cot_text.lower()
    for pattern in forbidden_phrases:
        if re.search(pattern, text_lower):
            logger.warning(
                "check_forbidden_phrases: forbidden pattern '%s' found.", pattern
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Step 9 — DeBERTa faithfulness score
# ---------------------------------------------------------------------------


def check_faithfulness(
    answer_text: str,
    original_profile: str,
    threshold: float = 0.8,
) -> tuple[bool, float]:
    """Compute a faithfulness score using DeBERTa NLI on (profile, answer).

    The original profile is the premise and the answer sentence is the
    hypothesis. The quality score is 1 - contradiction_probability.
    Checking the answer directly is more discriminative than checking the full
    CoT: a contradicting answer is a direct signal of hallucination.

    Args:
        answer_text: Content of the [ANSWER]...[EOA] block (hypothesis).
        original_profile: Raw profile text (premise).
        threshold: Minimum quality score to pass the check.

    Returns:
        A tuple (passed: bool, quality_score: float) where quality_score
        is in [0, 1].
    """
    nli_pipe = _get_deberta_pipe()

    # DeBERTa NLI expects a dict with text / text_pair
    result = nli_pipe(
        {"text": original_profile, "text_pair": answer_text},
        return_all_scores=True,
        top_k=None,
        truncation="only_first",
    )

    # result is a list of dicts: [{"label": "CONTRADICTION", "score": ...}, ...]
    scores = {item["label"].upper(): item["score"] for item in result}
    contradiction_prob = scores.get("CONTRADICTION")
    quality_score = 1.0 - contradiction_prob

    passed = quality_score >= threshold
    if not passed:
        logger.warning(
            "check_faithfulness: quality score %.4f below threshold %.4f "
            "(contradiction prob %.4f).",
            quality_score,
            threshold,
            contradiction_prob,
        )
    return passed, quality_score


# ---------------------------------------------------------------------------
# Step 10 — BERT validator classification check
# ---------------------------------------------------------------------------


def check_bert_label(
    cot_text: str,
    ground_truth_label: int,
    config: ValidatorConfig,
) -> tuple[bool, float]:
    """Verify that the BERT validator predicts the correct label from the CoT.

    The CoT should carry sufficient directional reasoning signal to support
    the correct classification without explicitly stating the answer. A
    passing CoT will lead the validator to assign confidence above
    config.confidence_threshold on the ground-truth class.

    Args:
        cot_text: Cleaned CoT string.
        ground_truth_label: Expected integer label (domain-defined).
        config: Domain-specific ValidatorConfig (provides model path,
            label_map, and confidence_threshold).

    Returns:
        A tuple (passed: bool, confidence: float) where confidence is the
        validator's predicted probability on the ground-truth class.
    """
    bert_pipe = _get_bert_validator_pipe(config.bert_model_path)
    result = bert_pipe(cot_text, return_all_scores=True, top_k=None)

    # result is a list of dicts: [{"label": "positive", "score": ...}, ...]
    scores = {item["label"].lower(): item["score"] for item in result}

    top_label_str = max(scores, key=scores.get)
    predicted_int = config.label_map.get(top_label_str)

    gt_label_str = [k for k, v in config.label_map.items() if v == ground_truth_label]
    confidence = scores.get(gt_label_str[0], 0.0) if gt_label_str else 0.0

    passed = confidence >= config.confidence_threshold
    if not passed:
        logger.warning(
            "check_bert_label [%s]: predicted %s (gt=%d), confidence on correct class %.4f.",
            config.name,
            predicted_int,
            ground_truth_label,
            confidence,
        )
    return passed, confidence


# ---------------------------------------------------------------------------
# Master validation function
# ---------------------------------------------------------------------------


def validate_cot(
    cot_text: str,
    original_profile: str,
    ground_truth_label: int,
    config: ValidatorConfig,
    verbose: bool = False,
) -> dict:
    """Run the full 10-layer CoT validation pipeline.

    Binary gates (layers 2-8) must all pass. Continuous scores
    (layers 9-10) are combined into a final weighted quality score using
    the weights defined in config.

    Args:
        cot_text: Raw CoT string as returned by the teacher LLM.
        original_profile: Original profile text used as NLI premise and
            paired with cot_text for the DeBERTa token count check.
        ground_truth_label: Expected classification label (domain-defined).
        config: Domain-specific ValidatorConfig (FINANCE_VALIDATOR_CONFIG
            or HEALTHCARE_VALIDATOR_CONFIG).
        verbose: If True, log a detailed report of all check results.

    Returns:
        A dict with the following keys:
            - approved (bool): True if all checks pass.
            - cleaned_cot (str): Full cleaned CoT text (steps + answer block).
            - cot_text (str): Reasoning steps only (up to last [EORS]).
            - answer_text (str): Content of the [ANSWER]...[EOA] block.
            - gates (dict): Pass/fail result for each binary gate.
            - faithfulness_score (float): DeBERTa quality score.
            - bert_confidence (float): BERT validator confidence on correct class.
            - quality_score (float): Final weighted continuous score.
    """
    # --- Step 1: Clean ---
    cleaned = clean_cot_text(cot_text)
    cot_steps = _extract_cot_steps(cleaned)
    answer_text = _extract_answer_text(cleaned)

    # --- Steps 2-8: Binary gates ---
    gates = {
        "eors_tokens": check_eors_tokens(cleaned),
        "step_tokens": check_step_tokens(cleaned),
        "es_tokens": check_es_tokens(cleaned),
        "answer_block": check_answer_block(cleaned),
        "sentence_structure": check_sentence_structure(cleaned),
        "token_count": check_token_count(cot_steps),
        "forbidden_phrases": check_forbidden_phrases(
            cot_steps, config.forbidden_phrases
        ),
    }

    all_gates_passed = all(gates.values())

    # --- Steps 9-10: Continuous scores (only run if gates pass) ---
    faithfulness_score = 0.0
    bert_confidence = 0.0
    faithfulness_passed = False
    bert_passed = False

    if all_gates_passed:
        faithfulness_passed, faithfulness_score = check_faithfulness(
            answer_text, original_profile, threshold=config.faithfulness_threshold
        )
        # Pass only the reasoning steps so the explicit verdict in the answer
        # block does not trivialise the BERT classification check.
        bert_passed, bert_confidence = check_bert_label(
            cot_steps, ground_truth_label, config
        )

    # --- Final quality score of the generated cot profile ---
    quality_score = (
        config.faithfulness_weight * faithfulness_score
        + config.confidence_weight * bert_confidence
    )

    approved = all_gates_passed and faithfulness_passed and bert_passed

    result = {
        "approved": approved,
        "cleaned_cot": cleaned,
        "cot_text": cot_steps,
        "answer_text": answer_text,
        "gates": gates,
        "faithfulness_score": round(faithfulness_score, 4),
        "bert_confidence": round(bert_confidence, 4),
        "quality_score": round(quality_score, 4),
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("CoT Validation Report [%s]", config.name)
        logger.info("=" * 60)
        for gate_name, gate_result in gates.items():
            status = "PASS ✓" if gate_result else "FAIL ✗"
            logger.info("  %-25s %s", gate_name, status)
        logger.info("  %-25s %.4f", "faithfulness_score", faithfulness_score)
        logger.info("  %-25s %.4f", "bert_confidence", bert_confidence)
        logger.info("  %-25s %.4f", "quality_score", quality_score)
        logger.info("  %-25s %s", "APPROVED", "YES ✓" if approved else "NO ✗")
        logger.info("=" * 60)

    return result


def validate_cot_batch(
    cot_texts: list[str],
    original_profiles: list[str],
    ground_truth_labels: list[int],
    config: ValidatorConfig,
    verbose: bool = False,
    profile_ids: list | None = None,
) -> list[dict]:
    """Run the validation pipeline for an entire batch in one call.

    Binary gates (layers 2-8) are applied per-item (cheap CPU ops). Neural
    inference (DeBERTa faithfulness + BERT validator, layers 9-10) is executed
    as a single batched forward pass over all items that passed the gates,
    which is more efficient than calling the pipelines one-by-one.

    Args:
        cot_texts: List of raw CoT strings from the teacher LLM.
        original_profiles: List of original profile texts (one per CoT).
        ground_truth_labels: List of expected labels (domain-defined).
        config: Domain-specific ValidatorConfig (FINANCE_VALIDATOR_CONFIG
            or HEALTHCARE_VALIDATOR_CONFIG).
        verbose: If True, log a one-line validation report per item.
        profile_ids: Optional list of profile IDs (one per item). When
            provided, IDs are used in log messages instead of batch indices.

    Returns:
        List of result dicts (same structure as validate_cot) in the same
        order as the input lists.
    """
    n = len(cot_texts)

    # --- Step 1: Clean all texts and extract sub-parts ---
    cleaned_texts = [clean_cot_text(t) for t in cot_texts]
    cot_steps_list = [_extract_cot_steps(c) for c in cleaned_texts]
    answer_texts = [_extract_answer_text(c) for c in cleaned_texts]

    # --- Steps 2-8: Binary gates (per-item, no model inference) ---
    gates_list = [
        {
            "eors_tokens": check_eors_tokens(c),
            "step_tokens": check_step_tokens(c),
            "es_tokens": check_es_tokens(c),
            "answer_block": check_answer_block(c),
            "sentence_structure": check_sentence_structure(c),
            "token_count": check_token_count(cot_steps_list[i]),
            "forbidden_phrases": check_forbidden_phrases(
                cot_steps_list[i], config.forbidden_phrases
            ),
        }
        for i, c in enumerate(cleaned_texts)
    ]
    all_gates_passed = [all(g.values()) for g in gates_list]

    # --- Steps 8-9: Batched neural inference ---
    faithfulness_scores = [0.0] * n
    bert_confidences = [0.0] * n
    faithfulness_passed_list = [False] * n
    bert_passed_list = [False] * n

    passing_indices = [i for i, passed in enumerate(all_gates_passed) if passed]

    print_memory_usage("before validation")

    if passing_indices:
        # DeBERTa: single forward pass over all gate-passing items
        nli_pipe = _get_deberta_pipe()
        nli_inputs = [
            {"text": original_profiles[i], "text_pair": answer_texts[i]}
            for i in passing_indices
        ]
        with torch.no_grad():
            nli_results = nli_pipe(
                nli_inputs, top_k=None, return_all_scores=True, truncation="only_first"
            )

        # BERT validator: single forward pass over all gate-passing items
        # Pass only the reasoning steps so the explicit verdict in the answer
        # block does not trivialise the classification check.
        bert_pipe = _get_bert_validator_pipe(config.bert_model_path)
        bert_inputs = [cot_steps_list[i] for i in passing_indices]
        with torch.no_grad():
            bert_results = bert_pipe(
                bert_inputs,
                top_k=None,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )

        print_memory_usage("after validation")

        for result_idx, item_idx in enumerate(passing_indices):
            label = ground_truth_labels[item_idx]

            # DeBERTa result
            nli_scores = {
                item["label"].upper(): item["score"] for item in nli_results[result_idx]
            }
            contradiction_prob = nli_scores.get("CONTRADICTION", 0.0)
            quality = 1.0 - contradiction_prob
            faithfulness_scores[item_idx] = quality
            faithfulness_passed_list[item_idx] = (
                quality >= config.faithfulness_threshold
            )
            if not faithfulness_passed_list[item_idx]:
                pid = profile_ids[item_idx] if profile_ids else item_idx
                logger.warning(
                    "check_faithfulness [ID %s]: quality score %.4f below threshold %.4f.",
                    pid,
                    quality,
                    config.faithfulness_threshold,
                )

            # BERT validator result
            fb_scores = {
                item["label"].lower(): item["score"]
                for item in bert_results[result_idx]
            }
            top_label_str = max(fb_scores, key=fb_scores.get)
            predicted_int = config.label_map.get(top_label_str)
            gt_label_str = [k for k, v in config.label_map.items() if v == label]
            confidence = fb_scores.get(gt_label_str[0], 0.0) if gt_label_str else 0.0
            bert_confidences[item_idx] = confidence
            bert_passed_list[item_idx] = confidence >= config.confidence_threshold
            if not bert_passed_list[item_idx]:
                # print(top_label_str, predicted_int, gt_label_str, confidence)
                pid = profile_ids[item_idx] if profile_ids else item_idx
                logger.warning(
                    "check_bert_label [%s, ID %s]: predicted %s (gt=%d), confidence %.4f.",
                    config.name,
                    pid,
                    predicted_int,
                    label,
                    confidence,
                )

    # --- Assemble results ---
    results = []
    for i in range(n):
        approved = (
            all_gates_passed[i] and faithfulness_passed_list[i] and bert_passed_list[i]
        )
        quality_score = (
            config.faithfulness_weight * faithfulness_scores[i]
            + config.confidence_weight * bert_confidences[i]
        )
        result = {
            "approved": approved,
            "cleaned_cot": cleaned_texts[i],
            "cot_text": cot_steps_list[i],
            "answer_text": answer_texts[i],
            "gates": gates_list[i],
            "faithfulness_score": round(faithfulness_scores[i], 4),
            "bert_confidence": round(bert_confidences[i], 4),
            "quality_score": round(quality_score, 4),
        }

        if verbose:
            pid = profile_ids[i] if profile_ids else i
            gate_summary = " | ".join(
                f"{name}={'✓' if result else '✗'}"
                for name, result in gates_list[i].items()
            )
            logger.info(
                "[%s] ID %s — %s | faith=%.4f bert=%.4f quality=%.4f | %s",
                config.name,
                pid,
                gate_summary,
                faithfulness_scores[i],
                bert_confidences[i],
                quality_score,
                "APPROVED ✓" if approved else "FAILED ✗",
            )

        results.append(result)

    return results
