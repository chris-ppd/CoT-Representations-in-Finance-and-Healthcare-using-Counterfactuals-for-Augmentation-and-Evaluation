"""
Hook Integration Tests
======================
Six tests verifying the forward-hook hidden state extraction pipeline for Qwen3-4B.

Tests 1, 2, 5 — require a real CUDA GPU + Qwen3-4B (skipped when unavailable).
Tests 3, 4, 6 — operate on saved artefacts or pure Python; no GPU required.

Run all:         pytest tests/test_hook_integration.py -v
Skip GPU tests:  pytest tests/test_hook_integration.py -v -k "not (1 or 2 or 5)"

Tokenisation note (Tests 3 & 4)
--------------------------------
cot_text ends with "[EORS]" and answer_text starts with "[ANSWER]".  Encoding the
concatenation as a single string causes BPE to merge the closing "]" and opening "["
into the token "][" (ID 1457), destroying the 4th EORS sequence.  Encoding the two
fields separately and concatenating the ID lists avoids this boundary artefact and
reproduces the token sequence as it was during generation.
"""

import json
import re
from pathlib import Path

import pytest
import torch

from src.cot.hidden_state_extractor import HookBuffer, _find_extraction_positions

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HIDDEN_DIM = 2560
MODEL_NAME = "Qwen/Qwen3-4B"
ID2LABEL = {0: "repaid", 1: "defaulted"}

INDEX_PATH = (
    PROJECT_ROOT / "data/hidden_states/finbench/ld1/test/hidden_states_index.jsonl"
)
COT_JSONL_PATH = (
    PROJECT_ROOT / "data/processed/finbench/ld1/ld1_cot_test_Qwen3-4b-test-logger.jsonl"
)
COT_STATES_PATH = PROJECT_ROOT / "data/hidden_states/finbench/ld1/test/cot_states.pt"
ANSWER_STATES_PATH = (
    PROJECT_ROOT / "data/hidden_states/finbench/ld1/test/answer_states.pt"
)
PROMPT_PATH = PROJECT_ROOT / "prompts/finbench_ld1_prompt.txt"


# ---------------------------------------------------------------------------
# Checking hook for Test 2
# ---------------------------------------------------------------------------


class _VerifyingHookBuffer(HookBuffer):
    """HookBuffer subclass that validates captured == output[0][:, 0, :] per step."""

    def __init__(self) -> None:
        super().__init__()
        self.extraction_mismatch = False

    def _hook_fn(self, _module, _input, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            return  # prefill pass — skip

        captured = hidden[:, 0, :].detach().cpu().float()
        expected = (
            output[0][:, 0, :].detach().cpu().float()
            if isinstance(output, tuple)
            else hidden[:, 0, :].detach().cpu().float()
        )
        if not torch.allclose(captured, expected, atol=1e-5):
            self.extraction_mismatch = True

        self.step_hidden_states.append(captured)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tokenizer():
    """Qwen3-4B tokenizer — no GPU required."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="session")
def model_and_tokenizer(tokenizer):
    """Full Qwen3-4B model in 8-bit + tokenizer, loaded once per session."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — skipping model-dependent integration test")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def index_entries() -> list[dict]:
    with open(INDEX_PATH) as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="session")
def cot_data() -> dict[str, dict]:
    data: dict[str, dict] = {}
    with open(COT_JSONL_PATH) as f:
        for line in f:
            d = json.loads(line)
            data[str(d["id"])] = d
    return data


@pytest.fixture(scope="session")
def eors_eoa_sequences(tokenizer):
    """All BPE variants for [EORS] and [EOA] markers."""
    eors = tuple(
        {
            tuple(tokenizer.encode("[EORS]", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EORS]", add_special_tokens=False)),
            tuple(tokenizer.encode("[EORS]\n", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EORS]\n", add_special_tokens=False)),
        }
    )
    eoa = tuple(
        {
            tuple(tokenizer.encode("[EOA]", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EOA]", add_special_tokens=False)),
            tuple(tokenizer.encode("[EOA]\n", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EOA]\n", add_special_tokens=False)),
        }
    )
    return eors, eoa


# ---------------------------------------------------------------------------
# Test 1 — Hook fires only during decoding
# ---------------------------------------------------------------------------


def test_1_hook_fires_only_during_decoding(model_and_tokenizer):
    """
    A prefill-only forward pass (model(**inputs)) must capture 0 states.
    model.generate() must capture exactly n_generated states (one per token).
    """
    model, tokenizer = model_and_tokenizer
    inputs = tokenizer(
        ["The capital of France is"], return_tensors="pt", padding=True
    ).to(model.device)
    input_length = inputs.input_ids.shape[1]

    buf = HookBuffer()

    # --- prefill only ---
    buf.attach(model)
    with torch.no_grad():
        _ = model(**inputs)
    buf.detach()

    assert len(buf.step_hidden_states) == 0, (
        f"Prefill-only forward pass captured {len(buf.step_hidden_states)} "
        "hidden states; expected 0."
    )

    # --- autoregressive generation ---
    buf.reset()
    buf.attach(model)
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
    finally:
        buf.detach()

    # The last token in output is either EOS or the max_new_tokens boundary token —
    # HuggingFace appends it without a forward pass, so hook fires n_generated - 1 times
    n_generated = output.shape[1] - input_length
    assert len(buf.step_hidden_states) == n_generated - 1, (
        f"Expected {n_generated - 1} hook captures after model.generate(), "
        f"got {len(buf.step_hidden_states)}."
    )


# ---------------------------------------------------------------------------
# Test 2 — Hook captures the correct layer output
# ---------------------------------------------------------------------------


def test_2_hook_captures_correct_layer_output(model_and_tokenizer):
    """
    At every decoding step, hidden[:, 0, :] must equal output[0][:, 0, :].
    Shape (batch_size, hidden_dim) must be consistent across all steps, even
    when different profiles generate different numbers of tokens.
    """
    model, tokenizer = model_and_tokenizer
    texts = [
        "The first client has a stable income.",
        "This borrower has a very high debt-to-income ratio and a recent default.",
        "Applicant has an excellent credit score and no outstanding debt.",
        "Customer recently lost their job and has missed three consecutive payments.",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    batch_size = len(texts)

    buf = _VerifyingHookBuffer()
    buf.attach(model)
    try:
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
    finally:
        buf.detach()

    assert not buf.extraction_mismatch, (
        "Hook stored a tensor that differed from output[0][:, 0, :] by more "
        "than atol=1e-5 at one or more decoding steps."
    )
    assert len(buf.step_hidden_states) > 0, "No decoding steps were captured."
    for step_idx, step_tensor in enumerate(buf.step_hidden_states):
        assert step_tensor.shape == (batch_size, HIDDEN_DIM), (
            f"Step {step_idx}: expected shape ({batch_size}, {HIDDEN_DIM}), "
            f"got {tuple(step_tensor.shape)}."
        )


# ---------------------------------------------------------------------------
# Test 3 — Position alignment between master index and CoT text
# ---------------------------------------------------------------------------


def test_3_position_alignment(index_entries, cot_data, eors_eoa_sequences, tokenizer):
    """
    Re-tokenising saved cot_text + answer_text and re-running
    _find_extraction_positions must reproduce the eors_token_positions and
    eoa_token_position stored in the master index for every indexed profile.
    """
    eors_seqs, eoa_seqs = eors_eoa_sequences
    failures: list[str] = []

    for entry in index_entries:
        pid = entry["profile_id"]
        saved_eors: list[int] = entry["eors_token_positions"]
        saved_eoa: int = entry["eoa_token_position"]

        if pid not in cot_data:
            failures.append(f"profile_id={pid}: not found in CoT jsonl")
            continue

        record = cot_data[pid]
        # Encode separately to avoid BPE merging "]" + "[" at the boundary.
        token_ids = tokenizer.encode(
            record["cot_text"], add_special_tokens=False
        ) + tokenizer.encode(record["answer_text"], add_special_tokens=False)

        eors_pos, eoa_pos = _find_extraction_positions(token_ids, eors_seqs, eoa_seqs)

        if eors_pos != saved_eors:
            failures.append(
                f"profile_id={pid}: [EORS] mismatch — "
                f"index={saved_eors}, re-extracted={eors_pos}"
            )
        if len(eoa_pos) != 1 or eoa_pos[0] != saved_eoa:
            failures.append(
                f"profile_id={pid}: [EOA] mismatch — "
                f"index={saved_eoa}, re-extracted={eoa_pos}"
            )

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Test 4 — _find_extraction_positions unit logic
# ---------------------------------------------------------------------------


def test_4_find_extraction_positions_logic(eors_eoa_sequences, tokenizer, cot_data):
    """
    Sub-test A: sequence with exactly 4 EORS + 1 EOA → last-token indices correct.
    Sub-test B: partial matches (no full sequence) → empty results.
    Sub-test C: empty input → empty results.
    Sub-test D: real re-tokenised profile text → exactly 4 EORS + 1 EOA hits.
    """
    # Synthetic token IDs that won't occur in real Qwen3-4B output
    EORS_SEQ = (100_001, 200_002, 300_003)
    EOA_SEQ = (400_004, 500_005)
    eors_seqs_syn = (EORS_SEQ,)
    eoa_seqs_syn = (EOA_SEQ,)

    # ---- Sub-test A: exact positions ----
    # Layout: prefix(2) | EORS | mid1(2) | EORS | mid2(1) | EORS | mid3(3) | EORS | mid4(1) | EOA | suffix(1)
    prefix = [1, 2]
    mids = [[3, 4], [5], [6, 7, 8], [9]]
    suffix = [10]
    real_ids: list[int] = list(prefix)
    for mid in mids:
        real_ids += list(EORS_SEQ) + mid
    real_ids += list(EOA_SEQ) + suffix

    # Compute expected last-token positions
    expected_eors: list[int] = []
    pos = len(prefix)
    for mid in mids:
        expected_eors.append(pos + len(EORS_SEQ) - 1)
        pos += len(EORS_SEQ) + len(mid)
    expected_eoa = pos + len(EOA_SEQ) - 1

    eors_pos_a, eoa_pos_a = _find_extraction_positions(
        real_ids, eors_seqs_syn, eoa_seqs_syn
    )
    assert eors_pos_a == expected_eors, (
        f"Sub-test A [EORS]: expected {expected_eors}, got {eors_pos_a}"
    )
    assert eoa_pos_a == [expected_eoa], (
        f"Sub-test A [EOA]: expected [{expected_eoa}], got {eoa_pos_a}"
    )

    # ---- Sub-test B: partial EORS sequence only — no full match ----
    partial_ids = [100_001, 200_002, 999] * 5  # prefix of EORS, wrong 3rd token
    eors_pos_b, eoa_pos_b = _find_extraction_positions(
        partial_ids, eors_seqs_syn, eoa_seqs_syn
    )
    assert eors_pos_b == [], f"Sub-test B [EORS]: expected [], got {eors_pos_b}"
    assert eoa_pos_b == [], f"Sub-test B [EOA]: expected [], got {eoa_pos_b}"

    # ---- Sub-test C: empty input ----
    eors_pos_c, eoa_pos_c = _find_extraction_positions([], eors_seqs_syn, eoa_seqs_syn)
    assert eors_pos_c == [], f"Sub-test C [EORS]: expected [], got {eors_pos_c}"
    assert eoa_pos_c == [], f"Sub-test C [EOA]: expected [], got {eoa_pos_c}"

    # ---- Sub-test D: real re-tokenised profile (profile_id=1) ----
    eors_seqs_real, eoa_seqs_real = eors_eoa_sequences
    record = cot_data["1"]
    # Encode separately — same reason as Test 3.
    token_ids = tokenizer.encode(
        record["cot_text"], add_special_tokens=False
    ) + tokenizer.encode(record["answer_text"], add_special_tokens=False)
    eors_pos_d, eoa_pos_d = _find_extraction_positions(
        token_ids, eors_seqs_real, eoa_seqs_real
    )
    assert len(eors_pos_d) == 4, (
        f"Sub-test D [EORS]: expected 4 hits, got {len(eors_pos_d)}: {eors_pos_d}"
    )
    assert len(eoa_pos_d) == 1, (
        f"Sub-test D [EOA]: expected 1 hit, got {len(eoa_pos_d)}: {eoa_pos_d}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Hidden states encode profile-specific content
# ---------------------------------------------------------------------------


def test_5_hidden_states_encode_profile_content(
    model_and_tokenizer, cot_data, eors_eoa_sequences
):
    """
    Profile A and Profile B (A with one numeric field changed, same label) must be
    closer in hidden-state space than A and Profile C (opposite label).
    dist(A, B) < dist(A, C) measured as mean per-step L2 distance across 4 [EORS] states.
    """
    from src.cot.cot_generator import _format_messages, _slice_generated
    from src.cot.hidden_state_extractor import extract_eors_eoa_states

    model, tokenizer = model_and_tokenizer
    eors_seqs, eoa_seqs = eors_eoa_sequences
    eos_ids = {
        tid
        for tid in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if tid is not None
    }

    if not PROMPT_PATH.exists():
        pytest.skip(f"Prompt file not found: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text()

    # Profile A: label 0 (repaid)
    record_a = cot_data["1"]
    text_a, label_a = record_a["text"], record_a["label"]

    # Profile B: Profile A with the first dollar amount incremented by $1,000 — same label.
    # re.sub finds the first "$NNN" or "$N,NNN" pattern and bumps it so the text is
    # semantically similar but not byte-identical, exercising profile-specific encoding.
    text_b = re.sub(
        r"\$(\d[\d,]*)",
        lambda m: f"${int(m.group(1).replace(',', '')) + 1000:,}",
        text_a,
        count=1,
    )
    label_b = label_a

    # Profile C: label 1 (defaulted) — opposite label to A
    record_c = cot_data["2"]
    text_c, label_c = record_c["text"], record_c["label"]

    # Profile D: independent label 0 profile — same label as A but different person
    record_d = cot_data["3"]
    text_d, label_d = record_d["text"], record_d["label"]

    profiles = [
        (text_a, label_a),
        (text_b, label_b),
        (text_c, label_c),
        (text_d, label_d),
    ]

    # Build chat-formatted texts (mirrors generate_cot_batch_with_hooks internals)
    all_texts = [
        tokenizer.apply_chat_template(
            _format_messages(text, prompt, label, ID2LABEL),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=None,
        )
        for text, label in profiles
    ]
    inputs = tokenizer(
        all_texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    buf = HookBuffer()
    buf.reset()
    buf.attach(model)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
    finally:
        buf.detach()

    cot_states_list: list[torch.Tensor] = []
    for i in range(len(profiles)):
        gen_ids = _slice_generated(output_ids, inputs, i)
        cot_states, _, _, _, err = extract_eors_eoa_states(
            buf,
            profile_idx=i,
            generated_ids=gen_ids,
            eors_sequences=eors_seqs,
            eoa_sequences=eoa_seqs,
            eos_token_ids=eos_ids,
            hidden_dim=HIDDEN_DIM,
        )
        if cot_states is None:
            pytest.skip(
                f"Profile index {i} extraction failed: {err}. "
                "Model did not produce a valid 4-step CoT on this run."
            )
        cot_states_list.append(cot_states)

    states_a, states_b, states_c, states_d = cot_states_list

    # Mean per-step L2 distance across 4 [EORS] rows
    dist_ab = (states_a - states_b).norm(dim=-1).mean().item()
    dist_ac = (states_a - states_c).norm(dim=-1).mean().item()
    dist_ad = (states_a - states_d).norm(dim=-1).mean().item()

    assert dist_ab < dist_ac, (
        f"Expected dist(A, B) < dist(A, C): same-label pair should be closer. "
        f"dist_AB={dist_ab:.4f}, dist_AC={dist_ac:.4f}."
    )

    assert dist_ad < dist_ac, (
        f"Expected dist(A, D) < dist(A, C): independent same-label profile should be "
        f"closer than opposite-label. dist_AD={dist_ad:.4f}, dist_AC={dist_ac:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Tensor shape and content sanity
# ---------------------------------------------------------------------------


def test_6_tensor_shape_and_content_sanity(index_entries):
    """
    Consolidated .pt tensors have the expected shape.
    Every row referenced by the index is non-zero.
    Every row absent from the index is all zeros (discarded profiles).
    """
    cot_states = torch.load(COT_STATES_PATH, weights_only=True)
    answer_states = torch.load(ANSWER_STATES_PATH, weights_only=True)

    n_profiles = cot_states.shape[0]
    assert cot_states.shape == (n_profiles, 4, HIDDEN_DIM), (
        f"cot_states: expected (n, 4, {HIDDEN_DIM}), got {tuple(cot_states.shape)}"
    )
    assert answer_states.shape == (n_profiles, HIDDEN_DIM), (
        f"answer_states: expected (n, {HIDDEN_DIM}), got {tuple(answer_states.shape)}"
    )

    indexed_rows: set[int] = set()
    for entry in index_entries:
        row = entry["profile_row_idx"]
        indexed_rows.add(row)

        assert not cot_states[row].eq(0).all(), (
            f"profile_id={entry['profile_id']} (row={row}): "
            "cot_states row is unexpectedly all zeros."
        )
        assert not answer_states[row].eq(0).all(), (
            f"profile_id={entry['profile_id']} (row={row}): "
            "answer_states row is unexpectedly all zeros."
        )

    unindexed_rows = set(range(n_profiles)) - indexed_rows
    for row in unindexed_rows:
        assert cot_states[row].eq(0).all(), (
            f"Row {row} is not in the index but cot_states contains non-zero values."
        )
        assert answer_states[row].eq(0).all(), (
            f"Row {row} is not in the index but answer_states contains non-zero values."
        )
