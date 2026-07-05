"""
Batched Chain-of-Thought generation for client profiles using Qwen3-4B (teacher LLM).

Loads a YAML config that specifies the dataset (FinBench or ER-REASON), the active
sub-dataset (e.g. ld1, ld2 for FinBench), and paths to the processed profiles, prompt
template, and CoT output file. Generated CoT profiles are appended to a .jsonl output
file alongside the original profile id, text, and label.

Supports checkpoint resumption: profiles whose IDs already appear in the output file
are skipped, allowing interrupted runs to continue without reprocessing.

Hidden state extraction (optional)
-----------------------------------
When ``hidden_states.enabled: true`` is set in the YAML config (or via the
``--extract-hidden-states`` CLI flag), the generator attaches a forward hook to
the last transformer layer of Qwen3-4B and captures the hidden state at every
decoding step for the full batch.  After generation, per-profile [EORS] and
[EOA] positions are located and the 5 corresponding hidden state vectors are
extracted (4 × [EORS] + 1 × [EOA]), sanity-checked, and saved.  CoT validation
then determines whether they are promoted from temp/ to final/.

Key differences from standard batched generation when extraction is enabled:
  - [STEP], [EORS], [ES], [ANSWER], [EOA] are NOT registered as special tokens
    — they stay as multi-token sequences so the model generates them correctly
    from few-shot examples.  Adding them to the vocabulary caused Qwen3 to
    substitute them with <tool_call>/<tool_response> by triggering its tool-call
    mode.
  - Decoding uses skip_special_tokens=False to avoid stripping any markers.
  - [EORS] and [EOA] positions are found via sequence matching on their BPE
    encodings (both bare and space-prefixed forms); the hidden state at the last
    token of each match is captured.
  - Failed profiles (extraction or validation) are collected per-round and
    retried as a new batch, mirroring the existing retry structure.

Left-padding note
-----------------
The tokenizer always uses padding_side="left".  With left-padding, every
sequence in a batch is padded to the same length (max_input_len) on the LEFT.
model.generate() therefore returns outputs where the first max_input_len
tokens are always the (left-padded) input, and generated tokens start at index
max_input_len for every profile.  The correct slice is `output[max_input_len:]`
(using `inputs.input_ids[i].shape[0]`), NOT `attention_mask.sum()` — that
gives the unpadded prompt length and would include trailing prompt tokens in
the generated slice.
"""

import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.cot.cot_validator import (
    CC3_VALIDATOR_CONFIG,
    FINANCE_VALIDATOR_CONFIG,
    HEALTHCARE_VALIDATOR_CONFIG,
    ValidatorConfig,
    validate_cot_batch,
)
from src.cot.hidden_state_extractor import (
    HookBuffer,
    delete_per_profile_files,
    discard_from_temp,
    extract_eors_eoa_states,
    promote_to_final,
    save_hidden_states_temp,
    update_master_index,
)
from utils.memory_printer import print_memory_usage

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_SUB_DATASET_VALIDATOR: dict[str, ValidatorConfig] = {
    "ld1": FINANCE_VALIDATOR_CONFIG,
    "cc3": CC3_VALIDATOR_CONFIG,
    "er-reason": HEALTHCARE_VALIDATOR_CONFIG,
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def load_model(config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the teacher model and tokenizer from HuggingFace.

    [ES], [STEP], [EORS] are intentionally NOT added to the tokenizer
    vocabulary — they remain multi-token sequences.  Adding them as special
    tokens causes Qwen3 to substitute them with <tool_call>/<tool_response>
    during generation because novel token IDs (with unlearned embeddings)
    break Qwen3's few-shot pattern matching and trigger its tool-call mode.
    Hidden state extraction uses sequence matching on the multi-token encoding
    of "[ES]" instead of a single token ID lookup.
    """
    teacher_cfg = config["teacher_model"]
    model_name = teacher_cfg["name"]
    quantization = teacher_cfg.get("quantization")

    quant_cfg = None
    if quantization == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def load_prompt(prompt_path: str) -> str:
    with open(resolve_path(prompt_path), "r") as f:
        return f.read()


def load_jsonl_batches(filepath: str, batch_size: int):
    """Lazy batch loader — only batch_size records live in memory at a time."""
    batch = []
    with open(resolve_path(filepath), "r") as f:
        for line in f:
            batch.append(json.loads(line.strip()))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def load_processed_ids(output_path: Path) -> set:
    """Read already-processed IDs from the output file to support resumption."""
    processed_ids = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                data_point = json.loads(line.strip())
                processed_ids.add(data_point["id"])
    return processed_ids


def _count_jsonl_lines(filepath: str) -> int:
    with open(resolve_path(filepath), "r") as f:
        return sum(1 for _ in f)


def _count_split_index_entries(index_path: Path, cot_states_path: Path) -> int:
    """Count index entries that belong to the current split's consolidated tensor."""
    if not index_path.exists():
        return 0
    count = 0
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("cot_states_path") == str(cot_states_path):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_messages(
    profile_text: str,
    prompt: str,
    label: int,
    id2label: dict,
) -> list[dict]:
    label_str = id2label[label]
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"#### Patient profile ####\n{profile_text}\n\n"
                f"#### Ground truth label: {label_str} ####\n\n"
                "#### CoT Reasoning Steps and final answer derived from the reasoning steps ####"
            ),
        },
    ]


def _decode_preserving_special_tokens(
    token_ids: list[int],
    tokenizer: AutoTokenizer,
) -> str:
    """Decode while keeping [STEP], [EORS], [ES] but stripping EOS/BOS/PAD.

    Uses skip_special_tokens=False (preserves additional_special_tokens such
    as [ES]) and removes only the HuggingFace core token IDs by hand.
    """
    to_skip = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id}
    to_skip.discard(None)
    filtered = [tid for tid in token_ids if tid not in to_skip]
    return tokenizer.decode(filtered, skip_special_tokens=False)


def _slice_generated(
    output: torch.Tensor,
    inputs,
    profile_idx: int,
) -> list[int]:
    """Return the generated token IDs for one profile in a left-padded batch.

    With left-padding, every profile in the batch is padded to the same
    max_input_len on the left.  model.generate() returns:
        output[i] = [PAD..., prompt_tokens, gen_tok_1, ..., gen_tok_n]
                     |-------- max_input_len --------|
    so the generated tokens always start at max_input_len regardless of the
    individual profile length.  inputs.input_ids[i].shape[0] == max_input_len
    for every i, making it the correct and consistent split point.
    """
    max_input_len = inputs.input_ids[profile_idx].shape[0]
    return output[profile_idx][max_input_len:].tolist()


# ---------------------------------------------------------------------------
# Standard batched generation (no hook, extraction disabled)
# ---------------------------------------------------------------------------


def generate_cot_batch(
    batch_texts: list[str],
    batch_labels: list[int],
    id2label: dict[int, str],
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> list[str]:
    all_texts = []
    for profile_text, label in zip(batch_texts, batch_labels):
        messages = _format_messages(profile_text, prompt, label, id2label)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=None,  # explicitly disable tool calling
        )
        all_texts.append(text)

    inputs = tokenizer(
        all_texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    print_memory_usage("before batch generation")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    print_memory_usage("after batch generation")

    cot_texts = [
        _decode_preserving_special_tokens(
            _slice_generated(output_ids, inputs, i), tokenizer
        )
        for i in range(len(batch_texts))
    ]
    del output_ids, inputs
    torch.cuda.empty_cache()
    return cot_texts


def _generate_and_validate(
    data_points: list[dict],
    id2label: dict[int, str],
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    validator_config: ValidatorConfig,
    skip_validation: bool = False,
) -> list[tuple[dict, dict]]:
    batch_texts = [dp["text"] for dp in data_points]
    batch_labels = [dp["label"] for dp in data_points]
    cot_texts = generate_cot_batch(
        batch_texts, batch_labels, id2label, prompt, model, tokenizer, max_new_tokens
    )
    if skip_validation:
        validations = [
            {
                "approved": True,
                "cot_text": cot_text,
                "answer_text": None,
                "quality_score": None,
                "faithfulness_score": None,
                "bert_confidence": None,
            }
            for cot_text in cot_texts
        ]
        return list(zip(data_points, validations))
    validations = validate_cot_batch(
        cot_texts=cot_texts,
        original_profiles=[dp["text"] for dp in data_points],
        ground_truth_labels=[dp["label"] for dp in data_points],
        config=validator_config,
        verbose=True,
        profile_ids=[dp["id"] for dp in data_points],
    )
    return list(zip(data_points, validations))


# ---------------------------------------------------------------------------
# Batched generation WITH hook (extraction enabled)
# ---------------------------------------------------------------------------


def generate_cot_batch_with_hooks(
    batch_texts: list[str],
    batch_labels: list[int],
    id2label: dict[int, str],
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    hook_buffer: HookBuffer,
) -> tuple[list[str], list[list[int]]]:
    """Batched generation that simultaneously captures per-step hidden states.

    The hook fires once per decoding step, storing a (batch_size, hidden_dim)
    tensor for every token generated across all profiles.  After generation,
    each profile's [EORS] and [EOA] positions are resolved from its raw generated IDs and
    matched against the corresponding hook buffer entries.

    Returns:
        cot_texts         — decoded CoT strings, one per profile.
        all_generated_ids — raw generated token IDs per profile (includes EOS);
                            used for per-profile [EORS] and [EOA] position lookup.
    """
    all_texts = []
    for profile_text, label in zip(batch_texts, batch_labels):
        messages = _format_messages(profile_text, prompt, label, id2label)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=None,  # ← explicitly disable tool calling
        )
        all_texts.append(text)

    inputs = tokenizer(
        all_texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    hook_buffer.reset()
    hook_buffer.attach(model)
    print_memory_usage("before batch generation (with hooks)")
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        # Always detach — even on exception — to prevent GPU memory leaks.
        hook_buffer.detach()
    print_memory_usage("after batch generation (with hooks)")

    cot_texts: list[str] = []
    all_generated_ids: list[list[int]] = []

    for i in range(len(batch_texts)):
        gen_ids = _slice_generated(output_ids, inputs, i)
        cot_texts.append(_decode_preserving_special_tokens(gen_ids, tokenizer))
        all_generated_ids.append(gen_ids)

    del output_ids, inputs
    torch.cuda.empty_cache()
    return cot_texts, all_generated_ids


# ---------------------------------------------------------------------------
# Extraction-enabled main loop
# ---------------------------------------------------------------------------


def _run_extraction_loop(
    input_path: str,
    output_path: Path,
    prompt: str,
    id2label: dict[int, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    batch_size: int,
    num_batches: int | None,
    max_retries: int,
    hs_cfg: dict,
    path_suff: str | None,
    validator_config: ValidatorConfig,
    dataset_name: str = "",
    skip_validation: bool = False,
) -> None:
    """Batched CoT generation with per-profile hidden state extraction.

    Retry structure mirrors the standard batch loop: failed profiles are
    collected each round and retried as a new batch, up to max_retries times.
    A profile fails if hidden state extraction fails OR CoT validation fails.

    After each batch, approved profiles are written into two consolidated
    tensors (cot_states.pt and answer_states.pt) stored under
    {hs_output_dir}/{split}/.  Per-profile .pt / _answer.pt / _meta.json
    files are deleted immediately after being merged into those tensors.
    The master index at final/hidden_states_index.jsonl is the single source
    of truth: each entry records the consolidated tensor paths and the row
    index within them so the student training pipeline can do O(1) lookups.
    Note: final_dir and split_dir should be the same.
    """
    hs_output_dir = resolve_path(hs_cfg["output_dir"])
    temp_dir = hs_output_dir / "temp"
    split = path_suff
    _prefix = f"{dataset_name}_{split}" if dataset_name else split

    final_dir = hs_output_dir / split
    index_path = final_dir / f"{_prefix}_hidden_states_index.jsonl"
    hidden_dim = hs_cfg.get("hidden_dim", 2560)

    split_dir = hs_output_dir / split
    cot_states_path = split_dir / f"{_prefix}_cot_states.pt"
    answer_states_path = split_dir / f"{_prefix}_answer_states.pt"

    temp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale temp files left by a previous interrupted run.
    stale = list(temp_dir.glob("*"))
    if stale:
        logger.info(
            f"Cleaning {len(stale)} stale temp file(s) from a previous interrupted run: "
            f"{[p.name for p in stale]}"
        )
        for p in stale:
            p.unlink()

    # Pre-allocate consolidated tensors sized to the full split.
    # If a previous run already produced partial tensors, reload them so
    # that resumption continues writing at the correct row offset.
    n_profiles = _count_jsonl_lines(input_path)
    next_row_idx = _count_split_index_entries(index_path, cot_states_path)
    if cot_states_path.exists() and answer_states_path.exists():
        cot_states_tensor = torch.load(cot_states_path, weights_only=True)
        answer_states_tensor = torch.load(answer_states_path, weights_only=True)
        if cot_states_tensor.shape != (n_profiles, 4, hidden_dim):
            cot_states_tensor = torch.zeros(n_profiles, 4, hidden_dim)
            answer_states_tensor = torch.zeros(n_profiles, hidden_dim)
            next_row_idx = 0
    else:
        cot_states_tensor = torch.zeros(n_profiles, 4, hidden_dim)
        answer_states_tensor = torch.zeros(n_profiles, hidden_dim)

    logger.info(
        f"Split '{split}': {n_profiles} total profiles, "
        f"{next_row_idx} already consolidated."
    )

    # BPE tokenization is context-dependent: the same marker string encodes to
    # different token IDs depending on whether it is space-prefixed and whether
    # the trailing character (e.g. '\n') merges into the closing ']' token.
    # In practice, [EORS] always appears as " [EORS]" (space-prefixed) and the
    # closing ']' is either token 60 (standalone) or 921 (merged with '\n'),
    # depending on what the model generates next.  We collect all four variants
    # (bare/spaced × no-newline/newline) so _find_extraction_positions is robust
    # to any trailing whitespace the model appends after each marker.
    eors_sequences = tuple(
        {
            tuple(tokenizer.encode("[EORS]", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EORS]", add_special_tokens=False)),
            tuple(tokenizer.encode("[EORS]\n", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EORS]\n", add_special_tokens=False)),
        }
    )
    assert all(eors_sequences), "[EORS] encoded to empty sequence — check tokenizer."

    eoa_sequences = tuple(
        {
            tuple(tokenizer.encode("[EOA]", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EOA]", add_special_tokens=False)),
            tuple(tokenizer.encode("[EOA]\n", add_special_tokens=False)),
            tuple(tokenizer.encode(" [EOA]\n", add_special_tokens=False)),
        }
    )
    assert all(eoa_sequences), "[EOA] encoded to empty sequence — check tokenizer."

    eos_token_ids = {
        tid
        for tid in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if tid is not None
    }
    print(
        f"[EORS] sequences: {[list(s) for s in eors_sequences]}  |  "
        f"[EOA] sequences: {[list(s) for s in eoa_sequences]}  |  "
        f"EOS/PAD ids: {eos_token_ids}"
    )

    processed_ids = load_processed_ids(output_path)
    hook_buffer = HookBuffer()
    total = 0
    failed_points: list[dict] = []

    def _process_batch(data_points: list[dict], f_out) -> list[dict]:
        """Run one batch through generation → extraction → validation.

        Returns the sub-list of data points that still need retrying.
        """
        nonlocal total, next_row_idx
        batch_texts = [dp["text"] for dp in data_points]
        batch_labels = [dp["label"] for dp in data_points]
        cot_texts, all_generated_ids = generate_cot_batch_with_hooks(
            batch_texts,
            batch_labels,
            id2label,
            prompt,
            model,
            tokenizer,
            max_new_tokens,
            hook_buffer,
        )

        still_failing: list[dict] = []

        # ---- Phase 1: extraction (sequential, no model inference) ----
        # Each tuple: (data_point, cot_text, eors_pos, eoa_pos)
        extracted: list[tuple[dict, str, list, int]] = []
        for b, data_point in enumerate(data_points):
            pid = data_point["id"]
            cot_states, answer_state, eors_pos, eoa_pos, hs_err = (
                extract_eors_eoa_states(
                    hook_buffer,
                    profile_idx=b,
                    generated_ids=all_generated_ids[b],
                    eors_sequences=eors_sequences,
                    eoa_sequences=eoa_sequences,
                    eos_token_ids=eos_token_ids,
                    hidden_dim=hidden_dim,
                )
            )
            if cot_states is None:
                logger.info(f"ID {pid}: extraction failed — {hs_err}")
                still_failing.append(data_point)
                continue
            save_hidden_states_temp(
                pid, cot_states, answer_state, eors_pos, eoa_pos, temp_dir
            )
            extracted.append((data_point, cot_texts[b], eors_pos, eoa_pos))

        if not extracted:
            return still_failing

        # ---- Phase 2: batched validation (one DeBERTa + BERT forward pass) ----
        if skip_validation:
            validations = [
                {
                    "approved": True,
                    "cot_text": cot_text,
                    "answer_text": None,
                    "quality_score": None,
                    "faithfulness_score": None,
                    "bert_confidence": None,
                }
                for _, cot_text, _, _ in extracted
            ]
        else:
            validations = validate_cot_batch(
                cot_texts=[cot_text for _, cot_text, _, _ in extracted],
                original_profiles=[dp["text"] for dp, _, _, _ in extracted],
                ground_truth_labels=[dp["label"] for dp, _, _, _ in extracted],
                config=validator_config,
                verbose=True,
                profile_ids=[dp["id"] for dp, _, _, _ in extracted],
            )

        for (data_point, _, eors_pos, eoa_pos), validation in zip(
            extracted, validations
        ):
            pid = data_point["id"]
            if validation["approved"]:
                pt_path, answer_pt_path, _ = promote_to_final(pid, temp_dir, final_dir)

                # Merge into consolidated tensors and remove per-profile files.
                cot_states_tensor[next_row_idx] = torch.load(pt_path, weights_only=True)
                answer_states_tensor[next_row_idx] = torch.load(
                    answer_pt_path, weights_only=True
                )
                delete_per_profile_files(pid, final_dir)

                # Save tensors to disk BEFORE writing the index entry so that
                # the tensor always reflects what the index claims. A mid-batch
                # crash would otherwise leave index entries pointing to zero rows.
                _save_consolidated()

                update_master_index(
                    pid,
                    index_path,
                    eors_pos,
                    eoa_pos,
                    cot_states_path,
                    answer_states_path,
                    next_row_idx,
                )
                next_row_idx += 1

                record = dict(data_point)
                record["cot_text"] = validation["cot_text"]
                record["answer_text"] = validation["answer_text"]
                record["quality_score"] = validation["quality_score"]
                record["faithfulness_score"] = validation["faithfulness_score"]
                record["bert_confidence"] = validation["bert_confidence"]
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                total += 1
                logger.info(
                    f"ID {pid}: approved — row {next_row_idx - 1} in consolidated tensors"
                )
            else:
                discard_from_temp(pid, temp_dir)
                logger.info(f"ID {pid}: CoT validation failed — queued for retry")
                still_failing.append(data_point)

        return still_failing

    def _save_consolidated() -> None:
        torch.save(cot_states_tensor, cot_states_path)
        torch.save(answer_states_tensor, answer_states_path)

    with open(output_path, "a") as f_out:
        # ---- main pass ----
        for ind, batch in enumerate(load_jsonl_batches(input_path, batch_size)):
            if num_batches is not None and ind >= num_batches:
                break
            unprocessed = [dp for dp in batch if dp["id"] not in processed_ids]
            if not unprocessed:
                continue
            new_failed = _process_batch(unprocessed, f_out)
            _save_consolidated()
            failed_points.extend(new_failed)
            logger.info(
                f"Batch {ind} done — {total} approved total, "
                f"{len(failed_points)} queued for retry so far"
            )

        logger.info(
            f"\nMain pass complete. {total} approved, "
            f"{len(failed_points)} queued for retry."
        )

        # ---- retry rounds ----
        retry_round = 0
        while failed_points and retry_round < max_retries:
            retry_round += 1
            logger.info(
                f"\n--- Retry round {retry_round}/{max_retries}: "
                f"{len(failed_points)} profile(s) remaining ---"
            )
            next_failed: list[dict] = []
            for i in range(0, len(failed_points), batch_size):
                next_failed.extend(
                    _process_batch(failed_points[i : i + batch_size], f_out)
                )
            _save_consolidated()
            logger.info(
                f"Retry round {retry_round} complete — {len(next_failed)} still failing"
            )
            failed_points = next_failed

    if failed_points:
        logger.info(
            f"\nWARNING: {len(failed_points)} profile(s) permanently failed after "
            f"{max_retries} retry rounds. IDs: {[dp['id'] for dp in failed_points]}"
        )
    logger.info(f"\nDone! {total} CoT + hidden state pairs saved.")
    logger.info(f"Master index: {index_path}")
    logger.info(f"Consolidated tensors: {cot_states_path}, {answer_states_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_cot_generation(
    config_path: str,
    path_suff: str | None = None,
    teacher_model_suff: str | None = None,
    extract_hidden_states: bool | None = None,
    skip_validation: bool = False,
) -> None:
    """Run CoT generation, with optional hidden state extraction.

    Args:
        config_path:           Path to the YAML config file.
        path_suff:             Suffix inserted into the output file name
                               (e.g. "train", "val", "test").
        teacher_model_suff:    Suffix appended after the teacher model name
                               in the output file name.
        extract_hidden_states: If not None, overrides ``hidden_states.enabled``
                               in the YAML config (CLI flag takes precedence).
        skip_validation:       If True, bypass CoT validation and save all
                               generated profiles (scores will be null).
    """

    config = load_config(config_path)

    active = config["dataset"]["active_sub_dataset"]
    sub_cfg = config["dataset"]["sub_datasets"][active]
    input_path = sub_cfg["processed_profiles_path"]
    prompt_path = sub_cfg["prompt_path"]
    cot_output_path = sub_cfg["cot_output_path"]

    if not teacher_model_suff or not path_suff:
        print(
            "Please provide --teacher-model-suff and path-suff, e.g. --teacher-model-suff Qwen3-8b --path-suff test \nExiting."
        )
        sys.exit(1)

    if path_suff:
        cot_output_path = cot_output_path.replace(
            ".", f"_{path_suff}_{teacher_model_suff}."
        )
    output_path = resolve_path(cot_output_path)
    input_path = input_path.replace(".", f"_{path_suff}.")

    print_memory_usage("before model load")
    model, tokenizer = load_model(config)
    prompt = load_prompt(prompt_path)
    print_memory_usage("after model load")

    teacher_cfg = config["teacher_model"]
    batch_size = teacher_cfg["batch_size"]
    max_new_tokens = teacher_cfg["max_new_tokens"]
    num_batches = teacher_cfg.get("num_batches", None)
    max_retries = teacher_cfg.get("max_retries", 3)
    id2label: dict[int, str] = {int(k): v for k, v in sub_cfg["id2label"].items()}
    validator_config = _SUB_DATASET_VALIDATOR[active]

    hs_cfg = config.get("hidden_states", {})
    extraction_enabled = (
        extract_hidden_states
        if extract_hidden_states is not None
        else hs_cfg.get("enabled", False)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        logger.info(f"Resuming — {len(processed_ids)} profiles already processed.")

    if extraction_enabled:
        logger.info("Hidden state extraction ENABLED — batched generation with hook.")
        _run_extraction_loop(
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            id2label=id2label,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            num_batches=num_batches,
            max_retries=max_retries,
            hs_cfg=hs_cfg,
            path_suff=path_suff,
            validator_config=validator_config,
            dataset_name=active,
            skip_validation=skip_validation,
        )
        return

    # ----------------------------------------------------------------
    # Standard batched generation (extraction disabled)
    # ----------------------------------------------------------------
    total = 0
    failed_points: list[dict] = []

    with open(output_path, "a") as f_out:
        for ind, batch in enumerate(load_jsonl_batches(input_path, batch_size)):
            if num_batches is not None and ind >= num_batches:
                break
            unprocessed = [dp for dp in batch if dp["id"] not in processed_ids]
            if not unprocessed:
                continue
            for data_point, validation in _generate_and_validate(
                unprocessed,
                id2label,
                prompt,
                model,
                tokenizer,
                max_new_tokens,
                validator_config,
                skip_validation=skip_validation,
            ):
                if validation["approved"]:
                    data_point["cot_text"] = validation["cot_text"]
                    data_point["answer_text"] = validation["answer_text"]
                    data_point["quality_score"] = validation["quality_score"]
                    data_point["faithfulness_score"] = validation["faithfulness_score"]
                    data_point["bert_confidence"] = validation["bert_confidence"]
                    f_out.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                    f_out.flush()
                    total += 1
                    print(f"ID {data_point['id']} approved and saved")
                else:
                    failed_points.append(data_point)
                    print(f"ID {data_point['id']} failed validation, queued for retry")
            print(
                f"Batch {ind} finished — "
                f"{len(failed_points)} total queued for retry so far"
            )

        print(
            f"\nMain pass complete. {total} approved, "
            f"{len(failed_points)} failed and queued for retry."
        )

        retry_round = 0
        while failed_points and retry_round < max_retries:
            retry_round += 1
            print(
                f"\n--- Retry round {retry_round}/{max_retries}: "
                f"{len(failed_points)} points remaining ---"
            )
            next_failed: list[dict] = []
            for i in range(0, len(failed_points), batch_size):
                retry_batch = failed_points[i : i + batch_size]
                for data_point, validation in _generate_and_validate(
                    retry_batch,
                    id2label,
                    prompt,
                    model,
                    tokenizer,
                    max_new_tokens,
                    validator_config,
                    skip_validation=skip_validation,
                ):
                    if validation["approved"]:
                        data_point["cot_text"] = validation["cot_text"]
                        data_point["answer_text"] = validation["answer_text"]
                        data_point["quality_score"] = validation["quality_score"]
                        data_point["faithfulness_score"] = validation[
                            "faithfulness_score"
                        ]
                        data_point["bert_confidence"] = validation["bert_confidence"]
                        f_out.write(json.dumps(data_point, ensure_ascii=False) + "\n")
                        f_out.flush()
                        total += 1
                        print(
                            f"ID {data_point['id']} approved on retry round {retry_round}"
                        )
                    else:
                        next_failed.append(data_point)
                        print(
                            f"ID {data_point['id']} still failing "
                            f"(retry round {retry_round})"
                        )
            print(
                f"Retry round {retry_round} complete — "
                f"{len(next_failed)} points still failing"
            )
            failed_points = next_failed

    if failed_points:
        print(
            f"WARNING: {len(failed_points)} point(s) could not be approved after "
            f"{max_retries} retry rounds. IDs: {[dp['id'] for dp in failed_points]}"
        )
    print(f"Done! {total} new CoT texts saved to {output_path}")
