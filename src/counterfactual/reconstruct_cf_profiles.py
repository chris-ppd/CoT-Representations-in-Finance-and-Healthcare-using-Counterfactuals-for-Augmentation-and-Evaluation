"""
Reconstruct plain counterfactual profiles from counterfactual CoT reasoning text.

Input:  CF JSONL files produced by cf_generator.py — each entry has
        counterfactual_cot_text, counterfactual_label, original_id, id.
Output: JSONL with one reconstructed plain-text profile per entry, matching
        the style of the original text field.

Processing per entry:
  1. Strip CoT structural markers ([STEP], [ES], [EORS], [ANSWER], [EOA]).
  2. Call Qwen in batches to synthesize a concise plain profile.
  3. Tokenize output with bert-base-uncased; log and skip if > 512 tokens.
  4. Log and skip if Qwen returns an empty string.
  5. Write approved entries; resume support via id scanning on restart.
"""

import json
import logging
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.memory_printer import print_memory_usage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

_SPECIAL_TOKENS = ["[STEP]", "[ES]", "[EORS]", "[ANSWER]", "[EOA]"]
_BERT_TOKENIZER_ID = "bert-base-uncased"

_qwen_model: AutoModelForCausalLM | None = None
_qwen_tokenizer: AutoTokenizer | None = None
_bert_tokenizer = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_qwen(model_name: str):
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        logger.info(f"Loading Qwen: {model_name}")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qwen_tokenizer.padding_side = "left"
        if _qwen_tokenizer.pad_token is None:
            _qwen_tokenizer.pad_token = _qwen_tokenizer.eos_token
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,  # model name from config only — teacher_model_suff is for output naming only
            quantization_config=quant_cfg,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print_memory_usage("after Qwen load")
    return _qwen_model, _qwen_tokenizer


def _get_bert_tokenizer():
    global _bert_tokenizer
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained(_BERT_TOKENIZER_ID)
    return _bert_tokenizer


def _strip_special_tokens(text: str) -> str:
    """Remove CoT structural markers and collapse whitespace."""
    for tok in _SPECIAL_TOKENS:
        text = text.replace(tok, "")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Batched Qwen generation — mirrors cf_generator._generate_qwen_batch
# ---------------------------------------------------------------------------


def _generate_qwen_batch(
    messages_list: list[list[dict]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> list[str]:
    chat_texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            tools=None,
        )
        for msgs in messages_list
    ]
    inputs = tokenizer(
        chat_texts, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    max_input_len = inputs.input_ids.shape[1]

    print_memory_usage(f"before generate (batch={len(messages_list)})")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    print_memory_usage(f"after generate (batch={len(messages_list)})")

    to_skip = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id}
    to_skip.discard(None)

    results = []
    for i in range(len(messages_list)):
        gen_ids = output_ids[i][max_input_len:].tolist()
        filtered = [tid for tid in gen_ids if tid not in to_skip]
        results.append(tokenizer.decode(filtered, skip_special_tokens=True).strip())

    del output_ids, inputs
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def _check_token_budget(text: str) -> tuple[bool, int]:
    """Return (within_budget, token_count) using bert-base-uncased."""
    tok = _get_bert_tokenizer()
    count = len(tok(text, truncation=False, add_special_tokens=True)["input_ids"])
    return count <= 512, count


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_system_prompt(path: str) -> str:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Reconstruction system prompt not found.\n"
            f"Expected path (relative to project root): {path}\n"
            f"Full path: {resolved}\n"
            f"Create this file before running reconstruction."
        )
    return resolved.read_text(encoding="utf-8").strip()


def _load_processed_ids(output_path: Path) -> set:
    """Scan output JSONL for id fields to support resumption."""
    processed: set = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    processed.add(json.loads(line)["id"])
    return processed


def _derive_output_filename(input_path: Path, suff: str | None = None) -> str:
    """Derive output filename from input by removing '_cot' segment from stem.

    e.g. ld1_counterfactual_cot_train.jsonl, suff=None  → ld1_counterfactual_train.jsonl
         ld1_counterfactual_cot_train.jsonl, suff=Qwen3-4b → ld1_counterfactual_train_Qwen3-4b.jsonl
    """
    stem = input_path.stem.replace("_cot_", "_")
    if suff:
        stem = f"{stem}_{suff}"
    return stem + ".jsonl"


# ---------------------------------------------------------------------------
# Main reconstruction loop
# ---------------------------------------------------------------------------


def run_cf_reconstruction(
    config_path: str,
    input_path: str,
    output_path: str | None = None,
    model_name: str | None = None,
    teacher_model_suff: str | None = None,
    batch_size: int | None = None,
    num_batches: int | None = None,
    path_suff: str = "train",
) -> None:
    """Reconstruct plain CF profiles for one split.

    Args:
        config_path:        Path to the CF config YAML containing a reconstruction block.
        input_path:         Path to input CF JSONL (counterfactual_cot split file).
        output_path:        Override output file path; if None, derived from config + input name.
        model_name:         Override model name/path; if None, read from config generation.teacher_model.name.
        teacher_model_suff: Suffix appended to output filename only — never used for model loading.
        batch_size:         Profiles per Qwen call; overrides config if given.
        num_batches:        Limit to first N batches (N*batch_size profiles); overrides config.
        path_suff:          Split identifier used for logging (e.g. "train", "val", "test").
    """
    config = load_config(config_path)
    rec_cfg = config["reconstruction"]
    sub_dataset: str = config["dataset"]["sub_dataset"]

    system_prompt = _load_system_prompt(rec_cfg["system_prompt_path"])

    _batch_size = (
        batch_size if batch_size is not None else rec_cfg.get("batch_size", 80)
    )
    _max_new_tokens: int = rec_cfg.get("max_new_tokens", 512)
    _num_batches: int | None = (
        num_batches if num_batches is not None else rec_cfg.get("num_batches")
    )

    input_file = Path(input_path)
    if output_path is not None:
        out_file = Path(output_path)
    else:
        out_dir = _resolve(rec_cfg["output_dir"])
        out_file = out_dir / _derive_output_filename(input_file, teacher_model_suff)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # model name from config only — teacher_model_suff is for output naming only
    _model_name = model_name or config["generation"]["teacher_model"]["name"]
    qwen, qwen_tok = _load_qwen(_model_name)

    processed_ids = _load_processed_ids(out_file)
    if processed_ids:
        logger.info(f"Resuming — {len(processed_ids)} profiles already written.")

    all_data: list[dict] = []
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["id"] in processed_ids:
                continue
            all_data.append(entry)

    total = (
        min(len(all_data), _num_batches * _batch_size)
        if _num_batches is not None
        else len(all_data)
    )

    logger.info("=" * 65)
    logger.info(f"CF reconstruction  |  sub_dataset={sub_dataset}, split={path_suff}")
    logger.info(f"  model         : {model_name}")
    logger.info(f"  batch_size    : {_batch_size}")
    logger.info(f"  max_new_tokens: {_max_new_tokens}")
    logger.info(f"  total profiles: {total}")
    logger.info(f"  output        : {out_file}")
    logger.info("=" * 65)

    written = 0
    skipped_empty = 0
    skipped_budget = 0

    with open(out_file, "a") as f_out:
        for batch_idx, batch_start in enumerate(range(0, total, _batch_size)):
            if _num_batches is not None and batch_idx >= _num_batches:
                logger.info(f"Reached num_batches={_num_batches} limit, stopping.")
                break

            batch = all_data[batch_start : batch_start + _batch_size]
            n = len(batch)
            logger.info(
                f"Batch {batch_idx + 1} — "
                f"profiles {batch_start + 1}–{batch_start + n} / {total}"
            )

            messages_list = [
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": _strip_special_tokens(
                            entry["counterfactual_cot_text"]
                        ),
                    },
                ]
                for entry in batch
            ]

            responses = _generate_qwen_batch(
                messages_list, qwen, qwen_tok, _max_new_tokens
            )

            batch_written = 0
            for entry, response in zip(batch, responses):
                if not response:
                    logger.warning(
                        f"ID {entry['id']}: Qwen returned empty string, skipping."
                    )
                    skipped_empty += 1
                    continue

                ok, token_count = _check_token_budget(response)
                if not ok:
                    logger.warning(
                        f"ID {entry['id']}: output exceeds 512 BERT tokens "
                        f"({token_count}), skipping."
                    )
                    skipped_budget += 1
                    continue

                record = {
                    "id": entry["id"],
                    "original_id": entry["original_id"],
                    "counterfactual_text": response,
                    "counterfactual_cot_text": entry["counterfactual_cot_text"],
                    "original_label": entry["original_label"],
                    "counterfactual_label": entry["counterfactual_label"],
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                written += 1
                batch_written += 1

            logger.info(
                f"Batch {batch_idx + 1} done — {batch_written} written, "
                f"running totals: written={written}, "
                f"skipped_empty={skipped_empty}, skipped_budget={skipped_budget}"
            )

    logger.info(
        f"Reconstruction complete — written={written}, "
        f"skipped_empty={skipped_empty}, skipped_budget={skipped_budget}, "
        f"output={out_file}"
    )
