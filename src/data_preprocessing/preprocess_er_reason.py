"""
ER-REASON preprocessing pipeline.

Transforms raw ER-REASON CSV records into structured text profiles (≤512 tokens)
suitable for CoT generation, mirroring the role of FinBench profiles.

Pipeline
--------
Step 1  Load & filter CSV — keep Discharge / Admit, drop Transfer, map labels 0/1.
Step 2  Outer batch loop with resume — scan output JSONL for already-processed
        encounterkeys, skip those profiles.
Step 3  LLM summarization — one dedicated batched GPU call per column.
        For each of the 7 historical columns, all N profile texts for that column
        are processed together in a single model.generate() call.  column_batch_size
        controls how many columns are grouped per outer loop chunk (memory
        management only); total LLM calls is always 7, one per column.
        Returns an (N, 7) grid of 1-2 sentence summaries; None where input was None.
Step 4  Medical history consolidation — single LLM call per batch of N profiles.
        The 7 per-column summaries are consolidated into one coherent paragraph.
Step 5  Structured profile assembly — demographics + current visit + medical history.
Step 6  Token budget check — BERT tokenizer, ≤512 tokens; discard on failure.
Step 7  Save passing profiles to output JSONL.

LLM backend: local Qwen3-4B with 8-bit quantization via BitsAndBytesConfig.
Mirrors the load_model() pattern from src/cot/cot_generator.py exactly.
"""

import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils.memory_printer import print_memory_usage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

LABEL_COL = "eddisposition"
LABEL_MAP = {"Discharge": 0, "Admit": 1}
VALID_DISPOSITIONS = set(LABEL_MAP.keys())

# 7 historical text columns — order defines column indices 0–6 throughout
HIST_COLS = [
    "Discharge_Summary_Text",
    "Progress_Note_Text",
    "HP_Note_Text",
    "Echo_Text",
    "Imaging_Text",
    "Consult_Text",
    "ECG_Text",
]

HIST_COL_LABELS = [
    "Discharge Summary",
    "Progress Note",
    "History & Physical",
    "Echocardiogram",
    "Imaging",
    "Consultation",
    "ECG",
]

METADATA_COLS = [
    "patientdurablekey",
    "encounterkey",
    "note_count",
    "ArrivalYearKey",
    "DepartureYearKeyValue",
    "DepartureYearKey",
    "DispositionYearKeyValue",
    "birthYear",
    "Discharge_Summary_Year",
    "Progress_Note_Year",
    "HP_Note_Year",
    "Echo_Year",
    "Imaging_Year",
    "Consult_Year",
    "ED_Provider_Notes_Year",
    "ECG_Year",
]

BERT_TOKENIZER_ID = "bert-base-uncased"
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# LLM prompt constants
# ---------------------------------------------------------------------------

_SUMMARIZATION_SYSTEM = (
    "You are a clinical note summarizer for a research pipeline. "
    "Compress the clinical note into exactly 1-2 concise sentences that preserve "
    "the key clinical findings. Focus on: primary findings, relevant history, "
    "and significant results. Omit administrative details, patient identifiers, "
    "and routine normal findings."
)

_CONSOLIDATION_SYSTEM = (
    "You are a clinical summarizer for a medical research pipeline. "
    "Synthesize multiple brief clinical note summaries into one coherent medical "
    "history paragraph. Write exactly one sentence per note type; omit note types "
    "with no summary. Be concise and clinically precise."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_field(row: pd.Series, col: str, default: str = "N/A") -> str:
    """Return a clean string for a demographic field; fall back to default on null."""
    if col not in row.index:
        return default
    v = row[col]
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s.lower() not in ("", "nan", "none", "null", "n/a") else default


def _get_text_field(row: pd.Series, col: str) -> str | None:
    """Return stripped note text or None if the cell is absent / null / blank."""
    if col not in row.index:
        return None
    v = row[col]
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return None if s.lower() in ("", "nan", "none", "null", "n/a") else s


def _to_json_safe(v):
    """Convert pandas/numpy scalars to JSON-serialisable Python types."""
    if v is None:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


# ---------------------------------------------------------------------------
# Step 1 — Load & filter
# ---------------------------------------------------------------------------


def load_and_filter(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)
    df = df[df[LABEL_COL].isin(VALID_DISPOSITIONS)].copy()
    df["label"] = df[LABEL_COL].map(LABEL_MAP)
    df = df.reset_index(drop=True)
    logger.info(
        "Loaded %d rows; kept %d after filtering to Discharge/Admit.", n_raw, len(df)
    )
    return df


# ---------------------------------------------------------------------------
# Step 2 — Resume support
# ---------------------------------------------------------------------------


def load_processed_keys(output_path: str) -> set[str]:
    """Return the set of encounterkeys already written to the output JSONL."""
    processed: set[str] = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ek = record.get("metadata", {}).get("encounterkey")
                if ek is not None:
                    processed.add(str(ek))
            except json.JSONDecodeError:
                continue
    logger.info(
        "Found %d already-processed encounterkeys in %s.", len(processed), output_path
    )
    return processed


# ---------------------------------------------------------------------------
# Local LLM — load and generate
# ---------------------------------------------------------------------------


def load_llm(
    model_name: str,
    quantization: str | None = "8bit",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load local LLM with optional 8-bit quantization.

    Mirrors load_model() in src/cot/cot_generator.py exactly.
    """
    quant_cfg = (
        BitsAndBytesConfig(load_in_8bit=True) if quantization == "8bit" else None
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


def _generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Run one batched model.generate() call over pre-formatted chat prompts.

    Uses left-padding so all sequences in the batch have the same length.
    The generated slice for each item starts at max_input_len (consistent with
    cot_generator.py's _slice_generated approach).
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    max_input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    print_memory_usage(f"peak after generation (batch={len(prompts)})")

    to_skip = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id}
    to_skip.discard(None)

    results = []
    for i in range(len(prompts)):
        gen_ids = output_ids[i][max_input_len:].tolist()
        filtered = [tid for tid in gen_ids if tid not in to_skip]
        results.append(tokenizer.decode(filtered, skip_special_tokens=False).strip())

    del output_ids, inputs
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Step 3 — LLM summarization (per-column batched GPU generation)
# ---------------------------------------------------------------------------


def _summarize_one_column(
    col_idx: int,
    notes_grid: list[list[str | None]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_note_chars: int,
    max_new_tokens: int,
) -> list[str | None]:
    """Summarize one historical column for all N profiles in a single GPU batch.

    Each profile gets a clean, dedicated prompt for this specific note type.
    GPU processes all N non-null profiles for this column in parallel.

    Returns a list of N summaries (None where the input note was None).
    """
    n = len(notes_grid)
    col_label = HIST_COL_LABELS[col_idx]
    result: list[str | None] = [None] * n

    # Collect non-null texts and their profile indices
    active_indices: list[int] = []
    active_texts: list[str] = []
    for row_idx in range(n):
        text = notes_grid[row_idx][col_idx]
        if text is not None:
            active_indices.append(row_idx)
            active_texts.append(text[:max_note_chars] if max_note_chars else text)

    if not active_texts:
        return result

    # One focused prompt per profile: "summarize this {col_label} note"
    prompts = []
    for text in active_texts:
        messages = [
            {"role": "system", "content": _SUMMARIZATION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Summarize the following {col_label} note in 1-2 sentences, "
                    f"preserving key clinical findings:\n\n{text}"
                ),
            },
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                tools=None,
            )
        )

    print_memory_usage(f"before {col_label} batch ({len(prompts)} profiles)")
    summaries = _generate_batch(model, tokenizer, prompts, max_new_tokens)

    for row_idx, summary in zip(active_indices, summaries):
        result[row_idx] = summary if summary.strip() else None

    return result


def summarize_notes_batch(
    notes_grid: list[list[str | None]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    column_batch_size: int,
    max_note_chars: int,
    max_new_tokens: int,
) -> list[list[str | None]]:
    """Summarize all 7 historical columns; one batched GPU call per column.

    column_batch_size controls how many columns are processed per outer loop chunk
    (memory management only — e.g. column_batch_size=2 locally, 7 on the cluster).
    Total LLM calls = always 7, one per column.

    Returns:
        (N, 7) grid of 1-2 sentence summaries; None where input was None.
    """
    n = len(notes_grid)
    result: list[list[str | None]] = [[None] * 7 for _ in range(n)]

    for col_start in range(0, 7, column_batch_size):
        col_end = min(col_start + column_batch_size, 7)
        for col_idx in range(col_start, col_end):
            col_summaries = _summarize_one_column(
                col_idx, notes_grid, model, tokenizer, max_note_chars, max_new_tokens
            )
            for row_idx in range(n):
                result[row_idx][col_idx] = col_summaries[row_idx]

    return result


# ---------------------------------------------------------------------------
# Step 4 — Medical history consolidation
# ---------------------------------------------------------------------------


def consolidate_histories_batch(
    summaries_grid: list[list[str | None]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> list[str]:
    """Consolidate per-column summaries into one medical history paragraph per patient.

    Builds one prompt per patient and runs a single _generate_batch call (batch_size=N).
    Mirrors the _summarize_one_column pattern — no JSON required.
    Returns N strings (empty string if all summaries for a patient were None).
    """
    n = len(summaries_grid)
    histories: list[str] = [""] * n

    active_indices: list[int] = []
    prompts: list[str] = []

    for i, summaries in enumerate(summaries_grid):
        notes_text = "\n".join(
            f"- {HIST_COL_LABELS[j]}: {summaries[j]}"
            for j in range(7)
            if summaries[j] is not None
        )
        if not notes_text:
            continue
        messages = [
            {"role": "system", "content": _CONSOLIDATION_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Consolidate the following clinical note summaries into one concise "
                    "medical history paragraph:\n\n" + notes_text
                ),
            },
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                tools=None,
            )
        )
        active_indices.append(i)

    if not prompts:
        return histories

    results = _generate_batch(model, tokenizer, prompts, max_new_tokens)
    for idx, history in zip(active_indices, results):
        histories[idx] = history.strip()

    return histories


# ---------------------------------------------------------------------------
# Step 5 — Structured profile assembly
# ---------------------------------------------------------------------------


def assemble_raw_profile(row: pd.Series) -> str:
    """Assemble demographics + current visit (no historical notes)."""
    return (
        f"{_str_field(row, 'Age')} year old {_str_field(row, 'sex')}, "
        f"{_str_field(row, 'firstrace')}, speaks {_str_field(row, 'preferredlanguage')}.\n"
        f"Chief Complaint: {_str_field(row, 'primarychiefcomplaintname')}\n"
        f"ED Diagnosis: {_str_field(row, 'primaryeddiagnosisname')}\n"
        f"Acuity Level: {_str_field(row, 'acuitylevel')}\n"
        f"Current Visit: {_str_field(row, 'One_Sentence_Extracted')}"
    )


def assemble_profile(row: pd.Series, medical_history: str) -> str:
    """Assemble the full structured profile (raw profile + medical history block)."""
    base = assemble_raw_profile(row)
    if medical_history:
        return base + f"\nMedical History: {medical_history}"
    return base


# ---------------------------------------------------------------------------
# Step 6 — Token budget check
# ---------------------------------------------------------------------------

_bert_tokenizer_cache = None


def _get_bert_tokenizer():
    global _bert_tokenizer_cache
    if _bert_tokenizer_cache is None:
        _bert_tokenizer_cache = AutoTokenizer.from_pretrained(BERT_TOKENIZER_ID)
    return _bert_tokenizer_cache


def check_token_budget(text: str) -> bool:
    tokenizer = _get_bert_tokenizer()
    count = len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
    if count > MAX_TOKENS:
        logger.warning("Token budget exceeded: %d tokens (max %d).", count, MAX_TOKENS)
        return False
    return True


# ---------------------------------------------------------------------------
# Step 7 — Save
# ---------------------------------------------------------------------------


def _build_metadata(row: pd.Series, summaries: list[str | None]) -> dict:
    meta = {
        col: _to_json_safe(row[col]) if col in row.index else None
        for col in METADATA_COLS
    }
    # Human-readable names of historical columns included in the medical history
    meta["included_history_cols"] = [
        HIST_COL_LABELS[j] for j in range(7) if summaries[j] is not None
    ]
    return meta


def save_record(
    f, row: pd.Series, text: str, label: int, summaries: list[str | None]
) -> None:
    record = {
        "id": _to_json_safe(row["encounterkey"]),
        "text": text,
        "label": label,
        "metadata": _build_metadata(row, summaries),
    }
    f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline (Steps 1–7)
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str,
    output_path: str,
    model_name: str,
    quantization: str | None,
    batch_size: int,
    column_batch_size: int,
    run_name: str,
    max_note_chars: int = 3000,
    max_new_tokens: int = 512,
    num_batches: int | None = None,
) -> None:
    """Run the full ER-REASON preprocessing pipeline.

    Args:
        input_path:        Path to the raw ER-REASON CSV.
        output_path:       Path to the output JSONL (appended; resume-safe).
        model_name:        HuggingFace model name for local LLM summarization.
        quantization:      '8bit' for BitsAndBytes 8-bit quantization, or None.
        batch_size:        Number of profiles per outer batch (N).
        column_batch_size: Number of historical columns per outer loop chunk.
                           Controls memory usage; total LLM calls is always 7.
        run_name:          Experiment label used in log messages.
        max_note_chars:    Truncate input notes to this length before LLM calls.
                           Set 0 to disable truncation.
        max_new_tokens:    Maximum new tokens per LLM generation call.
        num_batches:       Limit the main pass to this many batches. None = full dataset.
    """
    df = load_and_filter(input_path)

    print_memory_usage("before LLM load")
    model, tokenizer = load_llm(model_name, quantization)
    print_memory_usage("after LLM load")

    # Step 2: resume — skip already-processed encounters
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed_keys = load_processed_keys(output_path)
    if processed_keys:
        mask = ~df["encounterkey"].astype(str).isin(processed_keys)
        n_skip = int((~mask).sum())
        df = df[mask].reset_index(drop=True)
        logger.info(
            "Skipping %d already-processed records; %d remaining.", n_skip, len(df)
        )

    total_pass = 0
    total_token_fail = 0
    n_total = len(df)
    _max_chars = max_note_chars if max_note_chars > 0 else 10**9

    logger.info("=" * 65)
    logger.info("ER-REASON preprocessing  |  run=%s", run_name)
    logger.info("  Input          : %s", input_path)
    logger.info("  Output         : %s", output_path)
    logger.info("  LLM model      : %s", model_name)
    logger.info("  Quantization   : %s", quantization or "none")
    logger.info("  Batch size     : %d", batch_size)
    logger.info("  Column batch   : %d", column_batch_size)
    logger.info(
        "  Max note chars : %s", _max_chars if _max_chars < 10**9 else "unlimited"
    )
    logger.info("  Max new tokens : %d", max_new_tokens)
    logger.info("  Total records  : %d", n_total)
    logger.info("=" * 65)

    with open(output_path, "a", encoding="utf-8") as f_out:
        for batch_idx, batch_start in enumerate(range(0, n_total, batch_size)):
            if num_batches is not None and batch_idx >= num_batches:
                logger.info("Reached num_batches=%d limit, stopping.", num_batches)
                break
            batch = df.iloc[batch_start : batch_start + batch_size]
            n = len(batch)
            logger.info(
                "[%s] Batch %d–%d / %d",
                run_name,
                batch_start + 1,
                batch_start + n,
                n_total,
            )

            # Step 3: build (N, 7) notes grid and LLM-summarize per column
            notes_grid: list[list[str | None]] = [
                [_get_text_field(row, col) for col in HIST_COLS]
                for _, row in batch.iterrows()
            ]
            summaries_grid = summarize_notes_batch(
                notes_grid,
                model,
                tokenizer,
                column_batch_size,
                _max_chars,
                max_new_tokens,
            )

            # Step 4: consolidate per patient into one paragraph
            histories = consolidate_histories_batch(
                summaries_grid, model, tokenizer, max_new_tokens
            )

            # Steps 5 & 6: assemble + token budget gate
            batch_pass = 0
            passing_keys: list = []
            failing_keys: list = []

            for i, (_, row) in enumerate(batch.iterrows()):
                profile = assemble_profile(row, histories[i])
                label = int(row["label"])
                ek = _to_json_safe(row.get("encounterkey"))

                if not check_token_budget(profile):
                    failing_keys.append(ek)
                    total_token_fail += 1
                    continue

                # Step 7: save
                save_record(f_out, row, profile, label, summaries_grid[i])
                f_out.flush()
                batch_pass += 1
                passing_keys.append(ek)

            total_pass += batch_pass

            if passing_keys:
                logger.info(
                    "  Passing encounterkeys (%d): %s", len(passing_keys), passing_keys
                )
            if failing_keys:
                logger.warning(
                    "  Token-failed encounterkeys (%d): %s",
                    len(failing_keys),
                    failing_keys,
                )
            logger.info("  → pass=%d | token_fail=%d", batch_pass, len(failing_keys))

    logger.info("=" * 65)
    logger.info("Pipeline complete  |  run=%s", run_name)
    logger.info(
        "  Pass           : %d / %d  (%.1f%%)",
        total_pass,
        n_total,
        100 * total_pass / max(n_total, 1),
    )
    logger.info("  Token fail     : %d", total_token_fail)
    logger.info("  Output         : %s", output_path)
    logger.info("=" * 65)
