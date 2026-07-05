"""
Unified BERT fine-tuning entry point for the whole thesis.

Supports any HuggingFace BERT-family checkpoint (finbert, medbert, bert-base, …)
on any binary-classification dataset, configured entirely through a YAML file.

Two roles
---------
validator   Concatenate train + val + test original profiles, then stratified
            80/20 train/eval.  The model sees the full data distribution and
            learns to classify from *original* text — used inside validate_cot.

student     Load from the CoT split files (cot_train / cot_val / cot_test).
            Each CoT file already contains both the original text and the CoT
            text for every validated record, so both modes train on exactly the
            same set of IDs — results are directly comparable.

Two input modes (student role only)
------------------------------------
original    Use the ``text`` field from the CoT .jsonl file.
cot         Use the ``cot_text`` field from the CoT .jsonl file.

Loss configuration
------------------
use_weighted_loss: true   → WeightedLossTrainer (CrossEntropyLoss with
                             per-class weights from class_weight config key).
use_weighted_loss: false  → Standard HuggingFace Trainer.

class_weight options (YAML key ``training.class_weight``):
  null / "balanced"            — sklearn inverse-frequency weighting.
  {"0": 0.5, "1": 2.0}        — explicit per-class weights (string keys OK).

Outputs
-------
  <output_dir>/<run_name>/   model weights + tokenizer + trainer_state.json
  <output_dir>/<run_name>/run_manifest.json  flags, split IDs, final metrics

Checkpoint note
---------------
``AutoModelForSequenceClassification.from_pretrained(checkpoint)`` always
starts from the HuggingFace pretrained weights — no local checkpoint is
resumed unless ``trainer.train(resume_from_checkpoint=...)`` is called
explicitly.  By default this file trains from scratch on the pretrained
weights every time.
"""

import csv
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import wandb
from src.data.injection_data_utils import (
    get_step_states,
    load_hidden_states,
    validate_dataset,
)
from src.models.bert_with_injection import BertWithInjection

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# logging.getLogger("transformers").setLevel(logging.ERROR)
# logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# logger = logging.getLogger("finetune_bert")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------

load_dotenv()
ENTITY_NAME = os.getenv("WANDB_ENTITY")
PROJECT_NAME = os.getenv("WANDB_PROJECT")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(yaml_path: Path) -> dict[str, Any]:
    with open(yaml_path, "r") as fh:
        return yaml.safe_load(fh)


def apply_overrides(config: dict, overrides: dict) -> None:
    """Merge non-None CLI overrides into the training block of config in-place."""
    for key, value in overrides.items():
        if value is not None:
            config["training"][key] = value


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _log_split_stats(name: str, records: list[dict]) -> None:
    n0 = sum(1 for r in records if r["label"] == 0)
    n1 = sum(1 for r in records if r["label"] == 1)
    logger.info(
        "%-12s : %4d records  (class-0=%d | class-1=%d)", name, len(records), n0, n1
    )


# ---------------------------------------------------------------------------
# CoT step stripping
# ---------------------------------------------------------------------------

_STEP_PATTERN = re.compile(r"\[STEP\](.*?)\[EORS\]", re.DOTALL)


def _strip_last_cot_step(text: str, record_id=None) -> str:
    """Remove the 4th [STEP]...[EORS] block from a 4-step CoT profile.

    Returns the original text unchanged if the profile does not have exactly
    4 blocks (with a warning), so callers can always use the return value safely.
    """
    matches = list(_STEP_PATTERN.finditer(text))
    if len(matches) != 4:
        logger.warning(
            "_strip_last_cot_step: expected 4 [STEP]...[EORS] blocks, "
            "found %d — returning text unchanged (record_id=%r)",
            len(matches),
            record_id,
        )
        return text
    stripped = (text[: matches[3].start()] + text[matches[3].end() :]).strip()
    remaining = list(_STEP_PATTERN.finditer(stripped))
    if len(remaining) != 3:
        logger.warning(
            "_strip_last_cot_step: after removal expected 3 blocks, "
            "found %d (record_id=%r)",
            len(remaining),
            record_id,
        )
    return stripped


# ---------------------------------------------------------------------------
# Injection hidden-state path resolution
# ---------------------------------------------------------------------------


def _hidden_state_paths(
    directory: str | Path,
    dataset: str,
    split: str,
    is_cf: bool,
    model_suff: str,
) -> tuple[str, str]:
    """Resolve (tensor_path, index_path) for a hidden-state directory.

    Filename convention:
        cot/{split}/  → {dataset}_{split}_cot_states.pt
                         {dataset}_{split}_hidden_states_index.jsonl
        cf/{split}/   → {dataset}_cf_{split}_{model_suff}_cot_states.pt
                         {dataset}_cf_{split}_{model_suff}_hidden_states_index.jsonl
    """
    d = Path(directory)
    if is_cf:
        stem = f"{dataset}_cf_{split}_{model_suff}"
    else:
        stem = f"{dataset}_{split}"
    tensor_path = str(d / f"{stem}_cot_states.pt")
    index_path = str(d / f"{stem}_hidden_states_index.jsonl")
    return tensor_path, index_path


def _load_injection_hidden_states(
    injection_cfg: dict,
    project_root: Path,
    dataset: str,
    augment: bool,
    need_cf_test: bool,
) -> dict[str, tuple]:
    """Load every hidden-state index/tensor pair needed for injection mode.

    cot/{train,val,test} are always loaded (1:1 with the record splits).
    cf/{train,val} are loaded only when augment=True (augmented splits may
    contain CF rows). cf/test is loaded only when need_cf_test=True (mirrors
    the existing cf_test_path condition for mode in (original, cot)).

    Returns a dict keyed "cot_train"/"cot_val"/"cot_test"/"cf_train"/"cf_val"/
    "cf_test", each value an (id_to_row, tensor, tensor_path) 3-tuple — or
    (None, None, None) for a cf_* split that wasn't needed.
    """
    model_suff = injection_cfg.get("teacher_model_suff", "Qwen3-4b")
    hs_cfg = injection_cfg["hidden_states"]

    result: dict[str, tuple] = {}
    for split in ("train", "val", "test"):
        hs_dir = project_root / hs_cfg["cot"][split]
        tensor_path, index_path = _hidden_state_paths(
            hs_dir, dataset, split, is_cf=False, model_suff=model_suff
        )
        result[f"cot_{split}"] = (
            *load_hidden_states(index_path, tensor_path),
            tensor_path,
        )

    cf_splits_needed = set()
    if augment:
        cf_splits_needed.update(("train", "val", "test"))
    if need_cf_test:
        cf_splits_needed.add("test")

    for split in ("train", "val", "test"):
        if split in cf_splits_needed:
            hs_dir = project_root / hs_cfg["cf"][split]
            tensor_path, index_path = _hidden_state_paths(
                hs_dir, dataset, split, is_cf=True, model_suff=model_suff
            )
            result[f"cf_{split}"] = (
                *load_hidden_states(index_path, tensor_path),
                tensor_path,
            )
        else:
            result[f"cf_{split}"] = (None, None, None)

    return result


# ---------------------------------------------------------------------------
# Validator data loading  (all original splits → stratified train/eval)
# ---------------------------------------------------------------------------


def load_validator_data(
    dataset_cfg: dict,
    eval_fraction: float,
    seed: int,
    project_root: Path,
) -> tuple[list[dict], list[dict]]:
    """Concatenate original train/val/test, return (train_records, eval_records).

    Uses a stratified split to preserve the class ratio in both halves.
    """
    all_records: list[dict] = []
    for split_key in ("train", "val", "test"):
        rel_path = dataset_cfg.get(split_key)
        if not rel_path:
            continue
        abs_path = project_root / rel_path
        if not abs_path.exists():
            logger.warning(
                "Split '%s' not found at %s — skipping.", split_key, abs_path
            )
            continue
        records = _load_jsonl(str(abs_path))
        _log_split_stats(split_key, records)
        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No records loaded. Check dataset.train/val/test paths in the YAML."
        )

    logger.info("Concatenated: %d total records", len(all_records))

    rng = np.random.default_rng(seed)
    class_0 = [r for r in all_records if r["label"] == 0]
    class_1 = [r for r in all_records if r["label"] == 1]

    train_records: list[dict] = []
    eval_records: list[dict] = []
    for cls_records in (class_0, class_1):
        n_eval = max(1, int(len(cls_records) * eval_fraction))
        idx = rng.permutation(len(cls_records))
        eval_records.extend(cls_records[i] for i in idx[:n_eval])
        train_records.extend(cls_records[i] for i in idx[n_eval:])

    _log_split_stats("train (strat)", train_records)
    _log_split_stats("eval  (strat)", eval_records)
    return train_records, eval_records


# ---------------------------------------------------------------------------
# Student data loading  (CoT split files → original or cot text)
# ---------------------------------------------------------------------------


def load_student_data(
    dataset_cfg: dict,
    mode: str,
    project_root: Path,
    augment: bool = False,
    drop_last_step: bool = False,
    extra_fields: list[str] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train / val / test from the CoT or augmented split files.

    Non-augmented (``augment=False``):
        Reads from ``cot_train`` / ``cot_val`` / ``cot_test`` keys.  Each file
        contains validated records with ``text``, ``cot_text``, and ``label``.
        Records without a non-empty ``cot_text`` are dropped so both modes
        operate on the same IDs and are directly comparable.

    Augmented (``augment=True``):
        Reads from ``aug_{mode}_train`` / ``aug_{mode}_val`` / ``aug_{mode}_test``
        keys (e.g. ``aug_cot_train`` or ``aug_original_train``).  These files
        carry only a single text field (``cot_text`` for mode=cot, ``text`` for
        mode=original); the cot_text filter is skipped.

    Args:
        dataset_cfg:    The ``dataset`` block from the YAML config.
        mode:           ``'original'`` or ``'cot'``.
        project_root:   Absolute project root for resolving relative paths.
        augment:        When True, load augmented split files instead of CoT files.
        drop_last_step: When True and mode=='cot', strip the 4th [STEP]...[EORS]
                        block from each CoT text before training.
        extra_fields:   Additional raw-record fields to carry over into the
                        processed record (e.g. ``["original_id", "is_counterfactual"]``
                        for injection mode, so CF rows in augmented splits stay
                        resolvable by ``get_step_states``). Ignored if absent
                        from a given raw record.

    Returns:
        ``(train_records, val_records, test_records)`` — each record is a
        minimal dict ``{id, text, label}`` ready for ``BertDataset``.
    """
    result: dict[str, list[dict]] = {}

    if augment:
        split_keys = {
            "train": f"aug_{mode}_train",
            "val": f"aug_{mode}_val",
            "test": f"aug_{mode}_test",
        }
    else:
        split_keys = {
            "train": "cot_train",
            "val": "cot_val",
            "test": "cot_test",
        }
    text_field = "cot_text" if mode == "cot" else "text"

    for logical_split, cfg_key in split_keys.items():
        file_path = dataset_cfg.get(cfg_key)
        if not file_path:
            logger.warning(
                "dataset.%s not set in config — split '%s' will be empty.",
                cfg_key,
                logical_split,
            )
            result[logical_split] = []
            continue

        abs_path = project_root / file_path
        if not abs_path.exists():
            logger.warning(
                "File not found: %s — split '%s' will be empty.",
                abs_path,
                logical_split,
            )
            result[logical_split] = []
            continue

        raw_records = _load_jsonl(str(abs_path))

        # Filter to records with a valid text field.  Non-augmented splits
        # always filter on cot_text so both modes train on the same IDs.
        if augment:
            source_records = [r for r in raw_records if r.get(text_field, "").strip()]
        else:
            source_records = [r for r in raw_records if r.get("cot_text", "").strip()]
            n_dropped = len(raw_records) - len(source_records)
            if n_dropped:
                logger.warning(
                    "Split '%s': %d / %d records lack cot_text and were dropped.",
                    logical_split,
                    n_dropped,
                    len(raw_records),
                )

        processed: list[dict] = []
        stripped_count = 0
        skipped_count = 0
        preview_logged = 0

        for r in source_records:
            text = r[text_field]
            if mode == "cot" and drop_last_step:
                new_text = _strip_last_cot_step(text, record_id=r.get("id"))
                if new_text != text:
                    stripped_count += 1
                else:
                    skipped_count += 1
                text = new_text
                if preview_logged < 5:
                    logger.info(
                        "Strip preview [split=%s, id=%r] (first 5 only):\n%s",
                        logical_split,
                        r.get("id"),
                        text,
                    )
                    preview_logged += 1
            processed_record = {"id": r["id"], "label": r["label"], "text": text}
            if extra_fields:
                for field in extra_fields:
                    if field in r:
                        processed_record[field] = r[field]
            processed.append(processed_record)

        if mode == "cot" and drop_last_step:
            logger.info(
                "Split '%s': last CoT step stripped for %d records; "
                "%d skipped (did not have exactly 4 blocks)",
                logical_split,
                stripped_count,
                skipped_count,
            )

        _log_split_stats(logical_split, processed)
        result[logical_split] = processed

    return result.get("train", []), result.get("val", []), result.get("test", [])


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class BertDataset(Dataset):
    """Generic binary-classification dataset for any BERT-family model.

    Args:
        records:    List of dicts with keys ``id``, ``text``, ``label``.
        tokenizer:  HuggingFace tokenizer.
        max_length: Maximum token sequence length.
        label_map:  Dict mapping our int label → model label id.
                    Defaults to identity {0: 0, 1: 1}.
        cot_index, cf_index, cot_states, cf_cot_states:
                    Injection-mode hidden-state lookups (see
                    ``injection_data_utils.get_step_states``). When
                    ``cot_states`` is None (default), no ``step_states`` key
                    is added to items — non-injection modes are unaffected.
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int,
        label_map: dict[int, int] | None = None,
        cot_index: dict[str, int] | None = None,
        cf_index: dict[str, int] | None = None,
        cot_states=None,
        cf_cot_states=None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or {0: 0, 1: 1}
        self.cot_index = cot_index
        self.cf_index = cf_index
        self.cot_states = cot_states
        self.cf_cot_states = cf_cot_states

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        label = self.label_map[record["label"]]
        encoding = self.tokenizer(
            record["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        if self.cot_states is not None or self.cf_cot_states is not None:
            item["step_states"] = get_step_states(
                record,
                self.cot_index,
                self.cf_index,
                self.cot_states,
                self.cf_cot_states,
            )
        return item


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _compute_metrics(eval_pred) -> dict[str, float]:
    """Return a compute_metrics closure reporting accuracy, f1_micro, f1_macro, f1_weighted."""

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def _compute_class_weights(
    train_records: list[dict],
    class_weight_cfg,
) -> torch.Tensor:
    """Derive a per-class weight tensor from the training records.

    Args:
        train_records:    Training records (dicts with a ``label`` key).
        class_weight_cfg: One of:
            ``None`` / ``"balanced"`` — sklearn inverse-frequency weighting.
            ``dict`` e.g. ``{"0": 0.5, "1": 2.0}`` — explicit weights
            (string keys are auto-cast to int).
    """
    labels = np.array([r["label"] for r in train_records])
    classes = np.unique(labels)

    if class_weight_cfg is None or class_weight_cfg == "balanced":
        weights = compute_class_weight("balanced", classes=classes, y=labels)
    elif isinstance(class_weight_cfg, dict):
        cw = {int(k): float(v) for k, v in class_weight_cfg.items()}
        weights = compute_class_weight(cw, classes=classes, y=labels)
    else:
        raise ValueError(f"Unrecognised class_weight value: {class_weight_cfg!r}")

    weight_tensor = torch.tensor(weights, dtype=torch.float)
    for cls, w in zip(classes, weights):
        logger.info("Class weight — label %d: %.4f", cls, w)
    return weight_tensor


# ---------------------------------------------------------------------------
# Weighted-loss Trainer
# ---------------------------------------------------------------------------


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies per-class weights to CrossEntropyLoss."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Post-training cleanup
# ---------------------------------------------------------------------------


def cleanup_model_weights(output_dir: str) -> dict:
    """Promote best checkpoint files to output_dir root, delete the rest.

    Returns the best-metric / best-step dict extracted from trainer_state.json.
    """
    keep_files = {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "trainer_state.json",
    }

    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            for filename in ("model.safetensors", "config.json", "trainer_state.json"):
                src = os.path.join(item_path, filename)
                dst = os.path.join(output_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            shutil.rmtree(item_path)
            logger.info("Promoted + deleted checkpoint dir: %s", item)

    for filename in list(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath) and filename not in keep_files:
            os.remove(filepath)

    best_info: dict = {}
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as fh:
            state = json.load(fh)
        best_info = {
            "best_metric": state.get("best_metric"),
            "best_step": state.get("best_global_step"),
        }
        logger.info(
            "Best metric: %.4f at step %s",
            best_info["best_metric"] or 0.0,
            best_info["best_step"],
        )
    return best_info


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def save_run_manifest(
    save_dir: str,
    run_name: str,
    config_path: str,
    role: str,
    mode: str,
    train_records: list[dict],
    val_records: list[dict],
    test_records: list[dict],
    best_info: dict,
    training_cfg: dict,
    model_checkpoint: str,
    test_metrics: dict | None = None,
    injection_position: str | None = None,
    augment: bool = False,
) -> None:
    manifest = {
        "run_name": run_name,
        "config": config_path,
        "role": role,
        "mode": mode,
        "injection_position": injection_position,
        "augment": augment,
        "model_checkpoint": model_checkpoint,
        "train_ids": [r["id"] for r in train_records],
        "val_ids": [r["id"] for r in val_records],
        "test_ids": [r["id"] for r in test_records] if test_records else None,
        "best_metric": best_info.get("best_metric"),
        "best_step": best_info.get("best_step"),
        "test_metrics": test_metrics,
        "hyperparams": {
            "epochs": training_cfg.get("epochs"),
            "batch_size": training_cfg.get("batch_size"),
            "lr": training_cfg.get("lr"),
            "warmup_ratio": training_cfg.get("warmup_ratio"),
            "max_len": training_cfg.get("max_len"),
            "seed": training_cfg.get("seed"),
            "eval_split": training_cfg.get("eval_split"),
            "class_weight": training_cfg.get("class_weight"),
            "use_weighted_loss": training_cfg.get("use_weighted_loss"),
            "primary_metric": training_cfg.get("primary_metric", "f1_macro"),
            "early_stopping_patience": training_cfg.get("early_stopping_patience", 3),
        },
    }
    manifest_path = os.path.join(save_dir, "run_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Run manifest saved to %s", manifest_path)


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------


def evaluate_on_set(
    trainer: Trainer,
    records: list[dict],
    tokenizer,
    max_len: int,
    label_map: dict[int, int],
    num_labels: int,
    id2label: dict[int, str],
    wandb_prefix: str,
    cot_index: dict[str, int] | None = None,
    cf_index: dict[str, int] | None = None,
    cot_states=None,
    cf_cot_states=None,
) -> dict:
    """Evaluate trainer on a record set, log metrics to wandb, and return them.

    Builds a ``BertDataset`` from ``records``, calls ``trainer.predict()``,
    computes accuracy, F1 macro, and AP per class, then fires a single
    ``wandb.log()`` call under ``wandb_prefix``.

    ``cot_index``/``cf_index``/``cot_states``/``cf_cot_states`` are forwarded
    to ``BertDataset`` for injection mode; left as ``None`` they are no-ops.

    Returns a dict with keys: ``accuracy``, ``f1_macro``, ``f1_weighted``,
    and ``ap`` (``{class_idx: float}``).
    """
    dataset = BertDataset(
        records,
        tokenizer,
        max_length=max_len,
        label_map=label_map,
        cot_index=cot_index,
        cf_index=cf_index,
        cot_states=cot_states,
        cf_cot_states=cf_cot_states,
    )
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    probs = softmax(logits, axis=-1)
    preds = np.argmax(logits, axis=-1)

    metrics: dict = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }
    logger.info("%s metrics:", wandb_prefix)
    for k, v in metrics.items():
        logger.info("  %-20s %.4f", k, v)

    ap_per_class: dict[int, float] = {}
    ap_named: dict[str, float] = {}
    pr_plots: dict = {}
    for class_idx in range(num_labels):
        class_name = id2label.get(class_idx, f"class_{class_idx}")
        binary_labels = (labels == class_idx).astype(int)
        class_probs = probs[:, class_idx]
        ap = float(average_precision_score(binary_labels, class_probs))
        ap_per_class[class_idx] = ap
        ap_named[class_name] = ap
        logger.info("  AP %-20s %.4f", class_name, ap)
        pr_plots[f"{wandb_prefix}/PR_curve_{class_name}"] = wandb.plot.pr_curve(
            binary_labels,
            np.column_stack([1 - class_probs, class_probs]),
            labels=["not_target", class_name],
            classes_to_plot=[1],
        )

    ap_table = wandb.Table(
        columns=["class", "AP"],
        data=[
            [id2label.get(i, f"class_{i}"), ap_per_class[i]] for i in range(num_labels)
        ],
    )
    wandb.log(
        {
            **{f"{wandb_prefix}/{k}": v for k, v in metrics.items()},
            **{f"{wandb_prefix}/AP_{name}": score for name, score in ap_named.items()},
            **pr_plots,
            f"{wandb_prefix}/AP_comparison": wandb.plot.bar(
                ap_table,
                label="class",
                value="AP",
                title="Average Precision per class",
            ),
        }
    )

    metrics["ap"] = ap_per_class
    return metrics


# ---------------------------------------------------------------------------
# Experiment 1 CSV results
# ---------------------------------------------------------------------------


def _append_csv(
    path: Path,
    columns: list[str],
    row: dict,
    match_by: tuple[str, ...] = ("dataset", "student_model"),
) -> None:
    """Update the matching row in a pre-initialized CSV file in-place.

    Reads the full file, locates the row where every match_by column equals
    the corresponding value from row, updates only non-empty columns, then
    writes the whole file back.  Never overwrites a populated cell with an
    empty string — logs at INFO and prints when that would have happened.
    If no matching row is found, logs a warning and does nothing (the
    orchestrator is responsible for pre-initializing the file with all rows).
    """
    if not path.exists():
        logger.warning(
            "_append_csv: %s not found — orchestrator must pre-initialize CSVs before training",
            path,
        )
        return

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames: list[str] = list(reader.fieldnames or columns)
        all_rows = list(reader)

    match_values = {k: row[k] for k in match_by}
    matched_idx: int | None = None
    for i, existing in enumerate(all_rows):
        if all(existing.get(k) == v for k, v in match_values.items()):
            matched_idx = i
            break

    if matched_idx is None:
        logger.warning(
            "_append_csv: no row matched %s in %s — skipping",
            match_values,
            path,
        )
        return

    updated_cols: dict[str, str] = {}
    for col, value in row.items():
        if col in match_by:
            continue
        if value == "":
            existing_value = all_rows[matched_idx].get(col, "")
            if existing_value != "":
                msg = (
                    f"_append_csv: skipping empty update for '{col}' "
                    f"(existing value {existing_value!r} preserved) in {path}"
                )
                logger.info(msg)
                print(msg)
            continue
        all_rows[matched_idx][col] = str(value)
        updated_cols[col] = str(value)

    row_id = ", ".join(f"{k}={match_values[k]}" for k in match_by)
    cols_str = ", ".join(f"{k}={v}" for k, v in updated_cols.items())
    logger.info(
        "_append_csv: updated row (%s) — %s", row_id, cols_str or "<no columns updated>"
    )

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def _append_experiment1_results(
    results_dir: Path,
    dataset: str,
    student_model: str,
    standard_metrics: dict | None,
    cf_metrics: dict | None,
    aug_metrics: dict | None,
) -> None:
    """Update the matching (dataset, student_model) row in the two Experiment 1 CSVs.

    The CSVs are pre-initialized by the orchestrator with all rows present and
    metric columns empty.  Non-augmented runs fill Standard / CF columns;
    augmented runs fill Augmented columns only.  CF Drop is computed only when
    both standard and CF metrics are available.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    def _f1(m: dict | None) -> float | str:
        return round(m["f1_macro"], 4) if m is not None else ""

    def _ap(m: dict | None, class_idx: int) -> float | str:
        if m is None:
            return ""
        value = m["ap"].get(class_idx)
        return round(value, 4) if value is not None else ""

    cf_drop: float | str = ""
    if standard_metrics is not None and cf_metrics is not None:
        cf_drop = round(standard_metrics["f1_macro"] - cf_metrics["f1_macro"], 4)

    primary_cols = [
        "dataset",
        "student_model",
        "Standard F1",
        "CF F1",
        "CF Drop",
        "Augmented F1",
    ]
    _append_csv(
        results_dir / "primary_results.csv",
        primary_cols,
        {
            "dataset": dataset,
            "student_model": student_model,
            "Standard F1": _f1(standard_metrics),
            "CF F1": _f1(cf_metrics),
            "CF Drop": cf_drop,
            "Augmented F1": _f1(aug_metrics),
        },
    )

    ap_cols = [
        "dataset",
        "student_model",
        "Standard AP0",
        "Standard AP1",
        "CF AP0",
        "CF AP1",
        "Aug AP0",
        "Aug AP1",
    ]
    _append_csv(
        results_dir / "ap_per_class_results.csv",
        ap_cols,
        {
            "dataset": dataset,
            "student_model": student_model,
            "Standard AP0": _ap(standard_metrics, 0),
            "Standard AP1": _ap(standard_metrics, 1),
            "CF AP0": _ap(cf_metrics, 0),
            "CF AP1": _ap(cf_metrics, 1),
            "Aug AP0": _ap(aug_metrics, 0),
            "Aug AP1": _ap(aug_metrics, 1),
        },
    )
    logger.info("Experiment 1 results appended to %s", results_dir)


# ---------------------------------------------------------------------------
# Experiment 2 CSV results (hidden-state injection)
# ---------------------------------------------------------------------------


def _append_experiment2_results(
    results_dir: Path,
    dataset: str,
    student_model: str,
    standard_metrics: dict | None,
    cf_metrics: dict | None,
    aug_metrics: dict | None,
) -> None:
    """Same shape as _append_experiment1_results, written to experiment2_*.csv."""
    results_dir.mkdir(parents=True, exist_ok=True)

    def _f1(m: dict | None) -> float | str:
        return round(m["f1_macro"], 4) if m is not None else ""

    def _ap(m: dict | None, class_idx: int) -> float | str:
        if m is None:
            return ""
        value = m["ap"].get(class_idx)
        return round(value, 4) if value is not None else ""

    cf_drop: float | str = ""
    if standard_metrics is not None and cf_metrics is not None:
        cf_drop = round(standard_metrics["f1_macro"] - cf_metrics["f1_macro"], 4)

    primary_cols = [
        "dataset",
        "student_model",
        "Standard F1",
        "CF F1",
        "CF Drop",
        "Augmented F1",
    ]
    _append_csv(
        results_dir / "experiment2_primary_results.csv",
        primary_cols,
        {
            "dataset": dataset,
            "student_model": student_model,
            "Standard F1": _f1(standard_metrics),
            "CF F1": _f1(cf_metrics),
            "CF Drop": cf_drop,
            "Augmented F1": _f1(aug_metrics),
        },
    )

    ap_cols = [
        "dataset",
        "student_model",
        "Standard AP0",
        "Standard AP1",
        "CF AP0",
        "CF AP1",
        "Aug AP0",
        "Aug AP1",
    ]
    _append_csv(
        results_dir / "experiment2_ap_per_class_results.csv",
        ap_cols,
        {
            "dataset": dataset,
            "student_model": student_model,
            "Standard AP0": _ap(standard_metrics, 0),
            "Standard AP1": _ap(standard_metrics, 1),
            "CF AP0": _ap(cf_metrics, 0),
            "CF AP1": _ap(cf_metrics, 1),
            "Aug AP0": _ap(aug_metrics, 0),
            "Aug AP1": _ap(aug_metrics, 1),
        },
    )
    logger.info("Experiment 2 results appended to %s", results_dir)


def _append_gate_values(
    results_dir: Path,
    dataset: str,
    student_model: str,
    gate_init: dict[str, float],
    gate_final: dict[str, float],
) -> None:
    """Update the matching (dataset, student_model) row in experiment2_gate_values.csv."""
    results_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "student_model",
        "gate_1_init",
        "gate_2_init",
        "gate_3_init",
        "gate_1_final",
        "gate_2_final",
        "gate_3_final",
    ]
    _append_csv(
        results_dir / "experiment2_gate_values.csv",
        columns,
        {
            "dataset": dataset,
            "student_model": student_model,
            "gate_1_init": round(gate_init["gate_1"], 4),
            "gate_2_init": round(gate_init["gate_2"], 4),
            "gate_3_init": round(gate_init["gate_3"], 4),
            "gate_1_final": round(gate_final["gate_1"], 4),
            "gate_2_final": round(gate_final["gate_2"], 4),
            "gate_3_final": round(gate_final["gate_3"], 4),
        },
    )
    logger.info("Gate values appended to %s", results_dir)


# ---------------------------------------------------------------------------
# Main fine-tuning function
# ---------------------------------------------------------------------------


def finetune(
    config_path: str | Path,
    role: str,
    mode: str,
    run_name: str,
    project_root: Path | None = None,
    output_dir_override: str | None = None,
    hparam_overrides: dict | None = None,
    augment: bool = False,
    cf_test_path: str | None = None,
    dataset_name: str | None = None,
    student_model_name: str | None = None,
    results_dir: Path | None = None,
    drop_last_step: bool = False,
    injection_position: str | None = None,
    dry_run: bool = False,
    results_experiment: int = 1,
) -> None:
    """Fine-tune a BERT-family model according to a YAML config.

    Always starts from the HuggingFace pretrained weights specified in the
    config — no local checkpoint is resumed unless you explicitly pass
    ``resume_from_checkpoint`` to trainer.train().

    Args:
        config_path:          Path to the YAML config file.
        role:                 ``'validator'`` or ``'student'``.
        mode:                 ``'original'`` or ``'cot'`` (student role only).
        run_name:             Name saved in WandB and used as checkpoint subdir.
        project_root:         Absolute path to the project root.
        output_dir_override:  Override ``output.base_dir`` from the YAML.
        hparam_overrides:     Optional dict of training hyperparameter overrides
                              from CLI flags (None values are ignored).
        augment:              When True, load augmented training/val/test splits
                              (``aug_{mode}_*`` YAML keys) instead of CoT splits.
        cf_test_path:         Optional path to a CF test JSONL.  When provided
                              (and augment is False), a second post-training
                              evaluation is run under the ``test_cf/`` prefix.
        dataset_name:         Dataset identifier written to Experiment 1 CSVs.
        student_model_name:   Model identifier written to Experiment 1 CSVs.
        injection_position:   ``'pre_group'`` or ``'post_group'``. Required when
                              ``mode == 'injection'``; ignored (with a warning)
                              otherwise.
        dry_run:              When True, load config/data, run injection hidden-
                              state validation (mode=='injection' only), print
                              the report, and return before any training or
                              WandB side effects.
        results_experiment:   ``1`` (default) writes to the Experiment 1 CSVs
                              via ``_append_experiment1_results``. ``2`` writes
                              to the Experiment 2 CSVs via
                              ``_append_experiment2_results`` (+ gate values
                              for non-augmented injection runs).
    """
    if role not in ("validator", "student"):
        raise ValueError(f"--role must be 'validator' or 'student', got '{role}'.")
    if mode not in ("original", "cot", "injection"):
        raise ValueError(
            f"--mode must be 'original', 'cot', or 'injection', got '{mode}'."
        )
    if mode == "injection" and injection_position is None:
        raise ValueError("--injection-position is required when --mode injection")
    if mode != "injection" and injection_position is not None:
        logger.warning(
            "--injection-position=%r given but --mode=%r (not 'injection') — ignoring",
            injection_position,
            mode,
        )

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    config_path = Path(config_path)
    config = load_config(config_path)
    if hparam_overrides:
        apply_overrides(config, hparam_overrides)

    model_cfg = config["model"]
    training_cfg = config["training"]
    dataset_cfg = config["dataset"]
    output_cfg = config.get("output", {})
    injection_cfg: dict = config.get("injection", {}) if mode == "injection" else {}

    base_dir = output_dir_override or output_cfg.get(
        "base_dir", "model_weights/bert_model"
    )
    abs_base_dir = str(project_root / base_dir)

    model_checkpoint = model_cfg["checkpoint"]
    num_labels = model_cfg.get("num_labels", 2)
    raw_label_map = model_cfg.get("label_map", {"0": 0, "1": 1})
    label_map = {int(k): int(v) for k, v in raw_label_map.items()}

    # --------------------------------------------------------------- dry-run
    # Validate injection data wiring and exit before any WandB side effects
    # or training.  Only meaningful for mode == "injection"; for other modes
    # this is a no-op report (nothing injection-specific to validate).
    if dry_run:
        logger.info("=" * 65)
        logger.info("DRY RUN — mode=%s (no training, no WandB run)", mode)
        logger.info("=" * 65)
        if mode == "injection":
            gate_init_value = injection_cfg.get("gate_init")
            assert isinstance(gate_init_value, float), (
                f"injection.gate_init must be a float, got {gate_init_value!r}"
            )
            logger.info("  injection.gate_init = %s ✓ (float)", gate_init_value)

            _dataset_stem = dataset_name or config_path.stem
            hs = _load_injection_hidden_states(
                injection_cfg,
                project_root,
                _dataset_stem,
                augment=augment,
                need_cf_test=bool(cf_test_path),
            )

            split_keys = (
                {
                    "train": "aug_original_train",
                    "val": "aug_original_val",
                    "test": "aug_original_test",
                }
                if augment
                else {"train": "cot_train", "val": "cot_val", "test": "cot_test"}
            )

            logger.info(
                "[Hidden state directories used — dataset=%s, augment=%s]",
                _dataset_stem,
                augment,
            )
            for split, cfg_key in split_keys.items():
                cot_path = hs.get(f"cot_{split}", (None, None, None))[2]
                logger.info(
                    "  split=%-5s | jsonl=%s | source=cot | path=%s",
                    split,
                    cfg_key,
                    cot_path or "<not loaded>",
                )
                cf_relevant = augment or (split == "test" and cf_test_path)
                if cf_relevant:
                    cf_path = hs.get(f"cf_{split}", (None, None, None))[2]
                    cf_jsonl_label = cfg_key if augment else cf_test_path
                    logger.info(
                        "  split=%-5s | jsonl=%s | source=cf  | path=%s",
                        split,
                        cf_jsonl_label,
                        cf_path or "<not loaded>",
                    )

            for split, cfg_key in split_keys.items():
                jsonl_rel = dataset_cfg.get(cfg_key)
                if not jsonl_rel:
                    logger.warning(
                        "dataset.%s not set — skipping validation for split '%s'",
                        cfg_key,
                        split,
                    )
                    continue
                cot_index, cot_states, _ = hs[f"cot_{split}"]
                cf_index, cf_states, _ = hs[f"cf_{split}"]
                validate_dataset(
                    str(project_root / jsonl_rel),
                    cot_index,
                    cf_index,
                    cot_states,
                    cf_states,
                    original_jsonl_path=(
                        str(project_root / dataset_cfg["cot_train"])
                        if augment and split == "train" and dataset_cfg.get("cot_train")
                        else None
                    ),
                )

            if cf_test_path:
                cot_index, cot_states, _ = hs["cot_test"]
                cf_index, cf_states, _ = hs["cf_test"]
                validate_dataset(
                    cf_test_path, cot_index, cf_index, cot_states, cf_states
                )
        else:
            logger.info(
                "  mode=%s has no injection-specific validation — dry-run is a no-op.",
                mode,
            )
        logger.info("Dry run complete — exiting before trainer.train().")
        return

    # WandB — init before extracting training vars so sweep config can override them
    _cw = training_cfg.get("class_weight")
    if isinstance(_cw, dict):
        _cw_str = "-".join(str(v) for v in _cw.values())
    elif _cw is not None:
        _cw_str = str(_cw)
    else:
        _cw_str = "default"
    _auto_name = f"lr{training_cfg.get('lr')}_cw{_cw_str}_seed{training_cfg.get('seed')}".replace(
        ".", "_"
    )
    run = wandb.init(
        entity=ENTITY_NAME,
        project=PROJECT_NAME,
        name=run_name or _auto_name,
        group=f"{config_path.stem}_{role}",
        job_type="sweep",
    )

    # When launched by a wandb agent the sweep params arrive via ${args} → CLI →
    # apply_overrides, so training_cfg is already correct.  All we need here is
    # the run_name fallback (wandb.run.name equals the _auto_name we passed in).
    if wandb.run.sweep_id is not None:
        if run_name is None:
            run_name = wandb.run.name
    elif run_name is None:
        run.finish()
        print("Please provide --run-name, e.g. --run-name run1\nExiting.")
        return

    save_dir = os.path.join(abs_base_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    epochs = training_cfg["epochs"]
    batch_size = training_cfg["batch_size"]
    lr = float(training_cfg["lr"])
    warmup_ratio = training_cfg["warmup_ratio"]
    max_len = training_cfg["max_len"]
    eval_split = training_cfg.get("eval_split", 0.2)
    seed = training_cfg.get("seed", 42)
    class_weight_cfg = training_cfg.get("class_weight", None)
    use_weighted_loss = training_cfg.get("use_weighted_loss", True)
    primary_metric = training_cfg.get("primary_metric", "f1_macro")
    early_stopping_patience = training_cfg.get("early_stopping_patience", 3)

    # ------------------------------------------------------------------ banner
    if role == "validator":
        _ds_train = dataset_cfg.get("train", "n/a")
        _ds_val = dataset_cfg.get("val", "n/a")
        _ds_test = dataset_cfg.get("test", "n/a")
    elif augment:
        _banner_mode = "original" if mode == "injection" else mode
        _ds_train = dataset_cfg.get(f"aug_{_banner_mode}_train", "n/a")
        _ds_val = dataset_cfg.get(f"aug_{_banner_mode}_val", "n/a")
        _ds_test = dataset_cfg.get(f"aug_{_banner_mode}_test", "n/a")
    else:
        _ds_train = dataset_cfg.get("cot_train", "n/a")
        _ds_val = dataset_cfg.get("cot_val", "n/a")
        _ds_test = dataset_cfg.get("cot_test", "n/a")

    logger.info("=" * 65)
    logger.info(
        "BERT fine-tuning  |  role=%-10s  mode=%-10s  augment=%s",
        role,
        mode,
        augment,
    )
    logger.info("=" * 65)
    logger.info("  Checkpoint     : %s", model_checkpoint)
    logger.info("  Config         : %s", config_path)
    logger.info("  Output         : %s", save_dir)
    logger.info("  Dataset train  : %s", _ds_train)
    logger.info("  Dataset val    : %s", _ds_val)
    logger.info("  Dataset test   : %s", _ds_test)
    logger.info("  Epochs         : %d", epochs)
    logger.info("  Batch size     : %d", batch_size)
    logger.info("  Learning rate  : %.2e", lr)
    logger.info("  Warmup ratio   : %.0f%%", 100 * warmup_ratio)
    logger.info("  Max seq len    : %d", max_len)
    logger.info("  Seed           : %d", seed)
    logger.info("  Weighted loss  : %s", use_weighted_loss)
    logger.info("  Class weights  : %s", class_weight_cfg)
    logger.info("  Primary metric : %s", primary_metric)
    logger.info("  Augment        : %s", augment)
    if mode == "cot":
        logger.info("  Drop last step : %s", drop_last_step)
    elif mode == "injection":
        logger.info(
            "  Drop last step : N/A (handled implicitly via 3-step hidden-state slicing)"
        )
    else:
        logger.info("  Drop last step : N/A (mode=original, no CoT steps used)")
    if cf_test_path:
        logger.info("  CF test path   : %s", cf_test_path)
    logger.info("  Device         : %s", "GPU" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        logger.info("  GPU            : %s", torch.cuda.get_device_name(0))
    logger.info("=" * 65)

    # Log run metadata and all resolved hyperparams to WandB config so every
    # run is fully self-describing — especially useful when comparing sweep runs.
    # Keys already set by the sweep agent (e.g. seed) are skipped to avoid
    # overwriting them, which would require allow_val_change=True and suppress
    # legitimate conflict warnings.
    _sweep_keys = set(dict(wandb.config))
    _run_config = {
        "role": role,
        "mode": mode,
        "augment": augment,
        "drop_last_step": drop_last_step,
        "cf_test_path": cf_test_path or "",
        "dataset_name": dataset_name or "",
        "student_model_name": student_model_name or "",
        "config_file": str(config_path),
        "model_checkpoint": model_checkpoint,
        "model_weights_dir": save_dir,
        "dataset_train": _ds_train,
        "dataset_val": _ds_val,
        "dataset_test": _ds_test,
        # All resolved training params (sweep/CLI values take precedence over YAML)
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "max_len": max_len,
        "seed": seed,
        "eval_split": eval_split,
        "class_weight": str(class_weight_cfg),
        "use_weighted_loss": use_weighted_loss,
        "early_stopping_patience": early_stopping_patience,
        "primary_metric": primary_metric,
    }
    wandb.config.update({k: v for k, v in _run_config.items() if k not in _sweep_keys})

    # ------------------------------------------------------------------- data
    test_records: list[dict] = []

    if role == "validator":
        train_records, eval_records = load_validator_data(
            dataset_cfg, eval_split, seed, project_root
        )
    else:  # student
        # injection mode trains on the untouched original text field — the
        # "drop step 4" equivalent is handled entirely via get_step_states
        # slicing to the first 3 teacher steps, not via text stripping.
        _load_mode = "original" if mode == "injection" else mode
        _extra_fields = (
            ["original_id", "is_counterfactual"] if mode == "injection" else None
        )
        train_records, eval_records, test_records = load_student_data(
            dataset_cfg,
            _load_mode,
            project_root,
            augment=augment,
            drop_last_step=drop_last_step,
            extra_fields=_extra_fields,
        )

    if not train_records:
        raise RuntimeError(
            "Training split is empty — check your config and data paths."
        )
    if not eval_records:
        raise RuntimeError("Eval split is empty — check your config and data paths.")

    # ------------------------------------------------ injection hidden states
    hs: dict[str, tuple] = {}
    if mode == "injection":
        hs = _load_injection_hidden_states(
            injection_cfg,
            project_root,
            dataset_name or config_path.stem,
            augment=augment,
            need_cf_test=bool(cf_test_path),
        )

    # -------------------------------------------------------- class weights
    class_weights: torch.Tensor | None = None
    if use_weighted_loss:
        logger.info("Computing class weights from training split...")
        class_weights = _compute_class_weights(train_records, class_weight_cfg)

    # ------------------------------------------------ model / tokenizer
    logger.info("Loading tokenizer and model from '%s'...", model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model_kwargs: dict = {
        "num_labels": num_labels,
        "ignore_mismatched_sizes": True,
    }
    if "id2label" in model_cfg:
        id2label = {int(k): v for k, v in model_cfg["id2label"].items()}
        model_kwargs["id2label"] = id2label
        model_kwargs["label2id"] = {v: k for k, v in id2label.items()}

    if mode == "injection":
        model = BertWithInjection(
            bert_model_name=model_checkpoint,
            num_labels=num_labels,
            injection_position=injection_position,
            gate_init=injection_cfg["gate_init"],
            layer_groups=injection_cfg.get("layer_groups"),
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, **model_kwargs
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %.1fM total  |  %.1fM trainable",
        total_params / 1e6,
        trainable_params / 1e6,
    )

    # ------------------------------------------------------ dataset objects
    _cot_train_idx, _cot_train_states, _ = hs.get("cot_train", (None, None, None))
    _cf_train_idx, _cf_train_states, _ = hs.get("cf_train", (None, None, None))
    _cot_val_idx, _cot_val_states, _ = hs.get("cot_val", (None, None, None))
    _cf_val_idx, _cf_val_states, _ = hs.get("cf_val", (None, None, None))

    train_dataset = BertDataset(
        train_records,
        tokenizer,
        max_length=max_len,
        label_map=label_map,
        cot_index=_cot_train_idx,
        cf_index=_cf_train_idx,
        cot_states=_cot_train_states,
        cf_cot_states=_cf_train_states,
    )
    eval_dataset = BertDataset(
        eval_records,
        tokenizer,
        max_length=max_len,
        label_map=label_map,
        cot_index=_cot_val_idx,
        cf_index=_cf_val_idx,
        cot_states=_cot_val_states,
        cf_cot_states=_cf_val_states,
    )
    logger.info(
        "Datasets — train: %d  |  eval: %d  |  test (held-out): %d",
        len(train_dataset),
        len(eval_dataset),
        len(test_records),
    )
    wandb.config.update(
        {
            "n_train": len(train_records),
            "n_eval": len(eval_records),
            "n_test": len(test_records),
        }
    )

    # ---------------------------------------------- training arguments
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    logging_steps = max(1, steps_per_epoch // 4)

    logger.info(
        "Steps: total=%d  |  warmup=%d  |  per-epoch=%d  |  log-every=%d",
        total_steps,
        warmup_steps,
        steps_per_epoch,
        logging_steps,
    )

    training_args = TrainingArguments(
        output_dir=abs_base_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=primary_metric,
        greater_is_better=True,
        logging_dir=os.path.join(abs_base_dir, "logs"),
        logging_steps=logging_steps,
        save_total_limit=1,
        seed=seed,
        report_to="wandb",
        fp16=torch.cuda.is_available(),
    )

    # --------------------------------------------------------------- trainer
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    if use_weighted_loss and class_weights is not None:
        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_metrics,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=_compute_metrics,
            callbacks=callbacks,
        )

    # ---------------------------------------------------------------- train
    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("=" * 65)
    logger.info("Training complete")
    logger.info("  Global steps  : %d", train_result.global_step)
    logger.info("  Training loss : %.4f", train_result.training_loss)
    logger.info("=" * 65)

    # ------------------------------------------------- injection diagnostics
    if mode == "injection":
        gate_final = model.get_gate_values()
        proj_norms = {
            f"proj_{k}_weight_norm": getattr(model, f"proj_{k}").weight.norm().item()
            for k in (1, 2, 3)
        }
        wandb.log(
            {
                **{f"gates/{name}": value for name, value in gate_final.items()},
                **proj_norms,
            }
        )
        logger.info("Final gate values: %s", gate_final)
        logger.info("Projection weight norms: %s", proj_norms)

        # One standalone forward pass for residual / hidden-state norm diagnostics —
        # never exercised by Trainer.compute_loss, so a single example is sufficient.
        model.eval()
        with torch.no_grad():
            _diag_item = eval_dataset[0]
            _device = next(model.parameters()).device
            _diag_batch = {
                k: v.unsqueeze(0).to(_device)
                for k, v in _diag_item.items()
                if k in ("input_ids", "attention_mask", "step_states")
            }
            _, injection_norms = model(**_diag_batch, return_injection_norms=True)
        wandb.log({f"injection_norms/{k}": v for k, v in injection_norms.items()})
        logger.info("Injection norms (single-example diagnostic): %s", injection_norms)
        model.train()

    # ------------------------------------------------------------ save
    trainer.save_model(abs_base_dir)
    tokenizer.save_pretrained(abs_base_dir)

    best_info = cleanup_model_weights(abs_base_dir)

    # Move cleaned files into the named run subdirectory
    for filename in (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "trainer_state.json",
    ):
        src = os.path.join(abs_base_dir, filename)
        dst = os.path.join(save_dir, filename)
        if os.path.exists(src):
            shutil.move(src, dst)

    tokenizer.save_pretrained(save_dir)
    logger.info("Weights saved to %s", save_dir)

    # --------------------------------------------------------- test eval
    id2label: dict[int, str] = {
        int(k): v
        for k, v in model_cfg.get("id2label", {"0": "class_0", "1": "class_1"}).items()
    }

    primary_test_records = test_records if test_records else eval_records
    if not test_records:
        logger.info(
            "No held-out test split available — using eval set (%d records) "
            "for test metrics and PR curves.",
            len(eval_records),
        )

    _eval_kwargs: dict = dict(
        trainer=trainer,
        tokenizer=tokenizer,
        max_len=max_len,
        label_map=label_map,
        num_labels=num_labels,
        id2label=id2label,
    )

    standard_metrics: dict | None = None
    cf_eval_metrics: dict | None = None
    aug_metrics: dict | None = None

    _cot_test_idx, _cot_test_states, _ = hs.get("cot_test", (None, None, None))
    _cf_test_idx, _cf_test_states, _ = hs.get("cf_test", (None, None, None))

    if augment:
        logger.info(
            "Running augmented test evaluation (%d records)...",
            len(primary_test_records),
        )
        aug_metrics = evaluate_on_set(
            records=primary_test_records,
            wandb_prefix="test_augmented",
            cot_index=_cot_test_idx,
            cf_index=_cf_test_idx,
            cot_states=_cot_test_states,
            cf_cot_states=_cf_test_states,
            **_eval_kwargs,
        )
        test_metrics = aug_metrics
    else:
        logger.info(
            "Running standard test evaluation (%d records)...",
            len(primary_test_records),
        )
        standard_metrics = evaluate_on_set(
            records=primary_test_records,
            wandb_prefix="test_standard",
            cot_index=_cot_test_idx,
            cf_index=None,
            cot_states=_cot_test_states,
            cf_cot_states=None,
            **_eval_kwargs,
        )
        test_metrics = standard_metrics

        if cf_test_path:
            raw_cf = _load_jsonl(cf_test_path)
            cf_records: list[dict] = []
            _cf_text_key = (
                "counterfactual_cot_text" if mode == "cot" else "counterfactual_text"
            )
            _cf_stripped_count = 0
            _cf_skipped_count = 0
            _cf_preview_logged = 0
            for r in raw_cf:
                text = r.get(_cf_text_key, "")
                if not text.strip():
                    logger.warning(
                        "CF record id=%r has empty/missing '%s' — skipping.",
                        r.get("id", "<unknown>"),
                        _cf_text_key,
                    )
                    continue
                if mode == "cot" and drop_last_step:
                    new_text = _strip_last_cot_step(text, record_id=r.get("id"))
                    if new_text != text:
                        _cf_stripped_count += 1
                    else:
                        _cf_skipped_count += 1
                    text = new_text
                    if _cf_preview_logged < 5:
                        logger.info(
                            "CF strip preview [id=%r] (first 5 only):\n%s",
                            r.get("id"),
                            text,
                        )
                        _cf_preview_logged += 1
                cf_records.append(
                    {
                        "id": r.get("id", ""),
                        "original_id": r.get("original_id"),
                        "label": r["counterfactual_label"],
                        "text": text,
                    }
                )
            if mode == "cot" and drop_last_step:
                logger.info(
                    "CF test: last CoT step stripped for %d records; "
                    "%d skipped (did not have exactly 4 blocks)",
                    _cf_stripped_count,
                    _cf_skipped_count,
                )
            _log_split_stats("cf_test", cf_records)
            if cf_records:
                logger.info(
                    "Running CF test evaluation (%d records)...", len(cf_records)
                )
            cf_eval_metrics = evaluate_on_set(
                records=cf_records,
                wandb_prefix="test_cf",
                cot_index=None,
                cf_index=_cf_test_idx,
                cot_states=None,
                cf_cot_states=_cf_test_states,
                **_eval_kwargs,
            )
    # ---------------------------------------------------------- manifest
    save_run_manifest(
        save_dir=save_dir,
        run_name=run_name,
        config_path=str(config_path),
        role=role,
        mode=mode,
        train_records=train_records,
        val_records=eval_records,
        test_records=test_records,
        best_info=best_info,
        training_cfg=training_cfg,
        model_checkpoint=model_checkpoint,
        test_metrics=test_metrics,
        injection_position=injection_position if mode == "injection" else None,
        augment=augment,
    )

    # ------------------------------------------------ results CSV
    if dataset_name and student_model_name:
        if results_experiment == 2:
            _append_experiment2_results(
                results_dir=project_root / "results" / f"{results_dir}",
                dataset=dataset_name,
                student_model=student_model_name,
                standard_metrics=standard_metrics,
                cf_metrics=cf_eval_metrics,
                aug_metrics=aug_metrics,
            )
            if mode == "injection" and not augment:
                _append_gate_values(
                    results_dir=project_root / "results" / f"{results_dir}",
                    dataset=dataset_name,
                    student_model=student_model_name,
                    gate_init={
                        "gate_1": injection_cfg["gate_init"],
                        "gate_2": injection_cfg["gate_init"],
                        "gate_3": injection_cfg["gate_init"],
                    },
                    gate_final=model.get_gate_values(),
                )
        else:
            _append_experiment1_results(
                results_dir=project_root / "results" / f"{results_dir}",
                dataset=dataset_name,
                student_model=student_model_name,
                standard_metrics=standard_metrics,
                cf_metrics=cf_eval_metrics,
                aug_metrics=aug_metrics,
            )

    run.finish()
    logger.info("Done. Model ready at: %s", save_dir)
