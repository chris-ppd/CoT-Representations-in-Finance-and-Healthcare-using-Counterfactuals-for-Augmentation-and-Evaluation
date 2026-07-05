"""
LIME attribution for counterfactual generation (Script 1 of 2).

Runs LIME (Ribeiro et al., 2016) on the fine-tuned FinBERT/MedBERT validator as a
black-box classifier to extract the most important token occurrences per profile.
With bow=False each occurrence is treated as an independent feature — correctly
handling repeated tokens (e.g. two "1000"s in the same profile).

Each profile is split into 4 steps × 2 sentences (8 sentences total) via [EORS]
and [ES] delimiters.  One independent explain_instance call is made per sentence,
with structural tokens stripped from the input.  Output is keyed by (step, sentence)
to map directly to CF generator sentence slots.

Runtime: ~2-3 s per profile on GPU. Run once per split.
"""

import json
import logging
import re
import string
from pathlib import Path

import numpy as np
import torch
import yaml
from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, pipeline

from src.cot.cot_validator import (
    FINANCE_VALIDATOR_CONFIG,
    HEALTHCARE_VALIDATOR_CONFIG,
    ValidatorConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

_SUB_DATASET_VALIDATOR: dict[str, ValidatorConfig] = {
    "ld1": FINANCE_VALIDATOR_CONFIG,
    "er-reason": HEALTHCARE_VALIDATOR_CONFIG,
}

_stop_words: set[str] | None = None

STRUCTURAL_TOKENS: frozenset[str] = frozenset({"STEP", "EORS", "ES", "EOA", "ANSWER"})


def _get_stop_words() -> set[str]:
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_validator_pipe(model_path: str, device: str):
    """Load fine-tuned BERT validator on the requested device for LIME's black box."""
    resolved = str(resolve_path(model_path))
    logger.info(f"Loading validator for LIME: {resolved} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    tokenizer.truncation_side = "left"
    return pipeline(
        "text-classification",
        model=resolved,
        tokenizer=tokenizer,
        top_k=None,
        device=device,
    )


def _make_predict_proba(
    validator_pipe, label_map: dict[str, int], lime_batch_size: int
):
    """Wrap HuggingFace pipeline as the predict_proba expected by LIME.

    Returns shape (n_samples, 2).  LIME calls this with all num_samples perturbed
    texts at once; pipeline batches them internally at lime_batch_size.
    """
    n_classes = max(label_map.values()) + 1

    def predict_proba(texts) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        results = validator_pipe(
            texts, truncation=True, max_length=512, batch_size=lime_batch_size
        )
        probs = np.zeros((len(texts), n_classes))
        for i, score_list in enumerate(results):
            for score_dict in score_list:
                idx = label_map.get(score_dict["label"])
                if idx is not None:
                    probs[i, idx] = score_dict["score"]
        return probs

    return predict_proba


def load_processed_ids(output_path: Path) -> set:
    processed: set = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                processed.add(json.loads(line.strip())["id"])
    return processed


def run_lime(config_path: str, split: str) -> None:
    """Run per-sentence LIME attribution for one split and save results to JSONL.

    Each profile's cot_text is split into 4 steps × 2 sentences (8 sentences
    total) via [EORS] and [ES] delimiters.  One independent explain_instance
    call is made per sentence with structural tokens stripped from the input.
    Output is keyed by (step, sentence) to map directly to CF generator slots.

    Args:
        config_path: Path to a counterfactual YAML config (e.g. cf_ld1.yaml).
        split:       Dataset split — "train", "val", or "test".
    """
    config = load_config(config_path)
    sub_dataset: str = config["dataset"]["sub_dataset"]
    split_cfg: dict = config["dataset"]["splits"][split]
    input_path = resolve_path(split_cfg["cot_input"])
    output_path = resolve_path(split_cfg["lime_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lime_cfg: dict = config.get("lime", {})
    num_features: int = lime_cfg.get(
        "num_features", 20
    )  # per sentence; 30-70 token sentences need far fewer than full profiles
    num_samples: int = lime_cfg.get("num_samples", 512)
    lime_batch_size: int = lime_cfg.get("batch_size", 64)

    validator_config = _SUB_DATASET_VALIDATOR[sub_dataset]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validator_pipe = _load_validator_pipe(validator_config.bert_model_path, device)
    predict_proba = _make_predict_proba(
        validator_pipe, validator_config.label_map, lime_batch_size
    )

    explainer = LimeTextExplainer(bow=False)
    stop_words = _get_stop_words()

    processed_ids = load_processed_ids(output_path)
    if processed_ids:
        logger.info(f"Resuming — {len(processed_ids)} profiles already processed.")

    total = 0
    with open(input_path, "r") as f_in, open(output_path, "a") as f_out:
        for line in f_in:
            data_point = json.loads(line.strip())
            pid = data_point["id"]
            if pid in processed_ids:
                continue

            cot_text_only: str = data_point["cot_text"].split("[ANSWER]")[0].strip()

            # Split into steps then sentences — 8 independent LIME calls per profile.
            # Filter empty strings after each split to avoid phantom slots from trailing
            # delimiters (e.g. "…[EORS]".split("[EORS]") yields a trailing "").
            # NOTE: predict_proba feeds individual sentences to FinBERT/MedBERT which
            # was trained on full profiles — known distribution shift; token attribution
            # scores remain directionally meaningful despite the shorter input.
            steps = [s for s in cot_text_only.split("[EORS]") if s.strip()]
            sentence_features: list[dict] = []

            for step_idx, step in enumerate(steps):
                step_body = step.replace("[STEP]", "").strip()
                sentences = [s for s in step_body.split("[ES]") if s.strip()]

                for sent_idx, sentence in enumerate(sentences):
                    sentence_clean = sentence.strip()

                    try:
                        explanation = explainer.explain_instance(
                            sentence_clean,
                            predict_proba,
                            num_features=num_features,
                            num_samples=num_samples,
                            top_labels=1,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"ID {pid} step {step_idx} sent {sent_idx}: "
                            f"LIME failed — {exc}. Skipping."
                        )
                        sentence_features.append(
                            {"step": step_idx, "sentence": sent_idx, "top_features": []}
                        )
                        continue

                    top_label = explanation.top_labels[0]
                    local_exp = explanation.local_exp[top_label]
                    as_list: list[str] = list(
                        explanation.domain_mapper.indexed_string.as_list
                    )

                    features: list[dict] = []
                    for feature_id, weight in local_exp:
                        if weight <= 0:
                            continue
                        if feature_id >= len(as_list):
                            continue
                        token = as_list[feature_id]
                        if not token.strip() or token.lower() in stop_words:
                            continue
                        if all(c in string.punctuation for c in token.strip()):
                            continue
                        if token.strip() in STRUCTURAL_TOKENS:
                            continue
                        if re.match(r"^[\s\.\[\]\{\}\(\)\n]+$", token.strip()):
                            continue
                        features.append(
                            {
                                "token": token,
                                "score": float(weight),
                                "position_idx": int(
                                    feature_id
                                ),  # as_list index within sentence
                            }
                        )

                    assert all(f["score"] > 0 for f in features), (
                        f"ID {pid} step {step_idx} sent {sent_idx}: "
                        f"negative scores leaked — {features}"
                    )
                    features.sort(key=lambda x: x["score"], reverse=True)
                    sentence_features.append(
                        {
                            "step": step_idx,
                            "sentence": sent_idx,
                            "top_features": features,
                        }
                    )

            record = {"id": pid, "sentence_features": sentence_features}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            total += 1

            if total % 10 == 0:
                logger.info(f"LIME: processed {total} profiles...")

    logger.info(f"Done! {total} LIME attribution records saved to {output_path}")
