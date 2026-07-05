"""
Fine-tune FinBERT (ProsusAI/finbert) on the FinBench LD1 dataset.
TO-DO: Rrefactor this file so it accepts any bert model for any dataset

All three dataset splits (train, val, test) are concatenated before
fine-tuning so the validator has seen the full data distribution.
A stratified 10% hold-out is reserved for in-training evaluation only
(accuracy / F1 monitoring); no data is withheld from learning.


The fine-tuned model and tokenizer are saved to:
  models_weights/finbert_finbench_ld1/

Usage
-----
    # from project root
    python src/models/finetune_finbert_ld1.py

    # with overrides
    python src/models/finetune_finbert_ld1.py \\
        --epochs 5 --batch-size 16 --lr 2e-5 --output-dir models_weights/my_run
"""

# PROBABLY REDUNDANAAAAAAAAAAAAAAAAAAAAAT (DO NOT REMOVE THIS IMPORTANT!)

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy HuggingFace / tokenizer warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger("finetune_finbert_ld1")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Configure wandb
load_dotenv()

ENTITY_NAME = os.getenv("WANDB_ENTITY")
PROJECT_NAME = os.getenv("WANDB_PROJECT")


FINBERT_MODEL_ID = "ProsusAI/finbert"

LABEL_TO_FINBERT_ID: dict[int, int] = {0: 0, 1: 1}

# Processed data paths relative to the project root
DATA_PATHS: dict[str, str] = {
    "train": "data/processed/finbench/ld1/ld1_processed_train.jsonl",
    "val": "data/processed/finbench/ld1/ld1_processed_val.jsonl",
    "test": "data/processed/finbench/ld1/ld1_processed_test.jsonl",
}

# ---------------------------------------------------------------------------
# Checkpoint clean up
# ---------------------------------------------------------------------------


def cleanup_model_weights(output_dir: str) -> None:
    """Keep only model.safetensors, config.json, tokenizer files and trainer_state.json.
    Also prints the best F1 macro the checkpoint was saved at.
    """

    # files to keep in the root output dir
    keep_files = {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
    }

    # copy model files out of checkpoint dir to root output dir
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            for filename in ["model.safetensors", "config.json", "trainer_state.json"]:
                src = os.path.join(item_path, filename)
                dst = os.path.join(output_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"Copied {filename} from {item}")
            # now safe to delete the checkpoint dir
            shutil.rmtree(item_path)
            print(f"Deleted checkpoint dir: {item}")

    # delete unnecessary files from root dir
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath) and filename not in keep_files:
            os.remove(filepath)
            print(f"Deleted: {filename}")

    # check best f1 macro from trainer_state.json
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, "r") as f:
            state = json.load(f)
        best_metric = state.get("best_metric", None)
        best_step = state.get("best_global_step", None)
        print(f"\nBest F1 macro : {best_metric:.4f}")
        print(f"Best global step : {best_step}")
    else:
        print("trainer_state.json not found!")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    """Read every line of a JSONL file and return parsed records."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_concatenated_data(
    data_paths: dict[str, str],
    project_root: Path,
) -> list[dict]:
    """Load and concatenate all FinBench LD1 splits.

    Args:
        data_paths: Mapping split_name → relative file path.
        project_root: Absolute path to the project root directory.

    Returns:
        List of record dicts, each containing ``id``, ``text``, and ``label``.
    """
    all_records: list[dict] = []
    label_counts = {0: 0, 1: 0}

    for split_name, rel_path in data_paths.items():
        abs_path = project_root / rel_path
        if not abs_path.exists():
            logger.warning(
                "Split '%s' not found at %s — skipping.", split_name, abs_path
            )
            continue

        records = _load_jsonl(str(abs_path))
        n_repaid = sum(1 for r in records if r["label"] == 0)
        n_defaulted = sum(1 for r in records if r["label"] == 1)
        label_counts[0] += n_repaid
        label_counts[1] += n_defaulted

        logger.info(
            "Loaded split '%-5s': %4d records  (repaid=%d | defaulted=%d)",
            split_name,
            len(records),
            n_repaid,
            n_defaulted,
        )
        all_records.extend(records)

    total = len(all_records)
    if total == 0:
        raise RuntimeError("No records loaded. Check DATA_PATHS and project root.")

    logger.info(
        "Concatenated dataset: %d records  (repaid=%d | defaulted=%d)",
        total,
        label_counts[0],
        label_counts[1],
    )
    logger.info(
        "Class balance — repaid: %.1f%%  |  defaulted: %.1f%%",
        100 * label_counts[0] / total,
        100 * label_counts[1] / total,
    )
    return all_records


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def stratified_split(
    records: list[dict],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Stratified random split that preserves the class ratio.

    Args:
        records: Full concatenated record list.
        eval_fraction: Fraction reserved for in-training evaluation.
        seed: Random seed for reproducibility.

    Returns:
        ``(train_records, eval_records)`` tuple.
    """
    rng = np.random.default_rng(seed)

    class_0 = [r for r in records if r["label"] == 0]
    class_1 = [r for r in records if r["label"] == 1]

    train_records: list[dict] = []
    eval_records: list[dict] = []

    for cls_records in (class_0, class_1):
        n_eval = max(1, int(len(cls_records) * eval_fraction))
        shuffled = rng.permutation(len(cls_records))
        eval_records.extend(cls_records[i] for i in shuffled[:n_eval])
        train_records.extend(cls_records[i] for i in shuffled[n_eval:])

    n_train_rep = sum(1 for r in train_records if r["label"] == 0)
    n_train_def = sum(1 for r in train_records if r["label"] == 1)
    n_eval_rep = sum(1 for r in eval_records if r["label"] == 0)
    n_eval_def = sum(1 for r in eval_records if r["label"] == 1)

    logger.info(
        "Train split: %d records  (repaid=%d | defaulted=%d)",
        len(train_records),
        n_train_rep,
        n_train_def,
    )
    logger.info(
        "Eval  split: %d records  (repaid=%d | defaulted=%d)  [%.0f%% held out]",
        len(eval_records),
        n_eval_rep,
        n_eval_def,
        100 * eval_fraction,
    )
    return train_records, eval_records


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class FinBenchDataset(Dataset):
    """PyTorch Dataset wrapping FinBench LD1 profiles.

    Maps our binary labels to FinBERT's 3-class internal IDs:
        0 (repaid)    → 2 (``"positive"``)
        1 (defaulted) → 0 (``"negative"``)

    Args:
        records: List of record dicts (keys: ``id``, ``text``, ``label``).
        tokenizer: Hugging Face tokenizer for FinBERT.
        max_length: Maximum token sequence length (default 512).
        label_map: Override for the default ``LABEL_TO_FINBERT_ID`` mapping.
    """

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int,
        label_map: dict[int, int] | None = None,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or LABEL_TO_FINBERT_ID

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        text = record["text"]
        label = self.label_map[
            record["label"]
        ]  # Maps our binary original label to finbert id2label. For now they are similar but we will use other BERT models later

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze the batch dimension added by return_tensors="pt"
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _compute_metrics(eval_pred) -> dict[str, float]:
    """Accuracy and macro-averaged F1 for the Trainer evaluation loop."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


# ---------------------------------------------------------------------------
# Class weights + weighted loss trainer
# ---------------------------------------------------------------------------


def _compute_class_weights(
    train_records: list[dict], class_weight: dict | None = None
) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training records.

    Uses sklearn's ``'balanced'`` strategy::

        weight_i = n_total / (n_classes * n_samples_i)

    This gives a higher weight to the minority class (defaulted, label 1)
    so the loss penalises misclassifying a defaulter more than misclassifying
    a repaid borrower.

    Args:
        train_records: Training records after the stratified split.

    Returns:
        Float tensor of shape ``(num_labels,)`` ready to pass directly to
        ``nn.CrossEntropyLoss(weight=...)``.
    """
    if class_weight is None:
        class_weight = "balanced"
    labels = np.array([r["label"] for r in train_records])
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight, classes=classes, y=labels)
    weight_tensor = torch.tensor(weights, dtype=torch.float)

    for cls, w in zip(classes, weights):
        logger.info(
            "Class weight — label %d (%s): %.4f",
            cls,
            "repaid" if cls == 0 else "defaulted",
            w,
        )
    return weight_tensor


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies per-class weights to ``CrossEntropyLoss``.

    The standard ``Trainer.compute_loss`` ignores class imbalance.  This
    subclass replaces it with a weighted cross-entropy so that errors on the
    minority class (defaulted, label 1) are penalised more heavily.

    The weight tensor is moved to the correct device on every forward pass,
    so it works transparently on both CPU and GPU.

    Args:
        class_weights: Float tensor of shape ``(num_labels,)`` — one weight
            per output class.
        *args / **kwargs: Forwarded verbatim to ``transformers.Trainer``.
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Custom TrainerCallBack to avoid overfitting
# ---------------------------------------------------------------------------


# class OverfitStoppingCallback(TrainerCallback):
#     def __init__(self, patience: int = 2):
#         self.patience = patience
#         self.overfit_count = 0

#     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
#         eval_loss = metrics.get("eval_loss", None)

#         # get train loss from the last logged value in state
#         train_loss = None
#         for log in reversed(state.log_history):
#             if "loss" in log and "eval_loss" not in log:
#                 train_loss = log["loss"]
#                 break

#         if eval_loss is not None and train_loss is not None:
#             if eval_loss > train_loss:
#                 self.overfit_count += 1
#                 logger.info(
#                     f"Overfit detected ({self.overfit_count}/{self.patience}): "
#                     f"eval_loss={eval_loss:.4f} > train_loss={train_loss:.4f}"
#                 )
#                 if self.overfit_count >= self.patience:
#                     logger.info("Overfit stopping triggered — blocking model save!")
#                     control.should_training_stop = True
#                     control.should_save = False
#             else:
#                 self.overfit_count = 0  # reset if gap closes

#         return control


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def finetune(
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    class_weight: dict | None,
    max_len: int,
    eval_split: float,
    seed: int,
    output_dir: str,
    project_root: Path | None = None,
    run_name: str = None,
) -> None:
    """Fine-tune FinBERT on the concatenated FinBench LD1 dataset.

    Args:
        epochs: Number of training epochs.
        batch_size: Per-device batch size for both training and evaluation.
        lr: Peak AdamW learning rate.
        warmup_ratio: Fraction of total steps used for LR warm-up.
        class_weight: The weights of each class to calculated a weighted training loss. If None, the 'balanced' class weights of scikit learn are used
        max_len: Maximum tokenizer sequence length.
        eval_split: Fraction of concatenated data reserved for evaluation
            during training (monitoring only — not withheld from learning).
        seed: Random seed for data shuffling and training.
        output_dir: Directory to save the fine-tuned model and tokenizer.
        project_root: Absolute path to the project root. Defaults to the
            directory two levels above this file.
        run_name: Name of the train run to save in wandb and as a checkpoint to load the model
    #"""
    if run_name is None:
        print("PLease provide the name of your run to save, e.g. --run-name run1 \n")
        print("Exiting program...")
        return None

    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    abs_output_dir = str(project_root / output_dir)

    # Start a new wandb run to track this script. TODO: Maybe change this later to not track the run all the time
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=ENTITY_NAME,
        # Set the wandb project where this run will be logged.
        project=PROJECT_NAME,
        name=run_name,
    )

    logger.info("=" * 60)
    logger.info("FinBERT fine-tuning — FinBench LD1")
    logger.info("=" * 60)
    logger.info("  Base model    : %s", FINBERT_MODEL_ID)
    logger.info("  Output dir    : %s", abs_output_dir)
    logger.info("  Epochs        : %d", epochs)
    logger.info("  Batch size    : %d", batch_size)
    logger.info("  Learning rate : %.2e", lr)
    logger.info("  Warmup ratio  : %.0f%%", 100 * warmup_ratio)
    logger.info("  Max seq len   : %d tokens", max_len)
    logger.info("  Eval split    : %.0f%%", 100 * eval_split)
    logger.info("  Random seed   : %d", seed)
    logger.info("  Device        : %s", "GPU" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        logger.info("  GPU           : %s", torch.cuda.get_device_name(0))
    logger.info("=" * 60)

    # ------------------------------------------------------------------ data
    all_records = load_concatenated_data(DATA_PATHS, project_root)
    train_records, eval_records = stratified_split(all_records, eval_split, seed)

    # Compute class weights from the training split so the loss penalises
    # errors on the minority class (defaulted, label 1) more heavily.
    logger.info("Computing class weights from training split...")
    # Convert the keys to integers
    class_weight = {int(k): v for k, v in class_weight.items()}
    class_weights = _compute_class_weights(train_records, class_weight=class_weight)

    # --------------------------------------------------------- model / tokenizer
    logger.info("Loading tokenizer and model from '%s'...", FINBERT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL_ID,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %.1fM total params  |  %.1fM trainable",
        total_params / 1e6,
        trainable_params / 1e6,
    )
    logger.info("Label schema (FinBERT id → label): %s", model.config.id2label)

    # ---------------------------------------------------------- dataset objects
    train_dataset = FinBenchDataset(train_records, tokenizer, max_length=max_len)
    eval_dataset = FinBenchDataset(eval_records, tokenizer, max_length=max_len)
    logger.info(
        "Dataset objects created — train: %d  |  eval: %d",
        len(train_dataset),
        len(eval_dataset),
    )

    # -------------------------------------------------- training arguments
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    logging_steps = max(1, steps_per_epoch // 4)  # log ~4× per epoch

    logger.info(
        "Steps: total=%d  |  warmup=%d  |  steps/epoch=%d  |  logging every %d steps",
        total_steps,
        warmup_steps,
        steps_per_epoch,
        logging_steps,
    )

    training_args = TrainingArguments(
        output_dir=abs_output_dir,
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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=os.path.join(abs_output_dir, "logs"),
        logging_steps=logging_steps,
        save_total_limit=1,  # keep only 1 checkpoints on disk
        seed=seed,
        report_to="wandb",  # disable W&B / MLflow / TensorBoard
        fp16=torch.cuda.is_available(),
    )

    # -----------------------------------------------------------------  trainer
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            # OverfitStoppingCallback(patience=2),
        ],
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    # -----------------------------------------------------------  log summary
    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("  Global steps   : %d", train_result.global_step)
    logger.info("  Training loss  : %.4f", train_result.training_loss)
    logger.info("=" * 60)

    # ---------------------------------------------------------- save artifacts
    Path(abs_output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(abs_output_dir)
    tokenizer.save_pretrained(abs_output_dir)
    logger.info("Model and tokenizer saved to %s", abs_output_dir)

    # ----------------------------------------------------------- final eval
    # logger.info("Running final evaluation on held-out eval split...")
    # eval_results = trainer.evaluate()
    # logger.info("Final eval metrics:")
    # for key, value in sorted(eval_results.items()):
    #     if isinstance(value, float):
    #         logger.info("  %-35s %.4f", key, value)
    #     else:
    #         logger.info("  %-35s %s", key, value)

    logger.info("Done. Fine-tuned model ready at: %s", abs_output_dir)

    # Save the model weights and clean up unecessary files
    save_dir = os.path.join(abs_output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    trainer.save_model(abs_output_dir)
    cleanup_model_weights(abs_output_dir)

    # move cleaned files to your custom named folder
    for filename in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "trainer_state.json",
    ]:
        src = os.path.join(abs_output_dir, filename)
        dst = os.path.join(save_dir, filename)
        if os.path.exists(src):
            shutil.move(src, dst)

    run.finish()
