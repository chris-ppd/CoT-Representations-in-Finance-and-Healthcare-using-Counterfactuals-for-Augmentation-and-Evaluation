"""
Data utilities for Experiment 2 (hidden-state injection into BERT).

Provides four public functions:

  load_hidden_states   — Load a preallocated .pt tensor and its .jsonl index,
                         returning only entries where extraction_ok is True.

  get_step_states      — Resolve a single profile entry (from any of the three
                         jsonl shapes used in this project) to a (3, 2560) tensor
                         of teacher CoT hidden states.

  validate_dataset     — Dry-run check: attempt to resolve every entry in a jsonl
                         file, collect failures, and print a human-readable summary.

  check_gate_init      — Duck-typed gate-value checker for BertWithInjection
                         (no model import — reads via get_gate_values()).

No dependency on src/models/bert_with_injection.py — pure data plumbing.
"""

import json
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# load_hidden_states
# ---------------------------------------------------------------------------


def load_hidden_states(
    index_path: str,
    tensor_path: str,
) -> tuple[dict[str, int], torch.Tensor]:
    """Load a preallocated hidden-state tensor and its profile-id row index.

    Args:
        index_path:  Path to a .jsonl index file.  Each line has at minimum:
                     {"profile_id": str, "profile_row_idx": int, "extraction_ok": bool}
        tensor_path: Path to a .pt file containing shape (n_profiles, 4, 2560).

    Returns:
        (id_to_row, tensor) — id_to_row maps profile_id str → row int and
        contains ONLY entries where extraction_ok is True.  Entries with
        extraction_ok=False are skipped and logged at WARNING level.
    """
    tensor: torch.Tensor = torch.load(tensor_path, weights_only=True)

    id_to_row: dict[str, int] = {}
    with open(index_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            profile_id = str(entry["profile_id"])
            if entry.get("extraction_ok", False):
                id_to_row[profile_id] = int(entry["profile_row_idx"])
            else:
                logger.warning(
                    "Skipping profile_id %r (extraction_ok=False) in index %s",
                    profile_id,
                    index_path,
                )

    return id_to_row, tensor


# ---------------------------------------------------------------------------
# get_step_states
# ---------------------------------------------------------------------------


def get_step_states(
    entry: dict,
    cot_index: dict[str, int],
    cf_index: Optional[dict[str, int]],
    cot_states: torch.Tensor,
    cf_cot_states: Optional[torch.Tensor],
) -> torch.Tensor:
    """Resolve a single profile entry to its first 3 teacher CoT step hidden states.

    Three different jsonl formats are handled transparently:

    * cot_{split}.jsonl          — {"id": "0", ...}
                                   Real profile; no is_counterfactual field.
    * counterfactual_{split}.jsonl — {"id": "cf_0", "original_id": 0, ...}
                                   CF profile; NO is_counterfactual field;
                                   identified by the "cf_"-prefixed id.
    * augmented_original_{split}.jsonl — {"id": "0"|"cf_0", "is_counterfactual": bool, ...}
                                   Explicit flag; CF rows also carry original_id.

    The dispatch condition is an OR of both signals so that every format is
    covered without relying on a single field being present in all files:

        is_cf = entry.get("is_counterfactual", False)
             or str(entry.get("id", "")).startswith("cf_")

    Args:
        entry:         Single profile dict from any of the above jsonl shapes.
        cot_index:     id_to_row for real CoT profiles (from load_hidden_states).
        cf_index:      id_to_row for CF profiles, or None if the caller has no CF data.
        cot_states:    (n_cot, 4, 2560) tensor for real profiles.
        cf_cot_states: (n_cf, 4, 2560) tensor for CF profiles, or None.

    Returns:
        Tensor of shape (3, 2560) — first 3 CoT step states (step 4 excluded
        upstream to prevent label leakage).

    Raises:
        ValueError: CF-shaped entry encountered but cf_index / cf_cot_states is None.
        KeyError:   Resolved lookup_id is absent from the appropriate index dict.
    """
    is_cf: bool = entry.get("is_counterfactual", False) or str(
        entry.get("id", "")
    ).startswith("cf_")

    if is_cf:
        lookup_id = str(entry["original_id"])
        if cf_index is None or cf_cot_states is None:
            raise ValueError(
                f"Entry id={entry.get('id')!r} (original_id={entry.get('original_id')!r}) "
                f"is a counterfactual profile but cf_index / cf_cot_states were not provided."
            )
        if lookup_id not in cf_index:
            raise KeyError(
                f"original_id {lookup_id!r} not found in cf_index "
                f"(entry id={entry.get('id')!r}, original_id={entry.get('original_id')!r})"
            )
        row_idx = cf_index[lookup_id]
        return cf_cot_states[row_idx, :3, :]
    else:
        lookup_id = str(entry["id"])
        if lookup_id not in cot_index:
            raise KeyError(
                f"id {lookup_id!r} not found in cot_index "
                f"(entry id={entry.get('id')!r})"
            )
        row_idx = cot_index[lookup_id]
        return cot_states[row_idx, :3, :]


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------


def validate_dataset(
    jsonl_path: str,
    cot_index: dict[str, int],
    cf_index: Optional[dict[str, int]],
    cot_states: torch.Tensor,
    cf_cot_states: Optional[torch.Tensor],
    original_jsonl_path: Optional[str] = None,
) -> dict:
    """Dry-run: attempt to resolve every entry in a jsonl file, report failures.

    Does not raise on individual resolution failures — catches KeyError,
    ValueError, and TypeError per entry and accumulates them in the returned
    summary.  The whole run fails fast only for genuine I/O errors.

    Args:
        jsonl_path:          Path to the jsonl file to validate.
        cot_index:           id_to_row for real profiles.
        cf_index:            id_to_row for CF profiles, or None.
        cot_states:          Tensor for real profiles.
        cf_cot_states:       Tensor for CF profiles, or None.
        original_jsonl_path: If provided, count its entries for attrition reporting.

    Returns:
        {
            "jsonl_path":        str,
            "n_entries":         int,
            "n_resolved":        int,
            "n_unresolved":      int,
            "unresolved":        [{"id": ..., "original_id": ..., "error": str}, ...],
            "shape_check_passed": bool,   # True iff every resolved entry had shape (3, 2560)
            "attrition":         {"original_count": int, "this_count": int} | None,
        }
    """
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]

    n_entries = len(entries)
    n_resolved = 0
    n_unresolved = 0
    unresolved: list[dict] = []
    shape_check_passed = True

    for entry in entries:
        try:
            states = get_step_states(
                entry, cot_index, cf_index, cot_states, cf_cot_states
            )
            if tuple(states.shape) != (3, 2560):
                shape_check_passed = False
            n_resolved += 1
        except (KeyError, ValueError, TypeError) as exc:
            n_unresolved += 1
            unresolved.append(
                {
                    "id": entry.get("id"),
                    "original_id": entry.get("original_id"),
                    "error": str(exc),
                }
            )

    attrition: Optional[dict] = None
    if original_jsonl_path is not None:
        with open(original_jsonl_path, "r", encoding="utf-8") as fh:
            original_count = sum(1 for line in fh if line.strip())
        attrition = {"original_count": original_count, "this_count": n_entries}

    # Human-readable summary
    resolved_mark = "✓" if n_unresolved == 0 else "✗"
    shape_mark = "✓" if shape_check_passed else "✗"
    print(f"[{jsonl_path}]")
    print(
        f"  {resolved_mark} {n_resolved}/{n_entries} profiles resolved to a valid hidden state row"
    )
    print(f"  {shape_mark} shape (3, 2560) confirmed for all resolved profiles")
    if attrition is not None:
        delta = attrition["original_count"] - attrition["this_count"]
        print(
            f"    original: {attrition['original_count']} -> "
            f"{attrition['this_count']} ({delta} augmented)"
        )
    if unresolved:
        id_strs = [str(u["id"]) for u in unresolved[:20]]
        suffix = f" ... and {len(unresolved) - 20} more" if len(unresolved) > 20 else ""
        print(f"    UNRESOLVED IDs: [{', '.join(id_strs)}{suffix}]")

    return {
        "jsonl_path": jsonl_path,
        "n_entries": n_entries,
        "n_resolved": n_resolved,
        "n_unresolved": n_unresolved,
        "unresolved": unresolved,
        "shape_check_passed": shape_check_passed,
        "attrition": attrition,
    }


# ---------------------------------------------------------------------------
# check_gate_init
# ---------------------------------------------------------------------------


def check_gate_init(
    model,
    expected_init: float,
    tol: float = 1e-6,
) -> dict[str, bool]:
    """Verify BertWithInjection gate parameters are close to their initial value.

    Duck-typed: calls model.get_gate_values() rather than importing the class,
    so this file has no dependency on src/models/bert_with_injection.py.

    Args:
        model:         Object with a get_gate_values() method returning
                       {"gate_1": float, "gate_2": float, "gate_3": float}.
        expected_init: Expected initial value (e.g. 0.1).
        tol:           Absolute tolerance (default 1e-6).

    Returns:
        {"gate_1": bool, "gate_2": bool, "gate_3": bool} — True if within tol.
    """
    gate_values: dict[str, float] = model.get_gate_values()
    results: dict[str, bool] = {}
    for name, value in gate_values.items():
        ok = abs(value - expected_init) < tol
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name} = {value:.8f} (expected {expected_init})")
        results[name] = ok
    return results
