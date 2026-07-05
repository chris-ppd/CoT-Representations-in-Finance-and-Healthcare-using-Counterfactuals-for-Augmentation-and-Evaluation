"""
Smoke tests for BertWithInjection.

Run with:
    pytest test/test_bert_with_injection.py -v
or standalone:
    python test/test_bert_with_injection.py

Requires an internet connection on first run to download bert-base-uncased;
subsequent runs use the local HuggingFace cache.  No GPU required.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.bert_with_injection import BertWithInjection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BERT_MODEL = "bert-base-uncased"
BATCH_SIZE = 4
SEQ_LEN = 32
NUM_LABELS = 2
GATE_INIT = 0.1
VOCAB_SIZE = 30522  # bert-base-uncased vocabulary size


# ---------------------------------------------------------------------------
# Shared input factory
# ---------------------------------------------------------------------------


def _make_inputs(seed: int = 0) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return (input_ids, attention_mask, step_states, labels) from a fixed seed."""
    torch.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long)
    step_states = torch.randn(BATCH_SIZE, 3, 2560)
    labels = torch.randint(0, NUM_LABELS, (BATCH_SIZE,))
    return input_ids, attention_mask, step_states, labels


# ---------------------------------------------------------------------------
# Core forward + backward test (run for both injection positions)
# ---------------------------------------------------------------------------


def _run_forward_backward(injection_position: str) -> BertWithInjection:
    """Instantiate model, run a forward+backward pass, assert correctness.

    Returns the model so the caller can inspect gate values after backward.
    """
    model = BertWithInjection(
        BERT_MODEL,
        num_labels=NUM_LABELS,
        injection_position=injection_position,
        gate_init=GATE_INIT,
    )
    model.train()

    input_ids, attention_mask, step_states, labels = _make_inputs(seed=0)

    out = model(input_ids, attention_mask, step_states, labels)

    # --- shape / loss checks ---
    assert out.logits.shape == (BATCH_SIZE, NUM_LABELS), (
        f"[{injection_position}] Expected logits shape ({BATCH_SIZE}, {NUM_LABELS}), "
        f"got {out.logits.shape}"
    )
    assert out.loss is not None, (
        f"[{injection_position}] loss should not be None when labels are provided"
    )
    assert out.loss.shape == (), (
        f"[{injection_position}] loss must be a scalar tensor, got shape {out.loss.shape}"
    )
    assert out.loss.requires_grad, (
        f"[{injection_position}] loss.requires_grad must be True"
    )

    # --- dict-style vs attribute access parity ---
    assert torch.equal(out.logits, out["logits"]), (
        f"[{injection_position}] outputs['logits'] and outputs.logits must be identical"
    )

    # --- backward pass ---
    out.loss.backward()

    # All THREE injection points must receive gradients.
    # An off-by-one in the layer_groups loop could silently leave points 2 or 3
    # unreached — checking all three is the only way to catch that.
    for k in (1, 2, 3):
        proj_grad = getattr(model, f"proj_{k}").weight.grad
        gate_grad = getattr(model, f"gate_{k}").grad
        norm_grad = getattr(model, f"norm_{k}").weight.grad

        assert proj_grad is not None, (
            f"[{injection_position}] proj_{k}.weight.grad is None — "
            f"injection point {k} was not reached or is disconnected from the loss"
        )
        assert gate_grad is not None, (
            f"[{injection_position}] gate_{k}.grad is None — "
            f"injection point {k} was not reached or is disconnected from the loss"
        )
        assert norm_grad is not None, (
            f"[{injection_position}] norm_{k}.weight.grad is None — "
            f"injection point {k} was not reached or is disconnected from the loss"
        )

    # --- gate values unchanged (no optimiser step was taken) ---
    gates = model.get_gate_values()
    for name, value in gates.items():
        assert abs(value - GATE_INIT) < 1e-5, (
            f"[{injection_position}] Expected {name} ≈ {GATE_INIT} "
            f"(no optimiser step taken), got {value}"
        )

    return model


def test_post_group_forward_backward() -> None:
    print("\n--- post_group: forward + backward ---")
    model = _run_forward_backward("post_group")
    print(f"  gate values: {model.get_gate_values()}")
    print("  PASSED")


def test_pre_group_forward_backward() -> None:
    print("\n--- pre_group: forward + backward ---")
    model = _run_forward_backward("pre_group")
    print(f"  gate values: {model.get_gate_values()}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Invalid injection_position raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_injection_position_raises() -> None:
    print("\n--- invalid injection_position raises ValueError ---")
    try:
        BertWithInjection(BERT_MODEL, num_labels=2, injection_position="mid_group")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        print(f"  ValueError raised as expected: {exc}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# pre_group vs post_group must produce different logits
# ---------------------------------------------------------------------------


def test_pre_post_produce_different_logits() -> None:
    """Verify the injection_position flag is not silently ignored.

    Both models receive identical weights (via state_dict copy) and the same
    inputs.  The only difference is when the injection is applied relative to
    each layer group.  If the two positions produce identical logits, one branch
    is a no-op or the two branches are accidentally identical.
    """
    print("\n--- pre_group vs post_group: logits must differ ---")

    # Instantiate with post_group, copy state dict to pre_group model so that
    # the only difference is injection_position (not random weight init).
    model_post = BertWithInjection(
        BERT_MODEL, num_labels=NUM_LABELS, injection_position="post_group", gate_init=GATE_INIT
    )
    model_pre = BertWithInjection(
        BERT_MODEL, num_labels=NUM_LABELS, injection_position="pre_group", gate_init=GATE_INIT
    )
    # Load identical weights — injection_position is a plain attribute, not in state_dict
    model_pre.load_state_dict(model_post.state_dict())

    model_pre.eval()
    model_post.eval()

    input_ids, attention_mask, step_states, _ = _make_inputs(seed=0)

    with torch.no_grad():
        out_pre = model_pre(input_ids, attention_mask, step_states)
        out_post = model_post(input_ids, attention_mask, step_states)

    assert not torch.allclose(out_pre.logits, out_post.logits), (
        "pre_group and post_group produced identical logits despite using the same "
        "inputs and weights — one branch may be a no-op or both branches are "
        "accidentally equivalent"
    )

    max_diff = (out_pre.logits - out_post.logits).abs().max().item()
    print(f"  pre vs post logit max absolute difference: {max_diff:.6f}")
    print("  PASSED")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_post_group_forward_backward()
    test_pre_group_forward_backward()
    test_invalid_injection_position_raises()
    test_pre_post_produce_different_logits()
    print("\nAll tests passed.")
