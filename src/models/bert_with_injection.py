"""
BertWithInjection: BERT-family student model with chain-of-thought teacher
hidden-state injection.

The teacher (Qwen3-4B) generates 4-step structured reasoning for each profile
and we extract the hidden state at the [EORS] token marking the end of each
step, yielding one 2560-dimensional vector per step.  Only the first 3 steps
are used here — the 4th is excluded for label-leakage reasons handled upstream.
Callers must pass ``step_states`` already sliced to shape ``(batch, 3, 2560)``.

At three points in BERT's 12-layer transformer stack — one per group of
consecutive layers — a projected, LayerNorm-normalised, and gated residual
derived from the corresponding teacher step state is added to the student's
hidden representation.  The gate parameters (``gate_1``, ``gate_2``,
``gate_3``) are scalar ``nn.Parameter`` tensors initialised to ``gate_init``
(default 0.1), so injection starts small and grows only where the teacher
signal is useful.

``injection_position`` controls injection timing:
  ``"post_group"`` — residual is added AFTER the last layer of each group,
                     so the group's own computation runs unmodified first.
  ``"pre_group"``  — residual is added BEFORE the first layer of each group,
                     conditioning the entire group's self-attention and FFN.

The injection formula for group index k (0-indexed):
    residual = gate_{k+1} * norm_{k+1}(proj_{k+1}(step_states[:, k, :]))
    hidden_states = hidden_states + residual.unsqueeze(1)  # broadcast seq_len

``BertWithInjection`` returns ``SequenceClassifierOutput`` and is a drop-in
replacement for ``AutoModelForSequenceClassification`` in HuggingFace
``Trainer`` / ``WeightedLossTrainer`` pipelines.
"""

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

TEACHER_HIDDEN_SIZE: int = 2560  # Qwen3-4B hidden dimension


class BertWithInjection(nn.Module):
    """BERT encoder with per-group teacher hidden-state injection.

    Args:
        bert_model_name:    HuggingFace model name or local path.  May point to
                            a ``BertForSequenceClassification`` checkpoint — the
                            classification head is silently ignored (HF will emit
                            a warning about unused weights; this is expected).
        num_labels:         Number of output classes.
        injection_position: ``"pre_group"`` or ``"post_group"``.
        gate_init:          Initial value for all three learnable scalar gates.
        layer_groups:       List of exactly 3 groups of 1-indexed BERT layer
                            numbers.  The position of each group in this list
                            (index 0, 1, or 2) determines which CoT step it is
                            paired with — group at index k uses
                            ``step_states[:, k, :]`` and
                            ``proj_{k+1}`` / ``norm_{k+1}`` / ``gate_{k+1}``.
                            Default: ``[[1,2,3,4],[5,6,7,8],[9,10,11,12]]``.
    """

    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        injection_position: str = "post_group",
        gate_init: float = 0.1,
        layer_groups: list[list[int]] | None = None,
    ) -> None:
        super().__init__()

        if injection_position not in ("pre_group", "post_group"):
            raise ValueError(
                f"injection_position must be 'pre_group' or 'post_group', "
                f"got {injection_position!r}"
            )

        self.bert = BertModel.from_pretrained(bert_model_name)
        # Some checkpoint formats (e.g. BiomedNLP-BiomedBERT stored in .bin)
        # save weight matrices in column-major order, producing non-contiguous
        # tensors with stride (1, D) instead of (D, 1). Computation is
        # numerically identical (PyTorch handles strides correctly), but
        # safetensors.save_file() rejects non-contiguous tensors, which crashes
        # every save_strategy="epoch" checkpoint write. Normalise at init so
        # the layout is always C-contiguous regardless of checkpoint format.
        for param in self.bert.parameters():
            param.data = param.data.contiguous()

        hidden_size: int = self.bert.config.hidden_size

        self.injection_position = injection_position
        self.layer_groups: list[list[int]] = (
            layer_groups
            if layer_groups is not None
            else [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        )

        # Precompute: 1-indexed layer number → group index (0, 1, 2)
        # Used in forward() to avoid per-step list scanning.
        self._pre_inject_at: dict[int, int] = {
            group[0]: k for k, group in enumerate(self.layer_groups)
        }
        self._post_inject_at: dict[int, int] = {
            group[-1]: k for k, group in enumerate(self.layer_groups)
        }

        # Injection components — one set per CoT step.
        # Explicitly named (not ModuleList) for easy per-gate logging and debugging.
        self.proj_1 = nn.Linear(TEACHER_HIDDEN_SIZE, hidden_size)
        self.proj_2 = nn.Linear(TEACHER_HIDDEN_SIZE, hidden_size)
        self.proj_3 = nn.Linear(TEACHER_HIDDEN_SIZE, hidden_size)

        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.norm_3 = nn.LayerNorm(hidden_size)

        # Scalar gates: start near zero so early injection has minimal effect.
        self.gate_1: nn.Parameter = nn.Parameter(torch.tensor(gate_init))
        self.gate_2: nn.Parameter = nn.Parameter(torch.tensor(gate_init))
        self.gate_3: nn.Parameter = nn.Parameter(torch.tensor(gate_init))

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    # ------------------------------------------------------------------
    # Injection helper
    # ------------------------------------------------------------------

    def _inject(
        self,
        hidden_states: torch.Tensor,
        step_states: torch.Tensor,
        group_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and add the teacher residual for a given group.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``
            step_states:   ``(batch, 3, TEACHER_HIDDEN_SIZE)`` — full tensor;
                           this method reads ``[:, group_idx, :]``.
            group_idx:     0-indexed position in ``self.layer_groups``.

        Returns:
            ``(updated_hidden_states, residual)`` — residual has shape
            ``(batch, hidden_size)`` and is exposed for norm diagnostics.
        """
        j = group_idx  # 0-indexed step; attributes are 1-indexed
        proj = getattr(self, f"proj_{j + 1}")
        norm = getattr(self, f"norm_{j + 1}")
        gate = getattr(self, f"gate_{j + 1}")

        # (batch, hidden_size) — project, normalise, scale by gate
        residual = gate * norm(proj(step_states[:, j, :]))
        # Broadcast uniformly over seq_len: (batch, 1, hidden_size)
        return hidden_states + residual.unsqueeze(1), residual

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_injection_norms: bool = False,
    ) -> SequenceClassifierOutput | tuple:
        """Manual layer-by-layer BERT forward pass with teacher-state injection.

        Args:
            input_ids:      ``(batch, seq_len)`` token ids.
            attention_mask: ``(batch, seq_len)`` — 1 for real tokens, 0 for padding.
            step_states:    ``(batch, 3, 2560)`` — teacher hidden states for the 3
                            CoT steps; must be pre-sliced to 3 by the caller.
            labels:         ``(batch,)`` int class indices.  If provided, a standard
                            cross-entropy loss is computed and returned.  During
                            ``WeightedLossTrainer`` training, ``labels`` are popped
                            before ``model(**inputs)`` is called, so ``labels=None``
                            is the normal training-time path — the weighted loss is
                            then computed externally from ``outputs.logits``.

        Returns:
            ``SequenceClassifierOutput`` with ``.loss`` (or ``None``) and
            ``.logits`` of shape ``(batch, num_labels)``.  Supports both attribute
            access (``outputs.logits``) and dict-style access (``outputs["logits"]``),
            matching ``AutoModelForSequenceClassification`` behaviour.
        """
        # 1. Embed input tokens (position + token-type ids use defaults)
        hidden_states: torch.Tensor = self.bert.embeddings(input_ids=input_ids)

        # 2. Build the additive attention bias expected by BertLayer
        extended_attn_mask: torch.Tensor = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        # 3. Manual loop over all transformer layers with conditional injection
        injection_norms: dict[str, float] = {}

        for i, layer_module in enumerate(self.bert.encoder.layer):
            layer_num = i + 1  # 1-indexed for group lookup
            # "pre" means we inject the residual signal at the beginning of each layer group

            if (
                self.injection_position == "pre_group"
                and layer_num in self._pre_inject_at
            ):
                group_idx = self._pre_inject_at[layer_num]
                if return_injection_norms:
                    injection_norms[f"group_{group_idx + 1}_hidden_norm_pre"] = (
                        hidden_states.norm(dim=-1).mean().item()
                    )
                hidden_states, residual = self._inject(
                    hidden_states, step_states, group_idx
                )
                if return_injection_norms:
                    injection_norms[f"group_{group_idx + 1}_residual_norm"] = (
                        residual.norm(dim=-1).mean().item()
                    )

            # transformers ≥5.0: BertLayer.forward returns a plain Tensor,
            # not a tuple — do NOT index with [0] or the batch dim is stripped.
            hidden_states = layer_module(
                hidden_states, attention_mask=extended_attn_mask
            )

            if (
                self.injection_position == "post_group"
                and layer_num in self._post_inject_at
            ):
                group_idx = self._post_inject_at[layer_num]
                if return_injection_norms:
                    injection_norms[f"group_{group_idx + 1}_hidden_norm_pre"] = (
                        hidden_states.norm(dim=-1).mean().item()
                    )
                hidden_states, residual = self._inject(
                    hidden_states, step_states, group_idx
                )
                if return_injection_norms:
                    injection_norms[f"group_{group_idx + 1}_residual_norm"] = (
                        residual.norm(dim=-1).mean().item()
                    )

        # 4. CLS pooling → dropout → linear classifier
        cls_output: torch.Tensor = hidden_states[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits: torch.Tensor = self.classifier(cls_output)

        # 5. Optional loss — fallback for standalone / direct use.
        #    WeightedLossTrainer computes its own loss from outputs.logits,
        #    so this branch is not exercised during normal training.
        loss: torch.Tensor | None = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        outputs = SequenceClassifierOutput(loss=loss, logits=logits)
        if return_injection_norms:
            return outputs, injection_norms
        return outputs

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def get_gate_values(self) -> dict[str, float]:
        """Return the current scalar gate values for monitoring / logging."""
        return {
            "gate_1": self.gate_1.item(),
            "gate_2": self.gate_2.item(),
            "gate_3": self.gate_3.item(),
        }
