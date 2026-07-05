"""
Forward-hook pipeline for extracting last-layer hidden states from Qwen3
at the [EORS] (end-of-reasoning-step) and [EOA] (end-of-answer) token positions
during batched CoT generation.

Public API
----------
HookBuffer                  Accumulates per-profile hidden states for every
                            decoding step across a full batch.
extract_eors_eoa_states()   Maps one profile's buffer contents to a (4, hidden_dim)
                            CoT tensor and a (hidden_dim,) answer tensor, with all
                            sanity checks.  Asserts exactly 5 extraction points:
                            4 × [EORS] + 1 × [EOA].
save_hidden_states_temp()   Saves .pt + _answer.pt + _meta.json to a temp directory.
promote_to_final()          Moves files from temp/ to {split}}/ after CoT validation.
discard_from_temp()         Deletes temp files for a rejected profile.
delete_per_profile_files()  Removes per-profile .pt, _answer.pt, _meta.json from
                            final/ after the profile is written into the consolidated
                            tensors.
update_master_index()       Appends one entry to the master index .jsonl.
                            References the consolidated tensor paths and the row
                            index within them; does NOT reference per-profile files.

Batched hook design
-------------------
The hook attaches to model.model.layers[-1] and fires on EVERY forward pass.

  Prefill pass  (prompt processing):
      hidden shape = (batch_size, prompt_seq_len, hidden_dim)
      prompt_seq_len > 1  → skip, nothing stored.

  Decoding steps (autoregressive, one token at a time with KV cache):
      hidden shape = (batch_size, 1, hidden_dim)
      seq_len == 1  → capture and store hidden[:, 0, :] as shape
                       (batch_size, hidden_dim) on CPU.

After model.generate() returns, step_hidden_states is a list of length
n_decoding_steps, each element a (batch_size, hidden_dim) CPU float32 tensor.

Per-profile extraction
----------------------
For profile b in the batch:
  1. Read all_generated_ids[b] — the raw generated token IDs (includes EOS).
  2. Truncate at the first EOS/PAD token to get real_ids.
  3. Find positions in real_ids matching [EORS] and [EOA] BPE sequences.
  4. Assert exactly 4 [EORS] positions and 1 [EOA] position (5 total).
  5. For each [EORS] position p, retrieve step_hidden_states[p][b] → (hidden_dim,).
     Stack into (4, hidden_dim) CoT states tensor.
  6. For [EOA] position q, retrieve step_hidden_states[q][b] → (hidden_dim,) answer
     state tensor.  This encodes the teacher's final decision signal, absent from the
     CoT reasoning steps due to the forbidden-phrases constraint.

Post-EOS steps are present in step_hidden_states but will never contain [EORS] or
[EOA], so the position-based lookup is safe for variable-length batches.
"""

import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hook buffer
# ---------------------------------------------------------------------------


class HookBuffer:
    """Accumulates last-layer hidden states for each decoding step of a batch.

    Stores one (batch_size, hidden_dim) CPU float32 tensor per decoding step.
    Reset before every generation call to prevent state bleed between batches
    or retry attempts.
    """

    def __init__(self) -> None:
        # List length = n_decoding_steps; each element shape (batch_size, hidden_dim)
        self.step_hidden_states: list[torch.Tensor] = []
        self._handle = None

    def reset(self) -> None:
        """Clear the buffer.  Call before every model.generate() call."""
        self.step_hidden_states = []

    def attach(self, model) -> None:
        """Register the forward hook on the last transformer layer."""
        last_layer = model.model.layers[-1]
        self._handle = last_layer.register_forward_hook(self._hook_fn)

    def detach(self) -> None:
        """Remove the hook.  Always call after generation to prevent leaks."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook_fn(self, _module, _input, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden shape: (batch_size, seq_len, hidden_dim)
        # seq_len == 1 only during autoregressive decoding (not the prefill pass).
        if hidden.shape[1] == 1:
            # Store (batch_size, hidden_dim) on CPU in float32 to avoid
            # accumulating GPU tensors across the full generation loop.
            self.step_hidden_states.append(hidden[:, 0, :].detach().cpu().float())

    def get_profile_states(self, profile_idx: int, n_steps: int) -> list[torch.Tensor]:
        """Return hidden states for one profile across the first n_steps.

        Args:
            profile_idx: Index of the profile within the batch (0-based).
            n_steps:     Number of real decoding steps for this profile
                         (i.e. generated length before EOS).

        Returns:
            List of length n_steps; each element is a (hidden_dim,) tensor.
        """
        return [step[profile_idx] for step in self.step_hidden_states[:n_steps]]


# ---------------------------------------------------------------------------
# Hidden state extraction (single profile, called per-profile after generation)
# ---------------------------------------------------------------------------


def _find_extraction_positions(
    real_ids: list[int],
    eors_sequences: tuple[tuple[int, ...], ...],
    eoa_sequences: tuple[tuple[int, ...], ...],
) -> tuple[list[int], list[int]]:
    """Scan real_ids for [EORS] and [EOA] BPE token sequences.

    Accepts multiple candidate sequences for each marker because BPE merges a
    preceding space into the opening bracket (▁[ vs [), so the same text token
    has different IDs depending on context.  Both bare and space-prefixed forms
    are tried at every position; after a match the scanner advances past it to
    prevent double-counting.

    Returns:
        eors_positions  — list of last-token indices for each [EORS] match.
        eoa_positions   — list of last-token indices for each [EOA] match.
    """
    eors_positions: list[int] = []
    eoa_positions: list[int] = []

    i = 0
    while i < len(real_ids):
        matched = False
        for seq in eors_sequences:
            n = len(seq)
            if i + n <= len(real_ids) and tuple(real_ids[i : i + n]) == seq:
                eors_positions.append(i + n - 1)
                i += n
                matched = True
                break
        if not matched:
            for seq in eoa_sequences:
                n = len(seq)
                if i + n <= len(real_ids) and tuple(real_ids[i : i + n]) == seq:
                    eoa_positions.append(i + n - 1)
                    i += n
                    matched = True
                    break
        if not matched:
            i += 1

    return eors_positions, eoa_positions


def extract_eors_eoa_states(
    hook_buffer: HookBuffer,
    profile_idx: int,
    generated_ids: list[int],
    eors_sequences: tuple[tuple[int, ...], ...],
    eoa_sequences: tuple[tuple[int, ...], ...],
    eos_token_ids: set[int],
    hidden_dim: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    list[int] | None,
    int | None,
    str | None,
]:
    """Extract the 4 [EORS] and 1 [EOA] hidden states for one profile.

    Args:
        hook_buffer:    Buffer populated during model.generate().
        profile_idx:    Index of this profile within the batch.
        generated_ids:  Raw generated token IDs for this profile (may include
                        EOS and post-EOS padding).
        eors_sequences: Tuple of candidate token-ID sequences for "[EORS]".
                        Both bare and space-prefixed BPE forms should be passed.
        eoa_sequences:  Tuple of candidate token-ID sequences for "[EOA]".
                        Both bare and space-prefixed BPE forms should be passed.
        eos_token_ids:  Set of token IDs that signal end-of-generation
                        (eos_token_id, pad_token_id — None values removed).
        hidden_dim:     Expected hidden dimension (e.g. 2560 for Qwen3-4B).

    Returns:
        On success: (cot_states, answer_state, eors_positions, eoa_position, None)
            cot_states      — torch.Tensor shape (4, hidden_dim), float32.
                              Each row is the teacher's accumulated reasoning
                              state after completing one full step.
            answer_state    — torch.Tensor shape (hidden_dim,), float32.
                              Teacher's internal state at the moment of committing
                              to the final verdict.
            eors_positions  — list of 4 flat decoding-step indices (last token
                              of each [EORS]).
            eoa_position    — flat decoding-step index (last token of [EOA]).
        On failure: (None, None, None, None, error_message)
    """
    # Truncate at first EOS/PAD to find the real generated content
    eos_pos = next(
        (i for i, tid in enumerate(generated_ids) if tid in eos_token_ids),
        len(generated_ids),
    )
    real_ids = generated_ids[:eos_pos]
    n_real = len(real_ids)

    if n_real > len(hook_buffer.step_hidden_states):
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: n_real={n_real} exceeds "
                f"captured steps={len(hook_buffer.step_hidden_states)}."
            ),
        )

    eors_positions, eoa_positions = _find_extraction_positions(
        real_ids, eors_sequences, eoa_sequences
    )

    if len(eors_positions) != 4:
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: expected 4 [EORS] tokens, found {len(eors_positions)}."
            ),
        )

    if len(eoa_positions) != 1:
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: expected 1 [EOA] token, found {len(eoa_positions)}."
            ),
        )

    eoa_position = eoa_positions[0]

    # Ordering sanity check: [EOA] must follow the last [EORS]
    if eoa_position <= eors_positions[-1]:
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: [EOA] position {eoa_position} does not follow "
                f"last [EORS] position {eors_positions[-1]}."
            ),
        )

    profile_states = hook_buffer.get_profile_states(profile_idx, n_real)

    # Extract CoT hidden states at each [EORS] position
    try:
        cot_states = torch.stack(
            [profile_states[pos] for pos in eors_positions]
        )  # (4, hidden_dim)
    except IndexError as exc:
        return (
            None,
            None,
            None,
            None,
            (f"Profile {profile_idx}: index error stacking [EORS] states — {exc}"),
        )

    # Extract answer hidden state at [EOA] position
    try:
        answer_state = profile_states[eoa_position]  # (hidden_dim,)
    except IndexError as exc:
        return (
            None,
            None,
            None,
            None,
            (f"Profile {profile_idx}: index error accessing [EOA] state — {exc}"),
        )

    # Shape checks
    if cot_states.shape != (4, hidden_dim):
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: unexpected CoT states shape {tuple(cot_states.shape)}, "
                f"expected (4, {hidden_dim})."
            ),
        )
    if answer_state.shape != (hidden_dim,):
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: unexpected answer state shape {tuple(answer_state.shape)}, "
                f"expected ({hidden_dim},)."
            ),
        )

    # No zero-norm vectors across all 5 extraction points
    all_states = torch.cat(
        [cot_states, answer_state.unsqueeze(0)], dim=0
    )  # (5, hidden_dim)
    norms = all_states.norm(dim=-1)
    if (norms == 0).any():
        bad = norms.eq(0).nonzero(as_tuple=True)[0].tolist()
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: zero-norm hidden state(s) at extraction point(s) {bad}."
            ),
        )

    # No duplicate vectors (pairwise L2 across all 5 points)
    dists = F.pdist(all_states, p=2)
    if (dists < 1e-6).any():
        return (
            None,
            None,
            None,
            None,
            (
                f"Profile {profile_idx}: duplicate hidden state vectors detected "
                f"(pairwise L2 < 1e-6)."
            ),
        )

    return cot_states, answer_state, eors_positions, eoa_position, None


# ---------------------------------------------------------------------------
# Temp / final save helpers
# ---------------------------------------------------------------------------


def save_hidden_states_temp(
    profile_id: int,
    cot_states: torch.Tensor,
    answer_state: torch.Tensor,
    eors_positions: list[int],
    eoa_position: int,
    temp_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save CoT states, answer state, and metadata to the temp directory.

    Files written:
        {profile_id}.pt         — (4, hidden_dim) float32 tensor (CoT states).
        {profile_id}_answer.pt  — (hidden_dim,) float32 tensor (answer state).
        {profile_id}_meta.json  — extraction metadata.

    Returns (pt_path, answer_pt_path, meta_path).
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    pt_path = temp_dir / f"{profile_id}.pt"
    answer_pt_path = temp_dir / f"{profile_id}_answer.pt"
    meta_path = temp_dir / f"{profile_id}_meta.json"

    torch.save(cot_states, pt_path)
    torch.save(answer_state, answer_pt_path)

    meta = {
        "profile_id": str(profile_id),
        "step_count": 4,
        "eors_token_positions": eors_positions,
        "eoa_token_position": eoa_position,
        "hidden_dim": cot_states.shape[-1],
        "cot_shape": list(cot_states.shape),
        "answer_shape": list(answer_state.shape),
        "extraction_ok": True,
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    return pt_path, answer_pt_path, meta_path


def promote_to_final(
    profile_id: int,
    temp_dir: Path,
    final_dir: Path,
) -> tuple[Path, Path, Path]:
    """Move .pt, _answer.pt, and _meta.json from temp/ to final/ after validation passes."""
    final_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for suffix in (".pt", "_answer.pt", "_meta.json"):
        src = temp_dir / f"{profile_id}{suffix}"
        dst = final_dir / f"{profile_id}{suffix}"
        shutil.move(str(src), str(dst))
        moved.append(dst)
    return moved[0], moved[1], moved[2]


def discard_from_temp(profile_id: int, temp_dir: Path) -> None:
    """Delete temp files for a rejected or failed profile."""
    for suffix in (".pt", "_answer.pt", "_meta.json"):
        path = temp_dir / f"{profile_id}{suffix}"
        if path.exists():
            path.unlink()


def delete_per_profile_files(profile_id: int, final_dir: Path) -> None:
    """Remove per-profile .pt, _answer.pt, _meta.json from final/.

    Called after the profile's hidden states have been written into the
    consolidated tensors, making the individual files redundant.
    """
    for suffix in (".pt", "_answer.pt", "_meta.json"):
        path = final_dir / f"{profile_id}{suffix}"
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Master index
# ---------------------------------------------------------------------------


def update_master_index(
    profile_id: int,
    index_path: Path,
    eors_positions: list[int],
    eoa_position: int,
    cot_states_path: Path,
    answer_states_path: Path,
    profile_row_idx: int,
) -> None:
    """Append one entry to the master index .jsonl file (incremental, resumable).

    The index is the single source of truth for the student training pipeline.
    Given a profile ID, the student can resolve its hidden states in O(1) by
    reading this index and slicing the consolidated tensor at ``profile_row_idx``.

    Entry format::

        {
          "profile_id": "1",
          "step_count": 4,
          "eors_token_positions": [p0, p1, p2, p3],
          "eoa_token_position": q,
          "cot_states_path": ".../train/cot_states.pt",
          "answer_states_path": ".../train/answer_states.pt",
          "profile_row_idx": 0,
          "extraction_ok": true
        }
    """

    # Include the cot_states and answer_states paths as relative paths
    cot_states_path = str(cot_states_path).split(
        "cot_representations_finance_healthcare"
    )[1]
    answer_states_path = str(answer_states_path).split(
        "cot_representations_finance_healthcare"
    )[1]

    entry = {
        "profile_id": str(profile_id),
        "step_count": 4,
        "eors_token_positions": eors_positions,
        "eoa_token_position": eoa_position,
        "cot_states_path": str(cot_states_path),
        "answer_states_path": str(answer_states_path),
        "profile_row_idx": profile_row_idx,
        "extraction_ok": True,
    }
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
        fh.flush()
