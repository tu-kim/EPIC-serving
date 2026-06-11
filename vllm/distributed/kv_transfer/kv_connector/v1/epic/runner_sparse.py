# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker/runner-side sparse-forward helpers for EPIC (Phase 2b, S4/S5).

The runner consumes the scheduler's per-request sparse plan
(``SchedulerOutput.epic_sparse_positions`` / ``epic_seq_len``, stamped by the
S3 scheduler patch) and rewrites two things before the model forward:

  * **RoPE positions** (S4): the M tokens actually forwarded are at the
    *logical* positions ``epic_sparse_positions[req_id]`` (which may be
    non-contiguous), NOT the contiguous ``computed_prefix + arange`` the vanilla
    runner builds. The token-id gather and slot_mapping both derive from
    ``positions`` downstream, so overwriting positions is sufficient for them to
    follow (gpu_model_runner.py :1915 / :2093).
  * **seq_lens** (S5): a sparse request attends over the full reused KV span
    ``N = epic_seq_len[req_id]``, not ``num_computed + num_scheduled`` (which is
    smaller / wrong because external A/B KV overlaps M).

These functions are pure (numpy / list math, no torch device ops) so they are
unit-testable on CPU without a GPU runner. The runner calls them to compute the
row-range overwrites and then applies them to its captured buffers in place
(no fresh tensor allocation -> CUDA-graph safe).

Vanilla invariance: every helper here is only ever called when
``scheduler_output.epic_sparse_positions`` is non-empty. With EPIC sparse
forward OFF that dict is empty and the runner never enters this path, so the
default forward is byte-for-byte unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SparseRowEdit:
    """One request's sparse overwrite, resolved to absolute row indices.

    Attributes:
        req_id: the request this edit applies to.
        row_start: first packed-token row (inclusive) for this request.
        row_end: one-past-last packed-token row (exclusive). The request owns
            rows ``[row_start, row_end)``; ``row_end - row_start == len(positions)``.
        positions: logical RoPE positions for those rows (M's logical positions,
            possibly non-contiguous). Length == ``row_end - row_start``.
        seq_len: N, the full logical span this request attends over (for S5
            seq_lens override).
    """

    req_id: str
    row_start: int
    row_end: int
    positions: list[int]
    seq_len: int


def build_sparse_row_edits(
    req_ids: list[str],
    cu_num_tokens: list[int],
    epic_sparse_positions: dict[str, list[int]],
    epic_seq_len: dict[str, int],
) -> list[SparseRowEdit]:
    """Resolve per-request sparse plans to absolute row-range edits.

    Args:
        req_ids: req_id for each request index, in batch order (==
            ``input_batch.req_ids[:num_reqs]``).
        cu_num_tokens: end-exclusive cumulative token counts per request, e.g.
            ``[2, 7, 10]`` for per-req counts ``[2, 5, 3]`` (==
            ``query_start_loc[1:num_reqs+1]``). ``len(cu_num_tokens) == len(req_ids)``.
        epic_sparse_positions: req_id -> M's logical positions (the value the S3
            scheduler stamped; ``len == num_scheduled_tokens[req_id]``).
        epic_seq_len: req_id -> N (full logical span).

    Returns:
        One ``SparseRowEdit`` per sparse request in the batch, in batch order.
        Non-sparse requests (not in ``epic_sparse_positions``) are skipped.

    Raises:
        ValueError: if a sparse request's position vector length does not match
            the number of rows the scheduler allocated for it (a contract
            violation between the S3 scheduler patch and this runner patch).
    """
    edits: list[SparseRowEdit] = []
    prev = 0
    for i, req_id in enumerate(req_ids):
        end = cu_num_tokens[i]
        positions = epic_sparse_positions.get(req_id)
        if positions is not None:
            num_rows = end - prev
            if len(positions) != num_rows:
                raise ValueError(
                    f"EPIC sparse positions for {req_id!r} have length "
                    f"{len(positions)} but the request owns {num_rows} forward "
                    f"rows ([{prev}, {end})); the scheduler must set "
                    f"num_scheduled_tokens == len(sparse_positions)."
                )
            edits.append(
                SparseRowEdit(
                    req_id=req_id,
                    row_start=prev,
                    row_end=end,
                    positions=list(positions),
                    # epic_seq_len is stamped alongside positions; default to the
                    # last position + 1 (== N, since positions[-1] == N-1) when a
                    # request somehow lacks an explicit seq_len entry.
                    seq_len=int(
                        epic_seq_len.get(
                            req_id,
                            (positions[-1] + 1) if positions else 0,
                        )
                    ),
                )
            )
        prev = end
    return edits
