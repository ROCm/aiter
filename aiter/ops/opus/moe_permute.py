# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE token permute / unpermute for training (gfx950 OPUS), forward and backward.

    x_perm, plan = moe_permute(x, expert_ids, num_experts)   x [N, H] bf16, expert_ids [N] int32
    out          = moe_unpermute(y_perm, plan, weights)      y_perm [R, H] bf16, weights [N] fp32 or None

One expert per token (the rows an expert-owner rank receives from the dispatch): there is no top-k.
Both are differentiable (torch.autograd.Function). The grouped GEMMs between them take their
group information from the plan: ``plan.grouped_layout`` is the M-grouped layout (forward, dgrad),
``plan.group_rows`` the K-grouped layout (wgrad) of aiter.ops.opus.gemm_op_a16w16_grouped.

Five kernels, each one workgroup per destination row:
    plan  expert_ids -> the row of every token (two launches: count, rows)
    K2  permute forward     x_perm[row[i]] = x[i]                    K3  permute backward  dx[i] = d_perm[row[i]]
    K4  unpermute forward   out[i] = weights[i] * y_perm[row[i]]     K5  unpermute backward (d_perm and d_weights)
moe_unpermute without weights is the copy: forward the kernel of K3, backward the kernel of K2.

Rows: experts in ascending order, every expert's rows contiguous and in token order, each group padded
to GROUP_ALIGN (128) rows; an expert without tokens takes no row. R, the rows of x_perm / y_perm and of the
plan's [R] tables, is the rows in use (see exact of moe_permute_plan), a multiple of GROUP_ALIGN, which the
M-grouped GEMMs require of their row count. Padding rows (inside a group, and every row past the last group) are zero in
x_perm / d_perm and -1 in plan.sorted_token_ids / plan.grouped_layout. plan.num_valid_ids [1] is the number of
rows in use (up to the end of the last group).

expert_ids and weights may be [N] or [N, 1]; the plan's tables are [N] / [R], d_weights has the shape of weights.
An id outside 0 .. E - 1 gets no row (plan.reverse_sorted -1). moe_permute / moe_unpermute expect every id to be
an expert; a token without a row reads as a zero row (dx, out) and its d_weights is not written.

Numerics (the kernels and the reference in op_tests/test_opus_moe_permute.py follow the same rules):
  - K2 / K3 (and moe_unpermute without weights) move the bf16 bits unchanged;
  - K4 / K5: the product weights[i] * row is computed in fp32 and rounded to bf16 once (nearest even);
  - K5: d_weights[i] = dot(d_out[i], y_perm[row[i]]) is accumulated as documented at DW_THREADS (no fused
    multiply-add).

Memory traffic: the kernels are bound by it. Every row is read once and written once per call. The outputs
[R, H] (x_perm, d_perm) are stored non-temporal, the outputs [N, H] (dx, out) with system scope; a source
larger than the last-level cache is loaded non-temporal. None of this changes a value.
"""

from typing import NamedTuple

import torch

from ...jit.core import compile_ops

GROUP_ALIGN = 128  # rows every expert's group is padded to (OPUS_A16W16_GROUP_ALIGN of the grouped GEMMs)
# The plan kernels take the tokens in blocks of PLAN_BLOCK_TOKENS (one thread per token) and have one thread
# per expert: up to PLAN_MAX_EXPERTS experts (OPUS_MOE_PERMUTE_PLAN_* of the kernels).
PLAN_BLOCK_TOKENS = 256
PLAN_MAX_EXPERTS = 256
PLAN_COUNT_BATCH = 64  # the counts workspace holds a multiple of this many blocks per expert

# Accumulation structure of the d_weights dot product over H elements, fixed by K5:
# the row is cut into chunks of DW_VEC elements; thread i of DW_THREADS adds the products of chunks
# i, i + DW_THREADS, ... in ascending element order into one fp32 partial sum; the partial sums of
# the threads that have a chunk are then added in ascending thread order. H is a multiple of DW_VEC.
DW_THREADS = 256
DW_VEC = 8


class MoEPermutePlan(NamedTuple):
    """The row of every token: the output of the plan kernels. On the device of expert_ids."""

    sorted_token_ids: torch.Tensor  # [R] int32   token of the row, -1: padding
    reverse_sorted: torch.Tensor  # [N] int32     row of token i, -1: its id is not an expert
    grouped_layout: torch.Tensor  # [R] int32     expert of the row, -1: padding (M-grouped layout)
    group_rows: torch.Tensor  # [E] int32         rows of each expert, without padding (K-grouped layout)
    num_valid_ids: torch.Tensor  # [1] int32      rows in use: up to the end of the last group; the rest is the tail


def moe_permute_rows(n: int, num_experts: int) -> int:
    """The most rows N tokens of num_experts experts can take, whatever their experts are: N rounded up to
    GROUP_ALIGN (all of them in one group) and one more group of padding for every further expert."""
    return (n + GROUP_ALIGN - 1) // GROUP_ALIGN * GROUP_ALIGN + GROUP_ALIGN * (num_experts - 1)


# ---- kernel entries (csrc/opus_moe_permute/opus_moe_permute.cu) ------------------


@compile_ops("module_moe_permute_opus", fc_name="opus_moe_permute_plan", develop=True)
def _plan_launch(
    expert_ids: torch.Tensor,
    counts: torch.Tensor,
    reverse_sorted: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    grouped_layout: torch.Tensor,
    group_rows: torch.Tensor,
    num_valid_ids: torch.Tensor,
) -> None: ...


# scatter copy: x_perm[r] = x[sorted_token_ids[r]], padding rows 0
@compile_ops("module_moe_permute_opus", fc_name="opus_moe_permute_fwd", develop=True)
def _permute_fwd_launch(x: torch.Tensor, sorted_token_ids: torch.Tensor, x_perm: torch.Tensor) -> None: ...


# gather copy: dx[i] = d_perm[reverse_sorted[i]]
@compile_ops("module_moe_permute_opus", fc_name="opus_moe_permute_bwd", develop=True)
def _permute_bwd_launch(d_perm: torch.Tensor, reverse_sorted: torch.Tensor, dx: torch.Tensor) -> None: ...


@compile_ops("module_moe_permute_opus", fc_name="opus_moe_unpermute_fwd", develop=True)
def _unpermute_fwd_launch(
    y_perm: torch.Tensor, reverse_sorted: torch.Tensor, weights: torch.Tensor, out: torch.Tensor
) -> None: ...


@compile_ops("module_moe_permute_opus", fc_name="opus_moe_unpermute_bwd", develop=True)
def _unpermute_bwd_launch(
    d_out: torch.Tensor,
    y_perm: torch.Tensor,
    weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    d_perm: torch.Tensor,
    d_weights: torch.Tensor,
) -> None: ...


# ---- public API ------------------------------------------------------------


def _is_per_token(t: torch.Tensor, dtype: torch.dtype) -> bool:
    return t.dtype == dtype and (t.dim() == 1 or (t.dim() == 2 and t.shape[1] == 1))


def moe_permute_plan(expert_ids: torch.Tensor, num_experts: int, exact: bool = True) -> MoEPermutePlan:
    """The plan for expert_ids (int32 [N] or [N, 1]). Not differentiable.
    exact True (default): R is the rows in use (the groups with their padding, a multiple of GROUP_ALIGN;
    GROUP_ALIGN when no id is an expert), so nothing behind the last group is allocated in x_perm / y_perm and
    in what the grouped GEMMs make of them. The [R] tables are views of tables of moe_permute_rows(N, E) rows;
    num_valid_ids is read back for it, so the call waits for the plan kernels.
    exact False: R = moe_permute_rows(N, E), the upper bound known before the launch; nothing is read back."""
    if not _is_per_token(expert_ids, torch.int32):
        raise ValueError("expert_ids must be int32 [N] or [N, 1]")
    if not 0 < num_experts <= PLAN_MAX_EXPERTS:
        raise ValueError(f"num_experts must be in 1 .. {PLAN_MAX_EXPERTS}")
    n = expert_ids.shape[0]
    rows = moe_permute_rows(n, num_experts)
    if rows >= 2**30:  # the plan kernels address the [R] tables with 32-bit byte offsets
        raise ValueError("too many rows")
    dev = expert_ids.device

    def i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device=dev)

    plan = MoEPermutePlan(i32(rows), i32(n), i32(rows), i32(num_experts), i32(1))
    with torch.cuda.device(dev):
        _plan_launch(
            expert_ids.contiguous(),
            # workspace: the expert counts of every token block
            i32(-(-(-(-n // PLAN_BLOCK_TOKENS)) // PLAN_COUNT_BATCH) * PLAN_COUNT_BATCH * num_experts),
            plan.reverse_sorted,
            plan.sorted_token_ids,
            plan.grouped_layout,
            plan.group_rows,
            plan.num_valid_ids,
        )
    if exact:
        used = max(int(plan.num_valid_ids), GROUP_ALIGN)
        plan = plan._replace(sorted_token_ids=plan.sorted_token_ids[:used], grouped_layout=plan.grouped_layout[:used])
    return plan


class _MoEPermuteRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sorted_token_ids, reverse_sorted):
        x_perm = x.new_empty((sorted_token_ids.numel(), x.shape[1]))
        with torch.cuda.device(x.device):
            _permute_fwd_launch(x.contiguous(), sorted_token_ids, x_perm)
        ctx.reverse_sorted = reverse_sorted
        return x_perm

    @staticmethod
    def backward(ctx, d_perm):
        dx = d_perm.new_empty((ctx.reverse_sorted.numel(), d_perm.shape[1]))
        with torch.cuda.device(d_perm.device):
            _permute_bwd_launch(d_perm.contiguous(), ctx.reverse_sorted, dx)
        return dx, None, None


class _MoEUnpermuteRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_perm, weights, sorted_token_ids, reverse_sorted):
        out = y_perm.new_empty((reverse_sorted.numel(), y_perm.shape[1]))
        y_perm = y_perm.contiguous()
        with torch.cuda.device(y_perm.device):
            if weights is None:
                _permute_bwd_launch(y_perm, reverse_sorted, out)  # the gather copy of K3
            else:
                weights = weights.contiguous()
                _unpermute_fwd_launch(y_perm, reverse_sorted, weights, out)
                ctx.save_for_backward(y_perm, weights)
        ctx.sorted_token_ids = sorted_token_ids
        return out

    @staticmethod
    def backward(ctx, d_out):
        d_out = d_out.contiguous()
        d_perm = d_out.new_empty((ctx.sorted_token_ids.numel(), d_out.shape[1]))
        with torch.cuda.device(d_out.device):
            if not ctx.saved_tensors:  # no weights: the scatter copy of K2
                _permute_fwd_launch(d_out, ctx.sorted_token_ids, d_perm)
                return d_perm, None, None, None
            y_perm, weights = ctx.saved_tensors
            d_weights = torch.empty_like(weights)
            _unpermute_bwd_launch(d_out, y_perm, weights, ctx.sorted_token_ids, d_perm, d_weights)
        return d_perm, d_weights, None, None


def _check_rows(name: str, t: torch.Tensor, rows: int, what: str):
    if t.dim() != 2 or t.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bf16 [rows, H]")
    if t.shape[1] % DW_VEC:
        raise ValueError(f"{name}: H ({t.shape[1]}) must be a multiple of {DW_VEC}")
    if t.shape[0] != rows:
        raise ValueError(f"{name} has {t.shape[0]} rows, {what} {rows}")


def moe_permute(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    weights: torch.Tensor | None = None,
    plan: MoEPermutePlan | None = None,
    exact: bool = True,
) -> tuple[torch.Tensor, MoEPermutePlan]:
    """x_perm [R, H]: row plan.reverse_sorted[i] = x[i], padding rows 0. Gradient: dx[i] = d_perm[row of i].
    weights: not read (moe_unpermute takes them). plan: reuse one already computed for these expert_ids
    (expert_ids, num_experts and exact are then not read). exact: of moe_permute_plan (R = the rows in use)."""
    if plan is None:
        plan = moe_permute_plan(expert_ids, num_experts, exact)
    _check_rows("x", x, plan.reverse_sorted.numel(), "expert_ids has")
    x_perm = _MoEPermuteRows.apply(x, plan.sorted_token_ids, plan.reverse_sorted)
    return x_perm, plan


def moe_unpermute(y_perm: torch.Tensor, plan: MoEPermutePlan, weights: torch.Tensor | None = None) -> torch.Tensor:
    """out [N, H]: out[i] = weights[i] * y_perm[plan.reverse_sorted[i]] (weights None: the row itself).
    Gradients: d_perm[row of i] = weights[i] * d_out[i] (padding rows 0; weights None: d_out[i] itself),
    d_weights[i] = dot(d_out[i], y_perm[row of i]). weights: fp32 [N] or [N, 1]."""
    _check_rows("y_perm", y_perm, plan.sorted_token_ids.numel(), "the plan has")
    if weights is not None and not (
        _is_per_token(weights, torch.float32) and weights.shape[0] == plan.reverse_sorted.numel()
    ):
        raise ValueError("weights must be fp32 [N] or [N, 1], one per token")
    return _MoEUnpermuteRows.apply(y_perm, weights, plan.sorted_token_ids, plan.reverse_sorted)
