# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""W4A16 MoE for RDNA3.5 (gfx115x) on the FlyDSL WMMA kernels in
``kernels/moe_w4a16_rdna.py``.

Weights are int4 with a group size that is a multiple of 16, with or without
zero points, packed by ``pack_w4a16_weights``. Activations are bfloat16 or
float16.
"""

import torch
import triton
import triton.language as tl

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.moe_w4a16_rdna import (
    BLOCK_M,
    MAX_REDUCE_ROWS,
    ROW_PATH_MAX_TOKENS,
    TILE_N,
    _build_gemm,
    _build_gemm_reduce,
    _build_gemm_row,
    _check_coef,
    _check_w,
    _expert_stride_dwords,
    gate_up_tile_order,
    lds_capacity_bytes,
    pack_w4a16_weights,
    pad_expert_stride,
    reduce_lds_bytes,
    row_lds_bytes,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, current_stream_arg

__all__ = [
    "BLOCK_M",
    "MAX_REDUCE_ROWS",
    "ROW_PATH_MAX_TOKENS",
    "TILE_N",
    "flydsl_rdna_moe",
    "flydsl_rdna_moe_gemm",
    "flydsl_rdna_moe_gemm_reduce",
    "flydsl_rdna_moe_path",
    "flydsl_rdna_moe_supported",
    "gate_up_tile_order",
    "moe_align_block_size_rows",
    "pack_w4a16_weights",
    "pad_expert_stride",
    "topk_weighted_sum",
]


# gfx115x targets with a known LDS capacity; the kernels are tested on gfx1151.
_SUPPORTED_GFX = ("gfx1150", "gfx1151")


def flydsl_rdna_moe_supported() -> bool:
    return get_gfx_runtime() in _SUPPORTED_GFX


# Buffer offsets are 32-bit, so a tensor must span less than this.
_MAX_SPAN_BYTES = 2**31


def _check_act(out: torch.Tensor, x: torch.Tensor) -> None:
    """The kernels store the activation dtype without a conversion."""
    assert x.dtype in (torch.bfloat16, torch.float16), "bf16 or fp16 activations"
    assert out.dtype == x.dtype, "output dtype must match the activations"


def _check_packed(w: torch.Tensor, coef: torch.Tensor, n_cols: int) -> None:
    assert w.size(1) * TILE_N == n_cols, "packed weights do not match the width"
    assert coef.shape[:2] == w.shape[:2], "weights and coefficients do not match"


def _check_args(dense, packed) -> None:
    """The kernels index raw data pointers: ``dense`` tensors must be
    contiguous, the packed ``[E, tiles, tile stride]`` weights dense within an
    expert, and all on one device."""
    device = dense[0].device
    for t in dense:
        assert t.device == device, "tensors must be on one device"
        assert t.is_contiguous(), "tensors must be contiguous"
        assert t.numel() * t.element_size() < _MAX_SPAN_BYTES, "tensor too large"
    for t in packed:
        assert t.device == device, "tensors must be on one device"
        assert t.dim() == 3 and t.stride(2) == 1 and t.stride(1) == t.size(2)
        assert t.stride(0) >= t.size(1) * t.size(2), "packed layout"
        span = t.size(0) * t.stride(0) * t.element_size()
        assert span < _MAX_SPAN_BYTES, "packed weights too large"


def _launch(launch, *args) -> None:
    """Run a ``@flyc.jit`` launcher through a compiled handle cached per
    device and tensor dtype, rank and unit-stride axis. The kernels read only
    the data pointers of their tensors."""
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    key = (tensors[0].device,) + tuple(
        (t.dtype, t.dim(), t.stride(-1) == 1) for t in tensors
    )
    stream = current_stream_arg(tensors[0].device)
    _run_compiled(launch, *args, stream, specialization_key=key)


def moe_align_block_size_rows(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the routed rows ``r = token * top_k + k`` by expert into blocks of
    ``block_size`` rows: ``(sorted_ids, expert_ids, num_valid)``.

    Padding rows hold ``topk_ids.numel()`` and blocks past the last expert
    hold expert -1.
    """
    from aiter.ops.moe_op import moe_align_block_size

    numel = topk_ids.numel()
    max_blocks = -(-(numel + num_experts * (block_size - 1)) // block_size)
    device = topk_ids.device
    sorted_ids = torch.full(
        (max_blocks * block_size,), numel, dtype=torch.int32, device=device
    )
    expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32, device=device)
    token_nums = torch.empty((max_blocks,), dtype=torch.int32, device=device)
    num_valid = torch.empty((1,), dtype=torch.int32, device=device)
    moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, token_nums, num_valid
    )
    return sorted_ids, expert_ids, num_valid


def flydsl_rdna_moe_gemm_reduce(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    coef: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    group_size: int,
    coef_pairs: bool = True,
) -> None:
    """GEMM2 with the top-k sum: ``out[t] = sum_e weights[t, e] *
    W[expert_ids[t * top_k + e]] @ x[t * top_k + e]``.

    ``weights`` is float32 ``[tokens, top_k]`` with at most
    ``MAX_REDUCE_ROWS`` experts per token. ``coef_pairs`` is False for weights
    packed without zero points.
    """
    _check_coef(coef, x.shape[1], group_size, coef_pairs)
    _check_w(w, x.shape[1])
    _check_args((out, x, expert_ids, weights), (w, coef))
    _check_act(out, x)
    assert weights.dtype == torch.float32
    n_tokens, top_k = weights.shape
    assert out.size(0) == n_tokens, "one output row per token"
    assert x.size(0) == expert_ids.numel() == weights.numel(), "routed rows"
    _check_packed(w, coef, out.size(1))
    launch = _build_gemm_reduce(
        x.shape[1],
        out.shape[1],
        group_size,
        top_k,
        _expert_stride_dwords(w),
        _expert_stride_dwords(coef),
        coef_pairs,
        x.dtype == torch.float16,
    )
    _launch(launch, out, x, w, coef, expert_ids, weights, n_tokens, 32)


def flydsl_rdna_moe_gemm(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    coef: torch.Tensor,
    sorted_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_valid: torch.Tensor | None,
    x_row_div: int,
    group_size: int,
    silu_mul: bool = False,
    coef_pairs: bool = True,
) -> None:
    """Run ``out[r] = W[expert(r)] @ x[r // x_row_div]`` for all routed rows.

    ``sorted_ids`` / ``expert_ids`` / ``num_valid`` come from
    ``moe_align_block_size`` with block size ``BLOCK_M``. When ``sorted_ids``
    is None, ``expert_ids[r]`` is the expert of row ``r`` and no grouping is
    needed; this is the cheap routing for a few rows with distinct experts.

    With ``silu_mul`` the weights are in ``gate_up_tile_order`` and ``out``
    receives ``silu(gate) * up``, half as many columns as the weights have.
    ``coef_pairs`` is False for weights packed without zero points.
    """
    _check_coef(coef, x.shape[1], group_size, coef_pairs)
    _check_w(w, x.shape[1])
    route = [t for t in (sorted_ids, expert_ids, num_valid) if t is not None]
    _check_args((out, x, *route), (w, coef))
    _check_act(out, x)
    n_rows, N = out.shape
    if silu_mul:
        N *= 2
    _check_packed(w, coef, N)
    assert -(-n_rows // x_row_div) <= x.size(0), "x has too few rows"
    if sorted_ids is None:
        assert expert_ids.numel() == n_rows, "one expert id per row"
    else:
        assert sorted_ids.numel() == expert_ids.numel() * BLOCK_M, "sorted ids"
    build_args = (
        x.shape[1],
        N,
        x_row_div,
        group_size,
        silu_mul,
        _expert_stride_dwords(w),
        _expert_stride_dwords(coef),
        coef_pairs,
        x.dtype == torch.float16,
    )
    if sorted_ids is None:
        launch = _build_gemm_row(*build_args)
        _launch(launch, out, x, w, coef, expert_ids, expert_ids.numel())
        return
    launch = _build_gemm(*build_args)
    _launch(
        launch,
        out,
        x,
        w,
        coef,
        sorted_ids,
        expert_ids,
        num_valid,
        n_rows,
        expert_ids.numel(),
    )


@triton.jit
def _topk_weighted_sum_kernel(
    out_ptr,
    rows_ptr,
    weights_ptr,
    ids_ptr,
    K,
    out_stride,
    rows_stride,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_IDS: tl.constexpr,
):
    # Column blocks vary fastest: concurrent programs then read different
    # 256 B DRAM channels instead of the same offset of 4 KB-strided rows.
    t = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = cols < K
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        w = tl.load(weights_ptr + t * TOP_K + k)
        row = tl.load(rows_ptr + (t * TOP_K + k) * rows_stride + cols, mask=mask)
        prod = w * row.to(tl.float32)
        if HAS_IDS:
            # Routes without an expert (id -1) add nothing, even if not finite.
            prod = tl.where(tl.load(ids_ptr + t * TOP_K + k) >= 0, prod, 0.0)
        acc += prod
    tl.store(
        out_ptr + t * out_stride + cols, acc.to(out_ptr.dtype.element_ty), mask=mask
    )


def topk_weighted_sum(
    out: torch.Tensor,
    rows: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
) -> None:
    """``out[t] = sum_k topk_weights[t, k] * rows[t * top_k + k]``, in f32.
    With ``topk_ids``, routes with a negative id are left out."""
    M, top_k = topk_weights.shape
    K = out.size(1)
    assert rows.stride(1) == 1 and out.stride(1) == 1
    block_k = 256
    _topk_weighted_sum_kernel[(triton.cdiv(K, block_k), M)](
        out,
        rows,
        topk_weights.contiguous(),
        topk_weights if topk_ids is None else topk_ids.contiguous(),
        K,
        out.stride(0),
        rows.stride(0),
        TOP_K=top_k,
        BLOCK_K=block_k,
        HAS_IDS=topk_ids is not None,
        num_warps=8,
    )


def flydsl_rdna_moe_path(
    M: int, K: int, N: int, top_k: int, group_size: int, weights_dtype
) -> str:
    """Kernels ``flydsl_rdna_moe`` runs: ``"reduce"`` (row GEMM1, GEMM2 with
    the top-k sum), ``"row"`` (row GEMMs, ``topk_weighted_sum``) or
    ``"grouped"``. The row and reduce kernels need their operands in LDS."""
    lds = lds_capacity_bytes()
    if M > ROW_PATH_MAX_TOKENS or row_lds_bytes(K, group_size) > lds:
        return "grouped"
    if (
        top_k <= MAX_REDUCE_ROWS
        and weights_dtype == torch.float32
        and reduce_lds_bytes(N, top_k) <= lds
    ):
        return "reduce"
    return "row" if row_lds_bytes(N, group_size) <= lds else "grouped"


def flydsl_rdna_moe(
    output: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    coef1: torch.Tensor,
    w2: torch.Tensor,
    coef2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace: torch.Tensor,
    group_size: int,
    num_experts: int,
    N: int,
    activation=None,
    w1_pairs: bool = True,
    w2_pairs: bool = True,
) -> None:
    """``output[t] = sum_e topk_weights[t, e] * (W2[e] @ act(W1[e] @ x[t]))``.

    ``w1`` / ``w2`` and their coefficients are in the ``pack_w4a16_weights``
    layout. ``activation(out, gate_up)`` is applied between the GEMMs; when
    it is None, ``w1`` is in ``gate_up_tile_order`` and GEMM1 applies
    SiLU-and-multiply itself. ``N`` is the width of the activation output.
    Routes with expert id -1 contribute nothing; other ids must be below
    ``num_experts``. ``workspace`` holds the intermediates, ``M * top_k *
    (gate_up + N + K)`` elements of the activation dtype.
    """
    silu_mul = activation is None
    M, K = x.shape
    top_k = topk_ids.size(1)
    gate_up = w1.size(1) * TILE_N
    rows = M * top_k
    widths = (gate_up, N, K)
    sizes = [rows * n for n in widths]
    _check_args((output, x, topk_weights, topk_ids, workspace), ())
    _check_act(output, x)
    assert workspace.dtype == x.dtype, "workspace dtype must match the activations"
    assert output.shape == x.shape, "output shape must match x"
    assert topk_weights.shape == topk_ids.shape, "top-k weights and ids shape"
    assert topk_ids.size(0) == M, "one top-k row per token"
    assert workspace.numel() >= sum(sizes), "workspace too small"
    assert w1.size(0) == w2.size(0) == num_experts, "expert count"
    _check_packed(w2, coef2, K)
    assert not silu_mul or gate_up == 2 * N, "fused SiLU needs gate_up = 2 N"
    gate_up_out, act_out, down_out = (
        t.view(rows, n)
        for t, n in zip(workspace.view(-1)[: sum(sizes)].split(sizes), widths)
    )
    ids = topk_ids.to(torch.int32)
    path = flydsl_rdna_moe_path(M, K, N, top_k, group_size, topk_weights.dtype)
    if path != "grouped":
        # Few tokens seldom share an expert, so every row is its own block
        # and the alignment kernel is not needed.
        route = (None, ids.view(-1), None)
    else:
        # The alignment indexes per-expert arrays with the ids: send routes
        # without an expert (-1) to expert 0; the top-k sum drops them.
        route = moe_align_block_size_rows(ids.clamp(min=0), BLOCK_M, num_experts)
    flydsl_rdna_moe_gemm(
        act_out if silu_mul else gate_up_out,
        x,
        w1,
        coef1,
        *route,
        top_k,
        group_size,
        silu_mul=silu_mul,
        coef_pairs=w1_pairs,
    )
    if not silu_mul:
        activation(act_out, gate_up_out)
    if path == "reduce":
        # GEMM2 also does the weighted sum over the experts of a token.
        flydsl_rdna_moe_gemm_reduce(
            output,
            act_out,
            w2,
            coef2,
            route[1],
            topk_weights,
            group_size,
            coef_pairs=w2_pairs,
        )
        return
    flydsl_rdna_moe_gemm(
        down_out, act_out, w2, coef2, *route, 1, group_size, coef_pairs=w2_pairs
    )
    # Rows are in routing order, so the top-k sum needs no permutation.
    topk_weighted_sum(output, down_out, topk_weights, topk_ids)
