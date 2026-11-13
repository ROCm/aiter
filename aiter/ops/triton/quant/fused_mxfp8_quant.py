# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fused_mxfp8_quant import (
    _fused_dual_rmsnorm_mxfp8_quant_kernel,
    _fused_flatten_mxfp8_quant_kernel,
    _fused_rms_mxfp8_kernel,
    _k3_mla_640_gather_kernel,
    _k3_mla_640_pack_kernel,
)

__all__ = [
    "fused_dual_rmsnorm_mxfp8_quant",
    "fused_flatten_mxfp8_quant",
    "fused_rms_mxfp8_quant",
    "k3_mla_640_concat_and_cache",
    "k3_mla_640_cp_gather_and_upconvert",
    "k3_mla_640_q_pack",
]

_QUANT_BLOCK_SIZE = 32


def fused_rms_mxfp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    y: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + MXFP8 (1x32 e8m0) quant in a single Triton launch.

    Args:
        x: (M, K) bf16 or fp16.
        weight: (K,) bf16 or fp16 RMSNorm weight.
        eps: RMSNorm epsilon.
        y: optional preallocated FP8 e4m3fn output (M, K).
        scale: optional preallocated uint8 e8m0 output (M, K // 32).

    Returns:
        y (M, K) fp8 e4m3fn, scale (M, K // 32) uint8.
    """
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}"
    M, K = x.shape
    assert weight.shape == (K,), f"weight shape {weight.shape} != ({K},)"
    assert K % _QUANT_BLOCK_SIZE == 0
    Ns = K // _QUANT_BLOCK_SIZE
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    if y is None:
        y = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty((M, Ns), dtype=torch.uint8, device=x.device)

    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _fused_rms_mxfp8_kernel[grid](
        x,
        weight,
        y,
        scale,
        M,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        eps,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )
    return y, scale


def fused_dual_rmsnorm_mxfp8_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps_q: float,
    eps_k: float | None = None,
    yq: torch.Tensor | None = None,
    sq: torch.Tensor | None = None,
    yk: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual RMSNorm in a single Triton launch.

    - Q side: RMSNorm(q, q_weight, eps_q) -> MXFP8 (FP8 e4m3fn + uint8 e8m0 1x32).
    - K side: RMSNorm(k, k_weight, eps_k) -> bf16.

    Replaces the CK `fused_qk_rmsnorm_group_quant` kernel for the MXFP8 GEMM
    path on V4 (Task #77): one launch instead of two (fused_rms_mxfp8_quant +
    rmsnorm2d_fwd_), eliminating the ~6us/layer launch-overhead regression.

    Args:
        q: (M, KQ) bf16 or fp16 — Q-side input (e.g. q_lora).
        k: (M, KK) bf16 or fp16 — K-side input (e.g. kv_pre).
        q_weight: (KQ,) bf16 or fp16 — Q RMSNorm weight.
        k_weight: (KK,) bf16 or fp16 — K RMSNorm weight.
        eps_q: Q RMSNorm epsilon.
        eps_k: K RMSNorm epsilon; defaults to eps_q.
        yq, sq, yk: optional pre-allocated outputs.

    Returns:
        yq (M, KQ) fp8 e4m3fn, sq (M, KQ // 32) uint8 e8m0, yk (M, KK) bf16.
    """
    assert q.dim() == 2, f"q must be 2D, got {q.dim()}"
    assert k.dim() == 2, f"k must be 2D, got {k.dim()}"
    M, KQ = q.shape
    Mk, KK = k.shape
    assert M == Mk, f"q rows {M} != k rows {Mk}"
    assert q_weight.shape == (KQ,), f"q_weight shape {q_weight.shape} != ({KQ},)"
    assert k_weight.shape == (KK,), f"k_weight shape {k_weight.shape} != ({KK},)"
    assert (
        KQ % _QUANT_BLOCK_SIZE == 0
    ), f"KQ={KQ} must be a multiple of {_QUANT_BLOCK_SIZE}"
    if eps_k is None:
        eps_k = eps_q

    Ns = KQ // _QUANT_BLOCK_SIZE
    BLOCK_SIZE_KQ = triton.next_power_of_2(KQ)
    BLOCK_SIZE_KK = triton.next_power_of_2(KK)

    if yq is None:
        yq = torch.empty((M, KQ), dtype=torch.float8_e4m3fn, device=q.device)
    if sq is None:
        sq = torch.empty((M, Ns), dtype=torch.uint8, device=q.device)
    if yk is None:
        yk = torch.empty((M, KK), dtype=k.dtype, device=k.device)

    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _fused_dual_rmsnorm_mxfp8_quant_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        yq,
        sq,
        yk,
        M,
        KQ,
        KK,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        yq.stride(0),
        yq.stride(1),
        sq.stride(0),
        sq.stride(1),
        yk.stride(0),
        yk.stride(1),
        eps_q,
        eps_k,
        BLOCK_SIZE_KQ=BLOCK_SIZE_KQ,
        BLOCK_SIZE_KK=BLOCK_SIZE_KK,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )
    return yq, sq, yk


def fused_flatten_mxfp8_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten the last two dimensions of x and apply per-1x32 MXFP8 quant along
    the flattened axis (FP8 e4m3 values + uint8 e8m0 scales).

    Equivalent in shape to `fused_flatten_fp8_group_quant` but emits MXFP8
    1x32 (e8m0) scales using the same recipe as `dynamic_mxfp8_quant`.

    Args:
        x: Input tensor of shape (M, N1, N2). N2 must be a multiple of 32.
        quant_dtype: FP8 dtype to cast quantized values to (defaults to
            torch.float8_e4m3fn).

    Returns:
        Tuple of:
            out: FP8 tensor of shape (M, N1 * N2).
            out_scales: e8m0 (uint8) scale tensor of shape
                (M, (N1 * N2) // 32).
    """
    assert x.dim() == 3, f"x must be 3D, got {x.dim()}"
    M, N1, N2 = x.shape
    assert (
        N2 % _QUANT_BLOCK_SIZE == 0
    ), f"N2={N2} must be a multiple of {_QUANT_BLOCK_SIZE}"

    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), _QUANT_BLOCK_SIZE)
    N = N1 * N2

    out = torch.empty((M, N), dtype=quant_dtype, device=x.device)
    out_scales = torch.empty(
        (M, N // _QUANT_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )

    grid = (M, N1)
    _fused_flatten_mxfp8_quant_kernel[grid](
        x,
        out,
        out_scales,
        *x.stride(),
        *out.stride(),
        *out_scales.stride(),
        N2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
    )

    return out, out_scales


K3_MLA_640_REC = 640
K3_MLA_640_NOPE = 512
K3_MLA_640_ROPE = 64


def _k3_mla_640_cache(cache: torch.Tensor, what: str = "kv_cache") -> torch.Tensor:
    """The cache as uint8 [num_blocks, block_size, 640]. It may be a strided view
    into a packed block row; a flat [rows, 640] grid is the same with page size 1."""
    if cache.ndim == 2:
        cache = cache.unsqueeze(1)
    if (
        cache.ndim != 3
        or cache.element_size() != 1
        or cache.shape[2] != K3_MLA_640_REC
        or cache.stride(2) != 1
        or cache.stride(1) != K3_MLA_640_REC
        or cache.stride(0) % K3_MLA_640_REC
        or cache.stride(0) < cache.shape[1] * K3_MLA_640_REC
    ):
        raise ValueError(
            f"{what} must be 1-byte [num_blocks, block_size, 640] or [rows, 640] "
            f"with contiguous records and a block stride that is a multiple of "
            f"640, got {tuple(cache.shape)} {cache.dtype} strides {cache.stride()}"
        )
    return cache.view(torch.uint8)


def _k3_rows(x: torch.Tensor, rows: int, width: int, what: str) -> torch.Tensor:
    """bf16 [rows, width] with unit inner stride (a view whenever x allows it)."""
    if x.dtype != torch.bfloat16 or x.numel() != rows * width:
        raise ValueError(
            f"{what} must be bf16 with {rows} x {width} elements, got "
            f"{x.dtype} {tuple(x.shape)}"
        )
    x = x.reshape(rows, width)
    return x if x.stride(1) == 1 else x.contiguous()


def k3_mla_640_concat_and_cache(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Quantize and write fp8_k3_mla_640 records (the call shape of vLLM
    ops.concat_and_cache_mla).

    kv_c [T, 512] and k_pe [T, 64] (rope already applied) are bf16; token t goes
    to slot_mapping[t] = block * block_size + pos, a negative slot is skipped.
    """
    cache = _k3_mla_640_cache(kv_cache)
    T = slot_mapping.shape[0]
    if T == 0:
        return
    nope = _k3_rows(kv_c, T, K3_MLA_640_NOPE, "kv_c")
    rope = _k3_rows(k_pe, T, K3_MLA_640_ROPE, "k_pe")
    _k3_mla_640_pack_kernel[(T,)](
        nope,
        rope,
        slot_mapping.to(torch.int64).contiguous(),
        cache,
        nope.stride(0),
        rope.stride(0),
        cache.stride(0),
        BLOCK_SIZE=cache.shape[1],
        HAS_SLOTS=True,
        num_warps=1,
    )


def k3_mla_640_q_pack(
    q_nope: torch.Tensor, q_pe: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 q_nope [..., 512] and q_pe [..., 64] -> uint8 [..., 640], one
    fp8_k3_mla_640 record per (token, head): what sparse_mla_fwd reads as q."""
    lead = q_nope.shape[:-1]
    rows = q_nope.numel() // K3_MLA_640_NOPE
    nope = _k3_rows(q_nope, rows, K3_MLA_640_NOPE, "q_nope")
    rope = _k3_rows(q_pe, rows, K3_MLA_640_ROPE, "q_pe")
    if out is None:
        out = torch.empty(
            (*lead, K3_MLA_640_REC), dtype=torch.uint8, device=q_nope.device
        )
    elif (
        out.shape != (*lead, K3_MLA_640_REC)
        or out.dtype != torch.uint8
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be contiguous uint8 {(*lead, K3_MLA_640_REC)}, got "
            f"{out.dtype} {tuple(out.shape)}"
        )
    if rows:
        _k3_mla_640_pack_kernel[(rows,)](
            nope,
            rope,
            out,
            out,
            nope.stride(0),
            rope.stride(0),
            K3_MLA_640_REC,
            BLOCK_SIZE=1,
            HAS_SLOTS=False,
            num_warps=1,
        )
    return out


def k3_mla_640_cp_gather_and_upconvert(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """Gather fp8_k3_mla_640 records and dequantize them to bf16 (the signature
    of vLLM ops.cp_gather_and_upconvert_fp8_kv_cache).

    Request r fills dst rows [workspace_starts[r], workspace_starts[r + 1]) (the
    last one up to dst.shape[0]) with its tokens from position seq_starts[r]
    (0 without it) on, token p read from block block_table[r, p // block_size].

    Args:
        src_cache: uint8 [num_blocks, block_size, 640].
        dst: bf16 [total_tokens, 576] workspace: 512 nope | 64 rope.
        block_table: int32 [num_reqs, max_blocks].
        workspace_starts: int32 [num_reqs].
        batch_size: num_reqs.
        seq_starts: optional int32 [num_reqs].
    """
    cache = _k3_mla_640_cache(src_cache, "src_cache")
    width = K3_MLA_640_NOPE + K3_MLA_640_ROPE
    if dst.dtype != torch.bfloat16 or dst.ndim != 2 or dst.shape[1] != width:
        raise ValueError(f"dst must be bf16 [total_tokens, {width}]")
    if dst.stride(1) != 1 or block_table.ndim != 2 or block_table.stride(1) != 1:
        raise ValueError("dst and block_table must be contiguous along their rows")
    for name, t in (
        ("block_table", block_table),
        ("workspace_starts", workspace_starts),
        ("seq_starts", seq_starts),
    ):
        if t is not None and (t.dtype != torch.int32 or t.stride(-1) != 1):
            raise ValueError(f"{name} must be int32 with unit stride, got {t.dtype}")
    if dst.shape[0] == 0 or batch_size == 0:
        return
    has_seq_starts = seq_starts is not None
    _k3_mla_640_gather_kernel[(dst.shape[0],)](
        cache,
        dst,
        block_table,
        workspace_starts,
        seq_starts if has_seq_starts else workspace_starts,
        batch_size,
        cache.stride(0),
        block_table.stride(0),
        dst.stride(0),
        BLOCK_SIZE=cache.shape[1],
        HAS_SEQ_STARTS=has_seq_starts,
        num_warps=1,
    )
