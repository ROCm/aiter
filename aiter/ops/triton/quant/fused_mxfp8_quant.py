# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fused_mxfp8_quant import (
    _fused_deepseek_v4_dequant_gather_k_cache_kernel,
    _fused_deepseek_v4_quantize_and_insert_k_kernel,
    _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_kernel,
    _fused_deepseek_v4_mxfp8_quant_q_pack_kernel,
    _fused_dual_rmsnorm_mxfp8_quant_kernel,
    _fused_flatten_mxfp8_quant_kernel,
    _fused_rms_mxfp8_kernel,
)

__all__ = [
    "fused_deepseek_v4_compress_norm_rope_store",
    "fused_deepseek_v4_compress_norm_rope_store_two_stage",
    "fused_deepseek_v4_dequantize_and_gather_k_cache",
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "fused_deepseek_v4_quantize_and_insert_k_cache",
    "fused_deepseek_v4_mxfp8_quant_q_pack",
    "fused_dual_rmsnorm_mxfp8_quant",
    "fused_flatten_mxfp8_quant",
    "fused_rms_mxfp8_quant",
]

_QUANT_BLOCK_SIZE = 32

# DSv4 a8w8 Q operand: 448 NoPE (UE8M0 fp8, 64-element groups) + 64 RoPE
# (bf16, never quantized). The decode kernel reads the scale bytes inline,
# each duplicated, because its scaled-MMA blocks are half the quant group.
_V4_DIM_NOPE = 448
_V4_DIM_ROPE = 64
_V4_DIM_QK = _V4_DIM_NOPE + _V4_DIM_ROPE
_FP8_GROUP_SIZE = 64
_V4_NUM_TILES = _V4_DIM_NOPE // _FP8_GROUP_SIZE




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


def fused_deepseek_v4_mxfp8_quant_q_pack(q: torch.Tensor):
    """``[..., 512]`` bf16 Q -> ``(packed [..., 512] fp8, rope [..., 64] bf16)``.

    The Q form ``_pa_decode_sparse_v4`` reads on gfx1250 (a8w8). RoPE is never
    quantized, which is why this returns a pair.

    A zeroed (padding) head packs to zero data with the 1e-4 floor's exponent in
    its scale bytes -- harmless, since a zero mantissa dequantizes to zero
    whatever the exponent says. What must never appear is an UNWRITTEN scale
    byte: 0xFF is E8M0 NaN and a NaN scale poisons a whole score row through
    ``0 * NaN``.
    """
    if q.shape[-1] != _V4_DIM_QK:
        raise RuntimeError(
            f"q last dim must be {_V4_DIM_QK} (448 NoPE + 64 RoPE), got "
            f"{q.shape[-1]}"
        )
    q = q.contiguous()
    lead = q.shape[:-1]
    rows = 1
    for d in lead:
        rows *= d
    packed = torch.empty((*lead, _V4_DIM_QK), dtype=torch.uint8, device=q.device)
    rope = torch.empty((*lead, _V4_DIM_ROPE), dtype=q.dtype, device=q.device)
    _fused_deepseek_v4_mxfp8_quant_q_pack_kernel[(rows,)](
        q, packed, rope, float(torch.finfo(torch.float8_e4m3fn).max),
        _V4_DIM_NOPE, _V4_DIM_ROPE, _V4_DIM_QK, _FP8_GROUP_SIZE, _V4_NUM_TILES,
        num_warps=4,
    )
    return packed.view(torch.float8_e4m3fn), rope


# The ALIGNED paged KV record. 640 = 5 * 128, so every token starts on a
# 128-byte boundary and TDM reads it on the direct global->L2->LDS path.
_V4_REC_ALIGNED = 640
_V4_SC_IN_REC = _V4_DIM_NOPE  # 448
_V4_ROPE_IN_REC = 512  # the 2buff row is 512 B; RoPE follows it


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache_block_size: int,
    eps: float,
    padded_heads: int,
    apply_q_norm: bool = True,
    pack_q: bool = True,
    write_q: bool = True,
    q_packed_out: torch.Tensor | None = None,
    q_rope_out: torch.Tensor | None = None,
    use_fnuz: bool = False,
):
    """The whole DSv4 decode producer in one launch.

    Q gets RMSNorm (no weight) then GPT-J RoPE, and -- when ``pack_q`` -- the
    UE8M0 fp8 pack off the same registers, so packing costs no second pass.
    KV gets RoPE, UE8M0 quant and an insert into the aligned paged cache.

    Args:
        q: ``[T, H, 512]`` bf16.
        kv: ``[T, 512]`` bf16, the compressed KV for these tokens.
        kv_cache: ``[nb, block, 640]`` uint8, or any view whose dim-0 stride is
            the block stride -- a per-layer view of a pooled allocation is one.
        kv_slot_mapping: ``[T]``, ``-1`` for a token with no cache slot.
        positions: ``[T]`` RoPE positions, shared by Q and KV.
        cos_sin_cache: ``[max_pos, 64]`` fp32, laid out cos || sin.
        padded_heads: the head count the decode kernel expects; slots at or
            past ``H`` are zero-filled.

    Returns:
        ``q_bf16`` when ``pack_q`` is False, else
        ``(q_bf16, q_packed, q_rope)``.

    ``write_q=False`` skips the bf16 Q store entirely -- the returned tensor is
    then UNINITIALISED and only its shape is meaningful. Use it only when the
    consumer reads the packed pair alone (a pure-decode step on the aiter
    path); anything that reads Q as bf16, prefill included, must leave it on.
    """
    t, h, dim = q.shape
    if dim != _V4_DIM_QK:
        raise RuntimeError(f"q must be [T, H, {_V4_DIM_QK}], got {tuple(q.shape)}")
    if padded_heads < h:
        raise RuntimeError(f"padded_heads {padded_heads} < H {h}")
    if kv.dim() != 2 or kv.shape != (t, _V4_DIM_QK):
        raise RuntimeError(
            f"kv must be [{t}, {_V4_DIM_QK}], got {tuple(kv.shape)}"
        )
    if kv_cache.shape[-1] != _V4_REC_ALIGNED:
        raise RuntimeError(
            f"kv_cache records must be {_V4_REC_ALIGNED} B, got "
            f"{kv_cache.shape[-1]}"
        )
    q = q.contiguous()
    kv = kv.contiguous()

    q_out = torch.empty(t, padded_heads, dim, dtype=q.dtype, device=q.device)
    if pack_q:
        # Caller-owned buffers when supplied. Allocating here puts them in
        # whatever pool is active -- the CUDA graph pool during capture -- and
        # a caller that holds them past the step then races the allocator.
        if q_packed_out is not None:
            if q_packed_out.shape != (t, padded_heads, dim):
                raise RuntimeError(
                    f"q_packed_out must be {(t, padded_heads, dim)}, got "
                    f"{tuple(q_packed_out.shape)}"
                )
            q_packed = q_packed_out.view(torch.uint8)
        else:
            q_packed = torch.empty(
                t, padded_heads, dim, dtype=torch.uint8, device=q.device
            )
        if q_rope_out is not None:
            if q_rope_out.shape != (t, padded_heads, _V4_DIM_ROPE):
                raise RuntimeError(
                    f"q_rope_out must be {(t, padded_heads, _V4_DIM_ROPE)}, "
                    f"got {tuple(q_rope_out.shape)}"
                )
            q_rope = q_rope_out
        else:
            q_rope = torch.empty(
                t, padded_heads, _V4_DIM_ROPE, dtype=q.dtype, device=q.device
            )
    else:
        # unused under PACK_Q=False, but the launch still has to typecheck
        q_packed = q_rope = q_out

    fp8_max = 224.0 if use_fnuz else float(torch.finfo(torch.float8_e4m3fn).max)
    # One extra slot along dim 1 carries the KV row for the token.
    _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_kernel[
        (t, padded_heads + 1)
    ](
        q,
        q_out,
        q_packed,
        q_rope,
        kv,
        kv_cache,
        kv_slot_mapping,
        positions,
        cos_sin_cache,
        kv_cache.stride(0),
        kv_cache_block_size,
        float(eps),
        fp8_max,
        h,
        padded_heads,
        _V4_DIM_QK,
        _V4_DIM_NOPE,
        _V4_DIM_ROPE,
        _FP8_GROUP_SIZE,
        _V4_NUM_TILES,
        _V4_REC_ALIGNED,
        _V4_SC_IN_REC,
        _V4_ROPE_IN_REC,
        bool(apply_q_norm),
        bool(pack_q),
        bool(write_q),
        bool(use_fnuz),
        num_warps=4,
    )
    if pack_q:
        return q_out, q_packed.view(torch.float8_e4m3fn), q_rope
    return q_out


def _rec_geometry(k_cache):
    """(rec_bytes, sc_in_rec, rope_in_rec, sc_step) from the record size.

    584 -> packed: fp8 [0,448) | bf16 [448,576), scales in a per-block region
    640 -> aligned: fp8 [0,448) | scales [448,462) | pad | bf16 [512,640)

    The PACKED stride between tokens is 576, not 584: its scales are not in the
    record, they are in a region after the block's token data, and 584 is only
    the per-token accounting size. Returning 584 here walks every token 8 bytes
    too far.

    ``sc_step`` is 2 on the aligned record because every group's scale is
    written twice -- one byte per 32 columns, which is what the decode kernel's
    scaled MMA reads, against a 64-element quant group.
    """
    # len(shape) rather than .dim(): vLLM's warmup feeds this a
    # TritonWarmupTensor, which carries a shape but is not a torch.Tensor.
    rec = int(k_cache.shape[-1]) if len(k_cache.shape) == 3 else 584
    return (640, 448, 512, 2) if rec == 640 else (576, 0, 448, 1)


def fused_deepseek_v4_quantize_and_insert_k_cache(
    k: torch.Tensor,  # [num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_bytes] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    block_size: int = 64,
    is_ue8m0: bool = True,
    use_fnuz: bool = False,
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576

    ``use_fnuz=True`` selects FNUZ E4M3 cache encoding and is only valid on
    platforms whose FP8 format is FNUZ. ``use_fnuz=False`` selects OCP E4M3,
    which is used by OCP-encoded caches even on gfx942.
    """
    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
    assert k.dtype == torch.bfloat16, f"K must be bf16, got {k.dtype}"
    assert is_ue8m0, "Only support ue8m0 quantization."

    # NOTE: When using DP, slot_mapping.shape[0] can be less than k.shape[0] due to
    # padding. Always use slot_mapping.shape[0] as the token count.
    num_tokens = slot_mapping.shape[0]
    block_stride = k_cache.stride(0)  # bytes per block

    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    if use_fnuz:
        if not True:
            raise ValueError("use_fnuz=True requires a platform using FNUZ FP8")
        FP8_MAX = 240.0
    else:
        FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    _REC_B, _SC_OFF, _ROPE_OFF, _SC_STEP = _rec_geometry(k_cache)
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    grid = (num_tokens,)

    _fused_deepseek_v4_quantize_and_insert_k_kernel[grid](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        input_dim=512,
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        rec_bytes=_REC_B,
        sc_in_rec=_SC_OFF,
        rope_in_rec=_ROPE_OFF,
        sc_step=_SC_STEP,
        block_stride=block_stride,
        fp8_max=FP8_MAX,
        n_quant_blocks=8,
        use_fnuz=use_fnuz,
    )


# One workgroup row per request; the second grid axis splits the gather.
_V4_GATHER_NUM_WORKERS = 128


def fused_deepseek_v4_dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int = 0,
    use_fnuz: bool = False,
) -> None:
    """Dequantize the paged K cache into ``out``, gathering per request.

    Args:
        out: ``[num_reqs, max_len, 512]`` bf16, written in place.
        k_cache: the paged cache; dim-0 stride is the block stride. A 3-D
            cache's last dim selects the record layout (640 aligned, else
            packed).
        seq_lens: ``[num_reqs]`` sequence lengths.
        gather_lens: ``[num_reqs]`` gather counts, or None to use seq_lens.
        block_table: ``[num_reqs, max_blocks_per_seq]``.
        offset: first token position to read.
        use_fnuz: fp8 flavour of the cache's encoder.
    """
    num_reqs = seq_lens.shape[0]
    rec_bytes, sc_in_rec, rope_in_rec, sc_step = _rec_geometry(k_cache)
    _fused_deepseek_v4_dequant_gather_k_cache_kernel[
        (num_reqs, _V4_GATHER_NUM_WORKERS)
    ](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=448,
        bf16_dim=64,
        scale_dim=8,
        quant_block=64,
        cache_block_size=block_size,
        token_data_size=576,
        rec_bytes=rec_bytes,
        sc_in_rec=sc_in_rec,
        rope_in_rec=rope_in_rec,
        sc_step=sc_step,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=float(torch.finfo(torch.float8_e4m3fn).max),
        n_quant_blocks=7,
        use_fnuz=use_fnuz,
    )


# ---------------------------------------------------------------------------
# DSv4 KV compressor launchers (ported from vLLM)
# ---------------------------------------------------------------------------
from functools import lru_cache  # noqa: E402
from typing import Any  # noqa: E402

from aiter.ops.triton.utils._triton import arch_info  # noqa: E402

# The compressor sanitises cache NaNs only where the decode path needs it;
# vLLM gated this on its own _ON_GFX950, so resolve the same thing from aiter.
_ON_GFX950 = arch_info.get_arch() == "gfx950"

from aiter.ops.triton._triton_kernels.quant.fused_mxfp8_quant import (  # noqa: E402
    _compress_gather_split_sparse_attn,
    _finalize_norm_rope_quant_store_sparse_attn,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)

def fused_deepseek_v4_compress_norm_rope_store(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Shared triton launcher for the fused compress+norm+RoPE+insert path.

    Picks one of the three kernels in this module based on ``head_dim`` and
    ``use_fp4_cache``. Identical launch signature for all three.
    """
    if head_dim != 512:
        # Only the sparse-attn compressor was ported. The indexer variants
        # (head_dim 128, plain and mxfp4) still live in vLLM behind its
        # VllmTritonJitKernel warmup wrapper, which has no aiter equivalent --
        # callers on that path must keep using vLLM's launcher.
        raise NotImplementedError(
            f"aiter's fused_deepseek_v4_compress_norm_rope_store supports head_dim 512, "
            f"got {head_dim}; use vLLM's launcher for the indexer path"
        )
    kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
    num_warps = 8
    kernel_kwargs = {"SANITIZE_CACHE_NANS": _ON_GFX950}

    kernel[(num_actual,)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # KV cache
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # paged KV cache block size (tokens per block)
        # constexprs
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SC_IN_REC=448 if token_stride == 640 else 0,
        ROPE_IN_REC=512 if token_stride == 640 else 448,
        SC_STEP=2 if token_stride == 640 else 1,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **kernel_kwargs,
        **pdl_kwargs,
    )


# =============================================================================
# DeepseekV4 Attention path (head=512, nope=448 FP8 + rope=64 bf16)
# =============================================================================

@lru_cache(maxsize=1)
def _n_cu() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count

def _pick_compress_num_splits(
    num_actual: int, compress_ratio: int, head_dim: int
) -> int:
    """Occupancy-targeted column splits for the cr>=128 head=512 compressor.

    Sizes the per-token fan-out so (estimated computing tokens) * num_splits ~
    #CU, capped by head tiling at a 32-wide min tile, as a power-of-2 divisor of
    head_dim.
    """
    max_splits = head_dim // 32
    est_compute = max(1, num_actual // compress_ratio)
    target = -(-_n_cu() // est_compute)  # ceil(#CU / est_compute)
    ns = 1
    while ns * 2 <= min(target, max_splits) and head_dim % (ns * 2) == 0:
        ns *= 2
    return ns

def _launch_two_stage_sparse_attn_compressor(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    head_dim: int,
    rope_head_dim: int,
    num_actual: int,
    compress_scratch: torch.Tensor,
) -> None:
    num_splits = _pick_compress_num_splits(num_actual, compress_ratio, head_dim)
    head_tile = head_dim // num_splits
    scratch = compress_scratch[:num_actual]
    _compress_gather_split_sparse_attn[(num_actual * num_splits,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        positions,
        slot_mapping,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        scratch,
        scratch.stride(0),
        HEAD_SIZE=head_dim,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        NUM_SPLITS=num_splits,
        HEAD_TILE=head_tile,
    )
    _finalize_norm_rope_quant_store_sparse_attn[(num_actual,)](
        scratch,
        scratch.stride(0),
        positions,
        slot_mapping,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        kv_slot_mapping,
        kv_cache.shape[1],
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        COMPRESS_RATIO=compress_ratio,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SC_IN_REC=448 if token_stride == 640 else 0,
        ROPE_IN_REC=512 if token_stride == 640 else 448,
        SC_STEP=2 if token_stride == 640 else 1,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        SANITIZE_CACHE_NANS=_ON_GFX950,
    )

def fused_deepseek_v4_compress_norm_rope_store_two_stage(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    num_decode_tokens: int,
    compress_scratch: torch.Tensor,
) -> None:
    """Two-stage split compressor dispatch for head=512 cr>=128 (no-overlap)

    Run the occupancy-fanned two-stage split for prefill [num_decodee_tokens:]
    to fill the CUs, and use the original single-pass launcher
    for decode [0, num_decode_tokens)
    """
    num_decodes = min(max(num_decode_tokens, 0), num_actual)
    num_prefills = num_actual - num_decodes
    if num_prefills > 0:
        _launch_two_stage_sparse_attn_compressor(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices[num_decodes:],
            positions=positions[num_decodes:],
            slot_mapping=slot_mapping[num_decodes:],
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=compress_ratio,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            kv_slot_mapping=k_cache_metadata.slot_mapping[num_decodes:],
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=scale_dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            num_actual=num_prefills,
            compress_scratch=compress_scratch,
        )
    if num_decodes > 0:
        fused_deepseek_v4_compress_norm_rope_store(
            state_cache=state_cache,
            num_actual=num_decodes,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=overlap,
            use_fp4_cache=use_fp4_cache,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=scale_dim,
        )


# =============================================================================
# Indexer path (head=128, all FP8, single quant block)
# =============================================================================
