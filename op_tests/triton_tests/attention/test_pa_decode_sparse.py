# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# from __future__ import annotations

import pytest
import torch
import triton

from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import checkAllclose


def _sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Per-batch sparse multi-head attention with sink in the denominator only.

    Shapes:
        q:           [B, M, H, D]
        kv:          [B, N, D]
        attn_sink:   [H]
        topk_idxs:   [B, M, K] int32, -1 means skip
    Returns:
        [B, M, H, D] same dtype as q.
    """
    B, M, H, _D = q.shape
    K = topk_idxs.shape[-1]
    device = q.device
    out_dtype = q.dtype

    valid = topk_idxs != -1
    safe_idxs = topk_idxs.clamp(min=0).long()
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, K)
    kv_gathered = kv[batch_idx, safe_idxs]  # [B, M, K, D]
    kv_f32 = kv_gathered.float()
    kv_f32 = torch.where(
        valid.unsqueeze(-1), kv_f32, torch.zeros((), dtype=kv_f32.dtype, device=device)
    )

    q_f32 = q.float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q_f32, kv_f32) * float(softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    sink = attn_sink.float().view(1, 1, H, 1).expand(B, M, H, 1)
    combined = torch.cat([scores, sink], dim=-1)
    cmax = combined.amax(dim=-1, keepdim=True)
    cmax = torch.where(
        cmax == float("-inf"),
        torch.zeros((), dtype=cmax.dtype, device=device),
        cmax,
    )
    weights = (combined - cmax).exp()
    denom = weights.sum(dim=-1, keepdim=True)
    weights = weights / denom.clamp(min=1e-30)
    weights_kv = weights[..., :K]
    out = torch.einsum("bmhk,bmkd->bmhd", weights_kv, kv_f32)
    return out.to(out_dtype)


def pa_decode_sparse_reference(
    q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale
):
    """Pure-torch reference that materialises per-token KV via gather."""
    T = q.size(0)
    indptr = kv_indptr.to(torch.int64)
    spans = (indptr[1:] - indptr[:T]).clamp(min=0)
    k_dim = int(spans.max().item()) if T > 0 else 1
    if k_dim == 0:
        k_dim = 1
    topk_idxs = torch.full((T, k_dim), -1, device=q.device, dtype=torch.int32)
    for t in range(T):
        s = int(indptr[t].item())
        n = int(spans[t].item())
        if n > 0:
            topk_idxs[t, :n] = kv_indices[s : s + n].to(torch.int32)
    return _sparse_attn_torch(
        q.unsqueeze(0),
        unified_kv.unsqueeze(0),
        attn_sink,
        topk_idxs.unsqueeze(0),
        softmax_scale,
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------


def _make_inputs(
    T: int,
    H: int,
    D: int,
    kv_len_per_token: int,
    total_pages: int,
    dtype=torch.bfloat16,
    seed: int = 0,
    include_sentinels: bool = False,
    variable_len: bool = False,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    q = torch.randn(T, H, D, dtype=dtype, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=dtype, device=device) * 0.5
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    # Per-token kv_len: fixed or random in [1, kv_len_per_token].
    if variable_len:
        kv_lens = torch.randint(
            low=1,
            high=kv_len_per_token + 1,
            size=(T,),
            device=device,
            dtype=torch.int64,
        )
    else:
        kv_lens = torch.full((T,), kv_len_per_token, device=device, dtype=torch.int64)

    indptr = torch.zeros(T + 1, device=device, dtype=torch.int64)
    indptr[1:] = kv_lens.cumsum(0)
    total_indices = int(indptr[-1].item())

    indices = torch.randint(
        low=0,
        high=total_pages,
        size=(total_indices,),
        device=device,
        dtype=torch.int32,
    )
    if include_sentinels and total_indices > 0:
        # Sprinkle a few -1 sentinels.
        n_sentinel = max(1, total_indices // 16)
        sentinel_pos = torch.randperm(total_indices, device=device)[:n_sentinel]
        indices[sentinel_pos] = -1

    indptr = indptr.to(torch.int32)
    softmax_scale = float(D) ** -0.5
    return q, unified_kv, indices, indptr, attn_sink, softmax_scale


# ---------------------------------------------------------------------------
# skip_reduce: the wrapper hands back the pre-reduce split-K partials and the
# caller is responsible for the log-sum-exp combine + sink fold. This mirrors
# the _pa_decode_sparse_reduce kernel in pure torch so we can validate the
# partials against the dense reference.
# ---------------------------------------------------------------------------


def _wrapper_main_kernel_params(T: int, H: int, D: int):
    """Reproduce the (use_exp2, block_k) the wrapper picks for the main kernel.

    Must stay in sync with ``pa_decode_sparse``'s USE_EXP2 and block_k logic.
    """
    use_gluon = arch_info.get_arch() == "gfx1250"
    use_exp2 = True
    if use_gluon:
        if H >= 128:
            block_h = 128
        elif H >= 64:
            if T >= 2048:
                block_h = 64
            elif T >= 32:
                block_h = 32
            else:
                block_h = 16
        elif H >= 32:
            if T >= 256:
                block_h = 32
            else:
                block_h = 16
        else:
            block_h = triton.next_power_of_2(H)
    else:
        block_h = triton.next_power_of_2(min(H, 16))
    if use_gluon:
        block_k = 16
        if block_h == 128:
            block_k = 32
    else:
        block_k = 16 if D >= 256 else 32
    return use_exp2, block_k


def _reduce_partials_torch(
    acc_partial, m_partial, l_partial, attn_sink, kv_indptr, block_k, use_exp2
):
    """Pure-torch port of _pa_decode_sparse_reduce.

    Shapes:
        acc_partial: [T, KV_SPLITS, H_padded, D] fp32
        m_partial:   [T, KV_SPLITS, H_padded]    fp32
        l_partial:   [T, KV_SPLITS, H_padded]    fp32
    Returns [T, H, D] in attn_sink-implied output dtype (bf16/fp16 caller casts).
    """
    T, kv_splits, _, D = acc_partial.shape
    H = attn_sink.shape[0]
    device = acc_partial.device

    expfn = torch.exp2 if use_exp2 else torch.exp
    LOG2E = 1.4426950408889634
    sink_scale = LOG2E if use_exp2 else 1.0

    indptr = kv_indptr.to(torch.int64)
    kv_lens = (indptr[1 : T + 1] - indptr[:T]).clamp(min=0)
    seg_ids = torch.arange(kv_splits, device=device)
    sink = attn_sink.float() * sink_scale  # [H]

    out = torch.empty(T, H, D, dtype=torch.float32, device=device)
    for t in range(T):
        n = int(kv_lens[t].item())
        # Match the kernel's tiles_per_segment / act_num_segments masking so we
        # ignore the stale (uninitialised) partial-buffer slots that the split
        # kernel early-returned on.
        if n <= 0:
            act_num_segments = 0
        else:
            tiles_per_segment = triton.cdiv(n, kv_splits * block_k)
            act_num_segments = triton.cdiv(n, tiles_per_segment * block_k)
        seg_mask = seg_ids < act_num_segments  # [KV_SPLITS]

        m_p = m_partial[t, :, :H].clone()  # [KV_SPLITS, H]
        l_p = l_partial[t, :, :H]
        a_p = acc_partial[t, :, :H, :]  # [KV_SPLITS, H, D]
        m_p = torch.where(seg_mask[:, None], m_p, torch.full_like(m_p, float("-inf")))

        m_max = m_p.max(dim=0).values  # [H]
        is_dead = m_p == float("-inf")  # [KV_SPLITS, H]
        alpha = torch.where(is_dead, torch.zeros_like(m_p), expfn(m_p - m_max[None, :]))
        l_comb = torch.where(is_dead, torch.zeros_like(l_p), l_p * alpha).sum(0)  # [H]
        acc_comb = torch.where(
            is_dead[:, :, None], torch.zeros_like(a_p), a_p * alpha[:, :, None]
        ).sum(
            0
        )  # [H, D]

        m_final = torch.maximum(m_max, sink)
        alpha_kv = expfn(m_max - m_final)
        alpha_sink = expfn(sink - m_final)
        l_final = l_comb * alpha_kv + alpha_sink
        acc_final = acc_comb * alpha_kv[:, None]
        denom = l_final.clamp(min=1e-30)
        out[t] = torch.where(
            l_final[:, None] > 0.0,
            acc_final / denom[:, None],
            torch.zeros_like(acc_final),
        )
    return out


@pytest.mark.parametrize("T", [1, 64, 256, 2048])
@pytest.mark.parametrize("H", [16, 32, 64, 128])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("kv_len", [136, 388, 1024])
@pytest.mark.parametrize("var_len", [True, False])
@pytest.mark.parametrize("sentinels", [False])
@pytest.mark.parametrize("skip_reduce", [False])
def test_pa_decode_sparse_vs_reference(
    T, H, D, kv_len, var_len, sentinels, skip_reduce
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    pages = T * kv_len
    q, ukv, indices, indptr, sink, scale = _make_inputs(
        T,
        H,
        D,
        kv_len,
        pages,
        include_sentinels=sentinels,
        variable_len=var_len,
    )

    ref = pa_decode_sparse_reference(q, ukv, indices, indptr, sink, scale)
    result = pa_decode_sparse(
        q,
        ukv,
        indices,
        indptr,
        sink,
        scale,
        has_invalid=sentinels,
        skip_reduce=skip_reduce,
    )

    if isinstance(result, tuple):
        # skip_reduce with the split-K path active (kv_splits > 1): the wrapper
        # returns raw partials, so do the log-sum-exp combine + sink fold here.
        acc_partial, m_partial, l_partial = result
        use_exp2, block_k = _wrapper_main_kernel_params(T, H, D)
        out = _reduce_partials_torch(
            acc_partial, m_partial, l_partial, sink, indptr, block_k, use_exp2
        ).to(q.dtype)
    else:
        # kv_splits == 1 (skip_reduce is a no-op) or skip_reduce=False: the
        # wrapper already returns the final output.
        out = result

    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            out.to(torch.bfloat16),
            ref.to(torch.bfloat16),
            atol=5e-3,
            rtol=5e-3,
            tol_err_ratio=tol_err_ratio,
            msg="pa_decode_sparse output",
        )
        <= tol_err_ratio
    )


# ---------------------------------------------------------------------------
# FP8 KV cache quantization helpers
# ---------------------------------------------------------------------------

_FP8_GROUP_SIZE = 64
_FP8_DTYPE = torch.float8_e4m3fnuz


def _quantize_kv_fp8(unified_kv, group_size=_FP8_GROUP_SIZE):
    """Quantize bf16/fp16 unified_kv to (fp8, scales) with 1xGROUP_SIZE block scaling.

    Returns (kv_fp8, kv_scales) where kv_fp8 is float8_e4m3fnuz and
    kv_scales is [total_pages, D // group_size] fp32.
    """
    total_pages, D = unified_kv.shape
    assert D % group_size == 0
    num_groups = D // group_size
    kv_f32 = unified_kv.float().view(total_pages, num_groups, group_size)
    amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    fp8_max = torch.finfo(_FP8_DTYPE).max
    scales = (amax / fp8_max).squeeze(-1)  # [total_pages, num_groups]
    kv_scaled = kv_f32 / amax * fp8_max
    kv_fp8 = kv_scaled.view(total_pages, D).to(_FP8_DTYPE)
    return kv_fp8, scales.to(torch.float32)


def _dequant_kv_fp8(kv_fp8, kv_scales, group_size=_FP8_GROUP_SIZE):
    """Dequantize for reference comparison."""
    total_pages, D = kv_fp8.shape
    num_groups = D // group_size
    kv_f32 = kv_fp8.float().view(total_pages, num_groups, group_size)
    scales_expanded = kv_scales.unsqueeze(-1).expand(
        total_pages, num_groups, group_size
    )
    return (kv_f32 * scales_expanded).view(total_pages, D)


# ---------------------------------------------------------------------------
# DSv4 "2buff" packed-fp8 KV layout — byte-identical to what the gfx1250 MLA-v4
# asm decode kernel reads (aiter.mla.mla_decode_fwd_v4_nm ->
# _ZN5aiter35mla_a8w8_qh64_1tg_16mx4_64nx1_sparseE). Mirrors
# op_tests/test_mla_v4_kargpreld.py::_native_to_2buff_for_asm and ATOM's
# atom/model_ops/v4_kernels/v4_quant.py (V4_* constants):
#
#   packed row [512 B] = [ NoPE 448 x fp8-e4m3
#                        | 14 E8M0 scale bytes, each 64-elt group's scale
#                          written TWICE (s0,s0,s1,s1,...,s6,s6)
#                        | 50 B pad ]
#   rope plane [64] bf16, a separate tensor
# ---------------------------------------------------------------------------

_V4_DIM_NOPE = 448
_V4_DIM_ROPE = 64
_V4_DIM_QK = _V4_DIM_NOPE + _V4_DIM_ROPE  # 512
_V4_TILE = 64
_V4_NUM_TILES = _V4_DIM_NOPE // _V4_TILE  # 7
_V4_FP8 = torch.float8_e4m3fn  # OCP e4m3, what the asm .co consumes


def v4_pack_2buff(x_bf16):
    """``[..., 512]`` bf16 (NoPE||RoPE) -> ``(packed [..., 512] fp8, rope [..., 64] bf16)``."""
    assert x_bf16.shape[-1] == _V4_DIM_QK
    lead = x_bf16.shape[:-1]
    nope = x_bf16[..., :_V4_DIM_NOPE].float()
    rope = x_bf16[..., _V4_DIM_NOPE:].contiguous()

    tiled = nope.reshape(*lead, _V4_NUM_TILES, _V4_TILE)
    fp8_max = float(torch.finfo(_V4_FP8).max)
    # amax/fp8_max rounded UP to a power of two, exactly as E8M0 stores it.
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    nope_fp8 = (tiled / scale.unsqueeze(-1)).to(_V4_FP8).reshape(*lead, _V4_DIM_NOPE)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)

    packed = torch.zeros((*lead, _V4_DIM_QK), dtype=torch.uint8, device=x_bf16.device)
    packed[..., :_V4_DIM_NOPE] = nope_fp8.view(torch.uint8)
    # the kernel reads each group's scale twice (its scaled-MMA blocks are 32
    # elements wide, the quant group is 64), so duplicate every byte
    packed[..., _V4_DIM_NOPE : _V4_DIM_NOPE + 2 * _V4_NUM_TILES] = (
        e8m0.repeat_interleave(2, dim=-1)
    )
    return packed.view(_V4_FP8), rope


def v4_unpack_2buff(packed, rope):
    """Inverse of ``v4_pack_2buff`` -> ``[..., 512]`` bf16."""
    lead = packed.shape[:-1]
    u8 = packed.view(torch.uint8)
    nope = (
        u8[..., :_V4_DIM_NOPE]
        .view(_V4_FP8)
        .float()
        .reshape(*lead, _V4_NUM_TILES, _V4_TILE)
    )
    # one byte per group: read the first of each duplicated pair
    exps = u8[..., _V4_DIM_NOPE : _V4_DIM_NOPE + 2 * _V4_NUM_TILES : 2].to(torch.int32)
    scale = torch.pow(2.0, (exps - 127).float())
    out = torch.empty((*lead, _V4_DIM_QK), dtype=torch.bfloat16, device=packed.device)
    out[..., :_V4_DIM_NOPE] = (
        (nope * scale.unsqueeze(-1)).reshape(*lead, _V4_DIM_NOPE).to(torch.bfloat16)
    )
    out[..., _V4_DIM_NOPE:] = rope
    return out


@pytest.mark.parametrize("T", [1, 32, 512])
@pytest.mark.parametrize("H", [16, 128])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("kv_len", [136, 384])
@pytest.mark.parametrize("var_len", [True, False])
@pytest.mark.parametrize("q_packed", [True, False])
def test_pa_decode_sparse_fp8_vs_reference(T, H, D, kv_len, var_len, q_packed):
    """DSv4 2buff packed-fp8 KV pool, the layout the MLA-v4 asm decode kernel
    reads. ``q_packed`` toggles full a8w8 parity (packed fp8 Q + bf16 RoPE
    plane) against a8w16 (plain bf16 Q, fp8 KV)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if arch_info.get_arch() != "gfx1250":
        pytest.skip("the DSv4 2buff packed-fp8 path is gfx1250-only")

    pages = T * kv_len
    q_bf16, ukv_bf16, indices, indptr, sink, scale = _make_inputs(
        T,
        H,
        D,
        kv_len,
        pages,
        variable_len=var_len,
    )

    kv_packed, kv_rope = v4_pack_2buff(ukv_bf16)
    if q_packed:
        q_arg, q_rope = v4_pack_2buff(q_bf16)
        q_ref = v4_unpack_2buff(q_arg, q_rope)
    else:
        q_arg, q_rope = q_bf16, None
        q_ref = q_bf16

    # Reference: dequantize exactly the bytes the kernel sees, then dense torch.
    ukv_ref = v4_unpack_2buff(kv_packed, kv_rope)
    ref = pa_decode_sparse_reference(q_ref, ukv_ref, indices, indptr, sink, scale)

    out = pa_decode_sparse(
        q_arg,
        kv_packed,
        indices,
        indptr,
        sink,
        scale,
        has_invalid=False,
        unified_kv_rope=kv_rope,
        q_rope=q_rope,
    )

    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            out.to(torch.bfloat16),
            ref.to(torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
            tol_err_ratio=tol_err_ratio,
            msg="pa_decode_sparse v4 2buff output",
        )
        <= tol_err_ratio
    )


def _asm_v4_decode():
    """The MLA-v4 asm decode entry, or None where it is not available.

    It is dispatched from a prebuilt ``.co`` plus a row in
    ``hsa/gfx1250/mla_v4/mla_v4_asm.csv``, so a tree without those assets (or a
    host that is not gfx1250) simply has no asm side to compare against.
    """
    if arch_info.get_arch() != "gfx1250":
        return None
    try:
        import aiter.mla
    except ImportError:
        return None
    return getattr(aiter.mla, "mla_decode_fwd_v4_nm", None)


@pytest.mark.parametrize("T", [1, 32, 512])
@pytest.mark.parametrize("H", [128])
@pytest.mark.parametrize("kv_len", [136, 384])
def test_pa_decode_sparse_v4_2buff_vs_asm(T, H, kv_len):
    """Hand the SAME packed buffers to the gluon kernel and to the MLA-v4 asm
    decode kernel. This is the format check: if the gluon kernel read the
    NoPE / scale / pad / RoPE regions differently the two would diverge."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    mla_decode_fwd_v4_nm = _asm_v4_decode()
    if mla_decode_fwd_v4_nm is None:
        pytest.skip("MLA-v4 asm decode is not available in this tree")

    D = _V4_DIM_QK
    pages = T * kv_len
    q_bf16, ukv_bf16, indices, indptr, sink, _ = _make_inputs(
        T, H, D, kv_len, pages, variable_len=False
    )
    # the asm kernel hardcodes 1/sqrt(512) and ignores the argument
    scale = float(D) ** -0.5

    kv_packed, kv_rope = v4_pack_2buff(ukv_bf16)
    q_packed, q_rope = v4_pack_2buff(q_bf16)

    out_gluon = pa_decode_sparse(
        q_packed,
        kv_packed,
        indices,
        indptr,
        sink,
        scale,
        has_invalid=False,
        unified_kv_rope=kv_rope,
        q_rope=q_rope,
    )

    # page_size=1, one query row per sequence (decode) -> qo_indptr = arange.
    device = q_bf16.device
    qo_indptr = torch.arange(0, T + 1, dtype=torch.int32, device=device)
    out_asm = torch.empty((T, H, D), dtype=torch.bfloat16, device=device)
    try:
        mla_decode_fwd_v4_nm(
            q_packed,
            q_rope,
            kv_packed.view(-1, 1, 1, D),
            kv_rope.view(-1, 1, 1, _V4_DIM_ROPE),
            out_asm,
            qo_indptr,
            indptr,
            indices,
            1,  # max_seqlen_q
            sink=sink,
            sm_scale=scale,
        )
    except RuntimeError as e:
        # The csv only ships a .co for gqa in {16, 64, 128} at qSeqLen=1; a
        # tree carrying a different subset has nothing to compare here.
        pytest.skip(f"no MLA-v4 asm decode variant for gqa={H}: {e}")

    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            out_gluon.float(),
            out_asm.float(),
            atol=2e-2,
            rtol=2e-2,
            tol_err_ratio=tol_err_ratio,
            msg="pa_decode_sparse v4 2buff gluon vs asm",
        )
        <= tol_err_ratio
    )


@pytest.mark.parametrize("T", [1, 32])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("kv_len", [100])
@pytest.mark.parametrize("var_len", [True, False])
def test_pa_decode_sparse_fp8_uniform_vs_reference(T, H, D, kv_len, var_len):
    """Legacy 1buff uniform pool: whole-head fp8 + a separate fp32 kv_scales."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    pages = T * kv_len
    q, ukv_bf16, indices, indptr, sink, scale = _make_inputs(
        T,
        H,
        D,
        kv_len,
        pages,
        variable_len=var_len,
    )

    # Quantize KV to fp8 + scales
    kv_fp8, kv_scales = _quantize_kv_fp8(ukv_bf16)

    # Reference: dequant back to bf16, run the torch reference
    ukv_deq = _dequant_kv_fp8(kv_fp8, kv_scales).to(q.dtype)
    ref = pa_decode_sparse_reference(q, ukv_deq, indices, indptr, sink, scale)

    # Triton kernel with fp8 kv + kv_scales
    out = pa_decode_sparse(
        q,
        kv_fp8,
        indices,
        indptr,
        sink,
        scale,
        kv_scales=kv_scales,
        has_invalid=False,
    )

    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            out.to(torch.bfloat16),
            ref.to(torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
            tol_err_ratio=tol_err_ratio,
            msg="pa_decode_sparse output",
        )
        <= tol_err_ratio
    )


def make_packed_cache(num_tokens, D, dtype):
    device = "cuda"
    rope = 64  # DSv4 RoPE dim, stored bf16
    block = 256  # packed cache page size
    nope = D - rope  # NoPE dim, stored fp8 e4m3 OCP
    nb = triton.cdiv(num_tokens, block)
    if dtype == "bf16":
        cache = (torch.randn(nb, block, D, device=device) * 0.4).to(torch.bfloat16)
        return cache, cache.reshape(nb * block, D).float()

    # per token: [nope fp8 (1B) | rope bf16 (2B) | 8 UE8M0 scale bytes]
    data_bytes = nope + rope * 2
    scale_bytes = 8
    row_bytes = data_bytes + scale_bytes
    cache = torch.zeros(nb, block, row_bytes, dtype=torch.uint8, device=device)
    flat = cache.view(nb, block * row_bytes)
    data = flat[:, : block * data_bytes].view(nb, block, data_bytes)
    scales_region = flat[:, block * data_bytes :].view(nb, block, scale_bytes)
    nope_fp8 = (torch.randn(nb, block, nope, device=device) * 0.4).to(
        torch.float8_e4m3fn
    )
    data[:, :, :nope] = nope_fp8.view(torch.uint8)
    rope_bf16 = (torch.randn(nb, block, rope, device=device) * 0.4).to(torch.bfloat16)
    data[:, :, nope:data_bytes] = rope_bf16.view(torch.uint8).view(nb, block, rope * 2)
    num_groups = nope // 64
    exps = torch.randint(
        124, 130, (nb, block, num_groups), device=device, dtype=torch.uint8
    )
    scales_region[:, :, :num_groups] = exps
    scales = torch.exp2(exps.float() - 127.0).repeat_interleave(64, dim=2)
    kv_deq = torch.cat([nope_fp8.float() * scales, rope_bf16.float()], dim=2)
    return cache, kv_deq.reshape(nb * block, D)


def widen_to_int32_overflow(cache, kv_deq):
    """Re-lay ``cache`` as a strided view whose span exceeds a 32-bit offset.

    Same nelement() and same contents, but the dim-0 pitch is stretched so the
    last block sits past 2**31 bytes. Only the blocks themselves are written;
    the padding between them is left uninitialised, so the pool costs its
    address space but not the time to fill it.
    """
    nb, block, row = cache.shape
    itemsize = cache.element_size()
    pitch = triton.cdiv(2**31, max(1, nb - 1) * itemsize)
    pitch = max(pitch, block * row)
    # the packed fp8 cache is viewed as bfloat16, which needs an even stride
    pitch += pitch % 2
    pool = torch.empty(
        pitch * (nb - 1) + block * row, dtype=cache.dtype, device=cache.device
    )
    view = pool.as_strided((nb, block, row), (pitch, row, 1))
    view.copy_(cache)
    assert view.stride(0) * itemsize * (nb - 1) >= 2**31
    return view, kv_deq


def two_loop_reference(
    q,
    main_deq,
    main_idx,
    main_indptr,
    extra_deq,
    extra_idx,
    extra_indptr,
    attn_sink,
    softmax_scale,
):
    """Reference for the SWA(main) + top-k(extra) two-loop: concatenate the two
    dequantized pools, merge the two ragged index sets (extra slots shifted past
    the main pool), then reuse ``pa_decode_sparse_reference``.
    """
    main_pages = main_deq.shape[0]
    combined = torch.cat([main_deq, extra_deq], dim=0).to(q.dtype)
    T = main_indptr.numel() - 1
    mi, mp = main_idx.long(), main_indptr.long()
    ei, ep = extra_idx.long(), extra_indptr.long()
    rows, lens = [], []
    for tok in range(T):
        row = torch.cat(
            [mi[mp[tok] : mp[tok + 1]], ei[ep[tok] : ep[tok + 1]] + main_pages]
        )
        rows.append(row)
        lens.append(row.numel())
    combined_idx = torch.cat(rows).to(torch.int32)
    combined_indptr = torch.zeros(T + 1, dtype=torch.int32, device=q.device)
    combined_indptr[1:] = torch.tensor(lens, device=q.device).cumsum(0)
    return pa_decode_sparse_reference(
        q, combined, combined_idx, combined_indptr, attn_sink, softmax_scale
    )


@pytest.mark.parametrize("T", [1, 32, 128])
@pytest.mark.parametrize("H", [16])
@pytest.mark.parametrize("D", [512])
@pytest.mark.parametrize("main_len", [128])
@pytest.mark.parametrize("extra_len", [8, 256])
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("strided_cache", [False, True])
def test_pa_decode_sparse_two_loop(T, H, D, main_len, extra_len, dtype, strided_cache):
    """gfx950 vLLM DSv4 decode path: SWA (main) + top-k (extra) two-loop over
    packed caches. fp8 (fp8_ds_mla) is the vLLM production format; bf16 is also
    exercised. Skipped off gfx950 (extra_* is a packed-only gluon path)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if arch_info.get_arch() != "gfx950":
        pytest.skip("two-loop (extra_*) is a gfx950 packed-cache-only path")
    if strided_cache:
        # The pool has to span >2 GiB for the offsets to overflow, so pin the
        # regression to one shape -- the fp8 production format at the largest T
        # -- rather than paying it on all 24 combinations.
        if dtype != "fp8" or T != 128:
            pytest.skip("strided-cache case is pinned to the fp8 T=128 shape")
        if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
            pytest.skip("needs ~3 GiB free for the >2 GiB strided pool")

    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device) * 0.125
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1
    softmax_scale = float(D) ** -0.5

    # main = contiguous SWA window per query
    main_cache, main_deq = make_packed_cache(T * main_len, D, dtype)
    query_base = (torch.arange(T, device=device) * main_len)[:, None]
    main_idx = (
        (query_base + torch.arange(main_len, device=device)).to(torch.int32).reshape(-1)
    )
    main_indptr = torch.arange(
        0, T * main_len + 1, main_len, dtype=torch.int32, device=device
    )
    # extra = scattered top-k over a pool
    extra_pool = T * extra_len
    extra_cache, extra_deq = make_packed_cache(extra_pool, D, dtype)
    if strided_cache:
        extra_cache, extra_deq = widen_to_int32_overflow(extra_cache, extra_deq)
    extra_idx = torch.randint(
        0, extra_pool, (T, extra_len), device=device, dtype=torch.int32
    ).reshape(-1)
    extra_indptr = torch.arange(
        0, T * extra_len + 1, extra_len, dtype=torch.int32, device=device
    )

    ref = two_loop_reference(
        q,
        main_deq,
        main_idx,
        main_indptr,
        extra_deq,
        extra_idx,
        extra_indptr,
        attn_sink,
        softmax_scale,
    )
    out = pa_decode_sparse(
        q,
        main_cache,
        main_idx,
        main_indptr,
        attn_sink,
        softmax_scale,
        extra_cache=extra_cache,
        extra_indices=extra_idx,
        extra_indptr=extra_indptr,
    )

    tol = 1e-2 if dtype == "fp8" else 5e-3
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# EXPERIMENTAL uniform-MX layout: the whole head (RoPE included) quantized to
# e4m3 in 64-element groups, with the E8M0 scales in their own plane. Not the
# asm kernel's format -- this exists to measure what the packing would cost if
# it were designed for the MX WMMA path from the start.
# ---------------------------------------------------------------------------


def v4_pack_mx(x_bf16):
    """``[..., 512]`` bf16 -> ``(e4m3 [..., 512], E8M0 scales [..., 16] uint8)``.

    8 quant groups of 64 over the full head. Each group's scale byte is stored
    twice so the 16 bytes are the MX scale operand as-is (MX blocks are 32).
    """
    lead = x_bf16.shape[:-1]
    d = x_bf16.shape[-1]
    ngroups = d // _V4_TILE
    tiled = x_bf16.float().reshape(*lead, ngroups, _V4_TILE)
    fp8_max = float(torch.finfo(_V4_FP8).max)
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    vals = (tiled / scale.unsqueeze(-1)).to(_V4_FP8).reshape(*lead, d)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    return vals, e8m0.repeat_interleave(2, dim=-1)


def v4_unpack_mx(vals, scales16):
    """Inverse of ``v4_pack_mx`` -> ``[..., 512]`` bf16."""
    lead = vals.shape[:-1]
    d = vals.shape[-1]
    ngroups = d // _V4_TILE
    exps = scales16[..., ::2].to(torch.int32)
    scale = torch.pow(2.0, (exps - 127).float())
    deq = vals.float().reshape(*lead, ngroups, _V4_TILE) * scale.unsqueeze(-1)
    return deq.reshape(*lead, d).to(torch.bfloat16)


@pytest.mark.parametrize("T", [1, 32, 512])
@pytest.mark.parametrize("H", [16, 128])
@pytest.mark.parametrize("kv_len", [136, 384])
@pytest.mark.parametrize("var_len", [True, False])
def test_pa_decode_sparse_mx_vs_reference(T, H, kv_len, var_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if arch_info.get_arch() != "gfx1250":
        pytest.skip("the uniform-MX path is gfx1250-only")

    D = _V4_DIM_QK
    pages = T * kv_len
    q_bf16, ukv_bf16, indices, indptr, sink, scale = _make_inputs(
        T, H, D, kv_len, pages, variable_len=var_len
    )
    q_mx, q_s = v4_pack_mx(q_bf16)
    kv_mx, kv_s = v4_pack_mx(ukv_bf16)

    # Reference dequantizes exactly the bytes the kernel reads.
    ref = pa_decode_sparse_reference(
        v4_unpack_mx(q_mx, q_s),
        v4_unpack_mx(kv_mx, kv_s),
        indices,
        indptr,
        sink,
        scale,
    )
    out = pa_decode_sparse(
        q_mx,
        kv_mx,
        indices,
        indptr,
        sink,
        scale,
        has_invalid=False,
        kv_mx_scales=kv_s,
        q_mx_scales=q_s,
    )

    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            out.to(torch.bfloat16),
            ref.to(torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
            tol_err_ratio=tol_err_ratio,
            msg="pa_decode_sparse uniform-MX output",
        )
        <= tol_err_ratio
    )
