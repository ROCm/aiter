# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from aiter.ops.triton._triton_kernels.attention.block_lut import (
    block_attn_mask_to_lut_kernel,
)
from aiter.ops.triton._triton_kernels.attention.sol_attn_pool import (
    POOL_AUX_NONE,
    POOL_AUX_SQUARE_MEAN,
    POOL_AUX_VARIANCE,
    sol_attn_pool,
)

# Default tile geometry of the gfx950 Sol-Attn ASM kernels. SOL_ATTN_TS_KV in particular is the
# pooling block size, which each row fixes rather than reading, so pooling with
# any other value is silently wrong rather than an error -- which is why BLOCK_M/BLOCK_N below must
# name the geometry of the row that will actually be dispatched, not a preference. gfx950 also ships
# a 64x64 FP8 row; mha_v4_sol(block_tile=...) drives routing and dispatch from one value so the
# two cannot drift apart.
SOL_ATTN_TS_QO = 256
SOL_ATTN_TS_KV = 128

# E8M0 block scales store a raw exponent biased by 127, so a stored byte s means a multiplier of
# 2**(s - 127). Code 255 is the format's NaN, so usable codes stop at 254.
_E8M0_BIAS = 127
_E8M0_MAX_CODE = 254
_E8M0_GROUP = 32

# Packed formats whose stored codes are not element addressable, so a pooled operand has to be
# built from the tensor the packer was given rather than from the codes it produced. The name says
# which packer the pooled operand goes through, so it has to be the one the row reads: a V named
# *_fp6_p is packed in the token order an FP6 P operand contracts over (AttentionPack.V_FOR_FP6_P),
# which is the only packed V order a Sol-Attn row reads.
SOL_ATTN_PACKED_K_FORMATS = ("mxfp4", "mxfp6")
SOL_ATTN_PACKED_V_FORMATS = ("mxfp4_fp6_p", "mxfp6_fp6_p")
SOL_ATTN_PACKED_FORMATS = tuple(
    dict.fromkeys(SOL_ATTN_PACKED_K_FORMATS + SOL_ATTN_PACKED_V_FORMATS)
)


def _e8m0_dequantize(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply an E8M0 1x32 scale image to its data, in fp32.

    The scale carries one exponent per 32 channels of the last axis, so it is expanded back over
    that axis before multiplying.
    """
    if data.shape[-1] != scale.shape[-1] * _E8M0_GROUP:
        raise ValueError(
            f"E8M0 scale covers {scale.shape[-1]} groups of {_E8M0_GROUP} but data has "
            f"{data.shape[-1]} channels"
        )
    factor = torch.exp2(scale.float() - _E8M0_BIAS)
    return data.float() * factor.repeat_interleave(_E8M0_GROUP, dim=-1)


def _e8m0_quantize(
    x: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an fp32 tensor to (data, E8M0 1x32 scale), one shared exponent per 32 channels.

    The exponent is the smallest one that brings the group's largest magnitude inside ``dtype``'s
    range, which is what makes this the inverse of :func:`_e8m0_dequantize` up to the element
    rounding. It is derived from a log2 rather than frexp so the whole thing stays a handful of
    elementwise ops and therefore traceable; the data is clamped afterwards because a log2 that
    rounds the exponent one low for a group whose max sits exactly on a representable boundary
    would otherwise saturate on the cast.
    """
    lead, channels = x.shape[:-1], x.shape[-1]
    if channels % _E8M0_GROUP:
        raise ValueError(
            f"E8M0 quantization needs a multiple of {_E8M0_GROUP} channels"
        )
    groups = channels // _E8M0_GROUP
    limit = torch.finfo(dtype).max

    grouped = x.reshape(*lead, groups, _E8M0_GROUP)
    # Floored at the smallest positive fp32 rather than left at 0: an all-zero group would take
    # log2(0) = -inf, and while the clamp below would still land it on code 0, the intermediate
    # -inf is not worth carrying through the graph.
    amax = (
        grouped.abs()
        .amax(dim=-1, keepdim=True)
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    code = (
        (torch.ceil(torch.log2(amax) - math.log2(limit)) + _E8M0_BIAS)
        .clamp(0, _E8M0_MAX_CODE)
        .to(torch.int32)
    )
    factor = torch.exp2(code.float() - _E8M0_BIAS)
    data = (grouped / factor).clamp(-limit, limit).to(dtype)
    return (
        data.reshape(*lead, channels).contiguous(),
        code.squeeze(-1).to(torch.uint8).contiguous(),
    )


def block_attn_mask_to_ragged_lut(
    block_attn_mask: torch.Tensor,
    num_heads: int | None = None,
    return_none_if_dense: bool = False,
    BLOCK_KB: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Convert a dense block attention mask to a ragged look-up table of KV block
    indices per (batch, head, q_block). Used for block-sparse attention with no
    per-iteration branching in the kernel.

    block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for
        same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks)
        for per-head masks. True = may attend, False = must not attend.
    num_heads: Required when block_attn_mask is 3D (number of Q heads). Ignored when 4D.
    return_none_if_dense: If True and the mask is all True (dense), return None so the
        caller can pass block_lut=None to fav3_sage_wrapper_func and use the dense path.
        Avoids building a very large LUT that can trigger munmap_chunk on MI300X/ROCm.
    Returns:
        kv_block_indices: 1D int32, concatenation of all KV block index lists.
        lut_start: 1D int32, length batch * num_heads * num_q_blocks. Index
            idx = batch_idx * (num_heads * num_q_blocks) + head_idx * num_q_blocks + q_block_idx.
        lut_count: 1D int32, same length as lut_start.
        When return_none_if_dense is True and the mask is all True, returns None instead.
    """
    device = block_attn_mask.device

    # 3D -> 4D: expand and fall through to 4D path
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        batch, num_q_blocks, num_kv_blocks = block_attn_mask.shape
        if return_none_if_dense and block_attn_mask.all():
            return None
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(
            batch, num_heads, num_q_blocks, num_kv_blocks
        )

    # 4D: (batch, num_heads, num_q_blocks, num_kv_blocks) — GPU vectorized path
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    if return_none_if_dense and block_attn_mask.all():
        return None

    counts = block_attn_mask.sum(dim=-1, dtype=torch.int32)
    lut_count = counts.reshape(-1)
    lut_start = torch.cumsum(lut_count, dim=0, dtype=torch.int32) - lut_count

    # NOTE: Overallocating the LUT is a waste of memory, but the
    # alternative lut_count.sum(), will cause graph break with torch compile.
    # TODO(sparse refresh): entries past the used range stay uninitialized, and an empty row's
    # lut_start points at them. Kernels that speculatively read LUT[lut_start] before testing
    # lut_count must clamp the derived offsets, as the FP8 prologue does, or this must be zeroed.
    max_count = batch * num_heads * num_q_blocks * num_kv_blocks
    kv_block_indices = torch.empty(max_count, dtype=torch.int32, device=device)
    block_attn_mask_to_lut_kernel(
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        BLOCK_KB=BLOCK_KB,
    )

    return kv_block_indices, lut_start, lut_count


def _sol_attn_pool_kv_quant(
    k_quant: torch.Tensor, v_quant: torch.Tensor, BLOCK_N: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool stored (already quantized) K/V into one row per KV block, in the SOURCE dtype.

    The block mean is accumulated in fp32 over the raw stored values and rounded back to the
    source dtype, so K's and V's PER-TENSOR descales stay valid for the pooled tensors
    (mean(x) * descale == mean(x * descale)). The row applies those descales itself, so a separate
    scale on either mean tensor would corrupt the exact/approximate mix. Rounding a mean of
    in-range values back to fp8 cannot overflow.

    This identity is what ties pooling to per-tensor scaling: a block-granular scale image
    (E8M0 per 1x32) spans several scale blocks per pooled row and has no single descale to
    inherit, so a non-per-tensor variant needs its own pooled-operand scale contract rather than
    this function.

    Returns (mean_k, mean_v), BSHD (batch, num_kv_blocks, nheads_kv, d) and contiguous. The
    allocation stops at num_kv_blocks: the approximate pass processes whole 128-block groups, but
    it masks the overhang columns before it loads their pooled rows, so it never reads past the end.
    """
    return (
        _sol_attn_pool_reuse_descale(k_quant, BLOCK_N),
        _sol_attn_pool_reuse_descale(v_quant, BLOCK_N),
    )


# An 8-bit float K's block variance can reach its format maximum squared (448^2 for e4m3), far past
# what the format holds, so it is stored divided by this. |k| <= max bounds the variance by max^2,
# and max^2 / 512 stays below max for every FP8 format the kernels take.
SOL_ATTN_FP8_VAR_DIVISOR = 512.0


def _sol_attn_pool_reuse_descale(
    x_quant: torch.Tensor, BLOCK_N: int, variance: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pool one operand's stored codes, in its own dtype, keeping the source descale valid.

    Only correct for a descale that does not vary along the sequence axis; see
    :func:`_sol_attn_pool_kv_quant` for why, and :func:`_sol_attn_pool_mx` for the case where it
    does. Rounding a mean of in-range values back to the source dtype cannot overflow.

    `variance` also returns the per-block population variance from the same fp32 copy, in the
    same layout, divided by SOL_ATTN_FP8_VAR_DIVISOR for an 8-bit operand; the mean is unaffected by
    asking for it. It keeps the operand's dtype, except that an integer one's is stored as e4m3,
    since an integer cannot hold it.
    """
    var_dtype = (
        x_quant.dtype if x_quant.dtype.is_floating_point else torch.float8_e4m3fn
    )
    var_multiplier = (
        1.0 / SOL_ATTN_FP8_VAR_DIVISOR if x_quant.element_size() == 1 else 1.0
    )
    if _sol_attn_pool_kernel_ok(x_quant):
        mean, var = sol_attn_pool(
            x_quant,
            BLOCK_N,
            x_quant.dtype,
            aux=POOL_AUX_VARIANCE if variance else POOL_AUX_NONE,
            aux_dtype=var_dtype,
            aux_multiplier=var_multiplier,
        )
        return (mean, var) if variance else mean
    xf = x_quant.float()
    mean = _sol_attn_block_mean(xf, BLOCK_N)
    if not x_quant.dtype.is_floating_point:
        # Casting to an integer dtype TRUNCATES toward zero, which would pull every pooled row
        # toward zero and shrink the approximate pass's scores. The float8 dtypes round on cast, so
        # only the integer operands (i8fp8's K) need this made explicit.
        mean = mean.round()
    mean = mean.to(x_quant.dtype).contiguous()
    if not variance:
        return mean
    var = _sol_attn_block_variance(xf, BLOCK_N) * var_multiplier
    return mean, var.to(var_dtype).contiguous()


def _sol_attn_pool_kernel_ok(x: torch.Tensor) -> bool:
    """Whether :func:`sol_attn_pool` covers this pooling; the torch path takes the rest."""
    head_dim = x.shape[-1]
    return x.is_cuda and head_dim > 0 and head_dim & (head_dim - 1) == 0


def _sol_attn_block_variance(xf: torch.Tensor, BLOCK_N: int) -> torch.Tensor:
    """Population variance of an fp32 BSHD tensor over each BLOCK_N-token KV block.

    A short last block is taken over its real tokens, as in :func:`_sol_attn_block_mean`.
    """
    batch, seqlen_k, nhead_kv, channels = xf.shape
    full = seqlen_k // BLOCK_N
    var = (
        xf[:, : full * BLOCK_N]
        .reshape(batch, full, BLOCK_N, nhead_kv, channels)
        .var(dim=2, unbiased=False)
    )
    if seqlen_k % BLOCK_N:
        tail = xf[:, full * BLOCK_N :].var(dim=1, unbiased=False, keepdim=True)
        var = torch.cat([var, tail], dim=1)
    return var


def _sol_attn_block_mean(xf: torch.Tensor, BLOCK_N: int) -> torch.Tensor:
    """Mean of an fp32 BSHD tensor over each BLOCK_N-token KV block.

    A short last block is divided by its real token count, not by BLOCK_N. Note that the KERNEL
    applies a constant BLOCK_N factor instead, which is why routing forces a partial tail block to
    stay exact rather than letting it reach the approximate pass.
    """
    batch, seqlen_k, nhead_kv, channels = xf.shape
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    pad = num_kv_blocks * BLOCK_N - seqlen_k
    counts = torch.full(
        (num_kv_blocks,), BLOCK_N, dtype=torch.float32, device=xf.device
    )
    if pad:
        counts[-1] = BLOCK_N - pad
        xf = F.pad(xf, (0, 0, 0, 0, 0, pad))
    xf = xf.reshape(batch, num_kv_blocks, BLOCK_N, nhead_kv, channels)
    return xf.sum(dim=2) / counts.view(1, num_kv_blocks, 1, 1)


def _sol_attn_pool_mx(
    x_quant: torch.Tensor, x_scale: torch.Tensor, BLOCK_N: int, variance: bool = False
) -> tuple[torch.Tensor, ...]:
    """Pool an E8M0-scaled operand into one row per KV block, WITH a pooled scale of its own.

    Per-tensor and per-channel descales survive pooling untouched, because pooling runs along the
    sequence axis and neither of those varies along it: mean(x) * descale == mean(x * descale), so
    :func:`_sol_attn_pool_kv_quant` can pool the stored codes and reuse the source descale. An E8M0
    1x32 scale does vary per token, so that identity fails and the codes are not poolable at all --
    a pooled row spans BLOCK_N different scale blocks per channel group. This function therefore
    pools in DEQUANTIZED space and requantizes, which is why the approximate pass needs the pooled
    scale slots in the mode-2 kernarg.

    Returns (mean_data, mean_scale, mean_dequantized). The third is what routing has to score on:
    for a per-tensor operand the stored codes are proportional to the real values so routing on
    them is scale invariant, but here the scale varies from row to row and from group to group, so
    scoring the raw codes would compare values that are not on a common footing.

    `variance` appends (var_data, var_scale) for the per-block population variance of the
    dequantized operand, quantized the same way and in the same layouts as the mean. A variance has
    no single unit here any more than the mean does, so it gets a scale image of its own.
    """
    dequantized = _e8m0_dequantize(x_quant, x_scale)
    pooled = _sol_attn_block_mean(dequantized, BLOCK_N)
    mean_data, mean_scale = _e8m0_quantize(pooled, x_quant.dtype)
    # Reconstructed from the requantized pair rather than returning `pooled`: the kernel loads the
    # rounded values, and routing has to see exactly what it will load or a block sitting within
    # rounding distance of the threshold can be selected here and skipped there.
    result = (mean_data, mean_scale, _e8m0_dequantize(mean_data, mean_scale))
    if not variance:
        return result
    var = _sol_attn_block_variance(dequantized, BLOCK_N)
    return result + _e8m0_quantize(var, x_quant.dtype)


def _e2m3_decode(code: torch.Tensor) -> torch.Tensor:
    """MXFP6 E2M3 codes (bit 5 is the sign) -> fp32 magnitudes.

    Exponent 0 is the subnormal rung m/8; the three normal rungs are 2**(e-1) * (1 + m/8), so the
    format tops out at 7.5. Written as the inverse of ``_e2m3_encode_torch`` in the fp6 packer,
    which aiter has but never had a decoder for.
    """
    bits = code.to(torch.int32)
    exponent = (bits >> 3) & 3
    mantissa = (bits & 7).float()
    magnitude = torch.where(
        exponent == 0,
        mantissa / 8.0,
        torch.exp2((exponent - 1).float()) * (1.0 + mantissa / 8.0),
    )
    return torch.where((bits & 32) != 0, -magnitude, magnitude)


def _fp6_quant_dequantize(x: torch.Tensor) -> torch.Tensor:
    """The values an MXFP6 operand actually presents to the kernel, in fp32.

    Two roundings, applied in the packer's order: an E8M0 exponent per 32 channels chosen as
    exp2(amax) - 129, then E2M3 on each element scaled by it. The packer's field permutation is
    deliberately skipped -- it only decides which byte a code lands in, and the encode is
    elementwise, so it cannot move a value.
    """
    from aiter.ops.triton.quant.mxfp6_fmha_pack import _e2m3_encode_torch

    lead, channels = list(x.shape[:-1]), x.shape[-1]
    if channels % _E8M0_GROUP:
        raise ValueError(
            f"MXFP6 needs a multiple of {_E8M0_GROUP} channels, got {channels}"
        )
    grouped = x.float().reshape(*lead, channels // _E8M0_GROUP, _E8M0_GROUP)
    amax = grouped.abs().amax(dim=-1)
    exponent = (amax.contiguous().view(torch.int32) >> 23) & 0xFF
    # -129 rather than -127: the E2M3 grid runs to 7.5, so the group exponent is picked to land
    # amax on the top rung instead of on 1.0 the way an fp8 E8M0 group would.
    biased = torch.where(amax == 0, torch.zeros_like(exponent), exponent - 129)
    scale = torch.exp2(biased.float()).unsqueeze(-1)
    codes = _e2m3_encode_torch(grouped / scale)
    return (_e2m3_decode(codes) * scale).reshape(*lead, channels)


def _sol_attn_pool_fp6_v(
    v: torch.Tensor, BLOCK_N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool an MXFP6 V into one row per KV block, WITH a pooled E8M0 scale of its own.

    Same dequant-pool-requant as :func:`_sol_attn_pool_mx` and for the same reason -- an E8M0 1x32
    scale varies along the very axis pooling reduces -- but MXFP6 codes are not element addressable
    (six bits, tile-packed), so this takes the unquantized V and rounds it itself rather than
    reading stored codes back.

    Requantizing through the ordinary V packer is what keeps the kernel side cheap: the packer
    tiles at 128 rows for any sequence length, so a pooled tensor of num_kv_blocks rows comes back
    in exactly the layout V already has (96-byte rows, 12288-byte tiles, 512 scale bytes per tile).
    The approximate pass can then read the pooled tensor with the recipe's own V reader.

    Returns (mean_data, mean_scale, mean_dequantized); the third is what routing scores on.
    """
    from aiter.ops.triton.quant.mxfp6_fmha_pack import pack_fp6_v_data_scale_views

    pooled = _sol_attn_block_mean(_fp6_quant_dequantize(v), BLOCK_N)
    mean_data, mean_scale = pack_fp6_v_data_scale_views(pooled)
    return mean_data, mean_scale, _fp6_quant_dequantize(pooled)


def _sol_attn_pad_to_tile(x: torch.Tensor, BLOCK_N: int) -> torch.Tensor:
    """Zero-pad a pooled BSHD tensor's row axis up to a whole packing tile.

    The MXFP4 readers walk whole tiles and rely on the bitmap's set tail bits to mask the overhang,
    which is enough for the DATA but not for the SCALE: a full tile of scale bytes may be read
    whatever the real pooled height is. An uninitialized E8M0 byte of 0xFF is 2**128, which reaches
    QK as inf before any mask is applied, so the rows have to exist and be zero before the packer
    sees them.
    """
    rows = x.shape[1]
    pad = -rows % BLOCK_N
    if not pad:
        return x
    return torch.cat([x, x.new_zeros(x.shape[0], pad, *x.shape[2:])], dim=1)


def _sol_attn_pool_mxfp4_k(
    k_source: torch.Tensor, BLOCK_N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool an MXFP4 K into one row per KV block, WITH a pooled E8M0 scale of its own.

    Takes the pre-quantization K rather than the stored codes, for the same reason
    :func:`_sol_attn_pool_fp6_v` does: MXFP4 codes are not element addressable. They are four bits
    packed two per byte and then permuted, so reading them back means inverting the packer's
    permutation, and K's logical view is not the layout the codes sit in.

    Pooling the source and quantizing once also happens to be the more accurate of the two orders
    (relative L2 against the ideal pooled mean 0.12 versus 0.17 for rounding before pooling), though
    the difference does not survive to the output: both land within 0.003 cosine end to end.

    Returns (mean_data, mean_scale, mean_pooled). The third is the unrounded pooled K, which is what
    routing scores on and what a reference should pool over. Routing on it rather than on the
    requantized values is sound here where it would not be for :func:`_sol_attn_pool_mx`: the packer
    applies an orthogonal Hadamard rotation to both Q and K, so (QR)(KR)^T == QK^T leaves the proxy
    unchanged, and the requantized values cannot be read back to score anyway.
    """
    from aiter.ops.mha_v4_quant import mxfp4_k_view, quantize_mxfp4_k

    pooled = _sol_attn_block_mean(k_source.float(), BLOCK_N).to(torch.bfloat16)
    blocks = pooled.shape[1]
    raw, scale = quantize_mxfp4_k(_sol_attn_pad_to_tile(pooled, BLOCK_N))
    # The view is built from the unpadded scale so it presents the real pooled height, while the
    # scale handed to the kernarg keeps its padded rows, which is what the scale read needs.
    return mxfp4_k_view(raw, scale[:, :blocks]), scale, pooled


def _sol_attn_pack_k(
    x: torch.Tensor, fmt: str, BLOCK_N: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a pooled-height BSHD tensor through `fmt`'s K packer, zero-padded to a whole tile.

    Returns (view, scale): the view presents the real pooled height, the scale keeps the padded
    one, for the reason :func:`_sol_attn_pool_mxfp4_k` gives.
    """
    from aiter.ops.mha_v4_quant import (
        mxfp4_k_view,
        mxfp6_k_view,
        quantize_mxfp4_k,
        quantize_mxfp6_k,
    )

    blocks = x.shape[1]
    padded = _sol_attn_pad_to_tile(x, BLOCK_N)
    if fmt == "mxfp4":
        raw, scale = quantize_mxfp4_k(padded)
        return mxfp4_k_view(raw, scale[:, :blocks]), scale
    raw, scale_raw = quantize_mxfp6_k(padded)
    view, scale = mxfp6_k_view(
        raw, scale_raw, padded.shape[0], padded.shape[1], padded.shape[2]
    )
    return view[:, :blocks], scale


def _sol_attn_pool_packed_k(
    k_source: torch.Tensor,
    fmt: str,
    BLOCK_N: int,
    variance: bool = False,
    shift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Pool a packed (MXFP4 or MXFP6) K into one row per KV block, WITH a pooled E8M0 scale.

    Source-side pooling as in :func:`_sol_attn_pool_mxfp4_k`. Returns (mean_data, mean_scale,
    mean_pooled), and with `variance` also (var_data, var_scale) in mean_k's layouts.

    The variance is the one the kernel's Jensen term needs, which is of K as the kernel holds it:
    in the basis the packer's Hadamard rotation moves it to, where q_d^2 * var[d] is summed. A
    variance is not linear, so it cannot be taken unrotated and left to the packer the way the mean
    can. It is taken in the rotated basis and rotated back before packing, so the packer's own
    rotation lands it where the kernel reads it; the normalized Hadamard is its own inverse.

    `shift` is the vector the exact pass's packer subtracted from every token. A block mean is
    linear, so the pooled K of the shifted source is the pooled K minus `shift`, taken here on the
    pooled rows rather than on a shifted copy of the whole source. A variance does not see it.
    """
    pooled = _sol_attn_block_mean(k_source.float(), BLOCK_N)
    if shift is not None:
        pooled = pooled - shift.float().unsqueeze(1)
    pooled = pooled.to(torch.bfloat16)
    result = (*_sol_attn_pack_k(pooled, fmt, BLOCK_N), pooled)
    if not variance:
        return result
    from aiter.ops.mha_v4_quant import rotate_activation_hd128

    def rotate(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        rotate_activation_hd128(out, x)
        return out

    rotated_var = _sol_attn_block_variance(
        rotate(k_source.contiguous()).float(), BLOCK_N
    )
    return result + _sol_attn_pack_k(
        rotate(rotated_var.to(torch.bfloat16).contiguous()), fmt, BLOCK_N
    )


def _sol_attn_pool_fp6_p_v(
    v_source: torch.Tensor, fmt: str, BLOCK_N: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool an MXFP4 or MXFP6 V for the FP6-P rows into one row per KV block, with its own scale.

    Source-side pooling as in :func:`_sol_attn_pool_mxfp4_k`, through the FP6-P V packers, whose
    scale already carries the slack the one-tile-ahead scale gather needs.
    """
    from aiter.ops.mha_v4_quant import (
        mxfp4_v_view,
        quantize_v_mxfp4_fp6_p,
        quantize_v_mxfp6_fp6_p,
    )

    pooled = _sol_attn_block_mean(v_source.float(), BLOCK_N).to(torch.bfloat16)
    blocks = pooled.shape[1]
    padded = _sol_attn_pad_to_tile(pooled, BLOCK_N)
    if fmt == "mxfp4_fp6_p":
        raw, scale = quantize_v_mxfp4_fp6_p(padded)
        return mxfp4_v_view(raw, scale, blocks), scale, pooled
    view, scale = quantize_v_mxfp6_fp6_p(padded)
    return view[:, :blocks], scale, pooled


def _sol_attn_pool_q(
    q: torch.Tensor, BLOCK_M: int, second_moment: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Pool Q into one representative row per query tile (the paper's Q-bar), fp32.

    q: (batch, seqlen_q, nheads_q, d) -> (batch, num_q_tiles, nheads_q, d)

    `second_moment` also returns the per-tile mean of q^2, same shape; the mean is unaffected.
    """
    if _sol_attn_pool_kernel_ok(q):
        mean, sq_mean = sol_attn_pool(
            q,
            BLOCK_M,
            torch.float32,
            aux=POOL_AUX_SQUARE_MEAN if second_moment else POOL_AUX_NONE,
            aux_dtype=torch.float32,
        )
        return (mean, sq_mean) if second_moment else mean
    batch, seqlen_q, nhead_q, d = q.shape
    num_q_tiles = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    pad = num_q_tiles * BLOCK_M - seqlen_q
    counts = torch.full((num_q_tiles,), BLOCK_M, dtype=torch.float32, device=q.device)
    if pad:
        counts[-1] = BLOCK_M - pad
    qf = q.float()
    if pad:
        qf = F.pad(qf, (0, 0, 0, 0, 0, pad))
    qf = qf.reshape(batch, num_q_tiles, BLOCK_M, nhead_q, d)
    counts = counts.view(1, num_q_tiles, 1, 1)
    mean = qf.sum(dim=2) / counts
    if not second_moment:
        return mean
    # One fused reduction; (qf * qf).sum would write a second full-size fp32 copy of Q first.
    return mean, torch.linalg.vector_norm(qf, dim=2).square() / counts


def _log_expm1(x: torch.Tensor) -> torch.Tensor:
    """log(exp(x) - 1) for x >= 0, finite everywhere: x itself once expm1 would overflow."""
    y = x.clamp(1e-30, 20.0)
    return torch.log(torch.expm1(y)) + (x - y)


def _sol_attn_route(
    q_mean: torch.Tensor,
    k_mean: torch.Tensor,
    beta: float,
    partial_tail: bool,
    q_sq_mean: torch.Tensor | None = None,
    k_var: torch.Tensor | None = None,
    logit_scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Query-dependent threshold routing, paper Eq. (3)-(5) and (7).

        proxy[b, h, i, j] = q_mean[b, i, h, :] . k_mean[b, j, h, :]
        tau[b, h, i]      = mean_j(proxy) + beta * population_std_j(proxy)
        selected          = proxy > tau

    With q_sq_mean, k_var and logit_scale (the "error" router) the proxy is instead the log of
    the tile-to-block mass, to second order, times the relative variance of the weights inside
    the block, i.e. how much the pooled estimate stands to get wrong by skipping it:

        c        = logit_scale, so c * q . k is the softmax logit in nats
        var_q    = q_sq_mean - q_mean^2
        sig2     = c^2 * q_sq_mean . k_var          row-averaged in-block logit variance
        mass     = c * q_mean . k_mean + 0.5 * sig2 + 0.5 * c^2 * var_q . k_mean^2
        proxy    = mass + log(expm1(sig2))

    A block whose keys barely spread scores low whatever its mass, since pooling it is nearly
    exact once the kernel takes the second-order term. That proxy is not scale invariant, which is
    why it needs c.

    The std is the population std, which is what the Eq. (5) closed form over the pooled-key first
    and second moments computes. The first proxy's selection is invariant to a positive rescale of
    the logits (mu and sigma scale with proxy), so the softmax scale and the per-tensor descales are
    deliberately NOT applied to it: host and kernel cannot disagree about routing because of a
    scale factor.

    That invariance is exact in real arithmetic but not in fp32: routing on quantized values here
    and on dequantized values in a reference rounds the last bit differently, so a block whose proxy
    sits within ~1e-7 * sigma of tau can land on either side (measured: 1 block in 4096 for seqlen
    4096, GQA 4). Both answers are equally valid, but it means the mask is the HOST's to own:
    compare a kernel run against a reference evaluated on the mask this function returned, never
    against a mask the reference rerouted itself.

    partial_tail: whether the last KV block holds fewer than BLOCK_N real tokens. This is a
    property of the shapes (seqlen_k % BLOCK_N != 0), not of the values, and is passed in rather
    than recovered from a token-count tensor so that routing stays free of host-side reads of
    device data and therefore traceable under torch.compile.

    Returns (batch, nheads_q, num_q_tiles, num_kv_blocks) bool, True == compute exactly.
    """
    g = q_mean.shape[2] // k_mean.shape[2]
    k_rep = k_mean.float().repeat_interleave(g, dim=2)
    if k_var is None:
        proxy = torch.einsum("bihd,bjhd->bhij", q_mean.float(), k_rep)
    else:
        c = logit_scale
        if isinstance(c, torch.Tensor):
            c = c.float().reshape(())
        q_m = q_mean.float()
        q_sq = q_sq_mean.float()
        q_var = (q_sq - q_m.square()).clamp_min(0.0)
        kv_rep = k_var.float().repeat_interleave(g, dim=2)
        # The three mass terms as one product over concatenated features; the operands are one
        # row per tile or block, so the concatenation is cheap next to a (tiles x blocks) op.
        half_c2 = 0.5 * c * c
        lhs = torch.cat([c * q_m, half_c2 * q_sq, half_c2 * q_var], dim=-1)
        rhs = torch.cat([k_rep, kv_rep, k_rep.square()], dim=-1)
        mass = torch.einsum("bihd,bjhd->bhij", lhs, rhs)
        sig2 = torch.einsum("bihd,bjhd->bhij", (c * c) * q_sq, kv_rep)
        proxy = mass + _log_expm1(sig2)
    mu = proxy.mean(dim=-1, keepdim=True)
    sigma = proxy.std(dim=-1, unbiased=False, keepdim=True)
    selected = proxy > (mu + beta * sigma)
    # A partial tail block reaching the approximate branch would be scaled by the kernel's constant
    # block-size factor, which is only exact for a full block.
    if partial_tail:
        selected[..., -1] = True
    # Keep the highest-proxy block for a row that a nearly flat proxy would otherwise clear
    # entirely. This is an ACCURACY choice, not a requirement: the kernel handles an empty row and
    # lands it on the pooled-only softmax rather than a zero tile. Keeping one exact block just gives
    # that row a real softmax max to work from. Applied unconditionally because the term is empty for
    # rows that already selected something, so guarding it on empty.any() would trade a cheap
    # elementwise op for a device sync and a graph break.
    empty = ~selected.any(dim=-1, keepdim=True)
    return selected | (empty & F.one_hot(proxy.argmax(dim=-1), proxy.shape[-1]).bool())


def sol_prepare(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    v_quant: torch.Tensor,
    beta: float | None = None,
    BLOCK_M: int = SOL_ATTN_TS_QO,
    BLOCK_N: int = SOL_ATTN_TS_KV,
    num_heads: int | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    k_source: torch.Tensor | None = None,
    v_source: torch.Tensor | None = None,
    k_packed_format: str | None = None,
    v_packed_format: str | None = None,
    block_attn_mask: torch.Tensor | None = None,
    force_block_mask: torch.Tensor | None = None,
    k_variance: bool = False,
    router: str = "mean",
    routing_scale: float | torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_shift: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Build every host-side input of the gfx950 Sol-Attn kernel (arXiv 2607.24027) from Q and the
    quantized K/V: the pooled K/V of the approximate branch and the routed block selection in both
    of the forms the kernel consumes (ragged LUT + bitmap).

    Every output shape here is a function of the INPUT SHAPES alone, and no host-side branch reads
    device data, so this whole function is traceable under torch.compile(fullgraph=True). That is a
    hard requirement, not an accident: it is what lets a caller compile an attention layer around
    Sol-Attn without wrapping the routing in its own opaque custom op.

    q: (batch, seqlen_q, nheads_q, d), any dtype; only used for routing, which is scale invariant,
        so the stored fp8 Q can be passed directly. A Q stored in one of SOL_ATTN_PACKED_FORMATS
        cannot: its codes are neither element addressable nor even d wide, so pass its
        pre-quantization tensor. Scale invariance is what makes the two interchangeable, and the
        MXFP4 and MXFP8 packers' Hadamard rotation is orthogonal, so it leaves the proxy alone too.
        An E8M0-scaled Q's codes are addressable but not proportional to Q, since the scale
        varies per token and channel group; pass its scale as q_scale.
    k_quant: (batch, seqlen_k, nheads_kv, d) already quantized, typically fp8 e4m3.
    v_quant: (batch, seqlen_k, nheads_kv, d_v) already quantized.
    beta: routing threshold, tau = mean_j(proxy) + beta * population_std_j(proxy). Pass this to
        have the selection routed from the pooled proxy, or pass block_attn_mask to supply one.
    block_attn_mask: (batch, nheads_q, num_q_tiles, num_kv_blocks) bool, an ALREADY CHOSEN
        selection to use instead of routing, True == compute that block exactly. Exactly one of
        beta and this may be given. Supplying a selection does not make the approximate branch go
        away: the blocks left unselected are still swept from the pooled K/V, which is the whole
        difference between this and a keep-or-drop sparse launch over the same mask.

        A partial tail block is forced on here exactly as routing forces it, because the
        approximate branch is defined for whole blocks only and so cannot
        represent a short one. That is a kernel requirement rather than a preference, so it is
        applied to a supplied mask rather than rejected.
    force_block_mask: bool, broadcastable to block_attn_mask's shape and ending in num_kv_blocks,
        OR-ed into whichever selection was used. Unlike block_attn_mask this composes with beta
        rather than replacing it: route as normal, then insist on these blocks as well.

        It exists for a caller whose sequence is not homogeneous. Routing compares each block
        against a threshold built from the mean and standard deviation over ALL blocks, so a
        minority of the sequence -- one modality in a packed multimodal sequence, say -- competes
        against a distribution the majority defines, and loses on blocks its own queries needed.
        Naming those blocks costs one exact block each, which is cheap exactly when it is needed,
        because the modality that gets outvoted is the small one.
    num_heads: optional cross-check on nheads_q.
    k_scale, v_scale: that operand's E8M0 1x32 scale image, (batch, seqlen_k, nheads_kv, d // 32)
        uint8, for the block-granular recipes. Pass it ONLY when the operand's scale mode is
        E8M0_PER_1X32; a per-tensor or per-channel descale survives pooling untouched (it does not
        vary along the sequence axis that pooling reduces) and must be left as None so the pooled
        operand reuses it. Supplying one switches that operand to pooling in dequantized space and
        returns a pooled scale for it.
    k_packed_format, v_packed_format: name a format in SOL_ATTN_PACKED_K_FORMATS /
        SOL_ATTN_PACKED_V_FORMATS when that operand's stored codes are not element addressable,
        i.e. sub-byte codes stored in a permuted order. A V format names the packing as well,
        and must be the one the dispatched row reads: *_fp6_p for the FP6-P rows.
        Such an operand cannot be pooled from k_quant / v_quant at all, so the matching
        k_source / v_source must carry the tensor the packer was given and the operand's own
        *_scale must be left as None -- there is nothing to pool it from. k_quant / v_quant are
        still required, and still describe the exact pass.
    k_source, v_source: that operand's pre-quantization tensor, required exactly when the matching
        *_packed_format is named and rejected otherwise.
    k_variance: also return mean_k_var, for a row whose pooled logits take the second-order term
        0.5 * scale^2 * sum_d q_d^2 * var_k[d]. With k_scale or a packed K it also returns
        mean_k_var_scale; a packed K's variance is taken from k_source in the basis its packer
        rotates to, which is the one the kernel sums over.
    router: how beta routing scores blocks. "mean" (the default) is the paper's q_mean . k_mean.
        "error" scores each block by its second-order mass times the relative variance of the
        weights inside it, which spends the exact budget where the pooled estimate is least
        reliable; see _sol_attn_route. It is meant for a kernel that takes mean_k_var. It needs
        K's per-block variance, so not a packed K. With k_scale it scores the dequantized pooled
        mean and variance, as the kernel will load them. It does not change the default outputs.
    routing_scale: required by router="error" and rejected otherwise: the multiplier that turns
        q . k into the softmax logit in nats, for q as routing sees it (dequantized by q_scale
        when that is given) and k as k_quant or, with k_scale, dequantized. softmax_scale for a
        BF16 Q/K; softmax_scale * q_descale * k_descale for a per-tensor quantized one;
        softmax_scale / m for an MXFP8 Q quantized with the multiplier m folded in. A float or a
        one-element tensor, so a device-side descale needs no host read.
    q_scale: Q's E8M0 1x32 scale image, (batch, seqlen_q, nheads_q, d // 32) uint8, for an
        MXFP8 Q. Routing then scores Q dequantized; nothing else reads Q.
    k_shift: (batch, nheads_kv, d), the vector a packed K's quantizer subtracted from every token
        (the `mean` its quantize_*_k took, for K smoothing). The stored codes are of
        k_source - k_shift, so pooling subtracts it from k_source's block means to match; the
        Jensen variance is unaffected by a shift. It needs k_packed_format: an addressable K pools
        its stored codes, which already carry the shift.

    Returns a dict with:
        mean_k, mean_v: pooled K/V in K's / V's own dtype, BSHD
            (batch, num_kv_blocks, nheads_kv, d) and contiguous, reusing the SOURCE descales
            unless the matching *_scale argument was given.
        mean_k_scale, mean_v_scale: the pooled operand's own E8M0 scale, same layout as the input
            scale with seqlen_k replaced by num_kv_blocks, or None when the source descale was
            reused. These go in the mode-2 kernarg's pooled scale slots.
        mean_k_var: the per-block population variance of k_quant, in mean_k's dtype and layout,
            when k_variance is set; otherwise None. For an 8-bit K it is stored divided by
            SOL_ATTN_FP8_VAR_DIVISOR, which the kernel undoes, and an int8 K's is e4m3. With
            k_scale it is the dequantized variance, quantized the way mean_k is. mha_v4_packed
            takes it as mean_k_var.
        mean_k_var_scale: mean_k_var's E8M0 scale in mean_k_scale's layout when k_variance is set
            and K is block-scaled (k_scale given, or packed); otherwise None. mha_v4_packed takes it as mean_k_var_scale.
        mean_k_pooled, mean_v_pooled: the pooled operand as the packer saw it, before quantization,
            for a packed operand whose mean_k / mean_v cannot be read back and dequantized. None
            for the addressable recipes, where dequantizing the pooled tensor with its own scale
            recovers this. A reference should pool over these rather than over the source K/V.
        block_bitmap: uint32 (num_work_items, bitmap_Ds), contiguous, bit j of word j // 32 set
            == KV block j selected for that work item. bitmap_Ds is
            (BLOCK_N // 32) * ceil(num_kv_blocks / BLOCK_N) and the bits above num_kv_blocks are
            SET; see the packing comment below.
        kv_block_indices, lut_start, lut_count: the ragged LUT, int32. kv_block_indices is
            overallocated by block_attn_mask_to_ragged_lut; only the spans are meaningful.
        num_kv_blocks, bitmap_Ds, num_q_tiles: kernarg scalars / grid geometry.
        block_attn_mask: the selection actually used, for reference comparisons. This is the routed
            mask, or the supplied one after the partial-tail block was forced on.

    block_bitmap, lut_start and lut_count are all indexed by the kernel's
        lut_idx = (b * nheads_q + h) * num_q_tiles + q_tile
    and are laid out contiguously in that order, so the bitmap and the LUT cannot disagree: both are
    derived from one boolean mask.
    """
    if (beta is None) == (block_attn_mask is None):
        raise ValueError(
            "pass exactly one of beta (route the selection from the pooled proxy) or "
            "block_attn_mask (use an already chosen one)"
        )
    if q.dim() != 4 or k_quant.dim() != 4 or v_quant.dim() != 4:
        raise ValueError("q, k_quant and v_quant must be 4D (batch, seqlen, nheads, d)")
    batch, seqlen_q, nhead_q, _ = q.shape
    seqlen_k, nhead_kv = k_quant.shape[1], k_quant.shape[2]
    if k_quant.shape[0] != batch or v_quant.shape[:3] != k_quant.shape[:3]:
        raise ValueError("k_quant and v_quant must share (batch, seqlen_k, nheads_kv)")
    if nhead_q % nhead_kv != 0:
        raise ValueError("nheads_q must be a multiple of nheads_kv")
    if num_heads is not None and num_heads != nhead_q:
        raise ValueError(f"num_heads {num_heads} does not match q's {nhead_q}")
    for name, fmt, source, scale, formats in (
        ("k", k_packed_format, k_source, k_scale, SOL_ATTN_PACKED_K_FORMATS),
        ("v", v_packed_format, v_source, v_scale, SOL_ATTN_PACKED_V_FORMATS),
    ):
        if fmt is not None and fmt not in formats:
            raise ValueError(f"{name}_packed_format {fmt!r} is not one of {formats}")
        if (fmt is None) != (source is None):
            raise ValueError(
                f"{name}_source and {name}_packed_format go together: a packed operand's codes "
                f"are not element addressable, so pooling it needs the tensor the packer was given"
            )
        if fmt is not None and scale is not None:
            raise ValueError(
                f"{name}_scale does not apply to a packed operand: {name}_packed_format "
                f"{fmt!r} pools from {name}_source and quantizes it again, so there is no stored "
                f"scale to pool"
            )
        if source is not None and source.shape[:3] != k_quant.shape[:3]:
            raise ValueError(
                f"{name}_source must share (batch, seqlen_k, nheads_kv) with k_quant"
            )

    if router not in ("mean", "error"):
        raise ValueError(f"router must be 'mean' or 'error', got {router!r}")
    error_router = router == "error"
    if error_router and block_attn_mask is not None:
        raise ValueError(
            "router='error' routes from beta; a supplied block_attn_mask is not routed"
        )
    if error_router != (routing_scale is not None):
        raise ValueError(
            "routing_scale goes with router='error': the error proxy is not scale invariant and "
            "needs it, and the mean proxy is and ignores it"
        )
    if error_router and k_packed_format is not None:
        raise ValueError(
            "router='error' needs K's per-block variance, and a packed K's codes have no "
            "element to take a variance of"
        )
    if q_scale is not None and (q_scale.dim() != 4 or q_scale.shape[:3] != q.shape[:3]):
        raise ValueError(
            "q_scale must be Q's E8M0 1x32 scale image, (batch, seqlen_q, nheads_q, d // 32)"
        )
    if k_shift is not None:
        if k_packed_format is None:
            raise ValueError(
                "k_shift goes with k_packed_format: an addressable K pools its stored codes, "
                "which already carry the shift its quantizer subtracted"
            )
        expected_shift = (batch, nhead_kv, q.shape[3])
        if tuple(k_shift.shape) != expected_shift:
            raise ValueError(
                f"k_shift must be (batch, nheads_kv, d) = {expected_shift}, got "
                f"{tuple(k_shift.shape)}"
            )

    num_q_tiles = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N

    # Each operand pools independently, because a recipe can be block-granular on one and not the
    # other: mxfp8 has an E8M0 K and a per-tensor V, f8f6 the reverse.
    mean_k_scale = mean_v_scale = None
    mean_k_pooled = mean_v_pooled = None
    mean_k_var = mean_k_var_scale = None
    k_routing_var = None
    if k_packed_format is not None:
        pooled_k = _sol_attn_pool_packed_k(
            k_source, k_packed_format, BLOCK_N, variance=k_variance, shift=k_shift
        )
        mean_k, mean_k_scale, mean_k_pooled = pooled_k[:3]
        if k_variance:
            mean_k_var, mean_k_var_scale = pooled_k[3:]
        k_routing = mean_k_pooled
    elif k_scale is None and (k_variance or error_router):
        mean_k, stored_var = _sol_attn_pool_reuse_descale(
            k_quant, BLOCK_N, variance=True
        )
        k_routing = mean_k
        if k_variance:
            mean_k_var = stored_var
        if error_router:
            # The variance the kernel will load, back in K's own units.
            k_routing_var = stored_var.float()
            if k_quant.element_size() == 1:
                k_routing_var = k_routing_var * SOL_ATTN_FP8_VAR_DIVISOR
    elif k_scale is None:
        mean_k = _sol_attn_pool_reuse_descale(k_quant, BLOCK_N)
        k_routing = mean_k
    elif k_variance or error_router:
        mean_k, mean_k_scale, k_routing, stored_var, stored_var_scale = (
            _sol_attn_pool_mx(k_quant, k_scale, BLOCK_N, variance=True)
        )
        if k_variance:
            mean_k_var, mean_k_var_scale = stored_var, stored_var_scale
        if error_router:
            k_routing_var = _e8m0_dequantize(stored_var, stored_var_scale)
    else:
        mean_k, mean_k_scale, k_routing = _sol_attn_pool_mx(k_quant, k_scale, BLOCK_N)
    if v_packed_format is not None:
        mean_v, mean_v_scale, mean_v_pooled = _sol_attn_pool_fp6_p_v(
            v_source, v_packed_format, BLOCK_N
        )
    elif v_scale is None:
        mean_v = _sol_attn_pool_reuse_descale(v_quant, BLOCK_N)
    else:
        mean_v, mean_v_scale, _ = _sol_attn_pool_mx(v_quant, v_scale, BLOCK_N)

    partial_tail = seqlen_k % BLOCK_N != 0
    if block_attn_mask is None:
        if q_scale is not None:
            q = _e8m0_dequantize(q, q_scale)
        # Route on the pooled values the kernel will actually load, i.e. after the rounding.
        if error_router:
            q_mean, q_sq_mean = _sol_attn_pool_q(q, BLOCK_M, second_moment=True)
            block_attn_mask = _sol_attn_route(
                q_mean,
                k_routing,
                beta,
                partial_tail=partial_tail,
                q_sq_mean=q_sq_mean,
                k_var=k_routing_var,
                logit_scale=routing_scale,
            )
        else:
            block_attn_mask = _sol_attn_route(
                _sol_attn_pool_q(q, BLOCK_M),
                k_routing,
                beta,
                partial_tail=partial_tail,
            )
    else:
        expected = (batch, nhead_q, num_q_tiles, num_kv_blocks)
        if tuple(block_attn_mask.shape) != expected:
            raise ValueError(
                f"block_attn_mask must be {expected} "
                f"(batch, nheads_q, num_q_tiles, num_kv_blocks), got "
                f"{tuple(block_attn_mask.shape)}"
            )
        if block_attn_mask.dtype != torch.bool:
            raise ValueError(
                f"block_attn_mask must be bool, got {block_attn_mask.dtype}"
            )
        if partial_tail:
            tail = F.one_hot(
                torch.tensor(num_kv_blocks - 1, device=block_attn_mask.device),
                num_kv_blocks,
            ).bool()
            block_attn_mask = block_attn_mask | tail

    if force_block_mask is not None:
        if force_block_mask.dtype != torch.bool:
            raise ValueError(
                f"force_block_mask must be bool, got {force_block_mask.dtype}"
            )
        if force_block_mask.shape[-1] != num_kv_blocks:
            raise ValueError(
                f"force_block_mask must end in num_kv_blocks={num_kv_blocks}, got "
                f"{tuple(force_block_mask.shape)}"
            )
        block_attn_mask = block_attn_mask | force_block_mask

    kv_block_indices, lut_start, lut_count = block_attn_mask_to_ragged_lut(
        block_attn_mask
    )
    lut_start = lut_start.to(torch.int32)
    lut_count = lut_count.to(torch.int32)

    # Bitmap: same mask, packed 32 blocks per uint32 word, num_work_items rows of bitmap_Ds words in
    # lut_idx order.
    #
    # The row length is rounded up to whole BLOCK_N-block groups (BLOCK_N // 32 words), not to a
    # single word: one group is read per approximate tile as a single aligned load of that width, so
    # a row that is not a whole number of groups misaligns every tile after the first.
    # The padding bits are SET, because a set bit means "already computed exactly" and so masks that
    # column out of the approximate pass; that is what clips the last tile's overhang, and it is why
    # the kernel needs no masked-tail path at all. Clearing them instead would let the pooled rows
    # past num_kv_blocks contribute spurious mass.
    num_work_items = batch * nhead_q * num_q_tiles
    bitmap_Ds = (BLOCK_N // 32) * ((num_kv_blocks + BLOCK_N - 1) // BLOCK_N)
    bits = block_attn_mask.reshape(num_work_items, num_kv_blocks)
    if bitmap_Ds * 32 != num_kv_blocks:
        bits = F.pad(bits, (0, bitmap_Ds * 32 - num_kv_blocks), value=True)
    weights = (1 << torch.arange(32, device=q.device, dtype=torch.int64)).view(1, 1, 32)
    block_bitmap = (
        (bits.reshape(num_work_items, bitmap_Ds, 32).to(torch.int64) * weights)
        .sum(dim=-1)
        .to(torch.uint32)
        .contiguous()
    )

    return {
        "mean_k": mean_k,
        "mean_v": mean_v,
        "mean_k_scale": mean_k_scale,
        "mean_v_scale": mean_v_scale,
        "mean_k_pooled": mean_k_pooled,
        "mean_v_pooled": mean_v_pooled,
        "mean_k_var": mean_k_var,
        "mean_k_var_scale": mean_k_var_scale,
        "block_bitmap": block_bitmap,
        "kv_block_indices": kv_block_indices,
        "lut_start": lut_start,
        "lut_count": lut_count,
        "num_kv_blocks": num_kv_blocks,
        "bitmap_Ds": bitmap_Ds,
        "num_q_tiles": num_q_tiles,
        "block_attn_mask": block_attn_mask,
    }
