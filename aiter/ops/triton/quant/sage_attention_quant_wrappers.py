import functools

import torch
import triton
import triton.language as tl

import aiter
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    map_dims,
)
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import (
    _compute_delta_s_kernel,
    _q_smooth_int8_kernel,
    _rot_k_only_kernel,
    _rot_q_kernel,
    _rotate_quantize_k_kernel,
    _rotate_quantize_q_kernel,
    sage_quant_kernel,
    sage_quant_v_fp4_colmajor_kernel,
    sage_quant_v_kernel,
    sage_quant_v_mxfp4_colmajor_kernel,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.quant.f4f4_solo import (
    quantize_f4f4_solo_k,
    quantize_f4f4_solo_v,
)


def fused_sage_quant_mxfp4(
    q,
    k,
    v,
    BLOCK_M,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
    q_smoothing=False,
    layout="bshd",
):
    if layout == "bhsd":
        b, _h_qo, _qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )

    elif layout == "bshd":
        b, _qo_len, _h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")

    # padded_head_dim = max(16, 1 << (head_dim - 1).bit_length())
    sm_scale = head_dim**-0.5

    q_fp4, q_scale, k_fp4, k_scale, delta_s = smooth_rotate_downcast_qk(
        q,
        k,
        BLOCK_SIZE_M=BLOCK_M,
        hadamard_rotation=hadamard_rotation,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )

    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(FP8_TYPE).max
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    BLOCK_K = 1024
    K_NUM_BLKS = (kv_len + BLOCK_K - 1) // BLOCK_K

    # V tensor per channel quantization
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)
    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_d_v,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLOCK_K,
        num_stages=5,
        num_warps=8,
    )

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s


def sage_quant_mxfp4(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    USE_RNE=False,
    R=None,
    BLOCK_R=32,
    smooth_k=True,
    return_lse=False,
):
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = v.shape

        stride_bz_v, stride_h_v, stride_seq_v, stride_d_v = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    # Apply K tensor smoothing following SageAttention approach
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)

    # padded_head_dim = max(16, 1 << (head_dim - 1).bit_length())

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # Capture un-rotated K mean before rotation_smooth_qk so we can build the
    # ring-attention LSE compensation in natural log units below.
    if return_lse and smooth_k:
        k_mean = k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
    else:
        k_mean = None

    q_orig = q
    q, k, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
        smooth_k=smooth_k,
    )

    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_d_v,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    downcast_func = downcast_to_mxfp

    q_fp4, q_scale = downcast_func(q, torch.uint8, axis=-1)
    k_fp4, k_scale = downcast_func(k, torch.uint8, axis=-1)

    if not return_lse:
        return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s

    # K-smoothing shifts every qk_ij by a row-wise constant
    # delta_lse_i = sm_scale * Q_i . k_mean^T (in natural log units).
    # Adding it back to the kernel's softmax_lse recovers the LSE for un-smoothed
    # K, which is what FA-style ring-attention merges require.
    if k_mean is None:
        delta_lse = torch.zeros(
            (b, h_qo, qo_len), device=q_orig.device, dtype=torch.float32
        )
    else:
        if layout == "bhsd":
            q_bhsd = q_orig
            kmean_bhsd = k_mean
        else:
            q_bhsd = q_orig.transpose(1, 2)
            kmean_bhsd = k_mean.transpose(1, 2)
        if h_qo != h_kv:
            assert (
                h_qo % h_kv == 0
            ), f"GQA ratio must be integer, got h_qo={h_qo}, h_kv={h_kv}"
            kmean_bhsd = kmean_bhsd.repeat_interleave(h_qo // h_kv, dim=1)
        delta_lse = (q_bhsd.to(torch.float32) * kmean_bhsd.to(torch.float32)).sum(
            dim=-1
        ) * sm_scale

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s, delta_lse


def _apply_int8_q_smoothing(q, k, BLKQ, layout, sm_scale):
    """Center Q per block and compute delta_s bias for INT8 Sage v1 (no Hadamard)."""
    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_NUM_BLKS = (s_q + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (s_k + BLKQ - 1) // BLKQ

    q_mean = torch.empty((b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device)
    delta_s = torch.empty(
        (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
    )
    q_out = torch.empty_like(q)

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_qob, stride_qom, stride_qoh, stride_qod = map_dims(q_out.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)

    sm_scale_log2 = sm_scale * 1.4426950408889634
    grid_q = (b * h_q, Q_NUM_BLKS, triton.cdiv(d, 32))
    _q_smooth_int8_kernel[grid_q](
        q,
        q_out,
        q_mean,
        sm_scale_log2,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qob,
        stride_qoh,
        stride_qom,
        stride_qod,
        q_mean.stride(0),
        q_mean.stride(1),
        q_mean.stride(2),
        q_mean.stride(3),
        h_q,
        s_q,
        d,
        BLOCK_M=BLKQ,
        BLOCK_D=32,
    )

    grid_delta = (b * h_q, Q_NUM_BLKS, K_NUM_BLKS)
    _compute_delta_s_kernel[grid_delta](
        q_mean,
        k,
        delta_s,
        q_mean.stride(0),
        q_mean.stride(1),
        q_mean.stride(2),
        q_mean.stride(3),
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        delta_s.stride(0),
        delta_s.stride(1),
        delta_s.stride(2),
        delta_s.stride(3),
        h_q,
        h_k,
        s_k,
        d,
        BLOCK_N=BLKQ,
    )
    return q_out, delta_s


_F4F4_V_KPERM_CACHE = {}


def _f4f4_v_kperm(device):
    """Cached int32 [64] 'meas' kv-column permutation for the f4f4 col-major V pack
    (col c holds kv-token kperm[c]). Built once per device so it is not recreated per
    call (and stays out of any CUDA-graph capture region)."""
    kp = _F4F4_V_KPERM_CACHE.get(device)
    if kp is None:
        s = torch.arange(64, device=device)
        j = s % 32
        pi = 4 * (j // 8) + 16 * ((j // 4) % 2) + (j % 4)
        tau64 = 32 * (s // 32) + pi
        kperm = torch.empty(64, dtype=torch.long, device=device)
        kperm[tau64] = s  # kperm[col] = tau64^{-1}(col)
        kp = kperm.to(torch.int32).contiguous()
        _F4F4_V_KPERM_CACHE[device] = kp
    return kp


FP4_V_TILE_TOKENS = 128
FP4_V_PACKED_BYTES_PER_TOKEN = 64
FP4_V_BUFFER_SLACK_BYTES = 64


def fp4_v_padded_sequence(sequence):
    """Round a V sequence length up to the 128-token FP4 packing tile."""
    return ((sequence + FP4_V_TILE_TOKENS - 1) // FP4_V_TILE_TOKENS) * FP4_V_TILE_TOKENS


def fp4_v_raw_buffer_size(batch, sequence, heads):
    """Return bytes for the packed FP4 V backing buffer, including view slack."""
    return (
        batch * fp4_v_padded_sequence(sequence) * heads * FP4_V_PACKED_BYTES_PER_TOKEN
        + FP4_V_BUFFER_SLACK_BYTES
    )


def sage_quant_v_f4f4(v, layout="bshd"):
    """Pack per-channel FP4 V into a padded, slack-backed col-major LDS layout."""
    if layout == "bshd":
        b, kv_len, h_kv, head_dim = v.shape
        v_tok = v.permute(0, 2, 1, 3)
    elif layout == "bhsd":
        b, h_kv, kv_len, head_dim = v.shape
        v_tok = v
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")

    tile = FP4_V_TILE_TOKENS
    assert head_dim == 128, f"f4f4 requires head_dim=128, got {head_dim}"
    padded_kv_len = fp4_v_padded_sequence(kv_len)
    nT = padded_kv_len // tile
    amax = v_tok.abs().amax(dim=-2).to(torch.float32)
    v_descale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax)).contiguous()
    kperm = _f4f4_v_kperm(v.device)
    buf = torch.empty(
        fp4_v_raw_buffer_size(b, kv_len, h_kv),
        dtype=torch.uint8,
        device=v.device,
    )
    packed = buf[: b * h_kv * padded_kv_len * FP4_V_PACKED_BYTES_PER_TOKEN].view(
        b, h_kv, nT * tile * FP4_V_PACKED_BYTES_PER_TOKEN
    )
    sage_quant_v_fp4_colmajor_kernel[(b * h_kv * nT * 8,)](
        v_tok,
        packed,
        v_descale,
        kperm,
        v_tok.stride(0),
        v_tok.stride(1),
        v_tok.stride(2),
        v_tok.stride(3),
        packed.stride(0),
        packed.stride(1),
        v_descale.stride(0),
        v_descale.stride(1),
        h_kv,
        nT,
        kv_len,
    )
    v_fp4_view = torch.as_strided(
        buf,
        (b, kv_len, h_kv, 128),
        (
            h_kv * padded_kv_len * FP4_V_PACKED_BYTES_PER_TOKEN,
            FP4_V_PACKED_BYTES_PER_TOKEN,
            padded_kv_len * FP4_V_PACKED_BYTES_PER_TOKEN,
            1,
        ),
    )
    return v_fp4_view, v_descale


@torch.library.custom_op("aiter::pack_v_mxfp4_colmajor_raw", mutates_args=())
def pack_v_mxfp4_colmajor_raw(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack V into contiguous payload and ASM-order E8M0 scale buffers.

    Each 128-token tile contributes 512 scale bytes: four 32-token blocks times
    128 channels, arranged in the gather order consumed by the F4F4/F6F4 kernels.
    """
    batch, sequence, heads, head_dim = value.shape
    if head_dim != 128 or not value.is_contiguous():
        raise ValueError("MXFP4 V packing requires contiguous hd128 BSHD input")
    padded_sequence = fp4_v_padded_sequence(sequence)
    tiles = padded_sequence // FP4_V_TILE_TOKENS
    raw = torch.empty(
        fp4_v_raw_buffer_size(batch, sequence, heads),
        dtype=torch.uint8,
        device=value.device,
    )
    scale = torch.empty(
        (batch, heads, tiles * 512), dtype=torch.uint8, device=value.device
    )
    value_bhsd = value.permute(0, 2, 1, 3)
    payload = raw[: batch * heads * tiles * 8192].view(batch, heads, tiles * 8192)
    kperm = _f4f4_v_kperm(value.device)
    sage_quant_v_mxfp4_colmajor_kernel[(batch * heads * tiles * 16,)](
        value_bhsd,
        payload,
        scale,
        kperm,
        value_bhsd.stride(0),
        value_bhsd.stride(1),
        value_bhsd.stride(2),
        value_bhsd.stride(3),
        payload.stride(0),
        payload.stride(1),
        scale.stride(0),
        scale.stride(1),
        heads,
        tiles,
        sequence,
        num_warps=1,
        num_stages=1,
    )
    return raw, scale


@pack_v_mxfp4_colmajor_raw.register_fake
def _pack_v_mxfp4_colmajor_raw_fake(value):
    batch, sequence, heads, _ = value.shape
    tiles = fp4_v_padded_sequence(sequence) // FP4_V_TILE_TOKENS
    return (
        value.new_empty(
            (fp4_v_raw_buffer_size(batch, sequence, heads),), dtype=torch.uint8
        ),
        value.new_empty((batch, heads, tiles * 512), dtype=torch.uint8),
    )


def sage_quant_v_mxfp4(value):
    """Return true-MXFP4 V data view and kernel-ready E8M0 block-scale image."""
    batch, sequence, heads, _ = value.shape
    padded_sequence = fp4_v_padded_sequence(sequence)
    raw, scale = pack_v_mxfp4_colmajor_raw(value)
    view = torch.as_strided(
        raw,
        (batch, sequence, heads, 128),
        (heads * padded_sequence * 64, 64, padded_sequence * 64, 1),
    )
    return view, scale


def sage_quant_f4f4(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    USE_RNE=False,
    R=None,
    BLOCK_R=32,
):
    """Quantize rotated MXFP4 Q/K plus true-MXFP4 V for the F4F4 ASM kernel."""
    del FP8_TYPE, FP8_MAX, BLKK, USE_RNE
    if layout != "bshd":
        raise ValueError(f"f4f4 requires bshd layout, got {layout}")
    _, _, _, head_dim = q.shape
    _, kv_len, _, _ = v.shape

    tile = 128
    assert head_dim == 128, f"f4f4 requires head_dim=128, got {head_dim}"
    assert (
        kv_len % tile == 0
    ), f"f4f4 col-major V pack requires kv_len % {tile} == 0, got {kv_len}"

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # Q/K: identical to sage_quant_mxfp4 (hadamard rotation + smoothing -> mxfp4).
    q, k, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )
    q_fp4, q_scale = downcast_to_mxfp(q, torch.uint8, axis=-1)
    k_fp4, k_scale = downcast_to_mxfp(k, torch.uint8, axis=-1)

    v_fp4_view, v_descale = sage_quant_v_mxfp4(v)
    return q_fp4, q_scale, k_fp4, k_scale, v_fp4_view, v_descale, delta_s


def sage_quant_mxfp6(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    R=None,
    BLOCK_R=32,
    f6f4=False,
    q_packer=None,
    k_packer=None,
):
    """MXFP6-E2M3 QK quantize (+ V) for the aiter mxfp6 (f6f8) / f6f4 fmha kernels.

    Rotates/smooths Q,K (Hadamard R, folding sm_scale*log2e into Q) then packs both to
    MXFP6-E2M3: Q -> [...,96] data + E8M0 scale; K -> kernel-ready LDS-order view with the
    E8M0 K-scale in the per-tile tail. By default Q/K are packed with the in-tree Triton
    packers (quantize_fp6_lastdim_triton / quantize_fp6_k_lds_order_triton); pass q_packer /
    k_packer callables to override (e.g. a bench that swaps the packer via AITER_MXFP6_PACK
    or forces the numpy path). The V operand is selected by f6f4:
      * f6f4=False (f6f8): raw fp8 V via sage_quant_v_kernel (per-channel descale).
            * f6f4=True: true-MXFP4 V with per-(channel, 32-token) E8M0 scales.
    Only the selected V operand is computed (no wasted fp8 quant on the f6f4 path).
    Returns (q_fp6, q_scale, k_view, k_scale, v_quantized, v_scale, delta_s). bshd only.
    """
    if q_packer is None or k_packer is None:
        import os as _os

        from aiter.ops.triton.quant import mxfp6_fmha_pack as _hp

        # Default to the fused TRITON packers (single in-graph kernels; hide the all-to-all far
        # better under torch.compile than the many-kernel torch packs). Set AITER_MXFP6_QK_TRITON=0
        # for the pure-torch (traceable ATen) packers.
        _use_triton_qk = _os.environ.get("AITER_MXFP6_QK_TRITON", "1") != "0"
        if _use_triton_qk:
            _default_q_packer = _hp.quantize_fp6_lastdim_triton

            def _default_k_packer(_k):
                return _hp.quantize_fp6_k_lds_order_triton(_k, tile=128)

        else:
            _default_q_packer = _hp.quantize_fp6_lastdim_torch

            def _default_k_packer(_k):
                return _hp.quantize_fp6_k_lds_order_torch(_k, tile=128)

    assert layout == "bshd", f"sage_quant_mxfp6 expects bshd, got {layout}"
    b, _qo_len, _h_qo, head_dim = q.shape
    _, kv_len, h_kv, _ = v.shape
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q, k, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )

    # V operand: true-MXFP4 (f6f4) or raw fp8 (f6f8) -- only the selected one.
    if f6f4:
        v_quantized, v_scale = sage_quant_v_mxfp4(v)
    else:
        v_quantized = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)
        K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK
        v_scale = v.abs().amax(dim=1).to(torch.float32) / FP8_MAX
        grid = (b * h_kv * K_NUM_BLKS,)
        sage_quant_v_kernel[grid](
            v,
            v_quantized,
            v_scale,
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            v_scale.stride(0),
            v_scale.stride(1),
            b,
            h_kv,
            K_NUM_BLKS,
            kv_len,
            D=head_dim,
            BLK_K=BLKK,
            num_stages=3,
            num_warps=8,
        )

    # Q -> base fp6 pack; K -> coalesced LDS-order pack (E8M0 K-scale in the tile tail).
    # Use caller-supplied packers when given (overridable), else the in-tree Triton packers.
    q_fp6, q_scale = q_packer(q) if q_packer is not None else _default_q_packer(q)
    k_view, k_scale = k_packer(k) if k_packer is not None else _default_k_packer(k)
    return q_fp6, q_scale, k_view, k_scale, v_quantized, v_scale, delta_s


def sage_quant(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
    layout="bshd",
    smooth_k=True,
    q_smoothing=False,
    return_lse=False,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
):
    """
    Quantize Q and K tensors to INT8 with per-block scaling.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        FP8_TYPE: Floating-point type for the quantized V tensor
        FP8_MAX: Maximum value for the quantized V tensor
        BLKQ: Block size for Q quantization
        BLKK: Block size for K quantization
        sm_scale: Softmax scale factor (defaults to head_dim^-0.5)
        layout: Either "bshd" or "bhsd"
        smooth_k: Whether to apply SageAttention-style smoothing to K tensor (default: True)
        q_smoothing: Whether to center Q per block and return delta_s correction (default: False)
        return_lse: If True, additionally return a per-query-row LSE correction
            term that compensates for K smoothing (default: False)
        hadamard_rotation: Apply normalized Hadamard rotation to Q/K before INT8 quant
        R: Optional pre-built Hadamard matrix (BLOCK_R x BLOCK_R)
        BLOCK_R: Hadamard tile size; required when hadamard_rotation=True and R is None
    Returns:
        q_int8: Quantized Q tensor
        q_scale: Per-block scales for Q
        k_int8: Quantized K tensor
        k_scale: Per-block scales for K
        v_fp8: Quantized V tensor
        v_scale: Per-(B,H,D) scales for V
        delta_s (when q_smoothing=True): [B,H,Q_blks,seqlen_k] bias for Q smoothing
        delta_lse (when return_lse=True): float32 (B, H_q, S_q) ring-attention LSE fixup
    """
    q_int8 = torch.empty_like(q, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty_like(k, dtype=torch.int8, device=k.device)
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    Q_NUM_BLKS = (qo_len + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    q_orig = q
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    delta_s = None
    k_mean = None
    if hadamard_rotation:
        if R is None:
            assert (
                BLOCK_R is not None
            ), "if using hadamard rotation, BLOCK_R must be provided when R is None"
            R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (
                BLOCK_R**0.5
            )
        else:
            BLOCK_R = R.shape[-1]
        if head_dim % BLOCK_R != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by BLOCK_R ({BLOCK_R})"
            )

        if return_lse and smooth_k:
            k_mean = k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

        q, k, _ = rotation_smooth_qk(
            q,
            k,
            BLKQ,
            R=R,
            BLOCK_R=BLOCK_R,
            q_smoothing=False,
            sm_scale=None,
            layout=layout,
            smooth_k=False,
        )
        if q_smoothing:
            if smooth_k:
                if k_mean is None:
                    k_mean = k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
                k = k - k_mean
            q, delta_s = _apply_int8_q_smoothing(q, k, BLKQ, layout, sm_scale)
        elif smooth_k:
            if k_mean is None:
                k_mean = k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
            k = k - k_mean
    elif q_smoothing:
        if smooth_k:
            k_mean = k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
            k = k - k_mean
        q, delta_s = _apply_int8_q_smoothing(q, k, BLKQ, layout, sm_scale)
    elif smooth_k:
        k_mean = k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
        k = k - k_mean

    q_scale = torch.empty((b, h_qo, Q_NUM_BLKS), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, K_NUM_BLKS), device=q.device, dtype=torch.float32)

    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    q_task_count = b * h_qo * Q_NUM_BLKS
    k_task_count = b * h_kv * K_NUM_BLKS
    v_task_count = b * h_kv * K_NUM_BLKS

    grid = (q_task_count + k_task_count + v_task_count,)

    # call sage_quant_kernel
    sage_quant_kernel[grid](
        q,
        q_int8,
        q_scale,
        k,
        k_int8,
        k_scale,
        v,
        v_fp8,
        v_scale,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        (1.0 if q_smoothing else (sm_scale * 1.4426950408889634)),
        q_task_count,
        k_task_count,
        b,
        h_qo,
        h_kv,
        Q_NUM_BLKS,
        K_NUM_BLKS,
        qo_len,
        kv_len,
        triton.next_power_of_2(kv_len),
        FP8_MAX=FP8_MAX,
        INT8_MAX=torch.iinfo(q_int8.dtype).max,
        D=head_dim,
        BLK_Q=BLKQ,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    out = [q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale]
    if q_smoothing:
        out.append(delta_s)
    if return_lse:
        if k_mean is None:
            delta_lse = torch.zeros(
                (b, h_qo, qo_len), device=q_orig.device, dtype=torch.float32
            )
        else:
            if layout == "bhsd":
                q_bhsd = q_orig
                kmean_bhsd = k_mean
            else:
                q_bhsd = q_orig.transpose(1, 2)
                kmean_bhsd = k_mean.transpose(1, 2)

            if h_qo != h_kv:
                assert (
                    h_qo % h_kv == 0
                ), f"GQA ratio must be integer, got h_qo={h_qo}, h_kv={h_kv}"
                kmean_bhsd = kmean_bhsd.repeat_interleave(h_qo // h_kv, dim=1)

            delta_lse = (q_bhsd.to(torch.float32) * kmean_bhsd.to(torch.float32)).sum(
                dim=-1
            ) * sm_scale
        out.append(delta_lse)
    return tuple(out)


def rotation_smooth_qk(
    q,
    k,
    BLOCK_SIZE_M,
    R=None,
    BLOCK_R=32,
    q_smoothing=False,
    sm_scale=None,
    layout="bhsd",
    smooth_k=True,
):
    if R is None:  # Generate Hadamard Matrix R if not given
        assert (
            BLOCK_R is not None
        ), "if not passing R (hadamard matrix), BLOCK_R (size of the hadamard matrix) must be provided."
        R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (
            BLOCK_R**0.5
        )
    else:
        BLOCK_R = R.shape[-1]

    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    # shapes
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_rot = torch.empty_like(q)
    K_rot = torch.empty_like(k)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    if q_smoothing:
        q_mean = torch.empty(
            (b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device
        )
        delta_s = torch.empty(
            (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
        )
    else:
        q_mean = None
        delta_s = None

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_qob, stride_qom, stride_qoh, stride_qod = map_dims(Q_rot.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)
    stride_kob, stride_kon, stride_koh, stride_kod = map_dims(K_rot.stride(), bshd)
    # rotate q and optionally smooth
    grid_q = (b * h_q, Q_NUM_BLKS, d // BLOCK_R)
    _rot_q_kernel[grid_q](
        q,
        Q_rot,
        q_mean,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qob,
        stride_qoh,
        stride_qom,
        stride_qod,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        R.stride(0),
        R.stride(1),
        h_q,
        s_q,
        d,
        q_smoothing=q_smoothing,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=BLOCK_R,
    )

    # rotate k
    grid_k = (b * h_k, K_NUM_BLKS, d // BLOCK_R)
    _rot_k_only_kernel[grid_k](
        k,
        K_rot,
        R,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_kob,
        stride_koh,
        stride_kon,
        stride_kod,
        R.stride(0),
        R.stride(1),
        h_k,
        s_k,
        d,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=BLOCK_R,
    )

    # smooth k
    if smooth_k:
        K_rot = K_rot - K_rot.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

    if q_smoothing:
        # compute delta s that needs to be added due to q smoothing
        # Q x K = Q x H x H.T x K
        # = ((Q x H - q_mean + q_mean) x H.T x K
        # = Q_rot x K_rot + q_mean x K_rot
        # = Q_rot x K_rot + delta_s
        grid_delta = (b * h_q, Q_NUM_BLKS, K_NUM_BLKS)
        _compute_delta_s_kernel[grid_delta](
            q_mean,
            K_rot,
            delta_s,
            q_mean.stride(0),
            q_mean.stride(1),
            q_mean.stride(2),
            q_mean.stride(3),
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            delta_s.stride(0),
            delta_s.stride(1),
            delta_s.stride(2),
            delta_s.stride(3),
            h_q,
            h_k,
            s_k,
            d,
            BLOCK_N=BLOCK_SIZE_M,
        )

    return Q_rot, K_rot, delta_s


def smooth_rotate_downcast_qk(
    q,
    k,
    BLOCK_SIZE_M,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
    q_smoothing=False,
    sm_scale=None,
    layout="bhsd",
):
    if hadamard_rotation:
        if R is None:
            assert (
                BLOCK_R is not None
            ), "if using hadamard rotation, BLOCK_R (size of the hadamard matrix) must be provided."
            R = create_hadamard_matrix(BLOCK_R, device=q.device, dtype=q.dtype) / (
                BLOCK_R**0.5
            )
        else:
            BLOCK_R = R.shape[-1]

    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    # shapes
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    if q_smoothing:
        q_mean = torch.empty(
            (b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device
        )
        delta_s = torch.empty(
            (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
        )
    else:
        q_mean = None
        delta_s = None

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)

    Q_q = torch.empty((*q.shape[:-1], d // 2), dtype=torch.uint8, device=q.device)
    Q_descale = torch.empty(
        (*q.shape[:-1], d // 32), dtype=torch.uint8, device=q.device
    )
    K_q = torch.empty((*k.shape[:-1], d // 2), dtype=torch.uint8, device=k.device)
    K_descale = torch.empty(
        (*k.shape[:-1], d // 32), dtype=torch.uint8, device=k.device
    )

    stride_qqb, stride_qqm, stride_qqh, stride_qqd = map_dims(Q_q.stride(), bshd)
    stride_kqb, stride_kqn, stride_kqh, stride_kqd = map_dims(K_q.stride(), bshd)

    stride_qsb, stride_qsm, stride_qsh, stride_qsd = map_dims(Q_descale.stride(), bshd)
    stride_ksb, stride_ksn, stride_ksh, stride_ksd = map_dims(K_descale.stride(), bshd)

    grid_q = (b * h_q * Q_NUM_BLKS,)
    _rotate_quantize_q_kernel[grid_q](
        q,
        Q_q,
        Q_descale,
        q_mean,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qqb,
        stride_qqm,
        stride_qqh,
        stride_qqd,
        stride_qsb,
        stride_qsm,
        stride_qsh,
        stride_qsd,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        b,
        h_q,
        s_q,
        d,
        q_smoothing=q_smoothing,
        hadamard_rotation=hadamard_rotation,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_R=BLOCK_R,
        D=d,
        num_warps=4,
        num_stages=5,
    )

    grid_k = (b * h_k * K_NUM_BLKS,)
    _rotate_quantize_k_kernel[grid_k](
        q,
        Q_q,
        Q_descale,
        q_mean,
        k,
        K_q,
        K_descale,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_qqb,
        stride_qqm,
        stride_qqh,
        stride_qqd,
        stride_qsb,
        stride_qsm,
        stride_qsh,
        stride_qsd,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_kqb,
        stride_kqn,
        stride_kqh,
        stride_kqd,
        stride_ksb,
        stride_ksn,
        stride_ksh,
        stride_ksd,
        b,
        h_q,
        h_k,
        s_q,
        s_k,
        d,
        q_smoothing=q_smoothing,
        hadamard_rotation=hadamard_rotation,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_R=BLOCK_R,
        D=d,
        num_warps=4,
        num_stages=5,
    )

    if q_smoothing:
        # 3. Compute Smoothing Delta S
        # Grid: Each Q-block x Each K-block
        grid_delta = (b * h_q, Q_NUM_BLKS, K_NUM_BLKS)
        _compute_delta_s_kernel[grid_delta](
            q_mean,
            k,
            delta_s,
            q_mean.stride(0),
            q_mean.stride(1),
            q_mean.stride(2),
            q_mean.stride(3),
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            delta_s.stride(0),
            delta_s.stride(1),
            delta_s.stride(2),
            delta_s.stride(3),
            h_k,
            h_q,
            s_k,
            d,
            BLOCK_N=BLOCK_SIZE_M,
        )

    return Q_q, Q_descale, K_q, K_descale, delta_s


@functools.lru_cache(maxsize=16)
def create_hadamard_matrix(block_size, device="cuda", dtype=torch.bfloat16):
    """
    Returns a Hadamard matrix of size block_size x block_size. Remember to normalize with sqrt(block_size) for it to be orthogonal.
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"

    # Base case: H_1 = [1]
    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    # Recursive construction: H_{2n} = [H_n   H_n  ]
    #                                   [H_n  -H_n ]
    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)

    # Build the full matrix (unnormalized)
    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half

    # The unnormalized matrix satisfies H_unnorm @ H_unnorm.T = block_size * I
    # remember to divide by sqrt(block_size) to get orthogonal matrix
    return H


def create_random_hadamard_matrix(block_size, device="cuda", dtype=torch.float32):
    # 1. Generate the deterministic Hadamard matrix (H)
    H = create_hadamard_matrix(block_size, device=device, dtype=dtype) / (
        block_size**0.5
    )
    # 2. Create the random diagonal matrix D (represented as a vector for efficiency)
    # This generates random +1 or -1 for each column
    random_signs = (
        torch.randint(0, 2, (block_size,), device=device, dtype=torch.int) * 2 - 1
    )
    # 3. Apply the random signs (H @ D)
    # Multiplying by a diagonal matrix on the right is equivalent to scaling columns
    H_tilde = H * random_signs
    return H_tilde


SOLATTN_BLOCK_Q = 64        # the solo kernel's Q tile


SOLATTN_BLOCK_KV = 128      # the solo kernel's KV tile


_solattn_stash: dict = {}


_solattn_beta_cache: dict = {}


_solattn_calls: list = [0]      # attention-call counter, for the routing window


@triton.jit
def _solattn_threshold_pack_kernel(
    SHAT, BETA, MASK, NK, NW, NQ,
    ss_row, sm_row,
    BLOCK: tl.constexpr, H: tl.constexpr,
):
    """Row statistics, query-dependent threshold, and bit packing in one launch.

    Deliberately does NOT compute the proxy GEMM. A fused version that did was measured 68% slower
    than handing the GEMM to rocBLAS -- 1176x588x128 per head is small, and a hand-rolled tl.dot at
    the tile sizes this needs cannot match the library. What is worth fusing is the tail: the mean,
    variance, max, compare and bit pack were six torch launches over a 13.8 MB proxy map.

    The row max is tracked so the top block is always selected. An empty row would leave the FMHA
    kernel with a zero-length block list, an L of zero, and a NaN output.
    """
    row = tl.program_id(0)
    beta = tl.load(BETA + (row // NQ) % H)
    base = SHAT + row * ss_row

    ssum = tl.zeros([BLOCK], dtype=tl.float32)
    ssq = tl.zeros([BLOCK], dtype=tl.float32)
    smax = tl.full([BLOCK], float("-inf"), tl.float32)
    for j0 in range(0, NK, BLOCK):
        offs = j0 + tl.arange(0, BLOCK)
        m = offs < NK
        s = tl.load(base + offs, mask=m, other=0.0)
        ssum += tl.where(m, s, 0.0)
        ssq += tl.where(m, s * s, 0.0)
        smax = tl.maximum(smax, tl.where(m, s, float("-inf")))
    mu = tl.sum(ssum) / NK
    var = tl.sum(ssq) / NK - mu * mu
    tau = mu + beta * tl.sqrt(tl.maximum(var, 0.0))
    rmax = tl.max(smax)

    bit = (1 << tl.arange(0, 32)).to(tl.int32)
    for w in range(0, NW):
        offs = w * 32 + tl.arange(0, 32)
        m = offs < NK
        s = tl.load(base + offs, mask=m, other=float("-inf"))
        sel = ((s > tau) | (s >= rmax)) & m
        tl.store(MASK + row * sm_row + w, tl.sum(tl.where(sel, bit, 0)).to(tl.int32))


def _solattn_route_fused(shat, beta, nq, nheads):
    """Threshold + pack a proxy map [rows, NK] into a [rows, NW] int32 bitmask."""
    rows, nk = shat.shape
    nw = (nk + 31) // 32
    nw = nw + (nw & 1)                    # whole 64-bit windows: the FMHA kernel walks them in pairs
    mask = torch.empty((rows, nw), dtype=torch.int32, device=shat.device)
    _solattn_threshold_pack_kernel[(rows,)](
        shat, beta.contiguous(), mask, nk, nw, nq,
        shat.stride(0), mask.stride(0),
        BLOCK=256, H=nheads, num_warps=4,
    )
    return mask


def _solattn_bitmask_mode() -> bool:
    """Emit a bitmask (fused expansion in the kernel) rather than a ragged index list."""
    import os

    return os.environ.get("AITER_SOLATTN_BITMASK", "1") != "0"


def _solattn_fused_mode() -> bool:
    """Use the single-launch Triton router once beta is calibrated."""
    import os

    return os.environ.get("AITER_SOLATTN_FUSED", "1") != "0"


def solattn_take_lut(b, hq, sq):
    """Peek at the block list built by the most recent sage_quant_f4f4_solo call, if it matches.

    Deliberately non-destructive. A SOLATTN_LUT_HBM code object always reads the LUT pointers, so
    handing it a 656-byte kernarg on a repeat launch would send it after uninitialized pointers.
    The stash is cleared by the next quantization call instead.
    """
    if _solattn_stash.get("shape") != (b, hq, sq):
        return None
    return _solattn_stash.get("lut")


def _solattn_maybe_route(q, k, layout, sm_scale):
    """Build a Sol-Attn block list from bf16 Q/K and stash it for the kernel launch.

    AITER_SOLATTN_DENSITY is the target mean density in (0, 1]; >= 1 disables routing. beta is
    calibrated per head to that density, which is what the paper's shared standardized cutoff
    controls. Per-head rather than global because heads differ by ~2.6x in recovered attention mass
    at fixed density on real Wan tensors.
    """
    import os

    target = float(os.environ.get("AITER_SOLATTN_DENSITY", "1"))
    mass_env = float(os.environ.get("AITER_SOLATTN_MASS", "0"))
    _solattn_stash.clear()
    if not (0.0 < target < 1.0) and mass_env <= 0.0:
        return

    # Call window. Diffusion output is far more sensitive to the early denoising steps, where
    # attention is also most diffuse and least worth sparsifying, so the paper leaves the first 20%
    # of steps dense. AITER_SOLATTN_FIRST/LAST bound the attention-call index that gets routed;
    # outside the window this returns with an empty stash and the launch is dense.
    _solattn_stash["calls"] = _solattn_calls[0] = _solattn_calls[0] + 1
    idx = _solattn_calls[0] - 1
    first = int(os.environ.get("AITER_SOLATTN_FIRST", "0"))
    last = int(os.environ.get("AITER_SOLATTN_LAST", "0"))
    if idx < first or (last > 0 and idx >= last):
        return
    if layout == "bhsd":
        b, hq, sq, d = q.shape
        hk, sk = k.shape[1], k.shape[2]
    else:
        b, sq, hq, d = q.shape
        sk, hk = k.shape[1], k.shape[2]
    nk = sk // SOLATTN_BLOCK_KV
    if nk < 2:
        return

    def _pool(x, blk, seq_dim):
        """Block-mean along the sequence axis IN THE NATIVE LAYOUT.

        Pooling used to permute to bhsd and call .float() first, which materializes a ~190 MB fp32
        copy of each of Q and K and dominated the whole routing pass. Reshaping the sequence axis in
        place is a pure view, and mean(dtype=float32) accumulates in fp32 without ever writing the
        upcast tensor.
        """
        n = x.shape[seq_dim] // blk
        shape = list(x.shape)
        shape[seq_dim : seq_dim + 1] = [n, blk]
        pooled = x.narrow(seq_dim, 0, n * blk).reshape(shape).mean(
            dim=seq_dim + 1, dtype=torch.float32
        )
        # -> [b, h, n, d]
        return pooled if seq_dim == 2 else pooled.permute(0, 2, 1, 3)

    seq_dim = 2 if layout == "bhsd" else 1
    if sq % SOLATTN_BLOCK_Q:
        pad = (-sq) % SOLATTN_BLOCK_Q
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad) if seq_dim == 1 else (0, 0, 0, pad))
    qbar = _pool(q, SOLATTN_BLOCK_Q, seq_dim)
    kbar = _pool(k, SOLATTN_BLOCK_KV, seq_dim)
    if hq != hk:
        kbar = kbar.repeat_interleave(hq // hk, dim=1)
    qbar = qbar.contiguous()
    kbar = kbar.contiguous()
    shat = torch.einsum("bhid,bhjd->bhij", qbar, kbar) * sm_scale
    key = (b, hq, sq, sk, round(target, 4), round(mass_env, 4),
           os.environ.get("AITER_SOLATTN_HEAD_DENSITY", ""))
    entry = _solattn_beta_cache.get(key)
    if entry is not None and _solattn_bitmask_mode() and _solattn_fused_mode():
        # Steady state: beta is known, so the whole tail is one Triton launch.
        entry[1] += 1
        nq = shat.shape[2]
        packed = _solattn_route_fused(shat.reshape(-1, shat.shape[-1]), entry[0], nq, hq)
        _solattn_stash["lut"] = (packed, None, None)
        _solattn_stash["shape"] = (b, hq, sq)
        return

    mu = shat.mean(dim=-1, keepdim=True)
    sigma = shat.std(dim=-1, unbiased=False, keepdim=True)

    # beta is a per-head scalar that drifts slowly across layers and denoising steps, so bisecting
    # it on every call is the single most expensive thing routing did (30 passes over the proxy map,
    # ~1.6 ms). Calibrate once per (shape, target) and reuse; AITER_SOLATTN_RECAL forces a refresh
    # every N calls if the drift ever matters.
    period = int(os.environ.get("AITER_SOLATTN_RECAL", "0"))
    stale = entry is None or (period > 0 and entry[1] % period == 0)
    if stale:
        # AITER_SOLATTN_MASS equalizes retained PROXY attention mass across heads instead of
        # density. Heads differ enormously in how concentrated their attention is, so a common
        # density over-spends on the sparse heads and starves the diffuse ones; matching mass
        # instead reaches the same output cosine for 15-18% less work, and it needs nothing the
        # router does not already have -- softmax over the proxy row is the criterion.
        mass_target = float(os.environ.get("AITER_SOLATTN_MASS", "0"))
        if mass_target > 0.0:
            pr = torch.softmax(shat, dim=-1)
            floor = float(os.environ.get("AITER_SOLATTN_DMIN", "0.05"))
            ceil = float(os.environ.get("AITER_SOLATTN_DMAX", "0.60"))
        # AITER_SOLATTN_HEAD_DENSITY gives an explicit per-head budget, for testing whether a
        # non-uniform allocation beats a flat one at matched quality.
        per_head = os.environ.get("AITER_SOLATTN_HEAD_DENSITY", "")
        if per_head:
            tgt_vec = torch.tensor([float(x) for x in per_head.split(",")],
                                   device=shat.device)[:hq]
        else:
            tgt_vec = torch.full((hq,), target, device=shat.device)
        lo = torch.full((hq,), -4.0, device=shat.device)
        hi = torch.full((hq,), 6.0, device=shat.device)
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            sel = shat > mu + mid.view(1, -1, 1, 1) * sigma
            if mass_target > 0.0:
                # bisect on retained mass, but keep the resulting density inside [floor, ceil]
                got = (pr * sel).sum(dim=-1).mean(dim=(0, 2))
                dens = sel.float().mean(dim=(0, 2, 3))
                over = (got > mass_target) | (dens > ceil)
                over = over & ~(dens < floor)
            else:
                over = sel.float().mean(dim=(0, 2, 3)) > tgt_vec
            lo = torch.where(over, mid, lo)
            hi = torch.where(over, hi, mid)
        beta = 0.5 * (lo + hi)
        _solattn_beta_cache[key] = [beta, 1]
    else:
        beta = entry[0]
        entry[1] += 1
    mask = shat > mu + beta.view(1, -1, 1, 1) * sigma
    # Force each row's best block on unconditionally. The guarded form (`if bool(empty.any())`)
    # needed a device-to-host sync every call, which cost more than the block it protects; the
    # top block is above threshold in all but degenerate rows anyway, so the density change is
    # nil and the guarantee the kernel needs -- never an empty list -- still holds.
    mask.scatter_(-1, shat.argmax(dim=-1, keepdim=True), True)

    flat = mask.reshape(-1, nk)
    if _solattn_bitmask_mode():
        # One bit per KV block, 64-bit windows. The kernel expands this into its block list with a
        # lane-mask compaction, which is ~12 instructions per 64 blocks against the 0.170 ms that
        # torch.nonzero costs here -- the single largest item in the routing budget.
        nw64 = (nk + 63) // 64
        pad = nw64 * 64 - nk
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        w = flat.reshape(-1, nw64 * 2, 32).int()
        shifts = (torch.arange(32, device=w.device, dtype=torch.int32)).view(1, 1, 32)
        packed = (w << shifts).sum(dim=2).to(torch.int32).contiguous()
        _solattn_stash["lut"] = (packed, None, None)
    else:
        count = flat.sum(dim=1).to(torch.int32)
        start = torch.zeros_like(count)
        start[1:] = torch.cumsum(count, dim=0)[:-1].to(torch.int32)
        idx = flat.nonzero(as_tuple=True)[1].to(torch.int32).contiguous()
        _solattn_stash["lut"] = (idx, start.contiguous(), count.contiguous())
    _solattn_stash["shape"] = (b, hq, sq)


def sage_quant_f4f4_solo(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ,
    BLKK,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    USE_RNE=False,
    R=None,
    BLOCK_R=32,
):
    """Quantize Q/K/V for the dedicated coalesced f4f4-solo kernel.

    Q uses the same rotation, smoothing, and MXFP4 downcast as
    :func:`sage_quant_f4f4`. K and V use the solo kernel's compact LDS-order
    tile images. Outputs always use bshd logical descriptors:

      * Q: uint8 ``[b, sq, hq, 64]``; Q scale: uint8 ``[b, sq, hq, 4]``.
      * K: uint8 ``[b, sk, hk, 64]``, seq stride 64 and head stride
        ``nT*8192``; K scale: uint8 ``[b, sk, hk, 4]``.
      * V: uint8 ``[b, sk, hk, 128]``, seq stride 64 and head stride
        ``nT*8192``; V scale image: uint8 ``[b, hk, nT*512]``.

    K/V are overlapping strided descriptors over tile images. Consumers must
    pass them through unchanged; making either view contiguous destroys the
    kernel ABI. ``FP8_TYPE``, ``FP8_MAX``, ``BLKK``, and ``USE_RNE`` remain in
    the signature for parity with the other Sage quantizers.
    """
    if layout == "bshd":
        b, sq, hq, head_dim = q.shape
        bk, sk, h_kv, kd = k.shape
        bv, sv, hv, vd = v.shape
    elif layout == "bhsd":
        b, hq, sq, head_dim = q.shape
        bk, h_kv, sk, kd = k.shape
        bv, hv, sv, vd = v.shape
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")

    if head_dim != 128 or kd != 128 or vd != 128:
        raise ValueError(
            f"f4f4-solo requires Q/K/V head_dim=128, got {head_dim}/{kd}/{vd}"
        )
    if (bk, bv) != (b, b) or sv != sk or hv != h_kv:
        raise ValueError(
            "Q/K/V batch, K/V sequence, or K/V head dimensions do not match"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    # Sol-Attn routing (experimental; AITER_SOLATTN_DENSITY unset => untouched dense behaviour).
    # This is the only point in the call chain that still holds bf16 Q/K, which is what the pooled
    # proxy needs, so the block list is built here and handed to the kernel through a per-call stash
    # rather than the return tuple -- callers of this function are pinned and unpack exactly seven
    # values. See SOL_ATTN_PLAN.md; requires a SOLATTN_LUT_HBM code object in the slot.
    _solattn_maybe_route(q, k, layout, sm_scale)

    q_rot, k_rot, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )
    if layout == "bhsd":
        q_rot = q_rot.permute(0, 2, 1, 3)
        k_rot = k_rot.permute(0, 2, 1, 3)
        v_bshd = v.permute(0, 2, 1, 3)
    else:
        v_bshd = v

    q_fp4, q_scale = downcast_to_mxfp(q_rot, torch.uint8, axis=-1)
    k_fp4, k_scale = quantize_f4f4_solo_k(k_rot)
    v_fp4, v_scale = quantize_f4f4_solo_v(v_bshd)
    return q_fp4, q_scale, k_fp4, k_scale, v_fp4, v_scale, delta_s
