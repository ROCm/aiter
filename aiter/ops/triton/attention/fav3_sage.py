# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple, Union
import torch
import aiter
import triton
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    sage_fwd,
)

# The ``map_dims`` layout helper lives in attention/utils.py; the Sparge / VFA
# block-sparse mask, ragged-LUT, and m_init preparation helpers live in
# attention/block_sparse.py. They are re-exported here so existing
# ``from aiter.ops.triton.attention.fav3_sage import ...`` call sites keep working.
from aiter.ops.triton.attention.utils import map_dims  # noqa: F401
from aiter.ops.triton.attention.block_sparse import (  # noqa: F401
    block_attn_mask_to_ragged_lut,
    fill_block_map_triton,
    fill_causal_mask_triton,
    get_block_map_meansim,
    block_attn_mask_to_ragged_lut_topn_front,
    build_attention_lut,
    compute_m_proxy_topn,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant

from aiter.ops.triton.utils._triton import arch_info


def get_sage_fwd_configs():
    arch = arch_info.get_arch()
    if arch == "gfx950":
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "waves_per_eu": 2,
            "PRE_LOAD_V": False,
            "num_stages": 3,
            "num_warps": 8,
        }
    elif arch == "gfx942":
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "waves_per_eu": 2,
            "PRE_LOAD_V": False,
            "num_stages": 2,
            "num_warps": 8,
        }
    else:
        # return tuned config for MI300X by default
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "waves_per_eu": 2,
            "PRE_LOAD_V": False,
            "num_stages": 2,
            "num_warps": 8,
        }


def build_tile_schedule(
    lut_count: torch.Tensor,
    descending: bool = True,
) -> torch.Tensor:
    """Order the block-sparse q-block tiles by their KV-block work for load balance.

    The block-sparse kernel launches one program per
    ``(batch, head, q_block)`` segment, and each program loops over
    ``lut_count[seg]`` KV blocks (``seg = b*(H*Q) + h*Q + q_block``). When the
    per-segment counts are very uneven (e.g. some heads near-dense, others
    sparse) a naive launch leaves heavy tiles running in the tail while light
    tiles have long finished.

    Returns an int32 permutation of segment indices sorted by ``lut_count``
    (descending by default = longest-processing-time-first). Used internally by
    :func:`fav3_sage_func` when ``sparge_load_balancing=True``: the persistent
    atomic work queue hands these out heaviest-first, shrinking the makespan
    tail.
    """
    order = torch.argsort(lut_count.to(torch.int64), descending=descending, stable=True)
    return order.to(torch.int32).contiguous()


class _FAv3SageWrapperFunc(torch.autograd.Function):
    """
    Sage Attention v1 wrapper that maintains high-precision inputs/outputs.

    This wrapper allows users to pass BF16/FP32 tensors and automatically handles
    the quantization internally, maintaining backward compatibility with
    high-precision training workflows.

    Forward: BF16/FP32 -> Int8 (Q & K) + FP16 V -> sage_attn -> FP32 output
    Backward: not supported yet
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        deterministic: bool,
        sm_margin: int,
        return_lse: bool = True,
        layout: str = "bshd",
        config: Optional[dict] = None,
        block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        smooth_k: bool = True,
        freeze_softmax_max_count: int = -1,
        sparge_load_balancing: bool = False,
    ):
        # 1. Dimension Mapping & Config Setup
        bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd_map)
        _, seqlen_k, num_kv_heads, _ = map_dims(k.shape, bshd_map)

        if config is None:
            config = get_sage_fwd_configs()

        BLKQ, BLKK = config["BLOCK_M"], config["BLOCK_N"]
        num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
        num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

        if block_lut is not None:
            kv_block_indices, lut_start, lut_count = block_lut
            use_block_sparse = True
            if causal or window_size != (-1, -1):
                raise NotImplementedError(
                    "The Triton block-sparse attention path selected by block_lut "
                    "does not support causal or sliding-window masking; "
                    "require causal=False and window_size=(-1, -1)."
                )
        else:
            kv_block_indices = lut_start = lut_count = None
            use_block_sparse = False

        # 2. Validation: Early Exit for unsupported features
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0 or sm_margin != 0:
            raise NotImplementedError(
                "softcap/sm_margin not supported in FP8 high-precision API"
            )

        if (q.requires_grad or k.requires_grad or v.requires_grad) and not return_lse:
            raise ValueError(
                "return_lse must be True during training (requires_grad=True)"
            )

        # 3. Quantization
        # Note: softmax_scale is integrated into quantization descaling
        softmax_scale = softmax_scale or (head_dim**-0.5)
        fp8_dtype = aiter.dtypes.fp8
        fp8_max = torch.finfo(fp8_dtype).max

        sq_result = sage_quant(
            q,
            k,
            v,
            fp8_dtype,
            fp8_max,
            sm_scale=softmax_scale,
            BLKQ=BLKQ,
            BLKK=BLKK,
            layout=layout,
            smooth_k=smooth_k,
            return_lse=return_lse,
        )
        if return_lse:
            q_int8, q_descale, k_int8, k_descale, v_fp8, v_descale, sage_lse_delta = (
                sq_result
            )
        else:
            q_int8, q_descale, k_int8, k_descale, v_fp8, v_descale = sq_result
            sage_lse_delta = None

        # 4. Verify Descale Shapes (Grouped scaling for GQA/MQA)
        num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
        num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

        expected_q_ds = (batch, num_q_heads, num_q_blocks)
        expected_k_ds = (batch, num_kv_heads, num_k_blocks)

        assert (
            q_descale.shape == expected_q_ds
        ), f"q_descale shape {q_descale.shape} != {expected_q_ds}"
        assert (
            k_descale.shape == expected_k_ds
        ), f"k_descale shape {k_descale.shape} != {expected_k_ds}"

        # 5. Execution
        out, softmax_lse = fav3_sage_func(
            q_int8,
            k_int8,
            v_fp8,
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            window_size,
            attention_chunk,
            softcap,
            sm_margin,
            return_lse,
            layout,
            config,
            kv_block_indices=kv_block_indices,
            lut_start=lut_start,
            lut_count=lut_count,
            use_block_sparse=use_block_sparse,
            freeze_softmax_max_count=freeze_softmax_max_count,
            sparge_load_balancing=sparge_load_balancing,
        )

        if return_lse:
            # Recover the un-smoothed LSE. The kernel computed the LSE
            # against (K - k_mean); adding delta = sm_scale * Q . k_mean^T
            # shifts it back so it is consistent with a kernel call on the
            # un-smoothed K (required for correct ring-attention merging).
            if sage_lse_delta is not None:
                softmax_lse = softmax_lse + sage_lse_delta.to(softmax_lse.dtype)
            return out, softmax_lse

        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        return (
            None,  # q
            None,  # k
            None,  # v
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # deterministic
            None,  # sm_margin
            None,  # return_lse
            None,  # layout
            None,  # config
            None,  # block_lut
            None,  # smooth_k
            None,  # freeze_softmax_max_count
            None,  # sparge_load_balancing
        )


def fav3_sage_wrapper_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    return_lse: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    smooth_k: bool = True,
    freeze_softmax_max_count: int = -1,
    sparge_load_balancing: bool = False,
):
    """
    SageAttention v1 high-precision entry point.

    This function accepts high-precision (BF16/FP32) tensors and internally
    quantizes them to Int8/BF16 for computation. The output and gradients remain
    in high precision (FP32 for output, input dtype for gradients).

    This API is designed for seamless integration with existing training code
    that uses BF16/FP32 tensors, providing FP8 acceleration without requiring
    manual quantization.

    Args:
        q: Query tensor [batch, seqlen, num_q_heads, head_dim] (BF16/FP32)
        k: Key tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP32)
        v: Value tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP32)
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Sliding window attention size (left, right)
        attention_chunk: Chunking parameter (0 or 1 only)
        softcap: Softcapping value (not yet supported)
        deterministic: Whether to use deterministic backward (not yet supported)
        sm_margin: SM margin parameter (not yet supported)
        return_lse: return softmax_lse if True, otherwise return None
        layout: bshd or bhsd layout for the inputs
        config: Optional kernel configuration dict with keys BLOCK_M, BLOCK_N,
                waves_per_eu, PRE_LOAD_V, num_stages, num_warps
        block_lut: Optional ragged LUT for block-sparse attention,
                (kv_block_indices, lut_start, lut_count) from block_attn_mask_to_ragged_lut.
                When None, dense attention is used.
        smooth_k: Whether to apply k-smoothing to the K tensor
        freeze_softmax_max_count: number of inner-loop K-block iterations after
                which the online-softmax running max is frozen (block-sparse only;
                -1 disables). See fav3_sage_func / build_attention_lut.
        sparge_load_balancing: when True (block-sparse only), enable the
                persistent + atomic longest-processing-time tile schedule to
                balance work across CUs. See fav3_sage_func for details.

    Returns:
        out: Output tensor [batch, seqlen, num_q_heads, head_dim] or [batch, num_q_heads, seqlen, head_dim] (FP32)

    Note:
        - Supports GQA/MQA (num_q_heads != num_kv_heads)
        - Automatically handles grouped quantization for GQA/MQA queries
        - backward is not yet supported
        - softcap is not yet supported in FP8 mode
    """

    # Check that inputs are high precision
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"sage_attn_v1_func expects high-precision inputs (fp16/bf16/fp32), got q.dtype={q.dtype}. "
        f"If you already have Int8 tensors, use sage_attn_v1_func() with q_descale/k_descale parameters instead."
    )
    assert k.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"sage_attn_v1_func expects high-precision inputs (fp16/bf16/fp32), got k.dtype={k.dtype}. "
        f"If you already have Int8 tensors, use sage_attn_v1_func() with q_descale/k_descale parameters instead."
    )
    assert v.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], f"sage_attn_v1_func expects high-precision inputs (fp16/bf16/fp32), got v.dtype={v.dtype}. "

    if sm_margin != 0:
        raise NotImplementedError(
            "sm_margin != 0 not supported in Sage Attention v1 API"
        )

    return _FAv3SageWrapperFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        sm_margin,
        return_lse,
        layout,
        config,
        block_lut,
        smooth_k,
        freeze_softmax_max_count,
        sparge_load_balancing,
    )


def fav3_sage_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    sm_margin: int = 0,
    return_lse: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
    kv_block_indices: Optional[torch.Tensor] = None,
    lut_start: Optional[torch.Tensor] = None,
    lut_count: Optional[torch.Tensor] = None,
    use_block_sparse: bool = False,
    freeze_softmax_max_count: int = -1,
    sparge_load_balancing: bool = False,
    m_init: Optional[torch.Tensor] = None,
):
    """
    SageAttention v1.

    Args:
        q: Query tensor [batch, seqlen, num_q_heads, head_dim] (int8)
        k: Key tensor [batch, seqlen, num_kv_heads, head_dim] (int8)
        v: Value tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP16)
        q_descale: Descale factors for Q (float32)
        k_descale: Descale factors for K (float32)
        v_descale: Descale factors for V (float32)
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Sliding window attention size (left, right)
        attention_chunk: Chunking parameter (0 or 1 only)
        softcap: Softcapping value (not yet supported)
        sm_margin: SM margin parameter (not yet supported)
        return_lse: return softmax_lse if True, otherwise return None
        layout: bshd or bhsd layout for the inputs
        config: Optional kernel configuration dict with keys BLOCK_M, BLOCK_N,
                waves_per_eu, PRE_LOAD_V, num_stages, num_warps
        kv_block_indices: Optional ragged LUT for block-sparse attention.
        lut_start: Optional start index for the ragged LUT
        lut_count: Optional count of the ragged LUT
        use_block_sparse: Whether to use block-sparse attention
        freeze_softmax_max_count: number of inner-loop K-block iterations after
                which the online-softmax running max stops being updated. Once
                frozen, the kernel skips the per-block max reduction and the acc
                rescale, computing ``p = exp(qk - m)``, ``l += rowsum(p)`` and
                ``acc += p @ v`` with ``m`` held fixed (VFA-style). ``-1``
                (default) disables freezing and keeps the exact online softmax.
                Only takes effect on the block-sparse path.
        m_init: optional precomputed per-row running-max estimate (VFA), fp32
                ``[batch, nheads_q, num_q_blocks, BLKQ]``. When provided the
                kernel skips the online-softmax rowmax reduction and the acc
                rescale entirely (frozen max from the first block), computing
                ``p = exp2(qk - m_init)``, ``l += rowsum(p)`` and
                ``acc += p @ v``. Works for the dense non-causal path and the
                block-sparse path. Mutually exclusive with causal/sliding-window
                masking and with ``freeze_softmax_max_count``. See
                :func:`compute_m_proxy_topn` for the estimator. ``None``
                (default) runs the exact online softmax.

    Returns:
        out: Output tensor [batch, seqlen, num_q_heads, head_dim] or [batch, num_q_heads, seqlen, head_dim] (FP32)
    """

    # --- 1. Layout & Dimension Mapping ---
    # bshd: [0,1,2,3], bhsd: [0,2,1,3]
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    batch, seqlen_q, nheads_q, head_size_qk = map_dims(q.shape, bshd_map)
    _, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd_map)
    _, seqlen_v, nheads_v, head_size_v = map_dims(v.shape, bshd_map)

    # --- 2. Feature & Input Validation ---
    if attention_chunk not in (0, 1) or softcap != 0.0 or sm_margin != 0:
        raise NotImplementedError(
            "Feature (chunking/softcap/sm_margin) not supported in this API."
        )

    assert q.dtype == torch.int8 and k.dtype == torch.int8, "Q and K must be int8"
    assert seqlen_k == seqlen_v, f"K/V seqlen mismatch: {seqlen_k} vs {seqlen_v}"
    assert nheads_k == nheads_v, f"K/V head mismatch: {nheads_k} vs {nheads_v}"
    assert (
        nheads_q % nheads_k == 0
    ), f"GQA/MQA error: {nheads_q} not divisible by {nheads_k}"

    # --- 3. Configuration & Descale Setup ---
    if config is None:
        config = get_sage_fwd_configs()

    BLKQ, BLKK = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
    num_k_blocks = (seqlen_k + BLKK - 1) // BLKK

    assert q_descale.shape == (batch, nheads_q, num_q_blocks)
    assert k_descale.shape == (batch, nheads_k, num_k_blocks)

    # --- 4. Output Allocation ---
    out_dtype = torch.bfloat16
    if layout == "thd":
        out = torch.zeros(
            (q.shape[0], q.shape[1], v.shape[-1]), dtype=out_dtype, device=q.device
        )
        softmax_lse = (
            torch.zeros((nheads_q, q.shape[0]), device=q.device, dtype=torch.float32)
            if return_lse
            else None
        )
    else:
        out_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
        out = torch.zeros(out_shape, dtype=out_dtype, device=q.device)
        softmax_lse = (
            torch.zeros(
                (batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32
            )
            if return_lse
            else None
        )

    # --- 5. Stride Extraction ---
    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd_map)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd_map)
    stride_vb, stride_vn, stride_vh, stride_vd = map_dims(v.stride(), bshd_map)
    stride_ob, stride_om, stride_oh, stride_od = map_dims(out.stride(), bshd_map)

    stride_lse_z, stride_lse_h, stride_lse_m = (
        softmax_lse.stride() if return_lse else (0, 0, 0)
    )
    stride_qsz, stride_qsh, stride_qsblk = q_descale.stride()
    stride_ksz, stride_ksh, stride_ksblk = k_descale.stride()
    stride_vsz, stride_vsh, _ = v_descale.stride()

    # --- 6. Padding & Metadata ---
    padded_d_model_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_model_v = max(16, 1 << (head_size_v - 1).bit_length())

    window_size_left, window_size_right = int(window_size[0]), int(window_size[1])
    use_sliding_window = window_size_left != -1 or window_size_right != -1

    if use_block_sparse and use_sliding_window:
        raise NotImplementedError(
            "Sliding window and block-sparse attention cannot be enabled "
            "together; set window_size=(-1, -1) when use_block_sparse=True."
        )

    if freeze_softmax_max_count >= 0 and not use_block_sparse:
        raise ValueError(
            "freeze_softmax_max_count is only meaningful with "
            "use_block_sparse=True; leave it at -1 (disabled) otherwise."
        )

    # VFA: a precomputed frozen running-max estimate. Incompatible with the
    # masking paths (causal/sliding window) and with the warm-up freeze, which
    # derive their own running max.
    use_precomputed_max = m_init is not None
    if use_precomputed_max:
        if causal or use_sliding_window:
            raise NotImplementedError(
                "m_init (VFA precomputed max) does not support causal or "
                "sliding-window masking; require causal=False and "
                "window_size=(-1, -1)."
            )
        if freeze_softmax_max_count >= 0:
            raise ValueError(
                "m_init (VFA precomputed max) and freeze_softmax_max_count are "
                "mutually exclusive; pass only one."
            )
        assert m_init.dtype == torch.float32, "m_init must be fp32"
        assert m_init.shape == (batch, nheads_q, num_q_blocks, BLKQ), (
            f"m_init shape {tuple(m_init.shape)} does not match expected "
            f"{(batch, nheads_q, num_q_blocks, BLKQ)}"
        )
        stride_mz, stride_mh, stride_mblk, stride_mr = m_init.stride()
    else:
        m_init = torch.zeros(1, dtype=torch.float32, device=q.device)
        stride_mz = stride_mh = stride_mblk = stride_mr = 0

    if use_block_sparse:
        if kv_block_indices is None or lut_start is None or lut_count is None:
            raise ValueError(
                "kv_block_indices, lut_start, and lut_count must be provided "
                "when use_block_sparse=True"
            )
        if causal:
            raise NotImplementedError(
                "The Triton block-sparse attention path selected by block_lut "
                "does not support causal masking."
                "require causal=False."
            )
    else:
        kv_block_indices = torch.zeros(1, dtype=torch.int32, device=q.device)
        lut_start = torch.zeros(1, dtype=torch.int32, device=q.device)
        lut_count = torch.zeros(1, dtype=torch.int32, device=q.device)

    if sparge_load_balancing and not use_block_sparse:
        raise ValueError(
            "sparge_load_balancing=True is only supported on the block-sparse "
            "path (use_block_sparse=True)."
        )

    # Sparge load balancing: sort the (batch, head, q_block) tiles by descending
    # KV-block count (longest-processing-time first) and run them through a
    # persistent kernel whose resident programs share a global atomic work queue.
    # This evens out the makespan when per-q-block KV-block counts are very
    # uneven (e.g. some heads near-dense, others sparse).
    use_tile_schedule = bool(sparge_load_balancing)
    tile_schedule = None
    tile_counter = None
    n_tiles = 0
    if use_tile_schedule:
        tile_schedule = build_tile_schedule(lut_count)
        n_tiles = tile_schedule.numel()
        # Atomic work-queue counter; must start at zero on every launch.
        tile_counter = torch.zeros(1, dtype=torch.int32, device=q.device)
        # One resident workgroup per CU saturates this register-heavy kernel
        # (occupancy ~1); surplus programs just find the queue drained and exit.
        num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_programs = min(n_tiles, num_cus)

    # --- 7. Kernel Launch ---
    if use_tile_schedule:

        def grid(META):
            return (num_programs,)

    else:

        def grid(META):
            return (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)

    sage_fwd[grid](
        q,
        k,
        v,
        None,
        q_descale,
        k_descale,
        v_descale,
        stride_qsz,
        stride_qsh,
        stride_qsblk,
        stride_ksz,
        stride_ksh,
        stride_ksblk,
        stride_vsz,
        stride_vsh,
        m_init,
        stride_mz,
        stride_mh,
        stride_mblk,
        stride_mr,
        softmax_lse,
        out,
        None,
        None,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        0,
        0,
        0,
        0,  # stride_bz, stride_bh, stride_bm, stride_bn
        0,
        0,  # stride_az, stride_ah
        0,
        0,
        0,
        0,  # stride_sz, stride_sh, stride_sm, stride_sn
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        None,
        None,
        None,
        None,
        kv_block_indices,
        lut_start,
        lut_count,
        num_q_blocks,
        dropout_p=0.0,
        philox_seed=None,
        philox_offset_base=None,
        RETURN_LSE=return_lse,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        IS_CAUSAL=causal,
        USE_SLIDING_WINDOW=use_sliding_window,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        IS_VARLEN=False,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False,
        USE_ALIBI=False,
        ENABLE_DROPOUT=False,
        USE_EXP2=True,
        RETURN_SCORES=False,
        USE_SEQUSED=False,
        USE_BLOCK_SPARSE=use_block_sparse,
        FREEZE_SOFTMAX_MAX_COUNT=freeze_softmax_max_count,
        USE_PRECOMPUTED_MAX=use_precomputed_max,
        TILE_SCHEDULE=tile_schedule,
        TILE_COUNTER=tile_counter,
        NUM_TILES=n_tiles,
        USE_TILE_SCHEDULE=use_tile_schedule,
        **config,
    )

    if return_lse:
        return out, softmax_lse
    else:
        return out, None


def fav3_sage_vfa_wrapper_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    layout: str = "bshd",
    config: Optional[dict] = None,
    n_sample_blocks: int = 16,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """High-precision VFA API: handles quantization and the ``m_init`` estimate.

    Thin convenience wrapper around :func:`fav3_sage_func`. It quantizes the
    high-precision Q/K/V, builds the per-row frozen-max estimate ``m_init``, and
    runs the (now unified) sage kernel with the VFA precomputed-max path.

    The per-row frozen-max estimate ``m_init`` is the top-``n_sample_blocks``
    blocks per q-block ranked by a SpargeAttn mean-pooled block-score and then
    evaluated with real K rows (a lower bound on the true per-row max; no safety
    margin). See :func:`compute_m_proxy_topn`.

    When ``block_lut`` (a ragged ``(kv_block_indices, lut_start, lut_count)``
    LUT from :func:`block_attn_mask_to_ragged_lut`) is provided, the
    block-sparse path runs (the hot loop visits only attended K blocks); the
    same guided ``m_init`` estimate is used.
    """
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert k.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float32]

    if config is None:
        config = get_sage_fwd_configs()

    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    _, _, _, head_dim = map_dims(q.shape, bshd_map)
    softmax_scale = softmax_scale or (head_dim ** -0.5)

    BLKQ, BLKK = config["BLOCK_M"], config["BLOCK_N"]
    fp8_dtype = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8_dtype).max

    q_int8, q_descale, k_int8, k_descale, v_fp8, v_descale = sage_quant(
        q, k, v,
        fp8_dtype, fp8_max,
        sm_scale=softmax_scale,
        BLKQ=BLKQ,
        BLKK=BLKK,
        layout=layout,
    )

    use_block_sparse = block_lut is not None
    if use_block_sparse:
        kv_block_indices, lut_start, lut_count = block_lut
    else:
        kv_block_indices = lut_start = lut_count = None

    m_init = compute_m_proxy_topn(
        q, k, q_int8, k_int8, q_descale, k_descale,
        BLKQ=BLKQ, BLKK=BLKK, layout=layout,
        n_blocks=n_sample_blocks,
    )

    out, lse = fav3_sage_func(
        q_int8, k_int8, v_fp8,
        q_descale, k_descale, v_descale,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        layout=layout,
        config=config,
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
        use_block_sparse=use_block_sparse,
        m_init=m_init,
    )
    if return_lse:
        return out, lse
    return out
