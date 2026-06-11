# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple, Union
import torch
import aiter
import triton
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    sage_fwd,
    map_dims,
    triton_bmm_pool_sim_simmean,
    triton_fill_block_map_kernel,
    triton_fill_causal_mask,
)
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
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


def get_pool_sim_triton_simmean(
    x: torch.Tensor,
    block_size: int,
    simthreshd1: torch.Tensor,
    attention_scored_only: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mean-pool each block and flag internally-similar blocks.

    Args:
        x: ``(B, H, N, D)`` tensor.
        block_size: number of tokens per block.
        simthreshd1: ``(H,)`` per-head similarity threshold.
        attention_scored_only: when ``True``, skip the intra-block similarity
            test entirely and return ``None`` for ``sim_blocks``.

    Steps:
        1. Pool (mean) within each block.
        2. Compute the mean pairwise cosine similarity within each block.
        3. Flag blocks whose mean self-similarity exceeds ``simthreshd1``.

    Note how the 3rd dimension ``N`` is reduced to ``nblock = N // block_size``;
    this keeps the downstream block-attention proxy at ``O(nblock^2)`` instead
    of the full ``O(N^2)``.

    Returns:
        pool: ``(B, H, nblock, D)`` tensor.
        sim_blocks: ``(B, H, nblock)`` bool tensor, or ``None`` when
            ``attention_scored_only`` is set.
    """
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    if attention_scored_only:
        sim_blocks = None
        # The kernel needs a valid pointer; pass `pool` as an unused placeholder.
        sim_arg = pool
    else:
        sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
        sim_arg = sim_blocks
    grid = (B, H, nblock)
    triton_bmm_pool_sim_simmean[grid](
        x, pool, sim_arg, simthreshd1, N=N, D=D, BS=block_size, SKIP_SIM=attention_scored_only
    )
    return pool, sim_blocks


def fill_block_map_triton(
    final_map: torch.Tensor,
    num_to_select: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """Scatter the top-``num_to_select`` ranked K blocks per (B, H, Q) into ``final_map``."""
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map


def fill_causal_mask_triton(mask: torch.Tensor, BqdivBk: float) -> torch.Tensor:
    """Fill a 2-D ``(nq, nk)`` block-level causal mask for a Q/K block-size ratio."""
    assert mask.dim() == 2
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


def get_block_map_meansim(
    q: torch.Tensor,
    k: torch.Tensor,
    is_causal: bool = False,
    BLKQ: int = 64,
    BLKK: int = 64,
    simthreshd1: float = 0.1,
    cdfthreshd: float = 0.9,
    attention_sink: bool = False,
    attention_scored_only: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Build a SpargeAttn block-sparse mask via the mean-similarity proxy.

    Mean-pools Q/K into per-block representatives, forms the block-level score
    ``pooled_q @ pooled_k^T * d**-0.5``, softmaxes over key blocks, and keeps the
    smallest set of top key blocks whose cumulative probability reaches
    ``cdfthreshd`` per query block.  Blocks that fail the intra-block similarity
    test are forced on (their mean is not a faithful summary, so they must be
    evaluated exactly).

    Args:
        attention_scored_only: when ``True``, skip the intra-block similarity
            test and block-selection logic entirely, returning
            ``(None, pooled_score)`` where ``pooled_score`` is the raw
            block-level score ``pooled_q @ pooled_k^T * d**-0.5``.

    Returns:
        A ``(final_map, pooled_score)`` tuple.  ``final_map`` is a
        ``(B, H, num_q_blocks, num_k_blocks)`` bool mask (or ``None`` when
        ``attention_scored_only`` is set), and ``pooled_score`` is the
        block-level score with positions that ``final_map`` masks out set to
        ``-inf``.
    """
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_q, sim_q = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1, attention_scored_only)
    pooled_k, sim_k = get_pool_sim_triton_simmean(k, BLKK, simthreshd1, attention_scored_only)
    pooled_score = pooled_q @ pooled_k.transpose(-1, -2) * q.shape[-1] ** -0.5
    if attention_scored_only:
        return None, pooled_score

    neg_inf = pooled_score.new_full((), float("-inf"))
    sim_k = sim_k.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_q = sim_q.unsqueeze(-1).expand(-1, -1, -1, nk)

    prob = torch.where(sim_k, pooled_score, neg_inf)
    causal_mask = None
    if is_causal:
        causal_mask = fill_causal_mask_triton(
            torch.empty(nq, nk, device=q.device, dtype=torch.bool), BLKQ / BLKK
        )
        prob = torch.where(causal_mask[None, None], prob, neg_inf)
    prob = prob.softmax(-1)

    # Keep the smallest set of top key blocks whose cumulative mass reaches cdfthreshd.
    sorted_score = torch.sort(prob, dim=-1, descending=True)
    cdf = sorted_score.values.cumsum(dim=-1)
    H, K = cdf.shape[1], cdf.shape[-1]
    ge = cdf >= cdfthreshd.view(1, H, 1, 1)
    idx = ge.to(torch.uint8).argmax(dim=-1)
    num_to_select = torch.where(ge.any(dim=-1), idx, idx.new_full((), K))

    final_map = fill_block_map_triton(
        torch.zeros_like(prob, dtype=torch.bool), num_to_select, sorted_score.indices
    )
    final_map = final_map | ~sim_k | ~sim_q
    if is_causal:
        final_map = final_map * causal_mask[None, None]
    if attention_sink:
        final_map[:, :, :, 0] = 1
    return final_map, pooled_score


def _num_text_blocks(text_len: int, block_m: int, block_n: int) -> Tuple[int, int]:
    """Number of (q, k) blocks spanned by ``text_len`` trailing text tokens."""
    return (
        (text_len + block_m - 1) // block_m,
        (text_len + block_n - 1) // block_n,
    )


def _assemble_full_block_mask(
    image_block_mask: torch.Tensor,
    image_len_q: int,
    image_len_k: int,
    text_len: int,
    block_m: int,
    block_n: int,
) -> torch.Tensor:
    """Append dense text rows/columns to an image-only block mask.

    Returns the full ``(B, H, n_iq + n_text_q, n_ik + n_text_k)`` mask in which
    all Q rows attend to the text K columns, all text Q rows attend to
    everything, and any partial image/text boundary block is forced dense (so
    spillover tokens are never dropped).  A no-op when ``text_len == 0``.
    """
    if text_len == 0:
        return image_block_mask

    B, H, n_iq, n_ik = image_block_mask.shape
    n_text_q, n_text_k = _num_text_blocks(text_len, block_m, block_n)

    full = torch.zeros(
        (B, H, n_iq + n_text_q, n_ik + n_text_k),
        dtype=image_block_mask.dtype,
        device=image_block_mask.device,
    )
    full[:, :, :n_iq, :n_ik] = image_block_mask
    full[:, :, :, -n_text_k:] = True  # every Q row attends to text K cols
    full[:, :, -n_text_q:, :] = True  # text Q rows attend to everything
    if image_len_q % block_m != 0:  # partial boundary blocks -> dense
        full[:, :, image_len_q // block_m, :] = True
    if image_len_k % block_n != 0:
        full[:, :, :, image_len_k // block_n] = True
    return full


def block_attn_mask_to_ragged_lut_topn_front(
    block_attn_mask: torch.Tensor,
    pooled_score: torch.Tensor,
    sample_n: int,
    num_heads: Optional[int] = None,
    force_front_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a ragged LUT with the top-``sample_n`` scored K blocks emitted first.

    Like :func:`aiter.ops.triton.attention.utils.block_attn_mask_to_ragged_lut`,
    this turns a block attention mask into a ragged
    ``(kv_block_indices, lut_start, lut_count)`` LUT, but within each
    ``(batch, head, q_block)`` segment it emits blocks in this order:

      1. the ``sample_n`` highest ``pooled_score`` blocks, descending score;
      2. the ``force_front_mask`` blocks (e.g. text), ascending block index;
      3. the remaining attended blocks, ascending block index.

    Building the LUT directly from the mask lets us write each segment out in the
    desired order in a single pass -- no separate reorder of a pre-built LUT. The
    valid entries are compactly packed into the prefix of ``kv_block_indices``
    (indexed by ``lut_start``/``lut_count``); the buffer itself is over-allocated
    to a static size so the whole function is CUDA-graph-safe (no data-dependent
    shapes or device syncs), matching
    :func:`aiter.ops.triton.attention.utils.block_attn_mask_to_ragged_lut`.

    This pairs with the ``freeze_softmax_max_count`` block-sparse path
    (:func:`fav3_sage_func`): the online-softmax running max is frozen after the
    first few inner-loop iterations, so visiting the highest-scoring (and thus
    likely highest-max) tiles -- plus any always-attended ``force_front`` tiles --
    first makes the frozen max a tight estimate. See ``fav3_sage_vfa.py`` for the
    analogous pooled-score top-N block selection.

    Args:
        block_attn_mask: ``(B, num_q_blocks, num_k_blocks)`` (shared across heads)
            or ``(B, H, num_q_blocks, num_k_blocks)`` bool mask. True = attend.
        pooled_score: fp32 ``(B, H, num_q_blocks, num_k_blocks)`` block-level
            attention score (e.g. from :func:`get_block_map_meansim`). Only blocks
            that are attended in ``block_attn_mask`` are ever selected.
        sample_n: number of top-scored tiles to emit first per segment. ``<= 0``
            emits only the ``force_front`` tiles ahead of the rest.
        num_heads: number of Q heads; required when ``block_attn_mask`` is 3D.
        force_front_mask: optional bool mask, same shape/broadcast as
            ``block_attn_mask``, of blocks to place immediately after the sampled
            tiles. Excluded from the top-``sample_n`` selection; only the
            attended ones are emitted.

    Returns:
        ``kv_block_indices`` (1D int32), ``lut_start`` (1D int32) and
        ``lut_count`` (1D int32), indexed by
        ``idx = b * (H * num_q_blocks) + h * num_q_blocks + q_block``.
        ``kv_block_indices`` is over-allocated to the static size ``B*H*Q*K``;
        only the compact prefix selected by ``lut_start``/``lut_count`` is valid
        (the remainder is unused padding).
    """
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        B, Q, K = block_attn_mask.shape
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(B, num_heads, Q, K)
        if force_front_mask is not None and force_front_mask.dim() == 3:
            force_front_mask = force_front_mask.unsqueeze(1).expand(B, num_heads, Q, K)

    B, H, Q, K = block_attn_mask.shape
    assert pooled_score.shape[:3] == (B, H, Q) and pooled_score.shape[-1] == K, (
        f"pooled_score shape {tuple(pooled_score.shape)} does not match mask "
        f"{(B, H, Q, K)}"
    )
    device = block_attn_mask.device

    attended = block_attn_mask.to(torch.bool)
    lut_count = attended.sum(-1).to(torch.int32).reshape(-1)
    lut_start = torch.cumsum(lut_count, 0) - lut_count

    if force_front_mask is None:
        force_front = torch.zeros_like(attended)
    else:
        # Only attended blocks can be emitted at all.
        force_front = force_front_mask.to(torch.bool).expand(B, H, Q, K) & attended

    neg_inf = pooled_score.new_full((), float("-inf"))
    masked_score = torch.where(attended, pooled_score.to(torch.float32), neg_inf)

    # Mark the top-``sample_n`` attended, non-force-front blocks per (B, H, Q) row.
    is_topn = torch.zeros((B, H, Q, K), dtype=torch.bool, device=device)
    n = min(sample_n, K)
    if n > 0:
        sample_score = torch.where(force_front, neg_inf, masked_score)
        topk = sample_score.topk(n, dim=-1)
        # A row with fewer than n candidates pads topk with -inf entries; mark
        # only the finite (genuinely attended, non-force-front) selections.
        is_topn.scatter_(-1, topk.indices, topk.values > neg_inf)

    # Per-row ordering of the K blocks by (priority, tiebreak):
    #   0 = attended & top-n      -> descending score (highest-max first)
    #   1 = attended & force-front -> ascending block index
    #   2 = attended, the rest    -> ascending block index
    #   3 = not attended          -> sorts past the per-row count, so dropped
    col = torch.arange(K, device=device).view(1, 1, 1, K)
    priority = torch.where(
        ~attended,
        3,
        torch.where(is_topn, 0, torch.where(force_front, 1, 2)),
    )
    tiebreak = torch.where(is_topn, -masked_score, col.to(torch.float32))

    # Lexicographic (priority, tiebreak) sort per row via two stable sorts.
    o1 = torch.argsort(tiebreak, dim=-1, stable=True)
    order = torch.gather(
        o1, -1, torch.argsort(torch.gather(priority, -1, o1), dim=-1, stable=True)
    )

    # The first ``count`` entries of each row are exactly the attended blocks in
    # the desired order; pack them row-major into the ragged index list. Done as
    # a scatter into an over-allocated, statically-sized buffer (rather than
    # boolean-mask indexing, which triggers a ``nonzero`` device sync and a
    # data-dependent output shape) so the function stays CUDA-graph-safe.
    R = B * H * Q
    rows = order.reshape(R, K)
    col = torch.arange(K, device=device)
    keep = col[None, :] < lut_count[:, None]
    # Destination of each entry in the packed buffer. Dropped (unattended)
    # entries are routed to a scratch sink slot so they can never clobber a
    # valid neighbouring row's position.
    sink = R * K
    dest = torch.where(
        keep,
        lut_start[:, None].to(torch.long) + col[None, :],
        col.new_full((), sink),
    )
    packed = torch.empty(sink + 1, dtype=torch.int32, device=device)
    packed.scatter_(0, dest.reshape(-1), rows.reshape(-1).to(torch.int32))
    kv_block_indices = packed[:sink]
    return kv_block_indices, lut_start, lut_count


def build_attention_lut(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    simthreshd1: float,
    cdfthreshd: float,
    mode: str = "both",
    n_sample: int = 8,
    is_causal: bool = False,
    static_block_mask: Optional[torch.Tensor] = None,
    text_len: int = 0,
    block_m: int = 128,
    block_n: int = 128,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Build a block-sparse ragged LUT for SpargeAttn and/or VFA.

    One entry point that turns Q/K into a ready-to-use ragged LUT plus the
    matching ``freeze_softmax_max_count``. ``mode`` selects the block set and
    ordering:

      * ``"sparge"``: keep only the SpargeAttn-selected (meansim) blocks and emit
        them in ascending order. ``freeze_softmax_max_count = -1`` (the online
        softmax max is never frozen); ``n_sample`` is ignored.
      * ``"vfa"``: keep *all* K blocks (block-causal when ``is_causal``) but
        front-load the top-``n_sample`` blocks by pooled score, so the frozen max
        sees the most important tiles first.
      * ``"both"``: keep only the SpargeAttn-selected blocks *and* front-load the
        top-``n_sample`` of them by pooled score.

    Text handling (``text_len > 0``): every q-block additionally attends to the
    trailing dense text K blocks. For ``"vfa"``/``"both"`` those text blocks are
    appended to the front immediately *after* the ``n_sample`` sampled tiles, and
    the returned ``freeze_softmax_max_count`` is ``n_sample + n_text_blocks`` so
    the max is frozen only once both the sampled and text tiles are visited.

    Causality is handled at block granularity (whole K blocks are kept/dropped);
    the block-sparse kernel does not apply an intra-block diagonal mask.

    Args mirror :func:`compute_sparge_block_mask`, plus ``mode`` and ``n_sample``.

    Returns:
        ``(block_lut, freeze_softmax_max_count)`` where ``block_lut`` is the
        ragged ``(kv_block_indices, lut_start, lut_count)`` tuple to pass as
        ``block_lut`` to :func:`fav3_sage_wrapper_func` (or
        ``kv_block_indices``/``lut_start``/``lut_count`` to :func:`fav3_sage_func`
        with ``use_block_sparse=True`` and ``freeze_softmax_max_count=...``).
    """
    if mode not in ("vfa", "sparge", "both"):
        raise ValueError(f"mode must be 'vfa', 'sparge' or 'both', got {mode!r}")

    image_q = q[:, :, : q.shape[2] - text_len, :] if text_len > 0 else q
    image_k = k[:, :, : k.shape[2] - text_len, :] if text_len > 0 else k
    image_len_q = q.shape[2] - text_len
    image_len_k = k.shape[2] - text_len
    n_text_k = _num_text_blocks(text_len, block_m, block_n)[1] if text_len > 0 else 0

    # "vfa" ranks every block (no sparsity selection), so it only needs the
    # pooled score; "sparge"/"both" also need the meansim sparsity mask.
    image_mask, image_score = get_block_map_meansim(
        image_q,
        image_k,
        is_causal=is_causal,
        BLKQ=block_m,
        BLKK=block_n,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        attention_scored_only=(mode == "vfa"),
    )

    if mode == "vfa":
        # Keep all blocks (block-causal when requested) as sampling candidates.
        B, H, n_iq, n_ik = image_score.shape
        if is_causal:
            causal = fill_causal_mask_triton(
                torch.empty(n_iq, n_ik, device=q.device, dtype=torch.bool),
                block_m / block_n,
            )
            image_mask = causal[None, None].expand(B, H, n_iq, n_ik).clone()
        else:
            image_mask = torch.ones((B, H, n_iq, n_ik), dtype=torch.bool, device=q.device)

    if static_block_mask is not None:
        image_mask = image_mask | static_block_mask[None, None, ...]

    full_mask = _assemble_full_block_mask(
        image_mask, image_len_q, image_len_k, text_len, block_m, block_n
    )

    if mode == "sparge":
        return block_attn_mask_to_ragged_lut(full_mask), -1

    # vfa / both: front-load the top-n sampled image tiles, then the dense text
    # tiles. Scores live only over the image region; text/text-row positions stay
    # -inf so they are never picked as sampled tiles (text is forced front).
    B, H, n_iq, n_ik = image_mask.shape
    n_tq, n_tk = full_mask.shape[-2], full_mask.shape[-1]
    full_score = full_mask.new_full(
        (B, H, n_tq, n_tk), float("-inf"), dtype=torch.float32
    )
    full_score[:, :, :n_iq, :n_ik] = image_score.to(torch.float32)

    force_front = None
    if n_text_k > 0:
        force_front = torch.zeros((B, H, n_tq, n_tk), dtype=torch.bool, device=q.device)
        force_front[:, :, :, -n_text_k:] = True

    block_lut = block_attn_mask_to_ragged_lut_topn_front(
        full_mask, full_score, n_sample, force_front_mask=force_front
    )
    return block_lut, n_sample + n_text_k
