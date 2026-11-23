import math

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.attention.mha_mxfp8 import (
    _mha_fwd_subtiled,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def get_mha_fwd_config(
    seqlen_q, seqlen_k, causal, unequal_seqlens, sliding_window
) -> dict:
    cfg_dir = resolve_config_dir("attention", "MHA", backend="gluon")
    configs = load_config_json(f"{cfg_dir}/mxfp8.json")
    # The causal mask of an unaligned diagonal needs BLOCK_M <= BLOCK_N.
    if sliding_window:
        key = "sliding_window"
    elif causal and unequal_seqlens:
        key = "causal_unequal_seqs"
    else:
        key = "default"
    return dict(configs[key])


def get_wmma_layouts(BLOCK_M, head_dim_pow2, num_warps):
    tiles_per_warp = BLOCK_M // num_warps // 16
    assert tiles_per_warp > 0
    num_warps_log2 = int(math.log2(num_warps))
    tiles_per_warp_log2 = int(math.log2(tiles_per_warp))
    base = 1
    warp_bases = []
    reg_bases = []
    for i in range(tiles_per_warp_log2):
        reg_bases.append([base, 0])
        base *= 2
    for i in range(num_warps_log2):
        warp_bases.append([base, 0])
        base *= 2

    WMMA_LAYOUT_QK = gl.amd.AMDWMMALayout(
        3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, 128],
    )
    WMMA_LAYOUT_O = gl.amd.AMDWMMALayout(
        3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, 128],
    )
    return WMMA_LAYOUT_QK, WMMA_LAYOUT_O


def strides_4d(t):
    # Varlen HTD tensors have no batch dim: use a zero batch stride.
    return (0, *t.stride()) if t.dim() == 3 else t.stride()


def mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    o: torch.Tensor | None = None,
    sm_scale: float | None = None,
    causal: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    window_size: tuple[int, int] = (-1, -1),
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Multi-head attention forward with per-tensor fp8 (e4m3) Q, K and V.

    q:         [B, HQ, SQ, D] (layout "bhsd") dtype float8_e4m3fn.
    k, v:      [B, HK, SK, D] dtype float8_e4m3fn. HQ must be a multiple of HK
               (grouped query attention; HK == HQ is plain MHA). SK may differ
               from SQ.
    q_scale, k_scale, v_scale: mx scales. q_scale is [B, HQ, SQ, D // 32]
               and k_scale is [B, HK, SK, D // 32]. v_scale is quantized along
               the sequence: [B, HK, cdiv(SK, 32), D].
    o:         optional output buffer, same shape as q, dtype bfloat16. The
               kernel downcasts the fp32 accumulator on the way out, so
               bfloat16 is the only supported output dtype.
    sm_scale:  softmax scale; defaults to 1 / sqrt(D).
    causal:    causal mask aligned to the bottom-right corner: query i attends
               to keys j <= i + (SK - SQ). If SQ > SK, the first SQ - SK
               query rows have no visible key and their output is zero.
    cu_seqlens_q, cu_seqlens_k: optional int32 [B + 1] cumulative sequence
               lengths. If set, the inputs are varlen with layout "htd":
               q [HQ, total_q, D], k and v [HK, total_k, D],
               q_scale [HQ, total_q, D // 32], k_scale [HK, total_k, D // 32]
               and v_scale [HK, cdiv(total_k, 32), D]. v_scale is quantized
               along the packed total_k axis, so a sequence can start at any
               token. Causal masking applies to each sequence. Each sequence
               must have at least one query and one key.
    max_seqlen_q, max_seqlen_k: maximum sequence lengths. Required for varlen.
    window_size: (left, right) local attention window. Only a left window
               with causal=True is supported, so right must be -1. If
               left > 0, query i attends to keys
               i + (SK - SQ) - left <= j <= i + (SK - SQ). A left of 0 or -1
               means no window.
    sink:      optional [HQ] attention sink logits, or None. Each one acts as
               an extra softmax column with no value vector. The logits are
               not multiplied by sm_scale.

    Returns:
    o:         same shape as q, dtype bfloat16.
    """

    arch = get_arch()
    assert arch == "gfx1250", "Only supported on gfx1250"

    varlen = cu_seqlens_q is not None
    assert varlen == (
        cu_seqlens_k is not None
    ), "set both cu_seqlens_q and cu_seqlens_k, or neither"
    num_dims = 3 if varlen else 4
    for t, name in zip((q, k, v), ("q", "k", "v")):
        assert t.dtype == torch.float8_e4m3fn, f"{name} must be float8_e4m3fn"
        assert t.dim() == num_dims, f"{name} must be {num_dims}D, got {t.dim()}D"
    assert k.shape == v.shape, "k and v must have the same shape"

    if varlen:
        for t, name in zip(
            (cu_seqlens_q, cu_seqlens_k), ("cu_seqlens_q", "cu_seqlens_k")
        ):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.device == q.device, f"{name} must be on the device of q"
        assert (
            cu_seqlens_q.numel() == cu_seqlens_k.numel()
        ), "cu_seqlens_q and cu_seqlens_k must have the same length"
        assert (
            max_seqlen_q is not None and max_seqlen_k is not None
        ), "varlen requires max_seqlen_q and max_seqlen_k"
        batch = cu_seqlens_q.numel() - 1
        num_q_heads, _, head_dim = q.shape
        num_k_heads, _, _ = k.shape
        seqlen_q, seqlen_k = max_seqlen_q, max_seqlen_k
    else:
        batch, num_q_heads, seqlen_q, head_dim = q.shape
        _, num_k_heads, seqlen_k, _ = k.shape
        assert k.shape[0] == batch, "k/v must match q in batch"
    assert k.shape[-1] == head_dim, "k/v must match q in head_dim"
    assert (
        num_q_heads % num_k_heads == 0
    ), f"num_q_heads ({num_q_heads}) must be a multiple of num_k_heads ({num_k_heads})"

    window_left, window_right = (int(w) for w in window_size)
    assert window_right == -1, "only a left window is supported, right must be -1"
    assert window_left >= -1, f"window_size left must be >= -1, got {window_left}"
    # A left window of 0 or -1 means no window.
    sliding_window = max(window_left, 0)
    assert causal or not sliding_window, "a sliding window requires causal=True"

    if sink is not None:
        assert sink.shape == (
            num_q_heads,
        ), f"sink must have shape ({num_q_heads},), got {tuple(sink.shape)}"
        assert sink.is_floating_point(), "sink must be a floating-point tensor"
        assert sink.device == q.device, "sink must be on the device of q"

    head_dim_pow2 = triton.next_power_of_2(head_dim)
    assert head_dim_pow2 >= 64, "tested only on head_dim_pow2 >= 64"
    unequal_seqlens = varlen or seqlen_q != seqlen_k

    config = get_mha_fwd_config(
        seqlen_q, seqlen_k, causal, unequal_seqlens, sliding_window
    )
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    NUM_BUFFERS = config["NUM_BUFFERS"]

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    if o is None:
        o = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    else:
        assert o.shape == q.shape, "output must have the same shape as q"
        # The kernel stores acc.to(bfloat16) through a descriptor typed by o.
        assert o.dtype == torch.bfloat16, "output must be bfloat16"

    stride_q_z, stride_q_h, stride_q_m, stride_q_d = strides_4d(q)
    stride_q_scale_z, stride_q_scale_h, stride_q_scale_m, stride_q_scale_d = strides_4d(
        q_scale
    )
    stride_k_z, stride_k_h, stride_k_n, stride_k_d = strides_4d(k)
    stride_k_scale_z, stride_k_scale_h, stride_k_scale_n, stride_k_scale_d = strides_4d(
        k_scale
    )
    stride_v_z, stride_v_h, stride_v_n, stride_v_d = strides_4d(v)
    stride_v_scale_z, stride_v_scale_h, stride_v_scale_n, stride_v_scale_d = strides_4d(
        v_scale
    )
    stride_o_z, stride_o_h, stride_o_m, stride_o_d = strides_4d(o)

    if not varlen and seqlen_k % 2 == 0 and head_dim == head_dim_pow2:
        preshuffle_factor_k = 2
    else:
        preshuffle_factor_k = 1
    preshuffle_factor_v = 1
    stride_k_n *= preshuffle_factor_k
    # stride_v_n *= preshuffle_factor_v

    WMMA_LAYOUT_QK, WMMA_LAYOUT_O = get_wmma_layouts(
        BLOCK_M, head_dim_pow2, config["num_warps"]
    )

    grid_m = triton.cdiv(seqlen_q, BLOCK_M)
    grid = (num_q_heads, grid_m, batch)
    _mha_fwd_subtiled[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        q_scale_ptr=q_scale,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        o_ptr=o,
        stride_q_z=stride_q_z,
        stride_q_h=stride_q_h,
        stride_q_m=stride_q_m,
        stride_q_d=stride_q_d,
        stride_q_scale_z=stride_q_scale_z,
        stride_q_scale_h=stride_q_scale_h,
        stride_q_scale_m=stride_q_scale_m,
        stride_q_scale_d=stride_q_scale_d,
        stride_k_z=stride_k_z,
        stride_k_h=stride_k_h,
        stride_k_n=stride_k_n,
        stride_k_d=stride_k_d,
        stride_k_scale_z=stride_k_scale_z,
        stride_k_scale_h=stride_k_scale_h,
        stride_k_scale_n=stride_k_scale_n,
        stride_k_scale_d=stride_k_scale_d,
        stride_v_z=stride_v_z,
        stride_v_h=stride_v_h,
        stride_v_n=stride_v_n,
        stride_v_d=stride_v_d,
        stride_v_scale_z=stride_v_scale_z,
        stride_v_scale_h=stride_v_scale_h,
        stride_v_scale_n=stride_v_scale_n,
        stride_v_scale_d=stride_v_scale_d,
        stride_o_z=stride_o_z,
        stride_o_h=stride_o_h,
        stride_o_m=stride_o_m,
        stride_o_d=stride_o_d,
        SM_SCALE=sm_scale,
        CAUSAL=causal,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        SEQLEN_Q=seqlen_q,
        SEQLEN_K=seqlen_k,
        UNEQUAL_SEQLENS=unequal_seqlens,
        VARLEN=varlen,
        SLIDING_WINDOW=sliding_window,
        sink_ptr=sink,
        ENABLE_SINK=sink is not None,
        cu_seqlens_q_ptr=cu_seqlens_q,
        cu_seqlens_k_ptr=cu_seqlens_k,
        grid_m=grid_m,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        HEAD_DIM_POW2=head_dim_pow2,
        NUM_BUFFERS=NUM_BUFFERS,
        EVEN_N=not varlen and seqlen_k % BLOCK_N == 0,
        SHORT_SEQ=not varlen and triton.cdiv(seqlen_k, BLOCK_N) < 3,
        PRESHUFFLE_FACTOR_K=preshuffle_factor_k,
        PRESHUFFLE_FACTOR_V=preshuffle_factor_v,
        WMMA_LAYOUT_QK=WMMA_LAYOUT_QK,
        WMMA_LAYOUT_O=WMMA_LAYOUT_O,
        num_warps=config["num_warps"],
        waves_per_eu=0,
    )

    return o
