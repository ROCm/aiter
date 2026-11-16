import math

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.attention.mha_mxfp8 import (
    _mha_fwd_subtiled,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def get_mha_fwd_config(seqlen_q, seqlen_k, causal) -> dict:
    cfg_dir = resolve_config_dir("attention", "MHA", backend="gluon")
    configs = load_config_json(f"{cfg_dir}/mxfp8.json")
    key = "causal_unequal_seqs" if causal and seqlen_q < seqlen_k else "default"
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
) -> torch.Tensor:
    """
    Multi-head attention forward with per-tensor fp8 (e4m3) Q, K and V.

    q:         [B, HQ, SQ, D] (layout "bhsd") dtype float8_e4m3fn.
    k, v:      [B, HK, SK, D] dtype float8_e4m3fn. HQ must be a multiple of HK
               (grouped query attention; HK == HQ is plain MHA). SK may differ
               from SQ; with causal=True it must satisfy SK >= SQ.
    q_scale, k_scale, v_scale: mx scales
    o:         optional output buffer, same shape as q, dtype bfloat16. The
               kernel downcasts the fp32 accumulator on the way out, so
               bfloat16 is the only supported output dtype.
    sm_scale:  softmax scale; defaults to 1 / sqrt(D).
    causal:    causal mask aligned to the bottom-right corner: query i attends
               to keys j <= i + (SK - SQ).

    Returns:
    o:         same shape as q, dtype bfloat16.
    """

    arch = get_arch()
    assert arch == "gfx1250", "Only supported on gfx1250"

    for t, name in zip((q, k, v), ("q", "k", "v")):
        assert t.dtype == torch.float8_e4m3fn, f"{name} must be float8_e4m3fn"
        assert t.dim() == 4, f"{name} must be 4D, got {t.dim()}D"

    batch, num_q_heads, seqlen_q, head_dim = q.shape
    _, num_k_heads, seqlen_k, _ = k.shape
    assert k.shape == v.shape, "k and v must have the same shape"
    assert (
        k.shape[0] == batch and k.shape[3] == head_dim
    ), "k/v must match q in batch and head_dim"
    assert (
        num_q_heads % num_k_heads == 0
    ), f"num_q_heads ({num_q_heads}) must be a multiple of num_k_heads ({num_k_heads})"
    # With the bottom-right causal alignment, SQ > SK would leave the first
    # SQ - SK query rows without any visible key.
    assert (
        not causal or seqlen_k >= seqlen_q
    ), f"causal requires seqlen_k ({seqlen_k}) >= seqlen_q ({seqlen_q})"

    config = get_mha_fwd_config(seqlen_q, seqlen_k, causal)
    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    NUM_BUFFERS = config["NUM_BUFFERS"]

    head_dim_pow2 = triton.next_power_of_2(head_dim)
    assert head_dim_pow2 >= 64, "tested only on head_dim_pow2 >= 64"
    causal_offset = seqlen_k - seqlen_q
    use_causal_offset = causal_offset > 0

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim**0.5)

    if o is None:
        o = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    else:
        assert o.shape == q.shape, "output must have the same shape as q"
        # The kernel stores acc.to(bfloat16) through a descriptor typed by o.
        assert o.dtype == torch.bfloat16, "output must be bfloat16"

    stride_q_z, stride_q_h, stride_q_m, stride_q_d = (q.stride(i) for i in range(4))
    stride_q_scale_z, stride_q_scale_h, stride_q_scale_m, stride_q_scale_d = (
        q_scale.stride(i) for i in range(4)
    )
    stride_k_z, stride_k_h, stride_k_n, stride_k_d = (k.stride(i) for i in range(4))
    stride_k_scale_z, stride_k_scale_h, stride_k_scale_n, stride_k_scale_d = (
        k_scale.stride(i) for i in range(4)
    )
    stride_v_z, stride_v_h, stride_v_n, stride_v_d = (v.stride(i) for i in range(4))
    stride_v_scale_z, stride_v_scale_h, stride_v_scale_n, stride_v_scale_d = (
        v_scale.stride(i) for i in range(4)
    )
    stride_o_z, stride_o_h, stride_o_m, stride_o_d = (o.stride(i) for i in range(4))

    if seqlen_k % 2 == 0 and head_dim == head_dim_pow2:
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
        USE_CAUSAL_OFFSET=use_causal_offset,
        CAUSAL_OFFSET=causal_offset,
        grid_m=grid_m,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=head_dim,
        HEAD_DIM_POW2=head_dim_pow2,
        NUM_BUFFERS=NUM_BUFFERS,
        EVEN_N=seqlen_k % BLOCK_N == 0,
        SHORT_SEQ=triton.cdiv(seqlen_k, BLOCK_N) < 3,
        PRESHUFFLE_FACTOR_K=preshuffle_factor_k,
        PRESHUFFLE_FACTOR_V=preshuffle_factor_v,
        WMMA_LAYOUT_QK=WMMA_LAYOUT_QK,
        WMMA_LAYOUT_O=WMMA_LAYOUT_O,
        num_warps=config["num_warps"],
        waves_per_eu=0,
    )

    return o
