import math

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import kv_async_copy_ok
from aiter.ops.triton._gluon_kernels.gfx950.attention.mha_fp8 import (
    _mha_varlen_fp8_combine,
    _mha_varlen_fwd_fp8,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.mha_fp8 import _mha_fwd_subtiled
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def get_mha_fwd_config(seqlen_q, seqlen_k, causal) -> dict:
    cfg_dir = resolve_config_dir("attention", "MHA", backend="gluon")
    configs = load_config_json(f"{cfg_dir}/DEFAULT.json")
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

    if head_dim_pow2 == 64:
        wmma_kwidth = 64
    else:
        wmma_kwidth = 128

    WMMA_LAYOUT_QK = gl.amd.AMDWMMALayout(
        3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, wmma_kwidth],
    )
    WMMA_LAYOUT_O = gl.amd.AMDWMMALayout(
        3,
        transposed=True,
        warp_bases=warp_bases,
        reg_bases=reg_bases,
        instr_shape=[16, 16, wmma_kwidth],
    )
    return WMMA_LAYOUT_QK, WMMA_LAYOUT_O


def mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor | float,
    k_scale: torch.Tensor | float,
    v_scale: torch.Tensor | float,
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
    q_scale, k_scale, v_scale: per-tensor dequant scales, fp32 scalars (python
               float or a 1-element / 0-dim tensor).
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

    def _as_scale(s, name):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32, device=q.device)
        assert s.numel() == 1, f"{name} must be a per-tensor (scalar) scale"
        return s.to(device=q.device, dtype=torch.float32)

    q_scale = _as_scale(q_scale, "q_scale")
    k_scale = _as_scale(k_scale, "k_scale")
    v_scale = _as_scale(v_scale, "v_scale")

    if o is None:
        o = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    else:
        assert o.shape == q.shape, "output must have the same shape as q"
        # The kernel stores acc.to(bfloat16) through a descriptor typed by o.
        assert o.dtype == torch.bfloat16, "output must be bfloat16"

    stride_q_z, stride_q_h, stride_q_m, stride_q_d = (q.stride(i) for i in range(4))
    stride_k_z, stride_k_h, stride_k_n, stride_k_d = (k.stride(i) for i in range(4))
    stride_v_z, stride_v_h, stride_v_n, stride_v_d = (v.stride(i) for i in range(4))
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
        stride_k_z=stride_k_z,
        stride_k_h=stride_k_h,
        stride_k_n=stride_k_n,
        stride_k_d=stride_k_d,
        stride_v_z=stride_v_z,
        stride_v_h=stride_v_h,
        stride_v_n=stride_v_n,
        stride_v_d=stride_v_d,
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
        EVEN_N=(seqlen_k % BLOCK_N == 0) and (triton.cdiv(seqlen_k, BLOCK_N) >= 3),
        PRESHUFFLE_FACTOR_K=preshuffle_factor_k,
        PRESHUFFLE_FACTOR_V=preshuffle_factor_v,
        WMMA_LAYOUT_QK=WMMA_LAYOUT_QK,
        WMMA_LAYOUT_O=WMMA_LAYOUT_O,
        num_warps=config["num_warps"],
        waves_per_eu=0,
    )

    return o


def _get_varlen_config(causal: bool) -> dict:
    cfg_dir = resolve_config_dir("attention", "MHA", backend="gluon")
    configs = load_config_json(f"{cfg_dir}/mha.json")["fwd_varlen_fp8"]
    return dict(configs["causal" if causal else "non_causal"])


def _max_offset(t: torch.Tensor) -> int:
    return sum((n - 1) * s for n, s in zip(t.shape, t.stride()))


def mha_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    config: dict | None = None,
):
    """FP8 varlen MHA forward (gfx950).

    Args:
        q, k: [T, H, d_qk] float8_e4m3fn. v: [T, H, d_v] float8_e4m3fn. The last dim is contiguous.
        cu_seqlens_q, cu_seqlens_k: [batch + 1] int32 cumulative sequence lengths.
        max_seqlen_q, max_seqlen_k: largest sequence lengths in the batch.
        q_descale, k_descale, v_descale: one-element fp32 tensors, the per-tensor dequant scales.
        softmax_scale: defaults to 1 / sqrt(d_qk).
        causal: apply the causal mask.
        return_lse: also return the log-sum-exp.
        out, lse: optional preallocated outputs.
        config: optional kernel config. The default comes from the JSON config.

    Returns:
        out [Tq, H, d_v] bfloat16, and with ``return_lse`` also lse [H, Tq] float32 (natural log,
        -inf for a row that sees no key).
    """
    arch = get_arch()
    assert arch == "gfx950", f"mha_varlen_fwd is only supported on gfx950, got {arch}"
    assert q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
    Tq, H, DQK = q.shape
    DV = v.shape[-1]
    D_PE = DQK - DV
    assert (DV, D_PE) == (
        128,
        64,
    ), f"needs d_v 128 and d_qk - d_v 64, got d_qk {DQK}, d_v {DV}"
    assert (
        k.shape[1] == H and v.shape[1] == H
    ), "q, k and v must have the same head count"
    for t in (q, k, v):
        assert t.stride(-1) == 1, "the last dim must be contiguous"
        assert t.device == q.device, "q, k and v must be on the same device"
    for cu in (cu_seqlens_q, cu_seqlens_k):
        assert (
            cu.dtype == torch.int32 and cu.is_contiguous() and cu.device == q.device
        ), "cu_seqlens_q and cu_seqlens_k must be contiguous int32 on the device of q"
    assert cu_seqlens_k.numel() == cu_seqlens_q.numel(), "cu_seqlens sizes differ"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(DQK)
    cfg = dict(config) if config is not None else _get_varlen_config(causal)
    if out is None:
        out = torch.empty((Tq, H, DV), dtype=torch.bfloat16, device=q.device)
    assert (
        out.shape == (Tq, H, DV) and out.dtype == torch.bfloat16
    ), f"out must be [{Tq}, {H}, {DV}] bfloat16, got {list(out.shape)} {out.dtype}"
    assert (
        out.device == q.device and out.stride(-1) == 1
    ), "out must be on the device of q with a contiguous last dim"
    if return_lse:
        if lse is None:
            lse = torch.empty((H, Tq), dtype=torch.float32, device=q.device)
        assert (
            lse.shape == (H, Tq) and lse.dtype == torch.float32
        ), f"lse must be [{H}, {Tq}] float32, got {list(lse.shape)} {lse.dtype}"
        assert (
            lse.device == q.device and lse.stride(-1) == 1
        ), "lse must be on the device of q with a contiguous last dim"
    batch = cu_seqlens_q.numel() - 1
    BLOCK_M = cfg.pop("BLOCK_M")
    BLOCK_N = cfg.pop("BLOCK_N")
    num_warps = cfg.pop("num_warps")
    waves_per_eu = cfg.pop("waves_per_eu")
    target_ctas = cfg.pop("TARGET_CTAS", 0)
    max_splits = cfg.pop("MAX_SPLITS", 1)
    kv_align = math.gcd(16, k.stride(0), k.stride(1), v.stride(0), v.stride(1))
    assert (
        kv_align == 16 and k.data_ptr() % 16 == 0 and v.data_ptr() % 16 == 0
    ), "k and v rows and heads must be 16-byte aligned"
    assert kv_async_copy_ok(DV, D_PE, BLOCK_N, num_warps, 8, kv_align)
    ctas = triton.cdiv(int(max_seqlen_q), BLOCK_M) * H * batch
    # Short queries over a long context leave most CUs idle: split the keys until the launch has about
    # target_ctas programs, with at least 4 key blocks per split. The partials are merged below.
    num_splits = 1
    if not causal and target_ctas > ctas:
        num_splits = max(
            1, min(max_splits, target_ctas // ctas, int(max_seqlen_k) // (4 * BLOCK_N))
        )
        num_splits = 1 << (
            num_splits.bit_length() - 1
        )  # power of 2 for the combine kernel
    if num_splits > 1:
        o_dst = torch.empty(
            (num_splits, Tq, H, DV), dtype=torch.float32, device=q.device
        )
        lse_dst = torch.empty((num_splits, H, Tq), dtype=torch.float32, device=q.device)
    else:
        o_dst = out
        lse_dst = lse if return_lse else out
    # Use int64 base offsets only when an element offset can pass int32.
    int64_offsets = max(_max_offset(t) for t in (q, k, v, o_dst, lse_dst, out)) >= 2**31
    grid = (ctas * num_splits,)
    _mha_varlen_fwd_fp8[grid](
        q,
        k,
        v,
        o_dst,
        lse_dst,
        q_descale,
        k_descale,
        v_descale,
        cu_seqlens_q,
        cu_seqlens_k,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o_dst.stride(-3),
        o_dst.stride(-2),
        lse_dst.stride(-2) if (return_lse or num_splits > 1) else 0,
        o_dst.stride(0) if num_splits > 1 else 0,
        lse_dst.stride(0) if num_splits > 1 else 0,
        SM_SCALE=float(softmax_scale),
        CAUSAL=bool(causal),
        NUM_Q_HEADS=H,
        NUM_K_HEADS=k.shape[1],
        MAX_SEQLEN_Q=int(max_seqlen_q),
        num_splits=num_splits,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM_NOPE=DV,
        HEAD_DIM_PE=D_PE,
        HEAD_DIM_V=DV,
        RETURN_LSE=bool(return_lse or num_splits > 1),
        INT64_OFFSETS=int64_offsets,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    if num_splits > 1:
        _mha_varlen_fp8_combine[(Tq * H,)](
            o_dst,
            lse_dst,
            out,
            lse if return_lse else lse_dst,
            cu_seqlens_q,
            batch,
            o_dst.stride(0),
            o_dst.stride(1),
            o_dst.stride(2),
            lse_dst.stride(0),
            lse_dst.stride(1),
            out.stride(0),
            out.stride(1),
            lse.stride(0) if return_lse else 0,
            NUM_Q_HEADS=H,
            NUM_SPLITS=num_splits,
            HEAD_DIM_V=DV,
            RETURN_LSE=bool(return_lse),
            INT64_OFFSETS=int64_offsets,
            num_warps=1,
        )
    return (out, lse) if return_lse else out
