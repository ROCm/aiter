import functools
import math

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.attention.mha_fp8 import (
    _mha_fwd_mla_varlen,
    _mha_fwd_subtiled,
)
from aiter.ops.triton.attention.utils import lse_merge
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.device_info import get_num_sms


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


def get_mla_varlen_config(causal, seqlen_k, total_q, num_heads) -> dict:
    cfg_dir = resolve_config_dir("attention", "MHA", backend="gluon")
    configs = load_config_json(f"{cfg_dir}/DEFAULT.json")["mla_varlen"]
    if causal:
        cfg = configs["causal"]
        programs = -(-total_q // cfg["BLOCK_M"]) * num_heads
        num_sms = get_num_sms()
        if programs < num_sms or (
            programs < 2 * num_sms and seqlen_k <= configs["causal_small_max_seqlen_k"]
        ):
            cfg = configs["causal_small"]
        elif seqlen_k >= configs["causal_long_min_seqlen_k"]:
            cfg = configs["causal_long"]
        return dict(cfg, NUM_SPLITS=1)
    if seqlen_k <= configs["short_max_seqlen_k"]:
        return dict(configs["short"], NUM_SPLITS=1)
    cfg = dict(configs["default"], NUM_SPLITS=1)
    programs = -(-total_q // cfg["BLOCK_M"]) * num_heads
    while (
        cfg["NUM_SPLITS"] < configs["split_kv_max_num_splits"]
        and programs * cfg["NUM_SPLITS"] * 2 <= get_num_sms()
        and seqlen_k // (cfg["NUM_SPLITS"] * 2 * cfg["BLOCK_N"])
        >= configs["split_kv_min_tiles_per_split"]
    ):
        cfg["NUM_SPLITS"] *= 2
    return cfg


@functools.cache
def _mla_varlen_layouts(block_m, num_warps):
    """WMMA layouts of the 128-dim QK and PV products and of the 64 PE dims."""
    layout = get_wmma_layouts(block_m, 128, num_warps)[0]
    layout_pe = gl.amd.AMDWMMALayout(
        3,
        transposed=True,
        warp_bases=layout.warp_bases,
        reg_bases=layout.reg_bases,
        instr_shape=[16, 16, 64],
    )
    return layout, layout_pe


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


def mha_fwd_mla_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_pe: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    sm_scale: float | None = None,
    causal: bool = False,
    out: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
    return_lse: bool = False,
    o_init: torch.Tensor | None = None,
    lse_init: torch.Tensor | None = None,
    config: dict | None = None,
):
    """
    fp8 attention of packed variable-length sequences with MLA heads

    q:         [total_q, HQ, 192] float8_e4m3fn: 128 dims, then 64 PE dims.
    k:         [total_k, HK, 128] float8_e4m3fn.
    v:         [total_k, HK, 64 or 128] float8_e4m3fn, with the strides of k.
    k_pe:      [total_k, 1 or HK, 64] float8_e4m3fn PE dims of k.
    cu_seqlens_q, cu_seqlens_k: [B + 1] int32 start offsets of the sequences.
    q_scale:   optional [B, HQ] fp32 dequant scales; k_scale and v_scale [B, HK].
               Give all three scales or none. None means a scale of 1.
    sm_scale:  softmax scale, 192 ** -0.5 by default. It must not be negative. A zero
               scale (also from a zero q or k scale) gives equal weights to the keys.
    causal:    causal mask aligned to the bottom-right corner of each sequence.
    out:       optional [total_q, HQ, dv] output; bf16 by default.
    lse_out:   optional [HQ, total_q] fp32 natural-log LSE; return_lse allocates one.
    o_init, lse_init: optional result of the same queries over other keys; the
               output merges it.

    Returns (out, lse_out).
    """
    assert get_arch() == "gfx1250", "Only supported on gfx1250"
    for t in (q, k, v, k_pe):
        assert t.dtype == torch.float8_e4m3fn and t.dim() == 3
    total_q, num_heads, _ = q.shape
    num_k_heads, head_dim_v = k.shape[1], v.shape[2]
    num_seqs = cu_seqlens_q.numel() - 1
    assert q.shape[2] == 192 and k.shape[2] == 128 and k_pe.shape[2] == 64
    assert head_dim_v in (64, 128) and v.stride() == k.stride()
    assert k_pe.shape[1] in (1, num_k_heads) and num_heads % num_k_heads == 0
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    assert sm_scale >= 0, "sm_scale must be a number that is not negative"
    has_scales = q_scale is not None
    assert (k_scale is not None) == has_scales and (v_scale is not None) == has_scales
    if has_scales:
        assert v_scale.stride() == k_scale.stride()
    if out is None:
        out = q.new_empty((total_q, num_heads, head_dim_v), dtype=torch.bfloat16)
    assert out.shape == (total_q, num_heads, head_dim_v)
    for t in (q, k, v, k_pe, out):
        assert t.data_ptr() % 16 == 0 and t.stride(-1) == 1
        assert all(s * t.element_size() % 16 == 0 for s in t.stride()[:-1])
    if lse_out is None and (return_lse or o_init is not None):
        lse_out = torch.empty(
            (num_heads, total_q), dtype=torch.float32, device=q.device
        )
    if o_init is not None:
        if o_init.data_ptr() != out.data_ptr():
            out.copy_(o_init)
        if lse_init.data_ptr() != lse_out.data_ptr():
            lse_out.copy_(lse_init)
    if total_q == 0:
        return out, lse_out

    cfg = get_mla_varlen_config(causal, max_seqlen_k, total_q, num_heads)
    cfg.update(config or {})
    BLOCK_M, num_splits = cfg["BLOCK_M"], cfg["NUM_SPLITS"]
    qblk_per_seq = -(-max_seqlen_q // BLOCK_M)
    num_qblk = total_q // BLOCK_M + num_seqs
    if 0 < num_seqs * qblk_per_seq <= num_qblk:
        num_qblk = num_seqs * qblk_per_seq
    else:
        qblk_per_seq = 0
    if num_splits > 1:
        # One partial per split. lse_merge also merges o_init as one more partial.
        num_parts = num_splits + (o_init is not None)
        o_k = torch.empty(
            (num_parts, total_q, num_heads, head_dim_v),
            dtype=torch.float32,
            device=q.device,
        )
        lse_k = torch.empty(
            (num_parts, num_heads, total_q), dtype=torch.float32, device=q.device
        )
        if o_init is not None:
            o_k[num_splits].copy_(out)
            lse_k[num_splits].copy_(lse_out)
    else:
        o_k = out.unsqueeze(0)
        lse_k = (out if lse_out is None else lse_out).unsqueeze(0)
    wmma_layout, wmma_layout_pe = _mla_varlen_layouts(BLOCK_M, cfg["num_warps"])
    fast_rowsum = bool(cfg.get("ROWSUM_MMA", lse_out is None and o_init is None))
    _mha_fwd_mla_varlen[(num_qblk * num_heads, num_splits)](
        q,
        k,
        k_pe,
        v,
        o_k,
        lse_k,
        cu_seqlens_q,
        cu_seqlens_k,
        q_scale if has_scales else q,
        k_scale if has_scales else q,
        v_scale if has_scales else q,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k_pe.stride(0),
        0 if k_pe.shape[1] == 1 else k_pe.stride(1),
        o_k.stride(1),
        o_k.stride(2),
        o_k.stride(0),
        lse_k.stride(1),
        lse_k.stride(2),
        lse_k.stride(0),
        *(q_scale.stride() if has_scales else (0, 0)),
        *(k_scale.stride() if has_scales else (0, 0)),
        sm_scale,
        num_seqs,
        num_heads,
        num_heads // num_k_heads,
        qblk_per_seq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=cfg["BLOCK_N"],
        HEAD_DIM=128,
        PE_DIM=64,
        HEAD_DIM_V=head_dim_v,
        CAUSAL=causal,
        NUM_SPLITS=num_splits,
        RETURN_LSE=num_splits > 1 or lse_out is not None,
        HAS_INIT=num_splits == 1 and o_init is not None,
        HAS_SCALES=has_scales,
        WMMA_LAYOUT=wmma_layout,
        WMMA_LAYOUT_PE=wmma_layout_pe,
        PV_SPLIT_K=fast_rowsum,
        ROWSUM_MMA=fast_rowsum,
        NUM_K_BUFFERS=int(cfg.get("NUM_K_BUFFERS", 3)),
        EARLY_K=bool(cfg.get("EARLY_K", True)),
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
        enable_fp_fusion=True,
    )
    if num_splits > 1:
        lse_merge(
            o_k,
            lse_k,
            out.unsqueeze(0),
            None if lse_out is None else lse_out.unsqueeze(0),
        )
    return out, lse_out
