import functools
import math

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import kv_async_copy_ok
from aiter.ops.triton._gluon_kernels.gfx950.attention.mha_fp8 import (
    _mha_varlen_fp8_combine,
    _mha_varlen_fwd_fp8,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.mha_fp8 import (
    _mha_fwd_subtiled,
    _mha_varlen_fwd_kernel,
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


def get_mha_varlen_config(causal, seqlen_k, total_q, num_heads) -> dict:
    cfg_dir = resolve_config_dir("attention", "MHA", backend="gluon")
    configs = load_config_json(f"{cfg_dir}/DEFAULT.json")["varlen"]
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
def _mha_varlen_layouts(block_m, num_warps):
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
    assert arch in ("gfx1250", "gfx1250-strict"), "Only supported on the gfx1250 family"

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


def _descale_strides(t: torch.Tensor, name: str, rows: int, heads: int, device):
    assert (
        t.dtype == torch.float32 and t.device == device
    ), f"{name} must be float32 on {device}, got {t.dtype} on {t.device}"
    if t.numel() == 1:
        return (0, 0)
    assert t.shape == (
        rows,
        heads,
    ), f"{name} must have one element or be [{rows}, {heads}], got {list(t.shape)}"
    return t.stride()


def _mha_varlen_fwd_gfx1250(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    q_descale,
    k_descale,
    v_descale,
    softmax_scale,
    causal,
    return_lse,
    out,
    lse,
    config,
    k_pe,
    o_init,
    lse_init,
):
    D_NOPE, D_PE = 128, 64
    if k_pe is None:
        assert (
            k.dim() == 3 and k.shape[2] == D_NOPE + D_PE
        ), f"k must be [T, H, {D_NOPE + D_PE}] when k_pe is not given, got {list(k.shape)}"
        k, k_pe = k[..., :D_NOPE], k[..., D_NOPE:]
    for t in (q, k, v, k_pe):
        assert t.dtype == torch.float8_e4m3fn and t.dim() == 3
        assert t.device == q.device, "q, k, v and k_pe must be on the same device"
    total_q, num_heads, _ = q.shape
    num_k_heads, head_dim_v = k.shape[1], v.shape[2]
    num_seqs = cu_seqlens_q.numel() - 1
    assert (
        q.shape[2] == D_NOPE + D_PE and k.shape[2] == D_NOPE and k_pe.shape[2] == D_PE
    ), f"needs d_qk {D_NOPE + D_PE} with {D_PE} rope dims, got q {list(q.shape)}"
    assert head_dim_v in (64, 128), f"d_v must be 64 or 128, got {head_dim_v}"
    assert v.shape[:2] == k.shape[:2], "v must have the tokens and heads of k"
    assert k_pe.shape[0] == k.shape[0] and k_pe.shape[1] in (1, num_k_heads)
    assert num_heads % num_k_heads == 0
    for cu in (cu_seqlens_q, cu_seqlens_k):
        assert (
            cu.dtype == torch.int32 and cu.is_contiguous() and cu.device == q.device
        ), "cu_seqlens_q and cu_seqlens_k must be contiguous int32 on the device of q"
    assert cu_seqlens_k.numel() == cu_seqlens_q.numel(), "cu_seqlens sizes differ"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    assert softmax_scale >= 0, "softmax_scale must not be negative"
    has_scales = q_descale is not None
    assert (k_descale is not None) == has_scales and (
        v_descale is not None
    ) == has_scales, "give all of q_descale, k_descale and v_descale, or none"
    if has_scales:
        qs_strides = _descale_strides(
            q_descale, "q_descale", num_seqs, num_heads, q.device
        )
        ks_strides = _descale_strides(
            k_descale, "k_descale", num_seqs, num_k_heads, q.device
        )
        vs_strides = _descale_strides(
            v_descale, "v_descale", num_seqs, num_k_heads, q.device
        )
        assert vs_strides == ks_strides, "v_descale must have the strides of k_descale"
    else:
        qs_strides = ks_strides = (0, 0)
    if out is None:
        out = q.new_empty((total_q, num_heads, head_dim_v), dtype=torch.bfloat16)
    assert out.shape == (
        total_q,
        num_heads,
        head_dim_v,
    ), f"out must be [{total_q}, {num_heads}, {head_dim_v}], got {list(out.shape)}"
    for t in (q, k, v, k_pe, out):
        assert t.data_ptr() % 16 == 0 and t.stride(-1) == 1
        assert all(s * t.element_size() % 16 == 0 for s in t.stride()[:-1])
    assert (o_init is None) == (lse_init is None), "give o_init and lse_init together"
    if lse is None and (return_lse or o_init is not None):
        lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
    if lse is not None:
        assert lse.shape == (num_heads, total_q) and lse.dtype == torch.float32
        assert lse.stride(-1) == 1
    if o_init is not None:
        if o_init.data_ptr() != out.data_ptr():
            out.copy_(o_init)
        if lse_init.data_ptr() != lse.data_ptr():
            lse.copy_(lse_init)
    if total_q == 0 or num_seqs == 0:
        return (out, lse) if return_lse else out

    cfg = get_mha_varlen_config(causal, max_seqlen_k, total_q, num_heads)
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
            lse_k[num_splits].copy_(lse)
    else:
        o_k = out.unsqueeze(0)
        lse_k = (out if lse is None else lse).unsqueeze(0)
    wmma_layout, wmma_layout_pe = _mha_varlen_layouts(BLOCK_M, cfg["num_warps"])
    fast_rowsum = bool(cfg.get("ROWSUM_MMA", lse is None and o_init is None))
    _mha_varlen_fwd_kernel[(num_qblk * num_heads, num_splits)](
        q,
        k,
        k_pe,
        v,
        o_k,
        lse_k,
        cu_seqlens_q,
        cu_seqlens_k,
        q_descale if has_scales else q,
        k_descale if has_scales else q,
        v_descale if has_scales else q,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_pe.stride(0),
        0 if k_pe.shape[1] == 1 else k_pe.stride(1),
        o_k.stride(1),
        o_k.stride(2),
        o_k.stride(0),
        lse_k.stride(1),
        lse_k.stride(2),
        lse_k.stride(0),
        *qs_strides,
        *ks_strides,
        softmax_scale,
        num_seqs,
        num_heads,
        num_heads // num_k_heads,
        qblk_per_seq,
        BLOCK_M=BLOCK_M,
        BLOCK_N=cfg["BLOCK_N"],
        HEAD_DIM=D_NOPE,
        PE_DIM=D_PE,
        HEAD_DIM_V=head_dim_v,
        CAUSAL=causal,
        NUM_SPLITS=num_splits,
        RETURN_LSE=num_splits > 1 or lse is not None,
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
            None if lse is None else lse.unsqueeze(0),
            row_bounds=cu_seqlens_q,
        )
    return (out, lse) if return_lse else out


def _get_varlen_config(causal: bool) -> dict:
    cfg_dir = resolve_config_dir("attention", "MHA", backend="gluon")
    configs = load_config_json(f"{cfg_dir}/DEFAULT.json")["fwd_varlen_fp8"]
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
    k_pe: torch.Tensor | None = None,
    o_init: torch.Tensor | None = None,
    lse_init: torch.Tensor | None = None,
):
    """FP8 varlen MHA forward (gfx950, gfx1250).

    Args:
        q, k: [T, H, d_qk] float8_e4m3fn. v: [T, H, d_v] float8_e4m3fn. The last dim is contiguous.
        cu_seqlens_q, cu_seqlens_k: [batch + 1] int32 cumulative sequence lengths.
        max_seqlen_q, max_seqlen_k: largest sequence lengths in the batch. They set the launch
            size. The wrapper does not compare them with cu_seqlens, because that needs a device
            sync. If a value is less than the real maximum, the rows after it are not written,
            or are wrong when the keys are split.
        q_descale, k_descale, v_descale: one-element fp32 tensors on the device of q, the
            per-tensor dequant scales.
        softmax_scale: defaults to 1 / sqrt(d_qk).
        causal: apply the causal mask.
        return_lse: also return the log-sum-exp.
        out, lse: optional preallocated outputs.
        config: optional kernel config. The default comes from the JSON config.

        k_pe, o_init, lse_init: gfx1250 only; separate k rope dims, and a prior (out, lse) to merge.
    Returns:
        out [Tq, H, d_v] bfloat16, and with ``return_lse`` also lse [H, Tq] float32 (natural log,
        -inf for a row that sees no key).

    Limits: gfx950: d_v 128, rope 0 or 64. gfx1250: d_qk 192 (64 rope), d_v 64 or 128.
    """
    arch = get_arch()
    if arch == "gfx1250":
        return _mha_varlen_fwd_gfx1250(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            return_lse,
            out,
            lse,
            config,
            k_pe,
            o_init,
            lse_init,
        )
    assert (
        arch == "gfx950"
    ), f"mha_varlen_fwd is only supported on gfx950 and gfx1250, got {arch}"
    assert (
        k_pe is None and o_init is None and lse_init is None
    ), "k_pe, o_init and lse_init are only supported on gfx1250"
    assert q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
    Tq, H, DQK = q.shape
    DV = v.shape[-1]
    D_PE = DQK - DV
    assert DV == 128 and D_PE in (
        0,
        64,
    ), f"needs d_v 128 and d_qk - d_v 0 or 64, got d_qk {DQK}, d_v {DV}"
    Tk = k.shape[0]
    assert k.shape == (
        Tk,
        H,
        DQK,
    ), f"k must be [{Tk}, {H}, {DQK}] like q, got {list(k.shape)}"
    assert v.shape == (
        Tk,
        H,
        DV,
    ), f"v must be [{Tk}, {H}, {DV}] with the token count of k, got {list(v.shape)}"
    for t in (q, k, v):
        assert t.stride(-1) == 1, "the last dim must be contiguous"
        assert t.device == q.device, "q, k and v must be on the same device"
    for cu in (cu_seqlens_q, cu_seqlens_k):
        assert (
            cu.dtype == torch.int32 and cu.is_contiguous() and cu.device == q.device
        ), "cu_seqlens_q and cu_seqlens_k must be contiguous int32 on the device of q"
    assert cu_seqlens_k.numel() == cu_seqlens_q.numel(), "cu_seqlens sizes differ"
    assert (
        0 <= max_seqlen_q <= Tq and 0 <= max_seqlen_k <= Tk
    ), f"max_seqlen_q {max_seqlen_q} and max_seqlen_k {max_seqlen_k} must be in [0, {Tq}] and [0, {Tk}]"
    for name, t in (
        ("q_descale", q_descale),
        ("k_descale", k_descale),
        ("v_descale", v_descale),
    ):
        assert (
            t.dtype == torch.float32 and t.numel() == 1 and t.device == q.device
        ), f"{name} must be a one-element float32 tensor on {q.device}, got {list(t.shape)} {t.dtype} on {t.device}"
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
    if ctas == 0:
        return (out, lse) if return_lse else out
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
