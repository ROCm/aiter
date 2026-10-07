# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the single-launch FlyDSL Mega-mHC seam (DeepSeek-V4.1, gfx950).

Same arguments and returns as the Triton seam
``aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm`` plus
``out_dtype`` ("bf16" or "fp8") and a ``config`` knob override. See
``aiter/ops/flydsl/kernels/mega_mhc.py`` for the kernel.
"""

import os
import threading
import weakref

import torch

from aiter import dtypes

__all__ = ["MEGA_MHC_DEFAULTS", "flydsl_mega_mhc", "get_mega_mhc_config"]

_GFX = ("gfx950",)
_N = 4

# knob defaults merged under every policy decision and user override
MEGA_MHC_DEFAULTS = {
    "BLOCK_M": 16,
    "WARP_SPLIT": "cols",
    "WARPS_PER_WG": 8,
    "NUM_KSPLIT": 1,
    "TILE_K": 64,
    "X1_LDS_SLOTS": 0,  # bf16 x1 chunks per warp kept in LDS for the finish
    "PERSIST_WGS": 0,  # > 0 caps the grid; each WG walks token blocks w, w + G, ...
    "FN_EARLY": 4,  # streams (of 4) of chunk 0's fn loaded ahead of the R/y prefetch
    "DIST_FINISH": False,  # every split rescales its own x1 columns (bf16, KS > 1)
    "DIST_SPIN": 1024,  # polls a waiting split spins before it hands its columns off
    "LATE_DESC": False,  # build late-use descriptors after the first loads are issued
    "SHUFFLE_DPP": False,  # Sinkhorn gate lane exchange by DPP instead of ds_swizzle
    "SEG128": False,  # 128 B-line stream layout (8 lanes per row line, TILE_K = 64)
    "NT_LD": False,  # nt hint on the R / y stream loads
    "NT_ST": False,  # nt hint on the R' / FP8 q stream stores
}

_TOKENS_CFG = {
    "BLOCK_M": 64,
    "WARP_SPLIT": "tokens",
    "WARPS_PER_WG": 4,
    "NUM_KSPLIT": 1,
    "TILE_K": 64,
}

_MAX_SPLIT = 20  # 40/80 splits measured no faster at decode (more partial rows)

# Wave quantization: a one-WG-per-CU kernel runs
# ceil(blocks / CUs) rounds, and a nearly empty last round costs ~65% of a full one
# (T = 4160: 180 us vs 111 us at T = 4096). A split-K kernel (items = blocks * KS) has
# a 25% fixed overhead at exact fill but smooth cost, so it wins while the last
# round of the KS = 1 kernel is less than FILL_MAX full. (NUM_KSPLIT, FILL_MAX)
_QUANT_SPLIT_BF16 = (5, 0.73)
_QUANT_SPLIT_FP8 = (10, 0.95)

# Distributed bf16 finish: at T = 1..3 the publish round trip costs more than the
# one-to-three-token rescale it replaces (+0.2-0.3 us); from T = 4 it wins.
_DIST_MIN_T = 4

# LATE_DESC pays only from nine token blocks up (T > 128) to ~24 blocks (T <= 384).
_LATE_DESC_T = (128, 384)

# SEG128 + nt: nt on the stream loads pays only once the bytes the seam
# moves (R, y read + R', out written) no longer fit the 256 MB MALL; below that the eager
# (MALL-hot) kernel is up to 44% slower with it. Measured boundaries at H = 5120: bf16 loses at
# T = 2496 (256 MB) and wins from 2560 (262 MB); fp8 loses at 2752 (270 MB), wins at 3072 (301 MB).
_NT_LD_MIN_BYTES = {False: 262e6, True: 300e6}  # keyed on out_fp8
_SEG128_BF16_DECODE_MIN_T = 256  # bf16 10-way decode split: T = 160 / 208 gain < 1%


def _seam_bytes(T: int, H: int, out_fp8: bool) -> float:
    """HBM bytes of one post-mix seam: R (4H) + y read, R' (4H) + bf16 x1 / FP8 q + scales."""
    return T * H * (2 * 4 + 2 + 2 * 4 + (1 + 4 / 32 if out_fp8 else 2))


def _with_seg128(T: int, H: int, cfg: dict, out_fp8: bool, decode_split: bool) -> dict:
    """``cfg`` with the 128 B-line stream layout and nt hints where they pay (see
    ``get_mega_mhc_config``)."""
    if cfg["TILE_K"] != 64:
        return cfg
    ks = cfg["NUM_KSPLIT"]
    nt_ld = _seam_bytes(T, H, out_fp8) >= _NT_LD_MIN_BYTES[out_fp8]
    if decode_split:
        if ks == 5 or (ks == 10 and not out_fp8 and T < _SEG128_BF16_DECODE_MIN_T):
            return cfg
        return dict(cfg, SEG128=True, NT_ST=True)
    if ks == 5 and not nt_ld:  # bf16 wave-quantization split at T = 2464..2559
        return cfg
    return dict(cfg, SEG128=True, NT_ST=True, NT_LD=nt_ld)


def get_mega_mhc_config(
    T: int, H: int, arch: str, cu_num: int, out_fp8: bool = False
) -> dict:
    """Launch policy keyed on (arch, cu_num) and T; thresholds tuned on MI355X at H = 5120.

    * Up to ``64 * cu_num`` tokens: 16-token blocks, 8 warps splitting the columns, and
      the largest legal ``NUM_KSPLIT <= 20`` with ``blocks * NUM_KSPLIT <= cu_num``
      (decode: 20; T = 384: 10; T = 1024: 4; T >= 4096: 1). ``TILE_K = 64`` wherever H
      divides, else 32.
    * Wave quantization: with ``NUM_KSPLIT == 1`` and a last round of workgroups less
      than 73% (bf16) / 95% (FP8) full, a 5-way (bf16) / 10-way (FP8) split-K kernel
      is used instead. A (nearly) exactly full FP8 grid with two or more rounds walks
      its blocks with ``PERSIST_WGS = cu_num``.
    * FP8 with ``NUM_KSPLIT == 1`` loads one fn stream early (``FN_EARLY = 1``); the
      FP8 32-token split kernel uses ``FN_EARLY = 2``; bf16 keeps all four.
    * bf16 with ``NUM_KSPLIT == 1`` and more than ~0.6 * cu_num token blocks keeps the
      largest fitting ``X1_LDS_SLOTS`` of x1 in LDS for the finish.
    * bf16 split-K from T = 4 uses ``DIST_FINISH`` at the largest split count whose
      workgroups are all resident (``dist_residency_error``); never with a CU mask in
      the environment. A split that waits ``DIST_SPIN`` polls hands its columns to the
      finisher, so a violated residency assumption costs time, not correctness.
    * Beyond ``64 * cu_num`` tokens: bf16 uses 64-token blocks with 4 token-split warps
      sharing each fn tile through LDS, while the last 64-token wave is >= 95% full
      (else the column kernel above); FP8 uses 32-token blocks, 8 column warps and a
      10-way split (16 tokens / 4 warps when H does not divide).
    * ``SHUFFLE_DPP`` everywhere except the bf16 ``NUM_KSPLIT == 1`` column kernel.
    * ``LATE_DESC`` for the decode split at 129 <= T <= 384 (FP8 only with 20 splits).
    * ``TILE_K == 64`` kernels use ``SEG128`` + ``NT_ST``, and ``NT_LD`` once the seam's
      HBM footprint exceeds the 256 MB MALL; not for the 5-way decode split, the bf16
      10-way split below T = 256, or the bf16 5-way split below the ``NT_LD`` footprint.
    """
    from aiter.ops.flydsl.kernels.mega_mhc import check_config, max_x1_lds_slots

    if arch not in _GFX:
        raise RuntimeError(f"[flydsl_mega_mhc] unsupported arch {arch}")
    cfg = dict(MEGA_MHC_DEFAULTS, SHUFFLE_DPP=True)
    tok_round = 64 * cu_num
    tok_fill = T / (-(-T // tok_round) * tok_round)  # fill of the last WG wave
    if T >= tok_round and (out_fp8 or tok_fill >= 0.95):
        if out_fp8:
            cfg.update(
                BLOCK_M=32,
                WARPS_PER_WG=8,
                NUM_KSPLIT=10,
                TILE_K=64,
                FN_EARLY=2,
            )
            try:
                check_config(H, cfg)
            except ValueError:  # H not divisible by 10 * 8 * 64: narrower warps
                cfg.update(
                    BLOCK_M=16, WARPS_PER_WG=4, FN_EARLY=MEGA_MHC_DEFAULTS["FN_EARLY"]
                )
        else:
            cfg.update(_TOKENS_CFG)
        cfg = _with_seg128(T, H, cfg, out_fp8, False)
        check_config(H, cfg)
        return cfg
    nblk = -(-T // 16)
    w = cfg["WARPS_PER_WG"]
    ks = 1
    for k in range(1, _MAX_SPLIT + 1):
        if H % (k * w * 32) == 0 and nblk * k <= cu_num:
            ks = k
    rounds = -(-nblk // cu_num)
    fill = nblk / (rounds * cu_num)  # how full the last round of WGs is at KS = 1
    persist = False
    decode_split = ks > 1  # the fill-the-GPU split, not the wave-quantization one
    if ks == 1:
        want, fill_max = _QUANT_SPLIT_FP8 if out_fp8 else _QUANT_SPLIT_BF16
        # below ~0.6 * CUs blocks one round is latency-bound: a split does not pay
        busy = (8 * nblk >= 5 * cu_num) if out_fp8 else (5 * nblk > 3 * cu_num)
        if (rounds > 1 or busy) and fill < fill_max:
            for k in (want, 5, 2):
                if H % (k * w * 64) == 0:
                    ks = k
                    break
        # an exactly filled FP8 grid walks its blocks in persistent workgroups: the
        # next block's first tile loads before this one finishes (2.3-3.1% at 8192)
        persist = out_fp8 and rounds > 1 and fill >= fill_max
    tk = 64 if H % (ks * w * 64) == 0 else 32
    cfg.update(BLOCK_M=16, NUM_KSPLIT=ks, TILE_K=tk)
    if ks == 1 and not out_fp8 and 5 * nblk > 3 * cu_num:
        cfg["X1_LDS_SLOTS"] = max_x1_lds_slots(H, cfg)
    if persist:
        cfg["PERSIST_WGS"] = cu_num
    elif ks == 1 and out_fp8:
        cfg["FN_EARLY"] = 1
    if not out_fp8 and ks > 1 and T >= _DIST_MIN_T:
        cfg = _with_dist_finish(T, H, cfg, cu_num)
    if not out_fp8 and ks == 1:
        cfg["SHUFFLE_DPP"] = False  # 0.8-1.2% slower with DPP
    if (
        decode_split
        and _LATE_DESC_T[0] < T <= _LATE_DESC_T[1]
        and (not out_fp8 or ks == _MAX_SPLIT)
    ):
        cfg["LATE_DESC"] = True
    cfg = _with_seg128(T, H, cfg, out_fp8, decode_split)
    try:
        check_config(H, cfg)
    except ValueError:
        if T < tok_round or out_fp8:
            raise
        cfg = dict(
            MEGA_MHC_DEFAULTS, SHUFFLE_DPP=True, **_TOKENS_CFG
        )  # H too narrow for 8 col warps
        cfg = _with_seg128(T, H, cfg, out_fp8, False)
        check_config(H, cfg)
    return cfg


def _with_dist_finish(T: int, H: int, cfg: dict, cu_num: int) -> dict:
    """``cfg`` with ``DIST_FINISH`` at the largest split count <= its own that is resident
    (``_dist_guard_error``) and legal; ``cfg`` unchanged if there is none."""
    from aiter.ops.flydsl.kernels.mega_mhc import DIST_MAX_KS, check_config

    w = cfg["WARPS_PER_WG"]
    for k in range(min(cfg["NUM_KSPLIT"], DIST_MAX_KS), 1, -1):
        if H % (k * w * 32):
            continue
        cand = dict(
            cfg,
            NUM_KSPLIT=k,
            TILE_K=64 if H % (k * w * 64) == 0 else 32,
            DIST_FINISH=True,
        )
        if _dist_guard_error(T, cand, cu_num):
            continue
        try:
            check_config(H, cand)
        except ValueError:
            continue
        return cand
    return cfg


_CAPTURE_HINT = (
    "run one eager flydsl_mega_mhc call on the capture stream with the largest "
    "token count and the same weights before capturing"
)


class _Scratch:
    """Per-(device, stream) split-K scratch, grown on demand by eager calls only.

    Counters are zeroed once at allocation; the kernel's finisher re-arms them,
    so CUDA-graph replays and back-to-back launches need no memset. Buffers
    replaced by a larger one are retained, never freed: an earlier captured
    graph may still launch on them.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.bufs = {}
        self.retired = []

    def get(self, device, stream_id, n_part_floats, n_counters, capturing):
        key = (device, stream_id)
        with self.lock:
            part, cnt = self.bufs.get(key, (None, None))
            grow_part = part is None or part.numel() < n_part_floats
            grow_cnt = cnt is None or cnt.numel() < n_counters
            if (grow_part or grow_cnt) and capturing:
                raise RuntimeError(
                    "[flydsl_mega_mhc] split-K scratch is not allocated for this "
                    f"stream during CUDA-graph capture: {_CAPTURE_HINT}"
                )
            if grow_part:
                if part is not None:
                    self.retired.append(part)
                part = torch.empty(
                    max(n_part_floats, 32), dtype=torch.float32, device=device
                )
            if grow_cnt:
                if cnt is not None:
                    self.retired.append(cnt)
                cnt = torch.zeros(max(n_counters, 32), dtype=torch.int32, device=device)
            self.bufs[key] = (part, cnt)
            return part, cnt


_SCRATCH = _Scratch()
_FN_CACHE: dict = {}


def _prepack_fn(fn: torch.Tensor, capturing: bool) -> torch.Tensor:
    """fn (24, 4H) fp32 -> bf16 [hi, lo] in MFMA B-operand order, cached per weight tensor.

    Keyed on the tensor object (dropped when it is freed, so a new weight at a
    recycled address never hits a stale pack) and checked against its version
    counter and address, so an in-place update repacks. Packing only runs
    eagerly: during capture the pack would be recorded, not computed.
    """
    key = id(fn)
    hit = _FN_CACHE.get(key)
    stamp = (fn.data_ptr(), fn._version)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    if capturing:
        raise RuntimeError(
            "[flydsl_mega_mhc] fn is not pre-packed during CUDA-graph capture: "
            f"{_CAPTURE_HINT}"
        )
    hi = fn.to(torch.bfloat16)
    lo = (fn - hi.float()).to(torch.bfloat16)
    # B operands of mfma_f32_16x16x32_bf16 in register order:
    # [H/32 chunk, stream, op, lane = kg*16 + r, 8] with op 0/1 = rows 0..15 hi/lo
    # and op 2 = rows 16..23 hi (r < 8) | lo (r >= 8); see kernels/mega_mhc.py
    H = fn.shape[1] // _N
    hi = hi.view(24, _N, H)
    lo = lo.view(24, _N, H)
    ops = torch.stack([hi[:16], lo[:16], torch.cat([hi[16:], lo[16:]])])
    x = ops.view(3, 16, _N, H // 32, 4, 8)  # op, r, s, ch, kg, i
    packed = x.permute(3, 2, 0, 4, 1, 5).contiguous()
    if hit is None:
        weakref.finalize(fn, _FN_CACHE.pop, key, None)
    _FN_CACHE[key] = (stamp, packed)
    return packed


_CU_MASK_ENV = ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")


def _dist_guard_error(T: int, cfg: dict, cu_num: int) -> str | None:
    """Why DIST_FINISH must not run (residency, CU mask env), or None."""
    from aiter.ops.flydsl.kernels.mega_mhc import dist_residency_error

    if not cfg.get("DIST_FINISH"):
        return None
    masked = [e for e in _CU_MASK_ENV if os.environ.get(e)]
    if masked:
        return f"a CU mask is set ({', '.join(masked)}): residency is unknown"
    return dist_residency_error(T, cfg, cu_num)


def _check_dist_guard(T: int, cfg: dict, cu_num: int) -> None:
    err = _dist_guard_error(T, cfg, cu_num)
    if err:
        raise ValueError(
            f"[flydsl_mega_mhc] DIST_FINISH needs every split workgroup resident: {err}"
        )


def _cu_num(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _arch(device) -> str:
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


def flydsl_mega_mhc(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None,
    sublayer_out: torch.Tensor | None = None,
    post_layer_mix: torch.Tensor | None = None,
    comb_res_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    residual_out: torch.Tensor | None = None,
    out_dtype: str = "bf16",
    config: dict | None = None,
):
    """Single-launch delayed mHC seam.

    Returns ``(residual_out, post_mix (T, 4, 1), comb_mix (T, 4, 4), layer_input,
    next_pre_mix (T, 4))``. ``layer_input`` is a (T, H) bf16 tensor, or for
    ``out_dtype="fp8"`` a tuple ``(q (T, H) fp8 e4m3, scale (T, H/32) fp32)``
    with ``layer_input ~= q * scale`` per 32-column group.

    ``residual_out`` must be preallocated when ``sublayer_out`` is given; with no
    post-mix (Engram seam) the residual is returned as is and must not be passed.

    CUDA graphs: before capturing, run one eager call on the capture stream with
    the largest token count and the same ``fn`` (as serving warm-up does). A call
    during capture that would need to pack ``fn`` or allocate split-K scratch
    raises ``RuntimeError``, so a replay is a single kernel. ``fn`` must not be
    updated in place after capture: the captured graph keeps the pack it saw.
    """
    from aiter.ops.flydsl.kernels.mega_mhc import (
        check_config,
        compile_mega_mhc,
        grid_size,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    assert out_dtype in (
        "bf16",
        "fp8",
    ), f"out_dtype must be bf16 or fp8, got {out_dtype}"
    assert residual.dim() == 3 and residual.dtype == torch.bfloat16
    assert residual.is_contiguous(), "residual must be contiguous"
    T, n, H = residual.shape
    assert n == _N, f"the Mega-mHC kernel is specialised for hc_mult=4, got {n}"
    assert H % 32 == 0
    assert fn.shape == (24, _N * H) and fn.dtype == torch.float32 and fn.is_contiguous()
    assert hc_scale.shape == (3,) and hc_base.shape == (24,)
    assert hc_scale.dtype == hc_base.dtype == torch.float32
    assert hc_scale.is_contiguous() and hc_base.is_contiguous()
    assert norm_weight is not None and norm_weight.shape == (H,)
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
    device = residual.device
    arch = _arch(device)
    if arch not in _GFX:
        raise RuntimeError(f"[flydsl_mega_mhc] supports {_GFX}, got {arch}")
    assert (
        T * _N * H * 2 < 2**31
    ), "residual exceeds the 2 GiB a buffer descriptor offset addresses"

    has_post = sublayer_out is not None
    identity_pre = pre_mix is None
    if has_post:
        assert post_layer_mix is not None and comb_res_mix is not None
        assert sublayer_out.shape == (T, H) and sublayer_out.dtype == torch.bfloat16
        assert sublayer_out.is_contiguous()
        assert post_layer_mix.numel() == T * _N and post_layer_mix.is_contiguous()
        assert comb_res_mix.numel() == T * _N * _N and comb_res_mix.is_contiguous()
        assert post_layer_mix.dtype == comb_res_mix.dtype == torch.float32
        assert residual_out is not None, "pass the preallocated residual_out"
        assert (
            residual_out.shape == residual.shape
            and residual_out.dtype == torch.bfloat16
        )
        assert residual_out.is_contiguous()
    else:
        assert residual_out is None, "no post-mix: the residual is returned as is"
        residual_out = residual
    if not identity_pre:
        assert pre_mix.numel() == T * _N and pre_mix.dtype == torch.float32
        assert pre_mix.is_contiguous()
    for t in (
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        pre_mix,
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
        residual_out,
    ):
        assert t is None or t.device == device, "all tensors must be on the same device"

    post_out = torch.empty(T, _N, 1, dtype=torch.float32, device=device)
    comb_out = torch.empty(T, _N, _N, dtype=torch.float32, device=device)
    next_pre = torch.empty(T, _N, dtype=torch.float32, device=device)
    out_fp8 = out_dtype == "fp8"
    if out_fp8:
        layer_q = torch.empty(T, H, dtype=dtypes.fp8, device=device)
        layer_s = torch.empty(T, H // 32, dtype=torch.float32, device=device)
        layer_input = (layer_q, layer_s)
    else:
        layer_q = torch.empty(T, H, dtype=torch.bfloat16, device=device)
        layer_s = layer_q
        layer_input = layer_q
    if T == 0:
        return residual_out, post_out, comb_out, layer_input, next_pre

    if config is None:
        cfg = get_mega_mhc_config(T, H, arch, _cu_num(device), out_fp8)
    else:
        unknown = sorted(set(config) - set(MEGA_MHC_DEFAULTS))
        if unknown:
            raise ValueError(f"[flydsl_mega_mhc] unknown config keys: {unknown}")
        cfg = dict(MEGA_MHC_DEFAULTS)
        cfg.update(config)
        check_config(H, cfg, out_fp8)
        _check_dist_guard(T, cfg, _cu_num(device))

    capturing = torch.cuda.is_current_stream_capturing()
    fn_arg = _prepack_fn(fn, capturing)
    nblk, n_wg = grid_size(T, cfg)
    ks = cfg["NUM_KSPLIT"]
    stream = torch.cuda.current_stream(device)
    if ks > 1:
        n_cnt_blk = n_wg // ks  # includes the XCD-mapping padding blocks
        partials, counters = _SCRATCH.get(
            device, stream.stream_id, T * ks * 32, n_cnt_blk * 32, capturing
        )
    else:
        n_cnt_blk = nblk
        partials, counters = next_pre, next_pre  # unused

    launcher = compile_mega_mhc(
        H=H,
        HAS_POST=has_post,
        IDENTITY_PRE=identity_pre,
        OUT_FP8=out_fp8,
        SINKHORN_ITERS=int(sinkhorn_repeat),
        FP8_MAX=float(torch.finfo(dtypes.fp8).max),
        **cfg,
    )
    dummy = residual
    ptrs = [
        residual,
        sublayer_out if has_post else dummy,
        post_layer_mix if has_post else dummy,
        comb_res_mix if has_post else dummy,
        dummy if identity_pre else pre_mix,
        fn_arg,
        hc_scale,
        hc_base,
        norm_weight,
        residual_out,
        layer_q,
        layer_s,
        post_out,
        comb_out,
        next_pre,
        partials,
        counters,
    ]
    _run_compiled(
        launcher,
        *[ptr_arg(t) for t in ptrs],
        T,
        n_cnt_blk,
        n_wg,
        float(rms_eps),
        float(hc_pre_eps),
        float(hc_sinkhorn_eps),
        float(hc_post_mult_value),
        float(norm_eps),
        stream,
    )
    return residual_out, post_out, comb_out, layer_input, next_pre
