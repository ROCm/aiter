# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the single-launch FlyDSL Mega-mHC seam (DeepSeek-V4.1, gfx950).

Same arguments and returns as the Triton seam
``aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm`` plus
``out_dtype`` ("bf16" or "fp8") and a ``config`` knob override. See
``aiter/ops/flydsl/kernels/mega_mhc.py`` for the kernel.
"""

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
    "WARPS_PER_SIMD": 2,
    "NUM_KSPLIT": 1,
    "TILE_K": 64,
    "COHERENCE": "none",
    "NT_STREAMS": False,  # opt-in only: no policy range wins in eager and graph
    "FN_PREPACKED": True,
    "SINKHORN_RCP": True,
}

_MAX_SPLIT = 20  # 40/80 splits measured no faster at decode (more partial rows)


def get_mega_mhc_config(
    T: int, H: int, arch: str, cu_num: int, out_fp8: bool = False
) -> dict:
    """HIP-style launch policy keyed on (arch, cu_num) and T (gfx950 sweep, MI355X).

    * Up to ``64 * cu_num`` tokens: 16-token blocks, 8 warps splitting the columns.
      Column splits fill the GPU: the largest legal ``NUM_KSPLIT <= 20`` with
      ``blocks * NUM_KSPLIT <= cu_num`` (decode: 20; T=384: 10; T=1024: 4; T >= 4096: 1).
      Splits finish on one XCD (``COHERENCE="xcd"``), measured faster than the
      agent-scope path at every size. ``TILE_K=64`` (whole 128 B lines per k-step)
      wherever H divides, else 32.
    * Beyond that, fn's L2 traffic (2 MB per token block) dominates: 64-token blocks
      with 4 token-split warps sharing each fn tile through LDS (bf16). The FP8 output
      instead keeps column-split warps with 10 column splits (32-token blocks, 8
      warps; 16-token/4-warp when H does not allow it): its finisher only rescales
      the group scales, so the split-K tail is cheap there, while the bf16 finisher
      re-reads and rewrites the whole staged collapse. The 32-token/8-warp point
      measured 2-4% faster than 16-token/4-warp at T = 16384..32768 (P1 sweep,
      ``sweep/p1_triton_geometry.md``).

    ``NT_STREAMS`` stays off at every size (D1, ``sweep/d1_nt_streams_decode.md``):
    paired off/on timing finds no (dtype, T) range where it wins by >= 2% in both
    the eager kernel time and the CUDA-graph per-launch time. Decode (T <= 192) is
    neutral to slower in eager; only the ks=10 window (T = 208..384) is faster in
    graph replay (-4..-8%) but not in eager; it is 8-40% slower from T = 416 on.
    """
    from aiter.ops.flydsl.kernels.mega_mhc import check_config

    if arch not in _GFX:
        raise RuntimeError(f"[flydsl_mega_mhc] unsupported arch {arch}")
    cfg = dict(MEGA_MHC_DEFAULTS)
    if T >= 64 * cu_num:
        if out_fp8:
            cfg.update(
                BLOCK_M=32, WARPS_PER_WG=8, NUM_KSPLIT=10, TILE_K=64, COHERENCE="xcd"
            )
            try:
                check_config(H, cfg)
            except ValueError:  # H not divisible by 10 * 8 * 64: narrower warps
                cfg.update(BLOCK_M=16, WARPS_PER_WG=4)
        else:
            cfg.update(
                BLOCK_M=64,
                WARP_SPLIT="tokens",
                WARPS_PER_WG=4,
                NUM_KSPLIT=1,
                TILE_K=64,
                COHERENCE="none",
            )
        check_config(H, cfg)
        return cfg
    nblk = -(-T // 16)
    w = cfg["WARPS_PER_WG"]
    ks = 1
    for k in range(1, _MAX_SPLIT + 1):
        if H % (k * w * 32) == 0 and nblk * k <= cu_num:
            ks = k
    tk = 64 if H % (ks * w * 64) == 0 else 32
    cfg.update(
        BLOCK_M=16, NUM_KSPLIT=ks, TILE_K=tk, COHERENCE="xcd" if ks > 1 else "none"
    )
    check_config(H, cfg)
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
        cfg = get_mega_mhc_config(T, H, arch, _cu_num(device), out_dtype == "fp8")
    else:
        cfg = dict(MEGA_MHC_DEFAULTS)
        cfg.update(config)
        check_config(H, cfg)

    capturing = torch.cuda.is_current_stream_capturing()
    fn_arg = _prepack_fn(fn, capturing) if cfg["FN_PREPACKED"] else fn
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
