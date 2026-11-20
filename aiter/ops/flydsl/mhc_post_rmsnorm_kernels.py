# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrappers of two hidden-5120 mHC kernels (DeepSeek-V4.1-Flash, gfx950).

``flydsl_mhc_post_pre_rmsnorm``: the delayed mHC seam in one launch (post-mix, gate
projection, Sinkhorn gates, collapse with the carried pre gate, RMSNorm), with
optional fused MXFP8 output and the mean of the four new streams (the auxiliary
hidden state a speculative draft reads). Same arguments as ``flydsl_mega_mhc`` and
``aiter.ops.mega_mhc.mega_mhc`` except that ``fn`` comes pre-packed by
``flydsl_mhc_pack_fn``. Tuned for decode row counts; see
``flydsl_mhc_post_pre_rmsnorm_supported`` for the shapes it takes,
``flydsl_mhc_post_pre_rmsnorm_preferred`` for where it is faster than Mega-MHC and
``warmup_flydsl_mhc_post_pre_rmsnorm`` to compile the variants before capture.

``flydsl_mhc_post_collapse_rmsnorm``: the step-end tail of the mHC residual in one
launch (post-mix of the last sub-layer, collapse with the carried pre gate, final
RMSNorm), optionally also storing the new residual and the mean of the streams.

Both kernels are specialised for hc_mult = 4 and hidden = 5120 and validated on
gfx950 only; the ``*_supported`` helpers let callers fall back.
"""

import os
import struct
import threading

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

__all__ = [
    "flydsl_mhc_pack_fn",
    "flydsl_mhc_post_collapse_rmsnorm",
    "flydsl_mhc_post_collapse_rmsnorm_supported",
    "flydsl_mhc_post_pre_rmsnorm",
    "flydsl_mhc_post_pre_rmsnorm_preferred",
    "flydsl_mhc_post_pre_rmsnorm_supported",
    "prepare_flydsl_mhc_post_pre_rmsnorm",
    "warmup_flydsl_mhc_post_pre_rmsnorm",
]

_GFX = ("gfx950",)
_N = 4
_H = 5120
_N_MIX = 24
_CHUNKS = _H // 32
_BLOCK_M = 16
_PSLOT = 32
_CTRL_INTS = 16 + _BLOCK_M
# Seam row limit: the per-block counters are sized for it.
MHC_POST_PRE_RMSNORM_MAX_TOKENS = 4096
_OUT_DTYPES = ("bf16", "mxfp8")
_CU_MASK_ENV = ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")

# (row blocks of 16 at most, split count, waves per workgroup), from graph-replay
# sweeps on MI355X (256 CUs, 8 XCDs). Past the table: the largest split of 4 / 2 / 1
# that fits the XCD bound below, 8 waves. Every entry compiles without scratch
# (private_segment_fixed_size 0) for all 16 variants: 16-wave workgroups get only
# 128 VGPRs and spill once a wave owns more than one chunk, so none is used.
_TUNED = ((8, 20, 8), (16, 16, 10), (24, 10, 8), (32, 8, 5), (48, 5, 8))
# Workgroup b runs on XCD b % 8 (the kernel's split-K mapping relies on it).
_XCDS = 8
# A wave owning more chunks than this stages x1 in the output instead of registers.
_STAGE_CHUNKS = 8
# Bounded wait of a split-K workgroup before it hands off: None is the kernel's
# DIST_SPIN (1024 polls). Tests set 0 to force the hand-off path.
_SEAM_SPIN = None


def _require(cond: bool, msg: str, exc: type = ValueError) -> None:
    if not cond:
        raise exc(msg)


def _f32_bits(x: float) -> int:
    return struct.unpack("<i", struct.pack("<f", float(x)))[0]


def _cu_num(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _cu_masked() -> bool:
    return any(os.environ.get(e) for e in _CU_MASK_ENV)


def _xcd_fits(blocks: int, ks: int, cu_num: int) -> bool:
    """Every real workgroup of ``blocks`` row blocks x ``ks`` splits has a CU of its
    XCD: row block b and all its splits run on XCD b % 8."""
    return -(-blocks // _XCDS) * ks <= cu_num // _XCDS


def _seam_config(m: int, cu_num: int) -> tuple[int, int, bool]:
    """(split count, waves, stage x1) for ``m`` rows.

    Split-K workgroups of a row block wait for the block's last arrival. The wait
    is bounded (a split that waits too long hands its columns to the last arrival
    and exits), so residency is a speed matter, not a correctness one. For speed
    every real workgroup gets a CU at once: a row block and all its splits run on
    one XCD, so the bound is per XCD (``_xcd_fits``), not ``blocks * splits <= CUs``
    (51 blocks x 5 splits fit the CU count but put 35 workgroups on 3 of the XCDs'
    32 CUs: 816 rows took 54 us instead of 39 us with 4 splits). The grid is padded to
    whole groups of 8 row blocks; padding workgroups exit at once.
    """
    blocks = -(-m // _BLOCK_M)
    ks = waves = None
    for bound, k, w in _TUNED:
        if blocks <= bound:
            if _xcd_fits(blocks, k, cu_num):
                ks, waves = k, w
            break
    if ks is None:
        ks = max(k for k in (1, 2, 4) if k == 1 or _xcd_fits(blocks, k, cu_num))
        waves = 8
    return ks, waves, _CHUNKS // ks // waves > _STAGE_CHUNKS


def flydsl_mhc_post_pre_rmsnorm_supported(
    num_tokens: int, hidden_size: int, hc_mult: int, device=None
) -> bool:
    """True when ``flydsl_mhc_post_pre_rmsnorm`` takes this shape on this GPU:
    gfx950, hc_mult 4, hidden 5120, 0 <= num_tokens <= 4096.

    A CU mask (``HSA_CU_MASK`` / ``ROC_GLOBAL_CU_MASK``) does not change the answer:
    the split-K wait is bounded, so the kernel stays correct with fewer CUs, but
    it was not tuned for that (``flydsl_mhc_post_pre_rmsnorm_preferred`` is False).
    """
    del device  # aiter assumes a homogeneous node
    return (
        get_gfx_runtime() in _GFX
        and hc_mult == _N
        and hidden_size == _H
        and 0 <= num_tokens <= MHC_POST_PRE_RMSNORM_MAX_TOKENS
    )


# Measured on MI355X, CUDA-graph replay of the post-mix seam with carried pre gate,
# bf16 and mxfp8 output, against HIP mega_mhc and flydsl_mega_mhc.
# Plain seam: faster than both at 129 ... 256 rows (4-7 %), within 5 % (mostly
# faster) at 16 ... 128 rows, slower from 288 rows.
_PREFERRED_TOKENS = (129, 256)
# aux_mean seam: faster than the cheapest alternative (a Mega-MHC seam plus the
# stream mean of its residual_out) up to this many rows (1.04x at 1024 rows,
# 0.85x at 1152).
_AUX_PREFERRED_MAX = 1024


def flydsl_mhc_post_pre_rmsnorm_preferred(
    num_tokens: int, aux_mean: bool = False
) -> bool:
    """Measured dispatch rule against ``mega_mhc`` / ``flydsl_mega_mhc`` on gfx950.

    Without ``aux_mean``: 129 ... 256 rows. With ``aux_mean`` the alternative is a
    Mega-MHC seam plus the mean of its ``residual_out``: 1 ... 1024 rows. False
    under a CU mask (not tuned there). Callers must also check
    ``flydsl_mhc_post_pre_rmsnorm_supported``.
    """
    if _cu_masked():
        return False
    if aux_mean:
        return 0 < num_tokens <= _AUX_PREFERRED_MAX
    return _PREFERRED_TOKENS[0] <= num_tokens <= _PREFERRED_TOKENS[1]


def flydsl_mhc_post_collapse_rmsnorm_supported(
    num_tokens: int, hidden_size: int, hc_mult: int, device=None
) -> bool:
    """True when ``flydsl_mhc_post_collapse_rmsnorm`` takes this shape on this GPU:
    gfx950, hc_mult 4, hidden 5120, any num_tokens >= 0 (the residual must stay
    below 2 GiB)."""
    del device
    return (
        get_gfx_runtime() in _GFX
        and hc_mult == _N
        and hidden_size == _H
        and 0 <= num_tokens
        and num_tokens * _N * _H * 2 < 2**31
    )


def flydsl_mhc_pack_fn(fn: torch.Tensor) -> torch.Tensor:
    """fp32 ``fn`` (24, 4 * 5120) -> the seam's bf16 hi / lo MFMA B operands.

    Run once per layer weight, at load time (not under graph capture). Returns a
    contiguous bf16 tensor (4, 160, 3, 4, 16, 8): per (stream, 32-column chunk)
    hi of mixes 0-15, lo of mixes 0-15, and hi | lo of mixes 16-23.
    """
    _require(
        fn.shape == (_N_MIX, _N * _H), f"fn must be ({_N_MIX}, {_N * _H})", ValueError
    )
    _require(fn.dtype == torch.float32, "fn must be fp32", TypeError)
    with torch.no_grad():
        fn = fn.contiguous()
        hi = fn.to(torch.bfloat16)
        lo = (fn - hi.float()).to(torch.bfloat16)
        hi = hi.view(_N_MIX, _N, _CHUNKS, 4, 8)
        lo = lo.view(_N_MIX, _N, _CHUNKS, 4, 8)
        # [op, mix, stream, chunk, group, j] -> [stream, chunk, op, group, mix, j]
        ops = torch.stack([hi[:16], lo[:16], torch.cat([hi[16:], lo[16:]])])
        return ops.permute(2, 3, 0, 4, 1, 5).contiguous()


class _Scratch:
    """Per-(device, stream) split-K scratch, allocated at full size once.

    The counters are zeroed at allocation and re-arm themselves in the kernel, so
    graph replays and back-to-back launches need no memset. Calls sharing a stream
    run serially and share it; seams on two streams use separate scratch, and may
    overlap: a split that cannot get its block finished in time hands off instead
    of waiting.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.bufs = {}

    def get(self, device, stream_id: int):
        key = (device, stream_id)
        with self.lock:
            buf = self.bufs.get(key)
            if buf is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "[flydsl_mhc_post_pre_rmsnorm] split-K scratch is not "
                        "allocated for this stream during CUDA-graph capture: call "
                        "prepare_flydsl_mhc_post_pre_rmsnorm (or one eager seam) on "
                        "the capture stream first"
                    )
                part = torch.empty(
                    _cu_num(device) * _BLOCK_M * _PSLOT,
                    dtype=torch.float32,
                    device=device,
                )
                ctrl = torch.zeros(
                    MHC_POST_PRE_RMSNORM_MAX_TOKENS // _BLOCK_M * _CTRL_INTS,
                    dtype=torch.int32,
                    device=device,
                )
                buf = self.bufs[key] = (part, ctrl)
            return buf


_SCRATCH = _Scratch()


def _device_of(device):
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def prepare_flydsl_mhc_post_pre_rmsnorm(device=None) -> None:
    """Allocate the seam's split-K scratch on the current stream of ``device``.

    Needed once per stream before CUDA-graph capture on that stream (one eager seam
    call, or ``warmup_flydsl_mhc_post_pre_rmsnorm``, does the same). Weight
    independent.
    """
    device = _device_of(device)
    _SCRATCH.get(device, torch.cuda.current_stream(device).stream_id)


def warmup_flydsl_mhc_post_pre_rmsnorm(
    max_tokens: int,
    *,
    out_dtype: str = "bf16",
    aux_mean: bool = False,
    has_post: bool = True,
    identity_pre: bool = False,
    sinkhorn_repeat: int = 20,
    device=None,
) -> list[tuple[int, int, bool]]:
    """Compile (and run once, on zero inputs) every split configuration the seam
    picks for 1 ... ``max_tokens`` rows of one variant, on the current stream.

    Each configuration and variant is a separate kernel that otherwise compiles on
    its first call; call this before CUDA-graph capture, once per variant the model
    captures (``sinkhorn_repeat`` is a compile-time constant too). Also prepares the
    stream's split-K scratch. Returns the (split count, waves, stage x1)
    configurations it compiled.
    """
    _require(
        not torch.cuda.is_current_stream_capturing(),
        "warm up before CUDA-graph capture",
        RuntimeError,
    )
    _require(
        0 <= max_tokens <= MHC_POST_PRE_RMSNORM_MAX_TOKENS,
        f"max_tokens must be in [0, {MHC_POST_PRE_RMSNORM_MAX_TOKENS}]",
    )
    device = _device_of(device)
    cu = _cu_num(device)
    rows = {}
    for blocks in range(1, -(-max_tokens // _BLOCK_M) + 1):
        rows.setdefault(
            _seam_config(blocks * _BLOCK_M, cu), min(blocks * _BLOCK_M, max_tokens)
        )
    if not rows:
        prepare_flydsl_mhc_post_pre_rmsnorm(device)
        return []
    T = max(rows.values())
    with torch.no_grad():
        z = torch.zeros(T, _N, _H, dtype=torch.bfloat16, device=device)
        fn_packed = torch.zeros(
            _N, _CHUNKS, 3, 4, 16, 8, dtype=torch.bfloat16, device=device
        )
        sc = torch.zeros(3, device=device)
        base = torch.zeros(_N_MIX, device=device)
        w = torch.zeros(_H, dtype=torch.bfloat16, device=device)
        pre = torch.zeros(T, _N, device=device)
        y = torch.zeros(T, _H, dtype=torch.bfloat16, device=device)
        pm = torch.zeros(T, _N, 1, device=device)
        cm = torch.zeros(T, _N, _N, device=device)
        rout = torch.empty_like(z)
        for t in rows.values():
            post = (y[:t], pm[:t], cm[:t]) if has_post else (None, None, None)
            flydsl_mhc_post_pre_rmsnorm(
                z[:t], fn_packed, sc, base, 1e-6, 1e-6, 1e-6, 1.0,
                sinkhorn_repeat, None if identity_pre else pre[:t], *post, w, 1e-6,
                residual_out=rout[:t] if has_post else None,
                out_dtype=out_dtype, aux_mean=aux_mean,
            )  # fmt: skip
    return list(rows)


def _check_post(T, H, sublayer_out, post_layer_mix, comb_res_mix):
    _require(
        post_layer_mix is not None and comb_res_mix is not None,
        "sublayer_out needs post_layer_mix and comb_res_mix",
    )
    _require(sublayer_out.shape == (T, H), f"sublayer_out must be ({T}, {H})")
    _require(
        sublayer_out.dtype == torch.bfloat16, "sublayer_out must be bf16", TypeError
    )
    _require(sublayer_out.is_contiguous(), "sublayer_out must be contiguous")
    _require(
        post_layer_mix.numel() == T * _N and post_layer_mix.is_contiguous(),
        f"post_layer_mix must be {T * _N} contiguous values",
    )
    _require(
        comb_res_mix.numel() == T * _N * _N and comb_res_mix.is_contiguous(),
        f"comb_res_mix must be {T * _N * _N} contiguous values",
    )
    _require(
        post_layer_mix.dtype == comb_res_mix.dtype == torch.float32,
        "post_layer_mix and comb_res_mix must be fp32",
        TypeError,
    )


def _check_residual(residual):
    _require(residual.dim() == 3, "residual must be (T, 4, H)")
    _require(residual.dtype == torch.bfloat16, "residual must be bf16", TypeError)
    _require(residual.is_contiguous(), "residual must be contiguous")
    T, n, H = residual.shape
    _require(n == _N and H == _H, f"specialised for (T, {_N}, {_H}), got {(T, n, H)}")
    _require(
        get_gfx_runtime() in _GFX,
        f"supports {_GFX}, got {get_gfx_runtime()}",
        RuntimeError,
    )
    return T


def _check_norm_weight(norm_weight):
    _require(
        norm_weight is not None and norm_weight.shape == (_H,),
        f"norm_weight must be ({_H},)",
    )
    _require(norm_weight.dtype == torch.bfloat16, "norm_weight must be bf16", TypeError)
    _require(norm_weight.is_contiguous(), "norm_weight must be contiguous")


def _check_device(device, *tensors):
    for t in tensors:
        _require(
            t is None or t.device == device, "all tensors must be on the same device"
        )


def flydsl_mhc_post_pre_rmsnorm(
    residual: torch.Tensor,
    fn_packed: torch.Tensor,
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
    aux_mean: bool = False,
):
    """Delayed mHC seam (hidden 5120, hc_mult 4) in one launch.

    Args:
        residual: (T, 4, 5120) bf16 residual streams entering the seam.
        fn_packed: ``flydsl_mhc_pack_fn(fn)`` of the fp32 (24, 4 * 5120) gate weight.
        hc_scale, hc_base: (3,) and (24,) fp32.
        pre_mix: (T, 4) fp32 pre gate carried from the previous seam; None collapses
            to stream 0 (the first layer).
        sublayer_out, post_layer_mix, comb_res_mix: (T, 5120) bf16 output of the
            previous sub-layer and its seam's (T, 4[, 1]) / (T, 4, 4) fp32 gates; all
            None skips the post-mix (the residual is used as is).
        norm_weight: (5120,) bf16 RMSNorm weight of the next sub-layer.
        residual_out: (T, 4, 5120) bf16 buffer for the new residual (post-mix only;
            allocated when None). Must not alias ``residual``.
        out_dtype: "bf16" or "mxfp8" (also e4m3 codes and row-major E8M0 scales of the
            normed layer input, bit-exact with quantizing the bf16 output per 32
            columns: scale byte = clamp(ceil(log2(amax / 448)) + 127, 0, 254)).
        aux_mean: also return the bf16 mean of the four new streams.

    Returns:
        ``(residual_out, post_mix (T, 4, 1), comb_mix (T, 4, 4), layer_input,
        next_pre_mix (T, 4))`` plus ``(mean (T, 5120),)`` with ``aux_mean``.
        ``layer_input`` is the bf16 (T, 5120) norm, or ``(norm, codes e4m3 (T, 5120),
        scales uint8 (T, 160))`` for "mxfp8". Without a post-mix ``residual_out`` is
        ``residual``.

    Calls on one stream share a split-K scratch and must run serially. Before CUDA
    graph capture on a stream, call ``warmup_flydsl_mhc_post_pre_rmsnorm`` (compiles
    the variants too), ``prepare_flydsl_mhc_post_pre_rmsnorm`` or one eager seam on
    it. Split-K workgroups of a row block wait for the block's last arrival for a
    bounded time and then hand their columns to it, so the kernel completes even
    when they are not all resident (CU masks, concurrent kernels), only slower.
    """
    from aiter.ops.flydsl.kernels.mhc_post_pre_rmsnorm import (
        DIST_SPIN,
        compile_mhc_post_pre_rmsnorm,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    _require(out_dtype in _OUT_DTYPES, f"out_dtype must be one of {_OUT_DTYPES}")
    T = _check_residual(residual)
    _require(
        T <= MHC_POST_PRE_RMSNORM_MAX_TOKENS,
        f"at most {MHC_POST_PRE_RMSNORM_MAX_TOKENS} tokens, got {T}",
    )
    _require(
        fn_packed.dtype == torch.bfloat16
        and fn_packed.shape == (_N, _CHUNKS, 3, 4, 16, 8)
        and fn_packed.is_contiguous(),
        "fn_packed must come from flydsl_mhc_pack_fn",
    )
    _require(
        hc_scale.shape == (3,) and hc_base.shape == (_N_MIX,),
        "hc_scale (3,) and hc_base (24,)",
    )
    _require(
        hc_scale.dtype == hc_base.dtype == torch.float32,
        "hc_scale and hc_base must be fp32",
        TypeError,
    )
    _require(
        hc_scale.is_contiguous() and hc_base.is_contiguous(),
        "hc_scale / hc_base must be contiguous",
    )
    _require(int(sinkhorn_repeat) >= 1, "sinkhorn_repeat must be >= 1")
    _check_norm_weight(norm_weight)
    device = residual.device
    has_post = sublayer_out is not None
    if has_post:
        _check_post(T, _H, sublayer_out, post_layer_mix, comb_res_mix)
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        _require(
            residual_out.shape == residual.shape
            and residual_out.dtype == torch.bfloat16
            and residual_out.is_contiguous(),
            "residual_out must be a contiguous bf16 tensor shaped like residual",
        )
        _require(
            residual_out.data_ptr() != residual.data_ptr() or T == 0,
            "residual_out must not alias residual",
        )
    else:
        _require(
            post_layer_mix is None and comb_res_mix is None and residual_out is None,
            "no post-mix: post_layer_mix, comb_res_mix and residual_out must be None",
        )
        residual_out = residual
    ident_pre = pre_mix is None
    if not ident_pre:
        _require(
            pre_mix.numel() == T * _N
            and pre_mix.dtype == torch.float32
            and pre_mix.is_contiguous(),
            f"pre_mix must be {T * _N} contiguous fp32 values",
        )
    _check_device(
        device,
        fn_packed,
        hc_scale,
        hc_base,
        norm_weight,
        pre_mix,
        sublayer_out,
        post_layer_mix,
        comb_res_mix,
        residual_out,
    )

    quant = out_dtype == "mxfp8"
    post = torch.empty(T, _N, 1, dtype=torch.float32, device=device)
    comb = torch.empty(T, _N, _N, dtype=torch.float32, device=device)
    next_pre = torch.empty(T, _N, dtype=torch.float32, device=device)
    norm = torch.empty(T, _H, dtype=torch.bfloat16, device=device)
    codes = (
        torch.empty(T, _H, dtype=torch.float8_e4m3fn, device=device) if quant else None
    )
    scales = (
        torch.empty(T, _H // 32, dtype=torch.uint8, device=device) if quant else None
    )
    mean = torch.empty(T, _H, dtype=torch.bfloat16, device=device) if aux_mean else None
    layer_input = (norm, codes, scales) if quant else norm
    res = (residual_out, post, comb, layer_input, next_pre)
    if aux_mean:
        res = res + (mean,)
    if T == 0:
        return res

    stream = torch.cuda.current_stream(device)
    part, ctrl = _SCRATCH.get(device, stream.stream_id)
    ks, waves, stage = _seam_config(T, _cu_num(device))
    spin = DIST_SPIN if _SEAM_SPIN is None else _SEAM_SPIN
    launcher = compile_mhc_post_pre_rmsnorm(
        ks,
        waves,
        has_post,
        ident_pre,
        quant,
        aux_mean,
        int(sinkhorn_repeat),
        stage,
        spin,
    )

    def ptr(t):
        return ptr_arg(residual if t is None else t)

    _run_compiled(
        launcher,
        *[
            ptr(t)
            for t in (
                residual,
                fn_packed,
                sublayer_out,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                hc_scale,
                hc_base,
                norm_weight,
                residual_out,
                post,
                comb,
                next_pre,
                norm,
                codes,
                scales,
                mean,
                part,
                ctrl,
            )
        ],
        T,
        _f32_bits(rms_eps),
        _f32_bits(hc_pre_eps),
        _f32_bits(hc_sinkhorn_eps),
        _f32_bits(hc_post_mult_value),
        _f32_bits(norm_eps),
        stream,
        # one compiled handle per device: a handle does not launch on another one
        specialization_key=device.index,
    )
    return res


def flydsl_mhc_post_collapse_rmsnorm(
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    sublayer_out: torch.Tensor | None = None,
    post_layer_mix: torch.Tensor | None = None,
    comb_res_mix: torch.Tensor | None = None,
    *,
    residual_out: torch.Tensor | None = None,
    keep_residual: bool = False,
    aux_mean: bool = False,
):
    """mHC step-end tail (hidden 5120, hc_mult 4) in one launch.

    ``R' = post-mix(residual, sublayer_out)`` (skipped when ``sublayer_out`` is
    None), ``x1 = bf16(sum_j pre_mix_j * R'_j)``, ``out = bf16(x1 * rsqrt(mean(x1^2)
    + norm_eps) * norm_weight)``.

    Args:
        residual: (T, 4, 5120) bf16.
        pre_mix: (T, 4) fp32 pre gate carried from the last seam.
        norm_weight: (5120,) bf16 final norm weight.
        sublayer_out, post_layer_mix, comb_res_mix: last sub-layer output (T, 5120)
            bf16 and its seam's fp32 gates; all None when ``residual`` already is
            the new residual.
        residual_out / keep_residual: store R' (post-mix only) into ``residual_out``
            (allocated when None and ``keep_residual``).
        aux_mean: also return the bf16 mean of the four streams of R'.

    Returns:
        ``(layer_input (T, 5120) bf16, R' or None, mean or None)``.
    """
    from aiter.ops.flydsl.kernels.mhc_post_collapse_rmsnorm import (
        compile_mhc_post_collapse_rmsnorm,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    T = _check_residual(residual)
    _require(T * _N * _H * 2 < 2**31, "residual must stay below 2 GiB")
    _require(
        pre_mix.numel() == T * _N
        and pre_mix.dtype == torch.float32
        and pre_mix.is_contiguous(),
        f"pre_mix must be {T * _N} contiguous fp32 values",
    )
    _check_norm_weight(norm_weight)
    device = residual.device
    has_post = sublayer_out is not None
    keep = keep_residual or residual_out is not None
    if has_post:
        _check_post(T, _H, sublayer_out, post_layer_mix, comb_res_mix)
        if keep:
            if residual_out is None:
                residual_out = torch.empty_like(residual)
            _require(
                residual_out.shape == residual.shape
                and residual_out.dtype == torch.bfloat16
                and residual_out.is_contiguous(),
                "residual_out must be a contiguous bf16 tensor shaped like residual",
            )
    else:
        _require(
            post_layer_mix is None and comb_res_mix is None and not keep,
            "no post-mix: post_layer_mix, comb_res_mix and residual_out must be "
            "None and keep_residual False",
        )
    _check_device(
        device, pre_mix, norm_weight, sublayer_out, post_layer_mix, comb_res_mix
    )
    _check_device(device, residual_out)

    out = torch.empty(T, _H, dtype=torch.bfloat16, device=device)
    mean = torch.empty(T, _H, dtype=torch.bfloat16, device=device) if aux_mean else None
    if T:

        def ptr(t):
            return ptr_arg(out if t is None else t)

        _run_compiled(
            compile_mhc_post_collapse_rmsnorm(has_post, keep, aux_mean),
            *[
                ptr(t)
                for t in (
                    residual,
                    sublayer_out,
                    post_layer_mix,
                    comb_res_mix,
                    pre_mix,
                    norm_weight,
                    residual_out,
                    out,
                    mean,
                )
            ],
            T,
            _f32_bits(norm_eps),
            torch.cuda.current_stream(device),
            specialization_key=device.index,
        )
    return out, residual_out, mean
