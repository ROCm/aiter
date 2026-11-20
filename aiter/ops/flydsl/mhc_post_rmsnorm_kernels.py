# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrappers of two hidden-5120 mHC kernels (DeepSeek-V4.1-Flash, gfx950).

``flydsl_mhc_post_pre_rmsnorm``: the delayed mHC seam in one launch (post-mix, gate
projection, Sinkhorn gates, collapse with the carried pre gate, RMSNorm), with
optional fused MXFP8 output and the mean of the four new streams (the auxiliary
hidden state a speculative draft reads). Same arguments as ``flydsl_mega_mhc`` and
``aiter.ops.mega_mhc.mega_mhc`` except that ``fn`` comes pre-packed by
``flydsl_mhc_pack_fn``. Tuned for decode row counts (16 ... 768); see
``flydsl_mhc_post_pre_rmsnorm_supported`` for the shapes it takes and
``flydsl_mhc_post_pre_rmsnorm_preferred`` for where it is faster than Mega-MHC.

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
# sweeps at 16 ... 768 rows on MI355X (256 CUs). Past the table: the largest split
# of 1 / 2 / 4 that keeps blocks * splits within the CU count.
_TUNED = ((8, 20, 8), (16, 16, 10), (24, 10, 16), (32, 8, 5), (51, 5, 8))


def _require(cond: bool, msg: str, exc: type = ValueError) -> None:
    if not cond:
        raise exc(msg)


def _f32_bits(x: float) -> int:
    return struct.unpack("<i", struct.pack("<f", float(x)))[0]


def _cu_num(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _cu_masked() -> bool:
    return any(os.environ.get(e) for e in _CU_MASK_ENV)


def _seam_config(m: int, cu_num: int, cu_masked: bool) -> tuple[int, int]:
    """(split count, waves) for ``m`` rows.

    Split-K workgroups of a row block wait for the block's last arrival, so every
    split must be resident: ``blocks * splits`` stays within the CU count, and
    with a CU mask (residency unknown) the kernel runs unsplit.
    """
    blocks = -(-m // _BLOCK_M)
    if cu_masked:
        return 1, 16
    for bound, ks, waves in _TUNED:
        if blocks <= bound:
            if blocks * ks <= cu_num:
                return ks, waves
            break
    ks = max(k for k in (1, 2, 4) if blocks * k <= cu_num or k == 1)
    return ks, 8 if ks > 1 else 16


def flydsl_mhc_post_pre_rmsnorm_supported(
    num_tokens: int, hidden_size: int, hc_mult: int, device=None
) -> bool:
    """True when ``flydsl_mhc_post_pre_rmsnorm`` takes this shape on this GPU:
    gfx950, hc_mult 4, hidden 5120, 0 <= num_tokens <= 4096."""
    del device  # aiter assumes a homogeneous node
    return (
        get_gfx_runtime() in _GFX
        and hc_mult == _N
        and hidden_size == _H
        and 0 <= num_tokens <= MHC_POST_PRE_RMSNORM_MAX_TOKENS
    )


# Rows where the plain seam beats both Mega-MHC kernels (MI355X, CUDA-graph replay of
# the post-mix seam, bf16 and mxfp8 output): 6-15 % faster at 160 ... 256 rows, within
# 2 % at 16 ... 128 rows, 11-25 % slower from 288 rows.
_PREFERRED_TOKENS = (129, 256)


def flydsl_mhc_post_pre_rmsnorm_preferred(
    num_tokens: int, aux_mean: bool = False
) -> bool:
    """Measured dispatch rule against ``mega_mhc`` / ``flydsl_mega_mhc`` on gfx950.

    With ``aux_mean`` the alternative is a Mega-MHC seam plus a separate post-mix and
    mean launch, which this kernel beats at every supported row count; without it,
    prefer this kernel for 129 ... 256 rows. Callers must also check
    ``flydsl_mhc_post_pre_rmsnorm_supported``.
    """
    if aux_mean:
        return 0 < num_tokens <= MHC_POST_PRE_RMSNORM_MAX_TOKENS
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
    graph replays and back-to-back launches need no memset. Calls sharing a
    stream run serially; seams on two streams use separate scratch.
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


def prepare_flydsl_mhc_post_pre_rmsnorm(device=None) -> None:
    """Allocate the seam's split-K scratch on the current stream of ``device``.

    Needed once per stream before CUDA-graph capture on that stream (one eager seam
    call does the same). Weight independent.
    """
    device = torch.device("cuda") if device is None else torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _SCRATCH.get(device, torch.cuda.current_stream(device).stream_id)


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
    graph capture on a stream, call ``prepare_flydsl_mhc_post_pre_rmsnorm`` or one
    eager seam on it. Split-K workgroups of a row block wait for each other, which
    needs them resident at once: with ``HSA_CU_MASK`` / ``ROC_GLOBAL_CU_MASK`` set
    the kernel runs without split-K.
    """
    from aiter.ops.flydsl.kernels.mhc_post_pre_rmsnorm import (
        compile_mhc_post_pre_rmsnorm,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

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
    ks, waves = _seam_config(T, _cu_num(device), _cu_masked())
    launcher = compile_mhc_post_pre_rmsnorm(
        ks, waves, has_post, ident_pre, quant, aux_mean, int(sinkhorn_repeat)
    )
    dummy = residual.data_ptr()

    def ptr(t):
        return dummy if t is None else t.data_ptr()

    _run_compiled(
        launcher,
        residual.data_ptr(),
        fn_packed.data_ptr(),
        ptr(sublayer_out),
        ptr(post_layer_mix),
        ptr(comb_res_mix),
        ptr(pre_mix),
        hc_scale.data_ptr(),
        hc_base.data_ptr(),
        norm_weight.data_ptr(),
        residual_out.data_ptr(),
        post.data_ptr(),
        comb.data_ptr(),
        next_pre.data_ptr(),
        norm.data_ptr(),
        ptr(codes),
        ptr(scales),
        ptr(mean),
        part.data_ptr(),
        ctrl.data_ptr(),
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
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

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
        dummy = out.data_ptr()

        def ptr(t):
            return dummy if t is None else t.data_ptr()

        _run_compiled(
            compile_mhc_post_collapse_rmsnorm(has_post, keep, aux_mean),
            residual.data_ptr(),
            ptr(sublayer_out),
            ptr(post_layer_mix),
            ptr(comb_res_mix),
            pre_mix.data_ptr(),
            norm_weight.data_ptr(),
            ptr(residual_out),
            out.data_ptr(),
            ptr(mean),
            T,
            _f32_bits(norm_eps),
            torch.cuda.current_stream(device),
            specialization_key=device.index,
        )
    return out, residual_out, mean
