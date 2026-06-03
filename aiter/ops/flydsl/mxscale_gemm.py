# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Low-level FlyDSL backend wrapper for the gfx1250 MXScale dense GEMM kernel.

``flydsl_mxscale_gemm`` is the backend launch primitive covering ``data_format``
``"fp8"`` (MXFP8 E4M3 + E8M0).

This module is the codegen/launch layer. The public GEMM entry is
``aiter.gemm_a8w8_bpreshuffle``; MXFP8 routes here when its tuned config uses a
``flydsl_mxscale_*`` kernel name. The codegen / tuning knobs below are
tuner-internal and are *not* part of the public op surface.

The B weight is prepared with the ordinary ``shuffle_weight`` 16x16 byte
preshuffle (shape-preserving); MXScale needs no bespoke weight metadata. K is
never padded — the selected kernel must divide K; only M is padded at runtime.
"""

from __future__ import annotations

import functools
import re
from typing import Optional

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

from .mxscale_layout import (
    SCALE_BLOCK,
    align_up,
    mxscale_pack_factors,
    preshuffle_mxscale_activation,
    preshuffle_mxscale_scale_for_kernel,
    to_kernel_uint8,
    validate_mxscale_kernel_shape,
)
from .utils import is_flydsl_available

# Sentinels for keeping the import surface small when flydsl is missing.
_compile_mxscale_gemm = None  # type: ignore[assignment]
_run_compiled = None  # type: ignore[assignment]
_fx = None  # type: ignore[assignment]

_TARGET_GFX = "gfx1250"
_VALID_FORMATS = ("fp8", "a8w4")
_VALID_OUT_DTYPES = ("bf16", "f16", "f32")

_TORCH_DTYPE_FROM_NAME = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}
_NAME_FROM_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_FROM_NAME.items()}


def _resolve_target_device(*tensors: Optional[Tensor]) -> torch.device:
    cuda_devices = []
    for tensor in tensors:
        if tensor is None or not tensor.is_cuda:
            continue
        if tensor.device not in cuda_devices:
            cuda_devices.append(tensor.device)
    if len(cuda_devices) > 1:
        devices = ", ".join(str(device) for device in cuda_devices)
        raise ValueError(f"all MXScale tensors must use one CUDA device, got {devices}")
    if cuda_devices:
        return cuda_devices[0]
    if not torch.cuda.is_available():
        raise RuntimeError("flydsl_mxscale_gemm requires an available CUDA device")
    return torch.device("cuda", torch.cuda.current_device())


def _to_target_device(tensor: Tensor, device: torch.device) -> Tensor:
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


def _lazy_import_flydsl():
    """Import flydsl-dependent symbols lazily so the module loads without flydsl."""
    global _compile_mxscale_gemm, _run_compiled, _fx
    if _compile_mxscale_gemm is not None:
        return
    if not is_flydsl_available():
        raise RuntimeError(
            "flydsl is not installed; install the matching flydsl wheel to use "
            "flydsl_mxscale_gemm."
        )
    import flydsl.expr as fx_mod

    from .kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm as _compile
    from .kernels.tensor_shim import _run_compiled as _runner

    _compile_mxscale_gemm = _compile
    _run_compiled = _runner
    _fx = fx_mod


# ---------------------------------------------------------------------------
# Internal codegen constants
# ---------------------------------------------------------------------------
#
# Experimental / arch-locked / debug knobs of the underlying kernel, pinned at
# safe defaults. They are NOT tuner-searched, NOT part of any signature, and NOT
# encoded in the kernel name. They are applied to ``compile_mxscale_gemm`` only
# if that kernel still accepts them (see ``mxscale_compile_kwargs``), so the
# kernel can freely drop / rename / add an experimental knob and this
# integration — and every caller — stays unaffected: dropping or renaming a
# pinned knob silently falls back to the kernel's own default, and a newly added
# knob simply keeps its default. The public ops never see any of this.
_FIXED_CODEGEN = {
    "use_scale_opsel": False,
    "wave_specialized_tdm": True,
    "l2_prefetch_distance": 0,
    "waves_per_eu": 0,
    "inst_prefetch": False,
    "expert_sched_mode": True,
    "atomic_barrier_enable": False,
    "b_streaming": False,
    # Must stay "tdm": mxscale_layout.preshuffle_e8m0_scale_wmma produces the
    # tdm (non-coalesced) scale layout. vgpr / vgpr_ab_split need a different
    # preshuffle and would silently corrupt scales if selected here.
    "scale_load_path": "tdm",
    "fp8_schedule": "auto",
}


@functools.lru_cache(maxsize=1)
def _compile_accepted_params() -> frozenset:
    """Parameter names accepted by the current ``compile_mxscale_gemm`` kernel."""
    import inspect

    from .kernels.gemm_fp8fp4_gfx1250 import compile_mxscale_gemm

    return frozenset(inspect.signature(compile_mxscale_gemm).parameters)


def mxscale_compile_kwargs(
    *,
    data_format: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    split_k: int,
    cluster_m: int = 1,
    cluster_n: int = 1,
) -> dict:
    """Build the ``compile_mxscale_gemm`` kwargs from the tuner perf knobs.

    Tuner-searched knobs (always passed): tile_m/n/k, m_warp, n_warp,
    num_buffers, split_k, cluster_m, cluster_n. ``use_tdm_store`` is derived from
    ``split_k``. Experimental flags from ``_FIXED_CODEGEN`` are added only when
    the kernel still accepts them, keeping the glue robust to kernel updates.
    """
    kwargs = {
        "data_format": data_format,
        "out_dtype": out_dtype,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "m_warp": m_warp,
        "n_warp": n_warp,
        "num_buffers": num_buffers,
        "split_k": split_k,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
    }
    accepted = _compile_accepted_params()
    # Experimental flags + the derived use_tdm_store go through the accepted
    # filter so the kernel can drop / rename any of them without breaking us.
    kwargs.update({k: v for k, v in _FIXED_CODEGEN.items() if k in accepted})
    if "use_tdm_store" in accepted:
        # split_k > 1 accumulates via buffer stores (atomic add), not TDM store.
        kwargs["use_tdm_store"] = split_k == 1
    # waves_per_eu == 0 means "no hint"; the kernel builder wants None for that.
    # Done here (post-filter) so dropping the param from the kernel can't break us.
    if kwargs.get("waves_per_eu", None) == 0:
        kwargs["waves_per_eu"] = None
    return kwargs


# ---------------------------------------------------------------------------
# kernelName encode / decode
# ---------------------------------------------------------------------------
#
# The kernel name encodes ONLY the tuner perf knobs — experimental codegen
# flags are pinned in ``_FIXED_CODEGEN`` and never serialized:
#   flydsl_mxscale_{fmt}_{out}_t{tm}x{tn}x{tk}_mw{mw}_nw{nw}_buf{buf}_sk{sk}
#     _cm{cm}_cn{cn}_gfx1250
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_mxscale_"
    r"(?P<fmt>fp8|a8w4)_"
    r"(?P<out>bf16|f16|f32)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_buf(?P<num_buffers>\d+)_"
    r"sk(?P<split_k>\d+)_cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)_"
    r"(?P<target_gfx>gfx1250)$"
)


def flydsl_mxscale_kernel_name(
    *,
    data_format: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    split_k: int,
    cluster_m: int = 1,
    cluster_n: int = 1,
    target_gfx: str = _TARGET_GFX,
) -> str:
    """Encode a kernel name from the tuner perf knobs (no experimental flags)."""
    if data_format not in _VALID_FORMATS:
        raise ValueError(
            f"data_format must be one of {_VALID_FORMATS}, got {data_format!r}"
        )
    if out_dtype not in _VALID_OUT_DTYPES:
        raise ValueError(
            f"out_dtype must be one of {_VALID_OUT_DTYPES}, got {out_dtype!r}"
        )
    return (
        f"flydsl_mxscale_{data_format}_{out_dtype}_"
        f"t{tile_m}x{tile_n}x{tile_k}_mw{m_warp}_nw{n_warp}_buf{num_buffers}_"
        f"sk{split_k}_cm{cluster_m}_cn{cluster_n}_{target_gfx}"
    )


def parse_flydsl_mxscale_kernel_name(name: str) -> Optional[dict]:
    """Parse a mxscale kernel name into its perf-knob config, or None on miss.

    Returns ``kind="mxscale"`` plus the tuner perf knobs. Experimental codegen
    flags are not encoded — apply ``mxscale_compile_kwargs`` to expand to the
    full ``compile_mxscale_gemm`` argument set.
    """
    m = _KERNEL_NAME_RE.fullmatch(name)
    if m is None:
        return None
    return {
        "kind": "mxscale",
        "data_format": m.group("fmt"),
        "out_dtype": m.group("out"),
        "tile_m": int(m.group("tile_m")),
        "tile_n": int(m.group("tile_n")),
        "tile_k": int(m.group("tile_k")),
        "m_warp": int(m.group("m_warp")),
        "n_warp": int(m.group("n_warp")),
        "num_buffers": int(m.group("num_buffers")),
        "split_k": int(m.group("split_k")),
        "cluster_m": int(m.group("cluster_m")),
        "cluster_n": int(m.group("cluster_n")),
        "target_gfx": m.group("target_gfx"),
    }


# ---------------------------------------------------------------------------
# Public runtime entry
# ---------------------------------------------------------------------------


def _resolve_out_dtype(
    out: Optional[Tensor], out_dtype: Optional[str]
) -> tuple[str, torch.dtype]:
    """Resolve (name, torch.dtype) for the output, checking out/out_dtype agree."""
    if out is not None:
        torch_dt = out.dtype
        name = _NAME_FROM_TORCH_DTYPE.get(torch_dt)
        if name is None:
            raise ValueError(
                f"out tensor dtype {torch_dt} not supported; expected one of "
                f"{list(_TORCH_DTYPE_FROM_NAME.values())}"
            )
        if out_dtype is not None and out_dtype != name:
            raise ValueError(
                f"out_dtype={out_dtype!r} conflicts with out.dtype={torch_dt}"
            )
        return name, torch_dt
    if out_dtype is None:
        out_dtype = "bf16"
    if out_dtype not in _TORCH_DTYPE_FROM_NAME:
        raise ValueError(
            f"out_dtype must be one of {_VALID_OUT_DTYPES}, got {out_dtype!r}"
        )
    return out_dtype, _TORCH_DTYPE_FROM_NAME[out_dtype]


def mxscale_out_dtype_name(dtype: torch.dtype) -> str:
    """Return the MXScale kernel-name dtype token for a torch output dtype."""
    name = _NAME_FROM_TORCH_DTYPE.get(dtype)
    if name is None:
        raise ValueError(
            f"unsupported MXScale out dtype {dtype}; expected bf16/f16/f32"
        )
    return name


def flydsl_mxscale_gemm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    *,
    data_format: str = "fp8",
    out: Optional[Tensor] = None,
    out_dtype: Optional[str] = None,
    tile_m: int = 128,
    tile_n: int = 32,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 2,
    num_buffers: int = 2,
    split_k: int = 1,
    cluster_m: int = 1,
    cluster_n: int = 1,
    kernel_name: Optional[str] = None,
) -> Tensor:
    """Run a FlyDSL gfx1250 MXScale GEMM (data_format ∈ {"fp8", "a8w4"}).

    Only structural args and the tuner perf knobs (tile / warp / num_buffers /
    split_k / cluster_m / cluster_n) are accepted. All experimental codegen
    flags are pinned internally in ``_FIXED_CODEGEN`` and are intentionally not
    part of this signature.

    Parameters
    ----------
    A : (M, K) FP8 E4M3 byte storage.
    B : (N, K // pack_b) weight pre-shuffled with the ordinary 16x16
        ``shuffle_weight`` (shape-preserving; no K padding, no metadata).
    A_scale : (M, K // 32) E8M0 (uint8 storage).
    B_scale : (N, K // 32) E8M0 (uint8 storage), raw row-major.

    Notes
    -----
    * If ``out`` is provided it must have shape ``(M, N)``. The wrapper pads M
      at runtime; K is never padded. N and K must be divided by the selected
      kernel tile (``validate_mxscale_kernel_shape`` raises otherwise).
    * ``split_k > 1`` uses buffer-store atomic accumulation (derived internally)
      and zero-fills the padded output before launch.
    * ``kernel_name`` may be provided when the dispatch was already resolved
      against a tuned CSV; the perf knobs from it override the keyword arguments.
      When ``None`` a kernel name is synthesised from the kwargs.
    """
    if data_format not in _VALID_FORMATS:
        raise ValueError(
            f"data_format must be one of {_VALID_FORMATS}, got {data_format!r}"
        )
    cur_gfx = get_gfx()
    if cur_gfx != _TARGET_GFX:
        raise RuntimeError(
            f"flydsl_mxscale_gemm requires {_TARGET_GFX}, current arch is {cur_gfx!r}"
        )
    _lazy_import_flydsl()

    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(
            f"A and B must be 2-D, got A.shape={tuple(A.shape)}, B.shape={tuple(B.shape)}"
        )
    M = A.shape[0]

    # If a kernel name was provided, take its perf knobs verbatim.
    if kernel_name is not None:
        parsed = parse_flydsl_mxscale_kernel_name(kernel_name)
        if parsed is None:
            raise ValueError(f"unrecognised mxscale kernel_name: {kernel_name!r}")
        if parsed["data_format"] != data_format:
            raise ValueError(
                f"kernel_name data_format={parsed['data_format']!r} != "
                f"data_format={data_format!r}"
            )
        tile_m = parsed["tile_m"]
        tile_n = parsed["tile_n"]
        tile_k = parsed["tile_k"]
        m_warp = parsed["m_warp"]
        n_warp = parsed["n_warp"]
        num_buffers = parsed["num_buffers"]
        split_k = parsed["split_k"]
        cluster_m = parsed["cluster_m"]
        cluster_n = parsed["cluster_n"]
        out_dtype = parsed["out_dtype"]

    # Reconcile out.dtype with the out_dtype string (a kernel_name parse above
    # may already have set out_dtype); they must agree.
    out_dtype_name, out_torch_dtype = _resolve_out_dtype(out, out_dtype)

    # All MX operands are 1-byte (fp8 / e8m0); reject mistyped tensors up front
    # since to_kernel_uint8 would otherwise reinterpret wider dtypes silently.
    for name, t in (("A", A), ("B", B), ("A_scale", A_scale), ("B_scale", B_scale)):
        if t.element_size() != 1:
            raise ValueError(f"{name} must be 1-byte storage, got dtype {t.dtype}")

    # ----- Recover logical shapes (no padding). B is shape-preserving-shuffled. -----
    pack_a, pack_b = mxscale_pack_factors(data_format)
    K = A.shape[1]
    N = B.shape[0]
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
    if K != B.shape[1] * pack_b:
        raise ValueError(
            f"A and B contraction dimensions disagree: A.shape[1]={K} "
            f"vs B.shape[1]={B.shape[1]} (pack_b={pack_b})"
        )
    if A_scale.shape != (M, K // SCALE_BLOCK):
        raise ValueError(
            f"A_scale shape must be {(M, K // SCALE_BLOCK)}, got {tuple(A_scale.shape)}"
        )
    if B_scale.shape != (N, K // SCALE_BLOCK):
        raise ValueError(
            f"B_scale shape must be {(N, K // SCALE_BLOCK)}, got {tuple(B_scale.shape)}"
        )

    # K is never padded: the kernel must divide (N, K). Raises (no fallback).
    validate_mxscale_kernel_shape(
        N=N,
        K=K,
        tile_n=tile_n,
        tile_k=tile_k,
        num_buffers=num_buffers,
        split_k=split_k,
    )
    # The host scale preshuffle below produces the "tdm" scale layout. If a kernel
    # update drops scale_load_path (filtered out -> kernel default) and that
    # default is not "tdm", the layouts would silently disagree — guard loudly.
    assert _FIXED_CODEGEN["scale_load_path"] == "tdm" and (
        "scale_load_path" in _compile_accepted_params()
    ), "preshuffle_e8m0_scale_wmma assumes scale_load_path='tdm'; kernel changed it"
    target_device = _resolve_target_device(A, B, A_scale, B_scale, out)

    if out is not None and tuple(out.shape) != (M, N):
        raise ValueError(f"out shape must be {(M, N)}, got {tuple(out.shape)}")
    if out is not None and out.device != target_device:
        raise ValueError(
            f"out must be on the MXScale launch device {target_device}, "
            f"got {out.device}"
        )

    # ----- Runtime-private A prepare and B-scale kernel swizzle (M pad only) -----
    compile_shape = {
        "M": align_up(M, tile_m),
        "N": N,
        "K": K,
        "K_scale": K // SCALE_BLOCK,
        "pack_a": pack_a,
        "pack_b": pack_b,
    }
    b_p = B
    b_s_p = preshuffle_mxscale_scale_for_kernel(
        B_scale,
        tile_n=tile_n,
        tile_k=tile_k,
        n_warp=n_warp,
    )
    a_p, a_s_p = preshuffle_mxscale_activation(
        A,
        A_scale,
        data_format=data_format,
        tile_m=tile_m,
        tile_k=tile_k,
        m_warp=m_warp,
    )

    a_dev = _to_target_device(a_p, target_device)
    b_dev = _to_target_device(b_p, target_device)
    a_s_dev = _to_target_device(a_s_p, target_device)
    b_s_dev = _to_target_device(b_s_p, target_device)

    # ----- Allocate padded out + (optional) zero-init for split-K -----
    # split_k > 1 accumulates partials via atomic add. Rounding every partial
    # into a bf16/f16 output buffer loses precision (error grows with split_k),
    # so accumulate in an f32 scratch buffer and cast once at the end -- this
    # matches the single-shot (split_k == 1) output precision.
    acc_f32_scratch = split_k > 1 and out_dtype_name != "f32"
    compile_out_dtype = "f32" if acc_f32_scratch else out_dtype_name
    out_buf = torch.empty(
        (compile_shape["M"], compile_shape["N"]),
        dtype=torch.float32 if acc_f32_scratch else out_torch_dtype,
        device=a_dev.device,
    )
    if split_k > 1:
        out_buf.zero_()

    # ----- Compile + launch -----
    compile_kwargs = mxscale_compile_kwargs(
        data_format=data_format,
        out_dtype=compile_out_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        split_k=split_k,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
    )
    launch_fn = _compile_mxscale_gemm(
        M=compile_shape["M"],
        N=compile_shape["N"],
        K=compile_shape["K"],
        **compile_kwargs,
    )
    stream = _fx.Stream(torch.cuda.current_stream(device=a_dev.device))
    _run_compiled(
        launch_fn,
        out_buf.contiguous().view(-1),
        to_kernel_uint8(a_dev),
        to_kernel_uint8(b_dev),
        to_kernel_uint8(a_s_dev),
        to_kernel_uint8(b_s_dev),
        compile_shape["M"],
        compile_shape["N"],
        stream,
    )

    # ----- Slice padded buffer back to (M, N) + cast f32 scratch to target -----
    sliced = (
        out_buf
        if (compile_shape["M"], compile_shape["N"]) == (M, N)
        else out_buf[:M, :N]
    )
    if out is not None:
        out.copy_(sliced)
        return out
    if sliced.dtype != out_torch_dtype:
        return sliced.to(out_torch_dtype)
    return sliced.contiguous()


__all__ = [
    "flydsl_mxscale_gemm",
    "flydsl_mxscale_kernel_name",
    "parse_flydsl_mxscale_kernel_name",
    "mxscale_out_dtype_name",
]
