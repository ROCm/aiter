# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP8 linear-layer GEMMs with Transformer Engine's ``MXFP8BlockScaling`` numerics.

The recipe (TE ``MXFP8BlockScaling(fp8_format=E4M3)``, as Megatron-LM runs it
with ``--fp8-format e4m3 --fp8-recipe mxfp8``):

* every operand of every linear GEMM (fprop, dgrad, wgrad) is E4M3 with one
  E8M0 (power of two) scale per 32 consecutive values **along the GEMM's
  reduction axis**: fprop ``Y = X W^T`` quantizes X and W along K (rowwise),
  dgrad ``dX = dY W`` quantizes dY rowwise and W columnwise (along N), wgrad
  ``dW = dY^T X`` quantizes dY and X columnwise (along the tokens);
* each copy is quantized from the 16-bit tensor (never fp8 -> fp8); weights
  from their bf16 staging copy of the fp32 master; on GPU through aiter's
  Triton ``convert_to_mxfp8(scale_mode="te")``, else the torch port below;
* scale: ``e = float_to_e8m0(amax * fp32(1 / 448))``, the biased exponent
  rounded **up** (``cvt.rp.satfinite.ue8m0``), so ``amax / 2^(e-127) <= 448``;
* values: ``fp8_e4m3(x * 2^(127-e))``, round to nearest even, saturating;
* products accumulate in fp32; fprop and dgrad write the activation dtype
  (bf16), wgrad accumulates fp32 (``main_grad``).

``mxfp8_mm(a, b)`` computes ``a @ b`` that way, quantizing ``a`` along its
columns and ``b`` along its rows (the reduction axis of the product), so the
same call is the fprop, dgrad or wgrad of a linear layer depending on which
tensors it gets. Implementations (``set_mxfp8_gemm_impl`` /
``AITER_MXFP8_GEMM``):

* ``emulate``: dequantize to fp32 and matmul in fp32. The reference; the only
  path off GPU and for non-bf16 inputs.
* ``qdq``: the dequantized operands are exact in bf16 (4 significant bits, a
  power-of-two scale, the bf16 exponent range), so a bf16 GEMM with fp32
  accumulation on them is the MXFP8 GEMM up to summation order, with an fp32
  output.
* ``auto`` (default): aiter's FP8 MFMA GEMM with 1x32 E8M0 scales
  (``gemm_a8w8_mxfp8_bpreshuffle``, gfx950; bf16 out) when the result is
  wanted in bf16 (``out16=True``: fprop, dgrad), else (wgrad, fp32 out) or
  where that kernel is missing / rejects the shape, ``qdq``; with
  ``AITER_MXFP8_WGRAD=hipblaslt`` the fp32-out GEMMs try hipBLASLt's MXFP8
  GEMM first.

``AITER_MXFP8_FASTPATH`` (default on) selects numerics-neutral speedups, e.g.
``qdq`` in one Triton pass instead of a cast plus eager dequantization; off
restores the plain path, which the tests compare with it bit for bit.
"""

import os
import weakref
from collections import Counter

import torch
import torch.nn.functional as F

MXFP8_BLOCK = 32
E4M3_MAX = 448.0
# TE's Quantized_Limits<fp8e4m3>::max_norm_rcp: 1.0 / 448 in double, stored as float
_E4M3_MAX_RCP = torch.tensor(1.0 / E4M3_MAX, dtype=torch.float32).item()
_F32_MANT = 23

MXFP8_GEMM_IMPLS = ("auto", "qdq", "emulate")
_impl_override = None

# Which implementation ran each mxfp8_mm call since the last clear().
LAST_IMPLS: Counter = Counter()

# Numerics-neutral speedups (one-pass Triton qdq, ...). Off restores the plain
# path, which the fast one must match bit for bit.
_fastpath_override = None

# wgrad (fp32 out) implementation: "qdq" (bf16 GEMM on dequantized operands) or
# "hipblaslt" (hipBLASLt MXFP8 GEMM, VEC32_UE8M0 scales, fp32 out; opt-in)
MXFP8_WGRAD_IMPLS = ("qdq", "hipblaslt")
_TRITON_ERRORS = (ImportError, RuntimeError, ValueError, AssertionError)


def set_mxfp8_fastpath(on):
    """Force the numerics-neutral fast paths on or off; ``None`` restores
    ``AITER_MXFP8_FASTPATH`` (default on)."""
    global _fastpath_override
    _fastpath_override = None if on is None else bool(on)


def mxfp8_fastpath():
    if _fastpath_override is not None:
        return _fastpath_override
    return os.environ.get("AITER_MXFP8_FASTPATH", "1") not in ("0", "off", "false")


# Quantized operands keyed by (id(tensor), axis, op), each entry validated by a
# weakref, the tensor's version counter and data pointer: the per-call cache is
# live inside ``mxfp8_quant_cache()``; ``AITER_MXFP8_WEIGHT_CACHE=1`` keeps
# weight operands across calls too (one fp8 copy per weight and layout).
_call_cache = None
_weight_cache: dict = {}


class mxfp8_quant_cache:
    """Context in which a tensor's quantized operand (per axis and use) is
    computed once and reused; nested scopes share the outer cache."""

    def __enter__(self):
        global _call_cache
        self.outer = _call_cache
        if _call_cache is None:
            _call_cache = {}
        return self

    def __exit__(self, *exc):
        global _call_cache
        _call_cache = self.outer


def _cache_lookup(cache, key, anchor):
    hit = cache.get(key)
    if hit is None:
        return None
    ref, version, ptr, val = hit
    if ref() is anchor and anchor._version == version and anchor.data_ptr() == ptr:
        return val
    return None


def _cached(anchor, key, fn, weight=False):
    """``fn()``, reused for the same ``anchor`` tensor (unchanged) and ``key``."""
    if anchor is None or not mxfp8_fastpath():
        return fn()
    cache = _call_cache
    if weight and os.environ.get("AITER_MXFP8_WEIGHT_CACHE", "0") == "1":
        cache = _weight_cache
        for k in [k for k, v in cache.items() if v[0]() is None]:
            del cache[k]
    if cache is None:
        return fn()
    key = (id(anchor), *key)
    val = _cache_lookup(cache, key, anchor)
    if val is not None:
        LAST_IMPLS["quant_cache_hit"] += 1
        return val
    val = fn()
    cache[key] = (weakref.ref(anchor), anchor._version, anchor.data_ptr(), val)
    return val


def mxfp8_wgrad_impl():
    impl = os.environ.get("AITER_MXFP8_WGRAD", "qdq")
    if impl not in MXFP8_WGRAD_IMPLS:
        raise ValueError(f"AITER_MXFP8_WGRAD={impl!r}, expected {MXFP8_WGRAD_IMPLS}")
    return impl


def set_mxfp8_gemm_impl(impl):
    """Force an implementation for every ``mxfp8_mm``; ``None`` restores
    ``AITER_MXFP8_GEMM`` (default ``auto``)."""
    global _impl_override
    if impl is not None and impl not in MXFP8_GEMM_IMPLS:
        raise ValueError(
            f"mxfp8 GEMM impl {impl!r}, expected one of {MXFP8_GEMM_IMPLS}"
        )
    _impl_override = impl


def mxfp8_gemm_impl():
    impl = _impl_override or os.environ.get("AITER_MXFP8_GEMM", "auto")
    if impl not in MXFP8_GEMM_IMPLS:
        raise ValueError(
            f"AITER_MXFP8_GEMM={impl!r}, expected one of {MXFP8_GEMM_IMPLS}"
        )
    return impl


def float_to_e8m0(v):
    """TE ``ptx::float_to_e8m0`` on a non-negative fp32 tensor: the biased
    exponent of ``v`` rounded up, saturating at 254; NaN -> 255, 0 -> 0."""
    v = v.float()
    bits = v.view(torch.int32)
    exp = (bits >> _F32_MANT) & 0xFF
    man = bits & 0x7FFFFF
    up = (man > 0) & (exp != 0xFE) & ~((exp == 0) & (man <= 0x400000))
    e = exp + up.to(exp.dtype)
    e = torch.where(torch.isinf(v), torch.full_like(e, 0xFE), e)
    e = torch.where(torch.isnan(v), torch.full_like(e, 0xFF), e)
    return e.to(torch.uint8)


def _exp2_biased(e, negate=False):
    """fp32 ``2^(e-127)`` (or ``2^(127-e)``) of an E8M0 tensor, as TE's
    ``exp2f`` / ``exp2f_rcp``: exact powers of two, 2^-127 as a subnormal."""
    e = e.to(torch.int32)
    bits = ((254 - e) if negate else e) << _F32_MANT
    tiny = (e == 254) if negate else (e == 0)
    bits = torch.where(tiny, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(e == 255, torch.full_like(bits, 0x7FC00000), bits)
    return bits.view(torch.float32)


def _blocks(x, dim):
    """x moved so ``dim`` is last and split into [..., n / 32, 32]."""
    x = x.movedim(dim, -1)
    n = x.shape[-1]
    if n % MXFP8_BLOCK:
        raise ValueError(f"MXFP8 quantizes axes of a multiple of 32, got {n}")
    return x.reshape(*x.shape[:-1], n // MXFP8_BLOCK, MXFP8_BLOCK)


def _quantize_triton(x, dim):
    """aiter's Triton MXFP8 convert in its TE scale mode (``scale_mode="te"``);
    rows and columns are zero-padded to its 64x64 tile, which adds whole zero
    blocks only."""
    from aiter.ops.triton.quant.quant_mxfp8 import convert_to_mxfp8

    xt = x.movedim(dim, -1)
    lead, n = xt.shape[:-1], xt.shape[-1]
    x2 = xt.reshape(-1, n)
    rows = x2.shape[0]
    x2 = _pad_to(_pad_to(x2, 0, 64), 1, 64).contiguous()
    q, s = convert_to_mxfp8(x2, torch.float8_e4m3fn, scale_mode="te")
    q = q[:rows, :n].reshape(*lead, n).movedim(-1, dim)
    s = s[:rows, : n // MXFP8_BLOCK].reshape(*lead, -1).movedim(-1, dim)
    return q, s


def mxfp8_quantize(x, dim=-1, backend=None):
    """E4M3 values and E8M0 scales of ``x`` in blocks of 32 along ``dim``.

    Returns ``(q, scale)``: ``q`` float8_e4m3fn shaped like ``x``; ``scale``
    uint8 (biased exponent) shaped like ``x`` with ``dim`` divided by 32.
    Bit-exact with TE's MXFP8 cast of the same tensor. ``backend``: "triton"
    (``convert_to_mxfp8(scale_mode="te")``), "torch", or None for triton on
    GPU for fp32 / bf16 inputs (torch where it fails to build or run).
    """
    if x.shape[dim] % MXFP8_BLOCK:
        raise ValueError(f"MXFP8 quantizes axes of a multiple of 32, got {x.shape}")
    if backend is None:
        backend = (
            "triton"
            if x.is_cuda and x.dtype in (torch.float32, torch.bfloat16)
            else "torch"
        )
        if backend == "triton":
            try:
                out = _quantize_triton(x, dim)
                LAST_IMPLS["quant:triton_te"] += 1
                return out
            except (ImportError, RuntimeError, ValueError, AssertionError) as err:
                LAST_IMPLS[f"quant:torch({type(err).__name__})"] += 1
                backend = "torch"
    elif backend == "triton":
        return _quantize_triton(x, dim)
    xb = _blocks(x, dim).float()
    amax = xb.abs().amax(-1, keepdim=True)
    e = float_to_e8m0(amax * _E4M3_MAX_RCP)
    scaled = xb * _exp2_biased(e, negate=True)
    q = scaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    q = q.reshape(*q.shape[:-2], -1).movedim(-1, dim)
    return q, e.squeeze(-1).movedim(-1, dim)


def mxfp8_dequantize(q, scale, dim=-1, dtype=torch.float32):
    qb = _blocks(q.float(), dim)
    out = qb * _exp2_biased(scale.movedim(dim, -1)).unsqueeze(-1)
    return out.reshape(*out.shape[:-2], -1).movedim(-1, dim).to(dtype)


def _pad_to(x, dim, mult):
    pad = (-x.shape[dim]) % mult
    if not pad:
        return x
    widths = [0, 0] * (x.dim() - 1 - dim % x.dim()) + [0, pad]
    return F.pad(x, widths)


def _rows_triton(x2, dequant=None):
    """TE-mode Triton cast of a 2-D tensor (any strides) along its last dim.

    Returns ``(y, scale)`` sliced from the kernel's 64x64-padded outputs:
    ``y`` fp8, or the dequantized values in ``dequant``; ``scale`` uint8 with
    one column per (zero-padded) block of 32.
    """
    from aiter.ops.triton.quant.quant_mxfp8 import convert_to_mxfp8

    rows, n = x2.shape
    xp = _pad_to(_pad_to(x2, 0, 64), 1, 64)
    y, s = convert_to_mxfp8(
        xp, torch.float8_e4m3fn, scale_mode="te", dequant_dtype=dequant
    )
    return y[:rows, :n], s[:rows, : -(-n // MXFP8_BLOCK)]


def _qdq_triton(x, dim, dtype):
    """One-pass Triton ``mxfp8_qdq`` of a 2-D tensor, in the memory layout the
    plain path produces (row-major along ``dim``, rows of the length padded to
    32), so a GEMM on it sees the same operand."""
    xt = x.movedim(dim, -1)
    n = xt.shape[-1]
    n32 = -(-n // MXFP8_BLOCK) * MXFP8_BLOCK
    y, _ = _rows_triton(_pad_to(xt, 1, MXFP8_BLOCK), dequant=dtype)
    if y.stride(0) != n32:
        y = y.contiguous()
    return y[:, :n].movedim(-1, dim)


def mxfp8_qdq(x, dim=-1, dtype=torch.float32):
    """``x`` quantized to MXFP8 along ``dim`` and dequantized (in ``dtype``).

    A ragged ``dim`` is zero-padded to a multiple of 32 for the quantization
    and cut back: the zeros leave the last block's amax unchanged.
    """
    if (
        mxfp8_fastpath()
        and dtype == torch.bfloat16
        and x.is_cuda
        and x.dim() == 2
        and x.dtype in (torch.float32, torch.bfloat16)
    ):
        try:
            out = _qdq_triton(x, dim, dtype)
            LAST_IMPLS["qdq:triton_te"] += 1
            return out
        except _TRITON_ERRORS as err:
            LAST_IMPLS[f"qdq:torch({type(err).__name__})"] += 1
    n = x.shape[dim]
    xp = _pad_to(x, dim, MXFP8_BLOCK)
    out = mxfp8_dequantize(*mxfp8_quantize(xp, dim), dim=dim, dtype=dtype)
    return out.narrow(dim, 0, n) if xp.shape[dim] != n else out


def mxfp8_qdq_dtype(dtype):
    """Dtype that holds every dequantized MXFP8 value of a ``dtype`` tensor
    exactly: bf16 for bf16, else fp32."""
    return torch.bfloat16 if dtype == torch.bfloat16 else torch.float32


def _mm_f32(a, b):
    """a @ b with fp32 accumulation and output."""
    if a.dtype == b.dtype == torch.bfloat16 and a.is_cuda:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (RuntimeError, TypeError):
            pass
    return a.float() @ b.float()


def _quant_rows(x2):
    """(fp8, scale) of a 2-D tensor along its last dim, contiguous. The fast
    path reads strided views (no transpose copy); the values are the same."""
    if mxfp8_fastpath() and x2.is_cuda and x2.dtype in (torch.float32, torch.bfloat16):
        try:
            q, s = _rows_triton(x2)
            LAST_IMPLS["quant:triton_te"] += 1
            return q.contiguous(), s.contiguous()
        except _TRITON_ERRORS as err:
            LAST_IMPLS[f"quant:torch({type(err).__name__})"] += 1
    q, s = mxfp8_quantize(x2.contiguous(), -1)
    return q, s.contiguous()


def _native_mm(a, b, anchors=None):
    """aiter FP8 MFMA GEMM (1x32 E8M0 scales, bf16 out) on ``a [M, K] @ b [K, N]``.

    K is padded to 128 and N to 16 with zeros, which leaves every block's scale
    and the product unchanged. ``anchors`` = (tensor_a, axis_a, tensor_b,
    axis_b, b_is_weight) names the tensors the operands are views of, for the
    quantized-operand cache.
    """
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("the native MXFP8 GEMM path is gfx950-only")
    ta, xa, tb, xb, b_weight = anchors or (None,) * 5
    M, N = a.shape[0], b.shape[1]

    def quant_a():
        aq, a_s = _quant_rows(_pad_to(a, 1, 128))
        return aq.view(dtypes.fp8), a_s.view(dtypes.fp8_e8m0)

    def quant_b():
        bq, b_s = _quant_rows(_pad_to(_pad_to(b.t(), 1, 128), 0, 16))
        bq = shuffle_weight(bq.view(dtypes.fp8), layout=(16, 16))
        return bq, b_s.view(dtypes.fp8_e8m0)

    aq, a_s = _cached(ta, (xa, "native_a"), quant_a)
    bq, b_s = _cached(tb, (xb, "native_b"), quant_b, weight=b_weight)
    out = gemm_a8w8_mxfp8_bpreshuffle(aq, bq, a_s, b_s, dtypes.bf16)
    return out[:M, :N]


_NATIVE_ERRORS = (
    ImportError,
    NotImplementedError,
    ValueError,
    RuntimeError,
    AssertionError,
)
_HIPB_READY = False


def _hipb_mm_fp32(a, b):
    """``a [M, K] @ b [K, N]`` on hipBLASLt with 1x32 E8M0 scales on both
    operands along K (VEC32_UE8M0) and an fp32 output. K is padded to 128 and
    M, N to 32 with zeros (no block's scale or the product changes)."""
    global _HIPB_READY
    import aiter
    from aiter import dtypes
    from aiter.ops.gradlib import HipblasLtMatmulScaleMode as ScaleMode

    if not _HIPB_READY:
        aiter.hipb_create_extension()
        _HIPB_READY = True
    M, N = a.shape[0], b.shape[1]
    a = _pad_to(_pad_to(a, 1, 128), 0, MXFP8_BLOCK)
    bt = _pad_to(_pad_to(b.t(), 1, 128), 0, MXFP8_BLOCK)
    aq, a_s = (t.contiguous() for t in _rows_triton(a))
    bq, b_s = (t.contiguous() for t in _rows_triton(bt))
    out = aiter.hipb_mm(
        aq.view(dtypes.fp8),
        bq.view(dtypes.fp8).t(),
        solution_index=-1,
        out_dtype=torch.float32,
        scaleA=a_s,
        scaleB=b_s,
        scaleA_mode=ScaleMode.VEC32_UE8M0,
        scaleB_mode=ScaleMode.VEC32_UE8M0,
    )
    return out[:M, :N]


def mxfp8_mm(a, b, out16=False, impl=None, anchors=None):
    """``a @ b`` (2-D) as an MXFP8 GEMM: ``a`` quantized along its columns and
    ``b`` along its rows, in blocks of 32 of the shared K axis.

    Inputs are quantized as given: round them to the activation dtype first,
    as TE quantizes bf16 activations, gradients and weight staging copies.
    K is zero-padded to a multiple of 32 (zeros change no block's amax and add
    nothing to the sum). ``out16=True`` returns the product rounded to the
    inputs' 16-bit dtype (fprop / dgrad), else fp32 (wgrad). ``impl``
    overrides ``mxfp8_gemm_impl()``. ``anchors`` = (tensor_a, axis_a,
    tensor_b, axis_b, b_is_weight): the tensors ``a`` / ``b`` are views of and
    the axis their blocks run along, for the quantized-operand cache (None for
    either tensor: not cached).
    """
    impl = impl or mxfp8_gemm_impl()
    bf16 = a.dtype == b.dtype == torch.bfloat16
    od = a.dtype if out16 and a.dtype in (torch.bfloat16, torch.float16) else None
    if impl != "emulate" and not (a.is_cuda and bf16):
        impl = "emulate"
    a, b = _pad_to(a, 1, MXFP8_BLOCK), _pad_to(b, 0, MXFP8_BLOCK)
    if impl == "auto" and od is not None:
        try:
            out = _native_mm(a, b, anchors)
            LAST_IMPLS["native"] += 1
            return out
        except _NATIVE_ERRORS as err:
            LAST_IMPLS[f"native_skipped:{type(err).__name__}"] += 1
    if impl == "auto" and od is None and mxfp8_wgrad_impl() == "hipblaslt":
        try:
            out = _hipb_mm_fp32(a, b)
            LAST_IMPLS["hipblaslt_fp32"] += 1
            return out
        except _NATIVE_ERRORS as err:
            LAST_IMPLS[f"hipblaslt_skipped:{type(err).__name__}"] += 1
    impl = "emulate" if impl == "emulate" else "qdq"
    LAST_IMPLS[impl] += 1
    dt = mxfp8_qdq_dtype(a.dtype) if impl == "qdq" else torch.float32
    ta, xa, tb, xb, b_weight = anchors or (None,) * 5
    aq = _cached(ta, (xa, "qdq", dt), lambda: mxfp8_qdq(a, 1, dt))
    bq = _cached(tb, (xb, "qdq", dt), lambda: mxfp8_qdq(b, 0, dt), weight=b_weight)
    out = _mm_f32(aq, bq)
    return out if od is None else out.to(od)


class _MXFP8Matmul(torch.autograd.Function):
    """``a @ b`` as a TE MXFP8 linear: operands and the incoming gradient
    rounded to ``cd`` and quantized along each GEMM's reduction axis; the
    output and ``da`` (fprop, dgrad) rounded to ``cd``, ``db`` (wgrad) fp32."""

    @staticmethod
    def forward(ctx, a, b, cd, impl):
        ctx.save_for_backward(a, b)
        ctx.cd, ctx.impl = cd, impl
        return mxfp8_mm(a.to(cd), b.to(cd), out16=True, impl=impl).float()

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        cd, impl = ctx.cd, ctx.impl
        g = g.to(cd)
        da = db = None
        if ctx.needs_input_grad[0]:
            da = mxfp8_mm(g, b.to(cd).t(), out16=True, impl=impl).float()
        if ctx.needs_input_grad[1]:
            db = mxfp8_mm(a.to(cd).t(), g, impl=impl)
        return da, db, None, None


def mxfp8_matmul(a, b, cd=torch.bfloat16, impl="emulate"):
    """Differentiable ``a @ b`` (a: [..., K], b: [K, N]) with the MXFP8 linear
    recipe in forward and backward (see ``_MXFP8Matmul``); fp32 tensors out.
    ``cd`` is the activation dtype; ``impl`` defaults to the fp32 emulation."""
    lead = a.shape[:-1]
    out = _MXFP8Matmul.apply(a.reshape(-1, a.shape[-1]), b, cd, impl)
    return out.view(*lead, b.shape[-1])
