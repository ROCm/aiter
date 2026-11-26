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
  from their bf16 staging copy of the fp32 master;
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
  where that kernel is missing / rejects the shape, ``qdq``.
"""

import os
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


def mxfp8_quantize(x, dim=-1):
    """E4M3 values and E8M0 scales of ``x`` in blocks of 32 along ``dim``.

    Returns ``(q, scale)``: ``q`` float8_e4m3fn shaped like ``x``; ``scale``
    uint8 (biased exponent) shaped like ``x`` with ``dim`` divided by 32.
    Bit-exact with TE's MXFP8 cast of the same tensor.
    """
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


def mxfp8_qdq(x, dim=-1, dtype=torch.float32):
    """``x`` quantized to MXFP8 along ``dim`` and dequantized (in ``dtype``).

    A ragged ``dim`` is zero-padded to a multiple of 32 for the quantization
    and cut back: the zeros leave the last block's amax unchanged.
    """
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


def _native_mm(a, b):
    """aiter FP8 MFMA GEMM (1x32 E8M0 scales, bf16 out) on ``a [M, K] @ b [K, N]``.

    K is padded to 128 and N to 16 with zeros, which leaves every block's scale
    and the product unchanged.
    """
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8_bpreshuffle
    from aiter.ops.shuffle import shuffle_weight

    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("the native MXFP8 GEMM path is gfx950-only")
    M, N = a.shape[0], b.shape[1]
    a = _pad_to(a, 1, 128)
    bt = _pad_to(_pad_to(b.t(), 1, 128), 0, 16)
    aq, a_s = mxfp8_quantize(a.contiguous(), -1)
    bq, b_s = mxfp8_quantize(bt.contiguous(), -1)
    out = gemm_a8w8_mxfp8_bpreshuffle(
        aq.view(dtypes.fp8),
        shuffle_weight(bq.view(dtypes.fp8), layout=(16, 16)),
        a_s.contiguous().view(dtypes.fp8_e8m0),
        b_s.contiguous().view(dtypes.fp8_e8m0),
        dtypes.bf16,
    )
    return out[:M, :N]


_NATIVE_ERRORS = (
    ImportError,
    NotImplementedError,
    ValueError,
    RuntimeError,
    AssertionError,
)


def mxfp8_mm(a, b, out16=False, impl=None):
    """``a @ b`` (2-D) as an MXFP8 GEMM: ``a`` quantized along its columns and
    ``b`` along its rows, in blocks of 32 of the shared K axis.

    Inputs are quantized as given: round them to the activation dtype first,
    as TE quantizes bf16 activations, gradients and weight staging copies.
    K is zero-padded to a multiple of 32 (zeros change no block's amax and add
    nothing to the sum). ``out16=True`` returns the product rounded to the
    inputs' 16-bit dtype (fprop / dgrad), else fp32 (wgrad). ``impl``
    overrides ``mxfp8_gemm_impl()``.
    """
    impl = impl or mxfp8_gemm_impl()
    bf16 = a.dtype == b.dtype == torch.bfloat16
    od = a.dtype if out16 and a.dtype in (torch.bfloat16, torch.float16) else None
    if impl != "emulate" and not (a.is_cuda and bf16):
        impl = "emulate"
    a, b = _pad_to(a, 1, MXFP8_BLOCK), _pad_to(b, 0, MXFP8_BLOCK)
    if impl == "auto" and od is not None:
        try:
            out = _native_mm(a, b)
            LAST_IMPLS["native"] += 1
            return out
        except _NATIVE_ERRORS as err:
            LAST_IMPLS[f"native_skipped:{type(err).__name__}"] += 1
    impl = "emulate" if impl == "emulate" else "qdq"
    LAST_IMPLS[impl] += 1
    dt = mxfp8_qdq_dtype(a.dtype) if impl == "qdq" else torch.float32
    out = _mm_f32(mxfp8_qdq(a, 1, dt), mxfp8_qdq(b, 0, dt))
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
