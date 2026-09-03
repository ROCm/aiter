# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dense weight-only int4 GEMM (a16w4) for gfx1201 (Navi 48).

    out = x @ dequant(weight)
    dequant(weight)[k, n] = (nibble(k, n) - zeros[k // G, n]) * scales[k // G, n]

Tensor layout, both dtypes:

    x        [M, K]        bf16 or fp16
    weight   [K/8, N]      int32, 8 nibbles of ONE column packed along K
    scales   [K/G, N]      same dtype as x
    zeros    [K/G, N]      same dtype as x
    out      [M, N]        same dtype as x

Note the weight packing is along **K**, whereas an AWQ/GPTQ checkpoint ships
[K, N//8] packed along **N**. Use `pack_a16w4_weight()` on the raw nibbles, or
repack at load time -- never per token.

The fp16 path additionally needs MAGIC-permuted nibbles and +1024-biased
zeros; `prepare_a16w4_weight()` applies both. The two weight layouts are NOT
interchangeable and nothing downstream can detect a mix-up, so the dtype of
`x` is the only switch.
"""

import functools

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

MD_NAME = "module_gemm_a16w4"

SUPPORTED_GFX = ("gfx1201",)

# int4 quantisation group along K. Mirrors a16w4::kGroupSize in
# csrc/gemm_a16w4/include/gemm_a16w4_launch.h -- fixed by the checkpoint
# format, not a tuning knob.
GROUP_SIZE = 128
# int4 nibbles packed per int32 of `weight`.
PACK_K = 8

# +1024 magic bias for the fp16 dequant path. NOT arbitrary: fp16 has a
# 10-bit mantissa, so 0x6400 | n == 1024 + n exactly for any 4-bit n, and
# 1024 + n and 1024 + z both lie in the binade [1024, 2048) where fp16
# subtraction is exact by Sterbenz's lemma. The dequant therefore rounds ONCE
# (the multiply by the scale), which is why the fp16 kernels measure ~15x
# LOWER max error than the bf16 ones rather than higher.
MAGIC_BIAS = 1024.0
# k -> bit position, p(k) = 4*(k>>1) + 16*(k&1), so the kernel's four
# extractions ((packed >> 4j) & 0x000F000F) | 0x64006400 yield k = 0..7 in
# order.
_MAGIC_SHIFTS = [4 * (j >> 1) + 16 * (j & 1) for j in range(8)]


# develop=True is load-bearing, not a debug flag: it converts the torch.Tensor
# arguments to the pybind aiter_tensor_t ABI the C++ side takes, AND calls
# module._set_current_hip_stream(), which is what makes aiter::getCurrentHIPStream()
# return the caller's stream instead of the null stream.
@compile_ops(MD_NAME, fc_name="gemm_a16w4", develop=True)
def _gemm_a16w4(
    x: Tensor,
    weight: Tensor,
    scales: Tensor,
    zeros: Tensor,
    out: Tensor,
    workspace: Tensor,
) -> None: ...


@compile_ops(MD_NAME)
def gemm_a16w4_unsupported_reason(M: int, N: int, K: int, is_fp16: bool) -> str: ...


@compile_ops(MD_NAME)
def gemm_a16w4_workspace_elems(M: int, N: int, K: int, is_fp16: bool) -> int: ...


def is_a16w4_available() -> bool:
    """True when the running GPU is one this kernel was built and tuned for."""
    try:
        return get_gfx_runtime() in SUPPORTED_GFX
    except Exception:  # noqa: BLE001 - no GPU / unknown arch is just "no"
        return False


@functools.lru_cache(maxsize=256)
def _plan(m: int, n: int, k: int, is_fp16: bool) -> tuple[str, int]:
    """(unsupported_reason, workspace_elems) for one shape.

    Both are pure host arithmetic on the C++ side, but they still cross the
    pybind boundary, so they are cached: a decode step is ~30 us end to end
    and would otherwise pay for two module calls per linear layer.
    """
    reason = gemm_a16w4_unsupported_reason(m, n, k, is_fp16)
    if reason:
        return reason, 0
    return "", gemm_a16w4_workspace_elems(m, n, k, is_fp16)


def gemm_a16w4(
    x: Tensor,
    weight: Tensor,
    scales: Tensor,
    zeros: Tensor,
    out: Tensor | None = None,
    workspace: Tensor | None = None,
) -> Tensor:
    """Run the a16w4 GEMM, allocating `out` and the split-K scratch if needed.

    Args:
        x: [M, K] bf16 or fp16 activations, contiguous.
        weight: [K/8, N] int32 int4 weights packed along K. For fp16 `x` these
            must be MAGIC-permuted -- see `prepare_a16w4_weight()`.
        scales: [K/128, N], same dtype as `x`.
        zeros: [K/128, N], same dtype as `x`. For fp16 `x` these must already
            carry the +1024 magic bias.
        out: optional preallocated [M, N] output, same dtype as `x`.
        workspace: optional fp32 scratch of at least
            `gemm_a16w4_workspace_elems(M, N, K, is_fp16)` elements. Only the
            decode path uses it; prefill needs none.

    Returns:
        [M, N] tensor, `out` when one was passed.

    Raises:
        RuntimeError: on a non-gfx1201 GPU, or for a shape no tile covers
            (the reason names the failing constraint).

    Note:
        Pass both buffers on the decode path. The kernel itself is ~24 us at
        M=1 N=K=5120, so allocating a fresh output and scratch per call --
        even from the caching allocator -- measurably dominates it. A serving
        stack should hoist them the way it hoists any other per-layer buffer.
    """
    if not is_a16w4_available():
        raise RuntimeError(
            f"gemm_a16w4 is {'/'.join(SUPPORTED_GFX)} only; "
            f"running on {get_gfx_runtime()}"
        )

    m, k = x.shape
    n = weight.shape[1]
    is_fp16 = x.dtype == torch.float16

    reason, ws_elems = _plan(m, n, k, is_fp16)
    if reason:
        raise RuntimeError(
            f"gemm_a16w4 does not support M={m} N={n} K={k} ({x.dtype}): {reason}"
        )

    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if workspace is None:
        # Sized by the C++ side so the split-K factor has exactly one
        # definition. C++ re-checks numel(), so a short buffer raises rather
        # than corrupting memory.
        workspace = torch.empty(ws_elems, dtype=torch.float32, device=x.device)

    _gemm_a16w4(x, weight, scales, zeros, out, workspace)
    return out


def pack_a16w4_weight(nibbles: Tensor) -> Tensor:
    """[K, N] uint8 nibbles (0..15) -> [K/8, N] int32 packed along K.

    Nibble for row k lands at bit 4*(k%8), which is what the bf16 kernels
    read. One packed int32 is exactly the 8 elements a lane needs for a
    16x16x16 B fragment.

    One-off, at weight load. Not on the inference path.
    """
    k, n = nibbles.shape
    if k % PACK_K:
        raise ValueError(f"K must be a multiple of {PACK_K}, got {k}")
    v = nibbles.to(torch.int64).view(k // PACK_K, PACK_K, n)
    shifts = (torch.arange(PACK_K, device=nibbles.device, dtype=torch.int64) * 4).view(
        1, PACK_K, 1
    )
    return (v << shifts).sum(dim=1).to(torch.int32)


def repack_a16w4_weight_magic(weight: Tensor) -> Tensor:
    """[K/8, N] int32 packed along K -> the same, in MAGIC bit order.

    Pure bit shuffle inside each int32; no data movement between rows. Needed
    only by the fp16 kernels, whose dequant extracts two nibbles at a time out
    of the 16-bit halves of a dword.
    """
    src = weight.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(src)
    for k in range(PACK_K):
        out |= ((src >> (4 * k)) & 0xF) << _MAGIC_SHIFTS[k]
    return out.to(torch.int32)


def bias_a16w4_zeros_magic(zeros: Tensor) -> Tensor:
    """zeros -> fp16(zeros + 1024), the form the fp16 kernels subtract.

    Folding the bias into the scale instead -- w = h*s - (1024+z)*s as one
    v_pk_fma_f16 -- is one instruction cheaper and WRONG: it subtracts two
    ~1024*s quantities to make a ~15*s result, amplifying fp16's 2^-11
    relative error by 1024/15 to about 3.3%. That measured cosine 0.99947,
    which passes a loose tolerance. Do not "optimise" it back.
    """
    return (zeros.float() + MAGIC_BIAS).to(torch.float16)


def prepare_a16w4_weight(
    nibbles: Tensor, scales: Tensor, zeros: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor]:
    """Turn raw int4 nibbles + scales + zeros into kernel-ready tensors.

    This is the whole of `process_weights_after_loading()` for this op: it
    runs once per checkpoint, never per token.

    Args:
        nibbles: [K, N] uint8, values 0..15.
        scales: [K/128, N], any float dtype.
        zeros: [K/128, N], any float dtype.
        dtype: bf16 or fp16 -- must match the activations the GEMM will see.

    Returns:
        (weight, scales, zeros) in the layout `a16w4_gemm()` expects for
        `dtype`. For fp16 the weight is MAGIC-permuted and the zeros carry
        the +1024 bias; for bf16 neither transform is applied.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"a16w4 activations must be fp16 or bf16, got {dtype}")

    weight = pack_a16w4_weight(nibbles)
    if dtype == torch.float16:
        return (
            repack_a16w4_weight_magic(weight),
            scales.to(torch.float16),
            bias_a16w4_zeros_magic(zeros),
        )
    return weight, scales.to(torch.bfloat16), zeros.to(torch.bfloat16)
