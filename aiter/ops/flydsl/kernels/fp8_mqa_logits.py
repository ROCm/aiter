# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 MQA logits (DeepSeek lightning indexer) -- FlyDSL gfx942 kernel.

Cmpute for each query row ``m`` and KV position ``n``
inside that row's window ``[cu_starts[m], cu_ends[m])``::

    logits[m, n] = sum_h ReLU(<Q[m, h, :], K[n, :]> * kv_scale[n]) * weights[m, h]

The public ``flydsl_fp8_mqa_logits`` mirrors the Triton launcher
``aiter.ops.triton.attention.fp8_mqa_logits.fp8_mqa_logits`` exactly (same
arguments, same return tensor, same ``clean_logits`` semantics) so the two are
drop-in interchangeable in tests and benchmarks.
"""

# NOTE: do NOT add `from __future__ import annotations` to this file -- PEP 563
# stringizes annotations, which FlyDSL's kernel-argument typing relies on being
# real objects. (Matches the note in qk_norm_rope_quant.py.)

from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Stream

from .tensor_shim import _run_compiled

# Default per-arch tile / launch configuration. These mirror the gfx942 Triton
# path (BLOCK_KV=128, 4 waves, waves_per_eu=2) and are the starting point the
# real kernel will be tuned around.
DEFAULT_BLOCK_KV = 128
_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 2,
}


@lru_cache(maxsize=32)
def compile_fp8_mqa_logits(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int = DEFAULT_BLOCK_KV,
    paged: bool = False,
):
    """Return a cached, compiled FlyDSL launcher for the given shape config.

    PLACEHOLDER -- this is the ONLY unimplemented part of the module. Once the
    ``@flyc.kernel`` body is written this will build and cache the launcher
    specialized on the compile-time constants (``num_heads``, ``head_size``,
    ``block_kv``); the host-side ``flydsl_fp8_mqa_logits`` is already complete.

    Parameters
    ----------
    num_heads : int
        Number of indexer query heads (compile-time constant, power of two).
    head_size : int
        Head dimension D (compile-time constant, power of two; D in {64, 128}).
    block_kv : int
        KV tile width processed per inner-loop iteration.
    paged : bool
        Reserved for the Phase-2 paged variant. Must be False for now.
    """
    if paged:
        raise NotImplementedError(
            "Paged FlyDSL fp8_mqa_logits is Phase 2 and not implemented yet."
        )
    raise NotImplementedError(
        "FlyDSL fp8_mqa_logits kernel is not implemented yet (scaffolding only)."
    )


def _ptr_arg(launcher, t):
    """Pointer argument in the convention the launcher expects.

    Mirrors ``qk_norm_rope_quant``: direct-call launchers take a raw int
    data pointer, otherwise wrap it as a flydsl ``Uint8*`` void pointer.
    """
    if getattr(launcher, "_direct_call_state", None) is not None:
        return int(t.data_ptr())
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _stream_arg(launcher, stream):
    if getattr(launcher, "_direct_call_state", None) is not None:
        return stream
    return Stream(stream)


def flydsl_fp8_mqa_logits(
    Q,
    KV,
    kv_scales,
    weights,
    cu_starts,
    cu_ends,
    clean_logits=True,
    stream=None,
):
    """FlyDSL gfx942 FP8 MQA logits -- drop-in for the Triton ``fp8_mqa_logits``.

    This is the full host-side launcher: it validates shapes, allocates the
    output exactly like the Triton path, computes strides, binds the HIP
    stream, and dispatches the compiled kernel. The only piece not yet
    implemented is the kernel itself -- ``compile_fp8_mqa_logits`` is the sole
    source of ``NotImplementedError`` until the ``@flyc.kernel`` body lands.

    Q:            [seq_len, NUM_HEADS, HEAD_SIZE], dtype float8
    KV:           [seq_len_kv, HEAD_SIZE], dtype float8
    kv_scales:    [seq_len_kv], dtype float32
    weights:      [seq_len, NUM_HEADS], dtype float32
    cu_starts:    [seq_len], dtype int32, per-row window start (inclusive)
    cu_ends:      [seq_len], dtype int32, per-row window end (exclusive)
    clean_logits: bool. If True, positions outside [cu_starts[i], cu_ends[i])
                  in row i are written as -inf. If False, the kernel skips
                  those positions and the caller owns whatever is left there.
    stream:       optional HIP stream; defaults to the current stream.

    Returns
    -------
    logits: [seq_len, seq_len_kv], dtype float32.
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    assert num_heads & (num_heads - 1) == 0, "num q. heads should be power of 2."
    assert head_size & (head_size - 1) == 0, "head size should be power of 2."

    # Compile (or fetch cached) the kernel for this shape config first, so an
    # unimplemented kernel fails fast before we allocate the output buffer.
    launcher = compile_fp8_mqa_logits(
        num_heads=num_heads,
        head_size=head_size,
        block_kv=DEFAULT_BLOCK_KV,
        paged=False,
    )

    # Match the Triton launcher's -inf-prefill / padding behavior so the two
    # produce identically-shaped, identically-masked outputs.
    aligned_size = 256
    seq_len_kv_aligned = (
        (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    )
    if clean_logits:
        logits = torch.full(
            (seq_len, seq_len_kv_aligned),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv_aligned),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]

    if stream is None:
        stream = torch.cuda.current_stream()

    stride_q_s, stride_q_h, stride_q_d = Q.stride()
    stride_kv_s, stride_kv_d = KV.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    # Argument order the @flyc.kernel body will be authored against. Grid is
    # (seq_len,), one CTA per query row (kernel reverses row order internally).
    args = (
        _ptr_arg(launcher, Q),
        _ptr_arg(launcher, KV),
        _ptr_arg(launcher, kv_scales),
        _ptr_arg(launcher, weights),
        _ptr_arg(launcher, cu_starts),
        _ptr_arg(launcher, cu_ends),
        _ptr_arg(launcher, logits),
        seq_len,
        seq_len_kv,
        stride_q_s,
        stride_q_h,
        stride_q_d,
        stride_kv_s,
        stride_kv_d,
        stride_w_s,
        stride_w_h,
        stride_logits_s,
        stride_logits_k,
        int(clean_logits),
        _stream_arg(launcher, stream),
    )
    _run_compiled(launcher, *args)

    return logits
