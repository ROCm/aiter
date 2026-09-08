# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attention-residual (residual candidate gating) forward.

For every token the op scores L residual candidates with an RMS-normalized dot
product against the query, turns the scores into a softmax gate over the
candidate axis, and mixes the raw residuals with that gate::

    rstd_l  = 1 / sqrt(mean_d v[l, n, d]^2 + eps)
    logit_l = rstd_l * sum_d v[l, n, d] * (q_d * w_d)
    o[n]    = onorm( sum_l softmax_l(scale * logit_l) * v[l, n] )

This is a forward-only port of fla 0.5.2's ``fused_attnres`` (see
``attnres_fwd_kernel``): the kernel structure, math, and launch surface follow
fla, but the backward pass is not ported. The backward checkpoint tensors
(``o_pre`` and the per-candidate ``rstd`` / ``logit`` / softmax ``lse``) are kept
as kernel parameters so a backward kernel can be reintroduced later, but they are
passed as ``None`` with their save flags off here.

Both residual layouts are served by the single ``attnres_fwd_kernel`` via an
``IS_PACKED`` switch:

* ``layout="sequence"``: a ``Sequence`` of L independent ``[.., D]`` tensors, the
  native form of the fla ``fused_attnres`` API, gathered through a length-``L2``
  pointer table.
* ``layout="packed"``: one contiguous ``[.., L, D]`` tensor read with row strides.

:func:`attn_res_gate` exposes the same packed kernel under the inference contract
used by serving stacks (the candidate set is a packed ``[.., B, D]`` block plus a
separate ``prefix`` row, and the caller's ``prefix += hidden [+ hidden2]`` add can
be folded into the kernel) -- this mirrors ATOM's ``apply_attn_res``, including its
two-addend fold (``add_hidden``/``add_hidden2``, for an MoE layer's routed and
shared expert outputs) and its independently-epsilon'd output RMSNorm
(``output_rms_eps``, distinct from the per-candidate ``eps``). That output
RMSNorm can also emit a per-token FP8 activation directly (``out_quant_dtype``),
which is what lets several consumers of the same normed row share one quant.
"""

import logging
import os
from collections.abc import Sequence

import torch
import triton
from triton import knobs
from triton.runtime import driver

from aiter.ops.triton._triton_kernels.fusions.attn_res import (
    ATTN_RES_TRITON_AUTOTUNE,
    attnres_fwd_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import get_dtype_max

_LOGGER = AiterTritonLogger()

_TRUTHY = ("1", "true", "yes")

# Decode is host-bound: at N=1..128 attn_res_gate spends ~50us of Python per
# call against ~10-25us of device work, so whoever calls it in a per-layer decode
# loop is paying for Triton's launch path, not for the kernel. Measured split
# (MI350X/gfx950, N=1 B=8 H=7168, quant + close_block, scratch/
# probe_gate_launch_floor.py): 20.7us wrapper prologue, 26.5us JITFunction.run
# argument binding + specialization + cache lookup, 5.2us actual driver launch.
#
# _LAUNCH_CACHE removes the middle term. JITFunction.run re-derives the
# specialization of all 42 kernel parameters on every call (11 live tensors, each
# costing a dtype/alignment/2GB probe plus a knobs lookup) only to arrive at a
# cache key it already computed on the previous, identically-shaped call. We
# compute an equivalent key ourselves from the handful of things that can
# actually vary here, and on a hit invoke the CompiledKernel directly the same
# way JITFunction.run would.
#
# Set AITER_ATTN_RES_NO_LAUNCH_CACHE=1 to force every call back through
# JITFunction.run. Set AITER_ATTN_RES_VERIFY_LAUNCH_CACHE=1 to additionally
# resolve the kernel through Triton on every hit and assert it is the object the
# cache handed out -- that is the invariant the key has to satisfy, and it is
# what the launch-cache test asserts across the full flag matrix.
_LAUNCH_CACHE_ENABLED: bool = (
    os.getenv("AITER_ATTN_RES_NO_LAUNCH_CACHE", "0").lower() not in _TRUTHY
)
_LAUNCH_CACHE_VERIFY: bool = (
    os.getenv("AITER_ATTN_RES_VERIFY_LAUNCH_CACHE", "0").lower() in _TRUTHY
)
_LAUNCH_CACHE: dict = {}

_MAX_INT32 = 2**31 - 1


def _tensor_spec(t: torch.Tensor | None):
    """The properties of ``t`` that Triton specializes the compiled kernel on.

    Mirrors HIPBackend.get_tensor_specialization: the element type picks the
    pointer type, 16-byte alignment gates the ``tt.divisibility`` attribute, and
    a storage that fits in 32 bits gates ``tt.pointer_range`` (buffer ops). A
    tensor's own size is not enough for the last one -- a small view into a large
    KV/activation pool is over the limit -- so this has to look at the storage.
    """
    if t is None:
        return None
    return (
        t.dtype,
        t.data_ptr() % 16 == 0,
        t.untyped_storage().size() <= _MAX_INT32,
    )


def _fresh_tensor_spec(t: torch.Tensor | None):
    """:func:`_tensor_spec` for a tensor this module just allocated.

    Such a tensor owns its whole storage, so ``nbytes`` (one attribute) answers
    the 2GB question that would otherwise cost an UntypedStorage round trip, and
    the CUDA caching allocator hands back 512-byte-aligned blocks, so the
    divisibility bit is known. The launch-cache verification exercises this on
    every output tensor, so a wrong assumption here fails loudly rather than
    silently reusing the wrong kernel.
    """
    if t is None:
        return None
    return (t.dtype, True, t.nbytes <= _MAX_INT32)


def _int_spec(v: int):
    # Triton specializes integers on 16-divisibility and on being 1, and picks
    # i32 vs i64 by magnitude. Only the token count is unbounded enough to need
    # this; the other integer arguments are folded into the key by value.
    return (v % 16 == 0, v == 1, -(2**31) <= v <= _MAX_INT32)


_POW2_CACHE: dict[int, int] = {}


def _next_pow2(v: int) -> int:
    # triton.next_power_of_2 is a Python function call on a value that is the
    # same on every call of a given model; memoize it out of the launch path.
    p = _POW2_CACHE.get(v)
    if p is None:
        p = _POW2_CACHE[v] = triton.next_power_of_2(v)
    return p


_DTYPE_MAX_CACHE: dict[torch.dtype, float] = {}


def _dtype_max(dtype: torch.dtype) -> float:
    m = _DTYPE_MAX_CACHE.get(dtype)
    if m is None:
        m = _DTYPE_MAX_CACHE[dtype] = get_dtype_max(dtype)
    return m


# Static per-token-count launch tables (ATOM-style), replacing @triton.autotune.
# Dispatch rounds N UP to the smallest bucket >= N (ceil-to-bucket), so a handful
# of fixed sizes compile one config each -- bounded compile cost and CUDAGraph-
# capture safe (autotune would JIT many configs on a cold cache and invalidate the
# capture). N above the largest bucket falls through to the catch-all.
#
# Two separate tables because BL means something different per layout:
#
# * sequence: the AMD-safe gather scans all L2 padded pointer slots on EVERY BL
#   tile regardless of BL (see the kernel body), so splitting into more, smaller
#   tiles only multiplies that O(L2) scan instead of shrinking anything -- BL=L2
#   (a single tile, one scan) is the right choice here, independent of token count.
# * packed: the tile load is a plain strided read with no such cost, so BL can
#   (and should) be a small constant independent of the candidate count L. Tying
#   BL to L2 here -- the original design -- makes the [BL, BD] register tile scale
#   with L for no benefit: measured on MI350X/gfx950 at H=7168, L2=16 (Kimi-K3's
#   real worst case, 8 banked candidates + 1 prefix) spills 199 of 256 VGPR and is
#   4-6x slower than a fixed-BL=2 dispatch. ATOM's own attn_res kernel uses this
#   fixed-BL-by-token-count shape (not keyed by L at all); the values below are
#   copied from it since they're already validated in production.
_ATTN_RES_SEQ_CONFIGS = (
    # (max_tokens, num_warps, num_stages)
    (16, 8, 1),
    (64, 8, 1),
    (256, 8, 1),
    (1024, 16, 1),
)
_ATTN_RES_SEQ_CATCHALL = (16, 1)  # N > largest bucket

#
# Verified 2026-08-13 with a real search, not just inference from the ATOM
# origin: scratch/tune_attn_res.py wraps attnres_fwd_kernel with
# @triton.autotune (ATTN_RES_TRITON_AUTOTUNE=1, see
# _triton_kernels/fusions/attn_res.py -- an fla-style tuning escape hatch, off
# by default for the same CUDAGraph/compile-cost reasons noted above) and
# benchmarks the full (BL, num_warps, num_stages) grid across Kimi-K3's real
# (T, B) shapes. Result: the search reproduces BL=2/num_warps=8 almost
# everywhere this table already has it, and the one bucket where the search
# initially looked different (the N>2048 catchall) turned out to be noise on
# closer, same-process A/B (scratch/verify_catchall_2x2.py) -- every
# alternative tried was within ~5-10% either way with no consistent winner
# across N. No changes made; this table is already at/near the tiling
# optimum for this kernel.
_ATTN_RES_PACKED_CONFIGS = (
    # (max_tokens, num_warps, num_stages, BL)
    (8, 8, 2, 2),
    (64, 8, 2, 2),
    (512, 8, 2, 2),
    (2048, 4, 2, 2),
)
_ATTN_RES_PACKED_CATCHALL = (4, 2, 2)  # N > largest bucket


# SEPARATE is on for N above this. Below it the grid underfills the GPU and the
# packed path's BL=2 tile wins over the separated loop's serial full-H row loads.
# 256 clears the small-N regression zone while still covering the (256, 512] band
# that used to stay on the packed path.
_ATTN_RES_PREFILL_T = 256

# BL is how many candidate rows a workgroup loads per loop iteration: at BL=1 the
# next instruction consumes the row that was just loaded, so a wave never has more
# than one row in flight. That is normally covered by switching waves, but the grid
# here is one workgroup per token, so at mid token counts there are too few waves
# to switch to. Raising BL is the one lever that adds outstanding loads per wave
# without needing more workgroups.
#
# The table keys on the candidate count rather than on N, which is what the sweep
# says it depends on (scratch/tune_bl_nocm.sh; flat in N within a row). The losing
# cells are exactly the ones where BL reaches the full candidate count, because the
# separated loop then runs a single iteration: the wide tile pays the register price
# with no second iteration to overlap its loads against, which is the whole point of
# raising BL. Hence at least two iterations, BL = l2 // 2.
#
# The cap at 4 is the register file: the [BL, BD] tile is fp32, so at BD=8192 over
# 256 threads it costs BL*32 VGPRs per thread, and BL=4 already drops occupancy to
# 1 wave/SIMD.
_ATTN_RES_SEPARATE_BL_CONFIGS = (
    # (max_l2, BL)
    (2, 1),
    (4, 2),
)
_ATTN_RES_SEPARATE_BL_CATCHALL = 4  # l2 > largest bucket

# Above this N there are enough workgroups to cover memory latency by wave
# switching alone, so the extra registers stop paying for themselves.
_ATTN_RES_SEPARATE_BL_MAX_T = 16384

# All of the above holds *only when block_out is being written*, so BL>1 is gated
# on close_block; with that store off the same table is a sizeable regression and
# a flat BL=1 wins instead (scratch/sweep_bl_nocb.py).
#
# The reason it flips: with block_out on, that write stream saturates HBM on its
# own and the wide tile's register cost is free. With it off, per-workgroup traffic
# roughly halves, so saturating HBM needs waves in flight instead -- which is
# exactly what the wide tile gives up. The real predicate is therefore how much
# traffic each workgroup carries, and close_block is only its dominant term;
# re-check the gate if another flag ever moves comparable volume.

# Set to an int to force a BL and bypass the table (used by the sweeps above).
_ATTN_RES_SEPARATE_BL_OVERRIDE: int | None = None


def _pick_attn_res_separate_bl(tokens: int, l2: int, close_block: bool) -> int:
    if _ATTN_RES_SEPARATE_BL_OVERRIDE is not None:
        bl = _ATTN_RES_SEPARATE_BL_OVERRIDE
    elif not close_block or tokens > _ATTN_RES_SEPARATE_BL_MAX_T:
        bl = 1
    else:
        bl = _ATTN_RES_SEPARATE_BL_CATCHALL
        for max_l2, cand in _ATTN_RES_SEPARATE_BL_CONFIGS:
            if l2 <= max_l2:
                bl = cand
                break
    # The separated loop only covers the L-1 block_residual rows, so a wider tile
    # than that just burns registers on lanes that are masked off anyway.
    return max(1, min(bl, l2))


def _pick_attn_res_seq_config(tokens: int) -> tuple[int, int]:
    for max_tokens, num_warps, num_stages in _ATTN_RES_SEQ_CONFIGS:
        if tokens <= max_tokens:
            return num_warps, num_stages
    return _ATTN_RES_SEQ_CATCHALL


def _fast_reshape2d(t: torch.Tensor, d: int) -> torch.Tensor:
    # Skip reshape()/contiguous() (each a real dispatcher call) when the tensor
    # is already exactly the shape/layout the kernel wants -- the common case
    # for a caller like ATOM's AttnRes, whose inputs are already [N, D]
    # contiguous from the previous layer. Only pay for the general path
    # (leading-dim collapse and/or a real copy) when actually needed.
    if t.dim() == 2 and t.shape[1] == d and t.is_contiguous():
        return t
    return t.reshape(-1, d).contiguous()


def _fast_reshape3d(t: torch.Tensor, b: int, d: int) -> torch.Tensor:
    if t.dim() == 3 and t.shape[1] == b and t.shape[2] == d and t.is_contiguous():
        return t
    return t.reshape(-1, b, d).contiguous()


def _fast_flatten1d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1 and t.is_contiguous():
        return t
    return t.flatten().contiguous()


def _pick_attn_res_packed_config(tokens: int, l2: int) -> tuple[int, int, int]:
    for max_tokens, num_warps, num_stages, bl in _ATTN_RES_PACKED_CONFIGS:
        if tokens <= max_tokens:
            return num_warps, num_stages, min(bl, l2)
    num_warps, num_stages, bl = _ATTN_RES_PACKED_CATCHALL
    return num_warps, num_stages, min(bl, l2)


def _launch_tune_kwargs(num_warps: int, num_stages: int, bl: int | None = None) -> dict:
    # When ATTN_RES_TRITON_AUTOTUNE is on, attnres_fwd_kernel is itself wrapped by
    # @triton.autotune (see _triton_kernels/fusions/attn_res.py): BL/num_warps/
    # num_stages become meta-parameters the decorator supplies from its own config
    # search, so the launch must NOT also pass them explicitly (that would just be
    # a redundant/duplicate value, not an override). Otherwise, launch with the
    # wrapper's static per-token-count picks, as today.
    if ATTN_RES_TRITON_AUTOTUNE:
        return {}
    kwargs = {"num_warps": num_warps, "num_stages": num_stages}
    if bl is not None:
        kwargs["BL"] = bl
    return kwargs


# Positional order of attnres_fwd_kernel's parameters, so the cached-launch path
# can hand the CompiledKernel a tuple instead of rebuilding a 42-entry keyword
# dict. Asserted against the kernel below rather than trusted.
_KERNEL_ARG_ORDER = (
    "q",
    "res",
    "w",
    "ow",
    "o",
    "o_pre",
    "rstd",
    "logit",
    "lse",
    "res_packed",
    "prefix",
    "add_hidden",
    "add_hidden2",
    "prefix_out",
    "block_out",
    "o_scale",
    "N",
    "L",
    "stride_res_n",
    "stride_res_l",
    "stride_bo_n",
    "stride_bo_l",
    "L2",
    "D",
    "eps",
    "out_eps",
    "scale",
    "BL",
    "BD",
    "HAS_ONORM",
    "SAVE_OPRE",
    "SAVE_STATS",
    "IS_PACKED",
    "HAS_PREFIX",
    "DO_ADD",
    "DO_ADD2",
    "WRITE_PREF",
    "WRITE_BLOCK_CAT",
    "HAS_W",
    "QUANT_FP8",
    "FP8_MAX",
    "SEPARATE",
)

if ATTN_RES_TRITON_AUTOTUNE:
    # The kernel is wrapped by @triton.autotune, which owns BL and the launch
    # config; there is no single CompiledKernel to cache per key.
    _LAUNCH_CACHE_ENABLED = False
elif tuple(attnres_fwd_kernel.arg_names) != _KERNEL_ARG_ORDER:
    _LAUNCH_CACHE_ENABLED = False
    _LOGGER.info(
        "ATTN_RES_GATE: attnres_fwd_kernel signature changed, launch cache off"
    )


def _assert_launch_cache_key(cached, grid_n: int, kwargs: dict) -> None:
    # warmup() resolves the kernel through Triton's own specialization and cache
    # without launching, so it answers exactly the question the key has to get
    # right: would Triton have picked this same CompiledKernel?
    resolved = attnres_fwd_kernel.warmup(grid=(grid_n,), **kwargs)
    if resolved is not cached:
        raise AssertionError(
            "attn_res launch cache returned a different kernel than Triton would "
            f"have for grid={grid_n}; the cache key is too coarse"
        )


def _launch_hooks_idle() -> bool:
    # Triton keeps launch_enter_hook/launch_exit_hook as HookChain objects that
    # are non-None but empty until someone (a profiler, typically) registers a
    # callback, so "is None" is not the test. An empty chain has nothing to
    # receive launch metadata, which is what lets the fast path skip building it.
    enter = knobs.runtime.launch_enter_hook
    if getattr(enter, "calls", enter):
        return False
    exit_ = knobs.runtime.launch_exit_hook
    return not getattr(exit_, "calls", exit_)


def _launch_attn_res(key, grid_n: int, argv: tuple, tune_kwargs: dict) -> None:
    """Launch attnres_fwd_kernel, reusing the resolved kernel when we can.

    ``key`` is None when the cache is off, in which case this is just the plain
    ``attnres_fwd_kernel[grid](...)`` call. The fast path mirrors the tail of
    JITFunction.run; it is skipped whenever a launch hook is installed, since
    those need the launch metadata that only the full path builds.
    """
    if key is not None and _launch_hooks_idle():
        device = driver.active.get_current_device()
        per_device = _LAUNCH_CACHE.get(device)
        cached = per_device.get(key) if per_device is not None else None
        if cached is not None:
            if _LAUNCH_CACHE_VERIFY:
                _assert_launch_cache_key(
                    cached, grid_n, dict(zip(_KERNEL_ARG_ORDER, argv), **tune_kwargs)
                )
            try:
                cached.run(
                    grid_n,
                    1,
                    1,
                    driver.active.get_current_stream(device),
                    cached.function,
                    cached.packed_metadata,
                    None,  # launch_metadata: only built for hooks, excluded above
                    None,  # launch_enter_hook
                    None,  # launch_exit_hook
                    *argv,
                )
                return
            except TypeError:
                # CompiledKernel.run's parameter list is Triton's internal launcher
                # ABI, stable across every version this has been built against but
                # not a public contract. If it ever moves, an arity/type mismatch
                # here means nothing was launched, so retire the fast path for the
                # process and fall through to Triton's own launch.
                global _LAUNCH_CACHE_ENABLED
                _LAUNCH_CACHE_ENABLED = False
                _LAUNCH_CACHE.clear()
                _LOGGER.info(
                    "ATTN_RES_GATE: CompiledKernel launch ABI mismatch, launch cache off"
                )
                key = None
    else:
        device = None

    kwargs = dict(zip(_KERNEL_ARG_ORDER, argv))
    if ATTN_RES_TRITON_AUTOTUNE:
        del kwargs["BL"]  # supplied by the autotuner's config search
    kernel = attnres_fwd_kernel[(grid_n,)](**kwargs, **tune_kwargs)
    if key is not None:
        if device is None:
            device = driver.active.get_current_device()
        _LAUNCH_CACHE.setdefault(device, {})[key] = kernel


def _build_ptr_table(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    # Pad the per-source tuple to a power-of-2 length so Triton compiles one
    # kernel per L2 bucket instead of one per L. Padded slots reuse tensors[0]
    # and are masked out in the kernel.
    L2 = max(1, triton.next_power_of_2(len(tensors)))
    assert 1 <= len(tensors) <= L2
    for t in tensors:
        assert (
            t.data_ptr() % 16 == 0
        ), "attn_res residual sources must be 16-byte aligned"
    return tuple(tensors) + (tensors[0],) * (L2 - len(tensors))


def attn_res_fwd(
    query: torch.Tensor,
    residuals,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    *,
    layout: str = "sequence",
) -> torch.Tensor:
    """Attention-residual forward.

    Key parameters:
    - query: ``[.., D]`` scoring query, flattened internally.
    - residuals: sequence layout -> Sequence of L tensors each ``[.., D]``;
      packed layout -> a single ``[.., L, D]`` tensor, or a Sequence that will
      be stacked into one.
    - rms_weight: ``[D]`` per-channel weight folded into the score.
    - output_rms_weight: optional ``[D]`` weight enabling the output RMSNorm.
    - rms_eps: epsilon of both the per-candidate and the output RMSNorm.
    - scale: multiplies the logits before the softmax.
    - layout: "sequence" or "packed".

    Returns the mixed residual ``o`` of shape ``[.., D]``.
    """
    if layout not in ("sequence", "packed"):
        raise ValueError(f"layout must be 'sequence' or 'packed', got {layout!r}")

    _LOGGER.info(
        f"ATTN_RES: query={tuple(query.shape)} rms_weight={tuple(rms_weight.shape)} "
        f"layout={layout}"
    )

    has_onorm = output_rms_weight is not None
    q_flat = query.flatten().contiguous()
    w_flat = rms_weight.flatten().contiguous()
    ow_flat = output_rms_weight.flatten().contiguous() if has_onorm else None

    runner = _run_packed if layout == "packed" else _run_sequence
    return runner(q_flat, residuals, w_flat, ow_flat, rms_eps, scale, has_onorm)


def _run_sequence(q_flat, residuals, w_flat, ow_flat, rms_eps, scale, has_onorm):
    if not residuals[0].is_cuda:
        raise ValueError("Triton attn_res requires CUDA/ROCm tensors")
    output_shape = residuals[0].shape
    D = output_shape[-1]
    # The slot-scan gather hints 16-element alignment (tl.multiple_of) on each
    # row base, which only holds when the row stride D is a multiple of 16.
    assert (
        D % 16 == 0
    ), f"attn_res sequence layout requires D to be a multiple of 16, got D={D}"
    flat_residuals = tuple(r.reshape(-1, D).contiguous() for r in residuals)
    res = _build_ptr_table(flat_residuals)
    L = len(flat_residuals)
    N = flat_residuals[0].numel() // D
    dtype = flat_residuals[0].dtype
    device = flat_residuals[0].device

    o = torch.empty((N, D), device=device, dtype=dtype)
    L2 = max(1, triton.next_power_of_2(L))
    num_warps, num_stages = _pick_attn_res_seq_config(N)

    attnres_fwd_kernel[(N,)](
        q=q_flat,
        res=res,
        w=w_flat,
        ow=ow_flat,
        o=o,
        o_pre=None,
        rstd=None,
        logit=None,
        lse=None,
        res_packed=None,
        prefix=None,
        add_hidden=None,
        add_hidden2=None,
        prefix_out=None,
        block_out=None,
        o_scale=None,
        N=N,
        L=L,
        stride_res_n=0,
        stride_res_l=0,
        stride_bo_n=0,
        stride_bo_l=0,
        L2=L2,
        D=D,
        eps=rms_eps,
        out_eps=rms_eps,
        scale=scale,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=False,
        SAVE_STATS=False,
        IS_PACKED=False,
        HAS_PREFIX=False,
        DO_ADD=False,
        DO_ADD2=False,
        WRITE_PREF=False,
        WRITE_BLOCK_CAT=False,
        HAS_W=True,
        QUANT_FP8=False,
        FP8_MAX=1.0,
        **_launch_tune_kwargs(num_warps, num_stages, L2),
    )
    return o.view(output_shape)


def _run_packed(q_flat, residuals, w_flat, ow_flat, rms_eps, scale, has_onorm):
    if isinstance(residuals, (list, tuple)):
        L = len(residuals)
        output_shape = residuals[0].shape  # [.., D]
        packed = torch.stack([r.contiguous() for r in residuals], dim=-2)  # [.., L, D]
    else:
        packed = residuals
        L = packed.shape[-2]
        output_shape = packed.shape[:-2] + packed.shape[-1:]
    if not packed.is_cuda:
        raise ValueError("Triton attn_res requires CUDA/ROCm tensors")
    D = output_shape[-1]
    packed = packed.reshape(-1, L, D).contiguous()  # [N, L, D]
    N = packed.shape[0]
    dtype = packed.dtype
    device = packed.device

    o = torch.empty((N, D), device=device, dtype=dtype)
    L2 = max(1, triton.next_power_of_2(L))
    num_warps, num_stages, bl = _pick_attn_res_packed_config(N, L2)

    attnres_fwd_kernel[(N,)](
        q=q_flat,
        res=None,  # unused when IS_PACKED (sequence branch is dead); None keeps
        # the L2 dead pointer slots out of the kernarg segment
        w=w_flat,
        ow=ow_flat,
        o=o,
        o_pre=None,
        rstd=None,
        logit=None,
        lse=None,
        res_packed=packed,
        prefix=None,
        add_hidden=None,
        add_hidden2=None,
        prefix_out=None,
        block_out=None,
        o_scale=None,
        N=N,
        L=L,
        stride_res_n=packed.stride(0),
        stride_res_l=packed.stride(1),
        stride_bo_n=0,
        stride_bo_l=0,
        L2=L2,
        D=D,
        eps=rms_eps,
        out_eps=rms_eps,
        scale=scale,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=False,
        SAVE_STATS=False,
        IS_PACKED=True,
        HAS_PREFIX=False,
        DO_ADD=False,
        DO_ADD2=False,
        WRITE_PREF=False,
        WRITE_BLOCK_CAT=False,
        HAS_W=True,
        QUANT_FP8=False,
        FP8_MAX=1.0,
        **_launch_tune_kwargs(num_warps, num_stages, bl),
    )
    return o.view(output_shape)


def attn_res_gate(
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float = 1e-6,
    add_hidden: torch.Tensor | None = None,
    add_hidden2: torch.Tensor | None = None,
    *,
    output_rms_weight: torch.Tensor | None = None,
    output_rms_eps: float = 1e-6,
    scale: float = 1.0,
    close_block: bool = False,
    out_quant_dtype: torch.dtype | None = None,
):
    """Inference-shaped attention-residual gate over ``B + 1`` candidates.

    Same math as :func:`attn_res_fwd` on the packed layout, specialized for the
    decode/prefill contract (mirrors ATOM's ``apply_attn_res``): the candidate set
    is the ``B`` rows of ``block_residual`` plus ``prefix`` as the last candidate.

    Key parameters:
    - prefix: ``[.., D]`` running residual, used as the last candidate.
    - block_residual: ``[.., B, D]`` packed candidate block.
    - score_weight: ``[D]`` pre-folded ``rms_weight * query`` scoring vector.
    - eps: per-candidate RMSNorm epsilon.
    - add_hidden: optional ``[.., D]``; folds ``prefix = prefix + add_hidden``
      into the kernel, saving a launch and an HBM round trip.
    - add_hidden2: optional ``[.., D]``; folds a SECOND addend the same way
      (``prefix = prefix + add_hidden + add_hidden2``), e.g. so an MoE layer can
      hand over its routed and shared expert outputs unsummed. Requires
      ``add_hidden`` to also be given.
    - output_rms_weight: optional ``[D]``; folds the prenorm that would
      otherwise follow this call into the kernel.
    - output_rms_eps: epsilon of that output RMSNorm; independent of ``eps``
      since the caller's output-norm module may differ from the per-candidate
      one (only used when ``output_rms_weight`` is given).
    - scale: multiplies the logits before the softmax.
    - close_block: when True, additionally fuses
      ``torch.cat([block_residual, prefix_out.unsqueeze(-2)], dim=-2)`` into
      this same kernel pass (mirrors ATOM's ``AttnRes.maybe_close_block``
      block-banking step) instead of a separate ``torch.cat`` that would
      re-read ``block_residual`` from HBM. See ``block_out`` below.
    - out_quant_dtype: optional FP8 dtype; folds the per-token activation quant
      of the output RMSNorm result into this same kernel, so ``y`` comes back as
      an ``(fp8, scale)`` pair a GEMM can consume directly instead of a BF16
      tensor that each consumer quantizes for itself. Requires
      ``output_rms_weight`` (the quant input is that norm's result). The scale is
      ``[.., 1]`` fp32, one per token, derived as ``amax * (1 / finfo(dtype).max)``
      -- bit-exact against ``get_hip_quant(QuantType.per_Token)``, which is what
      a consumer runs on this activation when the fold is off. An all-zero row
      gets scale 0 and quantizes to zeros.
      ``block_out`` stays unquantized regardless: those rows return as scoring
      candidates, which need the unquantized values.

    Returns:
    - ``close_block=False`` (default): ``(y, prefix_out)``, unchanged for
      every existing caller. ``prefix_out`` is the summed prefix when
      ``add_hidden`` is given, otherwise ``prefix`` unchanged.
    - ``close_block=True``: ``(y, prefix_out, block_out)`` where ``block_out``
      is ``cat([block_residual, prefix_out.unsqueeze(-2)], dim=-2)``,
      ``[.., B + 1, D]``.
    - ``out_quant_dtype`` set: ``y`` above becomes the tuple
      ``(y_fp8, y_scale)``; the rest of the contract is unchanged.
    """
    if not prefix.is_cuda:
        raise ValueError("Triton attn_res requires CUDA/ROCm tensors")
    if block_residual.dtype != prefix.dtype:
        raise ValueError(
            f"prefix and block_residual must share a dtype, got {prefix.dtype} "
            f"and {block_residual.dtype}"
        )
    if add_hidden2 is not None and add_hidden is None:
        raise ValueError("add_hidden2 requires add_hidden")
    if out_quant_dtype is not None and output_rms_weight is None:
        raise ValueError("out_quant_dtype requires output_rms_weight")

    if _LOGGER.get_logger().isEnabledFor(logging.INFO):
        _LOGGER.info(
            f"ATTN_RES_GATE: prefix={tuple(prefix.shape)} "
            f"block_residual={tuple(block_residual.shape)}"
        )

    output_shape = prefix.shape  # [.., D]
    D = output_shape[-1]
    B = block_residual.shape[-2]
    L = B + 1  # candidates: the B packed rows plus the prefix

    br = _fast_reshape3d(block_residual, B, D)
    pf = _fast_reshape2d(prefix, D)
    sw = _fast_flatten1d(score_weight)
    N = pf.shape[0]
    if br.shape[0] != N:
        raise ValueError(
            f"prefix has {N} rows but block_residual has {br.shape[0]}; the "
            "leading dimensions must match"
        )

    has_onorm = output_rms_weight is not None
    ow = _fast_flatten1d(output_rms_weight) if has_onorm else sw

    quant = out_quant_dtype is not None
    y = torch.empty((N, D), device=pf.device, dtype=out_quant_dtype or pf.dtype)
    y_scale = (
        torch.empty((N, 1), device=pf.device, dtype=torch.float32) if quant else None
    )
    do_add = add_hidden is not None
    do_add2 = add_hidden2 is not None
    if do_add:
        hs = _fast_reshape2d(add_hidden, D)
        prefix_out = torch.empty_like(pf)
    else:
        # add_hidden / prefix_out are unused (DO_ADD / WRITE_PREF are off) but
        # Triton still needs a tensor argument, so reuse the prefix.
        hs = pf
        prefix_out = pf
    hs2 = _fast_reshape2d(add_hidden2, D) if do_add2 else pf

    if close_block:
        block_out = torch.empty((N, B + 1, D), device=br.device, dtype=br.dtype)
        bo = block_out
    else:
        block_out = None
        bo = br  # unused (WRITE_BLOCK_CAT off); reuse an existing tensor arg

    L2 = max(1, _next_pow2(L))
    num_warps, num_stages, bl = _pick_attn_res_packed_config(N, L2)
    # Prefill regime: separate the prefix candidate out of the loop and drop to BL=1
    # (see _ATTN_RES_PREFILL_T). The separated loop covers only the B block_residual
    # rows, so BL=1 is a flat 1-D row load with no candidate-axis raggedness.
    separate = N > _ATTN_RES_PREFILL_T
    if separate:
        bl = _pick_attn_res_separate_bl(N, _next_pow2(max(L - 1, 1)), close_block)

    res_stride_n, res_stride_l, _ = br.stride()
    bo_stride_n, bo_stride_l, _ = bo.stride()
    fp8_max = _dtype_max(out_quant_dtype) if quant else 1.0
    argv = (
        sw,  # q
        None,  # res: unused when IS_PACKED (sequence branch is dead); None keeps
        # the L2 dead pointer slots out of the kernarg segment
        sw,  # w
        ow,
        y,  # o
        None,  # o_pre
        None,  # rstd
        None,  # logit
        None,  # lse
        br,  # res_packed
        pf,  # prefix
        hs,  # add_hidden
        hs2,  # add_hidden2
        prefix_out,
        bo,  # block_out
        y_scale,  # o_scale
        N,
        L,
        res_stride_n,
        res_stride_l,
        bo_stride_n,
        bo_stride_l,
        L2,
        D,
        eps,
        output_rms_eps,  # out_eps
        scale,
        bl,  # BL
        _next_pow2(D),  # BD
        has_onorm,  # HAS_ONORM
        False,  # SAVE_OPRE
        False,  # SAVE_STATS
        True,  # IS_PACKED
        True,  # HAS_PREFIX
        do_add,  # DO_ADD
        do_add2,  # DO_ADD2
        do_add,  # WRITE_PREF
        close_block,  # WRITE_BLOCK_CAT
        False,  # HAS_W
        quant,  # QUANT_FP8
        fp8_max,  # FP8_MAX
        separate,  # SEPARATE
    )
    if _LAUNCH_CACHE_ENABLED:
        # Everything Triton would specialize on. The aliased slots (q/w, and the
        # hs/hs2/prefix_out/bo reuse of pf/br when their flags are off) are
        # covered by the flags already in the key, so they are not probed twice.
        key = (
            _tensor_spec(sw),
            _tensor_spec(ow),
            _fresh_tensor_spec(y),
            _fresh_tensor_spec(y_scale),
            _tensor_spec(br),
            _tensor_spec(pf),
            _tensor_spec(hs) if do_add else None,
            _tensor_spec(hs2) if do_add2 else None,
            _fresh_tensor_spec(prefix_out) if do_add else None,
            _fresh_tensor_spec(bo) if close_block else None,
            _int_spec(N),
            L,
            res_stride_n,
            res_stride_l,
            bo_stride_n,
            bo_stride_l,
            L2,
            D,
            # Triton specializes scalars too -- a float that happens to be 1.0
            # compiles to a different kernel than one that is not -- so the
            # scalars go in by value rather than by any derived property. They
            # are model constants, so this does not grow the cache.
            eps,
            output_rms_eps,
            scale,
            bl,
            has_onorm,
            do_add,
            do_add2,
            close_block,
            quant,
            fp8_max,
            separate,
            num_warps,
            num_stages,
        )
    else:
        key = None
    # BL rides in argv at its signature position, so it must not be repeated here.
    _launch_attn_res(key, N, argv, _launch_tune_kwargs(num_warps, num_stages))

    # prefix is [.., D] and the kernel writes [N, D], so the view is a no-op
    # whenever the caller already handed us a 2-D batch -- the decode case.
    y_v = y if y.shape == output_shape else y.view(output_shape)
    y_out = (y_v, y_scale) if quant else y_v
    if do_add:
        prefix_result = (
            prefix_out
            if prefix_out.shape == output_shape
            else prefix_out.view(output_shape)
        )
    else:
        prefix_result = prefix
    if not close_block:
        return y_out, prefix_result
    return y_out, prefix_result, block_out
