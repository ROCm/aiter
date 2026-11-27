# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The routed MoE tail (gather-reduce + routed scale + shared add) deferred into
the all-reduce that consumes the MoE output (gfx1250 TP4 decode).

At decode sizes the grouped MoE's last launch, the gather-reduce
(``kernels/moe_gather_reduce.py`` with a ``moe_output_epilogue``), only feeds the
next all-reduce + RMSNorm, which reads every row of it once. Inside
``moe_defer_tail()`` the grouped MoE call skips that launch: its output tensor is
returned unwritten and a ``MoeDeferredTail`` holding the launch's operands is
registered under the output's storage. The all-reduce that takes that tensor
(``take_deferred_moe_tail``) either computes the rows itself in its prologue
(the k5 one-shot kernel, ``CustomAllreduce.k5_fused_ar_rms_tail``: the same
bits) or, on any other path, runs the skipped launch first
(``MoeDeferredTail.materialize``). Any other reader must call
``materialize_deferred_moe_tail`` (or the caller must not defer).

The registry holds the operands (and the unwritten output) alive until the
consumer takes them, so no allocation can reuse their storage in between, eager
or under CUDA-graph capture. Python-side state: producer and consumer must run
eagerly (opaque custom ops), in program order.
"""

from __future__ import annotations

import contextlib
import threading

import torch

__all__ = [
    "MoeDeferredTail",
    "moe_defer_tail",
    "take_deferred_moe_tail",
    "materialize_deferred_moe_tail",
    "has_deferred_moe_tail",
]


class MoeDeferredTail:
    """``out[t] = bf16(sum_k gather_w[t, k] * rows_bf16[topids_to_rows[t, k]])``,
    then ``* scale`` (rounded) when ``scale != 1`` and ``+ addend`` (rounded) with an
    addend: the gather-reduce launch it stands for. ``before`` (e.g. the join of
    the stream producing ``addend``) must run before any launch that reads it.
    ``gemm2`` (``launch()``, from ``keep_gemm2``): the GEMM2 launch that writes
    ``grouped_out``, held back for a consumer that runs it inside its own launch
    (``CustomAllreduce.k5_fused_ar_rms_tail``); every other reader of
    ``grouped_out`` calls ``run_gemm2`` first."""

    __slots__ = (
        "out",
        "grouped_out",
        "topids_to_rows",
        "gather_w",
        "addend",
        "scale",
        "before",
        "gemm2",
    )

    def __init__(
        self, out, grouped_out, topids_to_rows, gather_w, addend, scale, before, gemm2=None
    ):
        self.out = out
        self.grouped_out = grouped_out
        self.topids_to_rows = topids_to_rows
        self.gather_w = gather_w
        self.addend = addend
        self.scale = float(scale)
        self.before = before
        self.gemm2 = gemm2

    def run_gemm2(self) -> None:
        """Launch the held-back GEMM2 (once): ``grouped_out`` is then written."""
        gemm2, self.gemm2 = self.gemm2, None
        if gemm2 is not None:
            gemm2.launch()

    def run_before(self) -> None:
        before, self.before = self.before, None
        if before is not None:
            before()

    def materialize(self) -> torch.Tensor:
        """Run the skipped gather-reduce into ``out`` (on the current stream)."""
        from aiter.ops.flydsl.grouped_moe_gfx1250 import flydsl_moe_gather_reduce

        import traceback

        caller = " <- ".join(
            f"{fr.name}" for fr in traceback.extract_stack(limit=6)[:-1][::-1]
        )
        _log_once(
            f"materialized_{self.out.shape[0]}",
            f"deferred MoE tail run before a non-k5 reader ({self.out.shape[0]} rows; {caller})",
            warn=True,
        )

        self.run_gemm2()
        self.run_before()
        flydsl_moe_gather_reduce(
            self.grouped_out,
            self.topids_to_rows,
            self.gather_w,
            out=self.out,
            addend=self.addend,
            addend_scale=self.scale,
        )
        return self.out

    def k5_operands(self):
        """(grouped rows flat, topids_to_rows, gather_w, addend or None) as the k5
        tail kernel reads them, or None when it cannot: bf16 rows of 6144 in a
        3-D (E, max_m, D) grouped output (split_k 1), topk 4, fp32 weights,
        everything contiguous and 16-byte aligned."""
        g, rows, w, add, out = (
            self.grouped_out,
            self.topids_to_rows,
            self.gather_w,
            self.addend,
            self.out,
        )
        tokens = out.shape[0]
        ok = (
            g.dim() == 3
            and g.dtype == torch.bfloat16
            and g.shape[-1] == out.shape[-1]
            and g.is_contiguous()
            and rows.dtype == torch.int32
            and tuple(rows.shape) == (tokens, 4)
            and rows.is_contiguous()
            and w.dtype == torch.float32
            and tuple(w.shape) == (tokens, 4)
            and w.is_contiguous()
            and (add is None or (add.dtype == torch.bfloat16 and add.is_contiguous()
                                 and add.shape == out.shape))
        )
        if not ok:
            _log_once(
                f"operands_{tokens}",
                "k5 tail operands refused: "
                f"g {tuple(g.shape)} {g.dtype} contig {g.is_contiguous()}, rows {tuple(rows.shape)} {rows.dtype} "
                f"contig {rows.is_contiguous()}, w {tuple(w.shape)} {w.dtype} contig {w.is_contiguous()}, add "
                + ("None" if add is None else f"{tuple(add.shape)} {add.dtype} contig {add.is_contiguous()}")
                + f", out {tuple(out.shape)}",
                warn=True,
            )
            return None
        ptrs = [g.data_ptr(), rows.data_ptr(), w.data_ptr()] + ([add.data_ptr()] if add is not None else [])
        if any(p % 16 for p in ptrs):
            _log_once(f"align_{tokens}", f"k5 tail operands misaligned: {[p % 16 for p in ptrs]}", warn=True)
            return None
        return g.view(-1, g.shape[-1]), rows, w, add


_LOGGED: set = set()


def _log_once(key: str, msg: str, warn: bool = False) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        from aiter import logger

        (logger.warning if warn else logger.info)("moe_deferred_tail: %s (first time)", msg)


# Pending tails by the data pointer of their (unwritten) output.
_PENDING: dict[int, MoeDeferredTail] = {}
_SCOPE = threading.local()
_WARNED = False


class _DeferScope:
    __slots__ = ("deferred",)

    def __init__(self):
        self.deferred = None


@contextlib.contextmanager
def moe_defer_tail():
    """Inside: the grouped MoE call whose gather-reduce takes a
    ``moe_output_epilogue`` (TP, no EP / fold) returns its output unwritten and
    registers the tail for its consumer. Yields a scope; ``scope.deferred`` is
    the ``MoeDeferredTail`` when that happened, else None (the call ran its
    gather-reduce as usual)."""
    scope = _DeferScope()
    prev = getattr(_SCOPE, "value", None)
    _SCOPE.value = scope
    try:
        yield scope
    finally:
        _SCOPE.value = prev


def _defer_scope():
    scope = getattr(_SCOPE, "value", None)
    if scope is None or scope.deferred is not None:
        return None
    return scope


def register_deferred_moe_tail(tail: MoeDeferredTail) -> bool:
    """Called by the grouped MoE in place of its gather-reduce launch: True if a
    ``moe_defer_tail`` scope took the tail (the output stays unwritten)."""
    scope = _defer_scope()
    if scope is None or tail.out.numel() == 0:
        return False
    if _PENDING:
        # An earlier tail nobody took: its consumer read the unwritten output (a
        # caller deferred where something else reads the MoE output). Write it now
        # and say so: the results so far are wrong.
        global _WARNED
        if not _WARNED:
            _WARNED = True
            from aiter import logger

            logger.warning(
                "moe_deferred_tail: a deferred MoE tail was never consumed by an all-reduce; "
                "its output was read unwritten. Do not defer where other readers exist."
            )
        materialize_deferred_moe_tail()
    _PENDING[tail.out.data_ptr()] = tail
    scope.deferred = tail
    return True


# Residual taps: a caller that wants the residual stream after a MoE layer (e.g. a speculative
# drafter's hidden-state tap) hands the all-reduce that consumes the layer's output (keyed by that
# output's storage, like a deferred tail) the buffer rows to write with its residual output.
_TAPS: dict[int, tuple] = {}
TAP_STATS = {"registered": 0, "written": 0}  # host-side counts (captures included), for engagement checks


def register_residual_tap(x: torch.Tensor, buf: torch.Tensor, rows) -> None:
    """The all-reduce + norm consuming ``x`` writes its residual output into ``buf[rows]``."""
    _TAPS[x.data_ptr()] = (buf, rows)
    TAP_STATS["registered"] += 1


def take_residual_tap(x: torch.Tensor):
    """The (buffer, rows) registered for ``x``, removed, or None."""
    if not _TAPS or not isinstance(x, torch.Tensor):
        return None
    return _TAPS.pop(x.data_ptr(), None)


# A deferred-tail launch: a caller that can run a deferred tail's GEMM2, the tail and the all-reduce
# that takes it as one launch registers it here; with none registered nothing below changes.
_TAIL_LAUNCH = None


def register_tail_launch(keep, fuse) -> None:
    """``keep(gemm2_args, gemm2_kwargs, tp_gather, tokens)``: an object with ``launch()`` holding the
    grouped MoE's GEMM2 back for the all-reduce, or None to launch it as usual. ``fuse(ca, tail,
    operands, res_inp, w, eps, gemma_norm, lamport_order, fp8_out)``: the all-reduce's outputs from
    one launch, or None (GEMM2 then runs and the all-reduce takes the tail as before)."""
    global _TAIL_LAUNCH
    if _TAIL_LAUNCH is not None and _TAIL_LAUNCH != (keep, fuse):
        import logging

        logging.getLogger("aiter").warning("deferred MoE tail: a different tail launch replaces the registered one")
    _TAIL_LAUNCH = (keep, fuse)


def keep_gemm2(args, kw, tp_gather, tokens: int):
    return None if _TAIL_LAUNCH is None else _TAIL_LAUNCH[0](args, kw, tp_gather, tokens)


def fused_tail(ca, tail, operands, res_inp, w, eps, gemma_norm, lamport_order, fp8_out):
    if _TAIL_LAUNCH is None:
        return None
    return _TAIL_LAUNCH[1](ca, tail, operands, res_inp, w, eps, gemma_norm, lamport_order, fp8_out)


def has_deferred_moe_tail() -> bool:
    return bool(_PENDING)


def take_deferred_moe_tail(x: torch.Tensor) -> MoeDeferredTail | None:
    """The pending tail whose output is ``x`` (same storage, offset, shape and
    dtype), removed from the registry: the caller computes or materializes it."""
    if not _PENDING or not isinstance(x, torch.Tensor):
        return None
    tail = _PENDING.get(x.data_ptr())
    if tail is None:
        return None
    o = tail.out
    _PENDING.pop(x.data_ptr())
    if x.shape != o.shape or x.dtype != o.dtype or x.stride() != o.stride():
        # another view of the output: write it, the caller reads it as is
        tail.materialize()
        return None
    return tail


def materialize_deferred_moe_tail(x: torch.Tensor | None = None) -> None:
    """Write the pending tail whose output is ``x`` (every pending tail when
    ``x`` is None) before a reader that does not take it."""
    if not _PENDING:
        return
    if x is None:
        for key in list(_PENDING):
            _PENDING.pop(key).materialize()
        return
    tail = _PENDING.pop(x.data_ptr(), None) if isinstance(x, torch.Tensor) else None
    if tail is not None:
        tail.materialize()
