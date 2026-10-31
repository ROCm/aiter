# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bisect ladder: which FlyDSL construct hangs the gfx1250 queue?

The ``6c272c3c`` page-4 QSA scorer was dispatched **once** on a healthy single
gfx1250 node (d01-3) and hung it: the launch returned (0.8 s), then
``torch.cuda.synchronize()`` never returned, the GPU stayed 100 % busy, and
``dmesg`` showed ``MES(0, 0) failed to respond to msg=REMOVE_QUEUE`` /
``Failed to evict queue 1`` / ``Failed to quiesce KFD`` - the queue never
completes and firmware cannot evict it, so **a hang costs the node a reset**.
There was no ``VM_PAGE_FAULT`` at that moment, so no address left its buffer:
the address arithmetic is not the suspect (see ``records/g2/builder2-log.md`` sec 2
and ``records/g2/builder3-log.md``).

That makes every GPU attempt expensive, so this driver never runs "the kernel"
first.  It runs a ladder of **minimal, independent FlyDSL kernels**, one per
process, each with a hard wall-clock timeout (default **10 s**), printing
    VARIANT <name> START
    VARIANT <name> OK | FAIL | TIMEOUT | SKIP <seconds>[ phase=ready|compile|kernel]

with ``flush=True``, plus ``bs_<name>.log`` (the child's own trace) and
``bs_<name>.json`` per variant and ``bisect_summary.json`` at the end.  The child
prints its own progress markers (``fixture`` / ``built`` / ``dispatch`` /
``dispatched`` / ``synchronized`` / ``readback``) *and* two marker lines the
parent budgets against, so a hang is localised inside a variant **and** in the
right phase:

    CHILD <name> READY        imports + fixtures + kernel traced; the queue has
                              not been touched      --ready-timeout, 180 s
    CHILD <name> DISPATCHED   compile + launch returned  --compile-timeout, 60 s
    (process exit)            synchronize + readback     --timeout, 10 s  <-- the
                              only phase that can wedge the GPU

A timeout in the first two phases is an import/JIT stall, never a device event;
only ``phase=kernel`` sets ``device_suspect`` in the summary (BISECT.md sec 1).

Every rung isolates **one** construct and then checks the *result* exactly (int32
/ fp32 values, no golden module needed), so a passing rung is evidence that the
construct both completes *and* does the right thing:

    probe   env, dev            versions / device health; no flydsl kernel
    launch  empty               body-less kernel: module load + kernarg + launch
    store   store_uncond_raw    plain dword store, all 256 lanes in bounds
            store_uncond_gridint  same, grid as Python ints (not ``fx.Int64``)
            store_pred_raw      same store under a lane-varying ``if col < 4``
            store_uncond_cop    copy-atom layer (``make_view`` + ``copy_atom_call``)
            store_pred_cop      copy-atom layer under the lane-varying ``if``
            meta_store_raw      the ``writer``-predicated int32 metadata stores
            meta_store_cop      same through the copy-atom layer
    load    load_scalar_raw     4 B int32 load (metadata path)
            load_vec_raw        2 x 16 B loads, summed (Q / KV path)
            load_vec_cop        same through the copy-atom layer
            nostore             loads + integer FMA chain, no store at all
    sync    barrier_probe       ``fx.gpu.barrier()`` in uniform control flow
    module  full_v1             the fixed module, tiny bf16 case (control)
            full_v1_cop         same module, ``QSA_G2_ACCESS=cop`` (old layer)
            v0_orig             frozen ``6c272c3c`` module - only with
                                ``--include-orig``, always last

Order matters because **the first TIMEOUT stops the ladder** (STOP RULE, see
``BISECT.md``): the ladder climbs from "nothing" (empty) through the two access
layers to the whole kernel, so whatever hangs is the *smallest* thing that
hangs, and everything before it is proven good.

Usage::

    python3 -u bisect.py                     # the whole ladder, 10 s per variant
    python3 -u bisect.py --list              # the plan, no GPU touched
    python3 -u bisect.py --dry-run           # validate driver + fixtures, no GPU
    python3 -u bisect.py --only empty,store_pred_cop
    python3 -u bisect.py --include-orig      # add the v0 control, last
    python3 -u bisect.py --timeout 10 --out /work/results/g2/builder/bisect
    bash bisect.sh 10 b1                     # container wrapper: same, with env

Exit codes: 0 all OK/SKIP, 2 a variant FAILed, 3 a variant TIMEOUT (device
suspect: stop, capture dmesg, ask for a reset - do not retry).
"""

from __future__ import annotations

import os
import sys

# This file must be called ``bisect.py`` (deliverable spec), and a script's own
# directory is ``sys.path[0]`` - so ``import bisect`` from anywhere later in the
# process (``random`` -> ``tempfile`` -> ``torch``) resolves to *this* file.
# Two consequences, both handled here and in the shim at the bottom:
#
#   * when this file is the entry point, drop the script directory before
#     anything imports torch (children are spawned by absolute path, so they hit
#     the same line);
#   * when this file is *imported as the stdlib module* by a sibling driver
#     (`python3 test_contract_cpu.py` also runs with `_g2` as sys.path[0]), the
#     ``__getattr__`` shim at the bottom re-exports the real ``bisect`` API so
#     the neighbours keep working - without it every driver in this directory
#     dies with ``ImportError: cannot import name 'bisect' from 'bisect'``.
_HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__" and sys.path and os.path.abspath(sys.path[0] or ".") == _HERE:
    del sys.path[0]

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

HERE = _HERE

KERNEL_REL = "aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py"
V0_REV = "6c272c3c"          # the build that hung d01-3
PAGE, HEADS, HDIM = 4, 4, 128
THREADS = 256
RC_OK, RC_FAIL, RC_SKIP, RC_TIMEOUT = 0, 2, 4, 3
LABEL = {RC_OK: "OK", RC_FAIL: "FAIL", RC_TIMEOUT: "TIMEOUT", RC_SKIP: "SKIP"}

STOP_RULE = (
    "STOP RULE (bisect): a TIMEOUT means the queue never completed, i.e. the "
    "device is very likely wedged. Do NOT retry, do NOT run the next variant, "
    "do NOT run the full ladder. Capture `dmesg | tail -40`, "
    "`rocm-smi --showpids`, the children's bs_*.log, and request a node reset. "
    "One timeout per session is the budget."
)


# ---------------------------------------------------------------------------
# Ladder description
# ---------------------------------------------------------------------------
class Variant:
    """One rung: what it isolates, and what OK / a hang there would mean."""

    def __init__(self, name, kind, what, ok_means, on_hang, env=None, timeout=None):
        self.name = name
        self.kind = kind              # probe | mini | module
        self.what = what              # the single construct under test
        self.ok_means = ok_means      # conclusion if it passes
        self.on_hang = on_hang        # conclusion + next action if it hangs
        self.env = env or {}
        self.timeout = timeout        # seconds; None -> --timeout
        self.kernel_path = None       # module variants: explicit file


def _ladder():
    V = Variant
    return [
        V("env", "probe",
          "python / torch / flydsl versions + device enumeration (no tensor op)",
          "the image carries the expected stack; continue",
          "the session cannot even enumerate the device: not a kernel problem"),
        V("dev", "probe",
          "one trivial torch GPU op (`zeros(64).sum()`) - the post-mortem's probe",
          "the device is healthy *before* any FlyDSL kernel; continue",
          "the device was already wedged when the ladder started: stop, this is "
          "an environment fault, do not blame the kernel, request a reset"),
        V("empty", "mini",
          "body-less `@flyc.kernel` with 2 unused tensor args + 2 unused Int32 "
          "kernargs, `known_block_size=[256,1,1]`, `grid=(Int64(1),Int64(1),1)`",
          "module load + kernarg packing + launch of a trivial kernel is fine",
          "the launch path itself (kernarg packing / grid / block / stream / "
          "code object) is the problem, not any memory access: rerun with "
          "`--only empty --no-stream` to split the stream out"),
        V("store_uncond_raw", "mini",
          "raw layer: one unconditional `buffer_store_dword` per lane, all 256 "
          "lanes in bounds (columns=256)",
          "the raw store primitive + a runtime-extent descriptor are fine",
          "a plain in-bounds unconditional dword store hangs => the store "
          "instruction/descriptor is the problem (suspicion (b)/(c))"),
        V("store_uncond_gridint", "mini",
          "identical to store_uncond_raw but `grid=(1,1,1)` as Python ints "
          "instead of `fx.Int64`",
          "`fx.Int64` grid dims are equivalent to int grid dims (suspicion (d) "
          "closed)",
          "`fx.Int64` grid dims are the difference: launch with Python-int grid "
          "values (one-line change in the `@flyc.jit` wrapper)"),
        V("store_pred_raw", "mini",
          "raw layer, same store but inside `if col < 4:` - true for lanes 0-3, "
          "false for the other 252",
          "a lane-varying `if` around a raw store is fine (suspicion (a) closed "
          "for the raw layer)",
          "a lane-varying `if` around a store hangs => replace the store "
          "predicates with the `_guarded` idiom of "
          "`mqa_logits/fp8_paged_mqa_logits_gfx1250.py:93-99` (see BISECT.md)"),
        V("store_uncond_cop", "mini",
          "copy-atom layer: `make_view(ptr + off)` + `memref_store_vec` + "
          "`copy_atom_call(BufferCopy(32), Int8)`, unconditional",
          "the copy-atom store path is fine on its own",
          "the copy-atom store layer hangs even without a predicate => the "
          "access layer is the hang, `raw` (proven above) is the fix"),
        V("store_pred_cop", "mini",
          "copy-atom layer under the lane-varying `if col < 4` - exactly what "
          "`6c272c3c` did (register-array allocation inside a divergent `if`)",
          "the `6c272c3c` store shape is fine in isolation",
          "the `6c272c3c` store shape is the hang: the raw layer (proven above) "
          "is the fix and it is already the module default"),
        V("meta_store_raw", "mini",
          "the `writer`-predicated metadata path: two int32 stores under "
          "`if (tid == 0) & (bid == 0)` (one lane of wave 0)",
          "the second store type (int32) + a 1-of-256 predicate are fine",
          "the metadata store path is the hang (suspicion (b): mixed store types "
          "/ 1-lane predicate)"),
        V("meta_store_cop", "mini",
          "same two int32 stores through the copy-atom layer",
          "the copy-atom metadata store path is fine in isolation",
          "the copy-atom layer hangs on the metadata store but not on the "
          "logits store => the copy-atom layer is the common factor"),
        V("load_scalar_raw", "mini",
          "raw layer: one 4 B int32 load from a known offset, value stored back "
          "and checked exactly (`== 7`)",
          "the scalar dword load + runtime-extent descriptor are fine",
          "the scalar metadata load path is the hang"),
        V("load_vec_raw", "mini",
          "raw layer: two 16 B loads (byte offset 0 and 16) summed, value stored "
          "back and checked exactly (`== 28`)",
          "16 B dwordx4 loads + descriptor + `s_waitcnt` are fine (suspicion (c) "
          "closed for the raw layer)",
          "16 B vector loads are the hang => the Q/KV load path (suspicion (c))"),
        V("load_vec_cop", "mini",
          "copy-atom layer: same two 16 B loads through `BufferCopy(128)` / "
          "`memref_load_vec`",
          "the copy-atom load path is fine in isolation",
          "the copy-atom layer hangs on loads too => the copy-atom layer is the "
          "common factor (and `raw` is already proven)"),
        V("nostore", "mini",
          "loads + an 8-deep integer FMA chain, **no store at all** (the "
          "compiler may dead-code-eliminate the loads; see BISECT.md)",
          "loads/compute with no memory side effect complete (or are removed "
          "entirely) - nothing here hangs",
          "loads/compute alone hang even with no store => pure load-path hang"),
        V("barrier_probe", "mini",
          "`fx.gpu.barrier()` in uniform control flow, then a dword store "
          "(SKIP if the API exposes no barrier)",
          "barriers are usable at all (this kernel does not need one)",
          "barriers hang: do not put one in this kernel"),
        V("full_v1", "module",
          "the module at HEAD (fixed access layer), tiny bf16 case: 1 row, 1 "
          "request, 4 columns; logits NaN-prefilled and checked exactly "
          "`[0, -inf, -inf, -inf]`, `visible == 1`, `row_starts == 0`",
          "the whole kernel runs and is correct on its first case => the fix "
          "holds; go to the full first-touch ladder",
          "the fix is not the fix: the ladder above has already proven which "
          "layer is safe - bisect the module against it"),
        V("full_v1_cop", "module",
          "the same module with `QSA_G2_ACCESS=cop` (the `6c272c3c` access "
          "layer), same case",
          "the old layer works too => the hang is *not* in the access layer and "
          "the ladder's other rungs must be read again",
          "reproduces the d01-3 hang with the old layer only => the access-layer "
          "change is exactly the fix", env={"QSA_G2_ACCESS": "cop"}),
        V("v0_orig", "module",
          "the frozen `6c272c3c` module (`git show`), same tiny case - the "
          "historical control; only with `--include-orig`, always last",
          "the control *does not* hang => the d01-3 hang was not this kernel "
          "(environment), stop and re-read the evidence",
          "reproduces d01-3 (expected): the night's work is confirmed and the "
          "fix above is the answer"),
    ]


VARIANTS = _ladder()
BY_NAME = {v.name: v for v in VARIANTS}


# ---------------------------------------------------------------------------
# Fixtures: deterministic, integer-exact, so the GPU check is exact
# ---------------------------------------------------------------------------
# (dtype, length) per tensor slot; every variant kernel takes exactly two
# tensors (A, B) and two Int32 kernargs (rows, columns).
FIXTURES = {
    "empty": {"A": ("i32", 1), "B": ("i32", 4)},
    "store_uncond_raw": {"A": ("i32", 256), "B": ("i32", 4)},
    "store_uncond_gridint": {"A": ("i32", 256), "B": ("i32", 4)},
    "store_pred_raw": {"A": ("f32", 4), "B": ("f32", 4)},
    "store_uncond_cop": {"A": ("i32", 256), "B": ("i32", 4)},
    "store_pred_cop": {"A": ("f32", 4), "B": ("f32", 4)},
    "meta_store_raw": {"A": ("i32", 1), "B": ("i32", 1)},
    "meta_store_cop": {"A": ("i32", 1), "B": ("i32", 1)},
    "load_scalar_raw": {"A": ("i32", 64), "B": ("i32", 1)},
    "load_vec_raw": {"A": ("i32", 64), "B": ("i32", 1)},
    "load_vec_cop": {"A": ("i32", 64), "B": ("i32", 1)},
    "nostore": {"A": ("i32", 64), "B": ("i32", 1)},
    "barrier_probe": {"A": ("i32", 1), "B": ("i32", 4)},
}

# Expected post-conditions; tensors are zero-filled, A is `arange(n)` for the
# load variants so the expected dword is known without a GPU.
EXPECT = {
    "empty": {},
    "store_uncond_raw": {"A": {i: 0x1234 for i in range(256)}, "B": {}},
    "store_uncond_gridint": {"A": {i: 0x1234 for i in range(256)}, "B": {}},
    "store_pred_raw": {"A": {i: 7.0 for i in range(4)}, "B": {}},
    "store_uncond_cop": {"A": {i: 0x1234 for i in range(256)}, "B": {}},
    "store_pred_cop": {"A": {i: 7.0 for i in range(4)}, "B": {}},
    "meta_store_raw": {"A": {0: 5}, "B": {0: 0}},
    "meta_store_cop": {"A": {0: 5}, "B": {0: 0}},
    "load_scalar_raw": {"A": {}, "B": {0: 7}},
    "load_vec_raw": {"A": {}, "B": {0: 28}},
    "load_vec_cop": {"A": {}, "B": {0: 28}},
    "nostore": {"A": {}, "B": {0: 0}},
    "barrier_probe": {"A": {0: 0x5A5A}, "B": {}},
}

# (rows, columns) kernargs per variant.
SCALARS = {
    "empty": (1, 1),
    "store_uncond_raw": (1, 256),
    "store_uncond_gridint": (1, 256),
    "store_pred_raw": (1, 4),
    "store_uncond_cop": (1, 256),
    "store_pred_cop": (1, 4),
    "meta_store_raw": (1, 1),
    "meta_store_cop": (1, 1),
    "load_scalar_raw": (64, 1),
    "load_vec_raw": (64, 4),
    "load_vec_cop": (64, 4),
    "nostore": (64, 4),
    "barrier_probe": (1, 1),
}


def _make_fixtures(name, dev):
    import torch

    out = []
    for slot in ("A", "B"):
        dt, n = FIXTURES[name][slot]
        if name in ("load_scalar_raw", "load_vec_raw", "load_vec_cop", "nostore") \
                and slot == "A":
            t = torch.arange(n, dtype=torch.int32)
        else:
            t = torch.zeros(n, dtype=torch.int32 if dt == "i32" else torch.float32)
        out.append(t.to(dev))
    return out


def _check(name, tensors):
    """Compare A/B with EXPECT; returns (ok, mismatch-detail)."""
    want = EXPECT[name]
    detail, bad = {}, {}
    for slot, t in zip(("A", "B"), tensors):
        got = t.tolist()
        for i, v in want.get(slot, {}).items():
            g = got[i]
            good = (g == v) if isinstance(v, float) else (int(g) == int(v))
            detail[f"{slot}[{i}]"] = {"want": v, "got": g, "ok": good}
            if not good:
                bad[f"{slot}[{i}]"] = detail[f"{slot}[{i}]"]
    return not bad, bad


# ---------------------------------------------------------------------------
# flydsl helpers (each kernel child re-implements the module's access layer)
# ---------------------------------------------------------------------------
def _fx():
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    return fx, flyc


def _imin(a, b):
    return (a <= b).select(a, b)


_AUX: dict = {}


def _aux():
    """``(ir, aux)`` for the raw ROCDL intrinsics, resolved once per process.

    Mirrors the kernel module's ``_raw()`` on purpose (flydsl moved the buffer
    cache policy from an i32 operand to an enum attribute between releases, and
    ``aiter/ops/flydsl/kernels/buffer_ops.py:44-48`` detects it the same way).
    Deliberately duplicated instead of imported: the ladder must keep working
    when the module under test does not.
    """
    if not _AUX:
        import inspect

        from flydsl._mlir import ir

        fx, _ = _fx()
        is_attr = (
            inspect.signature(fx.rocdl.RawPtrBufferLoadOp).parameters["aux"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        aux = (
            ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)
            if is_attr
            else fx.Int32(0).ir_value()
        )
        _AUX.update(ir=ir, aux=aux)
    return _AUX["ir"], _AUX["aux"]


def _desc(tensor, nbytes, access):
    """One pinned descriptor for *tensor*, extent *nbytes* (bytes, i64)."""
    fx, _ = _fx()
    ptr = fx.rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, fx.get_iter(tensor)),
        num_records_bytes=fx.Int64(nbytes),
    )
    return ptr if access == "cop" else fx.rocdl.get_buffer_rsrc(ptr)


def _load_vec(desc, byte_off, n, access):
    """``n`` dwords at byte offset ``byte_off`` (one buffer_load_dwordxN)."""
    fx, _ = _fx()
    if access == "cop":
        src = fx.make_view(desc + byte_off, fx.make_layout(n * 4, 1))
        reg = fx.make_rmem_tensor(fx.make_layout(n, 1), fx.Int32)
        fx.copy_atom_call(
            fx.make_copy_atom(fx.rocdl.BufferCopy(n * 32), fx.Int8), src, reg
        )
        return fx.Vector(fx.memref_load_vec(reg))
    ir, aux = _aux()
    return fx.Vector(
        fx.rocdl.raw_ptr_buffer_load(
            ir.VectorType.get([n], fx.Int32.ir_type),
            desc,
            fx.Int32(byte_off).ir_value(),
            fx.Int32(0).ir_value(),
            aux=aux,
        )
    )


def _load_1(desc, byte_off, access):
    fx, _ = _fx()
    if access == "cop":
        src = fx.make_view(desc + byte_off, fx.make_layout(4, 1))
        reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
        fx.copy_atom_call(fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Int8), src, reg)
        return fx.Int32(fx.Vector(fx.memref_load_vec(reg))[0])
    _, aux = _aux()
    return fx.Int32(
        fx.rocdl.raw_ptr_buffer_load(
            fx.Int32.ir_type,
            desc,
            fx.Int32(byte_off).ir_value(),
            fx.Int32(0).ir_value(),
            aux=aux,
        )
    )


def _store_1(desc, byte_off, value, elem, access):
    """One buffer_store_dword of *value* (fx.Int32 / fx.Float32) at *byte_off*."""
    fx, _ = _fx()
    if access == "cop":
        dst = fx.make_view(desc + byte_off, fx.make_layout(4, 1))
        reg = fx.make_rmem_tensor(fx.make_layout(1, 1), elem)
        fx.memref_store_vec(fx.Vector.from_elements([value], dtype=elem), reg)
        fx.copy_atom_call(fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Int8), reg, dst)
        return
    _, aux = _aux()
    fx.rocdl.raw_ptr_buffer_store(
        elem(value).ir_value(),
        desc,
        fx.Int32(byte_off).ir_value(),
        fx.Int32(0).ir_value(),
        aux=aux,
    )


def _dispatch(launcher, args):
    """Compile-once-then-launch, exactly like the module's ``_run``."""
    import flydsl.compiler as flyc

    cf = getattr(launcher, "_cf", None)
    if cf is None:
        cf = flyc.compile(launcher, *args)
        launcher._cf = cf
    cf(*args)


# ---------------------------------------------------------------------------
# Mini kernels: exactly one construct each (a factory per rung, so the traced
# body never branches on a Python value - that is a *traced* `if` in FlyDSL)
# ---------------------------------------------------------------------------
def _k_empty(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        pass

    return kern


def _k_store_uncond(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        col = fx.Int32(fx.block_idx.x) * THREADS + tid
        out = _desc(A, fx.Int64(rows) * fx.Int64(columns) * fx.Int64(4), access)
        _store_1(out, col * 4, fx.Int32(0x1234), fx.Int32, access)

    return kern


def _k_store_pred(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        col = fx.Int32(fx.block_idx.x) * THREADS + tid
        out = _desc(A, fx.Int64(rows) * fx.Int64(columns) * fx.Int64(4), access)
        # lane-varying predicate: true for 4 of 256 lanes
        if col < columns:
            _store_1(out, _imin(col, columns - 1) * 4, fx.Float32(7.0), fx.Float32,
                     access)

    return kern


def _k_meta_store(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        vis = _desc(A, fx.Int64(rows) * fx.Int64(4), access)
        rst = _desc(B, fx.Int64(rows) * fx.Int64(4), access)
        # 1-of-256 predicate, the module's `writer` shape
        if (tid == 0) & (bid == 0) & (fx.Int32(0) < rows):
            _store_1(vis, fx.Int32(0), fx.Int32(5), fx.Int32, access)
            _store_1(rst, fx.Int32(0), fx.Int32(0), fx.Int32, access)

    return kern


def _k_load_scalar(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        src = _desc(A, fx.Int64(rows) * fx.Int64(4), access)
        dst = _desc(B, fx.Int64(4), access)
        # Every lane loads the same dword: the load itself is not inside a branch,
        # so a hang here cannot be blamed on predication.
        v = _load_1(src, fx.Int32(7) * 4, access)      # A[7] == 7
        if fx.Int32(fx.thread_idx.x) == 0:
            _store_1(dst, fx.Int32(0), v, fx.Int32, access)

    return kern


def _k_load_vec(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        src = _desc(A, fx.Int64(rows) * fx.Int64(4), access)
        dst = _desc(B, fx.Int64(4), access)
        lo = _load_vec(src, fx.Int32(0), 4, access)     # A[0..3]
        hi = _load_vec(src, fx.Int32(16), 4, access)    # A[4..7]
        total = fx.Int32(0)
        for j in fx.range_constexpr(4):
            total = total + fx.Int32(lo[j]) + fx.Int32(hi[j])
        if fx.Int32(fx.thread_idx.x) == 0:
            _store_1(dst, fx.Int32(0), total, fx.Int32, access)

    return kern


def _k_nostore(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        src = _desc(A, fx.Int64(rows) * fx.Int64(4), access)
        total = fx.Int32(0)
        for it in fx.range_constexpr(8):
            v = _load_vec(src, fx.Int32(it * 16), 4, access)
            for j in fx.range_constexpr(4):
                total = total * fx.Int32(3) + fx.Int32(v[j])
        # no store anywhere: `total` is deliberately dropped.  A dead value is
        # removable, so this rung may degenerate to a body-less kernel - see
        # BISECT.md for how to read that.

    return kern


def _k_barrier(fname, access):
    fx, flyc = _fx()

    @flyc.kernel(name=fname, known_block_size=[THREADS, 1, 1])
    def kern(A, B, rows: fx.Int32, columns: fx.Int32):
        dst = _desc(A, fx.Int64(4), access)
        fx.gpu.barrier()
        if (fx.Int32(fx.thread_idx.x) == 0) & (fx.Int32(fx.block_idx.x) == 0):
            _store_1(dst, fx.Int32(0), fx.Int32(0x5A5A), fx.Int32, access)

    return kern


_KERNEL_FACTORY = {
    "empty": _k_empty,
    "store_uncond_raw": _k_store_uncond,
    "store_uncond_gridint": _k_store_uncond,
    "store_uncond_cop": _k_store_uncond,
    "store_pred_raw": _k_store_pred,
    "store_pred_cop": _k_store_pred,
    "meta_store_raw": _k_meta_store,
    "meta_store_cop": _k_meta_store,
    "load_scalar_raw": _k_load_scalar,
    "load_vec_raw": _k_load_vec,
    "load_vec_cop": _k_load_vec,
    "nostore": _k_nostore,
    "barrier_probe": _k_barrier,
}


def _access_of(name):
    return "cop" if name.endswith("_cop") else "raw"


def _mini_launcher(name):
    """Build (kernel, launcher) for a rung; returns the launcher."""
    fx, flyc = _fx()
    access = _access_of(name)
    if name == "barrier_probe" and not hasattr(fx.gpu, "barrier"):
        raise _Skip("flydsl exposes no fx.gpu.barrier()")
    fname = f"g2b3_{name}"
    kern = _KERNEL_FACTORY[name](fname, access)
    grid_x = fx.Int64(1) if name != "store_uncond_gridint" else 1
    grid_y = fx.Int64(1) if name != "store_uncond_gridint" else 1

    @flyc.jit
    def launch(A, B, rows, columns, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        kern._func.__name__ = fname
        kern(A, B, rows, columns).launch(
            grid=(grid_x, grid_y, 1), block=(THREADS, 1, 1), stream=stream
        )

    launch.kernel_name = fname
    launch.access = access
    return launch


class _Skip(Exception):
    """This rung cannot run on this toolchain (reported as SKIP)."""


# ---------------------------------------------------------------------------
# Child: run one variant
# ---------------------------------------------------------------------------
def _device():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/HIP device")
    return torch.device("cuda")


def child_probe(spec, mark, args):
    import torch

    print(f"CHILD {spec.name} READY", flush=True)
    print(f"CHILD {spec.name} DISPATCHED", flush=True)
    if spec.name == "env":
        info = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        try:
            import flydsl

            info["flydsl"] = getattr(flydsl, "__version__", "?")
            info["flydsl_file"] = getattr(flydsl, "__file__", "?")
        except Exception as exc:  # noqa: BLE001
            info["flydsl_error"] = f"{type(exc).__name__}: {exc}"
        if info["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            info.update({"gpu": p.name, "arch": getattr(p, "gcnArchName", None),
                         "cu": p.multi_processor_count})
        print("CHILD env " + json.dumps(info, default=str), flush=True)
        return {"ok": True, **info}

    dev = _device()
    mark("device")
    t0 = time.time()
    y = torch.zeros(64, dtype=torch.float32, device=dev).sum().item()
    torch.cuda.synchronize()
    mark("synchronized")
    return {"ok": y == 0.0, "sum": y, "seconds": round(time.time() - t0, 3)}


def child_mini(spec, mark, args):
    import torch

    dev = _device()
    tensors = _make_fixtures(spec.name, dev)
    rows, columns = SCALARS[spec.name]
    mark("fixture")
    launcher = _mini_launcher(spec.name)
    mark("built")
    print(f"CHILD {spec.name} READY", flush=True)
    call_args = [tensors[0], tensors[1], rows, columns]
    if not args.no_stream:
        call_args.append(torch.cuda.current_stream(dev))
    mark("dispatch")
    _dispatch(launcher, call_args)
    mark("dispatched")
    print(f"CHILD {spec.name} DISPATCHED", flush=True)
    torch.cuda.synchronize()
    mark("synchronized")
    ok, bad = _check(spec.name, tensors)
    mark("readback")
    print(f"CHILD {spec.name} access={launcher.access} check="
          f"{json.dumps(bad, default=str)[:800]}", flush=True)
    return {"ok": ok, "access": launcher.access, "mismatch": bad,
            "stream": "off" if args.no_stream else "torch"}


def _kernel_path():
    return os.path.abspath(
        os.path.join(HERE, "..", "qsa_paged_mqa_logits_page4_gfx1250.py"))


def _load_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("qsa_bisect_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, f"file:{path}"


def child_module(spec, mark, args):
    import torch

    dev = _device()
    path = spec.kernel_path or _kernel_path()
    mod, how = _load_module(path)
    print(f"CHILD {spec.name} module {how} "
          f"access={getattr(mod, '_ACCESS', '(none)')}", flush=True)
    mark("module")

    q = torch.zeros((1, HEADS, HDIM), dtype=torch.bfloat16, device=dev)
    cache = torch.zeros((1, PAGE, 1, HDIM), dtype=torch.bfloat16, device=dev)
    table = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    t2r = torch.zeros((1,), dtype=torch.int32, device=dev)
    qp = torch.full((1,), 3, dtype=torch.int32, device=dev)
    ctx = torch.full((1,), 16, dtype=torch.int32, device=dev)
    # NaN pre-fill: the kernel must overwrite *every* column of the row.
    logits = torch.full((1, 4), float("nan"), dtype=torch.float32, device=dev)
    vis = torch.full((1,), -1, dtype=torch.int32, device=dev)
    rst = torch.full((1,), -1, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    mark("fixture")
    print(f"CHILD {spec.name} READY", flush=True)

    mark("dispatch")
    t0 = time.time()
    out, out_vis = mod.qsa_paged_mqa_logits_page4(
        q, cache, table, t2r, qp, ctx, compress_ratio=PAGE,
        compressed_k_scale=None, logits=logits, visible_groups=vis, row_starts=rst,
    )
    dispatch_s = time.time() - t0
    mark("dispatched")
    print(f"CHILD {spec.name} DISPATCHED", flush=True)
    torch.cuda.synchronize()
    mark("synchronized")

    got = [float(v) for v in out[0].tolist()]
    want = [0.0, float("-inf"), float("-inf"), float("-inf")]
    vis_v, rst_v = int(out_vis[0]), int(rst[0])
    bad = {}
    if got != want:
        bad["logits"] = {"want": want, "got": got}
    if vis_v != 1:
        bad["visible"] = {"want": 1, "got": vis_v}
    if rst_v != 0:
        bad["row_starts"] = {"want": 0, "got": rst_v}
    mark("readback")
    print(f"CHILD {spec.name} got={got} visible={vis_v} row_starts={rst_v} "
          f"dispatch={dispatch_s:.2f}s", flush=True)
    return {"ok": not bad, "logits": got, "visible": vis_v, "row_starts": rst_v,
            "dispatch_s": round(dispatch_s, 2), "mismatch": bad, "module": how,
            "access": getattr(mod, "_ACCESS", None)}


def run_child(spec, args):
    out_dir = os.path.abspath(args.out or os.path.join(HERE, "bisect_out"))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    marks = {}

    def mark(what):
        marks[what] = round(time.time() - t0, 2)
        print(f"CHILD {spec.name} {what} {marks[what]:.2f}s", flush=True)

    try:
        if spec.kind == "probe":
            payload = child_probe(spec, mark, args)
        elif spec.kind == "mini":
            payload = child_mini(spec, mark, args)
        else:
            payload = child_module(spec, mark, args)
    except _Skip as exc:
        payload = {"ok": True, "skip": True, "reason": str(exc)}
        print(f"CHILD {spec.name} SKIP {exc}", flush=True)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    payload.update(variant=spec.name, kind=spec.kind, pid=os.getpid(),
                   marks=marks, total_s=round(time.time() - t0, 2))
    path = os.path.join(out_dir, f"bs_{spec.name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"CHILD {spec.name} DONE ok={payload.get('ok')} -> {path}", flush=True)
    if payload.get("skip"):
        return RC_SKIP
    return RC_OK if payload.get("ok") else RC_FAIL


# ---------------------------------------------------------------------------
# Parent: one process per variant, phase-aware hard timeout
# ---------------------------------------------------------------------------
# A fresh process pays the torch/flydsl import and the JIT compile before it can
# hang anything, so a single wall-clock timeout would either be far too long for
# "did the dispatch hang?" or would fire during an import.  The child therefore
# prints two markers and the parent budgets each phase separately:
#
#   CHILD <name> READY       imports + fixtures + kernel traced; nothing left
#                            that can touch the queue
#   CHILD <name> DISPATCHED  compile + launch returned; syncing now
#
# --ready-timeout (180 s) covers import, --compile-timeout (60 s) covers the JIT
# and the launch, and **--timeout (10 s) covers only the part that can wedge the
# GPU**: from `dispatched` to the process exit.  A timeout in the first two
# phases is a JIT/toolchain stall, *not* evidence of a wedged device; only a
# `kernel`-phase timeout sets `device_suspect`.
def _spawn(spec, args):
    out_dir = os.path.abspath(args.out or os.path.join(HERE, "bisect_out"))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"bs_{spec.name}.log")
    env = dict(os.environ)
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.update(spec.env)
    cmd = [sys.executable, "-u", os.path.abspath(__file__),
           "--variant", spec.name, "--out", out_dir]
    if args.no_stream:
        cmd.append("--no-stream")
    timeout = spec.timeout or args.timeout
    kwargs = {"start_new_session": True} if os.name == "posix" else {}
    t0 = time.time()
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
        cwd=HERE, **kwargs
    )
    ev_ready, ev_dispatched = threading.Event(), threading.Event()

    def _pump():
        for raw in iter(proc.stdout.readline, b""):
            log_fh.write(raw)
            log_fh.flush()
            if b" READY" in raw:
                ev_ready.set()
            if b" DISPATCHED" in raw:
                ev_dispatched.set()
            try:
                sys.stdout.write(raw.decode("utf-8", "replace"))
                sys.stdout.flush()
            except Exception:  # noqa: BLE001 - stdout is best effort
                pass

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()

    def _wait(ev, budget):
        """Wait for *ev*; True if set, False on budget exhaustion or child exit."""
        deadline = time.time() + budget
        while True:
            if ev.is_set():
                return True
            if proc.poll() is not None:
                return ev.is_set()
            if time.time() >= deadline:
                return False
            time.sleep(0.02)

    phase = None
    rc = None
    if not _wait(ev_ready, args.ready_timeout):
        if proc.poll() is None:
            phase = "ready"
        else:
            rc = proc.returncode            # died before READY: plain OK/FAIL
    if rc is None and phase is None:
        ready_s = time.time() - t0
        if not _wait(ev_dispatched, args.compile_timeout):
            if proc.poll() is None:
                phase = "compile"
            else:
                rc = proc.returncode        # died compiling/dispatching: FAIL
        compile_s = time.time() - t0 - ready_s
        if phase is None and rc is None:
            kernel_s = time.time() - t0 - ready_s - compile_s
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                phase = "kernel"
            kernel_s = time.time() - t0 - ready_s - compile_s
    secs = time.time() - t0
    if phase is not None:
        # A hung kernel keeps the child inside synchronize(); SIGKILL the group.
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), 9)
            else:  # pragma: no cover - the ladder runs in a Linux container
                proc.kill()
        except Exception:  # noqa: BLE001
            proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        pump.join(timeout=2)
        log_fh.close()
        return RC_TIMEOUT, secs, phase
    pump.join(timeout=5)
    log_fh.close()
    if rc == 0:
        return RC_OK, secs, None
    if rc == RC_SKIP:
        return RC_SKIP, secs, None
    return RC_FAIL, secs, None


def _select(args):
    names = [n.strip() for n in (args.only or "").split(",") if n.strip()]
    if not names:
        specs = [v for v in VARIANTS if v.name != "v0_orig"]
    else:
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown variant(s): {unknown}; try --list")
        specs = [BY_NAME[n] for n in names]
    if args.include_orig and not any(s.name == "v0_orig" for s in specs):
        specs.append(BY_NAME["v0_orig"])
    return specs


def _repo_root():
    d = HERE
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return None


def _materialize_v0(out_dir, rev=V0_REV):
    """``git show`` the frozen kernel into *out_dir* (no network, no checkout)."""
    repo = _repo_root()
    if repo is None:
        return None, "no .git above this file; cannot materialize the v0 control"
    dst = os.path.join(out_dir, f"kernel_{rev[:8]}.py")
    try:
        proc = subprocess.run(["git", "-C", repo, "show", f"{rev}:{KERNEL_REL}"],
                              capture_output=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return None, f"git show failed: {exc}"
    if proc.returncode != 0:
        return None, f"git show {rev} failed: {proc.stderr.decode()[:200]}"
    with open(dst, "wb") as fh:
        fh.write(proc.stdout)
    return dst, f"{rev}:{KERNEL_REL} -> {dst}"


def run_ladder(args):
    out_dir = os.path.abspath(args.out or os.path.join(HERE, "bisect_out"))
    os.makedirs(out_dir, exist_ok=True)
    specs = _select(args)
    origs = [s for s in specs if s.name == "v0_orig"]
    specs = [s for s in specs if s.name != "v0_orig"] + origs
    for spec in origs:                       # materialise the frozen control
        path, how = _materialize_v0(out_dir, args.orig_rev)
        if path is None:
            specs.remove(spec)
            print(f"VARIANT v0_orig SKIP {how}", flush=True)
        else:
            spec.kernel_path = path
            print(f"VARIANT v0_orig kernel {how}", flush=True)

    print(f"BISECT out={out_dir} kernel-timeout={args.timeout:g}s "
          f"compile-timeout={args.compile_timeout:g}s rungs={len(specs)} "
          f"stream={'off' if args.no_stream else 'torch'}", flush=True)
    results, stop = [], None
    for spec in specs:
        print(f"VARIANT {spec.name} START", flush=True)
        rc, secs, phase = _spawn(spec, args)
        where = f" phase={phase}" if phase else ""
        print(f"VARIANT {spec.name} {LABEL[rc]} {secs:.1f}s{where}", flush=True)
        results.append({"variant": spec.name, "rc": rc, "label": LABEL[rc],
                        "seconds": round(secs, 2), "timeout_phase": phase,
                        "what": spec.what, "ok_means": spec.ok_means})
        if rc == RC_TIMEOUT:
            stop = RC_TIMEOUT
            print("", flush=True)
            print(f"HANG-{phase.upper()} {spec.name}: {spec.what}", flush=True)
            if phase == "kernel":
                print(f"HANG means: {spec.on_hang}", flush=True)
                print(STOP_RULE, flush=True)
            else:
                print(f"NOT A DEVICE HANG: the child never reached `dispatched` "
                      f"({phase} phase: import or JIT/launch stall). The queue was "
                      f"probably never touched. Capture the log, do not retry "
                      f"blindly, and try `--only {spec.name} --compile-timeout 300` "
                      f"once to see whether it is a slow cold JIT.", flush=True)
            break
        if rc == RC_FAIL and args.stop_on_fail:
            stop = RC_FAIL
            print(f"FAIL {spec.name} (--stop-on-fail): {spec.what}", flush=True)
            break
    summary = {
        "out": out_dir,
        "timeout_s": args.timeout,
        "compile_timeout_s": args.compile_timeout,
        "ready_timeout_s": args.ready_timeout,
        "stream": "off" if args.no_stream else "torch",
        "variants": results,
        "results": {r["variant"]: r["label"] for r in results},
        "stopped_on": stop,
        "timeout_phase": next((r["timeout_phase"] for r in results
                               if r["timeout_phase"]), None),
        "device_suspect": any(r["rc"] == RC_TIMEOUT and r["timeout_phase"] == "kernel"
                              for r in results),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(out_dir, "bisect_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"BISECT summary -> {path}", flush=True)
    if stop == RC_TIMEOUT:
        if summary["device_suspect"]:
            print("BISECT STOPPED ON A KERNEL-PHASE TIMEOUT - capture dmesg and "
                  "request a node reset.", flush=True)
        else:
            print("BISECT STOPPED ON A NON-KERNEL TIMEOUT (import/JIT) - the device "
                  "is not implicated; report the log.", flush=True)
        return RC_TIMEOUT
    if any(r["rc"] == RC_FAIL for r in results):
        return RC_FAIL
    print("BISECT ALL OK - the full first-touch ladder may follow "
          "(FIRST_TOUCH.md).", flush=True)
    return RC_OK


# ---------------------------------------------------------------------------
# Dry run: validate the driver and the fixtures on the CPU (no GPU, no flydsl)
# ---------------------------------------------------------------------------
def _dry_run(args):
    import torch

    print("BISECT dry-run (no GPU, no flydsl): driver + fixture validation\n")
    rc = 0
    specs = _select(args)
    for spec in specs:
        detail = []
        if spec.kind == "mini":
            name = spec.name
            fx_mod = None
            try:
                import flydsl.expr  # noqa: F401

                fx_mod = "flydsl present"
            except Exception:  # noqa: BLE001
                fx_mod = "flydsl absent (kernels cannot be built here)"
            if name not in FIXTURES:
                detail.append("NO FIXTURE")
                rc = 1
            else:
                a, b = _make_fixtures(name, torch.device("cpu"))
                rows, columns = SCALARS[name]
                if name == "load_scalar_raw":
                    want = int(a[7])
                    got = EXPECT[name]["B"][0]
                    detail.append(f"A[7]={want} expect B[0]={got}")
                    rc |= 0 if want == got else 1
                elif name in ("load_vec_raw", "load_vec_cop"):
                    want = int(a[:8].sum())
                    got = EXPECT[name]["B"][0]
                    detail.append(f"sum(A[0:8])={want} expect B[0]={got}")
                    rc |= 0 if want == got else 1
                elif name == "nostore":
                    detail.append("no store: B must stay 0")
                detail.append(f"rows={rows} columns={columns}")
            detail.append(fx_mod)
        elif spec.kind == "module":
            path = spec.kernel_path or _kernel_path()
            if not os.path.exists(path):
                detail.append(f"MISSING {path}")
                rc = 1
            else:
                detail.append(f"kernel {os.path.basename(path)}")
                if spec.env:
                    detail.append(f"env {spec.env}")
        print(f"PLAN {spec.name:<22} {spec.kind:<7} " + "; ".join(detail))
        if spec.on_hang:
            print(f"     on hang: {spec.on_hang}")
    print(f"\ntimeout {args.timeout:g}s per rung; output {args.out or '(default)'}")
    print("dry-run OK" if rc == 0 else "dry-run FAILED")
    return rc


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", help="run this variant in this process (internal)")
    ap.add_argument("--only", help="comma-separated subset of variants")
    ap.add_argument("--out", help="output directory (logs + json)")
    ap.add_argument("--timeout", type=float,
                    default=float(os.environ.get("QSA_BISECT_TIMEOUT", "10")),
                    help="hard timeout (s) for the kernel phase only: from "
                         "`dispatched` to the process exit (default 10)")
    ap.add_argument("--compile-timeout", type=float,
                    default=float(os.environ.get("QSA_BISECT_COMPILE_TIMEOUT", "60")),
                    help="timeout (s) for the JIT + launch phase (default 60)")
    ap.add_argument("--ready-timeout", type=float,
                    default=float(os.environ.get("QSA_BISECT_READY_TIMEOUT", "180")),
                    help="timeout (s) for imports + fixtures (default 180)")
    ap.add_argument("--include-orig", action="store_true",
                    help="append the frozen v0 (6c272c3c) control, last")
    ap.add_argument("--orig-rev", default=V0_REV, help="git revision of the control")
    ap.add_argument("--no-stream", action="store_true",
                    help="do not pass a torch stream (use the flydsl default)")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="also stop on FAIL (default: only a TIMEOUT stops)")
    ap.add_argument("--list", action="store_true", help="print the ladder and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate driver + fixtures on the CPU, touch no GPU")
    args = ap.parse_args(argv)

    if args.list:
        for spec in VARIANTS + ([BY_NAME["v0_orig"]] if args.include_orig else []):
            print(f"{spec.name:<22} {spec.kind:<7} {spec.what}")
        return RC_OK
    if args.variant:
        spec = BY_NAME.get(args.variant)
        if spec is None:
            raise SystemExit(f"unknown variant {args.variant!r}")
        if spec.name == "v0_orig":
            spec.kernel_path = os.environ.get("QSA_BISECT_KERNEL")
        return run_child(spec, args)
    if args.dry_run:
        return _dry_run(args)
    return run_ladder(args)


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Stdlib compatibility shim (this file shadows the stdlib `bisect` module)
# ---------------------------------------------------------------------------
# `random` does `from bisect import bisect as _bisect` and `torch` imports
# `random`, so *every* driver in this directory that touches torch imports the
# stdlib module - and gets this file.  Re-export the real API from the C
# accelerator, lazily (PEP 562), so running this driver pays nothing for it.
_STDLIB_BISECT = ("bisect", "bisect_left", "bisect_right",
                  "insort", "insort_left", "insort_right")


def __getattr__(name):                  # only called for a *missing* attribute
    if name in _STDLIB_BISECT:
        return _stdlib_bisect()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _stdlib_bisect():
    try:
        import _bisect as c
    except ImportError:                 # pragma: no cover - CPython always has it
        return _stdlib_bisect_py()
    return {"bisect": c.bisect_right, "bisect_left": c.bisect_left,
            "bisect_right": c.bisect_right, "insort": c.insort_right,
            "insort_left": c.insort_left, "insort_right": c.insort_right}


def _stdlib_bisect_py():                # pragma: no cover - fallback only
    def bisect_left(a, x, lo=0, hi=None, *, key=None):
        if lo < 0:
            raise ValueError("lo must be non-negative")
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if (key(a[mid]) if key else a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def bisect_right(a, x, lo=0, hi=None, *, key=None):
        if lo < 0:
            raise ValueError("lo must be non-negative")
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < (key(a[mid]) if key else a[mid]):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def insort_left(a, x, lo=0, hi=None, *, key=None):
        i = bisect_left(a, x, lo, hi, key=key)
        a.insert(i, x)

    def insort_right(a, x, lo=0, hi=None, *, key=None):
        i = bisect_right(a, x, lo, hi, key=key)
        a.insert(i, x)

    return {"bisect": bisect_right, "bisect_left": bisect_left,
            "bisect_right": bisect_right, "insort": insort_right,
            "insort_left": insort_left, "insort_right": insort_right}
