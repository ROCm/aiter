# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stability / reliability stress harness for fused_moe_router.

This is NOT the op test. op_tests/test_fused_moe_router.py answers "is one
launch correct". This answers "does it stay correct under the conditions an
end-to-end serving run actually creates" -- which is where synchronisation and
stale-buffer bugs surface, because they need *history* to show up. It lives
here rather than in a pytest shard for two reasons: the soak suites run
hundreds of launches, and the RISKY ones fail by hanging or aborting the
process rather than by returning wrong numbers.

Three pieces of state outlive a single launch, and every suite here targets
one of them:

  * the barrier semaphore -- one u32 in the caller's workspace, zeroed once at
    allocation and rearmed only by the sense flip
    (see the workspace layout note in fused_moe_router_entry.cu). A flip bug
    corrupts the NEXT launch, so
    nothing that syncs between launches can see it.
  * the tok_scale scratch -- the rest of that workspace, uninitialised and
    never cleared.
  * the caller's output buffers -- reused across steps by vLLM's allocator, so
    the region past num_valid holds whatever was there before. The kernel is
    responsible for the tail fills.

The workspace is allocated per (device, max_tokens) by the Python accessor, so
which device and stream a launch runs on is also part of the state under test.

Suites marked RISKY run in a child process with a watchdog: a grid barrier
that fails to make progress hangs rather than returning wrong answers, and a
cross-device pointer dereference aborts rather than raising.

    python op_tests/multigpu_tests/test_fused_moe_router_stress.py       # all
    python op_tests/multigpu_tests/test_fused_moe_router_stress.py -l    # list
    python op_tests/multigpu_tests/test_fused_moe_router_stress.py -k barrier
    python op_tests/multigpu_tests/test_fused_moe_router_stress.py --soak 600
"""

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_OPTEST = _HERE.parent / "test_fused_moe_router.py"


def _load_optest():
    """Reuse the op test's reference harness rather than forking it."""
    spec = importlib.util.spec_from_file_location("fmr_optest", _OPTEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load_optest()

W_TOL = T.W_TOL

# The base shape everything varies off: 320 experts / topk 8 is the decode
# routing this kernel was built for.
E0, TOPK0, U0 = 320, 8, 16


def _fmt(errs):
    """Drop the metrics that are within tolerance, so a failure line shows only
    what actually went wrong."""
    return {k: v for k, v in errs.items() if (v > W_TOL if k.endswith("_w_err") else v)}


_fused = T._pin_path
_unpin = T._unpin_path
_fill_pat = T._fill_pat
_alloc_poisoned = T._alloc_poisoned
_tail_errs = T._tail_errs


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _case(M, E=E0, topk=TOPK0, unit=U0, mask=None, seed=None, bias_dtype=None):
    """Inputs plus the stock reference for one shape."""
    g, b, h = T._inputs(M, E, bias_dtype or T.dtypes.bf16, M if seed is None else seed)
    ref = T._run_stock(g, b, h, M, E, topk, unit, True, 1.0, mask)
    return g, b, h, ref


def _errs(ref, got, M, topk, unit):
    return T._compare(ref, got, M, topk, unit)


def _bad(errs):
    return T._failed(errs)


# --------------------------------------------------------------------------
# suites
# --------------------------------------------------------------------------

SUITES = {}


def suite(name, risky=False, timeout=300):
    def deco(fn):
        fn.risky = risky
        fn.timeout = timeout
        SUITES[name] = fn
        return fn

    return deco


@suite("barrier_churn")
def s_barrier_churn(cfg):
    """Rearm the barrier across launches whose participant count keeps changing.

    GRID is max(M, 16) capped at num_cu, so varying M varies the number of
    blocks the sense flip has to account for. Nothing syncs between launches:
    if launch N leaves the semaphore mid-sense, launch N+1 is the one that
    reads it wrong.
    """
    fails = []
    _fused(True)
    try:
        ms = [1, 2, 3, 7, 16, 17, 31, 33, 64, 65, 96, 100, 103, 128]
        pre = {m: _case(m) for m in ms}
        torch.cuda.synchronize()

        pending = []
        for i in range(cfg.iters):
            M = ms[(i * 7 + i // len(ms)) % len(ms)]  # non-repeating order
            g, b, h, ref = pre[M]
            got = _alloc_poisoned(ref, M, TOPK0, i)
            T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
            pending.append((i, M, ref, got))
        torch.cuda.synchronize()

        for i, M, ref, got in pending:
            e = _errs(ref, got, M, TOPK0, U0)
            if _bad(e):
                fails.append(f"launch {i} M={M}: {_fmt(e)}")
    finally:
        _unpin()
    return fails, f"{cfg.iters} unsynced launches, GRID churning over {len(ms)} shapes"


@suite("barrier_lds_churn")
def s_barrier_lds_churn(cfg):
    """Same, but also vary E and topk so the LDS layout changes per launch.

    total_routed_rows = M*topk drives the dynamic shmem request, which drives
    occupancy, which drives whether the grid is co-resident at all -- the
    precondition the barrier deadlocks without.
    """
    fails = []
    _fused(True)
    try:
        shapes = [
            (1, 256, 1, 16),
            (64, 256, 2, 32),
            (32, 320, 8, 16),
            (8, 512, 8, 64),
            (100, 320, 8, 128),
            (16, 512, 1, 16),
            (48, 288, 4, 32),
            (2, 320, 8, 16),
        ]
        pre = {s: _case(s[0], s[1], s[2], s[3]) for s in shapes}
        torch.cuda.synchronize()

        pending = []
        for i in range(cfg.iters):
            M, E, topk, unit = shapes[(i * 3) % len(shapes)]
            g, b, h, ref = pre[(M, E, topk, unit)]
            got = _alloc_poisoned(ref, M, topk, i)
            T._call_fused(g, b, h, got, E, topk, unit, True, 1.0, None, None)
            pending.append((i, M, topk, unit, ref, got))
        torch.cuda.synchronize()

        for i, M, topk, unit, ref, got in pending:
            e = _errs(ref, got, M, topk, unit)
            if _bad(e):
                fails.append(f"launch {i} M={M} E/topk/u={topk}/{unit}: {_fmt(e)}")
    finally:
        _unpin()
    return fails, f"{cfg.iters} unsynced launches over {len(shapes)} LDS shapes"


@suite("barrier_split_interleave")
def s_split_interleave(cfg):
    """Alternate fused and split launches with no sync between them.

    The split path never touches the semaphore, so a fused launch that follows
    a run of splits must still find the sense the previous *fused* launch left.
    In production this interleaving is exactly what a mixed prefill/decode
    batch stream produces.
    """
    fails = []
    try:
        ms = [1, 33, 64, 100, 128, 200]
        pre = {m: _case(m) for m in ms}
        torch.cuda.synchronize()

        pending = []
        for i in range(cfg.iters):
            M = ms[i % len(ms)]
            _fused(i % 3 != 0)  # 2 fused : 1 split, so runs of each occur
            g, b, h, ref = pre[M]
            got = _alloc_poisoned(ref, M, TOPK0, i)
            T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
            pending.append((i, M, ref, got))
        torch.cuda.synchronize()

        for i, M, ref, got in pending:
            e = _errs(ref, got, M, TOPK0, U0)
            if _bad(e):
                fails.append(
                    f"launch {i} M={M} ({'fused' if i%3 else 'split'}): " f"{_fmt(e)}"
                )
    finally:
        _unpin()
    return fails, f"{cfg.iters} launches alternating fused/split"


@suite("scratch_staleness")
def s_scratch(cfg):
    """tok_scale is process-wide, uninitialised, and grown but never shrunk.

    A large M sizes it and fills it; every later smaller M reuses the same
    allocation with the large run's scale bytes still in the rows above its
    own. Phase 3 must never read a row it did not write this launch.
    """
    fails = []
    _fused(True)
    try:
        # Descend so each launch inherits a buffer filled by a wider one, and
        # never sync in between -- a stale read is a race, not just a stale value.
        order = [128, 100, 64, 33, 16, 8, 4, 2, 1]
        pre = {m: _case(m) for m in order}
        torch.cuda.synchronize()

        for rep in range(cfg.reps):
            pending = []
            for M in order:
                g, b, h, ref = pre[M]
                got = _alloc_poisoned(ref, M, TOPK0, M + rep)
                T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
                pending.append((M, ref, got))
            torch.cuda.synchronize()
            for M, ref, got in pending:
                e = _errs(ref, got, M, TOPK0, U0)
                if _bad(e):
                    fails.append(f"rep {rep} M={M} (descending): {_fmt(e)}")

        # Then force a grow: a wider M than anything seen reallocates the
        # scratch, so the next narrow launch reads a *fresh* uninitialised
        # buffer rather than a warm one.
        for M in (192, 1):
            g, b, h, ref = _case(M)
            got = _alloc_poisoned(ref, M, TOPK0, 9)
            T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
            torch.cuda.synchronize()
            e = _errs(ref, got, M, TOPK0, U0)
            if _bad(e):
                fails.append(f"post-grow M={M}: {_fmt(e)}")
    finally:
        _unpin()
    return fails, f"{cfg.reps} descending sweeps + scratch regrow"


@suite("allocator_poison")
def s_allocator_poison(cfg):
    """Reproduce the real failure mode: outputs land on freed dirty memory.

    Rather than poisoning with fill_(), allocate garbage, free it so the
    caching allocator keeps the block, then allocate the output buffers -- they
    come back pointing at the same bytes. This is what a serving loop does
    every step, and it is the difference between "the kernel writes the right
    values" and "the kernel writes every value it is responsible for".
    """
    fails = []
    for fused in (True, False):
        _fused(fused)
        tag = "fused" if fused else "split"
        try:
            for M in (1, 33, 64, 128):
                g, b, h, ref = _case(M)
                shapes = {
                    k: (ref[k].shape, ref[k].dtype)
                    for k in ("sids", "sw", "seids", "nv", "o4", "osc")
                }
                # Dirty the pool with blocks of exactly the sizes about to be
                # requested, then drop them.
                junk = []
                for _ in range(4):
                    for shape, dt in shapes.values():
                        t = torch.empty(shape, dtype=dt)
                        t.view(torch.uint8).fill_(0xC3)
                        junk.append(t)
                del junk

                got = {
                    "ti": torch.empty(M, TOPK0, dtype=torch.int32),
                    "tw": torch.empty(M, TOPK0, dtype=T.dtypes.fp32),
                }
                for k, (shape, dt) in shapes.items():
                    got[k] = torch.empty(shape, dtype=dt)
                got["nv"].zero_()

                T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
                torch.cuda.synchronize()
                e = _errs(ref, got, M, TOPK0, U0)
                if _bad(e):
                    fails.append(f"{tag} M={M}: {_fmt(e)}")
                fails += _tail_errs(got, M, TOPK0, U0, f"{tag} M={M} dirty-alloc")
        finally:
            _unpin()
    return fails, "outputs allocated onto deliberately dirtied allocator blocks"


@suite("moe_buf_zero")
def s_moe_buf(cfg):
    """moe_buf is zeroed by the kernel because stage 2 accumulates atomically.

    A single missed element is a silent accuracy bug downstream, not a crash,
    and it is grid-strided so which elements get missed depends on GRID -- i.e.
    on M. Check every element, at every M that changes the grid, both paths.
    """
    fails = []
    for fused in (True, False):
        _fused(fused)
        tag = "fused" if fused else "split"
        try:
            for M in (1, 2, 15, 16, 17, 63, 64, 103, 104, 128, 200):
                g, b, h, ref = _case(M)
                got = _alloc_poisoned(ref, M, TOPK0, M)
                buf = torch.empty_like(ref["moe_buf"])
                buf.view(torch.uint8).fill_(0x5A)
                T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, buf)
                torch.cuda.synchronize()
                nz = int((buf != 0).sum())
                if nz:
                    fails.append(
                        f"{tag} M={M}: {nz}/{buf.numel()} moe_buf "
                        "elements left nonzero"
                    )
                e = _errs(ref, got, M, TOPK0, U0)
                if _bad(e):
                    fails.append(f"{tag} M={M} routing wrong with moe_buf: {_fmt(e)}")
        finally:
            _unpin()
    return fails, "every moe_buf element, 11 grid sizes x fused/split"


@suite("determinism_soak")
def s_determinism(cfg):
    """Identical input, many launches, bit-identical output required.

    A race that only loses a store occasionally produces a *correct-looking*
    result almost every time -- comparing against a tolerance-based reference
    hides it. Comparing launches against each other makes any single divergence
    a failure, which is what catches the once-in-a-thousand case.

    out_scale is compared only below num_valid: the swizzled rows above it are
    never written by design, so they still hold the caller's poison and would
    "diverge" every iteration by construction.
    """
    fails = []
    for fused in (True, False):
        _fused(fused)
        tag = "fused" if fused else "split"
        try:
            for M in (1, 64, 128):
                g, b, h, ref = _case(M)
                nv = int(ref["nv"][0])
                base = None
                for i in range(cfg.iters):
                    got = _alloc_poisoned(ref, M, TOPK0, i)
                    T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
                    torch.cuda.synchronize()
                    snap = {
                        k: got[k].clone()
                        for k in ("ti", "tw", "sids", "sw", "seids", "nv", "o4")
                    }
                    snap["osc"] = T._deswizzle(got["osc"], nv).clone()
                    if base is None:
                        base = snap
                        e = _errs(ref, got, M, TOPK0, U0)
                        if _bad(e):
                            fails.append(f"{tag} M={M} iter0 wrong: {_fmt(e)}")
                    else:
                        for k, v in snap.items():
                            if not torch.equal(v, base[k]):
                                n = int((v != base[k]).sum())
                                fails.append(
                                    f"{tag} M={M} iter {i}: {k} diverged "
                                    f"in {n} elements"
                                )
                                base = snap  # resync so one blip is not N failures
                                break
        finally:
            _unpin()
    return fails, f"{cfg.iters} repeats x 3 M x fused/split, bit-identical"


@suite("cuda_graph", risky=True, timeout=420)
def s_graph(cfg):
    """Capture and replay -- the case with no coverage anywhere else.

    Replay reissues the recorded launch without re-running the host code, so
    the semaphore's sense must be self-consistent across replays with nothing
    on the host to fix it up. The scratch pointers are baked into the captured
    node, so a regrow after capture would leave the graph pointing at freed
    memory: warm up first so capture sees a settled allocation.
    """
    fails = []
    _fused(True)
    try:
        for M in (1, 64, 100):
            g, b, h, ref = _case(M)
            got = _alloc_poisoned(ref, M, TOPK0, M)

            # Warm up on a side stream: capture requires the caching allocator
            # and the scratch statics to already be initialised.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)

            for r in range(cfg.replays):
                for k in ("sids", "sw", "seids", "o4", "osc"):
                    _fill_pat(got[k], r)
                got["nv"].zero_()
                graph.replay()
                torch.cuda.synchronize()
                e = _errs(ref, got, M, TOPK0, U0)
                if _bad(e):
                    fails.append(f"M={M} replay {r}: {_fmt(e)}")
                    break
                tf = _tail_errs(got, M, TOPK0, U0, f"M={M} replay {r}")
                if tf:
                    fails += tf
                    break
            del graph
    finally:
        _unpin()
    return fails, f"capture + {cfg.replays} replays x 3 M"


@suite("graph_interleave", risky=True, timeout=420)
def s_graph_interleave(cfg):
    """Graph replays interleaved with eager launches.

    Both share the one semaphore. A replay leaves the sense wherever the
    captured launch left it, and the eager launch that follows has to pick up
    from there -- vLLM does exactly this when it falls out of a captured batch
    size onto the eager path.
    """
    fails = []
    _fused(True)
    try:
        Mg = 64
        gg, bg, hg, refg = _case(Mg)
        gotg = _alloc_poisoned(refg, Mg, TOPK0, 0)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                T._call_fused(gg, bg, hg, gotg, E0, TOPK0, U0, True, 1.0, None, None)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            T._call_fused(gg, bg, hg, gotg, E0, TOPK0, U0, True, 1.0, None, None)

        eager_ms = [1, 17, 33, 96]
        pre = {m: _case(m) for m in eager_ms}
        torch.cuda.synchronize()

        for i in range(cfg.replays):
            graph.replay()
            M = eager_ms[i % len(eager_ms)]
            g, b, h, ref = pre[M]
            got = _alloc_poisoned(ref, M, TOPK0, i)
            T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
            torch.cuda.synchronize()
            e = _errs(ref, got, M, TOPK0, U0)
            if _bad(e):
                fails.append(f"eager M={M} after replay {i}: {_fmt(e)}")
            eg = _errs(refg, gotg, Mg, TOPK0, U0)
            if _bad(eg):
                fails.append(f"graph output after replay {i}: {_fmt(eg)}")
        del graph
    finally:
        _unpin()
    return fails, f"{cfg.replays} alternating replay/eager launches"


@suite("multi_stream", risky=True, timeout=900)
def s_multi_stream(cfg):
    """Concurrent launches on separate streams.

    The semaphore and the tok_scale scratch are one allocation each per
    (device, stream). Two launches that overlap on ONE stream key share both:
    their barrier arrivals sum into the same counter, and their phase-1 scale
    writes target the same tok_scale rows (it is indexed by token, so any
    overlap in token range aliases).

    Run over both launch configs, because they are exposed differently:

      * fused (1 launch) -- phase 1 and phase 3 are separated by the grid
        barrier inside a single kernel, so an interloper has to land inside
        that window.
      * split (2 launches) -- k1 writes tok_scale and k23 reads it from a
        SEPARATE launch (the k1/k23 pair in the entry). The gap between them is an
        ordinary stream gap, so another stream can be scheduled straight into
        it. The wider window in principle -- but both modes fail in practice,
        so run both rather than assuming split subsumes fused.

    Also mix M across and below the kSplitMinTokens=104 crossover so the
    natural (unpinned) dispatch is covered as well as the pinned ones.
    """
    fails = []
    streams = [torch.cuda.Stream() for _ in range(cfg.streams)]
    # Straddle the 104 crossover: the last entry also varies GRID, which sets
    # how many blocks the barrier waits on.
    ms = [16, 48, 103, 104, 128]
    try:
        for mode in ("fused", "split", "auto"):
            if mode == "auto":
                _unpin()
            else:
                _fused(mode == "fused")
            pre = {m: _case(m) for m in ms}
            torch.cuda.synchronize()

            for rep in range(cfg.reps):
                work = []
                for si, st in enumerate(streams):
                    M = ms[(si + rep) % len(ms)]
                    g, b, h, ref = pre[M]
                    got = _alloc_poisoned(ref, M, TOPK0, si)
                    st.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(st):
                        for _ in range(cfg.iters // cfg.streams):
                            T._call_fused(
                                g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None
                            )
                    work.append((si, M, ref, got, st))
                for _, _, _, _, st in work:
                    torch.cuda.current_stream().wait_stream(st)
                torch.cuda.synchronize()
                for si, M, ref, got, _ in work:
                    e = _errs(ref, got, M, TOPK0, U0)
                    if _bad(e):
                        fails.append(f"{mode} rep {rep} stream {si} M={M}: {_fmt(e)}")
    finally:
        _unpin()
    return fails, (
        f"{cfg.streams} concurrent streams x {cfg.reps} reps x "
        f"fused/split/auto, M straddling the split crossover"
    )


@suite("cu_pressure", risky=True, timeout=420)
def s_cu_pressure(cfg):
    """Fused launches while another stream competes for CUs.

    The grid barrier only terminates if every block of the launch is
    co-resident. The host checks that against an *idle* device
    (hipOccupancyMaxActiveBlocksPerMultiprocessor in the entry), which says
    nothing about a device already running someone else's kernel. In a serving
    process there is always someone else.
    """
    fails = []
    _fused(True)
    try:
        bg = torch.cuda.Stream()
        a = torch.randn(4096, 4096, dtype=T.dtypes.bf16)
        bmat = torch.randn(4096, 4096, dtype=T.dtypes.bf16)
        ms = [16, 64, 100]
        pre = {m: _case(m) for m in ms}
        torch.cuda.synchronize()

        for rep in range(cfg.reps):
            bg.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(bg):
                for _ in range(40):  # keep the CUs busy across the whole window
                    a @ bmat
            work = []
            for M in ms:
                g, b, h, ref = pre[M]
                got = _alloc_poisoned(ref, M, TOPK0, rep)
                T._call_fused(g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None)
                work.append((M, ref, got))
            torch.cuda.synchronize()
            torch.cuda.current_stream().wait_stream(bg)
            for M, ref, got in work:
                e = _errs(ref, got, M, TOPK0, U0)
                if _bad(e):
                    fails.append(f"rep {rep} M={M} under load: {_fmt(e)}")
    finally:
        _unpin()
    # Reaching here at all is half the result: a broken co-residency assumption
    # hangs, and the watchdog reports that instead of a failure list.
    return fails, f"{cfg.reps} reps under concurrent 4096^3 GEMM load"


@suite("multi_device", risky=True, timeout=420)
def s_multi_device(cfg):
    """Run on a second device after the first.

    The workspace accessor caches per device, so a second device must get its
    own buffer: sharing the first device's pointer would fault on dereference
    rather than return wrong numbers. Both launch configs, because they consume
    the workspace from a different number of launches, and M spans the split
    crossover so the natural dispatch is covered too.
    """
    n = torch.cuda.device_count()
    if n < 2:
        return [], f"skipped: needs 2 devices, found {n}"
    fails = []
    try:
        for mode in ("fused", "split", "auto"):
            if mode == "auto":
                _unpin()
            else:
                _fused(mode == "fused")
            for dev in range(min(n, 4)):
                torch.cuda.set_device(dev)
                torch.set_default_device(f"cuda:{dev}")
                try:
                    for M in (1, 64, 103, 104, 128):
                        g, b, h, ref = _case(M)
                        got = _alloc_poisoned(ref, M, TOPK0, M)
                        T._call_fused(
                            g, b, h, got, E0, TOPK0, U0, True, 1.0, None, None
                        )
                        torch.cuda.synchronize()
                        e = _errs(ref, got, M, TOPK0, U0)
                        if _bad(e):
                            fails.append(f"{mode} device {dev} M={M}: {_fmt(e)}")
                except Exception as exc:  # noqa: BLE001 - the fault IS the result
                    fails.append(
                        f"{mode} device {dev}: raised {type(exc).__name__}: {exc}"
                    )
            torch.cuda.set_device(0)
            torch.set_default_device("cuda:0")
    finally:
        _unpin()
    # A cross-device pointer dereference aborts the process rather than
    # raising, so a regression here surfaces as a CRASH from the watchdog
    # rather than as a failure list.
    return fails, (
        f"sequential use of {min(n, 4)} devices in one process, "
        f"fused/split/auto x 5 M"
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _run_one(name, cfg):
    fn = SUITES[name]
    t0 = time.time()
    fails, note = fn(cfg)
    return fails, note, time.time() - t0


def _child(name, cfg):
    fails, note, dt = _run_one(name, cfg)
    for f in fails:
        print(f"  FAIL {f}")
    print(f"__RESULT__ {len(fails)} {dt:.1f} {note}")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-k", "--filter", default="", help="substring match on suite name")
    p.add_argument("-l", "--list", action="store_true")
    p.add_argument("--iters", type=int, default=256)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--replays", type=int, default=64)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument(
        "--soak", type=float, default=0, help="loop the whole selection for N seconds"
    )
    p.add_argument(
        "--no-isolate",
        action="store_true",
        help="run RISKY suites inline (a hang wedges the run)",
    )
    p.add_argument("--only", help=argparse.SUPPRESS)  # child-process entry
    cfg = p.parse_args()

    if cfg.list:
        for n, fn in SUITES.items():
            print(f"  {'RISKY ' if fn.risky else '      '}{n}")
        return 0

    if cfg.only:
        return _child(cfg.only, cfg)

    if not T.SUPPORTED:
        print(f"SKIP: {T._skip_msg()}")
        return 0

    names = [n for n in SUITES if cfg.filter in n]
    if not names:
        print(f"no suite matches {cfg.filter!r}")
        return 1

    deadline = time.time() + cfg.soak if cfg.soak else None
    total_fail, rounds = 0, 0
    while True:
        rounds += 1
        for name in names:
            fn = SUITES[name]
            isolate = fn.risky and not cfg.no_isolate
            print(f"[{name}]{' (isolated)' if isolate else ''}", flush=True)
            if isolate:
                cmd = [
                    sys.executable,
                    __file__,
                    "--only",
                    name,
                    "--iters",
                    str(cfg.iters),
                    "--reps",
                    str(cfg.reps),
                    "--replays",
                    str(cfg.replays),
                    "--streams",
                    str(cfg.streams),
                ]
                try:
                    # check=False: a crashing child is a result to report, not
                    # an exception to raise.
                    r = subprocess.run(
                        cmd,
                        timeout=fn.timeout,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    print(f"  HANG after {fn.timeout}s -- no forward progress")
                    total_fail += 1
                    continue
                out = r.stdout + r.stderr
                res = [ln for ln in out.splitlines() if ln.startswith("__RESULT__")]
                for ln in out.splitlines():
                    if ln.startswith("  FAIL"):
                        print(ln)
                if not res:
                    print(f"  CRASH rc={r.returncode}")
                    print("  " + "\n  ".join(out.strip().splitlines()[-8:]))
                    total_fail += 1
                    continue
                nf, dt, note = res[0].split(None, 3)[1:]
                nf = int(nf)
                total_fail += nf
                print(f"  {'FAIL' if nf else 'ok'} ({dt}s) {note}")
            else:
                fails, note, dt = _run_one(name, cfg)
                for f in fails:
                    print(f"  FAIL {f}")
                total_fail += len(fails)
                print(f"  {'FAIL' if fails else 'ok'} ({dt:.1f}s) {note}")

        if deadline is None or time.time() >= deadline:
            break
        print(
            f"--- soak round {rounds} done, " f"{deadline - time.time():.0f}s left ---",
            flush=True,
        )

    print(
        f"\n{'FAILED' if total_fail else 'PASSED'}: "
        f"{total_fail} failure(s) over {rounds} round(s), {len(names)} suite(s)"
    )
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
