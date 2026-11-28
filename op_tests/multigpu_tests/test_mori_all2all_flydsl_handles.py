# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL handles from ``MoriAll2AllManager`` on one node (2..8 GPUs).

Each rank builds the EP group with ``aiter.init_dist_env`` and asks the
``"mori"`` all-to-all manager for handles, as ATOM does.

* ``scales``: a bf16-wire handle configured with scales that dispatches
  without them -- ATOM's handle for an FP8 block-scale model whose rows travel
  in bf16 (e.g. DeepSeek-R1): ``quant_dtype=bf16``, ``scale_dim=hidden//128``,
  ``scale_type_size=4``, ``dispatch(a1, w, None, ids)``. ``get_handle`` serves
  it with FlyDSL; dispatch returns no scales, the received rows, weights and
  ids equal the source tokens' (and a mori handle's for the same call), and
  combine equals an f32 sum over the distinct destination ranks (and mori's).
  The same no-scales dispatch + combine captured in a CUDA graph and replayed,
  with every output overwritten before each replay, gives the same received
  rows / weights / ids and combine result.
* ``lifetime``: a handle the caller dropped is returned again (same op, no new
  IPC allocation); another ``index`` gets its own op; ``destroy()`` unmaps
  every peer mapping and frees every allocation; a handle requested after
  ``destroy()`` works; ``aiter.destroy_dist_env()`` with live FlyDSL handles
  returns cleanly and releases their IPC memory (every mapping closed, every
  allocation freed).
* ``agreement``: when opening a peer's IPC handle fails on one rank, or one
  rank runs out of memory while allocating the op's buffers, building the op
  raises on every rank (no rank hangs in the next collective), and every
  allocation and peer mapping made so far is released.
* ``geometry``: launch geometry on the default path. A handle with tuned rows
  (MXFP8 wire at EP8 with hidden 7168, top-8, 32 local experts) has every
  tuned geometry's dispatch and combine kernel loaded when it is built, so
  dispatch + combine at 1 to 4096 tokens (every row's bucket, 1025..2048
  included) loads nothing new, and the results match the reference. With the
  CU count capped at 80 (MI308X) and 38 (a CPX partition), new handles use
  only geometries within it and give the same results. The gate does not
  depend on the current device (building a handle is collective), and it
  declines scale rows that are not whole 4-byte words (gpt-oss MXFP4 rows:
  90 B of e8m0 scales).

    python op_tests/multigpu_tests/test_mori_all2all_flydsl_handles.py [--world 8]
"""

import argparse
import gc
import os
import queue
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_flydsl_ep_intranode import _bitdiff, _reference

CASES = ("scales", "lifetime", "agreement", "geometry")


class _IpcCounter:
    """Counts HIP IPC allocations / peer mappings and can fail one mapping."""

    def __init__(self):
        from aiter.ops.flydsl.ipc_heap import IpcHeap

        self.cls = IpcHeap
        self.orig = {
            n: getattr(IpcHeap, n)
            for n in (
                "alloc",
                "free_device_mem",
                "open_mem_handle",
                "close_mem_handle",
            )
        }
        self.live_alloc, self.live_open = set(), set()
        self.n_alloc = 0
        self.fail_open_at = None  # 1-based count of open_mem_handle calls
        self.fail_alloc_at = None  # 1-based count of alloc calls
        self.n_open = 0

        def alloc(size, flags=None, fill=0):
            if (
                self.fail_alloc_at is not None
                and self.n_alloc + 1 == self.fail_alloc_at
            ):
                raise RuntimeError("injected hipErrorOutOfMemory")
            ptr = self.orig["alloc"](size, flags, fill)
            self.live_alloc.add(ptr)
            self.n_alloc += 1
            return ptr

        def free(ptr):
            self.live_alloc.discard(ptr)
            return self.orig["free_device_mem"](ptr)

        def open_(handle):
            self.n_open += 1
            if self.fail_open_at is not None and self.n_open == self.fail_open_at:
                raise RuntimeError("injected hipIpcOpenMemHandle failure")
            ptr = self.orig["open_mem_handle"](handle)
            self.live_open.add(ptr)
            return ptr

        def close(ptr):
            self.live_open.discard(ptr)
            return self.orig["close_mem_handle"](ptr)

        for name, fn in (
            ("alloc", alloc),
            ("free_device_mem", free),
            ("open_mem_handle", open_),
            ("close_mem_handle", close),
        ):
            setattr(IpcHeap, name, staticmethod(fn))

    def leaked(self, alloc_base, open_base):
        """(allocations, mappings) made since the bases and still live. An
        allocation is freed once its heap is closed and no tensor on it is
        left, so collect first."""
        gc.collect()
        torch.cuda.synchronize()
        return self.live_alloc - alloc_base, self.live_open - open_base


def _ids(T, K, E, g):
    return torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(
        torch.int32
    )


def _round_trip(handle, x, w, ids, epr, rank, launch):
    """dispatch(x, w, None, ids) + combine of (received row * rank factor).

    Returns (received rows / weights / ids sorted by source, count, scales,
    combine result). ``w[:, 0]`` encodes the source (rank, token)."""
    out = handle.dispatch(x, w, None, ids, *launch)
    torch.cuda.synchronize()
    n = int(out[4].reshape(-1)[0].item())
    order = torch.argsort(out[1][:n, 0])
    recv = tuple(t[:n][order].cpu() for t in (out[0], out[1], out[3]))
    cin = (out[0][: max(n, 1)].float() * (1.0 + rank / 8.0)).to(torch.bfloat16)
    res = handle.combine(cin, None, ids, *launch)[0][: x.shape[0]].clone()
    torch.cuda.synchronize()
    return recv, n, out[2], res


def _poison(*tensors):
    for t in tensors:
        t.view(torch.uint8).fill_(0xFF)


def _graph_round_trip(handle, x, w, ids, rank, launch, group, replays=3):
    """``_round_trip``'s dispatch(x, w, None, ids) + combine captured in one
    CUDA graph and replayed. Every output is overwritten on every rank before
    each replay; returns, per replay, ``_round_trip``'s received tuple and
    count and the combine result."""
    dist.barrier(group=group)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            out = handle.dispatch(x, w, None, ids, *launch)
            n_g = out[4].clone()  # combine resets the count
            cin = (out[0].float() * (1.0 + rank / 8.0)).to(torch.bfloat16)
            res = handle.combine(cin, None, ids, *launch)[0]
    torch.cuda.synchronize()
    results = []
    for _ in range(replays):
        _poison(out[0], out[1], out[3], n_g, res)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        n = int(n_g.reshape(-1)[0].item())
        order = torch.argsort(out[1][:n, 0])
        recv = tuple(t[:n][order].cpu() for t in (out[0], out[1], out[3]))
        results.append((recv, n, res[: x.shape[0]].clone()))
    del graph
    dist.barrier(group=group)
    return results


def _case_scales(manager, rank, world, dev, failures, has_mori):
    from aiter.dist.device_communicators import all2all

    hidden, epr, K, max_tok = 7168, 32, 8, 256
    E = epr * world
    kwargs = {
        "rank": rank,
        "num_ep_ranks": world,
        "input_dtype": torch.bfloat16,
        "quant_dtype": torch.bfloat16,
        "token_hidden_size": hidden,
        "scale_dim": hidden // 128,
        "scale_type_size": 4,
        "max_num_tokens_per_dp_rank": max_tok,
        "num_local_experts": epr,
        "num_experts_per_token": K,
        "gpu_per_node": world,
        "low_latency": False,
    }
    fly = manager.get_handle(kwargs)
    if not isinstance(fly, all2all._FlyDSLMoriCallSurface):
        failures.append(f"rank{rank} scales: handle is {type(fly).__name__}")
        return
    mori_h = manager.get_handle(kwargs, backend="mori") if has_mori else None
    # mori's launch keywords (block_num, rdma_block_num, warp_per_block), as
    # ATOM passes them; the FlyDSL handle accepts and ignores them.
    launch = (64, -1, 16)
    g = torch.Generator().manual_seed(77 + rank)
    for T in (1, 37, max_tok):
        tag = f"rank{rank} scales T={T}"
        ids = _ids(T, K, E, g).to(dev)
        w = torch.rand(T, K, generator=g)
        w[:, 0] = rank * max_tok + torch.arange(T, dtype=torch.float32)
        w = w.to(dev)
        x = (torch.randn(T, hidden, generator=g) * 4).to(torch.bfloat16).to(dev)
        all_rows = [None] * world
        dist.all_gather_object(
            all_rows, (ids.cpu(), x.cpu(), w.cpu()), group=manager.cpu_group
        )
        try:
            got, n, scales, res = _round_trip(fly, x, w, ids, epr, rank, launch)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{tag}: FlyDSL dispatch/combine raised {e!r}")
            dist.barrier(group=manager.cpu_group)
            continue
        if scales is not None:
            failures.append(f"{tag}: dispatch returned scales for a call without")
        src = [
            (r_i, t)
            for r_i, (r_ids, _, _) in enumerate(all_rows)
            for t in range(r_ids.shape[0])
            if bool((r_ids[t].long() // epr == rank).any())
        ]
        src.sort(key=lambda s: s[0] * max_tok + s[1])
        want = tuple(
            torch.stack([all_rows[r_i][j][t] for r_i, t in src]) if src else got[i][:0]
            for i, j in ((0, 1), (1, 2), (2, 0))
        )
        if n != len(src):
            failures.append(f"{tag}: received {n}, expected {len(src)}")
        else:
            for name, a, b in zip(("rows", "weights", "ids"), got, want):
                if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                    failures.append(f"{tag}: received {name} differ")
        ref = _reference(x.float(), ids, epr)
        bad = _bitdiff(res, ref)
        if bad:
            failures.append(f"{tag}: combine {bad} elements differ from reference")
        replays = _graph_round_trip(fly, x, w, ids, rank, launch, manager.cpu_group)
        for i, (got_g, n_g, res_g) in enumerate(replays):
            if n_g != n or any(
                not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(got_g, got)
            ):
                failures.append(f"{tag}: graph replay {i} dispatch differs")
            bad = _bitdiff(res_g, ref)
            if bad:
                failures.append(f"{tag}: graph replay {i} {bad} elements differ")
        if mori_h is not None:
            m_got, m_n, m_scales, m_res = _round_trip(
                mori_h, x, w, ids, epr, rank, launch
            )
            if m_scales is not None:
                failures.append(f"{tag}: mori returned scales")
            if m_n != n or any(
                not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(got, m_got)
            ):
                failures.append(f"{tag}: FlyDSL dispatch differs from mori")
            if not torch.allclose(res.float(), m_res.float(), atol=2e-2, rtol=2e-2):
                failures.append(f"{tag}: FlyDSL combine differs from mori")
        dist.barrier(group=manager.cpu_group)


def _small_kwargs(rank, world, hidden=2048, max_tok=64):
    return {
        "rank": rank,
        "num_ep_ranks": world,
        "input_dtype": torch.bfloat16,
        "quant_dtype": torch.bfloat16,
        "token_hidden_size": hidden,
        "scale_dim": 0,
        "scale_type_size": 0,
        "max_num_tokens_per_dp_rank": max_tok,
        "num_local_experts": 4,
        "num_experts_per_token": 4,
        "gpu_per_node": world,
        "low_latency": False,
    }


def _check_small(handle, rank, world, dev, tag, failures):
    g = torch.Generator().manual_seed(5 + rank)
    T, K, epr = 16, 4, 4
    ids = _ids(T, K, epr * world, g).to(dev)
    w = torch.rand(T, K, generator=g)
    w[:, 0] = rank * 64 + torch.arange(T, dtype=torch.float32)
    w = w.to(dev)
    x = torch.randn(T, 2048, generator=g).to(torch.bfloat16).to(dev)
    _, _, _, res = _round_trip(handle, x, w, ids, epr, rank, ())
    bad = _bitdiff(res, _reference(x.float(), ids, epr))
    if bad:
        failures.append(f"{tag}: combine {bad} elements differ")


def _case_lifetime(manager, rank, world, dev, failures, counter):
    tag = f"rank{rank} lifetime"
    kw = _small_kwargs(rank, world)
    h = manager.get_handle(kw)
    op = h._op
    n_alloc = counter.n_alloc
    _check_small(h, rank, world, dev, tag, failures)
    del h
    gc.collect()
    h = manager.get_handle(kw)
    if h._op is not op:
        failures.append(f"{tag}: a dropped handle was rebuilt")
    if counter.n_alloc != n_alloc:
        failures.append(f"{tag}: re-requesting a handle allocated IPC memory")
    h1 = manager.get_handle(kw, index=1)
    if h1._op is op:
        failures.append(f"{tag}: index=1 shares index 0's op")
    _check_small(h1, rank, world, dev, tag + " index=1", failures)
    del h, h1, op
    dist.barrier(group=manager.cpu_group)
    manager.destroy()  # also frees the earlier cases' FlyDSL ops
    left_a, left_o = counter.leaked(counter.base_alloc, counter.base_open)
    if left_o:
        failures.append(f"{tag}: {len(left_o)} peer mappings left open")
    if left_a:
        failures.append(f"{tag}: {len(left_a)} allocations not freed")
    h = manager.get_handle(kw)
    _check_small(h, rank, world, dev, tag + " after destroy", failures)
    # left live on purpose: destroy_dist_env() must cope with it


def _case_agreement(manager, rank, world, dev, failures, counter):
    kw = _small_kwargs(rank, world, hidden=1024, max_tok=32)
    # the last rank fails in the middle of the op's buffers: opening a peer's
    # IPC handle, or allocating (out of memory); every rank must raise
    for how, index in (("open", 7), ("alloc", 8)):
        tag = f"rank{rank} agreement ({how})"
        before_alloc, before_open = set(counter.live_alloc), set(counter.live_open)
        if rank == world - 1:
            if how == "open":
                counter.fail_open_at = counter.n_open + 3 * (world - 1) + 1
            else:
                counter.fail_alloc_at = counter.n_alloc + 4
        raised = None
        try:
            manager.get_handle(kw, index=index)
        except RuntimeError as e:
            raised = e
        finally:
            counter.fail_open_at = counter.fail_alloc_at = None
        if raised is None:
            failures.append(f"{tag}: building the op did not raise")
        raised = None  # its traceback holds the half-built op
        leaked_a, leaked_o = counter.leaked(before_alloc, before_open)
        if leaked_a or leaked_o:
            failures.append(
                f"{tag}: {len(leaked_a)} allocations / {len(leaked_o)} mappings "
                "leaked"
            )
        dist.barrier(group=manager.cpu_group)  # nobody hangs in a collective


def _loaded_geometries(op):
    """{"dispatch" / "combine": {(block_num, warp_num_per_block)}} of the
    kernels the op has compiled and loaded."""
    out = {"dispatch": set(), "combine": set()}
    for k in op._loaded:
        phase = "combine" if hasattr(k, "variant") else "dispatch"
        out[phase].add((k.block_num, k.warp_num_per_block))
    return out


def _jit_state(op):
    """Every JIT launcher of the op and how many artifacts it has compiled."""
    return {
        key: len(getattr(fn, "_mem_cache", None) or {})
        for cache in (op._disp_jit_cache, op._comb_jit_cache)
        for key, fn in cache.items()
    }


def _case_geometry(manager, rank, world, dev, failures):
    from aiter.dist.device_communicators import all2all
    from aiter.ops.flydsl.kernels import (
        flydsl_dispatch_combine_intranode_op as op_mod,
    )

    hidden, epr, K, max_tok = 7168, 32, 8, 4096
    E = epr * world
    kwargs = {
        "rank": rank,
        "num_ep_ranks": world,
        "input_dtype": torch.bfloat16,
        "quant_dtype": torch.float8_e4m3fn,
        "token_hidden_size": hidden,
        "scale_dim": hidden // 32,
        "scale_type_size": 1,
        "max_num_tokens_per_dp_rank": max_tok,
        "num_local_experts": epr,
        "num_experts_per_token": K,
        "gpu_per_node": world,
        "low_latency": False,
    }
    # the tuned rows' buckets for this shape at EP8, and their neighbours
    token_counts = (1, 5, 8, 9, 33, 100, 1024, 1025, 2048, 2049, 4096)
    g = torch.Generator().manual_seed(91 + rank)
    orig_cu = op_mod._device_cu_count
    try:
        for cu_cap, index in ((None, 0), (80, 1), (38, 2)):
            tag = f"rank{rank} geometry cu<={cu_cap or 'device'}"
            if cu_cap is not None:
                op_mod._device_cu_count = lambda i, c=cu_cap: min(orig_cu(i), c)
            num_cu = op_mod._device_cu_count(torch.cuda.current_device())
            h = manager.get_handle(kwargs, index=index)
            if not isinstance(h, all2all._FlyDSLMoriCallSurface):
                failures.append(f"{tag}: handle is {type(h).__name__}")
                return
            op = h._op
            table = op.cfg.tuning_table
            if world == 8 and (
                table is None or not table.dispatch or not table.combine
            ):
                failures.append(f"{tag}: no tuned rows for the EP8 shape")
            loaded = _loaded_geometries(op)
            jit = _jit_state(op)
            for phase, default in (
                (
                    "dispatch",
                    (
                        op_mod._DEFAULT_DISPATCH_BLOCK_NUM,
                        op_mod._DEFAULT_DISPATCH_WARP_NUM,
                    ),
                ),
                (
                    "combine",
                    (
                        op_mod._DEFAULT_COMBINE_BLOCK_NUM,
                        op_mod._DEFAULT_COMBINE_WARP_NUM,
                    ),
                ),
            ):
                rows = getattr(table, phase) if table is not None else {}
                want = {
                    (min(bn, num_cu), wpb)
                    for bn, wpb in (rows.values() if rows else [default])
                }
                if loaded[phase] != want:
                    failures.append(
                        f"{tag}: {phase} geometries loaded at build "
                        f"{sorted(loaded[phase])}, want {sorted(want)}"
                    )
            for T in token_counts:
                ids = _ids(T, K, E, g).to(dev)
                w = torch.rand(T, K, generator=g).to(dev)
                xf = (torch.randn(T, hidden, generator=g) * 4).clamp(-448, 448)
                x = xf.to(torch.float8_e4m3fn).to(dev)
                sc = torch.full((T, hidden // 32), 127, dtype=torch.uint8, device=dev)
                out = h.dispatch(x, w, sc, ids, 64, -1, 16)
                rows = max(1, int(out[4].reshape(-1)[0].item()))
                cin = (out[0][:rows].float() * (1.0 + rank / 8.0)).to(torch.bfloat16)
                res = h.combine(cin, None, ids, 64, -1, 16)[0][:T]
                torch.cuda.synchronize()
                bad = _bitdiff(res, _reference(x.float(), ids, epr))
                if bad:
                    failures.append(f"{tag} T={T}: combine {bad} elements differ")
                dist.barrier(group=manager.cpu_group)
            now = _jit_state(op)
            if now != jit or _loaded_geometries(op) != loaded:
                grown = {
                    k: (jit.get(k, 0), n) for k, n in now.items() if n != jit.get(k)
                }
                failures.append(f"{tag}: live calls compiled kernels: {grown}")
    finally:
        op_mod._device_cu_count = orig_cu

    # the gate
    fly_kwargs = all2all._flydsl_op_kwargs(
        **{k: v for k, v in kwargs.items() if k not in ("gpu_per_node", "low_latency")}
    )
    if not all2all._flydsl_intranode_supported(fly_kwargs):
        failures.append(f"rank{rank} geometry: gate declines the supported handle")
    torch.cuda.set_device((rank + 1) % world)
    try:
        # rank-independent: the op runs on the current device
        if not all2all._flydsl_intranode_supported(fly_kwargs):
            failures.append(f"rank{rank} geometry: gate declines rank != device")
    finally:
        torch.cuda.set_device(rank)
    gpt_oss = {
        **fly_kwargs,
        "dispatch_dtype": torch.float4_e2m1fn_x2,
        "hidden_dim": 2880,
        "scale_dim": 2880 // 32,
        "scale_type_size": 1,
        "num_experts_per_rank": 128 // world,
        "num_experts_per_token": 4,
    }
    if all2all._flydsl_intranode_supported(gpt_oss):
        failures.append(f"rank{rank} geometry: gate accepts 90-byte scale rows")
    dist.barrier(group=manager.cpu_group)


def _worker(rank, world, port, cases, result_q):
    import aiter
    from aiter.dist.device_communicators.all2all import has_mori
    from aiter.dist.parallel_state import get_ep_group
    from aiter.dist.utils import get_distributed_init_method

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    failures = []
    counter = _IpcCounter()
    try:
        aiter.init_dist_env(
            tensor_model_parallel_size=1,
            rankID=0,
            backend="cpu:gloo,cuda:nccl",
            distributed_init_method=get_distributed_init_method("127.0.0.1", port),
            data_parallel_size=world,
            data_parallel_rank=rank,
        )
        manager = get_ep_group().device_communicator.get_or_create_all2all_manager(
            "mori"
        )
        # IPC memory of other communicators (e.g. custom all-reduce) stays
        counter.base_alloc = set(counter.live_alloc)
        counter.base_open = set(counter.live_open)
        for case in cases:
            try:
                if case == "scales":
                    _case_scales(manager, rank, world, dev, failures, has_mori())
                elif case == "lifetime":
                    _case_lifetime(manager, rank, world, dev, failures, counter)
                elif case == "agreement":
                    _case_agreement(manager, rank, world, dev, failures, counter)
                elif case == "geometry":
                    _case_geometry(manager, rank, world, dev, failures)
            except Exception as e:  # noqa: BLE001
                failures.append(f"rank{rank} {case}: {type(e).__name__}: {e}")
        torch.cuda.synchronize()
        try:
            aiter.destroy_dist_env()
        except Exception as e:  # noqa: BLE001
            failures.append(f"rank{rank} destroy_dist_env: {type(e).__name__}: {e}")
        else:
            # Everything mapped since init, the FlyDSL heap the lifetime case
            # leaves live included, is released by the teardown.
            left_a, left_o = counter.leaked(counter.base_alloc, counter.base_open)
            if left_o or left_a:
                failures.append(
                    f"rank{rank} destroy_dist_env: {len(left_o)} peer mappings "
                    f"open, {len(left_a)} allocations not freed"
                )
    except Exception as e:  # noqa: BLE001
        failures.append(f"rank{rank}: {type(e).__name__}: {e}")
    result_q.put((rank, failures))


def run(world=None, cases=CASES, timeout=900):
    from aiter.dist.device_communicators.all2all import has_mori
    from aiter.dist.utils import get_open_port
    from aiter.jit.utils.chip_info import get_gfx_runtime

    gfx = get_gfx_runtime()
    if gfx not in ("gfx942", "gfx950"):
        print(f"skip: FlyDSL intra-node EP targets gfx942/gfx950, not {gfx}")
        return True
    if not has_mori():
        print("skip: MoriAll2AllManager needs mori installed")
        return True
    n_dev = torch.cuda.device_count()
    world = min(8, n_dev) if world is None else world
    if world < 2 or world > min(8, n_dev):
        print(f"skip: world {world} needs 2..8 GPUs, have {n_dev}")
        return True
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = get_open_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world, port, tuple(cases), q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = {}
    deadline = time.monotonic() + timeout
    while len(results) < world:
        try:
            r, f = q.get(timeout=5)
            results[r] = f
            continue
        except queue.Empty:
            pass
        dead = [
            i
            for i, p in enumerate(procs)
            if i not in results and p.exitcode not in (None, 0)
        ]
        if dead:  # a crashed rank leaves its peers waiting
            results[-1] = [f"rank(s) {dead} exited without reporting"]
            break
        if time.monotonic() > deadline:
            results[-1] = [f"timeout after {timeout}s"]
            break
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.kill()
    failures = [f for r in sorted(results) for f in results[r]]
    for f in failures[:50]:
        print("FAIL", f)
    print(
        f"MoriAll2AllManager FlyDSL handles: world={world} cases={list(cases)}: "
        f"{'FAILED' if failures else 'passed'}"
    )
    return not failures


def test_mori_all2all_flydsl_handles():
    assert run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--cases", type=str, default=",".join(CASES))
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    ok = run(world=a.world, cases=a.cases.split(","), timeout=a.timeout)
    sys.exit(0 if ok else 1)
