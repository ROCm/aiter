# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-process TP harness (peer access, no torchrun) for bench_mega_moe_TP.py
--single-process and tune_mega_moe_tp.py: the arena group, flydsl multi-device
patch and one-shot P2P AllGather / ReduceScatter / AllReduce."""

from __future__ import annotations

import functools
import pickle
import threading

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.mega_moe_tp.common import (
    AUX_SYS,
    MAX_TP,
    bf16x8_to_f32,
    bld,
    bst,
    g_ld_sys,
    g_st_sys,
    i32,
    pack_bf16x8,
    rsrc,
    traced,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

__all__ = ["P2PGroup", "PeerArenaGroup", "flydsl_multi_device"]

NT = 256
NCTA = 128
NCTA_MAX = 512
KIND_AG, KIND_RS, KIND_AR = 0, 1, 2
POLL_LIMIT = 1 << 24


class PeerArenaGroup:
    def __init__(self, devices):
        from aiter.ops.flydsl.kernels.symmetric_arena import _hip

        self.devices = [torch.device(d) for d in devices]
        self.world_size = len(self.devices)
        self._base: dict[int, int] = {}
        self._barrier = threading.Barrier(self.world_size)
        hip = _hip()
        prev = torch.cuda.current_device()
        for d in self.devices:
            torch.cuda.set_device(d)
            for peer in self.devices:
                if peer != d:
                    err = hip.hipDeviceEnablePeerAccess(peer.index, 0)
                    if err not in (0, 704):
                        raise RuntimeError(
                            f"hipDeviceEnablePeerAccess({d} -> {peer}) = {err}"
                        )
                    if err:
                        hip.hipGetLastError()
        torch.cuda.set_device(prev)

    def rank_of(self, device: torch.device) -> int:
        return self.devices.index(torch.device(device))

    def barrier(self, timeout: float = 60.0) -> None:
        try:
            self._barrier.wait(timeout)
        except threading.BrokenBarrierError:
            self._barrier.reset()
            raise TimeoutError(
                f"PeerArenaGroup.barrier: ranks did not meet within {timeout:.0f} s"
            ) from None

    def register(self, rank: int, base_ptr: int) -> None:
        self._base[rank] = base_ptr

    def base_ptrs(self) -> tuple[int, ...]:
        if len(self._base) != self.world_size:
            raise RuntimeError(
                f"{len(self._base)} of {self.world_size} ranks committed"
            )
        return tuple(self._base[r] for r in range(self.world_size))


def flydsl_multi_device() -> None:
    from flydsl.compiler import jit_executor, jit_function

    art_cls = jit_executor.CompiledArtifact
    if getattr(art_cls, "_mega_moe_multi_device", False):
        return
    lock = threading.RLock()
    jf_call = jit_function.JitFunction.__call__

    def _locked_call(self, *a, **k):
        with lock:
            return jf_call(self, *a, **k)

    jit_function.JitFunction.__call__ = _locked_call
    orig = art_cls._get_func_exe

    class _PerDevice:
        def __init__(self, art):
            self.art = art
            self.home = torch.cuda.current_device()
            self.fns, self.keep = {}, []

        def __call__(self, packed):
            dev = torch.cuda.current_device()
            fn = self.fns.get(dev)
            if fn is None:
                with lock:
                    fn = self.fns.get(dev) or self._build(dev)
            return fn(packed)

        def _build(self, dev):
            if dev == self.home:
                fn = orig(self.art)
            else:
                twin = pickle.loads(pickle.dumps(self.art))
                twin._post_load_processors = list(self.art._post_load_processors)
                fn = orig(twin)
                self.keep.append(twin)
            self.fns[dev] = fn
            return fn

    def _get_func_exe(self):
        disp = getattr(self, "_per_device_exe", None)
        if disp is None:
            disp = self._per_device_exe = _PerDevice(self)
        return disp

    art_cls._get_func_exe = _get_func_exe
    art_cls._mega_moe_multi_device = True


@functools.cache
def compile_p2p(kind: int, row_bytes: int, tp: int, device: int):
    UB = 16 if row_bytes % 16 == 0 else (8 if row_bytes % 8 == 0 else 4)
    UPR = row_bytes // UB
    TP = int(tp)
    NCTA_K, NT_K, UNR, push_aux = NCTA, NT, 4, AUX_SYS
    name = f"p2p_{['ag', 'rs', 'ar'][kind]}_r{row_bytes}_tp{TP}_a{push_aux}_c{NCTA_K}_t{NT_K}_u{UNR}"
    assert kind == KIND_AG or UB == 16
    const_expr = fx.const_expr

    def vty():
        return {16: T.vec(4, T.i32), 8: T.vec(2, T.i32), 4: T.i32}[UB]

    def i64(v):
        return fx.Int64(v)

    def tab(t, p):
        return i64(bld(rsrc(t), i32(p) * i32(8), 0, T.i64))

    BIG = 0x7FFFFFF0

    @traced
    def raise_flags(a, tid, phase, c, ep):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid < i32(TP):
            idx = ((i32(phase) * i32(MAX_TP) + a["rank"]) * i32(NCTA_K) + c) * i32(4)
            g_st_sys(tab(a["flags"], tid) + i64(idx), ep)

    @traced
    def wait_flags(a, tid, phase, c, ep):
        if tid < i32(TP):
            idx = ((i32(phase) * i32(MAX_TP) + tid) * i32(NCTA_K) + c) * i32(4)
            addr = tab(a["flags"], a["rank"]) + i64(idx)
            v = g_ld_sys(addr)
            n = i32(0)
            while (v < ep) & (n < i32(POLL_LIMIT)):
                rocdl.s_sleep(1)
                v = g_ld_sys(addr)
                n = n + i32(1)
        gpu.barrier()

    def stage_row0(a, p, bank, src_rank):
        return tab(a["stage"], p) + i64(
            bank * a["stage_sz"] + src_rank * a["mmax"] * i32(row_bytes)
        )

    def offs(tid, it, r0, nu):
        out = []
        for k in range_constexpr(UNR):
            u = it + i32(k * NT_K)
            out.append((u < nu).select((r0 * i32(UPR) + u) * i32(UB), i32(BIG)))
        return out

    @traced
    def send(a, tid, c, r0, nu, bank):
        rank, m = a["rank"], a["m"]
        nbytes = m * i32(row_bytes)
        dsts = [rsrc(stage_row0(a, i32(p), bank, rank), nbytes) for p in range(TP)]
        if const_expr(kind == KIND_AG):
            rs_in = rsrc(a["src"], nbytes)
            rs_out = rsrc(i64(a["out"]) + i64(rank * nbytes), nbytes)
            for it_ in range(tid, nu, i32(NT_K * UNR)):
                o = offs(tid, i32(it_), r0, nu)
                vs = [bld(rs_in, o[k], 0, vty()) for k in range(UNR)]
                for k in range_constexpr(UNR):
                    bst(vs[k], rs_out, o[k], 0)
                    for p in range_constexpr(TP):
                        if rank != i32(p):
                            bst(vs[k], dsts[p], o[k], 0, push_aux)
        else:
            rs_ins = [
                rsrc(i64(a["src"]) + i64(i32(p) * nbytes), nbytes) for p in range(TP)
            ]
            for it_ in range(tid, nu, i32(NT_K * UNR)):
                o = offs(tid, i32(it_), r0, nu)
                for p in range_constexpr(TP):
                    if rank != i32(p):
                        vs = [bld(rs_ins[p], o[k], 0, vty()) for k in range(UNR)]
                        for k in range_constexpr(UNR):
                            bst(vs[k], dsts[p], o[k], 0, push_aux)

    @traced
    def pull(a, tid, src0, dst0, r0, nu):
        nbytes = a["m"] * i32(row_bytes)
        rs_st = rsrc(src0, nbytes)
        rs_out = rsrc(dst0, nbytes)
        for it_ in range(tid, nu, i32(NT_K * UNR)):
            o = offs(tid, i32(it_), r0, nu)
            vs = [bld(rs_st, o[k], 0, vty(), AUX_SYS) for k in range(UNR)]
            for k in range_constexpr(UNR):
                bst(vs[k], rs_out, o[k], 0)

    @traced
    def reduce(a, tid, r0, nu, bank):
        rank, m = a["rank"], a["m"]
        nbytes = m * i32(row_bytes)
        own0 = i64(a["src"]) + i64(rank * nbytes)
        srcs = [
            rsrc(
                (rank == i32(p)).select(own0, stage_row0(a, rank, bank, i32(p))), nbytes
            )
            for p in range(TP)
        ]
        rs_out = rsrc(
            i64(a["out"]) + i64((rank * nbytes) if kind == KIND_AR else i32(0)), nbytes
        )
        dsts = [
            rsrc(stage_row0(a, i32(p), bank + i32(2), rank), nbytes) for p in range(TP)
        ]
        for it_ in range(tid, nu, i32(NT_K * UNR)):
            o = offs(tid, i32(it_), r0, nu)
            vs = [
                [bld(srcs[p], o[k], 0, vty(), AUX_SYS) for p in range(TP)]
                for k in range(UNR)
            ]
            for k in range_constexpr(UNR):
                acc = bf16x8_to_f32(vs[k][0])
                for p in range_constexpr(1, TP):
                    acc = [x + y for x, y in zip(acc, bf16x8_to_f32(vs[k][p]))]
                res = pack_bf16x8(acc)
                bst(res, rs_out, o[k], 0)
                if const_expr(kind == KIND_AR):
                    for p in range_constexpr(TP):
                        if rank != i32(p):
                            bst(res, dsts[p], o[k], 0, push_aux)

    @traced
    def body(a, tid, c, ep):
        rank, m = a["rank"], a["m"]
        nblk = i32(gpu.grid_dim.x)
        rpc = ceildiv(m, nblk)
        r0 = c * rpc
        nr = fx.max(fx.min(rpc, m - r0), i32(0))
        nu = nr * i32(UPR)
        bank = ep & i32(1)
        send(a, tid, c, r0, nu, bank)
        raise_flags(a, tid, 0, c, ep)
        wait_flags(a, tid, 0, c, ep)
        nbytes = m * i32(row_bytes)
        if const_expr(kind == KIND_AG):
            for p in range_constexpr(TP):
                if rank != i32(p):
                    pull(
                        a,
                        tid,
                        stage_row0(a, rank, bank, i32(p)),
                        i64(a["out"]) + i64(i32(p) * nbytes),
                        r0,
                        nu,
                    )
        else:
            reduce(a, tid, r0, nu, bank)
            if const_expr(kind == KIND_AR):
                raise_flags(a, tid, 1, c, ep)
                wait_flags(a, tid, 1, c, ep)
                for p in range_constexpr(TP):
                    if rank != i32(p):
                        pull(
                            a,
                            tid,
                            stage_row0(a, rank, bank + i32(2), i32(p)),
                            i64(a["out"]) + i64(i32(p) * nbytes),
                            r0,
                            nu,
                        )

    @traced
    def epoch_store(a, tid, c, ep):
        if tid == i32(0):
            g_st_sys(i64(a["eflag"]) + i64(c * i32(4)), ep)

    @flyc.kernel(name=name, known_block_size=[NT_K, 1, 1])
    def p2p_kernel(
        src: fx.Int64,
        out: fx.Int64,
        stage: fx.Int64,
        flags: fx.Int64,
        eflag: fx.Int64,
        rank: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
    ):
        a = {
            "src": src,
            "out": out,
            "stage": stage,
            "flags": flags,
            "eflag": eflag,
            "rank": rank,
            "m": m,
            "mmax": mmax,
            "stage_sz": mmax * i32(MAX_TP * row_bytes),
        }
        tid = i32(gpu.thread_id("x"))
        c = i32(gpu.block_id("x"))
        ep = g_ld_sys(i64(eflag) + i64(c * i32(4))) + i32(1)
        body(a, tid, c, ep)
        gpu.barrier()
        epoch_store(a, tid, c, ep)

    @flyc.jit
    def launch(
        src: fx.Int64,
        out: fx.Int64,
        stage: fx.Int64,
        flags: fx.Int64,
        eflag: fx.Int64,
        rank: fx.Int32,
        m: fx.Int32,
        mmax: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        p2p_kernel(src, out, stage, flags, eflag, rank, m, mmax).launch(
            grid=(fx.Int64(grid), 1, 1), block=(NT_K, 1, 1), stream=stream
        )

    return launch


MAX_SITES = 16
SITE_FLAG_INTS = 2 * MAX_TP * NCTA_MAX + NCTA_MAX


class _Site:
    def __init__(self, group, idx, kind, row_bytes):
        devices, tp, mmax = group.devices, group.tp, group.mmax
        nstage = 4 if kind == KIND_AR else 2
        self.stage = [
            torch.empty(nstage * MAX_TP * mmax * row_bytes, dtype=torch.uint8, device=d)
            for d in devices
        ]
        base = idx * SITE_FLAG_INTS
        self.flags = [pool[base : base + 2 * MAX_TP * NCTA_MAX] for pool in group.pool]
        self.eflag = [
            pool[base + 2 * MAX_TP * NCTA_MAX : base + SITE_FLAG_INTS]
            for pool in group.pool
        ]
        sp = [t.data_ptr() for t in self.stage] + [0] * (MAX_TP - tp)
        fp = [t.data_ptr() for t in self.flags] + [0] * (MAX_TP - tp)
        self._host = [torch.tensor(v, dtype=torch.int64).pin_memory() for v in (sp, fp)]
        self.stage_tab = [self._host[0].to(d, non_blocking=True) for d in devices]
        self.flag_tab = [self._host[1].to(d, non_blocking=True) for d in devices]


class P2PGroup:
    def __init__(self, devices, max_local_tokens: int):
        self.devices = [torch.device(d) for d in devices]
        self.tp = len(self.devices)
        self.mmax = int(max_local_tokens)
        self._sites: dict = {}
        self._lock = threading.Lock()
        self.pool = [
            torch.zeros(MAX_SITES * SITE_FLAG_INTS, dtype=torch.int32, device=d)
            for d in self.devices
        ]
        for d in self.devices:
            torch.cuda.synchronize(d)

    def site(self, key, kind, row_bytes):
        with self._lock:
            s = self._sites.get(key)
            if s is None:
                if len(self._sites) >= MAX_SITES:
                    raise RuntimeError(f"more than {MAX_SITES} P2P call sites")
                s = _Site(self, len(self._sites), kind, row_bytes)
                self._sites[key] = s
            return s

    def comm(self, rank: int) -> P2PComm:
        return P2PComm(self, rank)


class P2PComm:
    def __init__(self, group: P2PGroup, rank: int):
        self.group, self.rank = group, rank
        self.tp_size = group.tp
        self.device = group.devices[rank]

    def _launch(self, kind, x, out, rows_per_rank):
        row_bytes = x[0].numel() * x.element_size() if x.dim() > 1 else x.element_size()
        key = (kind, row_bytes, x.dtype, tuple(out.shape[1:]))
        s = self.group.site(key, kind, row_bytes)
        r = self.rank
        exe = compile_p2p(kind, row_bytes, self.tp_size, self.device.index)
        _run_compiled(
            exe,
            x.data_ptr(),
            out.data_ptr(),
            s.stage_tab[r].data_ptr(),
            s.flag_tab[r].data_ptr(),
            s.eflag[r].data_ptr(),
            r,
            rows_per_rank,
            self.group.mmax,
            max(1, min(NCTA, rows_per_rank)),
            torch.cuda.current_stream(self.device),
        )
        return out

    def all_gather(self, x: torch.Tensor, out: torch.Tensor):
        x = x.contiguous()
        return self._launch(KIND_AG, x, out, x.shape[0])

    def reduce_scatter(self, x: torch.Tensor, out: torch.Tensor):
        x = x.contiguous()
        return self._launch(KIND_RS, x, out, x.shape[0] // self.tp_size)

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor):
        x = x.contiguous()
        return self._launch(KIND_AR, x, out, x.shape[0] // self.tp_size)
