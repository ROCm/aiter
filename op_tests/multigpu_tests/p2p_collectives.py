# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-shot P2P collectives for one process driving every TP GPU.

The single-process mode of ``test_mega_moe_TP.py`` runs the split baseline's
AllGather / ReduceScatter / AllReduce on these kernels (the multi-process
one-shot collectives need one process per rank). Each call is one launch per
rank: the sender pushes its rows into every peer's staging buffer with
system-scope stores, raises one epoch flag per (peer, CTA), and the receiver
reads its staging back with cache-bypassing loads once its flags are up --
the same design as aiter's one-shot kernels.

Staging is per call site and double-buffered on the epoch parity; epochs live
on the device, so CUDA-graph replays stay correct.
"""

from __future__ import annotations

import functools
import threading

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

__all__ = ["P2PGroup"]

MAX_TP = 8
NT = 256
NCTA = 128
NCTA_MAX = 512
AUX_SYS = 1 | 16
KIND_AG, KIND_RS, KIND_AR = 0, 1, 2
POLL_LIMIT = 1 << 24  # a peer that never shows up: give up instead of hanging the GPU
traced = ASTRewriter.transform


def _u(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _attr(v):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), int(v))


@functools.cache
def compile_p2p(kind: int, row_bytes: int, tp: int, device: int, push_aux: int = AUX_SYS, ncta: int = 0, nt: int = 0, unr: int = 4):
    """``kind``: 0 AllGather (any bytes), 1 ReduceScatter / 2 AllReduce (bf16)."""
    UB = 16 if row_bytes % 16 == 0 else (8 if row_bytes % 8 == 0 else 4)
    UPR = row_bytes // UB
    TP = int(tp)
    NCTA_K = ncta or NCTA
    NT_K = nt or NT
    UNR = unr
    name = f"p2p_{['ag', 'rs', 'ar'][kind]}_r{row_bytes}_tp{TP}_a{push_aux}_c{NCTA_K}_t{NT_K}_u{UNR}"
    assert kind == KIND_AG or UB == 16
    const_expr = fx.const_expr

    def vty():
        return {16: T.vec(4, T.i32), 8: T.vec(2, T.i32), 4: T.i32}[UB]

    def i32(v):
        return fx.Int32(v)

    def i64(v):
        return fx.Int64(v)

    def uni(v):
        return i32(rocdl.readfirstlane(T.i32, _u(i32(v))))

    def uni64(v):
        v = i64(v)
        lo = uni(i32(v & i64(0xFFFFFFFF)))
        hi = uni(i32(v >> i64(32)))
        return (i64(hi) << i64(32)) | (i64(lo) & i64(0xFFFFFFFF))

    def rsrc(addr):
        return buffer_ops.create_buffer_resource_from_addr(_u(uni64(addr)))

    def bld(rs, off, ty, aux=0):
        return rocdl.raw_ptr_buffer_load(ty, rs, _u(i32(off)), _u(i32(0)), aux=_attr(aux))

    def bst(v, rs, off, aux=0):
        rocdl.raw_ptr_buffer_store(_u(v), rs, _u(i32(off)), _u(i32(0)), aux=_attr(aux))

    def gptr(addr):
        return _llvm.IntToPtrOp(_llvm.PointerType.get(address_space=1), _u(i64(addr))).result

    def ld_sys(addr):
        return i32(
            _llvm.LoadOp(
                T.i32, gptr(addr), alignment=4, volatile_=True,
                ordering=_llvm.AtomicOrdering.monotonic, syncscope="one-as",
            ).res
        )

    def st_sys(addr, v):
        _llvm.StoreOp(
            _u(i32(v)), gptr(addr), alignment=4,
            ordering=_llvm.AtomicOrdering.monotonic, syncscope="one-as",
        )

    def wait_vm0():
        _llvm.inline_asm(None, [], "s_waitcnt vmcnt(0)", "", has_side_effects=True)

    def tab(t, p):
        return i64(bld(rsrc(t), i32(p) * i32(8), T.i64))

    def bf16x8(d):
        dv = fx.Vector(d)
        out = []
        for q in range_constexpr(4):
            w = i32(dv[q])
            out += [(w << i32(16)).bitcast(fx.Float32), (w & i32(-65536)).bitcast(fx.Float32)]
        return out

    def bits(x):
        return i32(
            fx.Vector.from_elements([fx.Float32(x).to(fx.BFloat16)], fx.BFloat16).bitcast(fx.Int16)[0]
        ) & i32(0xFFFF)

    def pack8(f):
        return fx.Vector.from_elements(
            [bits(f[2 * q]) | (bits(f[2 * q + 1]) << i32(16)) for q in range(4)], fx.Int32
        )

    BIG = 0x7FFFFFF0

    def rsrc_n(addr, nbytes):
        return buffer_ops.create_buffer_resource_from_addr(
            _u(uni64(addr)), num_records_bytes=_u(i64(uni(nbytes)))
        )

    @traced
    def raise_flags(a, tid, phase, c, ep):
        wait_vm0()
        gpu.barrier()
        if tid < i32(TP):
            idx = ((i32(phase) * i32(MAX_TP) + a["rank"]) * i32(NCTA_K) + c) * i32(4)
            st_sys(tab(a["flags"], tid) + i64(idx), ep)

    @traced
    def wait_flags(a, tid, phase, c, ep):
        if tid < i32(TP):
            idx = ((i32(phase) * i32(MAX_TP) + tid) * i32(NCTA_K) + c) * i32(4)
            addr = tab(a["flags"], a["rank"]) + i64(idx)
            v = ld_sys(addr)
            n = i32(0)
            while (v < ep) & (n < i32(POLL_LIMIT)):
                rocdl.s_sleep(1)
                v = ld_sys(addr)
                n = n + i32(1)
        gpu.barrier()

    def stage_row0(a, p, bank, src_rank):
        return tab(a["stage"], p) + i64(bank * a["stage_sz"] + src_rank * a["mmax"] * i32(row_bytes))

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
        dsts = [rsrc_n(stage_row0(a, i32(p), bank, rank), nbytes) for p in range(TP)]
        if const_expr(kind == KIND_AG):
            rs_in = rsrc_n(a["src"], nbytes)
            rs_out = rsrc_n(i64(a["out"]) + i64(rank * nbytes), nbytes)
            for it_ in range(tid, nu, i32(NT_K * UNR)):
                o = offs(tid, i32(it_), r0, nu)
                vs = [bld(rs_in, o[k], vty()) for k in range(UNR)]
                for k in range_constexpr(UNR):
                    bst(vs[k], rs_out, o[k])
                    for p in range_constexpr(TP):
                        if rank != i32(p):
                            bst(vs[k], dsts[p], o[k], push_aux)
        else:
            rs_ins = [rsrc_n(i64(a["src"]) + i64(i32(p) * nbytes), nbytes) for p in range(TP)]
            for it_ in range(tid, nu, i32(NT_K * UNR)):
                o = offs(tid, i32(it_), r0, nu)
                for p in range_constexpr(TP):
                    if rank != i32(p):
                        vs = [bld(rs_ins[p], o[k], vty()) for k in range(UNR)]
                        for k in range_constexpr(UNR):
                            bst(vs[k], dsts[p], o[k], push_aux)

    @traced
    def pull(a, tid, src0, dst0, r0, nu):
        nbytes = a["m"] * i32(row_bytes)
        rs_st = rsrc_n(src0, nbytes)
        rs_out = rsrc_n(dst0, nbytes)
        for it_ in range(tid, nu, i32(NT_K * UNR)):
            o = offs(tid, i32(it_), r0, nu)
            vs = [bld(rs_st, o[k], vty(), AUX_SYS) for k in range(UNR)]
            for k in range_constexpr(UNR):
                bst(vs[k], rs_out, o[k])

    @traced
    def reduce(a, tid, r0, nu, bank):
        rank, m = a["rank"], a["m"]
        nbytes = m * i32(row_bytes)
        own0 = i64(a["src"]) + i64(rank * nbytes)
        srcs = [
            rsrc_n((rank == i32(p)).select(own0, stage_row0(a, rank, bank, i32(p))), nbytes)
            for p in range(TP)
        ]
        rs_out = rsrc_n(i64(a["out"]) + i64((rank * nbytes) if kind == KIND_AR else i32(0)), nbytes)
        dsts = [rsrc_n(stage_row0(a, i32(p), bank + i32(2), rank), nbytes) for p in range(TP)]
        for it_ in range(tid, nu, i32(NT_K * UNR)):
            o = offs(tid, i32(it_), r0, nu)
            vs = [[bld(srcs[p], o[k], vty(), AUX_SYS) for p in range(TP)] for k in range(UNR)]
            for k in range_constexpr(UNR):
                acc = bf16x8(vs[k][0])
                for p in range_constexpr(1, TP):
                    acc = [x + y for x, y in zip(acc, bf16x8(vs[k][p]))]
                res = pack8(acc)
                bst(res, rs_out, o[k])
                if const_expr(kind == KIND_AR):
                    for p in range_constexpr(TP):
                        if rank != i32(p):
                            bst(res, dsts[p], o[k], push_aux)

    @traced
    def body(a, tid, c, ep):
        rank, m = a["rank"], a["m"]
        nblk = i32(gpu.grid_dim.x)
        rpc = (m + nblk - i32(1)) // nblk
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
                    pull(a, tid, stage_row0(a, rank, bank, i32(p)), i64(a["out"]) + i64(i32(p) * nbytes), r0, nu)
        else:
            reduce(a, tid, r0, nu, bank)
            if const_expr(kind == KIND_AR):
                raise_flags(a, tid, 1, c, ep)
                wait_flags(a, tid, 1, c, ep)
                for p in range_constexpr(TP):
                    if rank != i32(p):
                        pull(a, tid, stage_row0(a, rank, bank + i32(2), i32(p)), i64(a["out"]) + i64(i32(p) * nbytes), r0, nu)

    @traced
    def epoch_store(a, tid, c, ep):
        if tid == i32(0):
            st_sys(i64(a["eflag"]) + i64(c * i32(4)), ep)

    @flyc.kernel(name=name, known_block_size=[NT_K, 1, 1])
    def p2p_kernel(
        src: fx.Int64, out: fx.Int64, stage: fx.Int64, flags: fx.Int64, eflag: fx.Int64,
        rank: fx.Int32, m: fx.Int32, mmax: fx.Int32,
    ):
        a = {"src": src, "out": out, "stage": stage, "flags": flags, "eflag": eflag,
             "rank": rank, "m": m, "mmax": mmax, "stage_sz": mmax * i32(MAX_TP * row_bytes)}
        tid = i32(gpu.thread_id("x"))
        c = i32(gpu.block_id("x"))
        ep = ld_sys(i64(eflag) + i64(c * i32(4))) + i32(1)
        body(a, tid, c, ep)
        gpu.barrier()
        epoch_store(a, tid, c, ep)

    @flyc.jit
    def launch(
        src: fx.Int64, out: fx.Int64, stage: fx.Int64, flags: fx.Int64, eflag: fx.Int64,
        rank: fx.Int32, m: fx.Int32, mmax: fx.Int32, grid: fx.Int32, stream: fx.Stream,
    ):
        p2p_kernel(src, out, stage, flags, eflag, rank, m, mmax).launch(
            grid=(fx.Int64(grid), 1, 1), block=(NT_K, 1, 1), stream=stream
        )

    return launch


MAX_SITES = 16
SITE_FLAG_INTS = 2 * MAX_TP * NCTA_MAX + NCTA_MAX  # flags (2 phases) + epochs


class _Site:
    """Staging + flags of one call site, on every rank. Flags and epochs come
    from the group's pool (zeroed once, up front); staging needs no
    initialization -- its bytes are always written before their flag -- so
    a site can be created lazily without launching anything on a peer that
    may be mid-collective."""

    def __init__(self, group, idx, kind, row_bytes):
        devices, tp, mmax = group.devices, group.tp, group.mmax
        nstage = 4 if kind == KIND_AR else 2
        self.kind, self.row_bytes, self.mmax = kind, row_bytes, mmax
        self.stage = [torch.empty(nstage * MAX_TP * mmax * row_bytes, dtype=torch.uint8, device=d) for d in devices]
        base = idx * SITE_FLAG_INTS
        self.flags = [pool[base : base + 2 * MAX_TP * NCTA_MAX] for pool in group.pool]
        self.eflag = [pool[base + 2 * MAX_TP * NCTA_MAX : base + SITE_FLAG_INTS] for pool in group.pool]
        sp = [t.data_ptr() for t in self.stage] + [0] * (MAX_TP - tp)
        fp = [t.data_ptr() for t in self.flags] + [0] * (MAX_TP - tp)
        # pinned + non_blocking: a pageable copy would wait for the peer's
        # stream, which may be spinning in a collective this rank has not
        # joined yet
        self._host = [torch.tensor(v, dtype=torch.int64).pin_memory() for v in (sp, fp)]
        self.stage_tab = [self._host[0].to(d, non_blocking=True) for d in devices]
        self.flag_tab = [self._host[1].to(d, non_blocking=True) for d in devices]


class P2PGroup:
    """Every rank's view; ``comm(rank)`` has TpCollectives' interface."""

    def __init__(self, devices, max_local_tokens: int):
        self.devices = [torch.device(d) for d in devices]
        self.tp = len(self.devices)
        self.mmax = int(max_local_tokens)
        self._sites: dict = {}
        self._lock = threading.Lock()
        self.pool = [torch.zeros(MAX_SITES * SITE_FLAG_INTS, dtype=torch.int32, device=d) for d in self.devices]
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

    def comm(self, rank: int) -> "P2PComm":
        return P2PComm(self, rank)


class P2PComm:
    enabled = True
    fell_back = False
    error = ""

    def __init__(self, group: P2PGroup, rank: int):
        self.group, self.rank = group, rank
        self.tp_size = group.tp
        self.device = group.devices[rank]
        self._calls = 0

    def describe(self) -> str:
        return "p2p-single-process"

    def _launch(self, kind, x, out, rows_per_rank):
        row_bytes = x[0].numel() * x.element_size() if x.dim() > 1 else x.element_size()
        key = (kind, row_bytes, x.dtype, tuple(out.shape[1:]))
        s = self.group.site(key, kind, row_bytes)
        r = self.rank
        import os

        knobs = [int(os.environ.get(k, d)) for k, d in (("P2P_AUX", AUX_SYS), ("P2P_NCTA", 0), ("P2P_NT", 0), ("P2P_UNR", 4))]
        exe = compile_p2p(kind, row_bytes, self.tp_size, self.device.index, *knobs)
        _run_compiled(
            exe,
            x.data_ptr(), out.data_ptr(), s.stage_tab[r].data_ptr(), s.flag_tab[r].data_ptr(),
            s.eflag[r].data_ptr(), r, rows_per_rank, self.group.mmax,
            max(1, min(knobs[1] or NCTA, rows_per_rank)),
            torch.cuda.current_stream(self.device),
        )
        return out

    def all_gather(self, x: torch.Tensor, out: torch.Tensor | None = None):
        x = x.contiguous()
        m = x.shape[0]
        if out is None:
            out = torch.empty((m * self.tp_size,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        return self._launch(KIND_AG, x, out, m)

    def reduce_scatter(self, x: torch.Tensor, out: torch.Tensor | None = None):
        x = x.contiguous()
        m = x.shape[0] // self.tp_size
        if out is None:
            out = torch.empty((m,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        return self._launch(KIND_RS, x, out, m)

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor | None = None):
        x = x.contiguous()
        m = x.shape[0] // self.tp_size
        if out is None:
            out = torch.empty_like(x)
        return self._launch(KIND_AR, x, out, m)
