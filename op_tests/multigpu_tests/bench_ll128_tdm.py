# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two hardware preconditions of an LL128 combine on gfx1250.

An LL128 row tags every 128 B line with the generation that wrote it, and the
reader trusts a line whose tag is current. That holds only if

  tear     a reader never sees a 128 B line half old and half new while a peer's
           TDM gather store (SCOPE_SYS, 8 rows per store, as gemm2 issues them)
           rewrites it, and
  reload   a reader that has the old line cached sees the new one once it
           reloads it with the cache policy the reduce would use.

Both run on the cco symmetric arena MegaMoE uses, rank r writing rank r+1 and
reading what rank r-1 writes into its own arena, writer and reader on separate
streams so they overlap.

    torchrun --standalone --nproc_per_node=4 bench_ll128_tdm.py
"""

import argparse
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
import torch
import torch.distributed as dist
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import tdm_prims as TDM
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.arena import SymmetricArena
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.tdm_gather_shim import (
    make_tensor_gather_descriptor,
    tensor_store_gather,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

WAVE = 32
BLOCK = 4 * WAVE
#: One gemm2 row of a 512-column tile in the LL128 layout: 2 segments x 640 B.
ROW_B = 1280
#: The bf16 comb_inp slot pitch at hidden 5120.
STRIDE_B = 16384
#: Rows per gather store, as gemm2 issues them.
GROUP = 8
LINE_B = 128
LINES = GROUP * ROW_B // LINE_B
SCOPE_DEV = 2 << 3
SCOPE_SYS = 3 << 3
#: Reloads (s_sleep 1 each) before the reload test gives up on a trial.
RELOAD_CAP = 1 << 16
#: Result words: lines checked, torn inside a 64 B half, torn between the two
#: 64 B halves, line value changes seen, reload trials failed, reloads spent.
R_LINES, R_TEAR64, R_SPLIT, R_CHANGES, R_FAILS, R_RELOADS = range(6)


def _lds_addr(ptr):
    return fx.Int64(fx.ptrtoint(ptr))


@flyc.jit
def _fill(lds, value, tid):
    """Every dword of the GROUP x ROW_B LDS tile := value."""
    for i in range(tid, GROUP * ROW_B // 4, BLOCK):
        comm_ops.store_i32_lds(lds + fx.Int64(i) * fx.Int64(4), value)


def _gather_store(base, lds, row0, rows, policy):
    """One TDM gather store of GROUP LDS rows to rows row0.. of base."""
    i8g = fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Global, 16)
    view = fx.Tensor(
        fx.make_view(
            fx.inttoptr(i8g, fx.Int64(base)),
            fx.make_layout((rows, ROW_B), (STRIDE_B, 1)),
        )
    )
    desc = make_tensor_gather_descriptor(
        view,
        fx.index_cast(T.index, lds),
        [fx.Int32(row0) + fx.Int32(i) for i in range(GROUP)],
        row_width=ROW_B,
        tensor_dim0=ROW_B,
        tensor_dim1=rows,
        stride=STRIDE_B,
        elem_bytes=1,
        index_size=32,
    )
    tensor_store_gather(desc, cache_policy=policy)
    rocdl.s_nop(0)


def _load_rows(base, row0, lds, policy):
    """One 2-D TDM load of GROUP rows (ROW_B each, STRIDE_B apart) into LDS."""
    TDM.tdm_load(
        TDM.tdm_group0(
            fx.Int32(arith.trunci(T.i32, arith.unwrap(fx.Int64(lds)))),
            fx.Int64(base) + fx.Int64(row0) * fx.Int64(STRIDE_B),
        ),
        TDM.tdm_group1(ROW_B // 4, GROUP, 4, STRIDE_B // 4),
        cache_policy=policy,
    )
    rocdl.s_nop(0)
    TDM.tdm_wait(0)


def _bar():
    fx.rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def _line_state(lds, line):
    """(v0, torn inside a 64 B half, halves uniform but different) of a line."""

    def ld(j):
        return fx.Int32(comm_ops.load_i32_lds(lds + fx.Int64(line * LINE_B + j * 4)))

    v0 = ld(0)
    v16 = ld(16)
    bad = fx.Int32(0)
    for j in range_constexpr(1, 16):
        bad = bad | fx.Int32((ld(j) != v0).select(fx.Int32(1), fx.Int32(0)))
    for j in range_constexpr(17, 32):
        bad = bad | fx.Int32((ld(j) != v16).select(fx.Int32(1), fx.Int32(0)))
    split = fx.Int32(
        ((bad == fx.Int32(0)) & (v16 != v0)).select(fx.Int32(1), fx.Int32(0))
    )
    return v0, bad, split


def build(rows, off_buf, off_vis, off_flag, writer_policy=SCOPE_SYS):
    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def tear_writer(arena: Int64, peer: Int32, iters: Int32):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid >> fx.Int32(5)).ir_value()))
        smem = fx.SharedAllocator(static=False)
        lds = _lds_addr(smem.allocate(GROUP * ROW_B, 128)._ptr)
        win = cco.CachedWindow(arena)
        dst = win.base + fx.Int64(peer) * win.stride + fx.Int64(off_buf)
        for g in range(fx.Int32(1), iters + fx.Int32(1), fx.Int32(1)):
            _fill(lds, g, tid)
            _bar()
            if wave == fx.Int32(0):
                _gather_store(dst, lds, bid * fx.Int32(GROUP), rows, writer_policy)
                TDM.tdm_wait(0)
            _bar()

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def tear_reader(arena: Int64, me: Int32, iters: Int32, policy: Int32, out: Int64):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid >> fx.Int32(5)).ir_value()))
        smem = fx.SharedAllocator(static=False)
        lds = _lds_addr(smem.allocate(GROUP * ROW_B, 128)._ptr)
        win = cco.CachedWindow(arena)
        src = win.base + fx.Int64(me) * win.stride + fx.Int64(off_buf)
        n_lines = fx.Int32(0)
        n_tear = fx.Int32(0)
        n_split = fx.Int32(0)
        n_change = fx.Int32(0)
        prev = fx.Int32(-1)
        line = fx.Int32((tid < fx.Int32(LINES)).select(tid, fx.Int32(0)))
        live = fx.Int32((tid < fx.Int32(LINES)).select(fx.Int32(1), fx.Int32(0)))
        for _ in range(fx.Int32(0), iters, fx.Int32(1)):
            if wave == fx.Int32(0):
                if policy == fx.Int32(SCOPE_SYS):
                    _load_rows(src, bid * fx.Int32(GROUP), lds, SCOPE_SYS)
                elif policy == fx.Int32(SCOPE_DEV):
                    _load_rows(src, bid * fx.Int32(GROUP), lds, SCOPE_DEV)
                else:
                    _load_rows(src, bid * fx.Int32(GROUP), lds, 0)
            _bar()
            v0, bad, split = _line_state(lds, line)
            n_lines = n_lines + live
            n_tear = n_tear + live * bad
            n_split = n_split + live * split
            n_change = n_change + live * fx.Int32(
                ((prev >= fx.Int32(0)) & (v0 != prev)).select(fx.Int32(1), fx.Int32(0))
            )
            prev = v0
            _bar()
        comm_ops.atomic_add_agent(out + fx.Int64(R_LINES * 4), n_lines)
        comm_ops.atomic_add_agent(out + fx.Int64(R_TEAR64 * 4), n_tear)
        comm_ops.atomic_add_agent(out + fx.Int64(R_SPLIT * 4), n_split)
        comm_ops.atomic_add_agent(out + fx.Int64(R_CHANGES * 4), n_change)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def vis_writer(arena: Int64, peer: Int32, me: Int32, trials: Int32):
        """Per trial t: wait for the reader's ready = t, write t + 1 everywhere,
        retire the store, then publish written = t on the reader."""
        tid = fx.Int32(fx.thread_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid >> fx.Int32(5)).ir_value()))
        smem = fx.SharedAllocator(static=False)
        lds = _lds_addr(smem.allocate(GROUP * ROW_B, 128)._ptr)
        win = cco.CachedWindow(arena)
        dst = win.base + fx.Int64(peer) * win.stride + fx.Int64(off_vis)
        ready = win.base + fx.Int64(me) * win.stride + fx.Int64(off_flag)
        written = win.base + fx.Int64(peer) * win.stride + fx.Int64(off_flag + 4)
        for t in range(fx.Int32(1), trials + fx.Int32(1), fx.Int32(1)):
            if tid == fx.Int32(0):
                comm_ops.spin_until_ge_i32_system(ready, t)
            _fill(lds, t + fx.Int32(1), tid)
            _bar()
            if wave == fx.Int32(0):
                _gather_store(dst, lds, fx.Int32(0), GROUP, writer_policy)
                TDM.tdm_wait(0)
                comm_ops.waitcnt_stores()
                if tid == fx.Int32(0):
                    comm_ops.store_i32_global_system_release(written, t)
            _bar()

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def vis_reader(
        arena: Int64, peer: Int32, me: Int32, trials: Int32, policy: Int32, out: Int64
    ):
        """Per trial t: load the rows with the default policy (caching the old
        value t), signal ready = t, wait for written = t, then reload with
        ``policy`` until every dword reads t + 1."""
        tid = fx.Int32(fx.thread_idx.x)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid >> fx.Int32(5)).ir_value()))
        smem = fx.SharedAllocator(static=False)
        lds = _lds_addr(smem.allocate(GROUP * ROW_B, 128)._ptr)
        flag = smem.allocate(16, 16)._ptr
        flag_lds = _lds_addr(flag)
        win = cco.CachedWindow(arena)
        src = win.base + fx.Int64(me) * win.stride + fx.Int64(off_vis)
        ready = win.base + fx.Int64(peer) * win.stride + fx.Int64(off_flag)
        written = win.base + fx.Int64(me) * win.stride + fx.Int64(off_flag + 4)
        line = fx.Int32((tid < fx.Int32(LINES)).select(tid, fx.Int32(0)))
        fails = fx.Int32(0)
        spent = fx.Int32(0)
        for t in range(fx.Int32(1), trials + fx.Int32(1), fx.Int32(1)):
            if wave == fx.Int32(0):
                _load_rows(src, fx.Int32(0), lds, 0)
                if tid == fx.Int32(0):
                    comm_ops.store_i32_global_system_release(ready, t)
                    comm_ops.spin_until_ge_i32_system(written, t)
            _bar()
            n = fx.Int32(0)
            ok = fx.Int32(0)
            while (ok == fx.Int32(0)) & (n < fx.Int32(RELOAD_CAP)):
                if wave == fx.Int32(0):
                    if policy == fx.Int32(SCOPE_SYS):
                        _load_rows(src, fx.Int32(0), lds, SCOPE_SYS)
                    elif policy == fx.Int32(SCOPE_DEV):
                        _load_rows(src, fx.Int32(0), lds, SCOPE_DEV)
                    else:
                        _load_rows(src, fx.Int32(0), lds, 0)
                if tid == fx.Int32(0):
                    comm_ops.store_i32_lds(flag_lds, fx.Int32(0))
                _bar()
                v0, bad, split = _line_state(lds, line)
                stale = (tid < fx.Int32(LINES)) & (
                    (bad != fx.Int32(0))
                    | (split != fx.Int32(0))
                    | (v0 != t + fx.Int32(1))
                )
                if stale:
                    comm_ops.store_i32_lds(flag_lds, fx.Int32(1))
                _bar()
                ok = fx.Int32(
                    (fx.Int32(comm_ops.load_i32_lds(flag_lds)) == fx.Int32(0)).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                )
                n = n + fx.Int32(1)
                if ok == fx.Int32(0):
                    fx.rocdl.s_sleep(1)
                _bar()
            spent = spent + n
            fails = fails + fx.Int32(
                (ok == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
            )
        if tid == fx.Int32(0):
            comm_ops.atomic_add_agent(out + fx.Int64(R_FAILS * 4), fails)
            comm_ops.atomic_add_agent(out + fx.Int64(R_RELOADS * 4), spent)

    @flyc.jit
    def run_tear_writer(
        arena: Int64,
        peer: Int32,
        iters: Int32,
        blocks: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        tear_writer(arena, peer, iters).launch(
            grid=(blocks, 1, 1), block=[BLOCK, 1, 1], stream=stream
        )

    @flyc.jit
    def run_tear_reader(
        arena: Int64,
        me: Int32,
        iters: Int32,
        policy: Int32,
        out: Int64,
        blocks: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        tear_reader(arena, me, iters, policy, out).launch(
            grid=(blocks, 1, 1), block=[BLOCK, 1, 1], stream=stream
        )

    @flyc.jit
    def run_vis_writer(
        arena: Int64,
        peer: Int32,
        me: Int32,
        trials: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        vis_writer(arena, peer, me, trials).launch(
            grid=(1, 1, 1), block=[BLOCK, 1, 1], stream=stream
        )

    @flyc.jit
    def run_vis_reader(
        arena: Int64,
        peer: Int32,
        me: Int32,
        trials: Int32,
        policy: Int32,
        out: Int64,
        stream=fx.Stream(None),  # noqa: B008
    ):
        vis_reader(arena, peer, me, trials, policy, out).launch(
            grid=(1, 1, 1), block=[BLOCK, 1, 1], stream=stream
        )

    return run_tear_writer, run_tear_reader, run_vis_writer, run_vis_reader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, default=256, help="writer/reader CTAs")
    p.add_argument("--tear_iters", type=int, default=4000)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--policies", default="24,16,0", help="reader cache policies")
    p.add_argument("--writer_policy", type=int, default=SCOPE_SYS)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="gloo")
    from mori.cco import Communicator

    rows = args.blocks * GROUP
    regions = [("buf", rows * STRIDE_B), ("vis", GROUP * STRIDE_B), ("flag", 256)]
    total = sum(n for _, n in regions) + 4096
    per_rank_vmm = (total + (2 << 20) - 1) // (2 << 20) * (2 << 20) + (4 << 20)
    uid = Communicator.get_unique_id() if rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=0)
    comm = Communicator.init(world, rank, objs[0], per_rank_vmm=per_rank_vmm)
    arena = SymmetricArena(comm, regions)
    arena.zero()
    torch.cuda.synchronize()
    comm.barrier()

    tw, tr, vw, vr = build(
        rows,
        arena.offset("buf"),
        arena.offset("vis"),
        arena.offset("flag"),
        args.writer_policy,
    )
    peer = (rank + 1) % world
    src_rank = (rank - 1) % world
    s_w = torch.cuda.Stream()
    s_r = torch.cuda.Stream()
    report = []
    for policy in [int(x) for x in args.policies.split(",")]:
        out = torch.zeros(8, dtype=torch.int32, device="cuda")
        arena.zero()
        torch.cuda.synchronize()
        comm.barrier()
        _run_compiled(
            tw, arena.handle, peer, args.tear_iters, args.blocks, fx.Stream(s_w)
        )
        _run_compiled(
            tr,
            arena.handle,
            rank,
            args.tear_iters,
            policy,
            out.data_ptr(),
            args.blocks,
            fx.Stream(s_r),
        )
        torch.cuda.synchronize()
        comm.barrier()
        arena.zero()
        torch.cuda.synchronize()
        comm.barrier()
        _run_compiled(vw, arena.handle, peer, rank, args.trials, fx.Stream(s_w))
        _run_compiled(
            vr,
            arena.handle,
            src_rank,
            rank,
            args.trials,
            policy,
            out.data_ptr(),
            fx.Stream(s_r),
        )
        torch.cuda.synchronize()
        comm.barrier()
        report.append((policy, out.cpu().tolist()))

    all_reports = [None] * world
    dist.all_gather_object(all_reports, report)
    if rank == 0:
        names = {SCOPE_SYS: "SCOPE_SYS", SCOPE_DEV: "SCOPE_DEV", 0: "default"}
        print(
            f"writer_policy={args.writer_policy} blocks={args.blocks} rows={rows} "
            f"row={ROW_B}B tear_iters={args.tear_iters} "
            f"trials={args.trials}"
        )
        for i, (policy, _) in enumerate(all_reports[0]):
            print(f"-- reader policy {names.get(policy, policy)}")
            for r in range(world):
                v = all_reports[r][i][1]
                print(
                    f"  rank{r}: lines={v[R_LINES]} torn_in_64B={v[R_TEAR64]} "
                    f"torn_between_64B={v[R_SPLIT]} changes_seen={v[R_CHANGES]} | "
                    f"reload fails={v[R_FAILS]}/{args.trials} "
                    f"avg_reloads={v[R_RELOADS] / max(1, args.trials):.2f}"
                )
    arena.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
