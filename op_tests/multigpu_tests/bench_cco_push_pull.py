#!/usr/bin/env python3
# gfx1250 跨卡 remote-WRITE(push) vs remote-READ(pull) 带宽 A/B 微基准。
#
# 用与现网 MoE dispatch 完全相同的跨卡原语:mori.cco 对称 arena +
# kernel 内 `cco.Window(handle).lsa_ptr(peer, off)` + buffer_load/buffer_store。
# push 与 pull 走同一套 buffer_load/store,唯一区别是哪一端指向 peer(lsa_ptr),
# 从而把"远程写 vs 远程读"的差异单独隔离出来。
#
# 运行(2 rank,非 /app 目录以免 /app/triton 命名空间遮蔽):
#   cd /tmp && FLYDSL_GPU_ARCH=gfx1250 torchrun --standalone --nproc_per_node=2 \
#       /app/aiter/op_tests/multigpu_tests/bench_cco_push_pull.py
#   可选:--nbytes 7168(fp8) / 14336(bf16)  --n_copy 2048  --blocks 512 --verify

import argparse
import os

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco
from mori.cco import Communicator

from aiter import get_gfx
from aiter.ops.flydsl.dispatch_combine_v2 import flydsl_prims as P
from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    SymmArena,
    from_gpu_ptr,
)
from aiter.ops.flydsl.dispatch_combine_v2.intranode_kernels import (
    WAVE,
    LANE_MASK,
    LOG2_WAVE,
    _LANE_STRIDE_I32,
)

_CHASE_N = 1 << 20  # 依赖链读的 permutation 表大小(4MB int32)


def _make_copy(*, pull: bool, nbytes: int, off: int, block_num: int, warps_per_block: int):
    """Build a cross-card copy kernel. pull=False -> push(write peer), pull=True -> read peer."""
    n_i32 = nbytes // 4
    threads = warps_per_block * WAVE

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def copy_k(arena: Int64, local_ptr: Int64, peer: Int32, n_items: Int32):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        gwid = bid * warps_per_block + warp
        gwn = block_num * warps_per_block
        window = cco.Window(arena)

        for i in range(gwid, n_items, gwn):
            peer_base = fx.Int64(window.lsa_ptr(peer, off)) + fx.Int64(i) * fx.Int64(nbytes)
            local_base = fx.Int64(local_ptr) + fx.Int64(i) * fx.Int64(nbytes)
            rsrc_peer = create_buffer_resource_from_addr(peer_base)
            rsrc_local = create_buffer_resource_from_addr(local_base)
            # pull 是编译期 python 常量:三元赋值(无 traced `if`),否则 AST 重写的
            # 内层 for 子函数捕获不到条件赋的名字。远程读:src=peer;远程写:dst=peer。
            rsrc_src = rsrc_peer if pull else rsrc_local
            rsrc_dst = rsrc_local if pull else rsrc_peer
            for chunk in range(lane * 4, n_i32, _LANE_STRIDE_I32):
                v = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                buffer_store(v, rsrc_dst, chunk)

    @flyc.jit
    def run(arena: Int64, local_ptr: Int64, peer: Int32, n_items: Int32, stream=fx.Stream(None)):
        copy_k(arena, local_ptr, peer, n_items).launch(
            grid=(block_num, 1, 1), block=[threads, 1, 1], stream=stream
        )

    return run


def _make_read_latency(*, off: int, hops: int):
    """单线程、单 outstanding:对 peer 的 permutation 表做 hops 次依赖链读。
    每次读地址依赖上次读回值 ⇒ 无法流水,直接暴露远程读 RTT。hops 编译期常量、python 展开。"""

    @flyc.kernel(known_block_size=[1, 1, 1])
    def readlat_k(arena: Int64, sink_ptr: Int64, peer: Int32):
        window = cco.Window(arena)
        peer_base = fx.Int64(window.lsa_ptr(peer, off))
        idx = fx.Int32(0)
        for _ in range(hops):
            idx = fx.Int32(P.load_i32_nt(peer_base, idx))
        P.store_i32_system(sink_ptr, 0, idx)  # sink,防 DCE

    @flyc.jit
    def run(arena: Int64, sink_ptr: Int64, peer: Int32, stream=fx.Stream(None)):
        readlat_k(arena, sink_ptr, peer).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream)

    return run


def _make_pingpong(*, off_flag: int, is_init: bool):
    """两 rank flag ping-pong:initiator 写 peer.flag 再自旋本地 flag,responder 反之。
    单线程,hops 次往返。总时/hops = 一个 RTT(写→对端可见→写回→可见);半程≈单向写可见延迟。"""

    @flyc.kernel(known_block_size=[1, 1, 1])
    def pp_k(arena: Int64, local_flag: Int64, peer: Int32, hops: Int32):
        window = cco.Window(arena)
        peer_flag = fx.Int64(window.lsa_ptr(peer, off_flag))
        for h in range(0, hops, 1):
            exp = h + fx.Int32(1)
            if is_init:
                P.store_i32_system(peer_flag, 0, exp)
                P.spin_until_eq_i32(local_flag, exp)
            else:
                P.spin_until_eq_i32(local_flag, exp)
                P.store_i32_system(peer_flag, 0, exp)

    @flyc.jit
    def run(arena: Int64, local_flag: Int64, peer: Int32, hops: Int32, stream=fx.Stream(None)):
        pp_k(arena, local_flag, peer, hops).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream)

    return run


def _bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def _time_once(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)  # ms


def _run_latency(comm, rank, peer, local, args):
    # arena: chase 表(peer 读)+ ping-pong flag
    arena = SymmArena(comm, [("buf", _CHASE_N * 4), ("flag", 64)])
    off_buf = arena.offset("buf")
    off_flag = arena.offset("flag")
    arena.zero()

    # chase 表 = 随机 permutation(依赖链地址不可预测,击穿预取/行局部性)
    perm = torch.randperm(_CHASE_N, device=local).to(torch.int32)
    from_gpu_ptr(arena.local_ptr("buf"), (_CHASE_N,), torch.int32).copy_(perm)
    sink = torch.zeros(1, dtype=torch.int32, device=local)
    stream = fx.Stream(torch.cuda.current_stream())
    comm.barrier()

    # ── 远程读延迟:两个 hop 数取斜率,消掉 launch/固定开销 ──
    h_lo, h_hi = 256, 2048
    rd_lo = _make_read_latency(off=off_buf, hops=h_lo)
    rd_hi = _make_read_latency(off=off_buf, hops=h_hi)
    for f, h in ((rd_lo, h_lo), (rd_hi, h_hi)):
        for _ in range(5):
            f(arena.handle, sink.data_ptr(), peer, stream)  # warmup
    torch.cuda.synchronize()
    t_lo = min(_time_once(lambda: rd_lo(arena.handle, sink.data_ptr(), peer, stream)) for _ in range(10))
    t_hi = min(_time_once(lambda: rd_hi(arena.handle, sink.data_ptr(), peer, stream)) for _ in range(10))
    read_ns = (t_hi - t_lo) * 1e6 / (h_hi - h_lo)  # ns / 依赖读

    # ── 远程写→对端可见延迟:flag ping-pong ──
    arena.zero("flag")
    comm.barrier()
    pp = _make_pingpong(off_flag=off_flag, is_init=(rank == 0))
    local_flag = arena.local_ptr("flag")
    comm.barrier()
    t_pp = _time_once(lambda: pp(arena.handle, local_flag, peer, args.hops, stream))
    rtt_ns = t_pp * 1e6 / args.hops  # ns / 往返
    comm.barrier()

    if rank == 0:
        print(
            f"[rank0] LATENCY | remote-READ(dependent) {read_ns:7.0f} ns/read "
            f"| ping-pong RTT {rtt_ns:7.0f} ns  (write→visible 半程 {rtt_ns/2:7.0f} ns)",
            flush=True,
        )
    arena.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bw", "latency"], default="bw")
    ap.add_argument("--hops", type=int, default=4000, help="latency: ping-pong 往返次数")
    ap.add_argument("--nbytes", type=int, default=14336, help="每 token 字节数(bf16 7168=14336, fp8=7168)")
    ap.add_argument("--n_copy", type=int, default=2048, help="每 kernel 跨卡搬运 token 数")
    ap.add_argument("--blocks", type=int, default=512)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    assert args.nbytes % 16 == 0, "nbytes 需 16B 对齐(vec4 i32)"

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    assert world == 2, "本基准固定双 rank"
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    os.environ.setdefault("FLYDSL_GPU_ARCH", get_gfx())
    peer = 1 - rank

    uid = Communicator.get_unique_id() if rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=0)
    uid = objs[0]
    comm = Communicator.init(world, rank, uid)

    if args.mode == "latency":
        _run_latency(comm, rank, peer, local, args)
        comm.barrier()
        comm.destroy()
        dist.destroy_process_group()
        return

    region = args.n_copy * args.nbytes
    arena = SymmArena(comm, [("buf", region)])
    off = arena.offset("buf")
    arena.zero()

    # 本地侧(非对称)数据:push 从这里读、pull 写到这里。用 rank 相关 pattern 便于校验。
    n_i32 = region // 4
    local_buf = torch.full((n_i32,), rank + 1, dtype=torch.int32, device=local)

    push_fn = _make_copy(pull=False, nbytes=args.nbytes, off=off, block_num=args.blocks, warps_per_block=args.warps)
    pull_fn = _make_copy(pull=True, nbytes=args.nbytes, off=off, block_num=args.blocks, warps_per_block=args.warps)
    stream = fx.Stream(torch.cuda.current_stream())

    def do_push():
        push_fn(arena.handle, local_buf.data_ptr(), peer, args.n_copy, stream)

    def do_pull():
        pull_fn(arena.handle, local_buf.data_ptr(), peer, args.n_copy, stream)

    if args.verify:
        # push:各 rank 把本地 pattern(rank+1)写进 peer.buf → peer.buf 应全为 (peer_of_writer+1)=(rank? ) 检查本rank buf。
        arena.zero()
        comm.barrier()
        do_push()
        torch.cuda.synchronize()
        comm.barrier()
        buf_view = torch.zeros(n_i32, dtype=torch.int32, device=local)
        from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import from_gpu_ptr
        got = from_gpu_ptr(arena.local_ptr("buf"), (n_i32,), torch.int32)
        expect = peer + 1  # 写进我 buf 的是 peer 的 local pattern
        ok = bool((got == expect).all().item())
        print(f"[rank{rank}] PUSH verify: buf=={expect}? {ok}", flush=True)
        comm.barrier()

    comm.barrier()
    ms_push = _bench(do_push, iters=args.iters)
    comm.barrier()
    ms_pull = _bench(do_pull, iters=args.iters)
    comm.barrier()

    moved = args.n_copy * args.nbytes  # 每 iter 单向搬运字节
    bw_push = moved / (ms_push * 1e-3) / 1e9
    bw_pull = moved / (ms_pull * 1e-3) / 1e9
    print(
        f"[rank{rank}] nbytes={args.nbytes} n_copy={args.n_copy} blk={args.blocks}x{args.warps}w "
        f"| PUSH(write) {ms_push*1e3:7.1f}us {bw_push:6.1f} GB/s "
        f"| PULL(read) {ms_pull*1e3:7.1f}us {bw_pull:6.1f} GB/s "
        f"| write/read={bw_push/bw_pull:.2f}x",
        flush=True,
    )

    comm.barrier()
    arena.close()
    comm.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
