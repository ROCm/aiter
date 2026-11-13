# SPDX-License-Identifier: MIT
"""Exercise all four logical rail nodes on two physical eight-GPU hosts.

Run with torchrun --nnodes=2 --nproc_per_node=8, one process per GPU. Rank
order is host-major. This checks the transport overlay needed by a 4x4
MegaMoE protocol; it does not run Stage1/GEMM/Stage2.
"""

from __future__ import annotations

import argparse
import inspect
import os
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
import torch
import torch.distributed as dist
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import rocdl
from flydsl.expr import gpu
from mori.cco import GDA_CONNECTION_RAIL, CCODevCommRequirements, Communicator, UniqueId

try:
    from aiter.ops.flydsl.kernels.mega_moe_internode.virtual_topology import (
        VirtualNodeTopology,
    )
except ImportError:
    # The probe can be copied with its small planner into a remote task dir
    # without replacing the installed AITER package during an active run.
    from virtual_topology import VirtualNodeTopology

WORLD = 16
LOGICAL_GPUS = 4
PHYSICAL_GPUS = 8
_DEFAULT_STREAM = fx.Stream(None)
PAYLOAD_WORDS = 512
PAYLOAD_BYTES = PAYLOAD_WORDS * 8
SHIFTS = 3
AREA_BYTES = ((PAYLOAD_BYTES + 8 + 255) // 256) * 256
SLOT_BYTES = 3 * AREA_BYTES
CREDIT_SLOT_BYTES = 128
WINDOW_BYTES = ((SHIFTS * (SLOT_BYTES + CREDIT_SLOT_BYTES) + 4095) // 4096) * 4096
COPY_THREADS = 256
_GDA_POST_OPTIONS = {}
_GDA_FLUSH_OPTIONS = {}
_GDA_HAS_RUNTIME_CTX = "ctx" in inspect.signature(cco.Gda.put).parameters
_GDA_DOORBELL = getattr(cco, "GdaDoorbell", None)
_CALLER_ORDERED = False


def _gda(dev_comm):
    if _GDA_DOORBELL is not None and _CALLER_ORDERED:
        return cco.DevComm(dev_comm).gda(0, doorbell=_GDA_DOORBELL.CALLER_ORDERED)
    return cco.DevComm(dev_comm).gda(0)


def _gda_context(qp):
    # Earlier transport runtimes have only the handle's static context zero.
    return {"ctx": qp} if _GDA_HAS_RUNTIME_CTX else {}


def _global_i64(addr):
    ptr_type = fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8)
    return fx.to_llvm_ptr(fx.inttoptr(ptr_type, fx.Int64(addr)))


def _load_flag(addr):
    return llvm_d.LoadOp(
        ir.IntegerType.get_signless(64),
        _global_i64(addr),
        alignment=8,
        ordering=llvm_d.AtomicOrdering.acquire,
        syncscope="one-as",
    ).result


def _store_flag(addr, value):
    llvm_d.StoreOp(
        fx.Int64(value).ir_value(),
        _global_i64(addr),
        alignment=8,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    )


def _send_off(shift: int) -> int:
    return (shift - 1) * SLOT_BYTES


def _bridge_off(shift: int) -> int:
    return _send_off(shift) + AREA_BYTES


def _recv_off(shift: int) -> int:
    return _send_off(shift) + 2 * AREA_BYTES


def _credit_bridge_off(shift: int) -> int:
    return SHIFTS * SLOT_BYTES + (shift - 1) * CREDIT_SLOT_BYTES


def _credit_off(shift: int) -> int:
    return _credit_bridge_off(shift) + 64


class _CudaArrayView:
    def __init__(
        self, ptr: int, shape: tuple[int, ...], strides: tuple[int, ...] | None = None
    ):
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": shape,
            "typestr": "<i8",
            "version": 3,
            "strides": strides,
        }


@flyc.kernel(known_block_size=[COPY_THREADS, 1, 1])
def lsa_copy_kernel(
    src_win_handle: fx.Int64,
    dst_win_handle: fx.Int64,
    src_lsa: fx.Int32,
    dst_lsa: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    generation: fx.Int64,
):
    src_win = cco.Window(src_win_handle)
    dst_win = cco.Window(dst_win_handle)
    src = src_win.lsa_ptr(src_lsa, src_off)
    dst = dst_win.lsa_ptr(dst_lsa, dst_off)
    tid = fx.Int32(fx.thread_idx.x)
    for word in range(tid, fx.Int32(PAYLOAD_WORDS), fx.Int32(COPY_THREADS)):
        offset = fx.Int64(word) * fx.Int64(8)
        value = fx.Int64(
            llvm_d.LoadOp(
                ir.IntegerType.get_signless(64),
                _global_i64(src + offset),
                alignment=8,
            ).result
        )
        llvm_d.StoreOp(value.ir_value(), _global_i64(dst + offset), alignment=8)
    rocdl.s_waitcnt(0)
    gpu.barrier()
    if tid == fx.Int32(0):
        _store_flag(
            dst_win.lsa_ptr(dst_lsa, dst_off + fx.Int64(PAYLOAD_BYTES)),
            generation,
        )


@flyc.kernel(known_block_size=[64, 1, 1])
def rail_copy_kernel(
    dev_comm: fx.Int64,
    src_win_handle: fx.Int64,
    dst_win_handle: fx.Int64,
    peer_rank: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    generation: fx.Int64,
    qp: fx.Int32,
):
    # All 64 lanes participate in MORI's warp-cooperative post and flush.
    gda = _gda(dev_comm)
    gda.put(
        peer_rank,
        dst_win_handle,
        dst_off,
        src_win_handle,
        src_off,
        fx.Int64(PAYLOAD_BYTES),
        coop=cco.CoopScope.WARP,
        **_gda_context(qp),
        **_GDA_POST_OPTIONS,
    )
    gda.put_value(
        peer_rank,
        dst_win_handle,
        dst_off + fx.Int64(PAYLOAD_BYTES),
        generation,
        coop=cco.CoopScope.WARP,
        **_gda_context(qp),
        **_GDA_POST_OPTIONS,
    )
    gda.flush_peer(
        peer_rank, coop=cco.CoopScope.WARP, **_gda_context(qp), **_GDA_FLUSH_OPTIONS
    )


@flyc.kernel(known_block_size=[64, 1, 1])
def lsa_flag_kernel(
    win_handle: fx.Int64,
    dst_lsa: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
):
    if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
        _store_flag(cco.Window(win_handle).lsa_ptr(dst_lsa, dst_off), generation)


@flyc.kernel(known_block_size=[64, 1, 1])
def rail_flag_kernel(
    dev_comm: fx.Int64,
    win_handle: fx.Int64,
    peer_rank: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
    qp: fx.Int32,
):
    gda = _gda(dev_comm)
    gda.put_value(
        peer_rank,
        win_handle,
        dst_off,
        generation,
        coop=cco.CoopScope.WARP,
        **_gda_context(qp),
        **_GDA_POST_OPTIONS,
    )
    gda.flush_peer(
        peer_rank, coop=cco.CoopScope.WARP, **_gda_context(qp), **_GDA_FLUSH_OPTIONS
    )


@flyc.kernel(known_block_size=[64, 1, 1])
def wait_flag_kernel(flag_ptr: fx.Int64, generation: fx.Int64):
    if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
        seen = fx.Int64(_load_flag(flag_ptr))
        while seen < generation:
            seen = fx.Int64(_load_flag(flag_ptr))


@flyc.jit
def run_lsa_copy(
    src_win_handle: fx.Int64,
    dst_win_handle: fx.Int64,
    src_lsa: fx.Int32,
    dst_lsa: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    generation: fx.Int64,
    stream=_DEFAULT_STREAM,
):
    lsa_copy_kernel(
        src_win_handle,
        dst_win_handle,
        src_lsa,
        dst_lsa,
        src_off,
        dst_off,
        generation,
    ).launch(grid=(1, 1, 1), block=(COPY_THREADS, 1, 1), stream=stream)


@flyc.jit
def run_rail_copy(
    dev_comm: fx.Int64,
    src_win_handle: fx.Int64,
    dst_win_handle: fx.Int64,
    peer_rank: fx.Int32,
    src_off: fx.Int64,
    dst_off: fx.Int64,
    generation: fx.Int64,
    qp: fx.Int32,
    stream=_DEFAULT_STREAM,
):
    rail_copy_kernel(
        dev_comm,
        src_win_handle,
        dst_win_handle,
        peer_rank,
        src_off,
        dst_off,
        generation,
        qp,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def run_lsa_flag(
    win_handle: fx.Int64,
    dst_lsa: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
    stream=_DEFAULT_STREAM,
):
    lsa_flag_kernel(win_handle, dst_lsa, dst_off, generation).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


@flyc.jit
def run_rail_flag(
    dev_comm: fx.Int64,
    win_handle: fx.Int64,
    peer_rank: fx.Int32,
    dst_off: fx.Int64,
    generation: fx.Int64,
    qp: fx.Int32,
    stream=_DEFAULT_STREAM,
):
    rail_flag_kernel(dev_comm, win_handle, peer_rank, dst_off, generation, qp).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


@flyc.jit
def run_wait_flag(
    flag_ptr: fx.Int64,
    generation: fx.Int64,
    stream=_DEFAULT_STREAM,
):
    wait_flag_kernel(flag_ptr, generation).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


def _pattern(rank: int, shift: int, generation: int) -> torch.Tensor:
    return (
        torch.arange(PAYLOAD_WORDS, dtype=torch.int64, device="cuda")
        + rank * 100000
        + shift * 10000
        + generation * 1000000
    )


def main() -> None:
    global PAYLOAD_WORDS, PAYLOAD_BYTES, AREA_BYTES, SLOT_BYTES, WINDOW_BYTES
    global _CALLER_ORDERED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--delay-rank", type=int, default=-1)
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument(
        "--mr-sizes-mib",
        default="",
        help="three independent MR sizes, e.g. 960,960,704 for DSv4 TPR16384",
    )
    parser.add_argument("--gda-contexts", type=int, default=1)
    parser.add_argument("--rail-team", action="store_true")
    parser.add_argument("--caller-ordered", action="store_true")
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=PAYLOAD_BYTES,
        help="bytes sent to each logical peer (262144 covers DSv4 A8W4 TPR32)",
    )
    parser.add_argument(
        "--multi-mr",
        action="store_true",
        help="register a second allocation and use it for rail destinations",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.payload_bytes < 16 or args.payload_bytes % 16:
        parser.error("--payload-bytes must be a positive multiple of 16")
    if args.delay_ms < 0 or (args.delay_ms and not 0 <= args.delay_rank < WORLD):
        parser.error("a positive --delay-ms needs --delay-rank in [0, 16)")
    if args.gda_contexts < 1:
        parser.error("--gda-contexts must be positive")
    if args.gda_contexts > 1 and not _GDA_HAS_RUNTIME_CTX:
        parser.error("multiple contexts require MORI's per-call ctx API")
    if args.rail_team and not hasattr(cco, "Team"):
        parser.error("--rail-team requires MORI Team.GDA")
    if (
        args.caller_ordered
        and _GDA_DOORBELL is None
        and "posted_before_flush" not in inspect.signature(cco.Gda.put).parameters
    ):
        parser.error("--caller-ordered requires MORI caller-ordered posting")
    if args.rail_team:
        _GDA_POST_OPTIONS["team"] = cco.Team.GDA
        _GDA_FLUSH_OPTIONS["team"] = cco.Team.GDA
    if args.caller_ordered:
        _CALLER_ORDERED = True
        # One warp posts payload and ready, then flushes that same QP. The
        # ready WQE follows the payload WQE before the caller rings the doorbell.
        if _GDA_DOORBELL is None:
            _GDA_POST_OPTIONS.update(defer_doorbell=True, posted_before_flush=True)
            _GDA_FLUSH_OPTIONS.update(posted_before_flush=True)
    expected_ccqe = os.environ.get("CCO_TEST_BITCODE_CCQE")
    if expected_ccqe is not None:
        from mori.jit.core import is_ccqe_enabled

        if expected_ccqe not in ("0", "1"):
            parser.error("CCO_TEST_BITCODE_CCQE must be 0 or 1")
        if is_ccqe_enabled() != bool(int(expected_ccqe)):
            raise RuntimeError("explicit device bitcode disagrees with Ionic CQ mode")

    def rail_peer(world_rank):
        return world_rank // PHYSICAL_GPUS if args.rail_team else world_rank

    # The module is compiled once per process after argument parsing. All
    # offsets and loop bounds must use the same payload geometry.
    PAYLOAD_BYTES = args.payload_bytes
    PAYLOAD_WORDS = PAYLOAD_BYTES // 8
    AREA_BYTES = ((PAYLOAD_BYTES + 8 + 255) // 256) * 256
    SLOT_BYTES = 3 * AREA_BYTES
    WINDOW_BYTES = ((SHIFTS * (SLOT_BYTES + CREDIT_SLOT_BYTES) + 4095) // 4096) * 4096
    if args.mr_sizes_mib:
        try:
            mr_sizes = tuple(int(v) << 20 for v in args.mr_sizes_mib.split(","))
        except ValueError:
            parser.error("--mr-sizes-mib needs three integer sizes")
        if len(mr_sizes) != 3 or min(mr_sizes) < WINDOW_BYTES:
            parser.error("three MR sizes must each fit the probe window")
        if not args.multi_mr:
            parser.error("--mr-sizes-mib requires --multi-mr")
    else:
        mr_sizes = (WINDOW_BYTES,) * (2 if args.multi_mr else 1)
    # The CCO LSA team must reflect physical topology. Logical nodes are
    # arithmetic groups inside the eight-rank physical LSA team.
    os.environ["MORI_CCO_FABRIC_DISABLE"] = "1"
    if "MORI_VPOD_ID" in os.environ:
        raise RuntimeError("MORI_VPOD_ID must be unset for the virtual4 overlay")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"[rank {rank}] process group init", flush=True)
    dist.init_process_group("gloo")
    print(f"[rank {rank}] process group ready", flush=True)
    if dist.get_world_size() != WORLD or local_rank != rank % PHYSICAL_GPUS:
        raise RuntimeError(
            "expected 2 physical nodes x 8 GPUs in host-major rank order"
        )
    topology = VirtualNodeTopology(WORLD, LOGICAL_GPUS, PHYSICAL_GPUS)

    print(f"[rank {rank}] unique ID broadcast", flush=True)
    uid_list = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid_list, src=0)
    print(f"[rank {rank}] unique ID ready", flush=True)
    uid = UniqueId.from_bytes(uid_list[0])
    failures = 0
    mr_failures = 0
    print(f"[rank {rank}] CCO init", flush=True)
    vmm_bytes = max(
        1 << 30,
        ((sum(mr_sizes) + (512 << 20) + (1 << 30) - 1) // (1 << 30)) * (1 << 30),
    )
    with Communicator.init(WORLD, rank, uid, per_rank_vmm=vmm_bytes) as comm:
        print(f"[rank {rank}] CCO alloc", flush=True)
        memories = []
        for index, size in enumerate(mr_sizes):
            print(f"[rank {rank}] CCO alloc MR {index} bytes={size}", flush=True)
            if args.mr_sizes_mib:
                # MORI's gpuDirectRDMACapable hipMemSetAccess can fail when
                # several processes on one host allocate large VMM regions
                # concurrently. Serialize local GPU lanes on both hosts.
                for lane in range(PHYSICAL_GPUS):
                    if local_rank == lane:
                        memory = comm.alloc_mem(size)
                    dist.barrier()
            else:
                memory = comm.alloc_mem(size)
                dist.barrier()
            memories.append(memory)
        print(f"[rank {rank}] CCO register", flush=True)
        windows = []
        for index, (memory, size) in enumerate(zip(memories, mr_sizes)):
            # ccoWindowRegister requires the base returned by alloc_mem; an
            # interior pointer is rejected as absent from MORI's allocTable.
            print(f"[rank {rank}] CCO register MR {index}", flush=True)
            windows.append(comm.register_window(memory.ptr, size))
            dist.barrier()
        window = windows[0]
        second_window = windows[1] if len(windows) > 1 else None
        credit_window = windows[2] if len(windows) > 2 else second_window or window
        req = CCODevCommRequirements()
        req.gda_connection_type = GDA_CONNECTION_RAIL
        req.gda_context_count = args.gda_contexts
        req.gda_signal_count = 0
        req.gda_counter_count = 0
        req.lsa_barrier_count = 0
        req.rail_gda_barrier_count = 0
        req.barrier_count = 0
        print(f"[rank {rank}] CCO devcomm", flush=True)
        dev_comm = comm.create_dev_comm(req)
        print(f"[rank {rank}] CCO ready", flush=True)
        if dev_comm.lsa_size != PHYSICAL_GPUS:
            raise RuntimeError(
                f"CCO LSA size {dev_comm.lsa_size}, expected {PHYSICAL_GPUS}"
            )

        dst_window = second_window if second_window else window
        dst_handle = dst_window.handle
        dst_ptr = dst_window.local_ptr
        credit_handle = credit_window.handle
        credit_ptr = credit_window.local_ptr
        view = torch.as_tensor(
            _CudaArrayView(
                window.local_ptr,
                (SHIFTS, 3, PAYLOAD_WORDS),
                (SLOT_BYTES, AREA_BYTES, 8),
            ),
            device="cuda",
        )
        recv_view = torch.as_tensor(
            _CudaArrayView(
                dst_ptr,
                (SHIFTS, 3, PAYLOAD_WORDS),
                (SLOT_BYTES, AREA_BYTES, 8),
            ),
            device="cuda",
        )
        torch.as_tensor(
            _CudaArrayView(window.local_ptr, (WINDOW_BYTES // 8,)), device="cuda"
        ).zero_()
        if second_window:
            torch.as_tensor(
                _CudaArrayView(dst_ptr, (WINDOW_BYTES // 8,)), device="cuda"
            ).zero_()
        if credit_window is not dst_window:
            torch.as_tensor(
                _CudaArrayView(credit_ptr, (WINDOW_BYTES // 8,)), device="cuda"
            ).zero_()
        for size, mr_window in zip(mr_sizes, windows):
            if size > WINDOW_BYTES:
                torch.as_tensor(
                    _CudaArrayView(mr_window.local_ptr + size - 8, (1,)),
                    device="cuda",
                ).zero_()
        torch.cuda.synchronize()
        dist.barrier()
        if args.mr_sizes_mib:
            rail_node = 1 - topology.physical_node(rank)
            remote_rank = rail_node * PHYSICAL_GPUS + topology.physical_lsa_rank(rank)
            for index, (size, mr_window) in enumerate(zip(mr_sizes, windows)):
                run_rail_flag(
                    dev_comm.ptr,
                    mr_window.handle,
                    rail_peer(remote_rank),
                    size - 8,
                    1000 + rank * 64 + index,
                    index % args.gda_contexts,
                )
            torch.cuda.synchronize()
            for index, (size, mr_window) in enumerate(zip(mr_sizes, windows)):
                expected = 1000 + remote_rank * 64 + index
                run_wait_flag(mr_window.local_ptr + size - 8, expected)
            torch.cuda.synchronize()
            for index, (size, mr_window) in enumerate(zip(mr_sizes, windows)):
                actual = int(
                    torch.as_tensor(
                        _CudaArrayView(mr_window.local_ptr + size - 8, (1,)),
                        device="cuda",
                    ).item()
                )
                expected = 1000 + remote_rank * 64 + index
                if actual != expected:
                    mr_failures += 1
                    print(
                        f"[rank {rank}] MR {index} high-address mismatch "
                        f"actual={actual} expected={expected}",
                        flush=True,
                    )
        print(f"[rank {rank}] window view ready", flush=True)
        for generation in range(args.rounds):
            qp = generation % args.gda_contexts
            if rank == args.delay_rank and args.delay_ms:
                time.sleep(args.delay_ms / 1000)
            # Ready/credit values are monotonic. Each source waits for all
            # credits before reusing its three send slots in the next round.
            flag = generation + 1
            for shift in (1, 2, 3):
                view[shift - 1, 0].copy_(_pattern(rank, shift, generation))
            torch.cuda.synchronize()

            for shift in (1, 2, 3):
                target_node = (
                    topology.logical_node(rank) + shift
                ) % topology.logical_nodes
                send = topology.aligned_peer(rank, target_node)
                print(
                    f"[rank {rank}] generation={generation} shift={shift} "
                    f"route={send.kind}",
                    flush=True,
                )
                if send.kind == "lsa":
                    run_lsa_copy(
                        window.handle,
                        dst_handle,
                        topology.physical_lsa_rank(rank),
                        topology.physical_lsa_rank(send.destination_rank),
                        _send_off(shift),
                        _recv_off(shift),
                        flag,
                    )
                else:
                    run_rail_copy(
                        dev_comm.ptr,
                        window.handle,
                        dst_handle,
                        rail_peer(
                            send.bridge_rank
                            if send.bridge_rank is not None
                            else send.destination_rank
                        ),
                        _send_off(shift),
                        (
                            _bridge_off(shift)
                            if send.kind == "rail_relay"
                            else _recv_off(shift)
                        ),
                        flag,
                        qp,
                    )
            torch.cuda.synchronize()
            print(f"[rank {rank}] generation={generation} sends complete", flush=True)

            for shift in (1, 2, 3):
                remote_source = (
                    1 - topology.physical_node(rank)
                ) * PHYSICAL_GPUS + topology.physical_lsa_rank(rank)
                source_node = topology.logical_node(remote_source)
                remote_target = (source_node + shift) % topology.logical_nodes
                incoming = topology.aligned_peer(remote_source, remote_target)
                if incoming.kind == "rail_relay" and incoming.bridge_rank == rank:
                    run_wait_flag(dst_ptr + _bridge_off(shift) + PAYLOAD_BYTES, flag)
                    run_lsa_copy(
                        dst_handle,
                        dst_handle,
                        topology.physical_lsa_rank(rank),
                        topology.physical_lsa_rank(incoming.destination_rank),
                        _bridge_off(shift),
                        _recv_off(shift),
                        flag,
                    )
            torch.cuda.synchronize()
            print(f"[rank {rank}] generation={generation} relays complete", flush=True)

            for shift in (1, 2, 3):
                run_wait_flag(dst_ptr + _recv_off(shift) + PAYLOAD_BYTES, flag)
            torch.cuda.synchronize()
            for shift in (1, 2, 3):
                source_node = (
                    topology.logical_node(rank) - shift
                ) % topology.logical_nodes
                source_rank = source_node * LOGICAL_GPUS + rank % LOGICAL_GPUS
                expected = _pattern(source_rank, shift, generation)
                actual = recv_view[shift - 1, 2]
                if not torch.equal(actual, expected):
                    failures += 1
                    print(
                        f"[rank {rank}] shift={shift} source={source_rank} "
                        f"mismatch={int((actual != expected).sum().item())}",
                        flush=True,
                    )

            # Ack each received payload only after verifying it. The reverse
            # path uses the same aligned rail and optional local relay.
            for shift in (1, 2, 3):
                source_node = (
                    topology.logical_node(rank) - shift
                ) % topology.logical_nodes
                credit = topology.aligned_peer(rank, source_node)
                if credit.kind == "lsa":
                    run_lsa_flag(
                        credit_handle,
                        topology.physical_lsa_rank(credit.destination_rank),
                        _credit_off(shift),
                        flag,
                    )
                else:
                    run_rail_flag(
                        dev_comm.ptr,
                        credit_handle,
                        rail_peer(
                            credit.bridge_rank
                            if credit.bridge_rank is not None
                            else credit.destination_rank
                        ),
                        (
                            _credit_bridge_off(shift)
                            if credit.kind == "rail_relay"
                            else _credit_off(shift)
                        ),
                        flag,
                        qp,
                    )
            torch.cuda.synchronize()
            for shift in (1, 2, 3):
                remote_source = (
                    1 - topology.physical_node(rank)
                ) * PHYSICAL_GPUS + topology.physical_lsa_rank(rank)
                credit_target = (
                    topology.logical_node(remote_source) - shift
                ) % topology.logical_nodes
                incoming = topology.aligned_peer(remote_source, credit_target)
                if incoming.kind == "rail_relay" and incoming.bridge_rank == rank:
                    run_wait_flag(credit_ptr + _credit_bridge_off(shift), flag)
                    run_lsa_flag(
                        credit_handle,
                        topology.physical_lsa_rank(incoming.destination_rank),
                        _credit_off(shift),
                        flag,
                    )
            torch.cuda.synchronize()
            for shift in (1, 2, 3):
                run_wait_flag(credit_ptr + _credit_off(shift), flag)
            torch.cuda.synchronize()

        total = torch.tensor([failures, mr_failures], dtype=torch.int64)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if rank == 0:
            mr_status = (
                f"{3 * WORLD - int(total[1])}/{3 * WORLD}" if args.mr_sizes_mib else "-"
            )
            print(
                f"virtual4 transport: {48 * args.rounds - int(total[0])}/"
                f"{48 * args.rounds} directed peer payloads passed; "
                f"payload_bytes={PAYLOAD_BYTES}; multi_mr={args.multi_mr}; "
                f"mr_sizes_mib={args.mr_sizes_mib or '-'}; "
                f"gda_contexts={args.gda_contexts}; ready_credit=True; "
                f"gda_contexts_exercised={min(args.rounds, args.gda_contexts)}; "
                f"rail_team={args.rail_team}; caller_ordered={args.caller_ordered}; "
                f"mr_high_offsets={mr_status} "
                f"failures={int(total.sum())}",
                flush=True,
            )
        if int(total.sum()):
            raise AssertionError(f"virtual4 transport failures={int(total.sum())}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
