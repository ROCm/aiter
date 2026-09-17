# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Dest-sharded chunked TP all-gather, dispatch-style ordered sends.

Each producer CTA owns one peer. Local rows go out in consecutive chunks
``[c*S, (c+1)*S)`` at ``rank * m_local + t``. After a chunk, the last CTA
for that destination ``atomic_add`` ``chunk_ready[c]`` on the peer.
Consumers wait ``chunk_ready[c] == epoch * npes`` (monotonic, like EP).
A planner CTA bumps the epoch and does an in-kernel ``launch_ready``
handshake so CUDA Graph replay does not need host zero + NCCL.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr

from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .dispatch import _copy_token_row
from .tp_incremental_payload import _copy_scale_row

_HERE = os.path.dirname(os.path.abspath(__file__))


def _register_chunk_source_dir():
    try:
        from flydsl.compiler.jit_function import EXTRA_SOURCE_DIRS
    except ImportError:
        EXTRA_SOURCE_DIRS = None
    if EXTRA_SOURCE_DIRS is not None and _HERE not in EXTRA_SOURCE_DIRS:
        EXTRA_SOURCE_DIRS.append(_HERE)
    cur = os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS", "")
    parts = [p for p in cur.split(":") if p]
    if _HERE not in parts:
        parts.append(_HERE)
        os.environ["FLYDSL_EXTRA_SOURCE_DIRS"] = ":".join(parts)


_register_chunk_source_dir()


def wait_chunk_payload(ready_base_i64, chunk_i32, expected_i32):
    """Spin until ``chunk_ready[chunk] == expected`` (agent scope)."""
    addr = fx.Int64(ready_base_i64) + fx.Int64(chunk_i32) * fx.Int64(4)
    comm_ops.spin_until_eq_i32(addr, expected_i32)


# fmt: off
@flyc.jit
def begin_tp_chunk_epoch(
    *, npes, rank, tid, is_owner, gate_epoch, addr_epoch_gate, addr_launch_ready,
    addr_p2p_launch_ready, addr_work_cursor, zero_work_cursor, addr_local_chunk_done,
    num_done_flags, total_threads,
):
# fmt: on
    if is_owner != fx.Int32(0):
        if tid < fx.Int32(npes):
            comm_ops.fence_system_release()
            crfa = buffer_ops.create_buffer_resource_from_addr
            peer = (tid + rank) % fx.Int32(npes)
            remote = buffer_ops.buffer_load(
                crfa(addr_p2p_launch_ready), peer, vec_width=1, dtype=fx.Int64
            )
            comm_ops.store_i32_system(fx.Int64(remote), rank, gate_epoch)
            comm_ops.spin_until_gt_i32(
                fx.Int64(addr_launch_ready) + fx.Int64(peer) * fx.Int64(4),
                gate_epoch - fx.Int32(1),
            )
            comm_ops.fence_system_acquire()
        fx.barrier()
        if const_expr(zero_work_cursor):
            if tid == fx.Int32(0):
                comm_ops.store_i32_system(addr_work_cursor, fx.Int32(0), fx.Int32(0))
        for flag in range(tid, num_done_flags, fx.Int32(total_threads)):
            comm_ops.store_i32_system(addr_local_chunk_done, flag, fx.Int32(0))
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.store_i32_system(addr_epoch_gate, fx.Int32(0), gate_epoch)
        fx.barrier()
    else:
        if tid == fx.Int32(0):
            comm_ops.spin_until_eq_i32(addr_epoch_gate, gate_epoch)
            comm_ops.fence_agent_acquire()
        fx.barrier()


# fmt: off
@flyc.jit
def emit_tp_chunk_payload(
    *, npes, rank, model_dim, row_major, producer_slot, num_producers, num_waves,
    chunk_rows, m_local, addr_in_tok, addr_in_sc, addr_p2p_rx, addr_p2p_sc,
    addr_p2p_chunk_ready, addr_local_chunk_done,
):
# fmt: on
    """Push this rank's rows to one peer, signaling after each consecutive chunk."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid // fx.Int32(64)
    blocks_per_dest = num_producers // npes
    row_bytes = model_dim
    scale_bytes = model_dim // 32
    row_i32 = row_bytes // 4
    scale_i32 = scale_bytes // 4
    row_safe_end = (row_i32 // 512) * 512
    destination = producer_slot // fx.Int32(blocks_per_dest)
    sub = producer_slot - destination * fx.Int32(blocks_per_dest)
    r_p2p_rx = crfa(addr_p2p_rx)
    r_p2p_sc = crfa(addr_p2p_sc)
    r_p2p_ready = crfa(addr_p2p_chunk_ready)
    peer_x = buffer_ops.buffer_load(r_p2p_rx, destination, vec_width=1, dtype=fx.Int64)
    peer_s = buffer_ops.buffer_load(r_p2p_sc, destination, vec_width=1, dtype=fx.Int64)
    peer_ready = buffer_ops.buffer_load(
        r_p2p_ready, destination, vec_width=1, dtype=fx.Int64
    )
    row0 = sub + warp * fx.Int32(blocks_per_dest)
    row_stride = fx.Int32(blocks_per_dest * num_waves)
    chunk_h = fx.Int32(chunk_rows)
    num_chunks = (m_local + chunk_h - fx.Int32(1)) // chunk_h
    for chunk in range(num_chunks):
        begin = chunk * chunk_h
        end = begin + chunk_h
        end = (end < m_local).select(end, m_local)
        for row in range(begin + row0, end, row_stride):
            if const_expr(row_major):
                dest_row = row * fx.Int32(npes) + rank
            else:
                dest_row = rank * m_local + row
            _copy_token_row(
                crfa(addr_in_tok + fx.Int64(row) * fx.Int64(row_bytes)),
                crfa(fx.Int64(peer_x) + fx.Int64(dest_row) * fx.Int64(row_bytes)),
                lane,
                fz_safe_end_i32=row_safe_end,
                fz_n_i32=row_i32,
            )
            _copy_scale_row(
                addr_in_sc + fx.Int64(row) * fx.Int64(scale_bytes),
                fx.Int64(peer_s) + fx.Int64(dest_row) * fx.Int64(scale_bytes),
                lane,
                scale_i32,
            )
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
            done_idx = destination * num_chunks + chunk
            finished = fx.Int32(
                comm_ops.atomic_add_agent(
                    addr_local_chunk_done + fx.Int64(done_idx) * fx.Int64(4),
                    fx.Int32(1),
                )
            )
            if finished == fx.Int32(blocks_per_dest - 1):
                comm_ops.fence_agent_acquire()
                comm_ops.fence_system_release()
                comm_ops.atomic_add_system(
                    fx.Int64(peer_ready) + fx.Int64(chunk) * fx.Int64(4),
                    fx.Int32(1),
                )
        fx.barrier()


DEFAULT_PRODUCER_BLOCKS = 32
DEFAULT_NUM_WAVES = 4


# fmt: off
@functools.cache
def compile_tp_chunk_push(
    *, npes: int, model_dim: int, row_major: bool, num_producers: int,
    num_waves: int = DEFAULT_NUM_WAVES,
):
    # fmt: on
    """Dest-sharded chunked AG. ``chunk_rows`` is a runtime arg so size can be swept."""
    if int(num_producers) % int(npes):
        raise ValueError(
            f"num_producers={num_producers} must be divisible by npes={npes}"
        )
    num_waves = int(num_waves)
    assert num_waves >= 1
    total_threads = num_waves * 64
    kernel_name = (
        f"tp_chunk_gather_p{npes}_h{model_dim}"
        f"_rm{int(row_major)}_np{num_producers}_w{num_waves}_epoch"
    )

    @fx.struct
    class EpochLds:
        pool: fx.Array[fx.Int8, 16, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[total_threads, 1, 1])
    def kernel(
        local_x: fx.Tensor,
        local_scale: fx.Tensor,
        p2p_rx: fx.Tensor,
        p2p_scale: fx.Tensor,
        p2p_chunk_ready: fx.Tensor,
        p2p_launch_ready: fx.Tensor,
        chunk_ready: fx.Tensor,
        launch_ready: fx.Tensor,
        local_chunk_done: fx.Tensor,
        epoch_gate: fx.Tensor,
        entry_count: fx.Tensor,
        rank: fx.Int32,
        m_local: fx.Int32,
        chunk_rows: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(EpochLds).peek()
        ticket_scratch = fx.recast_iter(fx.Int64, lds.pool.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(
                    fx.Int64(fx.ptrtoint(fx.get_iter(entry_count))), fx.Int64(1)
                )
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        gate_epoch = fx.Int32(ticket64 // fx.Int64(num_producers) + fx.Int64(1))
        payload_expected = gate_epoch * fx.Int32(npes)
        chunk_h = chunk_rows
        num_chunks = (m_local + chunk_h - fx.Int32(1)) // chunk_h
        begin_tp_chunk_epoch(
            npes=npes,
            rank=rank,
            tid=tid,
            is_owner=(fx.Int32(fx.block_idx.x) == fx.Int32(0)).select(
                fx.Int32(1), fx.Int32(0)
            ),
            gate_epoch=gate_epoch,
            addr_epoch_gate=fx.Int64(fx.ptrtoint(fx.get_iter(epoch_gate))),
            addr_launch_ready=fx.Int64(fx.ptrtoint(fx.get_iter(launch_ready))),
            addr_p2p_launch_ready=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_launch_ready))),
            addr_work_cursor=fx.Int64(0),
            zero_work_cursor=False,
            addr_local_chunk_done=fx.Int64(fx.ptrtoint(fx.get_iter(local_chunk_done))),
            num_done_flags=fx.Int32(npes) * num_chunks,
            total_threads=total_threads,
        )
        emit_tp_chunk_payload(
            npes=npes,
            rank=rank,
            model_dim=model_dim,
            row_major=row_major,
            producer_slot=fx.Int32(fx.block_idx.x),
            num_producers=num_producers,
            num_waves=num_waves,
            chunk_rows=chunk_rows,
            m_local=m_local,
            addr_in_tok=fx.Int64(fx.ptrtoint(fx.get_iter(local_x))),
            addr_in_sc=fx.Int64(fx.ptrtoint(fx.get_iter(local_scale))),
            addr_p2p_rx=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_rx))),
            addr_p2p_sc=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_scale))),
            addr_p2p_chunk_ready=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_chunk_ready))),
            addr_local_chunk_done=fx.Int64(fx.ptrtoint(fx.get_iter(local_chunk_done))),
        )
        if tid == fx.Int32(0):
            last = num_chunks - fx.Int32(1)
            wait_chunk_payload(
                fx.Int64(fx.ptrtoint(fx.get_iter(chunk_ready))),
                last,
                payload_expected,
            )
            comm_ops.fence_system_acquire()
        fx.barrier()

    @flyc.jit
    def launch(
        local_x: fx.Tensor,
        local_scale: fx.Tensor,
        p2p_rx: fx.Tensor,
        p2p_scale: fx.Tensor,
        p2p_chunk_ready: fx.Tensor,
        p2p_launch_ready: fx.Tensor,
        chunk_ready: fx.Tensor,
        launch_ready: fx.Tensor,
        local_chunk_done: fx.Tensor,
        epoch_gate: fx.Tensor,
        entry_count: fx.Tensor,
        rank: fx.Int32,
        m_local: fx.Int32,
        chunk_rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            local_x,
            local_scale,
            p2p_rx,
            p2p_scale,
            p2p_chunk_ready,
            p2p_launch_ready,
            chunk_ready,
            launch_ready,
            local_chunk_done,
            epoch_gate,
            entry_count,
            rank,
            m_local,
            chunk_rows,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{total_threads},{total_threads}",
            },
        ).launch(
            grid=(num_producers, 1, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch


def run_tp_chunk_push(
    workspace,
    local_x,
    local_scale,
    m_local,
    stream,
    *,
    chunk_rows: int,
    row_major: bool = False,
    num_producers: int | None = None,
    num_waves: int | None = None,
):
    """Chunked dest-sharded AG. Epoch handshake is in-kernel; do not host-zero between graph replays."""
    m_local = int(m_local)
    chunk_rows = int(chunk_rows)
    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")
    if m_local <= 0:
        return workspace
    if m_local > workspace.max_m_local:
        raise ValueError(
            f"m_local={m_local} exceeds workspace max_m_local={workspace.max_m_local}"
        )
    producers = int(num_producers or DEFAULT_PRODUCER_BLOCKS)
    waves = int(num_waves or DEFAULT_NUM_WAVES)
    launch = compile_tp_chunk_push(
        npes=workspace.npes,
        model_dim=workspace.model_dim,
        row_major=bool(row_major),
        num_producers=producers,
        num_waves=waves,
    )
    _run_compiled(
        launch,
        local_x.contiguous().view(torch.uint8),
        local_scale.contiguous().view(torch.uint8),
        workspace.p2p_rx,
        workspace.p2p_scale,
        workspace.p2p_chunk_ready,
        workspace.p2p_launch_ready,
        workspace.chunk_ready,
        workspace.launch_ready,
        workspace.local_chunk_done,
        workspace.epoch_gate,
        workspace.entry_count,
        fx.Int32(workspace.rank),
        fx.Int32(m_local),
        fx.Int32(chunk_rows),
        stream,
    )
    return workspace
