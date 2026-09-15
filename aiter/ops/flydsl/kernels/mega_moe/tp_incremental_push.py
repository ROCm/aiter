# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Bulk dest-sharded TP activation all-gather (Step A).

Each producer CTA owns one peer. Warps stripe local rows into that peer's
dense slab at ``rank * m_local + t``. Handshake is ``npes`` system atomics
after the whole slab, not per token. The fused incremental kernel still uses
``emit_tp_incremental_payload``.
"""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr import const_expr, range_constexpr
from mori.shmem import mori_shmem_create_tensor

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .dispatch import _copy_token_row
from .tp_incremental_payload import _copy_scale_row

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRODUCER_BLOCKS = 32
DEFAULT_NUM_WAVES = 4


def _register_push_source_dir():
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


_register_push_source_dir()


def _p2p_table(tensor, rank, npes, device):
    table = torch.zeros(npes, dtype=torch.int64, device=device)
    for peer in range(npes):
        table[peer] = ms.shmem_ptr_p2p(tensor.data_ptr(), rank, peer)
    return table


class TpIncrementalWorkspace:
    """Symmetric dense A/scale plus per-expert received counters."""

    def __init__(self, *, rank, npes, max_m_local, model_dim, num_experts, device):
        self.rank = int(rank)
        self.npes = int(npes)
        self.max_m_local = int(max_m_local)
        self.model_dim = int(model_dim)
        self.num_experts = int(num_experts)
        n_rows = self.npes * self.max_m_local + 1
        self.rx_u8 = mori_shmem_create_tensor((n_rows, model_dim), torch.uint8)
        self.rx_scale = mori_shmem_create_tensor((n_rows, model_dim // 32), torch.uint8)
        self.received = mori_shmem_create_tensor((num_experts,), torch.int32)
        self.ranks_done = mori_shmem_create_tensor((1,), torch.int32)
        self.local_prod_done = torch.zeros(1, dtype=torch.int32, device=device)
        self.work_cursor = torch.zeros(1, dtype=torch.int32, device=device)
        self.expert0_done = torch.zeros(1, dtype=torch.int32, device=device)
        self.overlap = torch.zeros(1, dtype=torch.int32, device=device)
        self.rx_u8.zero_()
        self.rx_scale.zero_()
        self.received.zero_()
        self.ranks_done.zero_()
        ms.shmem_barrier_all()
        self.p2p_rx = _p2p_table(self.rx_u8, self.rank, self.npes, device)
        self.p2p_scale = _p2p_table(self.rx_scale, self.rank, self.npes, device)
        self.p2p_received = _p2p_table(self.received, self.rank, self.npes, device)
        self.p2p_ranks_done = _p2p_table(self.ranks_done, self.rank, self.npes, device)

    def zero_handshake(self):
        """Clear per-replay flags. Caller must cross-rank sync before producers."""
        self.received.zero_()
        self.ranks_done.zero_()
        self.local_prod_done.zero_()
        self.work_cursor.zero_()
        self.expert0_done.zero_()
        self.overlap.zero_()

    @property
    def rx(self):
        return self.rx_u8.view(torch.float8_e4m3fn)

    def reset(self):
        self.rx_u8.zero_()
        self.rx_scale.zero_()
        self.received.zero_()
        self.ranks_done.zero_()
        self.local_prod_done.zero_()
        self.work_cursor.zero_()
        self.expert0_done.zero_()
        self.overlap.zero_()
        ms.shmem_barrier_all()


# fmt: off
@functools.cache
def compile_tp_incremental_push(
    *, npes: int, model_dim: int, row_major: bool, num_producers: int,
    num_waves: int = DEFAULT_NUM_WAVES,
):
    # fmt: on
    if int(num_producers) % int(npes):
        raise ValueError(
            f"num_producers={num_producers} must be divisible by npes={npes}"
        )
    num_waves = int(num_waves)
    assert num_waves >= 1
    total_threads = num_waves * 64
    blocks_per_dest = int(num_producers) // int(npes)
    row_bytes = int(model_dim)
    scale_bytes = int(model_dim) // 32
    row_i32 = row_bytes // 4
    scale_i32 = scale_bytes // 4
    row_safe_end = (row_i32 // 512) * 512
    kernel_name = (
        f"tp_bulk_gather_p{npes}_h{model_dim}"
        f"_rm{int(row_major)}_np{num_producers}_w{num_waves}_sc1"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[total_threads, 1, 1])
    def kernel(
        local_x: fx.Tensor,
        local_scale: fx.Tensor,
        p2p_rx: fx.Tensor,
        p2p_scale: fx.Tensor,
        p2p_ranks_done: fx.Tensor,
        ranks_done: fx.Tensor,
        local_prod_done: fx.Tensor,
        rank: fx.Int32,
        m_local: fx.Int32,
    ):
        crfa = buffer_ops.create_buffer_resource_from_addr
        tid = fx.thread_idx.x
        lane = tid & fx.Int32(63)
        warp = tid // fx.Int32(64)
        producer_slot = fx.Int32(fx.block_idx.x)
        destination = producer_slot // fx.Int32(blocks_per_dest)
        sub = producer_slot - destination * fx.Int32(blocks_per_dest)
        addr_in_tok = fx.Int64(fx.ptrtoint(fx.get_iter(local_x)))
        addr_in_sc = fx.Int64(fx.ptrtoint(fx.get_iter(local_scale)))
        r_p2p_rx = crfa(fx.Int64(fx.ptrtoint(fx.get_iter(p2p_rx))))
        r_p2p_sc = crfa(fx.Int64(fx.ptrtoint(fx.get_iter(p2p_scale))))
        r_p2p_ranks_done = crfa(fx.Int64(fx.ptrtoint(fx.get_iter(p2p_ranks_done))))
        peer_x = buffer_ops.buffer_load(r_p2p_rx, destination, vec_width=1, dtype=fx.Int64)
        peer_s = buffer_ops.buffer_load(r_p2p_sc, destination, vec_width=1, dtype=fx.Int64)
        row0 = sub + warp * fx.Int32(blocks_per_dest)
        row_stride = fx.Int32(blocks_per_dest * num_waves)
        for row in range(row0, m_local, row_stride):
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
            finished = fx.Int32(
                comm_ops.atomic_add_agent(
                    fx.Int64(fx.ptrtoint(fx.get_iter(local_prod_done))), fx.Int32(1)
                )
            )
            if finished == fx.Int32(num_producers - 1):
                for peer in range_constexpr(npes):
                    done_base = buffer_ops.buffer_load(
                        r_p2p_ranks_done, fx.Int32(peer), vec_width=1, dtype=fx.Int64
                    )
                    comm_ops.atomic_add_system(fx.Int64(done_base), fx.Int32(1))
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.spin_until_eq_i32(
                fx.Int64(fx.ptrtoint(fx.get_iter(ranks_done))), fx.Int32(npes)
            )
            comm_ops.fence_system_acquire()
        fx.barrier()

    @flyc.jit
    def launch(
        local_x: fx.Tensor,
        local_scale: fx.Tensor,
        p2p_rx: fx.Tensor,
        p2p_scale: fx.Tensor,
        p2p_ranks_done: fx.Tensor,
        ranks_done: fx.Tensor,
        local_prod_done: fx.Tensor,
        rank: fx.Int32,
        m_local: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            local_x,
            local_scale,
            p2p_rx,
            p2p_scale,
            p2p_ranks_done,
            ranks_done,
            local_prod_done,
            rank,
            m_local,
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


def run_tp_incremental_push(
    workspace: TpIncrementalWorkspace,
    local_x,
    local_scale,
    local_ids,
    m_local,
    stream,
    *,
    row_major: bool = False,
    num_producers: int | None = None,
    num_waves: int | None = None,
    publish_order=None,
):
    """Blit local FP8 rows into every peer's dense rx; wait until all ranks done."""
    del local_ids, publish_order
    m_local = int(m_local)
    if m_local <= 0:
        return workspace
    if m_local > workspace.max_m_local:
        raise ValueError(
            f"m_local={m_local} exceeds workspace max_m_local={workspace.max_m_local}"
        )
    producers = int(num_producers or DEFAULT_PRODUCER_BLOCKS)
    waves = int(num_waves or DEFAULT_NUM_WAVES)
    launch = compile_tp_incremental_push(
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
        workspace.p2p_ranks_done,
        workspace.ranks_done,
        workspace.local_prod_done,
        fx.Int32(workspace.rank),
        fx.Int32(m_local),
        stream,
    )
    return workspace
