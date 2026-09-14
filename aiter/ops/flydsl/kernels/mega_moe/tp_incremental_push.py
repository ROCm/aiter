# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Standalone min-expert TP activation push (Step 3 gather, no GEMM overlap)."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .tp_incremental_payload import emit_tp_incremental_payload
from .tp_incremental_schedule import publish_order_from_topk

PRODUCER_THREADS = 64


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
    *, npes: int, topk: int, model_dim: int, row_major: bool, num_producers: int,
):
    # fmt: on
    kernel_name = (
        f"tp_incr_push_p{npes}_k{topk}_h{model_dim}"
        f"_rm{int(row_major)}_np{num_producers}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[PRODUCER_THREADS, 1, 1])
    def kernel(
        local_x: fx.Tensor,
        local_scale: fx.Tensor,
        local_ids: fx.Tensor,
        publish_order: fx.Tensor,
        p2p_rx: fx.Tensor,
        p2p_scale: fx.Tensor,
        p2p_received: fx.Tensor,
        p2p_ranks_done: fx.Tensor,
        ranks_done: fx.Tensor,
        local_prod_done: fx.Tensor,
        rank: fx.Int32,
        m_local: fx.Int32,
    ):
        tid = fx.thread_idx.x
        emit_tp_incremental_payload(
            npes=npes,
            rank=rank,
            topk=topk,
            model_dim=model_dim,
            row_major=row_major,
            producer_slot=fx.Int32(fx.block_idx.x),
            num_producers=num_producers,
            m_local=m_local,
            addr_in_tok=fx.Int64(fx.ptrtoint(fx.get_iter(local_x))),
            addr_in_sc=fx.Int64(fx.ptrtoint(fx.get_iter(local_scale))),
            addr_ids=fx.Int64(fx.ptrtoint(fx.get_iter(local_ids))),
            addr_order=fx.Int64(fx.ptrtoint(fx.get_iter(publish_order))),
            addr_p2p_rx=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_rx))),
            addr_p2p_sc=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_scale))),
            addr_p2p_received=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_received))),
            addr_p2p_ranks_done=fx.Int64(fx.ptrtoint(fx.get_iter(p2p_ranks_done))),
            addr_local_prod_done=fx.Int64(fx.ptrtoint(fx.get_iter(local_prod_done))),
            skew_rank=fx.Int32(-1),
            skew_split=fx.Int32(0),
            skew_sleeps=fx.Int32(0),
            addr_expert0_done=fx.Int64(fx.ptrtoint(fx.get_iter(local_prod_done))),
            addr_overlap=fx.Int64(fx.ptrtoint(fx.get_iter(local_prod_done))),
        )
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
        local_ids: fx.Tensor,
        publish_order: fx.Tensor,
        p2p_rx: fx.Tensor,
        p2p_scale: fx.Tensor,
        p2p_received: fx.Tensor,
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
            local_ids,
            publish_order,
            p2p_rx,
            p2p_scale,
            p2p_received,
            p2p_ranks_done,
            ranks_done,
            local_prod_done,
            rank,
            m_local,
        ).launch(
            grid=(num_producers, 1, 1),
            block=(PRODUCER_THREADS, 1, 1),
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
):
    """Publish local FP8 rows into every peer's dense rx; wait until all ranks done."""
    m_local = int(m_local)
    if m_local <= 0:
        return workspace
    if m_local > workspace.max_m_local:
        raise ValueError(
            f"m_local={m_local} exceeds workspace max_m_local={workspace.max_m_local}"
        )
    topk = int(local_ids.shape[1])
    producers = int(num_producers or 8)
    order = publish_order_from_topk(local_ids).to(torch.int32).contiguous()
    launch = compile_tp_incremental_push(
        npes=workspace.npes,
        topk=topk,
        model_dim=workspace.model_dim,
        row_major=bool(row_major),
        num_producers=producers,
    )
    _run_compiled(
        launch,
        local_x.contiguous().view(torch.uint8),
        local_scale.contiguous().view(torch.uint8),
        local_ids.to(torch.int32).contiguous(),
        order,
        workspace.p2p_rx,
        workspace.p2p_scale,
        workspace.p2p_received,
        workspace.p2p_ranks_done,
        workspace.ranks_done,
        workspace.local_prod_done,
        fx.Int32(workspace.rank),
        fx.Int32(m_local),
        stream,
    )
    return workspace
