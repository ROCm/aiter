# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Per-expert payload handshake and min-expert TP activation push.

Step 2 host-prefill: ``payload_ready[e] == expected[e]`` before launch.
Step 3 producers push each local token once to every peer's dense row
``rank * m_local + t`` (or row-major for the negative control), then
``atomic_add`` ``received[e]`` for each unique topk expert.
"""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
from .dispatch import _copy_token_row
from .gemm_util import _buffer_load

_HERE = os.path.dirname(os.path.abspath(__file__))


def _register_payload_source_dir():
    """FlyDSL disk cache hashes EXTRA_SOURCE_DIRS, not imported function bodies."""
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


_register_payload_source_dir()


def wait_expert_payload(ready_base_i64, expected_rsrc, expert_i32):
    """Spin until ``payload_ready[expert] == expected[expert]`` (agent scope)."""
    want = _buffer_load(expected_rsrc, expert_i32, fx.Int32)
    addr = fx.Int64(ready_base_i64) + fx.Int64(expert_i32) * fx.Int64(4)
    comm_ops.spin_until_eq_i32(addr, want)


@flyc.jit
def _copy_scale_row(source_addr, destination_addr, lane, n_scale_i32):
    if lane < fx.Int32(n_scale_i32):
        value = buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource_from_addr(source_addr),
            lane,
            vec_width=1,
            dtype=T.i32,
        )
        buffer_ops.buffer_store(
            value,
            buffer_ops.create_buffer_resource_from_addr(destination_addr),
            lane,
        )


# fmt: off
@flyc.jit
def emit_tp_incremental_payload(
    *, npes, rank, topk, model_dim, row_major, producer_slot, num_producers, m_local,
    addr_in_tok, addr_in_sc, addr_ids, addr_order, addr_p2p_rx, addr_p2p_sc,
    addr_p2p_received, addr_p2p_ranks_done, addr_local_prod_done,
):
# fmt: on
    """Push this rank's tokens in min-expert order; each token is sent once."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    n_i32 = model_dim // 4
    safe_end = (n_i32 // 512) * 512
    row_bytes = model_dim
    n_scale_bytes = model_dim // 32
    n_scale_i32 = n_scale_bytes // 4
    r_order = crfa(addr_order)
    r_ids = crfa(addr_ids)
    r_p2p_rx = crfa(addr_p2p_rx)
    r_p2p_sc = crfa(addr_p2p_sc)
    r_p2p_received = crfa(addr_p2p_received)
    r_p2p_ranks_done = crfa(addr_p2p_ranks_done)

    for i in range(producer_slot, m_local, fx.Int32(num_producers)):
        local_row = buffer_ops.buffer_load(r_order, i, vec_width=1, dtype=fx.Int32)
        if const_expr(row_major):
            dest_row = local_row * fx.Int32(npes) + fx.Int32(rank)
        else:
            dest_row = fx.Int32(rank) * m_local + local_row
        src_tok = crfa(addr_in_tok + fx.Int64(local_row) * fx.Int64(row_bytes))
        src_sc = addr_in_sc + fx.Int64(local_row) * fx.Int64(n_scale_bytes)
        for peer in range_constexpr(npes):
            rx_base = buffer_ops.buffer_load(
                r_p2p_rx, fx.Int32(peer), vec_width=1, dtype=fx.Int64
            )
            sc_base = buffer_ops.buffer_load(
                r_p2p_sc, fx.Int32(peer), vec_width=1, dtype=fx.Int64
            )
            dst_tok = crfa(fx.Int64(rx_base) + fx.Int64(dest_row) * fx.Int64(row_bytes))
            dst_sc = fx.Int64(sc_base) + fx.Int64(dest_row) * fx.Int64(n_scale_bytes)
            _copy_token_row(
                src_tok,
                dst_tok,
                lane,
                fz_safe_end_i32=safe_end,
                fz_n_i32=n_i32,
            )
            _copy_scale_row(src_sc, dst_sc, lane, n_scale_i32)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
            for k in range_constexpr(topk):
                ek = buffer_ops.buffer_load(
                    r_ids,
                    local_row * fx.Int32(topk) + fx.Int32(k),
                    vec_width=1,
                    dtype=fx.Int32,
                )
                is_unique = fx.Boolean(True)
                for j in range_constexpr(k):
                    prev = buffer_ops.buffer_load(
                        r_ids,
                        local_row * fx.Int32(topk) + fx.Int32(j),
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                    is_unique = (prev != ek).select(is_unique, fx.Boolean(False))
                if is_unique:
                    for peer in range_constexpr(npes):
                        rec_base = buffer_ops.buffer_load(
                            r_p2p_received, fx.Int32(peer), vec_width=1, dtype=fx.Int64
                        )
                        comm_ops.atomic_add_system(
                            fx.Int64(rec_base) + fx.Int64(ek) * fx.Int64(4),
                            fx.Int32(1),
                        )
        fx.barrier()

    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        finished = fx.Int32(comm_ops.atomic_add_agent(addr_local_prod_done, fx.Int32(1)))
        if finished == fx.Int32(num_producers - 1):
            comm_ops.fence_agent_acquire()
            comm_ops.fence_system_release()
            for peer in range_constexpr(npes):
                done_base = buffer_ops.buffer_load(
                    r_p2p_ranks_done, fx.Int32(peer), vec_width=1, dtype=fx.Int64
                )
                comm_ops.atomic_add_system(fx.Int64(done_base), fx.Int32(1))
    fx.barrier()
