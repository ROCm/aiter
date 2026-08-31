# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One kernel that both gathers the TP activations and runs GEMM1.

This is an assembly of two halves that were each verified on their own:

* the P2P push from ``tp_gather.py``, proven bit-identical to
  ``dist.all_gather_into_tensor`` for m_local 1..256;
* the persistent GEMM1 tile loop from ``tp_gemm1.py``, proven bit-identical to
  the original contiguous loaders fed host-permuted data.

The staging half runs in one of two directions, chosen at compile time by
``pull``:

* ``pull=True`` (default) -- each staging CTA owns a SOURCE rank and reads that
  rank's rows out of its symmetric ``tx`` slab into our own receive slab. Nobody
  writes into anybody else's memory. That removes the ``payload_ready`` protocol
  entirely: ``emit_launch_rendezvous`` already proves every peer entered this
  round, and on a single stream that implies the peer's quantize kernel retired,
  which is exactly the readiness condition. What remains is a device-local
  counter so the GEMM half does not start before this rank's own staging CTAs
  finish. The price is that the quantized rows must live in a Mori symmetric
  buffer -- see ``TPActivationGather.tx_views``.
* ``pull=False`` -- the original push: each staging CTA owns a DESTINATION rank
  and writes our rows into that peer's receive slab, then the last one bumps
  every peer's ``payload_ready``.

Both directions move the same bytes to the same offsets, so the GEMM half sees a
byte-identical A operand and the two are a controlled A/B of nothing but the
trigger side.

The grid is the GEMM's (``num_cu * grid_mult``); the launch ticket taken by
``emit_ticket_and_roles`` decides which CTA owns the round and which ones stage.
Every CTA then waits once for all source ranks and runs its share of the tile
loop. The wait is per CTA, not per tile: under TP every tile's ``sort_block_m``
rows are scattered over all source ranks, so no tile can start early and
per-tile readiness would only add protocol overhead.

Wiring notes that are easy to get wrong:

* LDS is reused, not duplicated. ``emit_ticket_and_roles`` broadcasts the ticket
  through byte 0 of ``SharedStorage.pool``; the GEMM then uses the same pool for
  the A ping/pong buffers and the CShuffle tile. First use finishes before the
  second starts, exactly as MegaMoE does it.
* A and its scale come from the symmetric receive buffer. The host slices
  ``rx_x[parity]`` / ``rx_scale[parity]`` and passes those views, so the kernel's
  ``parity`` scalar is only used by the push side.
* Work is split statically (``for flat in range(block_idx.x, total_work,
  GRID_X)``), the same split ``tp_gemm1.py`` validated. The producer CTAs do the
  push first and then take an equal GEMM share, so they trail slightly. Measure
  before replacing this with an atomic work pool.

Two traps inherited from ``tp_gemm1.py``, unchanged and not up for debate:

* ``always_valid=True`` on the epilogue must stay. ``SiluQuantEpilogue`` reads
  its ``sorted_rsrc`` as a token table indexed by row slot, but what we hand it
  is ``trb_rsrc``, a row-base table indexed by tile. Only ``always_valid=True``
  keeps that read dead.
* ``trb_rsrc`` is built with ``max_size=True``, so there is no hardware bounds
  clamp: the host table must hold at least ``num_m_tiles`` int32 entries.

One behaviour that differs from the standalone GEMM1 test: there, the padding
token id equals ``total_rows - 1`` and clamps onto the zeroed pad row. Here the
symmetric buffer is sized for ``max_tok_per_rank`` while a call only fills
``tp_size * m_local`` rows, so the padding sentinel (``m_global``) usually lands
on a row holding a previous round's activations rather than on the pad row. That
is allocated, initially-zeroed memory, and the MFMA keeps rows independent, so
the stale bytes only ever reach padding output rows -- which stage1 callers mask
off by token id anyway.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.runtime.device import get_rocm_arch

from .. import buffer_ops
from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .collective_sched import copy_row, emit_launch_rendezvous, emit_ticket_and_roles
from .gemm1 import _LdsF32View, do_tile
from .gemm_util import (
    AS2RLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    _buffer_load,
    _make_buffer,
    _make_buffer_from_addr,
)
from .tp_gemm_util import TPAScaleLoader, TPATileLoader

_RESET_COUNTERS = 1  # push_done
_EPOCH_SLOT = 0


# fmt: off
@functools.cache
def compile_tp_fused_stage1(
    *,
    # staging half (tp_gather.py)
    model_dim: int, npes: int, rank: int, producer_blocks: int = 32, slots: int = 2,
    pull: bool = True,
    # GEMM half (tp_gemm1.py)
    inter_dim: int, experts: int, total_rows: int, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, num_cu: int = 256,
    grid_mult: int = 4, swizzle_a: bool = True, pipe_weights: bool = True,
    mfma_amajor: bool = False, async_a_copy: bool = False, waves_per_eu_hint: int = 2,
    swiglu_limit: float = 0.0,
):
    # fmt: on
    """Compile (and cache) the fused push+GEMM1 kernel."""
    arch = get_rocm_arch()
    if not str(arch).startswith("gfx95"):
        raise RuntimeError(f"tp_fused_stage1 targets gfx95x, got {arch}")

    assert num_waves > 1
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % num_waves == 0
    assert (2 * inter_dim) % tile_n == 0
    assert tile_k == 256 and model_dim % tile_k == 0
    assert model_dim % 512 == 0, "copy_row needs 16-byte granularity on both rows"
    assert producer_blocks % npes == 0
    assert 0 <= rank < npes
    assert not (pull and slots != 2), "pull needs double buffering; see TPActivationGather"

    NUM_WAVES = num_waves
    TOTAL_THREADS = NUM_WAVES * 64
    GRID_X = num_cu * grid_mult
    # Ticket 0 is the owner and tickets 1..producer_blocks are the producers, so
    # the grid must be able to hand out that many distinct roles.
    assert producer_blocks + 1 <= GRID_X, (
        f"producer_blocks={producer_blocks} + 1 exceeds grid {GRID_X}"
    )

    # --- staging half ------------------------------------------------------
    ROW_BYTES = model_dim
    SCALE_BYTES = model_dim // 32
    ROW_I32 = ROW_BYTES // 4
    SCALE_I32 = SCALE_BYTES // 4
    ROW_SAFE_END = (ROW_I32 // 512) * 512
    SCALE_SAFE_END = (SCALE_I32 // 512) * 512
    # Staging CTAs are partitioned by peer: under push a group writes to one
    # destination rank, under pull it reads from one source rank. Same count,
    # same split, opposite direction.
    BLOCKS_PER_PEER = producer_blocks // npes

    # --- GEMM half ---------------------------------------------------------
    n_per_wave = tile_n // NUM_WAVES
    N_TILES = (2 * inter_dim) // tile_n
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0
    A_K_STEP_BYTES = tile_k
    K_ITERS = model_dim // tile_k
    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    PAD_ROW = total_rows - 1  # last row of the symmetric slab is the zeroed pad row

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    kernel_name = (
        f"tp_fused_stage1_{'pull' if pull else 'push'}"
        f"_d{model_dim}_i{inter_dim}_n{npes}_pb{producer_blocks}"
        f"_t{sort_block_m}x{tile_n}x{tile_k}_w{NUM_WAVES}_g{GRID_X}"
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def fused_kernel(
        # outputs
        out: fx.Tensor, out_scale: fx.Tensor,
        # A side: this rank's parity slab of the symmetric receive buffers
        x: fx.Tensor, scale_x: fx.Tensor,
        # weights
        w: fx.Tensor, scale_w: fx.Tensor,
        # plan
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor,
        sorted_token_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        # push only: this rank's locally quantized input, pushed to every peer
        addr_x_q: fx.Int64, addr_x_scale: fx.Int64,
        # push only: p2p tables of every peer's receive slab
        addr_p2p_rx_x: fx.Int64, addr_p2p_rx_scale: fx.Int64,
        addr_p2p_payload_ready: fx.Int64,
        # pull only: p2p tables of every peer's source slab, and this rank's
        # own receive slab for the parity in play (the pull destination)
        addr_p2p_tx_x: fx.Int64, addr_p2p_tx_scale: fx.Int64,
        addr_rx_x: fx.Int64, addr_rx_scale: fx.Int64,
        # both modes
        addr_p2p_launch_ready: fx.Int64,
        # local flags / counters
        addr_payload_ready: fx.Int64, addr_launch_ready: fx.Int64,
        addr_epoch_gate: fx.Int64, addr_entry_count: fx.Int64, addr_reset: fx.Int64,
        # scalars
        m_local: fx.Int32, parity: fx.Int32, expected: fx.Int32,
        launch_epoch: fx.Int32, tokens: fx.Int32,
        x_slab_bytes: fx.Int32, scale_slab_bytes: fx.Int32,
        tx_slab_bytes: fx.Int32, tx_scale_slab_bytes: fx.Int32,
    ):
        # fmt: on
        tid = fx.thread_idx.x
        lane = tid & fx.Int32(63)
        warp = tid // fx.Int32(64)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # ---- role assignment, then the cross-rank launch handshake ---------
        gate_addr, gate_epoch, is_owner, is_producer, producer_slot = (
            emit_ticket_and_roles(
                tid=tid, lds_scratch=lds.pool, a_entry_count=addr_entry_count,
                a_epoch_gate=addr_epoch_gate, epoch_slot=_EPOCH_SLOT,
                launch_grid_x=GRID_X, producer_blocks=producer_blocks)
        )

        emit_launch_rendezvous(
            tid=tid, is_owner=is_owner, p_launch_ready=addr_p2p_launch_ready,
            a_launch_ready=addr_launch_ready, a_reset_counters=addr_reset,
            reset_count=_RESET_COUNTERS, gate_addr=gate_addr, gate_epoch=gate_epoch,
            launch_epoch=launch_epoch, npes=npes, rank=rank)

        # ---- stage all m_global rows into this rank's receive slab ---------
        if const_expr(slots == 1):
            slab = fx.Int32(0)
        else:
            slab = parity
        x_base_off = fx.Int64(slab) * fx.Int64(x_slab_bytes)
        s_base_off = fx.Int64(slab) * fx.Int64(scale_slab_bytes)
        tx_base_off = fx.Int64(slab) * fx.Int64(tx_slab_bytes)
        ts_base_off = fx.Int64(slab) * fx.Int64(tx_scale_slab_bytes)

        if const_expr(pull):
            # Pull: each staging CTA owns one SOURCE rank and copies that rank's
            # rows out of its symmetric tx slab into our own receive slab. The
            # rows land at exactly the same offsets push would have written, so
            # the GEMM half sees a byte-identical A operand either way.
            #
            # No readiness flag is needed for the data. emit_launch_rendezvous
            # above already proved every peer entered this round, and on a single
            # stream that implies the peer's quantize kernel -- which writes tx --
            # retired. Its fence_system_release before publishing the epoch wrote
            # that data back out of the peer's L2, so the acquire below is enough
            # to see it. That is the whole reason pull can drop payload_ready and
            # the npes system atomics that maintained it.
            if is_producer:
                comm_ops.fence_system_acquire()
                source = producer_slot // fx.Int32(BLOCKS_PER_PEER)
                sub = producer_slot - source * fx.Int32(BLOCKS_PER_PEER)
                tx_table = _make_buffer_from_addr(addr_p2p_tx_x, fx.Int64)
                ts_table = _make_buffer_from_addr(addr_p2p_tx_scale, fx.Int64)
                peer_x = _buffer_load(tx_table, source, fx.Int64) + tx_base_off
                peer_s = _buffer_load(ts_table, source, fx.Int64) + ts_base_off
                dest_base = source * m_local
                row0 = sub + warp * fx.Int32(BLOCKS_PER_PEER)
                row_stride = fx.Int32(BLOCKS_PER_PEER * num_waves)
                for row in range(row0, m_local, row_stride):
                    dest_row = dest_base + row
                    # fmt: off
                    copy_row(
                        buffer_ops.create_buffer_resource_from_addr(peer_x + fx.Int64(row) * fx.Int64(ROW_BYTES)),
                        buffer_ops.create_buffer_resource_from_addr(addr_rx_x + fx.Int64(dest_row) * fx.Int64(ROW_BYTES)),
                        lane, safe_end_i32=ROW_SAFE_END, n_i32=ROW_I32)
                    copy_row(
                        buffer_ops.create_buffer_resource_from_addr(peer_s + fx.Int64(row) * fx.Int64(SCALE_BYTES)),
                        buffer_ops.create_buffer_resource_from_addr(addr_rx_scale + fx.Int64(dest_row) * fx.Int64(SCALE_BYTES)),
                        lane, safe_end_i32=SCALE_SAFE_END, n_i32=SCALE_I32)
                    # fmt: on
                fx.rocdl.s_waitcnt(0)
                fx.barrier()
                if tid == fx.Int32(0):
                    comm_ops.fence_agent_release()
                    comm_ops.atomic_add_agent(addr_reset, fx.Int32(1))

            # Every CTA waits for this rank's staging to finish. The writers are
            # all on this device now, so the counter and the fence are agent
            # scope; push has to use system scope because its writers are remote.
            if tid == fx.Int32(0):
                mori_shmem.int32_wait_until_equals(
                    addr_reset, fx.Int32(producer_blocks))
                comm_ops.fence_agent_acquire()
            fx.barrier()
        else:
            if is_producer:
                destination = producer_slot // fx.Int32(BLOCKS_PER_PEER)
                sub = producer_slot - destination * fx.Int32(BLOCKS_PER_PEER)
                rx_table = _make_buffer_from_addr(addr_p2p_rx_x, fx.Int64)
                sc_table = _make_buffer_from_addr(addr_p2p_rx_scale, fx.Int64)
                peer_x = _buffer_load(rx_table, destination, fx.Int64) + x_base_off
                peer_s = _buffer_load(sc_table, destination, fx.Int64) + s_base_off
                dest_base = fx.Int32(rank) * m_local
                row0 = sub + warp * fx.Int32(BLOCKS_PER_PEER)
                row_stride = fx.Int32(BLOCKS_PER_PEER * num_waves)
                for row in range(row0, m_local, row_stride):
                    dest_row = dest_base + row
                    # fmt: off
                    copy_row(
                        buffer_ops.create_buffer_resource_from_addr(addr_x_q + fx.Int64(row) * fx.Int64(ROW_BYTES)),
                        buffer_ops.create_buffer_resource_from_addr(peer_x + fx.Int64(dest_row) * fx.Int64(ROW_BYTES)),
                        lane, safe_end_i32=ROW_SAFE_END, n_i32=ROW_I32)
                    copy_row(
                        buffer_ops.create_buffer_resource_from_addr(addr_x_scale + fx.Int64(row) * fx.Int64(SCALE_BYTES)),
                        buffer_ops.create_buffer_resource_from_addr(peer_s + fx.Int64(dest_row) * fx.Int64(SCALE_BYTES)),
                        lane, safe_end_i32=SCALE_SAFE_END, n_i32=SCALE_I32)
                    # fmt: on
                fx.rocdl.s_waitcnt(0)
                fx.barrier()
                # Last producer block on this rank publishes once to every peer, so
                # payload_ready advances by exactly npes per launch -- the step the
                # host's _expected_for() assumes.
                if tid == fx.Int32(0):
                    comm_ops.fence_system_release()
                    done = fx.Int32(comm_ops.atomic_add_agent(addr_reset, fx.Int32(1)))
                    if done == fx.Int32(producer_blocks - 1):
                        pr_table = _make_buffer_from_addr(addr_p2p_payload_ready, fx.Int64)
                        for pe in range_constexpr(npes):
                            remote = _buffer_load(pr_table, fx.Int32(pe), fx.Int64)
                            comm_ops.atomic_add_system(
                                remote + fx.Int64(parity) * fx.Int64(4), fx.Int32(1))

            # ---- every CTA waits once for all source ranks -----------------
            if tid == fx.Int32(0):
                mori_shmem.int32_wait_until_equals(
                    addr_payload_ready + fx.Int64(parity) * fx.Int64(4), expected)
                comm_ops.fence_system_acquire()
            fx.barrier()

        # ---- GEMM1 ---------------------------------------------------------
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        tok_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        os_rsrc = _make_buffer(
            out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes
        )

        sched = TileScheduler(
            expert_rsrc=expert_rsrc, inter_dim=inter_dim, expert_offset=0
        )
        n_wave_base = wave_id * fx.Int32(n_per_wave)

        # fmt: off
        a_gather = TPATileLoader(row_bytes=model_dim, sort_block_m=sort_block_m,
            k_step_bytes=A_K_STEP_BYTES, total_threads=TOTAL_THREADS,
            swizzle=swizzle_a, x_tensor=x, tok_rsrc=tok_rsrc, pad_row=PAD_ROW,
            total_rows=total_rows, async_copy=async_a_copy)
        a_scale = TPAScaleLoader(scale_rsrc=sx_rsrc, m_repeat=M_REPEAT,
            model_dim=model_dim, sort_block_m=sort_block_m,
            total_threads=TOTAL_THREADS, tok_rsrc=tok_rsrc, pad_row=PAD_ROW)
        # fmt: on

        a_s2r = AS2RLoader(k_step_bytes=A_K_STEP_BYTES, swizzle=swizzle_a)
        b_loader = BWeightLoader(
            w_rsrc=w_rsrc, num_acc_n=NUM_ACC_N, model_dim=model_dim, cache_modifier=0
        )
        b_scale = BScaleLoader(
            scale_rsrc=sw_rsrc, num_acc_n=NUM_ACC_N, model_dim=model_dim
        )
        mfma = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N)
        # fmt: off
        epi = SiluQuantEpilogue(out_rsrc=None, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc,
            tokens=0, inter_dim=inter_dim, m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N,
            sort_block_m=sort_block_m, tile_n=tile_n, num_waves=NUM_WAVES, lds_out=c_tile,
            swiglu_limit=swiglu_limit, always_valid=True, out_tensor=out)
        # fmt: on

        # A closure, not an inline body: the scf.for rewriter treats every name
        # assigned in a loop body as loop-carried state, and the loader/scheduler
        # objects have no IR representation.
        def do_scheduled_tile(flat):
            m_tile = flat // fx.Int32(N_TILES)
            n_tile = flat - m_tile * fx.Int32(N_TILES)
            n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
            expert = sched.expert_of(m_tile)
            # fmt: off
            do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader,
                b_scale, a_scale, mfma, epi, a_buf, a_scale_lds, a_lds_i32,
                K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
                mfma_amajor, async_a_copy, trb_rsrc)
            # fmt: on

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = (num_valid + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)
        total_work = num_m_tiles * fx.Int32(N_TILES)

        for flat in range(fx.block_idx.x, total_work, fx.Int32(GRID_X)):
            do_scheduled_tile(flat)

    # fmt: off
    @flyc.jit
    def launch(
        out: fx.Tensor, out_scale: fx.Tensor,
        x: fx.Tensor, scale_x: fx.Tensor,
        w: fx.Tensor, scale_w: fx.Tensor,
        tile_row_base: fx.Tensor, expert_ids: fx.Tensor,
        sorted_token_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        addr_x_q: fx.Int64, addr_x_scale: fx.Int64,
        addr_p2p_rx_x: fx.Int64, addr_p2p_rx_scale: fx.Int64,
        addr_p2p_payload_ready: fx.Int64,
        addr_p2p_tx_x: fx.Int64, addr_p2p_tx_scale: fx.Int64,
        addr_rx_x: fx.Int64, addr_rx_scale: fx.Int64,
        addr_p2p_launch_ready: fx.Int64,
        addr_payload_ready: fx.Int64, addr_launch_ready: fx.Int64,
        addr_epoch_gate: fx.Int64, addr_entry_count: fx.Int64, addr_reset: fx.Int64,
        m_local: fx.Int32, parity: fx.Int32, expected: fx.Int32,
        launch_epoch: fx.Int32, tokens: fx.Int32,
        x_slab_bytes: fx.Int32, scale_slab_bytes: fx.Int32,
        tx_slab_bytes: fx.Int32, tx_scale_slab_bytes: fx.Int32,
        stream: fx.Stream,
    ):
        fused_kernel(
            out, out_scale, x, scale_x, w, scale_w,
            tile_row_base, expert_ids, sorted_token_ids, num_valid_ids,
            addr_x_q, addr_x_scale,
            addr_p2p_rx_x, addr_p2p_rx_scale, addr_p2p_payload_ready,
            addr_p2p_tx_x, addr_p2p_tx_scale, addr_rx_x, addr_rx_scale,
            addr_p2p_launch_ready,
            addr_payload_ready, addr_launch_ready,
            addr_epoch_gate, addr_entry_count, addr_reset,
            m_local, parity, expected, launch_epoch, tokens,
            x_slab_bytes, scale_slab_bytes,
            tx_slab_bytes, tx_scale_slab_bytes,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(GRID_X, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)
    # fmt: on

    return launch


class TPFusedStage1Runner:
    """Host side of the fused kernel: owns the plan scratch and the round counter.

    The symmetric buffers, the p2p tables and the parity bookkeeping all live in
    the ``TPActivationGather`` handed in at construction; this class only adds the
    per-call scratch the GEMM half needs and drives one launch.
    """

    def __init__(self, *, gather, w, w_scale, model_dim, inter_dim, experts,
                 sort_block_m=32, swiglu_limit=0.0, pull=True, **cfg):
        self.gather = gather
        if pull and not gather.enable_pull:
            raise ValueError(
                "pull=True needs the symmetric source slab; construct "
                "TPActivationGather with enable_pull=True"
            )
        self.pull = bool(pull)
        # Weights are constant across calls; keep the uint8 views the kernel wants.
        self.w = w.view(torch.uint8)
        self.w_scale = w_scale.view(torch.uint8)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.sort_block_m = int(sort_block_m)
        self.swiglu_limit = float(swiglu_limit)
        self.cfg = dict(cfg)
        # tile_row_base is just arange(num_m_tiles) * sort_block_m, so one table
        # per tile count is reusable across calls.
        self._trb_cache = {}

    def _tile_row_base(self, n_tiles, device):
        key = int(n_tiles)
        trb = self._trb_cache.get(key)
        if trb is None:
            trb = torch.arange(key, dtype=torch.int32, device=device) * self.sort_block_m
            self._trb_cache[key] = trb
        return trb

    # fmt: off
    def run(self, *, x_q, x_scale, sorted_token_ids, expert_ids, num_valid_ids,
            max_sorted, stream=None):
        # fmt: on
        """Push, wait, and run GEMM1 in one launch. Returns (payload, out_scale)."""
        g = self.gather
        m_local = g._validate(x_q, x_scale)
        sbm = self.sort_block_m
        if int(max_sorted) % sbm != 0:
            raise ValueError(f"max_sorted={max_sorted} must be a multiple of {sbm}")

        launch = compile_tp_fused_stage1(
            model_dim=self.model_dim,
            npes=g.tp_size,
            rank=g.tp_rank,
            producer_blocks=g.producer_blocks,
            slots=g.slots,
            pull=self.pull,
            inter_dim=self.inter_dim,
            experts=self.experts,
            total_rows=g.rows,
            sort_block_m=sbm,
            num_waves=g.num_waves,
            swiglu_limit=self.swiglu_limit,
            **self.cfg,
        )

        dev = x_q.device
        out = torch.empty(
            (int(max_sorted), self.inter_dim), dtype=torch.float8_e4m3fn, device=dev
        )
        prows = ((int(max_sorted) + 255) // 256) * 256
        pcols = (((self.inter_dim // 32) + 7) // 8) * 8
        # +8192 mirrors the num_records slack the kernel gives os_rsrc, so the
        # buffer descriptor can never describe more memory than actually exists.
        out_scale_flat = torch.zeros(prows * pcols + 8192, dtype=torch.uint8, device=dev)

        # trb_rsrc uses max_size=True: no hardware bounds clamp, so the table has
        # to cover every tile the kernel can visit (+ slack).
        trb = self._tile_row_base(int(max_sorted) // sbm + 64, dev)

        parity = g.current_parity()
        expected = g._expected_for(parity)
        launch_epoch = g._round + 1
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())

        if self.pull:
            # Peers read this rank's rows out of the symmetric tx slab, so they
            # have to be there. A caller that quantized straight into
            # gather.tx_views() pays nothing here; anything else gets copied.
            g.stage_source(x_q, x_scale, parity)
            addr_p2p_tx_x = fx.Int64(g.p2p_tx_x.data_ptr())
            addr_p2p_tx_scale = fx.Int64(g.p2p_tx_scale.data_ptr())
        else:
            addr_p2p_tx_x = fx.Int64(0)
            addr_p2p_tx_scale = fx.Int64(0)

        # fmt: off
        _run_compiled(
            launch,
            out, out_scale_flat,
            g.rx_x[parity], g.rx_scale[parity],
            self.w, self.w_scale,
            trb, expert_ids, sorted_token_ids, num_valid_ids,
            fx.Int64(x_q.data_ptr()), fx.Int64(x_scale.data_ptr()),
            fx.Int64(g.p2p_rx_x.data_ptr()), fx.Int64(g.p2p_rx_scale.data_ptr()),
            fx.Int64(g.p2p_payload_ready.data_ptr()),
            addr_p2p_tx_x, addr_p2p_tx_scale,
            fx.Int64(g.rx_x[parity].data_ptr()),
            fx.Int64(g.rx_scale[parity].data_ptr()),
            fx.Int64(g.p2p_launch_ready.data_ptr()),
            fx.Int64(g.payload_ready.data_ptr()), fx.Int64(g.launch_ready.data_ptr()),
            fx.Int64(g.epoch_gate.data_ptr()), fx.Int64(g.entry_count.data_ptr()),
            fx.Int64(g.reset_counters.data_ptr()),
            fx.Int32(m_local), fx.Int32(parity), fx.Int32(expected),
            fx.Int32(launch_epoch), fx.Int32(int(max_sorted)),
            fx.Int32(g.slab_bytes("x")), fx.Int32(g.slab_bytes("scale")),
            fx.Int32(g.slab_bytes("tx_x") if self.pull else 0),
            fx.Int32(g.slab_bytes("tx_scale") if self.pull else 0),
            stream,
        )
        # fmt: on
        g._round += 1
        return out, out_scale_flat[: prows * pcols].view(prows, pcols)
