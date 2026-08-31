# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""In-kernel P2P activation all-gather for tensor-parallel MoE.

Replaces ``dist.all_gather_into_tensor`` with plain stores into peer memory from
inside a kernel. On one node the 8 GPUs are XGMI-connected and peer memory is
directly addressable, so the transfer costs no separate launch and no collective
rendezvous -- the ~17us fixed cost per collective measured in phase 1 goes away.

Result is bit-identical to the NCCL all-gather: destination row is
``tp_rank * m_local + local_row``, i.e. the same rank-major layout.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
import torch
from flydsl.expr import const_expr, range_constexpr

import mori.shmem as ms

from .. import buffer_ops
from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .collective_sched import copy_row, emit_launch_rendezvous, emit_ticket_and_roles
from .gemm_util import _buffer_load, _make_buffer_from_addr

# ``copy_row`` carries its own ``@flyc.jit``; the two ``emit_*`` helpers do not,
# so their ``if <device value>:`` statements would reach Python's ``__bool__``
# instead of being rewritten into device branches. Wrapping applies the same AST

_SUPPORTED_TP = (4, 8)
_RESET_COUNTERS = 1  # push_done


def _shmem_looks_initialised():
    """Best-effort check that Mori SHMEM is up on the current device.

    Reads Mori's own Python-level bookkeeping: ``_ensure_shmem_module()`` is the
    common entry point of all three init paths and records the HIP device id in
    ``_shmem_module_loaded_gpus``; ``shmem_finalize()`` removes it.

    This exists because calling ``ms.shmem_npes()`` on an uninitialised Mori
    SEGFAULTs the process -- it dereferences an unset GpuStates in C++, so there
    is nothing for Python to catch. Measured on this box: exit code 139, no
    output. A best-effort guard that turns that into a readable error is worth
    depending on two private names for.

    Returns None if Mori's internals no longer look like this, in which case the
    caller should skip the check rather than fail: this is a foot-gun guard, not
    a correctness mechanism.
    """
    try:
        import mori.shmem.api as _api

        loaded = getattr(_api, "_shmem_module_loaded_gpus", None)
        current = getattr(_api, "_current_hip_device", None)
        if loaded is None or current is None:
            return None
        return current() in loaded
    except Exception:  # noqa: BLE001 - never let the guard itself break construction
        return None


def _p2p_table(t, rank, npes, device):
    """i64[npes] table of intra-node P2P pointers to ``t`` on every peer."""
    table = torch.zeros(npes, dtype=torch.int64, device=device)
    for pe in range(npes):
        table[pe] = ms.shmem_ptr_p2p(t.data_ptr(), rank, pe)
    return table


# fmt: off
@functools.cache
def compile_tp_gather(*, model_dim: int, npes: int, rank: int, producer_blocks: int,
        num_waves: int = 4, slots: int = 2):
    """Push this rank's quantized rows into every peer's symmetric receive buffer."""
    TOTAL_THREADS = num_waves * 64
    ROW_BYTES = model_dim
    SCALE_BYTES = model_dim // 32
    ROW_I32 = ROW_BYTES // 4
    SCALE_I32 = SCALE_BYTES // 4
    ROW_SAFE_END = (ROW_I32 // 512) * 512
    SCALE_SAFE_END = (SCALE_I32 // 512) * 512
    BLOCKS_PER_DEST = producer_blocks // npes
    LAUNCH_GRID_X = 1 + producer_blocks
    EPOCH_SLOT = 0

    kernel_name = (
        f"tp_gather_d{model_dim}_n{npes}_pb{producer_blocks}"
        f"_w{num_waves}_s{slots}"
    )

    # Declared in the factory scope, not inside the kernel body -- this mirrors
    # mega_moe_stage1.py:160-163, where SharedStorage sits outside @flyc.kernel.
    @fx.struct
    class SharedStorage:
        scratch: fx.Array[fx.Int8, 64, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        addr_x_q: fx.Int64, addr_x_scale: fx.Int64,
        addr_p2p_rx_x: fx.Int64, addr_p2p_rx_scale: fx.Int64,
        addr_p2p_payload_ready: fx.Int64, addr_p2p_launch_ready: fx.Int64,
        addr_payload_ready: fx.Int64, addr_launch_ready: fx.Int64,
        addr_epoch_gate: fx.Int64, addr_entry_count: fx.Int64,
        addr_reset: fx.Int64,
        m_local: fx.Int32, x_slab_bytes: fx.Int32, scale_slab_bytes: fx.Int32,
        parity: fx.Int32, expected: fx.Int32, launch_epoch: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid & fx.Int32(63)
        warp = tid // fx.Int32(64)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        gate_addr, gate_epoch, is_owner, is_producer, producer_slot = emit_ticket_and_roles(
            tid=tid, lds_scratch=lds.scratch, a_entry_count=addr_entry_count,
            a_epoch_gate=addr_epoch_gate, epoch_slot=EPOCH_SLOT,
            launch_grid_x=LAUNCH_GRID_X, producer_blocks=producer_blocks)

        emit_launch_rendezvous(
            tid=tid, is_owner=is_owner, p_launch_ready=addr_p2p_launch_ready,
            a_launch_ready=addr_launch_ready, a_reset_counters=addr_reset,
            reset_count=1, gate_addr=gate_addr, gate_epoch=gate_epoch,
            launch_epoch=launch_epoch, npes=npes, rank=rank)

        if const_expr(slots == 1):
            slab = fx.Int32(0)
        else:
            slab = parity
        x_base_off = fx.Int64(slab) * fx.Int64(x_slab_bytes)
        s_base_off = fx.Int64(slab) * fx.Int64(scale_slab_bytes)

        if is_producer:
            destination = producer_slot // fx.Int32(BLOCKS_PER_DEST)
            sub = producer_slot - destination * fx.Int32(BLOCKS_PER_DEST)
            rx_table = _make_buffer_from_addr(addr_p2p_rx_x, fx.Int64)
            sc_table = _make_buffer_from_addr(addr_p2p_rx_scale, fx.Int64)
            peer_x = _buffer_load(rx_table, destination, fx.Int64) + x_base_off
            peer_s = _buffer_load(sc_table, destination, fx.Int64) + s_base_off
            dest_base = fx.Int32(rank) * m_local
            row0 = sub + warp * fx.Int32(BLOCKS_PER_DEST)
            row_stride = fx.Int32(BLOCKS_PER_DEST * num_waves)
            for row in range(row0, m_local, row_stride):
                dest_row = dest_base + row
                copy_row(
                    buffer_ops.create_buffer_resource_from_addr(addr_x_q + fx.Int64(row) * fx.Int64(ROW_BYTES)),
                    buffer_ops.create_buffer_resource_from_addr(peer_x + fx.Int64(dest_row) * fx.Int64(ROW_BYTES)),
                    lane, safe_end_i32=ROW_SAFE_END, n_i32=ROW_I32)
                copy_row(
                    buffer_ops.create_buffer_resource_from_addr(addr_x_scale + fx.Int64(row) * fx.Int64(SCALE_BYTES)),
                    buffer_ops.create_buffer_resource_from_addr(peer_s + fx.Int64(dest_row) * fx.Int64(SCALE_BYTES)),
                    lane, safe_end_i32=SCALE_SAFE_END, n_i32=SCALE_I32)
            fx.rocdl.s_waitcnt(0)
            fx.barrier()
            # Last producer block on this rank publishes once to every peer, so
            # payload_ready advances by exactly npes per launch, which is the
            # step the host's _expected_for() assumes.
            if tid == fx.Int32(0):
                comm_ops.fence_system_release()
                done = fx.Int32(comm_ops.atomic_add_agent(addr_reset, fx.Int32(1)))
                if done == fx.Int32(producer_blocks - 1):
                    pr_table = _make_buffer_from_addr(addr_p2p_payload_ready, fx.Int64)
                    for pe in range_constexpr(npes):
                        remote = _buffer_load(pr_table, fx.Int32(pe), fx.Int64)
                        comm_ops.atomic_add_system(
                            remote + fx.Int64(parity) * fx.Int64(4), fx.Int32(1))

        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(
                addr_payload_ready + fx.Int64(parity) * fx.Int64(4), expected)
            comm_ops.fence_system_acquire()
        fx.barrier()

    @flyc.jit
    def launch(
        addr_x_q: fx.Int64, addr_x_scale: fx.Int64,
        addr_p2p_rx_x: fx.Int64, addr_p2p_rx_scale: fx.Int64,
        addr_p2p_payload_ready: fx.Int64, addr_p2p_launch_ready: fx.Int64,
        addr_payload_ready: fx.Int64, addr_launch_ready: fx.Int64,
        addr_epoch_gate: fx.Int64, addr_entry_count: fx.Int64, addr_reset: fx.Int64,
        m_local: fx.Int32, x_slab_bytes: fx.Int32, scale_slab_bytes: fx.Int32,
        parity: fx.Int32, expected: fx.Int32, launch_epoch: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            addr_x_q, addr_x_scale, addr_p2p_rx_x, addr_p2p_rx_scale,
            addr_p2p_payload_ready, addr_p2p_launch_ready, addr_payload_ready,
            addr_launch_ready, addr_epoch_gate,
            addr_entry_count, addr_reset, m_local, x_slab_bytes, scale_slab_bytes,
            parity, expected, launch_epoch,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(LAUNCH_GRID_X, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch
# fmt: on


class TPActivationGather:
    """Owns the symmetric receive buffers and runs the push kernel.

    Preconditions, checked in ``__init__``:
      * ``mori.shmem.shmem_torch_process_group_init`` has already been called
        by the caller, over a communicator whose size equals ``tp_size``
      * every rank constructs this with identical arguments, in the same order
        (``mori_shmem_create_tensor`` is collective)

    Buffers are sized by ``max_tok_per_rank`` once at construction; Mori's
    symmetric allocation cannot be resized per call.
    """

    def __init__(
        self,
        *,
        model_dim,
        tp_size,
        tp_rank,
        max_tok_per_rank,
        device=None,
        num_waves=4,
        producer_blocks=32,
        double_buffer=True,
        enable_pull=True,
    ):
        if int(tp_size) not in _SUPPORTED_TP:
            raise ValueError(
                f"tp_size={tp_size} unsupported; expected one of {_SUPPORTED_TP}"
            )
        if not (0 <= int(tp_rank) < int(tp_size)):
            raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
        if int(max_tok_per_rank) <= 0:
            raise ValueError("max_tok_per_rank must be positive")
        if int(model_dim) % 512 != 0:
            # copy_row moves dwordx4 chunks and the scale row is model_dim/32
            # bytes; both need 16-byte granularity, which model_dim % 512 == 0
            # guarantees for the activation row (model_dim bytes) and the scale
            # row (model_dim/32 bytes) simultaneously.
            raise ValueError(f"model_dim={model_dim} must be a multiple of 512")
        if int(producer_blocks) % int(tp_size) != 0:
            raise ValueError(
                f"producer_blocks={producer_blocks} must be divisible by tp_size={tp_size}"
            )
        if enable_pull and not double_buffer:
            # Pull inverts the buffer-reuse hazard: instead of a peer writing into
            # our receive slab, we read the peer's source slab, so what has to be
            # ordered is the peer's *next* quantize against our current read. The
            # launch rendezvous cannot order that on its own, because the quantize
            # runs before the rendezvous on the peer's stream. Two slots close the
            # gap -- a peer only rewrites slab p in round N+2, and by then our
            # round-N kernel has provably retired. Checked before the first
            # collective allocation so a bad config raises instead of hanging.
            raise ValueError("enable_pull requires double_buffer=True")

        initialised = _shmem_looks_initialised()
        if initialised is False:
            raise RuntimeError(
                "Mori SHMEM is not initialised; call "
                "mori.shmem.shmem_torch_process_group_init(<pg name>) before "
                "constructing TPActivationGather. (Constructing anyway would "
                "SEGFAULT inside Mori rather than raise.)"
            )
        # initialised is None -> Mori's internals changed shape; skip the guard
        # rather than block a valid caller.
        if initialised:
            shmem_npes = int(ms.shmem_npes())
            if shmem_npes != int(tp_size):
                raise ValueError(
                    f"Mori SHMEM world size {shmem_npes} != tp_size {tp_size}; the "
                    "shmem communicator must be the TP group"
                )

        self.model_dim = int(model_dim)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.mtpr = int(max_tok_per_rank)
        self.num_waves = int(num_waves)
        self.producer_blocks = int(producer_blocks)
        self.slots = 2 if double_buffer else 1
        self.enable_pull = bool(enable_pull)
        dev = device or torch.device("cuda", torch.cuda.current_device())
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev

        self.scale_dim = self.model_dim // 32
        # +1 row: a zeroed PAD row the fused GEMM1 will clamp out-of-range token
        # ids to. Unused by this class, allocated now so the layout is stable.
        self.rows = self.tp_size * self.mtpr + 1
        self.row_bytes = self.model_dim
        self.scale_bytes = self.scale_dim

        # Slab byte counts are passed to the kernel as i32, and buffer resources
        # carry a 32-bit offset anyway (mega_moe_stage1.py:29).
        if self.rows * self.row_bytes >= (1 << 31):
            raise ValueError(
                f"symmetric slab {self.rows * self.row_bytes} bytes exceeds the i32 "
                f"offset ABI; lower max_tok_per_rank={self.mtpr}"
            )

        self.rx_x = ms.mori_shmem_create_tensor(
            (self.slots, self.rows, self.row_bytes), torch.uint8
        )
        self.rx_scale = ms.mori_shmem_create_tensor(
            (self.slots, self.rows, self.scale_bytes), torch.uint8
        )
        self.payload_ready = ms.mori_shmem_create_tensor((2,), torch.int32)
        self.launch_ready = ms.mori_shmem_create_tensor((self.tp_size,), torch.int32)
        for t in (self.rx_x, self.rx_scale, self.payload_ready, self.launch_ready):
            t.zero_()
        ms.shmem_barrier_all()

        # Pull needs the *source* rows to be readable by peers, which push does
        # not: under push a rank reads only its own quantized tensor. So the
        # quantized rows have to live in a symmetric buffer of their own, and the
        # producer of those rows has to write them there directly -- staging them
        # with a copy afterwards would put two extra launches on the critical
        # path and defeat the point of fusing. Hence tx_*: allocated here (a
        # collective call, so it cannot be deferred to the first forward) and
        # handed to the caller through tx_views().
        if self.enable_pull:
            self.tx_x = ms.mori_shmem_create_tensor(
                (self.slots, self.mtpr, self.row_bytes), torch.uint8
            )
            self.tx_scale = ms.mori_shmem_create_tensor(
                (self.slots, self.mtpr, self.scale_bytes), torch.uint8
            )
            for t in (self.tx_x, self.tx_scale):
                t.zero_()
            ms.shmem_barrier_all()
        else:
            self.tx_x = None
            self.tx_scale = None

        self.p2p_rx_x = _p2p_table(self.rx_x, self.tp_rank, self.tp_size, dev)
        self.p2p_rx_scale = _p2p_table(self.rx_scale, self.tp_rank, self.tp_size, dev)
        self.p2p_payload_ready = _p2p_table(
            self.payload_ready, self.tp_rank, self.tp_size, dev
        )
        self.p2p_launch_ready = _p2p_table(
            self.launch_ready, self.tp_rank, self.tp_size, dev
        )
        if self.enable_pull:
            self.p2p_tx_x = _p2p_table(self.tx_x, self.tp_rank, self.tp_size, dev)
            self.p2p_tx_scale = _p2p_table(
                self.tx_scale, self.tp_rank, self.tp_size, dev
            )
        else:
            self.p2p_tx_x = None
            self.p2p_tx_scale = None

        # Round counter kept on the host: parity is deterministic (every rank
        # calls the same number of times), so deriving it here avoids a
        # GPU->CPU sync per call. Measured cost of that sync: ~15us, 13-26% of
        # the gather.
        self._round = 0

        # Local (non-symmetric) per-launch state.
        self.epoch_gate = torch.zeros(10, dtype=torch.int32, device=dev)
        self.entry_count = torch.zeros(10, dtype=torch.int64, device=dev)
        # One 64-byte cache line per counter, like MegaMoE's work_head.
        self.reset_counters = torch.zeros(
            _RESET_COUNTERS * 16, dtype=torch.int32, device=dev
        )

    def slab_bytes(self, kind):
        """Bytes per parity slab, for the kernel's parity offset arithmetic."""
        if kind == "x":
            return self.rows * self.row_bytes
        if kind == "scale":
            return self.rows * self.scale_bytes
        if kind == "tx_x":
            return self.mtpr * self.row_bytes
        if kind == "tx_scale":
            return self.mtpr * self.scale_bytes
        raise ValueError(kind)

    def tx_views(self, m_local, parity):
        """Destination views the caller should quantize into for a pull round.

        Returns ``(x_q, x_scale)`` aliasing this rank's symmetric source slab, so
        the quantize kernel writes where the peers will read. The activation view
        carries float8_e4m3fn to match what a plain per_1x32_mx_quant would
        return; the scale view stays uint8, as it already is everywhere else.
        """
        if not self.enable_pull:
            raise RuntimeError("tx_views() needs enable_pull=True")
        m = int(m_local)
        if not (0 < m <= self.mtpr):
            raise ValueError(f"m_local={m} outside 1..{self.mtpr}")
        x = self.tx_x[parity, :m].view(torch.float8_e4m3fn)
        scale = self.tx_scale[parity, :m]
        return x, scale

    def is_tx_view(self, t, m_local, parity, kind):
        """True if ``t`` already aliases the tx slab, i.e. no staging copy is due."""
        if not self.enable_pull:
            return False
        base = self.tx_x if kind == "x" else self.tx_scale
        return (
            t.data_ptr() == base[parity].data_ptr()
            and t.shape[0] == int(m_local)
            and t.is_contiguous()
        )

    def stage_source(self, x_q, x_scale, parity):
        """Make sure this round's rows are in the symmetric slab peers will read.

        A caller that quantized into tx_views() already satisfies this and pays
        nothing. Callers that hand in an ordinary tensor (the tests and the bench
        do) get a copy, which is correct but costs two extra launches -- it is
        not the shape the production path should take.
        """
        m_local = int(x_q.shape[0])
        if self.is_tx_view(x_q, m_local, parity, "x") and self.is_tx_view(
            x_scale, m_local, parity, "scale"
        ):
            return
        dst_x, dst_scale = self.tx_views(m_local, parity)
        dst_x.view(torch.uint8).copy_(x_q.view(torch.uint8))
        dst_scale.copy_(x_scale.view(torch.uint8))

    def _validate(self, x_q, x_scale):
        m_local = int(x_q.shape[0])
        if m_local <= 0:
            raise ValueError("m_local must be positive")
        if m_local > self.mtpr:
            raise ValueError(f"m_local={m_local} exceeds max_tok_per_rank={self.mtpr}")
        if x_q.dtype != torch.float8_e4m3fn or not x_q.is_contiguous():
            raise ValueError("x_q must be contiguous float8_e4m3fn")
        if tuple(x_q.shape) != (m_local, self.model_dim):
            raise ValueError(
                f"x_q must be [{m_local}, {self.model_dim}], got {tuple(x_q.shape)}"
            )
        if x_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(
                f"x_scale must be uint8 or float8_e8m0fnu, got {x_scale.dtype}"
            )
        if not x_scale.is_contiguous():
            raise ValueError("x_scale must be contiguous")
        if tuple(x_scale.shape) != (m_local, self.scale_dim):
            raise ValueError(
                f"x_scale must be [{m_local}, {self.scale_dim}], got {tuple(x_scale.shape)}"
            )
        for name, t in (("x_q", x_q), ("x_scale", x_scale)):
            if t.device != self.device:
                raise ValueError(f"{name} is on {t.device}, expected {self.device}")
        return m_local

    def current_parity(self):
        """Parity the NEXT gather() will write. Host-derived, no device read."""
        return (self._round % 2) if self.slots == 2 else 0

    def _expected_for(self, parity):
        """How many source-rank publishes payload_ready[parity] holds after this round.

        With two slots a given parity is used every other round, so by the end
        of round ``self._round`` it has been used ``(self._round - parity) // 2
        + 1`` times. With a single slot parity is always 0 and the slot is used
        every round, so the count is ``self._round + 1``. Each use adds exactly
        ``tp_size`` publishes, one per source rank.
        """
        if self.slots == 1:
            rounds = self._round + 1
        else:
            rounds = (self._round - parity) // 2 + 1
        return rounds * self.tp_size

    def views(self, m_local, parity):
        """Views of the gathered result for a given round. Valid until the next gather."""
        n = self.tp_size * int(m_local)
        x = self.rx_x[parity, :n].view(torch.float8_e4m3fn)
        scale = self.rx_scale[parity, :n]
        return x, scale

    def gather(self, x_q, x_scale, stream=None):
        """Push this rank's rows to every peer and wait for theirs.

        Returns ``(rx_x, rx_scale)`` views of shape ``[tp_size*m_local, ...]``,
        valid until the next call (double buffering gives one round of slack).
        """
        m_local = self._validate(x_q, x_scale)
        launch = compile_tp_gather(
            model_dim=self.model_dim,
            npes=self.tp_size,
            rank=self.tp_rank,
            producer_blocks=self.producer_blocks,
            num_waves=self.num_waves,
            slots=self.slots,
        )
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        parity = self.current_parity()
        _run_compiled(
            launch,
            fx.Int64(x_q.data_ptr()),
            fx.Int64(x_scale.data_ptr()),
            fx.Int64(self.p2p_rx_x.data_ptr()),
            fx.Int64(self.p2p_rx_scale.data_ptr()),
            fx.Int64(self.p2p_payload_ready.data_ptr()),
            fx.Int64(self.p2p_launch_ready.data_ptr()),
            fx.Int64(self.payload_ready.data_ptr()),
            fx.Int64(self.launch_ready.data_ptr()),
            fx.Int64(self.epoch_gate.data_ptr()),
            fx.Int64(self.entry_count.data_ptr()),
            fx.Int64(self.reset_counters.data_ptr()),
            fx.Int32(m_local),
            fx.Int32(self.slab_bytes("x")),
            fx.Int32(self.slab_bytes("scale")),
            fx.Int32(parity),
            fx.Int32(self._expected_for(parity)),
            fx.Int32(self._round + 1),
            stream,
        )
        self._round += 1
        return self.views(m_local, parity)
