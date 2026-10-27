# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)

from .tensor_shim import buf_base_i64

GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64
SPLIT_K_SEMAPHORE_MAX_LEN = 256


def wait_vmcnt_and_barrier(vmcnt=0):
    rocdl.s_waitcnt(vmcnt=vmcnt)
    rocdl.s_barrier()


def _global_sync_ptr(ptr, offset, dtype):
    """Naturally aligned global pointer for split-K tickets and signals."""
    size = dtype.width // 8
    address = buf_base_i64(ptr) + fx.Int64(offset) * size
    ptr_type = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, size)
    return fx.inttoptr(ptr_type, address)


class BlockSwizzle:
    def __init__(self, NUM_XCDS, NUM_PIDS_THRESHOLD, GROUP_M):
        self.NUM_XCDS = NUM_XCDS
        self.NUM_PIDS_THRESHOLD = NUM_PIDS_THRESHOLD
        self.GROUP_M = GROUP_M

    @flyc.jit
    def swizzle(self, num_pid_m, num_pid_n, pid):
        simple_m = pid // num_pid_n
        simple_n = pid % num_pid_n
        if const_expr(self.GROUP_M <= 0):
            return simple_m, simple_n
        num_xcds = self.NUM_XCDS
        swizzle_threshold = self.NUM_PIDS_THRESHOLD
        num_wg = num_pid_m * num_pid_n
        linear_id = pid
        intra_xcd = linear_id // num_xcds
        xcd = linear_id % num_xcds
        wgid = xcd * (num_wg // num_xcds) + intra_xcd
        group_m = self.GROUP_M
        wgid_per_group = group_m * num_pid_n
        group_id = wgid // wgid_per_group
        intra_group = wgid % wgid_per_group
        first_pid_m = group_id * group_m
        remaining_m = num_pid_m - first_pid_m
        group_size_m = (remaining_m < group_m).select(remaining_m, group_m)
        swizzled_n = intra_group // group_size_m
        swizzled_m = first_pid_m + (intra_group % group_size_m)
        use_simple = (num_wg < swizzle_threshold) | ((num_wg % num_xcds) != 0)
        return (
            use_simple.select(simple_m, swizzled_m),
            use_simple.select(simple_n, swizzled_n),
        )


class SplitKReduction:
    """Deterministic fp32 split-K reduction.

    Every workgroup takes a ticket when it starts and stores its fp32 partial. The
    tile's last workgroup to start reduces: the others count themselves done once
    their stores have landed, the reducer waits for that count and adds every
    split's partial in split order, so the sum does not depend on arrival order.
    Waiting only on workgroups that took a ticket earlier, and so are already
    running, cannot deadlock. Partials use sc1 buffer stores and loads, which are
    coherent at agent scope (what LLVM emits for monotonic agent-scope accesses),
    so no cache-wide fence is needed. Per launch a tile's ticket word advances by
    SPLIT_K and its done word by SPLIT_K - 1 (buffers are keyed by split_k and never
    reset), so ticket // SPLIT_K is the launch index. Partials are thread-major
    float4 groups of the MMA fragments; ``tid`` and ``THREADS`` index the threads
    that hold the tile (k-wave 0 under slice-K).
    """

    SC1 = 16  # buffer cache policy: coherent at agent scope

    def __init__(self, SPLIT_K, TILE_ELEMS, THREADS):
        self.SPLIT_K = SPLIT_K
        self.TILE_ELEMS = TILE_ELEMS
        self.THREADS = THREADS

    @flyc.jit
    def init(self, semaphore, workspace, flag_ptr, tid, tile_idx, ks_idx):
        self.ticket_ptr = _global_sync_ptr(semaphore, tile_idx, fx.Int64)
        self.done_ptr = _global_sync_ptr(
            semaphore, SPLIT_K_SEMAPHORE_MAX_LEN + tile_idx, fx.Int64
        )
        self.slots = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(workspace, max_size=True),
            fx.make_layout(1, 1),
        )
        self.flag_ptr = flag_ptr
        self.tid = tid
        self.ks_idx = ks_idx
        self.tile_base = tile_idx * (self.SPLIT_K * self.TILE_ELEMS)
        self.copy_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(self.SC1), fx.Float32
        )

    @flyc.jit
    def take_ticket(self):
        # Issued early so the round trip hides under the K loop; only thread 0
        # holds the ticket.
        ticket = fx.Int64(0)
        if fx.thread_idx.x == 0:
            ticket = fx.atomic_add(self.ticket_ptr, fx.Int64(1), syncscope="agent")
        return ticket

    @flyc.jit
    def is_reducer(self, ticket):
        # The flag sits in shared C, which aliases the A/B stages: wait for the
        # K loop of every wave before writing it.
        gpu.barrier()
        if fx.thread_idx.x == 0:
            last = (ticket % self.SPLIT_K) == (self.SPLIT_K - 1)
            fx.ptr_store(last.select(fx.Int32(1), fx.Int32(0)), self.flag_ptr)
        gpu.barrier()
        return fx.ptr_load(self.flag_ptr) != 0

    def _slot(self, ks, group):
        offset = self.tile_base + ks * self.TILE_ELEMS
        return fx.slice(
            self.slots, (None, offset + (group * self.THREADS + self.tid) * 4)
        )

    def store_partial(self, frags):
        group = 0
        for frag in frags:
            for i in range_constexpr(fx.size(frag.shape).unpack() // 4):
                vec = fx.make_rmem_tensor(4, fx.Float32)
                vec.store(
                    fx.Vector.from_elements(
                        [frag[i * 4 + e] for e in range_constexpr(4)], fx.Float32
                    )
                )
                fx.copy(self.copy_atom, vec, self._slot(self.ks_idx, group))
                group += 1

    @flyc.jit
    def signal_done(self):
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if fx.thread_idx.x == 0:
            fx.atomic_add(self.done_ptr, fx.Int64(1), syncscope="agent")

    @flyc.jit
    def wait_for_partials(self, ticket):
        rocdl.s_waitcnt(vmcnt=0)  # this workgroup reads its own partial back too
        if fx.thread_idx.x == 0:
            target = (ticket // self.SPLIT_K + 1) * (self.SPLIT_K - 1)
            done = target - 1
            while done < target:
                done = fx.generic_load(
                    self.done_ptr,
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope="agent",
                )
        gpu.barrier()

    def accumulate(self, frag, first_group, init_frag):
        """Set ``frag`` to init + every split's partial, in split order;
        ``init_frag`` writes the bias (or zero) into ``frag``."""
        init_frag()
        for ks in range_constexpr(self.SPLIT_K):
            for i in range_constexpr(fx.size(frag.shape).unpack() // 4):
                partial = fx.make_rmem_tensor(4, fx.Float32)
                fx.copy(self.copy_atom, self._slot(ks, first_group + i), partial)
                for e in range_constexpr(4):
                    frag[i * 4 + e] = frag[i * 4 + e] + partial[e]


def make_lds_layout(rows, block_k):
    swizzle = fx.static(fx.SwizzleType.get(3, 3, 3))
    return fx.make_composed_layout(
        swizzle,
        fx.make_ordered_layout((rows, block_k), (1, 0)),
    )


def make_transposed_lds_layout(rows, block_k):
    # Preserve the 16-element groups required by ds_read_tr16 and XOR low K
    # bits into contiguous-dimension bits [4:6] to spread LDS bank accesses.
    base_layout = fx.make_ordered_layout((rows, block_k), (0, 1))
    if const_expr(rows == 64):
        trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 2))
        return fx.make_composed_layout(trans_swizzle, base_layout)
    if const_expr(rows == 128):
        trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 3))
        return fx.make_composed_layout(trans_swizzle, base_layout)
    if const_expr(rows == 256):
        trans_swizzle = fx.static(fx.SwizzleType.get(2, 4, 4))
        return fx.make_composed_layout(trans_swizzle, base_layout)
    return base_layout


def get_wave_lds_offset(tid, async_load_bytes):
    return rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )


def swizzled_col_idx(row, col, layout, block_k):
    elem_offset = fx.get_scalar(fx.crd2idx((row, col), layout))
    return elem_offset % block_k


def transposed_contiguous_idx(idx, k_idx, layout, rows):
    # The XOR swizzle is self-inverse. Given the physical contiguous
    # position written by direct-to-LDS DMA, select the logical global
    # vector that belongs at that position.
    elem_offset = fx.get_scalar(fx.crd2idx((idx, k_idx), layout))
    return elem_offset % rows
