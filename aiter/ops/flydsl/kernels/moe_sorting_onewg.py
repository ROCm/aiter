# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Technique source: the one-workgroup expert sort of amd-kernels
# (moe/mxfp4_h6144_i768_e128_top4/sort_a16w4.py).

"""One-workgroup MoE expert sort for E == 128 experts and up to 512 tokens (FlyDSL).

Same contract as ``moe_sorting_flydsl`` (``sorted_ids[row] = token | slot << 24`` with the padding sentinel
``(topk << 24) | M``, ``sorted_weights``, ``sorted_expert_ids`` per block, ``num_valid_ids = [padded rows, M]``),
for the case where no output clearing is wanted (``moe_buf`` empty, no EP mask).

Why a separate kernel: ``moe_sorting_flydsl``'s oneshot path stops at ~16 tokens and above that it launches the
multiphase kernels (HBM workspace, several workgroups, inter-CTA sync).  Up to 512 tokens (2048 pairs) the whole
sort fits one workgroup's LDS, so there is no cross-CTA state at all:

  * the sorted table is built in LDS: every row is first written as padding, the pairs land on their rows after
    one wave's scan of the 64 expert pairs, and rows ``[0, num_valid)`` leave in 16 B stores after all LDS reads
    (one CU carries every store, so scattered dword stores would be most of the kernel time);
  * a pair's row inside its expert is its LDS-atomic arrival rank (``rank=True``, from ``RANK_FROM_TOKENS``
    tokens) or the atomic cursor.  Consumers must not rely on the order inside an expert: every pair owns its own
    output slot, so placement order does not change any GEMM output bit.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx_runtime

from . import buffer_ops
from .tensor_shim import _run_compiled

THREADS = 256  # one thread per histogram / scan entry
NUM_EXPERTS = 128  # the wave scan covers 64 lanes x 2 experts
MAX_TOKENS = 512
# rank placement from this many tokens (inline routing: faster at 64..512 tokens, equal below)
RANK_FROM_TOKENS = 17
# compile tiers (max tokens; pairs per thread = topk * tier / THREADS)
TOKEN_TIERS = (64, 128, 192, 256, 384, 448, 512)

# inclusive wave scan over 64 lanes: row_shr 1/2/4/8, then row_bcast15 / row_bcast31 (DPP control, row mask, bound)
_WAVE_SCAN_DPP = (
    (0x111, 0xF, True),
    (0x112, 0xF, True),
    (0x114, 0xF, True),
    (0x118, 0xF, True),
    (0x142, 0xA, False),
    (0x143, 0xC, False),
)


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _lds_atomic_add_i32(ptr, value):
    """``old = *ptr; *ptr += value`` on an LDS i32 pointer (workgroup scope), returns ``old``.

    Raw ``llvm.AtomicRMWOp``: ``fx`` has no LDS atomic wrapper (the rank / cursor of a pair is the returned count).
    """
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


def _wave_incl_scan_i32(x):
    """Inclusive prefix sum of an i32 over the 64 lanes of a wave (DPP, no LDS).

    Raw ``rocdl.update_dpp``: no ``fx`` wave-scan primitive; the DPP chain keeps the scan off LDS (the whole
    kernel is ~2 us, an LDS scan adds barriers).
    """
    zero = _raw(fx.Int32(0))
    for ctrl, row_mask, bound in _WAVE_SCAN_DPP:
        t = _rocdl_ops.update_dpp(T.i32, zero, _raw(x), ctrl, row_mask, 0xF, bound)
        x = x + fx.Int32(t)
    return x


def max_sorted_rows(n_tokens: int, num_experts: int, topk: int, block_m: int) -> int:
    """Rows bound of the sorted table: every active expert pads by < block_m rows."""
    active = min(num_experts, n_tokens * topk)
    cumsum_max = n_tokens * topk + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


def _fits(tier, topk, block_m):
    """The block -> expert table needs a thread per block, the sorted table + counters LDS (gfx950: 160 KiB)."""
    ms = max_sorted_rows(tier, NUM_EXPERTS, topk, block_m)
    pairs = tier * topk
    lds = 4 * (
        2 * (ms + 4 * THREADS) + NUM_EXPERTS * 2 + 4 + THREADS + pairs * 2 + THREADS
    )
    return ms // block_m <= THREADS and lds <= 150 * 1024


def onewg_supported(M, num_experts, topk, unit_size, has_mask=False, moe_buf_numel=0):
    """Whether :func:`moe_sorting_onewg` covers this call (otherwise use ``moe_sorting_flydsl``)."""
    # verified on gfx950 only (wave64 DPP scan with row_bcast; gfx1250 is wave32)
    if get_gfx_runtime() != "gfx950":
        return False
    if not (
        num_experts == NUM_EXPERTS
        and 1 <= M <= MAX_TOKENS
        and unit_size in (16, 32, 64)
        and 1 <= topk <= 8
        and not has_mask
        and moe_buf_numel == 0
    ):
        return False
    return _fits(next(t for t in TOKEN_TIERS if M <= t), topk, unit_size)


@functools.lru_cache(maxsize=64)
def compile_moe_sorting_onewg(
    *, topk: int, block_m: int, max_tokens: int, rank: bool = True, device: int = 0
):
    """Histogram, padded-count scan and placement in one workgroup, no cross-CTA state.

    ``device`` only keys the cache: a compiled handle is bound to the device it was built on.
    """
    E = NUM_EXPERTS
    threads = THREADS
    assert (block_m & (block_m - 1)) == 0 and threads >= E
    bm_shift = block_m.bit_length() - 1
    PPT = (max_tokens * topk + threads - 1) // threads
    assert block_m <= 64
    tag = f"e{E}_top{topk}_bm{block_m}_ppt{PPT}_t{threads}" + ("_rank" if rank else "")
    # rank: invalid pairs count into their own slot past E (distinct addresses, never read)
    COUNT = E + threads * PPT if rank else E
    # cursor: invalid pairs advance their own slot past E too (no branch around the atomic)
    CURSOR = E if rank else E + threads * PPT
    # the sorted table in LDS (ids rows, then weight bits): padding everywhere first, the pairs on top, then
    # [0, num_valid) leaves in 16 B stores
    MS = max_sorted_rows(max_tokens, E, topk, block_m)
    # 16 B tiles per thread and array; row MS takes the invalid pairs, so the table holds MS + 1 rows (MS alone
    # fills whole tiles when MS / 4 is a multiple of the thread count, e.g. topk=1, block_m=16, 64 tokens)
    CH = ((MS + 4) // 4 + threads - 1) // threads
    TAB = CH * threads * 4  # rows per array
    assert TAB > MS
    NB = MS // block_m  # blocks of the bound: one block -> expert entry per thread
    assert NB <= threads

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, COUNT]  # rows per expert
        cursor: fx.Array[fx.Int32, CURSOR]  # next free sorted row per expert
        scan: fx.Array[fx.Int32, E]  # expert start (rank) / end (cursor) rows
        nvs: fx.Array[fx.Int32, 4]  # num_valid for every wave
        tab: fx.Array[
            fx.Int32, 2 * TAB, 16
        ]  # sorted ids rows, then weight bits (16 B aligned)
        blk: fx.Array[
            fx.Int32, threads + threads * PPT
        ]  # block -> expert; past threads: one dummy per pair

    @flyc.kernel(name=f"moe_sorting_onewg_{tag}", known_block_size=[threads, 1, 1])
    def moe_sorting_onewg_kernel(
        topk_ids: fx.Tensor,  # [n_tokens * topk] i32
        topk_w: fx.Tensor,  # [n_tokens * topk] f32
        sorted_ids: fx.Tensor,  # [max_sorted] i32
        sorted_w: fx.Tensor,  # [max_sorted] f32
        sorted_eids: fx.Tensor,  # [max_sorted / block_m] i32
        num_valid: fx.Tensor,  # [2] i32
        n_tok: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        if bx == 0:
            count, cursor, blk = smem.count.ptr, smem.cursor.ptr, smem.blk.ptr
            scan = smem.scan.ptr
            n_pairs = n_tok * topk
            last_pair = n_pairs - 1
            t_lt_E = tx < E
            active = (n_pairs < E).select(n_pairs, fx.Int32(E))
            ms_rows = (
                (n_pairs + active * (block_m - 1) + (block_m - 1)) >> bm_shift
            ) << bm_shift
            ids_r = buffer_ops.create_buffer_resource(
                sorted_ids, max_size=False, num_records_bytes=ms_rows * 4
            )
            w_r = buffer_ops.create_buffer_resource(
                sorted_w, max_size=False, num_records_bytes=ms_rows * 4
            )
            e_r = buffer_ops.create_buffer_resource(
                sorted_eids, max_size=False, num_records_bytes=(ms_rows >> bm_shift) * 4
            )

            # 0. this thread's pairs: one batch of loads (clamped index, validity)
            pairs = []
            for k in range_constexpr(PPT):
                p = tx + k * threads
                valid = p < n_pairs
                pc = valid.select(p, last_pair)
                pairs.append((p, valid, fx.Int32(topk_ids[pc]), fx.Float32(topk_w[pc])))

            lds16 = fx.logical_divide(
                fx.make_view(
                    fx.recast_iter(fx.Int32, smem.tab.ptr), fx.make_layout(2 * TAB, 1)
                ),
                fx.make_layout(4, 1),
            )
            lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

            def lds_store16(tile, vec4):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                r.store(vec4)
                fx.copy(lds_atom, r, fx.slice(lds16, (None, tile)))

            def lds_load16(tile):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
                return Vec(r.load())

            # every table row as padding (sentinel (topk << 24) | n_tokens, weight 0), under the id loads
            pad_ids = Vec.from_elements([n_tok | (topk << 24)] * 4, fx.Int32)
            pad_w = Vec.filled(4, 0, fx.Int32)
            for j in range_constexpr(CH):
                lds_store16(tx + j * threads, pad_ids)
                lds_store16(tx + (j + CH) * threads, pad_w)
            tab = smem.tab.ptr

            # 1. zero the counters
            if t_lt_E:
                count[tx] = 0
            fx.gpu.barrier()

            # 2. histogram (by scan slot); rank: the returned count is the pair's rank
            ranks = []
            for k, (p, valid, e, _w) in enumerate(pairs):
                if const_expr(rank):
                    slot = valid.select(e, fx.Int32(E + k * threads) + tx)
                    ranks.append(_lds_atomic_add_i32(count + slot, 1))
                else:
                    if valid:
                        _lds_atomic_add_i32(count + e, 1)
            fx.gpu.barrier()

            # 3. one wave scans 64 consecutive expert pairs; starts / ends stay in expert order
            if wave == 0:
                c0, c1 = fx.Int32(count[lane * 2]), fx.Int32(count[lane * 2 + 1])
                p0 = (c0 + (block_m - 1)) & ~(block_m - 1)
                p1 = (c1 + (block_m - 1)) & ~(block_m - 1)
                pair_end = _wave_incl_scan_i32(p0 + p1)
                pair_start = pair_end - (p0 + p1)
                for j in range_constexpr(2):
                    e_t = lane * 2 + j
                    start = pair_start + (p0 if const_expr(j == 1) else fx.Int32(0))
                    end = start + (p0 if const_expr(j == 0) else p1)
                    scan[e_t] = start if const_expr(rank) else end
                    if const_expr(not rank):
                        cursor[e_t] = start
                if lane == 63:
                    num_valid[0] = pair_end
                    num_valid[1] = n_tok
                    smem.nvs.ptr[0] = pair_end
            fx.gpu.barrier()

            # 4. every pair's row
            rows = []
            for k, (p, valid, e, w) in enumerate(pairs):
                if const_expr(rank):
                    rows.append(fx.Int32(scan[e]) + ranks[k])
                else:
                    slot = valid.select(e, fx.Int32(E + k * threads) + tx)
                    rows.append(_lds_atomic_add_i32(cursor + slot, 1))
            nv = fx.Int32(smem.nvs.ptr[0])

            # 5. place the pairs into the table (rows of a pair's expert, disjoint from its padding)
            for k, (p, valid, e, w) in enumerate(pairs):
                r = valid.select(
                    rows[k], fx.Int32(MS)
                )  # invalid pairs: a row past MS, never copied
                tab[r] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
                tab[r + TAB] = w.bitcast(fx.Int32)
                # a block's first row is always a pair (padding < block_m rows): its pair names the block
                first = valid & ((r & (block_m - 1)) == 0)
                blk[
                    first.select(r >> bm_shift, fx.Int32(threads + k * threads) + tx)
                ] = e
            fx.gpu.barrier()

            # 6. rows [0, num_valid) in 16 B stores: every LDS read first, then the stores
            tiles = []
            for j in range_constexpr(CH):
                t = tx + j * threads
                tiles.append((t, lds_load16(t), lds_load16(t + CH * threads)))
            blk_e = fx.Int32(blk[tx])
            buffer_ops.buffer_store(blk_e, e_r, tx, mask=tx < (nv >> bm_shift))
            for t, vi, vw in tiles:
                in_rows = t * 4 < nv
                buffer_ops.buffer_store(vi, ids_r, t * 4, mask=in_rows)
                buffer_ops.buffer_store(vw, w_r, t * 4, mask=in_rows)

    @flyc.jit
    def launch_moe_sorting_onewg(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        moe_sorting_onewg_kernel(
            topk_ids, topk_w, sorted_ids, sorted_w, sorted_eids, num_valid, n_tokens
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch_moe_sorting_onewg


def moe_sorting_onewg(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    num_experts,
    unit_size,
    rank=None,
):
    """Sort ``topk_ids`` [M, topk] (M <= 512, E == 128) into the padded per-expert layout of ``moe_sorting``.

    Precondition: every id of ``topk_ids`` is in ``[0, 128)`` (no -1 / dropped routes, no EP mask); ids outside
    that range index LDS out of bounds.  Output buffers are caller-allocated like ``aiter.moe_sorting_fwd`` (sizes from ``max_sorted_rows``).  The
    order of the pairs inside one expert is unspecified (``rank``: LDS-atomic arrival order).
    """
    M, topk = topk_ids.shape
    assert onewg_supported(
        M, num_experts, topk, unit_size
    ), "moe_sorting_onewg: unsupported shape"
    if rank is None:
        rank = M >= RANK_FROM_TOKENS
    with torch.cuda.device(topk_ids.device):
        launch = compile_moe_sorting_onewg(
            topk=topk,
            block_m=unit_size,
            max_tokens=next(t for t in TOKEN_TIERS if M <= t),
            rank=rank,
            device=torch.cuda.current_device(),
        )
        # through the cached handle: a direct @flyc.jit call enters the JIT (a full trace) on every call
        _run_compiled(
            launch,
            topk_ids.view(-1),
            topk_weights.view(-1),
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            int(M),
            torch.cuda.current_stream(topk_ids.device),
        )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids
