# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, I001
"""Stage2 kernel1: GEMM2 + weighted push into the node-local plane-slot inbox.

The GEMM2 core (``gemm2_compute_v2``) is MegaMoEV2's; this file adds the
EP16 epilogue that sends each weighted BF16 row to the GPU with the owner's
local index on this node and publishes its arrival flag.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T
from flydsl.runtime.device import get_rocm_arch

from ..mxfp4_gemm_common import (
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
import mori.cco.device.flydsl as cco

from . import comm_ops
from ..tensor_shim import (
    _run_compiled,
    buf_copy_store,
    ptr_buf_tensor,
)

from .gemm2 import (
    gemm2_compute_v2,
    issue_a_load_lds_dt,
    kStages,
)

_BUFFER_OFFSET_ABI_BYTES = 1 << 31
# Tile shape: 32-row M tiles (one Stage1 tile each), K step 256.
BM = 32
BK = 256
# Workgroups are dispatched round-robin over the 8 XCDs (SPX mode).
NUM_XCD = 8


# fmt: off
def p2p_scatter_epilog(lds_acc_base, accm, n_block_idx, wave, lane, *, N_OUT, BM, BN, npes, topk,
    log2_max_tok, mask_max_tok, recv_cap, comb_inp_nbytes, lds_packed_off, lds_weight_off,
    lds_peer_off, scatter_vec, gpus_per_node, arrival_delta, arrival_count_ptr, num_n_blocks,
    generation):
# fmt: on
    """CShuffle one GEMM2 tile into weighted BF16 rows and scatter them to peers."""
    kMChunks = BM // 16
    numAccN = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)

    # CPol (SIDefines.h:593, pre-gfx12 encoding): GLC=SC0=1, SLC=NT=2,
    # SCC=SC1=16.  The default 2 is non-temporal with no coherence scope, so
    # the payload sits dirty in this rank's L2 and someone has to write it
    # back before a peer can see it -- that writeback is what made the
    # per-row release RMW cost 41x.  17|2 = sc0|sc1|nt writes through to the
    # system coherence point instead, so no writeback is ever needed and the
    # outgoing payload stops competing with w2 for L2 as well.
    # Publishing implies write-through: there is no fence anywhere in this
    # path any more, so a payload left dirty in this rank's L2 would never
    # reach the peer.  The two are one decision, not two knobs.
    pay_cm = 19
    out_elem_bytes = 2
    token_nbytes = N_OUT * out_elem_bytes

    fx.barrier()

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        for J in range_constexpr(numAccN):
            col = wave * fx.Int32(wave_n) + J * 16 + lane_mod_16
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + col
                lds_base_fptr[idx] = fx.Float32(vec[v])

    fx.barrier()

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        row_byte_off = row * fx.Int32(4)
        p = fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_packed_off) + row_byte_off, T.i32, align=4)
        )
        weight = fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_weight_off) + row_byte_off, T.f32, align=4)
        )
        p = rocdl.readfirstlane(T.i32, p.ir_value())
        weight = rocdl.readfirstlane(T.f32, weight.ir_value())
        t = p & fx.Int32(0x00FFFFFF)
        if const_expr(topk <= 8):
            # Preserve the established DSV4-Pro instruction sequence.
            s = p >> fx.Int32(24)
        else:
            # Slots 8..15 set bit 31, so decode the high byte as unsigned.
            s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        owner_rank = t >> fx.Int32(log2_max_tok)
        dest_lid = t & fx.Int32(mask_max_tok)
        valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (
            owner_rank < fx.Int32(npes))
        # Local rank l proxies global ranks l and 8+l: the intra-node hop
        # always lands on the GPU whose local index matches the owner's, and
        # the plane says which of that proxy's two identities the token
        # belongs to. Only the remote plane is later shipped over the rail.
        dest_pe = owner_rank & fx.Int32(gpus_per_node - 1)
        dest_lid = (
            (owner_rank // fx.Int32(gpus_per_node))
            * fx.Int32(1 << log2_max_tok)
            + dest_lid
        )
        dest_pe_safe = valid.select(dest_pe, fx.Int32(0))
        peer_base = fx.ptr_load(
            lds_typed_ptr(
                fx.Int32(lds_peer_off) + dest_pe_safe * fx.Int32(8),
                T.i64,
                align=8,
            )
        )
        peer_base = rocdl.readfirstlane(T.i64, peer_base.ir_value())
        slot = dest_lid * fx.Int32(topk) + s

        row_base = slot * fx.Int32(token_nbytes)
        row_off = row_base + n_block_idx * fx.Int32(BN * out_elem_bytes)

        # Inactive lanes read safe LDS and issue a bounded OOB store.
        active = lane < fx.Int32(BN // scatter_vec)
        col = active.select(lane * fx.Int32(scatter_vec), fx.Int32(0))
        idx0 = row * fx.Int32(BN) + col
        v8 = fx.Vector(
            lds_vec_load(
                lds_acc_base,
                idx0 * fx.Int32(4),
                fx.Vector.make_type(scatter_vec, fx.Float32),
                fx.Float32,
                align=16,
            )
        )
        weighted_v8 = fx.Vector.from_elements(
            [v8[i] * weight for i in range_constexpr(scatter_vec)],
            fx.Float32,
        )
        pk = weighted_v8.to(fx.BFloat16)
        off = (valid & active).select(
            row_off + col * fx.Int32(out_elem_bytes),
            fx.Int32(comb_inp_nbytes),
        )
        output_buf = ptr_buf_tensor(
            peer_base,
            fx.BFloat16,
            unit_elems=scatter_vec,
            num_records_bytes=comb_inp_nbytes,
        )
        buf_copy_store(
            output_buf,
            off // fx.Int32(scatter_vec * 2),
            pk,
            fx.BFloat16,
            unit_elems=scatter_vec,
            cache_modifier=pay_cm,
        )


    # One (token, slot) row is written by num_n_blocks CTAs, one column
    # range each.  They count themselves on a rank-private counter and the
    # last arriver publishes the generation to the consumer.
    #
    # Ordering: the payload is stored sc0|sc1|nt, so it is at the system
    # coherence point once vmcnt retires.  One s_waitcnt for the whole
    # batch puts every row's payload there before any increment -- which is
    # why there is no fence anywhere in this path.  An earlier version put
    # a fence_system_release here: +399us.
    #
    # Neither publication store may be a *release* store.  store_i32_system
    # and store_i64_global_system are, and LLVM gives every one of those a
    # buffer_wbl2 -- a full L2 writeback, ~18k per generation, +6.7ms.  The
    # relaxed variant keeps the one-as scope and drops the writeback.
    #
    # The indices are re-decoded rather than carried across the store loop:
    # the counter needs t as well as slot, and keeping five values per row
    # live pushes SGPR past the 102 budget.
    #
    # p2p_scatter_epilog is a plain Python function -- the AST rewriter
    # never sees its `if`s -- so the dynamic branches live in a nested
    # @flyc.jit, the same reason everything else here uses .select().
    rocdl.s_waitcnt(0)
    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        p = fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_packed_off) + row * fx.Int32(4),
                          T.i32, align=4)
        )
        p = rocdl.readfirstlane(T.i32, p.ir_value())
        t = p & fx.Int32(0x00FFFFFF)
        if const_expr(topk <= 8):
            s = p >> fx.Int32(24)
        else:
            s = p.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        owner_rank = t >> fx.Int32(log2_max_tok)
        dest_lid = t & fx.Int32(mask_max_tok)
        valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (
            owner_rank < fx.Int32(npes))
        dest_pe = owner_rank & fx.Int32(gpus_per_node - 1)
        dest_lid = (
            (owner_rank // fx.Int32(gpus_per_node))
            * fx.Int32(1 << log2_max_tok)
            + dest_lid
        )
        dest_pe_safe = valid.select(dest_pe, fx.Int32(0))
        peer_base = fx.ptr_load(
            lds_typed_ptr(
                fx.Int32(lds_peer_off) + dest_pe_safe * fx.Int32(8),
                T.i64,
                align=8,
            )
        )
        peer_base = rocdl.readfirstlane(T.i64, peer_base.ir_value())

        @flyc.jit
        def _count_and_publish():
            if lane == fx.Int32(0):
              if valid:
                # Keyed by t, not by the destination-side slot: dest_lid
                # has dest_pe folded out of it, so owner ranks 3 and 4
                # share a dest_lid and would have shared a counter here.
                count_idx = t * fx.Int32(topk) + s
                prev = fx.Int32(comm_ops.atomic_add_agent(
                    arrival_count_ptr + fx.Int64(count_idx) * fx.Int64(4),
                    fx.Int32(1),
                ))
                if prev + fx.Int32(1) == num_n_blocks:
                    # Subtract rather than store 0: same effect, and it
                    # reuses the atomic that AR9 measured at +3.5us
                    # instead of a system store that carries a writeback.
                    comm_ops.atomic_add_agent(
                        arrival_count_ptr + fx.Int64(count_idx) * fx.Int64(4),
                        fx.Int32(0) - num_n_blocks,
                    )
                    comm_ops.store_i64_global_system_relaxed(
                        peer_base
                        + fx.Int64(arrival_delta)
                        + fx.Int64(dest_lid * fx.Int32(topk) + s)
                        * fx.Int64(8),
                        generation,
                    )

        _count_and_publish()


# fmt: off
def compile_mega_moe_stage2(*, model_dim: int, inter_dim: int, experts: int, topk: int, npes: int,
    max_tok: int, recv_cap: int, comb_inp_nbytes: int, BN: int, cu_num: int, gpus_per_node: int,
    arrival_delta: int, arrival_delta_step: int, epoch_planes: tuple, a_tile_map_plane: int,
    m_group: int = 1):
# fmt: on
    """Compile the persistent GEMM2 + node push kernel (stage2 kernel1).

    ``m_group`` is Stage1's tile group G: the G consecutive 32-row m-blocks
    of a group belong to one expert, and the kernel places them on one XCD
    so they share that XCD's L2 copy of the w2 tile.

    A is the FP4 GMM1 output, stored per physical 32-row tile and read through
    ``a_tile_map`` (sorted tile -> physical tile).  Every output row is
    weighted, cast to BF16 and pushed to the plane-slot inbox of the GPU with
    the owner's local index on this node; the last of the ``hidden / BN``
    CTAs writing a (token, slot) row publishes its arrival flag.

    ``i64_generation`` carries the address of the Stage-1 ``epoch_gate`` word.
    The kernel loads the generation there and selects the parity plane
    itself, so a captured CUDA graph sees a fresh generation/parity on every
    replay.  ``epoch_planes`` are the parity-plane byte sizes of (aq, ascale,
    eids, cumsum, stids, sweights); the host passes parity-0 pointers, a p2p
    table holding both parities back to back, and ``arrival_delta`` for
    parity 0 plus the per-parity step.  ``experts`` is the local expert count:
    Stage1 already stores local expert ids.
    """
    epoch_planes = tuple(int(v) for v in epoch_planes)
    if len(epoch_planes) != 6:
        raise ValueError("epoch_planes must hold six plane sizes")
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE stage2 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    assert max_tok > 0 and (max_tok & (max_tok - 1)) == 0, "max_tok must be power of two"
    assert model_dim % BN == 0
    assert inter_dim % BK == 0, f"inter_dim must be a multiple of {BK}"
    if cu_num <= 0:
        raise AssertionError(f"cu_num must be positive, got {cu_num}")
    scatter_vec = 8
    log2_max_tok = max_tok.bit_length() - 1
    mask_max_tok = max_tok - 1
    N_OUT = model_dim
    aStages = kStages + 1
    KH_TILE_A = BK // 2  # FP4: two values per byte
    # The f32 CShuffle slab and the A stages share one LDS union.
    compute_lds_bytes = max(BM * BN * 4, aStages * BM * KH_TILE_A)
    lds_packed_off = compute_lds_bytes
    lds_weight_off = lds_packed_off + BM * 4
    lds_peer_off = lds_weight_off + BM * 4
    lds_bytes = lds_peer_off + npes * 8
    if lds_bytes > 160 * 1024:
        raise ValueError(f"Stage2 LDS use {lds_bytes} exceeds 160 KiB")
    if not 0 < comb_inp_nbytes < _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("MegaMoE stage2 P2P buffer exceeds the 32-bit buffer-resource ABI")

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    num_n_blocks_c = model_dim // BN
    xcd_group = int(m_group)
    if xcd_group > 1 and (cu_num * num_n_blocks_c) % (NUM_XCD * xcd_group) != 0:
        xcd_group = 1  # remap would not be a bijection; keep the plain order

    kernel_name = (
        f"megamoe_k1_t{BM}x{BN}x{BK}_cu{cu_num}"
        + (f"_xg{xcd_group}" if xcd_group > 1 else "")
        + "_de_" + "_".join(str(v) for v in epoch_planes)
        # 符号名里不能有 '-':负步长写成 m 前缀
        + f"_ad{'m' if arrival_delta_step < 0 else ''}{abs(arrival_delta_step)}"
        + f"_atm{int(a_tile_map_plane)}"
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel_epilog_v2(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64):
    # fmt: on
        tx_i32 = fx.thread_idx.x
        bx_i32 = fx.block_idx.x
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        # Stage-1 k1 wrote epoch_gate earlier on this stream.
        # relaxed、agent scope:epoch_gate 是同一 stream 上更早的 kernel 写的,
        # kernel 边界已保证可见。system acquire 会带 buffer_inv,每个 CTA 入口
        # 冲一次 L2,persistent GEMM2 靠 L2 复用 w2,实测 +97us。
        i64_generation = fx.Int64(rocdl.readfirstlane(
            T.i64,
            fx.Int64(comm_ops.load_i64_global_agent_relaxed(i64_generation)).ir_value()))
        _par = i64_generation & fx.Int64(1)
        arg_aq = arg_aq + _par * fx.Int64(epoch_planes[0])
        arg_ascale = arg_ascale + _par * fx.Int64(epoch_planes[1])
        arg_eids = arg_eids + _par * fx.Int64(epoch_planes[2])
        arg_cumsum = arg_cumsum + _par * fx.Int64(epoch_planes[3])
        arg_stids = arg_stids + _par * fx.Int64(epoch_planes[4])
        arg_sweights = arg_sweights + _par * fx.Int64(epoch_planes[5])
        arg_a_tile_map = arg_a_tile_map + _par * fx.Int64(a_tile_map_plane)
        arrival_delta_rt = fx.Int64(arrival_delta) + _par * fx.Int64(
            arrival_delta_step)
        p2p_shift = fx.Int32(_par) * fx.Int32(npes)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))

        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        k_bytes = fx.Int32(i32_inter) // fx.Int32(2)
        # kernel-invariant scatter resources + peer-base table (loaded into registers once).
        trb_buf = ptr_buf_tensor(arg_trb, fx.Int32)
        stids_buf = ptr_buf_tensor(arg_stids, fx.Int32)
        sweights_buf = ptr_buf_tensor(arg_sweights, fx.Float32)
        p2p_table = ptr_buf_tensor(arg_p2p_comb_inp, fx.Int64)
        if tx_i32 < fx.Int32(npes):
            peer_base = p2p_table[tx_i32 + p2p_shift]
            fx.ptr_store(
                peer_base,
                lds_typed_ptr(
                    fx.Int32(lds_peer_off) + tx_i32 * fx.Int32(8),
                    T.i64,
                    align=8,
                ),
            )

        def issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(arg_aq, lds_base_i32, slot, slot, m_row0, wave, lane,
                    False, KH_TILE_A, k_bytes, BM=BM)

        # a_tile_map:A(h1)按物理 tile 存放,排序后的第 t 个 32 行 tile 的数据在
        # 物理 tile a_tile_map[t]。只有 A/A-scale 走这层间接;expert id、srcmap、
        # 输出位置仍按排序后的行号。
        a_map_buf = ptr_buf_tensor(arg_a_tile_map, fx.Int32)

        def run_unit(unit_bx, m_block_idx, a_row):
            # Map each Stage2 BM sub-tile to its Stage1 SBM metadata row.
            m_row = m_block_idx * fx.Int32(BM)
            sort_block_idx = m_row // fx.Int32(BM)
            row_in_sort_block = m_row - sort_block_idx * fx.Int32(BM)
            srcmap_row_base = (
                trb_buf[sort_block_idx]
                + row_in_sort_block
            )
            if tx_i32 < fx.Int32(BM):
                sorted_pos = srcmap_row_base + tx_i32
                packed = stids_buf[sorted_pos]
                weight = sweights_buf[sorted_pos]
                fx.ptr_store(
                    packed,
                    lds_typed_ptr(
                        fx.Int32(lds_packed_off) + tx_i32 * fx.Int32(4),
                        T.i32,
                        align=4,
                    ),
                )
                fx.ptr_store(
                    weight,
                    lds_typed_ptr(
                        fx.Int32(lds_weight_off) + tx_i32 * fx.Int32(4),
                        T.f32,
                        align=4,
                    ),
                )
            # fmt: off
            accm_vecs, m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(lds_base_i32, arg_ascale, arg_bq,
                arg_bscale, arg_eids, arg_aq, i32_max_m_blocks, unit_bx, lane, wave, i32_inter, i32_hidden,
                0, 0, BM=BM, BN=BN, BK=BK, use_nt=False, INTER_MAX=inter_dim, aStages=aStages,
                a_dtype="fp4", has_pad=False, SBM=BM, g2_bhoist=True, g2_ascale_pf=True,
                expert_offset=0, explicit_a_row=a_row)
            p2p_scatter_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, N_OUT=N_OUT,
                BM=BM, BN=BN, npes=npes, topk=topk,
                log2_max_tok=log2_max_tok, mask_max_tok=mask_max_tok, recv_cap=recv_cap,
                comb_inp_nbytes=comb_inp_nbytes, lds_packed_off=lds_packed_off,
                lds_weight_off=lds_weight_off, lds_peer_off=lds_peer_off, scatter_vec=scatter_vec,
                gpus_per_node=gpus_per_node, arrival_delta=arrival_delta_rt,
                arrival_count_ptr=arg_arrival_count, num_n_blocks=num_n_blocks,
                generation=i64_generation)
            # fmt: on

        def run_mapped_unit(unit_bx, m_block_idx):
            m_row0 = m_block_idx * fx.Int32(BM)
            a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
            issue_all_a_loads(a_row)
            rocdl.sched_barrier(0)
            run_unit(unit_bx, m_block_idx, a_row)

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = (cumsum0 + fx.Int32(BM - 1)) // fx.Int32(BM)

        if const_expr(xcd_group > 1):
            # Workgroups go round-robin over the XCDs (bx % 8).  The m_group
            # 32-row m-blocks of one Stage1 tile group (= one expert's rows)
            # read the same w2 tile for a given n_block; with the plain
            # bx = m_slot * num_n_blocks + n order they sit num_n_blocks
            # apart and land on different XCDs, so each L2 fetches the tile
            # from HBM again.  Put them on one XCD, m_group consecutive
            # dispatches apart.  Bijective because the grid is a multiple
            # of NUM_XCD * m_group (checked at compile time).
            xcd = bx_i32 % fx.Int32(NUM_XCD)
            q = bx_i32 // fx.Int32(NUM_XCD)
            h = q % fx.Int32(xcd_group)
            gidx = (q // fx.Int32(xcd_group)) * fx.Int32(NUM_XCD) + xcd
            n_block = gidx % fx.Int32(num_n_blocks)
            m_slot = (gidx // fx.Int32(num_n_blocks)) * fx.Int32(xcd_group) + h
        else:
            m_slot = bx_i32 // fx.Int32(num_n_blocks)
            n_block = bx_i32 - m_slot * fx.Int32(num_n_blocks)
        diff = total_m_blocks - m_slot
        rem = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
        n_iters = (rem + fx.Int32(cu_num - 1)) // fx.Int32(cu_num)
        for _it in range(fx.Int32(0), n_iters, fx.Int32(1)):
            m_block = m_slot + fx.Int32(_it) * fx.Int32(cu_num)
            unit_bx = m_block * fx.Int32(num_n_blocks) + n_block
            fx.barrier()  # separate prev-iter epilog LDS reads from this iter's A-load into the LDS union
            if fx.Int32(m_block) < total_m_blocks:
                run_mapped_unit(unit_bx, m_block)

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, stream: fx.Stream):
    # fmt: on
        # Persistent grid: cu_num CTAs per N block.
        kernel_epilog_v2(
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
            arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
            i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map,
        ).launch(grid=(cu_num * (model_dim // BN), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def peer_table_launcher(gpus_per_node: int, npes: int):
    """One-off kernel filling ``arg_p2p``: each peer's plane_slot_inbox base.

    CCO's ``lsa_ptr`` is device-only (the host cannot see a peer's VA), so the
    table is written by a tiny kernel.  The push target is the GPU with the
    owner's local index (``owner_rank & (gpus_per_node - 1)``), so only the
    first ``gpus_per_node`` entries are addressed; the rest hold peer 0's base
    so an index the valid bit already rejects is never a wild pointer.
    """
    local_peers = int(gpus_per_node)
    total = int(npes)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def fill_peer_table(window: fx.Int64, table: fx.Int64, offset: fx.Int64):
        if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
            for peer in range(local_peers):
                address = cco.Window(window).lsa_ptr(fx.Int32(peer), offset)
                comm_ops.store_i64_global_relaxed(
                    table + fx.Int64(peer * 8), address)
            fallback = cco.Window(window).lsa_ptr(fx.Int32(0), offset)
            for peer in range(local_peers, total):
                comm_ops.store_i64_global_relaxed(
                    table + fx.Int64(peer * 8), fallback)

    @flyc.jit
    def launch(window: fx.Int64, table: fx.Int64, offset: fx.Int64,
               stream: fx.Stream):
        fill_peer_table(window, table, offset).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


_G2_LAUNCH_CACHE = {}


def run_mega_moe_stage2(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
                        arg_sweights, arg_trb, arg_p2p, arg_arrival_count, arg_generation,
                        arg_a_tile_map, row_capacity, stream, **compile_kw):
    """Compile (once per configuration) and launch stage2 kernel1.

    ``compile_kw`` are the keyword arguments of :func:`compile_mega_moe_stage2`.
    """
    key = tuple(sorted(compile_kw.items()))
    launch = _G2_LAUNCH_CACHE.get(key)
    if launch is None:
        launch = compile_mega_moe_stage2(**compile_kw)
        _G2_LAUNCH_CACHE[key] = launch
    ptrs = (arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
            arg_sweights, arg_trb, arg_p2p)
    _run_compiled(
        launch, *(fx.Int64(int(p)) for p in ptrs), fx.Int32((row_capacity + BM - 1) // BM),
        fx.Int32(compile_kw["inter_dim"]), fx.Int32(compile_kw["model_dim"]),
        fx.Int64(int(arg_arrival_count)), fx.Int64(int(arg_generation)),
        fx.Int64(int(arg_a_tile_map)), stream,
    )
