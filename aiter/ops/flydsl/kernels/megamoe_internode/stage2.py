# SPDX-License-Identifier: MIT
"""Stage2 of MegaMoEInternode: two kernels.

kernel1 (``megamoe_k1_*``): GEMM2 + weighted push into the node-local
plane-slot inbox.  The GEMM2 core (``gemm2_compute_v2``) is MegaMoEV2's; this file adds the
EP16 epilogue that sends each weighted BF16 row to the GPU with the owner's
local index on this node and publishes its arrival flag.

kernel2 (``megamoe_k2_*``): node 内归约 -> RAIL 发送 -> node 间归约 + unpermute 输出。
和 kernel1 是两个独立 kernel、独立 ABI,共用同一份 CCO 初始化。

三个阶段,所有 CTA 依次经过,没有固定角色分区:

  阶段 1  每个 CTA 静态认领若干 chunk。读 plane_slot_inbox,按 slot mask
          归约成每 token 的 node partial 写 node_accumulator;属于
          remote_plane 的 chunk 归约完后,**由同一个 CTA** 直接
          put(aggregate=True) 写 WQE,不敲门铃,然后给 wqe_posted[qp] 加一。
          同一个 CTA 拥有整个 chunk,所以 put 的源地址天然连续。

  阶段 2  前 NUM_QP 个 CTA 各带一个 QP(wave0):等本 QP 的 WQE 全部写入
          (wqe_posted[qp]),给对端发各 chunk 的 return_group_ready,再
          flush 一次(敲门铃)。QP 之间没有顺序要求,不需要 barrier。

  阶段 3  所有 CTA 降级为 node 间归约。节点内累加阶段 1 已完成,
          remote_partial_rx 里是对端已累加好的 partial,所以这里只做
          相加 + unpermute 写出。按 chunk 到达驱动,不做全局 barrier。

shape 一律参数化,不写死。tokens 不被 num_qp*chunk 整除时走尾块路径 ——
参考实现是靠整除约束回避尾块的("no tail transport here"),这里不保留那条。
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr, rocdl, gpu
from flydsl.expr.typing import Int8, T, BFloat16, Float32
from flydsl.runtime.device import get_rocm_arch
from ..mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr, lds_vec_load, _fabs_f32 as fabs_f32
from . import communication as comm
from ..tensor_shim import _run_compiled, buf_copy_store, ptr_buf_tensor
from .gemm import gemm2_compute_v2, issue_a_load_lds_dt, kStages
from aiter.ops.flydsl.kernels import buffer_ops
from .communication import release as _rail, ccqe_enabled as _rail_ccqe


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
        if const_expr(comb_inp_nbytes >= 1 << 31):
            # mtpr > 4096: slot * token_nbytes passes the i32 byte range. The
            # row goes into the 64-bit descriptor base (slot is wave-uniform,
            # so no waterfall) and the descriptor covers one token row.
            row_ptr = fx.Int64(peer_base) + fx.Int64(
                valid.select(slot, fx.Int32(0))
            ) * fx.Int64(token_nbytes)
            off = (valid & active).select(
                n_block_idx * fx.Int32(BN * out_elem_bytes)
                + col * fx.Int32(out_elem_bytes),
                fx.Int32(token_nbytes),
            )
            output_buf = ptr_buf_tensor(
                row_ptr,
                fx.BFloat16,
                unit_elems=scatter_vec,
                num_records_bytes=token_nbytes,
            )
        else:
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
                prev = fx.Int32(comm.atomic_add_agent(
                    arrival_count_ptr + fx.Int64(count_idx) * fx.Int64(4),
                    fx.Int32(1),
                ))
                if prev + fx.Int32(1) == num_n_blocks:
                    # Subtract rather than store 0: same effect, and it
                    # reuses the atomic that AR9 measured at +3.5us
                    # instead of a system store that carries a writeback.
                    comm.atomic_add_agent(
                        arrival_count_ptr + fx.Int64(count_idx) * fx.Int64(4),
                        fx.Int32(0) - num_n_blocks,
                    )
                    comm.store_i64_global_system_relaxed(
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
    # Past the 32-bit buffer-resource range the push addresses one token row
    # per descriptor (see p2p_scatter_epilog), so only a row must fit.
    if not 0 < comb_inp_nbytes or not model_dim * 2 < _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("MegaMoE stage2 P2P row exceeds the 32-bit buffer-resource ABI")

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
            fx.Int64(comm.load_i64_global_agent_relaxed(i64_generation)).ir_value()))
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
                    KH_TILE_A, k_bytes, BM=BM)

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
                BM=BM, BN=BN, BK=BK, INTER_MAX=inter_dim, aStages=aStages, explicit_a_row=a_row)
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
                comm.store_i64_global_relaxed(
                    table + fx.Int64(peer * 8), address)
            fallback = cco.Window(window).lsa_ptr(fx.Int32(0), offset)
            for peer in range(local_peers, total):
                comm.store_i64_global_relaxed(
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


THREADS = 256


# scratch 布局(i32 槽位)。
SCRATCH_WQE_POSTED = 0          # [num_qp]   已写入 WQE 的 chunk 数
SCRATCH_FINAL_DONE = 65         # 阶段 3 的完成计数
SCRATCH_CHUNK_DONE = 256        # [2][max_chunks] 每个 (plane,chunk) 的完成计数
SCRATCH_P3_DONE = 512           # [max_chunks] 阶段3 每个 chunk 的 CTA 完成计数
SCRATCH_WORDS = 1024            # 宿主侧 scratch 的长度,编译期据此校验越界


def _ceil_div(a, b):
    return (a + b - 1) // b


def _ceil_div_rt(a, b):
    """运行时向上取整。尾块就是从这里来的,不再要求整除。"""
    return (a + b - fx.Int32(1)) // b


def _min_rt(a, b):
    return (a < b).select(a, b)


def _max_rt(a, b):
    return (a > b).select(a, b)


@flyc.jit
def _fp8_scale_for_leader(is_leader, local_max):
    e8m0 = fx.Int32(0)
    if is_leader:
        max_bits = local_max.bitcast(fx.Int32)
        working_bits = (local_max * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(
            fx.Int32
        )
        mantissa = working_bits & fx.Int32(0x7FFFFF)
        biased_exp = (working_bits >> fx.Int32(23)) & fx.Int32(0xFF)
        e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
        e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
        e8m0 = (max_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
    return e8m0



def _fp8_encode8(vals, lane, *, VEC):
    """VEC 个 f32 -> (打包好的 i32 列表, 本块 e8m0, 本 lane 是否块 leader)。

    逐字照搬 kernel1 的 push 端(p2p_scatter_epilog)。VEC=8 时一个
    32 元素的尺度块正好跨 4 个 lane,所以块内求最大值要跨 lane^1、lane^2。
    """
    group_lanes = 32 // VEC
    local_max = fabs_f32(vals[0])
    for q in range_constexpr(1, VEC):
        local_max = local_max.maximumf(fabs_f32(vals[q]))
    max_bits = local_max.bitcast(fx.Int32)
    for xor_lane in (1, 2):
        if xor_lane < group_lanes:
            remote_bits = rocdl.ds_bpermute(
                T.i32, (lane ^ fx.Int32(xor_lane)) * fx.Int32(4), max_bits)
            local_max = local_max.maximumf(
                fx.Int32(remote_bits).bitcast(Float32))
            max_bits = local_max.bitcast(fx.Int32)
    leader_lane = lane & fx.Int32(~(group_lanes - 1))
    is_leader = (lane & fx.Int32(group_lanes - 1)) == fx.Int32(0)
    leader_e8m0 = _fp8_scale_for_leader(is_leader, local_max)
    e8m0 = fx.Int32(
        rocdl.ds_bpermute(T.i32, leader_lane * fx.Int32(4), leader_e8m0))
    block_scale = (e8m0 << fx.Int32(23)).bitcast(Float32)
    pk_ty = T.vec(2, T.i16)
    words = []
    for word in range_constexpr(VEC // 4):
        packed_word = fx.Vector.filled(2, 0, fx.Int16).ir_value()
        for pair in range_constexpr(2):
            v = word * 4 + pair * 2
            packed_word = rocdl.cvt_scalef32_pk_fp8_f32(
                pk_ty, packed_word,
                vals[v].ir_value(), vals[v + 1].ir_value(),
                block_scale.ir_value(), pair)
        words.append(fx.Vector(packed_word).bitcast(fx.Int32)[0])
    return words, e8m0, is_leader


def _fp8_decode8(packed2, scale_f32):
    """2 个 i32(8 个 e4m3 字节) + 该块的 e8m0 尺度 -> Vector(8, f32)。

    照搬 moe_reduce.py:133-140 的形态(同格式):cvt_pk_f32_fp8 一次吃一个 32 位
    源加一个 word 选择位,所以 2 个 i32 要 4 次调用才出 8 个 f32。
    """
    _v2f32 = T.vec(2, T.f32)
    w = fx.Vector(packed2)
    words = (w[0], w[0], w[1], w[1])
    lanes = []
    for pi in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_pk_f32_fp8(res=_v2f32, src=words[pi], word_sel=bool(pi & 1))
        )
        lanes.append(pair[0] * scale_f32)
        lanes.append(pair[1] * scale_f32)
    return fx.Vector.from_elements(lanes, Float32)


def compile_stage2_node_combine(
    *,
    hidden: int,
    max_tokens: int,
    topk: int,
    rank: int,
    gpus_per_node: int = 8,
    num_qp: int = 8,
    return_chunk_tokens: int = 16,
    rail_quant: str = "none",
    inbox_off: int,
    inbox_bytes: int,
    slot_mask_off: int,
    accumulator_off: int,
    accumulator_bytes: int,
    rx_off: int,
    rx_bytes: int,
    partial_ready_off: int,
    group_ready_off: int,
    consumed_off: int,
    arrival_off: int,
    s2_window_off: int,
    epoch_off: int,
    parity_planes: tuple,
):
    """构建 kernel2。所有 *_off 是相对 window local_ptr 的字节偏移。

    generation 从 window 里 stage1 的 epoch_gate(epoch_off)读(同一条 stream
    上 stage1 已写好),*_off 传 parity 0 的偏移,
    kernel 自己按 generation & 1 加上 parity_planes 里对应的平面大小。
    CUDA graph capture 后每次 replay 才能拿到当代的 generation/parity。
    parity_planes 顺序: inbox, slot_mask, accumulator, rx, partial_ready,
    group_ready, consumed, arrival。
    """
    parity_planes = tuple(int(v) for v in parity_planes)
    if epoch_off <= 0 or len(parity_planes) != 8:
        raise ValueError("kernel2 needs the epoch_gate offset and eight parity_planes")

    if hidden <= 0 or hidden % 512:
        # 512 而非 256:一个 wave 一次覆盖 64 lane x 8 bf16 = 512 列,不整除时
        # 尾部那几列不会被写。上游本来就保证整除(bench/ABI 要求 hidden 在
        # [1024,8192] 且能被 512 整除),这里显式挡住,免得直接调用时静默少写。
        raise ValueError("hidden must be a positive multiple of 512")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    # 每个 *_off 都必须是 window 相对(已含 stage2_offset)。传成 region 相对
    # 不会崩也不会报错,只会把跨节点写落到别处 —— 一定要在编译期挡掉。
    for _name in ("inbox_off", "slot_mask_off", "accumulator_off", "rx_off",
                  "partial_ready_off", "group_ready_off"):
        if int(locals()[_name]) < int(s2_window_off):
            raise ValueError(
                f"{_name} 小于 s2_window_off,看起来是 region 相对偏移;"
                "kernel2 要的是相对 window local_ptr 的偏移")
    if int(num_qp) not in (1, 2, 4, 8):
        raise ValueError("num_qp must be one of 1,2,4,8")
    # 参考实现是 qp = wave。wave 数多于 QP 时有 wave 没活干,少于 QP 时
    # 一个 wave 顺序带 num_qp//waves 个 QP —— 两者都要求整除。
    _waves = THREADS // 64
    if int(num_qp) < _waves or int(num_qp) % _waves:
        raise ValueError(
            f"num_qp({num_qp}) 必须是 wave 数({_waves}) 的整数倍")
    if int(consumed_off) < int(s2_window_off):
        raise ValueError(
            "consumed_off 小于 s2_window_off,看起来是 region 相对偏移")
    if int(arrival_off) < int(s2_window_off):
        raise ValueError(
            "arrival_off 小于 s2_window_off,看起来是 region 相对偏移")
    if return_chunk_tokens < 4:
        # return_group_ready 只有 max_tokens/4 个槽位,粒度不能比 4 更细。
        raise ValueError("return_chunk_tokens must be >= 4")

    HIDDEN = int(hidden)
    MAX_TOKENS = int(max_tokens)
    TOPK = int(topk)
    CHUNK = int(return_chunk_tokens)
    # ceil,而不是整除:CHUNK*TOPK 小于 threads 时(比如 CHUNK=8)也要跑一轮,
    # 多出来的线程靠下面的上界判断落空。之前写成整除,CHUNK 一调就编译失败。
    WAIT_SPAN = CHUNK * int(topk)
    WAIT_ITERS = _ceil_div(WAIT_SPAN, THREADS)
    NUM_QP = int(num_qp)
    WAVES = THREADS // 64
    RECORD_BYTES = HIDDEN * 2               # 一个 token 的 BF16 partial,无 header
    # inbox 的记录宽度和 RECORD_BYTES 是两回事:RECORD_BYTES 只用于 rail 的
    # put/get(每 token 一条 accumulator),inbox 是每 (token, topk slot) 一条,
    # 由 kernel1 的 token_nbytes 定宽。两者可以独立量化。
    # kernel1: row_base = slot * token_nbytes, token_nbytes = N_OUT + N_OUT//32
    if rail_quant not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(
            f"rail_quant must be 'none' or 'fp8_blockwise_1x32' "
            f"(got {rail_quant!r})")
    RAIL_FP8 = rail_quant == "fp8_blockwise_1x32"
    if RAIL_FP8:
        if HIDDEN % 32:
            raise ValueError("fp8 rail 要求 hidden 能被 32 整除(e8m0 块大小)")
        # RECORD_BYTES 是 put/get 的每 token 字节数,量化后直接减半 ——
        # rail 上 78us 的线速载荷就是被这一行砍掉的。
        RECORD_BYTES = HIDDEN + HIDDEN // 32
        if RECORD_BYTES % 8:
            raise ValueError(
                f"rail 行宽 {RECORD_BYTES} 不是 8 的倍数,put 源地址会错位")
    INBOX_ROW_BYTES = (HIDDEN * 2)
    # 每 plane 的 inbox 字节;整个 inbox(2 plane)>= 2 GiB 时按 plane 建描述符。
    INBOX_PLANE_BYTES = MAX_TOKENS * TOPK * INBOX_ROW_BYTES
    INBOX_PER_PLANE = 2 * INBOX_PLANE_BYTES >= 1 << 31
    if INBOX_PLANE_BYTES >= 1 << 31:
        raise ValueError("inbox plane exceeds the 32-bit buffer-resource range")
    ACC_ROW_BYTES = RECORD_BYTES
    if INBOX_ROW_BYTES % 8:
        # 每 lane 取 VEC 个值要落成一次 dwordx2/x4,行首必须 8 字节对齐。
        raise ValueError(
            f"inbox 行宽 {INBOX_ROW_BYTES} 不是 8 的倍数,向量化 load 会错位")
    VEC = 8                                 # 每 lane 8 个 bf16 = 16 B
    CHUNKS_PER_WAVE = HIDDEN // (64 * VEC)  # 一个 wave 覆盖 512 列
    CHUNK_ITERS = _ceil_div(CHUNKS_PER_WAVE, WAVES)
    # 阶段1 的工作单元:一个 (plane, token, hidden 分片) 由**一个 wave** 做完。
    # HPARTS 就是 CHUNKS_PER_WAVE —— 每片正好 512 列,所以没有掩码浪费
    # (按 chunk 分片的老路子是 4 wave x 2 轮 = 8 个槽装 7 份活,1/8 空转)。
    HPARTS = CHUNKS_PER_WAVE
    # chunk 完成计数:[plane][chunk],给 rail 的 put 找"最后一个写完的 wave"。
    MAX_CHUNKS = _ceil_div(MAX_TOKENS, CHUNK)
    if (SCRATCH_CHUNK_DONE + 2 * MAX_CHUNKS > SCRATCH_P3_DONE
            or SCRATCH_P3_DONE + MAX_CHUNKS > SCRATCH_WORDS):
        raise ValueError(
            f"scratch 不够:需要 {SCRATCH_CHUNK_DONE + 2 * MAX_CHUNKS} 个 i32")

    node = int(rank) // int(gpus_per_node)
    remote_node = 1 - node
    local_plane = node
    remote_plane = remote_node

    kernel_name = (
        f"megamoe_k2_h{HIDDEN}_t{MAX_TOKENS}_k{TOPK}"
        f"_qp{NUM_QP}_c{CHUNK}_w{WAVES}"
        # rail 的 CQ 轮询按 CCQE 模式编译,模式必须进 kernel 名(= flydsl 缓存键)
        + ("_ccqe" if _rail_ccqe() else "")
        + ("_qr8" if RAIL_FP8 else "")
        + f"_de{epoch_off}_" + "_".join(str(v) for v in parity_planes)
    )

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64,
        arg_scratch: fx.Int64,
        local_tokens: fx.Int32,
    ):
        bx = fx.Int32(gpu.block_id("x"))
        grid = fx.Int32(gpu.grid_dim.x)
        tx = fx.Int32(gpu.thread_id("x"))
        wave = tx // fx.Int32(64)
        lane = tx % fx.Int32(64)
        # 同 kernel1:relaxed 读(不带 buffer_inv),readfirstlane 留在 SGPR。
        generation = fx.Int64(rocdl.readfirstlane(
            T.i64,
            fx.Int64(comm.load_i64_global_agent_relaxed(
                arena_ptr + fx.Int64(epoch_off))).ir_value()))
        _par = generation & fx.Int64(1)
        _inbox_off = fx.Int64(inbox_off) + _par * fx.Int64(parity_planes[0])
        _slot_mask_off = fx.Int64(slot_mask_off) + _par * fx.Int64(parity_planes[1])
        _accumulator_off = fx.Int64(accumulator_off) + _par * fx.Int64(parity_planes[2])
        _rx_off = fx.Int64(rx_off) + _par * fx.Int64(parity_planes[3])
        _partial_ready_off = fx.Int64(partial_ready_off) + _par * fx.Int64(parity_planes[4])
        _group_ready_off = fx.Int64(group_ready_off) + _par * fx.Int64(parity_planes[5])
        _consumed_off = fx.Int64(consumed_off) + _par * fx.Int64(parity_planes[6])
        _arrival_off = fx.Int64(arrival_off) + _par * fx.Int64(parity_planes[7])

        inbox_ptr = arena_ptr + _inbox_off
        accum_ptr = arena_ptr + _accumulator_off
        rx_ptr = arena_ptr + _rx_off
        mask_ptr = arena_ptr + _slot_mask_off
        ready_ptr = arena_ptr + _partial_ready_off
        arrival_ptr = arena_ptr + _arrival_off
        group_ready_ptr = arena_ptr + _group_ready_off

        # node_dest_slot_mask 是**本 rank 的 stage1** 写的(stage1.py:891/:1487
        # 都是本地 buffer_store),和 kernel2 同 rank 同 stream 串行 —— 没有任何
        # 跨 GPU 时序,所以不需要 system scope 的绕缓存读。原来两处都用
        # load_i32_global_system_relaxed:等待循环里每个 (token,slot) 读一次
        # (同一个 token 重复 16 次),归约循环里每个工作单元读一次,加起来近
        # 10 万次不可缓存的读。
        mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
            mask_ptr, num_records_bytes=fx.Int32(2 * MAX_TOKENS * 4))
        if const_expr(not INBOX_PER_PLANE):
            inbox_rsrc = buffer_ops.create_buffer_resource_from_addr(
                inbox_ptr, num_records_bytes=fx.Int32(inbox_bytes))
        accum_rsrc = buffer_ops.create_buffer_resource_from_addr(
            accum_ptr, num_records_bytes=fx.Int32(accumulator_bytes))
        rx_rsrc = buffer_ops.create_buffer_resource_from_addr(
            rx_ptr, num_records_bytes=fx.Int32(rx_bytes))
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_output_bf16, num_records_bytes=fx.Int32(MAX_TOKENS * HIDDEN * 2))

        total_chunks = _ceil_div_rt(local_tokens, fx.Int32(CHUNK))

        @flyc.jit
        def _wait_ready_relaxed(addr, expected):
            """cco 的 wait_ready 每次轮询都是 acquire load,每条带一次 L2
            invalidate。到达标志是**批量**等待(每 chunk 512 个,还被 ~10 个 CTA
            各查一遍),绝大多数一看就已就绪 —— 为此付 8 万次 invalidate,实测
            +39us。轮询改 relaxed,acquire 由调用方在 barrier 之后做**一次**。
            """
            word = fx.Int64(comm.load_i64_global_system_relaxed(addr))
            while word < expected:
                word = fx.Int64(comm.load_i64_global_system_relaxed(addr))



        # ------------------------------------------------------------------
        # 阶段 1:归约 + 写 WQE
        # ------------------------------------------------------------------
        # 两个 plane 都要归约(本端 plane 供阶段 3 用,对端 plane 要发出去)。
        # 分片按 (plane, token, hidden tile) 到 wave;**簿记按 (CTA, chunk)**。
        #
        # 上一版把完成计数放在每个工作单元后面,代价是每单元一次
        # rocdl.s_waitcnt(0) —— 它等的是全部未完成访存,于是下一个单元的
        # load 一个都发不出去,跨单元的访存重叠被切断。隔离 kernel 里单变量
        # 量过:同样的读,加上「每单元 s_waitcnt(0) + 一次 agent 原子」就从
        # 26.4us 变成 117.2us(4.4x)。而 stride=twaves(640) 大于一个 chunk
        # 的 224 个单元,一个 wave 在每个 chunk 里只做一个单元,这笔开销连摊
        # 都摊不掉。
        #
        # 现在 CTA bx 认领 chunk (bx % total_chunks),同一个 chunk 由
        # G 个 CTA 分担,G = 该 chunk 分到的 CTA 数。完成计数数的是 **CTA**:
        # 16 个 chunk x G=10 => 160 次原子,而不是 7168 次;s_waitcnt(0) 每
        # CTA 每 chunk 一次。活跃 CTA 数仍然是整个 grid。
        #
        # 对照(同一块 arena、同样的分片和软流水,不带任何簿记)是 2.2 TB/s。
        # 【设计点:不要再试"提早发 put"】两个方向都实测过,一律没有收益:
        # 先做会发 put 的 remote plane(PLANE_FIRST)-> 200.0 vs 201.2,零;
        # 把 chunk 分成 S 级让前几级的 put 早发(RAIL_STAGES)-> S=2 回退到
        # 221.6(基线 195.6)、S=4 回退到 280.5。结论:rail 的时间是**线上
        # 时间不是发射延迟**,在发射侧做流水不会有回报 —— 剩下的杠杆只有减
        # 字节(就是现在的 rail fp8,实测 -38us)。两条代码路径已按此结论删除。
        _PLANES = (0, 1)
        ck_local = bx % total_chunks
        sub = bx // total_chunks
        n_cta = (grid - ck_local + total_chunks - fx.Int32(1)) // total_chunks
        # 必须是 range_constexpr:AST 改写器只特判它。写成
        # `for plane in _PLANES:`(普通 Python 元组循环)会让嵌在里面的动态
        # range 不被改写,编译期报 "dynamic 'ArithValue' has no Python
        # integer representation"。
        for _pi in range_constexpr(2):
          plane = _PLANES[_pi]
          if const_expr(INBOX_PER_PLANE):
              # mtpr > 4096:整个 inbox 超出 i32 的 num_records(掩码越界读不再返回 0),
              # 每个 plane 单独一个描述符,记录号按 plane 内的 token 算。
              inbox_rsrc_p = buffer_ops.create_buffer_resource_from_addr(
                  inbox_ptr + fx.Int64(plane * INBOX_PLANE_BYTES),
                  num_records_bytes=fx.Int32(INBOX_PLANE_BYTES))
          # 运行期 range:ck_local, +total_chunks, ...
          # 用它而不是 `for _st in constexpr: if ck < total_chunks:` —— 动态
          # range 落在 scf.if 生成的分支函数里不会被 AST 改写(和
          # p2p_scatter_epilog 同一个坑),编译期就炸。
          # 和分级前逐字等价。
          for ck in range(ck_local, total_chunks, total_chunks):
              ck_first = ck * fx.Int32(CHUNK)
              ck_tok = _min_rt(fx.Int32(CHUNK), local_tokens - ck_first)
              # 每 CTA 每 chunk 一次,和老的 chunk 路径同粒度(实测只要
              # +2.7us)。按工作单元等会把 acquire fence 乘 7 倍 —— 那
              # 一版实测 +86.8us。认领同一 chunk 的 G 个 CTA 会重复等,
              # 第一个之后都是已就绪的缓存命中。
              for _wi in range_constexpr(WAIT_ITERS):
                w_flat = tx + fx.Int32(_wi * THREADS)
                w_token = ck_first + w_flat // fx.Int32(TOPK)
                w_slot = w_flat % fx.Int32(TOPK)
                if (w_token < local_tokens) & (
                        w_flat < fx.Int32(WAIT_SPAN)):
                    w_index = fx.Int32(plane * MAX_TOKENS) + w_token
                    w_mask = fx.Int32(buffer_ops.buffer_load(
                        mask_rsrc, w_index, vec_width=1, dtype=T.i32))
                    if ((w_mask >> w_slot) & fx.Int32(1)) != fx.Int32(0):
                        # 标志值就是 generation,wait_ready 是单调 u64 的
                        # ">= expected",上一代留下的字更小,所以这块区域
                        # 一次都不用清零。
                        _wait_ready_relaxed(
                            arrival_ptr
                            + fx.Int64(w_index * fx.Int32(TOPK) + w_slot)
                            * fx.Int64(8), generation)
              gpu.barrier()
              comm.fence_system_acquire()
              unit0 = sub * fx.Int32(WAVES) + wave
              ustep = n_cta * fx.Int32(WAVES)
              ulo = fx.Int32(0)
              uhi = ck_tok * fx.Int32(HPARTS)
              for u in range(ulo + unit0, uhi, ustep):
                  token = ck_first + u // fx.Int32(HPARTS)
                  hpart = u - (u // fx.Int32(HPARTS)) * fx.Int32(HPARTS)
                  token_index = fx.Int32(plane * MAX_TOKENS) + token
                  slot_mask = fx.Int32(buffer_ops.buffer_load(
                      mask_rsrc, token_index, vec_width=1, dtype=T.i32))
                  col = (hpart * fx.Int32(64) + lane) * fx.Int32(VEC)
                  totals = fx.Vector.filled(VEC, 0.0, Float32)
                  # MegaMoEv2 的流水(intranode_kernel.py:1010-1085):K 个
                  # load 先全部攒进 list,一个都不当场消费,发完才累加。
                  # 有效性走掩码不走分支 —— buffer_load(mask=) 把越界偏移
                  # 设成 0x7FFFFFFF,硬件边界检查返回 0。
                  vals = []
                  for slot in range_constexpr(TOPK):
                      ok = ((slot_mask >> fx.Int32(slot))
                            & fx.Int32(1)) != fx.Int32(0)
                      if const_expr(INBOX_PER_PLANE):
                          rec = (token * fx.Int32(TOPK)
                                 + fx.Int32(slot))
                          src = rec * fx.Int32(HIDDEN) + col
                          vals.append(buffer_ops.buffer_load(
                              inbox_rsrc_p, src // fx.Int32(2),
                              vec_width=4, dtype=T.i32, mask=ok))
                      else:
                          rec = (token_index * fx.Int32(TOPK)
                                 + fx.Int32(slot))
                          src = rec * fx.Int32(HIDDEN) + col
                          vals.append(buffer_ops.buffer_load(
                              inbox_rsrc, src // fx.Int32(2),
                              vec_width=4, dtype=T.i32, mask=ok))
                  for slot in range_constexpr(TOPK):
                      totals = totals + fx.Vector(vals[slot]).bitcast(
                          BFloat16).to(Float32)
                  # 【设计点:accumulator 不写穿】ACCUM_WT=1(sc0|sc1 写穿,
                  # 省掉每 chunk 那次 fence_system_release 的整 L2 写回)实测
                  # 194.5 vs 195.6 —— 零收益,开关已删,这里恒用默认策略。
                  _acm = 0
                  if const_expr(RAIL_FP8):
                      _vals = [fx.Float32(totals[i])
                               for i in range_constexpr(VEC)]
                      _w, _e8m0, _is_ldr = _fp8_encode8(_vals, lane, VEC=VEC)
                      _abyte = (token_index * fx.Int32(ACC_ROW_BYTES) + col)
                      buffer_ops.buffer_store(
                          fx.Vector.from_elements(_w, fx.Int32),
                          accum_rsrc, _abyte // fx.Int32(4),
                          cache_modifier=_acm)
                      # 尺度字节用 mask 而不是 if:分支里做 store 没问题,
                      # 但掩码版不产生分歧,省一次 exec mask 往返。
                      buffer_ops.buffer_store(
                          _e8m0.to(fx.Int8), accum_rsrc,
                          token_index * fx.Int32(ACC_ROW_BYTES)
                          + fx.Int32(HIDDEN) + col // fx.Int32(32),
                          mask=_is_ldr, cache_modifier=_acm)
                  # 17 = sc0|sc1(SIDefines.h:593 的 pre-gfx12 编码,GLC=SC0=1、
                  # SCC=SC1=16):写穿到系统一致点。NIC 要读 node_accumulator,
                  # 默认策略下这些写是脏在本 rank 的 L2 里,得靠下面那次
                  # fence_system_release 的 buffer_wbl2(整 L2 写回)刷出去 ——
                  # 每 chunk 一次。写穿之后那次 fence 就不需要了。
                  # 和 kernel1 payload 用 pay_cm=19 是同一条路子。
                  if const_expr(not RAIL_FP8):
                      out = fx.Vector.from_elements(
                          [fx.Float32(totals[i])
                           for i in range_constexpr(VEC)],
                          Float32,
                      ).to(BFloat16).bitcast(fx.Int32)
                      buffer_ops.buffer_store(
                          out, accum_rsrc,
                          (token_index * fx.Int32(HIDDEN) + col)
                          // fx.Int32(2),
                          cache_modifier=_acm)
              # token 段模式:本 CTA 的字节范围是连续的,自己发自己的 put,
              # 不等本 chunk 最慢的那个 CTA。WQE 只是写进发送队列
              # (aggregate 不敲门铃),所以 10 个 CTA 并发写没有问题;
              # 门铃仍由最后一个到达者通过 wqe_posted[qp] 触发 —— 它的
              # acq_rel 能看见其它 9 个 CTA 在各自 s_waitcnt(0) 之后的 release,
              # 也就保证了那 10 份 WQE 都已经写好。
              # --- 每 CTA 每 chunk 一次:这里才是排空点 ---
              rocdl.s_waitcnt(0)
              gpu.barrier()
              cnt_idx = fx.Int32(SCRATCH_CHUNK_DONE + plane * MAX_CHUNKS) + ck
              if wave == fx.Int32(0):
                  prev = fx.Int32(0)
                  if lane == fx.Int32(0):
                      prev = fx.Int32(comm.atomic_add_agent_acq_rel(
                          arg_scratch + fx.Int64(cnt_idx) * fx.Int64(4),
                          fx.Int32(1)))
                  # prev 只在 lane0 有效;广播成 wave uniform,下面那个 if 才
                  # 能整 wave 一起进(put 是 warp 级 collective)。
                  prev = fx.Int32(rocdl.readfirstlane(T.i32, prev.ir_value()))
                  if (prev + fx.Int32(1)) == n_cta:
                      # 系统可见性(NIC 要读 node_accumulator)由这**一次**
                      # fence 负责,每 chunk 一次,不是每个 wave 一次。
                      # 写穿模式下 accumulator 的写本来就落到系统一致点了,
                      # 这次整 L2 写回纯属多余。
                      comm.fence_system_release()
                      if lane == fx.Int32(0):
                          comm.store_i32_system(
                              arg_scratch, cnt_idx, fx.Int32(0))
                          for _t in range_constexpr(CHUNK):
                              tk = ck_first + fx.Int32(_t)
                              if tk < local_tokens:
                                  # relaxed:上面那次 fence 已经把 writeback
                                  # 做了,release store 会再来一遍 buffer_wbl2。
                                  comm.store_i64_global_system_relaxed(
                                      ready_ptr
                                      + fx.Int64(fx.Int32(plane * MAX_TOKENS) + tk)
                                      * fx.Int64(8), generation)
                      if const_expr(plane == remote_plane):
                          _qp = ck % fx.Int32(NUM_QP)
                          _rail.put(dev_comm, _qp, fx.Int32(remote_node),
                              arena_win,
                              _rx_off
                              + fx.Int64(ck_first) * fx.Int64(RECORD_BYTES),
                              arena_win,
                              _accumulator_off
                              + fx.Int64(remote_plane * MAX_TOKENS)
                              * fx.Int64(RECORD_BYTES)
                              + fx.Int64(ck_first) * fx.Int64(RECORD_BYTES),
                              fx.Int64(ck_tok) * fx.Int64(RECORD_BYTES),
                              aggregate=True)
                          rocdl.s_waitcnt(0)
                          if lane == fx.Int32(0):
                              comm.atomic_add_agent(
                                  arg_scratch
                                  + fx.Int64(SCRATCH_WQE_POSTED) * fx.Int64(4)
                                  + fx.Int64(_qp) * fx.Int64(4),
                                  fx.Int32(1))


        # ------------------------------------------------------------------
        # 阶段 2:rail 返回
        # ------------------------------------------------------------------
        # 整段只在一个 CTA 里跑,qp = wave。这样 gpu.barrier() 是合法的(CTA 的
        # 每个 wave 都会走到),"本批 WQE 已全部写入"就是一次 barrier,不需要
        # 任何跨 CTA 计数器。NUM_QP 多于 wave 数时,一个 wave 顺序带几个 QP。
        # 一个 CTA 带一个 QP。当初整段收在 CTA0 里是因为"本批 WQE 已全部
        # 写入"想用一次 gpu.barrier() 表达;但阶段1 各 CTA 写的
        # SCRATCH_WQE_POSTED[qp] 已经是跨 CTA 计数器,而 QP 之间没有顺序
        # 要求 —— 那个理由不成立了。这里不用任何 barrier。
        if bx < fx.Int32(NUM_QP):
          if wave == fx.Int32(0):
           _want = _max_rt(
               (total_chunks - bx + fx.Int32(NUM_QP - 1))
               // fx.Int32(NUM_QP), fx.Int32(0))
           _posted = fx.Int32(comm.load_i32_global_system_relaxed(
               arg_scratch + fx.Int64(SCRATCH_WQE_POSTED) * fx.Int64(4)
               + fx.Int64(bx) * fx.Int64(4)))
           while _posted < _want:
               _posted = fx.Int32(comm.load_i32_global_system_relaxed(
                   arg_scratch + fx.Int64(SCRATCH_WQE_POSTED) * fx.Int64(4)
                   + fx.Int64(bx) * fx.Int64(4)))
           comm.fence_system_acquire()
           for chunk in range(bx, total_chunks, fx.Int32(NUM_QP)):
               _rail.put_value(
                   dev_comm, bx, fx.Int32(remote_node), arena_win,
                   _group_ready_off + fx.Int64(chunk) * fx.Int64(8),
                   generation, aggregate=True)
           # 原来是 flush_async 拿 request、下一行立刻 wait_request —— 中间
           # 什么都没做,拆成两步没有收益。同步 flush 就是它。
           _rail.flush_peer(dev_comm, bx, fx.Int32(remote_node))
           if lane == fx.Int32(0):
               comm.store_i32_system(
                   arg_scratch, fx.Int32(SCRATCH_WQE_POSTED) + bx,
                   fx.Int32(0))

        # ------------------------------------------------------------------
        # 阶段 3:所有 CTA 做 node 间归约 + unpermute
        # ------------------------------------------------------------------
        # 节点内累加已在阶段 1 完成,remote_partial_rx 里也是对端累加好的,
        # 所以这里只是两者相加再按 token 序写出。
        # 和阶段1 同一个形状。老写法是"一个 CTA 一个 token",每个 token 一次
        # gpu.barrier() + fence_system_acquire() + s_waitcnt(0) + 一次
        # **system scope** acq_rel 原子(每条自带整 L2 写回,kernel1 上量过
        # ~370ns/条),512 个 token 就是 512 次,全撞在 L2 上;而且
        # CHUNK_ITERS=2 只用掉 7/8 个 wave 槽。实测 29.4us,而这段只搬
        # 11 MB —— 按探针量到的 2.2 TB/s 应该是 5us。
        #
        # 现在:CTA 认领 chunk,同步降到每 CTA 每 chunk 一次;chunk 内的
        # (token, hidden tile) 按 wave 铺开(HPARTS=7 整除,没有掩码空转)。
        for ck in range(bx % total_chunks, total_chunks, grid):
          ck_first = ck * fx.Int32(CHUNK)
          ck_tok = _min_rt(fx.Int32(CHUNK), local_tokens - ck_first)
          if tx == fx.Int32(0):
              # ready 位是阶段1 按整个 chunk 一起发布的,所以等首 token 即可,
              # 不用逐个等 32 次。
              _wait_ready_relaxed(
                  ready_ptr
                  + fx.Int64(fx.Int32(local_plane * MAX_TOKENS) + ck_first)
                  * fx.Int64(8), generation)
              _wait_ready_relaxed(
                  group_ready_ptr + fx.Int64(ck) * fx.Int64(8), generation)
          gpu.barrier()
          comm.fence_system_acquire()
          n_cta = (grid - ck + total_chunks - fx.Int32(1)) // total_chunks
          sub = bx // total_chunks
          for u in range(sub * fx.Int32(WAVES) + wave,
                         ck_tok * fx.Int32(HPARTS),
                         n_cta * fx.Int32(WAVES)):
              token = ck_first + u // fx.Int32(HPARTS)
              hpart = u - (u // fx.Int32(HPARTS)) * fx.Int32(HPARTS)
              col = (hpart * fx.Int32(64) + lane) * fx.Int32(VEC)
              local_index = ((fx.Int32(local_plane * MAX_TOKENS) + token)
                             * fx.Int32(HIDDEN) + col)
              out_index = token * fx.Int32(HIDDEN) + col
              if const_expr(RAIL_FP8):
                  _lrow = ((fx.Int32(local_plane * MAX_TOKENS) + token)
                           * fx.Int32(ACC_ROW_BYTES))
                  _rrow = token * fx.Int32(ACC_ROW_BYTES)
                  local_packed = buffer_ops.buffer_load(
                      accum_rsrc, (_lrow + col) // fx.Int32(4),
                      vec_width=2, dtype=T.i32)
                  remote_packed = buffer_ops.buffer_load(
                      rx_rsrc, (_rrow + col) // fx.Int32(4),
                      vec_width=2, dtype=T.i32)
                  _lsc = buffer_ops.buffer_load(
                      accum_rsrc,
                      _lrow + fx.Int32(HIDDEN) + col // fx.Int32(32),
                      vec_width=1, dtype=T.i8)
                  _rsc = buffer_ops.buffer_load(
                      rx_rsrc,
                      _rrow + fx.Int32(HIDDEN) + col // fx.Int32(32),
                      vec_width=1, dtype=T.i8)
                  local_values = _fp8_decode8(
                      local_packed,
                      ((fx.Uint8(_lsc).to(fx.Uint32).bitcast(fx.Int32))
                       << fx.Int32(23)).bitcast(Float32))
                  remote_values = _fp8_decode8(
                      remote_packed,
                      ((fx.Uint8(_rsc).to(fx.Uint32).bitcast(fx.Int32))
                       << fx.Int32(23)).bitcast(Float32))
              else:
                  local_packed = buffer_ops.buffer_load(
                      accum_rsrc, local_index // fx.Int32(2),
                      vec_width=4, dtype=T.i32)
                  remote_packed = buffer_ops.buffer_load(
                      rx_rsrc, out_index // fx.Int32(2),
                      vec_width=4, dtype=T.i32)
                  local_values = fx.Vector(local_packed).bitcast(
                      BFloat16).to(Float32)
                  remote_values = fx.Vector(remote_packed).bitcast(
                      BFloat16).to(Float32)
              result = fx.Vector.from_elements(
                  [fx.Float32(local_values[i]) + fx.Float32(remote_values[i])
                   for i in range_constexpr(VEC)],
                  Float32,
              ).to(BFloat16).bitcast(fx.Int32)
              buffer_ops.buffer_store(
                  result, out_rsrc, out_index // fx.Int32(2))
          # 每 CTA 每 chunk 一次。FINAL_DONE 是**本 GPU 内**的计数(CTA0 在
          # 收尾里读它),所以增量走 agent scope 就够 —— 系统可见性只在把信用
          # 发给对端那一刻才需要,那里本来就有一次 release。
          rocdl.s_waitcnt(0)
          gpu.barrier()
          if tx == fx.Int32(0):
              prev = fx.Int32(comm.atomic_add_agent_acq_rel(
                  arg_scratch + fx.Int64(SCRATCH_P3_DONE + ck) * fx.Int64(4),
                  fx.Int32(1)))
              if (prev + fx.Int32(1)) == n_cta:
                  comm.store_i32_system(
                      arg_scratch, fx.Int32(SCRATCH_P3_DONE) + ck, fx.Int32(0))
                  comm.atomic_add_system_acq_rel(
                      arg_scratch + fx.Int64(SCRATCH_FINAL_DONE) * fx.Int64(4),
                      ck_tok)
        rocdl.s_waitcnt(0)
        gpu.barrier()


        # ------------------------------------------------------------------
        # 收尾:return_consumed 信用交换
        # ------------------------------------------------------------------
        # remote_partial_rx 只有 2 个 parity 槽,第 N 代和第 N+2 代共用同一块。
        # 没有这个握手,两端代数可以自由漂移;而 wait_ready 是 >= 语义,对端跑快
        # 两代时它写的新 ready 会同时满足我们对旧代的等待 —— 不挂死,直接读到
        # 写了一半的 buffer。所以这一步不是可选的优化。
        if bx == fx.Int32(0):
          done_ptr = arg_scratch + fx.Int64(SCRATCH_FINAL_DONE) * fx.Int64(4)
          if tx == fx.Int32(0):
              done = fx.Int32(comm.load_i32_global_system(done_ptr))
              while done < local_tokens:
                  done = fx.Int32(comm.load_i32_global_system(done_ptr))
              # 本代计数已满,减回去,下一代重新从 0 开始。
              comm.atomic_add_system_acq_rel(
                  done_ptr, fx.Int32(0) - local_tokens)
          gpu.barrier()
          if wave == fx.Int32(0):
              # "你发给我的那块我读完了" -> 对端可以复用它的发送源。
              _rail.put_value(dev_comm, fx.Int32(0), fx.Int32(remote_node),
                              arena_win, _consumed_off, generation,
                              aggregate=True)
              _rail.flush_peer(dev_comm, fx.Int32(0), fx.Int32(remote_node))
              if lane == fx.Int32(0):
                  # 反过来等对端的信用:确认我们写进它 rx 的那块已被读完,
                  # 下一代才可以再往同一个 parity 槽写。system-acquire 是
                  # 必须的:这个字是对端 NIC 写的。
                  _credit_ptr = arena_ptr + _consumed_off
                  _credit = fx.Int64(
                      comm.load_i64_global_system(_credit_ptr))
                  while _credit < generation:
                      _credit = fx.Int64(
                          comm.load_i64_global_system(_credit_ptr))

    @flyc.jit
    def launch(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64,
        arg_scratch: fx.Int64,
        local_tokens: fx.Int32,
        blocks: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch, local_tokens,
            value_attrs={"rocdl.flat_work_group_size": f"{THREADS},{THREADS}"},
        ).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.kernel_name = kernel_name
    return launch


_CACHE = {}


def run_stage2_node_combine(
    dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
    local_tokens, blocks, stream, **compile_kw,
):
    """Compile (once per configuration) and launch stage2 kernel2."""
    key = tuple(sorted(compile_kw.items()))
    launch = _CACHE.get(key)
    if launch is None:
        launch = _CACHE[key] = compile_stage2_node_combine(**compile_kw)
    launch(
        fx.Int64(int(dev_comm)), fx.Int64(int(arena_win)),
        fx.Int64(int(arena_ptr)), fx.Int64(int(arg_output_bf16)),
        fx.Int64(int(arg_scratch)), fx.Int32(int(local_tokens)),
        fx.Int32(int(blocks)), stream,
    )
