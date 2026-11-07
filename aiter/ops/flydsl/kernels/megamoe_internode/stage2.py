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
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Float32, Int8, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops

from .. import dpp_utils
from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from ..mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr, lds_vec_load
from ..tensor_shim import _run_compiled, buf_copy_load, buf_copy_store, ptr_buf_tensor
from . import communication as comm
from .communication import ccqe_enabled as _rail_ccqe
from .communication import rail as _rail
from .gemm import gemm2_compute_v2, issue_a_load_lds_dt, kStages

_BUFFER_OFFSET_ABI_BYTES = 1 << 31
# Tile shape: 32-row M tiles (one Stage1 tile each), K step 256.
BM = 32
BK = 256
# Workgroups are dispatched round-robin over the 8 XCDs (SPX mode).
NUM_XCD = 8


# fmt: off
def p2p_scatter_epilog(lds_acc_base, accm, n_block_idx, wave, lane, *, N_OUT, BM, BN, npes, topk,
    log2_max_tok, mask_max_tok, recv_cap, comb_inp_nbytes, lds_packed_off, lds_weight_off,
    lds_peer_off, scatter_vec, gpus_per_node, lds_rowptr_off=None):
# fmt: on
    """CShuffle one GEMM2 tile into weighted BF16 rows and scatter them to peers.

    With ``lds_rowptr_off`` (SDMA stage2) each row's destination was resolved
    before the GEMM: an absolute row address in LDS (0 = no row), pointing at
    the local send buffer, this GPU's own landing rows, or -- past the send
    buffer -- the peer's landing row directly.
    """
    kMChunks = BM // 16
    numAccN = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)

    # SIDefines.h (pre-gfx12): SC0=1, NT=2, SC1=16. System write-through
    # makes payload stores peer-visible once vmcnt retires and avoids evicting w2.
    # Publication relies on this policy; a dirty-L2 payload would need a fence.
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
        if const_expr(lds_rowptr_off is not None):
            rp = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(fx.ptr_load(lds_typed_ptr(
                fx.Int32(lds_rowptr_off) + row * fx.Int32(8), T.i64, align=8))).ir_value()))
            has_row = rp != fx.Int64(0)
            active = lane < fx.Int32(BN // scatter_vec)
            col = active.select(lane * fx.Int32(scatter_vec), fx.Int32(0))
            idx0 = row * fx.Int32(BN) + col
            v8 = fx.Vector(lds_vec_load(lds_acc_base, idx0 * fx.Int32(4),
                fx.Vector.make_type(scatter_vec, fx.Float32), fx.Float32, align=16))
            # SDMA rows are FP8 (1x32 blocks, E8M0 scales after the H bytes):
            # half the bytes for the copy engines.  A 32-column block spans
            # 4 lanes; every lane runs the encode (it shuffles across lanes).
            vals = [fx.Float32(v8[i] * weight) for i in range_constexpr(scatter_vec)]
            words, e8m0 = _fp8_encode8_quad(vals)
            is_ldr = (lane & fx.Int32(3)) == fx.Int32(0)
            # A row without a destination keeps a mapped base (peer table entry
            # 0); its stores are masked off.
            base0 = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(fx.ptr_load(lds_typed_ptr(
                fx.Int32(lds_peer_off), T.i64, align=8))).ir_value()))
            row8 = N_OUT + N_OUT // 32
            rsrc = buffer_ops.create_buffer_resource_from_addr(
                has_row.select(rp, base0), num_records_bytes=fx.Int32(row8))
            colg = n_block_idx * fx.Int32(BN) + col
            ok = has_row & active
            buffer_ops.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc,
                colg // fx.Int32(4), mask=ok, cache_modifier=pay_cm)
            buffer_ops.buffer_store(e8m0.to(fx.Int8), rsrc,
                fx.Int32(N_OUT) + colg // fx.Int32(32), mask=ok & is_ldr,
                cache_modifier=pay_cm)
        else:
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


    # Drain this unit's write-through payload before the CTA moves on; kernel1
    # publishes arrival once per kernel (see the end of kernel_epilog_v2), and
    # that publish relies on every CTA's stores having retired.
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)


# fmt: off
# Words after kernel1's arrival counts for its per-n_block m-block tickets and
# done count; the host allocates them with the arrival counts.
K1_CLAIM_WORDS = 64
# Kernel1 CTAs resident per CU (91 VGPRs, 4 waves -> 5 per CU); the dynamic
# claim launches at most this many per CU.
K1_CTAS_PER_CU = 5
# i32 words the host allocates for the SDMA stage2 plan buffer.
SDMA_PLAN_WORDS_MAX = 1024


def compile_mega_moe_stage2(*, model_dim: int, inter_dim: int, experts: int, topk: int, npes: int,
    max_tok: int, recv_cap: int, comb_inp_nbytes: int, BN: int, cu_num: int, gpus_per_node: int,
    arrival_delta: int, arrival_delta_step: int, epoch_planes: tuple, a_tile_map_plane: int,
    m_group: int = 1, dynamic_claim: bool = False, resident_ctas: int = 0,
    fused: bool = False, remote_first: bool = False, local_experts: int = 0,
    tile_group: int = 1, expert_count_plane: int = 0, sdma: bool = False,
    sdma_chunk_tiles: int = 32, sdma_send_rows: int = 0, sdma_frac_q64: int = 64,
    sdma_put_bytes: int = 4 << 20, sdma_arrival_woff: int = 0, sdma_arrival_plane: int = 0,
    sdma_max_chunks: int = 0, gemm2_bm: int = 32, sdma_ramp_chunks: int = 0):
# fmt: on
    """Compile the persistent GEMM2 + node push kernel (stage2 kernel1).

    ``dynamic_claim`` makes each CTA take m-blocks from a per-n_block ticket
    instead of a fixed stride (ignored when the XCD remap is on); the grid is
    then capped at ``resident_ctas`` so no CTA starts after the work is gone.

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
    # GEMM2 M tile: GBM = 32*GS rows = GS consecutive 32-row Stage1 tiles of one
    # tile group (same expert, physically adjacent A/A-scale).  Tickets, the
    # remote-first tables, a_tile_map, the SDMA plan and chunks stay in 32-row
    # tiles; only the GEMM, its row metadata and the epilogue use GBM.
    GBM = int(gemm2_bm)
    GS = GBM // BM
    if GBM % BM or not 1 <= GS <= 4:
        raise ValueError(f"gemm2_bm must be 32, 64, 96 or 128, got {GBM}")
    if GS > 1 and (not (fused and remote_first) or int(tile_group) % GS):
        raise ValueError(f"gemm2_bm {GBM} needs the fused remote-first kernel and GS | tile_group")

    def _gs(x):
        return x if GS == 1 else x * fx.Int32(GS)
    log2_max_tok = max_tok.bit_length() - 1
    mask_max_tok = max_tok - 1
    N_OUT = model_dim
    aStages = kStages + 1
    KH_TILE_A = BK // 2  # FP4: two values per byte
    # The f32 CShuffle slab and the A stages share one LDS union.
    compute_lds_bytes = max(GBM * BN * 4, aStages * GBM * KH_TILE_A)
    lds_packed_off = compute_lds_bytes
    lds_weight_off = lds_packed_off + GBM * 4
    lds_peer_off = lds_weight_off + GBM * 4
    num_n_blocks_c = model_dim // BN
    xcd_group = int(m_group)
    if fused:
        # The fused kernel runs kernel2 on these CTAs once their GEMM2 work is
        # done, so every CTA must be resident from the start: a CTA still
        # waiting for a slot behind spinning kernel2 CTAs never gets one.
        if resident_ctas <= 0:
            raise ValueError("fused stage2 needs resident_ctas > 0")
        cu_num = min(int(cu_num), max(1, int(resident_ctas) // num_n_blocks_c))
    if xcd_group > 1 and (cu_num * num_n_blocks_c) % (NUM_XCD * xcd_group) != 0:
        xcd_group = 1  # remap would not be a bijection; keep the plain order
    # Without the XCD remap, each CTA keeps its static n_block (so its w2 column
    # slice stays hot across m-blocks) but claims m-blocks from that n_block's
    # ticket instead of a fixed stride: m-blocks differ in cost (per-expert row
    # counts), so a static split leaves a long tail of idle CUs at large TPR.
    # num_n_blocks tickets and a done count sit right after the arrival counts
    # (recv_cap * topk words); the last CTA to finish zeroes them.
    dynamic = bool(dynamic_claim) and xcd_group == 1
    if fused:
        # A persistent static stride is badly imbalanced (K3 TPR 1024: +22%),
        # so the fused kernel always claims m-blocks.
        dynamic, xcd_group = True, 1
    if dynamic and resident_ctas <= 0:
        raise ValueError("dynamic_claim needs resident_ctas > 0")
    lds_claim_off = lds_peer_off + npes * 8  # two i32 slots, double-buffered
    lds_bytes = lds_claim_off + (16 if dynamic else 0)
    # Remote-first claim (dynamic only): per local expert its local/remote tile
    # counts and the prefix sums that map a ticket to an m-block.
    remote_first = bool(remote_first) and dynamic
    LE = int(local_experts)
    if remote_first and LE <= 0:
        raise ValueError("remote_first needs local_experts")
    tab_nl = lds_bytes
    tab_nr = tab_nl + 4 * (LE + 1)
    tab_base = tab_nr + 4 * (LE + 1)
    tab_rp = tab_base + 4 * (LE + 1)
    tab_lp = tab_rp + 4 * (LE + 1)
    if remote_first:
        lds_bytes = tab_lp + 4 * (LE + 1)
    # SDMA stage2 (fused, remote-first only): see _sdma_plan / _sdma_comm.
    sdma = bool(sdma)
    if sdma and not (fused and remote_first):
        raise ValueError("sdma stage2 needs the fused remote-first kernel")
    SD_CT = int(sdma_chunk_tiles)
    SD_ROWS = int(sdma_send_rows)
    # Share of each peer's rows that goes by SDMA, in 1/64: the rest of that
    # peer's rows are stored over xGMI by the epilogue, so both paths spread
    # over every link (a cut by send-buffer position would pile the xGMI share
    # onto the highest-numbered peers).
    SD_FQ = int(sdma_frac_q64)
    if not 0 <= SD_FQ <= 64:
        raise ValueError("sdma_frac_q64 must be in [0, 64]")
    SD_RB = model_dim + model_dim // 32  # one FP8 output row + its E8M0 scales
    SD_CTR_B, SD_SIG_B, SD_INV_B = 128, 136, 256  # bytes into a peer's arrival region
    # Plan CTAs: each lays out a contiguous share of the tickets, the last to
    # finish adds the per-CTA offsets (one CTA alone took 142us at K3 TPR4096).
    SD_PLAN_CTAS = 32
    # i32 words of the plan buffer: [0,8) seg_off, [8,16) rbase, [16,24) rows
    # per dest, [24,32) of them by SDMA; i64 at 32: done generation; 40: plan-CTA arrival count;
    # [64, 64+8P) per-CTA row counts; [64+8P, 64+16P) per-CTA offsets.
    SD_W_CNT, SD_W_CTOT = 40, 64
    SD_W_COFF = SD_W_CTOT + 8 * SD_PLAN_CTAS
    SD_PLAN_WORDS = SD_W_COFF + 8 * SD_PLAN_CTAS
    if SD_PLAN_WORDS > SDMA_PLAN_WORDS_MAX:
        raise ValueError("sdma plan words exceed the host buffer")
    SD_CDONE_STRIDE = 32  # i32 words (128 B) between chunk-done counters
    # Chunk-done counters live in the window (the copy engine polls them): in
    # this parity's arrival region, after inv.  All puts are posted right after
    # the plan as engine-side waits + puts, one doorbell per peer.
    SD_PUT_B = int(sdma_put_bytes)
    SD_AWOFF, SD_APLANE, SD_MAXCH = int(sdma_arrival_woff), int(sdma_arrival_plane), int(sdma_max_chunks)
    SD_CDONE_B = (256 + 2 * max_tok * topk * 4 + 255) // 256 * 256
    if sdma:
        if SD_CT <= 0 or SD_ROWS <= 0:
            raise ValueError("sdma needs sdma_chunk_tiles > 0 and sdma_send_rows > 0")
        if gpus_per_node != 8:
            raise ValueError("sdma stage2 assumes 8 GPUs per node")
        if model_dim % 256 or SD_RB % 8:
            raise ValueError("sdma FP8 rows need hidden % 256 == 0 and 8-byte rows")
        if SD_AWOFF <= 0 or SD_MAXCH <= 0 or SD_PUT_B <= 0:
            raise ValueError("sdma needs the arrival window offset, max chunks and put bytes")
        if SD_CDONE_B + SD_MAXCH * SD_CDONE_STRIDE * 4 > SD_APLANE:
            raise ValueError("sdma chunk counters do not fit the arrival region")
        if SD_CT % GS:
            raise ValueError("sdma chunks must hold whole GEMM2 tile groups")
        lds_bytes = (lds_bytes + 7) // 8 * 8
        lds_rowptr_off = lds_bytes            # GBM i64 row destinations
        lds_splan_off = lds_rowptr_off + GBM * 8  # 32 i32: seg_off, rbase, rows, sdma rows (8 each)
        lds_coff_off = lds_splan_off + 32 * 4     # per plan-CTA offsets, 8 i32 each
        lds_bytes = lds_coff_off + SD_PLAN_CTAS * 8 * 4
        # The plan's scratch reuses the GEMM LDS union (it runs before the
        # CTA's first unit): per-thread totals, per-group sums, dest totals.
        SP_TOT, SP_GS, SP_SUM = 0, 256 * 8 * 4, 256 * 8 * 4 + 32 * 8 * 4
        SP_LAST, SP_RUN = 255, 192  # i32 slots in SP_GS: "last plan CTA", per-peer run start
        if SP_SUM + 8 * 4 > compute_lds_bytes:
            raise ValueError("sdma plan scratch does not fit the GEMM LDS union")
    SEARCH_STEPS = []
    _st = 1
    while _st * 2 < max(LE, 1):
        _st *= 2
    while _st >= 1:
        SEARCH_STEPS.append(_st)
        _st //= 2
    if lds_bytes > 160 * 1024:
        raise ValueError(f"Stage2 LDS use {lds_bytes} exceeds 160 KiB")
    # CTAs that fit on a CU (160 KiB LDS); a bigger GEMM2 tile lowers it, and
    # the persistent grid must shrink with it (a CTA that never becomes
    # resident behind spinning ones hangs the fused kernel).
    K1_OCC = max(1, min(K1_CTAS_PER_CU, (160 * 1024) // lds_bytes))
    res_eff = int(resident_ctas) // K1_CTAS_PER_CU * K1_OCC
    if fused:
        cu_num = min(int(cu_num), max(1, res_eff // num_n_blocks_c))
    # Past the 32-bit buffer-resource range the push addresses one token row
    # per descriptor (see p2p_scatter_epilog), so only a row must fit.
    if not 0 < comb_inp_nbytes or not model_dim * 2 < _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("MegaMoE stage2 P2P row exceeds the 32-bit buffer-resource ABI")

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    claim_word = 0  # arg_arrival_count holds only kernel1's K1_CLAIM_WORDS words
    # Dynamic: only resident CTAs, so none launches after the tickets run out
    # just to bump the same two atomics (+200us at K3 TPR 1024 with 7168 CTAs).
    ctas_per_nb = min(cu_num, max(1, res_eff // num_n_blocks_c)) if dynamic else cu_num
    grid_ctas = ctas_per_nb * num_n_blocks_c
    # n-block tickets, the done count, and (remote-first) the remote-half count.
    if num_n_blocks_c + 2 > K1_CLAIM_WORDS:
        raise ValueError(f"k1 needs {num_n_blocks_c + 2} claim words, has {K1_CLAIM_WORDS}")
    # SDMA: the last CTA only drives the copy engine and the one before it
    # plans the send layout before joining GEMM2; only GEMM CTAs count done.
    # SDMA: the last SD_PLAN_CTAS CTAs plan the send layout, then join GEMM2.
    SD_PLAN_BX = grid_ctas - SD_PLAN_CTAS
    n_workers = grid_ctas
    if sdma and grid_ctas < SD_PLAN_CTAS + 1:
        raise ValueError("sdma stage2 grid too small for its plan and comm CTAs")
    # SDMA chunk shape per half (remote, then local): SD_NS small chunks of one
    # compute round (the tickets the resident CTAs hold at once, which finish
    # together), big SD_CTB chunks in the middle, SD_NS small ones at the end.
    # The head starts the copy engine early, the tail shortens what is left to
    # ship once the half's GEMM2 is done (and so delays the signal and the rail
    # less); the middle keeps the engine waits (~14us each, serialised) few.
    # A half no longer than 2*SD_NS small chunks is all small.  SD_NS = 0 is
    # the uniform SD_CT split.
    SD_NS = int(sdma_ramp_chunks)
    SD_S = ctas_per_nb * GS if SD_NS else SD_CT
    SD_CTB = max(SD_CT, SD_S)
    if sdma and (SD_NS < 0 or SD_S % GS or SD_CTB % GS):
        raise ValueError("sdma ramp chunks must be >= 0 and hold whole GEMM2 tile groups")

    kernel_name = (
        f"megamoe_k1_t{GBM}x{BN}x{BK}_cu{cu_num}"
        + (f"_xg{xcd_group}" if xcd_group > 1 else "")
        + (f"_dynp{ctas_per_nb}" if dynamic else "")
        + (f"_rf{LE}g{int(tile_group)}" if remote_first else "")
        + (f"_sd{SD_CTB}r{SD_ROWS}q{SD_FQ}p{SD_PUT_B >> 10}" if sdma else "")
        + (f"s{SD_S}n{SD_NS}" if sdma and SD_NS else "")
        + "_de_" + "_".join(str(v) for v in epoch_planes)
        # 符号名里不能有 '-':负步长写成 m 前缀
        + f"_ad{'m' if arrival_delta_step < 0 else ''}{abs(arrival_delta_step)}"
        + f"_atm{int(a_tile_map_plane)}"

    )

    # The body is a device function so the fused stage2 kernel can inline it
    # with its own block index; kernel_epilog_v2 below is the standalone form.
    # fmt: off
    @flyc.jit
    def k1_body(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32, bx_i32: fx.Int32,
        arg_expert_count=None, boundary_hook=None, dev_comm=None, arg_sd_send=None,
        arg_sd_tb=None, arg_sd_plan=None, arg_sd_cdone=None, arena_win=None, arena_ptr=None):
    # fmt: on
        tx_i32 = fx.thread_idx.x


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
                    KH_TILE_A, k_bytes, BM=GBM)

        # a_tile_map:A(h1)按物理 tile 存放,排序后的第 t 个 32 行 tile 的数据在
        # 物理 tile a_tile_map[t]。只有 A/A-scale 走这层间接;expert id、srcmap、
        # 输出位置仍按排序后的行号。
        a_map_buf = ptr_buf_tensor(arg_a_tile_map, fx.Int32)
        if const_expr(sdma):
            sd_tb_buf = ptr_buf_tensor(arg_sd_tb, fx.Int32)

        def _sdma_row_target(packed, ticket):
            # Lanes 0..BM-1 of wave 0, one row each: where this row goes.  Rows
            # for dest p are numbered within the tile in row order (ballot), so
            # the plan's per-(tile, dest) prefix plus that rank is the row's
            # place in p's landing area and, for a peer, in our send buffer.
            packed = fx.Int32(packed)
            t24 = packed & fx.Int32(0x00FFFFFF)
            s = packed.shrui(fx.Int32(24)) & fx.Int32(0xFF)
            owner = t24 >> fx.Int32(log2_max_tok)
            valid = (t24 < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (owner < fx.Int32(npes))
            dpe = valid.select(owner & fx.Int32(gpus_per_node - 1), fx.Int32(0))
            slot = (((owner // fx.Int32(gpus_per_node)) * fx.Int32(1 << log2_max_tok)
                     + (t24 & fx.Int32(mask_max_tok))) * fx.Int32(topk) + s)
            rank = fx.Int32(0)
            upper = lane >= fx.Int32(32)  # lanes 32..63 hold the next 32-row subtile
            for _q in range_constexpr(gpus_per_node):
                pred = valid & (dpe == fx.Int32(_q))
                bal = fx.Int64(rocdl.ballot(T.i64, pred.ir_value()))
                below = fx.Int32(rocdl.mbcnt_lo(
                    T.i32, fx.Int32(bal).ir_value(), fx.Int32(0).ir_value()))
                if const_expr(GS > 1):
                    below_hi = fx.Int32(rocdl.mbcnt_hi(
                        T.i32, fx.Int32(bal >> fx.Int64(32)).ir_value(), fx.Int32(0).ir_value()))
                    below = upper.select(below_hi, below)
                rank = pred.select(below, rank)
            if const_expr(GS > 1):
                ticket = ticket + tx_i32 // fx.Int32(BM)
            tb = _sd_gtb(ticket, dpe)
            seg = fx.Int32(fx.ptr_load(lds_typed_ptr(
                fx.Int32(lds_splan_off) + dpe * fx.Int32(4), T.i32, align=4)))
            rbase = fx.Int32(fx.ptr_load(lds_typed_ptr(
                fx.Int32(lds_splan_off + 32) + dpe * fx.Int32(4), T.i32, align=4)))
            peer = fx.Int64(fx.ptr_load(lds_typed_ptr(
                fx.Int32(lds_peer_off) + dpe * fx.Int32(8), T.i64, align=8)))
            land = rbase + tb + rank
            srow = seg + tb + rank
            sd_cap = fx.Int32(fx.ptr_load(lds_typed_ptr(
                fx.Int32(lds_splan_off + 96) + dpe * fx.Int32(4), T.i32, align=4)))
            to_send = ((dpe != fx.Int32(i32_local_rank)) & (tb + rank < sd_cap)
                       & (srow < fx.Int32(SD_ROWS)))
            rowptr = to_send.select(
                fx.Int64(arg_sd_send) + fx.Int64(srow) * fx.Int64(SD_RB),
                peer + fx.Int64(land) * fx.Int64(SD_RB))
            fx.ptr_store(valid.select(rowptr, fx.Int64(0)), lds_typed_ptr(
                fx.Int32(lds_rowptr_off) + tx_i32 * fx.Int32(8), T.i64, align=8))
            if valid:
                # inv: the peer's (plane, token, slot) -> landing row.  Visible
                # with the arrival flags, which follow every CTA's drained stores.
                comm.store_i32_global_system_relaxed(
                    peer + arrival_delta_rt + fx.Int64(SD_INV_B)
                    + fx.Int64(slot) * fx.Int64(4), land)

        def run_unit(unit_bx, m_block_idx, a_row, ticket=None):
            # Row metadata (source token/slot, weight) sits at the A row's
            # physical arena row: the GBM rows of a unit are GS consecutive
            # 32-row tiles of one tile group, physically adjacent like A.
            if tx_i32 < fx.Int32(GBM):
                sorted_pos = a_row + tx_i32
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
                if const_expr(sdma):
                    _sdma_row_target(packed, ticket)
            # fmt: off
            accm_vecs, _m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(lds_base_i32, arg_ascale, arg_bq,
                arg_bscale, arg_eids, arg_aq, i32_max_m_blocks, unit_bx, lane, wave, i32_inter, i32_hidden,
                BM=GBM, BN=BN, BK=BK, INTER_MAX=inter_dim, aStages=aStages, explicit_a_row=a_row,
                eids_stride=GS)
            p2p_scatter_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, N_OUT=N_OUT,
                BM=GBM, BN=BN, npes=npes, topk=topk,
                log2_max_tok=log2_max_tok, mask_max_tok=mask_max_tok, recv_cap=recv_cap,
                comb_inp_nbytes=comb_inp_nbytes, lds_packed_off=lds_packed_off,
                lds_weight_off=lds_weight_off, lds_peer_off=lds_peer_off, scatter_vec=scatter_vec,
                gpus_per_node=gpus_per_node, lds_rowptr_off=lds_rowptr_off if sdma else None)
            # fmt: on

        def run_mapped_unit(unit_bx, m_block_idx):
            m_row0 = m_block_idx * fx.Int32(BM)
            a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
            issue_all_a_loads(a_row)
            rocdl.sched_barrier(0)
            run_unit(unit_bx, m_block_idx, a_row)

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        total_m_blocks = (cumsum0 + fx.Int32(BM - 1)) // fx.Int32(BM)

        if const_expr(dynamic):
            claim_ptr = arg_arrival_count + fx.Int64(claim_word * 4)
            dyn_n_block = bx_i32 % fx.Int32(num_n_blocks_c)
            ticket_ptr = claim_ptr + fx.Int64(dyn_n_block) * fx.Int64(4)

            def claim():
                # Thread 0 takes the next m-block of this CTA's n_block; the CTA
                # reads it back from LDS.
                if tx_i32 == fx.Int32(0):
                    fx.ptr_store(
                        fx.Int32(comm.atomic_add_agent(ticket_ptr, fx.Int32(1))),
                        lds_typed_ptr(fx.Int32(lds_claim_off), T.i32, align=4),
                    )
                fx.barrier()
                got = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
                    lds_typed_ptr(fx.Int32(lds_claim_off), T.i32, align=4)).ir_value()))
                fx.barrier()
                return got

            if const_expr(not remote_first):
                # The next ticket is fetched while this unit runs: thread 0 issues
                # the atomic before the A loads and parks the result in the other
                # LDS slot after them, so its latency hides behind the loads and
                # each unit keeps the single barrier the static loop had.
                m_block = claim()
                slot = fx.Int32(1)
                while m_block < total_m_blocks:
                    nxt = fx.Int32(0)
                    if tx_i32 == fx.Int32(0):
                        nxt = fx.Int32(comm.atomic_add_agent(ticket_ptr, fx.Int32(1)))
                    m_row0 = m_block * fx.Int32(BM)
                    a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
                    issue_all_a_loads(a_row)
                    if tx_i32 == fx.Int32(0):
                        fx.ptr_store(
                            nxt,
                            lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32, align=4),
                        )
                    rocdl.sched_barrier(0)
                    run_unit(m_block * num_n_blocks + dyn_n_block, m_block, a_row)
                    fx.barrier()  # epilog LDS reads vs next A-load; also publishes the slot
                    m_block = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
                        lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32,
                                      align=4)).ir_value()))
                    slot = fx.Int32(1) - slot

            if const_expr(remote_first):
                # Tickets map remote-source m-blocks first, then local ones.
                # Stage1's sorted view is, per local expert e, its local tiles
                # then its remote tiles (each rounded up to the tile group), so
                # ticket -> m-block is a search over per-expert prefix sums.
                ec_base = arg_expert_count + _par * fx.Int64(expert_count_plane)
                ec_buf = ptr_buf_tensor(ec_base, fx.Int32)

                def _alloc(c):
                    n = (c + fx.Int32(BM - 1)) // fx.Int32(BM)
                    if const_expr(int(tile_group) > 1):
                        g = fx.Int32(int(tile_group))
                        n = ((n + g - fx.Int32(1)) // g) * g
                    return n

                def _lds(off, i):
                    return lds_typed_ptr(fx.Int32(off) + i * fx.Int32(4), T.i32, align=4)

                if tx_i32 < fx.Int32(LE):
                    fx.ptr_store(_alloc(ec_buf[tx_i32]), _lds(tab_nl, tx_i32))
                    fx.ptr_store(_alloc(ec_buf[tx_i32 + fx.Int32(LE)]), _lds(tab_nr, tx_i32))
                fx.barrier()
                if tx_i32 < fx.Int32(LE + 1):
                    sl = fx.Int32(0)
                    sr = fx.Int32(0)
                    for _e in range_constexpr(LE):
                        before = fx.Int32(_e) < tx_i32
                        sl = sl + before.select(fx.Int32(fx.ptr_load(_lds(tab_nl, fx.Int32(_e)))), fx.Int32(0))
                        sr = sr + before.select(fx.Int32(fx.ptr_load(_lds(tab_nr, fx.Int32(_e)))), fx.Int32(0))
                    fx.ptr_store(sl + sr, _lds(tab_base, tx_i32))
                    fx.ptr_store(sr, _lds(tab_rp, tx_i32))
                    fx.ptr_store(sl, _lds(tab_lp, tx_i32))
                fx.barrier()
                remote_total = fx.Int32(rocdl.readfirstlane(
                    T.i32, fx.ptr_load(_lds(tab_rp, fx.Int32(LE))).ir_value()))

                def tmap(t, uniform=True):
                    rem = t < remote_total
                    tt = rem.select(t, t - remote_total)
                    pre = rem.select(fx.Int32(tab_rp), fx.Int32(tab_lp))
                    lo = fx.Int32(0)
                    for _si in range_constexpr(len(SEARCH_STEPS)):
                        cand = lo + fx.Int32(SEARCH_STEPS[_si])
                        safe = (cand < fx.Int32(LE)).select(cand, fx.Int32(LE - 1))
                        v = fx.Int32(fx.ptr_load(lds_typed_ptr(pre + safe * fx.Int32(4), T.i32, align=4)))
                        lo = ((cand < fx.Int32(LE)) & (v <= tt)).select(cand, lo)
                    pv = fx.Int32(fx.ptr_load(lds_typed_ptr(pre + lo * fx.Int32(4), T.i32, align=4)))
                    m = (fx.Int32(fx.ptr_load(_lds(tab_base, lo)))
                         + rem.select(fx.Int32(fx.ptr_load(_lds(tab_nl, lo))), fx.Int32(0))
                         + (tt - pv))
                    if const_expr(not uniform):
                        return m
                    return fx.Int32(rocdl.readfirstlane(T.i32, m.ir_value()))

                if const_expr(sdma):
                    # Chunk geometry of one half of L tickets (see SD_NS).
                    # Recomputed where used, like _sd_tpc (no live VGPRs).
                    def _sd_geo(L):
                        """(all_small, tail start, middle chunk count, chunks)."""
                        S, B = fx.Int32(SD_S), fx.Int32(SD_CTB)
                        head = fx.Int32(SD_NS * SD_S)
                        all_small = L <= fx.Int32(2 * SD_NS * SD_S)
                        ts = L - head
                        nmid = all_small.select(fx.Int32(0), (ts - head + B - fx.Int32(1)) // B)
                        nc = all_small.select((L + S - fx.Int32(1)) // S,
                                              fx.Int32(2 * SD_NS) + nmid)
                        return all_small, ts, nmid, nc

                    def _sd_chunk_of(off, L):
                        # Chunk (within the half) of ticket offset off.
                        all_small, ts, nmid, _ = _sd_geo(L)
                        head = fx.Int32(SD_NS * SD_S)
                        in_head = all_small | (off < head)
                        in_tail = off >= ts
                        c_head = off // fx.Int32(SD_S)
                        c_mid = fx.Int32(SD_NS) + (off - head) // fx.Int32(SD_CTB)
                        c_tail = fx.Int32(SD_NS) + nmid + (off - ts) // fx.Int32(SD_S)
                        return in_head.select(c_head, in_tail.select(c_tail, c_mid))

                    def _sd_chunk_start(ci, L):
                        # First ticket offset of chunk ci; the chunk count gives L.
                        all_small, ts, nmid, _nc = _sd_geo(L)
                        head = fx.Int32(SD_NS * SD_S)
                        s_head = ci * fx.Int32(SD_S)
                        s_mid = head + (ci - fx.Int32(SD_NS)) * fx.Int32(SD_CTB)
                        s_tail = ts + (ci - fx.Int32(SD_NS) - nmid) * fx.Int32(SD_S)
                        st = (all_small | (ci < fx.Int32(SD_NS))).select(
                            s_head, (ci < fx.Int32(SD_NS) + nmid).select(s_mid, s_tail))
                        return _min_rt(st, L)

                    def _sd_chunk_small(ci, L):
                        all_small, _, nmid, _ = _sd_geo(L)
                        return all_small | (ci < fx.Int32(SD_NS)) | (ci >= fx.Int32(SD_NS) + nmid)

                    def _sd_lloc():
                        return _max_rt(total_m_blocks - remote_total, fx.Int32(0))

                    def _sd_nrc():
                        return _sd_geo(remote_total)[3]
                    # Recomputed where used: kept live across the GEMM loop they spilled a VGPR.
                    def _sd_tpc():
                        return _max_rt((total_m_blocks + fx.Int32(SD_PLAN_CTAS - 1))
                                       // fx.Int32(SD_PLAN_CTAS), fx.Int32(1))

                    def _sd_gtb(tk, p):
                        # Rows for dest p in tickets before tk: the plan CTA's local
                        # prefix plus that CTA's offset; tk == total is the dest total.
                        at_end = tk == total_m_blocks
                        pcx = _min_rt(tk // _sd_tpc(), fx.Int32(SD_PLAN_CTAS - 1))
                        loc = fx.Int32(sd_tb_buf[at_end.select(fx.Int32(0), tk) * fx.Int32(8) + p])
                        off = fx.Int32(fx.ptr_load(lds_typed_ptr(
                            fx.Int32(lds_coff_off) + (pcx * fx.Int32(8) + p) * fx.Int32(4),
                            T.i32, align=4)))
                        tot = fx.Int32(fx.ptr_load(lds_typed_ptr(
                            fx.Int32(lds_splan_off + 64) + p * fx.Int32(4), T.i32, align=4)))
                        return at_end.select(tot, loc + off)

                    def _sd_cdone_woff(c):
                        # Window offset of chunk c's counter (this parity).
                        return (fx.Int64(SD_AWOFF + SD_CDONE_B) + _par * fx.Int64(SD_APLANE)
                                + fx.Int64(c) * fx.Int64(SD_CDONE_STRIDE * 4))

                    def _sd_cdone(c):
                        return fx.Int64(arena_ptr) + _sd_cdone_woff(c)

                    def _sd_nch():
                        return _sd_nrc() + _sd_geo(_sd_lloc())[3]

                    def _sd_lds_i32(off, i):
                        return fx.Int32(fx.ptr_load(lds_typed_ptr(
                            fx.Int32(off) + i * fx.Int32(4), T.i32, align=4)))

                    def _sd_lds_put(off, i, v):
                        fx.ptr_store(fx.Int32(v), lds_typed_ptr(
                            fx.Int32(off) + i * fx.Int32(4), T.i32, align=4))

                    def _sd_peer(p):
                        return fx.Int64(fx.ptr_load(lds_typed_ptr(
                            fx.Int32(lds_peer_off) + p * fx.Int32(8), T.i64, align=8)))

                    def _sdma_plan():
                        """One CTA lays out the send buffer before any GEMM2 row is
                        stored: per (ticket, dest) the exclusive prefix of rows
                        (tile_base, plus the end entry), per dest its run in the
                        send buffer (seg_off) and its landing base on the peer
                        (rbase, one remote atomic per dest).  Thread tx owns
                        tickets [tx*k, tx*k+k); counts go to tile_base and a
                        two-level LDS scan turns them into prefixes in place."""
                        n_tk = total_m_blocks
                        pc = bx_i32 - fx.Int32(SD_PLAN_BX)
                        c_base = pc * _sd_tpc()
                        n_loc = _min_rt(_max_rt(n_tk - c_base, fx.Int32(0)), _sd_tpc())
                        kper = (n_loc + fx.Int32(255)) // fx.Int32(256)
                        t_lo = c_base + tx_i32 * kper
                        t_hi = _min_rt(t_lo + kper, c_base + n_loc)
                        stids4 = ptr_buf_tensor(arg_stids, fx.Int32, unit_elems=4)
                        for _q in range_constexpr(8):
                            _sd_lds_put(SP_TOT, tx_i32 * fx.Int32(8) + fx.Int32(_q), fx.Int32(0))
                        for tk in range(t_lo, t_hi, fx.Int32(1)):
                            r0 = a_map_buf[tmap(tk, uniform=False)] * fx.Int32(32)
                            dsts = []
                            for _j in range_constexpr(BM // 4):
                                v4 = fx.Vector(buf_copy_load(
                                    stids4, r0 // fx.Int32(4) + fx.Int32(_j), fx.Int32, unit_elems=4))
                                for _e in range_constexpr(4):
                                    pk = fx.Int32(v4[_e])
                                    t24 = pk & fx.Int32(0x00FFFFFF)
                                    sk = pk.shrui(fx.Int32(24)) & fx.Int32(0xFF)
                                    own = t24 >> fx.Int32(log2_max_tok)
                                    ok = ((t24 < fx.Int32(recv_cap)) & (sk < fx.Int32(topk))
                                          & (own < fx.Int32(npes)))
                                    dsts.append(ok.select(own & fx.Int32(7), fx.Int32(8)))
                            for _q in range_constexpr(8):
                                cq = fx.Int32(0)
                                for _d in dsts:
                                    cq = cq + (_d == fx.Int32(_q)).select(fx.Int32(1), fx.Int32(0))
                                buf_copy_store(sd_tb_buf, tk * fx.Int32(8) + fx.Int32(_q), cq, fx.Int32)
                                ti = tx_i32 * fx.Int32(8) + fx.Int32(_q)
                                _sd_lds_put(SP_TOT, ti, _sd_lds_i32(SP_TOT, ti) + cq)
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        fx.barrier()
                        # Level 1: 32 groups of 8 threads, per dest.
                        gq = tx_i32 & fx.Int32(7)
                        gg = tx_i32 >> fx.Int32(3)
                        gsum = fx.Int32(0)
                        for _i in range_constexpr(8):
                            gsum = gsum + _sd_lds_i32(
                                SP_TOT, (gg * fx.Int32(8) + fx.Int32(_i)) * fx.Int32(8) + gq)
                        _sd_lds_put(SP_GS, gg * fx.Int32(8) + gq, gsum)
                        fx.barrier()
                        if tx_i32 < fx.Int32(8):
                            run = fx.Int32(0)
                            for _g in range_constexpr(32):
                                gv = _sd_lds_i32(SP_GS, fx.Int32(_g * 8) + tx_i32)
                                _sd_lds_put(SP_GS, fx.Int32(_g * 8) + tx_i32, run)
                                run = run + gv
                            _sd_lds_put(SP_SUM, tx_i32, run)
                        fx.barrier()
                        # Level 2: this thread's exclusive prefix per dest.
                        gfirst = tx_i32 - gq
                        exq = []
                        for _q in range_constexpr(8):
                            e = _sd_lds_i32(SP_GS, gg * fx.Int32(8) + fx.Int32(_q))
                            for _i in range_constexpr(7):
                                e = e + (fx.Int32(_i) < gq).select(_sd_lds_i32(
                                    SP_TOT, (gfirst + fx.Int32(_i)) * fx.Int32(8) + fx.Int32(_q)),
                                    fx.Int32(0))
                            exq.append(e)
                        fx.barrier()
                        for _q in range_constexpr(8):
                            _sd_lds_put(SP_TOT, tx_i32 * fx.Int32(8) + fx.Int32(_q), exq[_q])
                        for tk in range(t_lo, t_hi, fx.Int32(1)):
                            for _q in range_constexpr(8):
                                ti = tx_i32 * fx.Int32(8) + fx.Int32(_q)
                                cq = fx.Int32(sd_tb_buf[tk * fx.Int32(8) + fx.Int32(_q)])
                                e = _sd_lds_i32(SP_TOT, ti)
                                buf_copy_store(sd_tb_buf, tk * fx.Int32(8) + fx.Int32(_q), e,
                                               fx.Int32, cache_modifier=19)
                                _sd_lds_put(SP_TOT, ti, e + cq)
                        plan_buf = ptr_buf_tensor(arg_sd_plan, fx.Int32)
                        if tx_i32 < fx.Int32(8):
                            buf_copy_store(plan_buf, fx.Int32(SD_W_CTOT) + pc * fx.Int32(8) + tx_i32,
                                           _sd_lds_i32(SP_SUM, tx_i32), fx.Int32, cache_modifier=19)
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        fx.barrier()
                        # The last plan CTA to arrive turns the per-CTA counts into
                        # offsets and finishes the plan.
                        if tx_i32 == fx.Int32(0):
                            prev = fx.Int32(comm.atomic_add_agent_acq_rel(
                                fx.Int64(arg_sd_plan) + fx.Int64(SD_W_CNT * 4), fx.Int32(1)))
                            _sd_lds_put(SP_GS, fx.Int32(0), prev)
                        fx.barrier()
                        last = fx.Int32(rocdl.readfirstlane(T.i32, _sd_lds_i32(
                            SP_GS, fx.Int32(0)).ir_value())) == fx.Int32(SD_PLAN_CTAS - 1)
                        if tx_i32 == fx.Int32(0):
                            _sd_lds_put(SP_GS, fx.Int32(SP_LAST), last.select(fx.Int32(1), fx.Int32(0)))
                        if last:
                            # This parity's counters were last read by the copy
                            # engine two generations ago (done before any peer
                            # finished that kernel2).
                            for c in range(tx_i32, _sd_nch(), fx.Int32(256)):
                                comm.store_i32_global_system_relaxed(_sd_cdone(c), fx.Int32(0))
                        if last & (tx_i32 < fx.Int32(8)):
                            run = fx.Int32(0)
                            for _c in range_constexpr(SD_PLAN_CTAS):
                                cv = fx.Int32(comm.load_i32_global_system_relaxed(
                                    fx.Int64(arg_sd_plan)
                                    + fx.Int64((SD_W_CTOT + _c * 8) * 4) + fx.Int64(tx_i32) * fx.Int64(4)))
                                buf_copy_store(plan_buf, fx.Int32(SD_W_COFF + _c * 8) + tx_i32, run,
                                               fx.Int32, cache_modifier=19)
                                run = run + cv
                            _sd_lds_put(SP_SUM, tx_i32, run)
                        fx.barrier()
                        if last & (tx_i32 < fx.Int32(8)):
                            n_p = _sd_lds_i32(SP_SUM, tx_i32)
                            seg = fx.Int32(0)
                            for _q in range_constexpr(8):
                                n_q = _sd_lds_i32(SP_SUM, fx.Int32(_q))
                                seg = seg + ((fx.Int32(_q) < tx_i32)
                                             & (fx.Int32(_q) != fx.Int32(i32_local_rank))).select(
                                    (n_q * fx.Int32(SD_FQ)) >> fx.Int32(6), fx.Int32(0))
                            sd_n = (tx_i32 != fx.Int32(i32_local_rank)).select(
                                (n_p * fx.Int32(SD_FQ)) >> fx.Int32(6), fx.Int32(0))
                            # Landing rows on dest tx: claimed once per (us, dest);
                            # the dest zeroes its counter at the end of its kernel2.
                            rb = fx.Int32(comm.atomic_add_system(
                                _sd_peer(tx_i32) + arrival_delta_rt + fx.Int64(SD_CTR_B), n_p))
                            plan_buf = ptr_buf_tensor(arg_sd_plan, fx.Int32)
                            buf_copy_store(plan_buf, tx_i32, seg, fx.Int32, cache_modifier=19)
                            buf_copy_store(plan_buf, tx_i32 + fx.Int32(8), rb, fx.Int32,
                                           cache_modifier=19)
                            buf_copy_store(plan_buf, tx_i32 + fx.Int32(16), n_p, fx.Int32,
                                           cache_modifier=19)
                            buf_copy_store(plan_buf, tx_i32 + fx.Int32(24), sd_n, fx.Int32,
                                           cache_modifier=19)
                        if last & (tx_i32 == fx.Int32(0)):
                            # Nobody else touches the count again this forward.
                            buf_copy_store(plan_buf, fx.Int32(SD_W_CNT), fx.Int32(0), fx.Int32,
                                           cache_modifier=19)
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        fx.barrier()
                        if last & (tx_i32 == fx.Int32(0)):
                            comm.fence_system_release()
                            comm.store_i64_global_system_relaxed(
                                fx.Int64(arg_sd_plan) + fx.Int64(128), i64_generation)

                    def _sdma_signal(sd, p):
                        # 4 bytes into the peer's spare word, only to carry the
                        # trailing signal: the queue is FIFO, so it lands after
                        # every put before it.
                        sd.put_ptr(p, _sd_peer(p) + arrival_delta_rt + fx.Int64(SD_SIG_B),
                                   fx.Int64(arg_sd_plan), fx.Int64(4), fx.Int32(0),
                                   remote_signal=True, aggregate=True)

                    def _sdma_post():
                        """Post the whole forward's SDMA work at once, one lane per
                        peer: per chunk an engine-side wait (counter >= its units),
                        and a put each time ~SD_PUT_B bytes for that peer have
                        built up (or at the remote-half end / the last chunk); a
                        signal after the remote half and one at the end; then one
                        doorbell.  No CU polls anything: the copy engine starts a
                        put the moment its chunks are counted."""
                        p = tx_i32
                        if (p < fx.Int32(gpus_per_node)) & (p != fx.Int32(i32_local_rank)):
                            sd = cco.DevComm(dev_comm).sdma()
                            nrc = _sd_nrc()
                            nch = _sd_nch()
                            seg = _sd_lds_i32(lds_splan_off, p)
                            rb = _sd_lds_i32(lds_splan_off + 32, p)
                            cap = _sd_lds_i32(lds_splan_off + 96, p)
                            _sd_lds_put(SP_GS, fx.Int32(SP_RUN) + p, fx.Int32(0))
                            if nrc == fx.Int32(0):
                                _sdma_signal(sd, p)
                            for c in range(fx.Int32(0), nch, fx.Int32(1)):
                                rem = c < nrc
                                hl = rem.select(remote_total, _sd_lloc())
                                hb = rem.select(fx.Int32(0), remote_total)
                                ci = rem.select(c, c - nrc)
                                c_t0 = hb + _sd_chunk_start(ci, hl)
                                c_t1 = hb + _sd_chunk_start(ci + fx.Int32(1), hl)
                                need = (c_t1 - c_t0) * fx.Int32(num_n_blocks_c)
                                sd.post_wait_mem(p, fx.Int32(0), arena_win, _sd_cdone_woff(c), need,
                                                 cmp=cco.SdmaCmp.GE, aggregate=True)
                                hi = _min_rt(_sd_gtb(c_t1, p), cap)
                                run = _sd_lds_i32(SP_GS, fx.Int32(SP_RUN) + p)
                                # Small chunks (head / tail of a half) each get their own put.
                                edge = ((c == nrc - fx.Int32(1)) | (c == nch - fx.Int32(1))
                                        | _sd_chunk_small(ci, hl))
                                if edge | ((hi - run) * fx.Int32(SD_RB) >= fx.Int32(SD_PUT_B)):
                                    s_lo = seg + run
                                    s_hi = _min_rt(seg + hi, fx.Int32(SD_ROWS))
                                    if s_hi > s_lo:
                                        sd.put_ptr(
                                            p, _sd_peer(p) + fx.Int64(rb + run) * fx.Int64(SD_RB),
                                            fx.Int64(arg_sd_send) + fx.Int64(s_lo) * fx.Int64(SD_RB),
                                            fx.Int64(s_hi - s_lo) * fx.Int64(SD_RB), fx.Int32(0),
                                            aggregate=True)
                                    _sd_lds_put(SP_GS, fx.Int32(SP_RUN) + p, hi)
                                    if c == nrc - fx.Int32(1):
                                        _sdma_signal(sd, p)
                            _sdma_signal(sd, p)
                            sd.commit(p, fx.Int32(0))

                    is_plan = bx_i32 >= fx.Int32(SD_PLAN_BX)
                    if is_plan:
                        _sdma_plan()
                    if tx_i32 == fx.Int32(0):
                        pd = fx.Int64(comm.load_i64_global_system_relaxed(
                            fx.Int64(arg_sd_plan) + fx.Int64(128)))
                        while pd < i64_generation:
                            fx.rocdl.s_sleep(1)
                            pd = fx.Int64(comm.load_i64_global_system_relaxed(
                                fx.Int64(arg_sd_plan) + fx.Int64(128)))
                    fx.barrier()
                    if tx_i32 < fx.Int32(32):
                        _sd_lds_put(lds_splan_off, tx_i32, comm.load_i32_global_system_relaxed(
                            fx.Int64(arg_sd_plan) + fx.Int64(tx_i32) * fx.Int64(4)))
                    if tx_i32 < fx.Int32(SD_PLAN_CTAS * 8):
                        _sd_lds_put(lds_coff_off, tx_i32, comm.load_i32_global_system_relaxed(
                            fx.Int64(arg_sd_plan) + fx.Int64(SD_W_COFF * 4)
                            + fx.Int64(tx_i32) * fx.Int64(4)))
                    fx.barrier()
                    if is_plan & (fx.Int32(rocdl.readfirstlane(T.i32, _sd_lds_i32(
                            SP_GS, fx.Int32(SP_LAST)).ir_value())) == fx.Int32(1)):
                        _sdma_post()
                        fx.barrier()  # the poster's LDS run slots vs the first A load

                    def _sd_count(tk, local_half):
                        # After the unit's barrier: every wave drained its stores.
                        if tx_i32 == fx.Int32(0):
                            if const_expr(local_half):
                                c = _sd_nrc() + _sd_chunk_of(tk - remote_total, _sd_lloc())
                            else:
                                c = _sd_chunk_of(tk, remote_total)
                            # System scope: the copy engine reads the counter
                            # from memory, not from an XCD's L2.
                            comm.atomic_add_system(_sd_cdone(c), fx.Int32(GS))

                t = _gs(claim())
                slot = fx.Int32(1)
                while t < remote_total:
                    nxt = fx.Int32(0)
                    if tx_i32 == fx.Int32(0):
                        nxt = fx.Int32(comm.atomic_add_agent(ticket_ptr, fx.Int32(1)))
                    m_block = tmap(t)
                    m_row0 = m_block * fx.Int32(BM)
                    a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
                    issue_all_a_loads(a_row)
                    if tx_i32 == fx.Int32(0):
                        fx.ptr_store(
                            nxt,
                            lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32, align=4),
                        )
                    rocdl.sched_barrier(0)
                    run_unit((m_block // fx.Int32(GS) if GS > 1 else m_block) * num_n_blocks
                             + dyn_n_block, m_block, a_row, ticket=t)
                    fx.barrier()  # epilog LDS reads vs next A-load; also publishes the slot
                    if const_expr(sdma):
                        _sd_count(t, False)
                    t = _gs(fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
                        lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32,
                                      align=4)).ir_value())))
                    slot = fx.Int32(1) - slot
                # This CTA has no remote m-block left (its held ticket is local or
                # past the end).  The last CTA here tells the node peers that the
                # remote-source half of kernel1 has been pushed (word 8 + local
                # rank), so they can reduce and ship the remote plane while the
                # local half is still in GEMM2.
                if tx_i32 == fx.Int32(0):
                    rdone = fx.Int32(comm.atomic_add_agent(
                        arg_arrival_count + fx.Int64((claim_word + num_n_blocks_c + 1) * 4),
                        fx.Int32(1)))
                    if rdone == fx.Int32(n_workers - 1):
                        comm.fence_system_release()
                        for _p in range_constexpr(gpus_per_node):
                            peer = fx.Int64(fx.ptr_load(lds_typed_ptr(
                                fx.Int32(lds_peer_off + _p * 8), T.i64, align=8)))
                            comm.store_i64_global_system_relaxed(
                                peer + arrival_delta_rt
                                + fx.Int64(i32_local_rank + fx.Int32(gpus_per_node)) * fx.Int64(8),
                                i64_generation)
                if const_expr(boundary_hook is not None):
                    boundary_hook()
                while t < total_m_blocks:
                    nxt = fx.Int32(0)
                    if tx_i32 == fx.Int32(0):
                        nxt = fx.Int32(comm.atomic_add_agent(ticket_ptr, fx.Int32(1)))
                    m_block = tmap(t)
                    m_row0 = m_block * fx.Int32(BM)
                    a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
                    issue_all_a_loads(a_row)
                    if tx_i32 == fx.Int32(0):
                        fx.ptr_store(
                            nxt,
                            lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32, align=4),
                        )
                    rocdl.sched_barrier(0)
                    run_unit((m_block // fx.Int32(GS) if GS > 1 else m_block) * num_n_blocks
                             + dyn_n_block, m_block, a_row, ticket=t)
                    fx.barrier()  # epilog LDS reads vs next A-load; also publishes the slot
                    if const_expr(sdma):
                        _sd_count(t, True)
                    t = _gs(fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
                        lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32,
                                      align=4)).ir_value())))
                    slot = fx.Int32(1) - slot
        if const_expr(not dynamic):
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

        # Arrival is published once per kernel, not per row: every token's
        # top-k rows are spread over all experts, so a consumer could not start
        # a chunk before essentially all of kernel1 anyway.  Each CTA drained
        # its payload stores after every unit; the last CTA to finish clears
        # the claim words, then tells each node peer (word = our local rank in
        # its arrival region, value = generation, so nothing needs clearing).
        # Only CTAs that had work count: the static grid can be far larger
        # than the resident set, and idle CTAs arriving last would serialize
        # on the counter after the real work (DSv4.1 TPR 512: +40~60us).
        def _count_and_publish(target):
            done_ptr = arg_arrival_count + fx.Int64((claim_word + num_n_blocks_c) * 4)
            done = fx.Int32(comm.atomic_add_agent(done_ptr, fx.Int32(1)))
            if done == target - fx.Int32(1):
                for _w in range_constexpr(num_n_blocks_c + (2 if remote_first else 1)):
                    comm.store_i32_system(
                        arg_arrival_count + fx.Int64(claim_word * 4), fx.Int32(_w), fx.Int32(0))
                comm.fence_system_release()
                for _p in range_constexpr(gpus_per_node):
                    peer = fx.Int64(fx.ptr_load(lds_typed_ptr(
                        fx.Int32(lds_peer_off + _p * 8), T.i64, align=8)))
                    comm.store_i64_global_system_relaxed(
                        peer + arrival_delta_rt
                        + fx.Int64(i32_local_rank) * fx.Int64(8),
                        i64_generation)

        fx.barrier()
        # Keep constexpr and runtime guards separate for FlyDSL specialization.
        if const_expr(dynamic):  # noqa: SIM102
            if tx_i32 == fx.Int32(0):
                _count_and_publish(fx.Int32(n_workers))
        if const_expr(not dynamic):
            # m_slot spans [0, cu_num) with num_n_blocks CTAs each; with no
            # m-block at all CTA 0 (m_slot 0) publishes alone.
            no_work = total_m_blocks == fx.Int32(0)
            busy_slots = (total_m_blocks < fx.Int32(cu_num)).select(
                total_m_blocks, fx.Int32(cu_num))
            target = no_work.select(fx.Int32(1), busy_slots * fx.Int32(num_n_blocks_c))
            takes_part = (m_slot < total_m_blocks) | (no_work & (bx_i32 == fx.Int32(0)))
            if (tx_i32 == fx.Int32(0)) & takes_part:
                _count_and_publish(target)


    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel_epilog_v2(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32):
    # fmt: on
        k1_body(
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
            arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
            i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map, i32_local_rank,
            fx.block_idx.x)

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32,
        stream: fx.Stream):
    # fmt: on
        # Persistent grid: cu_num CTAs per N block.
        kernel_epilog_v2(
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
            arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
            i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map, i32_local_rank,
        ).launch(grid=(grid_ctas, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = kernel_name
    launch.body = k1_body
    launch.grid_ctas = grid_ctas
    launch.remote_first = remote_first
    launch.sd_plan_bx = SD_PLAN_BX
    launch.ctas_per_cu = K1_OCC
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
                address = fx.Int64(cco.Window(window).lsa_ptr(fx.Int32(peer), offset))
                comm.store_i64_global_relaxed(
                    table + fx.Int64(peer * 8), address)
            fallback = fx.Int64(cco.Window(window).lsa_ptr(fx.Int32(0), offset))
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
                        arg_a_tile_map, row_capacity, local_rank, stream, arg_expert_count=0,
                        **compile_kw):
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
        fx.Int64(int(arg_a_tile_map)), fx.Int32(int(local_rank)), stream,
    )


THREADS = 256


# scratch 布局(i32 槽位)。
SCRATCH_WQE_POSTED = 0          # [num_qp]   已写入 WQE 的 chunk 数
SCRATCH_RAIL_REQ = 16           # [num_qp] i64:part 1 敲门铃的 request,part 2 回收
SCRATCH_SDMA_EPOCH = 48         # SDMA stage2: forwards finished (signal running totals)
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


def _fp8_encode8_quad(vals):
    """_fp8_encode8 for a block held by one lane quad, without LDS shuffles.

    The 4 lanes of a 32-column block are a DPP quad: two quad_perm max steps
    (xor 1, xor 2) give every lane the block max in registers, and every lane
    derives the same E8M0 itself (the formula of _fp8_scale_for_leader), so
    nothing is broadcast.  Bit-identical output; 0 ds_bpermute instead of 3.
    """
    local_max = fabs_f32(vals[0])
    for q in range_constexpr(1, len(vals)):
        local_max = local_max.maximumf(fabs_f32(vals[q]))
    for off in (1, 2):
        local_max = local_max.maximumf(fx.Float32(dpp_utils.dpp_xor_f32(local_max, off)))
    max_bits = local_max.bitcast(fx.Int32)
    working_bits = (local_max * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
    mantissa = working_bits & fx.Int32(0x7FFFFF)
    biased_exp = (working_bits >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    e8m0 = (max_bits == fx.Int32(0)).select(fx.Int32(0), e8m0)
    block_scale = (e8m0 << fx.Int32(23)).bitcast(Float32)
    pk_ty = T.vec(2, T.i16)
    words = []
    for word in range_constexpr(len(vals) // 4):
        packed_word = fx.Vector.filled(2, 0, fx.Int16).ir_value()
        for pair in range_constexpr(2):
            v = word * 4 + pair * 2
            packed_word = rocdl.cvt_scalef32_pk_fp8_f32(
                pk_ty, packed_word,
                vals[v].ir_value(), vals[v + 1].ir_value(),
                block_scale.ir_value(), pair)
        words.append(fx.Vector(packed_word).bitcast(fx.Int32)[0])
    return words, e8m0


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
    sdma: bool = False,
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
    RED_BATCH = min(int(topk), 8)           # 归约每批在途的 topk load 数
    RED_BATCHES = _ceil_div(int(topk), RED_BATCH)
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

        + ("_sd" if sdma else "")
    )
    # SDMA stage2 (fused remote-first only): kernel1's rows land packed in the
    # inbox; inv (in the arrival region, see compile_mega_moe_stage2) maps each
    # (plane, token, slot) to its row, and each node peer's copy engine signals
    # once after the remote half and once at the end of every forward.
    sdma = bool(sdma)
    SD_CTR_B, SD_INV_B = 128, 256
    SD_RB8 = HIDDEN + HIDDEN // 32  # FP8 row + E8M0 scales (see p2p_scatter_epilog)
    LOCAL_RANK = int(rank) % int(gpus_per_node)
    if sdma and 256 + 2 * MAX_TOKENS * TOPK * 4 > 2 * MAX_TOKENS * TOPK * 8:
        raise ValueError("inv does not fit the arrival region")

    # Device-function body (the fused stage2 kernel inlines it with a virtual
    # block index and grid); ``kernel`` below is the standalone form.
    @flyc.jit
    def k2_body(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64,
        arg_scratch: fx.Int64,
        local_tokens: fx.Int32,
        bx: fx.Int32,
        grid: fx.Int32,
        part=0,
    ):
        if const_expr(sdma and part not in (1, 2)):
            raise ValueError("sdma kernel2 runs only as the fused parts 1 and 2")
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
        if const_expr(sdma):
            inv_rsrc = buffer_ops.create_buffer_resource_from_addr(
                arrival_ptr + fx.Int64(SD_INV_B),
                num_records_bytes=fx.Int32(2 * MAX_TOKENS * TOPK * 4))

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
        # 阶段 1:归约 + 写 WQE
        # Reduce the remote plane first and ring its rail as soon as it is
        # posted, so the transfer runs under the local plane's reduce.  G CTAs
        # share each chunk and count completion once per CTA, retaining overlap
        # between memory operations. Per-unit waits would serialize this
        # bandwidth path.
        _PLANES = (remote_plane, 1 - remote_plane)
        _rail_req = fx.Int64(0)
        # Every node peer's kernel1 publishes once, at its end (word = its
        # local rank).  Poll relaxed, acquire once.  part 1 (fused stage2,
        # remote-first kernel1) needs only the remote-source half, which is
        # published at words 8 + local rank.
        # part 0: the whole kernel2; part 1: remote plane + ring; part 2: rest.
        _flag0 = int(gpus_per_node) if part == 1 else 0
        if tx == fx.Int32(0):
            for _p in range_constexpr(int(gpus_per_node)):
                _wait_ready_relaxed(arrival_ptr + fx.Int64((_flag0 + _p) * 8), generation)
            if const_expr(sdma):
                # The flags cover inv, our own rows and any row a peer stored
                # directly; the copy-engine signal covers the rows it moved.
                # Forwards finished so far.  CTA 0 advances it only after
                # FINAL_DONE, which every kernel2 CTA feeds after both of its
                # waits, so both parts of every CTA read this forward's value.
                _epoch = fx.Int32(comm.load_i32_global_system_relaxed(
                    arg_scratch + fx.Int64(SCRATCH_SDMA_EPOCH * 4)))
                _sd = cco.DevComm(dev_comm).sdma()
                _want = fx.Int64(_epoch) * fx.Int64(2) + fx.Int64(part)
                for _p in range_constexpr(int(gpus_per_node)):
                    if const_expr(_p != LOCAL_RANK):
                        _wait_ready_relaxed(
                            fx.Int64(_sd.signal_slot(fx.Int32(_p), fx.Int32(0))), _want)
        gpu.barrier()
        comm.fence_system_acquire()
        ck_local = bx % total_chunks
        sub = bx // total_chunks
        n_cta = (grid - ck_local + total_chunks - fx.Int32(1)) // total_chunks
        # The rewriter needs range_constexpr to reach nested runtime loops.
        _PIS = {0: (0, 1), 1: (0,), 2: (1,)}[part]
        for _pii in range_constexpr(len(_PIS)):
          _pi = _PIS[_pii]
          plane = _PLANES[_pi]
          if const_expr(INBOX_PER_PLANE):
              # mtpr > 4096:整个 inbox 超出 i32 的 num_records(掩码越界读不再返回 0),
              # 每个 plane 单独一个描述符,记录号按 plane 内的 token 算。
              inbox_rsrc_p = buffer_ops.create_buffer_resource_from_addr(
                  inbox_ptr + fx.Int64(plane * INBOX_PLANE_BYTES),
                  num_records_bytes=fx.Int32(INBOX_PLANE_BYTES))
          # Keep a typed runtime range so branch rewriting preserves the loop.
          for ck in range(ck_local, total_chunks, total_chunks):
              ck_first = ck * fx.Int32(CHUNK)
              ck_tok = _min_rt(fx.Int32(CHUNK), local_tokens - ck_first)
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
                  # Loads go out RED_BATCH at a time, accumulated before the
                  # next batch: all TOPK at once held 64 VGPRs of loads and put
                  # kernel2 at 117 VGPRs, which pushes a fused stage2 kernel
                  # below kernel1's five CTAs per CU.
                  for _rb in range_constexpr(RED_BATCHES):
                      vals = []
                      if const_expr(sdma):
                          # inv first for the whole batch, then the rows.
                          _oks, _recs = [], []
                          for _rj in range_constexpr(RED_BATCH):
                              slot = _rb * RED_BATCH + _rj
                              if const_expr(slot < TOPK):
                                  ok = ((slot_mask >> fx.Int32(slot))
                                        & fx.Int32(1)) != fx.Int32(0)
                                  _oks.append(ok)
                                  _recs.append(fx.Int32(buffer_ops.buffer_load(
                                      inv_rsrc, token_index * fx.Int32(TOPK) + fx.Int32(slot),
                                      vec_width=1, dtype=T.i32, mask=ok)))
                          # Rows are FP8 + E8M0 (SD_RB8 bytes); a masked load
                          # reads 0, which decodes to 0.
                          _pk, _sc = [], []
                          for _k in range_constexpr(len(_oks)):
                              if const_expr(INBOX_PER_PLANE):
                                  # One-row descriptor: the packed row index can
                                  # pass the i32 byte range.  The row is
                                  # wave-uniform (one token-slot per wave).
                                  _ru = fx.Int32(rocdl.readfirstlane(T.i32, _recs[_k].ir_value()))
                                  _row_rsrc = buffer_ops.create_buffer_resource_from_addr(
                                      inbox_ptr + fx.Int64(_ru) * fx.Int64(SD_RB8),
                                      num_records_bytes=fx.Int32(SD_RB8))
                                  _b0 = fx.Int32(0)
                              else:
                                  _row_rsrc = inbox_rsrc
                                  _b0 = _recs[_k] * fx.Int32(SD_RB8)
                              _pk.append(buffer_ops.buffer_load(
                                  _row_rsrc, (_b0 + col) // fx.Int32(4),
                                  vec_width=2, dtype=T.i32, mask=_oks[_k]))
                              _sc.append(buffer_ops.buffer_load(
                                  _row_rsrc, _b0 + fx.Int32(HIDDEN) + col // fx.Int32(32),
                                  vec_width=1, dtype=T.i8, mask=_oks[_k]))
                          for _k in range_constexpr(len(_oks)):
                              totals = totals + _fp8_decode8(
                                  _pk[_k],
                                  ((fx.Uint8(_sc[_k]).to(fx.Uint32).bitcast(fx.Int32))
                                   << fx.Int32(23)).bitcast(Float32))
                      for _rj in range_constexpr(RED_BATCH if not sdma else 0):
                          slot = _rb * RED_BATCH + _rj
                          if const_expr(slot < TOPK):
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
                      for _v in vals:
                          totals = totals + fx.Vector(_v).bitcast(
                              BFloat16).to(Float32)
                      if const_expr(_rb + 1 < RED_BATCHES):
                          rocdl.sched_barrier(0)
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
              # Drain this CTA's stores before its acq_rel arrival. The last
              # arrival observes every partial before posting the chunk's WQE.
              fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
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
                      # Publish the whole chunk to the NIC with one system fence.
                      comm.fence_system_release()
                      if lane == fx.Int32(0):
                          comm.store_i32_system(
                              arg_scratch, cnt_idx, fx.Int32(0))
                          # Phase 3 reads only the chunk's first token.
                          # relaxed:上面那次 fence 已经把 writeback 做了,
                          # release store 会再来一遍 buffer_wbl2。
                          comm.store_i64_global_system_relaxed(
                              ready_ptr
                              + fx.Int64(fx.Int32(plane * MAX_TOKENS) + ck_first)
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
                          fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                          if lane == fx.Int32(0):
                              comm.atomic_add_agent(
                                  arg_scratch
                                  + fx.Int64(SCRATCH_WQE_POSTED) * fx.Int64(4)
                                  + fx.Int64(_qp) * fx.Int64(4),
                                  fx.Int32(1))
          if const_expr(_pi == 0):  # noqa: SIM102
            # One CTA's wave0 per QP rings it once all its WQEs are posted.
            # QPs are independent; this needs no CTA barrier.
            if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
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
              _rail_req = fx.Int64(_rail.flush_async(
                  dev_comm, bx, fx.Int32(remote_node)))
              if const_expr(part == 1):  # noqa: SIM102
                  if lane == fx.Int32(0):
                      comm.store_i64_global_relaxed(
                          arg_scratch + fx.Int64(SCRATCH_RAIL_REQ * 4)
                          + fx.Int64(bx) * fx.Int64(8), _rail_req)
        if const_expr(part != 1):
          # 阶段 2:rail 返回 —— the doorbell rang after the remote plane; reap
          # the completion here, after the local plane's reduce.
          if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
           if const_expr(part == 2):
               _rail_req = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(
                   comm.load_i64_global(
                       arg_scratch + fx.Int64(SCRATCH_RAIL_REQ * 4)
                       + fx.Int64(bx) * fx.Int64(8))).ir_value()))
           _rail.wait(dev_comm, bx, _rail_req)
           if lane == fx.Int32(0):
               comm.store_i32_system(
                   arg_scratch, fx.Int32(SCRATCH_WQE_POSTED) + bx,
                   fx.Int32(0))
          # 阶段 3:所有 CTA 做 node 间归约 + unpermute
          # Both node partials are already reduced. Add and unpermute them;
          # synchronize once per CTA/chunk and distribute its work across waves.
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
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
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
          fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
          gpu.barrier()
          # 收尾:return_consumed 信用交换
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
                if const_expr(sdma):
                    # Every node peer claimed its landing rows here before its
                    # data (which we have read): free them for generation + 2,
                    # and count this forward's two signals per peer.
                    comm.store_i32_global_system_relaxed(
                        arrival_ptr + fx.Int64(SD_CTR_B), fx.Int32(0))
                    comm.atomic_add_agent(
                        arg_scratch + fx.Int64(SCRATCH_SDMA_EPOCH * 4), fx.Int32(1))
            gpu.barrier()
            if wave == fx.Int32(0):
                # "你发给我的那块我读完了" -> 对端可以复用它的发送源。
                _rail.put_value(dev_comm, fx.Int32(0), fx.Int32(remote_node),
                                arena_win, _consumed_off, generation,
                                aggregate=True)
                # Ring now, reap our own completion after the peer's credit:
                # the two are independent, so waiting for our ACK first only
                # serializes one extra round trip in front of the credit wait.
                _consumed_req = _rail.flush_async(
                    dev_comm, fx.Int32(0), fx.Int32(remote_node))
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
                _rail.wait(dev_comm, fx.Int32(0), _consumed_req)

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64,
        arg_scratch: fx.Int64,
        local_tokens: fx.Int32,
    ):
        k2_body(dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
                local_tokens, fx.Int32(gpu.block_id("x")), fx.Int32(gpu.grid_dim.x))

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
    launch.body = k2_body
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


# ---------------------------------------------------------------------------
# Fused stage2: kernel1 and kernel2 in one launch.
#
# Every CTA runs the kernel1 body (GEMM2 + node push + the once-per-kernel
# arrival publish); CTAs below ``k2_ctas`` then run the kernel2 body with their
# own block index.  No role claim and no grid barrier: kernel2 only waits on
# the arrival flags, and kernel1 never waits on kernel2.  The grid is the
# resident CTA count (see ``fused`` in compile_mega_moe_stage2), so every
# kernel2 CTA is resident; consecutive indices spread it over the CUs.
# (All CTAs running kernel2 over the whole grid measured +5~29%.)
# ---------------------------------------------------------------------------
_FUSED_CACHE = {}


def compile_stage2_fused(k1_kw: dict, k2_kw: dict, k2_ctas: int):
    k1 = compile_mega_moe_stage2(**k1_kw, fused=True)
    k2 = compile_stage2_node_combine(**k2_kw)
    k1_body, k2_body = k1.body, k2.body
    grid_ctas = int(k1.grid_ctas)
    remote_first = bool(k1.remote_first)
    k2_ctas = int(k2_ctas)
    if k2_ctas > grid_ctas:
        raise ValueError(f"fused stage2 grid {grid_ctas} < kernel2 CTAs {k2_ctas}")
    sdma = bool(k1_kw.get("sdma", False))
    k1_occ = int(k1.ctas_per_cu)
    if sdma != bool(k2_kw.get("sdma", False)):
        raise ValueError("kernel1 and kernel2 must agree on sdma")
    if sdma and k2_ctas > k1.sd_plan_bx:
        # The plan CTAs (then GEMM2) and the comm CTA sit past kernel2's CTAs.
        raise ValueError(f"sdma fused stage2 grid {grid_ctas} leaves no room for {k2_ctas} kernel2 CTAs")
    kernel_name = "megamoe_s2f_" + k1.kernel_name[len("megamoe_"):] + "__" + k2.kernel_name[len("megamoe_"):]

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def fused(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32,
        dev_comm: fx.Int64, arena_win: fx.Int64, arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64, arg_scratch: fx.Int64, local_tokens: fx.Int32,
        arg_expert_count: fx.Int64, arg_sd_send: fx.Int64, arg_sd_tb: fx.Int64,
        arg_sd_plan: fx.Int64, arg_sd_cdone: fx.Int64):
    # fmt: on
        bx = fx.Int32(fx.block_idx.x)

        def remote_half_done():
            # Remote-first kernel1: once this CTA has no remote m-block left,
            # the kernel2 CTAs reduce the remote plane and ring the rail while
            # the other CTAs carry on with the local half of GEMM2.
            if bx < fx.Int32(k2_ctas):
                k2_body(dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
                        local_tokens, bx, fx.Int32(k2_ctas), part=1)

        k1_body(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
                arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
                i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map, i32_local_rank,
                fx.block_idx.x, arg_expert_count=arg_expert_count,
                boundary_hook=remote_half_done if remote_first else None,
                dev_comm=dev_comm, arg_sd_send=arg_sd_send, arg_sd_tb=arg_sd_tb,
                arg_sd_plan=arg_sd_plan, arg_sd_cdone=arg_sd_cdone,
                arena_win=arena_win, arena_ptr=arena_ptr)
        if bx < fx.Int32(k2_ctas):
            k2_body(dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
                    local_tokens, bx, fx.Int32(k2_ctas), part=2 if remote_first else 0)

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32,
        dev_comm: fx.Int64, arena_win: fx.Int64, arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64, arg_scratch: fx.Int64, local_tokens: fx.Int32,
        arg_expert_count: fx.Int64, arg_sd_send: fx.Int64, arg_sd_tb: fx.Int64,
        arg_sd_plan: fx.Int64, arg_sd_cdone: fx.Int64, stream: fx.Stream):
    # fmt: on
        fused(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
              arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
              i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map, i32_local_rank,
              dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch, local_tokens,
              arg_expert_count, arg_sd_send, arg_sd_tb, arg_sd_plan, arg_sd_cdone,
              # Keep kernel1's occupancy: the grid assumes K1_CTAS_PER_CU per CU.
              value_attrs={"rocdl.waves_per_eu": k1_occ,
                           "rocdl.flat_work_group_size": "256,256"},
              ).launch(grid=(grid_ctas, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = kernel_name
    return launch


def run_stage2_fused(*, k1_ptrs: dict, k1_compile: dict, dev_comm, arena_win, arena_ptr,
                     arg_output_bf16, arg_scratch, local_tokens, k2_ctas, stream,
                     k2_compile: dict):
    """Compile (once per configuration) and launch the fused stage2 kernel."""
    key = (tuple(sorted(k1_compile.items())), tuple(sorted(k2_compile.items())), int(k2_ctas))
    launch = _FUSED_CACHE.get(key)
    if launch is None:
        launch = _FUSED_CACHE[key] = compile_stage2_fused(k1_compile, k2_compile, k2_ctas)
    p = k1_ptrs
    ptrs = (p["arg_aq"], p["arg_ascale"], p["arg_bq"], p["arg_bscale"], p["arg_eids"],
            p["arg_cumsum"], p["arg_stids"], p["arg_sweights"], p["arg_trb"], p["arg_p2p"])
    _run_compiled(
        launch, *(fx.Int64(int(v)) for v in ptrs),
        fx.Int32((p["row_capacity"] + BM - 1) // BM),
        fx.Int32(k1_compile["inter_dim"]), fx.Int32(k1_compile["model_dim"]),
        fx.Int64(int(p["arg_arrival_count"])), fx.Int64(int(p["arg_generation"])),
        fx.Int64(int(p["arg_a_tile_map"])), fx.Int32(int(p["local_rank"])),
        fx.Int64(int(dev_comm)), fx.Int64(int(arena_win)), fx.Int64(int(arena_ptr)),
        fx.Int64(int(arg_output_bf16)), fx.Int64(int(arg_scratch)), fx.Int32(int(local_tokens)),
        fx.Int64(int(p.get("arg_expert_count", 0))),
        *(fx.Int64(int(p.get(k, 0))) for k in ("arg_sd_send", "arg_sd_tb", "arg_sd_plan",
                                               "arg_sd_cdone")),
        stream,
    )
