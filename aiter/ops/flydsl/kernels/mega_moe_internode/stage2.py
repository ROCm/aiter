# SPDX-License-Identifier: MIT
"""Stage2 of MegaMoEInternode: two kernels.

kernel1 (``megamoe_k1_*``) computes GEMM2 and pushes weighted BF16 rows into
the node-local plane-slot inbox. It uses MegaMoEV2's ``gemm2_compute_v2`` core;
the EP16 epilogue targets the GPU with the owner's local index on this node
and publishes its arrival flag.

kernel2 (``megamoe_k2_*``) reduces within the node, sends over the rail, then
reduces across nodes and unpermutes output. It has a separate kernel and ABI
from kernel1, sharing the same CCO initialization.

All CTAs pass through three phases rather than keeping fixed role partitions:

  Phase 1: Each CTA owns chunks, reads plane_slot_inbox with slot masks, and
           writes per-token node partials to node_accumulator. For remote-plane
           chunks, the same CTA posts a contiguous aggregate PUT WQE without
           ringing the doorbell, then increments wqe_posted[qp].

  Phase 2: The first NUM_QP CTAs each handle one QP in wave0. Wait for that
           QP's WQEs, publish return_group_ready for its chunks, and flush once
           to ring the doorbell. QPs need no ordering or shared barrier.

  Phase 3: All CTAs perform inter-node reduction. Phase 1 already reduced node
           partials; remote_partial_rx contains the peer's reduced values.
           Add and unpermute on chunk arrival without a global barrier.

Shapes are parameterized. Handle tail chunks when tokens is not divisible
by num_qp*chunk, rather than imposing the reference implementation's
"no tail transport here" divisibility restriction.
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
from .buffer_addressing import (
    region_local_address,
    region_local_dev_comm,
    region_lsa_address,
    region_window_address,
)
from .communication import RailGda
from .communication import ccqe_enabled as _rail_ccqe
from .gemm import gemm2_compute_v2, issue_a_load_lds_dt, kStages
from .layout import MAX_FUSED_MODEL_DIM
from .mega_moe_internode_config import GEMM2_BNS

# Keep constexpr and runtime guards separate (local SIM102/SIM114 exemptions).
# Raw loads/atomics preserve system scope, ordering and element-offset units;
# scaled MFMA SSA fragments retain the tuned instruction and rounding contract.
_BUFFER_OFFSET_ABI_BYTES = 1 << 31
# Tile shape: 32-row M tiles (one Stage1 tile each), K step 256.
BM = 32
BK = 256
# Workgroups are dispatched round-robin over the 8 XCDs (SPX mode).
NUM_XCD = 8


# fmt: off
def p2p_scatter_epilog(lds_acc_base, accm, n_block_idx, wave, lane, *, N_OUT, BM, BN, npes, topk,
    log2_max_tok, mask_max_tok, recv_cap, comb_inp_nbytes, lds_packed_off, lds_weight_off,
    lds_peer_off, scatter_vec, gpus_per_node, lds_rowptr_off=None, NW=4):
# fmt: on
    """CShuffle one GEMM2 tile into weighted BF16 rows and scatter them to peers.

    With ``lds_rowptr_off`` (SDMA stage2) each row's destination was resolved
    before the GEMM: an absolute row address in LDS (0 = no row), pointing at
    the local send buffer, this GPU's own landing rows, or -- past the send
    buffer -- the peer's landing row directly.

    NW=8 (two n_blocks per CTA, see gemm2_compute_v2): each four-wave half
    shuffles through its own BM x BN slab and stores its own n_block.
    """
    if NW == 8:
        lds_acc_base = lds_acc_base + (wave // fx.Int32(4)) * fx.Int32(BM * BN * 4)
        wave = wave % fx.Int32(4)
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
# One m-block ticket per N-block plus the done and remote-half done counters.
# Cover the largest supported hidden dimension at the smallest allowed BN.
# The host allocation and the specialization guard use this shared bound.
K1_CLAIM_WORDS = MAX_FUSED_MODEL_DIM // min(GEMM2_BNS) + 2
# Kernel1 CTAs resident per CU (91 VGPRs, 4 waves -> 5 per CU); the dynamic
# claim launches at most this many per CU.
K1_CTAS_PER_CU = 5
# i32 words the host allocates for the SDMA stage2 plan buffer.
SDMA_PLAN_WORDS_MAX = 1024


def tr_layout(max_tok: int, topk: int) -> dict:
    """Token-ready rail words, byte offsets into a rank's plane_slot_arrived
    plane (per parity), packed at its end; inv and the SDMA chunk counters sit
    at its start.

    cnt[tok]  i32: the rail rank adds -E (rows of tok its node peers compute),
              every peer +1 per finished row; it reaches 0 exactly once.
    q[pos]    i64: non-SDMA ready queue, gen32 << 32 | tok.
              SDMA uses q[tok] as the full compute-ready generation.
              Fixed owners keep their consumed bitmap/cursor in CTA LDS.
    tail/head i32: enqueue / claim counters (separate lines).
    posted[8], rung[8] i32: per QP batches posted / covered by a doorbell.
    req[8]    i64: per QP last flush request.  bdone[b] i32: tokens reduced.
    prog[8]   i64: SDMA landed-row prefix per source rank (gen32 << 32 | end).
    """
    plane = 2 * int(max_tok) * int(topk) * 8
    out, off = {}, 0
    for name, nbytes in (("cnt", max_tok * 4), ("q", max_tok * 8), ("tail", 4), ("head", 4),
                         ("scan", 4), ("state", max_tok * 4),
                         ("posted", 32), ("rung", 32), ("req", 64),
                         ("bdone", (max_tok // 4 + 1) * 4),
                         ("issued", (max_tok // 4 + 1) * 4), ("prog", 64)):
        off = (off + 127) // 128 * 128
        out[name] = off
        off += nbytes
    base = (plane - off) // 256 * 256
    if base < 256 + 2 * int(max_tok) * int(topk) * 4:
        raise ValueError("token-ready words do not fit the arrival plane")
    out = {k: base + v for k, v in out.items()}
    out["base"] = base
    return out


def compile_mega_moe_stage2(*, model_dim: int, inter_dim: int, experts: int, topk: int, npes: int,
    max_tok: int, recv_cap: int, comb_inp_nbytes: int, BN: int, cu_num: int, gpus_per_node: int,
    arrival_delta: int, arrival_delta_step: int, epoch_planes: tuple, a_tile_map_plane: int,
    m_group: int = 1, dynamic_claim: bool = False, resident_ctas: int = 0,
    fused: bool = False, remote_first: bool = False, local_experts: int = 0,
    tile_group: int = 1, expert_count_plane: int = 0, sdma: bool = False,
    sdma_chunk_tiles: int = 32, sdma_send_rows: int = 0, sdma_frac_q64: int = 64,
    sdma_put_bytes: int = 4 << 20, sdma_arrival_woff: int = 0, sdma_arrival_plane: int = 0,
    sdma_max_chunks: int = 0, gemm2_bm: int = 32, sdma_ramp_chunks: int = 0, waves: int = 4,
    sdma_remote_only: bool = False, token_ready: bool = False, a_dtype: str = "fp4",
    buffer_regions: tuple = (), group_sync_tail: int = 0):
# fmt: on
    """Compile the persistent GEMM2 + node push kernel (stage2 kernel1).

    ``waves`` = 8 (the single-kernel CTA shape, one CTA per CU at <= 256
    registers a wave): a CTA runs the n_block pair (2p, 2p+1) of each m-block
    it claims, waves 0-3 / 4-7 one n_block each over the shared A tile; the
    tickets are per n_block pair.  Fused dynamic xGMI kernel only.

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
    region_indices = {(stage, name): index for stage, name, index in buffer_regions}
    if buffer_regions and token_ready:
        raise NotImplementedError("separated buffers do not support token-ready")
    if a_dtype not in ("fp4", "fp8"):
        raise ValueError("a_dtype must be fp4 or fp8")
    a_pack = 1 if a_dtype == "fp8" else 2
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
    # tiles. Without remote-first, each ticket covers GBM rows. GEMM, row
    # metadata and epilogue use GBM on both paths.
    GBM = int(gemm2_bm)
    GS = GBM // BM
    if GBM % BM or not 1 <= GS <= 4:
        raise ValueError(f"gemm2_bm must be 32, 64 or 128, got {GBM}")
    if GS > 1 and int(tile_group) % GS:
        raise ValueError(f"gemm2_bm {GBM} needs GS | tile_group")

    def _gs(x):
        return x if GS == 1 else x * fx.Int32(GS)
    log2_max_tok = max_tok.bit_length() - 1
    mask_max_tok = max_tok - 1
    N_OUT = model_dim
    aStages = kStages + 1
    KH_TILE_A = BK // a_pack
    # The f32 CShuffle slab and the A stages share one LDS union.
    WAVES = int(waves)
    PAIR = WAVES == 8
    if WAVES not in (4, 8):
        raise ValueError(f"stage2 kernel1 waves must be 4 or 8, got {WAVES}")
    if PAIR and (not fused or (model_dim // BN) % 2):
        raise ValueError("8-wave kernel1 needs the fused kernel and an even n_block count")
    THREADS_K1 = 64 * WAVES
    NPAIR = 2 if PAIR else 1  # n_blocks per CTA unit
    compute_lds_bytes = max(NPAIR * GBM * BN * 4, aStages * GBM * KH_TILE_A)
    lds_packed_off = compute_lds_bytes
    lds_weight_off = lds_packed_off + GBM * 4
    lds_peer_off = lds_weight_off + GBM * 4
    num_n_blocks_c = model_dim // BN
    NB_CLAIM = num_n_blocks_c // NPAIR  # claim tickets / CTA columns (n_block pairs)
    xcd_group = int(m_group)
    if fused:
        # The fused kernel runs kernel2 on these CTAs once their GEMM2 work is
        # done, so every CTA must be resident from the start: a CTA still
        # waiting for a slot behind spinning kernel2 CTAs never gets one.
        if resident_ctas <= 0:
            raise ValueError("fused stage2 needs resident_ctas > 0")
        cu_num = min(int(cu_num), max(1, int(resident_ctas) // NB_CLAIM))
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
    ticket_bm = BM if remote_first and dynamic else GBM
    if dynamic and resident_ctas <= 0:
        raise ValueError("dynamic_claim needs resident_ctas > 0")
    lds_claim_off = lds_peer_off + npes * 8  # two i32 slots, double-buffered
    # token_ready: words 2..7 of the claim area are the rail hooks' broadcasts.
    TRK = bool(token_ready)
    # Fixed token owners retain one bit per assigned token in private CTA LDS.
    # Use a lower bound on the effective grid (resident_ctas is a capacity
    # scaled by K1_CTAS_PER_CU). Over-allocation also covers non-paired grids.
    TR_OWNER_GRID = max(1, int(resident_ctas) // K1_CTAS_PER_CU - NB_CLAIM + 1)
    TR_OWNER_ROUNDS = _ceil_div(max_tok, TR_OWNER_GRID * WAVES)
    TR_OWNER_WORDS = _ceil_div(TR_OWNER_ROUNDS, 32)
    tr_owner_bytes = (TR_OWNER_WORDS * WAVES + 2) * 4 if TRK and sdma else 0
    lds_bytes = lds_claim_off + ((64 + tr_owner_bytes if TRK else 16) if dynamic else 0)
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
    if TRK and not (fused and remote_first and dynamic):
        raise ValueError("token_ready needs the fused remote-first kernel")
    TRL = tr_layout(max_tok, topk) if TRK else None
    # Remote-only SDMA: the remote-source half (pushed first, awaited by the rail
    # reduce) goes by the copy engine; the local half, computed last, is stored
    # over xGMI by the GEMM2 CTAs so no engine tail follows the last unit.
    SD_RO = bool(sdma_remote_only)
    if SD_RO and not sdma:
        raise ValueError("sdma_remote_only needs the sdma stage2")
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
    SD_NODES = int(npes) // int(gpus_per_node)
    SD_CDONE_B = (256 + SD_NODES * max_tok * topk * 4 + 255) // 256 * 256
    if sdma:
        if SD_CT <= 0 or SD_ROWS <= 0:
            raise ValueError("sdma needs sdma_chunk_tiles > 0 and sdma_send_rows > 0")
        if gpus_per_node != 8:
            raise ValueError("sdma stage2 assumes 8 GPUs per node")
        if model_dim % 256 or SD_RB % 8:
            raise ValueError("sdma FP8 rows need hidden % 256 == 0 and 8-byte rows")
        if SD_AWOFF <= 0 or SD_MAXCH <= 0 or SD_PUT_B <= 0:
            raise ValueError("sdma needs the arrival window offset, max chunks and put bytes")
        if TRK and SD_CDONE_B + SD_MAXCH * SD_CDONE_STRIDE * 4 > TRL["base"]:
            raise ValueError("SDMA chunk counters overlap token-ready words")
        if TRK and 2 * max_tok * topk >= (1 << 28):
            raise ValueError("token-ready SDMA row indices exceed 28 bits")
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
        # One thread per ticket range, groups of 8 threads (THREADS_K1 // 8).
        SP_TOT, SP_GS = 0, THREADS_K1 * 8 * 4
        SP_SUM = SP_GS + (THREADS_K1 // 8) * 8 * 4
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
    K1_OCC = 1 if PAIR else max(1, min(K1_CTAS_PER_CU, (160 * 1024) // lds_bytes))
    res_eff = int(resident_ctas) // K1_CTAS_PER_CU * K1_OCC
    if fused:
        cu_num = min(int(cu_num), max(1, res_eff // NB_CLAIM))
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
    ctas_per_nb = min(cu_num, max(1, res_eff // NB_CLAIM)) if dynamic else cu_num
    grid_ctas = ctas_per_nb * NB_CLAIM
    if PAIR:
        # One CTA per CU, all of them: kernel2 (and later Stage1) runs on the
        # same grid.  Tickets are dynamic, so bx % NB_CLAIM may give a pair
        # one CTA more than another.
        grid_ctas = max(grid_ctas, res_eff)
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
        + ("_a8w4" if a_dtype == "fp8" else "")
        + (f"_xg{xcd_group}" if xcd_group > 1 else "")
        + (f"_dynp{ctas_per_nb}" if dynamic else "")
        + (f"_w{WAVES}" if PAIR else "")
        + (f"_rf{LE}g{int(tile_group)}" if remote_first else "")
        + (f"_sd{SD_CTB}r{SD_ROWS}q{SD_FQ}p{SD_PUT_B >> 10}" if sdma else "")
        + ("ro" if SD_RO else "") + ("_trpub" if TRK else "")
        + (f"s{SD_S}n{SD_NS}" if sdma and SD_NS else "")
        + "_de_" + "_".join(str(v) for v in epoch_planes)
        # Symbols cannot contain '-'; encode negative strides with an m prefix.
        + f"_ad{'m' if arrival_delta_step < 0 else ''}{abs(arrival_delta_step)}"
        + f"_atm{int(a_tile_map_plane)}"
        + ("_c1" if buffer_regions else "")

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
        arg_sd_tb=None, arg_sd_plan=None, arg_sd_cdone=None, arena_win=None, arena_ptr=None,
        lds_ptr=None, grp_sync=None, boundary_probe=None, arg_tr_mcnt=None,
        unit_hook=None, tail_hook=None, arg_tr_sdprog=None):
    # fmt: on
        # grp_sync (single fused kernel): (window offset of Stage1's per-group
        # h1 ready words, their parity plane, Stage1 tile group); each unit
        # waits for its physical group's GMM1 before loading A.
        tx_i32 = fx.thread_idx.x


        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        # Stage1 wrote epoch_gate earlier on this stream; the kernel boundary provides
        # visibility, so use a relaxed agent-scope load. A system acquire adds
        # buffer_inv at every CTA entry, disrupting GEMM2's L2 weight reuse (+97us measured).
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

        if const_expr(lds_ptr is None):
            lds_base_i32 = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr))
        else:
            # Single fused kernel: one LDS allocation shared with Stage1.
            lds_base_i32 = fx.Int32(fx.ptrtoint(lds_ptr))

        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        k_bytes = fx.Int32(i32_inter) // fx.Int32(a_pack)
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

        A_NW = min(WAVES, GBM // (4 if a_dtype == "fp8" else 8))

        def issue_all_a_loads(m_row0):
            if const_expr(A_NW < WAVES):
                if wave < fx.Int32(A_NW):
                    for slot in range_constexpr(kStages):
                        issue_a_load_lds_dt(arg_aq, lds_base_i32, slot, slot, m_row0, wave, lane,
                            KH_TILE_A, k_bytes, BM=GBM, NW=A_NW, a_dtype=a_dtype)
            else:
                for slot in range_constexpr(kStages):
                    issue_a_load_lds_dt(arg_aq, lds_base_i32, slot, slot, m_row0, wave, lane,
                        KH_TILE_A, k_bytes, BM=GBM, NW=WAVES, a_dtype=a_dtype)

        # a_tile_map maps sorted 32-row tile t to its physical A(h1) tile.
        # Only A/A-scale use this indirection; expert IDs, srcmap, and output positions
        # still use sorted row indices.
        a_map_buf = ptr_buf_tensor(arg_a_tile_map, fx.Int32)
        if const_expr(sdma):
            sd_tb_buf = ptr_buf_tensor(arg_sd_tb, fx.Int32)
            sd_arrival_ptr = fx.Int64(arena_ptr) + fx.Int64(SD_AWOFF) + _par * fx.Int64(SD_APLANE)
            sd_arrival_win = arena_win
            sd_arrival_off = fx.Int64(SD_AWOFF) + _par * fx.Int64(SD_APLANE)
            if const_expr(bool(buffer_regions)):
                sd_arrival_ptr = region_local_address(
                    arena_ptr, region_indices[2, "plane_slot_arrived"], _par)
                sd_arrival_win, sd_arrival_off = region_window_address(
                    arena_ptr, region_indices[2, "plane_slot_arrived"], _par)

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
            if const_expr(SD_RO):
                to_send = to_send & (ticket < remote_total)
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
                    + fx.Int64(slot) * fx.Int64(4),
                    (land | (fx.Int32(i32_local_rank) << fx.Int32(28))
                     | to_send.select(fx.Int32(-2147483648), fx.Int32(0)))
                    if TRK else land)

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
            if const_expr(PAIR):
                unit_bx = unit_bx + wave // fx.Int32(4)  # waves 4-7: the pair's second n_block
            # fmt: off
            accm_vecs, _m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(lds_base_i32, arg_ascale, arg_bq,
                arg_bscale, arg_eids, arg_aq, i32_max_m_blocks, unit_bx, lane, wave, i32_inter, i32_hidden,
                BM=GBM, BN=BN, BK=BK, INTER_MAX=inter_dim, aStages=aStages, explicit_a_row=a_row,
                eids_stride=GS, NW=WAVES, a_dtype=a_dtype)
            p2p_scatter_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, NW=WAVES, N_OUT=N_OUT,
                BM=GBM, BN=BN, npes=npes, topk=topk,
                log2_max_tok=log2_max_tok, mask_max_tok=mask_max_tok, recv_cap=recv_cap,
                comb_inp_nbytes=comb_inp_nbytes, lds_packed_off=lds_packed_off,
                lds_weight_off=lds_weight_off, lds_peer_off=lds_peer_off, scatter_vec=scatter_vec,
                gpus_per_node=gpus_per_node, lds_rowptr_off=lds_rowptr_off if sdma else None)
            # fmt: on

        if const_expr(grp_sync is not None):
            gs_ready = arena_ptr + fx.Int64(grp_sync[0]) + _par * fx.Int64(grp_sync[1])
            if const_expr(bool(buffer_regions)):
                gs_ready = region_local_address(arena_ptr, region_indices[1, "h1_grp_sync"], _par) + fx.Int64(group_sync_tail)
            gs_gen = fx.Int32(i64_generation)

        def wait_group(a_row):
            # The unit's GBM rows lie in one Stage1 tile group (GS | tile_group).
            if const_expr(grp_sync is not None):
                if tx_i32 == fx.Int32(0):
                    gaddr = gs_ready + fx.Int64(a_row // fx.Int32(32 * int(grp_sync[2]))) * fx.Int64(4)
                    gv = fx.Int32(comm.load_i32_global_system_relaxed(gaddr))
                    while gv != gs_gen:
                        fx.rocdl.s_sleep(1)
                        gv = fx.Int32(comm.load_i32_global_system_relaxed(gaddr))
                fx.barrier()

        def run_mapped_unit(unit_bx, m_block_idx):
            m_row0 = m_block_idx * fx.Int32(GBM)
            a_row = a_map_buf[m_row0 // fx.Int32(BM)] * fx.Int32(BM)
            wait_group(a_row)
            issue_all_a_loads(a_row)
            rocdl.sched_barrier(0)
            run_unit(unit_bx, m_block_idx, a_row)

        cumsum0 = global_typed_ptr(arg_cumsum, T.i32)[0]
        # Remote-first maps tickets in 32-row Stage1 tiles. The plain path
        # indexes contiguous GEMM2 units, each spanning GS Stage1 tiles.
        total_m_blocks = (cumsum0 + fx.Int32(ticket_bm - 1)) // fx.Int32(ticket_bm)

        if const_expr(dynamic):
            claim_ptr = arg_arrival_count + fx.Int64(claim_word * 4)
            dyn_n_block = bx_i32 % fx.Int32(NB_CLAIM)  # n_block (pair) of this CTA
            dyn_n0 = dyn_n_block * fx.Int32(NPAIR) if PAIR else dyn_n_block  # its first n_block
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
                    m_row0 = m_block * fx.Int32(GBM)
                    a_row = a_map_buf[m_row0 // fx.Int32(BM)] * fx.Int32(BM)
                    wait_group(a_row)
                    issue_all_a_loads(a_row)
                    if tx_i32 == fx.Int32(0):
                        fx.ptr_store(
                            nxt,
                            lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32, align=4),
                        )
                    rocdl.sched_barrier(0)
                    run_unit(m_block * num_n_blocks + dyn_n0, m_block, a_row)
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
                        return (sd_arrival_off + fx.Int64(SD_CDONE_B)
                                + fx.Int64(c) * fx.Int64(SD_CDONE_STRIDE * 4))

                    def _sd_cdone(c):
                        return sd_arrival_ptr + fx.Int64(SD_CDONE_B) + fx.Int64(c) * fx.Int64(SD_CDONE_STRIDE * 4)

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
                        kper = (n_loc + fx.Int32(THREADS_K1 - 1)) // fx.Int32(THREADS_K1)
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
                            for _g in range_constexpr(THREADS_K1 // 8):
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
                            for c in range(tx_i32, _sd_nch(), fx.Int32(THREADS_K1)):
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
                            sd = cco.DevComm(
                                region_local_dev_comm(arena_ptr, len(buffer_regions))
                                if buffer_regions else dev_comm).sdma()
                            nrc = _sd_nrc()
                            nch = _sd_nch()
                            seg = _sd_lds_i32(lds_splan_off, p)
                            rb = _sd_lds_i32(lds_splan_off + 32, p)
                            cap = _sd_lds_i32(lds_splan_off + 96, p)
                            if const_expr(SD_RO):
                                cap = _min_rt(cap, _sd_gtb(remote_total, p))
                                nch = nrc
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
                                sd.post_wait_mem(p, fx.Int32(0), sd_arrival_win, _sd_cdone_woff(c), need,
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
                                    if const_expr(TRK):  # noqa: SIM102
                                        if s_hi > s_lo:
                                            # This peer queue is FIFO: the prefix lands only
                                            # after the preceding row put. Each source word
                                            # is unique until all SDMA signals finish.
                                            src_prog = (fx.Int64(arg_tr_sdprog)
                                                + fx.Int64(p * fx.Int32(SD_MAXCH) + c) * fx.Int64(8))
                                            comm.store_i64_global_system_relaxed(src_prog,
                                                ((i64_generation & fx.Int64(0xFFFFFFFF)) << fx.Int64(32))
                                                | fx.Int64(rb + s_hi - seg))
                                            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                                            sd.put_ptr(p, _sd_peer(p) + arrival_delta_rt
                                                + fx.Int64(TRL["prog"]) + fx.Int64(i32_local_rank) * fx.Int64(8),
                                                src_prog, fx.Int64(8), fx.Int32(0), aggregate=True)
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
                            # One unit = GS tiles x NPAIR n_blocks of the chunk's count.
                            comm.atomic_add_system(_sd_cdone(c), fx.Int32(GS * NPAIR))

                def _tr_lw(i):
                    # LDS words 4..7 of the claim area, for the rail hooks.
                    return lds_typed_ptr(fx.Int32(lds_claim_off + 16 + 4 * i), T.i32, align=4)

                # The last of an m-block's NB_CLAIM column units (each CTA walks
                # one column group) finds the rows complete: every unit's pushes
                # drained at its barrier and released through the agent counter.
                # It adds one per row to the row's destination counter; whoever
                # brings it to 0 enqueues the token there (see tr_layout).  Split
                # in a thread-0 part and a whole-CTA part so the rail hook's
                # barrier serves both.
                def _tr_count_tx0(mb):
                    _mc = fx.Int64(arg_tr_mcnt) + fx.Int64(mb) * fx.Int64(4)
                    _c = fx.Int32(comm.atomic_add_agent_acq_rel(_mc, fx.Int32(1)))
                    _l = (_c + fx.Int32(1)) == fx.Int32(NB_CLAIM)
                    if _l:
                        comm.atomic_add_agent(_mc, fx.Int32(-NB_CLAIM))
                    fx.ptr_store(_l.select(fx.Int32(1), fx.Int32(0)), lds_typed_ptr(
                        fx.Int32(lds_claim_off + 12), T.i32, align=4))

                def _tr_count_rows():
                    _last = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lds_typed_ptr(
                        fx.Int32(lds_claim_off + 12), T.i32, align=4)).ir_value()))
                    if _last != fx.Int32(0):  # noqa: SIM102
                        if tx_i32 < fx.Int32(GBM):
                            pk = fx.Int32(fx.ptr_load(lds_typed_ptr(
                                fx.Int32(lds_packed_off) + tx_i32 * fx.Int32(4), T.i32, align=4)))
                            t24 = pk & fx.Int32(0x00FFFFFF)
                            sk = pk.shrui(fx.Int32(24)) & fx.Int32(0xFF)
                            own = t24 >> fx.Int32(log2_max_tok)
                            ok = ((t24 < fx.Int32(recv_cap)) & (sk < fx.Int32(topk))
                                  & (own < fx.Int32(npes)))
                            if ok:
                                dpe = own & fx.Int32(gpus_per_node - 1)
                                tok = t24 & fx.Int32(mask_max_tok)
                                trb = fx.Int64(fx.ptr_load(lds_typed_ptr(
                                    fx.Int32(lds_peer_off) + dpe * fx.Int32(8), T.i64,
                                    align=8))) + arrival_delta_rt
                                old = fx.Int32(comm.atomic_add_system(
                                    trb + fx.Int64(TRL["cnt"]) + fx.Int64(tok) * fx.Int64(4),
                                    fx.Int32(1)))
                                if old == fx.Int32(-1):
                                    if const_expr(sdma):
                                        # One unique last arrival publishes this token.
                                        # Its counter protects inv and direct-store rows;
                                        # the receiver separately checks SDMA landing.
                                        comm.store_i64_global_system_relaxed(
                                            trb + fx.Int64(TRL["q"]) + fx.Int64(tok) * fx.Int64(8),
                                            i64_generation)
                                    else:
                                        q = fx.Int32(comm.atomic_add_system(
                                            trb + fx.Int64(TRL["tail"]), fx.Int32(1)))
                                        comm.store_i64_global_system_relaxed(
                                            trb + fx.Int64(TRL["q"]) + fx.Int64(q) * fx.Int64(8),
                                            ((i64_generation & fx.Int64(0xFFFFFFFF)) << fx.Int64(32))
                                            | fx.Int64(tok))

                if const_expr(TRK and sdma):
                    for w in range(tx_i32, fx.Int32(TR_OWNER_WORDS * WAVES + 2),
                                   fx.Int32(THREADS)):
                        fx.ptr_store(fx.Int32(0), _tr_lw(12 + w))
                    fx.barrier()
                t = _gs(claim())
                slot = fx.Int32(1)
                while t < remote_total:
                    nxt = fx.Int32(0)
                    if tx_i32 == fx.Int32(0):
                        nxt = fx.Int32(comm.atomic_add_agent(ticket_ptr, fx.Int32(1)))
                    m_block = tmap(t)
                    m_row0 = m_block * fx.Int32(BM)
                    a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
                    wait_group(a_row)
                    issue_all_a_loads(a_row)
                    if tx_i32 == fx.Int32(0):
                        fx.ptr_store(
                            nxt,
                            lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32, align=4),
                        )
                    rocdl.sched_barrier(0)
                    run_unit((m_block // fx.Int32(GS) if GS > 1 else m_block) * num_n_blocks
                             + dyn_n0, m_block, a_row, ticket=t)
                    fx.barrier()  # epilog LDS reads vs next A-load; also publishes the slot
                    if const_expr(sdma):
                        _sd_count(t, False)
                    if const_expr(unit_hook is not None):
                        if const_expr(TRK):
                            # The hook emits IR synchronously in this iteration; it never stores the callback.
                            unit_hook(_tr_lw, lambda: _tr_count_tx0(m_block), _tr_count_rows)  # noqa: B023
                        else:
                            unit_hook(_tr_lw, None, None)
                    t = _gs(fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
                        lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32,
                                      align=4)).ir_value())))
                    slot = fx.Int32(1) - slot
                # This CTA has no remote m-block left (its held ticket is local or
                # past the end).  The last CTA here tells the node peers that the
                # remote-source half of kernel1 has been pushed (word 8 + local
                # rank), so they can reduce and ship the remote plane while the
                # local half is still in GEMM2.
                if const_expr(not TRK):  # noqa: SIM102
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
                if const_expr(boundary_hook is not None and boundary_probe is not None):
                    # Part 1 (remote plane reduce + rail post) runs once every
                    # node peer's remote half is in.  Until then keep doing local
                    # units, probing (local words only) after each one; with the
                    # tickets gone it runs anyway and waits inside.
                    rdy = (t < total_m_blocks).select(fx.Int32(0), fx.Int32(1))
                    while rdy == fx.Int32(0):
                        nxt = fx.Int32(0)
                        if tx_i32 == fx.Int32(0):
                            nxt = fx.Int32(comm.atomic_add_agent(ticket_ptr, fx.Int32(1)))
                        m_block = tmap(t)
                        m_row0 = m_block * fx.Int32(BM)
                        a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
                        wait_group(a_row)
                        issue_all_a_loads(a_row)
                        if tx_i32 == fx.Int32(0):
                            fx.ptr_store(
                                nxt,
                                lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32, align=4),
                            )
                        rocdl.sched_barrier(0)
                        run_unit((m_block // fx.Int32(GS) if GS > 1 else m_block) * num_n_blocks
                                 + dyn_n0, m_block, a_row, ticket=t)
                        fx.barrier()  # epilog LDS reads vs next A-load; also publishes the slot
                        if const_expr(sdma and not SD_RO):
                            _sd_count(t, True)
                        t = _gs(fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(
                            lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32,
                                          align=4)).ir_value())))
                        slot = fx.Int32(1) - slot
                        boundary_probe(lds_typed_ptr(fx.Int32(lds_claim_off + 8), T.i32, align=4))
                        fx.barrier()
                        r = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lds_typed_ptr(
                            fx.Int32(lds_claim_off + 8), T.i32, align=4)).ir_value()))
                        rdy = (t < total_m_blocks).select(r, fx.Int32(1))
                    boundary_hook()
                elif const_expr(boundary_hook is not None):
                    boundary_hook()
                while t < total_m_blocks:
                    nxt = fx.Int32(0)
                    if tx_i32 == fx.Int32(0):
                        nxt = fx.Int32(comm.atomic_add_agent(ticket_ptr, fx.Int32(1)))
                    m_block = tmap(t)
                    m_row0 = m_block * fx.Int32(BM)
                    a_row = a_map_buf[m_row0 // fx.Int32(32)] * fx.Int32(32) + m_row0 % fx.Int32(32)
                    wait_group(a_row)
                    issue_all_a_loads(a_row)
                    if tx_i32 == fx.Int32(0):
                        fx.ptr_store(
                            nxt,
                            lds_typed_ptr(fx.Int32(lds_claim_off) + slot * fx.Int32(4), T.i32, align=4),
                        )
                    rocdl.sched_barrier(0)
                    run_unit((m_block // fx.Int32(GS) if GS > 1 else m_block) * num_n_blocks
                             + dyn_n0, m_block, a_row, ticket=t)
                    fx.barrier()  # epilog LDS reads vs next A-load; also publishes the slot
                    if const_expr(sdma and not SD_RO):
                        _sd_count(t, True)
                    if const_expr(unit_hook is not None):
                        unit_hook(_tr_lw, None, None)
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

        # Source completion covers inv and direct-store payload, independent
        # of the receiver's remote-token reductions. SDMA consumers still
        # wait for the engine signal. Do not gate the local plane on a remote
        # reducer's tail; only drain that tail before returning to kernel2.
        if const_expr(dynamic and tail_hook is not None):
            tail_hook(_tr_lw)


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
    launch.token_ready = TRK
    launch.sd_plan_bx = SD_PLAN_BX
    launch.ctas_per_cu = K1_OCC
    launch.threads = THREADS_K1
    launch.lds_bytes = int(lds_bytes)
    return launch


def peer_table_launcher(gpus_per_node: int, npes: int, *, lsa_peer_base: int = 0):
    """One-off kernel filling ``arg_p2p``: each peer's plane_slot_inbox base.

    CCO's ``lsa_ptr`` is device-only (the host cannot see a peer's VA), so the
    table is written by a tiny kernel.  The push target is the GPU with the
    owner's local index (``owner_rank & (gpus_per_node - 1)``), so only the
    first ``gpus_per_node`` entries are addressed; the rest hold peer 0's base
    so an index the valid bit already rejects is never a wild pointer.
    """
    local_peers = int(gpus_per_node)
    total = int(npes)
    peer_base = int(lsa_peer_base)
    if peer_base < 0 or peer_base % local_peers:
        raise ValueError("LSA peer base must align to a logical GPU node")

    @flyc.kernel(known_block_size=[64, 1, 1])
    def fill_peer_table(window: fx.Int64, table: fx.Int64, offset: fx.Int64):
        if fx.Int32(fx.thread_idx.x) == fx.Int32(0):
            for peer in range(local_peers):
                address = fx.Int64(cco.Window(window).lsa_ptr(fx.Int32(peer_base + peer), offset))
                comm.store_i64_global_relaxed(
                    table + fx.Int64(peer * 8), address)
            fallback = fx.Int64(cco.Window(window).lsa_ptr(fx.Int32(peer_base), offset))
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


def stage2_scratch_layout(source_nodes, max_tokens, chunk_tokens, num_qp=8):
    """Allocate disjoint i32 scratch spans from topology and chunk geometry."""
    nodes, tokens, chunk_size, qps = map(
        int, (source_nodes, max_tokens, chunk_tokens, num_qp))
    if nodes < 2 or min(tokens, chunk_size, qps) <= 0:
        raise ValueError("Stage2 scratch needs at least two nodes and positive geometry")
    chunks = _ceil_div(tokens, chunk_size)
    peer_qps = (nodes - 1) * qps
    # One WQE counter and one i64 RAIL request per (peer, QP), followed
    # by independent scalar state. Requests must never alias SDMA's epoch.
    req = _ceil_div(peer_qps, 16) * 16
    sdma_epoch = req + 2 * peer_qps
    final = sdma_epoch + 1
    chunk = _ceil_div(final + 1, 64) * 64
    p3 = chunk + nodes * chunks
    return {"wqe": 0, "req": req, "sdma_epoch": sdma_epoch,
            "final": final, "chunk": chunk, "p3": p3,
            "words": _ceil_div(p3 + chunks, 64) * 64}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _ceil_div_rt(a, b):
    """Compute runtime ceiling division, supporting tail chunks without divisibility."""
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
    """Convert VEC f32 values to packed i32 words, E8M0 scale, and block-leader flag.

    Match kernel1's p2p_scatter_epilog push path. For VEC=8, each 32-element scale
    block spans four lanes, so reduce maxima across lane^1 and lane^2.
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
    """Decode two i32 words (eight E4M3 bytes) and their E8M0 scale to Vector(8, f32).

    Follow moe_reduce.py:133-140. cvt_pk_f32_fp8 accepts one 32-bit source and a
    word selector, requiring four calls to decode two i32 words into eight f32 values.
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
    source_nodes: int = 2,
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
    threads: int = THREADS,
    token_ready: bool = False,
    rail_idx_off: int = 0,
    rail_idx_plane: int = 0,
    buffer_regions: tuple = (),
    gda_context_base: int = 0,
    lsa_peer_base: int = 0,
    gda_peer_stride: int = 1,
    gda_peer_lane: int = 0,
    gda_physical_width: int = 0,
    recv_overflow_off: int = 0,
    recv_overflow_plane: int = 0,
    forward_status_off: int = 0,
):
    """Build kernel2 with a threads-wide CTA (512 for the fused 8-wave path).

    All *_off values are byte offsets relative to window local_ptr. Stage1 writes
    epoch_gate at epoch_off earlier on the same stream; read it for generation.
    Offsets refer to parity 0. Add each parity_planes size when generation & 1
    selects parity 1, so captured graph replays read the current generation/parity.
    parity_planes order: inbox, slot_mask, accumulator, rx, partial_ready,
    group_ready, consumed, arrival.
    """
    parity_planes = tuple(int(v) for v in parity_planes)
    RECV_STATUS = bool(recv_overflow_off)
    if RECV_STATUS and (not forward_status_off or not recv_overflow_plane):
        raise ValueError("receive overflow status needs both status and parity offsets")
    if RECV_STATUS and token_ready:
        raise NotImplementedError("receive overflow propagation needs fixed-order return readiness")
    _rail = RailGda(gda_context_base, peer_stride=gda_peer_stride, peer_lane=gda_peer_lane,
                    rank=rank, physical_width=gda_physical_width)
    lsa_peer_base = int(lsa_peer_base)
    if lsa_peer_base < 0 or lsa_peer_base % int(gpus_per_node):
        raise ValueError("LSA peer base must align to a logical GPU node")
    region_indices = {(stage, name): index for stage, name, index in buffer_regions}
    NODES = int(source_nodes)
    if NODES < 2 or not 0 <= int(rank) < NODES * int(gpus_per_node):
        raise ValueError("rank must belong to the Stage2 node topology")
    if NODES != 2 and token_ready:
        raise NotImplementedError("multi-node Stage2 uses fixed-order rail")
    if buffer_regions and token_ready:
        raise NotImplementedError("separated buffers do not support token-ready")
    if epoch_off <= 0 or len(parity_planes) != 8:
        raise ValueError("kernel2 needs the epoch_gate offset and eight parity_planes")

    if hidden <= 0 or hidden % 512:
        # Require divisibility by 512, not 256: one wave covers 64 lanes x 8 BF16 values.
        # Otherwise tail columns would go unwritten. The upstream ABI already requires
        # hidden in [1024,8192] divisible by 512; also reject invalid direct calls here.
        raise ValueError("hidden must be a positive multiple of 512")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    # Every *_off must be window-relative and include stage2_offset. Region-relative
    # offsets silently redirect inter-node writes, so reject them at compile time.
    for _name in ("inbox_off", "slot_mask_off", "accumulator_off", "rx_off",
                  "partial_ready_off", "group_ready_off"):
        if int(locals()[_name]) < int(s2_window_off):
            raise ValueError(
                f"{_name} 小于 s2_window_off,看起来是 region 相对偏移;"
                "kernel2 要的是相对 window local_ptr 的偏移")
    if int(num_qp) not in (1, 2, 4, 8):
        raise ValueError("num_qp must be one of 1,2,4,8")
    # The reference maps QP to wave. Extra waves idle when waves>QPs; otherwise
    # each wave handles num_qp//waves QPs sequentially. Both mappings require divisibility.
    THREADS = int(threads)
    if THREADS not in (256, 512):
        raise ValueError(f"kernel2 threads must be 256 or 512, got {THREADS}")
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
        # return_group_ready has only max_tokens/4 slots; its granularity cannot be below four.
        raise ValueError("return_chunk_tokens must be >= 4")

    HIDDEN = int(hidden)
    MAX_TOKENS = int(max_tokens)
    TOPK = int(topk)
    CHUNK = int(return_chunk_tokens)
    # Use ceiling division: even CHUNK*TOPK < threads (for example, CHUNK=8) needs
    # one iteration. Bounds checks discard extra threads. Floor division previously
    # caused compilation failures when CHUNK changed.
    NUM_QP = int(num_qp)
    WAVES = THREADS // 64
    RECORD_BYTES = HIDDEN * 2               # One token's BF16 partial, without a header.
    # RECORD_BYTES describes rail PUT/GET bytes per token accumulator. Inbox rows
    # are per (token, topk slot), sized by kernel1 token_nbytes; these formats can
    # be quantized independently.
    # kernel1: row_base = slot * token_nbytes, token_nbytes = N_OUT + N_OUT//32
    if rail_quant not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(
            f"rail_quant must be 'none' or 'fp8_blockwise_1x32' "
            f"(got {rail_quant!r})")
    RAIL_FP8 = rail_quant == "fp8_blockwise_1x32"
    if RAIL_FP8:
        if HIDDEN % 32:
            raise ValueError("fp8 rail 要求 hidden 能被 32 整除(e8m0 块大小)")
        # Quantization halves RECORD_BYTES per token for PUT/GET, reducing the rail's
        # measured 78us line-rate payload cost.
        RECORD_BYTES = HIDDEN + HIDDEN // 32
        if RECORD_BYTES % 8:
            raise ValueError(
                f"rail 行宽 {RECORD_BYTES} 不是 8 的倍数,put 源地址会错位")
    INBOX_ROW_BYTES = (HIDDEN * 2)
    # Inbox bytes per plane; use separate descriptors when the two-plane inbox is >=2GiB.
    INBOX_PLANE_BYTES = MAX_TOKENS * TOPK * INBOX_ROW_BYTES
    INBOX_PER_PLANE = NODES * INBOX_PLANE_BYTES >= 1 << 31
    if INBOX_PLANE_BYTES >= 1 << 31:
        raise ValueError("inbox plane exceeds the 32-bit buffer-resource range")
    ACC_ROW_BYTES = RECORD_BYTES
    if INBOX_ROW_BYTES % 8:
        # VEC values per lane use one dwordx2/x4 load, requiring 8B-aligned row starts.
        raise ValueError(
            f"inbox 行宽 {INBOX_ROW_BYTES} 不是 8 的倍数,向量化 load 会错位")
    VEC = 8                                 # Eight BF16 values per lane = 16B.
    RED_BATCH = min(int(topk), 8)           # Outstanding top-k loads per reduction batch.
    RED_BATCHES = _ceil_div(int(topk), RED_BATCH)
    # Reduce units interleaved per wave step (see the reduce loop).  The 4-wave
    # fused stage2 keeps 1: its VGPRs bound kernel1's CTAs per CU.  8 waves: at
    # most 24 row loads in flight per wave (K3, RED_BATCH 8: U=4 ran 7-70x slow
    # in 4 of 8 TPR1024 cases, U=2 0 of 6; DSv4.1, RED_BATCH 6: U=4 clean).
    RED_U = (4 if 4 * RED_BATCH <= 24 else 2) if THREADS == 512 else 1
    CHUNKS_PER_WAVE = HIDDEN // (64 * VEC)  # One wave covers 512 columns.
    CHUNK_ITERS = _ceil_div(CHUNKS_PER_WAVE, WAVES)
    # Phase 1 assigns each (plane, token, hidden partition) to one wave.
    # HPARTS=CHUNKS_PER_WAVE gives exactly 512 columns per partition, avoiding
    # masked idle work. The old chunk mapping used four waves x two iterations
    # for seven useful slots, leaving one eighth idle.
    HPARTS = CHUNKS_PER_WAVE
    # [plane][chunk] completion counts identify the last wave to post the rail PUT.
    MAX_CHUNKS = _ceil_div(MAX_TOKENS, CHUNK)
    scratch = stage2_scratch_layout(NODES, MAX_TOKENS, CHUNK, NUM_QP)
    S_WQE, S_REQ, S_FINAL, S_CHUNK, S_P3 = (
        scratch[k] for k in ("wqe", "req", "final", "chunk", "p3"))
    S_SDMA_EPOCH = scratch["sdma_epoch"]

    node = int(rank) // int(gpus_per_node)
    REMOTE_NODES = tuple(n for n in range(NODES) if n != node)
    remote_node = REMOTE_NODES[0]
    local_plane = node
    remote_plane = remote_node
    RX_PEER_BYTES = int(rx_bytes) // (NODES - 1)
    GROUP_PEER_BYTES = parity_planes[5] // (NODES - 1)

    kernel_name = (
        f"megamoe_k2_h{HIDDEN}_t{MAX_TOKENS}_k{TOPK}"
        f"_qp{NUM_QP}_c{CHUNK}_w{WAVES}"
        # Include rail CQ CCQE polling mode in the kernel name, which is the FlyDSL cache key.
        + ("_ccqe" if _rail_ccqe() else "")
        + ("_qr8" if RAIL_FP8 else "")
        + f"_de{epoch_off}_" + "_".join(str(v) for v in parity_planes)

        # Layout offsets are part of the compiled-handle cache identity.
        + "_scratch" + "x".join(str(scratch[k]) for k in
                                  ("req", "sdma_epoch", "final", "chunk", "p3"))
        + ("_sd" if sdma else "") + (f"_ru{RED_U}" if RED_U > 1 else "")
        + ("_trf" if token_ready and sdma else "_tr" if token_ready else "")
        + ("_c1" if buffer_regions else "")
        + (f"_vl{gda_peer_stride}l{gda_peer_lane}b{lsa_peer_base}" if gda_peer_stride != 1 else "")
        + ("_recv_status_poster" if RECV_STATUS else "")
        + (f"_ctx{gda_context_base}" if gda_context_base else "")
        + (f"_n{NODES}_r{rank}" if NODES != 2 else "")
    )
    # Token-ready rail (fused remote-first only, see tr_layout): the remote
    # plane goes out in completion order, CHUNK tokens a batch, each batch
    # with its token ids; part 2 reduces only the local plane and combines
    # the partner's batches through those ids.
    TRC = bool(token_ready)
    if TRC and (INBOX_PER_PLANE or CHUNK < WAVES or not RAIL_FP8):
        raise ValueError("token_ready needs fp8 rail, one inbox descriptor and CHUNK >= waves")
    if TRC and (int(rail_idx_off) < int(s2_window_off) or int(rail_idx_plane) < 8 * MAX_TOKENS):
        raise ValueError("token_ready needs the rail_idx region")
    TRL2 = tr_layout(MAX_TOKENS, TOPK) if TRC else None
    RIDX_OFF, RIDX_PLANE = int(rail_idx_off), int(rail_idx_plane)
    # SDMA stage2 (fused remote-first only): kernel1's rows land packed in the
    # inbox; inv (in the arrival region, see compile_mega_moe_stage2) maps each
    # (plane, token, slot) to its row, and each node peer's copy engine signals
    # once after the remote half and once at the end of every forward.
    sdma = bool(sdma)
    SD_CTR_B, SD_INV_B = 128, 256
    SD_RB8 = HIDDEN + HIDDEN // 32  # FP8 row + E8M0 scales (see p2p_scatter_epilog)
    LOCAL_RANK = int(rank) % int(gpus_per_node)
    if sdma and 256 + NODES * MAX_TOKENS * TOPK * 4 > NODES * MAX_TOKENS * TOPK * 8:
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


        # As in kernel1: relaxed load without buffer_inv; readfirstlane keeps the value in SGPRs.
        epoch_ptr = arena_ptr + fx.Int64(epoch_off)
        if const_expr(bool(buffer_regions)):
            epoch_ptr = region_local_address(arena_ptr, region_indices[1, "epoch_gate"], fx.Int64(0))
        generation = fx.Int64(rocdl.readfirstlane(
            T.i64,
            fx.Int64(comm.load_i64_global_agent_relaxed(
                epoch_ptr)).ir_value()))
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
        rx_win = accumulator_win = group_ready_win = consumed_win = arena_win
        credit_ptr = arena_ptr + _consumed_off
        if const_expr(bool(buffer_regions)):
            inbox_ptr = region_local_address(arena_ptr, region_indices[2, "plane_slot_inbox"], _par)
            accum_ptr = region_local_address(arena_ptr, region_indices[2, "node_accumulator"], _par)
            rx_ptr = region_local_address(arena_ptr, region_indices[2, "remote_partial_rx"], _par)
            mask_ptr = region_local_address(arena_ptr, region_indices[2, "node_dest_slot_mask"], _par)
            ready_ptr = region_local_address(arena_ptr, region_indices[2, "node_partial_ready"], _par)
            arrival_ptr = region_local_address(arena_ptr, region_indices[2, "plane_slot_arrived"], _par)
            group_ready_ptr = region_local_address(arena_ptr, region_indices[2, "return_group_ready"], _par)
            credit_ptr = region_local_address(arena_ptr, region_indices[2, "return_consumed"], _par)
            rx_win, _rx_off = region_window_address(arena_ptr, region_indices[2, "remote_partial_rx"], _par)
            accumulator_win, _accumulator_off = region_window_address(arena_ptr, region_indices[2, "node_accumulator"], _par)
            group_ready_win, _group_ready_off = region_window_address(arena_ptr, region_indices[2, "return_group_ready"], _par)
            consumed_win, _consumed_off = region_window_address(arena_ptr, region_indices[2, "return_consumed"], _par)

        # This rank's Stage1 writes node_dest_slot_mask with local buffer_store
        # (stage1.py:891/:1487). kernel2 follows on the same rank and stream, with no
        # cross-GPU ordering dependency, so cache-bypassing system loads are unnecessary.
        # Previously both sites used load_i32_global_system_relaxed: once per (token,slot)
        # while waiting, repeating a token up to 16 times, and once per reduction unit,
        # for nearly 100,000 uncacheable reads.
        mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
            mask_ptr, num_records_bytes=fx.Int32(NODES * MAX_TOKENS * 4))
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
                num_records_bytes=fx.Int32(NODES * MAX_TOKENS * TOPK * 4))

        @flyc.jit
        def _wait_ready_relaxed(addr, expected):
            """Poll arrival flags with relaxed loads; the caller acquires once after its barrier.

            CCO wait_ready acquires and invalidates L2 on every poll. Batched waits check
            512 flags per chunk, repeated by about ten CTAs, usually finding them ready.
            Avoid roughly 80,000 invalidations and their measured 39us overhead.
            """
            word = fx.Int64(comm.load_i64_global_system_relaxed(addr))
            while word < expected:
                word = fx.Int64(comm.load_i64_global_system_relaxed(addr))

        @flyc.jit
        def _node_recv_overflow():
            # Stage1 reservations are final once all local GEMM2 arrival
            # releases have been acquired. Only the rail poster waves and
            # CTA0's final status writer need this immutable node-wide vote.
            overflow = fx.Int32(0)
            for _p in range_constexpr(int(gpus_per_node)):
                if const_expr(bool(buffer_regions)):
                    overflow_ptr = region_lsa_address(
                        arena_ptr, region_indices[1, "recv_overflow"], _par,
                        fx.Int32(lsa_peer_base + _p),
                    )
                else:
                    overflow_ptr = fx.Int64(
                        cco.Window(arena_win).lsa_ptr(
                            fx.Int32(_p),
                            fx.Int64(recv_overflow_off)
                            + _par * fx.Int64(recv_overflow_plane),
                        )
                    )
                flag = fx.Int32(comm.load_i32_global_system_relaxed(overflow_ptr))
                flag = fx.Int32(rocdl.readfirstlane(T.i32, flag.ir_value()))
                overflow = overflow | (
                    flag != fx.Int32(0)
                ).select(fx.Int32(1), fx.Int32(0))
            return overflow

        # Phase 1: reduce and post WQEs.
        # Reduce the remote plane first and ring its rail as soon as it is posted,
        # overlapping transfer with local-plane reduction. G CTAs share each chunk
        # and report completion once per CTA, preserving memory-operation overlap.
        # Per-unit waits would serialize this bandwidth path.
        _PLANES = REMOTE_NODES + (local_plane,)
        _rail_req = fx.Int64(0)
        # Every node peer's kernel1 publishes once, at its end (word = its
        # local rank).  Poll relaxed, acquire once.  part 1 (fused stage2,
        # remote-first kernel1) needs only the remote-source half, which is
        # published at words 8 + local rank.
        # part 0: the whole kernel2; part 1: remote plane + ring; part 2: rest.
        _flag0 = int(gpus_per_node) if part == 1 else 0
        if const_expr(TRC and sdma and part != 1):  # noqa: SIM102
            if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
                _tr_owner_step(dev_comm, arena_win, _par, generation, arrival_ptr,
                               local_tokens, bx, lane, drain=True)
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
                    arg_scratch + fx.Int64(S_SDMA_EPOCH * 4)))
                _sd = cco.DevComm(
                    region_local_dev_comm(arena_ptr, len(buffer_regions))
                    if buffer_regions else dev_comm).sdma()
                _want = fx.Int64(_epoch) * fx.Int64(2) + fx.Int64(part)
                for _p in range_constexpr(int(gpus_per_node)):
                    if const_expr(_p != LOCAL_RANK):
                        _wait_ready_relaxed(
                            fx.Int64(_sd.signal_slot(fx.Int32(_p), fx.Int32(0))), _want)
        gpu.barrier()
        comm.fence_system_acquire()
        return_generation = generation
        if const_expr(RECV_STATUS):
            return_generation = generation * fx.Int64(2)
        ck_local = bx % total_chunks
        sub = bx // total_chunks
        n_cta = (grid - ck_local + total_chunks - fx.Int32(1)) // total_chunks
        # The rewriter needs range_constexpr to reach nested runtime loops.
        _PIS = {0: tuple(range(NODES)), 1: tuple(range(NODES - 1)),
                2: (NODES - 1,)}[part]
        for _pii in range_constexpr(len(_PIS)):
          _pi = _PIS[_pii]
          plane = _PLANES[_pi]
          peer_index = plane - int(plane > node)
          destination_peer_index = node - int(node > plane)
          if const_expr(INBOX_PER_PLANE):
              # At mtpr>4096, the full inbox exceeds i32 num_records and masked out-of-range
              # reads stop returning zero. Use one descriptor per plane with plane-local token indices.
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
              # RED_U units per wave per step, interleaved: their slot masks,
              # then their inv words, then their rows go out together, so the
              # three dependent loads of a unit are paid once per RED_U units
              # (part 1, DSv4.1 TPR4096: 60us at ~1.4 TB/s was this chain).
              for u in range(ulo + unit0, uhi, ustep * fx.Int32(RED_U)):
                  _ok_u, _tok, _tix, _col, _sm, _tot = [], [], [], [], [], []
                  for _j in range_constexpr(RED_U):
                      uj = u + fx.Int32(_j) * ustep
                      okj = uj < uhi
                      ujc = okj.select(uj, u)  # a valid unit; its work is masked off
                      token = ck_first + ujc // fx.Int32(HPARTS)
                      hpart = ujc - (ujc // fx.Int32(HPARTS)) * fx.Int32(HPARTS)
                      _ok_u.append(okj)
                      _tok.append(token)
                      _tix.append(fx.Int32(plane * MAX_TOKENS) + token)
                      _col.append((hpart * fx.Int32(64) + lane) * fx.Int32(VEC))
                  for _j in range_constexpr(RED_U):
                      _sm.append(_ok_u[_j].select(fx.Int32(buffer_ops.buffer_load(
                          mask_rsrc, _tix[_j], vec_width=1, dtype=T.i32)), fx.Int32(0)))
                      _tot.append(fx.Vector.filled(VEC, 0.0, Float32))
                  # Follow MegaMoEv2's pipeline (intranode_kernel.py:1010-1085): issue loads into
                  # a list before accumulation, using masks rather than branches for validity.
                  # buffer_load(mask=) maps invalid offsets to 0x7FFFFFFF, returning zero through
                  # hardware bounds checks. Issue RED_BATCH slots per unit and accumulate before
                  # the next batch. Loading all TOPK at once held 64 VGPRs of values, taking
                  # kernel2 to 117 VGPRs and reducing fused Stage2 occupancy below kernel1's
                  # five CTAs per CU.
                  for _rb in range_constexpr(RED_BATCHES):
                      if const_expr(sdma):
                          # inv first for the whole batch, then the rows.
                          _oks, _recs = [], []
                          for _j in range_constexpr(RED_U):
                              _oj, _rj_l = [], []
                              for _rj in range_constexpr(RED_BATCH):
                                  slot = _rb * RED_BATCH + _rj
                                  if const_expr(slot < TOPK):
                                      ok = ((_sm[_j] >> fx.Int32(slot))
                                            & fx.Int32(1)) != fx.Int32(0)
                                      _oj.append(ok)
                                      _rj_l.append(fx.Int32(buffer_ops.buffer_load(
                                          inv_rsrc, _tix[_j] * fx.Int32(TOPK) + fx.Int32(slot),
                                          vec_width=1, dtype=T.i32, mask=ok)))
                              _oks.append(_oj)
                              _recs.append(_rj_l)
                          # Rows are FP8 + E8M0 (SD_RB8 bytes); a masked load
                          # reads 0, which decodes to 0.
                          _pk, _sc = [], []
                          for _j in range_constexpr(RED_U):
                              _pj, _cj = [], []
                              for _k in range_constexpr(len(_oks[_j])):
                                  _rec = _recs[_j][_k]
                                  if const_expr(TRC):
                                      _rec = _rec & fx.Int32(0x0FFFFFFF)
                                  if const_expr(INBOX_PER_PLANE):
                                      # One-row descriptor: the packed row index can
                                      # pass the i32 byte range.  The row is
                                      # wave-uniform (one token-slot per wave).
                                      _ru = fx.Int32(rocdl.readfirstlane(T.i32, _rec.ir_value()))
                                      _row_rsrc = buffer_ops.create_buffer_resource_from_addr(
                                          inbox_ptr + fx.Int64(_ru) * fx.Int64(SD_RB8),
                                          num_records_bytes=fx.Int32(SD_RB8))
                                      _b0 = fx.Int32(0)
                                  else:
                                      _row_rsrc = inbox_rsrc
                                      _b0 = _rec * fx.Int32(SD_RB8)
                                  _pj.append(buffer_ops.buffer_load(
                                      _row_rsrc, (_b0 + _col[_j]) // fx.Int32(4),
                                      vec_width=2, dtype=T.i32, mask=_oks[_j][_k]))
                                  _cj.append(buffer_ops.buffer_load(
                                      _row_rsrc, _b0 + fx.Int32(HIDDEN) + _col[_j] // fx.Int32(32),
                                      vec_width=1, dtype=T.i8, mask=_oks[_j][_k]))
                              _pk.append(_pj)
                              _sc.append(_cj)
                          for _j in range_constexpr(RED_U):
                              for _k in range_constexpr(len(_pk[_j])):
                                  _tot[_j] = _tot[_j] + _fp8_decode8(
                                      _pk[_j][_k],
                                      ((fx.Uint8(_sc[_j][_k]).to(fx.Uint32).bitcast(fx.Int32))
                                       << fx.Int32(23)).bitcast(Float32))
                      else:
                          vals = []
                          for _j in range_constexpr(RED_U):
                              _vj = []
                              for _rj in range_constexpr(RED_BATCH):
                                  slot = _rb * RED_BATCH + _rj
                                  if const_expr(slot < TOPK):
                                      ok = ((_sm[_j] >> fx.Int32(slot))
                                            & fx.Int32(1)) != fx.Int32(0)
                                      if const_expr(INBOX_PER_PLANE):
                                          rec = (_tok[_j] * fx.Int32(TOPK)
                                                 + fx.Int32(slot))
                                          src = rec * fx.Int32(HIDDEN) + _col[_j]
                                          _vj.append(buffer_ops.buffer_load(
                                              inbox_rsrc_p, src // fx.Int32(2),
                                              vec_width=4, dtype=T.i32, mask=ok))
                                      else:
                                          rec = (_tix[_j] * fx.Int32(TOPK)
                                                 + fx.Int32(slot))
                                          src = rec * fx.Int32(HIDDEN) + _col[_j]
                                          _vj.append(buffer_ops.buffer_load(
                                              inbox_rsrc, src // fx.Int32(2),
                                              vec_width=4, dtype=T.i32, mask=ok))
                              vals.append(_vj)
                          for _j in range_constexpr(RED_U):
                              for _v in vals[_j]:
                                  _tot[_j] = _tot[_j] + fx.Vector(_v).bitcast(
                                      BFloat16).to(Float32)
                      if const_expr(_rb + 1 < RED_BATCHES):
                          rocdl.sched_barrier(0)
                  # Keep the default accumulator store policy. ACCUM_WT=1 used sc0|sc1
                  # write-through to avoid per-chunk fence_system_release L2 writeback, but
                  # measured 194.5 versus 195.6 with no meaningful gain; the switch was removed.
                  _acm = 0
                  for _j in range_constexpr(RED_U):
                      totals = _tot[_j]
                      token_index = _tix[_j]
                      col = _col[_j]
                      # Both planes use comm_quant's format. The local partial
                      # is quantized too, even when all routes stay on this node.
                      if const_expr(RAIL_FP8):
                          _vals = [fx.Float32(totals[i])
                                   for i in range_constexpr(VEC)]
                          _w, _e8m0, _is_ldr = _fp8_encode8(_vals, lane, VEC=VEC)
                          _abyte = (token_index * fx.Int32(ACC_ROW_BYTES) + col)
                          buffer_ops.buffer_store(
                              fx.Vector.from_elements(_w, fx.Int32),
                              accum_rsrc, _abyte // fx.Int32(4),
                              mask=_ok_u[_j], cache_modifier=_acm)
                          # Mask the scale-byte store rather than branching, avoiding divergence and an exec-mask round trip.
                          buffer_ops.buffer_store(
                              _e8m0.to(fx.Int8), accum_rsrc,
                              token_index * fx.Int32(ACC_ROW_BYTES)
                              + fx.Int32(HIDDEN) + col // fx.Int32(32),
                              mask=_is_ldr & _ok_u[_j], cache_modifier=_acm)
                      # 17=sc0|sc1 in pre-gfx12 encoding (SIDefines.h:593, GLC=SC0=1, SCC=SC1=16).
                      # Write-through reaches the system coherence point for NIC reads of
                      # node_accumulator. Default stores leave dirty data in local L2, requiring
                      # fence_system_release's full buffer_wbl2 writeback once per chunk.
                      # Write-through removes that requirement, as kernel1 payload pay_cm=19 does.
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
                              mask=_ok_u[_j], cache_modifier=_acm)
              # Token-segment mode gives this CTA a contiguous byte range for its PUT.
              # Drain this CTA's stores before its acq_rel arrival. The last arrival observes
              # every partial before posting the chunk's WQE.
              fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
              gpu.barrier()
              cnt_idx = fx.Int32(S_CHUNK + plane * MAX_CHUNKS) + ck
              if wave == fx.Int32(0):
                  prev = fx.Int32(0)
                  if lane == fx.Int32(0):
                      prev = fx.Int32(comm.atomic_add_agent_acq_rel(
                          arg_scratch + fx.Int64(cnt_idx) * fx.Int64(4),
                          fx.Int32(1)))
                  # prev is valid only in lane0. Broadcast it so the entire wave enters the
                  # following branch together; PUT is a warp-level collective.
                  prev = fx.Int32(rocdl.readfirstlane(T.i32, prev.ir_value()))
                  if (prev + fx.Int32(1)) == n_cta:
                      # Publish the whole chunk to the NIC with one system fence.
                      comm.fence_system_release()
                      if lane == fx.Int32(0):
                          comm.store_i32_system(
                              arg_scratch, cnt_idx, fx.Int32(0))
                          # Phase 3 reads only the chunk's first token.
                          # Use a relaxed store: the preceding fence already wrote back data;
                          # a release store would issue buffer_wbl2 again.
                          comm.store_i64_global_system_relaxed(
                              ready_ptr
                              + fx.Int64(fx.Int32(plane * MAX_TOKENS) + ck_first)
                              * fx.Int64(8), generation)
                      if const_expr(plane != local_plane):
                          _qp = ck % fx.Int32(NUM_QP)
                          _rail.put(dev_comm, _qp, fx.Int32(plane),
                              rx_win,
                              _rx_off
                              + fx.Int64(destination_peer_index * RX_PEER_BYTES)
                              + fx.Int64(ck_first) * fx.Int64(RECORD_BYTES),
                              accumulator_win,
                              _accumulator_off
                              + fx.Int64(plane * MAX_TOKENS)
                              * fx.Int64(RECORD_BYTES)
                              + fx.Int64(ck_first) * fx.Int64(RECORD_BYTES),
                              fx.Int64(ck_tok) * fx.Int64(RECORD_BYTES),
                              aggregate=True)
                          fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                          if lane == fx.Int32(0):
                              comm.atomic_add_agent(
                                  arg_scratch
                                  + fx.Int64(S_WQE + peer_index * NUM_QP) * fx.Int64(4)
                                  + fx.Int64(_qp) * fx.Int64(4),
                                  fx.Int32(1))
          if const_expr(plane != local_plane):  # noqa: SIM102
            # One CTA's wave0 per QP rings it once all its WQEs are posted.
            # QPs are independent; this needs no CTA barrier.
            if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
              _want = _max_rt(
                  (total_chunks - bx + fx.Int32(NUM_QP - 1))
                  // fx.Int32(NUM_QP), fx.Int32(0))
              _posted = fx.Int32(comm.load_i32_global_system_relaxed(
                  arg_scratch + fx.Int64(S_WQE + peer_index * NUM_QP) * fx.Int64(4)
                  + fx.Int64(bx) * fx.Int64(4)))
              while _posted < _want:
                  _posted = fx.Int32(comm.load_i32_global_system_relaxed(
                      arg_scratch + fx.Int64(S_WQE + peer_index * NUM_QP) * fx.Int64(4)
                      + fx.Int64(bx) * fx.Int64(4)))
              comm.fence_system_acquire()
              node_overflow = fx.Int32(0)
              if const_expr(RECV_STATUS):
                  node_overflow = _node_recv_overflow()
              for chunk in range(bx, total_chunks, fx.Int32(NUM_QP)):
                  _rail.put_value(
                      dev_comm, bx, fx.Int32(plane), group_ready_win,
                      _group_ready_off + fx.Int64(destination_peer_index * GROUP_PEER_BYTES)
                      + fx.Int64(chunk) * fx.Int64(8),
                      return_generation + fx.Int64(node_overflow), aggregate=True)
              _rail_req = fx.Int64(_rail.flush_async(
                  dev_comm, bx, fx.Int32(plane)))
              if const_expr(part == 1 or NODES != 2):  # noqa: SIM102
                  if lane == fx.Int32(0):
                      comm.store_i64_global_relaxed(
                          arg_scratch + fx.Int64(S_REQ * 4 + peer_index * NUM_QP * 8)
                          + fx.Int64(bx) * fx.Int64(8), _rail_req)
        if const_expr(part != 1):
          # Phase 2: reap rail completion. The remote plane already rang the doorbell;
          # reap here after local-plane reduction.
          if const_expr(TRC):
           # Token-ready: this QP's owner waits until all of its batches are
           # posted (by whichever CTA completed them), covers the rest with
           # one more doorbell and reaps the last request.
           if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
               _nbt = _ceil_div_rt(local_tokens, fx.Int32(CHUNK))
               _nbq = _max_rt((_nbt - bx + fx.Int32(NUM_QP - 1)) // fx.Int32(NUM_QP), fx.Int32(0))
               _tb = arrival_ptr
               if const_expr(not sdma):
                _pa = _tb + fx.Int64(TRL2["posted"]) + fx.Int64(bx) * fx.Int64(4)
                _posted = fx.Int32(comm.load_i32_global_system_relaxed(_pa))
                while _posted < _nbq:
                    _posted = fx.Int32(comm.load_i32_global_system_relaxed(_pa))
                _rung = fx.Int32(comm.load_i32_global_system_relaxed(
                    _tb + fx.Int64(TRL2["rung"]) + fx.Int64(bx) * fx.Int64(4)))
                if _posted > _rung:
                    comm.fence_system_acquire()
                    _rq = fx.Int64(_rail.flush_async(dev_comm, bx, fx.Int32(remote_node)))
                    if lane == fx.Int32(0):
                        comm.store_i64_global_system_relaxed(
                            _tb + fx.Int64(TRL2["req"]) + fx.Int64(bx) * fx.Int64(8), _rq)
               if _nbq > fx.Int32(0):
                   _rail.wait(dev_comm, bx, fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(
                       comm.load_i64_global_system_relaxed(
                           _tb + fx.Int64(TRL2["req"]) + fx.Int64(bx) * fx.Int64(8))).ir_value())))
          else:
           if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
            for _peer_index in range_constexpr(NODES - 1):
                if const_expr(part == 2 or NODES != 2):
                    _rail_req = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(
                        comm.load_i64_global(
                            arg_scratch + fx.Int64(S_REQ * 4 + _peer_index * NUM_QP * 8)
                            + fx.Int64(bx) * fx.Int64(8))).ir_value()))
                _rail.wait(dev_comm, bx, _rail_req)
                if lane == fx.Int32(0):
                    comm.store_i32_system(
                        arg_scratch, fx.Int32(S_WQE + _peer_index * NUM_QP) + bx,
                        fx.Int32(0))
          # Phase 3: all CTAs perform inter-node reduction and unpermute output.
          # Both node partials are already reduced. Add and unpermute them;
          # synchronize once per CTA/chunk and distribute its work across waves.
          if const_expr(TRC):
              # A batch holds tokens of any local chunk: wait for all of them.
              for _c in range(tx, total_chunks, fx.Int32(THREADS)):
                  _wait_ready_relaxed(
                      ready_ptr
                      + fx.Int64(fx.Int32(local_plane * MAX_TOKENS) + _c * fx.Int32(CHUNK))
                      * fx.Int64(8), generation)
              _ridx_rx = (arena_ptr + fx.Int64(RIDX_OFF + MAX_TOKENS * 4)
                          + _par * fx.Int64(RIDX_PLANE))
          for ck in range(bx % total_chunks, total_chunks, grid):
            ck_first = ck * fx.Int32(CHUNK)
            ck_tok = _min_rt(fx.Int32(CHUNK), local_tokens - ck_first)
            if tx == fx.Int32(0):
                # Phase 1 publishes readiness for the whole chunk together, so wait only on
                # its first token instead of performing 32 separate waits.
                if const_expr(not TRC):
                    _wait_ready_relaxed(
                        ready_ptr
                        + fx.Int64(fx.Int32(local_plane * MAX_TOKENS) + ck_first)
                        * fx.Int64(8), generation)
                for _peer_index in range_constexpr(NODES - 1):
                    _wait_ready_relaxed(
                        group_ready_ptr + fx.Int64(_peer_index * GROUP_PEER_BYTES)
                        + fx.Int64(ck) * fx.Int64(8), return_generation)
            gpu.barrier()
            comm.fence_system_acquire()
            if const_expr(RECV_STATUS):  # noqa: SIM102
                if (bx == fx.Int32(0)) & (tx == fx.Int32(0)):
                    overflow = _node_recv_overflow()
                    for _peer_index in range_constexpr(NODES - 1):
                        word = fx.Int64(comm.load_i64_global_system_relaxed(
                            group_ready_ptr + fx.Int64(_peer_index * GROUP_PEER_BYTES)
                            + fx.Int64(ck) * fx.Int64(8)))
                        overflow = overflow | fx.Int32(word & fx.Int64(1))
                    # The error bit is constant across this forward's chunks.
                    # CTA0 observes the local vote and every peer's first
                    # chunk, so one writer suffices for this rank's status.
                    if overflow != fx.Int32(0):
                        status_ptr = arena_ptr + fx.Int64(forward_status_off)
                        if const_expr(bool(buffer_regions)):
                            status_ptr = region_local_address(
                                arena_ptr, region_indices[1, "forward_status"], fx.Int64(0)
                            )
                        comm.store_i32_global_system_relaxed(status_ptr, fx.Int32(1))
            n_cta = (grid - ck + total_chunks - fx.Int32(1)) // total_chunks
            sub = bx // total_chunks
            for u in range(sub * fx.Int32(WAVES) + wave,
                           ck_tok * fx.Int32(HPARTS),
                           n_cta * fx.Int32(WAVES)):
                token = ck_first + u // fx.Int32(HPARTS)
                hpart = u - (u // fx.Int32(HPARTS)) * fx.Int32(HPARTS)
                col = (hpart * fx.Int32(64) + lane) * fx.Int32(VEC)
                rx_row = token
                if const_expr(TRC):
                    # rx row = batch position; its token came with the batch.
                    token = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(
                        comm.load_i32_global_system_relaxed(
                            _ridx_rx + fx.Int64(rx_row) * fx.Int64(4))).ir_value()))
                local_index = ((fx.Int32(local_plane * MAX_TOKENS) + token)
                               * fx.Int32(HIDDEN) + col)
                out_index = token * fx.Int32(HIDDEN) + col
                # FP8 mode decodes both node partials before summing; only
                # the remote partial crossed the rail.
                if const_expr(RAIL_FP8):
                    _lrow = ((fx.Int32(local_plane * MAX_TOKENS) + token)
                             * fx.Int32(ACC_ROW_BYTES))
                    _rrow = rx_row * fx.Int32(ACC_ROW_BYTES)
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
                totals = [fx.Float32(local_values[i]) + fx.Float32(remote_values[i])
                          for i in range_constexpr(VEC)]
                for _peer_index in range_constexpr(1, NODES - 1):
                    if const_expr(RAIL_FP8):
                        _rrow = (fx.Int32(_peer_index * RX_PEER_BYTES)
                                 + rx_row * fx.Int32(ACC_ROW_BYTES))
                        remote_packed = buffer_ops.buffer_load(
                            rx_rsrc, (_rrow + col) // fx.Int32(4),
                            vec_width=2, dtype=T.i32)
                        _rsc = buffer_ops.buffer_load(
                            rx_rsrc, _rrow + fx.Int32(HIDDEN) + col // fx.Int32(32),
                            vec_width=1, dtype=T.i8)
                        remote_values = _fp8_decode8(
                            remote_packed,
                            ((fx.Uint8(_rsc).to(fx.Uint32).bitcast(fx.Int32))
                             << fx.Int32(23)).bitcast(Float32))
                    else:
                        remote_packed = buffer_ops.buffer_load(
                            rx_rsrc,
                            fx.Int32(_peer_index * RX_PEER_BYTES // 4)
                            + out_index // fx.Int32(2), vec_width=4, dtype=T.i32)
                        remote_values = fx.Vector(remote_packed).bitcast(BFloat16).to(Float32)
                    totals = [totals[i] + fx.Float32(remote_values[i])
                              for i in range_constexpr(VEC)]
                result = fx.Vector.from_elements(totals, Float32).to(BFloat16).bitcast(fx.Int32)
                buffer_ops.buffer_store(
                    result, out_rsrc, out_index // fx.Int32(2))
            # Increment once per CTA/chunk. FINAL_DONE is local to this GPU, read by CTA0
            # at finalization, so agent scope suffices. System visibility is needed only
            # when sending peer credits, which already performs a release.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            if tx == fx.Int32(0):
                prev = fx.Int32(comm.atomic_add_agent_acq_rel(
                    arg_scratch + fx.Int64(S_P3 + ck) * fx.Int64(4),
                    fx.Int32(1)))
                if (prev + fx.Int32(1)) == n_cta:
                    comm.store_i32_system(
                        arg_scratch, fx.Int32(S_P3) + ck, fx.Int32(0))
                    comm.atomic_add_system_acq_rel(
                        arg_scratch + fx.Int64(S_FINAL) * fx.Int64(4),
                        ck_tok)
          fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
          gpu.barrier()
          # Finalization: exchange return_consumed credits.
          # remote_partial_rx has two parity slots, reused by generations N and N+2.
          # Without this handshake, nodes can drift by two generations and newer ready
          # values satisfy the old generation's >= wait, exposing a partially written
          # buffer without a hang. The handshake is therefore required for correctness.
          if bx == fx.Int32(0):
            done_ptr = arg_scratch + fx.Int64(S_FINAL) * fx.Int64(4)
            if tx == fx.Int32(0):
                done = fx.Int32(comm.load_i32_global_system(done_ptr))
                while done < local_tokens:
                    done = fx.Int32(comm.load_i32_global_system(done_ptr))
                # Subtract the completed generation's count so the next starts from zero.
                comm.atomic_add_system_acq_rel(
                    done_ptr, fx.Int32(0) - local_tokens)
                if const_expr(sdma):
                    # Every node peer claimed its landing rows here before its
                    # data (which we have read): free them for generation + 2,
                    # and count this forward's two signals per peer.
                    comm.store_i32_global_system_relaxed(
                        arrival_ptr + fx.Int64(SD_CTR_B), fx.Int32(0))
                    comm.atomic_add_agent(
                        arg_scratch + fx.Int64(S_SDMA_EPOCH * 4), fx.Int32(1))
            gpu.barrier()
            if wave == fx.Int32(0):
                # Send every acknowledgement before waiting for peer credits.
                # The receive arrays use sorted peer indices, excluding self.
                _consumed_requests = []
                for _peer_index in range_constexpr(NODES - 1):
                    _peer_node = REMOTE_NODES[_peer_index]
                    _destination_index = node - int(node > _peer_node)
                    _rail.put_value(dev_comm, fx.Int32(0), fx.Int32(_peer_node),
                                    consumed_win, _consumed_off + fx.Int64(_destination_index * 8),
                                    generation, aggregate=True)
                    _consumed_requests.append(_rail.flush_async(
                        dev_comm, fx.Int32(0), fx.Int32(_peer_node)))
                for _peer_index in range_constexpr(NODES - 1):
                    if lane == fx.Int32(0):
                        _credit_ptr = credit_ptr + fx.Int64(_peer_index * 8)
                        _credit = fx.Int64(comm.load_i64_global_system(_credit_ptr))
                        while _credit < generation:
                            _credit = fx.Int64(comm.load_i64_global_system(_credit_ptr))
                    _rail.wait(dev_comm, fx.Int32(0), _consumed_requests[_peer_index])

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

    @flyc.jit
    def k2_remote_ready(dev_comm: fx.Int64, arena_ptr: fx.Int64, arg_scratch: fx.Int64, word_ptr):
        """Non-blocking form of part 1's opening wait: thread 0 stores 1 into the
        LDS word ``word_ptr`` when every node peer has published its remote-source
        half (and, with SDMA, the copy engine signalled it), else 0.  The caller
        broadcasts the word (barrier) so the whole CTA takes the same branch."""
        tx = fx.Int32(gpu.thread_id("x"))
        if tx == fx.Int32(0):
            epoch_ptr = arena_ptr + fx.Int64(epoch_off)
            if const_expr(bool(buffer_regions)):
                epoch_ptr = region_local_address(arena_ptr, region_indices[1, "epoch_gate"], fx.Int64(0))
            generation = fx.Int64(comm.load_i64_global_agent_relaxed(
                epoch_ptr))
            _par = generation & fx.Int64(1)
            arrival_ptr = arena_ptr + fx.Int64(arrival_off) + _par * fx.Int64(parity_planes[7])
            if const_expr(bool(buffer_regions)):
                arrival_ptr = region_local_address(arena_ptr, region_indices[2, "plane_slot_arrived"], _par)
            ok = fx.Int32(1)
            for _p in range_constexpr(int(gpus_per_node)):
                w = fx.Int64(comm.load_i64_global_system_relaxed(
                    arrival_ptr + fx.Int64((int(gpus_per_node) + _p) * 8)))
                ok = (w < generation).select(fx.Int32(0), ok)
            if const_expr(sdma):
                _epoch = fx.Int32(comm.load_i32_global_system_relaxed(
                    arg_scratch + fx.Int64(S_SDMA_EPOCH * 4)))
                _sd = cco.DevComm(
                    region_local_dev_comm(arena_ptr, len(buffer_regions))
                    if buffer_regions else dev_comm).sdma()
                _want = fx.Int64(_epoch) * fx.Int64(2) + fx.Int64(1)
                for _p in range_constexpr(int(gpus_per_node)):
                    if const_expr(_p != LOCAL_RANK):
                        w = fx.Int64(comm.load_i64_global_system_relaxed(
                            fx.Int64(_sd.signal_slot(fx.Int32(_p), fx.Int32(0)))))
                        ok = (w < _want).select(fx.Int32(0), ok)
            fx.ptr_store(ok, word_ptr)

    def _tr_gen(arena_ptr):
        gen = fx.Int64(rocdl.readfirstlane(T.i64, fx.Int64(
            comm.load_i64_global_agent_relaxed(arena_ptr + fx.Int64(epoch_off))).ir_value()))
        return gen, gen & fx.Int64(1)

    def _tr_words(arena_ptr, par):
        return arena_ptr + fx.Int64(arrival_off) + par * fx.Int64(parity_planes[7])

    def _tr_enqueue(trb, tok, g32):
        q = fx.Int32(comm.atomic_add_system(trb + fx.Int64(TRL2["tail"]), fx.Int32(1)))
        comm.store_i64_global_system_relaxed(
            trb + fx.Int64(TRL2["q"]) + fx.Int64(q) * fx.Int64(8),
            (g32 << fx.Int64(32)) | fx.Int64(tok))

    @flyc.jit
    def k2_tr_init(arena_ptr: fx.Int64, local_tokens: fx.Int32, bx: fx.Int32, grid: fx.Int32):
        """Every CTA, before kernel1: clear the other parity's token-ready words
        (stream order: its last forward is over and the next one cannot start
        before ours ends), then add -E to this parity's counters (E = rows of
        the token its node peers will compute), enqueuing the tokens it closes."""
        tx = fx.Int32(gpu.thread_id("x"))
        gen, par = _tr_gen(arena_ptr)
        g32 = gen & fx.Int64(0xFFFFFFFF)
        cur = _tr_words(arena_ptr, par)
        oth = _tr_words(arena_ptr, fx.Int64(1) - par)
        step = grid * fx.Int32(THREADS)
        for t in range(bx * fx.Int32(THREADS) + tx, fx.Int32(MAX_TOKENS), step):
            comm.store_i32_global_system_relaxed(
                oth + fx.Int64(TRL2["cnt"]) + fx.Int64(t) * fx.Int64(4), fx.Int32(0))
        if bx == fx.Int32(0):
            for w in range(tx, fx.Int32(MAX_TOKENS // 4 + 1), fx.Int32(THREADS)):
                comm.store_i32_global_system_relaxed(
                    oth + fx.Int64(TRL2["bdone"]) + fx.Int64(w) * fx.Int64(4), fx.Int32(0))
                if const_expr(sdma):
                    comm.store_i32_global_system_relaxed(
                        oth + fx.Int64(TRL2["issued"]) + fx.Int64(w) * fx.Int64(4), fx.Int32(0))
            if tx < fx.Int32(8):
                comm.store_i32_global_system_relaxed(
                    oth + fx.Int64(TRL2["posted"]) + fx.Int64(tx) * fx.Int64(4), fx.Int32(0))
                comm.store_i32_global_system_relaxed(
                    oth + fx.Int64(TRL2["rung"]) + fx.Int64(tx) * fx.Int64(4), fx.Int32(0))
            if tx == fx.Int32(0):
                if const_expr(not sdma):
                    comm.store_i32_global_system_relaxed(oth + fx.Int64(TRL2["tail"]), fx.Int32(0))
                comm.store_i32_global_system_relaxed(oth + fx.Int64(TRL2["head"]), fx.Int32(0))
        mask_base = arena_ptr + fx.Int64(slot_mask_off) + par * fx.Int64(parity_planes[1])
        for t in range(bx * fx.Int32(THREADS) + tx, local_tokens, step):
            m = fx.Int32(comm.load_i32_global_system_relaxed(
                mask_base + fx.Int64(fx.Int32(remote_plane * MAX_TOKENS) + t) * fx.Int64(4)))
            e = fx.Int32(0)
            for _k in range_constexpr(TOPK):
                e = e + ((m >> fx.Int32(_k)) & fx.Int32(1))
            old = fx.Int32(comm.atomic_add_system(
                cur + fx.Int64(TRL2["cnt"]) + fx.Int64(t) * fx.Int64(4), fx.Int32(0) - e))
            if old == e:
                if const_expr(sdma):
                    comm.store_i64_global_system_relaxed(
                        cur + fx.Int64(TRL2["q"]) + fx.Int64(t) * fx.Int64(8), gen)
                else:
                    _tr_enqueue(cur, t, g32)

    @flyc.jit
    def _tr_landed(arena_ptr, par, cur, g32, pos):
        # One probe, never a wait while a CTA holds its next GEMM2 ticket.
        # The queue tail can precede the entry store. A compute-ready token
        # can also precede the last row of its SDMA chunk.
        qe = fx.Int64(comm.load_i64_global_system_relaxed(
            cur + fx.Int64(TRL2["q"]) + fx.Int64(pos) * fx.Int64(8)))
        good = fx.Int32(0)
        if qe.shrui(fx.Int64(32)) == g32:
            tok = fx.Int32(qe & fx.Int64(0xFFFFFFFF))
            ti = fx.Int32(remote_plane * MAX_TOKENS) + tok
            sm = fx.Int32(comm.load_i32_global_system_relaxed(
                arena_ptr + fx.Int64(slot_mask_off) + par * fx.Int64(parity_planes[1])
                + fx.Int64(ti) * fx.Int64(4)))
            good = fx.Int32(1)
            for sk in range_constexpr(TOPK):
                if ((sm >> fx.Int32(sk)) & fx.Int32(1)) != fx.Int32(0):
                    inv = fx.Int32(comm.load_i32_global_system_relaxed(
                        cur + fx.Int64(SD_INV_B)
                        + fx.Int64(ti * fx.Int32(TOPK) + fx.Int32(sk)) * fx.Int64(4)))
                    if inv < fx.Int32(0):
                        src = inv.shrui(fx.Int32(28)) & fx.Int32(7)
                        prog = fx.Int64(comm.load_i64_global_system_relaxed(
                            cur + fx.Int64(TRL2["prog"]) + fx.Int64(src) * fx.Int64(8)))
                        landed = ((prog.shrui(fx.Int64(32)) == g32)
                                  & ((prog & fx.Int64(0xFFFFFFFF))
                                     > fx.Int64(inv & fx.Int32(0x0FFFFFFF))))
                        good = landed.select(good, fx.Int32(0))
        return good

    @flyc.jit
    def _tr_landed_wave(arena_ptr, par, cur, g32, pos, lane):
        # One token per wave, one slot per lane. A single probe never holds
        # a GEMM2 ticket waiting for a copy-engine chunk to complete.
        qe = fx.Int64(0)
        if lane == fx.Int32(0):
            qe = fx.Int64(comm.load_i64_global_system_relaxed(
                cur + fx.Int64(TRL2["q"]) + fx.Int64(pos) * fx.Int64(8)))
        qe = fx.Int64(rocdl.readfirstlane(T.i64, qe.ir_value()))
        if const_expr(sdma):
            good = qe == g32  # full generation, token-indexed compute-ready word
            tok = pos
        else:
            good = qe.shrui(fx.Int64(32)) == g32
            tok = fx.Int32(qe & fx.Int64(0xFFFFFFFF))
        if good:
            ti = fx.Int32(remote_plane * MAX_TOKENS) + tok
            sm = fx.Int32(0)
            if lane == fx.Int32(0):
                sm = fx.Int32(comm.load_i32_global_system_relaxed(
                    arena_ptr + fx.Int64(slot_mask_off) + par * fx.Int64(parity_planes[1])
                    + fx.Int64(ti) * fx.Int64(4)))
            sm = fx.Int32(rocdl.readfirstlane(T.i32, sm.ir_value()))
            if lane < fx.Int32(TOPK):  # noqa: SIM102
                if ((sm >> lane) & fx.Int32(1)) != fx.Int32(0):
                    inv = fx.Int32(comm.load_i32_global_system_relaxed(
                        cur + fx.Int64(SD_INV_B)
                        + fx.Int64(ti * fx.Int32(TOPK) + lane) * fx.Int64(4)))
                    if inv < fx.Int32(0):
                        src = inv.shrui(fx.Int32(28)) & fx.Int32(7)
                        prog = fx.Int64(comm.load_i64_global_system_relaxed(
                            cur + fx.Int64(TRL2["prog"]) + fx.Int64(src) * fx.Int64(8)))
                        good = ((prog.shrui(fx.Int64(32)) == (g32 & fx.Int64(0xFFFFFFFF)))
                            & ((prog & fx.Int64(0xFFFFFFFF))
                               > fx.Int64(inv & fx.Int32(0x0FFFFFFF))))
        missing = fx.Int64(rocdl.ballot(T.i64, (~good).ir_value()))
        return (missing == fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))

    @flyc.jit
    def _tr_reduce(arena_ptr, par, cur, g32, pos, lane, wait_landing=True, queue_pos=None,
                   direct_token=None):
        """One wave: wait for queue entry pos, reduce its token's remote-plane
        rows (all HPARTS, RED_U at a time) into accumulator row pos, record
        the token id for the batch."""
        if const_expr(direct_token is not None):
            tok = direct_token
        else:
            qp = pos
            if const_expr(queue_pos is not None):
                qp = queue_pos
            qa = cur + fx.Int64(TRL2["q"]) + fx.Int64(qp) * fx.Int64(8)
            e = fx.Int64(0)
            if lane == fx.Int32(0):
                e = fx.Int64(comm.load_i64_global_system_relaxed(qa))
                while (e >> fx.Int64(32)) != g32:
                    fx.rocdl.s_sleep(1)
                    e = fx.Int64(comm.load_i64_global_system_relaxed(qa))
            if const_expr(sdma and wait_landing):  # noqa: SIM102
                if lane == fx.Int32(0):
                    landed = fx.Int32(_tr_landed(arena_ptr, par, cur, g32, pos))
                    while landed == fx.Int32(0):
                        fx.rocdl.s_sleep(1)
                        landed = fx.Int32(_tr_landed(arena_ptr, par, cur, g32, pos))
            tok = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(e & fx.Int64(0xFFFFFFFF)).ir_value()))
        tix = fx.Int32(remote_plane * MAX_TOKENS) + tok
        sm = fx.Int32(0)
        if lane == fx.Int32(0):
            sm = fx.Int32(comm.load_i32_global_system_relaxed(
                arena_ptr + fx.Int64(slot_mask_off) + par * fx.Int64(parity_planes[1])
                + fx.Int64(tix) * fx.Int64(4)))
        sm = fx.Int32(rocdl.readfirstlane(T.i32, sm.ir_value()))
        # Landing map is immutable once this token has passed readiness.
        # Fetch each valid row once rather than for every hidden-part group.
        tr_rows = []
        if const_expr(sdma):
            for slot in range_constexpr(TOPK):
                row = fx.Int32(0)
                if lane == fx.Int32(0):  # noqa: SIM102
                    if ((sm >> fx.Int32(slot)) & fx.Int32(1)) != fx.Int32(0):
                        rec = fx.Int32(comm.load_i32_global_system_relaxed(
                            cur + fx.Int64(SD_INV_B)
                            + fx.Int64(tix * fx.Int32(TOPK) + fx.Int32(slot)) * fx.Int64(4)))
                        row = (rec & fx.Int32(0x0FFFFFFF)) * fx.Int32(SD_RB8)
                tr_rows.append(fx.Int32(rocdl.readfirstlane(T.i32, row.ir_value())))
        inbox_rs = buffer_ops.create_buffer_resource_from_addr(
            arena_ptr + fx.Int64(inbox_off) + par * fx.Int64(parity_planes[0]),
            num_records_bytes=fx.Int32(inbox_bytes))
        acc_rs = buffer_ops.create_buffer_resource_from_addr(
            arena_ptr + fx.Int64(accumulator_off) + par * fx.Int64(parity_planes[2]),
            num_records_bytes=fx.Int32(accumulator_bytes))
        arow = (fx.Int32(remote_plane * MAX_TOKENS) + pos) * fx.Int32(ACC_ROW_BYTES)
        for _hg in range_constexpr(0, HPARTS, RED_U):
            hps = list(range(_hg, min(_hg + RED_U, HPARTS)))
            cols = [(fx.Int32(_hp * 64) + lane) * fx.Int32(VEC) for _hp in hps]
            tots = [fx.Vector.filled(VEC, 0.0, Float32) for _hp in hps]
            for _rb in range_constexpr(RED_BATCHES):
                vals, scales = [], []
                for _i in range_constexpr(len(hps)):
                    _vi = []
                    for _rj in range_constexpr(RED_BATCH):
                        slot = _rb * RED_BATCH + _rj
                        if const_expr(slot < TOPK):
                            ok = ((sm >> fx.Int32(slot)) & fx.Int32(1)) != fx.Int32(0)
                            if const_expr(sdma):
                                row = tr_rows[slot]
                                _vi.append(buffer_ops.buffer_load(inbox_rs,
                                    (row + cols[_i]) // fx.Int32(4), vec_width=2,
                                    dtype=T.i32, mask=ok, cache_modifier=17))
                                scales.append(buffer_ops.buffer_load(inbox_rs,
                                    row + fx.Int32(HIDDEN) + cols[_i] // fx.Int32(32),
                                    vec_width=1, dtype=T.i8, mask=ok, cache_modifier=17))
                            else:
                                src = (tix * fx.Int32(TOPK) + fx.Int32(slot)) * fx.Int32(HIDDEN) + cols[_i]
                                _vi.append(buffer_ops.buffer_load(
                                    inbox_rs, src // fx.Int32(2), vec_width=4, dtype=T.i32,
                                    mask=ok, cache_modifier=17))
                    vals.append(_vi)
                for _i in range_constexpr(len(hps)):
                    for _k in range_constexpr(len(vals[_i])):
                        if const_expr(sdma):
                            sc = scales[_i * len(vals[_i]) + _k]
                            tots[_i] = tots[_i] + _fp8_decode8(vals[_i][_k],
                                ((fx.Uint8(sc).to(fx.Uint32).bitcast(fx.Int32))
                                 << fx.Int32(23)).bitcast(Float32))
                        else:
                            tots[_i] = tots[_i] + fx.Vector(vals[_i][_k]).bitcast(BFloat16).to(Float32)
                if const_expr(_rb + 1 < RED_BATCHES):
                    rocdl.sched_barrier(0)
            for _i in range_constexpr(len(hps)):
                _vals = [fx.Float32(tots[_i][k]) for k in range_constexpr(VEC)]
                _w, _e8m0, _is_ldr = _fp8_encode8(_vals, lane, VEC=VEC)
                buffer_ops.buffer_store(
                    fx.Vector.from_elements(_w, fx.Int32), acc_rs, (arow + cols[_i]) // fx.Int32(4))
                buffer_ops.buffer_store(
                    _e8m0.to(fx.Int8), acc_rs, arow + fx.Int32(HIDDEN) + cols[_i] // fx.Int32(32),
                    mask=_is_ldr)
        if lane == fx.Int32(0):
            comm.store_i32_global_system_relaxed(
                arena_ptr + fx.Int64(RIDX_OFF) + par * fx.Int64(RIDX_PLANE)
                + fx.Int64(pos) * fx.Int64(4), tok)

    @flyc.jit
    def _tr_post(dev_comm, arena_win, par, gen, cur, b, local_tokens, lane, publish=True):
        """Wave 0 of the CTA that completed batch b: rows, ids, then its ready
        word, on QP b % NUM_QP; the QP's owner rings (see k2_tr_step)."""
        comm.fence_system_release()
        qp = b % fx.Int32(NUM_QP)
        pos0 = b * fx.Int32(CHUNK)
        nt = _min_rt(fx.Int32(CHUNK), local_tokens - pos0)
        rb = fx.Int64(RECORD_BYTES)
        _rail.put(dev_comm, qp, fx.Int32(remote_node), arena_win,
                  fx.Int64(rx_off) + par * fx.Int64(parity_planes[3]) + fx.Int64(pos0) * rb,
                  arena_win,
                  fx.Int64(accumulator_off) + par * fx.Int64(parity_planes[2])
                  + fx.Int64(remote_plane * MAX_TOKENS) * rb + fx.Int64(pos0) * rb,
                  fx.Int64(nt) * rb, aggregate=True)
        ridx = fx.Int64(RIDX_OFF) + par * fx.Int64(RIDX_PLANE)
        _rail.put(dev_comm, qp, fx.Int32(remote_node), arena_win,
                  ridx + fx.Int64(MAX_TOKENS * 4) + fx.Int64(pos0) * fx.Int64(4),
                  arena_win, ridx + fx.Int64(pos0) * fx.Int64(4),
                  fx.Int64(nt) * fx.Int64(4), aggregate=True)
        _rail.put_value(dev_comm, qp, fx.Int32(remote_node), arena_win,
                        fx.Int64(group_ready_off) + par * fx.Int64(parity_planes[5])
                        + fx.Int64(b) * fx.Int64(8), gen, aggregate=True)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        if const_expr(publish):  # noqa: SIM102
            if lane == fx.Int32(0):
                comm.atomic_add_agent(cur + fx.Int64(TRL2["posted"]) + fx.Int64(qp) * fx.Int64(4),
                                      fx.Int32(1))

    @flyc.jit
    def _tr_owner_step(dev_comm, arena_win, par, gen, cur, local_tokens, qp, lane, drain=False):
        # Only wave 0 of CTA qp touches this QP. Reducers publish bdone after
        # their drained output stores; this owner posts and rings serially.
        # No post/flush overlap and no posted/rung handoff are required.
        nb = _ceil_div_rt(local_tokens, fx.Int32(CHUNK))
        if const_expr(((MAX_TOKENS + CHUNK - 1) // CHUNK) <= NUM_QP):
            if qp < nb:
                issued = fx.Int32(0)
                if lane == fx.Int32(0):
                    issued = fx.Int32(comm.load_i32_global_system_relaxed(
                        cur + fx.Int64(TRL2["issued"]) + fx.Int64(qp) * fx.Int64(4)))
                issued = fx.Int32(rocdl.readfirstlane(T.i32, issued.ir_value()))
                if issued == fx.Int32(0):
                    done = fx.Int32(0)
                    if lane == fx.Int32(0):
                        done = fx.Int32(comm.load_i32_global_system_relaxed(
                            cur + fx.Int64(TRL2["bdone"]) + fx.Int64(qp) * fx.Int64(4)))
                    done = fx.Int32(rocdl.readfirstlane(T.i32, done.ir_value()))
                    want = _min_rt(fx.Int32(CHUNK), local_tokens - qp * fx.Int32(CHUNK))
                    if const_expr(drain):
                        while done < want:
                            fx.rocdl.s_sleep(1)
                            if lane == fx.Int32(0):
                                done = fx.Int32(comm.load_i32_global_system_relaxed(
                                    cur + fx.Int64(TRL2["bdone"]) + fx.Int64(qp) * fx.Int64(4)))
                            done = fx.Int32(rocdl.readfirstlane(T.i32, done.ir_value()))
                    if done == want:
                        comm.fence_system_acquire()
                        _tr_post(dev_comm, arena_win, par, gen, cur, qp, local_tokens,
                                 lane, publish=False)
                        rq = fx.Int64(_rail.flush_async(dev_comm, qp, fx.Int32(remote_node)))
                        if lane == fx.Int32(0):
                            comm.store_i64_global_system_relaxed(
                                cur + fx.Int64(TRL2["req"]) + fx.Int64(qp) * fx.Int64(8), rq)
                            comm.store_i32_global_system_relaxed(
                                cur + fx.Int64(TRL2["issued"]) + fx.Int64(qp) * fx.Int64(4),
                                fx.Int32(1))
        else:
            nb = _ceil_div_rt(local_tokens, fx.Int32(CHUNK))
            more = fx.Int32(1)
            while more != fx.Int32(0):
                more = fx.Int32(0)
                for b in range(qp, nb, fx.Int32(NUM_QP)):
                    issued = fx.Int32(0)
                    if lane == fx.Int32(0):
                        issued = fx.Int32(comm.load_i32_global_system_relaxed(
                            cur + fx.Int64(TRL2["issued"]) + fx.Int64(b) * fx.Int64(4)))
                    issued = fx.Int32(rocdl.readfirstlane(T.i32, issued.ir_value()))
                    if issued == fx.Int32(0):
                        done = fx.Int32(0)
                        if lane == fx.Int32(0):
                            done = fx.Int32(comm.load_i32_global_system_relaxed(
                                cur + fx.Int64(TRL2["bdone"]) + fx.Int64(b) * fx.Int64(4)))
                        done = fx.Int32(rocdl.readfirstlane(T.i32, done.ir_value()))
                        want = _min_rt(fx.Int32(CHUNK), local_tokens - b * fx.Int32(CHUNK))
                        if done == want:
                            comm.fence_system_acquire()
                            _tr_post(dev_comm, arena_win, par, gen, cur, b, local_tokens,
                                     lane, publish=False)
                            rq = fx.Int64(_rail.flush_async(dev_comm, qp, fx.Int32(remote_node)))
                            if lane == fx.Int32(0):
                                comm.store_i64_global_system_relaxed(
                                    cur + fx.Int64(TRL2["req"]) + fx.Int64(qp) * fx.Int64(8), rq)
                                comm.store_i32_global_system_relaxed(
                                    cur + fx.Int64(TRL2["issued"]) + fx.Int64(b) * fx.Int64(4),
                                    fx.Int32(1))
                        else:
                            more = fx.Int32(1)
                if const_expr(drain):
                    if more != fx.Int32(0):
                        fx.rocdl.s_sleep(1)
                else:
                    more = fx.Int32(0)

    @flyc.jit
    def k2_tr_step(dev_comm, arena_win, arena_ptr, local_tokens, bx, lw, drain, pre=None,
                   post=None):
        """One round of the rail role for a CTA between its GEMM2 units: the
        QP owner rings what was posted; then the CTA claims up to WAVES ready
        tokens (any queued ones when draining), one per wave, reduces them and
        posts every batch it completes.  Returns the tokens claimed (uniform).
        lw(i): LDS words 0..3 for the broadcasts.  pre(): extra thread-0 work
        and post(): extra whole-CTA work after the same barrier (kernel1's
        m-block count).  With nothing claimed it costs that one barrier."""
        tx = fx.Int32(gpu.thread_id("x"))
        wave = tx // fx.Int32(64)
        lane = tx % fx.Int32(64)
        gen, par = _tr_gen(arena_ptr)
        g32 = gen & fx.Int64(0xFFFFFFFF)
        cur = _tr_words(arena_ptr, par)
        if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
            _posted = fx.Int32(comm.load_i32_global_system_relaxed(
                cur + fx.Int64(TRL2["posted"]) + fx.Int64(bx) * fx.Int64(4)))
            _rung = fx.Int32(comm.load_i32_global_system_relaxed(
                cur + fx.Int64(TRL2["rung"]) + fx.Int64(bx) * fx.Int64(4)))
            if _posted > _rung:
                comm.fence_system_acquire()  # the posters' WQEs (as the per-chunk ring did)
                _rq = fx.Int64(_rail.flush_async(dev_comm, bx, fx.Int32(remote_node)))
                if lane == fx.Int32(0):
                    comm.store_i64_global_system_relaxed(
                        cur + fx.Int64(TRL2["req"]) + fx.Int64(bx) * fx.Int64(8), _rq)
                    comm.store_i32_global_system_relaxed(
                        cur + fx.Int64(TRL2["rung"]) + fx.Int64(bx) * fx.Int64(4), _posted)
        if tx == fx.Int32(0):
            if const_expr(pre is not None):
                pre()
            tail = fx.Int32(comm.load_i32_global_system_relaxed(cur + fx.Int64(TRL2["tail"])))
            head = fx.Int32(comm.load_i32_global_system_relaxed(cur + fx.Int64(TRL2["head"])))
            p0 = fx.Int32(0)
            n = fx.Int32(0)
            if const_expr(drain):
                # Out of GEMM2 tickets: claim even ahead of the queue tail; the
                # missing tokens only wait on other CTAs' GEMM2.
                if head < local_tokens:
                    p0 = fx.Int32(comm.atomic_add_agent(cur + fx.Int64(TRL2["head"]),
                                                        fx.Int32(WAVES)))
                    n = _max_rt(_min_rt(local_tokens - p0, fx.Int32(WAVES)), fx.Int32(0))
            else:
                # Between GEMM2 units only queued tokens: a CTA holding its next
                # ticket must never wait on a row that ticket (or one held by a
                # CTA waiting like it) computes.  One CAS, no retry.
                ready_n = _max_rt(_min_rt(_min_rt(tail, local_tokens) - head,
                                         fx.Int32(WAVES)), fx.Int32(0))
                if const_expr(sdma):
                    # First broadcast a snapshot; the waves check it together.
                    p0, n = head, ready_n
                else:
                    if ready_n > fx.Int32(0):
                        _old, _ok = fx.llvm.atomic_cas(
                            fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
                                        cur + fx.Int64(TRL2["head"])),
                            head, head + ready_n, syncscope=rocdl.SyncScope.Agent)
                        if _ok:
                            p0 = head
                            n = ready_n
            fx.ptr_store(p0, lw(0))
            fx.ptr_store(n, lw(1))
        gpu.barrier()
        if const_expr(post is not None):
            post()
        p0 = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(0)).ir_value()))
        n = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(1)).ir_value()))
        if const_expr(sdma and not drain):  # noqa: SIM102
            if n != fx.Int32(0):
                eligible = fx.Int32(0)
                if wave < n:
                    eligible = fx.Int32(_tr_landed_wave(
                        arena_ptr, par, cur, g32, p0 + wave, lane))
                if lane == fx.Int32(0):
                    fx.ptr_store(eligible, lw(4 + wave))
                gpu.barrier()
                if tx == fx.Int32(0):
                    ready_n = n
                    for w in range_constexpr(WAVES):
                        landed = fx.Int32(fx.ptr_load(lw(4 + w)))
                        ready_n = ((landed == fx.Int32(0)) & (fx.Int32(w) < ready_n)).select(
                            fx.Int32(w), ready_n)
                    claimed = fx.Int32(0)
                    if ready_n > fx.Int32(0):
                        _old, _ok = fx.llvm.atomic_cas(
                            fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
                                        cur + fx.Int64(TRL2["head"])),
                            p0, p0 + ready_n, syncscope=rocdl.SyncScope.Agent)
                        claimed = _ok.select(ready_n, fx.Int32(0))
                    fx.ptr_store(claimed, lw(1))
                gpu.barrier()
                n = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(1)).ir_value()))
        if n != fx.Int32(0):
            _tr_work(dev_comm, arena_win, arena_ptr, local_tokens, bx, lw, par, gen, g32, cur,
                     p0, n, tx, wave, lane, wait_landing=drain)
        return n

    @flyc.jit
    def _tr_work(dev_comm, arena_win, arena_ptr, local_tokens, bx, lw, par, gen, g32, cur,
                 p0, n, tx, wave, lane, wait_landing=True, queue_base=None, direct_token=None):
        if const_expr(direct_token is not None):
            offset = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(4 + wave)).ir_value()))
            if offset >= fx.Int32(0):
                _tr_reduce(arena_ptr, par, cur, g32, p0 + offset, lane,
                           wait_landing=False, direct_token=direct_token)
        elif const_expr(queue_base is not None):
            offset = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(4 + wave)).ir_value()))
            if offset >= fx.Int32(0):
                _tr_reduce(arena_ptr, par, cur, g32, p0 + offset, lane,
                           wait_landing=False, queue_pos=queue_base + wave)
        else:
            if wave < n:
                _tr_reduce(arena_ptr, par, cur, g32, p0 + wave, lane, wait_landing=wait_landing)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        gpu.barrier()
        if tx == fx.Int32(0):
            f0 = fx.Int32(0)
            f1 = fx.Int32(0)
            if n > fx.Int32(0):
                b0 = p0 // fx.Int32(CHUNK)
                b1 = (p0 + n - fx.Int32(1)) // fx.Int32(CHUNK)
                c0 = _min_rt(n, (b0 + fx.Int32(1)) * fx.Int32(CHUNK) - p0)
                z0 = _min_rt(fx.Int32(CHUNK), local_tokens - b0 * fx.Int32(CHUNK))
                r0 = fx.Int32(comm.atomic_add_agent_acq_rel(
                    cur + fx.Int64(TRL2["bdone"]) + fx.Int64(b0) * fx.Int64(4), c0))
                f0 = ((r0 + c0) == z0).select(b0 + fx.Int32(1), fx.Int32(0))
                if b1 != b0:
                    c1 = n - c0
                    z1 = _min_rt(fx.Int32(CHUNK), local_tokens - b1 * fx.Int32(CHUNK))
                    r1 = fx.Int32(comm.atomic_add_agent_acq_rel(
                        cur + fx.Int64(TRL2["bdone"]) + fx.Int64(b1) * fx.Int64(4), c1))
                    f1 = ((r1 + c1) == z1).select(b1 + fx.Int32(1), fx.Int32(0))
            if const_expr(not sdma):
                fx.ptr_store(f0, lw(2))
                fx.ptr_store(f1, lw(3))
        if const_expr(not sdma):
            gpu.barrier()
            f0 = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(2)).ir_value()))
            f1 = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(3)).ir_value()))
            gpu.barrier()
            if wave == fx.Int32(0):
                if f0 != fx.Int32(0):
                    _tr_post(dev_comm, arena_win, par, gen, cur, f0 - fx.Int32(1), local_tokens, lane)
                if f1 != fx.Int32(0):
                    _tr_post(dev_comm, arena_win, par, gen, cur, f1 - fx.Int32(1), local_tokens, lane)

    @flyc.jit
    def k2_tr_sdma_step(dev_comm, arena_win, arena_ptr, local_tokens, bx, lw, drain,
                        pre=None, post=None, owner_grid=256):
        # Fixed CTA/wave ownership replaces the global queue, scan and CAS.
        # Each service round probes one token per wave without waiting while
        # a CTA holds a GEMM ticket. The cursor covers all assigned tokens.
        tx = fx.Int32(gpu.thread_id("x"))
        wave, lane = tx // fx.Int32(64), tx % fx.Int32(64)
        gen, par = _tr_gen(arena_ptr)
        cur = _tr_words(arena_ptr, par)
        rounds = _ceil_div(MAX_TOKENS, owner_grid * WAVES)
        words = _ceil_div(rounds, 32)
        cursor_word = 12 + words * WAVES
        done_word = cursor_word + 1
        # Once all fixed-owner tokens have been reduced, GEMM2 may still
        # have tickets. Preserve producer publication and QP progress, but
        # stop probing completed tokens and packing empty service rounds.
        stride = fx.Int32(owner_grid * WAVES)
        want = (local_tokens // stride) * fx.Int32(WAVES) + _max_rt(
            _min_rt(local_tokens % stride - bx * fx.Int32(WAVES), fx.Int32(WAVES)),
            fx.Int32(0))
        done_before = fx.Int32(rocdl.readfirstlane(T.i32,
            fx.ptr_load(lw(done_word)).ir_value()))
        service = done_before < want
        if tx == fx.Int32(0):
            if const_expr(pre is not None):
                pre()
            if service:
                cursor = fx.Int32(fx.ptr_load(lw(cursor_word)))
                fx.ptr_store(cursor, lw(0))
                fx.ptr_store((cursor + fx.Int32(1)) % fx.Int32(rounds), lw(cursor_word))
        if const_expr(pre is not None):  # noqa: SIM114
            gpu.barrier()
        elif service:
            gpu.barrier()
        if const_expr(post is not None):
            post()
        n = fx.Int32(0)
        if service:
            cursor = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(0)).ir_value()))
            tok = bx * fx.Int32(WAVES) + wave + cursor * fx.Int32(owner_grid * WAVES)
            bit = fx.Int32(1) << (cursor % fx.Int32(32))
            bitmap_ptr = lw(12 + (cursor // fx.Int32(32)) * fx.Int32(WAVES) + wave)
            got = fx.Int32(0)
            if tok < local_tokens:
                bitmap = fx.Int32(0)
                if lane == fx.Int32(0):
                    bitmap = fx.Int32(fx.ptr_load(bitmap_ptr))
                bitmap = fx.Int32(rocdl.readfirstlane(T.i32, bitmap.ir_value()))
                if (bitmap & bit) == fx.Int32(0):
                    got = fx.Int32(_tr_landed_wave(arena_ptr, par, cur, gen, tok, lane))
                    if lane == fx.Int32(0):  # noqa: SIM102
                        if got != fx.Int32(0):
                            fx.ptr_store(bitmap | bit, bitmap_ptr)
            if lane == fx.Int32(0):
                fx.ptr_store(got, lw(4 + wave))
            gpu.barrier()
            if tx == fx.Int32(0):
                n = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    have = fx.Int32(fx.ptr_load(lw(4 + w)))
                    fx.ptr_store((have != fx.Int32(0)).select(n, fx.Int32(-1)), lw(4 + w))
                    n = n + have
                packed = fx.Int32(0)
                if n != fx.Int32(0):
                    packed = fx.Int32(comm.atomic_add_agent(cur + fx.Int64(TRL2["head"]), n))
                fx.ptr_store(packed, lw(0))
                fx.ptr_store(n, lw(1))
                done = fx.Int32(fx.ptr_load(lw(done_word)))
                fx.ptr_store(done + n, lw(done_word))
            gpu.barrier()
            packed = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(0)).ir_value()))
            n = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(1)).ir_value()))
            if n != fx.Int32(0):
                _tr_work(dev_comm, arena_win, arena_ptr, local_tokens, bx, lw, par, gen, gen, cur,
                         packed, n, tx, wave, lane, direct_token=tok)
        if (bx < fx.Int32(NUM_QP)) & (wave == fx.Int32(0)):
            _tr_owner_step(dev_comm, arena_win, par, gen, cur, local_tokens, bx, lane)
        if const_expr(drain):
            if n == fx.Int32(0):
                fx.rocdl.s_sleep(16)
            if tx == fx.Int32(0):
                done = fx.Int32(fx.ptr_load(lw(done_word)))
                # lw3 remains stable until all waves cross the next barrier.
                fx.ptr_store((done < want).select(fx.Int32(1), fx.Int32(0)), lw(3))
            gpu.barrier()
            return fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lw(3)).ir_value()))
        return n

    launch.kernel_name = kernel_name
    launch.body = k2_body
    launch.remote_ready = k2_remote_ready
    launch.tr_init = k2_tr_init
    launch.tr_step = k2_tr_sdma_step if sdma else k2_tr_step
    launch.token_ready = TRC
    launch.threads = THREADS
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
    from .stage2_pipeline import compile_stage2_pipeline

    pipeline = compile_stage2_pipeline(k1_kw, k2_kw, k2_ctas)
    k1, k2 = pipeline.k1, pipeline.k2
    grid_ctas = pipeline.grid
    k1_occ = int(k1.ctas_per_cu)
    kernel_name = "megamoe_s2f_" + k1.kernel_name[len("megamoe_"):] + "__" + k2.kernel_name[len("megamoe_"):]
    threads = pipeline.threads
    # 4 waves: keep kernel1's occupancy (the grid assumes k1_occ CTAs per CU);
    # 8 waves: one CTA per CU, two waves per SIMD.
    wpe = k1_occ if threads == 256 else 2

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def fused(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, arg_arrival_count: fx.Int64,
        i64_generation: fx.Int64, arg_a_tile_map: fx.Int64, i32_local_rank: fx.Int32,
        dev_comm: fx.Int64, arena_win: fx.Int64, arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64, arg_scratch: fx.Int64, local_tokens: fx.Int32,
        arg_expert_count: fx.Int64, arg_sd_send: fx.Int64, arg_sd_tb: fx.Int64,
        arg_sd_plan: fx.Int64, arg_sd_cdone: fx.Int64, arg_tr_mcnt: fx.Int64, arg_tr_sdprog: fx.Int64):
    # fmt: on
        pipeline.body(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum,
                      arg_stids, arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks,
                      i32_inter, i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map,
                      i32_local_rank, dev_comm, arena_win, arena_ptr, arg_output_bf16,
                      arg_scratch, local_tokens, arg_expert_count, arg_sd_send, arg_sd_tb,
                      arg_sd_plan, arg_sd_cdone, arg_tr_mcnt, arg_tr_sdprog)

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
        arg_sd_plan: fx.Int64, arg_sd_cdone: fx.Int64, arg_tr_mcnt: fx.Int64, arg_tr_sdprog: fx.Int64, stream: fx.Stream):
    # fmt: on
        fused(arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_stids,
              arg_sweights, arg_trb, arg_p2p_comb_inp, i32_max_m_blocks, i32_inter,
              i32_hidden, arg_arrival_count, i64_generation, arg_a_tile_map, i32_local_rank,
              dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch, local_tokens,
              arg_expert_count, arg_sd_send, arg_sd_tb, arg_sd_plan, arg_sd_cdone, arg_tr_mcnt, arg_tr_sdprog,
              value_attrs={"rocdl.waves_per_eu": wpe,
                           "rocdl.flat_work_group_size": f"{threads},{threads}"},
              ).launch(grid=(grid_ctas, 1, 1), block=(threads, 1, 1), stream=stream)

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
                                               "arg_sd_cdone", "arg_tr_mcnt", "arg_tr_sdprog")),
        stream,
    )
