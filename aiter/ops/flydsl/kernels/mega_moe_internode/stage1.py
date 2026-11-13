# SPDX-License-Identifier: MIT
"""One-launch EP16 dispatch + A4W4 GMM1 + SiLU Stage-1.

This is the first production-shape specialization.  It deliberately keeps the
MORI InterNodeV1 semantic boundary:

* the caller quantizes BF16 once per source token (MXFP4 + E8M0, outside
  this kernel); Stage-1 only copies the packed row into its record;
* one activation row is copied per selected destination rank on the source node;
* one record crosses RAIL per source token / remote node;
* the aligned proxy copies its activation once per selected destination rank;
* each destination rank expands every locally owned top-k slot into an expert
  route that gathers the shared activation row.

The 4096-byte record carries a 16-bit top-k-slot bitmap per EP rank.  This keeps
node/rank payload deduplication while preserving duplicate-rank routes and
their independent expert IDs and weights.

For the sparse production path, a full BM32 expert tile is release-published
to an in-kernel ready queue by its last route arrival.  Finished communication
CTAs immediately rejoin as GMM1 consumers, overlapping remaining fanout with
GMM1, SiLU and A4 requant.  Partial expert tails are padded and published only
after the eight communication EOS signals are acquired.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import rocdl as _mlir_rocdl
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    _device_trap,
)
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_workgroup as _atomic_add_wg,
)

from ..mxfp4_gemm_common import k_tiles_total_for
from . import communication as comm
from .buffer_addressing import (
    region_local_address,
    region_lsa_address,
    region_window_address,
)
from .communication import RailGda
from .communication import ccqe_enabled as _rail_ccqe
from .communication import rail as _default_rail

# GEMM helper changes must also invalidate this module's kernel cache.
from .gemm import (
    _bm_constants,
    _gemm1_body,
)
from .layout import MAX_FUSED_TOKENS_PER_RANK, Stage1ArenaLayout

BM = 32
BN = 256
BK = 256
THREADS = 256
WAVES = THREADS // 64
CCO_TICKET = 0
# Persistent grid.  The role split (comm / producer / compute tickets) is
# designed for 256 CTAs, one per CU.
WORKER_BLOCKS = 256


def _plane_bytes(layout: Stage1ArenaLayout, name: str) -> int:
    region = layout.region(name)
    if not region.shape or region.shape[0] != layout.parity_depth:
        raise ValueError(f"{name} is not parity indexed")
    return region.nbytes // layout.parity_depth


# Long kernel names can exceed disk-cache filename limits. Reuse launchers
# in-process; all window and weight addresses are runtime arguments.
_COMPILED: dict = {}


def compile_stage1(layout: Stage1ArenaLayout, stage2_layout, **kwargs):
    """Compile (or reuse) one EP16 Stage-1 persistent-kernel shape specialization."""
    key = (
        repr(layout),
        repr(stage2_layout),
        _rail_ccqe(),
        tuple(sorted(kwargs.items())),
    )
    launcher = _COMPILED.get(key)
    if launcher is None:
        launcher = _compile_stage1(layout, stage2_layout, **kwargs)
        _COMPILED[key] = launcher
    return launcher


def _compile_stage1(
    layout: Stage1ArenaLayout,
    stage2_layout,
    *,
    rank: int,
    stage2_window_offset: int = 0,
    fanout_ctas_per_dest: int = 2,
    data_send_ctas: int = 0,
    split_local: bool = False,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    waves: int = 4,
    group_ready: bool = False,
    buffer_regions: tuple = (),
    lsa_peer_base: int = 0,
    gda_peer_stride: int = 1,
    gda_peer_lane: int = 0,
    gda_physical_width: int = 0,
):
    """Compile one EP16 Stage-1 persistent-kernel shape specialization.

    ``group_ready`` (single fused kernel): GMM1 writes h1 write-through and
    counts its jobs per physical tile group; the last job of a group stores
    the generation into ``h1_grp_sync`` so GEMM2 can start on that group.

    ``arena_ptr`` is the base of the one registered two-kernel window.
    ``stage2_window_offset`` locates the logical :class:`Stage2ArenaLayout`
    inside that same window.  The launcher performs exactly one GPU launch.
    """

    # Select the GMM1 path and its associated switches from the tune table's split_local column.
    split_local = bool(split_local)
    worker_blocks = WORKER_BLOCKS
    early_local_gmm = split_local
    fan2_shards = 16 if split_local else 0
    extra_consumers = split_local
    static_g0 = not split_local
    # Unsplit groups publish row counts only after their batch stores retire.
    # Reservation completion fixes the final row counts before consumers pad tails.
    # Metadata/payload writes use system write-through, so publication needs no
    # repeated L2 writeback; consumers acquire once per job to preserve weight reuse.
    # Keep constexpr guards outside runtime guards: FlyDSL rewrites them separately.
    tile_ready = lite_sync = fan_wt = not split_local
    # ready_q: enqueue a group as soon as it fills. Dispatch reports arriving rows
    # in batches; the atomic that reaches 32*G rows appends the physical group ID
    # to the destination's gmm1_ready_q. Sealer phases 1/2 append local/remote tails.
    # GMM1 consumers claim jobs after their own fanout, without waiting for both
    # h1_local_eos and h1_queue_eos. Arrival counts require write-through payload
    # and metadata, as tile_ready does, so also enable fan_wt.
    # Without split (static_g0), consumer 0 pads tail arrival counts after all
    # reservations complete. Whichever atomic fills GROUP_ROWS enqueues the group;
    # see _rq_pad_tails.
    ready_q = True
    if ready_q:
        fan_wt = True
    # buffer_store cache_modifier, pre-gfx12 encoding: 1=sc0(GLC), 16=sc1(SCC), 2=nt(SLC).
    _WT_META = 17 if fan_wt else 0
    _WT_PAY = 19 if fan_wt else 0

    if not isinstance(layout, Stage1ArenaLayout):
        raise TypeError("layout must be Stage1ArenaLayout")
    HIDDEN = int(layout.hidden)
    A_DTYPE = layout.activation_dtype
    INTER = int(layout.inter)
    EXPERTS = int(layout.experts)
    WORLD = int(layout.world_size)
    GPUS_PER_NODE = int(layout.gpus_per_node)
    LOCAL_EXPERTS = int(layout.local_experts)
    TOPK = int(layout.topk)
    BM = int(layout.block_m)
    # Claim G adjacent BM32 tiles per expert; Stage2 retains the BM32 ABI.
    G = int(layout.tile_group)
    GBM = BM * G
    # CTA width: 4 waves (one per SIMD, up to 512 registers each) or 8 waves
    # (two per SIMD, <= 256 registers each; GMM1 gives each wave a 32-column
    # slice over all rows, see _gemm1_body).
    WAVES = int(waves)
    THREADS = 64 * WAVES
    if WAVES not in (4, 8):
        raise ValueError(f"stage1 waves must be 4 or 8, got {WAVES}")
    group_ready = bool(group_ready)
    GROUP_ROWS = BM * G
    BN = int(layout.block_n)
    if (
        WORLD not in (16, 32, 64)
        or GPUS_PER_NODE not in (4, 8)
        or WORLD // GPUS_PER_NODE not in (2, 4, 8)
    ):
        raise ValueError(
            "Stage-1 transport requires 2, 4 or 8 logical nodes of 4 or 8 GPUs"
        )
    if GPUS_PER_NODE == 4 and not buffer_regions:
        raise NotImplementedError(
            "four-GPU logical nodes require separated CCO buffers"
        )
    lsa_peer_base = int(lsa_peer_base)
    if lsa_peer_base < 0 or lsa_peer_base % GPUS_PER_NODE:
        raise ValueError("LSA peer base must align to a logical GPU node")
    _rail = (
        _default_rail
        if (gda_peer_stride, gda_peer_lane) == (1, 0)
        else RailGda(
            peer_stride=gda_peer_stride,
            peer_lane=gda_peer_lane,
            rank=rank,
            physical_width=gda_physical_width,
        )
    )
    if HIDDEN < 1024 or HIDDEN % 512:
        raise ValueError("Stage-1 hidden must be >= 1024 and divisible by 512")
    if HIDDEN > 8192:
        raise ValueError(
            "Stage-1 hidden must be <= 8192: one thread copies each 1x32 group"
        )
    if INTER <= 0 or INTER % BK:
        raise ValueError(f"Stage-1 inter must be positive and divisible by {BK}")
    if EXPERTS <= 0 or EXPERTS % WORLD:
        raise ValueError("Stage-1 experts must be positive and divisible by EP size")
    if LOCAL_EXPERTS > 256:
        raise ValueError("Stage-1 requires experts/world_size <= 256")
    if not 1 <= TOPK <= 16:
        raise ValueError("Stage-1 topk must be in [1,16]")
    if BN != 256 or BM not in (32, 64, 128):
        raise ValueError("Stage-1 requires BN=256 and BM in (32, 64, 128)")
    if activation not in ("silu", "situv2"):
        raise ValueError("Stage-1 activation must be silu or situv2")
    situ_beta, situ_linear_beta = float(situ_beta), float(situ_linear_beta)
    if activation == "situv2" and any(
        not math.isfinite(v) or v <= 0 for v in (situ_beta, situ_linear_beta)
    ):
        raise ValueError("Stage-1 situv2 requires finite positive beta / linear_beta")
    if swiglu_limit is not None and (
        activation != "silu" or not float(swiglu_limit) > 0.0
    ):
        raise ValueError(
            "Stage-1 swiglu_limit applies to silu only and must be positive"
        )
    # Keep the progress grid resident even when token capacity grows.
    MAX_TOKENS = int(layout.max_tokens)
    BOUNDED_RECV = layout.route_capacity < (
        layout.source_capacity * layout.max_routes_per_token_per_rank
    )
    COMPACT_INPUT = layout.input_capacity < layout.source_capacity
    COMM_CTAS = GPUS_PER_NODE
    PRODUCER_FIRST = COMM_CTAS
    PRODUCER_CTAS = min(128, MAX_TOKENS)
    ESSENTIAL_CTAS = PRODUCER_FIRST + PRODUCER_CTAS
    # Producers may rejoin GEMM; communication CTAs must retain their rail
    # credit/reclaim role until the protocol completes, otherwise progress stalls.
    COMPUTE_FIRST = min(8, ESSENTIAL_CTAS)
    INVALID_SOURCE = WORLD * MAX_TOKENS
    if not 1 <= MAX_TOKENS <= MAX_FUSED_TOKENS_PER_RANK:
        raise ValueError(
            f"Stage-1 max_tokens must be in [1, {MAX_FUSED_TOKENS_PER_RANK}]"
        )
    if fanout_ctas_per_dest < 1 or GPUS_PER_NODE * fanout_ctas_per_dest > worker_blocks:
        # Fanout may borrow waiting compute CTAs; the whole grid must stay resident.
        raise ValueError(
            "fanout_ctas_per_dest must satisfy 1 <= GPUS_PER_NODE*shards <= "
            f"{worker_blocks}"
        )
    if not 0 <= int(rank) < WORLD:
        raise ValueError(f"rank must be in [0, {WORLD})")
    if worker_blocks < ESSENTIAL_CTAS:
        raise ValueError(
            f"worker_blocks must be >= {ESSENTIAL_CTAS} so all progress roles are resident"
        )
    if worker_blocks == ESSENTIAL_CTAS:
        # Producers retire without rejoining in the chunked full path. With
        # zero GEMM consumers nobody would run GMM1.
        raise ValueError(
            "full Stage-1 requires at least one GEMM consumer after progress roles"
        )

    rank = int(rank)
    local_rank = rank % GPUS_PER_NODE
    node = rank // GPUS_PER_NODE
    NODES = WORLD // GPUS_PER_NODE
    REMOTE_NODES = tuple(peer for peer in range(NODES) if peer != node)
    stage2_window_offset = int(stage2_window_offset)
    # rail_soa layout shared by dispatch_staging[p] and remote_dispatch_rx[p].
    from .quant import rail_record_layout as _rail_record_layout

    REC_BYTES, REC_Q, REC_S, REC_I, REC_W = _rail_record_layout(HIDDEN, TOPK, A_DTYPE)
    # Each poster borrows the first fanout shard of a nonzero destination.
    # A fourth poster on a four-GPU node would steal dest0's first token
    # shard as well as T0's reserved shard, leaving those tokens unprocessed.
    RAIL_QPS = min(4, GPUS_PER_NODE - 1)
    # Assign fanout CTAs by (dest, shard). T0 handles CCO, credits, and sealing,
    # so it cannot also handle a dest0 shard; at least two shards are required.
    if fanout_ctas_per_dest < 2:
        raise ValueError(
            "fanout_ctas_per_dest must be >= 2 (T0 does not take a fanout shard)"
        )
    if stage2_window_offset < 0 or stage2_window_offset % 4096:
        raise ValueError(
            "stage2_window_offset must be non-negative and 4096-byte aligned"
        )

    wire = layout.wire
    record_bytes = wire.record_bytes
    # split_local selects between two GMM1 paths.
    # Split: count and group local-source fan1 rows separately from remote-source
    # fan2 rows. Sealer phase 1 seals local groups after eight local_eos values,
    # using h1 base 0; phase 2 seals remote groups after all comm_eos values.
    # Full groups enqueue during fanout, and each sealer phase publishes its tails.
    # h1_local_eos orders local sealing before final permutation and remote sealing.
    # fan2_shards: only fanout_shard < fan2_shards handles fan2, using that token
    # stride. Other fan CTAs enter local GMM1 after fan1, using rail wait time for compute.
    # lazy_pad: phase 1 publishes h1_local_eos without padding local tail rows;
    # phase 2 pads them. GMM1 reads A only through bounds-checked tile_row_input.
    # Before padding, these entries contain valid initial/previous-generation row
    # IDs. Stage2 discards their h1 values using INVALID_SOURCE and zero weights,
    # both installed before Stage1 completes.
    # extra_consumers: tickets 1..COMPUTE_FIRST-1, except the finisher, also enter
    # GMM1 after fanout. consumer_index remains a bijection over all consumers,
    # with extra consumers appended after the original range.
    # Without split: static_g0 fixes expert e's first group at physical group e,
    # tiles [e*G,e*G+G). Before launch_ready, each GPU initializer sets its map,
    # tile_expert, and map_ready, and starts tile_alloc at LE*G. Only overflow
    # groups are claimed remotely. Even a zero-row expert owns one fully padded
    # group with invalid sources. Seal tails in parallel over expert*SPL threads
    # using _seal_pad_par and _seal_perm(par=True).
    # Both paths write h1 by physical group and seal the inverse tile_src_of_dst
    # map for Stage2's indirect A reads. Rail posting CTAs (_POST_LO.._POST_HI)
    # post first and do not receive fanout tokens.
    data_send_ctas = int(data_send_ctas)
    if data_send_ctas not in (0, 32, 64):
        raise ValueError("data_send_ctas must be 0, 32 or 64")
    SEND_FIRST = GPUS_PER_NODE * fanout_ctas_per_dest
    SEND_END = SEND_FIRST + data_send_ctas
    if data_send_ctas and (WAVES != 8 or SEND_END > worker_blocks):
        raise ValueError("copy roles require eight waves and a resident grid")
    F2S = int(fan2_shards)
    if F2S and F2S > fanout_ctas_per_dest - 1:
        raise ValueError("fan2_shards must be <= fanout_ctas_per_dest - 1")
    _FIN_TICKET = local_rank
    _XC = (
        [t for t in range(1, COMPUTE_FIRST) if t != _FIN_TICKET]
        if extra_consumers
        else []
    )
    # Ticket q+1 posts QP q; T0 does not post.
    _POST_LO, _POST_HI = 1, RAIL_QPS + 1
    # Phase 1 runs on fan ticket 8*_SEG1_SHARD + local_rank. dest0 shard IDs shift
    # down by one, and posting CTAs have no tokens, so shift those ranks one more
    # slot to place them outside the fan2 pool.
    _SEG1_SHARD = 1
    if F2S:
        _s1 = F2S + (1 if local_rank == 0 or _POST_LO <= local_rank < _POST_HI else 0)
        if _s1 < fanout_ctas_per_dest:
            _SEG1_SHARD = _s1
    if REC_BYTES > record_bytes:
        raise ValueError("rail record does not fit one dispatch_staging record slot")
    if RAIL_QPS > int(layout.num_qp):
        raise ValueError("rail_qps must be <= num_qp")
    payload_dwords = wire.payload_bytes // 4
    scale_bytes = wire.scale_bytes
    scale_dwords = scale_bytes // 4
    dispatch_chunks = layout.dispatch_chunks
    max_route_tiles = layout.max_route_tiles
    max_tiles_per_expert = layout.max_tiles_per_expert
    GR_N = max_route_tiles // G + 1  # h1_grp_sync words per half (see layout)
    # GMM1 N width is arena.block_n (256; 128 was slower for every tested shape).
    # Count jobs using GNB.
    GBN = BN
    GNB = (2 * INTER) // GBN

    kh_tile = BK if A_DTYPE == "fp8" else BK // 2
    k_tiles_total = k_tiles_total_for(HIDDEN, BK)
    # Size LDS for GMM1's actual M tile, not the arena tile. With G>1, _gemm1_body
    # processes BM*G rows and needs BM*G*BN*4 accumulator bytes. Sizing for BM
    # silently writes out of bounds (measured relL2=1.55 at G=2).
    _, _, _, lds_bytes = _bm_constants(GBM, BN, kh_tile, k_tiles_total)
    assert lds_bytes % 16 == 0
    NC_MBOX_DW = lds_bytes // 4  # Ready-queue job broadcast after all GEMM1 LDS.
    lds_alloc_bytes = lds_bytes + 16
    # h1_output_q at mtpr > 4096 passes the i32 byte range: GMM1 then puts each
    # m-block's row offset into the 64-bit base (sizes <= 4096 keep the old ISA).
    H1_ROW_BASE = layout.region("h1_output_q").nbytes // layout.parity_depth >= 1 << 31
    INPUT_ROW_BASE = (
        layout.region("grouped_input_q").nbytes // layout.parity_depth >= 1 << 31
    )

    # Compile-time region helpers.  CCO offsets are relative to the physical
    # window; local addresses add arena_ptr at runtime.
    def off(name):
        return int(layout.region(name).offset)

    def plane(name):
        return _plane_bytes(layout, name)

    region_indices = {(stage, name): index for stage, name, index in buffer_regions}

    s2_parity_depth = int(stage2_layout.parity_depth)
    stage2_geometry = (
        int(stage2_layout.hidden),
        int(stage2_layout.topk),
        int(stage2_layout.max_tokens),
        int(stage2_layout.world_size),
        int(stage2_layout.gpus_per_node),
        int(stage2_layout.tile_n),
        int(stage2_layout.num_qp),
    )
    expected_stage2_geometry = (
        HIDDEN,
        TOPK,
        MAX_TOKENS,
        WORLD,
        GPUS_PER_NODE,
        BN,
        int(layout.num_qp),
    )
    if stage2_geometry != expected_stage2_geometry:
        raise ValueError(
            "Stage-1/Stage-2 arena geometry mismatch: "
            f"stage2={stage2_geometry}, expected={expected_stage2_geometry}"
        )
    if s2_parity_depth != 2:
        raise ValueError("Stage-2 metadata must be double buffered")

    def s2_off(name):
        return int(stage2_layout.region(name).offset)

    def s2_plane(name):
        region = stage2_layout.region(name)
        if not region.shape or region.shape[0] != s2_parity_depth:
            raise ValueError(f"Stage-2 region {name} is not parity indexed")
        return region.nbytes // s2_parity_depth

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_alloc_bytes, 16]

    # Names include only compile-time parameters that vary by shape or tune table.
    # Fixed switches are omitted. split_local represents its associated switches:
    # early local GMM, fan2, lazy padding, and extra consumers, or parallel sealing
    # and static group 0.
    kernel_name = (
        f"megamoe_stage1_{activation}"
        + (
            f"_b{str(situ_beta).replace('.', 'p')}_lb{str(situ_linear_beta).replace('.', 'p')}"
            if activation == "situv2" and (situ_beta, situ_linear_beta) != (1.0, 1.0)
            else ""
        )
        + ("_a8w4" if A_DTYPE == "fp8" else "")
        # Distinguish the restored M16 helper from prior M32 compiled handles.
        + ("_am16" if A_DTYPE == "fp8" and GBM == 128 else "")
        + (f"_ep{WORLD}_gpn{GPUS_PER_NODE}" if NODES != 2 else "")
        + (
            ""
            if swiglu_limit is None
            else f"_swl{float(swiglu_limit):g}".replace(".", "p")
        )
        + f"_r{rank}_h{HIDDEN}_i{INTER}_e{EXPERTS}_k{TOPK}"
        f"_mt{MAX_TOKENS}_rpc{layout.max_routes_per_token_per_rank}"
        + f"_cf{COMPUTE_FIRST}"
        + (f"_bm{BM}" if BM != 32 else "")
        + (f"_tg{G}" if G != 1 else "")
        + (f"_fos{fanout_ctas_per_dest}" if fanout_ctas_per_dest != 1 else "")
        + (f"_ds{data_send_ctas}_wtpub" if data_send_ctas else "")
        + ("_slg" if split_local else "_nsl")
        + (f"_w{WAVES}" if WAVES != 4 else "")
        + ("_gr" if group_ready else "")
        + ("_c1" if buffer_regions else "")
        + (
            f"_vl{gda_peer_stride}l{gda_peer_lane}b{lsa_peer_base}"
            if gda_peer_stride != 1
            else ""
        )
        + ("_rq_bounded_ac1_mgloop_c1" if ready_q else "")
        + ("_recv_status" if BOUNDED_RECV else "")
        # Compile rail CQ polling in CCQE mode.
        + ("_ccqe" if _rail_ccqe() else "")
    )

    # The body is a device function so the single fused kernel can inline it
    # ahead of stage2 with one shared LDS allocation (``lds_raw``).
    @flyc.jit
    def s1_body(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        x_q: fx.Int64,
        input_scale: fx.Int64,
        route_weights: fx.Int64,
        topk_ids: fx.Int64,
        w1q: fx.Int64,
        w1scale: fx.Int64,
        ntokens: fx.Int32,
        lds_raw=None,
    ):
        if const_expr(lds_raw is None):
            lds_raw = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        # cco_lsa_ptr(w, peer, off) = winBase + peer*(stride4G<<32) + off.
        # Each call reloads two window fields (~6.5% of routing time in ATT).
        # Cache the base and peer stride at entry; later peer addresses use integer arithmetic.
        if const_expr(not buffer_regions):
            arena_window = cco.Window(arena_win)
            arena_lsa_base = fx.Int64(arena_window.lsa_ptr(fx.Int32(0), fx.Int64(0)))
            arena_lsa_stride = (
                fx.Int64(arena_window.lsa_ptr(fx.Int32(1), fx.Int64(0)))
                - arena_lsa_base
            )

        def persistent_addr(name):
            address = arena_ptr + fx.Int64(off(name))
            if const_expr(bool(buffer_regions)):
                address = region_local_address(
                    arena_ptr, region_indices[1, name], fx.Int64(0)
                )
            return address

        tx = fx.Int32(gpu.thread_id("x"))
        lane = tx & fx.Int32(63)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))

        # An entry ticket advances on the GPU on every replay. All CTAs in an
        # ordered launch receive tickets from one worker_blocks-sized epoch.
        # Reusing a captured Python generation would accept stale ready flags.
        ticket_ptr = fx.recast_iter(fx.Int64, lds_raw)
        ticket_view = fx.make_view(ticket_ptr, fx.make_layout(1, 1))
        if tx == fx.Int32(0):
            ticket_lane = fx.Int64(
                comm.atomic_add_agent(persistent_addr("entry_count"), fx.Int64(1))
            )
            fx.ptr_store(Vec.from_elements([ticket_lane], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        # Derive generation on device from entry tickets because CUDA graph replay
        # freezes host arguments. Each CTA increments entry_count once; each forward
        # launches this kernel once with worker_blocks CTAs.
        generation = ticket64 // fx.Int64(worker_blocks) + fx.Int64(1)
        parity = fx.Int64(generation & fx.Int64(1))
        control_generation = generation

        def local_addr(name):
            address = arena_ptr + fx.Int64(off(name)) + parity * fx.Int64(plane(name))
            if const_expr(bool(buffer_regions)):
                address = region_local_address(
                    arena_ptr, region_indices[1, name], parity
                )
            return address

        def window_off(name):
            offset = fx.Int64(off(name)) + parity * fx.Int64(plane(name))
            if const_expr(bool(buffer_regions)):
                offset = region_window_address(
                    arena_ptr, region_indices[1, name], parity
                )[1]
            return offset

        def window_handle(name):
            handle = arena_win
            if const_expr(bool(buffer_regions)):
                handle = region_window_address(
                    arena_ptr, region_indices[1, name], parity
                )[0]
            return handle

        def stage2_addr(name):
            address = (
                arena_ptr
                + fx.Int64(stage2_window_offset + s2_off(name))
                + parity * fx.Int64(s2_plane(name))
            )
            if const_expr(bool(buffer_regions)):
                address = region_local_address(
                    arena_ptr, region_indices[2, name], parity
                )
            return address

        error_addr = persistent_addr("error_count")

        # Arrival tickets, rather than block IDs, ensure every progress role is
        # among the first resident CTAs on an oversubscribed persistent grid.
        ticket = fx.Int32(ticket64 % fx.Int64(worker_blocks))
        is_initializer = ticket == fx.Int32(0)

        is_cco = ticket == fx.Int32(CCO_TICKET)
        is_comm = ticket < fx.Int32(COMM_CTAS)

        # spin_relaxed uses relaxed polling; the caller performs one acquire afterward.
        _pl32 = comm.load_i32_global_system_relaxed
        _spin = comm.spin_until_ge_i64_system_rx

        def _cta_acquire():
            """Acquire for the whole CTA after polling: barrier, wave0 acquire, barrier.

            vL1D is shared per CU and L2 per XCD, so one buffer_inv covers the entire CTA
            instead of one per wave. The second barrier orders other waves' reads after
            invalidation. The first is needed where waves finish polling independently.
            """
            gpu.barrier()
            if wave == fx.Int32(0):
                comm.fence_system_acquire()
            gpu.barrier()

        def peer_local_addr(name, peer):
            # RX/credit/request arrays omit this node; their full allocation
            # stride includes all peers and both parities.
            peer_bytes = plane(name) // (NODES - 1)
            index = peer - int(peer > node)
            return local_addr(name) + fx.Int64(index * peer_bytes)

        def peer_window_off(name, peer):
            # A remote write indexes the destination's array by our node,
            # excluding that destination. The source's local index differs.
            peer_bytes = plane(name) // (NODES - 1)
            index = node - int(node > peer)
            return window_off(name) + fx.Int64(index * peer_bytes)

        def _rail_post(post_qp, with_credit):
            """Post QP post_qp's token records from one wave using rail_soa.

            Issue one PUT for the segment, this QP's ready words, and one doorbell without
            waiting. Save the request in remote_chunk_request[post_qp] for later reaping.
            with_credit waits for this QP's generation g-2 credit on the same parity;
            T0 has already waited for all credits outside this helper.
            """
            for remote_index in range_constexpr(NODES - 1):
                remote_node = REMOTE_NODES[remote_index]
                src_base = fx.Int64(off("dispatch_staging"))
                if const_expr(bool(buffer_regions)):
                    src_base = region_window_address(
                        arena_ptr, region_indices[1, "dispatch_staging"], fx.Int64(0)
                    )[1]
                comm.fence_system_release()
                if const_expr(with_credit):  # noqa: SIM102
                    if lane == fx.Int32(0):
                        for cchunk in range(
                            fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                        ):
                            _spin(
                                peer_local_addr("remote_chunk_credit", remote_node)
                                + fx.Int64(cchunk * fx.Int32(layout.num_qp) + post_qp)
                                * fx.Int64(8),
                                generation - fx.Int64(2),
                            )
                rows = fx.Int32((MAX_TOKENS + RAIL_QPS - 1) // RAIL_QPS)
                r0 = post_qp * rows
                left = ntokens - r0
                left = (left > fx.Int32(0)).select(left, fx.Int32(0))
                nr = (left < rows).select(left, rows)
                if nr > fx.Int32(0):
                    _rail.put(
                        dev_comm,
                        post_qp,
                        fx.Int32(remote_node),
                        window_handle("remote_dispatch_rx"),
                        peer_window_off("remote_dispatch_rx", remote_node)
                        + fx.Int64(r0 * fx.Int32(REC_BYTES)),
                        window_handle("dispatch_staging"),
                        src_base + fx.Int64(r0 * fx.Int32(REC_BYTES)),
                        fx.Int64(nr * fx.Int32(REC_BYTES)),
                        aggregate=True,
                    )
                # Consumers wait only for chunk 0's num_qp ready words. Each QP publishes at
                # least one ready word after its ordered data writes, so all data has arrived.
                for w in range_constexpr(layout.num_qp):
                    if post_qp == fx.Int32(w % RAIL_QPS):
                        _rail.put_value(
                            dev_comm,
                            post_qp,
                            fx.Int32(remote_node),
                            window_handle("remote_chunk_ready"),
                            peer_window_off("remote_chunk_ready", remote_node)
                            + fx.Int64(w * 8),
                            generation,
                            aggregate=True,
                        )
                request = _rail.flush_async(dev_comm, post_qp, fx.Int32(remote_node))
                if lane == fx.Int32(0):
                    comm.store_i64_global_system(
                        peer_local_addr("remote_chunk_request", remote_node)
                        + fx.Int64(post_qp) * fx.Int64(8),
                        request,
                    )

        # Role arrival counters remain host-readable through the CCO window's local_ptr
        # even if the kernel hangs. Write them directly into the arena so the host can
        # poll after launch without synchronizing and identify roles with no arriving CTA.

        # Owner initializes only counters/metadata.  Payload and generation
        # arrays are overwrite-before-publish and never need a hot-path memset.
        if is_initializer:
            expert_count = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            for item in range(tx, LOCAL_EXPERTS * (2 if split_local else 1), THREADS):
                buffer_ops.buffer_store(fx.Int32(0), expert_count, item)
            consumed = buffer_ops.create_buffer_resource_from_addr(
                local_addr("remote_chunk_consumed")
            )
            shard_done = buffer_ops.create_buffer_resource_from_addr(
                local_addr("fanout_shard_done")
            )
            if tx < fx.Int32(GPUS_PER_NODE * (2 if split_local else 1)):
                buffer_ops.buffer_store(fx.Int32(0), shard_done, tx)
            for item in range(
                tx,
                fx.Int32((NODES - 1) * dispatch_chunks * layout.num_qp),
                fx.Int32(THREADS),
            ):
                buffer_ops.buffer_store(fx.Int32(0), consumed, item)
            if tx == fx.Int32(0):
                for name in ("tile_alloc",):
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(local_addr(name)),
                        fx.Int32(0),
                    )
                if const_expr(BOUNDED_RECV):
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("recv_overflow")
                        ),
                        fx.Int32(0),
                    )
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(
                            persistent_addr("forward_status")
                        ),
                        fx.Int32(0),
                    )
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("recv_route_count")
                        ),
                        fx.Int32(0),
                    )
                if const_expr(COMPACT_INPUT):
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("grouped_input_alloc")
                        ),
                        fx.Int32(0),
                    )
                if const_expr(early_local_gmm):
                    _jh_res = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("gmm1_job_head")
                    )
                    buffer_ops.buffer_store(fx.Int32(0), _jh_res, fx.Int32(0))
                    buffer_ops.buffer_store(fx.Int32(0), _jh_res, fx.Int32(16))
                    if const_expr(ready_q):
                        # word 8: ready queue write head; word 9: consumer job claim head.
                        buffer_ops.buffer_store(fx.Int32(0), _jh_res, fx.Int32(8))
                        buffer_ops.buffer_store(fx.Int32(0), _jh_res, fx.Int32(9))
                if ntokens != fx.Int32(MAX_TOKENS):
                    comm.atomic_add_system(error_addr, fx.Int32(1))
            if const_expr(static_g0):
                # Static group 0: tx0 just cleared tile_alloc in this thread; now set it to LE*G.
                if tx == fx.Int32(0):
                    buffer_ops.buffer_store(
                        fx.Int32(LOCAL_EXPERTS * G),
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("tile_alloc")
                        ),
                        fx.Int32(0),
                    )
                sg_map = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("expert_tile_map")
                )
                sg_te = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_expert")
                )
                for sg_i in range(tx, fx.Int32(LOCAL_EXPERTS * G), fx.Int32(THREADS)):
                    sg_e = sg_i // fx.Int32(G)
                    sg_m = (
                        sg_e * fx.Int32(max_tiles_per_expert)
                        + sg_i
                        - sg_e * fx.Int32(G)
                    )
                    buffer_ops.buffer_store(sg_i, sg_map, sg_m)
                    buffer_ops.buffer_store(sg_e, sg_te, sg_i)
                    comm.store_i64_global_relaxed(
                        local_addr("expert_tile_map_ready")
                        + fx.Int64(sg_m) * fx.Int64(8),
                        fx.Int64(generation),
                    )
                if const_expr(tile_ready):
                    # Clear group arrival and reservation counts before publishing launch_ready.
                    # Words 0..7 count this rank's completed fan CTAs by destination; word 16 counts
                    # sources. Also initialize the static groups' expert IDs.
                    tr_done = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_done")
                    )
                    for tr_i in range(tx, fx.Int32(max_route_tiles), fx.Int32(THREADS)):
                        buffer_ops.buffer_store(fx.Int32(0), tr_done, tr_i)
                    if tx < fx.Int32(32):
                        buffer_ops.buffer_store(
                            fx.Int32(0),
                            buffer_ops.create_buffer_resource_from_addr(
                                local_addr("gmm1_job_head")
                            ),
                            tx,
                        )
                    if const_expr(G > 1):
                        for tr_e in range(
                            tx, fx.Int32(LOCAL_EXPERTS), fx.Int32(THREADS)
                        ):
                            buffer_ops.buffer_store(
                                tr_e,
                                buffer_ops.create_buffer_resource_from_addr(
                                    local_addr("tile_expert_group")
                                ),
                                tr_e,
                            )
            if const_expr(ready_q and not tile_ready):
                # Clear group arrivals before launch_ready; dispatch waits for it before atomics.
                rq_done = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_row_done")
                )
                for rq_i in range(tx, fx.Int32(max_route_tiles), fx.Int32(THREADS)):
                    buffer_ops.buffer_store(fx.Int32(0), rq_done, rq_i)
            if const_expr(data_send_ctas):
                js = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("send_job_state")
                )
                if tx < fx.Int32(32):
                    buffer_ops.buffer_store(fx.Int32(0), js, tx)
                if tx < fx.Int32(2 * GPUS_PER_NODE):
                    jd = tx % fx.Int32(GPUS_PER_NODE)
                    jn = (jd == fx.Int32(0)).select(
                        fx.Int32(fanout_ctas_per_dest - 1),
                        fx.Int32(fanout_ctas_per_dest),
                    )
                    buffer_ops.buffer_store(
                        jn,
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("send_join")
                        ),
                        tx,
                    )
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm.fence_system_release()
                # pub_relaxed: one fence publishes all clears, avoiding an L2 writeback per gate store.
                _pub_st = comm.store_i64_global_system_relaxed
                _pub_st(
                    persistent_addr("epoch_gate"),
                    control_generation,
                )
                _pub_st(local_addr("launch_ready"), control_generation)
        else:
            if tx == fx.Int32(0):
                _spin(
                    persistent_addr("epoch_gate"),
                    control_generation,
                )
            _cta_acquire()
        # Destination placement, step 1: compute and exchange counts.
        #
        # Counts depend only on topk_ids, available at entry. The plan can run alongside
        # quantization (~93us measured), leaving arithmetic offsets for payload transfer
        # instead of per-route cross-device expert_count atomics and tile claims/spins.
        #
        # Inter-node traffic uses the rail. Send only counts to the same-index peer,
        # which fans them out to eight destinations on its node. Destinations keep
        # row bases in their own arenas, readable via LSA by local producers and rail
        # forwarders. Remote sources need no row bases, eliminating the return rail hop.
        #
        # Counts need no clearing: each source owns its row and publishes generation
        # in plan_count_ready[src]. Destinations wait for equality with generation,
        # so stale previous-generation counts cannot satisfy the wait.
        # A bounded producer CTA pool owns tokens in a strided schedule. At capacity
        # 128 this matches the historical one-CTA-per-token mapping; larger capacities
        # reuse resident CTAs for later tokens rather than growing the grid.
        # The original split fanout used ticket as the token every iteration, writing
        # only 128 of 512 tokens and skipping tokens 0..7. That mapping assumed MORI's
        # 128-token, one-CTA-per-token, single-iteration shape.
        # Four waves own four QPs. Each chunk is one aggregate PUT per QP plus a
        # trailing ready value and one flush/doorbell. The same CTA receives the
        # reciprocal chunk and performs selected-rank proxy fanout.
        if is_cco:
            # One wave per QP; with more waves than QPs the extra waves repeat
            # a QP's waits (reads only).
            qp = wave % fx.Int32(layout.num_qp) if WAVES > layout.num_qp else wave

            for remote_index in range_constexpr(NODES - 1):
                remote_node = REMOTE_NODES[remote_index]
                # credit_async waits here for the peer's generation g-2 consumption credit,
                # just before overwriting the same remote_dispatch_rx parity plane.
                # Do not wait at kernel exit, which would require all peer fan2 work to finish.
                if lane == fx.Int32(0):
                    for cchunk in range(
                        fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                    ):
                        _spin(
                            peer_local_addr("remote_chunk_credit", remote_node)
                            + fx.Int64(cchunk * fx.Int32(layout.num_qp) + qp)
                            * fx.Int64(8),
                            generation - fx.Int64(2),
                        )
                # rail_soa reads contiguous [q|scale|ids|weights|pad] records, REC bytes/token,
                # written by rail_record_quant into dispatch_staging's registered parity-0 plane.
                # Each QP sends one contiguous token segment: one PUT, one ready word, and one
                # doorbell. Reap requests at the final credit phase. WQE posting costs ~15us
                # independent of size (TS3/TS5), so avoid separate PUTs by field.
                # The source is single-buffered: complete this PUT at kernel exit before the
                # next quantization. Tickets 1..RAIL_QPS post QPs at fanout entry; T0 does not.
                comm.fence_system_release()

                # After posting all transfers, wait for the batch with one acquire fence.
                # rail_soa publishes only chunk 0 ready words.
                if lane == fx.Int32(0):
                    _spin(
                        peer_local_addr("remote_chunk_ready", remote_node)
                        + fx.Int64(qp) * fx.Int64(8),
                        generation,
                    )
                _cta_acquire()
                gpu.barrier()
                rx_all = buffer_ops.create_buffer_resource_from_addr(
                    peer_local_addr("remote_dispatch_rx", remote_node)
                )
                # agent: the 256-thread CCO CTA parses all remote tokens with a strided loop.
                # Capacities above THREADS must process more than the first 256 records.
                for token in range(tx, ntokens, fx.Int32(THREADS)):
                    remote_record_available = fx.Int32(1)
                    remote_slot_mask = fx.Int32(0)
                    if remote_record_available != fx.Int32(0):
                        # Token IDs vary by lane, but descriptors must be scalar. Per-token descriptors
                        # generate readfirstlane waterfall loops (ATT5: up to 64 iterations per load,
                        # ~230k T0 cycles here). Use one RX descriptor with lane-specific offsets.
                        record_dword = token * fx.Int32(REC_BYTES // 4) + fx.Int32(
                            REC_I // 4
                        )
                        for slot in range_constexpr(TOPK):
                            expert = buffer_ops.buffer_load(
                                rx_all,
                                record_dword + fx.Int32(slot),
                                vec_width=1,
                                dtype=T.i32,
                            )
                            valid = (expert >= fx.Int32(0)) & (
                                expert < fx.Int32(EXPERTS)
                            )
                            owner = valid.select(
                                expert // fx.Int32(LOCAL_EXPERTS),
                                fx.Int32(0),
                            )
                            on_node = valid & (
                                (owner // fx.Int32(GPUS_PER_NODE)) == fx.Int32(node)
                            )
                            remote_slot_mask = on_node.select(
                                remote_slot_mask | (fx.Int32(1) << fx.Int32(slot)),
                                remote_slot_mask,
                            )
                    buffer_ops.buffer_store(
                        remote_slot_mask,
                        buffer_ops.create_buffer_resource_from_addr(
                            stage2_addr("node_dest_slot_mask")
                        ),
                        fx.Int32(remote_node * MAX_TOKENS) + token,
                    )
                if const_expr(group_ready):
                    # EOS on this rank's expert tiles does not join T0's mask
                    # parser. Publish all lanes' mask writes before the sealer
                    # hands the tables to concurrently executing Stage2 CTAs.
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        comm.store_i64_global_system(
                            local_addr("node_mask_ready")
                            + fx.Int64((remote_node - int(remote_node > node) + 1) * 8),
                            generation,
                        )
        # Split fanout CTAs cover the eight node-local destinations. Each writes
        # both its local source rank and the aligned remote source rank directly
        # into destination expert tiles; there is no rank inbox/sort.
        is_finisher = is_comm & (ticket == fx.Int32(local_rank))

        is_sealer = is_finisher

        def _peer_addr(dest, name):
            if const_expr(bool(buffer_regions)):
                if const_expr(lsa_peer_base != 0):
                    dest = dest + fx.Int32(lsa_peer_base)
                address = region_lsa_address(
                    arena_ptr, region_indices[1, name], parity, dest
                )
            else:
                address = (
                    arena_lsa_base
                    + fx.Int64(dest) * arena_lsa_stride
                    + fx.Int64(window_off(name))
                )
            return address

        def _tr_resv_signal(dest, cnt):
            """Publish tile_ready after P2 completes for this rank's last fan CTA to dest."""
            tr_prev = fx.Int32(
                comm.atomic_add_agent(
                    local_addr("gmm1_job_head") + fx.Int64(dest) * fx.Int64(4),
                    fx.Int32(1),
                )
            )
            if tr_prev + fx.Int32(1) == cnt:
                comm.atomic_add_agent(
                    local_addr("gmm1_job_head") + fx.Int64(dest) * fx.Int64(4),
                    fx.Int32(0) - cnt,
                )
                comm.atomic_add_system(
                    _peer_addr(dest, "gmm1_job_head") + fx.Int64(16 * 4), fx.Int32(1)
                )

        def _rq_push(dest, pg):
            """Append physical group pg to dest's ready queue after its data is visible.
            dest=None selects this GPU.
            """
            rq_h = (
                local_addr("gmm1_job_head")
                if dest is None
                else _peer_addr(dest, "gmm1_job_head")
            )
            rq_b = (
                local_addr("gmm1_ready_q")
                if dest is None
                else _peer_addr(dest, "gmm1_ready_q")
            )
            rq_q = fx.Int32(comm.atomic_add_system(rq_h + fx.Int64(8 * 4), fx.Int32(1)))
            comm.store_i64_global_system_relaxed(
                rq_b + fx.Int64(rq_q) * fx.Int64(8),
                (fx.Int64(generation) << fx.Int64(32)) | fx.Int64(pg),
            )

        def _rq_entry(addr, valid):
            # Terminal claims may exceed physical queue capacity, especially
            # with few N blocks. Clamp the address and mask the generation:
            # adjacent arena data must never be accepted as a ready group.
            value = fx.Int64(comm.load_i64_global_system_relaxed(addr))
            return valid.select(value, fx.Int64(0))

        def _rq_seal_tails(EO):
            """Enqueue each expert's partial tail group for EO=0 (local) or EO=LE (remote).
            Dispatch already enqueued full groups. The caller has acquired all arrivals
            for this segment, so counts are final.
            """
            if tx < fx.Int32(LOCAL_EXPERTS):
                rq_c = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("expert_count")
                    ),
                    tx + fx.Int32(EO),
                    vec_width=1,
                    dtype=T.i32,
                )
                if rq_c % fx.Int32(GROUP_ROWS) != fx.Int32(0):
                    rq_mi = (tx + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert) + (
                        rq_c // fx.Int32(GROUP_ROWS)
                    ) * fx.Int32(G)
                    _spin(
                        local_addr("expert_tile_map_ready")
                        + fx.Int64(rq_mi) * fx.Int64(8),
                        generation,
                    )
                    comm.fence_system_acquire()
                    rq_pg = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("expert_tile_map")
                        ),
                        rq_mi,
                        vec_width=1,
                        dtype=T.i32,
                    ) // fx.Int32(G)
                    _rq_push(None, rq_pg)

        def _alloc_tiles_for(count):
            """Return an expert's allocated tile count, rounded up to a multiple of G."""
            n = (count + fx.Int32(BM - 1)) // fx.Int32(BM)
            if const_expr(G > 1):
                n = ((n + fx.Int32(G - 1)) // fx.Int32(G)) * fx.Int32(G)
            if const_expr(static_g0):
                n = (n < fx.Int32(G)).select(fx.Int32(G), n)
            return n

        def _claim_tile_group(dest, map_index, local_expert, map_ready):
            """Claim G adjacent physical tiles, publish G map slots, and return the base.

            tile_alloc always increments by G, so bases are G-aligned. GMM1 can therefore
            treat [base,base+G) as one BM*G-row m_block. Later logical tiles in the group
            wait for map_ready in the else branch, as non-first rows did previously.
            """
            base = fx.Int32(
                comm.atomic_add_system(_peer_addr(dest, "tile_alloc"), fx.Int32(G))
            )
            if base >= fx.Int32(max_route_tiles - G + 1):
                comm.atomic_add_system(error_addr, fx.Int32(1))
                # llvm.intr_trap's noreturn CFG changes normal-path VGPR/scratch
                # allocation on gfx950. The shared s_trap 2 helper preserves
                # those resources while aborting before any map or payload store.
                _device_trap()
            map_res = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "expert_tile_map")
            )
            expert_res = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "tile_expert")
            )
            for j in range_constexpr(G):
                phys = base + fx.Int32(j)
                buffer_ops.buffer_store(
                    phys, map_res, map_index + fx.Int32(j), cache_modifier=_WT_META
                )
                buffer_ops.buffer_store(
                    local_expert, expert_res, phys, cache_modifier=_WT_META
                )
            if const_expr((tile_ready or ready_q) and G > 1):
                # local_expert is relative to the local/remote segment; GMM1 uses it for weights.
                buffer_ops.buffer_store(
                    local_expert,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_expert_group")
                    ),
                    base // fx.Int32(G),
                    cache_modifier=_WT_META,
                )
            if const_expr(fan_wt):
                # These stores are write-through; completion makes them peer-visible without an L2 writeback.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            for j in range_constexpr(G):
                # claim_relaxed: one fence already publishes map/tile_expert; G release stores
                # would each trigger a full L2 writeback.
                (comm.store_i64_global_system_relaxed)(
                    map_ready + fx.Int64(j * 8), generation
                )
            return base

        # route_gbatch follows MegaMoEv2 emit_dispatch_group. Each CTA batches
        # _GB_NB=THREADS/TOPK tokens, one (token, slot) per thread. LDS atomics assign
        # batch-local positions; one system atomic per (dest, expert) reserves a
        # contiguous row segment. Its owner claims any group head in that segment.
        # In ATT12, per-record routing spent 72% of cycles at s_barrier waiting for a
        # serial chain: 16 mask loads, the A1 atomic, and A2 map_ready.
        # There are four barriers per batch; no final barrier is needed. Next-batch
        # P0/P1 writes only CNT/HIT, whose P2/P3 readers precede this batch's P3 barrier.
        # P4 reads ROW, which next-batch P3 overwrites only after three barriers.
        # Limit batch tokens by THREADS//TOPK and GROUP_ROWS, so an expert crosses at
        # most one group head. Round to a multiple of WAVES for even P3 record assignment.
        # When TOPK does not divide THREADS, the last THREADS-_GB_LANES threads skip P1.
        _GB_NB = min(THREADS // TOPK, GROUP_ROWS) // WAVES * WAVES
        _GB_LANES = _GB_NB * TOPK
        # Dword distance from IDs to weights: (REC_W-REC_I) after record alignment,
        # or TOPK in the contiguous ids_first [ids|weights] array.
        _GB_WDW_REC = (REC_W - REC_I) // 4
        # Arrival tickets have already been read into registers. Reuse one
        # dword of the reserved 16-dword prefix for the batch acceptance vote.
        _GB_ACCEPT = 4
        _GB_CNT = 16
        _GB_BASE = _GB_CNT + LOCAL_EXPERTS
        _GB_HIT = _GB_BASE + LOCAL_EXPERTS
        _GB_PH = _GB_HIT + _GB_NB
        # meta_opt caches (group ID, physical base) for groups this CTA claimed in P2.
        # Parsing can then avoid remote map lookups.
        _GB_OWN = _GB_PH + 2 * LOCAL_EXPERTS
        _GB_OWNB = _GB_OWN + LOCAL_EXPERTS
        _GB_UNITS = payload_dwords // 4
        _GB_IT = (_GB_UNITS + 63) // 64
        assert _GB_NB >= WAVES and _GB_NB % WAVES == 0 and _GB_LANES <= THREADS
        assert (
            LOCAL_EXPERTS + _GB_NB <= THREADS
        ), "P0 clears HIT with threads [LE, LE+_GB_NB)"
        assert _GB_NB <= GROUP_ROWS, "token batch fits group-sized payload staging"
        assert payload_dwords % 4 == 0
        assert (
            layout.region("grouped_input_scale").nbytes // layout.parity_depth
            == int(layout.input_capacity) * scale_bytes
        ), "input scales must fit the grouped_input_scale region"

        def _gb_ptr(idx):
            return fx.add_offset(fx.recast_iter(fx.Int32, lds_raw), idx * fx.Int32(4))

        def _gb_ld(idx):
            return Vec(fx.make_view(_gb_ptr(idx), fx.make_layout(1, 1)).load())[0]

        def _gb_st(idx, value):
            fx.ptr_store(Vec.from_elements([value], fx.Int32), _gb_ptr(idx))

        def _route_gbatch(
            rs,
            tok0,
            tstride,
            dest,
            src_base,
            emit_masks=False,
            remote=False,
            resv_last=None,
            resv_n=None,
        ):
            # rs describes REC_BYTES-byte AoS records; batch token IDs are tok0+r*tstride.
            # In split_local, remote=True uses expert_count/map's [LE,2LE) half, counting
            # and grouping separately from local rows while sharing the same tile_alloc.
            _EO = LOCAL_EXPERTS if (split_local and remote) else 0
            # P0: clear batch counts and token hit flags.
            if tx < fx.Int32(LOCAL_EXPERTS):
                _gb_st(fx.Int32(_GB_CNT) + tx, fx.Int32(0))
            if (tx >= fx.Int32(LOCAL_EXPERTS)) & (
                tx < fx.Int32(LOCAL_EXPERTS + _GB_NB)
            ):
                _gb_st(fx.Int32(_GB_HIT - LOCAL_EXPERTS) + tx, fx.Int32(0))
            gpu.barrier()
            # P1: one (token, slot) per thread; assign a batch-local position for hits on dest.
            r = tx // fx.Int32(TOPK)
            slot = tx % fx.Int32(TOPK)
            tok = tok0 + r * tstride
            live = tok < ntokens
            if const_expr(_GB_LANES < THREADS):
                # Extra threads with r == _GB_NB point into the next batch and must be invalidated.
                live = live & (tx < fx.Int32(_GB_LANES))
            rec_dw = live.select(tok, fx.Int32(0)) * fx.Int32(REC_BYTES // 4)
            # ID dword base: REC_I in records, or token*[ids|weights] in ids_first storage.
            # Weights follow by _gb_wdw dwords; record IDs are padded to 16B.
            id0 = rec_dw + fx.Int32(REC_I // 4)
            _gb_wdw = _GB_WDW_REC
            expert = buffer_ops.buffer_load(rs, id0 + slot, vec_width=1, dtype=T.i32)
            lo = (fx.Int32(node * GPUS_PER_NODE) + dest) * fx.Int32(LOCAL_EXPERTS)
            hit = live & (expert >= lo) & (expert < lo + fx.Int32(LOCAL_EXPERTS))
            le = hit.select(expert - lo, fx.Int32(0))
            intra = fx.Int32(0)
            if hit:
                intra = fx.Int32(
                    _atomic_add_wg(
                        fx.Int64(fx.ptrtoint(_gb_ptr(fx.Int32(_GB_CNT) + le))),
                        fx.Int32(1),
                    )
                )
                _gb_st(fx.Int32(_GB_HIT) + r, fx.Int32(1))
            if const_expr(emit_masks):  # noqa: SIM102
                # Retain the original producer loop's Stage2 local-plane mask, once per token
                # with identical semantics. dest0 shards collectively cover all tokens.
                # The dest0 shard join publishes these writes for the single
                # kernel; the two-kernel path also has a stream boundary.
                if (dest == fx.Int32(0)) & (slot == fx.Int32(0)) & live:
                    m_lslot = fx.Int32(0)
                    m_err = fx.Int32(0)
                    for s2 in range_constexpr(TOPK):
                        ex = buffer_ops.buffer_load(
                            rs, id0 + fx.Int32(s2), vec_width=1, dtype=T.i32
                        )
                        ok = (ex >= fx.Int32(0)) & (ex < fx.Int32(EXPERTS))
                        owner = ok.select(ex // fx.Int32(LOCAL_EXPERTS), fx.Int32(0))
                        loc = ok & ((owner // fx.Int32(GPUS_PER_NODE)) == node)
                        m_lslot = loc.select(m_lslot | fx.Int32(1 << s2), m_lslot)
                        m_err = ok.select(m_err, m_err + fx.Int32(1))
                    if m_err != fx.Int32(0):
                        comm.atomic_add_system(error_addr, m_err)
                    buffer_ops.buffer_store(
                        m_lslot,
                        buffer_ops.create_buffer_resource_from_addr(
                            stage2_addr("node_dest_slot_mask")
                        ),
                        fx.Int32(node * MAX_TOKENS) + tok,
                    )
            gpu.barrier()
            if const_expr(BOUNDED_RECV):
                # Reserve all routes in this batch before any tile or payload
                # write. The atomic serializes concurrent source CTAs at the
                # destination, so accepted batches can never exceed the arena.
                if tx == fx.Int32(0):
                    batch_routes = fx.Int32(0)
                    for ex in range_constexpr(LOCAL_EXPERTS):
                        batch_routes += _gb_ld(fx.Int32(_GB_CNT + ex))
                    prev_routes = fx.Int32(
                        comm.atomic_add_system(
                            _peer_addr(dest, "recv_route_count"), batch_routes
                        )
                    )
                    accepted = prev_routes + batch_routes <= fx.Int32(
                        layout.route_capacity
                    )
                    _gb_st(
                        fx.Int32(_GB_ACCEPT),
                        accepted.select(fx.Int32(1), fx.Int32(0)),
                    )
                    if prev_routes + batch_routes > fx.Int32(layout.route_capacity):
                        comm.atomic_add_system(error_addr, fx.Int32(1))
                        # Keep the protocol alive: rejected batches allocate no
                        # rows or payload slots. Stage2 propagates the error and
                        # the stream reports failure after every peer has drained.
                        comm.atomic_add_system(
                            _peer_addr(dest, "recv_overflow"), fx.Int32(1)
                        )
                gpu.barrier()
                accepted = _gb_ld(fx.Int32(_GB_ACCEPT)) != fx.Int32(0)
                hit = hit & accepted
                # Each count is consumed by the same thread in P2. Each HIT
                # word is consumed by the same thread for compact allocation;
                # P2's existing barrier publishes both clears before P3.
                if tx < fx.Int32(LOCAL_EXPERTS):
                    count = _gb_ld(fx.Int32(_GB_CNT) + tx)
                    _gb_st(
                        fx.Int32(_GB_CNT) + tx,
                        accepted.select(count, fx.Int32(0)),
                    )
                if tx < fx.Int32(_GB_NB):
                    token_hit = _gb_ld(fx.Int32(_GB_HIT) + tx)
                    _gb_st(
                        fx.Int32(_GB_HIT) + tx,
                        accepted.select(token_hit, fx.Int32(0)),
                    )
            if const_expr(COMPACT_INPUT):
                # A token may match several experts on this destination rank,
                # yet gets exactly one payload slot. Reuse the batch hit word
                # to carry slot+1 into metadata and optional copy CTAs.
                # Keep the LDS load inside its thread bound; a DSL conjunction
                # may evaluate both operands before applying the predicate.
                if tx < fx.Int32(_GB_NB):  # noqa: SIM102
                    if _gb_ld(fx.Int32(_GB_HIT) + tx) != fx.Int32(0):
                        input_slot = fx.Int32(
                            comm.atomic_add_system(
                                _peer_addr(dest, "grouped_input_alloc"), fx.Int32(1)
                            )
                        )
                        if input_slot >= fx.Int32(layout.input_capacity):
                            comm.atomic_add_system(error_addr, fx.Int32(1))
                            _llvm_d.intr_trap()
                        _gb_st(fx.Int32(_GB_HIT) + tx, input_slot + fx.Int32(1))
                gpu.barrier()
            # P2: reserve one segment per expert with a system atomic. This thread claims
            # any group heads in its segment without waiting.
            if tx < fx.Int32(LOCAL_EXPERTS):
                n = _gb_ld(fx.Int32(_GB_CNT) + tx)
                if n > fx.Int32(0):
                    seg = fx.Int32(
                        comm.atomic_add_system(
                            _peer_addr(dest, "expert_count")
                            + fx.Int64(tx + fx.Int32(_EO)) * fx.Int64(4),
                            n,
                        )
                    )
                    _gb_st(fx.Int32(_GB_BASE) + tx, seg)
                    _gb_st(fx.Int32(_GB_OWN) + tx, fx.Int32(-1))
                    head = (
                        (seg + fx.Int32(GROUP_ROWS - 1)) // fx.Int32(GROUP_ROWS)
                    ) * fx.Int32(GROUP_ROWS)
                    if const_expr(static_g0):
                        head = (head == fx.Int32(0)).select(fx.Int32(GROUP_ROWS), head)
                    while head < seg + n:
                        m0 = (tx + fx.Int32(_EO)) * fx.Int32(
                            max_tiles_per_expert
                        ) + head // fx.Int32(BM)
                        own_base = _claim_tile_group(
                            dest,
                            m0,
                            tx,
                            _peer_addr(dest, "expert_tile_map_ready")
                            + fx.Int64(m0) * 8,
                        )
                        _gb_st(fx.Int32(_GB_OWN) + tx, head // fx.Int32(GROUP_ROWS))
                        _gb_st(fx.Int32(_GB_OWNB) + tx, own_base)
                        head += fx.Int32(GROUP_ROWS)
            gpu.barrier()
            if const_expr(tile_ready and resv_last is not None):  # noqa: SIM102
                # P2 system atomics completed before the barrier. No reservations follow fan2's last batch.
                if resv_last & (tx == fx.Int32(0)):
                    _tr_resv_signal(dest, resv_n)
            # P3: hit threads wait for map_ready and write metadata; each wave copies its records.
            # Issue all payload/scale loads for the wave first, overlapping group-base
            # resolution, then store the results after resolution completes.
            _pl = []

            def _issue_payload_loads():
                for rr in range_constexpr(_GB_NB // WAVES):
                    prec = wave + fx.Int32(rr * WAVES)
                    ptok = tok0 + prec * tstride
                    ptok_c = (ptok < ntokens).select(ptok, fx.Int32(0))
                    q0 = ptok_c * fx.Int32(REC_BYTES // 4) + fx.Int32(REC_Q // 4)
                    vals = []
                    for it in range_constexpr(_GB_IT):
                        d = (lane + fx.Int32(it * 64)) * fx.Int32(4)
                        dc = (d < fx.Int32(payload_dwords)).select(d, fx.Int32(0))
                        vals.append(
                            buffer_ops.buffer_load(
                                rs, q0 + dc, vec_width=4, dtype=T.i32
                            )
                        )
                    sv = None
                    sv = buffer_ops.buffer_load(
                        rs,
                        ptok_c * fx.Int32(REC_BYTES // 4)
                        + fx.Int32(REC_S // 4)
                        + (lane < fx.Int32(scale_dwords)).select(lane, fx.Int32(0)),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    _pl.append((prec, ptok, vals, sv))

            if const_expr(not data_send_ctas):
                _issue_payload_loads()
            # Resolve physical bases for at most two groups covered by [seg,seg+n).
            # P2 already claimed group heads, so this phase only waits, avoiding per-row
            # remote map reads. Two threads per expert resolve the first/last groups;
            # skip j=1 when both are the same. For groups this CTA claimed in P2, use
            # cached LDS bases; only other CTAs' groups require remote waits and reads.
            # This reduces worst-case serial remote round trips from four to two.
            if tx < fx.Int32(2 * LOCAL_EXPERTS):
                mo_e = tx % fx.Int32(LOCAL_EXPERTS)
                mo_j = tx // fx.Int32(LOCAL_EXPERTS)
                n = _gb_ld(fx.Int32(_GB_CNT) + mo_e)
                if n > fx.Int32(0):
                    seg = _gb_ld(fx.Int32(_GB_BASE) + mo_e)
                    g_first = seg // fx.Int32(GROUP_ROWS)
                    g_last = (seg + n - fx.Int32(1)) // fx.Int32(GROUP_ROWS)
                    gidx = (mo_j == fx.Int32(0)).select(g_first, g_last)
                    if (mo_j == fx.Int32(0)) | (g_last != g_first):
                        ph = fx.Int32(0)
                        # static_g0 fixes the first group's base at e*G, without remote map reads or waits.
                        sg_hit = gidx == fx.Int32(-1)
                        if const_expr(static_g0):
                            sg_hit = gidx == fx.Int32(0)
                        if _gb_ld(fx.Int32(_GB_OWN) + mo_e) == gidx:
                            ph = _gb_ld(fx.Int32(_GB_OWNB) + mo_e)
                        elif sg_hit:
                            ph = (mo_e + fx.Int32(_EO)) * fx.Int32(G)
                        else:
                            m = (mo_e + fx.Int32(_EO)) * fx.Int32(
                                max_tiles_per_expert
                            ) + gidx * fx.Int32(G)
                            _spin(
                                _peer_addr(dest, "expert_tile_map_ready")
                                + fx.Int64(m) * 8,
                                generation,
                            )
                            ph = buffer_ops.buffer_load(
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "expert_tile_map")
                                ),
                                m,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        _gb_st(
                            mo_j * fx.Int32(LOCAL_EXPERTS) + fx.Int32(_GB_PH) + mo_e,
                            ph,
                        )
            gpu.barrier()
            grow = fx.Int32(-1)
            if hit:
                row_slot = _gb_ld(fx.Int32(_GB_BASE) + le) + intra
                g_row = row_slot // fx.Int32(GROUP_ROWS)
                g_0 = _gb_ld(fx.Int32(_GB_BASE) + le) // fx.Int32(GROUP_ROWS)
                pbase = (g_row == g_0).select(
                    _gb_ld(fx.Int32(_GB_PH) + le),
                    _gb_ld(fx.Int32(_GB_PH + LOCAL_EXPERTS) + le),
                )
                g_end = (
                    _gb_ld(fx.Int32(_GB_BASE) + le)
                    + _gb_ld(fx.Int32(_GB_CNT) + le)
                    - fx.Int32(1)
                ) // fx.Int32(GROUP_ROWS)
                if (g_row > g_0) & (g_row < g_end):
                    pbase = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "expert_tile_map")
                        ),
                        (le + fx.Int32(_EO)) * fx.Int32(max_tiles_per_expert)
                        + g_row * fx.Int32(G),
                        vec_width=1,
                        dtype=T.i32,
                    )
                grow = (pbase + (row_slot // fx.Int32(BM)) % fx.Int32(G)) * fx.Int32(
                    BM
                ) + row_slot % fx.Int32(BM)
                weight = buffer_ops.buffer_load(
                    rs, id0 + fx.Int32(_gb_wdw) + slot, vec_width=1, dtype=T.f32
                )
                src = src_base + tok
                input_row = src
                if const_expr(COMPACT_INPUT):
                    input_row = _gb_ld(fx.Int32(_GB_HIT) + r) - fx.Int32(1)
                buffer_ops.buffer_store(
                    input_row,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_input")
                    ),
                    grow,
                    cache_modifier=_WT_META,
                )
                buffer_ops.buffer_store(
                    src | (slot << fx.Int32(24)),
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_source")
                    ),
                    grow,
                    cache_modifier=_WT_META,
                )
                buffer_ops.buffer_store(
                    weight,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_weight")
                    ),
                    grow,
                    cache_modifier=_WT_META,
                )
            if const_expr(not data_send_ctas):
                for prec, ptok, vals, sv in _pl:
                    if _gb_ld(fx.Int32(_GB_HIT) + prec) != fx.Int32(0):
                        input_row = src_base + ptok
                        if const_expr(COMPACT_INPUT):
                            input_row = _gb_ld(fx.Int32(_GB_HIT) + prec) - fx.Int32(1)
                        dst_payload = buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "grouped_input_q")
                            + fx.Int64(input_row) * fx.Int64(wire.payload_bytes),
                            num_records_bytes=wire.payload_bytes,
                        )
                        for it in range_constexpr(_GB_IT):
                            d = (lane + fx.Int32(it * 64)) * fx.Int32(4)
                            if d < fx.Int32(payload_dwords):
                                buffer_ops.buffer_store(
                                    vals[it], dst_payload, d, cache_modifier=_WT_PAY
                                )
                        if lane < fx.Int32(scale_dwords):
                            buffer_ops.buffer_store(
                                sv,
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "grouped_input_scale")
                                    + fx.Int64(input_row) * fx.Int64(scale_bytes)
                                ),
                                lane,
                                cache_modifier=_WT_PAY,
                            )
            if const_expr(not data_send_ctas):
                gpu.barrier()
            if const_expr(data_send_ctas):
                _send_enqueue(tok0, tstride, dest, src_base, remote)
            elif const_expr(tile_ready or ready_q):
                # Report group arrivals after all waves' remote metadata/payload stores complete.
                # [seg,seg+n) spans at most two groups, as assumed by _GB_PH's first/last resolution.
                # The atomic that reaches GROUP_ROWS is responsible for enqueueing the group.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                gpu.barrier()
                if tx < fx.Int32(LOCAL_EXPERTS):
                    tr_n = _gb_ld(fx.Int32(_GB_CNT) + tx)
                    if tr_n > fx.Int32(0):
                        tr_seg = _gb_ld(fx.Int32(_GB_BASE) + tx)
                        tr_g0 = tr_seg // fx.Int32(GROUP_ROWS)
                        tr_g1 = (tr_seg + tr_n - fx.Int32(1)) // fx.Int32(GROUP_ROWS)
                        tr_group = tr_g0
                        tr_remaining = tr_n
                        tr_amount = (tr_g0 + fx.Int32(1)) * fx.Int32(
                            GROUP_ROWS
                        ) - tr_seg
                        while tr_remaining > fx.Int32(0):
                            tr_amount = (tr_amount < tr_remaining).select(
                                tr_amount, tr_remaining
                            )
                            tr_ph = (tr_group == tr_g0).select(
                                _gb_ld(fx.Int32(_GB_PH) + tx),
                                _gb_ld(fx.Int32(_GB_PH + LOCAL_EXPERTS) + tx),
                            )
                            if (tr_group > tr_g0) & (tr_group < tr_g1):
                                tr_ph = buffer_ops.buffer_load(
                                    buffer_ops.create_buffer_resource_from_addr(
                                        _peer_addr(dest, "expert_tile_map")
                                    ),
                                    (tx + fx.Int32(_EO))
                                    * fx.Int32(max_tiles_per_expert)
                                    + tr_group * fx.Int32(G),
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            tr_old = fx.Int32(
                                comm.atomic_add_system(
                                    _peer_addr(dest, "tile_row_done")
                                    + fx.Int64(tr_ph) * fx.Int64(4),
                                    tr_amount,
                                )
                            )
                            if const_expr(ready_q):  # noqa: SIM102
                                if tr_old + tr_amount == fx.Int32(GROUP_ROWS):
                                    _rq_push(dest, tr_ph // fx.Int32(G))
                            tr_remaining -= tr_amount
                            tr_amount = fx.Int32(GROUP_ROWS)
                            tr_group += fx.Int32(1)

        # Copy roles consume only completed metadata batches. All descriptors
        # are local and write-through. No map lookup or remote reservation is
        # allowed in the copy loop. Generation tags avoid per-launch flag clears.
        SEND_NB = min(THREADS // TOPK, GROUP_ROWS) // WAVES * WAVES
        SEND_DW = (8 + SEND_NB + 4 * LOCAL_EXPERTS + 15) // 16 * 16
        SEND_C0 = 8 + SEND_NB
        SEND_P0 = SEND_C0 + LOCAL_EXPERTS
        SEND_C1 = SEND_P0 + LOCAL_EXPERTS
        SEND_P1 = SEND_C1 + LOCAL_EXPERTS

        def _send_join_done(dest, local_only=False):
            # One pending join counter: producers are registered at launch;
            # each queued copy registers before publication. Both retire once.
            # Thus old==1 has one unique owner and cannot race a late enqueue.
            jo = GPUS_PER_NODE if local_only else 0
            ja = local_addr("send_join") + fx.Int64(dest + fx.Int32(jo)) * fx.Int64(4)
            old = fx.Int32(comm.atomic_add_system(ja, fx.Int32(-1)))
            if old == fx.Int32(1):
                if const_expr(local_only):
                    comm.store_i64_global_system_relaxed(
                        _peer_addr(dest, "comm_eos")
                        + fx.Int64(GPUS_PER_NODE + local_rank) * fx.Int64(8),
                        generation,
                    )
                else:
                    for ci in range_constexpr(
                        (NODES - 1) * dispatch_chunks * layout.num_qp
                    ):
                        comm.atomic_add_system(
                            local_addr("remote_chunk_consumed") + fx.Int64(ci * 4),
                            fx.Int32(1),
                        )
                    comm.store_i64_global_system_relaxed(
                        _peer_addr(dest, "comm_eos") + fx.Int64(local_rank * 8),
                        generation,
                    )

        def _send_enqueue(tok0, stride, dest, src_base, remote):
            # The metadata batch is fully resolved before registering the copy.
            # All producers are still registered, so neither join can reach zero.
            if tx == fx.Int32(0):
                comm.atomic_add_agent(
                    local_addr("send_join") + fx.Int64(dest) * fx.Int64(4), fx.Int32(1)
                )
                if const_expr(split_local and not remote):
                    comm.atomic_add_agent(
                        local_addr("send_join")
                        + fx.Int64(dest + fx.Int32(GPUS_PER_NODE)) * fx.Int64(4),
                        fx.Int32(1),
                    )
                ji = fx.Int32(
                    comm.atomic_add_agent(local_addr("send_job_state"), fx.Int32(1))
                )
                _gb_st(fx.Int32(0), ji)
            gpu.barrier()
            ji = _gb_ld(fx.Int32(0))
            dr = buffer_ops.create_buffer_resource_from_addr(
                local_addr("send_job_desc") + fx.Int64(ji) * fx.Int64(SEND_DW * 4),
                num_records_bytes=SEND_DW * 4,
            )
            if tx == fx.Int32(0):
                for di, dv in enumerate(
                    [tok0, stride, dest, fx.Int32(int(remote)), src_base]
                ):
                    buffer_ops.buffer_store(
                        dv, dr, fx.Int32(di), cache_modifier=_WT_META
                    )
            if tx < fx.Int32(SEND_NB):
                hv = fx.Int32(0)
                if tx < fx.Int32(_GB_NB):
                    hv = _gb_ld(fx.Int32(_GB_HIT) + tx)
                buffer_ops.buffer_store(
                    hv, dr, fx.Int32(8) + tx, cache_modifier=_WT_META
                )
            if tx < fx.Int32(LOCAL_EXPERTS):
                n = _gb_ld(fx.Int32(_GB_CNT) + tx)
                c0 = fx.Int32(0)
                c1 = fx.Int32(0)
                p0 = fx.Int32(0)
                p1 = fx.Int32(0)
                if n > fx.Int32(0):
                    seg = _gb_ld(fx.Int32(_GB_BASE) + tx)
                    g0 = seg // fx.Int32(GROUP_ROWS)
                    c0 = (g0 + fx.Int32(1)) * fx.Int32(GROUP_ROWS) - seg
                    c0 = (c0 < n).select(c0, n)
                    c1 = n - c0
                    p0 = _gb_ld(fx.Int32(_GB_PH) + tx)
                    if c1 > fx.Int32(0):
                        p1 = _gb_ld(fx.Int32(_GB_PH + LOCAL_EXPERTS) + tx)
                    if c1 > fx.Int32(GROUP_ROWS):
                        # Negative C1 denotes multiple following groups. P1 is
                        # their first logical map index; all maps are published.
                        # Positive C1 retains the original count/physical pair.
                        c1 = -c1
                        p1 = (
                            tx
                            + fx.Int32(LOCAL_EXPERTS if split_local and remote else 0)
                        ) * fx.Int32(max_tiles_per_expert) + (
                            g0 + fx.Int32(1)
                        ) * fx.Int32(
                            G
                        )
                for di, dv in [
                    (SEND_C0, c0),
                    (SEND_P0, p0),
                    (SEND_C1, c1),
                    (SEND_P1, p1),
                ]:
                    buffer_ops.buffer_store(
                        dv, dr, fx.Int32(di) + tx, cache_modifier=_WT_META
                    )
            # Covers every metadata and descriptor writer, before ready release.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            if tx == fx.Int32(0):
                # All descriptor and row metadata stores are system write-through
                # and every writer drained before the CTA barrier above. Publish
                # readiness without flushing unrelated GEMM1 lines from this L2.
                comm.store_i64_global_system_relaxed(
                    local_addr("send_job_ready") + fx.Int64(ji) * fx.Int64(8),
                    generation,
                )

        def _send_copy_jobs():
            go = fx.Int32(1) == fx.Int32(1)
            while go:
                if tx == fx.Int32(0):
                    ji = fx.Int32(
                        comm.atomic_add_agent(
                            local_addr("send_job_state") + fx.Int64(4), fx.Int32(1)
                        )
                    )
                    ready_a = local_addr("send_job_ready") + fx.Int64(ji) * fx.Int64(8)
                    rv = fx.Int64(comm.load_i64_global_system_relaxed(ready_a))
                    ev = fx.Int64(
                        comm.load_i64_global_system_relaxed(local_addr("send_job_eos"))
                    )
                    while (rv < generation) & (ev < generation):
                        _mlir_rocdl.s_sleep(16)
                        rv = fx.Int64(comm.load_i64_global_system_relaxed(ready_a))
                        ev = fx.Int64(
                            comm.load_i64_global_system_relaxed(
                                local_addr("send_job_eos")
                            )
                        )
                    comm.fence_system_acquire()
                    total = fx.Int32(_pl32(local_addr("send_job_state")))
                    live = (rv >= generation) | (ji < total)
                    if live:
                        dr = buffer_ops.create_buffer_resource_from_addr(
                            local_addr("send_job_desc")
                            + fx.Int64(ji) * fx.Int64(SEND_DW * 4),
                            num_records_bytes=SEND_DW * 4,
                        )
                        for di in range_constexpr(5):
                            _gb_st(
                                fx.Int32(di + 1),
                                buffer_ops.buffer_load(
                                    dr, fx.Int32(di), vec_width=1, dtype=T.i32
                                ),
                            )
                    _gb_st(fx.Int32(0), live.select(ji, fx.Int32(-1)))
                gpu.barrier()
                ji = _gb_ld(fx.Int32(0))
                go = ji >= fx.Int32(0)
                if go:
                    tok0 = _gb_ld(fx.Int32(1))
                    stride = _gb_ld(fx.Int32(2))
                    dest = _gb_ld(fx.Int32(3))
                    remote = _gb_ld(fx.Int32(4))
                    src_base = _gb_ld(fx.Int32(5))
                    dr = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("send_job_desc")
                        + fx.Int64(ji) * fx.Int64(SEND_DW * 4),
                        num_records_bytes=SEND_DW * 4,
                    )
                    source_node = src_base // fx.Int32(MAX_TOKENS * GPUS_PER_NODE)
                    source_peer = source_node - (source_node > fx.Int32(node)).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                    remote_base = local_addr("remote_dispatch_rx") + fx.Int64(
                        source_peer
                    ) * fx.Int64(plane("remote_dispatch_rx") // (NODES - 1))
                    rs = buffer_ops.create_buffer_resource_from_addr(
                        (remote != fx.Int32(0)).select(remote_base, x_q)
                    )
                    # Wave-uniform live mask skips every inactive payload row.
                    # No CTA barrier, atomic, map wait or acquire inside copies.
                    for rr in range_constexpr(SEND_NB // WAVES):
                        prec = wave + fx.Int32(rr * WAVES)
                        hit = buffer_ops.buffer_load(
                            dr, fx.Int32(8) + prec, vec_width=1, dtype=T.i32
                        )
                        ptok = tok0 + prec * stride
                        if hit != fx.Int32(0):
                            input_row = src_base + ptok
                            if const_expr(COMPACT_INPUT):
                                input_row = hit - fx.Int32(1)
                            q0 = ptok * fx.Int32(REC_BYTES // 4) + fx.Int32(REC_Q // 4)
                            vals = []
                            for it in range_constexpr(_GB_IT):
                                d = (lane + fx.Int32(it * 64)) * fx.Int32(4)
                                dc = (d < fx.Int32(payload_dwords)).select(
                                    d, fx.Int32(0)
                                )
                                vals.append(
                                    buffer_ops.buffer_load(
                                        rs, q0 + dc, vec_width=4, dtype=T.i32
                                    )
                                )
                            sv = buffer_ops.buffer_load(
                                rs,
                                ptok * fx.Int32(REC_BYTES // 4)
                                + fx.Int32(REC_S // 4)
                                + (lane < fx.Int32(scale_dwords)).select(
                                    lane, fx.Int32(0)
                                ),
                                vec_width=1,
                                dtype=T.i32,
                            )
                            pay = buffer_ops.create_buffer_resource_from_addr(
                                _peer_addr(dest, "grouped_input_q")
                                + fx.Int64(input_row) * fx.Int64(wire.payload_bytes),
                                num_records_bytes=wire.payload_bytes,
                            )
                            for it in range_constexpr(_GB_IT):
                                d = (lane + fx.Int32(it * 64)) * fx.Int32(4)
                                if d < fx.Int32(payload_dwords):
                                    buffer_ops.buffer_store(
                                        vals[it], pay, d, cache_modifier=_WT_PAY
                                    )
                            if lane < fx.Int32(scale_dwords):
                                buffer_ops.buffer_store(
                                    sv,
                                    buffer_ops.create_buffer_resource_from_addr(
                                        _peer_addr(dest, "grouped_input_scale")
                                        + fx.Int64(input_row) * fx.Int64(scale_bytes)
                                    ),
                                    lane,
                                    cache_modifier=_WT_PAY,
                                )
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    gpu.barrier()
                    if tx < fx.Int32(LOCAL_EXPERTS):
                        for ci, pi in [(SEND_C0, SEND_P0), (SEND_C1, SEND_P1)]:
                            cnt = buffer_ops.buffer_load(
                                dr, fx.Int32(ci) + tx, vec_width=1, dtype=T.i32
                            )
                            if cnt > fx.Int32(0):
                                ph = buffer_ops.buffer_load(
                                    dr, fx.Int32(pi) + tx, vec_width=1, dtype=T.i32
                                )
                                old = fx.Int32(
                                    comm.atomic_add_system(
                                        _peer_addr(dest, "tile_row_done")
                                        + fx.Int64(ph) * fx.Int64(4),
                                        cnt,
                                    )
                                )
                                if old + cnt == fx.Int32(GROUP_ROWS):
                                    _rq_push(dest, ph // fx.Int32(G))
                            if const_expr(ci == SEND_C1):  # noqa: SIM102
                                if cnt < fx.Int32(0):
                                    logical = buffer_ops.buffer_load(
                                        dr, fx.Int32(pi) + tx, vec_width=1, dtype=T.i32
                                    )
                                    remaining = -cnt
                                    while remaining > fx.Int32(0):
                                        amount = (
                                            remaining < fx.Int32(GROUP_ROWS)
                                        ).select(remaining, fx.Int32(GROUP_ROWS))
                                        ph = buffer_ops.buffer_load(
                                            buffer_ops.create_buffer_resource_from_addr(
                                                _peer_addr(dest, "expert_tile_map")
                                            ),
                                            logical,
                                            vec_width=1,
                                            dtype=T.i32,
                                        )
                                        old = fx.Int32(
                                            comm.atomic_add_system(
                                                _peer_addr(dest, "tile_row_done")
                                                + fx.Int64(ph) * fx.Int64(4),
                                                amount,
                                            )
                                        )
                                        if old + amount == fx.Int32(GROUP_ROWS):
                                            _rq_push(dest, ph // fx.Int32(G))
                                        remaining -= amount
                                        logical += fx.Int32(G)
                    # Join only after every publication and RX read has retired.
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        if const_expr(split_local):  # noqa: SIM102
                            if remote == fx.Int32(0):
                                _send_join_done(dest, True)
                        _send_join_done(dest)

        # early_local_gmm seals padding and group maps over expert*SPL threads instead
        # of serially per expert. F8T2 phase 1 padding took ~12us and map/publication
        # ~11us, both on local GMM1's critical path.
        SPL = max(1, min(8, THREADS // max(1, LOCAL_EXPERTS)))

        def _seal_pad_par(EO):
            """Parallel _seal_pad for the non-split path (lite_sync, static_g0).
            Thread (e,l) handles rows l,l+SPL,... of each padded tile for expert e.
            The sealer has acquired all comm_eos values, so maps are published and
            per-tile map_ready polling is unnecessary.
            """
            expert_count = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            if tx < fx.Int32(LOCAL_EXPERTS * SPL):
                pe = tx // fx.Int32(SPL)
                pl = tx - pe * fx.Int32(SPL)
                count = buffer_ops.buffer_load(
                    expert_count, pe + fx.Int32(EO), vec_width=1, dtype=T.i32
                )
                alloc_tiles = _alloc_tiles_for(count)
                first_pad_tile = count // fx.Int32(BM)
                tile_map_res = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("expert_tile_map")
                )
                sources = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_row_source")
                )
                inputs = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_row_input")
                )
                weights = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_row_weight")
                )
                map_row0 = (pe + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert)
                for pad_tile in range(first_pad_tile, alloc_tiles, fx.Int32(1)):
                    map_index = map_row0 + pad_tile
                    physical = buffer_ops.buffer_load(
                        tile_map_res, map_index, vec_width=1, dtype=T.i32
                    )
                    head_physical = buffer_ops.buffer_load(
                        tile_map_res, map_row0, vec_width=1, dtype=T.i32
                    )
                    fallback_input = buffer_ops.buffer_load(
                        inputs, head_physical * fx.Int32(BM), vec_width=1, dtype=T.i32
                    )
                    # static_g0: a zero-row expert has no first row. Point padding at source row 0;
                    # its source is invalid, so the computed result is never consumed.
                    fallback_input = (count > fx.Int32(0)).select(
                        fallback_input, fx.Int32(0)
                    )
                    row_start = (pad_tile == first_pad_tile).select(
                        count % fx.Int32(BM), fx.Int32(0)
                    )
                    for row in range(row_start + pl, fx.Int32(BM), fx.Int32(SPL)):
                        dst = physical * fx.Int32(BM) + row
                        buffer_ops.buffer_store(fallback_input, inputs, dst)
                        buffer_ops.buffer_store(fx.Int32(INVALID_SOURCE), sources, dst)
                        buffer_ops.buffer_store(fx.Float32(0.0), weights, dst)

        def _seal_pad(EO):
            """Pad tail/empty tiles for expert_count[EO:EO+LE], EO=0 local/all or EO=LE remote."""
            expert_count = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            if tx < fx.Int32(LOCAL_EXPERTS):
                count = buffer_ops.buffer_load(
                    expert_count, tx + fx.Int32(EO), vec_width=1, dtype=T.i32
                )
                # With G>1, allocated tiles round up to a multiple of G. Pad both the last
                # real tile's tail rows and any extra empty tiles in the group.
                real_tiles = (count + fx.Int32(BM - 1)) // fx.Int32(BM)
                alloc_tiles = (
                    (real_tiles + fx.Int32(G - 1)) // fx.Int32(G)
                ) * fx.Int32(G)
                first_pad_tile = count // fx.Int32(BM)
                for pad_tile in range(first_pad_tile, alloc_tiles, fx.Int32(1)):
                    logical_tile = pad_tile
                    map_index = (tx + fx.Int32(EO)) * fx.Int32(
                        max_tiles_per_expert
                    ) + logical_tile
                    _spin(
                        local_addr("expert_tile_map_ready")
                        + fx.Int64(map_index) * fx.Int64(8),
                        generation,
                    )
                    tile_map_res = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("expert_tile_map")
                    )
                    physical = buffer_ops.buffer_load(
                        tile_map_res, map_index, vec_width=1, dtype=T.i32
                    )
                    sources = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_source")
                    )
                    inputs = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_input")
                    )
                    weights = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_weight")
                    )
                    # Use row 0 of this expert's first tile as fallback. Extra empty tiles never
                    # received their own row 0. Here count>0; otherwise alloc_tiles is zero and
                    # the loop is empty. Logical tile 0 therefore exists and is populated.
                    head_physical = buffer_ops.buffer_load(
                        tile_map_res,
                        (tx + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    fallback_input = buffer_ops.buffer_load(
                        inputs,
                        head_physical * fx.Int32(BM),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    row_start = (pad_tile == first_pad_tile).select(
                        count % fx.Int32(BM), fx.Int32(0)
                    )
                    for row in range(row_start, fx.Int32(BM), fx.Int32(1)):
                        dst = physical * fx.Int32(BM) + row
                        buffer_ops.buffer_store(fallback_input, inputs, dst)
                        buffer_ops.buffer_store(fx.Int32(INVALID_SOURCE), sources, dst)
                        buffer_ops.buffer_store(fx.Float32(0.0), weights, dst)

        def _seal_perm(segs, base_segs, check_tiles, publish_seg, par=False):
            """Build the expert-major permutation.
            segs concatenates map segments per expert: 0 is local/all, LE is remote.
            base_segs precedes the whole view; with split_local and no h1_phys, remote
            segments start after the total local tile count. For h1_phys+split_local,
            segs=(0,LE) keeps each expert contiguous, with local groups before remote groups.
            With par, thread (e,l) handles tiles/groups l,l+SPL,... for expert e and
            recomputes prefix sums, shortening serial load/store chains by a factor of SPL.
            """
            PS = SPL if par else 1
            pe = tx // fx.Int32(PS)
            pl = tx - pe * fx.Int32(PS)
            counts_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            seg_base = fx.Int32(0)
            for bi in range_constexpr(len(base_segs)):
                bo = base_segs[bi]
                for e in range_constexpr(LOCAL_EXPERTS):
                    seg_base = seg_base + _alloc_tiles_for(
                        buffer_ops.buffer_load(
                            counts_res, fx.Int32(bo + e), vec_width=1, dtype=T.i32
                        )
                    )
            sorted_e_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_expert_sorted")
            )
            inv_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_src_of_dst")
            )
            if tx < fx.Int32(LOCAL_EXPERTS * PS):
                my_base = seg_base
                all_tiles = fx.Int32(0)
                # range_constexpr, not a runtime loop: the accumulator is
                # loop-carried and this package only ever carries values
                # across unrolled loops.  One cache line of counts per segment.
                for si in range_constexpr(len(segs)):
                    so = segs[si]
                    for e in range_constexpr(LOCAL_EXPERTS):
                        c = buffer_ops.buffer_load(
                            counts_res, fx.Int32(so + e), vec_width=1, dtype=T.i32
                        )
                        n = _alloc_tiles_for(c)
                        my_base = my_base + (fx.Int32(e) < pe).select(n, fx.Int32(0))
                        all_tiles = all_tiles + n
                if tx == fx.Int32(0):
                    if const_expr(check_tiles is not None):  # noqa: SIM102
                        # Sum ceil(ceil(count/BM)/G)*G over all segments must equal tile_alloc.
                        # Each claim allocates exactly G tiles, so the permutation remains a bijection.
                        if seg_base + all_tiles != check_tiles:
                            comm.atomic_add_system(error_addr, fx.Int32(1))
                    if const_expr(publish_seg):
                        # Local-segment tile count, used to derive F5's local job count.
                        buffer_ops.buffer_store(
                            all_tiles,
                            buffer_ops.create_buffer_resource_from_addr(
                                local_addr("tile_alloc")
                            ),
                            fx.Int32(1),
                        )
                tile_map = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("expert_tile_map")
                )
                # The two group maps have regions only when G>1. Even resource construction
                # raises KeyError during tracing at G=1; all uses stay inside const_expr(G > 1).
                group_e_res = (
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_expert_group")
                    )
                    if G > 1
                    else None
                )
                run_base = my_base
                for si in range_constexpr(len(segs)):
                    so = segs[si]
                    map0 = (pe + fx.Int32(so)) * fx.Int32(max_tiles_per_expert)
                    my_tiles = _alloc_tiles_for(
                        buffer_ops.buffer_load(
                            counts_res, pe + fx.Int32(so), vec_width=1, dtype=T.i32
                        )
                    )
                    for j in range(pl, my_tiles, fx.Int32(PS)):
                        map_index = map0 + j
                        if const_expr(not lite_sync):
                            _spin(
                                local_addr("expert_tile_map_ready")
                                + fx.Int64(map_index) * fx.Int64(8),
                                generation,
                            )
                        src = buffer_ops.buffer_load(
                            tile_map, map_index, vec_width=1, dtype=T.i32
                        )
                        dst = run_base + j
                        buffer_ops.buffer_store(pe, sorted_e_res, dst)
                        buffer_ops.buffer_store(src, inv_res, dst)
                    if const_expr(G > 1):
                        # GMM1 uses groups as m_blocks, so provide group-level expert IDs and destinations.
                        # run_base is a G-aligned prefix sum, and source bases come from whole-group
                        # claims, aligning both sides to G. Use a separate group loop without modulo
                        # on the induction variable or conditional stores in its body.
                        for jg in range(pl, my_tiles // fx.Int32(G), fx.Int32(PS)):
                            j0 = jg * fx.Int32(G)
                            head_src = buffer_ops.buffer_load(
                                tile_map,
                                map0 + j0,
                                vec_width=1,
                                dtype=T.i32,
                            )
                            head_group = head_src // fx.Int32(G)
                            buffer_ops.buffer_store(pe, group_e_res, head_group)
                    run_base = run_base + my_tiles

        def _seal_seg1():
            """Seal split_local phase 1 after eight local_eos values confirm all local fan1.
            Pad local groups and publish their tile count plus early_local_gmm's group
            map and h1_local_eos. In early_local_gmm, fan CTAs with ticket>=8 call this:
            marker 7 means local EOS complete, 8 padding complete, and 9 published.
            Consumers use only 3/4/5/6. Otherwise the sealer calls it without markers,
            because slots 7..9 would conflict with T0.
            """
            # Phase 1: eight local_eos values mean all local rows have arrived and counts are final.
            # Seal local padding/permutation at h1 base 0, then wait for comm_eos to seal remote groups.
            if tx == fx.Int32(0):
                for peer in range_constexpr(GPUS_PER_NODE):
                    _spin(
                        local_addr("comm_eos") + fx.Int64((GPUS_PER_NODE + peer) * 8),
                        generation,
                    )
            _cta_acquire()
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            # With h1_phys, phase 2 builds the full sorted view, local then remote per expert.
            # Publish only the local tile count here; otherwise local tiles form a segment at base 0.
            if tx == fx.Int32(0):
                _lt = fx.Int32(0)
                _cr = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("expert_count")
                )
                # local_defer excludes the last D experts' local groups from the local segment.
                # Consumers derive its bound and the remote start from tile_alloc[1], keeping
                # group-map layout unchanged. These groups start the remote segment and share
                # a weight read with their remote groups, which already require all weights.
                for e in range_constexpr(LOCAL_EXPERTS):
                    _lt = _lt + _alloc_tiles_for(
                        buffer_ops.buffer_load(
                            _cr, fx.Int32(e), vec_width=1, dtype=T.i32
                        )
                    )
                buffer_ops.buffer_store(
                    _lt,
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_alloc")
                    ),
                    fx.Int32(1),
                )
            if const_expr(ready_q):
                _rq_seal_tails(0)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            if const_expr(early_local_gmm):  # noqa: SIM102
                if tx == fx.Int32(0):
                    comm.fence_system_release()
                    # The fence already published prior writes; a release store would write back L2 again.
                    (comm.store_i64_global_system_relaxed)(
                        local_addr("h1_local_eos"), generation
                    )

        dest = fx.Int32(0)
        # Shard each destination's fanout across CTAs to hide peer round trips.
        is_fanout = ticket < fx.Int32(GPUS_PER_NODE * fanout_ctas_per_dest)
        # T0 handles CCO, credits, and sealing on the critical path. Exclude it from
        # dest0 shard 0; distribute dest0 over the remaining fanout_ctas_per_dest-1 shards.
        is_fanout = is_fanout & (ticket != fx.Int32(0))
        if is_fanout:
            fan_stride = fx.Int32(fanout_ctas_per_dest)
            dest = ticket % fx.Int32(GPUS_PER_NODE)
            fanout_shard = ticket // fx.Int32(GPUS_PER_NODE)
            d0 = dest == fx.Int32(0)
            fanout_shard = d0.select(fanout_shard - fx.Int32(1), fanout_shard)
            fan_stride = d0.select(fx.Int32(fanout_ctas_per_dest - 1), fan_stride)
            # Shard ID/token stride for actual work; completion still uses fan_stride,
            # including posting CTAs with no tokens.
            fan_wshard = fanout_shard
            fan_tstride = fan_stride
            pn_dest = (dest >= fx.Int32(_POST_LO)) & (dest < fx.Int32(_POST_HI))
            pn_self = (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI))
            fan_tstride = pn_dest.select(fan_stride - fx.Int32(1), fan_stride)
            fan_wshard = pn_self.select(
                fx.Int32(1 << 20),
                pn_dest.select(fanout_shard - fx.Int32(1), fanout_shard),
            )
            # One CTA posts and rings each QP's doorbell, as kernel2 DBCTAS does: ticket q handles QP q.
            if ((ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI))) & (
                wave == fx.Int32(0)
            ):
                _rail_post(ticket - fx.Int32(1), True)
            if tx == fx.Int32(0):
                _spin(_peer_addr(dest, "launch_ready"), control_generation)
            gpu.barrier()
            for token in range(
                fan_wshard, fx.Int32(MAX_TOKENS), fan_tstride * fx.Int32(_GB_NB)
            ):
                _route_gbatch(
                    buffer_ops.create_buffer_resource_from_addr(x_q),
                    token,
                    fan_tstride,
                    dest,
                    fx.Int32(rank * MAX_TOKENS),
                    emit_masks=True,
                )
            if const_expr(group_ready):  # noqa: SIM102
                if dest == fx.Int32(0):
                    # Only dest0 shards write the local-source mask, but the
                    # sealer observes EOS from shards targeting local_rank.
                    # Join the actual writers rather than relying on their
                    # scheduling relative to those unrelated EOS publishers.
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        mask_join = local_addr("node_mask_ready") + fx.Int64(NODES * 8)
                        mask_prev = fx.Int64(
                            comm.atomic_add_agent_acq_rel(mask_join, fx.Int64(1))
                        )
                        if mask_prev + fx.Int64(1) == fx.Int64(fan_stride):
                            comm.atomic_add_agent(mask_join, -fx.Int64(fan_stride))
                            comm.store_i64_global_system(
                                local_addr("node_mask_ready"), generation
                            )
            if const_expr(split_local):
                # local_eos means this source rank's local fan1 half is complete for dest.
                # Use comm_eos-style shard counting and subtractive reset in counters [8,16).
                # It travels only over xGMI, independent of the rail, adding no inter-node wait cycle.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    if const_expr(data_send_ctas):
                        _send_join_done(dest, True)
                    else:
                        _ldone = (
                            local_addr("fanout_shard_done")
                            + fx.Int64(GPUS_PER_NODE * 4)
                            + fx.Int64(dest) * fx.Int64(4)
                        )
                        _lprev = fx.Int32(
                            comm.atomic_add_system_acq_rel(_ldone, fx.Int32(1))
                        )
                        if _lprev + fx.Int32(1) == fan_stride:
                            comm.atomic_add_agent(_ldone, fx.Int32(0) - fan_stride)
                            comm.fence_system_release()
                            (comm.store_i64_global_system_relaxed)(
                                _peer_addr(dest, "comm_eos")
                                + fx.Int64((GPUS_PER_NODE + local_rank) * 8),
                                generation,
                            )
            if const_expr(early_local_gmm):  # noqa: SIM102
                # Seal local groups after intra-node fanout; this does not wait
                # for remote fanout and stays outside the rail-post pool.
                if ticket == fx.Int32(GPUS_PER_NODE * _SEG1_SHARD + local_rank):
                    _seal_seg1()
            for remote_index in range_constexpr(NODES - 1):
                remote_node = REMOTE_NODES[remote_index]
                # One flush publishes all dispatch chunks together; wait once per CTA.
                if (tx < fx.Int32(layout.num_qp)) & (
                    fan_wshard < fx.Int32(F2S if F2S else 1 << 20)
                ):
                    _spin(
                        peer_local_addr("remote_chunk_ready", remote_node)
                        + fx.Int64(tx) * fx.Int64(8),
                        generation,
                    )
                _cta_acquire()
                f2_stride = fx.Int32(F2S) if F2S else fan_tstride
                f2_end = (
                    (fan_wshard < fx.Int32(F2S)).select(
                        fx.Int32(MAX_TOKENS), fx.Int32(0)
                    )
                    if F2S
                    else fx.Int32(MAX_TOKENS)
                )
                for token in range(fan_wshard, f2_end, f2_stride * fx.Int32(_GB_NB)):
                    _route_gbatch(
                        buffer_ops.create_buffer_resource_from_addr(
                            peer_local_addr("remote_dispatch_rx", remote_node)
                        ),
                        token,
                        f2_stride,
                        dest,
                        fx.Int32(
                            (remote_node * GPUS_PER_NODE + local_rank) * MAX_TOKENS
                        ),
                        remote=True,
                        resv_last=(
                            token + f2_stride * fx.Int32(_GB_NB) >= f2_end
                            if tile_ready and remote_node == REMOTE_NODES[-1]
                            else None
                        ),
                        resv_n=fan_stride,
                    )
                if const_expr(  # noqa: SIM102
                    tile_ready and remote_node == REMOTE_NODES[-1]
                ):
                    # Fan CTAs with no batch, including posting CTAs, must still report once to reach fan_stride.
                    if (fan_wshard >= f2_end) & (tx == fx.Int32(0)):
                        _tr_resv_signal(dest, fan_stride)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            # Reap this CTA's data request before publishing its shard. It must complete
            # before all consumed credits arrive, so T0's credit phase cannot poll the same QP CQ concurrently.
            if (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI)):
                if wave == fx.Int32(0):
                    post_qp = ticket - fx.Int32(1)
                    for remote_index in range_constexpr(NODES - 1):
                        remote_node = REMOTE_NODES[remote_index]
                        post_req = fx.Int64(
                            comm.load_i64_global(
                                peer_local_addr("remote_chunk_request", remote_node)
                                + fx.Int64(post_qp) * fx.Int64(8)
                            )
                        )
                        _rail.wait(dev_comm, post_qp, post_req)
                gpu.barrier()
            # k1 never touches fanout_shard_done, comm_eos, or remote_chunk_consumed.
            # They describe all destination records completing, whereas k1 completes only
            # the local-source half.
            if tx == fx.Int32(0):
                if const_expr(data_send_ctas):
                    _send_join_done(dest)
                    mp = fx.Int32(
                        comm.atomic_add_agent(
                            local_addr("send_job_state") + fx.Int64(8), fx.Int32(1)
                        )
                    )
                    if mp + fx.Int32(1) == fx.Int32(SEND_FIRST - 1):
                        comm.fence_system_release()
                        comm.store_i64_global_system_relaxed(
                            local_addr("send_job_eos"), generation
                        )
                else:
                    # Only the final shard per destination publishes EOS and increments credit.
                    # Each remote_chunk_consumed word must reach exactly GPUS_PER_NODE, not one
                    # increment per shard. Reset by subtraction, as Stage2's arrival protocol does,
                    # rather than relying on next-generation clearing.
                    # The system release RMW publishes this shard's drained remote writes. Its
                    # acquire lets the final arrival observe all shards, then the following
                    # release publishes completion through comm_eos.
                    # With fan_wt, stores are write-through and complete at s_waitcnt(0), avoiding
                    # release's full L2 writeback. Readers still acquire after observing comm_eos.
                    _prev = fx.Int32(
                        (
                            comm.atomic_add_system
                            if fan_wt
                            else comm.atomic_add_system_acq_rel
                        )(
                            local_addr("fanout_shard_done")
                            + fx.Int64(dest) * fx.Int64(4),
                            fx.Int32(1),
                        )
                    )
                    _publish = _prev + fx.Int32(1) == fan_stride
                    if _publish:
                        comm.atomic_add_agent(
                            local_addr("fanout_shard_done")
                            + fx.Int64(dest) * fx.Int64(4),
                            fx.Int32(0) - fan_stride,
                        )
                    if _publish:
                        if const_expr(lite_sync):
                            # The acq_rel RMW already published this CTA's writes and acquired other shards.
                            # Only local T0 reads consumed, using a system load and acquire fence. Fan RX
                            # reads completed at s_waitcnt(0), so credit has no race and a relaxed atomic suffices.
                            for consume_index in range_constexpr(
                                (NODES - 1) * dispatch_chunks * layout.num_qp
                            ):
                                comm.atomic_add_system(
                                    local_addr("remote_chunk_consumed")
                                    + fx.Int64(consume_index * 4),
                                    fx.Int32(1),
                                )
                        else:
                            comm.fence_system_release()
                            for consume_index in range_constexpr(
                                (NODES - 1) * dispatch_chunks * layout.num_qp
                            ):
                                comm.atomic_add_system_acq_rel(
                                    local_addr("remote_chunk_consumed")
                                    + fx.Int64(consume_index * 4),
                                    fx.Int32(1),
                                )
                        (comm.store_i64_global_system_relaxed)(
                            _peer_addr(dest, "comm_eos") + fx.Int64(local_rank) * 8,
                            generation,
                        )

        if const_expr(data_send_ctas):  # noqa: SIM102
            if (ticket >= fx.Int32(SEND_FIRST)) & (ticket < fx.Int32(SEND_END)):
                _send_copy_jobs()

        # The CCO CTA is also destination role zero, so delayed-credit progress
        # must run only after the common is_comm path above has contributed its
        # own consumed count. Otherwise the scoreboard can reach only seven.

        if is_cco:
            # One wave per QP posts its credits (a QP has one poster); the extra
            # waves of an 8-wave CTA only join the waits and barriers.
            qp = wave % fx.Int32(layout.num_qp) if WAVES > layout.num_qp else wave
            qp_poster = (
                (wave < fx.Int32(layout.num_qp)) if WAVES > layout.num_qp else None
            )
            for remote_index in range_constexpr(NODES - 1):
                remote_node = REMOTE_NODES[remote_index]
                for chunk in range(fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)):
                    # Do not credit a chunk until every destination fanout role has
                    # consumed its reciprocal payload.
                    consume_index = chunk * fx.Int32(layout.num_qp) + qp
                    if lane == fx.Int32(0):
                        consumed_count = fx.Int32(
                            comm.load_i32_global_system(
                                peer_local_addr("remote_chunk_consumed", remote_node)
                                + fx.Int64(consume_index) * fx.Int64(4)
                            )
                        )
                        while consumed_count < fx.Int32(GPUS_PER_NODE):
                            consumed_count = fx.Int32(
                                _pl32(
                                    peer_local_addr(
                                        "remote_chunk_consumed", remote_node
                                    )
                                    + fx.Int64(consume_index) * fx.Int64(4)
                                )
                            )
                        comm.fence_system_acquire()
                    _cta_acquire()
                    credit_byte = peer_window_off(
                        "remote_chunk_credit", remote_node
                    ) + fx.Int64(consume_index) * fx.Int64(8)
                    if const_expr(WAVES <= layout.num_qp):
                        _rail.put_value(
                            dev_comm,
                            qp,
                            fx.Int32(remote_node),
                            window_handle("remote_chunk_credit"),
                            credit_byte,
                            generation,
                            aggregate=True,
                        )
                        # Ring per chunk so the peer sees each credit as soon as it is
                        # earned; nothing here needs our own completion, so reap only
                        # the last request (one QP completes in order).
                        credit_req = _rail.flush_async(
                            dev_comm,
                            qp,
                            fx.Int32(remote_node),
                        )
                        if chunk == fx.Int32(dispatch_chunks - 1):
                            _rail.wait(dev_comm, qp, credit_req)
                    else:
                        if qp_poster:
                            _rail.put_value(
                                dev_comm,
                                qp,
                                fx.Int32(remote_node),
                                window_handle("remote_chunk_credit"),
                                credit_byte,
                                generation,
                                aggregate=True,
                            )
                            # Ring per chunk so the peer sees each credit as soon as it is
                            # earned; nothing here needs our own completion, so reap only
                            # the last request (one QP completes in order).
                            credit_req_w = _rail.flush_async(
                                dev_comm,
                                qp,
                                fx.Int32(remote_node),
                            )
                            if chunk == fx.Int32(dispatch_chunks - 1):
                                _rail.wait(dev_comm, qp, credit_req_w)

        # The role targeting this rank observes exactly eight EOS values; each
        # one covers a local source rank and its aligned remote source rank.
        if is_sealer:
            if const_expr(early_local_gmm):
                # Fan ticket 8+local_rank runs phase 1 after its fan1 work; see fanout.
                # Phase 2 permutation/sortcopy reads local padding and must follow phase 1.
                if tx == fx.Int32(0):
                    (comm.spin_until_ge_i64_sleep_rx)(
                        local_addr("h1_local_eos"), generation, 127
                    )
                _cta_acquire()
            if tx == fx.Int32(0):
                # Independent lane acquires followed by a CTA barrier do not
                # merge into one happens-before chain. The publishing thread
                # itself must acquire all eight communication-role releases.
                for peer in range_constexpr(GPUS_PER_NODE):
                    _spin(local_addr("comm_eos") + fx.Int64(peer * 8), generation)
            _cta_acquire()
            if const_expr(split_local):
                # Remote groups; also fill local tails deferred by lazy_pad.
                _seal_pad(LOCAL_EXPERTS)
                _seal_pad(0)
            else:
                _seal_pad_par(0)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            # Sealer subphases reuse T0 slots 5..7. Rank 0's sealer is T0, so omit ranks 0/8 in analysis.
            finish_scratch = fx.recast_iter(fx.Int32, lds_raw)
            if tx == fx.Int32(0):
                tiles = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_alloc")
                    ),
                    fx.Int32(0),
                    vec_width=1,
                    dtype=T.i32,
                )
                fx.ptr_store(Vec.from_elements([tiles], fx.Int32), finish_scratch)
            gpu.barrier()
            tiles = Vec(fx.make_view(finish_scratch, fx.make_layout(1, 1)).load())[0]
            # Expert-major view for Stage2.  expert_count is final here and
            # expert_tile_map[e][j] is dense in j, so one thread per expert
            # expands the permutation directly.  Each expert thread
            # recomputes its own tile base from the LOCAL_EXPERTS counts
            # rather than running a scan: the counts are one cache line
            # deep and this costs no LDS and no extra barrier.
            if const_expr(split_local):
                _seal_perm((0, LOCAL_EXPERTS), (), tiles, False)
                if const_expr(ready_q):
                    _rq_seal_tails(LOCAL_EXPERTS)
            else:
                _seal_perm((0,), (), tiles, False, par=True)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            if tx == fx.Int32(0):
                if const_expr(group_ready):
                    for mask_plane in range_constexpr(NODES):
                        _spin(
                            local_addr("node_mask_ready") + fx.Int64(mask_plane * 8),
                            generation,
                        )
                    comm.fence_system_acquire()
                buffer_ops.buffer_store(
                    tiles * fx.Int32(BM),
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("num_valid")
                    ),
                    fx.Int32(0),
                )
                comm.fence_system_release()
                # The fence already published prior writes; a release store would write back L2 again.
                (comm.store_i64_global_system_relaxed)(
                    local_addr("h1_queue_eos"), generation
                )

        def _run_gemm1_job(job):
            _gemm1_body(
                lds_raw,
                local_addr("grouped_input_q"),
                local_addr("grouped_input_scale"),
                w1q,
                w1scale,
                (
                    local_addr("tile_expert_group")
                    if G > 1
                    else local_addr("tile_expert")
                ),
                local_addr("tile_row_input"),
                local_addr("h1_output_q"),
                local_addr("h1_output_scale"),
                job,
                lane,
                wave,
                fx.Int32(layout.input_capacity),
                BM=GBM,
                BN=GBN,
                BK=BK,
                K=HIDDEN,
                N_OUT=2 * INTER,
                NE=LOCAL_EXPERTS,
                act=activation,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                NW=WAVES,
                aqin_row_base=INPUT_ROW_BASE,
                aqout_row_base=H1_ROW_BASE,
                h1_wt=group_ready,
                a_dtype=A_DTYPE,
            )
            if const_expr(group_ready):
                # Every wave's write-through h1 stores retired, then one count
                # per job; the group's last job publishes the generation.
                fx.rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    gr_g = job // fx.Int32(GNB)
                    gr_sync = local_addr("h1_grp_sync")
                    gr_old = fx.Int32(
                        comm.atomic_add_agent(
                            gr_sync + fx.Int64(gr_g) * fx.Int64(4), fx.Int32(1)
                        )
                    )
                    if gr_old == fx.Int32(GNB - 1):
                        # Relaxed: the other jobs' h1 stores retired write-through
                        # before their counts; no L2 write-back is needed.
                        comm.store_i32_global_system_relaxed(
                            gr_sync + fx.Int64(gr_g) * fx.Int64(4), fx.Int32(0)
                        )
                        comm.store_i32_global_system_relaxed(
                            gr_sync + fx.Int64(gr_g + fx.Int32(GR_N)) * fx.Int64(4),
                            fx.Int32(generation),
                        )

        # GMM1 consumers.
        # Originally only 120 tickets >= ESSENTIAL_CTAS computed GMM1. Eight comm and
        # 128 producer CTAs exited after their work, leaving 136/256 CTAs idle throughout
        # GMM1 (~797us, 46% of Stage1). All CTAs already crossed the same EOS gate,
        # so allowing them all to compute preserves ordering. The closed-form
        # (consumer_index, consumer_count) bookkeeping already supports this mapping.
        is_compute = ticket >= fx.Int32(COMPUTE_FIRST)
        for _xt in _XC:
            is_compute = is_compute | (ticket == fx.Int32(_xt))
        if is_compute:
            if const_expr(ready_q and not split_local):  # noqa: SIM102
                # Without split, expert_count is final once all source fan CTAs finish P2.
                # Consumer 0 adds missing rows to each tail group's arrival count, including
                # the fully padded static_g0 group for zero-row experts. Padding and real
                # arrivals update the same word; the atomic reaching GROUP_ROWS enqueues it.
                # The sealer later fills tile_row_* tails. GMM1 uses bounds-checked tile_row_input;
                # Stage2 discards padded h1 values through INVALID_SOURCE and zero weights.
                if ticket == fx.Int32(COMPUTE_FIRST):
                    if tx == fx.Int32(0):
                        pd_rf = fx.Int32(
                            _pl32(local_addr("gmm1_job_head") + fx.Int64(16 * 4))
                        )
                        while pd_rf < fx.Int32(GPUS_PER_NODE):
                            _mlir_rocdl.s_sleep(127)
                            pd_rf = fx.Int32(
                                _pl32(local_addr("gmm1_job_head") + fx.Int64(16 * 4))
                            )
                    _cta_acquire()
                    for pd_e in range(tx, fx.Int32(LOCAL_EXPERTS), fx.Int32(THREADS)):
                        pd_c = buffer_ops.buffer_load(
                            buffer_ops.create_buffer_resource_from_addr(
                                local_addr("expert_count")
                            ),
                            pd_e,
                            vec_width=1,
                            dtype=T.i32,
                        )
                        pd_k = pd_c // fx.Int32(GROUP_ROWS)
                        pd_r = pd_c - pd_k * fx.Int32(GROUP_ROWS)
                        if (pd_r != fx.Int32(0)) | (pd_c == fx.Int32(0)):
                            pd_ph = pd_e * fx.Int32(G)
                            if pd_k != fx.Int32(0):
                                pd_mi = pd_e * fx.Int32(
                                    max_tiles_per_expert
                                ) + pd_k * fx.Int32(G)
                                _spin(
                                    local_addr("expert_tile_map_ready")
                                    + fx.Int64(pd_mi) * fx.Int64(8),
                                    generation,
                                )
                                comm.fence_system_acquire()
                                pd_ph = buffer_ops.buffer_load(
                                    buffer_ops.create_buffer_resource_from_addr(
                                        local_addr("expert_tile_map")
                                    ),
                                    pd_mi,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            pd_add = fx.Int32(GROUP_ROWS) - pd_r
                            pd_old = fx.Int32(
                                comm.atomic_add_system(
                                    local_addr("tile_row_done")
                                    + fx.Int64(pd_ph) * fx.Int64(4),
                                    pd_add,
                                )
                            )
                            if pd_old + pd_add == fx.Int32(GROUP_ROWS):
                                _rq_push(None, pd_ph // fx.Int32(G))
            if const_expr(ready_q):
                # Claim job j and queue entry q=j//GNB. Wait for its generation marker, or
                # exit once global sealing confirms q exceeds the total groups. The mailbox
                # follows GMM1 LDS without overlapping the A ring.
                rq_box = fx.add_offset(fx.recast_iter(fx.Int32, lds_raw), NC_MBOX_DW)
                rq_view = fx.make_view(rq_box, fx.make_layout(1, 1))
                rq_qa = local_addr("gmm1_ready_q")
                rq_ha = local_addr("gmm1_job_head") + fx.Int64(9 * 4)
                rq_go = fx.Int32(1) == fx.Int32(1)
                while rq_go:
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        rq_j = fx.Int32(comm.atomic_add_agent(rq_ha, fx.Int32(1)))
                        rq_q = rq_j // fx.Int32(GNB)
                        rq_valid = rq_q < fx.Int32(GR_N)
                        rq_safe_q = rq_valid.select(rq_q, fx.Int32(0))
                        rq_ea = rq_qa + fx.Int64(rq_safe_q) * fx.Int64(8)
                        # 0: waiting; 1: entry published; 3: global sealing observed (recheck after acquire).
                        rq_v = _rq_entry(rq_ea, rq_valid)
                        rq_e = fx.Int64(
                            comm.load_i64_global_system_relaxed(
                                local_addr("h1_queue_eos")
                            )
                        )
                        rq_st = ((rq_v >> fx.Int64(32)) == fx.Int64(generation)).select(
                            fx.Int32(1),
                            (rq_e >= fx.Int64(generation)).select(
                                fx.Int32(3), fx.Int32(0)
                            ),
                        )
                        while rq_st == fx.Int32(0):
                            _mlir_rocdl.s_sleep(16)
                            rq_v = _rq_entry(rq_ea, rq_valid)
                            rq_e = fx.Int64(
                                comm.load_i64_global_system_relaxed(
                                    local_addr("h1_queue_eos")
                                )
                            )
                            rq_st = (
                                (rq_v >> fx.Int64(32)) == fx.Int64(generation)
                            ).select(
                                fx.Int32(1),
                                (rq_e >= fx.Int64(generation)).select(
                                    fx.Int32(3), fx.Int32(0)
                                ),
                            )
                        # Only EOS-based termination needs the final allocation
                        # count here. A ready entry is generation tagged; the
                        # payload/metadata acquire remains below before GMM1.
                        rq_in = rq_st == fx.Int32(1)
                        if rq_st == fx.Int32(3):
                            comm.fence_system_acquire()
                            rq_tot = fx.Int32(
                                _pl32(local_addr("tile_alloc"))
                            ) // fx.Int32(G)
                            rq_in = rq_q < rq_tot
                        # A successful generation-tagged read already owns this
                        # entry. Only EOS needs the final allocation count; an
                        # in-range EOS claim still waits for queue publication.
                        rq_w = (rq_v >> fx.Int64(32)) != fx.Int64(generation)
                        rq_w = rq_w & rq_in
                        while rq_w:
                            _mlir_rocdl.s_sleep(16)
                            rq_v = _rq_entry(rq_ea, rq_valid)
                            rq_w = (rq_v >> fx.Int64(32)) != fx.Int64(generation)
                        comm.fence_system_acquire()
                        fx.ptr_store(
                            Vec.from_elements(
                                [
                                    rq_in.select(
                                        fx.Int32(rq_v & fx.Int64(0xFFFFFFFF))
                                        * fx.Int32(GNB)
                                        + rq_j
                                        - rq_q * fx.Int32(GNB),
                                        fx.Int32(-1),
                                    )
                                ],
                                fx.Int32,
                            ),
                            rq_box,
                        )
                    gpu.barrier()
                    rq_job = fx.Int32(
                        rocdl.readfirstlane(T.i32, fx.Int32(Vec(rq_view.load())[0]))
                    )
                    rq_has = rq_job >= fx.Int32(0)
                    if rq_has:
                        _run_gemm1_job(rq_job)
                    rq_go = rq_has
            if tx == fx.Int32(0):
                # In the fused path, 248 CTAs wait here after fanout for global sealing.
                # Back off polling to avoid competing with active fanout and credit CTAs for memory.
                (comm.spin_until_ge_i64_sleep_rx)(
                    local_addr("h1_queue_eos"), generation, 127
                )
            _cta_acquire()

    @flyc.kernel(name=kernel_name, known_block_size=[THREADS, 1, 1])
    def kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        x_q: fx.Int64,
        input_scale: fx.Int64,
        route_weights: fx.Int64,
        topk_ids: fx.Int64,
        w1q: fx.Int64,
        w1scale: fx.Int64,
        ntokens: fx.Int32,
    ):
        s1_body(
            dev_comm,
            arena_win,
            arena_ptr,
            x_q,
            input_scale,
            route_weights,
            topk_ids,
            w1q,
            w1scale,
            ntokens,
        )

    @flyc.jit
    def launch_stage1(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        x_q: fx.Int64,
        input_scale: fx.Int64,
        route_weights: fx.Int64,
        topk_ids: fx.Int64,
        w1q: fx.Int64,
        w1scale: fx.Int64,
        ntokens: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            dev_comm,
            arena_win,
            arena_ptr,
            x_q,
            input_scale,
            route_weights,
            topk_ids,
            w1q,
            w1scale,
            ntokens,
            value_attrs={
                # With 256 four-wave CTAs, one per CU, the =2 hint caps VGPRs at 256 and causes
                # scratch spills in GMM1 (7 for K3 TPR128 and 29 for DSV4 TPR256).
                "rocdl.waves_per_eu": WAVES // 4,
                "rocdl.flat_work_group_size": f"{THREADS},{THREADS}",
            },
        ).launch(
            grid=(worker_blocks, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch_stage1.kernel_name = kernel_name
    launch_stage1.body = s1_body
    launch_stage1.lds_bytes = int(lds_alloc_bytes)
    launch_stage1.threads = THREADS
    launch_stage1.waves = WAVES
    launch_stage1.grid = int(worker_blocks)
    # Single fused kernel: where GEMM2 finds the tables-final gate and the
    # per-group h1 completion words (window byte offsets, parity-0 + plane).
    launch_stage1.group_sync = (
        {
            "eos_off": off("h1_queue_eos"),
            "eos_plane": plane("h1_queue_eos"),
            "sync_off": off("h1_grp_sync") + 4 * GR_N,
            "sync_plane": plane("h1_grp_sync"),
            "tile_group": G,
        }
        if group_ready
        else None
    )
    launch_stage1.epoch_off = int(off("epoch_gate"))
    launch_stage1.buffer_regions = buffer_regions
    launch_stage1.group_sync_tail = 4 * GR_N
    return launch_stage1


__all__ = ["compile_stage1"]
