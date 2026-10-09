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

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from . import comm_ops
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_workgroup as _atomic_add_wg,
)
from .gemm_common import k_tiles_total_for
# gemm1.py 的改动不进本文件 kernel 的 flydsl 缓存 key;改 gemm1 时同步改这行强制重编。
# gemm1 rev: megamoev2-style wait_lds_barrier pipeline, nofence barrier v1, next_claim v1
from .gemm1 import (
    _bm_constants,
    _gemm1_body,
)

# rail GDA 原语:MORI 公开绑定给不了 team/optFlags,见 gda_rail.py
#
# checked = 带 MORI device assert;release = assert 编译掉。kernel2
# (stage2_node_combine.py)当初就是因为 rail 关键路径才换成 release,而 stage1
# 一直用 checked。stage1 每代发 176 个 WQE(88 put + 88 put_value),kernel2 只发
# ~32 个,所以同样的 per-WQE assert 成本在 stage1 上被放大 5.5 倍。
from .gda_rail import checked as _rail, ccqe_enabled as _rail_ccqe
from .stage1_abi import Stage1ArenaLayout


BM = 32
BN = 256
BK = 256
THREADS = 256
WAVES = THREADS // 64
CCO_TICKET = 0




def _plane_bytes(layout: Stage1ArenaLayout, name: str) -> int:
    region = layout.region(name)
    if not region.shape or region.shape[0] != layout.parity_depth:
        raise ValueError(f"{name} is not parity indexed")
    return region.nbytes // layout.parity_depth


# 进程内编译缓存:stage1 kernel 名超过文件名长度上限,进不了 flydsl 的磁盘缓存,
# 每构造一次算子就要重编一次(大 TPR 约 25~30 秒)。launcher 只闭包编译期常量,
# 运行期地址全部走 launch 参数,所以同样的编译参数可以直接复用。
_COMPILED: dict = {}


def compile_megamoe_tile_ep16_stage1(layout: Stage1ArenaLayout, stage2_layout, **kwargs):
    """Compile (or reuse) one EP16 Stage-1 persistent-kernel shape specialization."""
    key = (repr(layout), repr(stage2_layout), _rail_ccqe(), tuple(sorted(kwargs.items())))
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
    worker_blocks: int = 512,
    waves_per_eu_hint: int = 2,
    fanout_shards: int = 1,
    compute_first: int = -1,
    split_local: bool = False,
    gmm1_bn: int = 0,
    activation: str = "silu",
    device_generation: bool = False,
    swiglu_limit: float | None = None,
):
    """Compile one EP16 Stage-1 persistent-kernel shape specialization.

    ``arena_ptr`` is the base of the one registered two-kernel window.
    ``stage2_window_offset`` locates the logical :class:`Stage2ArenaLayout`
    inside that same window.  The launcher performs exactly one GPU launch.
    """

    # split_local 两条 GMM1 路径各自的配套开关(见 mega_moe_tile_a4w4._compile_stage1)。
    split_local = bool(split_local)
    early_local_gmm = split_local
    fan2_shards = 16 if split_local else 0
    lazy_pad = split_local
    extra_consumers = split_local
    seal_fast = not split_local
    static_g0 = not split_local

    if not isinstance(layout, Stage1ArenaLayout):
        raise TypeError("layout must be Stage1ArenaLayout")
    HIDDEN = int(layout.hidden)
    INTER = int(layout.inter)
    EXPERTS = int(layout.experts)
    WORLD = int(layout.world_size)
    GPUS_PER_NODE = int(layout.gpus_per_node)
    LOCAL_EXPERTS = int(layout.local_experts)
    TOPK = int(layout.topk)
    BM = int(layout.block_m)
    # 成对(成组)认领:arena 的 tile 仍是 BM 行(Stage2 / MegaMoEv2 的 SBM
    # 依赖这个粒度),但 tile_alloc 一次固定分配 G 个,于是同一个 expert 的
    # G 个 tile 物理相邻,GMM1 可以按 BM*G 行做一个 m_block。
    if layout.dispatch_plan:
        raise ValueError("Stage-1 does not implement the dispatch-plan arena")
    G = int(layout.tile_group)
    # GMM1 以一个认领组(G 个 tile)为一个 m_block。
    GBM = BM * G
    GROUP_ROWS = BM * G
    BN = int(layout.block_n)
    if WORLD != 16 or GPUS_PER_NODE != 8:
        raise ValueError("Stage-1 transport requires EP16 on two 8-GPU nodes")
    if HIDDEN < 1024 or HIDDEN % 512:
        raise ValueError("Stage-1 hidden must be >= 1024 and divisible by 512")
    if HIDDEN > 8192:
        raise ValueError("Stage-1 hidden must be <= 8192: one thread copies each 1x32 group")
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
    if swiglu_limit is not None and (activation != "silu" or not float(swiglu_limit) > 0.0):
        raise ValueError("Stage-1 swiglu_limit applies to silu only and must be positive")
    # agent: token capacity来自arena specialization；producer CTA池保持固定，
    # 避免大batch把resident grid扩张到硬件无法同时驻留的规模。
    MAX_TOKENS = int(layout.max_tokens)
    COMM_CTAS = GPUS_PER_NODE
    PRODUCER_FIRST = COMM_CTAS
    # 量化池原本写死 128:512 个 token 由 128 个 CTA 各做 4 个,另外 128 个
    # CTA 在 fanout 之前完全空转。staging 段实测占 stage1 的 38%
    # ([[stage1 四段分解]]),所以池子大小是个直接旋钮。0 = 保持原默认。
    PRODUCER_CTAS = min(128, MAX_TOKENS)
    ESSENTIAL_CTAS = PRODUCER_FIRST + PRODUCER_CTAS
    # GMM1 消费者的起始 ticket。默认 ESSENTIAL_CTAS(=136),即 8 个 comm 和
    # 128 个 producer 干完自己的活就退出,GMM1 阶段 136/256 个 CTA 空转。
    # 实测把它降到 0(全员参与)GMM1 段确实快了 30%(797->555us),但总时间
    # 大幅回归:comm CTA 还欠着 rail 的 credit/reclaim,而 finisher 要等
    # h1_compute_done 收满,于是 GMM1 的收尾被 rail 拆链拖住。
    # 所以它是个**边界**而不是开关:8 = 放 producer 进来但不放 comm。

    COMPUTE_FIRST = (
        ESSENTIAL_CTAS if compute_first < 0 else min(int(compute_first), ESSENTIAL_CTAS)
    )
    INVALID_SOURCE = WORLD * MAX_TOKENS
    if not 1 <= MAX_TOKENS <= 4096:
        raise ValueError("Stage-1 max_tokens must be in [1, 4096]")
    if fanout_shards < 1 or GPUS_PER_NODE * fanout_shards > worker_blocks:
        # 原来只在 comm+producer 这 ESSENTIAL_CTAS 个 CTA 内铺开,理由是
        # producer 做完量化(实测 ~10us)就闲着,借用它们不减少 GMM1 消费者。
        # 但 fanout 是**跨 GPU 往返延迟**受限的(每条 record 约 5 次
        # atomic_add_system/spin 串行往返),并发度就是吞吐,而 ticket >=
        # ESSENTIAL_CTAS 的消费者在第一个 tile 发布前本来就只是在自旋。
        # 放开到 worker_blocks 后 shards>17 会让消费者先做完自己那片 fanout
        # 再入池 —— 是否划算由实测决定,所以这里只放开上界,不改默认。
        raise ValueError(
            "fanout_shards must satisfy 1 <= GPUS_PER_NODE*shards <= "
            f"{worker_blocks}"
        )
    if not 0 <= int(rank) < WORLD:
        raise ValueError("rank must be in [0, 16)")
    if worker_blocks < ESSENTIAL_CTAS:
        raise ValueError(
            f"worker_blocks must be >= {ESSENTIAL_CTAS} so all progress roles are resident"
        )
    if worker_blocks == ESSENTIAL_CTAS:
        # Producers retire without rejoining in the chunked full path. With
        # zero GEMM consumers the finisher would wait forever for h1_compute_done.
        raise ValueError("full Stage-1 requires at least one GEMM consumer after progress roles")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")

    rank = int(rank)
    local_rank = rank % GPUS_PER_NODE
    node = rank // GPUS_PER_NODE
    remote_node = 1 - node
    remote_source_rank = remote_node * GPUS_PER_NODE + local_rank
    stage2_window_offset = int(stage2_window_offset)
    # rail_soa 的 SoA 排布(发送端 dispatch_staging[p] 与接收端 remote_dispatch_rx[p] 共用)。
    from .rail_record_quant import rail_record_layout as _rail_record_layout

    REC_BYTES, REC_Q, REC_S, REC_I, REC_W = _rail_record_layout(HIDDEN, TOPK)
    RAIL_QPS = 4
    # fanout 按 (dest, shard) 分到 CTA;T0(CCO 收发 + credit + seal)不兼做 dest0 的分片,
    # 所以至少要 2 个分片。
    if fanout_shards < 2:
        raise ValueError("fanout_shards must be >= 2 (T0 does not take a fanout shard)")
    if stage2_window_offset < 0 or stage2_window_offset % 4096:
        raise ValueError("stage2_window_offset must be non-negative and 4096-byte aligned")

    wire = layout.wire
    record_bytes = wire.record_bytes
    # GMM1 的两条路径由 split_local 选:
    # split:本地来源行(fan1)和远端来源行(fan2)各自计数、各自成组,sealer 分两段封尾:
    #   段 1 在 8 个 local_eos 后封本地组(h1 段基址 0),段 2 在全部 comm_eos 后封远端组。
    #   early_local_gmm:段 1 封好本地组即发布 h1_local_eos + 物理组表
    #   gmm1_group_list[本地组..|远端组..];消费者先用本地头领本地 job(与 rail/fan2/段 2 重叠),
    #   过全局门后再用远端头领远端 job。每个消费者退出时 h1_compute_done +1。
    #   fan2_shards:每个 dest 只有 fanout_shard < fan2_shards 的 fan CTA 做 fan2(按新步长
    #   分 token),其余 fan CTA 做完 fan1 直接进 GMM1 本地段,把等 rail 的空档让给本地组。
    #   lazy_pad:段 1 不补本地组的尾行就发布 h1_local_eos,改由段 2 的 sealer 补。GMM1 只经
    #   tile_row_input 取 A,且是带界 buffer load;补齐前那些行里是上一代/初值的合法行号,
    #   算出来的 h1 行只会被 stage2 用 INVALID_SOURCE/权重 0 丢掉,而这两项在 stage1_done 前补齐。
    #   extra_consumers:ticket 1..COMPUTE_FIRST-1 里除 finisher 以外的 CTA 做完 fan 也进 GMM1。
    #   consumer_index 仍是 [0, 消费者数) 上的双射:额外的排在原来那段之后。
    # 不 split:
    #   static_g0:每个 expert 的第 0 组固定放在物理组 e(tile [e*G, e*G+G)),由各 GPU 的
    #   初始化者在发布 launch_ready 前写好 map/tile_expert/map_ready,tile_alloc 从 LE*G 起;只有
    #   溢出组(组号>=1)才远端认领。0 行的 expert 也占满一组(全 pad、source 无效)。
    #   seal_fast:并行封尾。
    # 两条路径共用:GMM1 按物理组写 h1,封尾额外写逆置换 tile_src_of_dst,stage2 按它间接读 A;
    # rail 发帖 CTA(ticket _POST_LO.._POST_HI)先发帖、不分 fan token。
    F2S = int(fan2_shards)
    if F2S and F2S > fanout_shards - 1:
        raise ValueError("fan2_shards must be <= fanout_shards - 1")
    _FIN_TICKET = local_rank
    _XC = (
        [t for t in range(1, COMPUTE_FIRST) if t != _FIN_TICKET]
        if extra_consumers
        else []
    )
    N_CONSUMERS = worker_blocks - COMPUTE_FIRST + len(_XC)
    # QP q 由 ticket q+1 发帖(T0 不发)。
    _POST_LO, _POST_HI = 1, RAIL_QPS + 1
    # 段 1 放在哪个 fan CTA(ticket = 8*_SEG1_SHARD + local_rank)。dest0 的分片号整体减 1、
    # 发帖 CTA 不分 token,所以这些 rank 要再往后挪一格才在 fan2 池外。
    _SEG1_SHARD = 1
    if F2S:
        _s1 = F2S + (1 if local_rank == 0 or _POST_LO <= local_rank < _POST_HI else 0)
        if _s1 < fanout_shards:
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
    # GMM1 自己的 N 块宽(与 arena 的 block_n 解耦;h1 按行存,N 块只决定哪个 CTA 写哪些列)。
    # 128 = 小算子 t128x128 的 N;job 数按 GNB 算。
    GBN = int(gmm1_bn) if gmm1_bn else BN
    if GBN not in (128, 256):
        raise ValueError("gmm1_bn must be 128 or 256")
    GNB = (2 * INTER) // GBN

    kh_tile = BK // 2
    k_tiles_total = k_tiles_total_for(HIDDEN, BK)
    # LDS 按 GMM1 实际的 M tiling 算,不是 arena 的 tile 粒度:G>1 时
    # _gemm1_body 以 BM*G 行跑,累加器要 BM*G*BN*4 字节。按 BM 算会
    # 静默写越界(G=2 实测 relL2=1.55)。
    _, _, _, lds_bytes = _bm_constants(GBM, BN, kh_tile, k_tiles_total)
    # next_claim(只在 early_local_gmm 下):GMM1 体内提前领下一个 job,
    # 映射好的 job 号经 LDS 信箱(GMM1 LDS 之后多分配的 16B,dword0=job,1=本段上界,2=本段偏移)
    # 交回领取循环。lead = 在倒数第几个 K 步开头发领位原子。
    gmm1_next_claim = bool(early_local_gmm)
    gmm1_next_claim_lead = 2
    if gmm1_next_claim:
        assert lds_bytes % 16 == 0
        assert gmm1_next_claim_lead >= 1
    NC_MBOX_DW = lds_bytes // 4  # 信箱在全部 GMM1 LDS(A 环/scale/累加器)之后
    lds_alloc_bytes = lds_bytes + (16 if gmm1_next_claim else 0)

    # Compile-time region helpers.  CCO offsets are relative to the physical
    # window; local addresses add arena_ptr at runtime.
    def off(name):
        return int(layout.region(name).offset)

    def plane(name):
        return _plane_bytes(layout, name)

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
    if stage2_layout.timeline_history_depth:
        raise ValueError("Stage-1 does not write the Stage-2 timeline history")
    if s2_parity_depth != 2:
        raise ValueError("Stage-2 metadata must be double buffered")

    def s2_off(name):
        return int(stage2_layout.region(name).offset)

    def s2_plane(name):
        region = stage2_layout.region(name)
        if not region.shape or region.shape[0] != s2_parity_depth:
            raise ValueError(f"Stage-2 region {name} is not parity indexed")
        return region.nbytes // s2_parity_depth

    # Resolve the shared metadata contract at compile time, not in the hot path.
    for required in (
        "node_dest_rank_mask",
        "source_token_count",
        "node_expected",
        "stage1_done",
    ):
        stage2_layout.region(required)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_alloc_bytes, 16]

    # 名字只带随 shape / tune 表变化的编译期参数;写死的开关不进名字。
    # split_local 一项代表它那一整套配套开关(early local GMM / fan2 / lazy pad /
    # extra consumers / next claim,或 seal fast / static g0)。
    kernel_name = (
        f"megamoe_tile_ep16_stage1_{activation}"
        + ("" if swiglu_limit is None else f"_swl{float(swiglu_limit):g}".replace(".", "p"))
        + f"_r{rank}_h{HIDDEN}_i{INTER}_e{EXPERTS}_k{TOPK}"
        f"_mt{MAX_TOKENS}_rpc{layout.max_routes_per_token_per_rank}_wb{worker_blocks}"
        + ("_devgen" if device_generation else "")
        + (f"_cf{COMPUTE_FIRST}" if compute_first >= 0 else "")
        + (f"_bm{BM}" if BM != 32 else "")
        + (f"_tg{G}" if G != 1 else "")
        + (f"_gbn{int(gmm1_bn)}" if gmm1_bn and int(gmm1_bn) != BN else "")
        + (f"_fos{fanout_shards}" if fanout_shards != 1 else "")
        + ("_slg" if split_local else "_nsl")
        # rail 的 CQ 轮询按 CCQE 模式编译
        + ("_ccqe" if _rail_ccqe() else "")
    )

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
        generation: fx.Int64,
    ):
        lds_raw = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        arena_window = cco.Window(arena_win)
        # cco_lsa_ptr(w, peer, off) = winBase + peer*(stride4G<<32) + off:
        # 每次调用都要从 window 结构里重新读两个字段(ATT 里 ~6.5% 的 route 时间)。
        # 入口算一次基址和对端步长,之后对端地址只剩整数运算。
        arena_lsa_base = fx.Int64(arena_window.lsa_ptr(fx.Int32(0), fx.Int64(0)))
        arena_lsa_stride = (
            fx.Int64(arena_window.lsa_ptr(fx.Int32(1), fx.Int64(0))) - arena_lsa_base
        )
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
                comm_ops.atomic_add_agent(
                    arena_ptr + fx.Int64(off("entry_count")), fx.Int64(1)
                )
            )
            fx.ptr_store(Vec.from_elements([ticket_lane], fx.Int64), ticket_ptr)
        gpu.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        if const_expr(device_generation):
            # cuda-graph 捕获要求 device_generation=True(forward() 里硬性检查),
            # 于是 generation 是在 device 上从到达票推出来的:entry_count 每个
            # CTA 加一,每次 forward 恰好一次 launch、worker_blocks 个 CTA。
            generation = (
                ticket64
                // fx.Int64(worker_blocks)
                + fx.Int64(1)
            )
        parity = fx.Int64(generation & fx.Int64(1))
        control_generation = (
            generation
        )

        def local_addr(name):
            return arena_ptr + fx.Int64(off(name)) + parity * fx.Int64(plane(name))

        def window_off(name):
            return fx.Int64(off(name)) + parity * fx.Int64(plane(name))

        publish_plane_slots = bool(
            getattr(stage2_layout, "include_plane_slots", False)
        )

        def stage2_addr(name):
            return (
                arena_ptr
                + fx.Int64(stage2_window_offset + s2_off(name))
                + parity * fx.Int64(s2_plane(name))
            )

        error_addr = arena_ptr + fx.Int64(off("error_count"))

        # Arrival tickets, rather than block IDs, ensure every progress role is
        # among the first resident CTAs on an oversubscribed persistent grid.
        ticket = fx.Int32(ticket64 % fx.Int64(worker_blocks))
        is_initializer = ticket == fx.Int32(0)

        is_cco = ticket == fx.Int32(CCO_TICKET)
        is_comm = ticket < fx.Int32(COMM_CTAS)

        # spin_relaxed:轮询用 relaxed load,循环外补一次 acquire(调用方负责)。
        _pl32 = comm_ops.load_i32_global_system_relaxed
        _spin = comm_ops.spin_until_ge_i64_system_rx

        def _rail_post(post_qp, with_credit):
            """rail_soa:一个 wave 为 QP post_qp 发它那段 token 的 record。

            整段一个 PUT + 本 QP 负责的 ready 字 + 一次门铃,不等完成;请求存进
            remote_chunk_request[post_qp] 由发起方稍后回收。with_credit:先等本 QP
            在 g-2(同 parity)的 credit(T0 自己在外面统一等过)。
            """
            src_base = fx.Int64(off("dispatch_staging"))
            comm_ops.fence_system_release()
            if const_expr(with_credit):
                if lane == fx.Int32(0):
                    for cchunk in range(
                        fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                    ):
                        _spin(
                            local_addr("remote_chunk_credit")
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
                    arena_win,
                    window_off("remote_dispatch_rx")
                    + fx.Int64(r0 * fx.Int32(REC_BYTES)),
                    arena_win,
                    src_base + fx.Int64(r0 * fx.Int32(REC_BYTES)),
                    fx.Int64(nr * fx.Int32(REC_BYTES)),
                    aggregate=True,
                )
            # 消费者只等 chunk 0 的 num_qp 个 ready 字;每个 QP 至少置一个且在
            # 自己的数据之后(同 QP 内有序)⇒ 全部数据已到。
            for w in range_constexpr(layout.num_qp):
                if post_qp == fx.Int32(w % RAIL_QPS):
                    _rail.put_value(
                        dev_comm,
                        post_qp,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_chunk_ready") + fx.Int64(w * 8),
                        generation,
                        aggregate=True,
                    )
            request = _rail.flush_async(dev_comm, post_qp, fx.Int32(remote_node))
            if lane == fx.Int32(0):
                comm_ops.store_i64_global_system(
                    local_addr("remote_chunk_request")
                    + fx.Int64(post_qp) * fx.Int64(8),
                    request,
                )

        # 角色到达计数。kernel 挂死时 host 读不到任何"完成后"的东西,但 CCO
        # window 的 local_ptr 是 host 可读的 —— 所以把计数直接打进 arena,
        # host 在 launch 之后不同步地轮询读,就能看出哪个角色一个 CTA 都没到。

        # Owner initializes only counters/metadata.  Payload and generation
        # arrays are overwrite-before-publish and never need a hot-path memset.
        if is_initializer:
            expert_count = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            queue_heads = buffer_ops.create_buffer_resource_from_addr(
                local_addr("h1_queue_head")
            )
            for item in range(tx, LOCAL_EXPERTS * (2 if split_local else 1), THREADS):
                buffer_ops.buffer_store(fx.Int32(0), expert_count, item)
            consumed = buffer_ops.create_buffer_resource_from_addr(
                local_addr("remote_chunk_consumed")
            )
            if const_expr(fanout_shards > 1):
                shard_done = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("fanout_shard_done")
                )
                if tx < fx.Int32(GPUS_PER_NODE * (2 if split_local else 1)):
                    buffer_ops.buffer_store(fx.Int32(0), shard_done, tx)
            for item in range(
                tx,
                fx.Int32(dispatch_chunks * layout.num_qp),
                fx.Int32(THREADS),
            ):
                buffer_ops.buffer_store(fx.Int32(0), consumed, item)
            if tx < fx.Int32(8):
                buffer_ops.buffer_store(fx.Int32(0), queue_heads, tx * fx.Int32(16))
            if tx == fx.Int32(0):
                for name in (
                    "tile_alloc",
                    "h1_queue_tail",
                    "h1_compute_done",
                    "h1_early_full_tiles",
                    "h1_gmm_started_before_all_comm_eos",
                    "h1_gmm_completed_before_all_comm_eos",
                ):
                    buffer_ops.buffer_store(
                        fx.Int32(0),
                        buffer_ops.create_buffer_resource_from_addr(local_addr(name)),
                        fx.Int32(0),
                    )
                if const_expr(early_local_gmm):
                    _jh_res = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("gmm1_job_head")
                    )
                    buffer_ops.buffer_store(fx.Int32(0), _jh_res, fx.Int32(0))
                    buffer_ops.buffer_store(fx.Int32(0), _jh_res, fx.Int32(16))
                if ntokens != fx.Int32(MAX_TOKENS):
                    comm_ops.atomic_add_system(error_addr, fx.Int32(1))
            if const_expr(static_g0):
                # 静态第 0 组:上面 tx0 刚把 tile_alloc 清零(同线程程序序在前),这里改成 LE*G。
                if tx == fx.Int32(0):
                    buffer_ops.buffer_store(
                        fx.Int32(LOCAL_EXPERTS * G),
                        buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                        fx.Int32(0),
                    )
                sg_map = buffer_ops.create_buffer_resource_from_addr(local_addr("expert_tile_map"))
                sg_te = buffer_ops.create_buffer_resource_from_addr(local_addr("tile_expert"))
                for sg_i in range(tx, fx.Int32(LOCAL_EXPERTS * G), fx.Int32(THREADS)):
                    sg_e = sg_i // fx.Int32(G)
                    sg_m = sg_e * fx.Int32(max_tiles_per_expert) + sg_i - sg_e * fx.Int32(G)
                    buffer_ops.buffer_store(sg_i, sg_map, sg_m)
                    buffer_ops.buffer_store(sg_e, sg_te, sg_i)
                    comm_ops.store_i64_global_relaxed(
                        local_addr("expert_tile_map_ready") + fx.Int64(sg_m) * fx.Int64(8),
                        fx.Int64(generation),
                    )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_system_release()
                # pub_relaxed:一次 fence 放行全部清零,两条门值不必各自再做整 L2 写回。
                _pub_st = (
                    comm_ops.store_i64_global_system_relaxed
                )
                _pub_st(
                    arena_ptr + fx.Int64(off("epoch_gate")),
                    control_generation,
                )
                _pub_st(
                    local_addr("launch_ready"), control_generation
                )
        else:
            if tx == fx.Int32(0):
                _spin(
                    arena_ptr + fx.Int64(off("epoch_gate")),
                    control_generation,
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()

        # ------------------------------------------------------------------
        # 目的端预计算放置 —— 第 1 步:算计数并交换。
        #
        # 计数只依赖 topk_ids,kernel 入口就有,所以整个 plan 可以和量化
        # (实测 ~93us)并行跑完;payload 阶段拿到的就是纯算术偏移,不再需要
        # 每条 route 一次跨设备原子(现状是 expert_count 原子 + tile 认领/自旋)。
        #
        # node 间只有 rail 一条路,所以只让**计数**过一跳 rail:发给同序号
        # 伙伴,由它在自己 node 内散给 8 个目的端。行基址完全不过 rail ——
        # 目的端把它留在自己 arena 里,node 内的推送方和 rail 转发方都直接
        # LSA 读,于是远端源根本不需要知道行基址,省掉了回程那一跳。
        #
        # 计数不需要清零:每个源只写自己那一行,再把 plan_count_ready[src]
        # 置成 generation,目的端等 `== generation`,天然免疫上一代的残留。
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # A bounded producer CTA pool owns tokens in a strided schedule.  At
        # capacity 128 this is identical to the historical one-CTA-per-token
        # mapping; larger capacities reuse the same resident CTA for later
        # tokens instead of increasing the resident grid.
        # ------------------------------------------------------------------
        # split fanout 原先把 token 直接写成 ticket,循环每轮重搬同一个
        # token:512 个只落 128 个,且从 ticket 起算(token 0..7 无人写)。
        # 那是 MORI「128 token / 一 CTA 一 token / 单轮」的形状。

        # ------------------------------------------------------------------
        # Four waves own four QPs.  Each chunk is one aggregate PUT per QP plus
        # a trailing ready value and one flush/doorbell.  The same CTA receives
        # the reciprocal chunk and performs selected-rank proxy fan-out.
        # ------------------------------------------------------------------
        if is_cco:
            qp = wave

            # credit_async:对端对 g-2(同 parity)那批数据的消费确认改到
            # 这里等 —— 只有覆盖对端同一块 remote_dispatch_rx 之前才需要。
            # 本 kernel 末尾不再等对端 credit(那要等对端整个 fan2 做完)。
            if lane == fx.Int32(0):
                for cchunk in range(
                    fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                ):
                    _spin(
                        local_addr("remote_chunk_credit")
                        + fx.Int64(cchunk * fx.Int32(layout.num_qp) + qp)
                        * fx.Int64(8),
                        generation - fx.Int64(2),
                    )
            # rail_soa:发送源是 k1 之前的 rail_record_quant 直接写进注册窗口
            # (dispatch_staging 的 parity-0 平面)的 token 连续 record
            # [q|scale|ids|weights|pad],REC 字节/token。每个 QP 负责一段连续
            # token,整段一个 PUT + 一个 ready 字 + 一次门铃,不等完成(请求在
            # 末尾 credit 段回收)。WQE 的投递成本 ~15µs 与大小无关(TS3/TS5),
            # 所以不再按字段拆 PUT。发送源单缓冲:本次 PUT 在本 kernel 末尾等掉,
            # 下一次量化在其后。
            # QP 由 ticket 1..RAIL_QPS 的发帖 CTA 发(见 fanout 入口),T0 不发。
            comm_ops.fence_system_release()

            # 全部发完之后一次性等齐,acquire fence 在批边界做一次。
            # rail_soa 只置 chunk 0 的 ready 字。
            if lane == fx.Int32(0):
                _spin(
                    local_addr("remote_chunk_ready") + fx.Int64(qp) * fx.Int64(8),
                    generation,
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            gpu.barrier()
            remote_masks = buffer_ops.create_buffer_resource_from_addr(
                stage2_addr("node_dest_rank_mask")
            )
            rx_all = buffer_ops.create_buffer_resource_from_addr(
                local_addr("remote_dispatch_rx")
            )
            # agent: CCO CTA用256线程跨步解析全部远端token；capacity大于
            # THREADS时不能只处理首批256条record。
            for token in range(tx, ntokens, fx.Int32(THREADS)):
                remote_record_available = fx.Int32(1)
                remote_mask = fx.Int32(0)
                remote_slot_mask = fx.Int32(0)
                if remote_record_available != fx.Int32(0):
                    # token 每 lane 不同:描述符必须是标量,按 token 建描述符会被
                    # 编译成 readfirstlane 瀑布循环(ATT5:每次 load 最多 64 轮,
                    # T0 在这里耗 ~230k 周期)。整个 rx 区一个描述符 + lane 偏移。
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
                            (owner // fx.Int32(GPUS_PER_NODE))
                            == fx.Int32(node)
                        )
                        remote_mask = on_node.select(
                            remote_mask
                            | (
                                fx.Int32(1)
                                << (owner % fx.Int32(GPUS_PER_NODE))
                            ),
                            remote_mask,
                        )
                        if const_expr(publish_plane_slots):
                            remote_slot_mask = on_node.select(
                                remote_slot_mask
                                | (fx.Int32(1) << fx.Int32(slot)),
                                remote_slot_mask,
                            )
                buffer_ops.buffer_store(
                    remote_mask,
                    remote_masks,
                    fx.Int32(remote_node * MAX_TOKENS) + token,
                )
                if const_expr(publish_plane_slots):
                    buffer_ops.buffer_store(
                        remote_slot_mask,
                        buffer_ops.create_buffer_resource_from_addr(
                            stage2_addr("node_dest_slot_mask")
                        ),
                        fx.Int32(remote_node * MAX_TOKENS) + token,
                    )

        # ------------------------------------------------------------------
        # Split fanout CTAs cover the eight node-local destinations. Each writes
        # both its local source rank and the aligned remote source rank directly
        # into destination expert tiles; there is no rank inbox/sort.
        # ------------------------------------------------------------------
        is_finisher = is_comm & (
            ticket == fx.Int32(local_rank)
        )

        # 两个 finisher 职责必须分开:封尾 partial tile 属于 dispatch 那一半,
        # 而「等 h1_compute_done 收满再发 stage1_done」属于 GMM1 那一半。
        is_sealer = is_finisher
        is_publisher = is_finisher

        def _peer_addr(dest, name):
            return (
                arena_lsa_base
                + fx.Int64(dest) * arena_lsa_stride
                + fx.Int64(window_off(name))
            )

        def _alloc_tiles_for(count):
            """一个 expert 实际被分配的 tile 数(向上取整到 G 的倍数)。"""
            n = (count + fx.Int32(BM - 1)) // fx.Int32(BM)
            if const_expr(G > 1):
                n = ((n + fx.Int32(G - 1)) // fx.Int32(G)) * fx.Int32(G)
            if const_expr(static_g0):
                n = (n < fx.Int32(G)).select(fx.Int32(G), n)
            return n

        def _is_group_head(row_in_tile, row_slot):
            """这一行是否是某个 G-tile 组的第一行(由它去认领整组)。"""
            if const_expr(G == 1):
                return row_in_tile == fx.Int32(0)
            return (row_slot % fx.Int32(GROUP_ROWS)) == fx.Int32(0)

        def _claim_tile_group(dest, map_index, local_expert, map_ready):
            """一次认领 G 个物理相邻 tile,发布 G 个 map 槽,返回组首 physical。

            tile_alloc 每次固定加 G,所以返回值必然是 G 的倍数 —— 这正是
            GMM1 能把 [base, base+G) 当作一个 BM*G 行 m_block 的前提。
            组内后续 logical tile 的首行会落到 else 分支上自旋等 map_ready,
            和原来「非首行等首行」的路径完全一致。
            """
            base = fx.Int32(
                comm_ops.atomic_add_system(
                    _peer_addr(dest, "tile_alloc"), fx.Int32(G)
                )
            )
            if base >= fx.Int32(max_route_tiles - G + 1):
                comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                base = fx.Int32(0)
            map_res = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "expert_tile_map")
            )
            expert_res = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "tile_expert")
            )
            # tile_row_done/tile_row_base 没有设备端读者,不写。
            for j in range_constexpr(G):
                phys = base + fx.Int32(j)
                buffer_ops.buffer_store(phys, map_res, map_index + fx.Int32(j))
                buffer_ops.buffer_store(local_expert, expert_res, phys)
            comm_ops.fence_system_release()
            for j in range_constexpr(G):
                # claim_relaxed:上面一次 fence 已放行 map/tile_expert,G 条 release store 各带一次整 L2 写回。
                (comm_ops.store_i64_global_system_relaxed)(
                    map_ready + fx.Int64(j * 8), generation
                )
            return base

        # route_batch:一条 record 的全部 route 并行做。wave0 的 lane j 负责
        # topk slot j:expert_count 原子、组头认领、非组头等 map_ready、写元数据
        # 都在各自 lane 上并发;scale 每线程读一次再散写到每条 route。
        # 每 (record,dest) 5 个 barrier,原来是 2+5*routes。
        # 死锁不变量:本 record 所有组头先认领完(A1),barrier 之后才有 lane
        # 去等外部 map_ready(A2);认领本身不等任何东西。
        _RB_ROW = 16
        _RB_SLOT = _RB_ROW + TOPK
        _RB_LE = _RB_ROW + 2 * TOPK
        _RB_PHYS = _RB_ROW + 3 * TOPK

        def _local_dest_slot_mask(token, dest, ids_rsrc=None, ids_base=None):
            """由 ids 数组直接算本 token 落到 dest 的 slot 位图(fan1_direct / rail_soa)。"""
            if ids_rsrc is None:
                ids_rsrc = buffer_ops.create_buffer_resource_from_addr(topk_ids)
                ids_base = token * fx.Int32(TOPK)
            scratch = fx.recast_iter(fx.Int32, lds_raw)
            if tx == fx.Int32(0):
                lo = (fx.Int32(node * GPUS_PER_NODE) + dest) * fx.Int32(
                    LOCAL_EXPERTS
                )
                slots = fx.Int32(0)
                for slot in range_constexpr(TOPK):
                    expert = buffer_ops.buffer_load(
                        ids_rsrc,
                        ids_base + fx.Int32(slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    hit = (expert >= lo) & (expert < lo + fx.Int32(LOCAL_EXPERTS))
                    slots = hit.select(slots | fx.Int32(1 << slot), slots)
                fx.ptr_store(Vec.from_elements([slots], fx.Int32), scratch)
            gpu.barrier()
            return Vec(fx.make_view(scratch, fx.make_layout(1, 1)).load())[0]

        def _dispatch_local_direct(token, dest, source_index):
            # x_q 指向 rail record 基址(量化直接写成 record)。
            rec_rs = buffer_ops.create_buffer_resource_from_addr(x_q)
            q_view = (rec_rs, token * fx.Int32(REC_BYTES // 4) + fx.Int32(REC_Q // 4))
            s_view = (rec_rs, token * fx.Int32(REC_BYTES) + fx.Int32(REC_S))
            _dispatch_view_batched(
                dest,
                source_index,
                _local_dest_slot_mask(token, dest),
                (
                    q_view[0],
                    q_view[1],
                    s_view[0],
                    s_view[1],
                    buffer_ops.create_buffer_resource_from_addr(topk_ids),
                    token * fx.Int32(TOPK),
                    buffer_ops.create_buffer_resource_from_addr(route_weights),
                    token * fx.Int32(TOPK),
                ),
            )

        def _dispatch_remote_soa(token, dest, source_index):
            # rail_soa:remote_dispatch_rx[parity] 与发送端同一 record 排布。
            rx = buffer_ops.create_buffer_resource_from_addr(
                local_addr("remote_dispatch_rx")
            )
            rec_dw = token * fx.Int32(REC_BYTES // 4)
            ids_base = rec_dw + fx.Int32(REC_I // 4)
            _dispatch_view_batched(
                dest,
                source_index,
                _local_dest_slot_mask(token, dest, rx, ids_base),
                (
                    rx, rec_dw + fx.Int32(REC_Q // 4),
                    rx, token * fx.Int32(REC_BYTES) + fx.Int32(REC_S),
                    rx, ids_base,
                    rx, rec_dw + fx.Int32(REC_W // 4),
                ),
            )

        def _dispatch_view_batched(dest, source_index, route_slots, view):
            # view = (q 描述符, q 起始 dword, scale 描述符, scale 起始字节,
            #         ids 描述符, ids 起始下标, weights 描述符, weights 起始下标)
            q_rs, q_dw0, s_rs, s_b0, id_rs, id_i0, w_rs, w_i0 = view
            if route_slots != fx.Int32(0):
                dst_payload = buffer_ops.create_buffer_resource_from_addr(
                    _peer_addr(dest, "grouped_input_q")
                    + fx.Int64(source_index) * fx.Int64(wire.payload_bytes),
                    num_records_bytes=wire.payload_bytes,
                )
                for dword in range(
                    tx * fx.Int32(4),
                    payload_dwords,
                    fx.Int32(THREADS * 4),
                ):
                    value = buffer_ops.buffer_load(
                        q_rs, q_dw0 + dword, vec_width=4, dtype=T.i32
                    )
                    buffer_ops.buffer_store(value, dst_payload, dword)
                rb_lds = fx.recast_iter(fx.Int32, lds_raw)

                def _rb_lane(base):
                    return fx.add_offset(
                        rb_lds, (fx.Int32(base) + lane) * fx.Int32(4)
                    )

                def _rb_load(ptr):
                    return Vec(
                        fx.make_view(ptr, fx.make_layout(1, 1)).load()
                    )[0]

                # A1:领行 + 组头认领。认领不等任何东西。
                if (wave == fx.Int32(0)) & (lane < fx.Int32(TOPK)):
                    slot_state = fx.Int32(-1)
                    le_state = fx.Int32(0)
                    phys_state = fx.Int32(-1)
                    if ((route_slots >> lane) & fx.Int32(1)) != fx.Int32(0):
                        dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                        expert = buffer_ops.buffer_load(
                            id_rs,
                            id_i0 + lane,
                            vec_width=1,
                            dtype=T.i32,
                        )
                        lo = dest_global * fx.Int32(LOCAL_EXPERTS)
                        hi = lo + fx.Int32(LOCAL_EXPERTS)
                        invalid = (expert < lo) | (expert >= hi)
                        if invalid:
                            comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                        local_expert = invalid.select(fx.Int32(0), expert - lo)
                        row_slot = fx.Int32(
                            comm_ops.atomic_add_system(
                                _peer_addr(dest, "expert_count")
                                + fx.Int64(local_expert) * fx.Int64(4),
                                fx.Int32(1),
                            )
                        )
                        slot_state = row_slot
                        le_state = local_expert
                        a1_rit = row_slot % fx.Int32(BM)
                        a1_map = (
                            local_expert * fx.Int32(max_tiles_per_expert)
                            + row_slot // fx.Int32(BM)
                        )
                        if _is_group_head(a1_rit, row_slot):
                            phys_state = _claim_tile_group(
                                dest,
                                a1_map,
                                local_expert,
                                _peer_addr(dest, "expert_tile_map_ready")
                                + fx.Int64(a1_map) * 8,
                            )
                    fx.ptr_store(
                        Vec.from_elements([slot_state], fx.Int32),
                        _rb_lane(_RB_SLOT),
                    )
                    fx.ptr_store(
                        Vec.from_elements([le_state], fx.Int32),
                        _rb_lane(_RB_LE),
                    )
                    fx.ptr_store(
                        Vec.from_elements([phys_state], fx.Int32),
                        _rb_lane(_RB_PHYS),
                    )
                # 必须是 CTA barrier 分开的两个区域:同一 wave 里「认领」和
                # 「等 map_ready」若写成互补的两个 if,编译器可合并成 if/else
                # 并先跑等待分支;同 token 重复 expert 时就会自等死锁。
                gpu.barrier()
                # A2:非组头等 map_ready;所有 lane 写元数据。
                if (wave == fx.Int32(0)) & (lane < fx.Int32(TOPK)):
                    grouped_row_lane = fx.Int32(-1)
                    a2_slot = _rb_load(_rb_lane(_RB_SLOT))
                    if a2_slot >= fx.Int32(0):
                        a2_le = _rb_load(_rb_lane(_RB_LE))
                        a2_rit = a2_slot % fx.Int32(BM)
                        a2_map = (
                            a2_le * fx.Int32(max_tiles_per_expert)
                            + a2_slot // fx.Int32(BM)
                        )
                        if _rb_load(_rb_lane(_RB_PHYS)) < fx.Int32(0):
                            _spin(
                                _peer_addr(dest, "expert_tile_map_ready")
                                + fx.Int64(a2_map) * 8,
                                generation,
                            )
                            phys_t = buffer_ops.buffer_load(
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "expert_tile_map")
                                ),
                                a2_map,
                                vec_width=1,
                                dtype=T.i32,
                            )
                            fx.ptr_store(
                                Vec.from_elements([phys_t], fx.Int32),
                                _rb_lane(_RB_PHYS),
                            )
                        grouped_row_lane = (
                            _rb_load(_rb_lane(_RB_PHYS)) * fx.Int32(BM) + a2_rit
                        )
                        weight = buffer_ops.buffer_load(
                            w_rs,
                            w_i0 + lane,
                            vec_width=1,
                            dtype=T.f32,
                        )
                        buffer_ops.buffer_store(
                            source_index,
                            buffer_ops.create_buffer_resource_from_addr(
                                _peer_addr(dest, "tile_row_input")
                            ),
                            grouped_row_lane,
                        )
                        buffer_ops.buffer_store(
                            source_index | (lane << fx.Int32(24)),
                            buffer_ops.create_buffer_resource_from_addr(
                                _peer_addr(dest, "tile_row_source")
                            ),
                            grouped_row_lane,
                        )
                        buffer_ops.buffer_store(
                            weight,
                            buffer_ops.create_buffer_resource_from_addr(
                                _peer_addr(dest, "tile_row_weight")
                            ),
                            grouped_row_lane,
                        )
                    fx.ptr_store(
                        Vec.from_elements([grouped_row_lane], fx.Int32),
                        _rb_lane(_RB_ROW),
                    )
                gpu.barrier()
                if tx < fx.Int32(scale_bytes):
                    scale = buffer_ops.buffer_load(
                        s_rs, s_b0 + tx, vec_width=1, dtype=T.i8
                    )
                    scale_res = buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "grouped_input_scale")
                    )
                    # 与 _dispatch_route 同一套 BM32 A-scale 预排布。
                    ku = tx // fx.Int32(8)
                    ikxdl = (tx % fx.Int32(8)) // fx.Int32(4)
                    k_lane = tx % fx.Int32(4)
                    for slot in range_constexpr(TOPK):
                        s_row = Vec(
                            fx.make_view(
                                fx.add_offset(
                                    rb_lds, fx.Int32((_RB_ROW + slot) * 4)
                                ),
                                fx.make_layout(1, 1),
                            ).load()
                        )[0]
                        if s_row >= fx.Int32(0):
                            s_phys = s_row // fx.Int32(BM)
                            s_rit = s_row % fx.Int32(BM)
                            a_sub = s_rit // fx.Int32(32)
                            a_row = s_rit % fx.Int32(32)
                            im_a = a_row // fx.Int32(16)
                            n_lane = a_row % fx.Int32(16)
                            dst_dword = (
                                (s_phys * fx.Int32(BM // 32) + a_sub)
                                * fx.Int32(scale_dwords * 32)
                                + ku * fx.Int32(64)
                                + k_lane * fx.Int32(16)
                                + n_lane
                            )
                            buffer_ops.buffer_store(
                                scale,
                                scale_res,
                                dst_dword * fx.Int32(4)
                                + ikxdl * fx.Int32(2)
                                + im_a,
                                offset_is_bytes=True,
                            )
                gpu.barrier()

        # route_gbatch:照 MegaMoEv2 emit_dispatch_group 的领位方式。一个 CTA 一批
        # _GB_NB = THREADS/TOPK 个 token,每线程一个 (token, slot);LDS 原子给片内
        # 序号,每个 (dest, expert) 只做一次 system 原子领一段连续行,段内若含组头
        # 就由领段的线程认领整组。ATT12:逐 record 路径 72% 周期是 s_barrier 陪等
        # 单线程串行链(mask 16 次串行 load、A1 原子、A2 map_ready)。
        # 每批 4 个 barrier;批末不需要 barrier:下一批 P0/P1 只写 CNT/HIT,
        # 它们的读者(P2/P3)都在本批 P3 末的 barrier 之前;P4 只读 ROW,
        # 而 ROW 只在下一批 P3 写,中间隔着三个 barrier。
        # 每批 token 数:受 THREADS//TOPK(每线程一个 (token, slot))和 GROUP_ROWS(一批每
        # expert 最多跨一个组头)两头限制,并取 WAVES 的倍数(P3 按 wave 均分 record)。
        # TOPK 不整除 THREADS 时末尾 THREADS-_GB_LANES 个线程在 P1 里不参与。
        _GB_NB = min(THREADS // TOPK, GROUP_ROWS) // WAVES * WAVES
        _GB_LANES = _GB_NB * TOPK
        # weights 相对 ids 的 dword 距离:record 内是 16B 补齐后的 REC_W-REC_I,
        # ids_first 的连续 [ids|weights] 数组里是 TOPK。
        _GB_WDW_REC = (REC_W - REC_I) // 4
        _GB_CNT = 16
        _GB_BASE = _GB_CNT + LOCAL_EXPERTS
        _GB_HIT = _GB_BASE + LOCAL_EXPERTS
        _GB_ROW = _GB_HIT + _GB_NB
        _GB_PH = _GB_ROW + THREADS
        # meta_opt:本 CTA 在 P2 自己认领的组(组号, 物理基址),解析时不必远端查。
        _GB_OWN = _GB_PH + 2 * LOCAL_EXPERTS
        _GB_OWNB = _GB_OWN + LOCAL_EXPERTS
        GB_P3 = 2
        _GB_UNITS = payload_dwords // 4
        _GB_IT = (_GB_UNITS + 63) // 64
        assert _GB_NB >= WAVES and _GB_NB % WAVES == 0 and _GB_LANES <= THREADS
        assert LOCAL_EXPERTS + _GB_NB <= THREADS, "P0 clears HIT with threads [LE, LE+_GB_NB)"
        assert _GB_NB <= GROUP_ROWS, "one batch may cross at most one group head per expert"
        assert payload_dwords % 4 == 0
        assert int(layout.source_capacity) <= max_route_tiles * BM, (
            "source-major scales must fit the grouped_input_scale region"
        )

        def _gb_ptr(idx):
            return fx.add_offset(
                fx.recast_iter(fx.Int32, lds_raw), idx * fx.Int32(4)
            )

        def _gb_ld(idx):
            return Vec(fx.make_view(_gb_ptr(idx), fx.make_layout(1, 1)).load())[0]

        def _gb_st(idx, value):
            fx.ptr_store(Vec.from_elements([value], fx.Int32), _gb_ptr(idx))

        def _route_gbatch(
            rs, tok0, tstride, dest, src_base, emit_masks=False,
            remote=False,
        ):
            # rs:AoS record 描述符(每 token REC_BYTES);本批 token = tok0 + r*tstride。
            # split_local 下远端来源(remote=True)用 expert_count/map 的后半 [LE,2LE):
            # 与本地来源分开计数、分开成组;物理 tile 仍从同一个 tile_alloc 认领。
            _EO = LOCAL_EXPERTS if (split_local and remote) else 0
            # P0:清本批计数和 token 命中标记。
            if tx < fx.Int32(LOCAL_EXPERTS):
                _gb_st(fx.Int32(_GB_CNT) + tx, fx.Int32(0))
            if (tx >= fx.Int32(LOCAL_EXPERTS)) & (
                tx < fx.Int32(LOCAL_EXPERTS + _GB_NB)
            ):
                _gb_st(fx.Int32(_GB_HIT - LOCAL_EXPERTS) + tx, fx.Int32(0))
            gpu.barrier()
            # P1:每线程一个 (token, slot),命中本 dest 的取片内序号。
            r = tx // fx.Int32(TOPK)
            slot = tx % fx.Int32(TOPK)
            tok = tok0 + r * tstride
            live = tok < ntokens
            if const_expr(_GB_LANES < THREADS):
                # 多出的线程 r == _GB_NB 会落到下一批的 token 上,必须整体失效。
                live = live & (tx < fx.Int32(_GB_LANES))
            rec_dw = live.select(tok, fx.Int32(0)) * fx.Int32(REC_BYTES // 4)
            # ids 的 dword 基址:record 内的 REC_I,或 ids_first 的连续数组(每 token
            # [ids|weights])。weights 在其后 _gb_wdw 个 dword(record 内 ids 段 16B 补齐)。
            id0 = rec_dw + fx.Int32(REC_I // 4)
            _gb_wdw = _GB_WDW_REC
            expert = buffer_ops.buffer_load(
                rs, id0 + slot, vec_width=1, dtype=T.i32
            )
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
            if const_expr(emit_masks):
                # 原 producer 循环唯一有用的产出:stage2 的本地平面掩码(每 token 一次,
                # 语义与原 producer 逐字相同)。dest0 的分片合起来恰好覆盖全部 token。
                # stage2 在 k2 之后跑,流序即可见,不需要 fence。
                if (dest == fx.Int32(0)) & (slot == fx.Int32(0)) & live:
                    m_rank = fx.Int32(0)
                    m_lslot = fx.Int32(0)
                    m_err = fx.Int32(0)
                    for s2 in range_constexpr(TOPK):
                        ex = buffer_ops.buffer_load(
                            rs, id0 + fx.Int32(s2), vec_width=1, dtype=T.i32
                        )
                        ok = (ex >= fx.Int32(0)) & (ex < fx.Int32(EXPERTS))
                        owner = ok.select(ex // fx.Int32(LOCAL_EXPERTS), fx.Int32(0))
                        m_rank = ok.select(m_rank | (fx.Int32(1) << owner), m_rank)
                        loc = ok & ((owner // fx.Int32(GPUS_PER_NODE)) == node)
                        m_lslot = loc.select(m_lslot | fx.Int32(1 << s2), m_lslot)
                        m_err = ok.select(m_err, m_err + fx.Int32(1))
                    if m_err != fx.Int32(0):
                        comm_ops.atomic_add_system(error_addr, m_err)
                    buffer_ops.buffer_store(
                        (m_rank >> fx.Int32(node * GPUS_PER_NODE)) & fx.Int32(0xFF),
                        buffer_ops.create_buffer_resource_from_addr(
                            stage2_addr("node_dest_rank_mask")
                        ),
                        fx.Int32(node * MAX_TOKENS) + tok,
                    )
                    if const_expr(publish_plane_slots):
                        buffer_ops.buffer_store(
                            m_lslot,
                            buffer_ops.create_buffer_resource_from_addr(
                                stage2_addr("node_dest_slot_mask")
                            ),
                            fx.Int32(node * MAX_TOKENS) + tok,
                        )
            gpu.barrier()
            # P2:每 expert 一次 system 原子领段;段内的组头由本线程认领(不等任何东西)。
            if tx < fx.Int32(LOCAL_EXPERTS):
                n = _gb_ld(fx.Int32(_GB_CNT) + tx)
                if n > fx.Int32(0):
                    seg = fx.Int32(
                        comm_ops.atomic_add_system(
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
                    gb_claim = head < seg + n
                    if const_expr(static_g0):
                        gb_claim = gb_claim & (head >= fx.Int32(GROUP_ROWS))
                    if gb_claim:
                        m0 = (tx + fx.Int32(_EO)) * fx.Int32(
                            max_tiles_per_expert
                        ) + head // fx.Int32(BM)
                        own_base = _claim_tile_group(
                            dest,
                            m0,
                            tx,
                            _peer_addr(dest, "expert_tile_map_ready") + fx.Int64(m0) * 8,
                        )
                        _gb_st(fx.Int32(_GB_OWN) + tx, head // fx.Int32(GROUP_ROWS))
                        _gb_st(fx.Int32(_GB_OWNB) + tx, own_base)
            gpu.barrier()
            # P3:命中线程等 map_ready、写元数据;然后每个 wave 拷自己那几个 record 的 payload。
            if const_expr(GB_P3 >= 1):
                # 先发出本 wave 全部 record 的 payload/scale load(与下面的组基址解析重叠),
                # 解析完再统一 store。
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
                                buffer_ops.buffer_load(rs, q0 + dc, vec_width=4, dtype=T.i32)
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

                _issue_payload_loads()
                if const_expr(GB_P3 >= 2):
                    # 每 expert 一个线程解析本批段落 [seg, seg+n) 涉及的至多 2 个组的物理基址
                    # (组头认领都已在 P2 完成,这里只等不认领)。逐行的远端 map 读没了。
                    # 每 expert 两个线程并发解析(j=0 首组,j=1 末组);末组与首组相同时 j=1 不做。
                    # 本 CTA 在 P2 认领的组直接用 LDS 里的基址,只有别人认领的组才远端等+读。
                    # 最坏由 4 次串行远端往返降到 2 次;认领都在 P2 完成,这里只等不认领。
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
                                # static_g0:第 0 组的物理基址是 e*G,不等也不读远端 map
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
                                    mo_j * fx.Int32(LOCAL_EXPERTS)
                                    + fx.Int32(_GB_PH)
                                    + mo_e,
                                    ph,
                                )
                    if tx < fx.Int32(0):
                        n = _gb_ld(fx.Int32(_GB_CNT) + tx)
                        if n > fx.Int32(0):
                            seg = _gb_ld(fx.Int32(_GB_BASE) + tx)
                            for j in range_constexpr(2):
                                gidx = (
                                    seg + (fx.Int32(0) if j == 0 else n - fx.Int32(1))
                                ) // fx.Int32(GROUP_ROWS)
                                m = (tx + fx.Int32(_EO)) * fx.Int32(
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
                                _gb_st(fx.Int32(_GB_PH + j * LOCAL_EXPERTS) + tx, ph)
                    gpu.barrier()
                grow = fx.Int32(-1)
                if hit:
                    row_slot = _gb_ld(fx.Int32(_GB_BASE) + le) + intra
                    if const_expr(GB_P3 >= 2):
                        g_row = row_slot // fx.Int32(GROUP_ROWS)
                        g_0 = _gb_ld(fx.Int32(_GB_BASE) + le) // fx.Int32(GROUP_ROWS)
                        pbase = (g_row == g_0).select(
                            _gb_ld(fx.Int32(_GB_PH) + le),
                            _gb_ld(fx.Int32(_GB_PH + LOCAL_EXPERTS) + le),
                        )
                        grow = (
                            pbase + (row_slot // fx.Int32(BM)) % fx.Int32(G)
                        ) * fx.Int32(BM) + row_slot % fx.Int32(BM)
                    else:
                        mp = (le + fx.Int32(_EO)) * fx.Int32(
                            max_tiles_per_expert
                        ) + row_slot // fx.Int32(BM)
                        _spin(
                            _peer_addr(dest, "expert_tile_map_ready") + fx.Int64(mp) * 8,
                            generation,
                        )
                        phys = buffer_ops.buffer_load(
                            buffer_ops.create_buffer_resource_from_addr(
                                _peer_addr(dest, "expert_tile_map")
                            ),
                            mp,
                            vec_width=1,
                            dtype=T.i32,
                        )
                        grow = phys * fx.Int32(BM) + row_slot % fx.Int32(BM)
                    weight = buffer_ops.buffer_load(
                        rs, id0 + fx.Int32(_gb_wdw) + slot, vec_width=1, dtype=T.f32
                    )
                    src = src_base + tok
                    buffer_ops.buffer_store(
                        src,
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_input")
                        ),
                        grow,
                    )
                    buffer_ops.buffer_store(
                        src | (slot << fx.Int32(24)),
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_source")
                        ),
                        grow,
                    )
                    buffer_ops.buffer_store(
                        weight,
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_weight")
                        ),
                        grow,
                    )
                _gb_st(fx.Int32(_GB_ROW) + tx, grow)
                for prec, ptok, vals, sv in _pl:
                    if _gb_ld(fx.Int32(_GB_HIT) + prec) != fx.Int32(0):
                        dst_payload = buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "grouped_input_q")
                            + fx.Int64(src_base + ptok) * fx.Int64(wire.payload_bytes),
                            num_records_bytes=wire.payload_bytes,
                        )
                        for it in range_constexpr(_GB_IT):
                            d = (lane + fx.Int32(it * 64)) * fx.Int32(4)
                            if d < fx.Int32(payload_dwords):
                                buffer_ops.buffer_store(vals[it], dst_payload, d)
                        if lane < fx.Int32(scale_dwords):
                            buffer_ops.buffer_store(
                                sv,
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "grouped_input_scale")
                                    + fx.Int64(src_base + ptok)
                                    * fx.Int64(scale_bytes)
                                ),
                                lane,
                            )
                gpu.barrier()
            else:
                grow = fx.Int32(-1)
                if hit:
                    row_slot = _gb_ld(fx.Int32(_GB_BASE) + le) + intra
                    mp = (le + fx.Int32(_EO)) * fx.Int32(
                        max_tiles_per_expert
                    ) + row_slot // fx.Int32(BM)
                    _spin(
                        _peer_addr(dest, "expert_tile_map_ready") + fx.Int64(mp) * 8,
                        generation,
                    )
                    phys = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "expert_tile_map")
                        ),
                        mp,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    grow = phys * fx.Int32(BM) + row_slot % fx.Int32(BM)
                    weight = buffer_ops.buffer_load(
                        rs, id0 + fx.Int32(_gb_wdw) + slot, vec_width=1, dtype=T.f32
                    )
                    src = src_base + tok
                    buffer_ops.buffer_store(
                        src,
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_input")
                        ),
                        grow,
                    )
                    buffer_ops.buffer_store(
                        src | (slot << fx.Int32(24)),
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_source")
                        ),
                        grow,
                    )
                    buffer_ops.buffer_store(
                        weight,
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "tile_row_weight")
                        ),
                        grow,
                    )
                _gb_st(fx.Int32(_GB_ROW) + tx, grow)
                for rr in range_constexpr(_GB_NB // WAVES):
                    prec = wave + fx.Int32(rr * WAVES)
                    if _gb_ld(fx.Int32(_GB_HIT) + prec) != fx.Int32(0):
                        ptok = tok0 + prec * tstride
                        dst_payload = buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "grouped_input_q")
                            + fx.Int64(src_base + ptok) * fx.Int64(wire.payload_bytes),
                            num_records_bytes=wire.payload_bytes,
                        )
                        q0 = ptok * fx.Int32(REC_BYTES // 4) + fx.Int32(REC_Q // 4)
                        vals = []
                        for it in range_constexpr(_GB_IT):
                            d = (lane + fx.Int32(it * 64)) * fx.Int32(4)
                            dc = (d < fx.Int32(payload_dwords)).select(d, fx.Int32(0))
                            vals.append(
                                buffer_ops.buffer_load(rs, q0 + dc, vec_width=4, dtype=T.i32)
                            )
                        for it in range_constexpr(_GB_IT):
                            d = (lane + fx.Int32(it * 64)) * fx.Int32(4)
                            if d < fx.Int32(payload_dwords):
                                buffer_ops.buffer_store(vals[it], dst_payload, d)
                        # scale 按源 token 行主序,和 payload 同一行号。
                        if lane < fx.Int32(scale_dwords):
                            sv = buffer_ops.buffer_load(
                                rs,
                                ptok * fx.Int32(REC_BYTES // 4)
                                + fx.Int32(REC_S // 4)
                                + lane,
                                vec_width=1,
                                dtype=T.i32,
                            )
                            buffer_ops.buffer_store(
                                sv,
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "grouped_input_scale")
                                    + fx.Int64(src_base + ptok)
                                    * fx.Int64(scale_bytes)
                                ),
                                lane,
                            )
                gpu.barrier()
            # P4:scale 按 (record, 字节) 摊到全 CTA,散写到每条 route 的 BM32 预排布。
            scale_res = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "grouped_input_scale")
            )
            for ps in range_constexpr(0):
                idx = tx + fx.Int32(ps * THREADS)
                if idx < fx.Int32(_GB_NB * scale_bytes):
                    sr = idx // fx.Int32(scale_bytes)
                    sbyte = idx % fx.Int32(scale_bytes)
                    stok = tok0 + sr * tstride
                    stok = (stok < ntokens).select(stok, fx.Int32(0))
                    scale = buffer_ops.buffer_load(
                        rs,
                        stok * fx.Int32(REC_BYTES) + fx.Int32(REC_S) + sbyte,
                        vec_width=1,
                        dtype=T.i8,
                    )
                    ku = sbyte // fx.Int32(8)
                    ikxdl = (sbyte % fx.Int32(8)) // fx.Int32(4)
                    k_lane = sbyte % fx.Int32(4)
                    for sslot in range_constexpr(TOPK):
                        s_row = _gb_ld(
                            fx.Int32(_GB_ROW + sslot) + sr * fx.Int32(TOPK)
                        )
                        if s_row >= fx.Int32(0):
                            s_phys = s_row // fx.Int32(BM)
                            s_rit = s_row % fx.Int32(BM)
                            a_sub = s_rit // fx.Int32(32)
                            a_row = s_rit % fx.Int32(32)
                            im_a = a_row // fx.Int32(16)
                            n_lane = a_row % fx.Int32(16)
                            dst_dword = (
                                (s_phys * fx.Int32(BM // 32) + a_sub)
                                * fx.Int32(scale_dwords * 32)
                                + ku * fx.Int32(64)
                                + k_lane * fx.Int32(16)
                                + n_lane
                            )
                            buffer_ops.buffer_store(
                                scale,
                                scale_res,
                                dst_dword * fx.Int32(4) + ikxdl * fx.Int32(2) + im_a,
                                offset_is_bytes=True,
                            )


        # early_local_gmm:封尾的 pad/组表按 expert×SPL 线程并行(原来每 expert 一个线程串行,
        # F8T2 段 1 pad ~12µs、组表+发布 ~11µs,且在本地 GMM1 的关键路径上)。
        SPL = max(1, min(8, THREADS // max(1, LOCAL_EXPERTS)))

        def _seal_pad_par(EO):
            """_seal_pad 的并行版(不含 diagnostic_no_arrival_rmw/dispatch_plan 分支,split_local 已排除)。
            线程 (e, l):expert e 的每个 pad tile 里第 l, l+SPL, ... 行。"""
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
                    _spin(
                        local_addr("expert_tile_map_ready")
                        + fx.Int64(map_index) * fx.Int64(8),
                        generation,
                    )
                    physical = buffer_ops.buffer_load(
                        tile_map_res, map_index, vec_width=1, dtype=T.i32
                    )
                    head_physical = buffer_ops.buffer_load(
                        tile_map_res, map_row0, vec_width=1, dtype=T.i32
                    )
                    fallback_input = buffer_ops.buffer_load(
                        inputs, head_physical * fx.Int32(BM), vec_width=1, dtype=T.i32
                    )
                    if const_expr(static_g0):
                        # 0 行的 expert 首行从未写过:pad 指向第 0 个 source 行(source 无效,结果不被读)
                        fallback_input = (count > fx.Int32(0)).select(fallback_input, fx.Int32(0))
                    row_start = (pad_tile == first_pad_tile).select(
                        count % fx.Int32(BM), fx.Int32(0)
                    )
                    for row in range(row_start + pl, fx.Int32(BM), fx.Int32(SPL)):
                        dst = physical * fx.Int32(BM) + row
                        buffer_ops.buffer_store(fallback_input, inputs, dst)
                        buffer_ops.buffer_store(fx.Int32(INVALID_SOURCE), sources, dst)
                        buffer_ops.buffer_store(fx.Float32(0.0), weights, dst)

        def _seal_pad(EO):
            """补齐 expert_count[EO:EO+LE] 这一类行的尾 tile/空 tile(EO=0 本地或全部,EO=LE 远端)。"""
            expert_count = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            if tx < fx.Int32(LOCAL_EXPERTS):
                count = buffer_ops.buffer_load(
                    expert_count, tx + fx.Int32(EO), vec_width=1, dtype=T.i32
                )
                # G>1 时一个 expert 认领的 tile 数被抬到 G 的倍数,所以要补的
                # 不只是最后一个真实 tile 的尾巴,还有整组里多出来的空 tile。
                real_tiles = (count + fx.Int32(BM - 1)) // fx.Int32(BM)
                alloc_tiles = (
                    (real_tiles + fx.Int32(G - 1)) // fx.Int32(G)
                ) * fx.Int32(G)
                first_pad_tile = count // fx.Int32(BM)
                for pad_tile in range(first_pad_tile, alloc_tiles, fx.Int32(1)):
                    logical_tile = pad_tile
                    map_index = (tx + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert) + logical_tile
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
                    # fallback 取本 expert 第 0 个 tile 的第 0 行:整组多出来的
                    # 空 tile 自己的第 0 行从来没有到达过,不能拿它当 fallback。
                    # 走到这里 count>0(否则 alloc_tiles==0,循环为空),所以
                    # logical tile 0 一定存在且已填。
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
            """expert-major 置换。segs:每个 expert 依次拼接的 map 行段(0=本地/全部,
            LE=远端);base_segs:排在整个视图前面的段(split_local 非 h1_phys 的远端段
            以本地段总数为基址)。h1_phys+split_local 时 segs=(0, LE):每个 expert 连续,
            组内先本地组后远端组。par:线程 (e, l) 做 expert e 的第 l, l+SPL, ... 个 tile/组
            (每条 lane 自己重算前缀和),把每 expert 串行的 load→store 依赖链缩成 1/SPL。"""
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
            perm_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_dst_of_src")
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
                        my_base = my_base + (fx.Int32(e) < pe).select(
                            n, fx.Int32(0)
                        )
                        all_tiles = all_tiles + n
                if tx == fx.Int32(0):
                    if const_expr(check_tiles is not None):
                        # Sum ceil(ceil(count/BM)/G)*G(全部段)必须等于 tile_alloc:
                        # 每次认领固定分配 G 个 tile,所以置换仍是双射。
                        if seg_base + all_tiles != check_tiles:
                            comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                    if const_expr(publish_seg):
                        # 本地段 tile 数(F5 的本地 job 数由它推出)。
                        buffer_ops.buffer_store(
                            all_tiles,
                            buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                            fx.Int32(1),
                        )
                tile_map = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("expert_tile_map")
                )
                # 按组的两张表只在 G>1 的 layout 里有 region;G=1 时连 resource
                # 都不能建(trace 期就 KeyError),用处也都在 const_expr(G > 1) 里。
                group_perm_res = (
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_group_perm")
                    )
                    if G > 1
                    else None
                )
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
                        _spin(
                            local_addr("expert_tile_map_ready")
                            + fx.Int64(map_index) * fx.Int64(8),
                            generation,
                        )
                        src = buffer_ops.buffer_load(
                            tile_map, map_index, vec_width=1, dtype=T.i32
                        )
                        dst = run_base + j
                        buffer_ops.buffer_store(dst, perm_res, src)
                        buffer_ops.buffer_store(pe, sorted_e_res, dst)
                        buffer_ops.buffer_store(src, inv_res, dst)
                    if const_expr(G > 1):
                        # GMM1 的 m_block 是「组」,expert id 和目的地都要按组
                        # 再给一份。run_base 是 G 倍数的前缀和,组首 src 由整组
                        # 认领得到,所以两边都对齐到 G。单独一个按组的循环:
                        # 不在归纳变量上做 %,也不在循环体里做条件写。
                        for jg in range(
                            pl, my_tiles // fx.Int32(G), fx.Int32(PS)
                        ):
                            j0 = jg * fx.Int32(G)
                            head_src = buffer_ops.buffer_load(
                                tile_map, map0 + j0, vec_width=1, dtype=T.i32,
                            )
                            head_group = head_src // fx.Int32(G)
                            buffer_ops.buffer_store(
                                (run_base + j0) // fx.Int32(G),
                                group_perm_res,
                                head_group,
                            )
                            buffer_ops.buffer_store(pe, group_e_res, head_group)
                    run_base = run_base + my_tiles

        def _seal_group_list(EO, base_segs, rev=False):
            """early_local_gmm:把 expert_count[EO:EO+LE] 这一类的物理组号(组首物理 tile // G)
            按 expert 顺序写进 gmm1_group_list;base_segs 的组数之和是本段起点(远端段接在本地段后)。
            rev:按 expert 倒序排(远端段用;本地段最后读的权重还在 MALL 里)。"""
            counts_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("expert_count")
            )
            if tx < fx.Int32(LOCAL_EXPERTS * SPL):
                ge = tx // fx.Int32(SPL)
                gl = tx - ge * fx.Int32(SPL)
                my_base = fx.Int32(0)
                for bi in range_constexpr(len(base_segs)):
                    for e in range_constexpr(LOCAL_EXPERTS):
                        my_base = my_base + _alloc_tiles_for(
                            buffer_ops.buffer_load(
                                counts_res, fx.Int32(base_segs[bi] + e), vec_width=1, dtype=T.i32
                            )
                        ) // fx.Int32(G)
                for e in range_constexpr(LOCAL_EXPERTS):
                    my_base = my_base + ((fx.Int32(e) > ge) if rev else (fx.Int32(e) < ge)).select(
                        _alloc_tiles_for(
                            buffer_ops.buffer_load(
                                counts_res, fx.Int32(EO + e), vec_width=1, dtype=T.i32
                            )
                        ) // fx.Int32(G),
                        fx.Int32(0),
                    )
                gl_map = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("expert_tile_map")
                )
                gl_list = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("gmm1_group_list")
                )
                gl_egrp = (
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_expert_group")
                    )
                    if G > 1
                    else None
                )
                gl_map0 = (ge + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert)
                gl_groups = _alloc_tiles_for(
                    buffer_ops.buffer_load(
                        counts_res, ge + fx.Int32(EO), vec_width=1, dtype=T.i32
                    )
                ) // fx.Int32(G)
                for gl_j in range(gl, gl_groups, fx.Int32(SPL)):
                    gl_mi = gl_map0 + gl_j * fx.Int32(G)
                    _spin(
                        local_addr("expert_tile_map_ready") + fx.Int64(gl_mi) * fx.Int64(8),
                        generation,
                    )
                    gl_grp = buffer_ops.buffer_load(
                        gl_map, gl_mi, vec_width=1, dtype=T.i32
                    ) // fx.Int32(G)
                    buffer_ops.buffer_store(gl_grp, gl_list, my_base + gl_j)
                    # GMM1 按 tile_expert_group[物理组] 取权重;本地组要在段 2 置换之前就能算,
                    # 所以这里先写(段 2 的 _seal_perm 会写同一个值)。
                    if const_expr(G > 1):
                        buffer_ops.buffer_store(ge, gl_egrp, gl_grp)

        def _seal_seg1():
            """split_local 段 1:等 8 个 local_eos(本 node 全部来源的 fan1 推完),补齐本地组,
            写本地段 tile 数(以及 early_local_gmm 的本地组表 + h1_local_eos)。
            early_local_gmm 下由 ticket>=8 的 fan CTA 调用,打点 7=本地 EOS 齐 8=pad 完 9=发布
            (消费者只用 3/4/5/6;非 early_local_gmm 时调用者是 sealer,槽 7..9 与 T0 冲突,不打)。"""
            # 段 1:8 个 local_eos 齐 = 全部本地来源行已落位,本地组计数定了。
            # 封本地组(pad + h1 段基址 0 的置换),再去等全局 comm_eos 封远端组。
            if tx == fx.Int32(0):
                for peer in range_constexpr(GPUS_PER_NODE):
                    _spin(
                        local_addr("comm_eos")
                        + fx.Int64((GPUS_PER_NODE + peer) * 8),
                        generation,
                    )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            if const_expr(early_local_gmm):
                if const_expr(not lazy_pad):
                    _seal_pad_par(0)
            else:
                _seal_pad(0)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            # h1_phys 下排序视图整体在段 2 算(每个 expert 先本地组后远端组);
            # 这里只发布本地段 tile 数。否则本地段自成一段、段基址 0。
            if tx == fx.Int32(0):
                _lt = fx.Int32(0)
                _cr = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("expert_count")
                )
                # local_defer:末尾 D 个 expert 的本地组不进本地段。消费者的本地上界和远端段起点
                # 都由 tile_alloc[1] 推出,组表布局不变,这些组就落在远端段开头,
                # 和它们的远端组一起在同一次权重读里做(远端段本来就要把权重全读一遍)。
                for e in range_constexpr(LOCAL_EXPERTS):
                    _lt = _lt + _alloc_tiles_for(
                        buffer_ops.buffer_load(_cr, fx.Int32(e), vec_width=1, dtype=T.i32)
                    )
                buffer_ops.buffer_store(
                    _lt,
                    buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                    fx.Int32(1),
                )
            if const_expr(early_local_gmm):
                _seal_group_list(0, ())
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(early_local_gmm):
                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    # fence 已经把前面的写整体放行;release store 会再做一次整 L2 写回。
                    (comm_ops.store_i64_global_system_relaxed)(
                        local_addr("h1_local_eos"), generation
                    )

        dest = fx.Int32(0)
        # fanout 的并发度:原本 is_comm = ticket < GPUS_PER_NODE,即 8 个 CTA
        # 各钉死一个目的 rank、串行走完全部 MAX_TOKENS。实测这一段占 stage1
        # 的 82%(8,338us 搬 15.6MB = 1.87 GB/s,而同一块 arena 流式读是
        # 3.6 TB/s)—— 是并发度不是带宽。分片后 CTA t 负责
        # dest = t % GPUS_PER_NODE、shard = t // GPUS_PER_NODE,token 按
        # shard 跨步。借用的是 producer CTA(量化只要 ~10us 就闲置),
        # 不动 GMM1 消费者池。
        is_fanout = ticket < fx.Int32(GPUS_PER_NODE * fanout_shards)
        # T0(CCO 收发 + credit + seal)是关键路径,不再兼做 dest0
        # 的 0 号分片;dest0 由其余 fanout_shards-1 个分片分掉。
        is_fanout = is_fanout & (ticket != fx.Int32(0))
        if is_fanout:
            fan_stride = fx.Int32(fanout_shards)
            if const_expr(fanout_shards == 1):
                dest = ticket
                fanout_shard = fx.Int32(0)
            else:
                dest = ticket % fx.Int32(GPUS_PER_NODE)
                fanout_shard = ticket // fx.Int32(GPUS_PER_NODE)
                d0 = dest == fx.Int32(0)
                fanout_shard = d0.select(
                    fanout_shard - fx.Int32(1), fanout_shard
                )
                fan_stride = d0.select(
                    fx.Int32(fanout_shards - 1), fan_stride
                )
            # 干活用的分片号/token 步长;完成计数仍用 fan_stride(含不分 token 的发帖 CTA)。
            fan_wshard = fanout_shard
            fan_tstride = fan_stride
            pn_dest = (dest >= fx.Int32(_POST_LO)) & (dest < fx.Int32(_POST_HI))
            pn_self = (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI))
            fan_tstride = pn_dest.select(fan_stride - fx.Int32(1), fan_stride)
            fan_wshard = pn_self.select(
                fx.Int32(1 << 20),
                pn_dest.select(fanout_shard - fx.Int32(1), fanout_shard),
            )
            # 每个 QP 一个 CTA 投递+敲门铃(照 kernel2 的 DBCTAS):ticket q 发 QP q。
            if (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI)):
                if wave == fx.Int32(0):
                    _rail_post(ticket - fx.Int32(1), True)
            if tx == fx.Int32(0):
                _spin(
                    _peer_addr(dest, "launch_ready"), control_generation
                )
            gpu.barrier()
            # 和 CCO CTA 那边同一个毛病:逐个 token 用单线程去看一个
            # 早已置位的 flag,每次一条 system scope load,串行依赖。
            # 实测所有 producer 在 ~98us 就发布完了,所以这些等待几乎
            # 全是白等。把它们提到循环前、用整个 CTA 一次看完。
            # 见 [[串行等 flag 的 700us]]。
            for token in range(
              fan_wshard, fx.Int32(MAX_TOKENS), fan_tstride * fx.Int32(_GB_NB)
            ):
                _route_gbatch(
                  buffer_ops.create_buffer_resource_from_addr(x_q),
                  token, fan_tstride, dest, fx.Int32(rank * MAX_TOKENS),
                  emit_masks=True,
                )
            for token in range(
              fanout_shard,
              fx.Int32(0),
              fan_stride,
            ):
                # 本地来源直接读量化输入(k1 之前已就绪),不等 producer。
                if token < ntokens:
                    _dispatch_local_direct(
                        token, dest, fx.Int32(rank * MAX_TOKENS) + token
                    )
            if const_expr(split_local):
                # local_eos:本源 rank 给 dest 的 fan1(本地来源那一半)全部推完。
                # 与 comm_eos 同一套分片计数+减法复位,计数器在 [8,16);只经 xGMI,
                # 不依赖 rail,所以不引入新的跨节点等待环。
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    _ldone = local_addr("fanout_shard_done") + fx.Int64(
                        GPUS_PER_NODE * 4
                    ) + fx.Int64(dest) * fx.Int64(4)
                    _lprev = fx.Int32(
                        comm_ops.atomic_add_system_acq_rel(_ldone, fx.Int32(1))
                    )
                    if _lprev + fx.Int32(1) == fan_stride:
                        comm_ops.atomic_add_agent(_ldone, fx.Int32(0) - fan_stride)
                        comm_ops.fence_system_release()
                        (comm_ops.store_i64_global_system_relaxed)(
                            _peer_addr(dest, "comm_eos")
                            + fx.Int64((GPUS_PER_NODE + local_rank) * 8),
                            generation,
                        )
            if const_expr(early_local_gmm):
                # F5:本地组封尾挪到 fan CTA(ticket 8+local_rank,非 T0/非 rail post)的 fan1 之后,
                # 不再等 sealer 自己做完 fan2(E1T:本地门 ~175µs ≈ 全局 comm_eos 时刻)。
                # 只等 node 内各源的 fan1,fan1 不依赖 fan2/封尾,无环。
                # fan2_shards 时挪到 fan2 池外(shard=F2S),不拖本 dest 的 fan2。
                if ticket == fx.Int32(GPUS_PER_NODE * _SEG1_SHARD + local_rank):
                    _seal_seg1()
            # 这里比原来保守:等**全部** chunk 而不是本 token 那个。
            # dispatch_chunks=2 且整批只有一次 flush,两者实际同时到,
            # 代价上限是半次传输(~10us),换掉每 token 一次的自旋。
            if (tx < fx.Int32(
                layout.num_qp
            )) & (fan_wshard < fx.Int32(F2S if F2S else 1 << 20)):
                _spin(
                    local_addr(
                        "remote_chunk_ready"
                    )
                    + fx.Int64(tx) * fx.Int64(8),
                    generation,
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            f2_stride = fx.Int32(F2S) if F2S else fan_tstride
            f2_end = (
                (fan_wshard < fx.Int32(F2S)).select(
                    fx.Int32(MAX_TOKENS), fx.Int32(0)
                )
                if F2S
                else fx.Int32(MAX_TOKENS)
            )
            for token in range(
              fan_wshard, f2_end, f2_stride * fx.Int32(_GB_NB)
            ):
                _route_gbatch(
                  buffer_ops.create_buffer_resource_from_addr(
                    local_addr("remote_dispatch_rx")
                  ),
                  token, f2_stride, dest,
                  fx.Int32(remote_source_rank * MAX_TOKENS),
                      remote=True,
                )
            for token in range(
              fanout_shard,
              fx.Int32(0),
              fan_stride,
            ):
                if token < ntokens:
                    _dispatch_remote_soa(
                        token,
                        dest,
                        fx.Int32(remote_source_rank * MAX_TOKENS) + token,
                    )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            # 在发布本分片之前回收自己的数据请求:consumed 到齐前必然完成,
            # T0 的 credit 段不会和这里同时轮询同一个 QP 的 CQ。
            if (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI)):
                if wave == fx.Int32(0):
                    post_qp = ticket - fx.Int32(1)
                    post_req = fx.Int64(
                        comm_ops.load_i64_global(
                            local_addr("remote_chunk_request")
                            + fx.Int64(post_qp) * fx.Int64(8)
                        )
                    )
                    _rail.wait(dev_comm, post_qp, post_req)
                gpu.barrier()
            # k1 一律不碰 fanout_shard_done / comm_eos /
            # remote_chunk_consumed:这三者都描述「本 dest 的全部 record
            # 都推完了」,而 k1 只推完了本地来源那一半。
            if (tx == fx.Int32(0)) & (
                fx.Int32(1) == fx.Int32(1)
            ):
                if const_expr(fanout_shards == 1):
                    _publish = fx.Int32(1) == fx.Int32(1)
                else:
                    # 每个 dest 的最后一个分片才发 EOS 和信用原子:
                    # remote_chunk_consumed 的每个字必须恰好累到
                    # GPUS_PER_NODE,分片不能各加一次。用减法复位
                    # (和 stage2 到达协议同一手法),不依赖下一代的清零。
                    # 本分片所有远端写(上面已 drain+barrier)经这条
                    # system release RMW 发布;最后到达者经它的
                    # acquire 看到全部分片,再由下面的 release 传给 comm_eos。
                    _prev = fx.Int32(
                        comm_ops.atomic_add_system_acq_rel(
                            local_addr("fanout_shard_done")
                            + fx.Int64(dest) * fx.Int64(4),
                            fx.Int32(1),
                        )
                    )
                    _publish = _prev + fx.Int32(1) == fan_stride
                    if _publish:
                        comm_ops.atomic_add_agent(
                            local_addr("fanout_shard_done")
                            + fx.Int64(dest) * fx.Int64(4),
                            fx.Int32(0) - fan_stride,
                        )
                if _publish:
                    comm_ops.fence_system_release()
                    for consume_index in range_constexpr(
                        dispatch_chunks * layout.num_qp
                    ):
                        comm_ops.atomic_add_system_acq_rel(
                            local_addr("remote_chunk_consumed")
                            + fx.Int64(consume_index * 4),
                            fx.Int32(1),
                        )
                    (comm_ops.store_i64_global_system_relaxed)(
                        _peer_addr(dest, "comm_eos")
                        + fx.Int64(local_rank) * 8,
                        generation,
                    )

        # The CCO CTA is also destination role zero, so delayed-credit progress
        # must run only after the common is_comm path above has contributed its
        # own consumed count. Otherwise the scoreboard can reach only seven.

        if is_cco:
            qp = wave
            for chunk in range(
                fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
            ):
                # Do not credit a chunk until every destination fanout role has
                # consumed its reciprocal payload.
                consume_index = chunk * fx.Int32(layout.num_qp) + qp
                if lane == fx.Int32(0):
                    consumed_count = fx.Int32(
                        comm_ops.load_i32_global_system(
                            local_addr("remote_chunk_consumed")
                            + fx.Int64(consume_index) * fx.Int64(4)
                        )
                    )
                    while consumed_count < fx.Int32(GPUS_PER_NODE):
                        consumed_count = fx.Int32(
                            _pl32(
                                local_addr("remote_chunk_consumed")
                                + fx.Int64(consume_index) * fx.Int64(4)
                            )
                        )
                    comm_ops.fence_system_acquire()
                gpu.barrier()
                comm_ops.fence_system_acquire()
                credit_byte = (
                    window_off("remote_chunk_credit")
                    + fx.Int64(consume_index) * fx.Int64(8)
                )
                _rail.put_value(
                    dev_comm,
                    qp,
                    fx.Int32(remote_node),
                    arena_win,
                    credit_byte,
                    generation,
                    aggregate=True,
                )
                credit_req = _rail.flush_async(
                    dev_comm,
                    qp,
                    fx.Int32(remote_node),
                )
                _rail.wait(dev_comm, qp, credit_req)

        # The role targeting this rank observes exactly eight EOS values; each
        # one covers a local source rank and its aligned remote source rank.
        if is_sealer:
            if const_expr(split_local and not early_local_gmm):
                _seal_seg1()
            if const_expr(early_local_gmm):
                # 段 1 由 ticket 8+local_rank 的 fan CTA 在它的 fan1 之后做(见 fanout 段);
                # 段 2 的置换/sortcopy 读本地 pad 行,必须排在段 1 之后。
                if tx == fx.Int32(0):
                    (comm_ops.spin_until_ge_i64_sleep_rx)(
                        local_addr("h1_local_eos"), generation, 127
                    )
                gpu.barrier()
                comm_ops.fence_system_acquire()
            if tx == fx.Int32(0):
                # Independent lane acquires followed by a CTA barrier do not
                # merge into one happens-before chain. The publishing thread
                # itself must acquire all eight communication-role releases.
                for peer in range_constexpr(GPUS_PER_NODE):
                    _spin(
                        local_addr("comm_eos") + fx.Int64(peer * 8), generation
                    )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            if const_expr(seal_fast):
                _seal_pad_par(LOCAL_EXPERTS if split_local else 0)
                if const_expr(lazy_pad):
                    _seal_pad_par(0)
            else:
                _seal_pad(LOCAL_EXPERTS if split_local else 0)
                if const_expr(lazy_pad):
                    _seal_pad(0)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            # sealer 内部分段(槽 5..7 本是 T0 专用;rank0 的 sealer 就是 T0,会覆盖,分析时丢掉 rank0/8)
            finish_scratch = fx.recast_iter(fx.Int32, lds_raw)
            if tx == fx.Int32(0):
                tiles = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                    fx.Int32(0),
                    vec_width=1,
                    dtype=T.i32,
                )
                fx.ptr_store(Vec.from_elements([tiles], fx.Int32), finish_scratch)
            gpu.barrier()
            tiles = Vec(
                fx.make_view(finish_scratch, fx.make_layout(1, 1)).load()
            )[0]
            # Expert-major view for Stage2.  expert_count is final here and
            # expert_tile_map[e][j] is dense in j, so one thread per expert
            # expands the permutation directly.  Each expert thread
            # recomputes its own tile base from the LOCAL_EXPERTS counts
            # rather than running a scan: the counts are one cache line
            # deep and this costs no LDS and no extra barrier.
            if const_expr(split_local):
                _seal_perm((0, LOCAL_EXPERTS), (), tiles, False, seal_fast)
                if const_expr(early_local_gmm):
                    _seal_group_list(LOCAL_EXPERTS, (0,), False)
            elif const_expr(split_local):
                _seal_perm((LOCAL_EXPERTS,), (0,), tiles, False)
            else:
                _seal_perm((0,), (), tiles, False, seal_fast)
            perm_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_dst_of_src")
            )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            # Stage2 indexes these two by absolute row, so they follow the
            # output rather than the source.  Partial-tile padding above has
            # already settled, so the copy sees final values.  The float32
            # weights are copied as raw bits.
            src_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_source")
            )
            wts_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_weight")
            )
            src_sorted_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_source_sorted")
            )
            wts_sorted_res = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_weight_sorted")
            )
            for row in range(
                tx,
                fx.Int32(0),
                fx.Int32(THREADS),
            ):
                src_tile = row // fx.Int32(BM)
                dst_row = (
                    buffer_ops.buffer_load(
                        perm_res, src_tile, vec_width=1, dtype=T.i32
                    )
                    * fx.Int32(BM)
                    + row
                    - src_tile * fx.Int32(BM)
                )
                buffer_ops.buffer_store(
                    buffer_ops.buffer_load(
                        src_res, row, vec_width=1, dtype=T.i32
                    ),
                    src_sorted_res,
                    dst_row,
                )
                buffer_ops.buffer_store(
                    buffer_ops.buffer_load(
                        wts_res, row, vec_width=1, dtype=T.i32
                    ),
                    wts_sorted_res,
                    dst_row,
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            # GMM1 的一个 m_block 覆盖 G 个物理 tile,所以作业数按组算。
            total_jobs = (tiles // fx.Int32(G)) * fx.Int32(
                GNB
            )
            # 非 pipeline 的 GMM1 按 tile_alloc 闭式跨步取作业,不读
            # h1_ready_queue / h1_ready_queue_generation,不再逐 job 写它们。
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                buffer_ops.buffer_store(
                    total_jobs,
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("h1_queue_tail")
                    ),
                    fx.Int32(0),
                )
                buffer_ops.buffer_store(
                    tiles * fx.Int32(BM),
                    buffer_ops.create_buffer_resource_from_addr(
                        local_addr("num_valid")
                    ),
                    fx.Int32(0),
                )
                comm_ops.fence_system_release()
                # fence 已经把前面的写整体放行;release store 会再做一次整 L2 写回。
                (comm_ops.store_i64_global_system_relaxed)(
                    local_addr("h1_queue_eos"), generation
                )

        def _run_gemm1_job(job, nc=None):
            # nc = (领位头地址, 本段上界, 本段偏移):GMM1 体内提前领下一个 job(next_claim)。
            # gemm1 rev: 3-stage A, 2-ahead DMA, wait_lds_barrier(vmcnt 24), ascale_gather v2, BN128 v2, lds alias scopes v1, epi acc pad4 v2, nofence barrier v1, next_claim v1, env-free v1(改 gemm1.py 时改这行,
            # 否则 flydsl 缓存 key 不变、继续跑旧 GMM1)
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
                x_q,
                job,
                lane,
                wave,
                # use_nt: 权重拷贝的 non-temporal 提示。BM=32 时整份 w1
                # (616MB) 放不进 LLC,但每个 (expert, n_block) 权重块会被
                # 该 expert 的 5 个 m_block 连着重读,NT 会把这段复用丢掉。
                # 隔离微基准: NT=True 363us / NT=False 237us。
                True,
                fx.Int32(layout.source_capacity),
                fx.Int32(max_route_tiles // G),
                (
                    (local_addr("tile_group_perm")
                        if G > 1
                        else local_addr("tile_dst_of_src"))
                ),
                nc[0] if nc is not None else fx.Int64(0),
                local_addr("gmm1_group_list") if nc is not None else fx.Int64(0),
                nc[1] if nc is not None else fx.Int32(0),
                nc[2] if nc is not None else fx.Int32(0),
                BM=GBM,
                expert_major=False,
                ascale_gather=True,
                BN=GBN,
                BK=BK,
                inline_quant=False,
                K=HIDDEN,
                N_OUT=2 * INTER,
                NE=LOCAL_EXPERTS,
                interleave=False,
                act=activation,
                swiglu_limit=swiglu_limit,
                # Match fused_moe's FlyDSL defaults and the Kimi K3 reference.
                # The standalone legacy GEMM helper defaults to 4/25 instead.
                situ_beta=1.0,
                situ_linear_beta=1.0,
                next_claim=nc is not None,
                next_claim_lds_dw=NC_MBOX_DW,
                next_claim_lead=gmm1_next_claim_lead,
            )

        # ------------------------------------------------------------------
        # GMM1 consumers.
        # ------------------------------------------------------------------
        # GMM1 消费者本来只有 ticket >= ESSENTIAL_CTAS 的 120 个,8 个 comm
        # 和 128 个 producer 干完自己的活就退出,整个 GMM1 阶段(实测 797us,
        # 占 stage1 的 46%)里 136/256 个 CTA 是空的。所有 CTA 本来就是在
        # EOS 门口一起放行的,所以让它们全部参与不改变任何顺序。
        # 记账((consumer_index, consumer_count) 闭式)本来就是通用的。
        is_compute = (
            fx.Int32(1) == fx.Int32(1)
            if const_expr(COMPUTE_FIRST == 0)
            else ticket >= fx.Int32(COMPUTE_FIRST)
        )
        for _xt in _XC:
            is_compute = is_compute | (ticket == fx.Int32(_xt))
        if is_compute:
            if const_expr(early_local_gmm):
                # F5:两段动态领 job。本地段:过 h1_local_eos(段 1 发布,只依赖 node 内 fan1)
                # 后用本地头领 [0, 本地组数*GNB);远端段:过 h1_queue_eos 后用远端头领
                # [0, 远端组数*GNB),组号表偏移本地组数。各段越界的那次领取直接丢弃。
                # 领取都在看到对应门之后,初始化者的清零经 launch_ready→fan1→comm_eos→
                # 段 1 的链先行。门口一律 s_sleep 退避(F4:无退避的轮询把封尾拖慢 5x)。
                # 槽:3=过本地门 5=本地段做完 6=过全局门(4 由下面的原收尾写)。
                el_sl = 127
                el_scr = fx.recast_iter(fx.Int32, lds_raw)
                el_view = fx.make_view(el_scr, fx.make_layout(1, 1))
                el_list = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("gmm1_group_list")
                )
                if tx == fx.Int32(0):
                    (comm_ops.spin_until_ge_i64_sleep_rx)(
                        local_addr("h1_local_eos"), generation, el_sl
                    )
                gpu.barrier()
                comm_ops.fence_system_acquire()
                if const_expr(gmm1_next_claim):
                    # 首领(过门后,不能提前):tx0 领位 + 本段上界 + 组表映射,写信箱。
                    # 之后每个 job 的下一领在 GMM1 体内发,job 后那一把 barrier 同时
                    # 发布信箱,省掉循环顶的 barrier + 领位 + 广播 barrier。
                    el_mbox = fx.add_offset(el_scr, NC_MBOX_DW)  # 按元素(i32)计
                    el_mview = fx.make_view(el_mbox, fx.make_layout(2, 1))
                    if tx == fx.Int32(0):
                        el_nj = fx.Int32(
                            comm_ops.atomic_add_agent(
                                local_addr("gmm1_job_head"), fx.Int32(1)
                            )
                        )
                        el_nn = (
                            fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("tile_alloc") + fx.Int64(4)
                                )
                            )
                            // fx.Int32(G)
                        ) * fx.Int32(GNB)
                        el_nv = el_nj < el_nn
                        el_ng = el_nv.select(el_nj // fx.Int32(GNB), fx.Int32(0))
                        el_np = buffer_ops.buffer_load(
                            el_list, el_ng, vec_width=1, dtype=T.i32
                        )
                        fx.ptr_store(
                            Vec.from_elements(
                                [
                                    el_nv.select(
                                        el_np * fx.Int32(GNB) + el_nj - el_ng * fx.Int32(GNB),
                                        fx.Int32(-1),
                                    ),
                                    el_nn,
                                ],
                                fx.Int32,
                            ),
                            el_mbox,
                        )
                    gpu.barrier()
                    el_nbound = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(Vec(el_mview.load())[1])))
                    el_na = fx.Int32(1) == fx.Int32(1)
                    while el_na:
                        el_njob = fx.Int32(
                            rocdl.readfirstlane(
                                T.i32, fx.Int32(Vec(el_mview.load())[0])
                            )
                        )
                        el_nhas = el_njob >= fx.Int32(0)
                        if el_nhas:
                            _run_gemm1_job(
                                el_njob,
                                (local_addr("gmm1_job_head"), el_nbound, fx.Int32(0)),
                            )
                            gpu.barrier()
                        el_na = el_nhas
                else:
                    el_la = fx.Int32(1) == fx.Int32(1)
                    while el_la:
                        gpu.barrier()
                        if tx == fx.Int32(0):
                            el_lj = fx.Int32(
                                comm_ops.atomic_add_agent(
                                    local_addr("gmm1_job_head"), fx.Int32(1)
                                )
                            )
                            el_ln = (
                                fx.Int32(
                                    comm_ops.load_i32_global_system(
                                        local_addr("tile_alloc") + fx.Int64(4)
                                    )
                                )
                                // fx.Int32(G)
                            ) * fx.Int32(GNB)
                            fx.ptr_store(
                                Vec.from_elements(
                                    [(el_lj < el_ln).select(el_lj, fx.Int32(-1))],
                                    fx.Int32,
                                ),
                                el_scr,
                            )
                        gpu.barrier()
                        el_ljob = fx.Int32(Vec(el_view.load())[0])
                        # dword0 同时是 A 槽 0 的第 0 行:全体 wave 读完 job 号之前,不能让任何 wave 的
                        # A DMA(序章写槽 0)覆盖它,否则慢 wave 读到激活字节,各 wave 对 has 的判断分叉 → barrier 死锁。
                        gpu.barrier()
                        el_lhas = el_ljob >= fx.Int32(0)
                        if el_lhas:
                            el_lg = el_ljob // fx.Int32(GNB)
                            el_lp = buffer_ops.buffer_load(
                                el_list, el_lg, vec_width=1, dtype=T.i32
                            )
                            _run_gemm1_job(
                                fx.Int32(
                                    rocdl.readfirstlane(
                                        T.i32,
                                        el_lp * fx.Int32(GNB) + el_ljob - el_lg * fx.Int32(GNB),
                                    )
                                )
                            )
                            gpu.barrier()
                        el_la = el_lhas
                if tx == fx.Int32(0):
                    (comm_ops.spin_until_ge_i64_sleep_rx)(
                        local_addr("h1_queue_eos"), generation, el_sl
                    )
                gpu.barrier()
                comm_ops.fence_system_acquire()
                if const_expr(gmm1_next_claim):
                    el_rbox = fx.add_offset(el_scr, NC_MBOX_DW)  # 按元素(i32)计
                    el_rview = fx.make_view(el_rbox, fx.make_layout(4, 1))
                    if tx == fx.Int32(0):
                        el_mj = fx.Int32(
                            comm_ops.atomic_add_agent(
                                local_addr("gmm1_job_head") + fx.Int64(16 * 4),
                                fx.Int32(1),
                            )
                        )
                        el_mlg = fx.Int32(
                            comm_ops.load_i32_global_system(
                                local_addr("tile_alloc") + fx.Int64(4)
                            )
                        ) // fx.Int32(G)
                        el_mn = (
                            fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("tile_alloc")
                                )
                            )
                            // fx.Int32(G)
                            - el_mlg
                        ) * fx.Int32(GNB)
                        el_mo = el_mlg * fx.Int32(GNB)
                        el_mv = el_mj < el_mn
                        el_mjj = el_mj + el_mo
                        el_mg = el_mv.select(el_mjj // fx.Int32(GNB), fx.Int32(0))
                        el_mp = buffer_ops.buffer_load(
                            el_list, el_mg, vec_width=1, dtype=T.i32
                        )
                        fx.ptr_store(
                            Vec.from_elements(
                                [
                                    el_mv.select(
                                        el_mp * fx.Int32(GNB) + el_mjj - el_mg * fx.Int32(GNB),
                                        fx.Int32(-1),
                                    ),
                                    el_mn,
                                    el_mo,
                                    fx.Int32(0),
                                ],
                                fx.Int32,
                            ),
                            el_rbox,
                        )
                    gpu.barrier()
                    el_rvals = Vec(el_rview.load())
                    el_mbound = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(el_rvals[1])))
                    el_moff = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(el_rvals[2])))
                    el_ma = fx.Int32(1) == fx.Int32(1)
                    while el_ma:
                        el_mjob = fx.Int32(
                            rocdl.readfirstlane(
                                T.i32, fx.Int32(Vec(el_rview.load())[0])
                            )
                        )
                        el_mhas = el_mjob >= fx.Int32(0)
                        if el_mhas:
                            _run_gemm1_job(
                                el_mjob,
                                (
                                    local_addr("gmm1_job_head") + fx.Int64(16 * 4),
                                    el_mbound,
                                    el_moff,
                                ),
                            )
                            gpu.barrier()
                        el_ma = el_mhas
                else:
                    el_ra = fx.Int32(1) == fx.Int32(1)
                    while el_ra:
                        gpu.barrier()
                        if tx == fx.Int32(0):
                            el_rj = fx.Int32(
                                comm_ops.atomic_add_agent(
                                    local_addr("gmm1_job_head") + fx.Int64(16 * 4),
                                    fx.Int32(1),
                                )
                            )
                            el_rlg = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("tile_alloc") + fx.Int64(4)
                                )
                            ) // fx.Int32(G)
                            el_rn = (
                                fx.Int32(
                                    comm_ops.load_i32_global_system(
                                        local_addr("tile_alloc")
                                    )
                                )
                                // fx.Int32(G)
                                - el_rlg
                            ) * fx.Int32(GNB)
                            fx.ptr_store(
                                Vec.from_elements(
                                    [(el_rj < el_rn).select(el_rj + el_rlg * fx.Int32(GNB), fx.Int32(-1))],
                                    fx.Int32,
                                ),
                                el_scr,
                            )
                        gpu.barrier()
                        el_rjob = fx.Int32(Vec(el_view.load())[0])
                        # dword0 同时是 A 槽 0 的第 0 行:全体 wave 读完 job 号之前,不能让任何 wave 的
                        # A DMA(序章写槽 0)覆盖它,否则慢 wave 读到激活字节,各 wave 对 has 的判断分叉 → barrier 死锁。
                        gpu.barrier()
                        el_rhas = el_rjob >= fx.Int32(0)
                        if el_rhas:
                            el_rg = el_rjob // fx.Int32(GNB)
                            el_rp = buffer_ops.buffer_load(
                                el_list, el_rg, vec_width=1, dtype=T.i32
                            )
                            _run_gemm1_job(
                                fx.Int32(
                                    rocdl.readfirstlane(
                                        T.i32,
                                        el_rp * fx.Int32(GNB) + el_rjob - el_rg * fx.Int32(GNB),
                                    )
                                )
                            )
                            gpu.barrier()
                        el_ra = el_rhas
            if tx == fx.Int32(0):
                # 融合版里 248 个 CTA 做完 fanout 就停在这里等全局封尾;
                # 退避轮询,不和仍在 fanout/发 credit 的 CTA 抢访存。
                (comm_ops.spin_until_ge_i64_sleep_rx)(
                    local_addr("h1_queue_eos"), generation, 127
                )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            # 融合版:GMM1 消费者(ticket>=COMPUTE_FIRST,从不写 T0 专用的 3..10 槽)
            # 3 = 过 h1_queue_eos 门,4 = 自己的 job 做完。
            tiles = buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_alloc")
                ),
                fx.Int32(0),
                vec_width=1,
                dtype=T.i32,
            )
            # early_local_gmm:job 已在上面两段领完,这里只剩 sortcopy + 完成计数(每 CTA +1)。
            total_jobs = (tiles // fx.Int32(G)) * fx.Int32(
                0 if early_local_gmm else GNB
            )
            consumer_index = (
                (ticket - fx.Int32(COMPUTE_FIRST))
            )
            for _xi, _xt in enumerate(_XC):
                consumer_index = (ticket == fx.Int32(_xt)).select(
                    fx.Int32(worker_blocks - COMPUTE_FIRST + _xi), consumer_index
                )
            consumer_count = (
                fx.Int32(N_CONSUMERS)
            )
            # 诊断:消费者在做任何 job 之前要先等 h1_queue_eos(全部 8 个
            # 通信角色 EOS)。这个戳把 flush_post->done 切成
            # 「fanout/等 EOS」与「GMM1 jobs」两段。单一写者:consumer 0 的 tx0。
            # 原 sealer 的逐行拷贝:每个 m_block 组(GBM 行)由一个消费者拷。
            # 目的行与 sealer 同一公式(按 tile 的 tile_dst_of_src)。可见性由
            # 下面本 CTA 的 release + h1_compute_done acq_rel 覆盖,publisher
            # 等齐全部 job 后才发 stage1_done。
            sc_perm = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_dst_of_src")
            )
            sc_src = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_source")
            )
            sc_wts = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_weight")
            )
            sc_src_o = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_source_sorted")
            )
            sc_wts_o = buffer_ops.create_buffer_resource_from_addr(
                local_addr("tile_row_weight_sorted")
            )
            for sc_g in range(
                consumer_index, tiles // fx.Int32(G), consumer_count
            ):
                if tx < fx.Int32(GBM):
                    sc_row = sc_g * fx.Int32(GBM) + tx
                    sc_tile = sc_row // fx.Int32(BM)
                    sc_dst = (
                        buffer_ops.buffer_load(
                            sc_perm, sc_tile, vec_width=1, dtype=T.i32
                        )
                        * fx.Int32(BM)
                        + sc_row % fx.Int32(BM)
                    )
                    buffer_ops.buffer_store(
                        buffer_ops.buffer_load(
                            sc_src, sc_row, vec_width=1, dtype=T.i32
                        ),
                        sc_src_o,
                        sc_dst,
                    )
                    buffer_ops.buffer_store(
                        buffer_ops.buffer_load(
                            sc_wts, sc_row, vec_width=1, dtype=T.i32
                        ),
                        sc_wts_o,
                        sc_dst,
                    )
            # 每 job 一次 s_waitcnt(0)(全流水排空)+ release fence
            # (自带整 L2 写回)+ system acq_rel 原子 —— 这正是 stage2
            # AR57 修掉的簿记粒度(546.8 -> 245.7us),以及
            # arrival-flag 协议里 release 每条自带 buffer_wbl2 的同一个病。
            # 6720 个 job 下实测每 job 182us,而一个 job 只有 58.7 MFLOP。
            #
            # 计数语义不变:finisher 等的是总数 == tiles*h1_n_blocks,
            # 所以一次加上本 CTA 的 job 数即可。**不能在运行期 for 循环里
            # 跨迭代累加标量**(AST rewriter 会把循环体抽成函数,跨迭代
            # 标量不可靠),所以 job 数用闭式算,不靠累加。
            # 循环内保留 gpu.barrier():它管的是 LDS 复用的 CTA 内序,
            # 与这里要批量化的系统级 ordering 无关。
            for job in range(
                consumer_index, total_jobs, consumer_count
            ):
                _run_gemm1_job(job)
                gpu.barrier()
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                _remaining = total_jobs - consumer_index
                _my_jobs = (_remaining > fx.Int32(0)).select(
                    (_remaining + consumer_count - fx.Int32(1))
                    // consumer_count,
                    fx.Int32(0),
                )
                if const_expr(early_local_gmm):
                    _my_jobs = fx.Int32(1)
                comm_ops.fence_system_release()
                comm_ops.atomic_add_system_acq_rel(
                    local_addr("h1_compute_done"), _my_jobs
                )

        if is_publisher:
            if tx == fx.Int32(0):
                tiles = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                    fx.Int32(0),
                    vec_width=1,
                    dtype=T.i32,
                )
                expected_jobs = (
                    tiles // fx.Int32(G)
                ) * fx.Int32(GNB)
                if const_expr(early_local_gmm):
                    # 每个 GMM1 消费者 CTA 退出时 +1。
                    expected_jobs = fx.Int32(N_CONSUMERS)
                completed = fx.Int32(0)
                while completed < expected_jobs:
                    completed = fx.Int32(
                        _pl32(local_addr("h1_compute_done"))
                    )
                comm_ops.fence_system_acquire()
                comm_ops.fence_system_release()
                comm_ops.store_i64_global_system(
                    stage2_addr("stage1_done"), generation
                )

    @flyc.jit
    def launch_megamoe_tile_ep16_stage1(
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
        generation: fx.Int64,
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
            generation,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(
            grid=(worker_blocks, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch_megamoe_tile_ep16_stage1.kernel_name = kernel_name
    launch_megamoe_tile_ep16_stage1.device_generation = bool(device_generation)
    launch_megamoe_tile_ep16_stage1.activation = activation
    launch_megamoe_tile_ep16_stage1.layout = layout
    launch_megamoe_tile_ep16_stage1.stage2_layout = stage2_layout
    launch_megamoe_tile_ep16_stage1.stage2_window_offset = stage2_window_offset
    launch_megamoe_tile_ep16_stage1.worker_blocks = worker_blocks
    launch_megamoe_tile_ep16_stage1.lds_bytes = lds_bytes
    launch_megamoe_tile_ep16_stage1.essential_ctas = ESSENTIAL_CTAS
    launch_megamoe_tile_ep16_stage1.gemm1_contraction = True
    launch_megamoe_tile_ep16_stage1.expert_major_output = True
    launch_megamoe_tile_ep16_stage1.cco_logical_doorbells = 2 * layout.num_qp * dispatch_chunks
    launch_megamoe_tile_ep16_stage1.single_gpu_launch = True
    launch_megamoe_tile_ep16_stage1.requires_resident_grid = True
    launch_megamoe_tile_ep16_stage1.architecture_contract = {
        "dispatch": "scoreboard_direct_to_expert_tile",
        "receive_comm_roles": 8,
        "cross_node_comm_roles": 1,
        "intra_node_comm_roles": 7,
        "allocation_counter": "alloc_count",
        "arrival_counter": "tile_arrived",
        "eos_tail": True,
        "uses_rank_inbox": False,
        "uses_source_activation_inbox": True,
        "uses_group_sort": False,
        "cross_node_dedup": "one_record_per_token_per_node",
        "destination_rank_payload": "one_source_indexed_activation_row_per_rank",
        "rank_route_encoding": "u16_topk_slot_mask_per_global_rank",
        "queue_publication": "post_8_role_eos_physical_major",
        "gmm_scheduler": f"post_eos_static_strided_{worker_blocks - ESSENTIAL_CTAS}_pure_compute_consumers",
        "input_scale_layout": "bm32_ku_ikxdl_klane_nlane_ima",
        "input_format": "mxfp4_e8m0_1x32",
        "producer_ctas": int(PRODUCER_CTAS),
        "compute_first_ticket": int(COMPUTE_FIRST),
        "fanout_shards": int(fanout_shards),
        "split_local": bool(split_local),
    }
    launch_megamoe_tile_ep16_stage1.output_regions = {
        name: layout.region(name).offset
        for name in (
            "h1_output_q",
            "h1_output_scale",
            "tile_expert",
            "tile_row_base",
            "num_valid",
            "tile_row_input",
            "tile_row_source",
            "tile_row_weight",
        )
    }
    return launch_megamoe_tile_ep16_stage1


__all__ = ["compile_megamoe_tile_ep16_stage1"]
