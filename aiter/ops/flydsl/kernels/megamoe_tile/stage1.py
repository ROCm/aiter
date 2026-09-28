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
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from . import comm_ops
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_workgroup as _atomic_add_wg,
)
from .gemm_common import MXFP4_SCALE_LAYOUT_TAG, k_tiles_total_for
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
import os as _os_rail
from .gda_rail import checked as _rail_checked, release as _rail_release

_RAIL_NDEBUG = _os_rail.environ.get("MEGAMOE_TK_S1_RAIL_NDEBUG", "0") != "0"
_rail = _rail_release if _RAIL_NDEBUG else _rail_checked
from .stage1_abi import (
    SPARSE_QP_GENERATION_SHIFT,
    SPARSE_QP_TOKEN_BITS,
    Stage1ArenaLayout,
)
from .stage2_abi import STAGE2_TIMELINE_INDEX


BM = 32
BN = 256
BK = 256
THREADS = 256
WAVES = THREADS // 64
# flydsl 的 cache key 只哈希源码,环境变量改了名字不变就会复用旧二进制
# ([[kernel 同一性看依赖闭包]])。把自旋退避值烧进 kernel 名字。
import os as _os_spin_tag
_SPIN_SLEEP_TAG = int(_os_spin_tag.environ.get("MEGAMOE_TK_SPIN_SLEEP", "0") or 0)
CCO_TICKET = 0

TEAM_RAIL = "rail"

# Diagnostic setup and primary kernels can share one non-parity epoch gate in
# consecutive launches.  Reserve a full three-bit phase field so every phase
# remains distinct across adjacent generations.
DIAGNOSTIC_PHASE_IDS = {
    "full": 0,
    "transport_only": 2,
    "fanout_only": 3,
    "dispatch_only": 5,
}
DIAGNOSTIC_CONTROL_GENERATION_BITS = 3


def _plane_bytes(layout: Stage1ArenaLayout, name: str) -> int:
    region = layout.region(name)
    if not region.shape or region.shape[0] != layout.parity_depth:
        raise ValueError(f"{name} is not parity indexed")
    return region.nbytes // layout.parity_depth


def compile_megamoe_tile_ep16_stage1(
    layout: Stage1ArenaLayout,
    stage2_layout,
    *,
    rank: int,
    stage2_window_offset: int = 0,
    worker_blocks: int = 512,
    work_shards: int = 8,
    waves_per_eu_hint: int = 2,
    enable_cco: bool = True,
    diagnostic_comm_only: bool = False,
    diagnostic_split_fanout: bool = False,
    diagnostic_wave_fanout: bool = False,
    diagnostic_no_arrival_rmw: bool = False,
    cco_chunks_per_flush: int = 1,
    rail_serial_doorbell: bool = False,
    cco_geometry: str = "chunked",
    diagnostic_phase: str = "full",
    tile_pipeline: bool = False,
    tile_pipeline_instrument: bool = False,
    tile_pipeline_fanout_shards: int = 16,
    timeline_instrument: bool = False,
    gmm1_batch_completion: bool = True,
    fanout_shards: int = 1,
    cco_defer_reciprocal_wait: bool = False,
    cco_hoist_staging_wait: bool = False,
    wave_fanout: bool = False,
    producer_ctas: int = 0,
    wide_staging_wait: bool = False,
    wide_fanout_wait: bool = False,
    compute_first: int = -1,
    lean_waitcnt: bool = False,
    fan1_direct: bool = False,
    t0_no_fanout: bool = False,
    credit_async: bool = False,
    rail_soa: bool = False,
    rail_qps: int = 1,
    rail_post_ctas: bool = False,
    route_gbatch: bool = False,
    ascale_gather: bool = False,
    rail_post_off_t0: bool = False,
    sortcopy_k2: bool = False,
    gb_p3: int = 0,
    rail_ids_first: bool = False,
    meta_opt: int = 0,
    split_local: bool = False,
    h1_phys: bool = False,
    k2_sorted_jobs: bool = False,
    gmm1_bn: int = 0,
    gate_sleep: int = 0,
    finisher_off_t0: bool = False,
    early_local_gmm: bool = False,
    fan2_shards: int = 0,
    post_nofan: bool = False,
    remote_rev: bool = False,
    local_defer: int = 0,
    seal_fast: bool = False,
    lazy_pad: bool = False,
    pub_relaxed: bool = False,
    claim_relaxed: bool = False,
    gmm1_use_nt: bool = True,
    gmm_tile_group: int = 0,
    dispatch_plan: bool = False,
    # 1=完整(算计数+等齐+算 plan);2=只发计数,不等不算。用来把挂死
    # 切成「发送端」和「等待端」两半(受控变体只能差一件事)。
    plan_stage: int = 1,
    plan_rail_ctx: int = 0,
    plan_spin_cycles: int = 0,
    claim_agent_probe: bool = False,
    fanout_probe: int = 0,
    spin_deadline_cycles: int = 0,
    live_mark: bool = False,
    split_fanout_only: int = 0,
    kernel_role: str = "fused",
    kernel_split_at: str = "source",
    launches_per_forward: int = 1,
    expert_major_output: bool = False,
    activation: str = "silu",
    device_generation: bool = False,
    swiglu_limit: float | None = None,
):
    """Compile one EP16 Stage-1 persistent-kernel shape specialization.

    ``arena_ptr`` is the base of the one registered two-kernel window.
    ``stage2_window_offset`` locates the logical :class:`Stage2ArenaLayout`
    inside that same window.  The launcher performs exactly one GPU launch.
    """

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
    if bool(dispatch_plan) != bool(layout.dispatch_plan):
        raise ValueError("dispatch_plan must match the arena layout")
    G = int(layout.tile_group)
    # 诊断用:GEMM 侧的分组可以和认领分组解耦。GG=1 表示「按组认领,但 GMM1
    # 仍按单个 tile 做 m_block」—— 用来把「认领/补齐」和「GEMM 侧接线」
    # 这两件事分开验(受控变体只能差一件事)。
    GG = G if int(gmm_tile_group) <= 0 else int(gmm_tile_group)
    if G % GG:
        raise ValueError("gmm_tile_group must divide tile_group")
    GBM = BM * GG
    # 一次认领覆盖的行数跟「认领分组 G」走,和 GEMM 的 M tiling(GBM,跟 GG 走)
    # 是两件事。诊断变体 GG=1 时两者不再相等,不能混用。
    GROUP_ROWS = BM * G
    # 规划者用最高 ticket,避开生产者池(见下面的断言)。
    PLAN_TICKET = int(worker_blocks) - 1
    # plan_stage: 1=完整 2=只发(自己敲门铃) 4=只发(不敲,搭 token 的门铃)
    #             5=完整(不敲)
    _plan_do_wait = int(plan_stage) in (1, 5)
    # 6=只做本地(直方图+LSA 写计数),不发 rail;7=只做直方图,连 LSA 写都不做。
    # 9=plan 块一行代码都不发,只保留 ABI 里多出来的 8 个 region。
    _plan_emit = int(plan_stage) != 9
    # 10=planner 只等 node 内 8 个源,完全不碰跨节点那一跳(诊断用)。
    _plan_wait_rail = int(plan_stage) != 10
    # >0 时 planner 的两处自旋改为限时:超时就把现场写进 plan_debug、
    # 记一次 error 并继续,让 kernel 正常退出而不是挂死到 host 超时。
    _PLAN_SPIN = int(plan_spin_cycles)
    _SPIN_DEADLINE = int(spin_deadline_cycles)
    _LIVE_MARK = bool(live_mark)
    # 竞态分诊:0=两组都开,1=只开 inter,2=只开 intra。两组单开都不崩、
    # 只有同开才崩 = 归属划分冲突。数值会错(数据缺一半),只看崩不崩。
    _SPLIT_ONLY = int(split_fanout_only)
    _PLAN_WAIT_SRCS = GPUS_PER_NODE if int(plan_stage) == 10 else WORLD
    _plan_send_local = int(plan_stage) != 7
    _plan_send_rail = int(plan_stage) not in (6, 7)
    _plan_own_doorbell = int(plan_stage) in (1, 2)
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
    if G > 1 and (tile_pipeline or diagnostic_no_arrival_rmw):
        # tile_pipeline 按单个 BM 行 tile 提前发布 job,no_arrival_rmw 按
        # ceil(count/BM) 回填 tile_row_done —— 两者都是 tile 粒度的,
        # 和「G 个 tile 合成一个 m_block」不相容。
        raise ValueError("tile_group>1 requires tile_pipeline=False and "
                         "diagnostic_no_arrival_rmw=False")
    if BN != 256 or BM not in (32, 64, 128):
        raise ValueError("Stage-1 requires BN=256 and BM in (32, 64, 128)")
    if activation not in ("silu", "situv2"):
        raise ValueError("Stage-1 activation must be silu or situv2")
    if swiglu_limit is not None and (activation != "silu" or not float(swiglu_limit) > 0.0):
        raise ValueError("Stage-1 swiglu_limit applies to silu only and must be positive")
    if device_generation and diagnostic_phase != "full":
        raise ValueError("device generation requires full Stage-1")
    # agent: token capacity来自arena specialization；producer CTA池保持固定，
    # 避免大batch把resident grid扩张到硬件无法同时驻留的规模。
    MAX_TOKENS = int(layout.max_tokens)
    COMM_CTAS = GPUS_PER_NODE
    PRODUCER_FIRST = COMM_CTAS
    # 量化池原本写死 128:512 个 token 由 128 个 CTA 各做 4 个,另外 128 个
    # CTA 在 fanout 之前完全空转。staging 段实测占 stage1 的 38%
    # ([[stage1 四段分解]]),所以池子大小是个直接旋钮。0 = 保持原默认。
    PRODUCER_CTAS = min(128, MAX_TOKENS)
    if producer_ctas:
        PRODUCER_CTAS = min(int(producer_ctas), MAX_TOKENS)
    ESSENTIAL_CTAS = PRODUCER_FIRST + PRODUCER_CTAS
    # GMM1 消费者的起始 ticket。默认 ESSENTIAL_CTAS(=136),即 8 个 comm 和
    # 128 个 producer 干完自己的活就退出,GMM1 阶段 136/256 个 CTA 空转。
    # 实测把它降到 0(全员参与)GMM1 段确实快了 30%(797->555us),但总时间
    # 大幅回归:comm CTA 还欠着 rail 的 credit/reclaim,而 finisher 要等
    # h1_compute_done 收满,于是 GMM1 的收尾被 rail 拆链拖住。
    # 所以它是个**边界**而不是开关:8 = 放 producer 进来但不放 comm。
    # 两 kernel 拆分。边界按 record 的**来源**切,不是按 dispatch/GMM1 切:
    #   k1 = 打包(输入已在 kernel 外量化) + rail 通信 + 本节点 token 的 node 内 push
    #   k2 = rail 来源 token 的 node 内 push + EOS/封尾/置换 + GMM1 + activation
    # 跨 kernel 的状态全在 persistent window 里,generation 管代际,不加 ABI。
    # 今天 dispatch 和 GMM1 本来就严格串行(tile_pipeline 恒 False),所以拆开
    # 不损失任何 overlap,只多一次 launch。
    if kernel_role not in ("fused", "k1", "k2", "k0"):
        raise ValueError("kernel_role must be one of fused, k1, k2, k0")
    if kernel_split_at not in ("source", "gmm", "noop"):
        raise ValueError("kernel_split_at must be source, gmm or noop")
    # k0:只做 plan 的计数交换,在 k1 之前单独发射。它不碰 k1 建立的任何状态,
    # 所以也不能等 epoch_gate —— 那个 gate 由 k1 的 initializer 写,k0 等它就是
    # 等一个还没启动的 kernel。放在 k1 里才会成环,见 planner-in-k1-deadlocks。
    _IN_K0 = kernel_role == "k0"
    _FUSED = kernel_role == "fused"
    # 纯探针:把 fanout 里每条 route 一次的跨设备领位原子换成本地 agent 原子。
    # 数值必然错(各源领到重叠的行),只用来量这一行的时间代价上界。
    _CLAIM_AGENT_PROBE = bool(claim_agent_probe)
    # fanout 簿记探针(数值必错,只量时间):
    #   1 = 跳过非 group-head 那条「跨设备自旋等 tile map + 读回 physical」
    #   2 = 去掉每条 route 结尾的 s_waitcnt(0)
    #   3 = 跳过 scale 的 112 次单字节 swizzle store
    _FANOUT_PROBE = int(fanout_probe)
    _IN_K1 = kernel_role in ("fused", "k1")
    _IN_K2 = kernel_role in ("fused", "k2")
    # 诊断(配合 MEGAMOE_TK_S1_ONLY + ATT):k1 每个 CTA 记下 ticket 与物理位置。
    _DIAG_HWID = (
        __import__("os").environ.get("MEGAMOE_TK_S1_HWID", "0") != "0"
        and kernel_role == "k1"
    )
    # 诊断(配合 MEGAMOE_TK_S1_ONLY,不开 ATT):k1 每个 CTA 在关键点用 s_memrealtime
    # (100MHz,全芯片统一)打时间戳,另把每条 record 的分段耗时累加在 LDS 里。
    # 写到独立输出 ts_out(host 分配,[worker_blocks, 32] int64),最后一次 replay 覆盖。
    # 槽位含义见 graph.py 的 MEGAMOE_S1_TS 导出。
    _DIAG_TS = (
        __import__("os").environ.get("MEGAMOE_TK_S1_TSTAMP", "0") != "0"
        and kernel_role in ("k1", "fused")
    )
    TS_SLOTS = 32
    TS_LDS = 1024
    # 两个切点:
    #   source = 按 record 来源切(k2 拿 rail 来源的 fanout + EOS/credit + GMM1)
    #   gmm    = 只把 GMM1 切出去(k1 做完整 dispatch),两 kernel 之间零协议穿越
    # 后者用来二分「协议穿越」和「GMM1 单独成 kernel」这两类问题。
    # noop = 决定性的单变量诊断:k1 干**全部**的活(等价于 fused),k2 是个
    # 什么都不做的空 launch。两个切点都挂,而 gmm 那个是零协议穿越的,所以
    # 共同因素只剩「同一个 persistent kernel 被 launch 两次」。这个模式把
    # 「双次 launch 的机制(ticket/epoch 协议)」和「活怎么分」彻底分开。
    _NOOP = kernel_split_at == "noop"
    _LATE = _IN_K2 if kernel_split_at == "source" else _IN_K1
    _do_produce = _IN_K1
    _do_fan1 = _IN_K1
    _do_fan2 = _LATE
    _do_eos = _LATE
    _do_credit = _LATE
    _do_seal = _LATE
    _do_compute = _IN_K1 if _NOOP else _IN_K2
    _do_publish = _IN_K1 if _NOOP else _IN_K2
    _K1_TOK = MAX_TOKENS if _do_fan1 else 0
    _K2_TOK = MAX_TOKENS if _do_fan2 else 0

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
    # split fanout 本身不要求 sparse_wqe(那是 tile_pipeline 的要求),它只是
    # 被这条门限连坐。producer_token 是 strided 推进的(+= PRODUCER_CTAS),
    # tokens > CTAs 本来就能覆盖,所以这里只保留另外两项。
    if MAX_TOKENS > 128 and cco_geometry == "sparse_wqe":
        raise ValueError(
            "max_tokens > 128 currently requires chunked transport"
        )
    if not 0 <= int(rank) < WORLD:
        raise ValueError("rank must be in [0, 16)")
    if worker_blocks < ESSENTIAL_CTAS:
        raise ValueError(
            f"worker_blocks must be >= {ESSENTIAL_CTAS} so all progress roles are resident"
        )
    if (worker_blocks == ESSENTIAL_CTAS and not diagnostic_comm_only
            and not diagnostic_split_fanout and diagnostic_phase == "full"):
        # Producers retire without rejoining in the chunked full path. With
        # zero GEMM consumers the finisher would wait forever for h1_compute_done.
        raise ValueError("full Stage-1 requires at least one GEMM consumer after progress roles")
    if work_shards not in (1, 2, 4, 8):
        raise ValueError("work_shards must be one of 1,2,4,8")
    if waves_per_eu_hint not in (1, 2, 3, 4):
        raise ValueError("waves_per_eu_hint must be one of 1,2,3,4")
    if not enable_cco:
        raise ValueError("strict EP16 Stage-1 requires real CCO transport")
    if diagnostic_split_fanout:
        if worker_blocks != 256:
            raise ValueError(
                "internodev1 split fanout requires worker_blocks=256"
            )
    if diagnostic_wave_fanout:
        if not diagnostic_split_fanout:
            raise ValueError(
                "diagnostic_wave_fanout requires split fanout mode"
            )
    if diagnostic_no_arrival_rmw:
        if not diagnostic_comm_only or not diagnostic_split_fanout:
            raise ValueError(
                "diagnostic_no_arrival_rmw requires split comm-only mode"
            )
    if cco_chunks_per_flush not in (1, 2, 4, 8, 16, 32, 64):
        # 实测门铃成本 13.8us/次且与载荷无关(与 kernel2 台账里独立测到的
        # 13.5~14.9us 吻合),而门铃不能并发敲 —— 所以唯一的杠杆是减少次数。
        # 下限是每 QP 一次。放开到 32/64 是为了把整轮 chunk 攒进一次门铃。
        raise ValueError(
            "cco_chunks_per_flush must be one of 1,2,4,8,16,32,64"
        )
    if cco_geometry not in ("chunked", "mori64x2", "sparse_wqe"):
        raise ValueError(
            "cco_geometry must be chunked, mori64x2, or sparse_wqe"
        )
    if diagnostic_phase not in (
        "full",
        "transport_only",
        "fanout_only",
        "dispatch_only",
    ):
        raise ValueError(
            "diagnostic_phase must be full, transport_only, fanout_only, "
            "or dispatch_only"
        )
    if diagnostic_phase != "full":
        if not diagnostic_comm_only or not diagnostic_split_fanout:
            raise ValueError(
                "non-full diagnostic_phase requires split comm-only mode"
            )
    if cco_geometry == "mori64x2":
        if not diagnostic_split_fanout:
            raise ValueError("mori64x2 geometry requires split fanout")
    if cco_geometry == "sparse_wqe":
        if not diagnostic_split_fanout or diagnostic_phase != "full":
            raise ValueError(
                "sparse_wqe requires split fanout with full phase"
            )
        if diagnostic_wave_fanout:
            raise ValueError(
                "sparse_wqe initially requires the CTA fanout path"
            )
    if tile_pipeline:
        if (
            diagnostic_comm_only
            or not diagnostic_split_fanout
            or diagnostic_wave_fanout
            or cco_geometry != "sparse_wqe"
            or diagnostic_phase != "full"
            or worker_blocks != 256
        ):
            raise ValueError(
                "tile_pipeline requires the full real-GMM1 256-CTA "
                "split sparse_wqe path"
            )
        if work_shards != 8:
            raise ValueError("tile_pipeline requires work_shards=8")
    if tile_pipeline_instrument and not tile_pipeline:
        raise ValueError("tile_pipeline_instrument requires tile_pipeline")
    if expert_major_output and tile_pipeline:
        # The permutation needs final expert counts, which only exist
        # after the eight comm EOS values; the streaming scheduler
        # publishes GMM1 jobs before that point.
        raise ValueError(
            "expert_major_output requires the post-EOS GMM1 scheduler "
            "(tile_pipeline=False)"
        )
    # Chunked transport records entry/completion; dispatch-flush markers are
    # available only on the sparse tile pipeline and remain zero otherwise.
    if timeline_instrument and diagnostic_phase != "full":
        raise ValueError(
            "timeline_instrument requires full Stage1"
        )
    if int(tile_pipeline_fanout_shards) not in (8, 12, 16):
        raise ValueError("tile_pipeline_fanout_shards must be 8, 12, or 16")

    rank = int(rank)
    local_rank = rank % GPUS_PER_NODE
    node = rank // GPUS_PER_NODE
    remote_node = 1 - node
    remote_source_rank = remote_node * GPUS_PER_NODE + local_rank
    stage2_window_offset = int(stage2_window_offset)
    # 默认路径:每条 record 的 route 并行做(_dispatch_record_batched),系统序
    # 在 shard 末尾一次发布。tile_pipeline(sparse_wqe)要逐 route 的
    # tile_row_done 到达计数提前发布满 tile,只能走逐 route 的旧路径。
    route_batch = not tile_pipeline
    # rail_soa 的 SoA 排布(发送端 dispatch_staging[p] 与接收端 remote_dispatch_rx[p] 共用)。
    from .rail_record_quant import rail_record_layout as _rail_record_layout

    REC_BYTES, REC_Q, REC_S, REC_I, REC_W = _rail_record_layout(HIDDEN, TOPK)
    # rail_ids_first:发送/接收平面在 record 区之后放一段连续的 [ids|weights] 数组
    # (每 token TOPK*8 B),先于 payload 单独 PUT。fan2 拿到 ids 就能领行/写元数据,
    # 与 payload 的 RDMA 传输重叠。dispatch_staging/remote_dispatch_rx 每平面
    # MAX_TOKENS*record_bytes,record 只用 MAX_TOKENS*REC_BYTES,尾部空着。
    SIDE_OFF = MAX_TOKENS * REC_BYTES
    SIDE_REC = TOPK * 8
    RAIL_QPS = int(rail_qps)
    if t0_no_fanout and (wave_fanout or fanout_shards < 2 or diagnostic_split_fanout):
        raise ValueError("t0_no_fanout needs the CTA-granular sharded fanout path")
    if credit_async and cco_geometry != "chunked":
        raise ValueError("credit_async is implemented for the chunked geometry only")
    if fan1_direct and not route_batch:
        raise ValueError("fan1_direct needs the batched route path (tile_pipeline=False)")
    if stage2_window_offset < 0 or stage2_window_offset % 4096:
        raise ValueError("stage2_window_offset must be non-negative and 4096-byte aligned")

    wire = layout.wire
    record_bytes = wire.record_bytes
    if rail_post_ctas and not rail_soa:
        raise ValueError("rail_post_ctas needs rail_soa")
    # rail_post_off_t0:QP0 也交给发帖 CTA(ticket q+1 发 QP q)。TSNP28:T0 的其余 wave
    # 同时在自旋轮询到达字,T0 自己发的 put 要 ~40µs,发帖 CTA 同样的 put ~8µs。
    if rail_post_off_t0 and not rail_post_ctas:
        raise ValueError("rail_post_off_t0 needs rail_post_ctas")
    # sortcopy_k2:tile_row_source/weight -> *_sorted 的逐行拷贝从 sealer(单 CTA,ATT14
    # EPLB 下 ~35µs、在 k1 关键路径上)挪到 k2 的 GMM1 消费者,按组分摊。
    if rail_ids_first:
        if not (rail_soa and route_gbatch and int(gb_p3) == 2 and rail_post_ctas):
            raise ValueError("rail_ids_first needs rail_soa, route_gbatch, gb_p3=2 and rail_post_ctas")
        if SIDE_REC % 16 or SIDE_OFF + MAX_TOKENS * SIDE_REC > MAX_TOKENS * record_bytes:
            raise ValueError("rail_ids_first side array does not fit the record plane")
        if cco_geometry != "chunked":
            raise ValueError("rail_ids_first borrows sparse_remote_qp_ready; chunked geometry only")
    if meta_opt and not (route_gbatch and int(gb_p3) == 2):
        raise ValueError("meta_opt refines the gb_p3=2 group-base resolve")
    # split_local:本地来源行(fan1)和远端来源行(fan2)各自计数、各自成组,sealer 分两段封尾:
    # 段 1 在 8 个 local_eos 后封本地组(h1 段基址 0),段 2 在全部 comm_eos 后封远端组。
    # h1_phys:GMM1 按物理组写 h1(不再按 expert-major 置换写),封尾额外写逆置换
    # tile_src_of_dst,stage2 按它间接读 A。排序视图与物理位置解耦,本地组可先算。
    if h1_phys and not (expert_major_output and not tile_pipeline):
        raise ValueError("h1_phys needs expert_major_output on the post-EOS scheduler")
    # k2_sorted_jobs:GMM1 的 job 按排序后(expert-major)的组号遍历,A 仍从物理组读、h1 仍写物理组。
    # 同一 expert 的 m 块相邻 ⇒ 并发执行,权重的第二次读命中 LLC(小算子 gemm1 的顺序)。
    if k2_sorted_jobs and not h1_phys:
        raise ValueError("k2_sorted_jobs maps sorted groups through tile_src_of_dst (needs h1_phys)")
    # early_local_gmm(F5,融合版):段 1 封好本地组即发布 h1_local_eos + 物理组表
    # gmm1_group_list[本地组..|远端组..];消费者先用本地头领本地 job(与 rail/fan2/段 2 重叠),
    # 过全局门后再用远端头领远端 job。每个消费者退出时 h1_compute_done +1。
    if early_local_gmm and not (
        kernel_role == "fused" and split_local and h1_phys and not k2_sorted_jobs
        and gmm1_batch_completion and not diagnostic_split_fanout
    ):
        raise ValueError("early_local_gmm needs the fused role with split_local+h1_phys and batch completion")
    # fan2_shards(F5):每个 dest 只有 fanout_shard < fan2_shards 的 fan CTA 做 fan2(按新步长
    # 分 token),其余 fan CTA 做完 fan1 直接进 GMM1 本地段,把等 rail 的空档让给本地组。
    # 完成计数不变:每个 fan CTA 仍在自己的最后一步给 fanout_shard_done[dest] +1。
    F2S = int(fan2_shards) if fan2_shards else 0
    if F2S and not (
        early_local_gmm and route_gbatch and wide_fanout_wait and fanout_shards > 1
        and F2S <= fanout_shards - (1 if t0_no_fanout else 0)
    ):
        raise ValueError("fan2_shards needs early_local_gmm, route_gbatch, wide_fanout_wait and <= fanout shards")
    # post_nofan:rail 发帖 CTA(ticket _POST_LO.._POST_HI,即 dest=ticket 的 0 号分片)先发帖、
    # fan1 晚起跑 ~55µs,拖住 local_eos。它们仍留在 fan 块里(要在发布分片前等自己 QP 的完成,
    # 见 rail_post_ctas 段)、仍计入 fanout_shard_done,但不分 token:这些 dest 的 token 步长
    # 变成 shards-1,其余分片号减 1。
    if local_defer and not (early_local_gmm and h1_phys and 0 < local_defer < layout.local_experts):
        raise ValueError("local_defer needs early_local_gmm + h1_phys and 0 < D < local_experts")
    # lazy_pad:段 1 不补本地组的尾行就发布 h1_local_eos,改由段 2 的 sealer 补。GMM1 只经
    # tile_row_input 取 A,且是带界 buffer load;补齐前那些行里是上一代/初值的合法行号,
    # 算出来的 h1 行只会被 stage2 用 INVALID_SOURCE/权重 0 丢掉,而这两项在 stage1_done 前补齐。
    if lazy_pad and not (early_local_gmm and h1_phys and split_local):
        raise ValueError("lazy_pad needs early_local_gmm + h1_phys + split_local")
    if seal_fast and not (early_local_gmm and h1_phys and not diagnostic_no_arrival_rmw and not dispatch_plan):
        raise ValueError("seal_fast needs early_local_gmm + h1_phys (parallel pad/perm skip the diagnostic and plan branches)")
    if post_nofan and not (
        rail_post_ctas and route_gbatch and fanout_shards > 1 and not wave_fanout and fan1_direct
    ):
        raise ValueError("post_nofan needs rail_post_ctas, route_gbatch, fan1_direct and sharded fanout")
    if early_local_gmm and GG != G:
        raise ValueError("early_local_gmm lists claim groups (G tiles) as GMM1 m-blocks; needs gmm_tile_group == G")
    if split_local and not (
        route_gbatch and rail_soa and fan1_direct and expert_major_output
        and not tile_pipeline and not diagnostic_no_arrival_rmw and not dispatch_plan
        and fanout_shards > 1 and not diagnostic_split_fanout
    ):
        raise ValueError(
            "split_local needs route_gbatch+rail_soa+fan1_direct+expert_major on the post-EOS scheduler"
        )
    if sortcopy_k2 and not (expert_major_output and not tile_pipeline):
        raise ValueError("sortcopy_k2 needs expert_major_output on the post-EOS GMM1 scheduler")
    _POST_T0 = 0 if rail_post_off_t0 else 1      # 第一个发帖 CTA 发的 QP = ticket - (1 - _POST_T0)
    _POST_LO, _POST_HI = 1, (RAIL_QPS + 1 if rail_post_off_t0 else RAIL_QPS)
    # 段 1 放在哪个 fan CTA(ticket = 8*_SEG1_SHARD + local_rank)。t0_no_fanout 时 dest0 的
    # 分片号整体减 1,所以 rank0 要再往后挪一格才在 fan2 池外。
    _SEG1_SHARD = 1
    if F2S:
        _s1 = F2S + (
            1
            if (t0_no_fanout and local_rank == 0)
            or (post_nofan and _POST_LO <= local_rank < _POST_HI)
            else 0
        )
        if _s1 < fanout_shards:
            _SEG1_SHARD = _s1
    if route_gbatch and not rail_soa:
        raise ValueError("route_gbatch reads AoS rail records; needs rail_soa")
    # ascale_gather:A-scale 按源 token 行主序写(每 (token,dest) 一次 16B 级连续写),
    # GMM1 按 tile_row_input gather 并在 LDS 里拼预排布。只有 route_gbatch 路径这么写。
    if ascale_gather and not route_gbatch:
        raise ValueError("ascale_gather needs route_gbatch (the only writer of source-major scales)")
    if rail_soa:
        if REC_BYTES > record_bytes:
            raise ValueError("rail record does not fit one dispatch_staging record slot")
        if not (fan1_direct and cco_geometry == "chunked" and wide_fanout_wait
                and cco_defer_reciprocal_wait and not wave_fanout
                and not diagnostic_split_fanout and route_batch):
            raise ValueError("rail_soa needs fan1_direct, chunked geometry, "
                             "wide_fanout_wait, deferred reciprocal wait and the batched route path")
        if rail_post_ctas and (diagnostic_split_fanout or wave_fanout):
            raise ValueError("rail_post_ctas needs tickets 1..RAIL_QPS-1 on the sharded fanout path")
        if RAIL_QPS not in (1, 2, 4) or RAIL_QPS > int(layout.num_qp):
            raise ValueError("rail_qps must be 1, 2 or 4 and <= num_qp")
    record_dwords = record_bytes // 4
    payload_dwords = wire.payload_bytes // 4
    scale_bytes = wire.scale_bytes
    scale_dwords = scale_bytes // 4
    records_per_chunk = wire.records_per_chunk
    dispatch_chunks = layout.dispatch_chunks
    if dispatch_chunks % cco_chunks_per_flush:
        raise ValueError("cco_chunks_per_flush must divide dispatch_chunks")
    records_per_qp = records_per_chunk // layout.num_qp
    qp_bytes = records_per_qp * record_bytes
    max_route_tiles = layout.max_route_tiles
    max_route_rows = layout.max_route_rows
    max_tiles_per_expert = layout.max_tiles_per_expert
    h1_n_blocks = layout.h1_n_blocks
    max_jobs = max_route_tiles * h1_n_blocks
    # GMM1 自己的 N 块宽(与 arena 的 block_n 解耦;h1 按行存,N 块只决定哪个 CTA 写哪些列)。
    # 128 = 小算子 t128x128 的 N;job 数按 GNB 算。tile_pipeline 的按 job 队列仍按 arena 的 n_blocks。
    GBN = int(gmm1_bn) if gmm1_bn else BN
    if GBN not in (128, 256) or (GBN != BN and tile_pipeline):
        raise ValueError("gmm1_bn must be 128/256 and 128 is post-EOS scheduler only")
    GNB = (2 * INTER) // GBN
    # 非 tile_pipeline 下原本写死 16 => 两组各 128 CTA,正好占满 256,不给
    # producer 留位置。chunked 下 producer 是独立 ticket
    # 区间,占满就必然重叠成环。这里跟着 fanout_shards 走,由上面的编译期
    # 断言保证 producer 池之后放得下两组。
    split_fanout_shards = (
        int(tile_pipeline_fanout_shards) if tile_pipeline else int(fanout_shards)
    )
    split_fanout_ctas = GPUS_PER_NODE * split_fanout_shards
    dedicated_compute_ctas = worker_blocks - 2 * split_fanout_ctas

    kh_tile = BK // 2
    k_tiles_total = k_tiles_total_for(HIDDEN, BK)
    # LDS 按 GMM1 实际的 M tiling 算,不是 arena 的 tile 粒度:G>1 时
    # _gemm1_body 以 BM*G 行跑,累加器要 BM*G*BN*4 字节。按 BM 算会
    # 静默写越界(G=2 实测 relL2=1.55)。
    _, _, _, lds_bytes = _bm_constants(GBM, BN, kh_tile, k_tiles_total)
    # MEGAMOE_TK_GMM1_NEXT_CLAIM=1(只在 early_local_gmm 下生效):GMM1 体内提前领下一个 job,
    # 映射好的 job 号经 LDS 信箱(GMM1 LDS 之后多分配的 16B,dword0=job,1=本段上界,2=本段偏移)
    # 交回领取循环。lead = 在倒数第几个 K 步开头发领位原子。
    gmm1_next_claim = bool(early_local_gmm) and (
        __import__("os").environ.get("MEGAMOE_TK_GMM1_NEXT_CLAIM", "1") != "0"
    )
    gmm1_next_claim_lead = (
        int(__import__("os").environ.get("MEGAMOE_TK_GMM1_NEXT_CLAIM_LEAD", "2"))
        if gmm1_next_claim
        else 2
    )
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
    timeline_history_depth = stage2_layout.timeline_history_depth
    if timeline_history_depth and not timeline_instrument:
        raise ValueError("timeline history requires timeline instrumentation")
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

    transport_tag = "cco" if enable_cco else "stub"
    kernel_name = (
        f"megamoe_tile_ep16_stage1_k3_a4w4_{activation}"
        + ("" if swiglu_limit is None else f"_swl{float(swiglu_limit):g}".replace(".", "p"))
        + f"_r{rank}_"
        f"mt{MAX_TOKENS}_rpc{layout.max_routes_per_token_per_rank}_"
        f"wb{worker_blocks}_ws{work_shards}_{transport_tag}_{MXFP4_SCALE_LAYOUT_TAG}_"
        "scoreboard_v14_tilequeue_rankslots_payload2_qpballot4_no_send_atomic_inputscale_abi3"
        + (f"_h{HIDDEN}" if HIDDEN != 7168 else "")
        + (f"_i{INTER}" if INTER != 3072 else "")
        + (f"_e{EXPERTS}" if EXPERTS != 896 else "")
        + (f"_k{TOPK}" if TOPK != 16 else "")
        + ("_devgen" if device_generation else "")
        + ("_diagnostic_comm_only" if diagnostic_comm_only else "")
        + (
            (
                "_internodev1_split128x2_grid256"
                if diagnostic_comm_only
                else (
                    f"_split{split_fanout_ctas}x2_tilepipe{worker_blocks}r"
                    if tile_pipeline
                    else "_split128x2_rejoin256_posteos"
                )
            )
            if diagnostic_split_fanout
            else ""
        )
        + ("_wave64x1_fullgrid" if diagnostic_wave_fanout else "")
        + ("_no_arrival_rmw" if diagnostic_no_arrival_rmw else "")
        + ("_wfan" if wave_fanout else "")
        + (f"_prod{PRODUCER_CTAS}" if producer_ctas else "")
        + ("_widewait" if wide_staging_wait else "")
        + ("_widefan" if wide_fanout_wait else "")
        + (f"_cf{COMPUTE_FIRST}" if compute_first >= 0 else "")
        + ("_leanwc" if lean_waitcnt else "")
        + ("_f1d" if fan1_direct else "")
        + ("_t0nf" if t0_no_fanout else "")
        + ("_cra" if credit_async else "")
        + (f"_soa{int(rail_qps)}" if rail_soa else "")
        + ("_pcta" if rail_post_ctas else "")
        + ("_gb" if route_gbatch else "")
        + ("_asg" if ascale_gather else "")
        + ("_pofft0" if rail_post_off_t0 else "")
        + ("_sck2" if sortcopy_k2 else "")
        + (f"_gp{int(gb_p3)}" if gb_p3 else "")
        + ("_idsf" if rail_ids_first else "")
        + (f"_mo{int(meta_opt)}" if meta_opt else "")
        + ("_slg" if split_local else "")
        + ("_h1p" if h1_phys else "")
        + ("_k2sj" if k2_sorted_jobs else "")
        + (f"_gbn{int(gmm1_bn)}" if gmm1_bn and int(gmm1_bn) != BN else "")
        + (f"_gs{int(gate_sleep)}" if gate_sleep else "")
        + ("_fo" if finisher_off_t0 else "")
        + ("_elg" if early_local_gmm else "")
        + (f"_f2s{int(fan2_shards)}" if fan2_shards else "")
        + ("_pnf" if post_nofan else "")
        + ("_rr" if remote_rev else "")
        + (f"_ld{int(local_defer)}" if local_defer else "")
        + ("_sf" if seal_fast else "")
        + ("_lp" if lazy_pad else "")
        + ("_prx" if pub_relaxed else "")
        + ("_crx" if claim_relaxed else "")
        + (f"_bm{BM}" if BM != 32 else "")
        + (f"_tg{G}" if G != 1 else "")
        + ("_plan" if dispatch_plan else "")
        + (f"_ps{plan_stage}" if dispatch_plan and plan_stage != 1 else "")
        + (f"_pctx{plan_rail_ctx}" if dispatch_plan else "")
        + (f"_spin{plan_spin_cycles}" if _PLAN_SPIN > 0 else "")
        + ("_claimprobe" if _CLAIM_AGENT_PROBE else "")
        + (f"_fp{_FANOUT_PROBE}" if _FANOUT_PROBE else "")
        + (f"_dl{spin_deadline_cycles}" if _SPIN_DEADLINE else "")
        + (f"_only{_SPLIT_ONLY}" if _SPLIT_ONLY else "")
        + (f"_gg{GG}" if GG != G else "")
        + ("" if gmm1_use_nt else "_gmmcached")
        + (f"_{kernel_role}" if kernel_role != "fused" else "")
        + (
            f"_at{kernel_split_at}"
            if kernel_role != "fused" and kernel_split_at != "source"
            else ""
        )
        + (
            f"_spinz{_SPIN_SLEEP_TAG}"
            if _SPIN_SLEEP_TAG
            else ""
        )
        + (
            f"_cco_flushb{cco_chunks_per_flush}"
            if cco_chunks_per_flush != 1 and cco_geometry == "chunked"
            else ""
        )
        + ("_cco_mori64x2" if cco_geometry == "mori64x2" else "")
        + ("_cco_sparse_wqe" if cco_geometry == "sparse_wqe" else "")
        + (
            f"_phase_{diagnostic_phase}"
            if diagnostic_phase != "full"
            else ""
        )
        + ("_overlapstats" if tile_pipeline_instrument else "")
        + ("_timeline" if timeline_instrument else "")
        + ("_expertmajor" if expert_major_output else "")
        + ("_perjobdone" if not gmm1_batch_completion else "")
        + (f"_fos{fanout_shards}" if fanout_shards != 1 else "")
        + ("_defrecv" if cco_defer_reciprocal_wait else "")
        + ("_hoistwait" if cco_hoist_staging_wait else "")
        + ("_serialdb" if rail_serial_doorbell else "")
        + ("_ndbgrail" if _RAIL_NDEBUG else "")
        + (f"_history{timeline_history_depth}" if timeline_history_depth else "")
        + (
            "_qpstream_intergate"
            if cco_geometry == "sparse_wqe"
            else ""
        )
    )

    if _DIAG_HWID:
        kernel_name = kernel_name + "_hwid"
    if _DIAG_TS:
        kernel_name = kernel_name + "_ts"
    # gemm1.py 读同一个开关;flydsl 缓存 key 不含 env,名字区分两份二进制。
    if __import__("os").environ.get("MEGAMOE_TK_GMM1_LDS_SCOPES", "1") != "0":
        kernel_name = kernel_name + "_lsc"
        if __import__("os").environ.get("MEGAMOE_TK_GMM1_NOFENCE_BAR", "1") != "0":
            kernel_name = kernel_name + "_nfb"
    # gemm1.py 尾声累加器去 bank 冲突排布(只对 BN=256 生效;LDS 由 _bm_constants 同步加大)。
    if BN == 256 and __import__("os").environ.get("MEGAMOE_TK_GMM1_EPI_SWZ", "1") != "0":
        kernel_name = kernel_name + "_esw"
    if gmm1_next_claim:
        kernel_name = kernel_name + f"_nxc{gmm1_next_claim_lead}"

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
        ts_out: fx.Int64,
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
            # 于是 generation 是在 device 上从到达票推出来的。entry_count 每个
            # CTA 加一,所以**每次 forward 发几个 kernel**必须算进分母:
            # 拆成两个 kernel 后,k1 拿 [0,256) -> gen 1,而 k2 拿 [256,512)
            # -> gen 2,k2 于是去等一个 k1 永远不会发布的代际,整条链死锁。
            # 这就是拆分版连 noop 模式都挂的根因。
            generation = (
                ticket64
                // fx.Int64(worker_blocks * launches_per_forward)
                + fx.Int64(1)
            )
        parity = fx.Int64(generation & fx.Int64(1))
        phase_id = DIAGNOSTIC_PHASE_IDS[diagnostic_phase]
        control_generation = (
            generation
            if const_expr(diagnostic_phase == "full")
            else (
                (
                    generation
                    << fx.Int64(DIAGNOSTIC_CONTROL_GENERATION_BITS)
                )
                + fx.Int64(phase_id)
            )
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
        timeline_addr = stage2_addr("timeline")
        if const_expr(timeline_history_depth > 0):
            # Only the timestamp address changes; no extra graph operation or
            # host synchronization is inserted between forward replays.
            timeline_addr = (
                arena_ptr + fx.Int64(stage2_window_offset + s2_off("timeline_history"))
                + (generation & fx.Int64(timeline_history_depth - 1))
                * fx.Int64(stage2_layout.region("timeline_history").nbytes // timeline_history_depth)
            )

        # Arrival tickets, rather than block IDs, ensure every progress role is
        # among the first resident CTAs on an oversubscribed persistent grid.
        ticket = fx.Int32(ticket64 % fx.Int64(worker_blocks))
        is_initializer = ticket == fx.Int32(0)
        if const_expr(_DIAG_HWID):
            # 给 ATT 把被 trace 的 wave 对回 ticket(角色)。写在 stage2 的 node_accumulator:
            # S1_ONLY 下 stage2 不跑,这块没人用,也不改 arena 布局。
            # 布局 [generation%64][ticket] x {gen_lo, blockIdx, HW_ID, XCC_ID}(u32)。
            if tx == fx.Int32(0):
                hwid_res = buffer_ops.create_buffer_resource_from_addr(
                    arena_ptr
                    + fx.Int64(stage2_window_offset + s2_off("node_accumulator"))
                )
                hwid_base = (
                    fx.Int32(generation & fx.Int64(63)) * fx.Int32(worker_blocks)
                    + ticket
                ) * fx.Int32(4)
                buffer_ops.buffer_store(fx.Int32(generation), hwid_res, hwid_base)
                buffer_ops.buffer_store(
                    fx.Int32(gpu.block_id("x")), hwid_res, hwid_base + fx.Int32(1)
                )
                buffer_ops.buffer_store(
                    fx.Int32(comm_ops.read_hw_id()), hwid_res, hwid_base + fx.Int32(2)
                )
                buffer_ops.buffer_store(
                    fx.Int32(comm_ops.read_xcc_id()), hwid_res, hwid_base + fx.Int32(3)
                )

        def _ts_val(slot, value):
            if const_expr(_DIAG_TS):
                if tx == fx.Int32(0):
                    comm_ops.store_i64_global_relaxed(
                        ts_out
                        + fx.Int64(ticket) * fx.Int64(TS_SLOTS * 8)
                        + fx.Int64(slot * 8),
                        fx.Int64(value),
                    )

        def _ts(slot):
            if const_expr(_DIAG_TS):
                _ts_val(slot, comm_ops.read_wall_clock())

        def _acc_ptr(k):
            return fx.add_offset(
                fx.recast_iter(fx.Int32, lds_raw), fx.Int32((TS_LDS + k) * 4)
            )

        def _acc_load(k):
            return Vec(fx.make_view(_acc_ptr(k), fx.make_layout(1, 1)).load())[0]

        def _acc_add(k, delta):
            # 只由 tx0 调用(调用方已在 tx==0 分支里)。
            fx.ptr_store(
                Vec.from_elements([_acc_load(k) + fx.Int32(delta)], fx.Int32),
                _acc_ptr(k),
            )

        def _acc_zero():
            if const_expr(_DIAG_TS):
                if tx == fx.Int32(0):
                    for k in range_constexpr(8):
                        fx.ptr_store(
                            Vec.from_elements([fx.Int32(0)], fx.Int32), _acc_ptr(k)
                        )

        def _acc_flush():
            if const_expr(_DIAG_TS):
                if tx == fx.Int32(0):
                    for k in range_constexpr(8):
                        _ts_val(20 + k, fx.Int64(_acc_load(k)))

        _ts_val(0, generation)
        _ts(1)
        if not _IN_K1:
            # k2 绝不能重跑初始化:那一段会把 k1 刚建好的 expert_count /
            # tile_alloc / remote_chunk_consumed / fanout_shard_done 全部清零。
            # k2 里 ticket0 因此走 else 分支去 acquire epoch_gate —— 那个值
            # k1 已经用同一个 control_generation 写好了,自旋立即通过。
            is_initializer = fx.Int32(0) == fx.Int32(1)
        if const_expr(timeline_instrument):
            if is_initializer & (tx == fx.Int32(0)):
                comm_ops.store_i64_global_relaxed(
                    timeline_addr
                    + fx.Int64(
                        STAGE2_TIMELINE_INDEX["stage1_entry"] * 8
                    ),
                    fx.Int64(comm_ops.read_wall_clock()),
                )
        transport_enabled = diagnostic_phase in (
            "full",
            "transport_only",
            "dispatch_only",
        )
        fanout_enabled = diagnostic_phase in (
            "full",
            "fanout_only",
            "dispatch_only",
        )
        is_cco = (ticket == fx.Int32(CCO_TICKET)) & (
            fx.Int32(1 if transport_enabled else 0) == fx.Int32(1)
        )
        if const_expr(diagnostic_split_fanout):
            # The tile pipeline can tune 8/12/16 shards per destination.
            # With fewer than 16, the unused fanout tickets wait directly on
            # the ready queue; with 16, all roles finish their own short
            # fanout slice and immediately rejoin as GMM1 consumers.  The
            # post-EOS reference retains the original 128+128 mapping.
            # ticket 0 是唯一的 CCO 传输 CTA(CCO_TICKET=0)。让它同时当
            # inter fanout worker 就成环:它在 fanout 里等的正是自己该去传的
            # chunk。sparse_wqe 路径没这问题(传输不由单个 CTA 做),chunked 有。
            # 两段 fanout 必须整体排在 producer 区间之后。原来 inter 从 1 起、
            # intra 从 128 起,与 producer [PRODUCER_FIRST, +PRODUCER_CTAS) 重叠
            # 57 个 CTA —— 一个 CTA 同时背 producer 和 fanout 两个角色,只要它
            # 在自己还欠 producer 份额时先进了 fanout 的等待就成环。非 split
            # 路径下 is_comm 只占 ticket 0..7,两者天然互斥,所以从没暴露。
            _SPLIT_BASE = PRODUCER_FIRST + PRODUCER_CTAS
            if _SPLIT_BASE + 2 * split_fanout_ctas > worker_blocks:
                raise ValueError(
                    "split fanout needs %d CTAs after the producer pool but "
                    "worker_blocks=%d (reduce fanout_shards)"
                    % (_SPLIT_BASE + 2 * split_fanout_ctas, worker_blocks)
                )
            inter_ticket = (ticket >= fx.Int32(_SPLIT_BASE)) & (
                ticket < fx.Int32(_SPLIT_BASE + split_fanout_ctas)
            )
            intra_ticket = (
                ticket >= fx.Int32(_SPLIT_BASE + split_fanout_ctas)
            ) & (ticket < fx.Int32(_SPLIT_BASE + 2 * split_fanout_ctas))
            # producer_token = ticket - PRODUCER_FIRST,所以低于 PRODUCER_FIRST
            # 的 ticket 会拿到**负的** token 索引,而 producer_active 里的
            #  对负数照样成立 —— 那几个 CTA 会去量化
            # 不存在的 token,对应 staging 永远写不出来,等它的人就永远等不到。
            # 起点区间必须正好是一个步长宽:producer_token = ticket -
            # PRODUCER_FIRST,循环按 += PRODUCER_CTAS 推进。上限写成 MAX_TOKENS
            # 会让起点区间(248)比步长(128)宽 —— token 覆盖既重叠又有缺口,
            # 一部分 token 永远不被量化,等 staging_ready 的人全部死等。
            producer_ticket = (ticket >= fx.Int32(PRODUCER_FIRST)) & (
                ticket < fx.Int32(PRODUCER_FIRST + PRODUCER_CTAS)
            )
            phase_fanout = fx.Int32(1 if fanout_enabled else 0)
            fanout_pred = phase_fanout == fx.Int32(1)
            is_inter_fanout = inter_ticket & fanout_pred
            is_intra_fanout = intra_ticket & fanout_pred
            if const_expr(_SPLIT_ONLY == 2):
                is_inter_fanout = fx.Int32(0) == fx.Int32(1)
            if const_expr(_SPLIT_ONLY == 1):
                is_intra_fanout = fx.Int32(0) == fx.Int32(1)
            is_comm = (inter_ticket | intra_ticket) & fanout_pred
            is_producer = producer_ticket & (
                fx.Int32(1 if diagnostic_phase == "full" else 0) == fx.Int32(1)
            )
        else:
            is_inter_fanout = fx.Int32(0) == fx.Int32(1)
            is_intra_fanout = fx.Int32(0) == fx.Int32(1)
            is_comm = ticket < fx.Int32(COMM_CTAS)
            is_producer = (ticket >= fx.Int32(PRODUCER_FIRST)) & (
                ticket < fx.Int32(PRODUCER_FIRST + PRODUCER_CTAS)
            )

        # rail 的**发送**相位属于 k1,而**信用/回收**相位必须跟着
        # remote_chunk_consumed 的信号走 —— 那个信号由 fanout 循环 2 产生,
        # 在 k2。两者共用一个 is_cco 就会让 k1 等一个永不到达的计数(死锁)。
        _never = fx.Int32(0) == fx.Int32(1)
        # 挂死定位:每个自旋点编号,超时就把 generation 写进 plan_debug[tag]
        # 并放行,让 kernel 正常退出把现场带回 host。默认 0 = 原样死等。
        def _mk_spin(tag):
            def _f(addr, expected):
                if const_expr(_SPIN_DEADLINE > 0):
                    _v = comm_ops.spin_until_ge_i64_bounded(
                        addr, expected, fx.Int64(_SPIN_DEADLINE)
                    )
                    if _v < fx.Int64(expected):
                        comm_ops.store_i64_global_system(
                            local_addr("plan_debug") + fx.Int64(tag * 8),
                            generation,
                        )
                else:
                    comm_ops.spin_until_ge_i64_system(addr, expected)
            return _f
        _spin_dbg_0 = _mk_spin(0)
        _spin_dbg_1 = _mk_spin(1)
        _spin_dbg_2 = _mk_spin(2)
        _spin_dbg_3 = _mk_spin(3)
        _spin_dbg_4 = _mk_spin(4)
        _spin_dbg_5 = _mk_spin(5)
        _spin_dbg_6 = _mk_spin(6)
        _spin_dbg_7 = _mk_spin(7)
        _spin_dbg_8 = _mk_spin(8)
        _spin_dbg_9 = _mk_spin(9)
        _spin_dbg_10 = _mk_spin(10)
        _spin_dbg_11 = _mk_spin(11)
        _spin_dbg_12 = _mk_spin(12)
        _spin_dbg_13 = _mk_spin(13)
        _spin_dbg_14 = _mk_spin(14)
        _spin_dbg_15 = _mk_spin(15)
        _spin_dbg_16 = _mk_spin(16)
        _spin_dbg_17 = _mk_spin(17)
        _spin_dbg_18 = _mk_spin(18)
        _spin_dbg_19 = _mk_spin(19)
        _spin_dbg_20 = _mk_spin(20)
        _spin_dbg_21 = _mk_spin(21)
        _spin_dbg_22 = _mk_spin(22)
        _spin_dbg_23 = _mk_spin(23)
        _spin_dbg_24 = _mk_spin(24)
        _spin_dbg_25 = _mk_spin(25)
        _spin_dbg_26 = _mk_spin(26)
        _spin_dbg_27 = _mk_spin(27)
        _spin_dbg_28 = _mk_spin(28)
        _spin_dbg_29 = _mk_spin(29)
        _spin_dbg_30 = _mk_spin(30)
        _spin_dbg_31 = _mk_spin(31)
        _spin_dbg_32 = _mk_spin(32)
        _spin_dbg_33 = _mk_spin(33)
        _spin_dbg_34 = _mk_spin(34)
        _spin_dbg_35 = _mk_spin(35)
        _spin_dbg_36 = _mk_spin(36)
        _spin_dbg_37 = _mk_spin(37)
        _spin_dbg_38 = _mk_spin(38)
        _spin_dbg_39 = _mk_spin(39)
        _spin_dbg_40 = _mk_spin(40)
        _spin_dbg_48 = _mk_spin(48)

        def _rail_post(post_qp, with_credit):
            """rail_soa:一个 wave 为 QP post_qp 发它那段 token 的 record。

            整段一个 PUT + 本 QP 负责的 ready 字 + 一次门铃,不等完成;请求存进
            remote_chunk_request[post_qp] 由发起方稍后回收。with_credit:先等本 QP
            在 g-2(同 parity)的 credit(T0 自己在外面统一等过)。
            """
            src_base = fx.Int64(off("dispatch_staging"))
            comm_ops.fence_system_release()
            _ts(28)
            if const_expr(with_credit and credit_async):
                if lane == fx.Int32(0):
                    for cchunk in range(
                        fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                    ):
                        _spin_dbg_48(
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
            if const_expr(rail_ids_first):
                # 先发本段 token 的连续 [ids|weights],再置 ids 就绪字(同 QP 内有序)。
                if nr > fx.Int32(0):
                    _rail.put(
                        dev_comm,
                        post_qp,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_dispatch_rx")
                        + fx.Int64(SIDE_OFF)
                        + fx.Int64(r0 * fx.Int32(SIDE_REC)),
                        arena_win,
                        src_base + fx.Int64(SIDE_OFF) + fx.Int64(r0 * fx.Int32(SIDE_REC)),
                        fx.Int64(nr * fx.Int32(SIDE_REC)),
                        aggregate=True,
                    )
                for w in range_constexpr(layout.num_qp):
                    if post_qp == fx.Int32(w % RAIL_QPS):
                        _rail.put_value(
                            dev_comm,
                            post_qp,
                            fx.Int32(remote_node),
                            arena_win,
                            window_off("sparse_remote_qp_ready") + fx.Int64(w * 8),
                            generation,
                            aggregate=True,
                        )
                # aggregate=True 只写 WQE 不敲门铃;不在这里 flush 的话 ids 和 payload 同一次门铃
                # 出发,ids 根本不会先到(ATTIF1:领行起点反而晚 11k 周期)。这个请求不单独回收:
                # 同 QP 完成有序,稍后 wait 数据请求即覆盖它。
                _rail.flush_async(dev_comm, post_qp, fx.Int32(remote_node))
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
            _ts(31)
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
            _ts(29)
            request = _rail.flush_async(dev_comm, post_qp, fx.Int32(remote_node))
            _ts(30)
            if lane == fx.Int32(0):
                comm_ops.store_i64_global_system(
                    local_addr("remote_chunk_request")
                    + fx.Int64(post_qp) * fx.Int64(8),
                    request,
                )
            _ts(5)
        is_cco_send = is_cco if _IN_K1 else _never
        # 规划者必须同时避开两类 ticket:
        #   - CCO:它要持续发 rail token,让它自旋等计数会把发送挡在后面;
        #   - **producer**:低 ticket 都是量化生产者。规划者一旦落在生产者
        #     范围内就会死锁 —— 它在入口等计数,于是它负责的 token 永远不被
        #     量化,CCO 的 token 发送卡在 staging_ready 上,计数永远发不出去,
        #     规划者继续等。实测 status=124 超时就是这条环。
        # 取最高 ticket:fanout 本来就要等 staging,早期空闲,阻塞它无代价。
        if PLAN_TICKET < PRODUCER_FIRST + PRODUCER_CTAS and PLAN_TICKET >= PRODUCER_FIRST:
            raise ValueError(
                "planner ticket %d falls inside the producer pool [%d,%d)"
                % (PLAN_TICKET, PRODUCER_FIRST, PRODUCER_FIRST + PRODUCER_CTAS)
            )
        is_planner = (
            (ticket == fx.Int32(PLAN_TICKET))
            if _FUSED
            else ((ticket == fx.Int32(1)) if _IN_K0 else _never)
        )
        is_cco_credit = is_cco if _do_credit else _never
        if not _do_produce:
            is_producer = _never
        if const_expr(rail_soa and route_gbatch):
            # rail_soa 的 record 由 k1 之前的量化 kernel 直接写进窗口,_rec_store 是空操作,
            # 且 fan1_direct/rail 发送都不读 dispatch_staging_ready。producer 循环只剩
            # 每 token 一次 fence_system_release + ready 写 + 两个 barrier,却让 ticket
            # 8..135 晚 ~60µs 才进 fanout(TSASG2:launch_ok 中位 66µs)。它顺带写的
            # stage2 本地平面掩码改由 _route_gbatch(emit_masks) 在 fan1 里写。
            is_producer = _never

        # 角色到达计数。kernel 挂死时 host 读不到任何"完成后"的东西,但 CCO
        # window 的 local_ptr 是 host 可读的 —— 所以把计数直接打进 arena,
        # host 在 launch 之后不同步地轮询读,就能看出哪个角色一个 CTA 都没到。
        if const_expr(_LIVE_MARK):
            if tx == fx.Int32(0):
                comm_ops.atomic_add_agent(
                    local_addr("plan_debug") + fx.Int64(49 * 8), fx.Int32(1)
                )
                if is_initializer:
                    comm_ops.atomic_add_agent(
                        local_addr("plan_debug") + fx.Int64(50 * 8), fx.Int32(1)
                    )
                if is_cco:
                    comm_ops.atomic_add_agent(
                        local_addr("plan_debug") + fx.Int64(51 * 8), fx.Int32(1)
                    )
                if is_inter_fanout:
                    comm_ops.atomic_add_agent(
                        local_addr("plan_debug") + fx.Int64(52 * 8), fx.Int32(1)
                    )
                if is_intra_fanout:
                    comm_ops.atomic_add_agent(
                        local_addr("plan_debug") + fx.Int64(53 * 8), fx.Int32(1)
                    )
                if is_producer:
                    comm_ops.atomic_add_agent(
                        local_addr("plan_debug") + fx.Int64(54 * 8), fx.Int32(1)
                    )
                if is_comm:
                    comm_ops.atomic_add_agent(
                        local_addr("plan_debug") + fx.Int64(55 * 8), fx.Int32(1)
                    )

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
            if tx < fx.Int32(work_shards):
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
                if const_expr(cco_geometry == "sparse_wqe"):
                    for name in (
                        "sparse_remote_consumed",
                        "sparse_remote_send_count",
                    ):
                        buffer_ops.buffer_store(
                            fx.Int32(0),
                            buffer_ops.create_buffer_resource_from_addr(
                                local_addr(name)
                            ),
                            fx.Int32(0),
                        )
                if ntokens != fx.Int32(MAX_TOKENS):
                    comm_ops.atomic_add_system(error_addr, fx.Int32(1))
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                comm_ops.fence_system_release()
                # pub_relaxed:一次 fence 放行全部清零,两条门值不必各自再做整 L2 写回。
                _pub_st = (
                    comm_ops.store_i64_global_system_relaxed
                    if pub_relaxed
                    else comm_ops.store_i64_global_system
                )
                _pub_st(
                    arena_ptr + fx.Int64(off("epoch_gate")),
                    control_generation,
                )
                _pub_st(
                    local_addr("launch_ready"), control_generation
                )
        else:
            if const_expr(not _IN_K0):
                if tx == fx.Int32(0):
                    _spin_dbg_1(
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
        if const_expr(dispatch_plan):
            _PLAN_SLICE = GPUS_PER_NODE * LOCAL_EXPERTS

            def _plan_peer(dest, name):
                return (
                    arena_lsa_base
                    + fx.Int64(dest) * arena_lsa_stride
                    + fx.Int64(window_off(name))
                )

            # k1 里计数搭 CCO 的发送 CTA;k0 里没有 CCO 角色划分,直接用 ticket 0,
            # 规划者用 ticket 1 —— 两个不同 CTA,免得发送方自己等自己。
            # 拆分模式下 plan 整个属于 k0;k1 里必须全关,否则 k1 的 planner
            # 照样在入口等 16 个源,环原封不动(实测 status=124)。
            _plan_sender = (
                is_cco_send
                if _FUSED
                else ((ticket == fx.Int32(0)) if _IN_K0 else _never)
            )
            _plan_gate = _plan_sender if const_expr(_plan_emit) else _never
            if _plan_gate:
                _hist = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("plan_local_hist")
                )
                for _e in range(tx, fx.Int32(EXPERTS), fx.Int32(THREADS)):
                    buffer_ops.buffer_store(fx.Int32(0), _hist, _e)
                rocdl.s_waitcnt(0)
                gpu.barrier()

                # 全局 expert 直方图。agent 作用域、打在本地 L2 上,和 fanout
                # 里那些打到对端的系统原子不是一回事。
                _ids = buffer_ops.create_buffer_resource_from_addr(topk_ids)
                for _k in range(tx, ntokens * fx.Int32(TOPK), fx.Int32(THREADS)):
                    _e = buffer_ops.buffer_load(_ids, _k, vec_width=1, dtype=T.i32)
                    if (_e >= fx.Int32(0)) & (_e < fx.Int32(EXPERTS)):
                        comm_ops.atomic_add_agent(
                            local_addr("plan_local_hist") + fx.Int64(_e) * fx.Int64(4),
                            fx.Int32(1),
                        )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                comm_ops.fence_agent_acquire()

                if const_expr(_plan_send_local):
                    # node 内 8 个目的端:直接 LSA 写它们的 plan_count[myrank][*]。
                    for _j in range(tx, fx.Int32(_PLAN_SLICE), fx.Int32(THREADS)):
                        _d = _j // fx.Int32(LOCAL_EXPERTS)
                        _le = _j % fx.Int32(LOCAL_EXPERTS)
                        _ge = (
                            fx.Int32(node * GPUS_PER_NODE) + _d
                        ) * fx.Int32(LOCAL_EXPERTS) + _le
                        _c = buffer_ops.buffer_load(_hist, _ge, vec_width=1, dtype=T.i32)
                        buffer_ops.buffer_store(
                            _c,
                            buffer_ops.create_buffer_resource_from_addr(
                                _plan_peer(_d, "plan_count")
                                + fx.Int64(local_rank * LOCAL_EXPERTS * 4)
                            ),
                            _le,
                        )
                    rocdl.s_waitcnt(0)
                    gpu.barrier()

                if const_expr(_plan_send_local or _plan_send_rail):
                    if tx == fx.Int32(0):
                        if const_expr(_plan_send_local):
                            comm_ops.fence_system_release()
                            for _d in range_constexpr(GPUS_PER_NODE):
                                comm_ops.store_i64_global_system(
                                    _plan_peer(fx.Int32(_d), "plan_count_ready")
                                    + fx.Int64(local_rank * 8),
                                    generation,
                                )
                            # 远端 8 个目的端的计数在直方图里天然连续(global expert id
                            # = dest_global*LOCAL_EXPERTS + le,而远端 8 个 rank 编号
                            # 连续),所以不需要打包,直接以那一段为源发出去。
                        if const_expr(_plan_send_rail):
                            _rail.put(
                                dev_comm,
                                fx.Int32(plan_rail_ctx),
                                fx.Int32(remote_node),
                                arena_win,
                                window_off("rail_count_inbox"),
                                arena_win,
                                window_off("plan_local_hist")
                                + fx.Int64(
                                    remote_node * GPUS_PER_NODE * LOCAL_EXPERTS * 4
                                ),
                                fx.Int64(_PLAN_SLICE * 4),
                                aggregate=True,
                            )
                            # 同一个 QP 上 WQE 按序完成,所以标志排在数据之后即可,
                            # 一次门铃带走两条(门铃是 13.8us 的常数,能省则省)。
                            _rail.put_value(
                                dev_comm,
                                fx.Int32(plan_rail_ctx),
                                fx.Int32(remote_node),
                                arena_win,
                                window_off("rail_count_inbox_ready"),
                                generation,
                                aggregate=True,
                            )
                            # 门铃是 13.8us 的硬常数,而且入口处额外敲一次会扰动
                            # token 传输(实测 relL2 0.0 -> 0.0026)。默认不自己敲,
                            # 让计数搭 CCO 本来就要敲的第一次 token 门铃。
                            if const_expr(_plan_own_doorbell):
                                _rail.flush_peer(
                                    dev_comm,
                                    fx.Int32(plan_rail_ctx),
                                    fx.Int32(remote_node),
                                )

            if is_planner and const_expr(_plan_do_wait and _plan_emit):
                _PARTNER = remote_node * GPUS_PER_NODE + local_rank
                if const_expr(_plan_wait_rail):
                    # 伙伴把它发给「我这个 node 的 8 个目的端」的计数一并送来,
                    # 由我在 node 内散开。于是每个目的端的 16 行计数全部由
                    # node 内写入填满 —— 远端源一次都不用再碰。
                    if const_expr(_PLAN_SPIN > 0):
                        if tx == fx.Int32(0):
                            _dbg = local_addr("plan_debug")
                            _v = comm_ops.spin_until_ge_i64_bounded(
                                local_addr("rail_count_inbox_ready"),
                                generation,
                                fx.Int64(_PLAN_SPIN),
                            )
                            comm_ops.store_i64_global_system(_dbg, _v)
                            comm_ops.store_i64_global_system(_dbg + fx.Int64(8), generation)
                            if _v < generation:
                                comm_ops.store_i64_global_system(_dbg + fx.Int64(16), fx.Int64(1))
                                comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                    else:
                        if tx == fx.Int32(0):
                            _spin_dbg_2(
                                local_addr("rail_count_inbox_ready"), generation
                            )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    _inbox = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("rail_count_inbox")
                    )
                    for _j in range(tx, fx.Int32(_PLAN_SLICE), fx.Int32(THREADS)):
                        _d = _j // fx.Int32(LOCAL_EXPERTS)
                        _le = _j % fx.Int32(LOCAL_EXPERTS)
                        _c = buffer_ops.buffer_load(_inbox, _j, vec_width=1, dtype=T.i32)
                        buffer_ops.buffer_store(
                            _c,
                            buffer_ops.create_buffer_resource_from_addr(
                                _plan_peer(_d, "plan_count")
                                + fx.Int64(_PARTNER * LOCAL_EXPERTS * 4)
                            ),
                            _le,
                        )
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    if tx == fx.Int32(0):
                        comm_ops.fence_system_release()
                        for _d in range_constexpr(GPUS_PER_NODE):
                            comm_ops.store_i64_global_system(
                                _plan_peer(fx.Int32(_d), "plan_count_ready")
                                + fx.Int64(_PARTNER * 8),
                                generation,
                            )

                # ---- 目的端 plan:等齐 16 个源的计数,算出每个 (源, expert)
                # 的行基址。补齐到 GROUP_ROWS 的倍数,和成组认领同一个粒度,
                # 这样 GMM1 的 m_block 仍然落在同一个 expert 内。
                if const_expr(_PLAN_SPIN > 0):
                    if tx < fx.Int32(_PLAN_WAIT_SRCS):
                        _v2 = comm_ops.spin_until_ge_i64_bounded(
                            local_addr("plan_count_ready") + fx.Int64(tx) * fx.Int64(8),
                            generation,
                            fx.Int64(_PLAN_SPIN),
                        )
                        comm_ops.store_i64_global_system(
                            local_addr("plan_debug") + fx.Int64(64) + fx.Int64(tx) * fx.Int64(8),
                            _v2,
                        )
                        if _v2 < generation:
                            comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                else:
                    if tx < fx.Int32(_PLAN_WAIT_SRCS):
                        _spin_dbg_3(
                            local_addr("plan_count_ready") + fx.Int64(tx) * fx.Int64(8),
                            generation,
                        )
                gpu.barrier()
                comm_ops.fence_system_acquire()

                _counts = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("plan_count")
                )
                _padded_lds = fx.recast_iter(fx.Int32, lds_raw)
                _my_total = fx.Int32(0)
                if tx < fx.Int32(LOCAL_EXPERTS):
                    for _src in range_constexpr(WORLD):
                        _my_total = _my_total + buffer_ops.buffer_load(
                            _counts,
                            fx.Int32(_src * LOCAL_EXPERTS) + tx,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    _my_padded = (
                        (_my_total + fx.Int32(GROUP_ROWS - 1)) // fx.Int32(GROUP_ROWS)
                    ) * fx.Int32(GROUP_ROWS)
                    fx.ptr_store(
                        Vec.from_elements([_my_padded], fx.Int32),
                        fx.add_offset(_padded_lds, tx * fx.Int32(4)),
                    )
                gpu.barrier()

                if tx < fx.Int32(LOCAL_EXPERTS):
                    # 跨 expert 的排他前缀和。LOCAL_EXPERTS 是编译期常量,
                    # 所以这是一段展开的 LDS 读,不是运行期循环里的标量累加
                    # (那个 AST rewriter 不可靠)。
                    _expert_base = fx.Int32(0)
                    _view = fx.make_view(_padded_lds, fx.make_layout(LOCAL_EXPERTS, 1))
                    for _e in range_constexpr(LOCAL_EXPERTS):
                        _pe = Vec(
                            fx.make_view(
                                fx.add_offset(_padded_lds, fx.Int32(_e * 4)),
                                fx.make_layout(1, 1),
                            ).load()
                        )[0]
                        _expert_base = _expert_base + (fx.Int32(_e) < tx).select(
                            _pe, fx.Int32(0)
                        )
                    _row_base = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("plan_row_base")
                    )
                    _sender_prefix = fx.Int32(0)
                    for _src in range_constexpr(WORLD):
                        buffer_ops.buffer_store(
                            _expert_base + _sender_prefix,
                            _row_base,
                            fx.Int32(_src * LOCAL_EXPERTS) + tx,
                        )
                        _sender_prefix = _sender_prefix + buffer_ops.buffer_load(
                            _counts,
                            fx.Int32(_src * LOCAL_EXPERTS) + tx,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    buffer_ops.buffer_store(
                        _my_total,
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("plan_expert_rows")
                        ),
                        tx,
                    )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        local_addr("plan_ready"), generation
                    )

        # ------------------------------------------------------------------
        # A bounded producer CTA pool owns tokens in a strided schedule.  At
        # capacity 128 this is identical to the historical one-CTA-per-token
        # mapping; larger capacities reuse the same resident CTA for later
        # tokens instead of increasing the resident grid.
        # ------------------------------------------------------------------
        # split fanout 原先把 token 直接写成 ticket,循环每轮重搬同一个
        # token:512 个只落 128 个,且从 ticket 起算(token 0..7 无人写)。
        # 那是 MORI「128 token / 一 CTA 一 token / 单轮」的形状。
        def _rec_store(*args, **kwargs):
            if const_expr(not rail_soa):
                buffer_ops.buffer_store(*args, **kwargs)

        _PROD_STRIDE = PRODUCER_CTAS
        producer_token = (ticket - fx.Int32(PRODUCER_FIRST)) % fx.Int32(
            _PROD_STRIDE
        )
        producer_active = is_producer & (producer_token < ntokens)
        _wdl46 = fx.Int64(comm_ops.read_wall_clock())
        while producer_active:
            if const_expr(_SPIN_DEADLINE > 0):
                _wdl46 = fx.Int64(comm_ops.read_wall_clock())
            token = producer_token
            record_addr = local_addr("dispatch_staging") + fx.Int64(token) * fx.Int64(
                record_bytes
            )
            record_rsrc = buffer_ops.create_buffer_resource_from_addr(
                record_addr, num_records_bytes=record_bytes
            )
            # 输入已由 k1 之外的 per_1x32_mx_quant 量化成 fp4 [T, H/2] +
            # e8m0 [T, H/32](与 MegaMoEv2 同形式),这里只把一行搬进 record。
            group = tx
            group_active = group < fx.Int32(HIDDEN // 32)
            if const_expr(rail_soa):
                # rail_soa:dispatch_staging 当前 parity 是 T0 的 SoA 发送源,
                # producer 不再写 record,只写 stage2 要的本地平面掩码。
                group_active = _never
            if group_active:
                input_q_rsrc = buffer_ops.create_buffer_resource_from_addr(x_q)
                input_scale_rsrc = (
                    buffer_ops.create_buffer_resource_from_addr(input_scale)
                )
                input_q_dw = (
                    token * fx.Int32(HIDDEN // 8) + group * fx.Int32(4)
                )
                packed = buffer_ops.buffer_load(
                    input_q_rsrc,
                    input_q_dw,
                    vec_width=4,
                    dtype=T.i32,
                )
                _rec_store(
                    packed,
                    record_rsrc,
                    group * fx.Int32(4),
                )
                input_e8m0 = buffer_ops.buffer_load(
                    input_scale_rsrc,
                    token * fx.Int32(HIDDEN // 32) + group,
                    vec_width=1,
                    dtype=T.i8,
                )
                _rec_store(
                    input_e8m0,
                    record_rsrc,
                    fx.Int32(wire.payload_bytes) + group,
                    offset_is_bytes=True,
                )

            ids_rsrc = buffer_ops.create_buffer_resource_from_addr(topk_ids)
            weights_rsrc = buffer_ops.create_buffer_resource_from_addr(route_weights)
            if (tx < fx.Int32(TOPK)):
                route = token * fx.Int32(TOPK) + tx
                expert = buffer_ops.buffer_load(ids_rsrc, route, vec_width=1, dtype=T.i32)
                weight = buffer_ops.buffer_load(
                    weights_rsrc, route, vec_width=1, dtype=T.f32
                )
                _rec_store(
                    expert,
                    record_rsrc,
                    fx.Int32(wire.ids_offset // 4) + tx,
                )
                _rec_store(
                    weight,
                    record_rsrc,
                    fx.Int32(wire.weights_offset // 4) + tx,
                )

            # Thread zero builds the node/rank masks used by Stage-2 and by the
            # sparse cross-node sender.  Duplicate destination ranks are legal:
            # the per-rank slot bitmaps below preserve every expert route while
            # the rank mask continues to deduplicate payload placement.
            rank_mask_scratch = fx.recast_iter(fx.Int32, lds_raw)
            rank_mask_view = fx.make_view(
                rank_mask_scratch, fx.make_layout(1, 1)
            )
            rank_mask_lane = fx.Int32(0)
            route_mask_lane = fx.Int64(0)
            route_error = fx.Int32(0)
            local_slot_mask_lane = fx.Int32(0)
            if (tx == fx.Int32(0)):
                for slot in range_constexpr(TOPK):
                    expert = buffer_ops.buffer_load(
                        ids_rsrc,
                        token * fx.Int32(TOPK) + fx.Int32(slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(EXPERTS))
                    owner = valid.select(expert // fx.Int32(LOCAL_EXPERTS), fx.Int32(0))
                    bit = fx.Int32(1) << owner
                    rank_mask_lane = valid.select(rank_mask_lane | bit, rank_mask_lane)
                    route_mask_lane = valid.select(
                        route_mask_lane | (fx.Int64(1) << fx.Int64(slot)), route_mask_lane
                    )
                    owner_is_local = valid & (
                        (owner // fx.Int32(GPUS_PER_NODE)) == node
                    )
                    if const_expr(publish_plane_slots):
                        local_slot_mask_lane = owner_is_local.select(
                            local_slot_mask_lane
                            | (fx.Int32(1) << fx.Int32(slot)),
                            local_slot_mask_lane,
                        )
                    if const_expr(cco_geometry == "sparse_wqe"):
                        invalid_non_padding = (expert < fx.Int32(-1)) | (
                            expert >= fx.Int32(EXPERTS)
                        )
                        route_error = invalid_non_padding.select(
                            route_error + fx.Int32(1), route_error
                        )
                    else:
                        route_error = valid.select(
                            route_error, route_error + fx.Int32(1)
                        )
                if route_error != fx.Int32(0):
                    comm_ops.atomic_add_system(error_addr, route_error)
                _rec_store(
                    fx.Int32(rank * MAX_TOKENS) + token,
                    record_rsrc,
                    fx.Int32(wire.source_offset // 4),
                )
                _rec_store(
                    fx.Int32(0),
                    record_rsrc,
                    fx.Int32(wire.source_offset // 4 + 1),
                )
                _rec_store(
                    fx.Vector.from_elements([route_mask_lane], fx.Int64).bitcast(fx.Int32),
                    record_rsrc,
                    fx.Int32(wire.route_mask_offset // 4),
                )
                # Zero the record tail so validation is deterministic.
                for dword in range_constexpr(wire.raw_bytes // 4, record_dwords):
                    _rec_store(fx.Int32(0), record_rsrc, fx.Int32(dword))
                mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    stage2_addr("node_dest_rank_mask")
                )
                local_mask = (rank_mask_lane >> fx.Int32(node * GPUS_PER_NODE)) & fx.Int32(0xFF)
                # This record belongs to the local aligned source rank.  The
                # reciprocal CCO receive, not this producer, owns the remote
                # source plane on this rank.
                buffer_ops.buffer_store(
                    local_mask,
                    mask_rsrc,
                    fx.Int32(node * MAX_TOKENS) + token,
                )
                if const_expr(publish_plane_slots):
                    buffer_ops.buffer_store(
                        local_slot_mask_lane,
                        buffer_ops.create_buffer_resource_from_addr(
                            stage2_addr("node_dest_slot_mask")
                        ),
                        fx.Int32(node * MAX_TOKENS) + token,
                    )
                # Broadcast rank_mask through the first LDS word.
                fx.ptr_store(
                    Vec.from_elements([rank_mask_lane], fx.Int32),
                    rank_mask_scratch,
                )
            # Eight lanes each pack the 16-bit slot masks for two EP ranks.
            # A destination rank obtains its multiplicity with popcount(mask)
            # and resolves every set bit through the existing topk ID/weight
            # arrays.  This occupies bytes [3952, 3984) of the 4096-B record.
            if (tx < fx.Int32(WORLD // 2)):
                rank0 = tx * fx.Int32(2)
                rank1 = rank0 + fx.Int32(1)
                slots0 = fx.Int32(0)
                slots1 = fx.Int32(0)
                for slot in range_constexpr(TOPK):
                    expert = buffer_ops.buffer_load(
                        ids_rsrc,
                        token * fx.Int32(TOPK) + fx.Int32(slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    valid = (expert >= fx.Int32(0)) & (
                        expert < fx.Int32(EXPERTS)
                    )
                    owner = valid.select(
                        expert // fx.Int32(LOCAL_EXPERTS), fx.Int32(0)
                    )
                    slot_bit = fx.Int32(1 << slot)
                    slots0 = (valid & (owner == rank0)).select(
                        slots0 | slot_bit, slots0
                    )
                    slots1 = (valid & (owner == rank1)).select(
                        slots1 | slot_bit, slots1
                    )
                _rec_store(
                    slots0 | (slots1 << fx.Int32(16)),
                    record_rsrc,
                    fx.Int32(wire.rank_slot_masks_offset // 4) + tx,
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(cco_geometry == "sparse_wqe"):
                rank_mask = Vec(rank_mask_view.load())[0]
                remote_rank_mask = (
                    rank_mask >> fx.Int32(remote_node * GPUS_PER_NODE)
                ) & fx.Int32(0xFF)
                if (wave == fx.Int32(0)) & (
                    remote_rank_mask != fx.Int32(0)
                ):
                    qp_id = token % fx.Int32(layout.num_qp)
                    record_offset = (
                        window_off("dispatch_staging")
                        + fx.Int64(token) * fx.Int64(record_bytes)
                    )
                    _rail.put(
                        dev_comm,
                        qp_id,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_dispatch_rx")
                        + fx.Int64(token) * fx.Int64(record_bytes),
                        arena_win,
                        record_offset,
                        fx.Int64(record_bytes),
                        aggregate=True,
                    )
                if tx == fx.Int32(0):
                    comm_ops.store_i64_global_system(
                        local_addr("sparse_remote_token_ready")
                        + fx.Int64(token) * fx.Int64(8),
                        (remote_rank_mask != fx.Int32(0)).select(
                            fx.Int64(1), fx.Int64(0)
                        ),
                    )
                # dispatch_staging_ready doubles as the cross-CTA WQE-posted
                # completion. The CCO coordinator observes it only after this
                # converged point, so postIdx reservation and descriptor stores
                # are both complete before any doorbell is rung.
                gpu.barrier()
            comm_ops.fence_system_release()
            if tx == fx.Int32(0):
                comm_ops.store_i64_global_system(
                    local_addr("dispatch_staging_ready")
                    + fx.Int64(token) * fx.Int64(8),
                    generation,
                )
                if const_expr(timeline_instrument):
                    if (ticket == fx.Int32(PRODUCER_FIRST)) & (
                        token == fx.Int32(0)
                    ):
                        comm_ops.store_i64_global_relaxed(
                            timeline_addr
                            + fx.Int64(
                                STAGE2_TIMELINE_INDEX[
                                    "stage1_producer_t0_done"
                                ]
                                * 8
                            ),
                            fx.Int64(comm_ops.read_wall_clock()),
                        )
                if const_expr(timeline_instrument):
                    if ticket == fx.Int32(PRODUCER_FIRST):
                        comm_ops.store_i64_global_relaxed(
                            timeline_addr
                            + fx.Int64(
                                STAGE2_TIMELINE_INDEX[
                                    "stage1_producer_last_done"
                                ]
                                * 8
                            ),
                            fx.Int64(comm_ops.read_wall_clock()),
                        )
                if const_expr(timeline_instrument):
                    if ticket == fx.Int32(
                        PRODUCER_FIRST + PRODUCER_CTAS - 1
                    ):
                        if token == fx.Int32(PRODUCER_CTAS - 1):
                            comm_ops.store_i64_global_relaxed(
                                timeline_addr
                                + fx.Int64(
                                    STAGE2_TIMELINE_INDEX[
                                        "stage1_producer_late_first"
                                    ]
                                    * 8
                                ),
                                fx.Int64(comm_ops.read_wall_clock()),
                            )
                        comm_ops.store_i64_global_relaxed(
                            timeline_addr
                            + fx.Int64(
                                STAGE2_TIMELINE_INDEX[
                                    "stage1_producer_late_done"
                                ]
                                * 8
                            ),
                            fx.Int64(comm_ops.read_wall_clock()),
                        )
            # agent: 所有wave完成ready发布后才复用LDS处理下一个token。
            gpu.barrier()
            producer_token = producer_token + fx.Int32(_PROD_STRIDE)
            producer_active = is_producer & (producer_token < ntokens)
            if const_expr(_SPIN_DEADLINE > 0):
                _over = (fx.Int64(comm_ops.read_wall_clock()) - _wdl46) // fx.Int64(
                    _SPIN_DEADLINE
                )
                producer_active = producer_active & (_over == fx.Int64(0))
        if const_expr(_SPIN_DEADLINE > 0):
            if (fx.Int64(comm_ops.read_wall_clock()) - _wdl46) >= fx.Int64(
                _SPIN_DEADLINE
            ):
                comm_ops.store_i64_global_system(
                    local_addr("plan_debug") + fx.Int64(46 * 8),
                    generation,
                )

        # ------------------------------------------------------------------
        # Four waves own four QPs.  Each chunk is one aggregate PUT per QP plus
        # a trailing ready value and one flush/doorbell.  The same CTA receives
        # the reciprocal chunk and performs selected-rank proxy fan-out.
        # ------------------------------------------------------------------
        if is_cco_send:
            qp = wave
            if const_expr(cco_geometry == "sparse_wqe"):
                if wave == fx.Int32(0):
                    # One wave serially owns all four doorbells. Producer CTAs
                    # may reserve/fill WQEs concurrently, but Ionic QPs share a
                    # doorbell mapping and must not be flushed concurrently.
                    for stream_qp in range_constexpr(layout.num_qp):
                        # The first 32 lanes wait for one producer each, then
                        # ballot their local send decisions into the terminal
                        # bitmap.  Publish and flush this QP immediately so its
                        # destination fanout can start while the following QPs
                        # are still waiting for producers.
                        token_flag = fx.Int32(0)
                        if lane < fx.Int32(MAX_TOKENS // layout.num_qp):
                            source_token = (
                                fx.Int32(stream_qp)
                                + lane * fx.Int32(layout.num_qp)
                            )
                            _spin_dbg_5(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(source_token) * fx.Int64(8),
                                generation,
                            )
                            token_flag = fx.Int32(
                                comm_ops.load_i64_global_system(
                                    local_addr("sparse_remote_token_ready")
                                    + fx.Int64(source_token) * fx.Int64(8)
                                )
                            )
                        comm_ops.fence_system_acquire()
                        token_mask = rocdl.ballot(
                            T.i64,
                            (lane < fx.Int32(MAX_TOKENS // layout.num_qp))
                            & (token_flag != fx.Int32(0)),
                        )
                        terminal_ready = (
                            generation
                            << fx.Int64(SPARSE_QP_GENERATION_SHIFT)
                        ) | (
                            token_mask
                            & fx.Int64(
                                (1 << SPARSE_QP_TOKEN_BITS) - 1
                            )
                        )
                        _rail.put_value(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                            arena_win,
                            window_off("sparse_remote_qp_ready")
                            + fx.Int64(stream_qp * 8),
                            terminal_ready,
                            aggregate=True,
                        )
                        if const_expr(timeline_instrument) and const_expr(
                            stream_qp == 0
                        ):
                            if lane == fx.Int32(0):
                                comm_ops.store_i64_global_relaxed(
                                    timeline_addr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage1_dispatch_flush_pre"
                                        ]
                                        * 8
                                    ),
                                    fx.Int64(comm_ops.read_wall_clock()),
                                )
                        request = _rail.flush_async(
                            dev_comm,
                            fx.Int32(stream_qp),
                            fx.Int32(remote_node),
                        )
                        if const_expr(timeline_instrument) and const_expr(
                            stream_qp == 0
                        ):
                            if lane == fx.Int32(0):
                                comm_ops.store_i64_global_relaxed(
                                    timeline_addr
                                    + fx.Int64(
                                        STAGE2_TIMELINE_INDEX[
                                            "stage1_dispatch_flush_post"
                                        ]
                                        * 8
                                    ),
                                    fx.Int64(comm_ops.read_wall_clock()),
                                )
                        if lane == fx.Int32(0):
                            comm_ops.store_i64_global_system(
                                local_addr("sparse_remote_request")
                                + fx.Int64(stream_qp * 8),
                                request,
                            )
                    for ready_qp in range_constexpr(layout.num_qp):
                        if lane == fx.Int32(0):
                            _spin_dbg_6(
                                local_addr("sparse_remote_qp_ready")
                                + fx.Int64(ready_qp * 8),
                                generation
                                << fx.Int64(SPARSE_QP_GENERATION_SHIFT),
                            )
                        comm_ops.fence_system_acquire()
                gpu.barrier()
                if tx == fx.Int32(0):
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        local_addr("sparse_remote_batch_ready"), generation
                    )
                gpu.barrier()
            if const_expr(cco_geometry == "mori64x2"):
                active_rail = wave < fx.Int32(2)
                if active_rail:
                    half = wave
                    # One lane owns one token-ready word; wave reconvergence
                    # proves all 64 contiguous records are packed.
                    token = half * fx.Int32(64) + lane
                    _spin_dbg_7(
                        local_addr("dispatch_staging_ready")
                        + fx.Int64(token) * fx.Int64(8),
                        generation,
                    )
                    comm_ops.fence_system_acquire()
                    src_byte = (
                        window_off("dispatch_staging")
                        + fx.Int64(half) * fx.Int64(64 * record_bytes)
                    )
                    dst_byte = (
                        window_off("remote_dispatch_rx")
                        + fx.Int64(half) * fx.Int64(64 * record_bytes)
                    )
                    _rail.put(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        dst_byte,
                        arena_win,
                        src_byte,
                        fx.Int64(64 * record_bytes),
                        aggregate=True,
                    )
                    _rail.put_value(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_chunk_ready")
                        + fx.Int64(half) * fx.Int64(8),
                        generation,
                        aggregate=True,
                    )
                    request = _rail.flush_async(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                    )
                    if lane == fx.Int32(0):
                        comm_ops.store_i64_global_system(
                            local_addr("remote_chunk_request")
                            + fx.Int64(half) * fx.Int64(8),
                            request,
                        )
                        _spin_dbg_8(
                            local_addr("remote_chunk_ready")
                            + fx.Int64(half) * fx.Int64(8),
                            generation,
                        )
                    comm_ops.fence_system_acquire()
                # wave2/3 are intentionally idle; this is the sole CTA-wide
                # rendezvous after both independent half-record transfers.
                gpu.barrier()

            cco_batch_count = (
                dispatch_chunks // cco_chunks_per_flush
                if cco_geometry == "chunked" and not rail_soa
                else 0
            )
            # 诊断:chunked 路径原本不写这两个戳(它们只埋在 sparse_wqe 分支里),
            # 于是生产配置下 timeline 只能给出一个总时长。这里补上传输段的两端。
            # is_cco 只有 ticket 0 一个 CTA,再限定 qp==0/lane==0 即单一写者,无竞争。
            if const_expr(timeline_instrument) and const_expr(
                cco_geometry == "chunked"
            ):
                if (qp == fx.Int32(0)) & (lane == fx.Int32(0)):
                    comm_ops.store_i64_global_relaxed(
                        timeline_addr
                        + fx.Int64(
                            STAGE2_TIMELINE_INDEX["stage1_dispatch_flush_pre"] * 8
                        ),
                        fx.Int64(comm_ops.read_wall_clock()),
                    )
            # 每 chunk 一次 barrier + 6 次 system 自旋,22 个 chunk 就是 22 次
            # barrier。扣掉门铃(13.8us x N)后的残差随 chunk 数线性增长
            # ~74us/chunk,说明每-chunk 的固定开销才是主项。hoist 模式把本 QP
            # 的全部 staging 等待提到循环外一次做完,循环内只剩发 WQE。
            if const_expr(cco_hoist_staging_wait and not rail_soa):
                # 实测(timeline):**全部** producer 在 ~98us 就发布完了自己的
                # token(第一个 producer 98.4us、最后一个 94.1us),可
                # staging_wait 要到 796us —— 那 700us 全花在这里:lane0 一次
                # 一个地去看 records_per_qp*dispatch_chunks = 134 个**早就置位
                # 了**的 flag,每次都是一条 system scope(非缓存)的 load,
                # 串行依赖。134 x ~5us 正好是 700us。
                #
                # 协议一个字都不用改:让 64 个 lane 各看一个 flag,一轮看 64 个。
                # 134 次串行往返变成 ceil(67/64)*2 = 4 轮。
                if const_expr(wide_staging_wait):
                    _wait_rounds = (records_per_qp + 63) // 64
                    for hchunk in range(
                        fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                    ):
                        hfirst = hchunk * fx.Int32(records_per_chunk)
                        for hround in range_constexpr(_wait_rounds):
                            hoff = fx.Int32(hround * 64) + lane
                            htoken = (
                                hfirst
                                + qp * fx.Int32(records_per_qp)
                                + hoff
                            )
                            # 原来 records_per_chunk 整除 MAX_TOKENS 时会把 ntokens 保护整个去掉。
                            # 那让「1 个 chunk」这个最优配置只在 ntokens == MAX_TOKENS 时成立:
                            # token 数一少,就会去等 producer 永远不置位的 staging flag
                            # (producer 循环上界是 ntokens),整条 rail 挂死。运行期比较不值钱。
                            hactive = (hoff < fx.Int32(records_per_qp)) & (
                                htoken < ntokens
                            )
                            if hactive:
                                _spin_dbg_9(
                                    local_addr("dispatch_staging_ready")
                                    + fx.Int64(htoken) * fx.Int64(8),
                                    generation,
                                )
                else:
                    for hchunk in range(
                        fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                    ):
                        hfirst = hchunk * fx.Int32(records_per_chunk)
                        for hitem in range_constexpr(records_per_qp):
                            htoken = (
                                hfirst
                                + qp * fx.Int32(records_per_qp)
                                + fx.Int32(hitem)
                            )
                            # 同 wide_staging_wait 处的说明。
                            hactive = htoken < ntokens
                            if (lane == fx.Int32(0)) & hactive:
                                _spin_dbg_10(
                                    local_addr("dispatch_staging_ready")
                                    + fx.Int64(htoken) * fx.Int64(8),
                                    generation,
                                )
                gpu.barrier()
            if const_expr(timeline_instrument) and const_expr(
                cco_geometry == "chunked"
            ):
                if (qp == fx.Int32(0)) & (lane == fx.Int32(0)):
                    comm_ops.store_i64_global_relaxed(
                        timeline_addr
                        + fx.Int64(
                            STAGE2_TIMELINE_INDEX["stage1_dispatch_stage_ready"]
                            * 8
                        ),
                        fx.Int64(comm_ops.read_wall_clock()),
                    )
            if const_expr(credit_async and cco_geometry == "chunked"):
                # credit_async:对端对 g-2(同 parity)那批数据的消费确认改到
                # 这里等 —— 只有覆盖对端同一块 remote_dispatch_rx 之前才需要。
                # 本 kernel 末尾不再等对端 credit(那要等对端整个 fan2 做完)。
                if lane == fx.Int32(0):
                    for cchunk in range(
                        fx.Int32(0), fx.Int32(dispatch_chunks), fx.Int32(1)
                    ):
                        _spin_dbg_48(
                            local_addr("remote_chunk_credit")
                            + fx.Int64(cchunk * fx.Int32(layout.num_qp) + qp)
                            * fx.Int64(8),
                            generation - fx.Int64(2),
                        )
            if const_expr(rail_soa):
                # rail_soa:发送源是 k1 之前的 rail_record_quant 直接写进注册窗口
                # (dispatch_staging 的 parity-0 平面)的 token 连续 record
                # [q|scale|ids|weights|pad],REC 字节/token。每个 QP 负责一段连续
                # token,整段一个 PUT + 一个 ready 字 + 一次门铃,不等完成(请求在
                # 末尾 credit 段回收)。WQE 的投递成本 ~15µs 与大小无关(TS3/TS5),
                # 所以不再按字段拆 PUT。发送源单缓冲:本次 PUT 在本 kernel 末尾等掉,
                # 下一次量化在其后。
                src_base = fx.Int64(off("dispatch_staging"))
                _ts(3)
                comm_ops.fence_system_release()
                _ts(4)
                if const_expr(enable_cco):
                    if const_expr(rail_post_ctas):
                        # QP0 由 T0 发,QP1.. 由 ticket 1.. 各自一个 CTA 发(见 fanout 入口)。
                        # rail_post_off_t0:T0 不发任何 QP。
                        if const_expr(not rail_post_off_t0):
                            if wave == fx.Int32(0):
                                _rail_post(fx.Int32(0), False)
                    else:
                        if wave < fx.Int32(RAIL_QPS):
                            _rail_post(wave, False)
            # agent: batch数量随token capacity增长，必须使用runtime循环，
            # 避免TPR4096把整段WQE/等待控制流静态复制数百次。
            for batch in range(
                fx.Int32(0), fx.Int32(cco_batch_count), fx.Int32(1)
            ):
                batch_first = batch * fx.Int32(cco_chunks_per_flush)
                for batch_item in range_constexpr(cco_chunks_per_flush):
                    chunk = batch_first + fx.Int32(batch_item)
                    first_token = chunk * fx.Int32(records_per_chunk)
                    # All four waves wait for their four source records.
                    for item in range_constexpr(records_per_qp):
                        token = (
                            first_token
                            + qp * fx.Int32(records_per_qp)
                            + fx.Int32(item)
                        )
                        # 同上:不再按 MAX_TOKENS 的整除性做 const 特化。
                        token_active = token < ntokens
                        if const_expr(not cco_hoist_staging_wait):
                            if (lane == fx.Int32(0)) & token_active:
                                _spin_dbg_11(
                                    local_addr("dispatch_staging_ready")
                                    + fx.Int64(token) * fx.Int64(8),
                                    generation,
                                )
                    if const_expr(not cco_hoist_staging_wait):
                        gpu.barrier()
                    src_byte = (
                        window_off("dispatch_staging")
                        + fx.Int64(
                            first_token + qp * fx.Int32(records_per_qp)
                        )
                        * fx.Int64(record_bytes)
                    )
                    dst_byte = (
                        window_off("remote_dispatch_rx")
                        + fx.Int64(
                            first_token + qp * fx.Int32(records_per_qp)
                        )
                        * fx.Int64(record_bytes)
                    )
                    if const_expr(enable_cco):
                        remaining_tokens = ntokens - first_token - qp * fx.Int32(records_per_qp)
                        positive_tokens = (remaining_tokens > fx.Int32(0)).select(remaining_tokens, fx.Int32(0))
                        payload_tokens = (positive_tokens < fx.Int32(records_per_qp)).select(positive_tokens, fx.Int32(records_per_qp))
                        # 同上:满块发送不能由整除性决定,否则
                        # ntokens < MAX_TOKENS 时会把没 staged 的尾巴一起发走。
                        # ntokens == MAX_TOKENS 且整除时 payload_tokens 恰好
                        # 等于 records_per_qp,与原来的 qp_bytes 逐字节相同。
                        payload_bytes = payload_tokens * fx.Int32(record_bytes)
                        # A partial final chunk may leave some QPs empty. All
                        # QPs still send a generation flag, but no payload may
                        # extend past the registered token slab.
                        if payload_bytes > fx.Int32(0):
                            _rail.put(
                                dev_comm,
                                qp,
                                fx.Int32(remote_node),
                                arena_win,
                                dst_byte,
                                arena_win,
                                src_byte,
                                fx.Int64(payload_bytes),
                                aggregate=True,
                            )
                        ready_byte = (
                            window_off("remote_chunk_ready")
                            + (
                                fx.Int64(chunk * layout.num_qp)
                                + fx.Int64(qp)
                            )
                            * fx.Int64(8)
                        )
                        _rail.put_value(
                            dev_comm,
                            qp,
                            fx.Int32(remote_node),
                            arena_win,
                            ready_byte,
                            generation,
                            aggregate=True,
                        )
                    else:
                        # EP8 single-node bring-up: treat this rank's staging
                        # slab as the aligned remote source.
                        src_local = buffer_ops.create_buffer_resource_from_addr(
                            local_addr("dispatch_staging")
                        )
                        dst_local = buffer_ops.create_buffer_resource_from_addr(
                            local_addr("remote_dispatch_rx")
                        )
                        base_dw = (
                            first_token + qp * fx.Int32(records_per_qp)
                        ) * fx.Int32(record_dwords)
                        for dword in range(
                            base_dw + lane * fx.Int32(4),
                            base_dw
                            + fx.Int32(records_per_qp * record_dwords),
                            fx.Int32(64 * 4),
                        ):
                            value = buffer_ops.buffer_load(
                                src_local,
                                dword,
                                vec_width=4,
                                dtype=T.i32,
                            )
                            buffer_ops.buffer_store(value, dst_local, dword)
                        rocdl.s_waitcnt(0)
                        gpu.barrier()
                        if lane == fx.Int32(0):
                            comm_ops.store_i64_global_system(
                                local_addr("remote_chunk_ready")
                                + (
                                    fx.Int64(chunk * layout.num_qp)
                                    + fx.Int64(qp)
                                )
                                * fx.Int64(8),
                                generation,
                            )

                if const_expr(enable_cco):
                    if const_expr(rail_serial_doorbell):
                        # sparse_wqe 分支早就是 wave0 串行敲全部 num_qp 个门铃,
                        # 理由写在它上面:Ionic 的 QP 共享同一个 doorbell 映射,
                        # 不可并发 flush。chunked 这边却是 4 个 wave 各敲各的 ctx
                        # —— 同一个文件里两个相反的假设。这个开关把 chunked 收成
                        # 和 sparse_wqe 一致,用来量并发门铃到底有没有争用。
                        #
                        # 两个 barrier 都必须在 if wave==0 之**外**:只有 wave0
                        # 进得去的块里调 gpu.barrier() 必死锁。
                        # 前一个:flushAsyncImpl 是非原子地读 wq->postIdx 的,
                        # wave0 替别人敲门铃之前,那几个 wave 的 put/put_value
                        # 必须已经全部 post 完。
                        gpu.barrier()
                        if wave == fx.Int32(0):
                            for sqp in range_constexpr(layout.num_qp):
                                request = _rail.flush_async(
                                    dev_comm,
                                    fx.Int32(sqp),
                                    fx.Int32(remote_node),
                                )
                                if lane == fx.Int32(0):
                                    request_index = (
                                        fx.Int64(
                                            batch_first * layout.num_qp
                                        )
                                        + fx.Int64(sqp)
                                    )
                                    comm_ops.store_i64_global_system(
                                        local_addr("remote_chunk_request")
                                        + request_index * fx.Int64(8),
                                        request,
                                    )
                        # 后一个:下一个 batch 的 WQE 不能在 wave0 还在敲门铃时
                        # 就开始 post,否则 NIC 会 fetch 一个刚 reserve 但没写完
                        # 的 WQE。
                        gpu.barrier()
                    else:
                        request = _rail.flush_async(
                            dev_comm,
                            qp,
                            fx.Int32(remote_node),
                        )
                        if lane == fx.Int32(0):
                            request_index = (
                                fx.Int64(batch_first * layout.num_qp)
                                + fx.Int64(qp)
                            )
                            comm_ops.store_i64_global_system(
                                local_addr("remote_chunk_request")
                                + request_index * fx.Int64(8),
                                request,
                            )

                # The batch is visible remotely after one doorbell; acquire
                # every reciprocal ready word before proxy fanout consumes it.
                #
                # 这个等待原本在发送循环**内**,于是两个节点按 chunk 锁步推进:
                # 发完 chunk i 就停下来等对端的 chunk i,才发 i+1。实测
                # 每 chunk 148us,而 64.5KB 在 47GB/s 下只要 1.4us —— 是锁步不是字节。
                # 这正是 stage2 AR37 修掉的「敲一个等一个」(那边值 291us)。
                # defer 模式把 19 个 chunk 全发完再统一等,ordering 在批边界做一次。
                # 消费侧的序不依赖这里:分片后的 fanout CTA 自己就在等
                # remote_chunk_ready(远端 proxy 循环),并各自 barrier。
                if const_expr(not cco_defer_reciprocal_wait):
                    for batch_item in range_constexpr(cco_chunks_per_flush):
                        chunk = batch_first + batch_item
                        if lane == fx.Int32(0):
                            _spin_dbg_12(
                                local_addr("remote_chunk_ready")
                                + (
                                    fx.Int64(chunk * layout.num_qp)
                                    + fx.Int64(qp)
                                )
                                * fx.Int64(8),
                                generation,
                            )
                        gpu.barrier()
                        comm_ops.fence_system_acquire()
                        gpu.barrier()

            if const_expr(timeline_instrument) and const_expr(
                cco_geometry == "chunked"
            ):
                if (qp == fx.Int32(0)) & (lane == fx.Int32(0)):
                    comm_ops.store_i64_global_relaxed(
                        timeline_addr
                        + fx.Int64(
                            STAGE2_TIMELINE_INDEX["stage1_dispatch_flush_post"] * 8
                        ),
                        fx.Int64(comm_ops.read_wall_clock()),
                    )
            if const_expr(cco_defer_reciprocal_wait):
                if const_expr(timeline_instrument):
                    if (qp == fx.Int32(0)) & (lane == fx.Int32(0)):
                        comm_ops.store_i64_global_relaxed(
                            timeline_addr
                            + fx.Int64(
                                STAGE2_TIMELINE_INDEX["stage1_dispatch_send_done"]
                                * 8
                            ),
                            fx.Int64(comm_ops.read_wall_clock()),
                        )
                # 全部发完之后一次性等齐,acquire fence 在批边界做一次。
                # rail_soa 只置 chunk 0 的 ready 字。
                for wchunk in range(
                    fx.Int32(0),
                    fx.Int32(1 if rail_soa else dispatch_chunks),
                    fx.Int32(1),
                ):
                    if lane == fx.Int32(0):
                        _spin_dbg_13(
                            local_addr("remote_chunk_ready")
                            + (
                                fx.Int64(wchunk * layout.num_qp)
                                + fx.Int64(qp)
                            )
                            * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()
                comm_ops.fence_system_acquire()
                gpu.barrier()
                _ts(6)
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
                if const_expr(cco_geometry == "sparse_wqe"):
                    ready_qp = token % fx.Int32(layout.num_qp)
                    ready_bit = token // fx.Int32(layout.num_qp)
                    terminal_ready = fx.Int64(
                        comm_ops.load_i64_global_system(
                            local_addr("sparse_remote_qp_ready")
                            + fx.Int64(ready_qp) * fx.Int64(8)
                        )
                    )
                    mask_bit_set = (
                        (
                            (
                                terminal_ready & fx.Int64(0xFFFFFFFF)
                            )
                            >> fx.Int64(ready_bit)
                        )
                        & fx.Int64(1)
                    ) != fx.Int64(0)
                    token_ready = (
                        (
                            terminal_ready
                            >> fx.Int64(SPARSE_QP_GENERATION_SHIFT)
                        )
                        >= generation
                    ) & mask_bit_set
                    remote_record_available = token_ready.select(
                        fx.Int32(1), fx.Int32(0)
                    )
                remote_mask = fx.Int32(0)
                remote_slot_mask = fx.Int32(0)
                if remote_record_available != fx.Int32(0):
                    # token 每 lane 不同:描述符必须是标量,按 token 建描述符会被
                    # 编译成 readfirstlane 瀑布循环(ATT5:每次 load 最多 64 轮,
                    # T0 在这里耗 ~230k 周期)。整个 rx 区一个描述符 + lane 偏移。
                    if const_expr(rail_soa):
                        record_dword = token * fx.Int32(REC_BYTES // 4) + fx.Int32(
                            REC_I // 4
                        )
                    else:
                        record_dword = token * fx.Int32(
                            record_bytes // 4
                        ) + fx.Int32(wire.ids_offset // 4)
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
            if const_expr(_DIAG_TS):
                gpu.barrier()
            _ts(7)

        # ------------------------------------------------------------------
        # Split fanout CTAs cover the eight node-local destinations. Each writes
        # both its local source rank and the aligned remote source rank directly
        # into destination expert tiles; there is no rank inbox/sort.
        # ------------------------------------------------------------------
        if const_expr(diagnostic_split_fanout):
            # 角色映射必须**不带** fanout_pred:fanout 阶段关掉的 role 里
            # is_inter/intra_fanout 为假,finisher 跟着消失,于是 sealer 消失,
            # tile 尾部 padding 行的 arg_mind 索引永远没人填,k2 gather 读到
            # 垃圾行号 -> illegal memory access。else 分支的 is_finisher 建在
            # is_comm 上,不依赖 fanout_pred,所以从没暴露过。
            split_active = inter_ticket | intra_ticket
            split_worker_raw = inter_ticket.select(
                ticket - fx.Int32(_SPLIT_BASE),
                ticket - fx.Int32(_SPLIT_BASE + split_fanout_ctas),
            )
            split_worker = split_active.select(
                split_worker_raw, fx.Int32(0)
            )
            split_dest = split_worker % fx.Int32(GPUS_PER_NODE)
            split_group = split_worker // fx.Int32(GPUS_PER_NODE)
            split_shard = split_active.select(
                split_group, fx.Int32(MAX_TOKENS)
            )
            is_finisher = (
                intra_ticket
                & (split_dest == fx.Int32(local_rank))
                & (split_group == fx.Int32(0))
            )
        else:
            split_worker = fx.Int32(0)
            split_dest = fx.Int32(0)
            split_group = fx.Int32(0)
            split_shard = fx.Int32(0)
            # finisher_off_t0:rank0 上 ticket==local_rank 就是 T0(已兼 rail/init/CCO),
            # 封尾+发布挪到 ticket 1+local_rank%7(永不是 T0)。
            is_finisher = is_comm & (
                ticket == fx.Int32(1 + local_rank % 7 if finisher_off_t0 else local_rank)
            )

        # 两个 finisher 职责必须分开:封尾 partial tile 属于 dispatch 那一半,
        # 而「等 h1_compute_done 收满再发 stage1_done」属于 GMM1 那一半。
        is_sealer = is_finisher if _do_seal else _never
        is_publisher = is_finisher if _do_publish else _never

        def _peer_addr(dest, name):
            return (
                arena_lsa_base
                + fx.Int64(dest) * arena_lsa_stride
                + fx.Int64(window_off(name))
            )

        def _alloc_tiles_for(count):
            """一个 expert 实际被分配的 tile 数(向上取整到 G 的倍数)。"""
            n = (count + fx.Int32(BM - 1)) // fx.Int32(BM)
            if const_expr(G == 1):
                return n
            return ((n + fx.Int32(G - 1)) // fx.Int32(G)) * fx.Int32(G)

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
            # tile_row_done 置零只服务 tile_pipeline 的到达计数;tile_row_base
            # 没有任何设备端读者。默认路径两者都不写。
            if const_expr(tile_pipeline):
                done_res = buffer_ops.create_buffer_resource_from_addr(
                    _peer_addr(dest, "tile_row_done")
                )
            for j in range_constexpr(G):
                phys = base + fx.Int32(j)
                buffer_ops.buffer_store(phys, map_res, map_index + fx.Int32(j))
                if const_expr(tile_pipeline):
                    buffer_ops.buffer_store(fx.Int32(0), done_res, phys)
                buffer_ops.buffer_store(local_expert, expert_res, phys)
            comm_ops.fence_system_release()
            for j in range_constexpr(G):
                # claim_relaxed:上面一次 fence 已放行 map/tile_expert,G 条 release store 各带一次整 L2 写回。
                (comm_ops.store_i64_global_system_relaxed if claim_relaxed else comm_ops.store_i64_global_system)(
                    map_ready + fx.Int64(j * 8), generation
                )
            return base

        def _enqueue_tile_jobs(dest, physical, early_full_tile):
            """Append one ready BM32 tile as one contiguous 24-job batch.

            This helper is called by exactly one thread: either the unique
            ``tile_row_done`` last-arriver for a full tile or one finisher
            thread for an EOS-sealed partial tile.  ``h1_queue_tail`` reserves
            space only; the generation word at the first slot of each batch is
            the release publication consumed by compute CTAs.
            """

            base = fx.Int32(
                comm_ops.atomic_add_system_acq_rel(
                    _peer_addr(dest, "h1_queue_tail"),
                    fx.Int32(h1_n_blocks),
                )
            )
            in_bounds = base <= fx.Int32(max_jobs - h1_n_blocks)
            if in_bounds:
                queue = buffer_ops.create_buffer_resource_from_addr(
                    _peer_addr(dest, "h1_ready_queue")
                )
                for nblock in range_constexpr(h1_n_blocks):
                    job = physical * fx.Int32(h1_n_blocks) + fx.Int32(nblock)
                    buffer_ops.buffer_store(
                        job,
                        queue,
                        base + fx.Int32(nblock),
                    )
                rocdl.s_waitcnt(0)
                comm_ops.fence_system_release()
                comm_ops.store_i64_global_system(
                    _peer_addr(dest, "h1_ready_queue_generation")
                    + fx.Int64(base) * fx.Int64(8),
                    generation,
                )
                if const_expr(tile_pipeline_instrument):
                    if early_full_tile:
                        comm_ops.atomic_add_system_acq_rel(
                            _peer_addr(dest, "h1_early_full_tiles"),
                            fx.Int32(1),
                        )
            else:
                comm_ops.atomic_add_system(error_addr, fx.Int32(1))

        def _all_comm_eos_seen():
            all_ready = fx.Int32(1)
            for peer in range_constexpr(GPUS_PER_NODE):
                observed = fx.Int64(
                    comm_ops.load_i64_global_system(
                        local_addr("comm_eos") + fx.Int64(peer * 8)
                    )
                )
                all_ready = (observed >= generation).select(
                    all_ready, fx.Int32(0)
                )
            return all_ready

        def _record_dest_slot_mask(record, dest):
            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            predicate_scratch = fx.recast_iter(fx.Int32, lds_raw)
            predicate_view = fx.make_view(
                predicate_scratch, fx.make_layout(1, 1)
            )
            if tx == fx.Int32(0):
                dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                packed_masks = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.rank_slot_masks_offset // 4)
                    + dest_global // fx.Int32(2),
                    vec_width=1,
                    dtype=T.i32,
                )
                shift = (dest_global & fx.Int32(1)) * fx.Int32(16)
                slots = (packed_masks >> shift) & fx.Int32(0xFFFF)
                fx.ptr_store(
                    Vec.from_elements([slots], fx.Int32),
                    predicate_scratch,
                )
            gpu.barrier()
            return Vec(predicate_view.load())[0]

        def _record_targets_dest(record, dest):
            return _record_dest_slot_mask(record, dest) != fx.Int32(0)

        def _dispatch_route(record, dest, source_index, topk_slot):
            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            scratch = fx.recast_iter(fx.Int32, lds_raw)
            scratch_view = fx.make_view(scratch, fx.make_layout(1, 1))
            if tx == fx.Int32(0):
                dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                expert = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.ids_offset // 4) + topk_slot,
                    vec_width=1,
                    dtype=T.i32,
                )
                valid = (expert >= dest_global * fx.Int32(LOCAL_EXPERTS)) & (
                    expert
                    < (dest_global + fx.Int32(1)) * fx.Int32(LOCAL_EXPERTS)
                )
                invalid = (expert < dest_global * fx.Int32(LOCAL_EXPERTS)) | (
                    expert
                    >= (dest_global + fx.Int32(1))
                    * fx.Int32(LOCAL_EXPERTS)
                )
                if invalid:
                    comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                fx.ptr_store(
                    Vec.from_elements(
                        [
                            valid.select(
                                expert
                                - dest_global * fx.Int32(LOCAL_EXPERTS),
                                fx.Int32(0),
                            )
                        ],
                        fx.Int32,
                    ),
                    scratch,
                )
            gpu.barrier()
            local_expert = Vec(scratch_view.load())[0]
            expert_count_addr = _peer_addr(dest, "expert_count")
            if tx == fx.Int32(0):
                row_slot_lane = fx.Int32(
                    comm_ops.atomic_add_system(
                        expert_count_addr + fx.Int64(local_expert) * fx.Int64(4),
                        fx.Int32(1),
                    )
                )
                fx.ptr_store(Vec.from_elements([row_slot_lane], fx.Int32), scratch)
            gpu.barrier()
            row_slot = Vec(scratch_view.load())[0]
            logical_tile = row_slot // fx.Int32(BM)
            row_in_tile = row_slot % fx.Int32(BM)
            map_index = local_expert * fx.Int32(max_tiles_per_expert) + logical_tile
            map_ready = _peer_addr(dest, "expert_tile_map_ready") + fx.Int64(map_index) * 8
            if _is_group_head(row_in_tile, row_slot):
                if tx == fx.Int32(0):
                    physical = _claim_tile_group(
                        dest, map_index, local_expert, map_ready
                    )
                    fx.ptr_store(Vec.from_elements([physical], fx.Int32), scratch)
            else:
                if tx == fx.Int32(0):
                    _spin_dbg_14(map_ready, generation)
                    physical = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "expert_tile_map")),
                        map_index,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    fx.ptr_store(Vec.from_elements([physical], fx.Int32), scratch)
            gpu.barrier()
            physical = Vec(scratch_view.load())[0]
            grouped_row = physical * fx.Int32(BM) + row_in_tile
            if tx < fx.Int32(scale_bytes):
                scale = buffer_ops.buffer_load(
                    rec, fx.Int32(wire.payload_bytes) + tx, vec_width=1, dtype=T.i8
                )
                # Exact BM32 A-scale preshuffle consumed by
                # package-local gemm1.issue_a_scale_load():
                # (ku, ikxdl, k_lane, n_lane, im_a).
                ku = tx // fx.Int32(8)
                ikxdl = (tx % fx.Int32(8)) // fx.Int32(4)
                k_lane = tx % fx.Int32(4)
                # BM=32*S 时每个 tile 占 S 个 32 行 scale chunk,gemm1 的
                # issue_a_scale_load() 从 m_row//32 起连读 S 个 chunk,所以
                # 这里也必须按 32 行子块落位,否则 BM>32 会读到错位的 scale。
                a_sub = row_in_tile // fx.Int32(32)
                a_row = row_in_tile % fx.Int32(32)
                im_a = a_row // fx.Int32(16)
                n_lane = a_row % fx.Int32(16)
                dst_dword = (
                    (physical * fx.Int32(BM // 32) + a_sub)
                    * fx.Int32(scale_dwords * 32)
                    + ku * fx.Int32(64)
                    + k_lane * fx.Int32(16)
                    + n_lane
                )
                dst_byte = dst_dword * fx.Int32(4) + ikxdl * fx.Int32(2) + im_a
                buffer_ops.buffer_store(
                    scale,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "grouped_input_scale")
                    ),
                    dst_byte,
                    offset_is_bytes=True,
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tx == fx.Int32(0):
                weight = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.weights_offset // 4) + topk_slot,
                    vec_width=1,
                    dtype=T.f32,
                )
                source_encoding = source_index | (topk_slot << fx.Int32(24))
                buffer_ops.buffer_store(
                    source_index,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_input")
                    ),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    source_encoding,
                    buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "tile_row_source")),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    weight,
                    buffer_ops.create_buffer_resource_from_addr(_peer_addr(dest, "tile_row_weight")),
                    grouped_row,
                )
                # 逐 route 路径(只剩 tile_pipeline)下这条 release 承担 EOS 排序:
                # 该路径的 fanout_shard_done 是 agent RMW,不带 release。
                comm_ops.fence_system_release()
                # tile_row_done 只在 tile_pipeline 下有读者(满 tile 立即入队)。
                # 默认路径 tile 在全部 comm_eos 之后由 sealer 统一发布,这条
                # system acq_rel 原子的结果没人用,每条 route 白付一次。
                if const_expr(tile_pipeline and not diagnostic_no_arrival_rmw):
                    completed = fx.Int32(
                        comm_ops.atomic_add_system_acq_rel(
                            _peer_addr(dest, "tile_row_done")
                            + fx.Int64(physical) * 4,
                            fx.Int32(1),
                        )
                    )
                    # The acq_rel RMW chain makes the unique last arriver
                    # observe all 32 row payload/scale/metadata releases.
                    # It can therefore publish this full tile immediately,
                    # before the eight communication roles reach EOS.
                    if completed == fx.Int32(BM - 1):
                        _enqueue_tile_jobs(dest, physical, True)
            gpu.barrier()

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

        def _dispatch_record_batched(record, dest, source_index):
            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            _dispatch_view_batched(
                dest,
                source_index,
                _record_dest_slot_mask(record, dest),
                (
                    rec, fx.Int32(0),
                    rec, fx.Int32(wire.payload_bytes),
                    rec, fx.Int32(wire.ids_offset // 4),
                    rec, fx.Int32(wire.weights_offset // 4),
                ),
            )

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
            t_pre = comm_ops.read_wall_clock() if _DIAG_TS else None
            if const_expr(rail_soa):
                # x_q 指向 rail record 基址(量化直接写成 record)。
                rec_rs = buffer_ops.create_buffer_resource_from_addr(x_q)
                q_view = (rec_rs, token * fx.Int32(REC_BYTES // 4) + fx.Int32(REC_Q // 4))
                s_view = (rec_rs, token * fx.Int32(REC_BYTES) + fx.Int32(REC_S))
            else:
                q_view = (
                    buffer_ops.create_buffer_resource_from_addr(x_q),
                    token * fx.Int32(HIDDEN // 8),
                )
                s_view = (
                    buffer_ops.create_buffer_resource_from_addr(input_scale),
                    token * fx.Int32(HIDDEN // 32),
                )
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
                t_pre,
            )

        def _dispatch_remote_soa(token, dest, source_index):
            # rail_soa:remote_dispatch_rx[parity] 与发送端同一 record 排布。
            rx = buffer_ops.create_buffer_resource_from_addr(
                local_addr("remote_dispatch_rx")
            )
            rec_dw = token * fx.Int32(REC_BYTES // 4)
            ids_base = rec_dw + fx.Int32(REC_I // 4)
            t_pre = comm_ops.read_wall_clock() if _DIAG_TS else None
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
                t_pre,
            )

        def _dispatch_view_batched(dest, source_index, route_slots, view, t_pre=None):
            # view = (q 描述符, q 起始 dword, scale 描述符, scale 起始字节,
            #         ids 描述符, ids 起始下标, weights 描述符, weights 起始下标)
            # t_pre:诊断打点时算 slot mask 之前的时刻(_DIAG_TS)。
            q_rs, q_dw0, s_rs, s_b0, id_rs, id_i0, w_rs, w_i0 = view
            if const_expr(_DIAG_TS):
                t_a = comm_ops.read_wall_clock()
                if tx == fx.Int32(0):
                    _acc_add(0, 1)
                    if const_expr(t_pre is not None):
                        _acc_add(3, t_a - t_pre)
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
                if const_expr(_DIAG_TS):
                    t_b = comm_ops.read_wall_clock()
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
                if const_expr(_DIAG_TS):
                    t_c = comm_ops.read_wall_clock()
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
                            _spin_dbg_14(
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
                if const_expr(_DIAG_TS):
                    t_d = comm_ops.read_wall_clock()
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
                if const_expr(_DIAG_TS):
                    t_e = comm_ops.read_wall_clock()
                    if tx == fx.Int32(0):
                        n_routes = fx.Int32(0)
                        for b in range_constexpr(TOPK):
                            n_routes = n_routes + (
                                (route_slots >> fx.Int32(b)) & fx.Int32(1)
                            )
                        _acc_add(1, 1)
                        _acc_add(2, n_routes)
                        _acc_add(4, t_b - t_a)
                        _acc_add(5, t_c - t_b)
                        _acc_add(6, t_d - t_c)
                        _acc_add(7, t_e - t_d)

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
        GB_P3 = int(gb_p3)
        _GB_UNITS = payload_dwords // 4
        _GB_IT = (_GB_UNITS + 63) // 64
        _GB_SP = (_GB_NB * scale_bytes + THREADS - 1) // THREADS
        if const_expr(route_gbatch):
            assert _GB_NB >= WAVES and _GB_NB % WAVES == 0 and _GB_LANES <= THREADS
            assert LOCAL_EXPERTS + _GB_NB <= THREADS, "P0 clears HIT with threads [LE, LE+_GB_NB)"
            assert _GB_OWNB + LOCAL_EXPERTS <= TS_LDS, "route_gbatch LDS overlaps TS accumulators"
            assert _GB_NB <= GROUP_ROWS, "one batch may cross at most one group head per expert"
            assert payload_dwords % 4 == 0
        if const_expr(ascale_gather):
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
            rs, tok0, tstride, dest, src_base, emit_masks=False, ids_side=False,
            remote=False,
        ):
            # rs:AoS record 描述符(每 token REC_BYTES);本批 token = tok0 + r*tstride。
            # split_local 下远端来源(remote=True)用 expert_count/map 的后半 [LE,2LE):
            # 与本地来源分开计数、分开成组;物理 tile 仍从同一个 tile_alloc 认领。
            _EO = LOCAL_EXPERTS if (split_local and remote) else 0
            if const_expr(_DIAG_TS):
                t_a = comm_ops.read_wall_clock()
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
            if const_expr(ids_side):
                id0 = fx.Int32(SIDE_OFF // 4) + live.select(tok, fx.Int32(0)) * fx.Int32(
                    SIDE_REC // 4
                )
            else:
                id0 = rec_dw + fx.Int32(REC_I // 4)
            _gb_wdw = TOPK if ids_side else _GB_WDW_REC
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
            if const_expr(_DIAG_TS):
                t_b = comm_ops.read_wall_clock()
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
                    if const_expr(meta_opt >= 1):
                        _gb_st(fx.Int32(_GB_OWN) + tx, fx.Int32(-1))
                    head = (
                        (seg + fx.Int32(GROUP_ROWS - 1)) // fx.Int32(GROUP_ROWS)
                    ) * fx.Int32(GROUP_ROWS)
                    if head < seg + n:
                        m0 = (tx + fx.Int32(_EO)) * fx.Int32(
                            max_tiles_per_expert
                        ) + head // fx.Int32(BM)
                        own_base = _claim_tile_group(
                            dest,
                            m0,
                            tx,
                            _peer_addr(dest, "expert_tile_map_ready") + fx.Int64(m0) * 8,
                        )
                        if const_expr(meta_opt >= 1):
                            _gb_st(fx.Int32(_GB_OWN) + tx, head // fx.Int32(GROUP_ROWS))
                            _gb_st(fx.Int32(_GB_OWNB) + tx, own_base)
            gpu.barrier()
            if const_expr(_DIAG_TS):
                t_c = comm_ops.read_wall_clock()
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
                    if const_expr(ascale_gather):
                        sv = buffer_ops.buffer_load(
                            rs,
                            ptok_c * fx.Int32(REC_BYTES // 4)
                            + fx.Int32(REC_S // 4)
                            + (lane < fx.Int32(scale_dwords)).select(lane, fx.Int32(0)),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    _pl.append((prec, ptok, vals, sv))

                if const_expr(not ids_side):
                    _issue_payload_loads()
                if const_expr(GB_P3 >= 2):
                    # 每 expert 一个线程解析本批段落 [seg, seg+n) 涉及的至多 2 个组的物理基址
                    # (组头认领都已在 P2 完成,这里只等不认领)。逐行的远端 map 读没了。
                    if const_expr(meta_opt >= 1):
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
                                    if _gb_ld(fx.Int32(_GB_OWN) + mo_e) == gidx:
                                        ph = _gb_ld(fx.Int32(_GB_OWNB) + mo_e)
                                    else:
                                        m = (mo_e + fx.Int32(_EO)) * fx.Int32(
                                            max_tiles_per_expert
                                        ) + gidx * fx.Int32(G)
                                        _spin_dbg_14(
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
                    if tx < fx.Int32(LOCAL_EXPERTS if meta_opt == 0 else 0):
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
                                _spin_dbg_14(
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
                        _spin_dbg_14(
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
                if const_expr(ids_side):
                    # 领行/元数据都已发出;此时才等本节点收到对端 payload(4 个 QP 的就绪字)。
                    if const_expr(_DIAG_TS):
                        t_g0 = comm_ops.read_wall_clock()
                    if tx < fx.Int32(layout.num_qp):
                        _spin_dbg_29(
                            local_addr("remote_chunk_ready") + fx.Int64(tx) * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    if const_expr(_DIAG_TS):
                        t_g1 = comm_ops.read_wall_clock()
                        if tx == fx.Int32(0):
                            # acc3 = P3 起点到 payload 门(领行后的元数据段);acc2 = 门内等待。
                            _acc_add(3, t_g0 - t_c)
                            _acc_add(2, t_g1 - t_g0)
                    _issue_payload_loads()
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
                        if const_expr(ascale_gather):
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
                    _spin_dbg_14(
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
                        if const_expr(ascale_gather):
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
            if const_expr(_DIAG_TS):
                t_d = comm_ops.read_wall_clock()
            # P4:scale 按 (record, 字节) 摊到全 CTA,散写到每条 route 的 BM32 预排布。
            scale_res = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "grouped_input_scale")
            )
            for ps in range_constexpr(0 if ascale_gather else _GB_SP):
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
            if const_expr(_DIAG_TS):
                t_e = comm_ops.read_wall_clock()
                if tx == fx.Int32(0):
                    _acc_add(0, _GB_NB)
                    _acc_add(1, 1)
                    _acc_add(4, t_b - t_a)
                    _acc_add(5, t_c - t_b)
                    _acc_add(6, t_d - t_c)
                    _acc_add(7, t_e - t_d)

        def _dispatch_record(record, dest, source_index):
            # The record is deduplicated at node/rank granularity, but every
            # matching top-k slot remains a distinct expert contribution.
            # The mask is uniform across the CTA, so barriers in
            # _dispatch_route are reached by all threads for every set bit.
            route_slots = _record_dest_slot_mask(record, dest)
            if route_slots != fx.Int32(0):
                # Second-level payload deduplication: one quantized activation
                # copy per (source token, destination rank), irrespective of
                # how many local experts that rank owns in the token's Top-K.
                # Route rows below point back to this fixed source-indexed row.
                rec = buffer_ops.create_buffer_resource_from_addr(
                    record, num_records_bytes=record_bytes
                )
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
                        rec, dword, vec_width=4, dtype=T.i32
                    )
                    buffer_ops.buffer_store(value, dst_payload, dword)
                # 这条 s_waitcnt(0) 是整条流水的排空,每条 record 一次
                # (shards=32 下每个 CTA ~32 次)。它是多余的:唯一**发布**这
                # 一行的动作是 `_dispatch_route` 末尾的 tile_row_done 到达原子,
                # 而那之前已经有一次 `rocdl.s_waitcnt(0)` + `gpu.barrier()`
                # (整 CTA 排空)再加 fence_system_release。payload 的写因此
                # 一样被排在到达信号之前。排空粒度本身值 4.4x,见
                # [[簿记粒度 vs 分片粒度]]。barrier 保留(便宜,且管 LDS 复用)。
                if const_expr(not lean_waitcnt):
                    rocdl.s_waitcnt(0)
                gpu.barrier()
                for topk_slot in range(
                    fx.Int32(0), fx.Int32(TOPK), fx.Int32(1)
                ):
                    if (
                        (route_slots >> topk_slot) & fx.Int32(1)
                    ) != fx.Int32(0):
                        _dispatch_route(
                            record, dest, source_index, topk_slot
                        )

        if const_expr(route_batch):
            _dispatch_record = _dispatch_record_batched

        def _dispatch_record_wave_multi(record, dest, source_index):
            """Place one record with one 64-lane wave, all routes, no barrier.

            与 CTA 粒度的 ``_dispatch_record`` 逐条等价(同样的 slot mask 去重、
            同样的 payload 单拷、同样的 per-route 认领与 tile 发布),差别只在
            并发粒度:一个 CTA 的 4 个 wave 同时放 4 条不同的 record。

            fanout 是**跨 GPU 往返延迟**受限的 —— 每条 route 串行走
            expert_count 原子、tile_alloc 原子或 map_ready 自旋、payload/scale
            的 s_waitcnt、tile_row_done 原子,约 5 次往返。CTA 粒度下这 5 次
            往返期间整个 CTA 的 256 条线程都在等;wave 粒度下 4 条 record 的
            往返互相重叠。

            全部控制流都是 **wave 内一致** 的(mask 与 expert 经 readlane 广播,
            topk 循环是 wave uniform),所以整段不需要也不能有 gpu.barrier()
            —— 4 个 wave 的迭代次数本来就不同。
            """
            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            slots_lane0 = fx.Int32(0)
            if lane == fx.Int32(0):
                dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                packed_masks = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.rank_slot_masks_offset // 4)
                    + dest_global // fx.Int32(2),
                    vec_width=1,
                    dtype=T.i32,
                )
                shift = (dest_global & fx.Int32(1)) * fx.Int32(16)
                slots_lane0 = (packed_masks >> shift) & fx.Int32(0xFFFF)
            route_slots = fx.Int32(rocdl.readlane(T.i32, slots_lane0, 0))

            if route_slots != fx.Int32(0):
                # payload 每 (source token, dest rank) 只拷一份,与 CTA 版一致。
                dst_payload = buffer_ops.create_buffer_resource_from_addr(
                    _peer_addr(dest, "grouped_input_q")
                    + fx.Int64(source_index) * fx.Int64(wire.payload_bytes),
                    num_records_bytes=wire.payload_bytes,
                )
                for dword in range(
                    lane * fx.Int32(4), payload_dwords, fx.Int32(64 * 4)
                ):
                    value = buffer_ops.buffer_load(
                        rec, dword, vec_width=4, dtype=T.i32
                    )
                    buffer_ops.buffer_store(value, dst_payload, dword)
                rocdl.s_waitcnt(0)

                for topk_slot in range(
                    fx.Int32(0), fx.Int32(TOPK), fx.Int32(1)
                ):
                    if ((route_slots >> topk_slot) & fx.Int32(1)) != fx.Int32(0):
                        expert_lane0 = fx.Int32(0)
                        if lane == fx.Int32(0):
                            dest_global = (
                                fx.Int32(node * GPUS_PER_NODE) + dest
                            )
                            expert = buffer_ops.buffer_load(
                                rec,
                                fx.Int32(wire.ids_offset // 4) + topk_slot,
                                vec_width=1,
                                dtype=T.i32,
                            )
                            # 谓词与 Int32 不混用:照抄 _dispatch_route 的
                            # valid/invalid 双表达式写法。
                            valid = (
                                expert
                                >= dest_global * fx.Int32(LOCAL_EXPERTS)
                            ) & (
                                expert
                                < (dest_global + fx.Int32(1))
                                * fx.Int32(LOCAL_EXPERTS)
                            )
                            invalid = (
                                expert
                                < dest_global * fx.Int32(LOCAL_EXPERTS)
                            ) | (
                                expert
                                >= (dest_global + fx.Int32(1))
                                * fx.Int32(LOCAL_EXPERTS)
                            )
                            if invalid:
                                comm_ops.atomic_add_system(
                                    error_addr, fx.Int32(1)
                                )
                            expert_lane0 = valid.select(
                                expert
                                - dest_global * fx.Int32(LOCAL_EXPERTS),
                                fx.Int32(0),
                            )
                        local_expert = fx.Int32(
                            rocdl.readlane(T.i32, expert_lane0, 0)
                        )

                        row_slot_lane0 = fx.Int32(0)
                        if lane == fx.Int32(0):
                            if const_expr(_CLAIM_AGENT_PROBE):
                                row_slot_lane0 = fx.Int32(
                                    comm_ops.atomic_add_agent(
                                        local_addr("expert_count")
                                        + fx.Int64(local_expert) * fx.Int64(4),
                                        fx.Int32(1),
                                    )
                                )
                            else:
                                row_slot_lane0 = fx.Int32(
                                    comm_ops.atomic_add_system(
                                        _peer_addr(dest, "expert_count")
                                        + fx.Int64(local_expert) * fx.Int64(4),
                                        fx.Int32(1),
                                    )
                                )
                        row_slot = fx.Int32(
                            rocdl.readlane(T.i32, row_slot_lane0, 0)
                        )
                        logical_tile = row_slot // fx.Int32(BM)
                        row_in_tile = row_slot % fx.Int32(BM)
                        map_index = (
                            local_expert * fx.Int32(max_tiles_per_expert)
                            + logical_tile
                        )
                        map_ready = (
                            _peer_addr(dest, "expert_tile_map_ready")
                            + fx.Int64(map_index) * 8
                        )
                        physical_lane0 = fx.Int32(0)
                        if lane == fx.Int32(0):
                            if _is_group_head(row_in_tile, row_slot):
                                physical_lane0 = _claim_tile_group(
                                    dest, map_index, local_expert, map_ready
                                )
                            else:
                                if const_expr(_FANOUT_PROBE == 1):
                                    physical_lane0 = logical_tile
                                else:
                                    _spin_dbg_15(
                                        map_ready, generation
                                    )
                                    physical_lane0 = buffer_ops.buffer_load(
                                        buffer_ops.create_buffer_resource_from_addr(
                                            _peer_addr(dest, "expert_tile_map")
                                        ),
                                        map_index,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                        physical = fx.Int32(
                            rocdl.readlane(T.i32, physical_lane0, 0)
                        )
                        grouped_row = (
                            physical * fx.Int32(BM) + row_in_tile
                        )

                        _n_scale = (
                            fx.Int32(0)
                            if const_expr(_FANOUT_PROBE == 3)
                            else fx.Int32(scale_bytes)
                        )
                        for scale_index in range(lane, _n_scale, fx.Int32(64)):
                            scale = buffer_ops.buffer_load(
                                rec,
                                fx.Int32(wire.payload_bytes) + scale_index,
                                vec_width=1,
                                dtype=T.i8,
                            )
                            ku = scale_index // fx.Int32(8)
                            ikxdl = (scale_index % fx.Int32(8)) // fx.Int32(4)
                            k_lane = scale_index % fx.Int32(4)
                            # BM=32*S 时每个 tile 占 S 个 32 行 scale chunk,gemm1 的
                            # issue_a_scale_load() 从 m_row//32 起连读 S 个 chunk,所以
                            # 这里也必须按 32 行子块落位,否则 BM>32 会读到错位的 scale。
                            a_sub = row_in_tile // fx.Int32(32)
                            a_row = row_in_tile % fx.Int32(32)
                            im_a = a_row // fx.Int32(16)
                            n_lane = a_row % fx.Int32(16)
                            dst_dword = (
                                (physical * fx.Int32(BM // 32) + a_sub)
                                * fx.Int32(scale_dwords * 32)
                                + ku * fx.Int32(64)
                                + k_lane * fx.Int32(16)
                                + n_lane
                            )
                            dst_byte = (
                                dst_dword * fx.Int32(4)
                                + ikxdl * fx.Int32(2)
                                + im_a
                            )
                            buffer_ops.buffer_store(
                                scale,
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "grouped_input_scale")
                                ),
                                dst_byte,
                                offset_is_bytes=True,
                            )
                        if const_expr(_FANOUT_PROBE != 2):
                            rocdl.s_waitcnt(0)

                        if lane == fx.Int32(0):
                            weight = buffer_ops.buffer_load(
                                rec,
                                fx.Int32(wire.weights_offset // 4) + topk_slot,
                                vec_width=1,
                                dtype=T.f32,
                            )
                            source_encoding = source_index | (
                                topk_slot << fx.Int32(24)
                            )
                            buffer_ops.buffer_store(
                                source_index,
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "tile_row_input")
                                ),
                                grouped_row,
                            )
                            buffer_ops.buffer_store(
                                source_encoding,
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "tile_row_source")
                                ),
                                grouped_row,
                            )
                            buffer_ops.buffer_store(
                                weight,
                                buffer_ops.create_buffer_resource_from_addr(
                                    _peer_addr(dest, "tile_row_weight")
                                ),
                                grouped_row,
                            )
                            comm_ops.fence_system_release()
                            if const_expr(not diagnostic_no_arrival_rmw):
                                completed = fx.Int32(
                                    comm_ops.atomic_add_system_acq_rel(
                                        _peer_addr(dest, "tile_row_done")
                                        + fx.Int64(physical) * 4,
                                        fx.Int32(1),
                                    )
                                )
                                # CTA 版在这里发布满 tile,legacy 的 wave 探针
                                # 丢了这一步(只 `_ = completed`),那会让
                                # tile_pipeline 失效、GMM1 只能等 EOS。
                                if const_expr(tile_pipeline):
                                    if completed == fx.Int32(BM - 1):
                                        _enqueue_tile_jobs(
                                            dest, physical, True
                                        )
                                else:
                                    _ = completed

        def _dispatch_record_wave(record, dest, source_index):
            """Place one record with one independent 64-lane wave.

            The legacy split probe assigns a whole 256-thread CTA to a token.
            This diagnostic body keeps the exact destination ABI and scoreboard
            protocol while replacing CTA scratch/barriers with lane-0 values
            broadcast through ``readlane``.  Four waves can consequently place
            four unrelated records at the same time.
            """

            rec = buffer_ops.create_buffer_resource_from_addr(
                record, num_records_bytes=record_bytes
            )
            packed_lane0 = fx.Int32(0)
            if lane == fx.Int32(0):
                found = fx.Int32(0)
                found_slot = fx.Int32(0)
                found_expert = fx.Int32(0)
                dest_global = fx.Int32(node * GPUS_PER_NODE) + dest
                for slot in range_constexpr(TOPK):
                    expert = buffer_ops.buffer_load(
                        rec,
                        fx.Int32(wire.ids_offset // 4 + slot),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    valid = (
                        expert >= dest_global * fx.Int32(LOCAL_EXPERTS)
                    ) & (
                        expert
                        < (dest_global + fx.Int32(1))
                        * fx.Int32(LOCAL_EXPERTS)
                    )
                    if valid & (found != fx.Int32(0)):
                        comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                    take = valid & (found == fx.Int32(0))
                    found_slot = take.select(fx.Int32(slot), found_slot)
                    found_expert = take.select(
                        expert - dest_global * fx.Int32(LOCAL_EXPERTS),
                        found_expert,
                    )
                    found = valid.select(fx.Int32(1), found)
                if found == fx.Int32(0):
                    comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                packed_lane0 = found_expert | (found_slot << fx.Int32(8))
            packed = fx.Int32(rocdl.readlane(T.i32, packed_lane0, 0))
            local_expert = packed & fx.Int32(0xFF)
            topk_slot = (packed >> fx.Int32(8)) & fx.Int32(0xFF)

            row_slot_lane0 = fx.Int32(0)
            if lane == fx.Int32(0):
                row_slot_lane0 = fx.Int32(
                    comm_ops.atomic_add_system(
                        _peer_addr(dest, "expert_count")
                        + fx.Int64(local_expert) * fx.Int64(4),
                        fx.Int32(1),
                    )
                )
            row_slot = fx.Int32(rocdl.readlane(T.i32, row_slot_lane0, 0))
            logical_tile = row_slot // fx.Int32(BM)
            row_in_tile = row_slot % fx.Int32(BM)
            map_index = (
                local_expert * fx.Int32(max_tiles_per_expert) + logical_tile
            )
            map_ready = (
                _peer_addr(dest, "expert_tile_map_ready")
                + fx.Int64(map_index) * 8
            )

            physical_lane0 = fx.Int32(0)
            if lane == fx.Int32(0):
                if _is_group_head(row_in_tile, row_slot):
                    physical_lane0 = _claim_tile_group(
                        dest, map_index, local_expert, map_ready
                    )
                else:
                    _spin_dbg_16(map_ready, generation)
                    physical_lane0 = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            _peer_addr(dest, "expert_tile_map")
                        ),
                        map_index,
                        vec_width=1,
                        dtype=T.i32,
                    )
            physical = fx.Int32(
                rocdl.readlane(T.i32, physical_lane0, 0)
            )
            grouped_row = physical * fx.Int32(BM) + row_in_tile

            dst_payload = buffer_ops.create_buffer_resource_from_addr(
                _peer_addr(dest, "grouped_input_q")
                + fx.Int64(source_index) * fx.Int64(wire.payload_bytes),
                num_records_bytes=wire.payload_bytes,
            )
            for dword in range(
                lane * fx.Int32(4),
                payload_dwords,
                fx.Int32(64 * 4),
            ):
                value = buffer_ops.buffer_load(
                    rec, dword, vec_width=4, dtype=T.i32
                )
                buffer_ops.buffer_store(value, dst_payload, dword)

            for scale_index in range(
                lane, fx.Int32(scale_bytes), fx.Int32(64)
            ):
                scale = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.payload_bytes) + scale_index,
                    vec_width=1,
                    dtype=T.i8,
                )
                ku = scale_index // fx.Int32(8)
                ikxdl = (scale_index % fx.Int32(8)) // fx.Int32(4)
                k_lane = scale_index % fx.Int32(4)
                # BM=32*S 时每个 tile 占 S 个 32 行 scale chunk,gemm1 的
                # issue_a_scale_load() 从 m_row//32 起连读 S 个 chunk,所以
                # 这里也必须按 32 行子块落位,否则 BM>32 会读到错位的 scale。
                a_sub = row_in_tile // fx.Int32(32)
                a_row = row_in_tile % fx.Int32(32)
                im_a = a_row // fx.Int32(16)
                n_lane = a_row % fx.Int32(16)
                dst_dword = (
                    (physical * fx.Int32(BM // 32) + a_sub)
                    * fx.Int32(scale_dwords * 32)
                    + ku * fx.Int32(64)
                    + k_lane * fx.Int32(16)
                    + n_lane
                )
                dst_byte = (
                    dst_dword * fx.Int32(4)
                    + ikxdl * fx.Int32(2)
                    + im_a
                )
                buffer_ops.buffer_store(
                    scale,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "grouped_input_scale")
                    ),
                    dst_byte,
                    offset_is_bytes=True,
                )
            rocdl.s_waitcnt(0)

            if lane == fx.Int32(0):
                weight = buffer_ops.buffer_load(
                    rec,
                    fx.Int32(wire.weights_offset // 4) + topk_slot,
                    vec_width=1,
                    dtype=T.f32,
                )
                source_encoding = source_index | (
                    topk_slot << fx.Int32(24)
                )
                buffer_ops.buffer_store(
                    source_index,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_input")
                    ),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    source_encoding,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_source")
                    ),
                    grouped_row,
                )
                buffer_ops.buffer_store(
                    weight,
                    buffer_ops.create_buffer_resource_from_addr(
                        _peer_addr(dest, "tile_row_weight")
                    ),
                    grouped_row,
                )
                comm_ops.fence_system_release()
                if const_expr(not diagnostic_no_arrival_rmw):
                    completed = fx.Int32(
                        comm_ops.atomic_add_system_acq_rel(
                            _peer_addr(dest, "tile_row_done")
                            + fx.Int64(physical) * 4,
                            fx.Int32(1),
                        )
                    )
                    _ = completed

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
                real_tiles = (count + fx.Int32(BM - 1)) // fx.Int32(BM)
                alloc_tiles = (
                    (real_tiles + fx.Int32(G - 1)) // fx.Int32(G)
                ) * fx.Int32(G)
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
                    _spin_dbg_36(
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
                if const_expr(diagnostic_no_arrival_rmw):
                    tile_count = (
                        count + fx.Int32(BM - 1)
                    ) // fx.Int32(BM)
                    tile_map = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("expert_tile_map")
                    )
                    row_done = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("tile_row_done")
                    )
                    for logical_tile in range(
                        fx.Int32(0), tile_count, fx.Int32(1)
                    ):
                        map_index = (
                            (tx + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert)
                            + logical_tile
                        )
                        _spin_dbg_35(
                            local_addr("expert_tile_map_ready")
                            + fx.Int64(map_index) * fx.Int64(8),
                            generation,
                        )
                        physical = buffer_ops.buffer_load(
                            tile_map,
                            map_index,
                            vec_width=1,
                            dtype=T.i32,
                        )
                        remaining = count - logical_tile * fx.Int32(BM)
                        valid_rows = (
                            remaining < fx.Int32(BM)
                        ).select(remaining, fx.Int32(BM))
                        buffer_ops.buffer_store(
                            valid_rows, row_done, physical
                        )
                # G>1 时一个 expert 认领的 tile 数被抬到 G 的倍数,所以要补的
                # 不只是最后一个真实 tile 的尾巴,还有整组里多出来的空 tile。
                if const_expr(dispatch_plan):
                    # S1 对拍:plan 是从 topk_ids 独立算出来的,必须和旧
                    # 协议一路原子加出来的 expert_count 逐 expert 相等。
                    _planned = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("plan_expert_rows")
                        ),
                        tx,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    if _planned != count:
                        comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                real_tiles = (count + fx.Int32(BM - 1)) // fx.Int32(BM)
                alloc_tiles = (
                    (real_tiles + fx.Int32(G - 1)) // fx.Int32(G)
                ) * fx.Int32(G)
                first_pad_tile = count // fx.Int32(BM)
                for pad_tile in range(first_pad_tile, alloc_tiles, fx.Int32(1)):
                    logical_tile = pad_tile
                    map_index = (tx + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert) + logical_tile
                    _spin_dbg_36(
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
                group_perm_res = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_group_perm")
                )
                group_e_res = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_expert_group")
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
                        _spin_dbg_37(
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
                        if const_expr(h1_phys):
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
                gl_egrp = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_expert_group")
                )
                gl_map0 = (ge + fx.Int32(EO)) * fx.Int32(max_tiles_per_expert)
                gl_groups = _alloc_tiles_for(
                    buffer_ops.buffer_load(
                        counts_res, ge + fx.Int32(EO), vec_width=1, dtype=T.i32
                    )
                ) // fx.Int32(G)
                for gl_j in range(gl, gl_groups, fx.Int32(SPL)):
                    gl_mi = gl_map0 + gl_j * fx.Int32(G)
                    _spin_dbg_37(
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
                    _spin_dbg_34(
                        local_addr("comm_eos")
                        + fx.Int64((GPUS_PER_NODE + peer) * 8),
                        generation,
                    )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            if const_expr(early_local_gmm):
                _ts(7)
                if const_expr(not lazy_pad):
                    _seal_pad_par(0)
            else:
                _seal_pad(0)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(early_local_gmm):
                _ts(8)
            # h1_phys 下排序视图整体在段 2 算(每个 expert 先本地组后远端组);
            # 这里只发布本地段 tile 数。否则本地段自成一段、段基址 0。
            if const_expr(not h1_phys):
                _seal_perm((0,), (), None, True)
            else:
                if tx == fx.Int32(0):
                    _lt = fx.Int32(0)
                    _cr = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("expert_count")
                    )
                    # local_defer:末尾 D 个 expert 的本地组不进本地段。消费者的本地上界和远端段起点
                    # 都由 tile_alloc[1] 推出,组表布局不变,这些组就落在远端段开头,
                    # 和它们的远端组一起在同一次权重读里做(远端段本来就要把权重全读一遍)。
                    for e in range_constexpr(LOCAL_EXPERTS - int(local_defer)):
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
                    (comm_ops.store_i64_global_system_relaxed if pub_relaxed else comm_ops.store_i64_global_system)(
                        local_addr("h1_local_eos"), generation
                    )
                _ts(9)

        dest = fx.Int32(0)
        shard = fx.Int32(0)
        # fanout_enabled 只是**诊断相位**开关(phase=="full" 时 k1/k2 都真),
        # 不是 kernel 角色开关。少了 _do_fan1/_do_fan2 这一项,split 路径会在
        # k2 里把 fanout 再跑一遍,每行放置两次 —— tile_alloc / expert_count
        # 精确翻倍,而消费者一边读 tile_alloc 一边被加,GMM1 的
        # expected_jobs 永远收不满(tag 41)。下面的传统路径一直是用
        # _do_fan1/_do_fan2 守的,所以从没暴露。
        if const_expr(
            diagnostic_split_fanout and fanout_enabled
            and (_do_fan1 or _do_fan2)
        ):
            dest = split_dest
            shard = split_shard
            if is_comm:
                if tx == fx.Int32(0):
                    _spin_dbg_17(
                        _peer_addr(dest, "launch_ready"), control_generation
                    )
            gpu.barrier()
            sparse_qp_token_mask = fx.Int64(0)
            if const_expr(cco_geometry == "sparse_wqe"):
                if is_inter_fanout:
                    qp_ready_scratch = fx.recast_iter(fx.Int64, lds_raw)
                    qp_ready_view = fx.make_view(
                        qp_ready_scratch, fx.make_layout(1, 1)
                    )
                    if tx == fx.Int32(0):
                        ready_qp = shard % fx.Int32(layout.num_qp)
                        observed = fx.Int64(
                            _spin_dbg_18(
                                local_addr("sparse_remote_qp_ready")
                                + fx.Int64(ready_qp) * fx.Int64(8),
                                generation
                                << fx.Int64(SPARSE_QP_GENERATION_SHIFT),
                            )
                        )
                        fx.ptr_store(
                            Vec.from_elements([observed], fx.Int64),
                            qp_ready_scratch,
                        )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    sparse_qp_token_mask = (
                        Vec(qp_ready_view.load())[0]
                        & fx.Int64(0xFFFFFFFF)
                    )
            completion_slot = is_inter_fanout.select(
                shard, fx.Int32(split_fanout_shards) + shard
            )

            if is_inter_fanout:
                if const_expr(diagnostic_wave_fanout):
                    if wave == fx.Int32(0):
                        for token in range(
                            shard,
                            ntokens,  # 上界必须是运行期 ntokens
                            fx.Int32(split_fanout_shards),
                        ):
                            if const_expr(cco_geometry == "mori64x2"):
                                ready_index = token // fx.Int32(64)
                            else:
                                chunk = token // records_per_chunk
                                qp_id = (
                                    token % records_per_chunk
                                ) // records_per_qp
                                ready_index = (
                                    chunk * fx.Int32(layout.num_qp) + qp_id
                                )
                            if lane == fx.Int32(0):
                                _spin_dbg_19(
                                    local_addr("remote_chunk_ready")
                                    + fx.Int64(ready_index) * fx.Int64(8),
                                    generation,
                                )
                            comm_ops.fence_system_acquire()
                            _dispatch_record_wave(
                                local_addr("remote_dispatch_rx")
                                + fx.Int64(token * record_bytes),
                                dest,
                                fx.Int32(
                                    remote_source_rank * MAX_TOKENS + token
                                ),
                            )
                else:
                    for token in range(
                        shard,
                        ntokens,  # 上界必须是运行期 ntokens
                        fx.Int32(split_fanout_shards),
                    ):
                        remote_record = (
                            local_addr("remote_dispatch_rx")
                            + fx.Int64(token * record_bytes)
                        )
                        if const_expr(cco_geometry == "sparse_wqe"):
                            ready_bit = token // fx.Int32(layout.num_qp)
                            if (
                                (sparse_qp_token_mask >> fx.Int64(ready_bit))
                                & fx.Int64(1)
                            ) != fx.Int64(0):
                                if _record_targets_dest(remote_record, dest):
                                    _dispatch_record(
                                        remote_record,
                                        dest,
                                        fx.Int32(
                                            remote_source_rank * MAX_TOKENS
                                            + token
                                        ),
                                    )
                        else:
                            if const_expr(cco_geometry == "mori64x2"):
                                ready_index = token // fx.Int32(64)
                            else:
                                chunk = token // records_per_chunk
                                qp_id = (
                                    token % records_per_chunk
                                ) // records_per_qp
                                ready_index = (
                                    chunk * fx.Int32(layout.num_qp) + qp_id
                                )
                            if tx == fx.Int32(0):
                                _spin_dbg_20(
                                    local_addr("remote_chunk_ready")
                                    + fx.Int64(ready_index) * fx.Int64(8),
                                    generation,
                                )
                            gpu.barrier()
                            _dispatch_record(
                                remote_record,
                                dest,
                                fx.Int32(
                                    remote_source_rank * MAX_TOKENS + token
                                ),
                            )
            if is_intra_fanout:
                if const_expr(diagnostic_wave_fanout):
                    if wave == fx.Int32(0):
                        for token in range(
                            shard,
                            ntokens,  # 上界必须是运行期 ntokens
                            fx.Int32(split_fanout_shards),
                        ):
                            if lane == fx.Int32(0):
                                _spin_dbg_21(
                                    local_addr("dispatch_staging_ready")
                                    + fx.Int64(token) * 8,
                                    generation,
                                )
                            comm_ops.fence_system_acquire()
                            _dispatch_record_wave(
                                local_addr("dispatch_staging")
                                + fx.Int64(token * record_bytes),
                                dest,
                                fx.Int32(rank * MAX_TOKENS + token),
                            )
                else:
                    for token in range(
                        shard,
                        ntokens,  # 上界必须是运行期 ntokens
                        fx.Int32(split_fanout_shards),
                    ):
                        if tx == fx.Int32(0):
                            _spin_dbg_22(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(token) * 8,
                                generation,
                            )
                        gpu.barrier()
                        local_record = (
                            local_addr("dispatch_staging")
                            + fx.Int64(token * record_bytes)
                        )
                        if const_expr(cco_geometry == "sparse_wqe"):
                            if _record_targets_dest(local_record, dest):
                                _dispatch_record(
                                    local_record,
                                    dest,
                                    fx.Int32(rank * MAX_TOKENS + token),
                                )
                        else:
                            _dispatch_record(
                                local_record,
                                dest,
                                fx.Int32(rank * MAX_TOKENS + token),
                            )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if const_expr(diagnostic_wave_fanout):
                if is_comm & (tx == fx.Int32(0)):
                    comm_ops.fence_system_release()
                    flag_index = dest * fx.Int32(32) + completion_slot
                    comm_ops.store_i64_global_system(
                        local_addr("fanout_done")
                        + fx.Int64(flag_index) * fx.Int64(8),
                        generation,
                    )
            else:
                if is_comm & (tx == fx.Int32(0)):
                    comm_ops.fence_system_release()
                    flag_index = dest * fx.Int32(32) + completion_slot
                    comm_ops.store_i64_global_system(
                        local_addr("fanout_done")
                        + fx.Int64(flag_index) * fx.Int64(8),
                        generation,
                    )
            gpu.barrier()

            # One local-domain coordinator per destination turns all unique
            # producer flags into the legacy one-EOS/one-consumed contribution.
            is_dest_coordinator = is_intra_fanout & (
                split_group == fx.Int32(0)
            )
            if is_dest_coordinator:
                if tx == fx.Int32(0):
                    for producer in range_constexpr(
                        2 * split_fanout_shards
                    ):
                        flag_index = dest * fx.Int32(32) + fx.Int32(producer)
                        _spin_dbg_23(
                            local_addr("fanout_done")
                            + fx.Int64(flag_index) * fx.Int64(8),
                            generation,
                        )
                    comm_ops.fence_system_acquire()
                    if const_expr(cco_geometry == "sparse_wqe"):
                        comm_ops.atomic_add_system_acq_rel(
                            local_addr("sparse_remote_consumed"),
                            fx.Int32(1),
                        )
                    else:
                        consumed_words = (
                            2
                            if cco_geometry == "mori64x2"
                            else dispatch_chunks * layout.num_qp
                        )
                        for consume_index in range_constexpr(
                            consumed_words
                        ):
                            comm_ops.atomic_add_system_acq_rel(
                                local_addr("remote_chunk_consumed")
                                + fx.Int64(consume_index * 4),
                                fx.Int32(1),
                            )
                    comm_ops.fence_system_release()
                    (comm_ops.store_i64_global_system_relaxed if pub_relaxed else comm_ops.store_i64_global_system)(
                        _peer_addr(dest, "comm_eos")
                        + fx.Int64(local_rank) * 8,
                        generation,
                    )
                gpu.barrier()
        else:
            # fanout 的并发度:原本 is_comm = ticket < GPUS_PER_NODE,即 8 个 CTA
            # 各钉死一个目的 rank、串行走完全部 MAX_TOKENS。实测这一段占 stage1
            # 的 82%(8,338us 搬 15.6MB = 1.87 GB/s,而同一块 arena 流式读是
            # 3.6 TB/s)—— 是并发度不是带宽。分片后 CTA t 负责
            # dest = t % GPUS_PER_NODE、shard = t // GPUS_PER_NODE,token 按
            # shard 跨步。借用的是 producer CTA(量化只要 ~10us 就闲置),
            # 不动 GMM1 消费者池。
            # 不做任何 fanout 的那一半绝不能进这个块:块首就对**对端**的
            # launch_ready 自旋,而对端的 launch_ready 只由它自己的初始化者
            # (k1)发布。k2 白白去等一个跟它无关的跨 rank 握手。
            if const_expr(_do_fan1 or _do_fan2):
                is_fanout = ticket < fx.Int32(GPUS_PER_NODE * fanout_shards)
                if const_expr(t0_no_fanout):
                    # T0(CCO 收发 + credit + seal)是关键路径,不再兼做 dest0
                    # 的 0 号分片;dest0 由其余 fanout_shards-1 个分片分掉。
                    is_fanout = is_fanout & (ticket != fx.Int32(0))
            else:
                is_fanout = _never
            if is_fanout:
                fan_stride = fx.Int32(fanout_shards)
                if const_expr(fanout_shards == 1):
                    dest = ticket
                    fanout_shard = fx.Int32(0)
                else:
                    dest = ticket % fx.Int32(GPUS_PER_NODE)
                    fanout_shard = ticket // fx.Int32(GPUS_PER_NODE)
                    if const_expr(t0_no_fanout):
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
                if const_expr(post_nofan):
                    pn_dest = (dest >= fx.Int32(_POST_LO)) & (dest < fx.Int32(_POST_HI))
                    pn_self = (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI))
                    fan_tstride = pn_dest.select(fan_stride - fx.Int32(1), fan_stride)
                    fan_wshard = pn_self.select(
                        fx.Int32(1 << 20),
                        pn_dest.select(fanout_shard - fx.Int32(1), fanout_shard),
                    )
                _acc_zero()
                _ts_val(2, dest + fx.Int32(1))
                _ts_val(19, fanout_shard + fx.Int32(1))
                if const_expr(rail_post_ctas):
                    # 每个 QP 一个 CTA 投递+敲门铃(照 kernel2 的 DBCTAS):ticket q 发 QP q。
                    if (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI)):
                        if wave == fx.Int32(0):
                            _rail_post(ticket - fx.Int32(1 - _POST_T0), True)
                if tx == fx.Int32(0):
                    _spin_dbg_24(
                        _peer_addr(dest, "launch_ready"), control_generation
                    )
                gpu.barrier()
                _ts(13)
                if const_expr(wave_fanout):
                    # 本 CTA 拥有的 token 集合是 {fanout_shard + k*fanout_shards}。
                    # 按 k mod WAVES 再切一刀分给 4 个 wave,步长因此是
                    # WAVES*fanout_shards。整段无 gpu.barrier():每个 wave 的
                    # 迭代次数不同,CTA 级 barrier 在这里必死锁。
                    for token in range(
                        fanout_shard + wave * fx.Int32(fanout_shards),
                        fx.Int32(MAX_TOKENS),
                        WAVES * fanout_shards,
                    ):
                        if lane == fx.Int32(0):
                            _spin_dbg_25(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(token) * 8,
                                generation,
                            )
                        comm_ops.fence_system_acquire()
                        _dispatch_record_wave_multi(
                            local_addr("dispatch_staging")
                            + fx.Int64(token * record_bytes),
                            dest,
                            fx.Int32(rank * MAX_TOKENS + token),
                        )
                    for token in range(
                        fanout_shard + wave * fx.Int32(fanout_shards),
                        fx.Int32(MAX_TOKENS),
                        WAVES * fanout_shards,
                    ):
                        chunk = token // records_per_chunk
                        qp_id = (token % records_per_chunk) // records_per_qp
                        if lane == fx.Int32(0):
                            _spin_dbg_26(
                                local_addr("remote_chunk_ready")
                                + fx.Int64((chunk * layout.num_qp + qp_id) * 8),
                                generation,
                            )
                        comm_ops.fence_system_acquire()
                        _dispatch_record_wave_multi(
                            local_addr("remote_dispatch_rx")
                            + fx.Int64(token * record_bytes),
                            dest,
                            fx.Int32(remote_source_rank * MAX_TOKENS + token),
                        )
                else:
                  # 和 CCO CTA 那边同一个毛病:逐个 token 用单线程去看一个
                  # 早已置位的 flag,每次一条 system scope load,串行依赖。
                  # 实测所有 producer 在 ~98us 就发布完了,所以这些等待几乎
                  # 全是白等。把它们提到循环前、用整个 CTA 一次看完。
                  # 见 [[串行等 flag 的 700us]]。
                  if const_expr(wide_fanout_wait and _do_fan1 and not fan1_direct):
                    _my_tokens = (MAX_TOKENS + fanout_shards - 2) // (
                        fanout_shards - 1 if t0_no_fanout else fanout_shards
                    ) + 1
                    for _r in range_constexpr(
                        (_my_tokens + THREADS - 1) // THREADS
                    ):
                        _idx = fx.Int32(_r * THREADS) + tx
                        _tok = fanout_shard + _idx * fan_stride
                        if _tok < fx.Int32(MAX_TOKENS):
                            _spin_dbg_27(
                                local_addr("dispatch_staging_ready")
                                + fx.Int64(_tok) * 8,
                                generation,
                            )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                  _ts(14)
                  if const_expr(route_gbatch):
                    for token in range(
                      fan_wshard, fx.Int32(_K1_TOK), fan_tstride * fx.Int32(_GB_NB)
                    ):
                      _route_gbatch(
                        buffer_ops.create_buffer_resource_from_addr(x_q),
                        token, fan_tstride, dest, fx.Int32(rank * MAX_TOKENS),
                        emit_masks=rail_soa,
                      )
                  for token in range(
                    fanout_shard,
                    fx.Int32(0 if route_gbatch else _K1_TOK),
                    fan_stride,
                  ):
                    if const_expr(fan1_direct):
                        # 本地来源直接读量化输入(k1 之前已就绪),不等 producer。
                        if token < ntokens:
                            _dispatch_local_direct(
                                token, dest, fx.Int32(rank * MAX_TOKENS) + token
                            )
                    else:
                        if const_expr(not wide_fanout_wait):
                            if tx == fx.Int32(0):
                                _spin_dbg_28(
                                    local_addr("dispatch_staging_ready")
                                    + fx.Int64(token) * 8,
                                    generation,
                                )
                            gpu.barrier()
                        _dispatch_record(
                            local_addr("dispatch_staging")
                            + fx.Int64(token * record_bytes),
                            dest,
                            fx.Int32(rank * MAX_TOKENS + token),
                        )
                  if const_expr(split_local and _do_fan1 and _do_fan2):
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
                            (comm_ops.store_i64_global_system_relaxed if pub_relaxed else comm_ops.store_i64_global_system)(
                                _peer_addr(dest, "comm_eos")
                                + fx.Int64((GPUS_PER_NODE + local_rank) * 8),
                                generation,
                            )
                  _ts(15)
                  if const_expr(early_local_gmm):
                    # F5:本地组封尾挪到 fan CTA(ticket 8+local_rank,非 T0/非 rail post)的 fan1 之后,
                    # 不再等 sealer 自己做完 fan2(E1T:本地门 ~175µs ≈ 全局 comm_eos 时刻)。
                    # 只等 node 内各源的 fan1,fan1 不依赖 fan2/封尾,无环。
                    # fan2_shards 时挪到 fan2 池外(shard=F2S),不拖本 dest 的 fan2。
                    if ticket == fx.Int32(GPUS_PER_NODE * _SEG1_SHARD + local_rank):
                      _seal_seg1()
                  if const_expr(wide_fanout_wait and _do_fan2):
                    # 这里比原来保守:等**全部** chunk 而不是本 token 那个。
                    # dispatch_chunks=2 且整批只有一次 flush,两者实际同时到,
                    # 代价上限是半次传输(~10us),换掉每 token 一次的自旋。
                    if (tx < fx.Int32(
                        layout.num_qp if rail_soa else dispatch_chunks * layout.num_qp
                    )) & (fan_wshard < fx.Int32(F2S if F2S else 1 << 20)):
                        _spin_dbg_29(
                            local_addr(
                                "sparse_remote_qp_ready"
                                if rail_ids_first
                                else "remote_chunk_ready"
                            )
                            + fx.Int64(tx) * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                  _ts(16)
                  if const_expr(route_gbatch):
                    f2_stride = fx.Int32(F2S) if F2S else fan_tstride
                    f2_end = (
                        (fan_wshard < fx.Int32(F2S)).select(
                            fx.Int32(_K2_TOK), fx.Int32(0)
                        )
                        if F2S
                        else fx.Int32(_K2_TOK)
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
                        ids_side=rail_ids_first,
                        remote=True,
                      )
                  for token in range(
                    fanout_shard,
                    fx.Int32(0 if route_gbatch else _K2_TOK),
                    fan_stride,
                  ):
                    if const_expr(rail_soa):
                        if token < ntokens:
                            _dispatch_remote_soa(
                                token,
                                dest,
                                fx.Int32(remote_source_rank * MAX_TOKENS) + token,
                            )
                    else:
                        if const_expr(not wide_fanout_wait):
                            chunk = token // records_per_chunk
                            qp_id = (
                                token % records_per_chunk
                            ) // records_per_qp
                            if tx == fx.Int32(0):
                                _spin_dbg_30(
                                    local_addr("remote_chunk_ready")
                                    + fx.Int64(
                                        (chunk * layout.num_qp + qp_id) * 8
                                    ),
                                    generation,
                                )
                            gpu.barrier()
                        _dispatch_record(
                            local_addr("remote_dispatch_rx")
                            + fx.Int64(token * record_bytes),
                            dest,
                            fx.Int32(remote_source_rank * MAX_TOKENS + token),
                        )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                _ts(17)
                if const_expr(rail_post_ctas):
                    # 在发布本分片之前回收自己的数据请求:consumed 到齐前必然完成,
                    # T0 的 credit 段不会和这里同时轮询同一个 QP 的 CQ。
                    if (ticket >= fx.Int32(_POST_LO)) & (ticket < fx.Int32(_POST_HI)):
                        if wave == fx.Int32(0):
                            post_qp = ticket - fx.Int32(1 - _POST_T0)
                            post_req = fx.Int64(
                                comm_ops.load_i64_global(
                                    local_addr("remote_chunk_request")
                                    + fx.Int64(post_qp) * fx.Int64(8)
                                )
                            )
                            _rail.wait(dev_comm, post_qp, post_req)
                        gpu.barrier()
                if const_expr(timeline_instrument):
                    if (ticket == fx.Int32(COMPUTE_FIRST)) & (
                        tx == fx.Int32(0)
                    ):
                        comm_ops.store_i64_global_relaxed(
                            timeline_addr
                            + fx.Int64(
                                STAGE2_TIMELINE_INDEX[
                                    "stage1_fanout_self_done"
                                ]
                                * 8
                            ),
                            fx.Int64(comm_ops.read_wall_clock()),
                        )
                # k1 一律不碰 fanout_shard_done / comm_eos /
                # remote_chunk_consumed:这三者都描述「本 dest 的全部 record
                # 都推完了」,而 k1 只推完了本地来源那一半。
                if (tx == fx.Int32(0)) & (
                    fx.Int32(1) == fx.Int32(1 if _do_eos else 0)
                ):
                    if const_expr(fanout_shards == 1):
                        _publish = fx.Int32(1) == fx.Int32(1)
                    else:
                        # 每个 dest 的最后一个分片才发 EOS 和信用原子:
                        # remote_chunk_consumed 的每个字必须恰好累到
                        # GPUS_PER_NODE,分片不能各加一次。用减法复位
                        # (和 stage2 到达协议同一手法),不依赖下一代的清零。
                        if const_expr(route_batch):
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
                        else:
                            _prev = fx.Int32(
                                comm_ops.atomic_add_agent(
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
                        (comm_ops.store_i64_global_system_relaxed if pub_relaxed else comm_ops.store_i64_global_system)(
                            _peer_addr(dest, "comm_eos")
                            + fx.Int64(local_rank) * 8,
                            generation,
                        )
                _ts(18)
                _acc_flush()

        # The CCO CTA is also destination role zero, so delayed-credit progress
        # must run only after the common is_comm path above has contributed its
        # own consumed count. Otherwise the scoreboard can reach only seven.
        if const_expr(diagnostic_phase == "transport_only"):
            if is_cco:
                consumed = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("remote_chunk_consumed")
                )
                consumed_words = (
                    2
                    if cco_geometry == "mori64x2"
                    else dispatch_chunks * layout.num_qp
                )
                if tx < fx.Int32(consumed_words):
                    buffer_ops.buffer_store(
                        fx.Int32(GPUS_PER_NODE), consumed, tx
                    )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                comm_ops.fence_system_release()

        if is_cco_credit:
            qp = wave
            if const_expr(cco_geometry == "sparse_wqe"):
                if wave == fx.Int32(0):
                    if lane == fx.Int32(0):
                        consumed_count = fx.Int32(
                            comm_ops.load_i32_global_system(
                                local_addr("sparse_remote_consumed")
                            )
                        )
                        _wdl45 = fx.Int64(comm_ops.read_wall_clock())
                        while consumed_count < fx.Int32(GPUS_PER_NODE):
                            consumed_count = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("sparse_remote_consumed")
                                )
                            )
                            if const_expr(_SPIN_DEADLINE > 0):
                                _over = (fx.Int64(comm_ops.read_wall_clock()) - _wdl45) // fx.Int64(
                                    _SPIN_DEADLINE
                                )
                                consumed_count = consumed_count + fx.Int32(_over) * fx.Int32(GPUS_PER_NODE)
                        if const_expr(_SPIN_DEADLINE > 0):
                            if (fx.Int64(comm_ops.read_wall_clock()) - _wdl45) >= fx.Int64(
                                _SPIN_DEADLINE
                            ):
                                comm_ops.store_i64_global_system(
                                    local_addr("plan_debug") + fx.Int64(45 * 8),
                                    generation,
                                )
                    comm_ops.fence_system_acquire()
                    _rail.put_value(
                        dev_comm,
                        fx.Int32(0),
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("sparse_remote_credit"),
                        generation,
                        aggregate=True,
                    )
                    credit_request = _rail.flush_async(
                        dev_comm,
                        fx.Int32(0),
                        fx.Int32(remote_node),
                    )
                    if const_expr(_LIVE_MARK):
                        if tx == fx.Int32(0):        # rail_enter
                            comm_ops.atomic_add_agent(
                                local_addr("plan_debug") + fx.Int64(56 * 8),
                                fx.Int32(1),
                            )
                    _rail.wait(
                        dev_comm,
                        fx.Int32(0),
                        credit_request,
                    )
                    if const_expr(_LIVE_MARK):
                        if tx == fx.Int32(0):        # rail_exit
                            comm_ops.atomic_add_agent(
                                local_addr("plan_debug") + fx.Int64(57 * 8),
                                fx.Int32(1),
                            )
                    if lane == fx.Int32(0):
                        _spin_dbg_31(
                            local_addr("sparse_remote_credit"), generation
                        )
                    comm_ops.fence_system_acquire()
                    for request_qp in range_constexpr(layout.num_qp):
                        request = fx.Int64(
                            comm_ops.load_i64_global(
                                local_addr("sparse_remote_request")
                                + fx.Int64(request_qp * 8)
                            )
                        )
                        if const_expr(_LIVE_MARK):
                            if tx == fx.Int32(0):        # rail_enter
                                comm_ops.atomic_add_agent(
                                    local_addr("plan_debug") + fx.Int64(58 * 8),
                                    fx.Int32(1),
                                )
                        _rail.wait(
                            dev_comm,
                            fx.Int32(request_qp),
                            request,
                        )
                        if const_expr(_LIVE_MARK):
                            if tx == fx.Int32(0):        # rail_exit
                                comm_ops.atomic_add_agent(
                                    local_addr("plan_debug") + fx.Int64(59 * 8),
                                    fx.Int32(1),
                                )
                gpu.barrier()
            if const_expr(cco_geometry == "mori64x2"):
                active_rail = wave < fx.Int32(2)
                if active_rail:
                    half = wave
                    if lane == fx.Int32(0):
                        consumed_count = fx.Int32(
                            comm_ops.load_i32_global_system(
                                local_addr("remote_chunk_consumed")
                                + fx.Int64(half) * fx.Int64(4)
                            )
                        )
                        _wdl44 = fx.Int64(comm_ops.read_wall_clock())
                        while consumed_count < fx.Int32(GPUS_PER_NODE):
                            consumed_count = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("remote_chunk_consumed")
                                    + fx.Int64(half) * fx.Int64(4)
                                )
                            )
                            if const_expr(_SPIN_DEADLINE > 0):
                                _over = (fx.Int64(comm_ops.read_wall_clock()) - _wdl44) // fx.Int64(
                                    _SPIN_DEADLINE
                                )
                                consumed_count = consumed_count + fx.Int32(_over) * fx.Int32(GPUS_PER_NODE)
                        if const_expr(_SPIN_DEADLINE > 0):
                            if (fx.Int64(comm_ops.read_wall_clock()) - _wdl44) >= fx.Int64(
                                _SPIN_DEADLINE
                            ):
                                comm_ops.store_i64_global_system(
                                    local_addr("plan_debug") + fx.Int64(44 * 8),
                                    generation,
                                )
                    comm_ops.fence_system_acquire()
                    _rail.put_value(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                        arena_win,
                        window_off("remote_chunk_credit")
                        + fx.Int64(half) * fx.Int64(8),
                        generation,
                        aggregate=True,
                    )
                    credit_req = _rail.flush_async(
                        dev_comm,
                        qp,
                        fx.Int32(remote_node),
                    )
                    if const_expr(_LIVE_MARK):
                        if tx == fx.Int32(0):        # rail_enter
                            comm_ops.atomic_add_agent(
                                local_addr("plan_debug") + fx.Int64(60 * 8),
                                fx.Int32(1),
                            )
                    _rail.wait(dev_comm, qp, credit_req)
                    if const_expr(_LIVE_MARK):
                        if tx == fx.Int32(0):        # rail_exit
                            comm_ops.atomic_add_agent(
                                local_addr("plan_debug") + fx.Int64(61 * 8),
                                fx.Int32(1),
                            )
                    if lane == fx.Int32(0):
                        _spin_dbg_32(
                            local_addr("remote_chunk_credit")
                            + fx.Int64(half) * fx.Int64(8),
                            generation,
                        )
                    original_request = fx.Int64(
                        comm_ops.load_i64_global(
                            local_addr("remote_chunk_request")
                            + fx.Int64(half) * fx.Int64(8)
                        )
                    )
                    if const_expr(_LIVE_MARK):
                        if tx == fx.Int32(0):        # rail_enter
                            comm_ops.atomic_add_agent(
                                local_addr("plan_debug") + fx.Int64(62 * 8),
                                fx.Int32(1),
                            )
                    _rail.wait(
                        dev_comm, qp, original_request
                    )
                    if const_expr(_LIVE_MARK):
                        if tx == fx.Int32(0):        # rail_exit
                            comm_ops.atomic_add_agent(
                                local_addr("plan_debug") + fx.Int64(63 * 8),
                                fx.Int32(1),
                            )
                gpu.barrier()

            credit_batch_count = (
                dispatch_chunks // cco_chunks_per_flush
                if cco_geometry == "chunked"
                else 0
            )
            for batch in range(
                fx.Int32(0), fx.Int32(credit_batch_count), fx.Int32(1)
            ):
                batch_first = batch * fx.Int32(cco_chunks_per_flush)
                # Do not credit any chunk in the batch until every destination
                # fanout role has consumed all B reciprocal payloads.
                for batch_item in range_constexpr(cco_chunks_per_flush):
                    chunk = batch_first + fx.Int32(batch_item)
                    consume_index = chunk * fx.Int32(layout.num_qp) + qp
                    if lane == fx.Int32(0):
                        consumed_count = fx.Int32(
                            comm_ops.load_i32_global_system(
                                local_addr("remote_chunk_consumed")
                                + fx.Int64(consume_index) * fx.Int64(4)
                            )
                        )
                        _wdl43 = fx.Int64(comm_ops.read_wall_clock())
                        while consumed_count < fx.Int32(GPUS_PER_NODE):
                            consumed_count = fx.Int32(
                                comm_ops.load_i32_global_system(
                                    local_addr("remote_chunk_consumed")
                                    + fx.Int64(consume_index) * fx.Int64(4)
                                )
                            )
                            if const_expr(_SPIN_DEADLINE > 0):
                                _over = (fx.Int64(comm_ops.read_wall_clock()) - _wdl43) // fx.Int64(
                                    _SPIN_DEADLINE
                                )
                                consumed_count = consumed_count + fx.Int32(_over) * fx.Int32(GPUS_PER_NODE)
                        if const_expr(_SPIN_DEADLINE > 0):
                            if (fx.Int64(comm_ops.read_wall_clock()) - _wdl43) >= fx.Int64(
                                _SPIN_DEADLINE
                            ):
                                comm_ops.store_i64_global_system(
                                    local_addr("plan_debug") + fx.Int64(43 * 8),
                                    generation,
                                )
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    _ts(8)
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
                if const_expr(_LIVE_MARK):
                    if tx == fx.Int32(0):        # rail_enter
                        comm_ops.atomic_add_agent(
                            local_addr("plan_debug") + fx.Int64(64 * 8),
                            fx.Int32(1),
                        )
                _rail.wait(dev_comm, qp, credit_req)
                _ts(9)
                if const_expr(_LIVE_MARK):
                    if tx == fx.Int32(0):        # rail_exit
                        comm_ops.atomic_add_agent(
                            local_addr("plan_debug") + fx.Int64(65 * 8),
                            fx.Int32(1),
                        )

                for batch_item in range_constexpr(
                    0 if credit_async else cco_chunks_per_flush
                ):
                    chunk = batch_first + batch_item
                    consume_index = fx.Int32(chunk * layout.num_qp) + qp
                    if lane == fx.Int32(0):
                        _spin_dbg_33(
                            local_addr("remote_chunk_credit")
                            + fx.Int64(consume_index) * fx.Int64(8),
                            generation,
                        )
                    gpu.barrier()

                # One retained data request belongs to the whole batch and is
                # reclaimed exactly once after all B reciprocal credits arrive.
                request_index = (
                    fx.Int64(batch_first * layout.num_qp) + fx.Int64(qp)
                )
                original_request = fx.Int64(
                    comm_ops.load_i64_global(
                        local_addr("remote_chunk_request")
                        + request_index * fx.Int64(8)
                    )
                )
                if const_expr(_LIVE_MARK):
                    if tx == fx.Int32(0):        # rail_enter
                        comm_ops.atomic_add_agent(
                            local_addr("plan_debug") + fx.Int64(66 * 8),
                            fx.Int32(1),
                        )
                if const_expr(rail_soa):
                    # rail_soa 只有 batch 0、wave < RAIL_QPS 发过数据请求。
                    if (batch == fx.Int32(0)) & (
                        qp < fx.Int32(
                            (_POST_T0 if rail_post_ctas else RAIL_QPS)
                        )
                    ):
                        _rail.wait(dev_comm, qp, original_request)
                else:
                    _rail.wait(dev_comm, qp, original_request)
                _ts(10)
                if const_expr(_LIVE_MARK):
                    if tx == fx.Int32(0):        # rail_exit
                        comm_ops.atomic_add_agent(
                            local_addr("plan_debug") + fx.Int64(67 * 8),
                            fx.Int32(1),
                        )

        # The role targeting this rank observes exactly eight EOS values; each
        # one covers a local source rank and its aligned remote source rank.
        if is_sealer:
            if const_expr(split_local and not early_local_gmm):
                _seal_seg1()
            if const_expr(early_local_gmm):
                # 段 1 由 ticket 8+local_rank 的 fan CTA 在它的 fan1 之后做(见 fanout 段);
                # 段 2 的置换/sortcopy 读本地 pad 行,必须排在段 1 之后。
                if tx == fx.Int32(0):
                    comm_ops.spin_until_ge_i64_sleep(
                        local_addr("h1_local_eos"), generation, 127
                    )
                gpu.barrier()
                comm_ops.fence_system_acquire()
            if tx == fx.Int32(0):
                # Independent lane acquires followed by a CTA barrier do not
                # merge into one happens-before chain. The publishing thread
                # itself must acquire all eight communication-role releases.
                for peer in range_constexpr(GPUS_PER_NODE):
                    _spin_dbg_34(
                        local_addr("comm_eos") + fx.Int64(peer * 8), generation
                    )
            gpu.barrier()
            comm_ops.fence_system_acquire()
            _ts(11)
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
            _ts(5)
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
            if const_expr(expert_major_output):
                # Expert-major view for Stage2.  expert_count is final here and
                # expert_tile_map[e][j] is dense in j, so one thread per expert
                # expands the permutation directly.  Each expert thread
                # recomputes its own tile base from the LOCAL_EXPERTS counts
                # rather than running a scan: the counts are one cache line
                # deep and this costs no LDS and no extra barrier.
                if const_expr(split_local and h1_phys):
                    _seal_perm((0, LOCAL_EXPERTS), (), tiles, False, seal_fast)
                    if const_expr(early_local_gmm):
                        _seal_group_list(LOCAL_EXPERTS, (0,), remote_rev)
                elif const_expr(split_local):
                    _seal_perm((LOCAL_EXPERTS,), (0,), tiles, False)
                else:
                    _seal_perm((0,), (), tiles, False)
                perm_res = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_dst_of_src")
                )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                _ts(6)
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
                    fx.Int32(0) if const_expr(sortcopy_k2) else tiles * fx.Int32(BM),
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
            total_jobs = (tiles // fx.Int32(GG)) * fx.Int32(
                h1_n_blocks if tile_pipeline else GNB
            )
            if const_expr(tile_pipeline):
                # Full tiles were already appended by their unique row-32
                # last-arriver.  After EOS, append exactly the partial expert
                # tails that were padded above.  Different threads may reserve
                # batches concurrently; per-batch generation is the publication
                # point, so reservation order need not equal ready order.
                row_done = buffer_ops.create_buffer_resource_from_addr(
                    local_addr("tile_row_done")
                )
                for physical in range(tx, tiles, fx.Int32(THREADS)):
                    completed_rows = buffer_ops.buffer_load(
                        row_done, physical, vec_width=1, dtype=T.i32
                    )
                    if completed_rows < fx.Int32(BM):
                        _enqueue_tile_jobs(
                            fx.Int32(local_rank), physical, False
                        )
                rocdl.s_waitcnt(0)
                gpu.barrier()
                if tx == fx.Int32(0):
                    final_tail = fx.Int32(
                        comm_ops.load_i32_global_system(
                            local_addr("h1_queue_tail")
                        )
                    )
                    if final_tail != total_jobs:
                        comm_ops.atomic_add_system(error_addr, fx.Int32(1))
                    buffer_ops.buffer_store(
                        tiles * fx.Int32(BM),
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("num_valid")
                        ),
                        fx.Int32(0),
                    )
                    comm_ops.fence_system_release()
                    # fence 已经把前面的写整体放行;release store 会再做一次整 L2 写回。
                    (comm_ops.store_i64_global_system_relaxed if pub_relaxed else comm_ops.store_i64_global_system)(
                        local_addr("h1_queue_eos"), generation
                    )
            else:
                # 非 pipeline 的 GMM1 按 tile_alloc 闭式跨步取作业,不读
                # h1_ready_queue / h1_ready_queue_generation,不再逐 job 写它们。
                rocdl.s_waitcnt(0)
                gpu.barrier()
                _ts(7)
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
                    (comm_ops.store_i64_global_system_relaxed if pub_relaxed else comm_ops.store_i64_global_system)(
                        local_addr("h1_queue_eos"), generation
                    )

        def _run_gemm1_job(job, nc=None):
            # nc = (领位头地址, 本段上界, 本段偏移):GMM1 体内提前领下一个 job(next_claim)。
            if const_expr(k2_sorted_jobs):
                sj_s = job // fx.Int32(GNB)
                sj_n = job - sj_s * fx.Int32(GNB)
                sj_p = buffer_ops.buffer_load(
                    buffer_ops.create_buffer_resource_from_addr(local_addr("tile_src_of_dst")),
                    sj_s * fx.Int32(GG),
                    vec_width=1,
                    dtype=T.i32,
                ) // fx.Int32(GG)
                job = fx.Int32(rocdl.readfirstlane(T.i32, sj_p * fx.Int32(GNB) + sj_n))
            # gemm1 rev: 3-stage A, 2-ahead DMA, wait_lds_barrier(vmcnt 24), ascale_gather v2, BN128 v2, lds alias scopes v1, epi acc pad4 v2, nofence barrier v1, next_claim v1(改 gemm1.py 时改这行,
            # 否则 flydsl 缓存 key 不变、继续跑旧 GMM1)
            _gemm1_body(
                lds_raw,
                local_addr("grouped_input_q"),
                local_addr("grouped_input_scale"),
                w1q,
                w1scale,
                (
                    local_addr("tile_expert_group")
                    if GG > 1
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
                gmm1_use_nt,
                fx.Int32(layout.source_capacity),
                fx.Int32(max_route_tiles // GG),
                (
                    (
                        local_addr("tile_group_perm")
                        if GG > 1
                        else local_addr("tile_dst_of_src")
                    )
                    if expert_major_output
                    else fx.Int64(0)
                ),
                nc[0] if nc is not None else fx.Int64(0),
                local_addr("gmm1_group_list") if nc is not None else fx.Int64(0),
                nc[1] if nc is not None else fx.Int32(0),
                nc[2] if nc is not None else fx.Int32(0),
                BM=GBM,
                expert_major=expert_major_output and not h1_phys,
                ascale_gather=ascale_gather,
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
        # GMM1 scheduler.  sparse_wqe production uses a ready-order queue:
        # full BM32 tiles are published by their last route arrival, so CTAs
        # that have completed their communication duty can overlap GMM1 with
        # the remaining fanout.  Partial tiles are sealed and appended at EOS.
        # Other geometries retain the post-EOS static reference scheduler.
        # ------------------------------------------------------------------
        if const_expr(not diagnostic_comm_only):
            # GMM1 消费者本来只有 ticket >= ESSENTIAL_CTAS 的 120 个,8 个 comm
            # 和 128 个 producer 干完自己的活就退出,整个 GMM1 阶段(实测 797us,
            # 占 stage1 的 46%)里 136/256 个 CTA 是空的。tile_pipeline 在生产
            # 路径恒为 False([[stage1 里 GMM1 零重叠]]),所有 CTA 本来就是在
            # EOS 门口一起放行的,所以让它们全部参与不改变任何顺序。
            # 记账((consumer_index, consumer_count) 闭式)本来就是通用的。
            is_compute = (
                fx.Int32(1) == fx.Int32(1)
                if const_expr(diagnostic_split_fanout or COMPUTE_FIRST == 0)
                else ticket >= fx.Int32(COMPUTE_FIRST)
            )
            if not _do_compute:
                is_compute = _never
            if is_compute:
                if const_expr(tile_pipeline and cco_geometry == "sparse_wqe"):
                    if is_inter_fanout:
                        # Per-QP receive lets this CTA place remote routes
                        # early.  Keep it out of the GMM queue until all four
                        # QPs have arrived, leaving the already-finished intra
                        # CTAs to consume early tiles without an inter-CTA
                        # queue-head/polling burst competing with transport.
                        if tx == fx.Int32(0):
                            _spin_dbg_38(
                                local_addr("sparse_remote_batch_ready"),
                                generation,
                            )
                        gpu.barrier()
                        comm_ops.fence_system_acquire()
                if const_expr(diagnostic_split_fanout):
                    # Each role drains its own communication stores before it
                    # joins either the streaming or post-EOS GMM scheduler.
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                if const_expr(tile_pipeline):
                    work_scratch = fx.recast_iter(fx.Int32, lds_raw)
                    work_view = fx.make_view(
                        work_scratch, fx.make_layout(1, 1)
                    )
                    queue = buffer_ops.create_buffer_resource_from_addr(
                        local_addr("h1_ready_queue")
                    )
                    work_shard = ticket & fx.Int32(work_shards - 1)
                    consumer_active = fx.Int32(1) == fx.Int32(1)
                    _wdl42 = fx.Int64(comm_ops.read_wall_clock())
                    while consumer_active:
                        if const_expr(_SPIN_DEADLINE > 0):
                            _wdl42 = fx.Int64(comm_ops.read_wall_clock())
                        gpu.barrier()
                        if tx == fx.Int32(0):
                            sequence = fx.Int32(
                                comm_ops.atomic_add_agent(
                                    local_addr("h1_queue_head")
                                    + fx.Int64(work_shard * fx.Int32(16 * 4)),
                                    fx.Int32(1),
                                )
                            )
                            qidx = (
                                work_shard
                                + sequence * fx.Int32(work_shards)
                            )
                            job = fx.Int32(-1)
                            if qidx < fx.Int32(max_jobs):
                                batch = (
                                    qidx // fx.Int32(h1_n_blocks)
                                ) * fx.Int32(h1_n_blocks)
                                ready = fx.Int64(
                                    comm_ops.load_i64_global_system(
                                        local_addr(
                                            "h1_ready_queue_generation"
                                        )
                                        + fx.Int64(batch) * fx.Int64(8)
                                    )
                                )
                                eos = fx.Int64(
                                    comm_ops.load_i64_global_system(
                                        local_addr("h1_queue_eos")
                                    )
                                )
                                while (ready < generation) & (
                                    eos < generation
                                ):
                                    ready = fx.Int64(
                                        comm_ops.load_i64_global_system(
                                            local_addr(
                                                "h1_ready_queue_generation"
                                            )
                                            + fx.Int64(batch) * fx.Int64(8)
                                        )
                                    )
                                    eos = fx.Int64(
                                        comm_ops.load_i64_global_system(
                                            local_addr("h1_queue_eos")
                                        )
                                    )
                                if ready < generation:
                                    # EOS release-orders every final enqueue.
                                    # A claimed slot below final_tail must have
                                    # a matching generation even if this CTA
                                    # observed EOS before reloading the marker.
                                    comm_ops.fence_system_acquire()
                                    final_tail = fx.Int32(
                                        comm_ops.load_i32_global_system(
                                            local_addr("h1_queue_tail")
                                        )
                                    )
                                    if qidx < final_tail:
                                        _spin_dbg_39(
                                            local_addr(
                                                "h1_ready_queue_generation"
                                            )
                                            + fx.Int64(batch) * fx.Int64(8),
                                            generation,
                                        )
                                        ready = generation
                                if ready >= generation:
                                    comm_ops.fence_system_acquire()
                                    job = buffer_ops.buffer_load(
                                        queue,
                                        qidx,
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                    if const_expr(tile_pipeline_instrument):
                                        if _all_comm_eos_seen() == fx.Int32(0):
                                            comm_ops.atomic_add_system_acq_rel(
                                                local_addr(
                                                    "h1_gmm_started_before_all_comm_eos"
                                                ),
                                                fx.Int32(1),
                                            )
                            fx.ptr_store(
                                Vec.from_elements([job], fx.Int32),
                                work_scratch,
                            )
                        gpu.barrier()
                        job = fx.Int32(Vec(work_view.load())[0])
                        has_work = job >= fx.Int32(0)
                        if has_work:
                            _run_gemm1_job(job)
                            rocdl.s_waitcnt(0)
                            gpu.barrier()
                            if tx == fx.Int32(0):
                                if const_expr(tile_pipeline_instrument):
                                    if _all_comm_eos_seen() == fx.Int32(0):
                                        comm_ops.atomic_add_system_acq_rel(
                                            local_addr(
                                                "h1_gmm_completed_before_all_comm_eos"
                                            ),
                                            fx.Int32(1),
                                        )
                                comm_ops.fence_system_release()
                                comm_ops.atomic_add_system_acq_rel(
                                    local_addr("h1_compute_done"), fx.Int32(1)
                                )
                        consumer_active = has_work
                        if const_expr(_SPIN_DEADLINE > 0):
                            _over = (fx.Int64(comm_ops.read_wall_clock()) - _wdl42) // fx.Int64(
                                _SPIN_DEADLINE
                            )
                            consumer_active = consumer_active & (_over == fx.Int64(0))
                    if const_expr(_SPIN_DEADLINE > 0):
                        if (fx.Int64(comm_ops.read_wall_clock()) - _wdl42) >= fx.Int64(
                            _SPIN_DEADLINE
                        ):
                            comm_ops.store_i64_global_system(
                                local_addr("plan_debug") + fx.Int64(42 * 8),
                                generation,
                            )
                else:
                    if const_expr(early_local_gmm):
                        # F5:两段动态领 job。本地段:过 h1_local_eos(段 1 发布,只依赖 node 内 fan1)
                        # 后用本地头领 [0, 本地组数*GNB);远端段:过 h1_queue_eos 后用远端头领
                        # [0, 远端组数*GNB),组号表偏移本地组数。各段越界的那次领取直接丢弃。
                        # 领取都在看到对应门之后,初始化者的清零经 launch_ready→fan1→comm_eos→
                        # 段 1 的链先行。门口一律 s_sleep 退避(F4:无退避的轮询把封尾拖慢 5x)。
                        # 槽:3=过本地门 5=本地段做完 6=过全局门(4 由下面的原收尾写)。
                        el_sl = int(gate_sleep) if gate_sleep else 127
                        el_scr = fx.recast_iter(fx.Int32, lds_raw)
                        el_view = fx.make_view(el_scr, fx.make_layout(1, 1))
                        el_list = buffer_ops.create_buffer_resource_from_addr(
                            local_addr("gmm1_group_list")
                        )
                        if tx == fx.Int32(0):
                            comm_ops.spin_until_ge_i64_sleep(
                                local_addr("h1_local_eos"), generation, el_sl
                            )
                        gpu.barrier()
                        comm_ops.fence_system_acquire()
                        _ts(3)
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
                        _ts(5)
                        if tx == fx.Int32(0):
                            comm_ops.spin_until_ge_i64_sleep(
                                local_addr("h1_queue_eos"), generation, el_sl
                            )
                        gpu.barrier()
                        comm_ops.fence_system_acquire()
                        _ts(6)
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
                        if const_expr(gate_sleep > 0):
                            # 融合版里 248 个 CTA 做完 fanout 就停在这里等全局封尾;
                            # 退避轮询,不和仍在 fanout/发 credit 的 CTA 抢访存。
                            comm_ops.spin_until_ge_i64_sleep(
                                local_addr("h1_queue_eos"), generation, int(gate_sleep)
                            )
                        else:
                            _spin_dbg_40(local_addr("h1_queue_eos"), generation)
                    gpu.barrier()
                    comm_ops.fence_system_acquire()
                    # 融合版:GMM1 消费者(ticket>=COMPUTE_FIRST,从不写 T0 专用的 3..10 槽)
                    # 3 = 过 h1_queue_eos 门,4 = 自己的 job 做完。
                    if const_expr(not early_local_gmm):
                        _ts(3)
                    tiles = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(
                            local_addr("tile_alloc")
                        ),
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    # early_local_gmm:job 已在上面两段领完,这里只剩 sortcopy + 完成计数(每 CTA +1)。
                    total_jobs = (tiles // fx.Int32(GG)) * fx.Int32(
                        0 if early_local_gmm else GNB
                    )
                    consumer_index = (
                        ticket
                        if const_expr(diagnostic_split_fanout)
                        else ticket - fx.Int32(COMPUTE_FIRST)
                    )
                    consumer_count = (
                        fx.Int32(256)
                        if const_expr(diagnostic_split_fanout)
                        else fx.Int32(worker_blocks - COMPUTE_FIRST)
                    )
                    # 诊断:消费者在做任何 job 之前要先等 h1_queue_eos(全部 8 个
                    # 通信角色 EOS)。这个戳把 flush_post->done 切成
                    # 「fanout/等 EOS」与「GMM1 jobs」两段。单一写者:consumer 0 的 tx0。
                    if const_expr(timeline_instrument):
                        if (consumer_index == fx.Int32(0)) & (tx == fx.Int32(0)):
                            comm_ops.store_i64_global_relaxed(
                                timeline_addr
                                + fx.Int64(
                                    STAGE2_TIMELINE_INDEX["stage1_gmm_gate_done"] * 8
                                ),
                                fx.Int64(comm_ops.read_wall_clock()),
                            )
                    if const_expr(sortcopy_k2):
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
                            consumer_index, tiles // fx.Int32(GG), consumer_count
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
                    if const_expr(gmm1_batch_completion):
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
                        _ts(4)
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
                    else:
                        for job in range(
                            consumer_index, total_jobs, consumer_count
                        ):
                            _run_gemm1_job(job)
                            rocdl.s_waitcnt(0)
                            gpu.barrier()
                            if tx == fx.Int32(0):
                                comm_ops.fence_system_release()
                                comm_ops.atomic_add_system_acq_rel(
                                    local_addr("h1_compute_done"), fx.Int32(1)
                                )

        if is_publisher:
            if tx == fx.Int32(0):
                if const_expr(not diagnostic_comm_only):
                    tiles = buffer_ops.buffer_load(
                        buffer_ops.create_buffer_resource_from_addr(local_addr("tile_alloc")),
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                    expected_jobs = (
                        tiles // fx.Int32(GG)
                    ) * fx.Int32(h1_n_blocks if tile_pipeline else GNB)
                    if const_expr(early_local_gmm):
                        # 每个 GMM1 消费者 CTA 退出时 +1。
                        expected_jobs = fx.Int32(worker_blocks - COMPUTE_FIRST)
                    completed = fx.Int32(0)
                    _wdl41 = fx.Int64(comm_ops.read_wall_clock())
                    while completed < expected_jobs:
                        completed = fx.Int32(
                            comm_ops.load_i32_global_system(local_addr("h1_compute_done"))
                        )
                        if const_expr(_SPIN_DEADLINE > 0):
                            _over = (fx.Int64(comm_ops.read_wall_clock()) - _wdl41) // fx.Int64(
                                _SPIN_DEADLINE
                            )
                            completed = completed + fx.Int32(_over) * expected_jobs
                    if const_expr(_SPIN_DEADLINE > 0):
                        if (fx.Int64(comm_ops.read_wall_clock()) - _wdl41) >= fx.Int64(
                            _SPIN_DEADLINE
                        ):
                            comm_ops.store_i64_global_system(
                                local_addr("plan_debug") + fx.Int64(41 * 8),
                                generation,
                            )
                comm_ops.fence_system_release()
                comm_ops.store_i64_global_system(
                    stage2_addr("stage1_done"), generation
                )
                if const_expr(timeline_instrument):
                    comm_ops.store_i64_global_relaxed(
                        timeline_addr
                        + fx.Int64(
                            STAGE2_TIMELINE_INDEX["stage1_done_publish"] * 8
                        ),
                        fx.Int64(comm_ops.read_wall_clock()),
                    )
        _ts(12)

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
        ts_out: fx.Int64,
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
            ts_out,
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
    launch_megamoe_tile_ep16_stage1.essential_ctas = (
        256 if diagnostic_split_fanout else ESSENTIAL_CTAS
    )
    launch_megamoe_tile_ep16_stage1.enable_cco = bool(enable_cco)
    launch_megamoe_tile_ep16_stage1.diagnostic_comm_only = bool(
        diagnostic_comm_only
    )
    launch_megamoe_tile_ep16_stage1.diagnostic_split_fanout = bool(
        diagnostic_split_fanout
    )
    launch_megamoe_tile_ep16_stage1.diagnostic_wave_fanout = bool(
        diagnostic_wave_fanout
    )
    launch_megamoe_tile_ep16_stage1.diagnostic_no_arrival_rmw = bool(
        diagnostic_no_arrival_rmw
    )
    launch_megamoe_tile_ep16_stage1.cco_chunks_per_flush = int(
        cco_chunks_per_flush
    )
    launch_megamoe_tile_ep16_stage1.cco_geometry = cco_geometry
    launch_megamoe_tile_ep16_stage1.gemm1_contraction = bool(
        not diagnostic_comm_only
    )
    launch_megamoe_tile_ep16_stage1.full_stage1_fusion = bool(
        not diagnostic_comm_only
    )
    launch_megamoe_tile_ep16_stage1.tile_pipeline = bool(tile_pipeline)
    launch_megamoe_tile_ep16_stage1.tile_pipeline_instrument = bool(
        tile_pipeline_instrument
    )
    launch_megamoe_tile_ep16_stage1.timeline_instrument = bool(
        timeline_instrument
    )
    launch_megamoe_tile_ep16_stage1.expert_major_output = bool(
        expert_major_output
    )
    launch_megamoe_tile_ep16_stage1.tile_pipeline_fanout_shards = int(
        split_fanout_shards
    )
    launch_megamoe_tile_ep16_stage1.cco_logical_doorbells = (
        5
        if cco_geometry == "sparse_wqe"
        else (
            4
            if cco_geometry == "mori64x2"
            else 2
            * layout.num_qp
            * (dispatch_chunks // cco_chunks_per_flush)
        )
    )
    launch_megamoe_tile_ep16_stage1.split_flag_bytes = (
        GPUS_PER_NODE * 32 * 8 if diagnostic_split_fanout else 0
    )
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
        "cross_node_dedup": (
            "skip_token_without_remote_route_one_data_wqe_per_remote_token"
            if cco_geometry == "sparse_wqe"
            else "one_record_per_token_per_node"
        ),
        "destination_rank_payload": "one_source_indexed_activation_row_per_rank",
        "rank_route_encoding": "u16_topk_slot_mask_per_global_rank",
        "sparse_token_readiness": (
            "four_streamed_qp_terminal_words_with_inter_compute_batch_gate"
            if cco_geometry == "sparse_wqe"
            else "not_applicable"
        ),
        "early_full_tile_enqueue": bool(tile_pipeline),
        "queue_publication": (
            "full_tile_last_arrival_plus_partial_tile_post_8_role_eos"
            if tile_pipeline
            else "post_8_role_eos_physical_major"
        ),
        "gmm_scheduler": (
            "bypassed_after_queue_publish"
            if diagnostic_comm_only
            else (
                "concurrent_ready_queue_8_shards_256_all_roles_rejoin"
                if tile_pipeline
                else (
                    "post_eos_static_strided_256_all_roles_rejoin"
                    if diagnostic_split_fanout
                    else f"post_eos_static_strided_{worker_blocks - ESSENTIAL_CTAS}_pure_compute_consumers"
                )
            )
        ),
        "input_scale_layout": "bm32_ku_ikxdl_klane_nlane_ima",
        "diagnostic_comm_only": bool(diagnostic_comm_only),
        "diagnostic_split_fanout": bool(diagnostic_split_fanout),
        "diagnostic_wave_fanout": bool(diagnostic_wave_fanout),
        "diagnostic_no_arrival_rmw": bool(diagnostic_no_arrival_rmw),
        "cco_chunks_per_flush": int(cco_chunks_per_flush),
        "wave_fanout": bool(wave_fanout),
        "producer_ctas": int(PRODUCER_CTAS),
        "wide_staging_wait": bool(wide_staging_wait),
        "wide_fanout_wait": bool(wide_fanout_wait),
        "compute_first_ticket": int(COMPUTE_FIRST),
        "lean_waitcnt": bool(lean_waitcnt),
        "route_batch": bool(route_batch),
        "fan1_direct": bool(fan1_direct),
        "t0_no_fanout": bool(t0_no_fanout),
        "credit_async": bool(credit_async),
        "rail_soa": bool(rail_soa),
        "rail_qps": int(rail_qps),
        "rail_post_ctas": bool(rail_post_ctas),
        "gmm1_use_nt": bool(gmm1_use_nt),
        "kernel_role": str(kernel_role),
        "kernel_split_at": str(kernel_split_at),
        "launches_per_forward": int(launches_per_forward),
        "cco_geometry": cco_geometry,
        "input_format": "mxfp4_e8m0_1x32",
        "tile_pipeline": bool(tile_pipeline),
        "tile_pipeline_instrument": bool(tile_pipeline_instrument),
        "tile_pipeline_fanout_shards": int(split_fanout_shards),
        "cco_logical_doorbells": (
            5
            if cco_geometry == "sparse_wqe"
            else (
                4
                if cco_geometry == "mori64x2"
                else 2
                * layout.num_qp
                * (dispatch_chunks // cco_chunks_per_flush)
            )
        ),
        "fanout_mapping": (
            "128_inter_plus_128_intra_ctas_one_active_wave_eight_records"
            if diagnostic_wave_fanout
            else (
                (
                    f"{split_fanout_ctas}_inter_plus_{split_fanout_ctas}_intra_"
                    f"{dedicated_compute_ctas}_dedicated_compute_ctas_"
                    f"dest_mod8_shard_div{split_fanout_shards}"
                    if tile_pipeline
                    else "128_inter_plus_128_intra_dest_mod8_shard_div8"
                )
                if diagnostic_split_fanout
                else "legacy_8_destination_ctas"
            )
        ),
        "fanout_completion": (
            (
                f"8x{2 * split_fanout_shards}_unique_generation_flags_"
                "then_one_eos_and_consumed_per_dest"
                if tile_pipeline
                else "8x32_unique_generation_flags_then_one_eos_and_consumed_per_dest"
            )
            if diagnostic_split_fanout
            else "one_role_one_eos_and_consumed_per_dest"
        ),
        "fanout_flag_storage": (
            "arena_parity_8x32_i64"
            if diagnostic_split_fanout
            else "none"
        ),
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
