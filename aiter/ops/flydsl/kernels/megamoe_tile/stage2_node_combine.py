# SPDX-License-Identifier: MIT
"""kernel2: node 内归约 -> RAIL 发送 -> node 间归约 + unpermute 输出。

和 kernel1(stage2_gemm_push.py)一样是独立文件、独立 ABI,共用同一份 CCO
初始化。megamoe_tile/stage2.py 只作参考,不改动。

三个阶段,所有 CTA 依次经过,没有固定角色分区:

  阶段 1  每个 CTA 静态认领若干 chunk。读 plane_slot_inbox,按 slot mask
          归约成每 token 的 node partial 写 node_accumulator;属于
          remote_plane 的 chunk 归约完后,**由同一个 CTA** 直接
          put(aggregate=True) 写 WQE,不敲门铃,然后给 wqe_posted[qp] 加一。
          同一个 CTA 拥有整个 chunk,所以 put 的源地址天然连续。

  阶段 2  CTA 0 的 wave0 串行敲门铃。这是硬件约束,不是选择:
          stage1.py 原注释 "Ionic QPs share a doorbell mapping and must not
          be flushed concurrently" —— WQE 可以多 CTA 并发写,doorbell 不能
          并发敲。flush_async 返回的 request 存进 scratch,不在发送路径上等。

  阶段 3  所有 CTA 降级为 node 间归约。节点内累加阶段 1 已完成,
          remote_partial_rx 里是对端已累加好的 partial,所以这里只做
          相加 + unpermute 写出。按 chunk 到达驱动,不做全局 barrier。

shape 一律参数化,不写死。tokens 不被 num_qp*chunk 整除时走尾块路径 ——
参考实现是靠整除约束回避尾块的("no tail transport here"),这里不保留那条。
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Float32, T

from aiter.ops.flydsl.kernels import buffer_ops
from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from . import comm_ops
# e8m0 的取整规则必须和 push 端**同一份代码**:两端各写一遍迟早会飘,
# 而飘掉的表现是数值慢慢不对,不是崩 —— 最难查的那一类。
from .stage2_gemm_push import _fp8_scale_for_leader
# 跨节点 rail 段用的三个 GDA 原语。MORI 公开绑定给不了(team 写死 WORLD、
# GDA 不透传 optFlags),见 gda_rail.py 顶部;其余 CCO 用法都是直连公开 API。
from .gda_rail import release as _rail

TEAM_RAIL = "rail"


# scratch 布局(i32 槽位),避免第一版就改 ABI。
SCRATCH_WQE_POSTED = 0          # [num_qp]   已写入 WQE 的 chunk 数
SCRATCH_FINAL_HEAD = 64         # 阶段 3 的取活游标
SCRATCH_FINAL_DONE = 65         # 阶段 3 的完成计数
SCRATCH_REQUEST = 128           # [num_qp] i64,flush_async 的 request(按 8 字节寻址)
SCRATCH_ARRIVAL_SEEN = 66       # arrival_probe: 检查过的标志字总数
SCRATCH_ARRIVAL_LATE = 67       # arrival_probe: 第一眼还没就绪的个数
SCRATCH_CHUNK_DONE = 256        # [2][max_chunks] 每个 (plane,chunk) 的完成计数
SCRATCH_P3_DONE = 512           # [max_chunks] 阶段3 每个 chunk 的 CTA 完成计数
SCRATCH_WQE_CHUNK = 144         # [max_chunks] 每个 chunk 的载荷 WQE 是否已入队
SCRATCH_STAMP = 768             # [4] i64 墙钟戳:入口 / 阶段1 后 / 阶段2 后 / 结束
SCRATCH_WORDS = 1024            # 宿主侧 _k2_scratch 的长度,编译期据此校验越界

# cta_stamp 的时间线:**单独一块 GM**(arg_timeline),不占 scratch。
# s_memrealtime 是全局共享的恒频计数器(不是每 CU 的 shader clock),所以不同
# CU 上的戳可以直接相减,不需要标定。
CTA_STAMP_ENTRY = 0             # 入口
CTA_STAMP_P1_END = 1            # 本 CTA 的阶段1 做完
CTA_STAMP_WQE = 2               # 本 CTA 最后一次 post 数据 WQE(aggregate,不敲门铃)
CTA_STAMP_DB_WAIT = 3           # 阶段2:开始自旋等本 QP 的 WQE_POSTED
CTA_STAMP_DB_DONE = 4           # 阶段2:flush_async + wait_request 返回(门铃已确认)
CTA_STAMP_P3_RDY = 5            # 阶段3:第一次等待被满足(远端数据到齐)
CTA_STAMP_P3_END = 6            # 阶段3 做完
CTA_STAMP_HWID = 7              # HW_ID(CU/SH/SE 等),不含 XCC_ID
CTA_STAMP_SPUN = 8              # 阶段2:WQE_POSTED 自旋满足,put_value 之前
CTA_STAMP_SLOTS = 9
CTA_STAMP_MAX_CTAS = 256        # grid 上限;超过的 CTA 不记(宿主侧 _two_kernel_prepare 里校验)
TIMELINE_WORDS = CTA_STAMP_MAX_CTAS * CTA_STAMP_SLOTS   # i64,宿主侧 _k2_timeline 的长度


def _ceil_div(a, b):
    return (a + b - 1) // b


def _ceil_div_rt(a, b):
    """运行时向上取整。尾块就是从这里来的,不再要求整除。"""
    return (a + b - fx.Int32(1)) // b


def _min_rt(a, b):
    return (a < b).select(a, b)


def _max_rt(a, b):
    return (a > b).select(a, b)


def _chunks_for_qp(total_chunks, qp, num_qp):
    """qp 认领的是 chunk == qp (mod num_qp),所以它的 chunk 数要按余数算。"""
    base = total_chunks // fx.Int32(num_qp)
    extra = (qp < (total_chunks % fx.Int32(num_qp))).select(
        fx.Int32(1), fx.Int32(0))
    return base + extra


def _fp8_encode8(vals, lane, *, VEC):
    """VEC 个 f32 -> (打包好的 i32 列表, 本块 e8m0, 本 lane 是否块 leader)。

    逐字照搬 kernel1 的 push 端(stage2_gemm_push.py:205-260)。VEC=8 时一个
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


@flyc.jit
def _reduce_one_token(
    plane, token, lane, wave, inbox_rsrc, accum_rsrc, mask_ptr,
    *, HIDDEN, TOPK, MAX_TOKENS, WAVES, VEC, CHUNK_ITERS,
):
    """把一个 token 的 topk 贡献按 slot mask 归约进 node_accumulator。

    关键:slot 循环用**运行时 range + 真 if**,不是 range_constexpr 全展开。
    展开版实测无效 —— 开不开 skip-absent、eplb 还是 mixed,四种组合时间全在
    139 µs,说明 `if peer_active` 被降成了谓词,访存一次没省。这里赌运行时
    循环能生成真分支;若实测仍不省(用 topk 扫描判定:时间是否正比于实际
    fan-in 而非 TOPK),就退到"预先压实贡献者列表 + 按 count 循环"。
    """
    token_index = fx.Int32(plane * MAX_TOKENS) + token
    slot_mask = fx.Int32(comm_ops.load_i32_global_system_relaxed(
        mask_ptr + fx.Int64(token_index) * fx.Int64(4)))

    CHUNKS_PER_WAVE = HIDDEN // (64 * VEC)
    for chunk_iter in range_constexpr(CHUNK_ITERS):
        col_chunk = wave + fx.Int32(chunk_iter * WAVES)
        active = col_chunk < fx.Int32(CHUNKS_PER_WAVE)
        col = (col_chunk * fx.Int32(64) + lane) * fx.Int32(VEC)
        totals = fx.Vector.filled(VEC, 0.0, Float32)
        for slot in range(fx.Int32(0), fx.Int32(TOPK), fx.Int32(1)):
            if ((slot_mask >> slot) & fx.Int32(1)) != fx.Int32(0):
                # arena 是 [plane][bsid][topk_slot][hidden],hidden 连续。
                src = ((token_index * fx.Int32(TOPK) + slot)
                       * fx.Int32(HIDDEN) + col)
                packed = buffer_ops.buffer_load(
                    inbox_rsrc, src // fx.Int32(2),
                    vec_width=4, dtype=T.i32, mask=active)
                totals = totals + fx.Vector(packed).bitcast(BFloat16).to(Float32)
        out = fx.Vector.from_elements(
            [fx.Float32(totals[i]) for i in range_constexpr(VEC)],
            Float32,
        ).to(BFloat16).bitcast(fx.Int32)
        dst = token_index * fx.Int32(HIDDEN) + col
        buffer_ops.buffer_store(
            out, accum_rsrc, dst // fx.Int32(2), mask=active)


def compile_stage2_node_combine(
    *,
    hidden: int,
    max_tokens: int,
    topk: int,
    rank: int,
    gpus_per_node: int = 8,
    num_qp: int = 8,
    return_chunk_tokens: int = 16,
    threads: int = 256,
    enable_rail: bool = True,
    stamp: bool = False,
    cta_stamp: bool = False,
    wait_remote: bool = True,
    inbox_quant: str = "none",
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
    consumed_off: int = 0,
    arrival_off: int = 0,
    arrival_wait: bool = False,
    arrival_probe: bool = False,
    s2_window_off: int,
    team: str = TEAM_RAIL,
):
    """构建 kernel2。所有 *_off 是相对 window local_ptr 的字节偏移。"""

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
    if enable_rail:
        # 参考实现是 qp = wave。wave 数多于 QP 时有 wave 没活干,少于 QP 时
        # 一个 wave 顺序带 num_qp//waves 个 QP —— 两者都要求整除。
        _waves = int(threads) // 64
        if int(num_qp) < _waves or int(num_qp) % _waves:
            raise ValueError(
                f"enable_rail 时 num_qp({num_qp}) 必须是 wave 数({_waves}) 的整数倍")
        if int(consumed_off) < int(s2_window_off):
            raise ValueError(
                "consumed_off 小于 s2_window_off,看起来是 region 相对偏移")
    if arrival_probe and not arrival_wait:
        raise ValueError("arrival_probe requires arrival_wait")
    if arrival_wait:
        if int(arrival_off) < int(s2_window_off):
            raise ValueError(
                "arrival_off 小于 s2_window_off,看起来是 region 相对偏移")
    if return_chunk_tokens < 4:
        # return_group_ready 只有 max_tokens/4 个槽位,粒度不能比 4 更细。
        raise ValueError("return_chunk_tokens must be >= 4")
    if threads % 64:
        raise ValueError("threads must be a multiple of 64")
    if team != TEAM_RAIL:
        raise ValueError("cross-node transport requires the rail team")

    HIDDEN = int(hidden)
    MAX_TOKENS = int(max_tokens)
    TOPK = int(topk)
    CHUNK = int(return_chunk_tokens)
    THREADS = int(threads)
    # ceil,而不是整除:CHUNK*TOPK 小于 threads 时(比如 CHUNK=8)也要跑一轮,
    # 多出来的线程靠下面的上界判断落空。之前写成整除,CHUNK 一调就编译失败。
    WAIT_SPAN = CHUNK * int(topk)
    WAIT_ITERS = _ceil_div(WAIT_SPAN, int(threads)) if arrival_wait else 0
    NUM_QP = int(num_qp)
    QP_PER_WAVE = max(1, NUM_QP // max(1, int(threads) // 64))
    WAVES = threads // 64
    RECORD_BYTES = HIDDEN * 2               # 一个 token 的 BF16 partial,无 header
    # inbox 的记录宽度和 RECORD_BYTES 是两回事:RECORD_BYTES 只用于 rail 的
    # put/get(每 token 一条 accumulator),inbox 是每 (token, topk slot) 一条,
    # 由 kernel1 的 token_nbytes 定宽。两者可以独立量化。
    if inbox_quant not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(
            f"inbox_quant must be 'none' or 'fp8_blockwise_1x32' "
            f"(got {inbox_quant!r})")
    INBOX_FP8 = inbox_quant == "fp8_blockwise_1x32"
    if INBOX_FP8:
        if HIDDEN % 32:
            raise ValueError("fp8 inbox 要求 hidden 能被 32 整除(e8m0 块大小)")
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
    INBOX_ROW_BYTES = (HIDDEN + HIDDEN // 32) if INBOX_FP8 else HIDDEN * 2
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
            or SCRATCH_P3_DONE + MAX_CHUNKS > SCRATCH_STAMP):
        raise ValueError(
            f"scratch 不够:需要 {SCRATCH_CHUNK_DONE + 2 * MAX_CHUNKS} 个 i32")

    node = int(rank) // int(gpus_per_node)
    remote_node = 1 - node
    local_plane = node
    remote_plane = remote_node

    kernel_name = (
        f"megamoe_k2_h{HIDDEN}_t{MAX_TOKENS}_k{TOPK}"
        f"_qp{NUM_QP}_c{CHUNK}_w{WAVES}"
        + ("" if enable_rail else "_norail")
        + ("" if wait_remote else "_nowait")
        + ("_aw1" if arrival_wait else "")
        + ("_qi8" if INBOX_FP8 else "")
        + ("_qr8" if RAIL_FP8 else "")
        + ("_st" if stamp else "")
        + ("_cs" if cta_stamp else "")
        + ("_ap1" if arrival_probe else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[threads, 1, 1])
    def kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64,
        arg_scratch: fx.Int64,
        generation: fx.Int64,
        local_tokens: fx.Int32,
        arg_timeline: fx.Int64,
    ):
        bx = fx.Int32(gpu.block_id("x"))
        grid = fx.Int32(gpu.grid_dim.x)
        tx = fx.Int32(gpu.thread_id("x"))
        wave = tx // fx.Int32(64)
        lane = tx % fx.Int32(64)

        inbox_ptr = arena_ptr + fx.Int64(inbox_off)
        accum_ptr = arena_ptr + fx.Int64(accumulator_off)
        rx_ptr = arena_ptr + fx.Int64(rx_off)
        mask_ptr = arena_ptr + fx.Int64(slot_mask_off)
        ready_ptr = arena_ptr + fx.Int64(partial_ready_off)
        arrival_ptr = arena_ptr + fx.Int64(arrival_off)
        group_ready_ptr = arena_ptr + fx.Int64(group_ready_off)

        # node_dest_slot_mask 是**本 rank 的 stage1** 写的(stage1.py:891/:1487
        # 都是本地 buffer_store),和 kernel2 同 rank 同 stream 串行 —— 没有任何
        # 跨 GPU 时序,所以不需要 system scope 的绕缓存读。原来两处都用
        # load_i32_global_system_relaxed:等待循环里每个 (token,slot) 读一次
        # (同一个 token 重复 16 次),归约循环里每个工作单元读一次,加起来近
        # 10 万次不可缓存的读。
        mask_rsrc = buffer_ops.create_buffer_resource_from_addr(
            mask_ptr, num_records_bytes=fx.Int32(2 * MAX_TOKENS * 4))
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
            word = fx.Int64(comm_ops.load_i64_global_system_relaxed(addr))
            while word < expected:
                word = fx.Int64(comm_ops.load_i64_global_system_relaxed(addr))

        @flyc.jit
        def _stamp(which):
            # 纯诊断:CTA0 的 thread0 记四个墙钟点。s_memrealtime 是恒频的,
            # 但频率不需要知道 —— 宿主侧按 (t3-t0) 归一化成占比,再乘上
            # 已经量到的 kernel 时间。store 走 relaxed,不引入任何 fence。
            if (bx == fx.Int32(0)) & (tx == fx.Int32(0)):
                comm_ops.store_i64_global_relaxed(
                    arg_scratch + fx.Int64((SCRATCH_STAMP + 2 * which) * 4),
                    fx.Int64(comm_ops.read_wall_clock()))

        @flyc.jit
        def _cta_stamp(which):
            # 每 CTA 的 thread0 记一个点,写进独立的 timeline GM。relaxed store,
            # 不引入任何 fence;整条时间线 8 个 store,对 ~195us 的 kernel 可忽略。
            if const_expr(cta_stamp):
                if tx == fx.Int32(0):
                    if bx < fx.Int32(CTA_STAMP_MAX_CTAS):
                        comm_ops.store_i64_global_relaxed(
                            arg_timeline
                            + (fx.Int64(bx) * fx.Int64(CTA_STAMP_SLOTS)
                               + fx.Int64(which)) * fx.Int64(8),
                            fx.Int64(comm_ops.read_wall_clock()))

        @flyc.jit
        def _cta_hwid():
            if const_expr(cta_stamp):
                if tx == fx.Int32(0):
                    if bx < fx.Int32(CTA_STAMP_MAX_CTAS):
                        comm_ops.store_i64_global_relaxed(
                            arg_timeline
                            + (fx.Int64(bx) * fx.Int64(CTA_STAMP_SLOTS)
                               + fx.Int64(CTA_STAMP_HWID)) * fx.Int64(8),
                            fx.Int64(comm_ops.read_hw_id()))

        _cta_hwid()
        _cta_stamp(CTA_STAMP_ENTRY)

        if const_expr(stamp):
            _stamp(0)

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
          # 运行期 range:ck_local, +total_chunks, ...
          # 用它而不是 `for _st in constexpr: if ck < total_chunks:` —— 动态
          # range 落在 scf.if 生成的分支函数里不会被 AST 改写(和
          # p2p_scatter_epilog 同一个坑),编译期就炸。
          # 和分级前逐字等价。
          for ck in range(ck_local, total_chunks, total_chunks):
              ck_first = ck * fx.Int32(CHUNK)
              ck_tok = _min_rt(fx.Int32(CHUNK), local_tokens - ck_first)
              # 认领同一个 chunk 的 CTA 数(各 chunk 最多差 1)。
              # 本 CTA 在这个 chunk 里负责的连续 token 段。tile 交织时一个
              # CTA 会碰到 chunk 里几乎所有 token,于是 10 个 CTA 各把这个
              # chunk 的 512 个标志重等一遍(10x 冗余),而且它写的字节不连续、
              # 没法自己发 put。连续段同时解决这两件事。
              t_lo = ck_tok * sub // n_cta
              t_hi = ck_tok * (sub + fx.Int32(1)) // n_cta
              my_tok = t_hi - t_lo
              if const_expr(arrival_wait):
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
                  comm_ops.fence_system_acquire()
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
                  scale_raw = []
                  for slot in range_constexpr(TOPK):
                      ok = ((slot_mask >> fx.Int32(slot))
                            & fx.Int32(1)) != fx.Int32(0)
                      rec = (token_index * fx.Int32(TOPK)
                             + fx.Int32(slot))
                      if const_expr(INBOX_FP8):
                          # 值区 1 字节/元素,尺度区紧跟其后。
                          vbyte = rec * fx.Int32(INBOX_ROW_BYTES) + col
                          vals.append(buffer_ops.buffer_load(
                              inbox_rsrc, vbyte // fx.Int32(4),
                              vec_width=2, dtype=T.i32, mask=ok))
                          # 掩码也必须盖住尺度:e8m0=0xFF 解出来是 +Inf,
                          # 缺席 slot 的值虽然是 0,0*Inf 会变成 NaN。
                          sbyte = (rec * fx.Int32(INBOX_ROW_BYTES)
                                   + fx.Int32(HIDDEN)
                                   + col // fx.Int32(32))
                          scale_raw.append(buffer_ops.buffer_load(
                              inbox_rsrc, sbyte,
                              vec_width=1, dtype=T.i8, mask=ok))
                      else:
                          src = rec * fx.Int32(HIDDEN) + col
                          vals.append(buffer_ops.buffer_load(
                              inbox_rsrc, src // fx.Int32(2),
                              vec_width=4, dtype=T.i32, mask=ok))
                  for slot in range_constexpr(TOPK):
                      if const_expr(INBOX_FP8):
                          e8m0 = (fx.Uint8(scale_raw[slot])
                                  .to(fx.Uint32).bitcast(fx.Int32))
                          bscale = (e8m0 << fx.Int32(23)).bitcast(Float32)
                          totals = totals + _fp8_decode8(vals[slot], bscale)
                      else:
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
                      prev = fx.Int32(comm_ops.atomic_add_agent_acq_rel(
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
                      comm_ops.fence_system_release()
                      if lane == fx.Int32(0):
                          comm_ops.store_i32_system(
                              arg_scratch, cnt_idx, fx.Int32(0))
                          for _t in range_constexpr(CHUNK):
                              tk = ck_first + fx.Int32(_t)
                              if tk < local_tokens:
                                  # relaxed:上面那次 fence 已经把 writeback
                                  # 做了,release store 会再来一遍 buffer_wbl2。
                                  comm_ops.store_i64_global_system_relaxed(
                                      ready_ptr
                                      + fx.Int64(fx.Int32(plane * MAX_TOKENS) + tk)
                                      * fx.Int64(8), generation)
                      if const_expr(enable_rail
                                    and plane == remote_plane):
                          _qp = ck % fx.Int32(NUM_QP)
                          _rail.put(dev_comm, _qp, fx.Int32(remote_node),
                              arena_win,
                              fx.Int64(rx_off)
                              + fx.Int64(ck_first) * fx.Int64(RECORD_BYTES),
                              arena_win,
                              fx.Int64(accumulator_off)
                              + fx.Int64(remote_plane * MAX_TOKENS)
                              * fx.Int64(RECORD_BYTES)
                              + fx.Int64(ck_first) * fx.Int64(RECORD_BYTES),
                              fx.Int64(ck_tok) * fx.Int64(RECORD_BYTES),
                              aggregate=True)
                          rocdl.s_waitcnt(0)
                          if lane == fx.Int32(0):
                              comm_ops.atomic_add_agent(
                                  arg_scratch
                                  + fx.Int64(SCRATCH_WQE_POSTED) * fx.Int64(4)
                                  + fx.Int64(_qp) * fx.Int64(4),
                                  fx.Int32(1))
        _cta_stamp(CTA_STAMP_P1_END)

        if const_expr(stamp):
            gpu.barrier()
            _stamp(1)

        # ------------------------------------------------------------------
        # 阶段 2:rail 返回 —— 形状照抄 megamoe_tile stage2.py:1026-1120
        # ------------------------------------------------------------------
        # 整段只在一个 CTA 里跑,qp = wave。这样 gpu.barrier() 是合法的(CTA 的
        # 每个 wave 都会走到),"本批 WQE 已全部写入"就是一次 barrier,不需要
        # 任何跨 CTA 计数器。NUM_QP 多于 wave 数时,一个 wave 顺序带几个 QP。
        if const_expr(enable_rail):
          # 一个 CTA 带一个 QP。当初整段收在 CTA0 里是因为"本批 WQE 已全部
          # 写入"想用一次 gpu.barrier() 表达;但阶段1 各 CTA 写的
          # SCRATCH_WQE_POSTED[qp] 已经是跨 CTA 计数器,而 QP 之间没有顺序
          # 要求 —— 那个理由不成立了。这里不用任何 barrier。
          if bx < fx.Int32(NUM_QP):
            if wave == fx.Int32(0):
             _cta_stamp(CTA_STAMP_DB_WAIT)
             _want = _max_rt(
                 (total_chunks - bx + fx.Int32(NUM_QP - 1))
                 // fx.Int32(NUM_QP), fx.Int32(0))
             _posted = fx.Int32(comm_ops.load_i32_global_system_relaxed(
                 arg_scratch + fx.Int64(SCRATCH_WQE_POSTED) * fx.Int64(4)
                 + fx.Int64(bx) * fx.Int64(4)))
             while _posted < _want:
                 _posted = fx.Int32(comm_ops.load_i32_global_system_relaxed(
                     arg_scratch + fx.Int64(SCRATCH_WQE_POSTED) * fx.Int64(4)
                     + fx.Int64(bx) * fx.Int64(4)))
             _cta_stamp(CTA_STAMP_SPUN)
             comm_ops.fence_system_acquire()
             for chunk in range(bx, total_chunks, fx.Int32(NUM_QP)):
                 _rail.put_value(
                     dev_comm, bx, fx.Int32(remote_node), arena_win,
                     fx.Int64(group_ready_off) + fx.Int64(chunk) * fx.Int64(8),
                     generation, aggregate=True)
             # 原来是 flush_async 拿 request、下一行立刻 wait_request —— 中间
             # 什么都没做,拆成两步没有收益。同步 flush 就是它。
             _rail.flush_peer(dev_comm, bx, fx.Int32(remote_node))
             _cta_stamp(CTA_STAMP_DB_DONE)
             if lane == fx.Int32(0):
                 comm_ops.store_i32_system(
                     arg_scratch, fx.Int32(SCRATCH_WQE_POSTED) + bx,
                     fx.Int32(0))
        if const_expr(stamp):
            _stamp(2)

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
              if const_expr(enable_rail and wait_remote):
                  _wait_ready_relaxed(
                      group_ready_ptr + fx.Int64(ck) * fx.Int64(8), generation)
              _cta_stamp(CTA_STAMP_P3_RDY)
          gpu.barrier()
          comm_ops.fence_system_acquire()
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
              prev = fx.Int32(comm_ops.atomic_add_agent_acq_rel(
                  arg_scratch + fx.Int64(SCRATCH_P3_DONE + ck) * fx.Int64(4),
                  fx.Int32(1)))
              if (prev + fx.Int32(1)) == n_cta:
                  comm_ops.store_i32_system(
                      arg_scratch, fx.Int32(SCRATCH_P3_DONE) + ck, fx.Int32(0))
                  comm_ops.atomic_add_system_acq_rel(
                      arg_scratch + fx.Int64(SCRATCH_FINAL_DONE) * fx.Int64(4),
                      ck_tok)
        rocdl.s_waitcnt(0)
        gpu.barrier()
        _cta_stamp(CTA_STAMP_P3_END)

        if const_expr(stamp):
            _stamp(3)

        # ------------------------------------------------------------------
        # 收尾:return_consumed 信用交换(照 stage2.py:1294-1324)
        # ------------------------------------------------------------------
        # remote_partial_rx 只有 2 个 parity 槽,第 N 代和第 N+2 代共用同一块。
        # 没有这个握手,两端代数可以自由漂移;而 wait_ready 是 >= 语义,对端跑快
        # 两代时它写的新 ready 会同时满足我们对旧代的等待 —— 不挂死,直接读到
        # 写了一半的 buffer。所以这一步不是可选的优化。
        if const_expr(enable_rail and wait_remote):
          if bx == fx.Int32(0):
            done_ptr = arg_scratch + fx.Int64(SCRATCH_FINAL_DONE) * fx.Int64(4)
            if tx == fx.Int32(0):
                done = fx.Int32(comm_ops.load_i32_global_system(done_ptr))
                while done < local_tokens:
                    done = fx.Int32(comm_ops.load_i32_global_system(done_ptr))
                # 本代计数已满,减回去,下一代重新从 0 开始。
                comm_ops.atomic_add_system_acq_rel(
                    done_ptr, fx.Int32(0) - local_tokens)
            gpu.barrier()
            if wave == fx.Int32(0):
                # "你发给我的那块我读完了" -> 对端可以复用它的发送源。
                _rail.put_value(dev_comm, fx.Int32(0), fx.Int32(remote_node),
                                arena_win, fx.Int64(consumed_off), generation,
                                aggregate=True)
                _rail.flush_peer(dev_comm, fx.Int32(0), fx.Int32(remote_node))
                if lane == fx.Int32(0):
                    # 反过来等对端的信用:确认我们写进它 rx 的那块已被读完,
                    # 下一代才可以再往同一个 parity 槽写。system-acquire 是
                    # 必须的:这个字是对端 NIC 写的。
                    _credit_ptr = arena_ptr + fx.Int64(consumed_off)
                    _credit = fx.Int64(
                        comm_ops.load_i64_global_system(_credit_ptr))
                    while _credit < generation:
                        _credit = fx.Int64(
                            comm_ops.load_i64_global_system(_credit_ptr))

    @flyc.jit
    def launch(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        arg_output_bf16: fx.Int64,
        arg_scratch: fx.Int64,
        generation: fx.Int64,
        local_tokens: fx.Int32,
        arg_timeline: fx.Int64,
        blocks: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
            generation, local_tokens, arg_timeline,
            value_attrs={"rocdl.flat_work_group_size": f"{threads},{threads}"},
        ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream)

    launch.kernel = kernel          # ISA dump / 资源统计用
    launch.kernel_name = kernel_name
    launch.num_qp = NUM_QP
    launch.return_chunk_tokens = CHUNK
    launch.record_bytes = RECORD_BYTES
    return launch


_CACHE = {}


def get_stage2_node_combine(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        _CACHE[key] = compile_stage2_node_combine(**kw)
    return _CACHE[key]


def run_stage2_node_combine(
    dev_comm, arena_win, arena_ptr, arg_output_bf16, arg_scratch,
    generation, local_tokens, blocks, stream, arg_timeline=0, **compile_kw,
):
    launch = get_stage2_node_combine(**compile_kw)
    launch(
        fx.Int64(int(dev_comm)), fx.Int64(int(arena_win)),
        fx.Int64(int(arena_ptr)), fx.Int64(int(arg_output_bf16)),
        fx.Int64(int(arg_scratch)), fx.Int64(int(generation)),
        fx.Int32(int(local_tokens)), fx.Int64(int(arg_timeline)),
        fx.Int32(int(blocks)), stream,
    )
    return launch
