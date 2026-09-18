# SPDX-License-Identifier: MIT
"""Public strict-two-kernel EP16 A4W4 MegaMoE operator.

This class intentionally mirrors :class:`MegaMoEV2` at its public boundary,
but specializes the K3 two-node deployment and accepts only ``quant='a4w4'``.
All allocation, CCO setup, window initialization and FlyDSL compilation happen
in the constructor.  A hot ``forward`` performs exactly two launcher calls:

1. fused BF16 quant + InterNodeV1 direct-to-expert-tile dispatch + GMM1 +
   SiLU + A4 requant;
2. fused weighted GMM2 + packed-BF16 direct LSA node-accumulator epilogue +
   InterNodeV1 combine.

There is no fallback to the former record-fanout cascade.  If either strict
kernel backend is unavailable, construction fails rather than silently
running a multi-kernel implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import re
from typing import Any, Callable

import torch
import torch.distributed as dist

from .stage1_abi import (
    MAX_FUSED_TOKENS_PER_RANK,
    Stage1ArenaLayout,
    TwoKernelArenaLayout,
    validate_fused_weight_addressing,
    validate_public_stage1_contract,
)
from .stage2_abi import STAGE2_TIMELINE_FIELDS, Stage2ArenaLayout


_STAGE1_MODULE = "aiter.ops.flydsl.kernels.megamoe_tile.stage1"
_STAGE1_FACTORY = "compile_megamoe_tile_ep16_stage1"
_STAGE2_MODULE = "aiter.ops.flydsl.kernels.megamoe_tile.stage2"
_STAGE2_FACTORY = "compile_megamoe_tile_ep16_stage2_a4w4"



def _bw_probe(nbytes, rank, label_pairs, reduce_shape=None):
    """同一个流式读 kernel 打在两块等大内存上,比达成带宽。

    kernel2 阶段1 只有 295 GB/s,而 CTA 数、cache policy、软流水、掩码 load
    全部改过、全部零反应,时间严格正比于字节数。剩下的解释只有一个:CCO 对称
    窗口这块内存本身就读不快(它要能被 peer 相干写入)。用同一个 kernel 打
    arena 和普通 torch 显存,这个问题就不用再猜。
    """
    import torch as _t
    import flydsl.compiler as _flyc
    import flydsl.expr as _fx
    from flydsl.expr import gpu as _gpu, range_constexpr as _rc
    from flydsl.expr.typing import T as _T
    from aiter.ops.flydsl.kernels import buffer_ops as _bo

    THREADS, VEC, UNROLL, BLOCKS = 256, 4, 8, 512
    ELEMS_PER_PASS = BLOCKS * THREADS * UNROLL * VEC

    @_flyc.kernel(name="megamoe_bwprobe2", known_block_size=[THREADS, 1, 1])
    def _k(base: _fx.Int64, out: _fx.Int64, n_iter: _fx.Int32, span: _fx.Int32):
        bx = _fx.Int32(_gpu.block_id("x"))
        tx = _fx.Int32(_gpu.thread_id("x"))
        rsrc = _bo.create_buffer_resource_from_addr(base, num_records_bytes=span)
        sink = _bo.create_buffer_resource_from_addr(
            out, num_records_bytes=_fx.Int32(65536))
        acc = _fx.Vector.filled(VEC, 0, _fx.Int32)
        for it in range(_fx.Int32(0), n_iter, _fx.Int32(1)):
            b = it * _fx.Int32(ELEMS_PER_PASS) + (bx * _fx.Int32(THREADS) + tx) * _fx.Int32(VEC)
            vs = []
            for u in _rc(UNROLL):
                vs.append(_bo.buffer_load(
                    rsrc, b + _fx.Int32(u * THREADS * VEC),
                    vec_width=VEC, dtype=_T.i32))
            for u in _rc(UNROLL):
                acc = acc + _fx.Vector(vs[u])
        # 依赖全部读到的数据,防止被优化掉;写进 64KB 的 sink,越界也安全
        _bo.buffer_store(acc, sink, (bx * _fx.Int32(THREADS) + tx) % _fx.Int32(4096))

    @_flyc.jit
    def _launch(base: _fx.Int64, out: _fx.Int64, n_iter: _fx.Int32,
                span: _fx.Int32, stream: _fx.Stream):
        _k(base, out, n_iter, span).launch(
            grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    @_flyc.kernel(name="megamoe_bwprobe_reduce", known_block_size=[THREADS, 1, 1])
    def _kr(base: _fx.Int64, out: _fx.Int64, n_tok: _fx.Int32, span: _fx.Int32,
            HIDDEN: _fx.Int32, TOPK: _fx.Int32):
        """逐字复刻阶段1 的访问模式,去掉 rail / arrival / 阶段3 的全部包袱。

        (token, hidden tile) 按 global wave id 分片,一个 wave 读 TOPK 个相隔
        HIDDEN*2 字节的 1 KB 块并累加。和阶段1 唯一的区别是这里不读 mask
        (无条件读满 TOPK)、不写 accumulator、没有完成计数。
        """
        bx = _fx.Int32(_gpu.block_id("x"))
        grid = _fx.Int32(_gpu.grid_dim.x)
        tx = _fx.Int32(_gpu.thread_id("x"))
        wave = tx // _fx.Int32(64)
        lane = tx % _fx.Int32(64)
        rsrc = _bo.create_buffer_resource_from_addr(base, num_records_bytes=span)
        sink = _bo.create_buffer_resource_from_addr(
            out, num_records_bytes=_fx.Int32(65536))
        HPARTS = 7
        gwave = bx * _fx.Int32(4) + wave
        twaves = grid * _fx.Int32(4)
        acc = _fx.Vector.filled(VEC, 0, _fx.Int32)
        for u in range(gwave, n_tok * _fx.Int32(HPARTS), twaves):
            tok = u // _fx.Int32(HPARTS)
            hp = u - tok * _fx.Int32(HPARTS)
            col = (hp * _fx.Int32(64) + lane) * _fx.Int32(8)
            vs = []
            for slot in _rc(16):
                src = (tok * TOPK + _fx.Int32(slot)) * HIDDEN + col
                vs.append(_bo.buffer_load(
                    rsrc, src // _fx.Int32(2), vec_width=VEC, dtype=_T.i32))
            for slot in _rc(16):
                acc = acc + _fx.Vector(vs[slot])
        _bo.buffer_store(acc, sink, (bx * _fx.Int32(THREADS) + tx) % _fx.Int32(4096))

    @_flyc.kernel(name="megamoe_bwprobe_reduce_bk", known_block_size=[THREADS, 1, 1])
    def _kb(base: _fx.Int64, out: _fx.Int64, n_tok: _fx.Int32, span: _fx.Int32,
            HIDDEN: _fx.Int32, TOPK: _fx.Int32):
        """_kr 加上阶段1 的每单元簿记:一次 s_waitcnt(0) + 一次 agent 原子。

        单变量:除了这两条,和 _kr 逐字一致。阶段1 的 chunk 完成计数就长这样,
        而 s_waitcnt(0) 等的是全部未完成访存 —— 下一个单元的 load 一个都发不
        出去。7168 个单元就是 7168 次全流水排空。
        """
        import flydsl.expr as _fxx
        from flydsl.expr import rocdl as _rocdl
        from aiter.ops.flydsl.kernels.megamoe_tile import comm_ops as _co
        bx = _fx.Int32(_gpu.block_id("x"))
        grid = _fx.Int32(_gpu.grid_dim.x)
        tx = _fx.Int32(_gpu.thread_id("x"))
        wave = tx // _fx.Int32(64)
        lane = tx % _fx.Int32(64)
        rsrc = _bo.create_buffer_resource_from_addr(base, num_records_bytes=span)
        sink = _bo.create_buffer_resource_from_addr(
            out, num_records_bytes=_fx.Int32(65536))
        HPARTS = 7
        gwave = bx * _fx.Int32(4) + wave
        twaves = grid * _fx.Int32(4)
        acc = _fx.Vector.filled(VEC, 0, _fx.Int32)
        for u in range(gwave, n_tok * _fx.Int32(HPARTS), twaves):
            tok = u // _fx.Int32(HPARTS)
            hp = u - tok * _fx.Int32(HPARTS)
            col = (hp * _fx.Int32(64) + lane) * _fx.Int32(8)
            vs = []
            for slot in _rc(16):
                src = (tok * TOPK + _fx.Int32(slot)) * HIDDEN + col
                vs.append(_bo.buffer_load(
                    rsrc, src // _fx.Int32(2), vec_width=VEC, dtype=_T.i32))
            for slot in _rc(16):
                acc = acc + _fx.Vector(vs[slot])
            _rocdl.s_waitcnt(0)
            prev = _fx.Int32(0)
            if lane == _fx.Int32(0):
                prev = _fx.Int32(_co.atomic_add_agent_acq_rel(
                    out + _fx.Int64(32768) + _fx.Int64(tok // _fx.Int32(32))
                    * _fx.Int64(4), _fx.Int32(1)))
            prev = _fx.Int32(_rocdl.readfirstlane(_T.i32, prev.ir_value()))
        _bo.buffer_store(acc, sink, (bx * _fx.Int32(THREADS) + tx) % _fx.Int32(4096))

    @_flyc.jit
    def _launch_b(base: _fx.Int64, out: _fx.Int64, n_tok: _fx.Int32,
                  span: _fx.Int32, H: _fx.Int32, K: _fx.Int32, stream: _fx.Stream):
        _kb(base, out, n_tok, span, H, K).launch(
            grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    @_flyc.jit
    def _launch_r(base: _fx.Int64, out: _fx.Int64, n_tok: _fx.Int32,
                  span: _fx.Int32, H: _fx.Int32, K: _fx.Int32, stream: _fx.Stream):
        _kr(base, out, n_tok, span, H, K).launch(
            grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    sink = _t.zeros(16384, dtype=_t.int32, device="cuda")
    stream = _t.cuda.current_stream()
    s_fx = _fx.Stream(stream.cuda_stream)
    n_iter = int(nbytes) // (ELEMS_PER_PASS * 4)
    real = n_iter * ELEMS_PER_PASS * 4
    out = []
    for label, ptr in label_pairs:
        args = (_fx.Int64(int(ptr)), _fx.Int64(int(sink.data_ptr())),
                _fx.Int32(n_iter), _fx.Int32(int(real)), s_fx)
        _launch(*args); _t.cuda.synchronize()
        ev0, ev1 = _t.cuda.Event(True), _t.cuda.Event(True)
        best = 1e9
        for _ in range(5):
            ev0.record(stream); _launch(*args); ev1.record(stream)
            _t.cuda.synchronize()
            best = min(best, ev0.elapsed_time(ev1) * 1e3)
        out.append((label, round(best, 1), round(real / best / 1e3, 1)))
    print("TWO_KERNEL_BW_PROBE " + repr(
        {"rank": rank, "MB": round(real / 1e6, 1), "us_and_GBs": out}), flush=True)

    # 同一块 arena,换成阶段1 的访问模式。字节数刻意算成和阶段1 一致。
    if reduce_shape is not None:
        _base, _ntok, _H, _K = reduce_shape
        rbytes = _ntok * _K * _H * 2
        args = (_fx.Int64(int(_base)), _fx.Int64(int(sink.data_ptr())),
                _fx.Int32(int(_ntok)), _fx.Int32(int(rbytes)),
                _fx.Int32(int(_H)), _fx.Int32(int(_K)), s_fx)
        _launch_r(*args); _t.cuda.synchronize()
        ev0, ev1 = _t.cuda.Event(True), _t.cuda.Event(True)
        best = 1e9
        for _ in range(5):
            ev0.record(stream); _launch_r(*args); ev1.record(stream)
            _t.cuda.synchronize()
            best = min(best, ev0.elapsed_time(ev1) * 1e3)
        print("TWO_KERNEL_BW_REDUCE_SHAPE " + repr({
            "rank": rank, "MB": round(rbytes / 1e6, 1),
            "us": round(best, 1), "GBs": round(rbytes / best / 1e3, 1)}), flush=True)
        # 单变量:同一个 kernel 加上每单元的 s_waitcnt(0) + 完成计数原子
        _launch_b(*args); _t.cuda.synchronize()
        best2 = 1e9
        for _ in range(5):
            ev0.record(stream); _launch_b(*args); ev1.record(stream)
            _t.cuda.synchronize()
            best2 = min(best2, ev0.elapsed_time(ev1) * 1e3)
        print("TWO_KERNEL_BW_REDUCE_BOOKKEEP " + repr({
            "rank": rank, "us": round(best2, 1),
            "GBs": round(rbytes / best2 / 1e3, 1),
            "slowdown_vs_plain": round(best2 / best, 2)}), flush=True)


@dataclass(frozen=True)
class _CcoRuntime:
    context: Any
    communicator: Any
    memory: Any
    window: Any
    dev_comm: Any
    per_rank_vmm: int
    # False when the caller supplied the Communicator: then this operator owns
    # only its window/memory and must not destroy the communicator.
    owns_communicator: bool = True


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)



from .window_view import (  # noqa: E402  (kept next to its only users)
    read_window_bytes as _read_window_bytes,
    read_window_u32 as _read_window_u32,
    read_window_u64 as _read_window_u64,
    window_tensor as _window_tensor,
    zero_window as _zero_window,
)

def _import_factory(module_name: str, factory_name: str) -> Callable[..., Any]:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, factory_name)
    except (ImportError, AttributeError) as error:
        raise NotImplementedError(
            "strict EP16 two-kernel backend is incomplete: expected "
            f"{module_name}:{factory_name}; the old cascade is not a fallback"
        ) from error


def _as_u8_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
    try:
        return tensor.view(torch.uint8)
    except RuntimeError as error:
        raise ValueError(f"{name} must have a byte-addressable packed layout") from error


from .rank_push_layout import RankPushWorkspace, summarize_rank_push_protocol


# 建议在 TPR > 1024 时开启 push 段的 fp8 通信量化;TPR 更小时是净亏的
# (实测 TPR=512:kernel1 +15.5us,kernel2 只省 5.0us,合计 +10.5us)。
# push 段开 fp8 的 mtpr 下界。取 1024,与 MegaMoEv2 当前的
# P2P_FP8_MIN_MTPR 同值,但**不 import 他们的常量**:一来那会把这个 kernel
# 绑进他们的依赖闭包(flydsl 的 cache key 递归哈希依赖源码,他们动一下我们
# 就得重编),二来那个数是按 model_dim 7168 调出来的,我们是 3584、每 token
# 字节数只有一半,交叉点本来就不该是同一个值 —— 这里先对齐,等自己量出
# 交叉点再改这一个数。
COMM_QUANT_PUSH_MIN_MTPR = 1024


def _comm_quant_kind(value: str) -> str:
    """'none'/'fp8' -> the kernels' p2p_quant_type/inbox_quant spelling.

    通信量化是**一个**对外开关,但内部分两段(kernel1 的 push、kernel2 的 rail),
    两段必须能单独开 —— 精度掉了要分得清是哪一段,见单变量实验的纪律。
    """
    text = str(value or "none")
    if text in ("none", ""):
        return "none"
    if text in ("fp8", "fp8_blockwise_1x32"):
        return "fp8_blockwise_1x32"
    raise ValueError(f"comm_quant must be 'none' or 'fp8' (got {value!r})")


def _resolve_stage1_worker_blocks(transport, requested, stage2_workers):
    """Keep the two persistent grids independent without changing legacy defaults."""
    workers = (256 if transport == "sparse_wqe" else int(stage2_workers)) if requested is None else int(requested)
    if not 1 <= workers <= 256:
        raise ValueError("stage1_worker_blocks must be in [1,256]")
    if transport == "sparse_wqe" and workers != 256:
        raise ValueError("sparse_wqe requires stage1_worker_blocks=256")
    return workers


class MegaMoETileA4W4:
    """K3 EP16 hierarchical MegaMoE with exactly two hot GPU launches.

    The instance supports one ordered in-flight forward on one CUDA stream, as
    does MegaMoEV2. ``rank`` is the global EP rank; tensor allocation always
    uses the current local CUDA device so ranks 8--15 work on node 1.

    Public defaults retain direct_atomic/token readiness. The Stage2 Graph
    runner selects rank_local and reducer tuning via the benchmark subclass;
    it does not change these defaults. See the current design and parameter
    rationale in scripts/megamoe_tile/FUSED_STAGE2_DESIGN_20260909.md.
    """

    quant_mode = "a4w4"
    activation = "silu"
    stage1_kernel_regex = r".*megamoe_tile_ep16_stage1.*"
    stage2_kernel_regex = r".*megamoe_tile_ep16_stage2.*"

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int,
        experts: int, topk: int, quant: str, w1: torch.Tensor, w1_scale: torch.Tensor,
        w2: torch.Tensor, w2_scale: torch.Tensor, max_tok_per_rank: int,
        mega_scheme: str = "fixedslot", swiglu_limit: float = 0.0,
        stage1_transport: str = "chunked",
        communicator=None,
        stage1_worker_blocks: int | None = None,
        max_routes_per_token_per_rank: int | None = None,
        stage2_rail_quant_type: str = "none",
        comm_quant: str = "none",
        stage2_gmm_work_swizzle: str = "token_major",
        stage2_window_n_groups: int = 2,
        stage2_ready_granularity: str = "group", activation: str = "silu",
        device_generation: bool = False):
    # fmt: on
        self._validate_static_contract(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            max_tok_per_rank=max_tok_per_rank,
            mega_scheme=mega_scheme,
            swiglu_limit=swiglu_limit,
            stage1_transport=stage1_transport,
            max_routes_per_token_per_rank=max_routes_per_token_per_rank,
            stage2_rail_quant_type=stage2_rail_quant_type,
            comm_quant=comm_quant,
            stage2_gmm_work_swizzle=stage2_gmm_work_swizzle,
            stage2_window_n_groups=stage2_window_n_groups,
            stage2_ready_granularity=stage2_ready_granularity,
            activation=activation,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("MegaMoETileA4W4 requires a ROCm CUDA device")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed Gloo must be initialized before collective CCO setup"
            )
        if dist.get_world_size() != int(world_size) or dist.get_rank() != int(rank):
            raise ValueError(
                "constructor rank/world_size must match the initialized process group"
            )

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = self.experts // self.world_size
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.mega_scheme = str(mega_scheme)
        self.swiglu_limit = float(swiglu_limit)
        self.activation = str(activation)
        self.device_generation = bool(device_generation)
        self.stage1_transport = str(stage1_transport)
        # 和 mega_moe_gfx1250 一致:communicator 由调用方建、由调用方销毁。
        # 不传时退回自建(旧行为),这样单测不必自己做 rendezvous。
        self._external_communicator = communicator
        self.stage2_rail_quant_type = str(stage2_rail_quant_type)
        self.comm_quant = str(comm_quant)
        self.stage2_gmm_work_swizzle = str(stage2_gmm_work_swizzle)
        self.stage2_window_n_groups = int(stage2_window_n_groups)
        self.stage2_ready_granularity = str(stage2_ready_granularity)
        self.max_routes_per_token_per_rank = (
            self.topk
            if max_routes_per_token_per_rank is None
            else int(max_routes_per_token_per_rank)
        )
        if self.mtpr > 128 and self.stage1_transport == "sparse_wqe":
            raise ValueError("sparse_wqe currently supports max_tok_per_rank <= 128")
        self.gpus_per_node = 8
        self.node = self.rank // self.gpus_per_node
        self.local_rank = self.rank % self.gpus_per_node
        self.peer_node = 1 - self.node
        self.device = torch.device("cuda", torch.cuda.current_device())
        # 160 is the default total persistent CTA budget, including service
        # roles. The reduce_push GEMM pool is grid - (1 RAIL + Q rank reducers
        # + R node reducers + F final reducers); the factory records its exact
        # boundary. Tune the total and Q/R/F together. CTA IDs select logical
        # roles and do not bind those roles to physical CUs.
        self.worker_blocks = int(getattr(self, "stage2_worker_blocks", 160))
        # Each stage runs separately and has its own full resident-grid budget.
        # None preserves existing callers; explicit values permit S1-only A/B.
        self.stage1_worker_blocks = _resolve_stage1_worker_blocks(
            self.stage1_transport, stage1_worker_blocks, self.worker_blocks
        )
        device_cus = torch.cuda.get_device_properties(self.device).multi_processor_count
        # The software grid barriers require all CTAs to make progress. Bound
        # the grid by available CUs as a conservative residency prerequisite;
        # actual residency also depends on compiled registers/LDS and contention.
        # Changing grid size mid-instance would also break monotonic epoch math.
        required_cus = max(self.worker_blocks, self.stage1_worker_blocks)
        if device_cus < required_cus:
            raise RuntimeError(
                f"strict persistent kernels require at least {required_cus} CUs, "
                f"got {device_cus}"
            )

        for name, tensor in (
            ("w1", w1),
            ("w1_scale", w1_scale),
            ("w2", w2),
            ("w2_scale", w2_scale),
        ):
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected current device {self.device}"
                )
        self._validate_weight_capacity(w1, w1_scale, w2, w2_scale)
        self._w1 = _as_u8_contiguous(w1, "w1")
        self._w1_scale = _as_u8_contiguous(w1_scale, "w1_scale")
        self._w2 = _as_u8_contiguous(w2, "w2")
        self._w2_scale = _as_u8_contiguous(w2_scale, "w2_scale")

        self.stage1_layout = Stage1ArenaLayout.create(
            hidden=self.model_dim,
            inter=self.inter_dim,
            experts=self.experts,
            world_size=self.world_size,
            gpus_per_node=self.gpus_per_node,
            topk=self.topk,
            max_tokens=self.mtpr,
            max_routes_per_token_per_rank=self.max_routes_per_token_per_rank,
        )
        stage2_node_accumulation_mode = getattr(
            self, "stage2_node_accumulation_mode", "rank_local"
        )
        stage2_rank_accumulation_mode = getattr(
            self, "stage2_rank_accumulation_mode", "reduce_push"
        )
        if self.mtpr > 128 and stage2_rank_accumulation_mode in (
            "staged_reduce",
            "staged_ring",
        ):
            raise ValueError(
                f"{stage2_rank_accumulation_mode} currently supports "
                "max_tok_per_rank <= 128"
            )
        # 用环境变量开关,避免改动构造签名和 bench CLI:接入面越小,
        # 与 candidate 基线的可比性越强(计时/捕获/对比口径完全不变)。
        import os as _os
        self._two_kernel_stage2 = _os.environ.get("MEGAMOE_TWO_KERNEL") == "1"
        self._two_kernel_bn = int(_os.environ.get("MEGAMOE_TK_BN", "128"))
        # 两段各自可覆盖:只开 push 就能单独量出 inbox 减半的收益,
        # 精度掉了也分得清是哪一段吃掉的。
        # push 段跟 MegaMoEv2 的门限走(mega_moe_config.py:102/330/345):
        # mtpr <= P2P_FP8_MIN_MTPR 就不量化。量化的 VALU 代价是每元素固定的,
        # 省下的字节要够多才摊得平 —— 实测 mtpr=512 时 kernel1 +15.5us 而
        # kernel2 只省 5.0us。显式设 MEGAMOE_TK_COMM_QUANT_PUSH 可以压过门限
        # (扫参要能强制两个方向,否则量不出交叉点在哪)。
        _push_env = _os.environ.get("MEGAMOE_TK_COMM_QUANT_PUSH")
        if _push_env is not None:
            self._k1_p2p_quant_type = _comm_quant_kind(_push_env)
            self._k1_p2p_gate = "env"
        elif self.mtpr > COMM_QUANT_PUSH_MIN_MTPR:
            self._k1_p2p_quant_type = _comm_quant_kind(self.comm_quant)
            self._k1_p2p_gate = "above_mtpr_gate"
        else:
            self._k1_p2p_quant_type = "none"
            self._k1_p2p_gate = "below_mtpr_gate"
        self._k2_rail_quant_type = _comm_quant_kind(
            _os.environ.get("MEGAMOE_TK_COMM_QUANT_RAIL", self.comm_quant))
        # kernel2 读 inbox 的格式必须跟着 kernel1 的 push 走,不是跟着 rail。
        self._k2_inbox_quant = self._k1_p2p_quant_type
        # num_qp / return_chunk_tokens 是随 shape 移动的最优点(CHUNK 决定 rail
        # 包大小),所以走 per-shape 查表。优先级 env > 表 > 内置默认:env 保留
        # 是因为单变量实验必须能压过表,否则扫参会被表悄悄改掉一个自变量。
        # 表的 token 口径 = GEMM 行数 = mtpr * topk(EP16 下 1 route/token/rank),
        # 不是 mtpr —— 与 kimik3_a4w4_tuned_fmoe.csv 的 key 语义一致。
        from .stage2_tune import lookup_stage2_tune as _lookup_s2_tune
        try:
            from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

            _s2_tuned = _lookup_s2_tune(
                gfx=get_gfx_runtime(),
                cu_num=get_cu_num(),
                token=self.mtpr * self.topk,
                model_dim=self.model_dim,
                inter_dim=self.inter_dim,
                expert=self.epr,
                topk=self.topk,
            )
        except Exception:
            # 查表永远不该让一次跑挂掉:查不到/读不了就回落到内置默认。
            _s2_tuned = None
        _s2_tuned = _s2_tuned or {}
        self._two_kernel_qp = int(
            _os.environ.get("MEGAMOE_TK_QP", _s2_tuned.get("num_qp", 8)))
        # 查不到表时按机制推:rail 有 num_qp 条队列,每条分到一个 chunk 时
        # 发射最均衡,即 chunk = mtpr / num_qp。实测 TPR=512 -> 64、
        # TPR=1024 -> 128 两点精确命中;TPR=128 上整条曲线只有 1.6% 跨度
        # (c16 203.6 / c32 202.3 / c64 205.6),规则给的 16 比最优差 1.3us,
        # 在那个区间怎么取都无所谓。**不是**按包字节数推 —— 实测证伪了:
        # 包字节坐标上 bf16 和 fp8 的最优点不重合(fp8 在 237KB 见底,
        # bf16 到 458KB 还在降),而 token 坐标上两者形状完全一致。
        _chunk_rule = max(4, self.mtpr // max(1, self._two_kernel_qp))
        self._two_kernel_chunk = int(
            _os.environ.get("MEGAMOE_TK_CHUNK",
                            _s2_tuned.get("return_chunk_tokens", _chunk_rule)))
        # 跑完必须能看出这次用的是哪一组值 —— run_ar.sh 的默认值不是最优配置
        # 这件事已经让一次基线看起来像 1.7x 回归。
        _env_hit = ("MEGAMOE_TK_QP" in _os.environ
                    or "MEGAMOE_TK_CHUNK" in _os.environ)
        self._two_kernel_tune_source = (
            "env" if _env_hit else ("table" if _s2_tuned else "default"))
        self._two_kernel_rail = _os.environ.get("MEGAMOE_TK_RAIL", "1") != "0"
        self._two_kernel_wait_remote = (
            _os.environ.get("MEGAMOE_TK_WAIT_REMOTE", "1") != "0")
        # 0 = 沿用 worker_blocks(现状)。g2_spart 要和实际 CU 数配套。
        self._two_kernel_k1_cu = int(_os.environ.get("MEGAMOE_TK_K1_CU", "0"))
        self._two_kernel_k1_spart = int(_os.environ.get("MEGAMOE_TK_K1_SPART", "402"))
        # 0 = 跟随 cu_num。单节点 179us 那次是 cu_num=256 / persist_cu=240。
        self._two_kernel_k1_pcu = int(_os.environ.get("MEGAMOE_TK_K1_PCU", "0"))
        # 流序只保证**本 rank** 的 kernel1 先于本 rank 的 kernel2,对
        # "别的 rank 推给我的 payload 到了没有"一无所知 —— 那条路上没有任何
        # 同步。所以这个开关不是"将来并发时才有意义",它补的是一个现在就存在
        # 的洞;今天关着也能过精度,靠的是 kernel2 比 kernel1 长 ~7x 的余量。
        # 打开时 kernel1 发 per-(token, slot) 标志,kernel2 归约前逐 token 等。
        self._two_kernel_arrival = (
            _os.environ.get("MEGAMOE_TK_ARRIVAL", "0") != "0")
        # 诊断:数 kernel2 第一眼就没就绪的标志占比。只有这个数非零,才说明
        # 等待真的挡住了什么;否则"打开了精度也对"只是重复了今天的时间余量。
        self._two_kernel_arrival_probe = (
            _os.environ.get("MEGAMOE_TK_ARRIVAL_PROBE", "0") != "0")
        # kernel1 推 payload 的 cache policy。-1 = 沿用默认(19/2,都带 NT)。
        # 消费端要把这 58.7 MB 再读一遍,NT 让它每次都落到 HBM。
        self._two_kernel_paycm = int(
            _os.environ.get("MEGAMOE_TK_PAYCM", "-1") or -1)
        self._two_kernel_bw_probe = (
            _os.environ.get("MEGAMOE_TK_BWPROBE", "0") != "0")
        # 诊断:CTA0 在四个相位边界记墙钟,用来把 K2_TIME 拆成阶段 1/2/3。
        # 不改变任何计算,精度照常校验。
        self._two_kernel_k2_stamp = (
            _os.environ.get("MEGAMOE_TK_K2_STAMP", "0") != "0")
        self._two_kernel_k2_cta_stamp = (
            _os.environ.get("MEGAMOE_TK_K2_CTASTAMP", "0") != "0")
        # kernel2 的 grid。归约按 wave 铺开之后,活干得完不再取决于 CTA 数,
        # 但**延迟掩藏**取决于每个 CU 上的 wave 数:160 CTA x 4 wave / 256 CU
        # = 2.5 wave/CU,基本没有东西可以用来盖住访存延迟。0 = 沿用
        # worker_blocks(现状)。
        self._two_kernel_k2_blocks = int(
            _os.environ.get("MEGAMOE_TK_K2_BLOCKS", "0") or 0)
        # payload 用 sc0|sc1|nt 写穿到系统一致点,于是一次 buffer_wbl2 都不需要。
        # 备选路线,和上面那次 fence 二选一。
        self._two_kernel_arrival_scst = (
            _os.environ.get("MEGAMOE_TK_ARRIVAL_SCSTORE", "0") != "0")
        # 1=只留 decode 循环,2=原子打到 CTA 私有下标(无争用)。都是故意写错的,
        # 只用来二分 7.1ms 到底出在哪一步。
        self._two_kernel_arrival_diag = int(
            _os.environ.get("MEGAMOE_TK_ARRIVAL_DIAG", "0"))
        # diag 模式下 kernel1 不发标志,消费端再等就是挂死。生产端开、消费端关,
        # 只量 kernel1 的成本。
        self._two_kernel_arrival_nowait = (
            _os.environ.get("MEGAMOE_TK_ARRIVAL_NOWAIT", "0") != "0")
        if self._two_kernel_arrival_scst and not self._two_kernel_arrival:
            raise ValueError(
                "MEGAMOE_TK_ARRIVAL_SCSTORE requires MEGAMOE_TK_ARRIVAL=1")
        if self._two_kernel_arrival_probe and not self._two_kernel_arrival:
            raise ValueError(
                "MEGAMOE_TK_ARRIVAL_PROBE requires MEGAMOE_TK_ARRIVAL=1")
        # Stage1 emits GMM1 output in expert-major tile order so GEMM2
        # keeps one expert's weights resident across consecutive
        # m-blocks.  The fused Stage2 reads h1_output through the
        # source-keyed tile_expert, so it must not see this layout.
        self._two_kernel_expert_major = (
            _os.environ.get("MEGAMOE_TK_EXPERT_MAJOR", "0") != "0")
        if self._two_kernel_expert_major and not self._two_kernel_stage2:
            raise ValueError(
                "MEGAMOE_TK_EXPERT_MAJOR requires MEGAMOE_TWO_KERNEL=1"
            )
        self.stage2_layout = Stage2ArenaLayout.create(
            hidden=self.model_dim,
            topk=self.topk,
            max_tokens=self.mtpr,
            world_size=self.world_size,
            gpus_per_node=self.gpus_per_node,
            include_route_slots=(stage2_node_accumulation_mode == "route_store"),
            include_rank_partials=(stage2_node_accumulation_mode == "rank_local"),
            include_staged_reduce=(
                stage2_node_accumulation_mode == "rank_local"
                and stage2_rank_accumulation_mode == "staged_reduce"
            ),
            include_staged_ring=False,
            include_rank_push=(stage2_rank_accumulation_mode == "reduce_push"),
            rail_quant_type=self.stage2_rail_quant_type,
            ready_granularity=self.stage2_ready_granularity,
            ready_group_tiles=int(getattr(self, "stage2_n_tile_group", 2)),
            timeline_history_depth=int(getattr(self, "timeline_history_depth", 0)),
            include_plane_slots=self._two_kernel_stage2,
        )
        # One physical registered window, two non-overlapping logical ABIs.
        # Stage1 writes Stage2 metadata directly through stage2_base; there is
        # no host copy or bridge launch between the two kernels.
        self.layout = TwoKernelArenaLayout.compose(
            self.stage1_layout, self.stage2_layout
        )
        self._runtime: _CcoRuntime | None = None
        self._closed = False
        try:
            self._runtime = self._initialize_cco_runtime()
            # Output is ordinary local memory. Stage2 overwrites every live row
            # before publishing completion; forward only returns a view.
            self._rank_push_workspace = None
            if self.stage2_layout.include_rank_push:
                self._rank_push_workspace = RankPushWorkspace.create(
                    max_tokens=self.mtpr, hidden=self.model_dim, topk=self.topk,
                    max_route_rows=self.stage1_layout.max_route_rows,
                    world_size=self.world_size,
                )
                # Private route payload is not registered or copied by CCO.
                # The public output prefix and both staging parities are disjoint.
                self._output_storage = torch.empty(
                    self._rank_push_workspace.total_bytes,
                    dtype=torch.uint8, device=self.device,
                )
                self._output = self._output_storage[:self._rank_push_workspace.output_bytes].view(
                    torch.bfloat16
                ).view(self.mtpr, self.model_dim)
            else:
                self._output = torch.empty(
                    (self.mtpr, self.model_dim),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            self._stage1 = self._compile_stage1()
            self._stage2 = self._compile_stage2()
            self._validate_launcher_contracts()
            self.stage1_kernel_name = getattr(
                self._stage1, "kernel_name", self.stage1_kernel_regex
            )
            self.stage2_kernel_name = getattr(
                self._stage2, "kernel_name", self.stage2_kernel_regex
            )
            self._generation = 0
            # Constructor-time clear/JIT must be globally complete before the
            # first generation can receive remote writes.
            torch.cuda.synchronize(self.device)
            self._runtime.communicator.barrier()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _validate_static_contract(
        *,
        rank: int,
        world_size: int,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        quant: str,
        max_tok_per_rank: int,
        mega_scheme: str,
        swiglu_limit: float,
        stage1_transport: str = "chunked",
        max_routes_per_token_per_rank: int | None = None,
        stage2_rail_quant_type: str = "none",
        comm_quant: str = "none",
        stage2_gmm_work_swizzle: str = "token_major",
        stage2_window_n_groups: int = 2,
        stage2_ready_granularity: str = "group",
        activation: str = "silu",
    ) -> None:
        if int(world_size) != 16:
            raise ValueError("the current transport requires world_size=16")
        if int(model_dim) < 1024 or int(model_dim) % 512:
            raise ValueError("model_dim must be >= 1024 and divisible by 512")
        if int(model_dim) > 8192:
            raise ValueError(
                "model_dim must be <= 8192 for the four-row 64-KiB return group"
            )
        if int(inter_dim) <= 0 or int(inter_dim) % 256:
            raise ValueError("inter_dim must be positive and divisible by 256")
        if int(experts) <= 0 or int(experts) % int(world_size):
            raise ValueError("experts must be positive and divisible by world_size")
        validate_fused_weight_addressing(
            hidden=model_dim,
            inter=inter_dim,
            experts=experts,
            world_size=world_size,
        )
        if not 1 <= int(topk) <= 16:
            raise ValueError("topk must be in [1,16]")
        if str(activation) not in ("silu", "situv2"):
            raise ValueError("activation must be silu or situv2")
        if not 1 <= int(max_tok_per_rank) <= MAX_FUSED_TOKENS_PER_RANK:
            raise ValueError("max_tok_per_rank must be in [1, 4096]")
        route_cap = (
            int(topk)
            if max_routes_per_token_per_rank is None
            else int(max_routes_per_token_per_rank)
        )
        if not 1 <= route_cap <= int(topk):
            raise ValueError(
                "max_routes_per_token_per_rank must be in [1, topk]"
            )
        _comm_quant_kind(comm_quant)  # raises on an unknown spelling
        if str(stage2_rail_quant_type) not in ("none", "fp8_blockwise"):
            raise ValueError(
                "stage2_rail_quant_type must be 'none' or 'fp8_blockwise'"
            )
        if str(stage2_gmm_work_swizzle) not in (
            "token_major",
            "n_major_window",
        ):
            raise ValueError(
                "stage2_gmm_work_swizzle must be token_major or n_major_window"
            )
        if not 1 <= int(stage2_window_n_groups) <= 64:
            raise ValueError("stage2_window_n_groups must be in [1,64]")
        if str(stage2_ready_granularity) not in ("token", "tile", "group"):
            raise ValueError("stage2_ready_granularity must be token, tile, or group")
        if not 0 <= int(rank) < int(world_size):
            raise ValueError("rank is outside world_size")
        if str(quant).lower() != "a4w4":
            raise ValueError("MegaMoETileA4W4 supports quant='a4w4' only")
        if str(mega_scheme) not in (
            "fixedslot",
            "hierarchical",
            "internode_v1",
        ):
            raise ValueError(
                "mega_scheme must be fixedslot, hierarchical, or internode_v1"
            )
        if float(swiglu_limit) != 0.0:
            raise ValueError(
                "swiglu_limit must remain 0.0 for silu/situv2"
            )
        if str(stage1_transport) not in ("chunked", "sparse_wqe"):
            raise ValueError(
                "stage1_transport must be 'chunked' or 'sparse_wqe'"
            )
        if int(max_tok_per_rank) > 128 and str(stage1_transport) == "sparse_wqe":
            raise ValueError("sparse_wqe currently supports max_tok_per_rank <= 128")

    def _validate_weight_capacity(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        expected_w1_bytes = self.epr * (2 * self.inter_dim) * self.model_dim // 2
        expected_w2_bytes = self.epr * self.model_dim * self.inter_dim // 2
        expected_w1_scale = (
            self.epr * (2 * self.inter_dim) * (self.model_dim // 32)
        )
        expected_w2_scale = self.epr * self.model_dim * (self.inter_dim // 32)
        actual = {
            "w1 bytes": w1.numel() * w1.element_size(),
            "w1_scale bytes": w1_scale.numel() * w1_scale.element_size(),
            "w2 bytes": w2.numel() * w2.element_size(),
            "w2_scale bytes": w2_scale.numel() * w2_scale.element_size(),
        }
        expected = {
            "w1 bytes": expected_w1_bytes,
            "w1_scale bytes": expected_w1_scale,
            "w2 bytes": expected_w2_bytes,
            "w2_scale bytes": expected_w2_scale,
        }
        mismatch = {
            name: (actual[name], expected[name])
            for name in actual
            if actual[name] != expected[name]
        }
        if mismatch:
            detail = ", ".join(
                f"{name}={got} (expected {want})"
                for name, (got, want) in mismatch.items()
            )
            raise ValueError(f"invalid native A4W4 weight capacity: {detail}")

    def _initialize_cco_runtime(self) -> _CcoRuntime:
        from mori.cco import (
            CCODevCommRequirements,
            Communicator,
            GDA_CONNECTION_RAIL,
            UniqueId,
        )

        external = getattr(self, "_external_communicator", None)
        if external is not None:
            return self._build_cco_runtime(external, context=None, owns=False)

        uid_payload = [
            bytes(Communicator.get_unique_id()) if self.rank == 0 else None
        ]
        dist.broadcast_object_list(uid_payload, src=0)
        uid = UniqueId.from_bytes(uid_payload[0])
        # All ranks use the same VMM capacity. Leave headroom for CCO mappings
        # without changing the registered logical window size.
        per_rank_vmm = max(
            128 * 1024 * 1024,
            _align_up(self.layout.total_bytes, 64 * 1024 * 1024),
        )
        context = Communicator.init(
            self.world_size,
            self.rank,
            uid,
            per_rank_vmm=per_rank_vmm,
        )
        communicator = context.__enter__()
        return self._build_cco_runtime(
            communicator, context=context, owns=True, per_rank_vmm=per_rank_vmm
        )

    def _build_cco_runtime(self, communicator, *, context, owns,
                           per_rank_vmm=0) -> _CcoRuntime:
        from mori.cco import CCODevCommRequirements, GDA_CONNECTION_RAIL

        try:
            memory = communicator.alloc_mem(self.layout.total_bytes)
            window = communicator.register_window(memory.ptr, memory.size)
            requirements = CCODevCommRequirements()
            requirements.gda_connection_type = GDA_CONNECTION_RAIL
            if self.stage1_layout.num_qp != self.stage2_layout.num_qp:
                raise AssertionError("Stage1/Stage2 must use the same CCO QP count")
            requirements.gda_context_count = max(
                self.stage1_layout.num_qp,
                self._two_kernel_qp if self._two_kernel_stage2 else 0,
            )
            requirements.gda_signal_count = 0
            requirements.gda_counter_count = 0
            requirements.lsa_barrier_count = 0
            requirements.rail_gda_barrier_count = 0
            requirements.barrier_count = 0
            dev_comm = communicator.create_dev_comm(requirements)

            # This launch is constructor-only and therefore excluded from the
            # strict hot-path trace contract.
            _zero_window(window.local_ptr, self.layout.total_bytes)
            torch.cuda.synchronize(self.device)
            communicator.barrier()
            return _CcoRuntime(
                context,
                communicator,
                memory,
                window,
                dev_comm,
                per_rank_vmm,
                owns,
            )
        except Exception:
            if context is not None:
                context.__exit__(None, None, None)
            raise

    def _compile_stage1(self):
        factory = _import_factory(_STAGE1_MODULE, _STAGE1_FACTORY)
        sparse = self.stage1_transport == "sparse_wqe"
        return factory(
            self.stage1_layout,
            self.stage2_layout,
            rank=self.rank,
            stage2_window_offset=self.layout.stage2_offset,
            worker_blocks=self.stage1_worker_blocks,
            waves_per_eu_hint=2,
            diagnostic_split_fanout=sparse,
            cco_geometry=self.stage1_transport,
            diagnostic_phase=getattr(self, "stage1_diagnostic_phase", "full"),
            activation=self.activation,
            device_generation=self.device_generation,
            tile_pipeline=sparse,
            tile_pipeline_fanout_shards=16,
            timeline_instrument=bool(
                getattr(self, "timeline_instrument", False)
            ),
            expert_major_output=bool(
                getattr(self, "_two_kernel_expert_major", False)
            ),
        )

    def _compile_stage2(self):
        factory = _import_factory(_STAGE2_MODULE, _STAGE2_FACTORY)
        # These are public/factory defaults. The breakdown benchmark overrides
        # this method to pass experiment knobs (including group_batch). Fixed
        # GEMM geometry/8 work shards match the Stage1 sort and Stage2 arena;
        # waves_per_eu=2 and final_combine_blocks=14 are resource choices, not
        # correctness constants or established performance optima.
        return factory(
            self.layout,
            rank=self.rank,
            BM=32,
            BN=256,
            BK=256,
            WORK_SHARDS=8,
            waves_per_eu_hint=2,
            team="rail",
            device_generation=self.device_generation,
            accumulator_dtype="bf16",
            final_combine_blocks=14,
            gmm_schedule="persistent_queue",
            return_chunk_tokens=int(
                getattr(self, "stage2_return_chunk_tokens", 8)
            ),
            bf16_atomic_kind="buffer",
            rail_return_schedule=getattr(
                self, "stage2_rail_return_schedule", "lockstep"
            ),
            epilogue_schedule="lane32_meta",
            n_tile_group=int(getattr(self, "stage2_n_tile_group", 2)),
            group_pipeline_schedule="a_double_buffer",
            node_accumulation_mode=getattr(
                self, "stage2_node_accumulation_mode", "rank_local"
            ),
            rank_accumulation_mode=getattr(
                self, "stage2_rank_accumulation_mode", "reduce_push"
            ),
            rank_reduce_blocks=int(getattr(self, "stage2_rank_reduce_blocks", 8)),
            rank_push_batch_size=int(getattr(self, "stage2_rank_push_batch_size", 1)),
            gemm_use_nt=bool(getattr(self, "stage2_gemm_use_nt", False)),
            rank_push_use_nt=bool(getattr(self, "stage2_rank_push_use_nt", False)),
            node_reduce_blocks=int(
                getattr(self, "stage2_node_reduce_blocks", 32)
            ),
            node_reduce_vec_bytes=int(
                getattr(self, "stage2_node_reduce_vec_bytes", 16)
            ),
            node_reduce_schedule=getattr(
                self, "stage2_node_reduce_schedule", "token"
            ),
            node_reduce_load_schedule=getattr(
                self,
                "stage2_node_reduce_load_schedule",
                "load_first",
            ),
            node_reduce_work_schedule=getattr(
                self,
                "stage2_node_reduce_work_schedule",
                "static_strided",
            ),
            node_reduce_rejoin_blocks=int(
                getattr(self, "stage2_node_reduce_rejoin_blocks", 0)
            ),
            rank_epilogue_lds_addressing=getattr(
                self, "stage2_rank_epilogue_lds_addressing", "expanded"
            ),
            rank_epilogue_barrier=getattr(
                self, "stage2_rank_epilogue_barrier", "per_row"
            ),
            timeline_instrument=bool(
                getattr(self, "timeline_instrument", False)
            ),
            rail_quant_type=self.stage2_rail_quant_type,
            gmm_work_swizzle=self.stage2_gmm_work_swizzle,
            window_n_groups=self.stage2_window_n_groups,
            ready_granularity=self.stage2_ready_granularity,
        )

    def _validate_launcher_contracts(self) -> None:
        for label, launcher, pattern in (
            ("Stage1", self._stage1, self.stage1_kernel_regex),
            ("Stage2", self._stage2, self.stage2_kernel_regex),
        ):
            if getattr(launcher, "single_gpu_launch", None) is not True:
                raise RuntimeError(
                    f"{label} launcher must declare single_gpu_launch=True"
                )
            kernel_name = getattr(launcher, "kernel_name", "")
            if not kernel_name or re.fullmatch(pattern, kernel_name) is None:
                raise RuntimeError(
                    f"{label} kernel_name={kernel_name!r} does not match {pattern!r}"
                )
        if not getattr(self, "diagnostic_only", False):
            sparse = self.stage1_transport == "sparse_wqe"
            expected_stage1 = {
                "cco_geometry": self.stage1_transport,
                "worker_blocks": self.stage1_worker_blocks,
                "diagnostic_split_fanout": sparse,
                "diagnostic_wave_fanout": False,
                "diagnostic_comm_only": False,
                "tile_pipeline": sparse,
                "tile_pipeline_fanout_shards": 16,
                "tile_pipeline_instrument": False,
                "gemm1_contraction": True,
                "full_stage1_fusion": True,
            }
            mismatch = {
                name: (getattr(self._stage1, name, "<missing>"), value)
                for name, value in expected_stage1.items()
                if getattr(self._stage1, name, "<missing>") != value
            }
            if mismatch:
                raise RuntimeError(
                    f"Stage1 transport contract mismatch: {mismatch}"
                )
            architecture = getattr(
                self._stage1, "architecture_contract", {}
            )
            expected_architecture = {
                "early_full_tile_enqueue": sparse,
                "tile_pipeline": sparse,
                "tile_pipeline_fanout_shards": 16,
                "queue_publication": (
                    "full_tile_last_arrival_plus_partial_tile_post_8_role_eos"
                    if sparse
                    else "post_8_role_eos_physical_major"
                ),
                "gmm_scheduler": (
                    "concurrent_ready_queue_8_shards_256_all_roles_rejoin"
                    if sparse
                    else f"post_eos_static_strided_{self.stage1_worker_blocks - 8 - min(128, self.mtpr)}_pure_compute_consumers"
                ),
            }
            bad_architecture = {
                name: (architecture.get(name, "<missing>"), value)
                for name, value in expected_architecture.items()
                if architecture.get(name, "<missing>") != value
            }
            if bad_architecture:
                raise RuntimeError(
                    f"Stage1 fusion architecture mismatch: {bad_architecture}"
                )
            expected_stage2 = {
                "diagnostic_mode": "full",
                "accumulator_dtype": "bf16",
                "final_combine_blocks": 14,
                "gmm_schedule": "persistent_queue",
                "return_chunk_tokens": int(
                    getattr(self, "stage2_return_chunk_tokens", 8)
                ),
                "node_ready_granularity": "token",
                "bf16_atomic_kind": "buffer",
                "gemm2_contraction": True,
                "communication_roles_enabled": True,
                "node_reduce_work_schedule": getattr(
                    self,
                    "stage2_node_reduce_work_schedule",
                    "static_strided",
                ),
                "node_reduce_rejoin_blocks": int(
                    getattr(self, "stage2_node_reduce_rejoin_blocks", 0)
                ),
                "rank_epilogue_lds_addressing": getattr(
                    self, "stage2_rank_epilogue_lds_addressing", "expanded"
                ),
                "rank_epilogue_barrier": getattr(
                    self, "stage2_rank_epilogue_barrier", "per_row"
                ),
                "rank_accumulation_mode": getattr(
                    self, "stage2_rank_accumulation_mode", "reduce_push"
                ),
                "rail_quant_type": self.stage2_rail_quant_type,
                "gmm_work_swizzle": self.stage2_gmm_work_swizzle,
                "window_n_groups": self.stage2_window_n_groups,
                "ready_granularity": self.stage2_ready_granularity,
                "n_tile_group": int(getattr(self, "stage2_n_tile_group", 2)),
            }
            stage2_mismatch = {
                name: (getattr(self._stage2, name, "<missing>"), value)
                for name, value in expected_stage2.items()
                if getattr(self._stage2, name, "<missing>") != value
            }
            if stage2_mismatch:
                raise RuntimeError(
                    f"Stage2 fusion contract mismatch: {stage2_mismatch}"
                )
            stage2_architecture = getattr(
                self._stage2, "architecture_contract", {}
            )
            expected_stage2_architecture = {
                name: expected_stage2[name]
                for name in (
                    "node_reduce_work_schedule",
                    "node_reduce_rejoin_blocks",
                )
            }
            stage2_architecture_mismatch = {
                name: (stage2_architecture.get(name, "<missing>"), value)
                for name, value in expected_stage2_architecture.items()
                if stage2_architecture.get(name, "<missing>") != value
            }
            if stage2_architecture_mismatch:
                raise RuntimeError(
                    "Stage2 fusion architecture mismatch: "
                    f"{stage2_architecture_mismatch}"
                )

    @staticmethod
    def _flydsl_stream(stream):
        import flydsl.expr as fx

        if stream is None:
            return fx.Stream(torch.cuda.current_stream())
        if isinstance(stream, torch.cuda.Stream):
            return fx.Stream(stream)
        return stream

    def _launch_stage1(
        self,
        x_bf16: torch.Tensor,
        wts: torch.Tensor,
        topk_ids: torch.Tensor,
        run_tokens: int,
        generation: int,
        stream,
        *,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("CCO runtime is closed")
        self._stage1(
            runtime.dev_comm.ptr,
            runtime.window.handle,
            runtime.window.local_ptr,
            x_bf16.data_ptr(),
            0 if input_scale is None else input_scale.data_ptr(),
            wts.data_ptr(),
            topk_ids.data_ptr(),
            self._w1.data_ptr(),
            self._w1_scale.data_ptr(),
            run_tokens,
            generation,
            stream=stream,
        )

    def _launch_stage2(
        self,
        run_tokens: int,
        generation: int,
        stream,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("CCO runtime is closed")
        if self._two_kernel_stage2:
            self._launch_two_kernel_stage2(run_tokens, generation, stream)
            return
        # The compiled launcher owns all Stage1/Stage2 arena offsets. Passing
        # only the composite window base prevents a host-side metadata bridge.
        self._stage2(
            runtime.dev_comm.ptr,
            runtime.window.handle,
            runtime.window.local_ptr,
            self._w2.data_ptr(),
            self._w2_scale.data_ptr(),
            generation,
            run_tokens,
            self.worker_blocks,
            self._output.data_ptr(),
            stream=stream,
        )

    def _two_kernel_prepare(self):
        """kernel1 的一次性附属结构:恒等 trb、零字、peer 基址表。

        peer 地址只有 device 侧能拿(cco.lsa_ptr),所以这张表由一个一次性的
        小 kernel 写出来,和 bench 里的做法一致。
        """
        if getattr(self, "_two_kernel_ready", False):
            return
        import torch as _torch
        from .kernel1_adapter import identity_trb, peer_table_launcher
        dev = self._w2.device
        s1 = self.layout.stage1
        self._k1_trb = identity_trb(s1.max_route_rows, 32, dev)
        self._k1_zero = _torch.zeros(64, dtype=_torch.int32, device=dev)
        # 每个 parity 一张表,各填一次即可(见下面的 _k1_filled)。
        self._k1_tables = [
            _torch.zeros(self.world_size, dtype=_torch.int64, device=dev),
            _torch.zeros(self.world_size, dtype=_torch.int64, device=dev),
        ]
        self._k1_filled = set()
        self._k1_fill = peer_table_launcher(self.gpus_per_node, self.world_size)
        self._k2_scratch = _torch.zeros(1024, dtype=_torch.int32, device=dev)   # 256 槽位 + [2][max_chunks] 的 chunk 完成计数
        # cta_stamp 的时间线单独一块 GM,不挤 scratch:[CTA_STAMP_MAX_CTAS][SLOTS] i64。
        # 不开 CTASTAMP 时也分配(2048 个 i64 = 16 KB),省掉一条空指针分支。
        from .stage2_node_combine import (
            CTA_STAMP_MAX_CTAS as _CSMAX, TIMELINE_WORDS as _TLW)
        self._k2_timeline = _torch.zeros(_TLW, dtype=_torch.int64, device=dev)
        if self._two_kernel_k2_cta_stamp:
            _blk = int(self._two_kernel_k2_blocks or self.worker_blocks)
            if _blk > _CSMAX:
                raise ValueError(
                    f"MEGAMOE_TK_K2_CTASTAMP 只记前 {_CSMAX} 个 CTA,"
                    f"当前 grid={_blk};调大 CTA_STAMP_MAX_CTAS 再跑")
        # One i32 per (packed source row, top-k slot) == per producer row.
        # recv_cap-keyed, not destination-keyed: see publish_arrival.
        self._k1_arrival_count = _torch.zeros(
            self.world_size * self.mtpr * self.topk,
            dtype=_torch.int32, device=dev)
        self._two_kernel_ready = True

    def _launch_two_kernel_stage2(self, run_tokens, generation, stream):
        import flydsl.expr as fx
        from .kernel1_adapter import (
            arrival_delta, build_kernel1_args, plane_slot_offset)
        from .stage2_gemm_push import run_mega_moe_stage2
        from .stage2_node_combine import run_stage2_node_combine

        self._two_kernel_prepare()
        runtime = self._runtime
        window = runtime.window
        arena = self.layout
        s2 = arena.stage2
        parity = int(generation) & 1
        # `stream` 已经过 _flydsl_stream,是 fx.Stream;再包一层会炸。
        s_fx = stream

        _k1_cu = self._two_kernel_k1_cu or self.worker_blocks

        # ---- kernel1: GEMM2 + push 到各 peer 的 plane_slot_inbox ----------
        k1_table = self._k1_tables[parity]
        if parity not in self._k1_filled:
            # 表内容跨代不变,只在每个 parity 第一次用时填。
            self._k1_fill(fx.Int64(window.handle), fx.Int64(k1_table.data_ptr()),
                          fx.Int64(plane_slot_offset(arena, parity)), s_fx)
            self._k1_filled.add(parity)
        _k1_args = build_kernel1_args(
                window=window, arena=arena, parity=parity,
                tensors={"bq": self._w2, "bs": self._w2_scale},
                trb=self._k1_trb, p2p_table=k1_table,
                zero_i32=self._k1_zero,
                expert_major=self._two_kernel_expert_major)
        _k1_pos = (arena.stage1.max_route_rows, int(self.inter_dim),
                   int(self.model_dim))

        if not getattr(self, "_k1_griddump", False):
            self._k1_griddump = True
            _s1 = arena.stage1
            _rv = _s1.region("num_valid")
            _nv = _read_window_u32(int(window.local_ptr) + _rv.offset
                         + parity * (_rv.nbytes // 2), 2)
            _re = _s1.region("tile_expert")
            _c0 = int(_nv[0])
            _tmb = (_c0 + 31) // 32
            _eids = _read_window_u32(int(window.local_ptr) + _re.offset
                           + parity * (_re.nbytes // 2),
                           max(1, min(_tmb, _re.nbytes // 2 // 4)))
            _bq, _bs = self._w2, self._w2_scale
            _u = list(_eids)[:_tmb]
            _runs, _cur, _n = [], (_u[0] if _u else -1), 1
            for _v in _u[1:]:
                if _v == _cur:
                    _n += 1
                else:
                    _runs.append(_n); _cur, _n = _v, 1
            _runs.append(_n)
            print("GRIDDUMP_RUNS rank=%d m_blocks=%d runs=%d avg_run=%.2f max_run=%d "
                  "uniq=%d head=%s"
                  % (self.rank, _tmb, len(_runs), sum(_runs) / len(_runs), max(_runs),
                     len(set(_u)), _u[:40]), flush=True)
            if self._two_kernel_expert_major:
                def _runs_of(seq):
                    out, cur, n = [], (seq[0] if seq else -1), 1
                    for v in seq[1:]:
                        if v == cur:
                            n += 1
                        else:
                            out.append(n); cur, n = v, 1
                    out.append(n)
                    return out
                _rs = _s1.region("tile_expert_sorted")
                _es = list(_read_window_u32(int(window.local_ptr) + _rs.offset
                                  + parity * (_rs.nbytes // 2), _tmb))[:_tmb]
                _rp = _s1.region("tile_dst_of_src")
                _pm = list(_read_window_u32(int(window.local_ptr) + _rp.offset
                                  + parity * (_rp.nbytes // 2), _tmb))[:_tmb]
                _sr = _runs_of(_es)
                _perm_ok = sorted(_es) == sorted(_u)
                _bij_ok = sorted(_pm) == list(range(_tmb))
                _mono_ok = all(_es[i] <= _es[i + 1] for i in range(len(_es) - 1))
                print("EMDUMP rank=%d m_blocks=%d runs=%d avg_run=%.2f max_run=%d "
                      "perm_ok=%s bijection_ok=%s expert_monotonic=%s head=%s"
                      % (self.rank, _tmb, len(_sr), sum(_sr) / len(_sr), max(_sr),
                         _perm_ok, _bij_ok, _mono_ok, _es[:40]), flush=True)
            print("GRIDDUMP_PROD rank=%d nvalid=%s cumsum0=%d total_m_blocks=%d "
                  "eid_min=%d eid_max=%d eids_cap=%d max_route_rows=%d "
                  "bq_dtype=%s bq_numel=%d bq_nbytes=%d bs_numel=%d bs_nbytes=%d"
                  % (self.rank, list(_nv), _c0, _tmb,
                     min(_eids), max(_eids), _re.nbytes // 2 // 4,
                     _s1.max_route_rows,
                     _bq.dtype, _bq.numel(), _bq.numel() * _bq.element_size(),
                     _bs.numel(), _bs.numel() * _bs.element_size()),
                  flush=True)
        _bin_dir = __import__("os").environ.get("MEGAMOE_TK_LOAD_BIN", "")
        if _bin_dir and not getattr(self, "_k1_bin_loaded", False):
            self._k1_bin_loaded = True
            import numpy as _np, pathlib as _pl
            _slots = {"aq": 0, "asc": 1, "bq": 2, "bs": 3, "experts": 4,
                      "nvalid": 5, "sources": 7, "weights": 8}
            _k1_args = list(_k1_args)
            self._k1_bins = []
            _d = _pl.Path(_bin_dir)
            _loaded = []
            for _name, _slot in _slots.items():
                _f = _d / ("rank%d_%s.bin" % (self.rank, _name))
                if not _f.exists():
                    raise FileNotFoundError(str(_f))
                _raw = _np.fromfile(str(_f), dtype=_np.uint8)
                _buf = torch.from_numpy(_raw).to(self._w2.device)
                self._k1_bins.append(_buf)
                _k1_args[_slot] = fx.Int64(int(_buf.data_ptr()))
                _loaded.append("%s:%d" % (_name, _raw.nbytes))
            self._k1_args_bin = tuple(_k1_args)
            print("TWO_KERNEL_K1_LOAD_BIN rank=%d %s" % (self.rank, ",".join(_loaded)),
                  flush=True)
        if getattr(self, "_k1_args_bin", None) is not None:
            _k1_args = self._k1_args_bin
        import os as _osc
        _copy_sel = _osc.environ.get("MEGAMOE_TK_K1_COPYIN", "")
        if _copy_sel and not getattr(self, "_k1_copied", False):
            self._k1_copied = True
            import ctypes as _ct
            _slots = {"aq": (0, "h1_output_q"), "asc": (1, "h1_output_scale"),
                      "eids": (4, "tile_expert"), "stids": (7, "tile_row_source"),
                      "swts": (8, "tile_row_weight")}
            _k1_args = list(_k1_args)
            self._k1_copies = []
            for _name in [x.strip() for x in _copy_sel.split(",") if x.strip()]:
                _slot, _region = _slots[_name]
                _r = arena.stage1.region(_region)
                _n = _r.nbytes // 2
                _src = int(window.local_ptr) + _r.offset + parity * _n
                _buf = torch.empty(_n, dtype=torch.uint8, device=self._w2.device)
                torch.cuda.synchronize()
                _buf.copy_(_window_tensor(_src, _n, torch.uint8))
                self._k1_copies.append(_buf)          # 保活
                _k1_args[_slot] = fx.Int64(int(_buf.data_ptr()))
            self._k1_args_copied = tuple(_k1_args)
            print("TWO_KERNEL_K1_COPYIN " + repr(
                {"copied": _copy_sel, "rank": self.rank}), flush=True)
        if getattr(self, "_k1_args_copied", None) is not None:
            _k1_args = self._k1_args_copied
        _k1_kw = dict(
            model_dim=self.model_dim, inter_dim=self.inter_dim,
            # rank=0 不是笔误:tile stage1 的 tile_expert 已经是 LOCAL id,
            # 而 MegaMoEv2 的 kernel 会再减一次 rank*experts。见 kernel1 的注释。
            experts=self.epr, topk=self.topk, rank=0,
            npes=self.world_size, max_tok=self.mtpr,
            recv_cap=self.world_size * self.mtpr,
            comb_inp_nbytes=2 * self.mtpr * self.topk * self.model_dim * 2,
            BM=32, SBM=32, BN=self._two_kernel_bn, BK=256, a_dtype="fp4",
            HIDDEN_MAX=self.model_dim, INTER_MAX=self.inter_dim,
            cu_num=_k1_cu, persist=True,
            persist_cu=(self._two_kernel_k1_pcu or _k1_cu), persist_strided=True,
            use_nt=False, g2_spart=self._two_kernel_k1_spart,
            p2p_quant_type=self._k1_p2p_quant_type,
            ep16_plane_slots=True, gpus_per_node=self.gpus_per_node,
            ep16_arrival_publish=self._two_kernel_arrival,
            arrival_delta=arrival_delta(arena, parity),
            arrival_count=int(self._k1_arrival_count.data_ptr()),
            generation=int(generation),
            arrival_sc_store=self._two_kernel_arrival_scst,
            arrival_diag=self._two_kernel_arrival_diag,
            pay_cm_override=self._two_kernel_paycm,
        )
        run_mega_moe_stage2(*_k1_args, *_k1_pos, s_fx, **_k1_kw)

        # ---- 可选:隔离计时 kernel1(与单节点 harness 同方法) -------------
        import os as _os2
        _k1t = int(_os2.environ.get("MEGAMOE_TK_K1_TIME", "0"))
        # SOLO_K1 打开时,必须等 stage1 已经停跑的那次 forward 才计时;
        # 否则计时那一刻 stage1 刚执行过,等于没隔离。
        # 注意 parity:generation 每次 forward 翻转,所以 stage1 至少要跑满
        # 两次(两个 parity 都写过)才能停,不然 kernel1 会读到没写过的那一半。
        _solo_n = int(_os2.environ.get("MEGAMOE_TK_SOLO_K1", "0"))
        _solo_ready = (_solo_n == 0) or (getattr(self, "_solo_s1_runs", 0) >= _solo_n)
        if _k1t > 0 and _solo_ready and not getattr(self, "_k1_timed", False) and not (
                torch.cuda.is_current_stream_capturing()):
            self._k1_timed = True
            import torch.distributed as _dist

            def _run_k1(_s):
                run_mega_moe_stage2(*_k1_args, *_k1_pos, _s, **_k1_kw)

            for _ in range(3):
                _run_k1(s_fx)
            torch.cuda.synchronize()
            if _dist.is_initialized():
                _dist.barrier()
            _cap = torch.cuda.Stream()
            _cap.wait_stream(torch.cuda.current_stream())
            _g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(_cap):
                with torch.cuda.graph(_g, stream=_cap):
                    _run_k1(fx.Stream(torch.cuda.current_stream().cuda_stream))
            for _ in range(3):
                _g.replay()
            torch.cuda.synchronize()
            if _dist.is_initialized():
                _dist.barrier()
            _ev = [(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True)) for _ in range(_k1t)]
            for _i in range(_k1t):
                _ev[_i][0].record()
                _g.replay()
                _ev[_i][1].record()
            torch.cuda.synchronize()
            _us = sorted(a.elapsed_time(b) * 1000.0 for a, b in _ev)
            _mine = dict(rank=self.rank, min_us=_us[0],
                         mean_us=sum(_us) / len(_us))
            if _dist.is_initialized():
                _all = [None] * _dist.get_world_size()
                _dist.all_gather_object(_all, _mine)
            else:
                _all = [_mine]
            print("TWO_KERNEL_K1_ISOLATED " + repr(dict(
                pooled_min_us=min(g["min_us"] for g in _all),
                min_of_means_us=min(g["mean_us"] for g in _all),
                mean_of_means_us=sum(g["mean_us"] for g in _all) / len(_all),
                slowest_mean_us=max(g["mean_us"] for g in _all),
                ranks=len(_all),
                per_rank=[(g["rank"], round(g["min_us"],1),
                           round(g["mean_us"],1))
                          for g in sorted(_all, key=lambda x: x["rank"])],
                )), flush=True)

        # ---- kernel2: node 内归约 -> rail 发送 -> node 间合并 -------------
        def s2_off(name):
            region = s2.region(name)
            return (int(arena.stage2_offset) + region.offset
                    + parity * (region.nbytes // s2.parity_depth))

        def s2_bytes(name):
            return s2.region(name).nbytes // s2.parity_depth

        if not getattr(self, "_nv_logged", False):
            self._nv_logged = True
            import torch as _t
            _s1 = arena.stage1
            _r = _s1.region("num_valid")
            _off = _r.offset + parity * (_r.nbytes // 2)
            _t.cuda.synchronize()
            _nv = _read_window_u32(int(window.local_ptr) + _off, 4)
            print("TWO_KERNEL_NUM_VALID rank=%d parity=%d values=%s "
                  "max_route_rows=%d" % (self.rank, parity, list(_nv),
                                         _s1.max_route_rows), flush=True)

        run_stage2_node_combine(
            int(runtime.dev_comm.ptr), int(window.handle), int(window.local_ptr),
            int(self._output.data_ptr()), int(self._k2_scratch.data_ptr()),
            int(generation), int(run_tokens),
            int(self._two_kernel_k2_blocks or self.worker_blocks), s_fx,
            arg_timeline=int(self._k2_timeline.data_ptr()),
            hidden=self.model_dim, max_tokens=self.mtpr, topk=self.topk,
            rank=self.rank, gpus_per_node=self.gpus_per_node,
            num_qp=self._two_kernel_qp,
            return_chunk_tokens=self._two_kernel_chunk,
            threads=256, enable_rail=self._two_kernel_rail,
            stamp=self._two_kernel_k2_stamp,
            cta_stamp=self._two_kernel_k2_cta_stamp,
            wait_remote=self._two_kernel_wait_remote,
            inbox_quant=self._k2_inbox_quant,
            rail_quant=self._k2_rail_quant_type,
            inbox_off=s2_off("plane_slot_inbox"),
            inbox_bytes=s2_bytes("plane_slot_inbox"),
            slot_mask_off=s2_off("node_dest_slot_mask"),
            accumulator_off=s2_off("node_accumulator"),
            accumulator_bytes=s2_bytes("node_accumulator"),
            rx_off=s2_off("remote_partial_rx"),
            rx_bytes=s2_bytes("remote_partial_rx"),
            partial_ready_off=s2_off("node_partial_ready"),
            group_ready_off=s2_off("return_group_ready"),
            consumed_off=s2_off("return_consumed"),
            arrival_off=s2_off("plane_slot_arrived"),
            arrival_wait=(self._two_kernel_arrival
                          and not self._two_kernel_arrival_nowait),
            arrival_probe=self._two_kernel_arrival_probe,
            s2_window_off=int(arena.stage2_offset),
        )

        # ---- 可选:隔离计时 kernel2(与 MEGAMOE_TK_K1_TIME 同方法) ---------
        # 没有这个数就只能从 full forward 里减 kernel2,而它的噪声 ±250us 比
        # 要优化的量还大。solo replay 时到达标志已经是本代的值,wait_ready 是
        # ">= generation" 立刻通过,所以量到的是**不含等待**的纯计算成本 ——
        # 正是调 kernel2 需要的基准。
        _k2t = int(_os2.environ.get("MEGAMOE_TK_K2_TIME", "0"))
        if _k2t > 0 and not getattr(self, "_k2_timed", False) and not (
                torch.cuda.is_current_stream_capturing()):
            self._k2_timed = True
            import torch.distributed as _dist2

            def _run_k2(_s):
                run_stage2_node_combine(
            int(runtime.dev_comm.ptr), int(window.handle), int(window.local_ptr),
            int(self._output.data_ptr()), int(self._k2_scratch.data_ptr()),
            int(generation), int(run_tokens),
            int(self._two_kernel_k2_blocks or self.worker_blocks), _s,
            arg_timeline=int(self._k2_timeline.data_ptr()),
            hidden=self.model_dim, max_tokens=self.mtpr, topk=self.topk,
            rank=self.rank, gpus_per_node=self.gpus_per_node,
            num_qp=self._two_kernel_qp,
            return_chunk_tokens=self._two_kernel_chunk,
            threads=256, enable_rail=self._two_kernel_rail,
            stamp=self._two_kernel_k2_stamp,
            cta_stamp=self._two_kernel_k2_cta_stamp,
            wait_remote=self._two_kernel_wait_remote,
            inbox_quant=self._k2_inbox_quant,
            rail_quant=self._k2_rail_quant_type,
            inbox_off=s2_off("plane_slot_inbox"),
            inbox_bytes=s2_bytes("plane_slot_inbox"),
            slot_mask_off=s2_off("node_dest_slot_mask"),
            accumulator_off=s2_off("node_accumulator"),
            accumulator_bytes=s2_bytes("node_accumulator"),
            rx_off=s2_off("remote_partial_rx"),
            rx_bytes=s2_bytes("remote_partial_rx"),
            partial_ready_off=s2_off("node_partial_ready"),
            group_ready_off=s2_off("return_group_ready"),
            consumed_off=s2_off("return_consumed"),
            arrival_off=s2_off("plane_slot_arrived"),
            arrival_wait=(self._two_kernel_arrival
                          and not self._two_kernel_arrival_nowait),
            arrival_probe=self._two_kernel_arrival_probe,
            s2_window_off=int(arena.stage2_offset),
                )

            for _ in range(3):
                _run_k2(s_fx)
            torch.cuda.synchronize()
            if _dist2.is_initialized():
                _dist2.barrier()
            _cap2 = torch.cuda.Stream()
            _cap2.wait_stream(torch.cuda.current_stream())
            _g2 = torch.cuda.CUDAGraph()
            with torch.cuda.stream(_cap2):
                with torch.cuda.graph(_g2, stream=_cap2):
                    _run_k2(fx.Stream(torch.cuda.current_stream().cuda_stream))
            for _ in range(3):
                _g2.replay()
            torch.cuda.synchronize()
            if _dist2.is_initialized():
                _dist2.barrier()
            _ev2 = [(torch.cuda.Event(enable_timing=True),
                     torch.cuda.Event(enable_timing=True)) for _ in range(_k2t)]
            for _i in range(_k2t):
                _ev2[_i][0].record()
                _g2.replay()
                _ev2[_i][1].record()
            torch.cuda.synchronize()
            _us2 = sorted(a.elapsed_time(b) * 1000.0 for a, b in _ev2)
            _mine2 = dict(rank=self.rank, min_us=_us2[0],
                          mean_us=sum(_us2) / len(_us2))
            if _dist2.is_initialized():
                _all2 = [None] * _dist2.get_world_size()
                _dist2.all_gather_object(_all2, _mine2)
            else:
                _all2 = [_mine2]
            print("TWO_KERNEL_K2_ISOLATED " + repr(dict(
                pooled_min_us=min(g["min_us"] for g in _all2),
                min_of_means_us=min(g["mean_us"] for g in _all2),
                mean_of_means_us=sum(g["mean_us"] for g in _all2) / len(_all2),
                slowest_mean_us=max(g["mean_us"] for g in _all2),
                ranks=len(_all2), rail=int(self._two_kernel_rail),
                per_rank=[(g["rank"], round(g["min_us"],1),
                           round(g["mean_us"],1))
                          for g in sorted(_all2, key=lambda x: x["rank"])],
                )), flush=True)
        if self._two_kernel_bw_probe:
            import torch as _t
            if not _t.cuda.is_current_stream_capturing():
                self._two_kernel_bw_probe = False
                # 探针的 buffer resource 用请求的大小做 num_records,所以
                # 请求量必须 <= 该 region 的真实大小 —— 否则越过 arena 末尾
                # 读到未映射的地址,硬件边界检查管不到,直接 illegal access。
                _cap = min(128 << 20, s2_bytes("plane_slot_inbox"))
                _plain = _t.empty(_cap + (16 << 20), dtype=_t.int8, device="cuda")
                _bw_probe(_cap, self.rank, [
                    ("arena_inbox",
                     int(window.local_ptr) + s2_off("plane_slot_inbox")),
                    ("plain_torch", int(_plain.data_ptr()))],
                    reduce_shape=(
                        int(window.local_ptr) + s2_off("plane_slot_inbox"),
                        int(run_tokens), int(self.model_dim), int(self.topk)))
                del _plain
        if self._two_kernel_k2_stamp:
            import torch as _t
            if not _t.cuda.is_current_stream_capturing():
                _t.cuda.synchronize()
                # 四个 i64 墙钟戳:入口 / 阶段1 后 / 阶段2 后 / 结束。频率不需要
                # 知道 —— 按 (t3-t0) 归一化成占比,再乘上 K2_TIME 量到的时间。
                _st = self._k2_scratch[768:776].view(_t.int64).tolist()   # SCRATCH_STAMP,和 stage2_node_combine.py 保持一致
                _t0, _t1, _t2, _t3 = _st
                _span = max(_t3 - _t0, 1)
                print("TWO_KERNEL_K2_STAMP " + repr({
                    "rank": self.rank, "generation": int(generation),
                    "ticks_total": _t3 - _t0,
                    "frac_phase1": (_t1 - _t0) / _span,
                    "frac_phase2": (_t2 - _t1) / _span,
                    "frac_phase3": (_t3 - _t2) / _span}), flush=True)
        if self._two_kernel_k2_cta_stamp:
            import torch as _t
            if not _t.cuda.is_current_stream_capturing():
                _t.cuda.synchronize()
                from .stage2_node_combine import (
                    CTA_STAMP_SLOTS as _CSSLOT, CTA_STAMP_MAX_CTAS as _CSMAX)
                _blk = int(self._two_kernel_k2_blocks or self.worker_blocks)
                _tl = self._k2_timeline.view(-1, _CSSLOT)[:_blk].tolist()
                # t0 = 最早起跑的 CTA,所有戳都相对它;未写过的槽是 0,原样留着。
                _base = min((r[0] for r in _tl if r[0]), default=0)
                _rows = []
                for _i, _r in enumerate(_tl):
                    _rows.append([_i, int(_r[7]) & 0xFFFFFFFF]
                                 + [(int(_v) - _base) if _v else -1
                                    for _v in list(_r[:7]) + [_r[8]]])
                print("TWO_KERNEL_K2_CTASTAMP " + repr({
                    "rank": self.rank, "generation": int(generation),
                    "blocks": _blk,
                    "cols": ["cta", "hw_id", "entry", "p1_end", "wqe",
                             "db_wait", "db_done", "p3_rdy", "p3_end", "spun"],
                    "ticks": _rows}), flush=True)
        if self._two_kernel_arrival_probe:
            import torch as _t
            if _t.cuda.is_current_stream_capturing():
                # 探针要 synchronize + D2H 读回计数,capture 期间两者都非法
                # (HIP: "operation not permitted when stream is capturing")。
                # 前几代 eager 的数已经够用,capture 之后不再打印。
                return
            _t.cuda.synchronize()
            _seen, _late = self._k2_scratch[66].item(), self._k2_scratch[67].item()
            print("TWO_KERNEL_ARRIVAL_PROBE " + repr({
                "rank": self.rank, "generation": int(generation),
                "checked": _seen, "not_ready_on_first_look": _late,
                "late_frac": (_late / _seen) if _seen else 0.0}), flush=True)

    def forward(
        self,
        x_bf16: torch.Tensor,
        wts: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        stream=None,
        slice_output: bool = True,
    ) -> torch.Tensor:
        """Launch Stage1 then Stage2 and return this source rank's BF16 rows."""

        if self._closed:
            raise RuntimeError("MegaMoETileA4W4 is closed")
        if not self.device_generation and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CUDA Graph capture requires device_generation=True")
        # TODO: enforce max_routes_per_token_per_rank in Stage1 while routing
        # metadata is already resident on device.  A host-side topk_ids scan
        # here would add a GPU launch plus synchronization to the two-launch
        # hot path.  Until the device check lands, compact capacities are a
        # trusted-input contract: callers must bound the number of expert IDs
        # owned by any one EP rank for every source token.
        run_tokens = validate_public_stage1_contract(
            x_bf16,
            wts,
            topk_ids,
            hidden=self.model_dim,
            topk=self.topk,
            max_tokens=self.mtpr,
        )
        if run_tokens != self.mtpr:
            raise ValueError(
                "the current fused EP16 protocol requires run_tokens to equal "
                f"the compiled capacity ({self.mtpr}), got {run_tokens}"
            )
        for name, tensor in (
            ("x_bf16", x_bf16),
            ("wts", wts),
            ("topk_ids", topk_ids),
        ):
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}"
                )
        launch_stream = self._flydsl_stream(stream)
        # This Python counter advances only when forward() executes, including
        # capture, not on graph.replay(). In device_generation mode Stage1 uses
        # entry_ticket // stage1_worker_blocks + 1 and publishes epoch_gate;
        # Stage2 loads that device epoch on the same stream. Thus each replay
        # gets fresh flags/parity even though these captured arguments are fixed.
        self._generation += 1
        generation = self._generation
        _solo = int(__import__("os").environ.get("MEGAMOE_TK_SOLO_K1", "0"))
        _seen = getattr(self, "_solo_s1_runs", 0)
        if _solo and _seen >= _solo:
            _skip_s1 = True
        else:
            _skip_s1 = False
            self._solo_s1_runs = _seen + 1
        if not _skip_s1:
            self._launch_stage1(
                x_bf16,
                wts,
                topk_ids,
                run_tokens,
                generation,
                launch_stream,
            )
        self._launch_stage2(run_tokens, generation, launch_stream)
        return self._output[:run_tokens] if slice_output else self._output

    forward_bf16 = forward
    __call__ = forward

    def debug_direct_tile_snapshot(self) -> dict[str, object]:
        """Copy completed-epoch protocol state to the host outside timing."""

        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")
        import hashlib
        import struct


        torch.cuda.synchronize(self.device)
        if self.device_generation:
            self._generation = int(_read_window_u64(
                int(self._runtime.window.local_ptr)
                + self.stage1_layout.offset("epoch_gate"), 1
            )[0])
        generation = int(self._generation)
        parity = generation & 1
        base = int(self._runtime.window.local_ptr)
        s2_base = base + int(self.layout.stage2_offset)

        def s1_ptr(name: str, *, parity_indexed: bool = True) -> int:
            offset = self.stage1_layout.offset(
                name, parity=parity if parity_indexed else None
            )
            return base + int(offset)

        def s2_ptr(name: str, *, parity_indexed: bool = True) -> int:
            offset = self.stage2_layout.offset(
                name, parity=parity if parity_indexed else None
            )
            return s2_base + int(offset)

        tile_alloc = int(_read_window_u32(s1_ptr("tile_alloc"), 1)[0])
        queue_tail = int(_read_window_u32(s1_ptr("h1_queue_tail"), 1)[0])
        compute_done = int(_read_window_u32(s1_ptr("h1_compute_done"), 1)[0])
        queue_jobs = list(
            _read_window_u32(s1_ptr("h1_ready_queue"), queue_tail)
        )
        queue_expected = list(range(queue_tail))
        queue_order_mismatch = sum(
            int(int(actual) != expected)
            for expected, actual in zip(queue_expected, queue_jobs)
        )
        queue_for_validation = (
            sorted(int(job) for job in queue_jobs)
            if getattr(self._stage1, "tile_pipeline", False)
            else queue_jobs
        )
        queue_mismatch = sum(
            int(int(actual) != expected)
            for expected, actual in zip(queue_expected, queue_for_validation)
        )
        early_full_tiles = int(
            _read_window_u32(s1_ptr("h1_early_full_tiles"), 1)[0]
        )
        gmm_started_before_eos = int(
            _read_window_u32(
                s1_ptr("h1_gmm_started_before_all_comm_eos"), 1
            )[0]
        )
        gmm_completed_before_eos = int(
            _read_window_u32(
                s1_ptr("h1_gmm_completed_before_all_comm_eos"), 1
            )[0]
        )
        raw_arrived = list(
            _read_window_u32(
                s1_ptr("tile_row_done"), self.stage1_layout.max_route_tiles
            )
        )
        # Inactive slots can contain an older generation; only [0,tile_alloc)
        # belongs to the completed epoch.
        arrived = [
            int(raw_arrived[index]) if index < tile_alloc else 0
            for index in range(self.stage1_layout.max_route_tiles)
        ]
        expert_count = list(
            _read_window_u32(
                s1_ptr("expert_count"), self.stage1_layout.local_experts
            )
        )
        comm_eos = list(
            _read_window_u64(s1_ptr("comm_eos"), self.gpus_per_node)
        )
        stage1_done = int(_read_window_u64(s2_ptr("stage1_done"), 1)[0])
        stage1_errors = int(
            _read_window_u32(
                s1_ptr("error_count", parity_indexed=False), 1
            )[0]
        )
        stage2_errors = int(
            _read_window_u32(
                s2_ptr("stage2_error_count", parity_indexed=False), 1
            )[0]
        )

        ntiles = self.model_dim // 256
        scoreboard_size = 2 * self.mtpr * ntiles
        node_expected_all = list(
            _read_window_u32(s2_ptr("node_expected"), scoreboard_size)
        )
        node_dest_rank_masks = list(
            _read_window_u32(
                s2_ptr("node_dest_rank_mask"), 2 * self.mtpr
            )
        )
        node_done_all = list(
            _read_window_u32(s2_ptr("node_done"), scoreboard_size)
        )
        token_scoreboard_size = 2 * self.mtpr
        node_token_done_raw = list(
            _read_window_u32(
                s2_ptr("node_token_done"), token_scoreboard_size * 16
            )
        )
        node_token_done_all = [
            int(node_token_done_raw[index * 16])
            for index in range(token_scoreboard_size)
        ]
        node_token_ready_all = list(
            _read_window_u64(
                s2_ptr("node_token_ready"), token_scoreboard_size
            )
        )
        node_partial_ready_all = (
            list(
                _read_window_u64(
                    s2_ptr("node_partial_ready"), token_scoreboard_size
                )
            )
            if (
                self.stage2_layout.include_route_slots
                or self.stage2_layout.include_rank_partials
            )
            else node_token_ready_all
        )
        compact_rank_return = (
            self.stage2_layout.include_rank_partials
            and getattr(self._stage2, "rail_return_schedule", "lockstep")
            == "compact"
        )
        rank_return_tx_slots = (
            list(
                _read_window_u32(
                    s2_ptr("rank_return_tx_slot"), self.mtpr
                )
            )
            if compact_rank_return
            else list(range(self.mtpr))
        )

        # Canonicalize Stage1 rows by the complete packed source key, removing
        # nondeterministic physical-tile allocation order from epoch comparison.
        num_valid = int(_read_window_u32(s1_ptr("num_valid"), 1)[0])
        packed_sources = list(
            _read_window_u32(s1_ptr("tile_row_source"), num_valid)
        )
        rank_local_active_sources: set[int] = set()
        rank_local_pending_nonzero = 0
        rank_local_ready_missing = 0
        rank_local_pending_values: list[int] = []
        tile_pending_nonzero = 0
        tile_group_arrival_mismatch = 0
        tile_rank_ready_missing = 0
        tile_reduce_queue_tail = 0
        tile_node_arrived_nonzero = 0
        node_ready_mask_full_count = 0
        tile_partial_ready_count = 0
        tile_partial_ready_planes = [0, 0]
        rank_return_counts = [0, 0, 0]
        rank_group_total = 0
        rank_group_pending_mismatch = 0
        rank_group_ready_missing = 0
        rank_accumulation_mode = getattr(self._stage2, "rank_accumulation_mode", "atomic")
        push_protocol = {}
        if self.stage2_layout.include_rank_partials:
            source_capacity = self.world_size * self.mtpr
            rank_pending_raw = list(
                _read_window_u32(
                    s2_ptr("rank_token_pending"), source_capacity * 16
                )
            )
            rank_pending = [
                int(rank_pending_raw[index * 16])
                for index in range(source_capacity)
            ]
            rank_ready = list(
                _read_window_u64(s2_ptr("rank_token_ready"), source_capacity)
            )
            rank_local_active_sources = {
                int(packed) & 0x00FFFFFF
                for packed in packed_sources
                if (int(packed) & 0x00FFFFFF) < source_capacity
            }
            rank_local_pending_nonzero = sum(
                int(rank_pending[source] != 0)
                for source in rank_local_active_sources
            )
            # agent: Ã¤Â¿ÂÃ§â€¢â„¢Ã¦Â´Â»Ã¨Â·Æ’ token Ã§Å¡â€ž pending Ã¥Ë†â€ Ã¥Â¸Æ’Ã¯Â¼Å’Ã¥Å’ÂºÃ¥Ë†â€ Ã¢â‚¬Å“Ã¥Â®Å’Ã¥â€¦Â¨Ã¦Å“ÂªÃ¥Ââ€˜Ã¥Â¸Æ’Ã¢â‚¬ÂÃ£â‚¬Â
            # Ã¢â‚¬Å“Ã¥ÂÂªÃ¦Â¼ÂÃ©Æ’Â¨Ã¥Ë†â€  n-groupÃ¢â‚¬ÂÃ¤Â»Â¥Ã¥ÂÅ Ã¨Â®Â¡Ã¦â€¢Â°Ã¤Â¸â€¹Ã¦ÂºÂ¢Ã¯Â¼Å’Ã¤Â¾Â¿Ã¤ÂºÅ½Ã¥Â®Å¡Ã¤Â½Â persistent hangÃ£â‚¬â€š
            rank_local_pending_values = [
                int(rank_pending[source]) for source in rank_local_active_sources
            ]
            rank_local_ready_missing = sum(
                int(int(rank_ready[source]) < generation)
                for source in rank_local_active_sources
            )
            if self.stage2_ready_granularity == "tile":
                ready_groups = self.stage2_layout.ready_group_count
                tile_pending = list(
                    _read_window_u32(
                        s2_ptr("rank_tile_pending"),
                        source_capacity * ready_groups,
                    )
                )
                tile_pending_nonzero = sum(
                    int(tile_pending[source * ready_groups + group] != 0)
                    for source in rank_local_active_sources
                    for group in range(ready_groups)
                )
                tile_group_arrival_mismatch = sum(
                    int(
                        int(tile_pending[source * ready_groups + group])
                        != int(rank_pending[source])
                    )
                    for source in rank_local_active_sources
                    for group in range(ready_groups)
                )
                tile_ready_values = list(
                    _read_window_u64(
                        s2_ptr("rank_tile_ready"),
                        source_capacity * ready_groups,
                    )
                )
                tile_rank_ready_missing = sum(
                    int(
                        int(tile_ready_values[source * ready_groups + group])
                        < generation
                    )
                    for source in rank_local_active_sources
                    for group in range(ready_groups)
                )
                tile_reduce_queue_tail = int(
                    _read_window_u32(s2_ptr("tile_reduce_queue_tail"), 1)[0]
                )
                node_arrived = list(
                    _read_window_u32(
                        s2_ptr("node_tile_arrived"),
                        2 * self.mtpr * ready_groups,
                    )
                )
                tile_node_arrived_nonzero = sum(int(v != 0) for v in node_arrived)
                ready_masks = list(
                    _read_window_u64(
                        s2_ptr("node_ready_mask"), 2 * self.mtpr
                    )
                )
                full_mask = (1 << ready_groups) - 1
                node_ready_mask_full_count = sum(
                    int(int(v) == full_mask) for v in ready_masks
                )
                partial_ready_values = list(
                    _read_window_u64(
                        s2_ptr("node_partial_ready"), 2 * self.mtpr
                    )
                )
                tile_partial_ready_count = sum(
                    int(int(v) >= generation) for v in partial_ready_values
                )
                tile_partial_ready_planes = [
                    sum(
                        int(int(v) >= generation)
                        for v in partial_ready_values[
                            plane * self.mtpr : (plane + 1) * self.mtpr
                        ]
                    )
                    for plane in range(2)
                ]
                rank_return_counts = list(
                    _read_window_u32(s2_ptr("rank_return_count"), 3)
                )
            elif self.stage2_ready_granularity == "group" and rank_accumulation_mode != "reduce_push":
                # Rank-group watermark: a single shared counter per group,
                # no per-source dimension. See publish_rank_group_watermark
                # in stage2.py.
                ready_groups = self.stage2_layout.ready_group_count
                rank_group_total = int(
                    _read_window_u32(s2_ptr("rank_group_total"), 1)[0]
                )
                pending_raw = list(
                    _read_window_u32(
                        s2_ptr("rank_group_pending"), ready_groups * 16
                    )
                )
                rank_group_pending = [
                    int(pending_raw[group * 16]) for group in range(ready_groups)
                ]
                rank_group_pending_mismatch = sum(
                    int(value != rank_group_total)
                    for value in rank_group_pending
                )
                rank_group_ready_values = list(
                    _read_window_u64(s2_ptr("rank_group_ready"), ready_groups)
                )
                rank_group_ready_missing = sum(
                    int(int(value) < generation)
                    for value in rank_group_ready_values
                )
                ready_masks = list(
                    _read_window_u64(
                        s2_ptr("node_ready_mask"), 2 * self.mtpr
                    )
                )
                full_mask = (1 << (self.stage2_layout.hidden // 256)) - 1
                node_ready_mask_full_count = sum(
                    int(int(v) == full_mask) for v in ready_masks
                )
                partial_ready_values = list(
                    _read_window_u64(
                        s2_ptr("node_partial_ready"), 2 * self.mtpr
                    )
                )
                tile_partial_ready_count = sum(
                    int(int(v) >= generation) for v in partial_ready_values
                )
                tile_partial_ready_planes = [
                    sum(
                        int(int(v) >= generation)
                        for v in partial_ready_values[
                            plane * self.mtpr : (plane + 1) * self.mtpr
                        ]
                    )
                    for plane in range(2)
                ]
            if rank_accumulation_mode == "reduce_push":
                # read_window_u32 uses hipMemcpy device-to-host and accepts any
                # device pointer, including this ordinary private allocation.
                # The old rank-group watermarks are unused by this protocol.
                workspace = self._rank_push_workspace
                private_base = (self._output.data_ptr() + workspace.workspace_offset
                                + parity * workspace.parity_stride)
                groups = self.stage2_layout.ready_group_count
                push_protocol = summarize_rank_push_protocol(
                    packed_sources=packed_sources, source_capacity=source_capacity,
                    topk=self.stage1_layout.topk, ready_groups=groups,
                    row_map=_read_window_u32(private_base + workspace.row_map_offset,
                                            workspace.row_map_bytes // 4),
                    route_counts=rank_pending,
                    local_arrivals=_read_window_u32(private_base + workspace.arrival_offset,
                                                   workspace.arrival_bytes // 4),
                    node_rank_masks=node_dest_rank_masks,
                    node_arrivals=_read_window_u32(s2_ptr("rank_push_arrived"),
                                                  2 * self.mtpr * groups),
                )
                ready_masks = _read_window_u64(s2_ptr("node_ready_mask"), 2 * self.mtpr)
                full_mask = (1 << ntiles) - 1
                node_ready_mask_full_count = sum(int(value) == full_mask for value in ready_masks)
                # Exact generation equality also rejects stale/future parity data.
                tile_partial_ready_planes = [
                    sum(int(value) == generation for value in node_partial_ready_all[
                        plane * self.mtpr:(plane + 1) * self.mtpr])
                    for plane in range(2)
                ]
                tile_partial_ready_count = sum(tile_partial_ready_planes)
        input_rows = list(
            _read_window_u32(s1_ptr("tile_row_input"), num_valid)
        )
        weight_bits = list(
            _read_window_u32(s1_ptr("tile_row_weight"), num_valid)
        )
        tile_count = (num_valid + self.stage1_layout.block_m - 1) // self.stage1_layout.block_m
        tile_experts = list(
            _read_window_u32(s1_ptr("tile_expert"), tile_count)
        )
        input_row_bytes = self.model_dim // 2
        h1_row_bytes = self.inter_dim // 2
        h1_scale_bytes = self.inter_dim // 32
        grouped_input = _read_window_bytes(
            s1_ptr("grouped_input_q"),
            self.stage1_layout.source_capacity * input_row_bytes,
        )
        scale_region = self.stage1_layout.region("grouped_input_scale")
        grouped_scale = _read_window_bytes(
            s1_ptr("grouped_input_scale"),
            scale_region.nbytes // self.stage1_layout.parity_depth,
        )
        h1_output = _read_window_bytes(
            s1_ptr("h1_output_q"), num_valid * h1_row_bytes
        )
        h1_scale_region = self.stage1_layout.region("h1_output_scale")
        h1_output_scale = _read_window_bytes(
            s1_ptr("h1_output_scale"),
            h1_scale_region.nbytes // self.stage1_layout.parity_depth,
        )
        rows = []
        low_sources = []
        for row, packed in enumerate(packed_sources):
            low_source = int(packed) & 0x00FFFFFF
            if low_source >= self.world_size * self.mtpr:
                continue
            expert = int(tile_experts[row // self.stage1_layout.block_m])
            rows.append((int(packed), row, expert, int(weight_bits[row])))
            low_sources.append(low_source)
        rows.sort(key=lambda item: item[0])
        valid_input_rows = [int(input_rows[row]) for _, row, _, _ in rows]
        metadata_sha = hashlib.sha256()
        input_sha = hashlib.sha256()
        gathered_input_sha = hashlib.sha256()
        input_map_sha = hashlib.sha256()
        input_scale_sha = hashlib.sha256()
        h1_sha = hashlib.sha256()
        h1_scale_sha = hashlib.sha256()
        per_key = {}
        duplicate_keys = 0
        previous_key = None
        for packed, row, expert, weight_raw in rows:
            if packed == previous_key:
                duplicate_keys += 1
            previous_key = packed
            header = struct.pack("<III", packed, expert, weight_raw)
            actual_row = int(input_rows[row])
            if 0 <= actual_row < self.stage1_layout.source_capacity:
                gathered_input_row = grouped_input[
                    actual_row * input_row_bytes : (actual_row + 1) * input_row_bytes
                ]
            else:
                gathered_input_row = b""
            input_row = gathered_input_row
            # Inverse of the exact BM32 GMM1 A-scale preshuffle:
            # (physical, ku, ikxdl, k_lane, n_lane, im_a).
            physical = row // self.stage1_layout.block_m
            row_in_tile = row % self.stage1_layout.block_m
            scale_bytes_per_row = self.model_dim // 32
            scale_dwords = scale_bytes_per_row // 4
            scale_row = bytearray(scale_bytes_per_row)
            for byte_index in range(scale_bytes_per_row):
                ku = byte_index // 8
                ikxdl = (byte_index % 8) // 4
                k_lane = byte_index % 4
                im_a = row_in_tile // 16
                n_lane = row_in_tile % 16
                dword = (
                    physical * (scale_dwords * self.stage1_layout.block_m)
                    + ku * 64
                    + k_lane * 16
                    + n_lane
                )
                source_byte = dword * 4 + ikxdl * 2 + im_a
                scale_row[byte_index] = grouped_scale[source_byte]
            scale_row = bytes(scale_row)
            h1_row = h1_output[row * h1_row_bytes : (row + 1) * h1_row_bytes]
            output_scale_row = bytearray(self.inter_dim // 32)
            output_chunk_dwords = (self.inter_dim // 256) * 64
            for scale_index in range(self.inter_dim // 32):
                n_block = scale_index // 4
                wave_group = scale_index % 4
                ku = n_block // 2
                ikxdl = n_block % 2
                sub = row_in_tile // 16
                m_lane = row_in_tile % 16
                dword = (
                    physical * output_chunk_dwords
                    + ku * 64
                    + wave_group * 16
                    + m_lane
                )
                source_byte = dword * 4 + ikxdl * 2 + sub
                output_scale_row[scale_index] = h1_output_scale[source_byte]
            output_scale_row = bytes(output_scale_row)
            metadata_sha.update(header)
            input_sha.update(struct.pack("<I", packed))
            input_sha.update(input_row)
            gathered_input_sha.update(struct.pack("<I", packed))
            gathered_input_sha.update(gathered_input_row)
            input_map_sha.update(struct.pack("<II", packed, actual_row))
            input_scale_sha.update(struct.pack("<I", packed))
            input_scale_sha.update(scale_row)
            h1_sha.update(struct.pack("<I", packed))
            h1_sha.update(h1_row)
            h1_scale_sha.update(struct.pack("<I", packed))
            h1_scale_sha.update(output_scale_row)
            per_key[packed] = (
                expert,
                weight_raw,
                hashlib.sha256(input_row).digest(),
                hashlib.sha256(scale_row).digest(),
                hashlib.sha256(h1_row).digest(),
                hashlib.sha256(output_scale_row).digest(),
                row,
                actual_row,
            )
        missing_low_sources = (
            self.world_size * self.mtpr - len(set(low_sources))
        )
        previous = getattr(self, "_debug_previous_canonical", None)
        first_diff = None
        h1_changed_by_expert = [0] * self.epr
        placement_stats = {
            "same_physical_row_total": 0,
            "same_physical_row_h1_changed": 0,
            "moved_row_total": 0,
            "moved_row_h1_changed": 0,
            "same_tile_different_row_total": 0,
            "same_tile_different_row_h1_changed": 0,
            "different_tile_same_row_lane_total": 0,
            "different_tile_same_row_lane_h1_changed": 0,
            "different_tile_different_row_lane_total": 0,
            "different_tile_different_row_lane_h1_changed": 0,
        }
        if previous is not None:
            all_keys = sorted(set(previous) | set(per_key))
            for key in all_keys:
                old = previous.get(key)
                new = per_key.get(key)
                h1_changed = old is not None and new is not None and old[4] != new[4]
                if old is not None and new is not None:
                    old_row, new_row = int(old[6]), int(new[6])
                    if old_row == new_row:
                        placement_stats["same_physical_row_total"] += 1
                        placement_stats["same_physical_row_h1_changed"] += int(h1_changed)
                    else:
                        placement_stats["moved_row_total"] += 1
                        placement_stats["moved_row_h1_changed"] += int(h1_changed)
                        old_tile, new_tile = old_row // 32, new_row // 32
                        old_lane, new_lane = old_row % 32, new_row % 32
                        if old_tile == new_tile:
                            prefix = "same_tile_different_row"
                        elif old_lane == new_lane:
                            prefix = "different_tile_same_row_lane"
                        else:
                            prefix = "different_tile_different_row_lane"
                        placement_stats[f"{prefix}_total"] += 1
                        placement_stats[f"{prefix}_h1_changed"] += int(h1_changed)
                if h1_changed:
                    expert_for_diff = int(new[0])
                    if 0 <= expert_for_diff < self.epr:
                        h1_changed_by_expert[expert_for_diff] += 1
                if first_diff is None and old != new:
                    first_diff = {
                        "packed_source": int(key),
                        "source": int(key) & 0x00FFFFFF,
                        "slot": int(key) >> 24,
                        "old_present": old is not None,
                        "new_present": new is not None,
                        "old_expert": None if old is None else int(old[0]),
                        "new_expert": None if new is None else int(new[0]),
                        "old_grouped_row": None if old is None else int(old[6]),
                        "new_grouped_row": None if new is None else int(new[6]),
                        "old_actual_row": None if old is None else int(old[7]),
                        "new_actual_row": None if new is None else int(new[7]),
                        "metadata_changed": (
                            old is None or new is None or old[:2] != new[:2]
                        ),
                        "input_q_changed": (
                            old is None or new is None or old[2] != new[2]
                        ),
                        "h1_q_changed": (
                            old is None or new is None or old[4] != new[4]
                        ),
                        "h1_scale_changed": (
                            old is None or new is None or old[5] != new[5]
                        ),
                        "input_scale_changed": (
                            old is None or new is None or old[3] != new[3]
                        ),
                    }
        self._debug_previous_canonical = per_key
        canonical_h1 = {
            "num_valid": num_valid,
            "valid_rows": len(rows),
            "duplicate_packed_keys": duplicate_keys,
            "missing_low_sources": missing_low_sources,
            "unique_input_rows": len(set(valid_input_rows)),
            "shared_input_route_rows": len(valid_input_rows)
            - len(set(valid_input_rows)),
            "metadata_sha256": metadata_sha.hexdigest(),
            "grouped_input_q_sha256": input_sha.hexdigest(),
            "gathered_input_q_sha256": gathered_input_sha.hexdigest(),
            "tile_row_input_sha256": input_map_sha.hexdigest(),
            "invalid_input_rows": sum(
                int(
                    int(input_rows[row]) < 0
                    or int(input_rows[row]) >= self.stage1_layout.source_capacity
                )
                for _, row, _, _ in rows
            ),
            "tile_row_input_identity_mismatch": sum(
                int(int(actual) != row)
                for row, actual in enumerate(input_rows)
            ),
            "grouped_input_scale_sha256": input_scale_sha.hexdigest(),
            "h1_output_q_sha256": h1_sha.hexdigest(),
            "h1_output_scale_sha256": h1_scale_sha.hexdigest(),
            "first_diff_vs_previous": first_diff,
            "h1_changed_rows": sum(h1_changed_by_expert),
            "h1_changed_by_expert": h1_changed_by_expert,
            "placement_stats": placement_stats,
        }

        # Optional untimed standalone replay of the exact current grouped H1
        # buffers. This distinguishes the persistent Stage1 wrapper from the
        # MXFP4 GMM1 body/physical-layout contract. It intentionally launches
        # a diagnostic kernel only when explicitly requested.
        if os.environ.get("MEGAMOE_DEBUG_REPLAY_H1", "0") == "1":
            if not hasattr(self, "_debug_h1_replay_launcher"):
                from .gemm1 import (
                    compile_gemm1_a4w4_port,
                )

                self._debug_h1_replay_launcher = compile_gemm1_a4w4_port(
                    BM=32,
                    use_nt=True,
                    inline_quant=False,
                    D_HIDDEN=self.model_dim,
                    D_INTER=self.inter_dim,
                    NE=self.epr,
                    TOPK=self.topk,
                    BN=256,
                    BK=256,
                    interleave=False,
                    act="silu",
                )
                self._debug_h1_replay_q = torch.empty(
                    (self.stage1_layout.max_route_rows, self.inter_dim // 2),
                    dtype=torch.uint8,
                    device=self.device,
                )
                self._debug_h1_replay_scale = torch.empty(
                    self.stage1_layout.max_route_rows * (self.inter_dim // 32),
                    dtype=torch.uint8,
                    device=self.device,
                )
            replay = self._debug_h1_replay_launcher
            replay(
                s1_ptr("grouped_input_q"),
                s1_ptr("grouped_input_scale"),
                self._w1.data_ptr(),
                self._w1_scale.data_ptr(),
                s1_ptr("tile_expert"),
                s1_ptr("num_valid"),
                s1_ptr("tile_row_input"),
                self.stage1_layout.source_capacity,
                tile_alloc * self.stage1_layout.h1_n_blocks,
                self._debug_h1_replay_q.data_ptr(),
                self._debug_h1_replay_scale.data_ptr(),
                self._output.data_ptr(),
                stream=torch.cuda.current_stream(self.device),
            )
            torch.cuda.synchronize(self.device)
            replay_raw = (
                self._debug_h1_replay_q[:num_valid]
                .contiguous()
                .cpu()
                .numpy()
                .tobytes()
            )
            replay_scale_raw = (
                self._debug_h1_replay_scale[: num_valid * h1_scale_bytes]
                .contiguous()
                .cpu()
                .numpy()
                .tobytes()
            )
            replay_sha = hashlib.sha256()
            replay_scale_sha = hashlib.sha256()
            fused_vs_replay_rows = 0
            fused_vs_replay_scale_rows = 0
            replay_per_key = {}
            for packed, row, _expert, _weight_raw in rows:
                replay_row = replay_raw[
                    row * h1_row_bytes : (row + 1) * h1_row_bytes
                ]
                fused_row = h1_output[
                    row * h1_row_bytes : (row + 1) * h1_row_bytes
                ]
                replay_sha.update(struct.pack("<I", packed))
                replay_sha.update(replay_row)
                replay_scale_row = bytearray(self.inter_dim // 32)
                for scale_index in range(self.inter_dim // 32):
                    n_block = scale_index // 4
                    wave_group = scale_index % 4
                    ku = n_block // 2
                    ikxdl = n_block % 2
                    replay_row_in_tile = row % self.stage1_layout.block_m
                    sub = replay_row_in_tile // 16
                    m_lane = replay_row_in_tile % 16
                    dword = (
                        (row // self.stage1_layout.block_m) * output_chunk_dwords
                        + ku * 64
                        + wave_group * 16
                        + m_lane
                    )
                    source_byte = dword * 4 + ikxdl * 2 + sub
                    replay_scale_row[scale_index] = replay_scale_raw[source_byte]
                replay_scale_row = bytes(replay_scale_row)
                replay_scale_sha.update(struct.pack("<I", packed))
                replay_scale_sha.update(replay_scale_row)
                digest = hashlib.sha256(replay_row).digest()
                scale_digest = hashlib.sha256(replay_scale_row).digest()
                replay_per_key[packed] = (digest, scale_digest)
                fused_vs_replay_rows += int(replay_row != fused_row)
                fused_vs_replay_scale_rows += int(
                    scale_digest != per_key[packed][5]
                )
            previous_replay = getattr(self, "_debug_previous_replay_h1", None)
            replay_changed_rows = None
            if previous_replay is not None:
                replay_changed_rows = sum(
                    int(previous_replay.get(key) != replay_per_key.get(key))
                    for key in set(previous_replay) | set(replay_per_key)
                )
            self._debug_previous_replay_h1 = replay_per_key
            canonical_h1["standalone_replay_sha256"] = replay_sha.hexdigest()
            canonical_h1["standalone_replay_scale_sha256"] = (
                replay_scale_sha.hexdigest()
            )
            canonical_h1["standalone_replay_changed_rows"] = replay_changed_rows
            canonical_h1["fused_vs_standalone_changed_rows"] = fused_vs_replay_rows
            canonical_h1["fused_vs_standalone_scale_changed_rows"] = (
                fused_vs_replay_scale_rows
            )
            canonical_h1["fused_vs_standalone_q_byte_mismatches"] = sum(
                int(left != right) for left, right in zip(h1_output, replay_raw)
            )
            canonical_h1["fused_vs_standalone_scale_byte_mismatches"] = sum(
                int(left != right)
                for left, right in zip(
                    h1_output_scale[: num_valid * h1_scale_bytes],
                    replay_scale_raw,
                )
            )
        local_base = self.node * self.mtpr * ntiles
        node_expected = []
        node_done = []
        node_ready = []
        for token in range(self.mtpr):
            start = local_base + token * ntiles
            end = start + ntiles
            expected_slice = node_expected_all[start:end]
            done_slice = node_done_all[start:end]
            node_expected.append(min(expected_slice))
            node_done.append(
                min(expected_slice)
                if self.stage2_layout.include_rank_partials
                else min(done_slice)
            )
            token_index = self.node * self.mtpr + token
            node_ready.append(
                int(
                    (node_dest_rank_masks[token_index] & 0xFF) == 0
                    or node_partial_ready_all[token_index] >= generation
                )
            )

        if compact_rank_return:
            remote_plane = 1 - self.node
            node_not_ready = 0
            for token in range(self.mtpr):
                token_index = remote_plane * self.mtpr + token
                if (node_dest_rank_masks[token_index] & 0xFF) == 0:
                    continue
                slot = int(rank_return_tx_slots[token])
                node_not_ready += int(
                    slot < 0
                    or int(
                        node_partial_ready_all[
                            remote_plane * self.mtpr + slot
                        ]
                    )
                    < generation
                )
        else:
            node_not_ready = sum(
                int(int(value) < generation)
                for value in node_partial_ready_all
            )

        snapshot = {
            # Fields consumed by the untimed harness validator.
            "rank_accumulation_mode": rank_accumulation_mode,
            **push_protocol,
            "comm_role_eos": comm_eos,
            "alloc_count": arrived,
            "tile_arrived": arrived,
            "tile_ready": [int(value > 0) for value in arrived],
            "tail_tile": [int(0 < value < 32) for value in arrived],
            "tail_sealed": [int(0 < value < 32) for value in arrived],
            "node_atomic_expected": node_expected,
            "node_atomic_done": node_done,
            "node_atomic_ready": node_ready,
            "node_ready_granularity": "token",
            "node_token_done_mismatch": sum(
                int(value != ntiles) for value in node_token_done_all
            ) if not self.stage2_layout.include_rank_partials else 0,
            "node_accumulation_mode": getattr(
                self._stage2, "node_accumulation_mode", "direct_atomic"
            ),
            "rank_local_active_tokens": len(rank_local_active_sources),
            "rank_local_pending_nonzero": rank_local_pending_nonzero,
            "rank_local_pending_min": (
                min(rank_local_pending_values) if rank_local_pending_values else 0
            ),
            "rank_local_pending_max": (
                max(rank_local_pending_values) if rank_local_pending_values else 0
            ),
            "rank_local_pending_sum": sum(rank_local_pending_values),
            "rank_local_ready_missing": rank_local_ready_missing,
            "tile_pending_nonzero": tile_pending_nonzero,
            "tile_group_arrival_mismatch": tile_group_arrival_mismatch,
            "tile_rank_ready_missing": tile_rank_ready_missing,
            "tile_reduce_queue_tail": tile_reduce_queue_tail,
            "tile_node_arrived_nonzero": tile_node_arrived_nonzero,
            "node_ready_mask_full_count": node_ready_mask_full_count,
            "tile_partial_ready_count": tile_partial_ready_count,
            "tile_partial_ready_planes": tile_partial_ready_planes,
            "rank_return_counts": rank_return_counts,
            "rank_group_total": rank_group_total,
            "rank_group_pending_mismatch": rank_group_pending_mismatch,
            "rank_group_ready_missing": rank_group_ready_missing,
            "node_expected_uniform_mismatch": sum(
                int(len(set(node_expected_all[start : start + ntiles])) != 1)
                for start in range(0, scoreboard_size, ntiles)
            ),
            "protocol_error_count": [stage1_errors + stage2_errors],
            # Extra diagnostics retained in the benchmark JSON.
            "generation": generation,
            "parity": parity,
            "stage1_done": stage1_done,
            "tile_alloc": tile_alloc,
            "queue_tail": queue_tail,
            "compute_done": compute_done,
            "queue_permutation_mismatch": queue_mismatch,
            "queue_order_identity_mismatch": queue_order_mismatch,
            "queue_sha256": hashlib.sha256(
                b"".join(struct.pack("<I", int(job)) for job in queue_jobs)
            ).hexdigest(),
            "early_full_tiles": early_full_tiles,
            "gmm_jobs_started_before_all_comm_eos": gmm_started_before_eos,
            "gmm_jobs_completed_before_all_comm_eos": gmm_completed_before_eos,
            "expert_count_sum": sum(int(value) for value in expert_count),
            "expert_count": [int(value) for value in expert_count],
            "node_expected_done_mismatch": (
                0
                if self.stage2_layout.include_rank_partials
                else sum(
                    int(int(expected) != int(done))
                    for expected, done in zip(node_expected_all, node_done_all)
                )
            ),
            "node_not_ready": node_not_ready,
            "node_route_store_not_ready": sum(
                int(int(value) < generation)
                for value in node_token_ready_all
            ),
            "stage1_error_count": stage1_errors,
            "stage2_error_count": stage2_errors,
            "stage1_full_fusion": bool(
                getattr(self._stage1, "full_stage1_fusion", False)
            ),
            "tile_pipeline": bool(
                getattr(self._stage1, "tile_pipeline", False)
            ),
            "tile_pipeline_instrument": bool(
                getattr(self._stage1, "tile_pipeline_instrument", False)
            ),
            "canonical_h1": canonical_h1,
        }
        if rank_accumulation_mode == "reduce_push":
            # These legacy completion counters do not describe rank push.
            # Omitting them prevents a generic consumer treating zero as proof.
            for name in ("rank_group_total", "rank_group_pending_mismatch",
                         "rank_group_ready_missing", "rank_local_pending_nonzero",
                         "rank_local_ready_missing"):
                snapshot.pop(name)
        return snapshot

    def debug_device_timeline(self, generation: int | None = None) -> dict[str, object]:
        """Read a completed generation after synchronization, outside timing.

        The optional ring tags every slot with its generation. Reject overwritten
        slots instead of silently attributing the latest fast replay to an older
        slow sample. Without the ring, only the current generation is supported.
        """

        if not (
            getattr(self._stage1, "timeline_instrument", False)
            and getattr(self._stage2, "timeline_instrument", False)
        ):
            raise RuntimeError("device timeline requires instrumented launchers")
        if self._runtime is None:
            raise RuntimeError("CCO runtime is closed")

        torch.cuda.synchronize(self.device)
        if self.device_generation:
            self._generation = int(_read_window_u64(
                int(self._runtime.window.local_ptr)
                + self.stage1_layout.offset("epoch_gate"), 1,
            )[0])
        latest_generation = int(self._generation)
        generation = latest_generation if generation is None else int(generation)
        depth = self.stage2_layout.timeline_history_depth
        if not 1 <= generation <= latest_generation or (
            latest_generation - generation >= (depth or 1)
        ):
            raise ValueError("requested timeline generation is outside the retained window")
        parity = generation & 1
        address = (
            int(self._runtime.window.local_ptr)
            + int(self.layout.stage2_offset)
            + int(self.stage2_layout.offset("timeline", parity=parity))
        )
        gmm_done_address = (
            int(self._runtime.window.local_ptr)
            + int(self.layout.stage2_offset)
            + int(
                self.stage2_layout.offset(
                    "timeline_gmm_worker_done", parity=parity
                )
            )
        )
        if depth:
            slot = generation & (depth - 1)
            base = int(self._runtime.window.local_ptr) + int(self.layout.stage2_offset)
            tag_address = base + self.stage2_layout.offset("timeline_history_generation") + slot * 8
            observed_generation = int(_read_window_u64(tag_address, 1)[0])
            if observed_generation != generation:
                raise RuntimeError(f"stale timeline slot: expected {generation}, got {observed_generation}")
            marker_region = self.stage2_layout.region("timeline_history")
            worker_region = self.stage2_layout.region("timeline_history_gmm_worker_done")
            address = base + marker_region.offset + slot * (marker_region.nbytes // depth)
            gmm_done_address = base + worker_region.offset + slot * (worker_region.nbytes // depth)
        values = list(_read_window_u64(address, len(STAGE2_TIMELINE_FIELDS)))
        if self.worker_blocks > self.stage2_layout.timeline_cta_capacity:
            raise RuntimeError("Stage2 grid exceeds timeline allocation capacity")
        gmm_done_all = list(_read_window_u64(
            gmm_done_address, self.stage2_layout.timeline_cta_capacity,
        ))
        gmm_first = (
            1
            + int(getattr(self._stage2, "rank_reduce_blocks", 0))
            + int(self._stage2.final_combine_blocks)
            + (
                int(self._stage2.node_reduce_blocks)
                if self._stage2.node_accumulation_mode
                in ("route_store", "rank_local")
                else 0
            )
        )
        gmm_first = int(getattr(self._stage2, "gmm_first_block", gmm_first))
        gmm_worker_done = [
            int(value)
            for value in gmm_done_all[gmm_first : int(self.worker_blocks)]
        ]
        ticks = {
            name: int(value)
            for name, value in zip(STAGE2_TIMELINE_FIELDS, values)
        }
        if not gmm_worker_done or any(value <= 0 for value in gmm_worker_done):
            raise RuntimeError("incomplete Stage2 GMM worker timeline")
        if depth and (ticks["stage1_entry"] <= 0 or ticks["stage2_entry"] <= 0
                      or any(value < ticks["stage2_entry"] for value in gmm_worker_done)):
            raise RuntimeError("incomplete or stale timeline history timestamps")
        ticks["stage2_first_gmm_worker_done"] = min(gmm_worker_done)
        ticks["stage2_all_gmm_done"] = max(gmm_worker_done)
        # The instrumented kernel already records every CTA into this array.
        # Decode all roles from that one completed-generation readback so full
        # EP16 diagnostics can distinguish a rank-push tail from RAIL/final
        # completion. These overlapping completion windows are not additive
        # phase durations; no device work or in-replay readback is added here.
        rank_end = 1 + int(getattr(self._stage2, "rank_reduce_blocks", 0))
        node_end = rank_end + int(self._stage2.node_reduce_blocks)
        role_ranges = {
            "rail": (0, 1),
            "rank_push": (1, rank_end),
            "node_reduce": (rank_end, node_end),
            "final": (node_end, gmm_first),
            "gemm": (gmm_first, int(self.worker_blocks)),
        }
        role_completion_ticks = {}
        for role, (begin, end) in role_ranges.items():
            if begin == end:
                continue
            values = [int(value) for value in gmm_done_all[begin:end]]
            if not values or min(values) <= ticks["stage2_entry"]:
                raise RuntimeError(f"incomplete or stale Stage2 {role} CTA timeline")
            role_completion_ticks[role] = {
                "first": min(values), "last": max(values),
                "first_cta": begin, "end_cta": end,
            }
            ticks[f"stage2_{role}_first_done"] = min(values)
            ticks[f"stage2_{role}_last_done"] = max(values)
        ticks["stage2_all_ctas_done"] = max(
            row["last"] for row in role_completion_ticks.values()
        )
        return {
            "generation": generation,
            "ticks": ticks,
            "role_completion_ticks": role_completion_ticks,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            torch.cuda.synchronize(self.device)
            if runtime.owns_communicator:
                # 自建的:context 退出会一并回收 window/memory/dev_comm。
                runtime.context.__exit__(None, None, None)
            else:
                # 调用方的 Communicator 要比我们活得久 —— 只还自己的那块,
                # 和 mega_moe_gfx1250 的 SymmetricArena.close() 一样。
                runtime.window.close()
                runtime.memory.close()

    def __enter__(self) -> "MegaMoETileA4W4":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# Descriptive alias used by the bring-up benchmark factory string.
HierarchicalMegaMoEV2 = MegaMoETileA4W4


__all__ = ["HierarchicalMegaMoEV2", "MegaMoETileA4W4"]
