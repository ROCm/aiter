# SPDX-License-Identifier: MIT
"""Public EP16 A4W4 MegaMoE operator with a two-kernel Stage2.

This class intentionally mirrors :class:`MegaMoEV2` at its public boundary,
but specializes the K3 two-node deployment and accepts only ``quant='a4w4'``.
The constructor allocates the CCO window and compiles Stage1. Stage2 prepares
its workspace and kernels on the first forward; warm up before graph capture.
A BF16 forward launches:

1. MXFP4 input quantization and dispatch-record packing;
2. InterNodeV1 dispatch + GMM1 + activation + A4 requant;
3. weighted GMM2 + node-local push;
4. node reduction + cross-node rail transfer + combine.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
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
from .stage2_abi import Stage2ArenaLayout
from .rank_push_layout import RankPushWorkspace
from .window_view import zero_window as _zero_window

_STAGE1_MODULE = "aiter.ops.flydsl.kernels.megamoe_tile.stage1"
_STAGE1_FACTORY = "compile_megamoe_tile_ep16_stage1"


class _CudaArrayView:
    """最小 __cuda_array_interface__ 包装:在注册窗口里开 torch 视图。"""

    def __init__(self, ptr: int, shape, typestr: str, strides=None):
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": tuple(shape),
            "typestr": typestr,
            "version": 3,
            "strides": None if strides is None else tuple(strides),
        }


def _window_view(ptr: int, shape, typestr: str, strides=None) -> torch.Tensor:
    """strides 以字节计(__cuda_array_interface__ 约定)。"""
    return torch.as_tensor(_CudaArrayView(ptr, shape, typestr, strides), device="cuda")


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
        raise ValueError(
            f"{name} must have a byte-addressable packed layout"
        ) from error


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


class MegaMoETileA4W4:
    """EP16 hierarchical MegaMoE (A4W4) over a two-node CCO window.

    The instance supports one ordered in-flight forward on one CUDA stream, as
    does MegaMoEV2. ``rank`` is the global EP rank; tensor allocation always
    uses the current local CUDA device so ranks 8--15 work on node 1.

    Each forward is input quant + one fused Stage1 kernel + two Stage2 kernels
    (stage2_gemm_push: GEMM2 + node push; stage2_node_combine: rail + combine).
    Shape-dependent tiles come from the stage1/stage2 tune tables.
    """

    quant_mode = "a4w4"
    activation = "silu"
    stage1_kernel_regex = r".*megamoe_tile_ep16_stage1.*"

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int,
        experts: int, topk: int, quant: str, w1: torch.Tensor, w1_scale: torch.Tensor,
        w2: torch.Tensor, w2_scale: torch.Tensor, max_tok_per_rank: int,
        mega_scheme: str = "fixedslot", swiglu_limit: float = 0.0,
        communicator=None,
        max_routes_per_token_per_rank: int | None = None,
        comm_quant: str = "none", activation: str = "silu",
        device_generation: bool = False,
        comm_quant_rail: str | None = None):
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
            max_routes_per_token_per_rank=max_routes_per_token_per_rank,
            comm_quant=comm_quant,
            comm_quant_rail=comm_quant_rail,
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
        # 和 mega_moe_gfx1250 一致:communicator 由调用方建、由调用方销毁。
        # 不传时退回自建(旧行为),这样单测不必自己做 rendezvous。
        self._external_communicator = communicator
        self.comm_quant = str(comm_quant)
        # 跨节点 rail 段单独的通信量化(None = 跟 comm_quant)。node 内 push 段按
        # mtpr 门限走 comm_quant;rail 段量化是跨节点字节减半,门限不同。
        self.comm_quant_rail = None if comm_quant_rail is None else str(comm_quant_rail)
        self.max_routes_per_token_per_rank = (
            self.topk
            if max_routes_per_token_per_rank is None
            else int(max_routes_per_token_per_rank)
        )
        self.gpus_per_node = 8
        self.node = self.rank // self.gpus_per_node
        self.local_rank = self.rank % self.gpus_per_node
        self.peer_node = 1 - self.node
        self.device = torch.device("cuda", torch.cuda.current_device())
        # 两个持久 grid 都是 256 CTA:stage1 的角色划分按 256 设计,stage2 kernel2
        # 用 worker_blocks,kernel1 查不到 gemm2_cu 时也沿用它。
        self.worker_blocks = 256
        self.stage1_worker_blocks = 256
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

        # GMM1 的 tile 切块(tile_group G -> BM=32*G、gmm1_bn)按 shape 查表,
        # 优先级 表 > 解析式默认(见 stage1_tune.py)。G 同时是 arena
        # 布局参数,所以必须在建 layout 之前定下来。
        from .stage1_tune import resolve_stage1_tile as _resolve_s1_tile

        self._s1_tile = _resolve_s1_tile(
            token=self.mtpr * self.topk,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            expert=self.epr,
            topk=self.topk,
        )
        self._s1_layout_kw = dict(
            hidden=self.model_dim,
            inter=self.inter_dim,
            experts=self.experts,
            world_size=self.world_size,
            gpus_per_node=self.gpus_per_node,
            topk=self.topk,
            max_tokens=self.mtpr,
            max_routes_per_token_per_rank=self.max_routes_per_token_per_rank,
            block_m=32,
            tile_group=self._s1_tile["tile_group"],
            dispatch_plan=False,
        )
        self.stage1_layout = Stage1ArenaLayout.create(**self._s1_layout_kw)
        # Stage2 固定走两 kernel(kernel1 GEMM2+push,kernel2 rail+combine)。
        self._two_kernel_stage2 = True
        # push 段跟 MegaMoEv2 的门限走(mega_moe_config.py:102/330/345):
        # mtpr <= P2P_FP8_MIN_MTPR 就不量化。量化的 VALU 代价是每元素固定的,
        # 省下的字节要够多才摊得平 —— 实测 mtpr=512 时 kernel1 +15.5us 而
        # kernel2 只省 5.0us。
        if self.mtpr > COMM_QUANT_PUSH_MIN_MTPR:
            self._k1_p2p_quant_type = _comm_quant_kind(self.comm_quant)
            self._k1_p2p_gate = "above_mtpr_gate"
        else:
            self._k1_p2p_quant_type = "none"
            self._k1_p2p_gate = "below_mtpr_gate"
        self._k2_rail_quant_type = _comm_quant_kind(
            self.comm_quant if self.comm_quant_rail is None else self.comm_quant_rail)
        # kernel2 读 inbox 的格式必须跟着 kernel1 的 push 走,不是跟着 rail。
        self._k2_inbox_quant = self._k1_p2p_quant_type
        # return_chunk_tokens 是随 shape 移动的最优点(CHUNK 决定 rail 包大小),
        # 所以走 per-shape 查表(表 > 内置规则);扫参换一张表(AITER_CONFIG_MEGAMOE_TILE_STAGE2)。
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
        # GEMM2 的 N tile 同样按 shape 查表。
        from .stage2_tune import resolve_gemm2_bn as _resolve_gemm2_bn
        from .stage2_tune import resolve_gemm2_cu as _resolve_gemm2_cu

        self._two_kernel_bn, self._two_kernel_bn_source = _resolve_gemm2_bn(
            _s2_tuned or None)
        # kernel1 的持久 grid 同样按 shape 查表(0 = 沿用 worker_blocks)。
        self._two_kernel_k1_cu, self._two_kernel_k1_cu_source = _resolve_gemm2_cu(
            _s2_tuned or None)
        # rail 队列数在所有实测 shape 上都是 8,是常量(stage2_tune.NUM_QP)。
        from .stage2_tune import NUM_QP as _S2_NUM_QP

        self._two_kernel_qp = _S2_NUM_QP
        # 查不到表时按机制推:rail 有 num_qp 条队列,每条分到一个 chunk 时
        # 发射最均衡,即 chunk = mtpr / num_qp。实测 TPR=512 -> 64、
        # TPR=1024 -> 128 两点精确命中;TPR=128 上整条曲线只有 1.6% 跨度
        # (c16 203.6 / c32 202.3 / c64 205.6),规则给的 16 比最优差 1.3us,
        # 在那个区间怎么取都无所谓。**不是**按包字节数推 —— 实测证伪了:
        # 包字节坐标上 bf16 和 fp8 的最优点不重合(fp8 在 237KB 见底,
        # bf16 到 458KB 还在降),而 token 坐标上两者形状完全一致。
        _chunk_rule = max(4, self.mtpr // max(1, self._two_kernel_qp))
        self._two_kernel_chunk = int(
            _s2_tuned.get("return_chunk_tokens", _chunk_rule))
        self._two_kernel_tune_source = "table" if _s2_tuned else "default"
        self._two_kernel_rail = True
        self._two_kernel_wait_remote = True
        # kernel1 的 grid 已在上面按 shape 查表。g2_spart 要和实际 CU 数配套。
        self._two_kernel_k1_spart = 402
        # 0 = 跟随 cu_num。单节点 179us 那次是 cu_num=256 / persist_cu=240。
        self._two_kernel_k1_pcu = 0
        # 流序只保证**本 rank** 的 kernel1 先于本 rank 的 kernel2,对
        # "别的 rank 推给我的 payload 到了没有"一无所知 —— 那条路上没有任何
        # 同步。所以这个开关不是"将来并发时才有意义",它补的是一个现在就存在
        # 的洞;今天关着也能过精度,靠的是 kernel2 比 kernel1 长 ~7x 的余量。
        # 打开时 kernel1 发 per-(token, slot) 标志,kernel2 归约前逐 token 等。
        self._two_kernel_arrival = True
        # Stage1 emits GMM1 output in expert-major tile order so GEMM2
        # keeps one expert's weights resident across consecutive
        # m-blocks.  The fused Stage2 reads h1_output through the
        # source-keyed tile_expert, so it must not see this layout.
        self._two_kernel_expert_major = True
        self.stage2_layout = Stage2ArenaLayout.create(
            hidden=self.model_dim,
            topk=self.topk,
            max_tokens=self.mtpr,
            world_size=self.world_size,
            gpus_per_node=self.gpus_per_node,
            # 区域集合与旧单 kernel stage2 的 rank_local/reduce_push 配置一致,
            # 窗口排布逐字节不变(加减没人读写的区域曾让 relL2 变化,原因未查清)。
            # 两 kernel 路径用 plane_slot_inbox,其余区域只是占位。
            include_route_slots=False,
            include_rank_partials=True,
            include_staged_reduce=False,
            include_staged_ring=False,
            include_rank_push=True,
            rail_quant_type="none",
            ready_granularity="group",
            ready_group_tiles=2,
            timeline_history_depth=0,
            include_plane_slots=True,
        )
        # One physical registered window, two non-overlapping logical ABIs.
        # Stage1 writes Stage2 metadata directly through stage2_base; there is
        # no host copy or bridge launch between the two kernels.
        # stage2 嵌进 stage1 的 RDMA 前缀之后:rail 目标全部落在窗口前部,不随路由上限增长。
        self.stage1_layout = Stage1ArenaLayout.create(
            **self._s1_layout_kw, embed_stage2_bytes=int(self.stage2_layout.total_bytes)
        )
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
                    max_route_rows=self._rank_push_route_rows(),
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
            # stage1 入口形式与 MegaMoEv2 一致:k1 不做量化,只吃 fp4+e8m0。
            # k1 的 T0 在入口直接从注册窗口 RDMA 发送 rail record
            # [q | scale | ids | weights]。发送源固定在 dispatch_staging 的
            # parity-0 平面(graph 捕获的是固定指针,parity 由 device epoch 定,
            # host 不知道);本次 PUT 在 k1 末尾就被回收,下一次量化在其后。
            # bf16 输入由 forward 先发射一次 rail_record_quant,量化直接写进窗口。
            from ..mega_moe.quant import BLOCK as _QBLOCK

            self._s1_quant_grid = (
                self.mtpr * (self.model_dim // 32) + _QBLOCK - 1
            ) // _QBLOCK
            # 量化直接写成按 token 连续的 rail record [q|scale|ids|weights|pad],
            # k1 每个 QP 整段一个 PUT。record 基址作为 x_q 传给 k1。
            from .rail_record_quant import get_rail_record_quant

            _hid, _mt, _tk = self.model_dim, self.mtpr, self.topk
            self._rail_rec_quant = get_rail_record_quant(_hid, _tk, 0)
            _rec, _oq, _os, _oi, _ow = self._rail_rec_quant.layout
            _base = int(self._runtime.window.local_ptr) + int(
                self.stage1_layout.offset("dispatch_staging", parity=0)
            )
            self._s1_quant_x = _window_view(
                _base + _oq, (_mt, _hid // 2), "|u1", (_rec, 1)
            )
            self._s1_quant_scale = _window_view(
                _base + _os, (_mt, _hid // 32), "|u1", (_rec, 1)
            )
            self._s1_rec_ids = _window_view(
                _base + _oi, (_mt, _tk), "<i4", (_rec, 4)
            )
            self._s1_rec_weights = _window_view(
                _base + _ow, (_mt, _tk), "<f4", (_rec, 4)
            )
            self._stage1 = self._compile_stage1()
            self._validate_launcher_contracts()
            self.stage1_kernel_name = getattr(
                self._stage1, "kernel_name", self.stage1_kernel_regex
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
        max_routes_per_token_per_rank: int | None = None,
        comm_quant: str = "none",
        comm_quant_rail: str | None = None,
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
        if comm_quant_rail is not None:
            _comm_quant_kind(comm_quant_rail)
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
        # 0.0 = 不 clamp;>0 只对 silu 有意义(DSV4:gate<=L, -L<=up<=L)。
        if float(swiglu_limit) != 0.0 and (
            str(activation) != "silu" or not float(swiglu_limit) > 0.0
        ):
            raise ValueError(
                "swiglu_limit must be 0.0, or positive with silu"
            )

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
        from mori.cco import Communicator, UniqueId

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

    def _rank_push_route_rows(self) -> int:
        """私有 rank-push route payload 的行容量。two-kernel 路径不发射单 kernel stage2,payload 用不到,
        只留最小值(输出切片不受影响);否则按 stage1 的 max_route_rows。"""
        return 32 if self._two_kernel_stage2 else int(self.stage1_layout.max_route_rows)

    def _rdma_prefix_end(self) -> int:
        """窗口里被 rail(GDA RDMA)读写的 region 的最大结束偏移(窗口坐标)。"""
        s1 = ("dispatch_staging", "remote_dispatch_rx", "remote_chunk_ready", "remote_chunk_credit",
              "sparse_remote_qp_ready", "sparse_remote_credit", "plan_local_hist", "rail_count_inbox",
              "rail_count_inbox_ready")
        s2 = ("node_accumulator", "remote_partial_rx", "return_group_ready", "return_consumed")
        end = 0
        for r in self.stage1_layout.regions:
            if r.name in s1:
                end = max(end, r.offset + r.nbytes)
        for r in self.stage2_layout.regions:
            if r.name in s2:
                end = max(end, int(self.layout.stage2_offset) + r.offset + r.nbytes)
        return end

    def _build_cco_runtime(self, communicator, *, context, owns,
                           per_rank_vmm=0) -> _CcoRuntime:
        from mori.cco import CCODevCommRequirements, GDA_CONNECTION_RAIL

        try:
            memory = communicator.alloc_mem(self.layout.total_bytes)
            # 只给窗口前部的 RDMA 前缀注册(RDMA MR);节点内 LSA 走 alloc_mem 的 P2P 平坦 VA 映射,覆盖整块分配。
            # 整窗注册在 TPR4096 直接 ENOMEM(7.7GB),且单 MR 超 ~1GiB 后 rail 写不落地(10-08 TPR1024 random 挂死)。
            reg_bytes = min(memory.size, _align_up(self._rdma_prefix_end(), 64 * 1024 * 1024))
            window = communicator.register_window(memory.ptr, reg_bytes)
            requirements = CCODevCommRequirements()
            requirements.gda_connection_type = GDA_CONNECTION_RAIL
            if self.stage1_layout.num_qp != self.stage2_layout.num_qp:
                raise AssertionError("Stage1/Stage2 must use the same CCO QP count")
            _ctx_base = max(
                self.stage1_layout.num_qp,
                self._two_kernel_qp if self._two_kernel_stage2 else 0,
            )
            requirements.gda_context_count = _ctx_base
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
        # 随 shape 变的三项来自 tune 表(见 stage1_tune.py);split_local 选 GMM1 的两条路径,
        # 各自的配套开关在 stage1 里由它推出。其余开关是实测最优的固定配置,已写死在 stage1 里。
        return factory(
            self.stage1_layout,
            self.stage2_layout,
            rank=self.rank,
            stage2_window_offset=self.layout.stage2_offset,
            worker_blocks=self.stage1_worker_blocks,
            waves_per_eu_hint=2,
            fanout_shards=int(self._s1_tile["fanout_shards"]),
            compute_first=self._s1_tile["compute_first"],
            split_local=bool(self._s1_tile["split_local"]),
            gmm1_bn=self._s1_tile["gmm1_bn"],
            activation=self.activation,
            device_generation=self.device_generation,
            swiglu_limit=(self.swiglu_limit or None),
        )

    def _validate_launcher_contracts(self) -> None:
        launcher, pattern = self._stage1, self.stage1_kernel_regex
        if getattr(launcher, "single_gpu_launch", None) is not True:
            raise RuntimeError("Stage1 launcher must declare single_gpu_launch=True")
        kernel_name = getattr(launcher, "kernel_name", "")
        if not kernel_name or re.fullmatch(pattern, kernel_name) is None:
            raise RuntimeError(
                f"Stage1 kernel_name={kernel_name!r} does not match {pattern!r}"
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
        _args = (
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
        )
        self._stage1(*_args, stream=stream)

    def _launch_stage2(
        self,
        run_tokens: int,
        generation: int,
        stream,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("CCO runtime is closed")
        self._launch_two_kernel_stage2(run_tokens, generation, stream)

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
        self._k1_fill = peer_table_launcher(self.gpus_per_node, self.world_size)
        # 两个 parity 的 peer 表连着放,kernel1 按 generation & 1 取 [parity * npes + pe]。
        self._k1_table2 = _torch.zeros(
            2 * self.world_size, dtype=_torch.int64, device=dev)
        self._k1_table2_filled = False
        self._k2_scratch = _torch.zeros(1024, dtype=_torch.int32, device=dev)   # 256 槽位 + [2][max_chunks] 的 chunk 完成计数
        # kernel2 的 timeline 参数指向一块 GM(不开打点时也要合法指针)。
        from .stage2_node_combine import TIMELINE_WORDS as _TLW
        self._k2_timeline = _torch.zeros(_TLW, dtype=_torch.int64, device=dev)
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
        # generation/parity 若按 host 整数传,CUDA graph capture 后就被冻结:
        # stage1 每次 replay 在 device 上前进,stage2 却一直用 capture 时的
        # parity,偶数代读到上一代的 stage1 输出。所以 host 一律给 parity 0 的
        # 基址,两个 kernel 自己从 epoch_gate 读 generation 并选平面。
        parity = 0
        epoch_off = int(arena.stage1.offset("epoch_gate"))
        # `stream` 已经过 _flydsl_stream,是 fx.Stream;再包一层会炸。
        s_fx = stream

        _k1_cu = self._two_kernel_k1_cu or self.worker_blocks

        # ---- kernel1: GEMM2 + push 到各 peer 的 plane_slot_inbox ----------
        if not self._k1_table2_filled:
            for _p in (0, 1):
                self._k1_fill(
                    fx.Int64(window.handle),
                    fx.Int64(self._k1_table2.data_ptr() + _p * self.world_size * 8),
                    fx.Int64(plane_slot_offset(arena, _p)), s_fx)
            self._k1_table2_filled = True
        _k1_args = build_kernel1_args(
                window=window, arena=arena, parity=parity,
                tensors={"bq": self._w2, "bs": self._w2_scale},
                trb=self._k1_trb, p2p_table=self._k1_table2,
                zero_i32=self._k1_zero,
                expert_major=True)
        _k1_pos = (arena.stage1.max_route_rows, int(self.inter_dim),
                   int(self.model_dim))
        _rinv = arena.stage1.region("tile_src_of_dst")
        run_mega_moe_stage2(
            *_k1_args, *_k1_pos, s_fx,
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
            ep16_arrival_publish=True,
            arrival_delta=arrival_delta(arena, parity),
            arrival_count=int(self._k1_arrival_count.data_ptr()),
            generation=int(window.local_ptr) + epoch_off,
            device_epoch=True,
            epoch_planes=tuple(
                arena.stage1.region(_n).nbytes // 2
                for _n in ("h1_output_q", "h1_output_scale", "tile_expert_sorted",
                           "num_valid", "tile_row_source_sorted", "tile_row_weight_sorted")
            ),
            arrival_delta_step=arrival_delta(arena, 1) - arrival_delta(arena, 0),
            # h1 按物理 tile 存放;stage2 经 tile_src_of_dst 间接读 A(与 stage1 的 h1_phys 成对)。
            a_tile_map=int(window.local_ptr) + _rinv.offset + parity * (_rinv.nbytes // 2),
            a_tile_map_plane=_rinv.nbytes // 2,
        )

        # ---- kernel2: node 内归约 -> rail 发送 -> node 间合并 -------------
        def s2_off(name):
            region = s2.region(name)
            return (int(arena.stage2_offset) + region.offset
                    + parity * (region.nbytes // s2.parity_depth))

        def s2_bytes(name):
            return s2.region(name).nbytes // s2.parity_depth

        run_stage2_node_combine(
            int(runtime.dev_comm.ptr), int(window.handle), int(window.local_ptr),
            int(self._output.data_ptr()), int(self._k2_scratch.data_ptr()),
            int(generation), int(run_tokens),
            int(self.worker_blocks), s_fx,
            arg_timeline=int(self._k2_timeline.data_ptr()),
            hidden=self.model_dim, max_tokens=self.mtpr, topk=self.topk,
            rank=self.rank, gpus_per_node=self.gpus_per_node,
            num_qp=self._two_kernel_qp,
            return_chunk_tokens=self._two_kernel_chunk,
            threads=256, enable_rail=True,
            wait_remote=True,
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
            arrival_wait=True,
            s2_window_off=int(arena.stage2_offset),
            epoch_off=epoch_off,
            parity_planes=tuple(
                s2_bytes(_n)
                for _n in ("plane_slot_inbox", "node_dest_slot_mask",
                           "node_accumulator", "remote_partial_rx",
                           "node_partial_ready", "return_group_ready",
                           "return_consumed", "plane_slot_arrived")),
        )

    def forward(
        self,
        x_bf16: torch.Tensor,
        wts: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        x_scale: torch.Tensor | None = None,
        stream=None,
        slice_output: bool = True,
    ) -> torch.Tensor:
        """Launch Stage1 then Stage2 and return this source rank's BF16 rows.

        ``x_bf16`` is either BF16 ``[tokens, hidden]`` (quantized here by a
        separate MXFP4 kernel) or packed FP4 ``[tokens, hidden // 2]`` with
        ``x_scale`` E8M0 ``[tokens, hidden // 32]``, which goes straight to
        dispatch.
        """

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
        input_is_fp4 = x_bf16.dtype in (
            torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8)
        )
        if input_is_fp4:
            if x_scale is None:
                raise ValueError("FP4 input requires x_scale")
            run_tokens = int(x_bf16.shape[0])
            if (
                x_bf16.ndim != 2
                or x_bf16.shape[1] != self.model_dim // 2
                or not x_bf16.is_contiguous()
            ):
                raise ValueError(
                    f"FP4 x must be contiguous [tokens, {self.model_dim // 2}]"
                )
            if (
                tuple(x_scale.shape) != (run_tokens, self.model_dim // 32)
                or x_scale.dtype != torch.uint8
                or not x_scale.is_contiguous()
            ):
                raise ValueError(
                    "x_scale must be contiguous uint8 E8M0 "
                    f"[{run_tokens}, {self.model_dim // 32}]"
                )
            validate_public_stage1_contract(
                torch.empty(
                    (run_tokens, self.model_dim),
                    dtype=torch.bfloat16,
                    device="meta",
                ),
                wts,
                topk_ids,
                hidden=self.model_dim,
                topk=self.topk,
                max_tokens=self.mtpr,
            )
        else:
            if x_bf16.dtype != torch.bfloat16:
                raise ValueError(
                    f"x must be bfloat16 or packed FP4, got {x_bf16.dtype}; "
                    "FP8 activations are not supported by the A4W4 GMM1"
                )
            if x_scale is not None:
                raise ValueError("x_scale is only valid with FP4 input")
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
        x_in, s_in = x_bf16, x_scale
        if topk_ids.dtype != torch.int32 or wts.dtype != torch.float32:
            raise ValueError("rail record expects int32 topk_ids and fp32 weights")
        if input_is_fp4:
            # 已量化输入(诊断 harness):按 record 跨度拷进窗口,必须同一条流。
            if stream is not None and not isinstance(stream, torch.cuda.Stream):
                raise ValueError("fp4 input needs stream=None or a torch.cuda.Stream")
            with torch.cuda.stream(
                stream if stream is not None else torch.cuda.current_stream()
            ):
                self._s1_quant_x[:run_tokens].copy_(x_bf16.view(torch.uint8))
                self._s1_quant_scale[:run_tokens].copy_(x_scale)
                self._s1_rec_ids[:run_tokens].copy_(topk_ids)
                self._s1_rec_weights[:run_tokens].copy_(wts)
        else:
            # 一次发射同时完成量化和 record 打包(替换原 quant,kernel 数不变)。
            self._rail_rec_quant(
                x_bf16.data_ptr(),
                topk_ids.data_ptr(),
                wts.data_ptr(),
                self._s1_quant_x.data_ptr(),
                run_tokens,
                self._s1_quant_grid,
                stream=launch_stream,
            )
        x_in, s_in = self._s1_quant_x, self._s1_quant_scale
        self._launch_stage1(
            x_in,
            wts,
            topk_ids,
            run_tokens,
            generation,
            launch_stream,
            input_scale=s_in,
        )
        self._launch_stage2(run_tokens, generation, launch_stream)
        return self._output[:run_tokens] if slice_output else self._output

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
                # 调用方的 Communicator 要比我们活得久 —— 只还自己建的
                # dev_comm/window/memory,和 mega_moe_gfx1250 的 SymmetricArena.close() 一样。
                runtime.dev_comm.close()
                runtime.window.close()
                runtime.memory.close()

    def __enter__(self) -> "MegaMoETileA4W4":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["MegaMoETileA4W4"]
