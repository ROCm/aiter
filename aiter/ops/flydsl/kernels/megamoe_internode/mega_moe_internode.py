# SPDX-License-Identifier: MIT
"""MegaMoE across two MI355 nodes (EP16, A4W4 or A8W4).

The public forward methods take BF16 or prequantized input, FP32 routing
weights and INT32 expert IDs. Each call uses exactly ``max_tok_per_rank``
rows; callers pad unused rows with zero input/weights and expert ID -1,
then slice the returned BF16 output to their active row count.

A forward quantizes input into rail records, then launches Stage1
(dispatch, GEMM1, activation and requantization) and fused Stage2
(GEMM2, node reduction, rail transfer and combine) on one stream. Optional
single-kernel and separate GEMM2/combine paths share the device routines.
Launchers are explicitly compiled once and cached per operator instance.
Device generation/parity state supports CUDA graph replay.

``layout.py`` defines the CCO window shared by both stages; ``gemm.py``
contains GEMM bodies, ``communication.py`` provides device communication,
and ``mega_moe_internode_config.py`` resolves per-shape tune parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import torch
import torch.distributed as dist
from flydsl._mlir import ir
from flydsl.utils import env
from typing_extensions import Self

from ..mega_moe.quant import BLOCK as _QUANT_BLOCK
from ..mega_moe.quant import per_1x32_mx_quant
from .layout import (
    MAX_FUSED_TOKENS_PER_RANK,
    Stage1ArenaLayout,
    Stage2ArenaLayout,
    TwoKernelArenaLayout,
    validate_fused_weight_addressing,
)
from .mega_moe_internode_config import (
    NUM_QP,
    STAGE2_SDMA_FRAC_Q64,
    STAGE2_SDMA_PUT_BYTES,
    STAGE2_SDMA_RAMP_CHUNKS,
    STAGE2_SDMA_REMOTE_ONLY,
    STAGE2_SDMA_SEND_ROWS_PER_ROUTE,
    STAGE2_TOKEN_READY,
    lookup_stage2_tune,
    resolve_gemm2_bm,
    resolve_gemm2_bn,
    resolve_gemm2_cu,
    resolve_gemm2_dynamic,
    resolve_sdma_chunk_tiles,
    resolve_single_kernel,
    resolve_stage1_tile,
    resolve_stage1_waves,
    resolve_stage2_fused,
    resolve_stage2_sdma,
    resolve_stage2_waves,
    single_kernel_possible,
)
from .quant import get_rail_record_quant
from .single_kernel import compile_single_kernel
from .stage1 import WORKER_BLOCKS, compile_stage1
from .stage2 import (
    BM,
    K1_CLAIM_WORDS,
    K1_CTAS_PER_CU,
    SCRATCH_WORDS,
    SDMA_PLAN_WORDS_MAX,
    compile_mega_moe_stage2,
    compile_stage2_fused,
    compile_stage2_node_combine,
    peer_table_launcher,
)

__all__ = ["MegaMoEInternode", "build_window_layout"]

logger = logging.getLogger(__name__)

WORLD_SIZE = 16
GPUS_PER_NODE = 8
# Stage1 tiles are 32 rows; GMM1 groups tile_group of them, GEMM2 reads them
# one at a time.
TILE_M = 32

_COMM_QUANT = {"none": "none", "fp8": "fp8_blockwise_1x32"}
# Largest single NIC registration known to work (TPR 4096 DSv4 registers 0.61 GiB).
_MAX_RDMA_MR_BYTES = 1 << 30
_FP4_DTYPES = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))

# Window regions read or written by the rail (GDA RDMA).  Only this prefix of
# the window is registered with the NIC (see _build_cco_runtime).
_RDMA_STAGE1_REGIONS = (
    "dispatch_staging",
    "remote_dispatch_rx",
    "remote_chunk_ready",
    "remote_chunk_credit",
)
_RDMA_STAGE2_REGIONS = (
    "node_accumulator",
    "remote_partial_rx",
    "return_group_ready",
    "return_consumed",
    "rail_idx",
)
# Parity-planed regions as stage2 kernel1 / kernel2 expect them, in order.
_K1_EPOCH_PLANES = (
    "h1_output_q",
    "h1_output_scale",
    "tile_expert_sorted",
    "num_valid",
    "tile_row_source",
    "tile_row_weight",
)
_K2_PARITY_PLANES = (
    "plane_slot_inbox",
    "node_dest_slot_mask",
    "node_accumulator",
    "remote_partial_rx",
    "node_partial_ready",
    "return_group_ready",
    "return_consumed",
    "plane_slot_arrived",
)


class _CudaArrayView:
    """Minimal ``__cuda_array_interface__`` to wrap a window VA in a tensor."""

    def __init__(self, ptr: int, shape, typestr: str, strides=None):
        self.__cuda_array_interface__ = {
            "data": (int(ptr), False),
            "shape": tuple(shape),
            "typestr": typestr,
            "version": 3,
            "strides": None if strides is None else tuple(strides),
        }


def _window_view(ptr: int, shape, typestr: str, strides=None) -> torch.Tensor:
    """Zero-copy view of window memory; ``strides`` are in bytes."""
    return torch.as_tensor(_CudaArrayView(ptr, shape, typestr, strides), device="cuda")


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def build_window_layout(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    max_tok_per_rank: int,
    max_routes_per_token_per_rank: int,
    tile_group: int,
    comm_quant: str = "fp8",
    data_send_ctas: int = 0,
    fanout_shards: int = 16,
    quant: str = "a4w4",
) -> TwoKernelArenaLayout:
    """Stage1 and Stage2 layouts sharing one registered window.

    Stage2 is embedded right after Stage1's RDMA prefix, so every rail
    target sits in the front of the window and does not move with the
    route capacity.
    """
    # Rail row width, as k2 sizes it (RECORD_BYTES): fp8 rows carry e8m0 scales.
    rail_record_bytes = (
        model_dim + model_dim // 32 if comm_quant == "fp8" else 2 * model_dim
    )
    stage2 = Stage2ArenaLayout.create(
        hidden=model_dim,
        topk=topk,
        max_tokens=max_tok_per_rank,
        world_size=WORLD_SIZE,
        gpus_per_node=GPUS_PER_NODE,
        rail_record_bytes=rail_record_bytes,
    )
    stage1 = Stage1ArenaLayout.create(
        hidden=model_dim,
        inter=inter_dim,
        experts=experts,
        world_size=WORLD_SIZE,
        gpus_per_node=GPUS_PER_NODE,
        topk=topk,
        max_tokens=max_tok_per_rank,
        max_routes_per_token_per_rank=max_routes_per_token_per_rank,
        activation_dtype="fp8" if quant == "a8w4" else "fp4",
        block_m=TILE_M,
        tile_group=tile_group,
        embed_stage2_bytes=int(stage2.total_bytes),
        # Each destination partitions T records into at most ceil(T/NB)+S
        # batches per phase. This conservative bound covers both fan1/fan2.
        send_queue_capacity=(
            2
            * GPUS_PER_NODE
            * (
                (
                    max_tok_per_rank
                    + max(8, min(512 // topk, TILE_M * tile_group) // 8 * 8)
                    - 1
                )
                // max(8, min(512 // topk, TILE_M * tile_group) // 8 * 8)
                + fanout_shards
            )
            if data_send_ctas
            else 0
        ),
    )
    return TwoKernelArenaLayout.compose(stage1, stage2)


@dataclass(frozen=True)
class _CcoRuntime:
    context: Any
    communicator: Any
    memory: Any
    window: Any
    dev_comm: Any
    # False when the caller supplied the Communicator: then this operator owns
    # only its window/memory/dev_comm and must not destroy the communicator.
    owns_communicator: bool


class MegaMoEInternode:
    """Fused dispatch, GEMM1, GEMM2 and combine over two nodes (EP16).

    One in-flight forward per instance, on one stream, as with MegaMoEV2.
    ``rank`` is the global EP rank; tensors are allocated on the current CUDA
    device.  ``torch.distributed`` must be initialized (it carries the CCO
    unique id when no ``communicator`` is given).

    Extra keyword arguments beyond MegaMoEV2's:

    * ``activation``: ``"silu"`` (``swiglu_limit`` > 0 clamps it) or
      ``"situv2"``;
    * ``comm_quant``: ``"fp8"`` (default) sends the cross-node rail payload
      as FP8 with 1x32 E8M0 scales, ``"none"`` sends BF16;
    * ``communicator``: a ``mori.cco`` Communicator owned by the caller
      (default: create one per instance);
    * ``max_routes_per_token_per_rank``: upper bound on the experts one
      token routes to within one EP rank (default ``topk``); it sizes the
      expert-tile arena.  Callers must respect it, it is not checked.

    ``mega_scheme`` and ``fanout_masks`` are accepted for MegaMoEV2
    compatibility; the internode dispatch has a single scheme and no fanout
    masks.
    """

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int,
        topk: int, quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor,
        w2_scale: torch.Tensor, max_tok_per_rank: int, mega_scheme: str = "fixedslot",
        swiglu_limit: float = 0.0, fanout_masks: tuple[int, ...] = (), activation: str = "silu",
        comm_quant: str = "fp8", communicator=None, max_routes_per_token_per_rank: int | None = None):
    # fmt: on
        del mega_scheme
        route_cap = topk if max_routes_per_token_per_rank is None else max_routes_per_token_per_rank
        self._validate_args(
            rank=rank, world_size=world_size, model_dim=model_dim, inter_dim=inter_dim,
            experts=experts, topk=topk, quant=quant, max_tok_per_rank=max_tok_per_rank,
            swiglu_limit=swiglu_limit, fanout_masks=fanout_masks, activation=activation,
            comm_quant=comm_quant, route_cap=route_cap,
        )
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before MegaMoEInternode")
        if dist.get_world_size() != int(world_size) or dist.get_rank() != int(rank):
            raise ValueError("rank/world_size must match the initialized process group")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = self.experts // self.world_size
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.swiglu_limit = float(swiglu_limit)
        self.activation = str(activation)
        self.quant = str(quant).lower()
        self._activation_dtype = "fp8" if self.quant == "a8w4" else "fp4"
        self.comm_quant = str(comm_quant)
        self.max_routes_per_token_per_rank = int(route_cap)
        self.device = torch.device("cuda", torch.cuda.current_device())
        device_cus = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._device_cus = device_cus
        # The software grid barriers of the persistent kernels need every CTA
        # resident; one CTA per CU is the conservative prerequisite.
        if device_cus < WORKER_BLOCKS:
            raise RuntimeError(f"persistent kernels need {WORKER_BLOCKS} CUs, got {device_cus}")

        for name, tensor in (("w1", w1), ("w1_scale", w1_scale), ("w2", w2), ("w2_scale", w2_scale)):
            if tensor.device != self.device:
                raise ValueError(f"{name} is on {tensor.device}, expected {self.device}")
        self._check_weight_bytes(w1, w1_scale, w2, w2_scale)
        self._w1 = _as_bytes(w1)
        self._w1_scale = _as_bytes(w1_scale)
        self._w2 = _as_bytes(w2)
        self._w2_scale = _as_bytes(w2_scale)

        # Tune-table lookups are keyed by GEMM rows = mtpr * topk, as in the
        # fused_moe tune tables.
        rows = self.mtpr * self.topk
        self._s1_tile = resolve_stage1_tile(
            token=rows, model_dim=self.model_dim, inter_dim=self.inter_dim,
            expert=self.epr, topk=self.topk,
        )
        s2_tuned = self._lookup_stage2_tune(rows)
        self._gemm2_bn, _ = resolve_gemm2_bn(s2_tuned or None)
        gemm2_cu, _ = resolve_gemm2_cu(s2_tuned or None)
        self._gemm2_cu = gemm2_cu or WORKER_BLOCKS
        self._stage2_fused = resolve_stage2_fused(self.mtpr)
        remote_first = bool(self._stage2_fused and self._s1_tile["split_local"])
        # SDMA needs the remote-first fused kernel (Stage1's local/remote
        # split); it is dropped at runtime setup if MORI gives no SDMA queue.
        self._stage2_sdma = bool(
            resolve_stage2_sdma(self.mtpr, s2_tuned or None)[0] and remote_first)
        # The single kernel needs the fused Stage2 body: below the fused-TPR
        # threshold it switches it on (remote-first when Stage1 splits local).
        if not self._stage2_fused and single_kernel_possible(
                self._s1_tile["tile_group"], self._stage2_sdma,
                self.model_dim // self._gemm2_bn):
            self._stage2_fused = True
            remote_first = bool(self._s1_tile["split_local"])
        # GEMM2 M tile (fused remote-first stage2 only; 32 elsewhere).
        self._s2_tuned = s2_tuned or None
        self._gemm2_bm, _ = resolve_gemm2_bm(
            self._s2_tuned, tile_group=int(self._s1_tile["tile_group"]),
            fused=remote_first, sdma=self._stage2_sdma)
        # Without a table row: one chunk per rail queue is the balanced
        # point (measured TPR 512 -> 64 and TPR 1024 -> 128 exactly).
        # Capped at 512 (the TPR 4096 value): k2 unrolls one ready store per chunk
        # token, so mtpr > 4096 adds chunks instead of widening them.
        self._chunk_tokens = int(s2_tuned.get(
            "return_chunk_tokens", max(4, min(512, self.mtpr // NUM_QP))))

        self.layout = build_window_layout(
            model_dim=self.model_dim, inter_dim=self.inter_dim, experts=self.experts,
            topk=self.topk, max_tok_per_rank=self.mtpr,
            max_routes_per_token_per_rank=self.max_routes_per_token_per_rank,
            tile_group=self._s1_tile["tile_group"], comm_quant=self.comm_quant,
            quant=self.quant,
            data_send_ctas=int(self._s1_tile["data_send_ctas"]),
            fanout_shards=int(self._s1_tile["fanout_shards"]),
        )
        # One NIC MR covers the rail prefix; past ~1 GiB rail writes stop
        # landing (ionic). Refuse here instead of hanging in k2.
        if _align_up(self._rdma_prefix_end(), 64 << 20) > _MAX_RDMA_MR_BYTES:
            raise ValueError(
                f"rail-registered prefix {self._rdma_prefix_end() / 2**30:.2f} GiB exceeds "
                f"{_MAX_RDMA_MR_BYTES / 2**30:.2f} GiB at max_tok_per_rank={self.mtpr}; "
                "use comm_quant='fp8' or a smaller max_tok_per_rank")
        self._runtime: _CcoRuntime | None = None
        self._closed = False
        try:
            self._runtime = self._initialize_cco_runtime(communicator)
            self._output = torch.empty(
                (self.mtpr, self.model_dim), dtype=torch.bfloat16, device=self.device)
            self._init_input_records()
            # One kernel for Stage1 + fused Stage2 when both run 8-wave CTAs;
            # Stage1 then publishes per-group GMM1 completion for GEMM2.
            s1_waves = resolve_stage1_waves(self._s1_tile["tile_group"])
            self._s2_waves = resolve_stage2_waves(
                self._stage2_fused, self._stage2_sdma, self.model_dim // self._gemm2_bn)
            self._single_kernel = resolve_single_kernel(s1_waves, self._s2_waves)
            self._stage1 = compile_stage1(
                self.layout.stage1, self.layout.stage2,
                rank=self.rank,
                stage2_window_offset=self.layout.stage2_offset,
                fanout_shards=int(self._s1_tile["fanout_shards"]),
                data_send_ctas=int(self._s1_tile["data_send_ctas"]),
                split_local=bool(self._s1_tile["split_local"]),
                activation=self.activation,
                swiglu_limit=(self.swiglu_limit or None),
                waves=s1_waves,
                group_ready=self._single_kernel,
            )
            self.stage1_kernel_name = self._stage1.kernel_name
            self._init_stage2()
            self._init_launchers()
            # Constructor-time clears and compiles must be complete everywhere
            # before the first generation can receive remote writes.
            torch.cuda.synchronize(self.device)
            self._runtime.communicator.barrier()
        except Exception:
            self.close()
            raise
    # construction

    @staticmethod
    def _validate_args(*, rank, world_size, model_dim, inter_dim, experts, topk, quant,
                       max_tok_per_rank, swiglu_limit, fanout_masks, activation, comm_quant,
                       route_cap) -> None:
        if str(quant).lower() not in ("a4w4", "a8w4"):
            raise ValueError("MegaMoEInternode supports quant=a4w4 or a8w4")
        if int(world_size) != WORLD_SIZE:
            raise ValueError(f"MegaMoEInternode requires world_size={WORLD_SIZE} (two nodes of {GPUS_PER_NODE})")
        if not 0 <= int(rank) < int(world_size):
            raise ValueError("rank is outside world_size")
        if not 1024 <= int(model_dim) <= 8192 or int(model_dim) % 512:
            raise ValueError("model_dim must be in [1024, 8192] and divisible by 512")
        if int(inter_dim) <= 0 or int(inter_dim) % 256:
            raise ValueError("inter_dim must be positive and divisible by 256")
        if int(experts) <= 0 or int(experts) % int(world_size):
            raise ValueError("experts must be positive and divisible by world_size")
        validate_fused_weight_addressing(
            hidden=model_dim, inter=inter_dim, experts=experts, world_size=world_size)
        if not 1 <= int(topk) <= 16:
            raise ValueError("topk must be in [1, 16]")
        if not 1 <= int(max_tok_per_rank) <= MAX_FUSED_TOKENS_PER_RANK:
            raise ValueError(f"max_tok_per_rank must be in [1, {MAX_FUSED_TOKENS_PER_RANK}]")
        if not 1 <= int(route_cap) <= int(topk):
            raise ValueError("max_routes_per_token_per_rank must be in [1, topk]")
        if str(activation) not in ("silu", "situv2"):
            raise ValueError("activation must be 'silu' or 'situv2'")
        # 0.0 = no clamp; > 0 only applies to silu (DSV4: gate <= L, -L <= up <= L).
        if float(swiglu_limit) != 0.0 and (str(activation) != "silu" or float(swiglu_limit) < 0):
            raise ValueError("swiglu_limit must be 0.0, or positive with silu")
        if tuple(fanout_masks):
            raise ValueError("fanout_masks are not supported by the internode dispatch")
        if str(comm_quant) not in _COMM_QUANT:
            raise ValueError(f"comm_quant must be one of {sorted(_COMM_QUANT)}, got {comm_quant!r}")

    def bind_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        """Point the next forwards at another layer's weights (same layouts as
        the constructor).  One instance serves every shape-identical MoE layer;
        the kernels read the weight addresses as runtime arguments, so a CUDA
        graph captured after this call keeps that layer's addresses."""
        for name, tensor in (("w1", w1), ("w1_scale", w1_scale), ("w2", w2), ("w2_scale", w2_scale)):
            if tensor.device != self.device:
                raise ValueError(f"{name} is on {tensor.device}, expected {self.device}")
        self._check_weight_bytes(w1, w1_scale, w2, w2_scale)
        self._w1 = _as_bytes(w1)
        self._w1_scale = _as_bytes(w1_scale)
        self._w2 = _as_bytes(w2)
        self._w2_scale = _as_bytes(w2_scale)
        self._k1_ptrs["arg_bq"] = self._w2.data_ptr()
        self._k1_ptrs["arg_bscale"] = self._w2_scale.data_ptr()

    def _check_weight_bytes(self, w1, w1_scale, w2, w2_scale) -> None:
        expected = {
            "w1": self.epr * 2 * self.inter_dim * self.model_dim // 2,
            "w1_scale": self.epr * 2 * self.inter_dim * (self.model_dim // 32),
            "w2": self.epr * self.model_dim * self.inter_dim // 2,
            "w2_scale": self.epr * self.model_dim * (self.inter_dim // 32),
        }
        tensors = {"w1": w1, "w1_scale": w1_scale, "w2": w2, "w2_scale": w2_scale}
        bad = [f"{name}={tensors[name].numel() * tensors[name].element_size()} (expected {want})"
               for name, want in expected.items()
               if tensors[name].numel() * tensors[name].element_size() != want]
        if bad:
            raise ValueError(f"invalid A4W4 weight sizes in bytes: {', '.join(bad)}")

    def _lookup_stage2_tune(self, rows: int) -> dict:
        try:
            from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

            tuned = lookup_stage2_tune(
                gfx=get_gfx_runtime(), cu_num=get_cu_num(), token=rows,
                model_dim=self.model_dim, inter_dim=self.inter_dim,
                expert=self.epr, topk=self.topk,
            )
        except Exception:
            # A missing or unreadable table falls back to the built-in rules.
            logger.debug("Stage2 tune lookup failed; using built-in rules", exc_info=True)
            tuned = None
        return tuned or {}

    def _rdma_prefix_end(self) -> int:
        """End offset (window coordinates) of the regions the rail touches."""
        end = 0
        for r in self.layout.stage1.regions:
            if r.name in _RDMA_STAGE1_REGIONS:
                end = max(end, r.offset + r.nbytes)
        for r in self.layout.stage2.regions:
            if r.name in _RDMA_STAGE2_REGIONS:
                end = max(end, self.layout.stage2_offset + r.offset + r.nbytes)
        return end

    def _initialize_cco_runtime(self, communicator) -> _CcoRuntime:
        from mori.cco import Communicator, UniqueId

        if communicator is not None:
            return self._build_cco_runtime(communicator, context=None)
        uid_payload = [bytes(Communicator.get_unique_id()) if self.rank == 0 else None]
        dist.broadcast_object_list(uid_payload, src=0)
        uid = UniqueId.from_bytes(uid_payload[0])
        # Same VMM capacity on every rank, with headroom for CCO mappings.
        per_rank_vmm = max(128 << 20, _align_up(self.layout.total_bytes, 64 << 20))
        context = Communicator.init(self.world_size, self.rank, uid, per_rank_vmm=per_rank_vmm)
        return self._build_cco_runtime(context.__enter__(), context=context)

    def _build_cco_runtime(self, communicator, *, context) -> _CcoRuntime:
        from mori.cco import GDA_CONNECTION_RAIL, CCODevCommRequirements

        try:
            memory = communicator.alloc_mem(self.layout.total_bytes)
            # Only the RDMA prefix is registered with the NIC; node-local LSA
            # goes through alloc_mem's P2P mapping of the whole allocation.
            # Registering the whole window hits ENOMEM at TPR 4096 (7.7 GB),
            # and rail writes stop landing once one MR exceeds ~1 GiB.
            reg_bytes = min(memory.size, _align_up(self._rdma_prefix_end(), 64 << 20))
            window = communicator.register_window(memory.ptr, reg_bytes)
            requirements = CCODevCommRequirements()
            requirements.gda_connection_type = GDA_CONNECTION_RAIL
            if self.layout.stage1.num_qp != self.layout.stage2.num_qp:
                raise AssertionError("Stage1/Stage2 must use the same CCO QP count")
            requirements.gda_context_count = max(self.layout.stage1.num_qp, NUM_QP)
            requirements.gda_signal_count = 0
            requirements.gda_counter_count = 0
            requirements.lsa_barrier_count = 0
            requirements.rail_gda_barrier_count = 0
            requirements.barrier_count = 0
            if self._stage2_sdma:
                requirements.sdma_queue_count = 1
            dev_comm = communicator.create_dev_comm(requirements)
            if self._stage2_sdma and int(getattr(dev_comm, "sdma_num_queue", 0) or 0) <= 0:
                # Puts on a comm without queues move nothing; keep the xGMI push.
                logger.warning("MegaMoEInternode: MORI gave no SDMA queue (MORI_ENABLE_SDMA "
                               "unset?); stage2 keeps the xGMI push")
                self._stage2_sdma = False
                # The default M tile follows the path (xGMI stalls at SDMA's 128).
                self._gemm2_bm, _ = resolve_gemm2_bm(
                    self._s2_tuned, tile_group=int(self._s1_tile["tile_group"]),
                    fused=True, sdma=False)
            _window_view(window.local_ptr, (self.layout.total_bytes,), "|i1").zero_()
            torch.cuda.synchronize(self.device)
            communicator.barrier()
            return _CcoRuntime(context, communicator, memory, window, dev_comm,
                               owns_communicator=context is not None)
        except Exception:
            if context is not None:
                context.__exit__(None, None, None)
            raise

    def _init_input_records(self) -> None:
        """Views of the rail records the input quant writes into the window.

        One record per token, ``[q | scale | topk_ids | weights | pad]``, in
        the parity-0 plane of dispatch_staging; stage1 sends each QP's token
        range with one PUT.  The plane is fixed because a captured graph
        holds fixed pointers, and the PUT is retired before stage1 ends.
        """
        hidden, mtpr, topk = self.model_dim, self.mtpr, self.topk
        self._record_quant = get_rail_record_quant(hidden, topk, self._activation_dtype)
        rec, off_q, off_s, off_i, off_w = self._record_quant.layout
        base = int(self._runtime.window.local_ptr) + int(
            self.layout.stage1.offset("dispatch_staging", parity=0))
        self._rec_q = _window_view(base + off_q, (mtpr, self.layout.stage1.wire.payload_bytes), "|u1", (rec, 1))
        self._rec_scale = _window_view(base + off_s, (mtpr, hidden // 32), "|u1", (rec, 1))
        self._rec_ids = _window_view(base + off_i, (mtpr, topk), "<i4", (rec, 4))
        self._rec_weights = _window_view(base + off_w, (mtpr, topk), "<f4", (rec, 4))
        self._quant_grid = (mtpr * (hidden // 32) + _QUANT_BLOCK - 1) // _QUANT_BLOCK

    def _init_stage2(self) -> None:
        """Constant launch arguments and private buffers of both stage2 kernels.

        All window pointers are parity 0: the kernels load the generation
        from stage1's epoch_gate and add the parity plane themselves, so a
        replayed graph sees the current generation.
        """
        layout, s1, s2 = self.layout, self.layout.stage1, self.layout.stage2
        base = int(self._runtime.window.local_ptr)
        dev = self.device
        self._epoch_off = int(s1.offset("epoch_gate"))

        def s1_addr(name):
            return base + s1.region(name).offset

        def s2_off(name):
            return layout.stage2_offset + s2.region(name).offset

        def s2_plane(name):
            return s2.region(name).nbytes // s2.parity_depth

        # Kernel1 reads stage1's metadata in sorted (expert-major) tile order
        # and finds each tile's A rows through tile_src_of_dst.  trb is the
        # identity sort-block map (one 32-row tile per sort block); one extra
        # entry because total_m_blocks rounds the row count up.
        tiles = (s1.max_route_rows + TILE_M - 1) // TILE_M
        self._k1_trb = torch.arange(tiles + 1, dtype=torch.int32, device=dev) * TILE_M
        # Kernel1's m-block tickets and done counts (see K1_CLAIM_WORDS).
        self._k1_arrival_count = torch.zeros(K1_CLAIM_WORDS, dtype=torch.int32, device=dev)
        # Each peer's plane_slot_inbox base for both parities, back to back.
        self._k1_peer_table = torch.zeros(2 * self.world_size, dtype=torch.int64, device=dev)
        fill = peer_table_launcher(GPUS_PER_NODE, self.world_size)
        import flydsl.expr as fx

        stream = fx.Stream(torch.cuda.current_stream())
        for parity in (0, 1):
            fill(fx.Int64(self._runtime.window.handle),
                 fx.Int64(self._k1_peer_table.data_ptr() + parity * self.world_size * 8),
                 fx.Int64(s2_off("plane_slot_inbox") + parity * s2_plane("plane_slot_inbox")),
                 stream)
        # The arrival word of a row sits at a fixed distance from its inbox
        # row on every peer, so kernel1 reuses the peer base it already has.
        arrival_delta = s2.region("plane_slot_arrived").offset - s2.region("plane_slot_inbox").offset
        arrival_delta_step = s2_plane("plane_slot_arrived") - s2_plane("plane_slot_inbox")
        a_tile_map = s1.region("tile_src_of_dst")
        self._k1_ptrs = {
            "arg_aq": s1_addr("h1_output_q"), "arg_ascale": s1_addr("h1_output_scale"),
            "arg_bq": self._w2.data_ptr(), "arg_bscale": self._w2_scale.data_ptr(),
            "arg_eids": s1_addr("tile_expert_sorted"), "arg_cumsum": s1_addr("num_valid"),
            # Read at the physical arena row (a_tile_map), not a sorted copy.
            "arg_stids": s1_addr("tile_row_source"),
            "arg_sweights": s1_addr("tile_row_weight"),
            "arg_trb": self._k1_trb.data_ptr(), "arg_p2p": self._k1_peer_table.data_ptr(),
            "arg_arrival_count": self._k1_arrival_count.data_ptr(),
            "arg_generation": base + self._epoch_off, "arg_a_tile_map": base + a_tile_map.offset,
            "row_capacity": s1.max_route_rows,
            "local_rank": self.rank % GPUS_PER_NODE,
            "arg_expert_count": s1_addr("expert_count"),
        }
        self._k1_compile = {
            "model_dim": self.model_dim, "inter_dim": self.inter_dim, "experts": self.epr,
            "topk": self.topk, "npes": self.world_size, "max_tok": self.mtpr,
            "a_dtype": self._activation_dtype,
            "recv_cap": self.world_size * self.mtpr,
            "comb_inp_nbytes": 2 * self.mtpr * self.topk * self.model_dim * 2,
            "BN": self._gemm2_bn, "cu_num": self._gemm2_cu, "gpus_per_node": GPUS_PER_NODE,
            "arrival_delta": arrival_delta, "arrival_delta_step": arrival_delta_step,
            "epoch_planes": tuple(s1.region(n).nbytes // 2 for n in _K1_EPOCH_PLANES),
            "a_tile_map_plane": a_tile_map.nbytes // 2,
            # Place a tile group's m-blocks on one XCD only for G=2 (K3 TPR128,
            # DSV4 TPR256: GEMM2 -14~22us).  At G=4 it measured neutral to
            # +12us (K3 TPR512), so G=4 keeps the plain order.
            "m_group": 2 if self._s1_tile["tile_group"] == 2 else 1,
            "dynamic_claim": resolve_gemm2_dynamic(self.mtpr),
            "resident_ctas": self._device_cus * K1_CTAS_PER_CU,
            # Fused stage2 overlaps the rail with GEMM2 by doing the
            # remote-source m-blocks first (needs Stage1's local/remote split).
            "remote_first": bool(self._stage2_fused and self._s1_tile["split_local"]),
            "local_experts": self.epr, "tile_group": int(self._s1_tile["tile_group"]),
            "expert_count_plane": s1.region("expert_count").nbytes // 2,
            "gemm2_bm": self._gemm2_bm,
        }

        if self._stage2_sdma:
            # SDMA stage2 (see compile_mega_moe_stage2): the send buffer, the
            # plan's per-(ticket, dest) row prefixes (+1 end entry), its words
            # and the per-chunk done counts (128 B apart).
            sd_tiles = (s1.max_route_rows + TILE_M - 1) // TILE_M
            sd_rows = int(STAGE2_SDMA_SEND_ROWS_PER_ROUTE * self.mtpr * self.topk)
            sd_ct = resolve_sdma_chunk_tiles(self.mtpr * self.topk, self._gemm2_bm // TILE_M)
            sd_chunks = sd_tiles // sd_ct + 2 + 4 * STAGE2_SDMA_RAMP_CHUNKS
            self._sd_send = torch.empty(sd_rows * self.model_dim * 2, dtype=torch.uint8, device=dev)
            self._sd_tile_base = torch.zeros((sd_tiles + 1) * 8, dtype=torch.int32, device=dev)
            self._sd_plan = torch.zeros(SDMA_PLAN_WORDS_MAX, dtype=torch.int32, device=dev)
            self._sd_cdone = torch.zeros(sd_chunks * 32, dtype=torch.int32, device=dev)
            self._k1_ptrs.update(
                arg_sd_send=self._sd_send.data_ptr(), arg_sd_tb=self._sd_tile_base.data_ptr(),
                arg_sd_plan=self._sd_plan.data_ptr(), arg_sd_cdone=self._sd_cdone.data_ptr())
            self._k1_compile.update(
                sdma=True, sdma_chunk_tiles=sd_ct, sdma_send_rows=sd_rows,
                sdma_frac_q64=STAGE2_SDMA_FRAC_Q64, sdma_put_bytes=STAGE2_SDMA_PUT_BYTES,
                sdma_arrival_woff=s2_off("plane_slot_arrived"),
                sdma_arrival_plane=s2_plane("plane_slot_arrived"), sdma_max_chunks=sd_chunks,
                sdma_ramp_chunks=STAGE2_SDMA_RAMP_CHUNKS,
                sdma_remote_only=STAGE2_SDMA_REMOTE_ONLY)

        self._k2_scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int32, device=dev)
        self._k2_compile = {
            "hidden": self.model_dim, "max_tokens": self.mtpr, "topk": self.topk, "rank": self.rank,
            "gpus_per_node": GPUS_PER_NODE, "num_qp": NUM_QP, "return_chunk_tokens": self._chunk_tokens,
            "rail_quant": _COMM_QUANT[self.comm_quant],
            "inbox_off": s2_off("plane_slot_inbox"), "inbox_bytes": s2_plane("plane_slot_inbox"),
            "slot_mask_off": s2_off("node_dest_slot_mask"),
            "accumulator_off": s2_off("node_accumulator"),
            "accumulator_bytes": s2_plane("node_accumulator"),
            "rx_off": s2_off("remote_partial_rx"), "rx_bytes": s2_plane("remote_partial_rx"),
            "partial_ready_off": s2_off("node_partial_ready"),
            "group_ready_off": s2_off("return_group_ready"),
            "consumed_off": s2_off("return_consumed"), "arrival_off": s2_off("plane_slot_arrived"),
            "s2_window_off": layout.stage2_offset, "epoch_off": self._epoch_off,
            "parity_planes": tuple(s2_plane(n) for n in _K2_PARITY_PLANES),
            **({"sdma": True} if self._stage2_sdma else {}),
        }
        s2_waves = self._s2_waves
        if s2_waves != 4:
            self._k1_compile["waves"] = s2_waves
            self._k2_compile["threads"] = 64 * s2_waves
        # Token-ready rail for the rail-bound low-topk SDMA path. High topk
        # stays on the established local critical path (K3 measured regression).
        token_ready = (STAGE2_TOKEN_READY and self._stage2_fused and self._s2_waves == 8
                       and self._k1_compile["remote_first"]
                       and self._stage2_sdma and self.topk <= 8 and self.mtpr >= 2048)
        if token_ready:
            # Token service shares the arrival plane with inv and SDMA counters.
            # Keep the ordinary reduction when this optional state cannot fit.
            from .stage2 import tr_layout
            try:
                tr_base = tr_layout(self.mtpr, self.topk)["base"]
            except ValueError:
                token_ready = False
            else:
                cdone_base = (256 + 2 * self.mtpr * self.topk * 4 + 255) // 256 * 256
                cdone_end = cdone_base + self._k1_compile["sdma_max_chunks"] * 128
                token_ready = (cdone_end <= tr_base
                               and self._k2_compile["inbox_bytes"] < (1 << 31)
                               and self._chunk_tokens >= s2_waves
                               and self.comm_quant == "fp8")
        if token_ready:
            tiles = (s1.max_route_rows + TILE_M - 1) // TILE_M
            # Column units finished per m-block; the last one counts its rows.
            self._tr_mcnt = torch.zeros(tiles + 1, dtype=torch.int32, device=dev)
            self._k1_ptrs["arg_tr_mcnt"] = self._tr_mcnt.data_ptr()
            self._tr_sdprog = torch.empty(
                (GPUS_PER_NODE, self._k1_compile["sdma_max_chunks"]),
                dtype=torch.int64, device=dev)
            self._k1_ptrs["arg_tr_sdprog"] = self._tr_sdprog.data_ptr()
            self._k1_compile["sdma_put_bytes"] = 2 << 20
            self._k1_compile["token_ready"] = True
            self._k2_compile.update(token_ready=True, rail_idx_off=s2_off("rail_idx"),
                                    rail_idx_plane=s2_plane("rail_idx"))


    def _init_launchers(self) -> None:
        """Fix the specialization once; compiled callables belong to this instance."""
        self._compiled = {}
        self._launchers = {"record_quant": self._record_quant}
        if self._single_kernel:
            self._launchers["single"] = compile_single_kernel(
                self._stage1, self._k1_compile, self._k2_compile, WORKER_BLOCKS
            )
        else:
            self._launchers["stage1"] = self._stage1
            if self._stage2_fused:
                self._launchers["stage2"] = compile_stage2_fused(
                    self._k1_compile, self._k2_compile, WORKER_BLOCKS
                )
            else:
                self._launchers["gemm2"] = compile_mega_moe_stage2(**self._k1_compile)
                self._launchers["combine"] = compile_stage2_node_combine(**self._k2_compile)

    def _compile_launcher(self, name, args) -> None:
        """Compile and execute once, or only compile during preload.

        Keep an execution cache intact across compile-only contexts. FlyDSL
        can leak an MLIR context on compilation failure; unwind it so a retry
        behaves the same as the shared tensor_shim compile path.
        """
        compile_only = env.compile.compile_only
        try:
            compiled = flyc.compile(self._launchers[name], *args)
        except Exception:
            try:
                while ir.Context.current is not None:
                    ir.Context.current.__exit__(None, None, None)
            except Exception:  # noqa: BLE001, S110
                pass
            raise
        if not compile_only and compiled is not None:
            self._compiled[name] = compiled

    # forward

    def quantize(self, x_bf16: torch.Tensor):
        """Return MXFP4 ``[T, H/2]`` or MXFP8 ``[T, H]`` and E8M0 ``[T, H/32]``."""
        return per_1x32_mx_quant(x_bf16, quant_mode=self._activation_dtype)

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        """Return this rank's BF16 output rows for BF16 input ``[mtpr, H]``."""
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if tuple(x_bf16.shape) != (self.mtpr, self.model_dim):
            raise ValueError(f"x_bf16 must be [{self.mtpr}, {self.model_dim}], got {tuple(x_bf16.shape)}")
        self._check_routing(x_bf16, wts, topk_ids)
        launch_stream = self._fx_stream(stream)
        # One launch quantizes x and packs it with ids/weights into the records.
        args = (
            x_bf16.data_ptr(), topk_ids.data_ptr(), wts.data_ptr(), self._rec_q.data_ptr(),
            self.mtpr, self._quant_grid, launch_stream,
        )
        compiled = self._compiled.get("record_quant")
        if compiled is None or env.compile.compile_only:
            self._compile_launcher("record_quant", args)
        else:
            compiled(*args)
        return self._run(wts, topk_ids, launch_stream, slice_output)

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        """Same as :meth:`forward` for input already quantized by :meth:`quantize`."""
        payload_dtype = (torch.float8_e4m3fn,) if self.quant == "a8w4" else _FP4_DTYPES
        payload_cols = self.layout.stage1.wire.payload_bytes
        if x_q.dtype not in payload_dtype or not x_q.is_contiguous():
            raise ValueError(f"x_q must be contiguous {self._activation_dtype} payload, got {x_q.dtype}")
        if tuple(x_q.shape) != (self.mtpr, payload_cols):
            raise ValueError(f"x_q must be [{self.mtpr}, {payload_cols}], got {tuple(x_q.shape)}")
        if (scales.dtype != torch.uint8 or not scales.is_contiguous()
                or tuple(scales.shape) != (self.mtpr, self.model_dim // 32)):
            raise ValueError(f"scales must be contiguous uint8 E8M0 [{self.mtpr}, {self.model_dim // 32}]")
        self._check_routing(x_q, wts, topk_ids)
        if stream is not None and not isinstance(stream, torch.cuda.Stream):
            raise ValueError("forward_prequant needs stream=None or a torch.cuda.Stream")
        # The copies into the records must be ordered before stage1 on the same stream.
        with torch.cuda.stream(stream if stream is not None else torch.cuda.current_stream()):
            self._rec_q.copy_(x_q.view(torch.uint8))
            self._rec_scale.copy_(scales)
            self._rec_ids.copy_(topk_ids)
            self._rec_weights.copy_(wts)
        return self._run(wts, topk_ids, self._fx_stream(stream), slice_output)

    forward_bf16 = forward
    __call__ = forward

    def _check_routing(self, x, wts, topk_ids) -> None:
        if self._closed:
            raise RuntimeError("MegaMoEInternode is closed")
        shape = (self.mtpr, self.topk)
        if wts.dtype != torch.float32 or not wts.is_contiguous() or tuple(wts.shape) != shape:
            raise ValueError(f"wts must be contiguous float32 {list(shape)}")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous() or tuple(topk_ids.shape) != shape:
            raise ValueError(f"topk_ids must be contiguous int32 {list(shape)}")
        for name, tensor in (("x", x), ("wts", wts), ("topk_ids", topk_ids)):
            if tensor.device != self.device:
                raise ValueError(f"{name} is on {tensor.device}, expected {self.device}")

    @staticmethod
    def _fx_stream(stream):
        import flydsl.expr as fx

        if stream is None:
            return fx.Stream(torch.cuda.current_stream())
        if isinstance(stream, torch.cuda.Stream):
            return fx.Stream(stream)
        return stream

    def _run(self, wts, topk_ids, stream, slice_output):
        runtime = self._runtime
        window = runtime.window
        s1_args = (
            runtime.dev_comm.ptr, window.handle, window.local_ptr,
            self._rec_q.data_ptr(), self._rec_scale.data_ptr(),
            wts.data_ptr(), topk_ids.data_ptr(),
            self._w1.data_ptr(), self._w1_scale.data_ptr(), self.mtpr,
        )
        p = self._k1_ptrs
        gemm2_args = (
            *(p[k] for k in ("arg_aq", "arg_ascale", "arg_bq", "arg_bscale", "arg_eids",
                             "arg_cumsum", "arg_stids", "arg_sweights", "arg_trb", "arg_p2p")),
            (p["row_capacity"] + BM - 1) // BM,
            self.inter_dim, self.model_dim,
            p["arg_arrival_count"], p["arg_generation"],
            p["arg_a_tile_map"], p["local_rank"],
        )
        output_args = (self._output.data_ptr(), self._k2_scratch.data_ptr(), self.mtpr)
        extra_args = (
            *(p.get(k, 0) for k in ("arg_expert_count", "arg_sd_send", "arg_sd_tb",
                                  "arg_sd_plan", "arg_sd_cdone", "arg_tr_mcnt", "arg_tr_sdprog")),
            stream,
        )
        if self._single_kernel:
            launches = (("single", (*s1_args, *gemm2_args, *output_args, *extra_args)),)
        else:
            launches = (("stage1", (*s1_args, stream)),)
            if self._stage2_fused:
                launches += (("stage2", (*gemm2_args, *s1_args[:3], *output_args, *extra_args)),)
            else:
                launches += (
                    ("gemm2", (*gemm2_args, stream)),
                    ("combine", (*s1_args[:3], *output_args, WORKER_BLOCKS, stream)),
                )
        for name, args in launches:
            compiled = self._compiled.get(name)
            if compiled is None or env.compile.compile_only:
                self._compile_launcher(name, args)
            else:
                compiled(*args)
        return self._output[: self.mtpr] if slice_output else self._output
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runtime, self._runtime = self._runtime, None
        if runtime is None:
            return
        torch.cuda.synchronize(self.device)
        if runtime.owns_communicator:
            # Leaving the context releases window, memory and dev_comm too.
            runtime.context.__exit__(None, None, None)
        else:
            # The caller's Communicator outlives us: release only our own objects.
            runtime.dev_comm.close()
            runtime.window.close()
            runtime.memory.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
    return tensor.view(torch.uint8)
