# SPDX-License-Identifier: MIT
"""Public strict-two-kernel EP16 A4W4 MegaMoE operator.

This class intentionally mirrors :class:`MegaMoEV2` at its public boundary,
but specializes the K3 two-node deployment and accepts only ``quant='a4w4'``.
All allocation, CCO setup, window initialization and FlyDSL compilation happen
in the constructor.  A hot ``forward`` performs exactly two launcher calls:

1. fused BF16 quant + InterNodeV1 direct-to-expert-tile dispatch + GMM1 +
   SiLU + A4 requant;
2. fused weighted GMM2 + direct LSA FP32 node-accumulator epilogue +
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
    Stage1ArenaLayout,
    TwoKernelArenaLayout,
    validate_public_stage1_contract,
)
from .stage2_abi import Stage2ArenaLayout


_STAGE1_MODULE = "aiter.ops.flydsl.kernels.megamoe_tile.stage1"
_STAGE1_FACTORY = "compile_megamoe_tile_ep16_stage1"
_STAGE2_MODULE = "aiter.ops.flydsl.kernels.megamoe_tile.stage2"
_STAGE2_FACTORY = "compile_megamoe_tile_ep16_stage2_a4w4"


@dataclass(frozen=True)
class _CcoRuntime:
    context: Any
    communicator: Any
    memory: Any
    window: Any
    dev_comm: Any
    per_rank_vmm: int


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
        raise ValueError(f"{name} must have a byte-addressable packed layout") from error


class MegaMoETileA4W4:
    """K3 EP16 hierarchical MegaMoE with exactly two hot GPU launches.

    The instance supports one ordered in-flight forward on one CUDA stream, as
    does MegaMoEV2. ``rank`` is the global EP rank; tensor allocation always
    uses the current local CUDA device so ranks 8--15 work on node 1.
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
        stage1_transport: str = "chunked"):
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
        self.stage1_transport = str(stage1_transport)
        self.gpus_per_node = 8
        self.node = self.rank // self.gpus_per_node
        self.local_rank = self.rank % self.gpus_per_node
        self.peer_node = 1 - self.node
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.worker_blocks = 160
        self.stage1_worker_blocks = (
            256 if self.stage1_transport == "sparse_wqe" else self.worker_blocks
        )
        device_cus = torch.cuda.get_device_properties(self.device).multi_processor_count
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
        )
        self.stage2_layout = Stage2ArenaLayout.create(
            hidden=self.model_dim,
            topk=self.topk,
            max_tokens=self.mtpr,
            world_size=self.world_size,
            gpus_per_node=self.gpus_per_node,
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
    ) -> None:
        expected = {
            "world_size": (world_size, 16),
            "model_dim": (model_dim, 7168),
            "inter_dim": (inter_dim, 3072),
            "experts": (experts, 896),
            "topk": (topk, 16),
            "max_tok_per_rank": (max_tok_per_rank, 128),
        }
        bad = {
            name: (int(got), want)
            for name, (got, want) in expected.items()
            if int(got) != want
        }
        if bad:
            detail = ", ".join(
                f"{name}={got} (expected {want})"
                for name, (got, want) in bad.items()
            )
            raise ValueError(f"MegaMoETileA4W4 supports only K3 EP16: {detail}")
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
                "the locked activation is SiLU; swiglu_limit must remain 0.0"
            )
        if str(stage1_transport) not in ("chunked", "sparse_wqe"):
            raise ValueError(
                "stage1_transport must be 'chunked' or 'sparse_wqe'"
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
        from mori.cco import (
            CCODevCommRequirements,
            Communicator,
            GDA_CONNECTION_RAIL,
            UniqueId,
        )

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
        try:
            memory = communicator.alloc_mem(self.layout.total_bytes)
            window = communicator.register_window(memory.ptr, memory.size)
            requirements = CCODevCommRequirements()
            requirements.gda_connection_type = GDA_CONNECTION_RAIL
            if self.stage1_layout.num_qp != self.stage2_layout.num_qp:
                raise AssertionError("Stage1/Stage2 must use the same CCO QP count")
            requirements.gda_context_count = self.stage1_layout.num_qp
            requirements.gda_signal_count = 0
            requirements.gda_counter_count = 0
            requirements.lsa_barrier_count = 0
            requirements.rail_gda_barrier_count = 0
            requirements.barrier_count = 0
            dev_comm = communicator.create_dev_comm(requirements)

            # This launch is constructor-only and therefore excluded from the
            # strict hot-path trace contract.
            from .cco import zero_window

            zero_window(window.local_ptr, self.layout.total_bytes)
            torch.cuda.synchronize(self.device)
            communicator.barrier()
            return _CcoRuntime(
                context,
                communicator,
                memory,
                window,
                dev_comm,
                per_rank_vmm,
            )
        except Exception:
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
            tile_pipeline=sparse,
            tile_pipeline_fanout_shards=16,
        )

    def _compile_stage2(self):
        factory = _import_factory(_STAGE2_MODULE, _STAGE2_FACTORY)
        return factory(
            self.layout,
            rank=self.rank,
            BM=32,
            BN=256,
            BK=256,
            WORK_SHARDS=8,
            waves_per_eu_hint=2,
            team="rail",
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
                    else "post_eos_static_strided_24_pure_compute_consumers"
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
                "the first strict EP16 kernels require exactly 128 tokens/rank"
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
        self._generation += 1
        generation = self._generation
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

        from .cco import read_window_bytes, read_window_u32, read_window_u64

        torch.cuda.synchronize(self.device)
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

        tile_alloc = int(read_window_u32(s1_ptr("tile_alloc"), 1)[0])
        queue_tail = int(read_window_u32(s1_ptr("h1_queue_tail"), 1)[0])
        compute_done = int(read_window_u32(s1_ptr("h1_compute_done"), 1)[0])
        queue_jobs = list(
            read_window_u32(s1_ptr("h1_ready_queue"), queue_tail)
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
            read_window_u32(s1_ptr("h1_early_full_tiles"), 1)[0]
        )
        gmm_started_before_eos = int(
            read_window_u32(
                s1_ptr("h1_gmm_started_before_all_comm_eos"), 1
            )[0]
        )
        gmm_completed_before_eos = int(
            read_window_u32(
                s1_ptr("h1_gmm_completed_before_all_comm_eos"), 1
            )[0]
        )
        raw_arrived = list(
            read_window_u32(
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
            read_window_u32(
                s1_ptr("expert_count"), self.stage1_layout.local_experts
            )
        )
        comm_eos = list(
            read_window_u64(s1_ptr("comm_eos"), self.gpus_per_node)
        )
        stage1_done = int(read_window_u64(s2_ptr("stage1_done"), 1)[0])
        stage1_errors = int(
            read_window_u32(
                s1_ptr("error_count", parity_indexed=False), 1
            )[0]
        )
        stage2_errors = int(
            read_window_u32(
                s2_ptr("stage2_error_count", parity_indexed=False), 1
            )[0]
        )

        ntiles = self.model_dim // 256
        scoreboard_size = 2 * self.mtpr * ntiles
        node_expected_all = list(
            read_window_u32(s2_ptr("node_expected"), scoreboard_size)
        )
        node_done_all = list(
            read_window_u32(s2_ptr("node_done"), scoreboard_size)
        )
        node_ready_all = list(
            read_window_u64(s2_ptr("node_tile_ready"), scoreboard_size)
        )

        # Canonicalize Stage1 rows by the complete packed source key, removing
        # nondeterministic physical-tile allocation order from epoch comparison.
        num_valid = int(read_window_u32(s1_ptr("num_valid"), 1)[0])
        packed_sources = list(
            read_window_u32(s1_ptr("tile_row_source"), num_valid)
        )
        input_rows = list(
            read_window_u32(s1_ptr("tile_row_input"), num_valid)
        )
        weight_bits = list(
            read_window_u32(s1_ptr("tile_row_weight"), num_valid)
        )
        tile_count = (num_valid + self.stage1_layout.block_m - 1) // self.stage1_layout.block_m
        tile_experts = list(
            read_window_u32(s1_ptr("tile_expert"), tile_count)
        )
        input_row_bytes = self.model_dim // 2
        h1_row_bytes = self.inter_dim // 2
        grouped_input = read_window_bytes(
            s1_ptr("grouped_input_q"),
            self.stage1_layout.source_capacity * input_row_bytes,
        )
        scale_region = self.stage1_layout.region("grouped_input_scale")
        grouped_scale = read_window_bytes(
            s1_ptr("grouped_input_scale"),
            scale_region.nbytes // self.stage1_layout.parity_depth,
        )
        h1_output = read_window_bytes(
            s1_ptr("h1_output_q"), num_valid * h1_row_bytes
        )
        h1_scale_region = self.stage1_layout.region("h1_output_scale")
        h1_output_scale = read_window_bytes(
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
            replay_sha = hashlib.sha256()
            fused_vs_replay_rows = 0
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
                digest = hashlib.sha256(replay_row).digest()
                replay_per_key[packed] = digest
                fused_vs_replay_rows += int(replay_row != fused_row)
            previous_replay = getattr(self, "_debug_previous_replay_h1", None)
            replay_changed_rows = None
            if previous_replay is not None:
                replay_changed_rows = sum(
                    int(previous_replay.get(key) != replay_per_key.get(key))
                    for key in set(previous_replay) | set(replay_per_key)
                )
            self._debug_previous_replay_h1 = replay_per_key
            canonical_h1["standalone_replay_sha256"] = replay_sha.hexdigest()
            canonical_h1["standalone_replay_changed_rows"] = replay_changed_rows
            canonical_h1["fused_vs_standalone_changed_rows"] = fused_vs_replay_rows
        local_base = self.node * self.mtpr * ntiles
        node_expected = []
        node_done = []
        node_ready = []
        for token in range(self.mtpr):
            start = local_base + token * ntiles
            end = start + ntiles
            expected_slice = node_expected_all[start:end]
            done_slice = node_done_all[start:end]
            ready_slice = node_ready_all[start:end]
            node_expected.append(min(expected_slice))
            node_done.append(min(done_slice))
            node_ready.append(
                int(all(int(value) >= generation for value in ready_slice))
            )

        return {
            # Fields consumed by the untimed harness validator.
            "comm_role_eos": comm_eos,
            "alloc_count": arrived,
            "tile_arrived": arrived,
            "tile_ready": [int(value > 0) for value in arrived],
            "tail_tile": [int(0 < value < 32) for value in arrived],
            "tail_sealed": [int(0 < value < 32) for value in arrived],
            "node_atomic_expected": node_expected,
            "node_atomic_done": node_done,
            "node_atomic_ready": node_ready,
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
            "node_expected_done_mismatch": sum(
                int(int(expected) != int(done))
                for expected, done in zip(node_expected_all, node_done_all)
            ),
            "node_not_ready": sum(
                int(int(value) < generation) for value in node_ready_all
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

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            torch.cuda.synchronize(self.device)
            runtime.context.__exit__(None, None, None)

    def __enter__(self) -> "MegaMoETileA4W4":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# Descriptive alias used by the bring-up benchmark factory string.
HierarchicalMegaMoEV2 = MegaMoETileA4W4


__all__ = ["HierarchicalMegaMoEV2", "MegaMoETileA4W4"]
