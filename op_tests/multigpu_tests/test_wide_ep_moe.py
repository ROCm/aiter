# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TestWideEpMoe EP16 A4W4/A8W4 correctness, performance, and profiling test.

Pipeline: prequantized dispatch (MORI InterNodeV1LL) -> AITER fused_moe
(A4W4 or A8W4, per_1x32, real expert_mask) -> BF16 combine.

Prerequisites
=============

Hardware and topology
---------------------

* AMD MI355/gfx950 GPUs only.
* Exactly two nodes with eight visible GPUs per node (EP16, global ranks 0..15).
* Kimi-K3: H3584/I3072/E896/TopK16 A4W4.
* DSV4: H7168/I3072/E384/TopK6 A8W4.
* Each GPU must have enough free VRAM for the 40 GiB MORI symmetric heap plus
  local A4W4 weights and test workspaces.  Check GPU users before launching.
* The two hosts must expose the RDMA rails selected by ``MORI_RDMA_DEVICES``.

Software and repository
-----------------------

* PyTorch must be a ROCm build with gfx950 support and distributed Gloo.
* The PyTorch build must expose packed MXFP4 as
  ``torch.float4_e2m1fn_x2`` (or the compatible one-byte storage ABI used by
  this AITER revision).
* MORI Python bindings and its InterNodeV1LL HIP kernels must be installed in
  the container.  ``mori.shmem`` must support torch process-group bootstrap.
* A working ROCm/hipcc and FlyDSL toolchain are required.  A clean checkout
  JIT-builds AITER modules such as ``module_aiter_core``, ``module_moe_asm``,
  ``module_moe_sorting_opus``, and ``module_quant`` on first use; one local
  worker builds while the other workers wait on the JIT baton.
* The tune file
  ``aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv`` must exist and
  contain the H3584/I3072/local-expert-56/TopK-compatible entries.
* The test-only wrapper and its custom op live in this file.  No TestWideEP
  implementation is exported from ``aiter.ops`` and MegaMoEV2 is not modified.

Network and process setup
-------------------------
* Use the same unused ``master_port`` and identical test arguments on both
  nodes.  Start node 0 and node 1 close together.
* ``torchrun`` starts one coordinator process per node.  The test itself uses
  ``torch.multiprocessing.spawn`` to create eight local GPU workers.
* Set ``PYTHONPATH`` to this checkout so the test cannot resolve a different
  editable AITER installation.

Input and numerical contract
----------------------------

* Dispatch input is packed FP4 plus one-byte E8M0 scale per 1x32 block.
* GEMM1 and GEMM2 use A4W4 with ``ActivationType.Situv2`` and
  ``QuantType.per_1x32``.
* Combine output is BF16.
* Inputs use deterministic random router scores and ``fused_topk``.  The
  unrelated ``AITER_MOE_EXPERT_BALANCE`` variable is not consumed.
* ``--accuracy-max-bs`` controls which cases build the Torch MoE reference.
  Cases above that threshold check shape, finite values and (when enabled)
  compiled/eager consistency, but are NOT reference-accuracy passes.
* A8W4 reuses ``torch_moe_stage1/2`` and the intermediate FP8/E8M0 quantizer
  from ``op_tests/test_moe_2stage.py``. Its ``strict_accuracy`` rejection rule
  is nonzero allclose mismatch AND ``logits_diff > 0.01``. The separate
  ``elementwise_5pct`` result reports whether at most 5% of elements exceed
  ``atol=rtol=0.01``; an AITER strict pass does not imply this stronger check.
* A8W4 reference dtype is selected automatically from
  the actual fused_moe metadata: FP32 for fused FP8-quant GEMM1, BF16 otherwise.
  Selection uses the received-buffer capacity, not TPR.
* The 5% elementwise result is retained as a diagnostic only.  The pass/fail
  gate matches ``op_tests/test_moe_2stage.py``: reject only when allclose has
  mismatches and ``logits_diff > 0.01``.
* For A8W4 cases covered by ``--accuracy-max-bs``, the test also checks the final
  output from the
  pre-dispatch source payload, scales, weights and global expert IDs. CPU/Gloo
  transports the source data and sums independent local-expert references;
  this oracle does not use MORI's received buffers or its combine algorithm.
* Profiler mode uses ten warmup iterations and records forty iterations,
  numbered groups ``iter0`` through ``iter39``.  The standard report uses GPU
  annotations from ``iter20`` through ``iter39``.

Standard two-node MI355 launch
==============================

Run from a checkout of this repository.  ``PYTHONPATH`` is explicit so a
container-wide editable AITER installation cannot silently resolve another
worktree.  Start node 0 first and node 1 immediately afterwards, using the same
master address, port, environment, and test arguments.

For Exp:

  PYTHONPATH=your_aiter_project_path \
  GLOO_SOCKET_IFNAME=your_bootstrap_interface \
  MORI_SOCKET_IFNAME=your_bootstrap_interface \
  MORI_DEVICE_NIC=ionic \
  MORI_RDMA_DEVICES='your_rdma_device_selection' \
  MORI_IB_GID_INDEX=1 \
  MORI_SHMEM_HEAP_SIZE=40G \
  MORI_EP_LAUNCH_CONFIG_MODE=AUTO \
  AITER_CONFIG_FMOE=aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv \
  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
    --master_addr=master_addr_ip --master_port=30001 \
    op_tests/multigpu_tests/test_wide_ep_moe.py \
    --bs-list 4,128 --accuracy-max-bs 128 --staged-only

The outer ``torchrun`` intentionally uses ``--nproc_per_node=1``.  This script
then spawns the default eight local GPU workers, producing global EP ranks
0..15.

Environment variables
---------------------

``PYTHONPATH``
    Selects this checkout instead of another editable AITER installation.
``GLOO_SOCKET_IFNAME``
    Pins the CPU process group to the inter-node data interface.
``MORI_SOCKET_IFNAME``
    Pins MORI control/bootstrap traffic to the same data interface.
``MORI_DEVICE_NIC=ionic``
    Selects the MI355 Ionic RDMA NIC backend.
``MORI_RDMA_DEVICES``
    Excludes the two control/BNXT-facing devices; MORI uses the remaining eight
    Ionic rails.
``MORI_IB_GID_INDEX=1``
    Selects the validated RoCE GID entry.
``MORI_SHMEM_HEAP_SIZE=40G``
    Reserves sufficient symmetric heap for EP16 and large token cases.
``MORI_EP_LAUNCH_CONFIG_MODE=AUTO``
    Lets MORI select its validated launch configuration.
``AITER_CONFIG_FMOE``
    Selects the Kimi-K3 H3584/I3072 local-expert tune table.

The test fixes ``gpu_per_node=8`` and ``num_qp_per_pe=2`` by default, and the
A4W4 preset selects SiTUv2 internally, so those settings do not need to be
repeated in the launch command.  This test uses CPU Gloo plus MORI and does not
create an NCCL process group.

``AITER_MOE_EXPERT_BALANCE`` is intentionally not used.

Compile profiler example
------------------------

Add the following arguments to both nodes to compile the public
``TestWideEpMoe.forward_prequant`` call, warm up ten times, profile forty
replays, and retain numbered iteration groups for tail-20 analysis::

  --torch-compile-cudagraph \
  --torch-profiler-dir trace_data/testwide_compile_trace

DSV4 A8W4 EP16 preset
---------------------

DSV4 uses H=7168, I=3072, 384 global experts, 24 local experts per EP16 rank,
TopK=6, FP8 activations, FP4 weights, per-1x32 E8M0 scales, SiLU, and the
gate/up-interleaved GEMM1 layout, and ``swiglu_limit=10.0`` from the model config.
The preset forwards the same limit to the operator and both independent
references. Kimi retains its default limit of 0.
Replace the tune file and add the preset on
both nodes::

  ATOM_MOE_GU_ITLV=1 \
  AITER_CONFIG_FMOE=aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv \
  torchrun ... op_tests/multigpu_tests/test_wide_ep_moe.py \
    --model-config dsv4 --bs-list 4,128 --accuracy-max-bs 128 \
    --staged-only

The source DSV4 table is tuned for EP8 with 48 local experts.  Its EP16 rows
preserve the same kernels and token buckets with 24 local experts.  MORI uses
the model's real TopK=6.  Before fused_moe, this test appends one zero-weight,
always-masked fake-expert slot, matching test_moe_ep.py.  The fused_moe EP
lookup removes that slot and therefore queries the tune table with TopK=6.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

import mori
import mori.shmem as ms
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    situv2,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
)
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils

_TEST_WIDE_EP_INSTANCES = {}


@triton.jit
def _append_fake_route_kernel(
    src_weights,
    src_ids,
    dst_weights,
    dst_ids,
    total_elements,
    routed_topk: tl.constexpr,
    fake_expert_id: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy MORI routes and append one zero-weight fake-expert route."""
    out_topk: tl.constexpr = routed_topk + 1
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < total_elements
    row = offsets // out_topk
    col = offsets - row * out_topk
    routed = valid & (col < routed_topk)
    src_offsets = row * routed_topk + col
    weights = tl.load(src_weights + src_offsets, mask=routed, other=0.0)
    ids = tl.load(src_ids + src_offsets, mask=routed, other=fake_expert_id)
    tl.store(dst_weights + offsets, weights, mask=valid)
    tl.store(dst_ids + offsets, ids, mask=valid)


@torch.library.custom_op("aiter::test_wide_ep_forward", mutates_args=())
def _test_wide_ep_forward(
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    instance_id: int,
    model_dim: int,
) -> torch.Tensor:
    """Keep dispatch, fused_moe and combine behind one Dynamo boundary."""
    op = _TEST_WIDE_EP_INSTANCES[instance_id]
    # MORI's combine result aliases its symmetric arena.  The clone creates the
    # single graph-pool-owned public output needed by cudagraph replay.
    return op._forward_prequant_impl(x_quant, x_scale, topk_weight, topk_ids).clone()


@_test_wide_ep_forward.register_fake
def _test_wide_ep_forward_fake(
    x_quant,
    x_scale,
    topk_weight,
    topk_ids,
    instance_id,
    model_dim,
):
    return x_quant.new_empty((topk_ids.shape[0], model_dim), dtype=torch.bfloat16)


@dataclass
class TestWideEpMoeContext:
    tokens: torch.Tensor
    weights: torch.Tensor
    scales: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens: torch.Tensor
    source_topk_ids: torch.Tensor
    source_tokens: int
    owner_id: int
    generation: int
    consumed: bool = False


class TestWideEpMoe:
    """Test-only EP16 MORI dispatch + AITER fused_moe + MORI combine wrapper."""

    def __init__(
        self,
        *,
        rank,
        world_size,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        w1,
        w1_scale,
        w2,
        w2_scale,
        max_tok_per_rank,
        gpu_per_node=8,
        swiglu_limit=0.0,
        activation=None,
        gate_mode=None,
    ):
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from aiter.ops.flydsl.moe_common import GateMode

        if get_gfx_runtime() != "gfx950":
            raise ValueError("TestWideEpMoe is supported only on gfx950")
        if quant not in ("a4w4", "a8w4"):
            raise ValueError("quant must be a4w4 or a8w4")
        if world_size != 16 or gpu_per_node != 8:
            raise ValueError("TestWideEpMoe requires EP16 (2 nodes x 8 GPUs)")
        if experts % world_size:
            raise ValueError("experts must be divisible by world_size")
        if max_tok_per_rank <= 0 or model_dim % 32:
            raise ValueError("invalid max_tok_per_rank or model_dim")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = self.experts // self.world_size
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.quant = quant
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.activation = activation or (
            ActivationType.Situv2 if quant == "a4w4" else ActivationType.Silu
        )
        self.gate_mode = GateMode(
            gate_mode or ("separated" if quant == "a4w4" else "interleave")
        )
        self.swiglu_limit = float(swiglu_limit)
        if not self.swiglu_limit >= 0:
            raise ValueError("swiglu_limit must be non-negative and not NaN")

        self.activation_dtype = dtypes.fp4x2 if quant == "a4w4" else dtypes.fp8
        self.capacity_mtpr = 1 << (self.mtpr - 1).bit_length()
        self.w1, self.w1_scale = w1, w1_scale
        self.w2, self.w2_scale = w2, w2_scale
        local_start = self.rank * self.epr
        self.expert_mask = torch.zeros(
            self.experts + 1, dtype=torch.int32, device=self.dev
        )
        self.expert_mask[local_start : local_start + self.epr] = 1

        if quant == "a4w4":
            os.environ["AITER_SITUV2_A8W4"] = "0"
            os.environ["AITER_SITUV2_A4W4"] = "1"
        else:
            os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")

        config = mori.ops.EpDispatchCombineConfig(
            data_type=self.activation_dtype,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=self.model_dim,
            scale_dim=self.model_dim // 32,
            scale_type_size=1,
            max_num_inp_token_per_rank=self.capacity_mtpr,
            num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk,
            max_token_type_size=2,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            gpu_per_node=gpu_per_node,
            num_qp_per_pe=2,
            rdma_block_num=int(os.environ.get("MORI_EP_RDMA_BLOCK_NUM", "64")),
            block_num=int(os.environ.get("MORI_EP_BLOCK_NUM", "96")),
            warp_num_per_block=int(os.environ.get("MORI_EP_WARP_PER_BLOCK", "8")),
        )
        self.op = mori.ops.EpDispatchCombineOp(config)
        self.owner_id = id(self)
        _TEST_WIDE_EP_INSTANCES[self.owner_id] = self
        self.generation = 0
        self.active_dispatch = None
        self._fmoe_route_weights = None
        self._fmoe_route_ids = None

    def prepare_torch_compile(self, x_quant, x_scale, weights, topk_ids):
        for tensor in (x_quant, x_scale, weights, topk_ids):
            torch._dynamo.mark_static_address(tensor, guard=True)

    def _validate_context(self, context):
        if not isinstance(context, TestWideEpMoeContext):
            raise TypeError("context must be TestWideEpMoeContext")
        if context.owner_id != self.owner_id or context.generation != self.generation:
            raise RuntimeError(
                "dispatch context is stale or belongs to another instance"
            )
        if context is not self.active_dispatch or context.consumed:
            raise RuntimeError("dispatch context is not active")

    def dispatch_prequant(self, x_quant, x_scale, weights, topk_ids):
        tokens = int(topk_ids.shape[0])
        if tokens > self.mtpr:
            raise ValueError(f"tokens={tokens} exceeds max_tok_per_rank={self.mtpr}")
        quant_width = self.model_dim // 2 if self.quant == "a4w4" else self.model_dim
        expected = {
            "x_quant": ((tokens, quant_width), self.activation_dtype),
            "x_scale": ((tokens, self.model_dim // 32), None),
            "weights": ((tokens, self.topk), torch.float32),
            "topk_ids": ((tokens, self.topk), torch.int32),
        }
        for name, tensor in (
            ("x_quant", x_quant),
            ("x_scale", x_scale),
            ("weights", weights),
            ("topk_ids", topk_ids),
        ):
            shape, dtype = expected[name]
            if tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous with shape {shape}")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(f"{name} must have dtype {dtype}")
            if tensor.device != self.dev:
                raise ValueError(f"{name} must be on {self.dev}")
        if x_scale.element_size() != 1:
            raise ValueError("x_scale must use one-byte E8M0 storage")
        if self.active_dispatch is not None and not self.active_dispatch.consumed:
            raise RuntimeError("complete the previous dispatch before starting another")

        recv = self.op.dispatch(x_quant, weights, x_scale, topk_ids)
        self.generation += 1
        # AITER's EP fused_moe ABI follows test_moe_ep.py: append one
        # always-masked fake-expert route after the real routed TopK.  There are
        # no shared experts in TestWideEpMoe.  Keep MORI configured with the
        # real TopK so the sentinel never participates in dispatch; only the
        # local fused_moe input uses [real TopK + one fake slot].
        route_shape = (recv[1].shape[0], self.topk + 1)
        if self._fmoe_route_weights is None:
            self._fmoe_route_weights = torch.empty(
                route_shape, dtype=recv[1].dtype, device=self.dev
            )
            self._fmoe_route_ids = torch.empty(
                route_shape, dtype=recv[3].dtype, device=self.dev
            )
        elif (
            tuple(self._fmoe_route_weights.shape) != route_shape
            or tuple(self._fmoe_route_ids.shape) != route_shape
        ):
            raise RuntimeError(
                "MORI dispatch capacity changed after route buffers were allocated"
            )
        total = route_shape[0] * route_shape[1]
        _append_fake_route_kernel[(triton.cdiv(total, 256),)](
            recv[1],
            recv[3],
            self._fmoe_route_weights,
            self._fmoe_route_ids,
            total,
            routed_topk=self.topk,
            fake_expert_id=self.experts,
            BLOCK_SIZE=256,
        )
        context = TestWideEpMoeContext(
            tokens=recv[0],
            weights=self._fmoe_route_weights,
            scales=recv[2],
            expert_ids=self._fmoe_route_ids,
            num_tokens=recv[4],
            source_topk_ids=topk_ids,
            source_tokens=tokens,
            owner_id=self.owner_id,
            generation=self.generation,
        )
        self.active_dispatch = context
        return context

    def fused_moe(self, context):
        from aiter.fused_moe import fused_moe

        self._validate_context(context)
        return fused_moe(
            context.tokens,
            self.w1,
            self.w2,
            context.weights,
            context.expert_ids,
            expert_mask=self.expert_mask,
            activation=self.activation,
            gate_mode=self.gate_mode.value,
            quant_type=QuantType.per_1x32,
            swiglu_limit=self.swiglu_limit,
            beta=DEFAULT_SITUV2_BETA,
            linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            a1_scale=context.scales,
            num_local_tokens=context.num_tokens[:1].to(dtypes.i32),
            dtype=torch.bfloat16,
        )

    def combine(self, local_output, context):
        self._validate_context(context)
        output, output_weights = self.op.combine(
            local_output, None, context.source_topk_ids
        )
        context.consumed = True
        return output[: context.source_tokens], output_weights

    def forward_prequant(self, x_quant, x_scale, weights, topk_ids):
        if torch.compiler.is_compiling():
            return _test_wide_ep_forward(
                x_quant,
                x_scale,
                weights,
                topk_ids,
                self.owner_id,
                self.model_dim,
            )
        return self._forward_prequant_impl(x_quant, x_scale, weights, topk_ids)

    def _forward_prequant_impl(self, x_quant, x_scale, weights, topk_ids):
        context = self.dispatch_prequant(x_quant, x_scale, weights, topk_ids)
        local_output = self.fused_moe(context)
        return self.combine(local_output, context)[0]


# Target EP16 A4W4 pipeline shape. The CSV-compatible result format follows
# aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv, while this benchmark
# intentionally uses the requested larger intermediate dimension (3072).
NETWORKS = {
    "kimi": {
        "model_dim": 3584,
        "inter_dim": 3072,
        "experts": 896,
        "topk": 16,
        "quant": "a4w4",
        "activation": ActivationType.Situv2,
        "gate_mode": "separated",
        "swiglu_limit": 0.0,
    },
    # DSV4's tune table is EP8 with 48 local experts: 48 * 8 = 384 global.
    # The EP16 conversion keeps 384 global experts and uses 24 per rank.
    "dsv4": {
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
        "quant": "a8w4",
        "activation": ActivationType.Silu,
        "gate_mode": "interleave",
        "swiglu_limit": 10.0,
    },
}
GPU_PER_NODE_DEFAULT = 8
A4W4_RTOL = 0.15
PROFILE_WARMUP_ITERS = 10
PROFILE_ITERS = 40


@dataclass(frozen=True)
class RunConfig:
    model: str
    batch_sizes: tuple[int, ...]
    iterations: int
    statistic_iterations: int
    seed: int
    accuracy_max_batch: int
    profiler_dir: str | None
    compile_cudagraph: bool
    staged_only: bool


@dataclass(frozen=True)
class TestState:
    x_quant: torch.Tensor
    x_scale: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w1_reference: torch.Tensor
    w1_reference_scale: torch.Tensor
    w2_reference: torch.Tensor
    w2_reference_scale: torch.Tensor
    expert_mask: torch.Tensor
    local_experts: int
    model_dim: int
    inter_dim: int
    quant: str
    activation: object
    world_size: int


def _setup_dist(rank, world_size, local_rank):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo", rank=rank, world_size=world_size)
    world_group = dist.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)
    ms.shmem_torch_process_group_init("default")
    return device


def _cleanup():
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


def _barrier():
    debug = os.environ.get("AITER_DEBUG_WIDE_EP", "0") == "1"
    rank = dist.get_rank() if dist.is_initialized() else -1
    if debug:
        print(f"[EP16-barrier rank={rank}] cuda sync 1 start", flush=True)
    torch.cuda.synchronize()
    if debug:
        print(
            f"[EP16-barrier rank={rank}] cuda sync 1 complete; shmem start", flush=True
        )
    ms.shmem_barrier_all()
    if debug:
        print(
            f"[EP16-barrier rank={rank}] shmem complete; cuda sync 2 start", flush=True
        )
    # shmem_barrier_all may enqueue device work. Drain it here so a
    # caller that records a CUDA event immediately after _barrier() does not
    # accidentally charge the barrier kernel to the following stage.
    torch.cuda.synchronize()
    if debug:
        print(f"[EP16-barrier rank={rank}] cuda sync 2 complete", flush=True)


def _reduce_float(value, op):
    if not dist.is_initialized():
        return float(value)
    # Process group is cpu:gloo only (no NCCL) -- reduce on CPU.
    result = torch.tensor(float(value), dtype=torch.float32, device="cpu")
    dist.all_reduce(result, op=op)
    return float(result.item())


def _collective_require(condition, message):
    """Make every rank fail together instead of stranding peers in collectives."""
    failed = _reduce_float(float(not bool(condition)), dist.ReduceOp.MAX)
    if failed:
        raise AssertionError(message)


def _make_local_inputs(
    tokens,
    model_dim,
    experts,
    topk,
    rank,
    seed,
    device,
):
    """Build deterministic random per-rank inputs and duplicate-free TopK routes."""
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.bfloat16, device=device, generator=generator
    )
    topk_ids = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((tokens, topk), dtype=torch.float32, device=device)
    fused_topk(x, scores, topk, True, topk_ids, topk_weights)
    return x.contiguous(), topk_weights.contiguous(), topk_ids.contiguous()


def _quantize_local_weights(
    model_dim, inter_dim, local_experts, rank, seed, device, quant="a4w4"
):
    """Per-rank local-expert bf16 weights -> a4w4 (per_1x32 mxfp4) quantized +
    shuffled, following test_moe_ep.py's a4w4_mxfp4 branch exactly. Returns both
    the kernel-ready shuffled tensors and the unshuffled quantized tensors (for
    the dequantized reference computation)."""
    generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    use_mxmoe_w1 = os.environ.get("MORI_MXMOE_W1_LAYOUT", "0") == "1"

    w1 = (
        torch.randn(
            (local_experts, 2 * inter_dim, model_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    if use_mxmoe_w1:
        from aiter.ops.quant import per_1x32_mx_quant_hip

        w1_qt, w1_scale = per_1x32_mx_quant_hip(
            w1.view(-1, model_dim), quant_dtype=dtypes.fp4x2
        )
        w1_qt = w1_qt.view(local_experts, 2 * inter_dim, model_dim // 2)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
        w1_qt = w1_qt.view(local_experts, 2 * inter_dim, model_dim // 2)

    w2 = (
        torch.randn(
            (local_experts, model_dim, inter_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(local_experts, model_dim, inter_dim // 2)

    # mxmoe GEMM1 consumes the A16W4 preshuffle, while the previous
    # flydsl_moe1_afp4 baseline consumes the generic layout.  Keep both paths
    # testable without rewriting the weight setup between regression runs.
    use_a16w4_layout = use_mxmoe_w1 or quant == "a8w4"
    w1_a = (
        shuffle_weight_a16w4(w1_qt, 16, quant == "a8w4")
        if use_a16w4_layout
        else shuffle_weight(w1_qt, layout=(16, 16))
    )
    w2_a = (
        shuffle_weight_a16w4(w2_qt, 16, False)
        if use_a16w4_layout
        else shuffle_weight(w2_qt, layout=(16, 16))
    )
    w1_s = (
        shuffle_scale_a16w4(w1_scale, local_experts, quant == "a8w4")
        if use_a16w4_layout
        else fp4_utils.e8m0_shuffle(w1_scale)
    )
    w2_s = (
        shuffle_scale_a16w4(w2_scale.view(-1, inter_dim // 32), local_experts, False)
        if use_a16w4_layout
        else fp4_utils.e8m0_shuffle(w2_scale)
    )
    w1_a.is_shuffled = True
    w2_a.is_shuffled = True

    return (w1_a, w1_s, w2_a, w2_s), (w1_qt, w1_scale, w2_qt, w2_scale)


def _dequant_weight(w_qt, w_scale, orig_shape):
    """mxfp4 -> f32, same formula as test_moe_ep.py's _dequant."""
    wf = fp4_utils.mxfp4_to_f32(w_qt).view(*orig_shape)
    sf = fp4_utils.e8m0_to_f32(w_scale).view(orig_shape[0], orig_shape[1], -1)
    sf = sf.unsqueeze(-1).expand(-1, -1, -1, 32).reshape(*orig_shape)
    return (wf * sf).to(torch.bfloat16)


def _dequant_tokens(tok_quant, scale, hidden_dim):
    """Dequantize dispatched MXFP4/MXFP8 tokens for the Torch reference."""
    n = tok_quant.shape[0]
    if tok_quant.dtype == dtypes.fp4x2:
        values = fp4_utils.mxfp4_to_f32(tok_quant).view(n, hidden_dim)
    elif tok_quant.dtype == dtypes.fp8:
        values = tok_quant.float().view(n, hidden_dim)
    else:
        raise ValueError(
            f"unsupported dispatched dtype for reference: {tok_quant.dtype}"
        )
    sf = fp4_utils.e8m0_to_f32(scale).view(n, hidden_dim // 32)
    sf = sf.unsqueeze(-1).expand(-1, -1, 32).reshape(n, hidden_dim)
    return (values * sf).to(torch.bfloat16)


def _torch_moe_reference(
    x,
    w1,
    w2,
    weights,
    global_ids,
    expert_mask,
    activation,
    *,
    swiglu_limit=0.0,
):
    """Local EP reference for SiLU or SiTUv2 gate/up activation."""
    compute_type = torch.float32
    batch, model_dim = x.shape
    topk = weights.shape[1]
    inter_dim = w2.shape[2]
    local_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
    local_hash[expert_mask == 0] = -1
    local_ids = local_hash[global_ids.long()]
    x_routes = x.to(compute_type).view(batch, 1, model_dim).expand(-1, topk, -1)
    out = torch.zeros((batch, topk, model_dim), dtype=compute_type, device=x.device)
    w1 = w1.to(compute_type)
    w2 = w2.to(compute_type)
    for expert_id in range(w1.shape[0]):
        mask = local_ids == expert_id
        if mask.any():
            gate, up = (x_routes[mask] @ w1[expert_id].transpose(0, 1)).split(
                [inter_dim, inter_dim], dim=-1
            )
            if swiglu_limit:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            hidden = (
                situv2(
                    gate,
                    up,
                    beta=DEFAULT_SITUV2_BETA,
                    linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
                )
                if activation == ActivationType.Situv2
                else F.silu(gate) * up
            )
            out[mask] = hidden @ w2[expert_id].transpose(0, 1)
    return (out * weights.view(batch, topk, 1)).sum(dim=1).to(x.dtype)


def _a8w4_stage1_reference_dtype(
    x_quant,
    w1,
    w2,
    weights,
    activation,
    gate_mode,
    requested="auto",
):
    # Query EP metadata with exactly the same arguments as fused_moe.
    # Do not select BF16/FP32 from TPR alone: a fixed-capacity receive buffer or
    # a tune-table update can change the GEMM1 kernel selected at runtime.
    metadata = get_2stage_cfgs(
        get_padded_M(x_quant.shape[0]),
        x_quant.shape[1],
        w2.shape[2] * 2,
        w1.shape[0],
        weights.shape[1],
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        activation,
        False,
        0,
        0,
        getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False),
        gate_mode,
        is_ep=True,
        has_stage2_bias=False,
        opus_weights_shuffled=(
            getattr(w1, "is_shuffled", False) and getattr(w2, "is_shuffled", False)
        ),
    )
    if requested == "auto":
        if metadata.run_1stage:
            raise ValueError(
                "automatic A8W4 reference currently requires a two-stage kernel"
            )
        if metadata.fuse_quant not in ("", "fp8"):
            raise ValueError(
                f"unsupported A8W4 intermediate quantization: {metadata.fuse_quant}"
            )
        requested = "fp32" if metadata.fuse_quant == "fp8" else "bf16"
    if requested not in ("bf16", "fp32"):
        raise ValueError(f"unsupported reference dtype: {requested}")
    return dtypes.fp32 if requested == "fp32" else dtypes.bf16


def _torch_a8w4_moe_reference(
    x_quant,
    x_scale,
    w1_quant,
    w1_scale,
    w2_quant,
    w2_scale,
    weights,
    global_ids,
    expert_mask,
    activation,
    *,
    stage1_dtype=dtypes.bf16,
    swiglu_limit=0.0,
):
    """AITER's A8W4 reference, including its --ref-dtype intermediate choice."""
    if x_quant.shape[0] == 0:
        return torch.empty(
            (0, w2_quant.shape[1]), dtype=dtypes.bf16, device=x_quant.device
        )
    # The reference path consumes unshuffled weights. Map only global
    # expert IDs owned by this rank; map non-local experts to -1 so the common
    # Torch reference skips them instead of incorrectly using expert 0.
    local_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
    local_hash[expert_mask == 0] = -1
    local_ids = local_hash[global_ids.long()]
    out1_ref = torch_moe_stage1(
        x_quant,
        w1_quant,
        w2_quant,
        weights,
        local_ids,
        dtype=stage1_dtype,
        activation=activation,
        quant_type=QuantType.per_1x32,
        a1_scale=x_scale,
        w1_scale=w1_scale,
        doweight=False,
        swiglu_limit=swiglu_limit,
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    )
    # Match AITER's A8W4 unit test: retain the requested reference dtype
    # through GEMM1 and activation, then apply 1x32 FP8 + E8M0 quantization.
    # The old reference fed floating-point activations directly into GEMM2 and
    # therefore omitted this quantization step.
    a2_quant, a2_scale = per_1x32_f8_scale_f8_quant(
        out1_ref,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
    )
    return torch_moe_stage2(
        a2_quant,
        w1_quant,
        w2_quant,
        weights,
        local_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


def _logits_diff(reference, actual):
    """Cosine-style error metric used by AITER's fused-MoE accuracy tests."""
    reference = reference.double()
    actual = actual.double()
    denominator = (reference.square() + actual.square()).sum()
    if float(denominator) == 0.0:
        return 0.0 if torch.equal(reference, actual) else float("inf")
    similarity = 2 * (reference * actual).sum() / denominator
    return float(1 - similarity)


def _check_a8w4_accuracy(
    reference,
    actual,
    *,
    bs,
    label,
    rank,
    strict_elementwise=False,
):
    """AITER strict_accuracy gate, with explicit all-rank error diagnostics."""
    _collective_require(
        reference.shape == actual.shape,
        f"TPR={bs} {label}: shape mismatch {reference.shape} != {actual.shape}",
    )
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if not _reduce_float(finite, dist.ReduceOp.MIN):
        raise AssertionError(f"TPR={bs} {label}: non-finite reference or output")
    mismatch = checkAllclose(
        reference,
        actual,
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0.05,
        printLog=False,
    )
    diff = actual.float() - reference.float()
    rel_l2 = float(
        torch.linalg.vector_norm(diff)
        / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
    )
    metrics = {
        "rel_l2": rel_l2,
        "mismatch_ratio": float(mismatch),
        "logits_diff": _logits_diff(reference, actual),
        "max_abs": float(diff.abs().max()) if diff.numel() else 0.0,
    }
    # Evaluate pass/fail on every rank before reducing the failure flag,
    # and report rank-wise maxima for all metrics. Do not describe compiled vs.
    # eager consistency as an independent reference-accuracy pass.
    failed = mismatch != 0 and metrics["logits_diff"] > 0.01
    failed = bool(_reduce_float(failed, dist.ReduceOp.MAX))
    metrics = {
        key: _reduce_float(value, dist.ReduceOp.MAX) for key, value in metrics.items()
    }
    if rank == 0:
        print(
            f"[A8W4-accuracy] TPR={bs} check={label} rank_max "
            f"relL2={metrics['rel_l2']:.8f} "
            f"mismatch_ratio={metrics['mismatch_ratio']:.8f} "
            f"logits_diff={metrics['logits_diff']:.8g} "
            f"max_abs={metrics['max_abs']:.8g} "
            f"elementwise_5pct={'PASS' if metrics['mismatch_ratio'] <= 0.05 else 'FAIL'} "
            f"aiter_strict={'FAIL' if failed else 'PASS'}",
            flush=True,
        )
    if failed or (strict_elementwise and metrics["mismatch_ratio"] > 0.05):
        raise AssertionError(
            f"TPR={bs} {label}: A8W4 accuracy failed "
            f"(strict_elementwise={strict_elementwise}): {metrics}"
        )
    return metrics


def _gather_source_inputs(tensors):
    # Gather only the source data before dispatch and transfer its raw
    # bytes through CPU/Gloo. This reference is independent of MORI's received
    # token order, expert IDs, scales, and combine routing.
    world = dist.get_world_size() if dist.is_initialized() else 1
    gathered = []
    for tensor in tensors:
        wire = tensor.detach().contiguous().view(torch.uint8).cpu()
        if world > 1:
            parts = [torch.empty_like(wire) for _ in range(world)]
            dist.all_gather(parts, wire)
            wire = torch.cat(parts, dim=0)
        gathered.append(wire.view(tensor.dtype))
    return tuple(gathered)


def _torch_a8w4_source_reference(
    source_inputs,
    w1_quant,
    w1_scale,
    w2_quant,
    w2_scale,
    expert_mask,
    activation,
    *,
    stage1_dtype,
    chunk_rows=512,
    swiglu_limit=0.0,
):
    """Independent source-token reference for the complete EP output.

    Each rank computes only its own experts. CPU/Gloo sums those contributions
    back to their source rank, without calling MORI dispatch/combine at all.
    The oracle includes the local BF16 output boundary but sums rank partials
    in FP32 before the final BF16 conversion (not MORI's reduction algorithm).
    """
    if chunk_rows <= 0:
        raise ValueError("reference chunk_rows must be positive")
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    tokens = source_inputs[0].shape[0]
    hidden = w2_quant.shape[1]
    device = source_inputs[0].device
    source_q, source_s, source_w, source_ids = _gather_source_inputs(source_inputs)
    local_rows = torch.nonzero(
        expert_mask.cpu()[source_ids.long()].bool().any(dim=1),
        as_tuple=True,
    )[0]
    partials = torch.zeros((world * tokens, hidden), dtype=torch.float32)
    for rows in local_rows.split(chunk_rows):
        if rows.numel() == 0:
            continue
        # Index through uint8 because PyTorch does not implement CPU
        # indexing for every float8/E8M0 representation used here.
        inputs = [
            data.view(torch.uint8)[rows].contiguous().view(data.dtype).to(device)
            for data in (source_q, source_s, source_w, source_ids)
        ]
        q, scales, weights, ids = inputs
        ref = _torch_a8w4_moe_reference(
            q,
            scales,
            w1_quant,
            w1_scale,
            w2_quant,
            w2_scale,
            weights,
            ids,
            expert_mask,
            activation,
            stage1_dtype=stage1_dtype,
            swiglu_limit=swiglu_limit,
        )
        partials[rows] = ref.float().cpu()
    partials = partials.view(world, tokens, hidden)
    # Gloo has no general reduce_scatter support here. Reduce once per
    # source rank to avoid all-gathering every expert rank's complete output;
    # only one reference shard remains resident in GPU memory.
    if world > 1:
        for source_rank in range(world):
            dist.reduce(partials[source_rank], dst=source_rank, op=dist.ReduceOp.SUM)
    return partials[rank].to(device=device, dtype=dtypes.bf16)


def _build_expert_mask(experts, local_expert_start, local_expert_end, device):
    expert_mask = torch.zeros((experts + 1,), dtype=dtypes.i32, device=device)
    expert_mask[local_expert_start:local_expert_end] = 1
    expert_mask[-1] = 0  # fake/padding expert id, never local
    return expert_mask


def _run_one_bs(
    bs,
    op,
    state,
    run,
    rank,
    perf_out,
):
    x_fp4, x_scale = state.x_quant, state.x_scale
    topk_weights, topk_ids = state.topk_weights, state.topk_ids
    w1_a, w2_a = state.w1, state.w2
    w1_qt, w1_scale = state.w1_reference, state.w1_reference_scale
    w2_qt, w2_scale = state.w2_reference, state.w2_reference_scale
    expert_mask, local_experts = state.expert_mask, state.local_experts
    model_dim, inter_dim = state.model_dim, state.inter_dim
    quant, activation, world_size = state.quant, state.activation, state.world_size
    iters, stat_iters = run.iterations, run.statistic_iterations
    accuracy_max_bs, staged_only = run.accuracy_max_batch, run.staged_only
    torch_profiler_dir = run.profiler_dir
    torch_compile_cudagraph = run.compile_cudagraph
    profile_warmup_iters, profile_iters = PROFILE_WARMUP_ITERS, PROFILE_ITERS
    rtol = A4W4_RTOL
    ref_dtype = "auto"
    # Match fused_moe's A8W4 strict_accuracy contract.  The elementwise 5%
    # metric remains useful in logs, but is intentionally not an independent
    # rejection gate.
    strict_elementwise = False
    end_to_end_reference = quant == "a8w4" and bs <= accuracy_max_bs
    x_fp4_bs = x_fp4[:bs].contiguous()
    x_scale_bs = x_scale[:bs].contiguous()
    topk_weights_bs = topk_weights[:bs].contiguous()
    topk_ids_bs = topk_ids[:bs].contiguous()

    out = None
    if not staged_only:
        # Public TestWideEpMoe contract: the operator owns the complete
        # inter-node dispatch -> fused_moe -> combine sequence.
        out = op.forward_prequant(
            x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs
        ).clone()
        torch.cuda.synchronize()
        _collective_require(
            out.shape == (bs, model_dim),
            f"TestWideEpMoe output shape {tuple(out.shape)} != {(bs, model_dim)}",
        )
        _collective_require(
            torch.isfinite(out.float()).all().item(),
            "TestWideEpMoe output has non-finite values",
        )

        # The public-call check and staged diagnostic are distinct MORI epochs.
        _barrier()

    # Diagnostic-only staged call through the private backend. This preserves
    # per-stage profiling without changing TestWideEpMoe's public forward API.
    backend = op
    debug_wide_ep = os.environ.get("AITER_DEBUG_WIDE_EP", "0") == "1"
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic dispatch start", flush=True)
    dispatched = backend.dispatch_prequant(
        x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs
    )
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic dispatch complete", flush=True)
    recv_tok_fp4 = dispatched.tokens
    recv_wts = dispatched.weights
    recv_scale = dispatched.scales
    recv_idx = dispatched.expert_ids
    recv_num_token = dispatched.num_tokens
    torch.cuda.synchronize()
    total_recv = int(recv_num_token[0].item())

    # Capture the two compute launch callables so GEMM1/GEMM2 can be timed
    # independently on the actual post-MORI routing distribution.
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    kernel_calls = []
    fused_moe_module.kernel_bench_callable = kernel_calls
    try:
        moe_out = backend.fused_moe(dispatched)
    finally:
        fused_moe_module.kernel_bench_callable = None
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic fused_moe complete", flush=True)
        print(
            f"[EP16-debug rank={rank}] combine ABI "
            f"recv_shape={tuple(recv_tok_fp4.shape)} recv_stride={recv_tok_fp4.stride()} "
            f"moe_shape={tuple(moe_out.shape)} moe_stride={moe_out.stride()} "
            f"moe_contiguous={moe_out.is_contiguous()} total_recv={total_recv} "
            f"source_tokens={bs}",
            flush=True,
        )
    if staged_only:
        torch.cuda.synchronize()
        if debug_wide_ep:
            print(f"[EP16-debug rank={rank}] fused_moe GPU sync complete", flush=True)

    # combine()'s indices/weights must be THIS rank's own [tokens, topk]
    # routing passed to dispatch() -- NOT dispatch()'s returned recv_idx/
    # recv_wts (ROCm/mori#475). weights=None: fused_moe already applied
    # topk weighting in stage2 (same convention as
    # test_dispatch_combine_internode.py's run_combine).
    if staged_only and os.environ.get("AITER_DEBUG_COMBINE_WITH_WEIGHTS", "0") == "1":
        # ABI diagnostic only: current MORI's official V1LL test passes the
        # dispatch-returned weights into combine. Production fused_moe already
        # applies them, so this path must not be used for numerical validation.
        combine_out, _combine_out_wts = backend.op.combine(
            moe_out, dispatched.weights, dispatched.source_topk_ids
        )
        dispatched.consumed = True
    else:
        combine_out, _combine_out_wts = backend.combine(moe_out, dispatched)
    torch.cuda.synchronize()
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic combine complete", flush=True)
    # Captured atomic GEMM2 calls below intentionally reuse their output buffer
    # for timing and therefore accumulate into ``moe_out``.  Preserve the
    # single-execution result before benchmarking for correctness checks.
    moe_out_correctness = moe_out.clone() if bs <= accuracy_max_bs else None
    diagnostic_out = combine_out[:bs]
    if out is not None and bs <= accuracy_max_bs:
        # GEMM2 uses atomic accumulation, so two otherwise identical launches
        # are not bitwise deterministic. A8W4 follows AITER's existing
        # checkAllclose + logits_diff contract instead of a hard max-delta gate.
        if quant == "a8w4":
            _check_a8w4_accuracy(
                out,
                diagnostic_out,
                bs=bs,
                label="public_vs_staged_consistency",
                rank=rank,
                strict_elementwise=strict_elementwise,
            )
        else:
            _collective_require(
                torch.allclose(out, diagnostic_out, rtol=1e-2, atol=2.5e-1),
                f"bs={bs} public/staged A4W4 outputs differ",
            )

    def _time_captured_kernel(call):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
        for start, end in zip(starts, ends):
            start.record()
            call()
            end.record()
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / 20

    kernel_us = (
        {}
        if staged_only
        else {name: _time_captured_kernel(call) * 1000 for name, call in kernel_calls}
    )
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] captured GEMM timing complete", flush=True)
    gemm1_us_local = kernel_us.get("stage1", 0.0)
    gemm2_us_local = kernel_us.get("stage2", 0.0)
    gemm1_us = _reduce_float(gemm1_us_local, dist.ReduceOp.SUM) / world_size
    gemm2_us = _reduce_float(gemm2_us_local, dist.ReduceOp.SUM) / world_size
    gemm_total_us_local = gemm1_us_local + gemm2_us_local
    gemm_total_us = gemm1_us + gemm2_us

    # ---- Quantization-aware local MoE and optional independent source reference ----
    rel_l2 = -1.0
    source_reference = None
    make_source_reference = None
    if bs <= accuracy_max_bs:
        if debug_wide_ep:
            print(f"[EP16-debug rank={rank}] torch reference start", flush=True)
        # The common A8W4 reference dequantizes weights lazily per expert;
        # A4W4 retains the direct dequantized Torch reference.
        if quant != "a8w4":
            recv_tok_bf16 = _dequant_tokens(recv_tok_fp4, recv_scale, model_dim)
            w1_deq = _dequant_weight(
                w1_qt, w1_scale, (local_experts, 2 * inter_dim, model_dim)
            )
            w2_deq = _dequant_weight(
                w2_qt, w2_scale, (local_experts, model_dim, inter_dim)
            )
        if quant == "a8w4":
            stage1_ref_dtype = _a8w4_stage1_reference_dtype(
                recv_tok_fp4,
                w1_a,
                w2_a,
                recv_wts,
                activation,
                op.gate_mode.value,
                ref_dtype,
            )
            if rank == 0:
                print(
                    f"[A8W4-reference] TPR={bs} capacity={recv_tok_fp4.shape[0]} "
                    f"requested={ref_dtype} stage1_dtype={stage1_ref_dtype} "
                    f"swiglu_limit={op.swiglu_limit}",
                    flush=True,
                )
            ref_moe_out = _torch_a8w4_moe_reference(
                recv_tok_fp4[:total_recv],
                recv_scale[:total_recv],
                w1_qt,
                w1_scale,
                w2_qt,
                w2_scale,
                recv_wts[:total_recv],
                recv_idx[:total_recv],
                expert_mask,
                activation,
                stage1_dtype=stage1_ref_dtype,
                swiglu_limit=op.swiglu_limit,
            )
        else:
            ref_moe_out = _torch_moe_reference(
                recv_tok_bf16[:total_recv],
                w1_deq,
                w2_deq,
                recv_wts[:total_recv],
                recv_idx[:total_recv],
                expert_mask,
                activation,
                swiglu_limit=op.swiglu_limit,
            )
        if quant == "a8w4":
            accuracy = _check_a8w4_accuracy(
                ref_moe_out,
                moe_out_correctness[:total_recv],
                bs=bs,
                label=f"local_moe_vs_quantized_reference_{stage1_ref_dtype}",
                rank=rank,
                strict_elementwise=strict_elementwise,
            )
            rel_l2 = accuracy["rel_l2"]
        else:
            rel_l2 = float(
                torch.linalg.vector_norm(
                    (moe_out_correctness[:total_recv] - ref_moe_out).float()
                )
                / torch.linalg.vector_norm(ref_moe_out.float())
            )
            rel_l2 = _reduce_float(rel_l2, dist.ReduceOp.MAX)
            if rel_l2 >= rtol:
                raise AssertionError(
                    f"bs={bs} moe relL2={rel_l2:.6f} exceeds rtol={rtol}"
                )
        _collective_require(
            diagnostic_out.shape == (bs, model_dim),
            f"combine output shape {tuple(diagnostic_out.shape)} != {(bs, model_dim)}",
        )
        _collective_require(
            torch.isfinite(diagnostic_out.float()).all().item(),
            "combine output has non-finite values",
        )
        if end_to_end_reference:

            def make_source_reference(inputs):
                return _torch_a8w4_source_reference(
                    inputs,
                    w1_qt,
                    w1_scale,
                    w2_qt,
                    w2_scale,
                    expert_mask,
                    activation,
                    stage1_dtype=stage1_ref_dtype,
                    swiglu_limit=op.swiglu_limit,
                )

            source_reference = make_source_reference(
                (x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs)
            )
            _check_a8w4_accuracy(
                source_reference,
                diagnostic_out,
                bs=bs,
                label="staged_full_pipeline_vs_source",
                rank=rank,
                strict_elementwise=strict_elementwise,
            )
            if out is not None:
                _check_a8w4_accuracy(
                    source_reference,
                    out,
                    bs=bs,
                    label="public_full_pipeline_vs_source",
                    rank=rank,
                    strict_elementwise=strict_elementwise,
                )
        if debug_wide_ep:
            print(f"[EP16-debug rank={rank}] torch reference complete", flush=True)

    if staged_only:
        if rank == 0:
            print(
                f"[EP16-staged-only] quant={quant} bs={bs} relL2={rel_l2:.6f} "
                f"reference={'CHECKED' if bs <= accuracy_max_bs else 'SKIPPED'}",
                flush=True,
            )
        return

    # ---- logical GEMM row count: (received row, local expert slot) pairs ----
    # actually computed by fused_moe's grouped GEMM -- exact, not an estimate.
    recv_idx_flat = recv_idx[:total_recv].reshape(-1)
    local_mask = expert_mask[recv_idx_flat.long()] == 1
    local_hits = int(local_mask.sum().item())
    # number of this rank's local experts that actually received >=1 token --
    # matches aiter's own MoE-GEMM benchmark convention (bench_moe_gemm_a4w4_cudagraph.py's
    # `routed = int((rdata.expt_data.hist > 0).sum())`): weight bytes should only be
    # counted for experts that were actually touched, not every local expert, since at
    # small bs some local experts may see zero tokens.
    active_local_experts = int(torch.unique(recv_idx_flat[local_mask]).numel())

    # Emit the per-rank payload metadata used to reproduce MORI's
    # bandwidth convention from rocprof kernel durations.
    print(
        f"[EP16-rank-meta] bs={bs} rank={rank} total_recv={total_recv} "
        f"local_hits={local_hits} active_local_experts={active_local_experts}",
        flush=True,
    )

    _barrier()

    # ---- perf: 3 timing brackets (dispatch / moe / combine) ----
    # CAVEAT (found via rocprofv3 ground truth, not yet resolved): these
    # torch.cuda.Event brackets do NOT necessarily match each op's real GPU
    # completion time. A profiled run (rocprofv3 --kernel-trace) on this exact
    # pipeline showed the real EpDispatchInterNodeV1Kernel/EpCombineInterNodeV1Kernel
    # durations can be far larger (and far more variable, up to ~150ms) than what
    # the bracket here measures (sub-ms), while the real gemm1/gemm2/quant/sorting
    # kernels inside fused_moe summed to only ~0.1ms per call versus a multi-ms
    # "moe" bracket -- i.e. the bracket boundaries likely don't line up with true
    # kernel start/end for these async/persistent-style mori kernels. Treat
    # dispatch_ms/moe_ms/combine_ms (and the derived GB/s/TFLOPS below) as a
    # measure of this script's host-observed critical path, not as validated
    # per-op GPU kernel time, until this is root-caused (check mori's
    # dispatch_combine.py for which stream these kernels run on and whether
    # they're fire-and-forget/polling rather than synchronous per call).
    # Use four independent events per iteration so one iteration's
    # combine end event is never reused as the next iteration's start event.
    # Keep the measured loop barrier-free: per-iteration host/SHMEM barriers
    # perturb V1LL's steady-state pipeline and amplify rank-arrival skew.
    n_events = 4 * iters
    events = [torch.cuda.Event(enable_timing=True) for _ in range(n_events)]
    torch.cuda.synchronize()
    dist.barrier()
    for i in range(iters):
        event_base = 4 * i
        events[event_base].record()
        dispatched = backend.dispatch_prequant(
            x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs
        )
        events[event_base + 1].record()
        moe_out = backend.fused_moe(dispatched)
        events[event_base + 2].record()
        backend.combine(moe_out, dispatched)
        events[event_base + 3].record()
    torch.cuda.synchronize()

    # Discard the first (iters - stat_iters) rounds as JIT/cache warmup; average
    # only the trailing stat_iters rounds (user-requested: iters=100, stat over
    # the last 20).
    keep = max(1, min(stat_iters, iters))
    dispatch_ms = [events[4 * i].elapsed_time(events[4 * i + 1]) for i in range(iters)][
        -keep:
    ]
    moe_ms = [events[4 * i + 1].elapsed_time(events[4 * i + 2]) for i in range(iters)][
        -keep:
    ]
    combine_ms = [
        events[4 * i + 2].elapsed_time(events[4 * i + 3]) for i in range(iters)
    ][-keep:]

    # mean, best-rank (min), worst-rank (max) across ranks: dispatch/combine/moe
    # are collective -- the group's real wall-clock latency is bounded by
    # whichever rank is slowest (stragglers are common under EP, since random
    # routing gives ranks uneven local_hits). mean alone hides that spread.
    dispatch_local = sum(dispatch_ms) / keep
    moe_local = sum(moe_ms) / keep
    combine_local = sum(combine_ms) / keep
    dispatch_avg = _reduce_float(dispatch_local, dist.ReduceOp.SUM) / world_size
    dispatch_min = _reduce_float(dispatch_local, dist.ReduceOp.MIN)
    dispatch_max = _reduce_float(dispatch_local, dist.ReduceOp.MAX)
    moe_avg = _reduce_float(moe_local, dist.ReduceOp.SUM) / world_size
    moe_min = _reduce_float(moe_local, dist.ReduceOp.MIN)
    moe_max = _reduce_float(moe_local, dist.ReduceOp.MAX)
    combine_avg = _reduce_float(combine_local, dist.ReduceOp.SUM) / world_size
    combine_min = _reduce_float(combine_local, dist.ReduceOp.MIN)
    combine_max = _reduce_float(combine_local, dist.ReduceOp.MAX)
    total_recv_avg = _reduce_float(float(total_recv), dist.ReduceOp.SUM) / world_size
    local_hits_avg = _reduce_float(float(local_hits), dist.ReduceOp.SUM) / world_size
    local_hits_max = _reduce_float(float(local_hits), dist.ReduceOp.MAX)
    active_experts_avg = (
        _reduce_float(float(active_local_experts), dist.ReduceOp.SUM) / world_size
    )
    active_experts_max = _reduce_float(float(active_local_experts), dist.ReduceOp.MAX)

    # ---- bandwidth / throughput conversions ----
    # dispatch/combine move total_recv rows across the fabric (XGMI+RDMA
    # combined, same convention as test_dispatch_combine_internode.py's
    # disp_total_bytes/comb_total_bytes: total_recv_num_token * hidden * elem_size).
    # GB/s reported for both mean (typical) and max (worst-rank/bottleneck) time.
    # Match AITER's tuning CSV convention: fp4x2 is accounted as one storage
    # byte per logical matrix element. This is an effective-BW convention,
    # not the physical nibble payload size.
    fp4_bytes_per_elem = 1.0
    bf16_bytes_per_elem = 2.0
    dispatch_bytes_local = total_recv * model_dim * fp4_bytes_per_elem
    combine_bytes_local = total_recv * model_dim * bf16_bytes_per_elem

    def _gbps(nbytes, ms):
        return nbytes / 1e9 / (ms / 1e3) if ms > 0 else 0.0

    # Match MORI's aggregation: compute payload/time for every rank and
    # iteration first, then average. Do not divide average payload by average
    # latency because mean(bytes/time) is not bytes_mean/time_mean.
    dispatch_gbps_local = (
        sum(_gbps(dispatch_bytes_local, ms) for ms in dispatch_ms) / keep
    )
    combine_gbps_local = sum(_gbps(combine_bytes_local, ms) for ms in combine_ms) / keep
    dispatch_gbps = _reduce_float(dispatch_gbps_local, dist.ReduceOp.SUM) / world_size
    dispatch_gbps_best = _reduce_float(dispatch_gbps_local, dist.ReduceOp.MAX)
    dispatch_gbps_worst = _reduce_float(dispatch_gbps_local, dist.ReduceOp.MIN)
    combine_gbps = _reduce_float(combine_gbps_local, dist.ReduceOp.SUM) / world_size
    combine_gbps_best = _reduce_float(combine_gbps_local, dist.ReduceOp.MAX)
    combine_gbps_worst = _reduce_float(combine_gbps_local, dist.ReduceOp.MIN)

    # Profile the public TestWideEpMoe forward API. In compile mode deliberately
    # wrap that public interface directly rather than introducing another
    # fused_moe facade in the implementation.
    profiled_call = op.forward_prequant
    profile_args = (x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs)
    pipeline_kind = "eager_full_pipeline"
    if torch_compile_cudagraph:
        op.prepare_torch_compile(*profile_args)
        profiled_call = torch.compile(op.forward_prequant, backend="cudagraphs")
        pipeline_kind = "torch_compile_cudagraph_test_wide_ep_forward"

        # Validate the compiled public callable itself.  The eager correctness
        # checks above do not prove that graph capture/replay preserves output.
        compiled_out = profiled_call(*profile_args).clone()
        torch.cuda.synchronize()
        _collective_require(
            compiled_out.shape == out.shape,
            f"compiled output shape {tuple(compiled_out.shape)} != {tuple(out.shape)}",
        )
        _collective_require(
            torch.isfinite(compiled_out.float()).all().item(),
            "compiled output has non-finite values",
        )
        compiled_rel_l2 = float(
            torch.linalg.vector_norm((compiled_out - out).float())
            / torch.linalg.vector_norm(out.float()).clamp_min(1e-30)
        )
        compiled_rel_l2 = _reduce_float(compiled_rel_l2, dist.ReduceOp.MAX)
        if quant == "a8w4":
            consistency = _check_a8w4_accuracy(
                out,
                compiled_out,
                bs=bs,
                label="first_compiled_vs_eager_consistency",
                rank=rank,
                strict_elementwise=strict_elementwise,
            )
            compiled_mismatch = consistency["mismatch_ratio"]
            compiled_logits_diff = consistency["logits_diff"]
            compile_failed = False
        else:
            compiled_mismatch = 0.0
            compiled_logits_diff = 0.0
            compile_failed = compiled_rel_l2 >= rtol
        if compile_failed:
            raise AssertionError(
                f"bs={bs} compiled/eager relL2={compiled_rel_l2:.6f} "
                f"mismatch_ratio={compiled_mismatch:.6f} "
                f"logits_diff={compiled_logits_diff:.6f}"
            )
        _barrier()
        if rank == 0:
            print(
                f"[EP16-torch-compile] bs={bs} output check PASS "
                f"relL2={compiled_rel_l2:.6f} "
                f"mismatch_ratio={compiled_mismatch:.6f} "
                f"logits_diff={compiled_logits_diff:.6f}",
                flush=True,
            )
        if source_reference is not None:
            _check_a8w4_accuracy(
                source_reference,
                compiled_out,
                bs=bs,
                label="first_compiled_full_pipeline_vs_source",
                rank=rank,
                strict_elementwise=strict_elementwise,
            )
    if torch_profiler_dir:
        from torch.profiler import ProfilerActivity, profile, record_function

        for _ in range(profile_warmup_iters):
            profiled_call(*profile_args)
        torch.cuda.synchronize()
        _barrier()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for profile_iter in range(profile_iters):
                with record_function(
                    f"megamoe_ep16_{pipeline_kind}_bs{bs}_iter{profile_iter}"
                ):
                    profiled_call(*profile_args)
            torch.cuda.synchronize()
        os.makedirs(torch_profiler_dir, exist_ok=True)
        prof.export_chrome_trace(
            os.path.join(torch_profiler_dir, f"rank{rank}_bs{bs}.json")
        )
        if rank == 0:
            print(
                f"[EP16-torch-profiler] bs={bs} traces={torch_profiler_dir} "
                f"iters={profile_iters} pipeline={pipeline_kind}",
                flush=True,
            )

    # MoE: exact logical-M FLOPs (grouped GEMM1 gate+up, GEMM2 down) -> TFLOPS.
    #
    # Memory bandwidth: "effective bandwidth = minimum required bytes / time"
    # (A + B + C, each counted once per GEMM) -- the same convention aiter's own
    # GEMM/MoE-GEMM benchmarks use (op_tests/test_gemm_a4w4.py: `(x.nbytes + w.nbytes)
    # / us`; op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py: mem_read(x,w,scales)
    # + mem_write(out); op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_cudagraph.py:
    # per-GEMM activation + local weight bytes + output,
    # with moe1/moe2 summed for the "total" row). fused_moe here is confirmed 2-stage
    # (its own log prints "using 2stage default" -- gemm1 gate+up and gemm2 down are
    # two separate kernel launches), so the intermediate hidden activation genuinely
    # round-trips through HBM: written once by gemm1 (its C), read back once by gemm2
    # (its A) -- not a guess, both gemms' bytes are summed the way aiter's own bench
    # sums moe1_bytes + moe2_bytes for "total":
    #   gemm1: A=recv tokens (model_dim) + B=all local w1 + C=hidden (inter_dim)
    #   gemm2: A=hidden (inter_dim) + B=all local w2 + C=output tokens (model_dim)
    # Weight bytes counted ONCE per GEMM (not per M-tile).  Match the tune-table
    # convention by counting every local expert weight, independent of the
    # random routing distribution for this batch.
    def _moe_flops(hits):
        return (
            2.0 * hits * model_dim * (2 * inter_dim)
            + 2.0 * hits * inter_dim * model_dim
        )

    def _moe_bytes(received_tokens):
        # Same aggregate convention as gemm_moe_tune.py: input + both local
        # weight matrices + final output, divided by GEMM1+GEMM2 duration.
        return (
            received_tokens * model_dim * fp4_bytes_per_elem
            + local_experts * (2 * inter_dim) * model_dim * fp4_bytes_per_elem
            + local_experts * model_dim * inter_dim * fp4_bytes_per_elem
            + received_tokens * model_dim * bf16_bytes_per_elem
        )

    moe_tflops = (
        _moe_flops(local_hits_avg) / (gemm_total_us * 1e6) if gemm_total_us > 0 else 0.0
    )
    local_moe_tflops = (
        _moe_flops(local_hits) / (gemm_total_us_local * 1e6)
        if gemm_total_us_local > 0
        else 0.0
    )
    moe_tflops_best = _reduce_float(local_moe_tflops, dist.ReduceOp.MAX)
    moe_tflops_worst = _reduce_float(local_moe_tflops, dist.ReduceOp.MIN)
    moe_gbps = _gbps(_moe_bytes(total_recv_avg), gemm_total_us / 1000)
    local_moe_gbps = _gbps(_moe_bytes(total_recv), gemm_total_us_local / 1000)
    moe_gbps_best = _reduce_float(local_moe_gbps, dist.ReduceOp.MAX)
    moe_gbps_worst = _reduce_float(local_moe_gbps, dist.ReduceOp.MIN)

    if rank == 0:
        print(
            f"[EP16-{quant}] bs={bs} total_recv~{total_recv_avg:.0f} local_hits~{local_hits_avg:.0f} "
            f"relL2={rel_l2:.6f} "
            f"(criterion={'aiter_a8w4_strict' if quant == 'a8w4' else f'relL2<{rtol}'}, "
            f"{'checked' if bs <= accuracy_max_bs else 'skipped'}, "
            f"stat over last {keep}/{iters} iters)\n"
            f"  dispatch: {dispatch_avg:.4f}/{dispatch_min:.4f}/{dispatch_max:.4f}ms mean/best/worst  "
            f"{dispatch_gbps:.2f}/{dispatch_gbps_best:.2f}/{dispatch_gbps_worst:.2f} GB/s mean/best/worst\n"
            f"  moe     : {moe_avg:.4f}/{moe_min:.4f}/{moe_max:.4f}ms mean/best/worst  "
            f"{moe_tflops:.2f}/{moe_tflops_best:.2f}/{moe_tflops_worst:.2f} TFLOPS mean/best/worst  "
            f"{moe_gbps:.2f}/{moe_gbps_best:.2f}/{moe_gbps_worst:.2f} GB/s mean/best/worst\n"
            f"  combine : {combine_avg:.4f}/{combine_min:.4f}/{combine_max:.4f}ms mean/best/worst  "
            f"{combine_gbps:.2f}/{combine_gbps_best:.2f}/{combine_gbps_worst:.2f} GB/s mean/best/worst\n"
            f"  kernels : gemm1={gemm1_us:.2f}us gemm2={gemm2_us:.2f}us "
            f"sum={gemm1_us + gemm2_us:.2f}us",
            flush=True,
        )
        if perf_out:
            record = {
                "category": f"ep16_{quant}_moe",
                "params": {
                    "world_size": world_size,
                    "bs": bs,
                    "experts": local_experts * world_size,
                    "local_experts": local_experts,
                    "topk": topk_ids.shape[1],
                    "model_dim": model_dim,
                    "inter_dim": inter_dim,
                    "quant_type": f"per_1x32_{quant}",
                },
                "stat_iters": keep,
                "total_iters": iters,
                "metrics": {
                    "dispatch_avg_ms": round(dispatch_avg, 4),
                    "dispatch_best_ms": round(dispatch_min, 4),
                    "dispatch_worst_ms": round(dispatch_max, 4),
                    "dispatch_gbps_mean": round(dispatch_gbps, 2),
                    "dispatch_gbps_best": round(dispatch_gbps_best, 2),
                    "dispatch_gbps_worst": round(dispatch_gbps_worst, 2),
                    "moe_avg_ms": round(moe_avg, 4),
                    "moe_best_ms": round(moe_min, 4),
                    "moe_worst_ms": round(moe_max, 4),
                    "moe_tflops_mean": round(moe_tflops, 2),
                    "moe_tflops_best": round(moe_tflops_best, 2),
                    "moe_tflops_worst": round(moe_tflops_worst, 2),
                    "moe_gbps_mean": round(moe_gbps, 2),
                    "moe_gbps_best": round(moe_gbps_best, 2),
                    "moe_gbps_worst": round(moe_gbps_worst, 2),
                    "combine_avg_ms": round(combine_avg, 4),
                    "combine_best_ms": round(combine_min, 4),
                    "combine_worst_ms": round(combine_max, 4),
                    "combine_gbps_mean": round(combine_gbps, 2),
                    "combine_gbps_best": round(combine_gbps_best, 2),
                    "combine_gbps_worst": round(combine_gbps_worst, 2),
                    "total_recv": round(total_recv_avg, 1),
                    "local_hits_mean": round(local_hits_avg, 1),
                    "local_hits_max": round(local_hits_max, 1),
                    "active_experts_mean": round(active_experts_avg, 1),
                    "active_experts_max": round(active_experts_max, 1),
                    "moe_rel_l2": None if rel_l2 < 0 else round(rel_l2, 6),
                    "gemm1_us": round(gemm1_us, 2),
                    "gemm2_us": round(gemm2_us, 2),
                },
                "ts": time.time(),
            }
            os.makedirs(os.path.dirname(os.path.abspath(perf_out)), exist_ok=True)
            with open(perf_out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")


def run_ep16(
    local_rank,
    run,
    gpu_per_node,
    node_rank,
    num_nodes,
):
    world_size = num_nodes * gpu_per_node
    rank = node_rank * gpu_per_node + local_rank
    device = _setup_dist(rank, world_size, local_rank)
    perf_out = os.environ.get("MORI_PERF_OUT") if rank == 0 else None
    op = None

    try:
        network = NETWORKS[run.model]
        experts = network["experts"]
        model_dim = network["model_dim"]
        inter_dim = network["inter_dim"]
        topk = network["topk"]
        quant = network["quant"]
        activation = network["activation"]
        gate_mode = network["gate_mode"]
        swiglu_limit = network["swiglu_limit"]
        if experts % world_size != 0:
            raise ValueError(
                f"experts={experts} must be divisible by world_size={world_size}"
            )
        local_experts = experts // world_size
        local_expert_start = rank * local_experts
        local_expert_end = local_expert_start + local_experts

        max_bs = max(run.batch_sizes)

        # ---- Build all tensors up front, before any dispatch/moe/combine/timing ----
        x, topk_weights, topk_ids = _make_local_inputs(
            max_bs,
            model_dim,
            experts,
            topk,
            rank,
            run.seed,
            device,
        )
        (w1_a, w1_s, w2_a, w2_s), (w1_qt, w1_scale, w2_qt, w2_scale) = (
            _quantize_local_weights(
                model_dim, inter_dim, local_experts, rank, run.seed, device, quant
            )
        )
        expert_mask = _build_expert_mask(
            experts, local_expert_start, local_expert_end, device
        )

        if rank == 0:
            print(
                f"[TestWideEpMoe] model={run.model} quant={quant} "
                f"activation={activation} routing='random' swiglu_limit={swiglu_limit}",
                flush=True,
            )

        from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

        quant_mode = "fp4" if quant == "a4w4" else "fp8"
        x_fp4, x_scale = per_1x32_mx_quant(x, quant_mode=quant_mode)

        op = TestWideEpMoe(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant=quant,
            activation=activation,
            gate_mode=gate_mode,
            swiglu_limit=swiglu_limit,
            w1=w1_a,
            w1_scale=w1_s,
            w2=w2_a,
            w2_scale=w2_s,
            max_tok_per_rank=max_bs,
        )
        state = TestState(
            x_quant=x_fp4,
            x_scale=x_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1_a,
            w1_scale=w1_s,
            w2=w2_a,
            w2_scale=w2_s,
            w1_reference=w1_qt,
            w1_reference_scale=w1_scale,
            w2_reference=w2_qt,
            w2_reference_scale=w2_scale,
            expert_mask=expert_mask,
            local_experts=local_experts,
            model_dim=model_dim,
            inter_dim=inter_dim,
            quant=quant,
            activation=activation,
            world_size=world_size,
        )

        _barrier()

        for bs in run.batch_sizes:
            _run_one_bs(bs, op, state, run, rank, perf_out)
    finally:
        if op is not None:
            _TEST_WIDE_EP_INSTANCES.pop(op.owner_id, None)
        _cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        choices=sorted(NETWORKS),
        default="kimi",
        help="built-in model/quantization preset",
    )
    parser.add_argument("--bs-list", default="128,512,1024,2048,4096")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--stat-iters",
        type=int,
        default=20,
        help="average over only the trailing N of --iters rounds (discard the rest as warmup)",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--accuracy-max-bs", type=int, default=512)
    parser.add_argument(
        "--torch-profiler-dir",
        default=None,
        help="export one full-pipeline torch.profiler Chrome trace per rank and BS",
    )
    parser.add_argument(
        "--torch-compile-cudagraph",
        action="store_true",
        help=(
            "apply torch.compile(backend='cudagraphs') directly to the public "
            "TestWideEpMoe.forward_prequant interface"
        ),
    )
    parser.add_argument(
        "--staged-only",
        action="store_true",
        help="run one synchronized dispatch/fused_moe/combine diagnostic and stop",
    )
    args = parser.parse_args()

    # This is a manually launched two-node test.  The generic multigpu CI
    # runner executes every script directly on one node, so leave successfully
    # unless the outer one-process-per-node torchrun rendezvous is present.
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ:
        print("SKIP: test_wide_ep_moe requires a two-node torchrun launch")
        return

    bs_list = [int(v) for v in args.bs_list.split(",") if v]
    if not bs_list or min(bs_list) <= 0:
        raise ValueError("--bs-list must contain positive integers")
    gpu_per_node = int(os.environ.get("GPU_PER_NODE", GPU_PER_NODE_DEFAULT))
    num_nodes = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["RANK"])
    if num_nodes != 2 or gpu_per_node != 8 or node_rank not in (0, 1):
        print(
            "SKIP: test_wide_ep_moe requires WORLD_SIZE=2, RANK=0/1, "
            "and GPU_PER_NODE=8"
        )
        return
    run = RunConfig(
        model=args.model_config,
        batch_sizes=tuple(bs_list),
        iterations=args.iters,
        statistic_iterations=args.stat_iters,
        seed=args.seed,
        accuracy_max_batch=args.accuracy_max_bs,
        profiler_dir=args.torch_profiler_dir,
        compile_cudagraph=args.torch_compile_cudagraph,
        staged_only=args.staged_only,
    )

    torch.multiprocessing.spawn(
        run_ep16,
        args=(
            run,
            gpu_per_node,
            node_rank,
            num_nodes,
        ),
        nprocs=gpu_per_node,
        join=True,
    )


if __name__ == "__main__":
    main()
