# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import GateMode, moe_sorting
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    invert_sorted_ids,
    sorted_sum,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.common import _ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


@dataclass
class Config:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    use_prefill: bool
    down_path: str = "default"
    down_output_padding_bytes: int = 0
    use_batch1_algorithm: bool = False

    def __post_init__(self):
        if self.down_path not in ("default", "1x4_64x256", "8x1_compact"):
            raise ValueError(f"Unsupported Down path: {self.down_path}")
        if self.down_output_padding_bytes not in (0, 32, 64, 128):
            raise ValueError("Down output padding must be 0, 32, 64 or 128 bytes")
        if self.down_path == "default" and self.down_output_padding_bytes != 0:
            raise ValueError("The default Down path does not support output padding")
        if self.down_path != "default" and not self.use_prefill:
            raise ValueError("Specialized Down paths require prefill")
        if self.use_batch1_algorithm and (
            self.use_prefill or self.down_path != "default"
        ):
            raise ValueError(
                "Direct decoding cannot use prefill or specialized Down paths"
            )

    def to_string(self):
        return (
            f"{self.BLOCK_M}_{self.BLOCK_N}_{self.BLOCK_K}_"
            f"{self.use_prefill}_{self.use_batch1_algorithm}:"
            f"{self.down_path}:{self.down_output_padding_bytes}"
        )

    @classmethod
    def from_string(cls, data: str):
        tile, down_path, padding = data.split(":")
        block_m, block_n, block_k, use_prefill, use_batch1_algorithm = tile.split("_")
        if use_prefill not in ("True", "False") or use_batch1_algorithm not in (
            "True",
            "False",
        ):
            raise ValueError(f"Invalid algorithm flags: {tile}")
        return cls(
            int(block_m),
            int(block_n),
            int(block_k),
            use_prefill == "True",
            down_path,
            int(padding),
            use_batch1_algorithm=use_batch1_algorithm == "True",
        )

    def unsupported_reason(self, problem: "_Problem") -> str | None:
        if not self.use_prefill:
            if self.use_batch1_algorithm:
                if not 2 <= problem.batch <= 8:
                    return "direct decoding requires 2 to 8 tokens"
                if (self.BLOCK_M, self.BLOCK_N, self.BLOCK_K) != (16, 16, 16):
                    return "direct decoding uses the fixed 16_16_16 config"
            else:
                if not 1 <= problem.batch <= 256:
                    return "decoding requires 1 to 256 tokens"
                if self.BLOCK_M != 16 or self.BLOCK_N not in (16, 64, 128):
                    return "decoding requires BM16 and BN64 or BN128"
                if self.BLOCK_K not in (16, 64):
                    return "decoding requires BK64"
                if problem.batch == 1 and (self.BLOCK_N, self.BLOCK_K) != (16, 16):
                    return "batch1 uses the fixed 16_16_16 config"
            if problem.quant_type == "mxfp4":
                if not self.use_batch1_algorithm and self.BLOCK_N not in (16, 64):
                    return "MXFP4 sorted decoding requires BLOCK_N=64"
                if problem.hidden_dim % 512 or problem.inter_dim % 128:
                    return "MXFP4 requires hidden_dim divisible by 512 and inter_dim by 128"
            elif problem.hidden_dim % 256 or problem.inter_dim % 64:
                return (
                    "decoding requires hidden_dim divisible by 256 and inter_dim by 64"
                )
            return None
        if problem.quant_type == "mxfp4":
            return "MXFP4 is supported by decoding kernels only"
        if self.down_path != "default":
            if self.BLOCK_M != 64:
                return "specialized Down paths require unchanged M64 Gate/Up metadata"
            # The native tuner's empty quant_type is a shape-only probe. The
            # runtime wrapper checks actual FP8 weight and activation support.
            if problem.quant_type not in ("", "ptpc", "per_tensor"):
                return "specialized Down paths require FP8 weights"
            if self.down_path == "8x1_compact":
                if problem.inter_dim not in (192, 256, 320, 384, 512, 640):
                    return (
                        "compact Down requires inter_dim in {192,256,320,384,512,640}"
                    )
                if not 0 < problem.experts <= 2048:
                    return "compact Down requires 1 to 2048 experts"
            scale_bytes = 0 if problem.quant_type == "per_tensor" else 256 * 4
            if 64 * problem.inter_dim + scale_bytes + 4 * 16 * 64 * 2 > 64 * 1024:
                return "specialized M64 Down exceeds the 64 KiB LDS capacity"
        if self.use_prefill and problem.quant_type == "no":
            if self.BLOCK_K not in (64, 128):
                return "BF16 prefill requires Gate/Up BLOCK_K=64 or 128"
            if 2 * self.BLOCK_M * self.BLOCK_K * 2 > 64 * 1024:
                return "BF16 Gate/Up ping-pong exceeds the 64 KiB LDS capacity"
            if self.BLOCK_M * problem.inter_dim * 2 > 64 * 1024:
                return "BF16 default Down exceeds the 64 KiB LDS capacity"
        if self.use_prefill and problem.gateup_dim % self.BLOCK_N != 0:
            return (
                f"gateup_dim={problem.gateup_dim} is not divisible by "
                f"BLOCK_N={self.BLOCK_N}"
            )
        if self.use_prefill and problem.inter_dim % 64 != 0:
            return (
                f"inter_dim={problem.inter_dim} is not divisible by the "
                "down BLOCK_K=64"
            )
        if self.use_prefill and problem.hidden_dim % (2 * self.BLOCK_K) != 0:
            return (
                f"hidden_dim={problem.hidden_dim} is not divisible by "
                f"2*BLOCK_K={2 * self.BLOCK_K} for the gateup pipeline"
            )
        if self.use_prefill and problem.model_dim % 256 != 0:
            return (
                f"model_dim={problem.model_dim} is not divisible by 256; "
                "the prefill down pipeline processes paired 128-wide tiles"
            )
        return None


@dataclass(frozen=True)
class _Problem:
    batch: int
    experts: int
    gateup_dim: int
    hidden_dim: int
    model_dim: int
    inter_dim: int
    topk: int
    quant_type: str

    @classmethod
    def from_inputs(
        cls,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_type: QuantType,
    ):
        experts, gateup_dim, hidden_dim = w1.shape
        model_dim, inter_dim = w2.shape[1], w2.shape[2]
        if w1.dtype == torch.float4_e2m1fn_x2:
            hidden_dim *= 2
            inter_dim *= 2
            if hidden_states.shape[1] != hidden_dim or model_dim != hidden_dim:
                raise ValueError("MXFP4 input, weight and output dimensions must match")
        assert gateup_dim == 2 * inter_dim
        return cls(
            batch=int(hidden_states.shape[0]),
            experts=experts,
            gateup_dim=gateup_dim,
            hidden_dim=hidden_dim,
            model_dim=model_dim,
            inter_dim=inter_dim,
            topk=topk_ids.shape[1],
            quant_type={
                QuantType.No: "no",
                QuantType.per_Token: "ptpc",
                QuantType.per_Tensor: "per_tensor",
                QuantType.per_1x32: "mxfp4",
            }[quant_type],
        )


def get_tune_space():
    prefill = [(256, 128), (128, 256), (128, 128)]
    configs = [
        Config(16, 16, 16, False),
        Config(16, 16, 16, False, use_batch1_algorithm=True),
        Config(16, 128, 64, False),
    ]
    if get_gfx() == "gfx950":
        return [config.to_string() for config in configs]
    configs.extend(Config(64, block_n, block_k, True) for block_n, block_k in prefill)
    configs.extend(
        Config(
            64, block_n, block_k, True, down_path=path, down_output_padding_bytes=128
        )
        for path in ("1x4_64x256", "8x1_compact")
        for block_n, block_k in prefill
    )
    return [config.to_string() for config in configs]


@cache
def _get_compiled_kernel(
    N,
    K,
    weight_dtype_str,
    quant_type_str,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage,
    alg,
    E,
    act_quant_type_str=None,
    BLOCK_TILE_SIZE_K=None,
    activation_str="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=0,
    fused_down_clear=False,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    mxfp4_gate_up_interleaved=True,
):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage import compile_gemm

    return compile_gemm(
        N=N,
        K=K,
        weight_dtype=weight_dtype_str,
        weight_quant_type=quant_type_str,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        tile_k=BLOCK_TILE_SIZE_K,
        stage=stage,
        alg=alg,
        E=E,
        USE_ATOMIC_WRITE=down_path == "default",
        act_quant_type=act_quant_type_str,
        activation=activation_str,
        swiglu_limit=swiglu_limit,
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        fused_down_clear=fused_down_clear,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
    )


def _launch(kernel_fn, *args):
    stream = torch.cuda.current_stream()
    prepared_args = [
        _ptr(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    _run_compiled(kernel_fn, *prepared_args, stream)


def _quant_per_tensor(x, scale=None, quant_dtype=torch.float8_e4m3fn, num_rows=None):
    assert scale is None
    assert num_rows is None

    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    xq = torch.empty_like(x, dtype=quant_dtype)
    flydsl_absmax()(x, amax)
    flydsl_quant_per_tensor(quant_dtype)(x, amax, xq)
    fmax = torch.finfo(quant_dtype).max
    xs = amax / fmax
    xs = xs.reshape(1).to(torch.float32)

    return xq, xs


def _empty_scale(device):
    return torch.empty(0, device=device)


def _validate_mxfp4_inputs(w1, w2, w1_scale, w2_scale, problem):
    if w1.shape[0] != w2.shape[0]:
        raise ValueError("MXFP4 weights must have the same expert count")
    for name, scale, channels, reduction in (
        ("w1_scale", w1_scale, problem.gateup_dim, problem.hidden_dim),
        ("w2_scale", w2_scale, problem.model_dim, problem.inter_dim),
    ):
        if scale is None or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(f"{name} must use an E8M0/uint8 dtype")
        groups = ((reduction // 32 + 7) // 8) * 8
        required = problem.experts * channels * groups
        if scale.numel() < required:
            raise ValueError(f"{name} needs at least {required} E8M0 scale entries")


def _gateup_output(hidden_states: torch.Tensor, problem: _Problem):
    return torch.empty(
        [problem.batch, problem.topk, problem.inter_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


def _run_prefill(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: QuantType,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config: Config,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.model_dim,
        hidden_states.dtype,
        config.BLOCK_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    act_quant_type_str = problem.quant_type
    quant_func = (
        aiter.get_hip_quant(aiter.QuantType.per_Token)
        if quant_type == QuantType.per_Token
        else _quant_per_tensor
    )

    if weight_dtype_str == "fp8":
        gateup_in, a_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=w1.dtype,
            num_rows=None,
        )
        a_scale = a_scale.to(torch.float32).contiguous()
    else:
        gateup_in = hidden_states
        a_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

    gemm1_out = _gateup_output(hidden_states, problem)
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=config.BLOCK_N,
        BLOCK_TILE_SIZE_K=config.BLOCK_K,
        stage="gateup",
        alg="prefill_1x4",
        E=problem.experts,
        act_quant_type_str=act_quant_type_str,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
    )
    task_num = int(sorted_expert_ids.shape[0])
    _launch(
        gateup_kernel,
        gateup_in,
        w1,
        gemm1_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        a_scale,
        problem.batch,
        task_num,
    )

    if weight_dtype_str == "fp8":
        down_in, down_in_scale = quant_func(
            gemm1_out.view(problem.batch * problem.topk, -1),
            scale=None,
            quant_dtype=w2.dtype,
            num_rows=None,
        )
    else:
        down_in = gemm1_out
        down_in_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

    output_row_size = (
        problem.model_dim
        + config.down_output_padding_bytes // hidden_states.element_size()
    )
    gemm2_out = torch.empty(
        [sorted_expert_ids.shape[0] * config.BLOCK_M, output_row_size],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=256 if config.down_path == "1x4_64x256" else 128,
        stage="down",
        alg="prefill_1x4",
        E=problem.experts,
        act_quant_type_str=act_quant_type_str,
        down_path=config.down_path,
        down_output_padding_bytes=config.down_output_padding_bytes,
    )
    down_args = (
        down_in,
        w2,
        gemm2_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device),
        down_in_scale,
        problem.batch,
        task_num,
    )
    if config.down_path == "8x1_compact":
        from aiter.ops.flydsl.kernels.moe_gemm_2stage.gemm2_8x1_compact import (
            allocate_task_buffers,
        )

        full_tasks, tail_tasks, counts = allocate_task_buffers(
            sorted_expert_ids, problem.experts
        )
        _launch(
            down_kernel,
            *down_args,
            full_tasks,
            tail_tasks,
            counts,
            full_tasks.shape[0],
            tail_tasks.shape[0],
        )
    else:
        _launch(down_kernel, *down_args)

    loc_ids = torch.empty(
        [problem.batch, problem.topk],
        dtype=torch.int32,
        device=hidden_states.device,
    )
    invert_sorted_ids(problem.topk)(
        sorted_ids,
        loc_ids,
        num_valid_ids,
        sorted_ids.shape[0],
        problem.batch,
    )
    sorted_sum(problem.topk, problem.model_dim, config.down_output_padding_bytes)(
        loc_ids, gemm2_out, cur_out, problem.batch
    )
    return cur_out


def _run_batch1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
    situ_beta: float,
    situ_linear_beta: float,
    mxfp4_gate_up_interleaved: bool,
):
    is_mxfp4 = w1.dtype == torch.float4_e2m1fn_x2
    topk_weight = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )
    gemm1_out = _gateup_output(hidden_states, problem)
    fused_down_clear = problem.model_dim == problem.hidden_dim
    allocate_output = torch.empty if fused_down_clear else torch.zeros
    cur_out = allocate_output(
        [problem.batch, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    route_count = problem.batch * problem.topk
    weight_dtype_str = (
        "bf16" if w1.dtype == torch.bfloat16 else "fp4" if is_mxfp4 else "fp8"
    )
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=64 if is_mxfp4 and problem.batch >= 4 else 32,
        stage="gateup",
        alg="batch1",
        E=None,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
        fused_down_clear=fused_down_clear,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        topk_ids,
        cur_out,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        route_count,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=32 if is_mxfp4 and problem.batch > 1 else 64,
        stage="down",
        alg="batch1",
        E=None,
    )
    _launch(
        down_kernel,
        gemm1_out,
        w2,
        cur_out,
        topk_ids,
        topk_weight,
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device),
        route_count,
    )
    return cur_out


def _run_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config: Config,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
    situ_beta: float,
    situ_linear_beta: float,
    mxfp4_gate_up_interleaved: bool,
):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.model_dim,
        hidden_states.dtype,
        config.BLOCK_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    grid = int(sorted_expert_ids.shape[0])
    if problem.batch * problem.topk <= problem.experts:
        grid = problem.batch * problem.topk

    gemm1_out = _gateup_output(hidden_states, problem)
    weight_dtype_str = (
        "bf16"
        if w1.dtype == torch.bfloat16
        else "fp4" if w1.dtype == torch.float4_e2m1fn_x2 else "fp8"
    )
    block_n = 64 if config.BLOCK_N == 16 else config.BLOCK_N
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        stage="gateup",
        alg="splitk",
        E=problem.experts,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        problem.batch,
        grid,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        stage="down",
        alg="splitk",
        E=problem.experts,
    )
    _launch(
        down_kernel,
        gemm1_out,
        w2,
        cur_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device),
        problem.batch,
        grid,
    )
    return cur_out


def run_flydsl_moe_gfx942(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config_string: str,
    swiglu_limit: float | None = None,
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    gate_mode: GateMode | str = GateMode.SEPARATED,
) -> torch.Tensor:
    if num_local_tokens is not None:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support num_local_tokens"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("w1", w1),
        ("w2", w2),
        ("topk_weight", topk_weight),
        ("topk_ids", topk_ids),
        ("w1_scale", w1_scale),
        ("w2_scale", w2_scale),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise NotImplementedError(
                f"gfx942 FlyDSL whole-graph backend requires contiguous {name}"
            )
    config = Config.from_string(config_string)
    gate_mode = GateMode(gate_mode)
    architecture = get_gfx()
    if architecture not in ("gfx942", "gfx950"):
        raise NotImplementedError(
            f"Unsupported FlyDSL MoE architecture: {architecture}"
        )
    if architecture == "gfx950" and config.use_prefill:
        raise NotImplementedError(
            "This FlyDSL backend supports decoding only on gfx950"
        )
    fp8_dtype = (
        torch.float8_e4m3fn if architecture == "gfx950" else torch.float8_e4m3fnuz
    )
    is_bf16 = (
        w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and quant_type == QuantType.No
    )
    is_fp8 = (
        w1.dtype == fp8_dtype
        and w2.dtype == fp8_dtype
        and quant_type in (QuantType.per_Token, QuantType.per_Tensor)
    )
    is_mxfp4 = (
        architecture == "gfx950"
        and w1.dtype == torch.float4_e2m1fn_x2
        and w2.dtype == torch.float4_e2m1fn_x2
        and quant_type == QuantType.per_1x32
    )
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation
        not in (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)
        or not (is_bf16 or is_fp8 or is_mxfp4)
    ):
        raise RuntimeError("Unsupported input for the gfx942 FlyDSL MoE backend")
    if gate_mode is not GateMode.SEPARATED and not (
        is_mxfp4 and gate_mode is GateMode.INTERLEAVE
    ):
        raise NotImplementedError("Interleaved Gate/Up requires MXFP4 weights")
    if is_fp8 and (w1_scale is None or w2_scale is None):
        raise ValueError("FP8 weights require both w1_scale and w2_scale")
    if is_bf16 and (w1_scale is not None or w2_scale is not None):
        raise NotImplementedError("BF16 weights do not support weight scales")

    if activation == ActivationType.Situv2:
        if config.use_prefill:
            raise NotImplementedError("SiTUv2 is supported by decoding kernels only")
        activation_str = "situv2"
    else:
        activation_str = "swiglu" if activation == ActivationType.Swiglu else "silu"
    problem = _Problem.from_inputs(hidden_states, w1, w2, topk_ids, quant_type)
    if is_mxfp4:
        _validate_mxfp4_inputs(w1, w2, w1_scale, w2_scale, problem)
    unsupported_reason = config.unsupported_reason(problem)
    if unsupported_reason is not None:
        raise RuntimeError(
            f"Unsupported gfx942 FlyDSL MoE config {config_string!r}: "
            f"{unsupported_reason}"
        )
    if config.use_prefill:
        return _run_prefill(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant_type,
            w1_scale,
            w2_scale,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
            config,
            problem,
            activation_str,
            swiglu_limit,
        )
    if problem.batch == 1 or config.use_batch1_algorithm:
        return _run_batch1(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            w1_scale,
            w2_scale,
            problem,
            activation_str,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
            gate_mode is GateMode.INTERLEAVE,
        )
    if 2 <= problem.batch <= 256:
        return _run_decode(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            w1_scale,
            w2_scale,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
            config,
            problem,
            activation_str,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
            gate_mode is GateMode.INTERLEAVE,
        )
    raise RuntimeError(f"Unsupported batch-size {problem.batch}")


def run_flydsl_moe_gfx942_impl(
    request: FusedMoeRequest,
    config_string: str,
) -> torch.Tensor:
    config = Config.from_string(config_string)
    if not (
        getattr(request.w1, "is_shuffled", False)
        and getattr(request.w2, "is_shuffled", False)
    ):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend requires preshuffled weights"
        )
    if request.bias1 is not None or request.bias2 is not None:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support per-expert bias"
        )
    if request.doweight_stage1:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support doweight_stage1=True"
        )
    if request.a1_scale is not None or request.a2_scale is not None:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support prequantized activations"
        )
    if request.hidden_pad or request.intermediate_pad:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support hidden/intermediate padding"
        )
    gate_mode = (
        GateMode.SEPARATED if request.gate_mode is None else GateMode(request.gate_mode)
    )
    is_mxfp4 = (
        request.w1.dtype == torch.float4_e2m1fn_x2
        and request.w2.dtype == torch.float4_e2m1fn_x2
        and request.quant_type == QuantType.per_1x32
    )
    if gate_mode is not GateMode.SEPARATED and not (
        is_mxfp4 and gate_mode is GateMode.INTERLEAVE
    ):
        raise NotImplementedError("Interleaved Gate/Up requires MXFP4 weights")
    if request.dtype not in (None, request.hidden_states.dtype):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support output dtype conversion"
        )
    if request.block_size_m not in (None, config.BLOCK_M):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support overriding block_size_m"
        )
    if request.ksplit != 0:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support split-K"
        )
    if request.w1.dtype == torch.bfloat16 or is_mxfp4:
        supported_q_dtypes_a = (None, torch.bfloat16)
    else:
        fp8_dtype = (
            torch.float8_e4m3fn if get_gfx() == "gfx950" else torch.float8_e4m3fnuz
        )
        supported_q_dtypes_a = (
            (None, fp8_dtype)
            if config.use_prefill
            else (None, fp8_dtype, torch.bfloat16)
        )
    if request.q_dtype_a not in supported_q_dtypes_a:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph activation dtype must match the weight mode"
        )
    if (
        request.q_dtype_w not in (None, request.w1.dtype)
        or request.w2.dtype != request.w1.dtype
    ):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph weight dtype must match the weight mode"
        )
    return run_flydsl_moe_gfx942(
        request.hidden_states,
        request.w1,
        request.w2,
        request.topk_weight,
        request.topk_ids,
        request.activation,
        request.quant_type,
        request.w1_scale,
        request.w2_scale,
        request.expert_mask,
        request.num_local_tokens,
        request.moe_sorting_dispatch_policy,
        config.to_string(),
        request.swiglu_limit,
        1.0 if request.beta is None else float(request.beta),
        1.0 if request.linear_beta is None else float(request.linear_beta),
        gate_mode,
    )
