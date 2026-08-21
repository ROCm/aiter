# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
from functools import cache
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    invert_sorted_ids,
    sorted_sum,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


@dataclass
class Config:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    use_prefill: bool

    def to_string(self):
        return (
            str(self.BLOCK_M)
            + "_"
            + str(self.BLOCK_N)
            + "_"
            + str(self.BLOCK_K)
            + "_"
            + str(self.use_prefill)
        )

    @classmethod
    def from_string(cls, data: str):
        parts = data.split("_")
        if len(parts) != 4:
            raise ValueError(f"Invalid config string: {data}")

        def parse_bool(value: str) -> bool:
            if value == "True":
                return True
            if value == "False":
                return False
            raise ValueError(f"Invalid boolean value in config string: {value}")

        return cls(
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
            parse_bool(parts[3]),
        )


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
        assert gateup_dim == 2 * inter_dim
        return cls(
            batch=int(hidden_states.shape[0]),
            experts=experts,
            gateup_dim=gateup_dim,
            hidden_dim=hidden_dim,
            model_dim=model_dim,
            inter_dim=inter_dim,
            topk=topk_ids.shape[1],
            quant_type=("ptpc" if quant_type == QuantType.per_Token else "per_tensor"),
        )


def get_tune_space():
    return [
        # decoding ignored BLOCK_N/BLOCK_K
        Config(16, 16, 16, False).to_string(),
        # Config(64, 256, 64, True).to_string(),
        # Config(64, 256, 128, True).to_string(),
        Config(64, 128, 256, True).to_string(),
        Config(64, 128, 128, True).to_string(),
    ]


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
):
    """Cache-compiled flydsl kernel via compile_gemm."""
    from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import compile_gemm

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
        USE_ATOMIC_WRITE=True,
        act_quant_type=act_quant_type_str,
        activation=activation_str,
        swiglu_limit=swiglu_limit,
    )


_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
}


def _ptr(t):
    return flyc.from_c_void_p(_TORCH_TO_FX[t.dtype], t.data_ptr())


def _launch(kernel_fn, *args):
    """Launch a FlyDSL JIT kernel on the current stream."""
    stream = torch.cuda.current_stream()
    prepared_args = [
        _ptr(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    _run_compiled(kernel_fn, *prepared_args, stream)


def _quant_per_tensor(x, scale=None, quant_dtype=torch.float8_e4m3fn, num_rows=None):
    assert scale is None
    assert num_rows is None

    amax = torch.empty(1, dtype=torch.float32, device=x.device)
    xq = torch.empty_like(x, dtype=quant_dtype)
    flydsl_absmax()(x, amax)
    flydsl_quant_per_tensor(quant_dtype)(x, amax, xq)
    fmax = torch.finfo(quant_dtype).max
    xs = amax / fmax
    xs = xs.reshape(1).to(torch.float32)

    return xq, xs


def _empty_scale(device):
    return torch.empty(0, device=device)


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
        None,
        None,
        0,
    )
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    act_quant_type_str = "ptpc"
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
        if quant_type == QuantType.per_Tensor:
            a_scale = a_scale.repeat(problem.batch, 1).contiguous()
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
        BLOCK_TILE_SIZE_K=None,
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

    gemm2_out = torch.empty(
        [sorted_expert_ids.shape[0] * config.BLOCK_M, problem.model_dim],
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
        BLOCK_TILE_SIZE_N=128,
        stage="down",
        alg="prefill_1x4",
        E=problem.experts,
    )
    _launch(
        down_kernel,
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
    sorted_sum(problem.topk, problem.model_dim)(
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
):
    topk_weight = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )
    gemm1_out = _gateup_output(hidden_states, problem)
    cur_out = torch.zeros(
        [1, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str="fp8",
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=32,
        stage="gateup",
        alg="batch1",
        E=None,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        topk_ids,
        topk_weight,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        problem.topk,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str="fp8",
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=64,
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
        problem.topk,
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
):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.hidden_dim,
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
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str="fp8",
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=64,
        stage="gateup",
        alg="splitk",
        E=problem.experts,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
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
        weight_dtype_str="fp8",
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=64,
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
) -> torch.Tensor:
    config = Config.from_string(config_string)
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation not in (ActivationType.Silu, ActivationType.Swiglu)
        or w1.dtype != torch.float8_e4m3fnuz
        or w2.dtype != torch.float8_e4m3fnuz
    ):
        raise RuntimeError("Unsupported input for the gfx942 FlyDSL MoE backend")
    if quant_type not in (QuantType.per_Token, QuantType.per_Tensor):
        raise RuntimeError(f"Unsupported quant_type: {quant_type}")

    activation_str = "swiglu" if activation == ActivationType.Swiglu else "silu"
    problem = _Problem.from_inputs(hidden_states, w1, w2, topk_ids, quant_type)
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
            config,
            problem,
            activation_str,
            swiglu_limit,
        )
    if problem.batch == 1:
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
        )
    raise RuntimeError(f"Unsupported batch-size {problem.batch}")


def run_flydsl_moe_gfx942_impl(
    request: FusedMoeRequest,
    config_string: str,
) -> torch.Tensor:
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
        config_string,
        request.swiglu_limit,
    )
