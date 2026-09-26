# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared host configuration, launcher caches, and no-dispatch MoE precompilation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.utils import env as flydsl_env

from aiter.ops.flydsl.kernels.moe_gemm_2stage import (
    precompile_moe_quant_kernels,
    precompile_moe_reduction_kernels,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.common import (
    device_context,
    get_device_cache_key,
    resolve_tile_k,
)
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled

if TYPE_CHECKING:
    from aiter import QuantType


@dataclass
class Config:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    use_prefill: bool

    def __post_init__(self):
        if min(self.BLOCK_M, self.BLOCK_N, self.BLOCK_K) <= 0:
            raise ValueError("MoE block sizes must be positive")

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

    def unsupported_reason(self, problem: _Problem) -> str | None:
        if (
            min(
                problem.batch,
                problem.experts,
                problem.gateup_dim,
                problem.hidden_dim,
                problem.model_dim,
                problem.inter_dim,
                problem.topk,
            )
            <= 0
        ):
            return "MoE dimensions must be positive"
        if not self.use_prefill:
            if problem.batch > 256:
                return "decode supports at most 256 tokens"
            if self.BLOCK_M != 16:
                return "decode requires BLOCK_M=16"
            # The baseline decode launchers use fixed N/K tiles, not config N/K.
            gateup_block_n = 32 if problem.batch == 1 else 64
            if problem.gateup_dim % gateup_block_n != 0:
                return f"gateup_dim must be divisible by {gateup_block_n} for decode"
            if problem.model_dim % 64 != 0:
                return "model_dim must be divisible by the decode down BLOCK_N=64"
            if problem.hidden_dim % 256 != 0:
                return "hidden_dim must be divisible by 256 for gateup split-K"
            if problem.inter_dim % 64 != 0:
                return "inter_dim must be divisible by the decode down BLOCK_K=64"
            return None
        if not (32 <= self.BLOCK_M <= 256 and self.BLOCK_M % 32 == 0):
            return "prefill BLOCK_M must be a multiple of 32 in [32, 256]"
        if self.BLOCK_N not in (128, 256):
            return "prefill gateup BLOCK_N must be 128 or 256"
        if self.BLOCK_K not in (128, 256):
            return "FP8 prefill gateup BLOCK_K must be 128 or 256"
        if problem.gateup_dim % self.BLOCK_N != 0:
            return (
                f"gateup_dim={problem.gateup_dim} is not divisible by "
                f"BLOCK_N={self.BLOCK_N}"
            )
        if problem.inter_dim % 64 != 0:
            return (
                f"inter_dim={problem.inter_dim} is not divisible by the down BLOCK_K=64"
            )
        if problem.hidden_dim % (2 * self.BLOCK_K) != 0:
            return (
                f"hidden_dim={problem.hidden_dim} is not divisible by "
                f"2*BLOCK_K={2 * self.BLOCK_K} for the gateup pipeline"
            )
        # The baseline reduction uses 32 threads for odd 256-column multiples.
        if problem.model_dim % 256 != 0:
            return "prefill down and reduction require model_dim divisible by 256"
        if 2 * self.BLOCK_M * self.BLOCK_K > 64 * 1024:
            return "prefill gateup ping-pong buffers exceed 64 KiB LDS"
        down_lds_bytes = self.BLOCK_M * problem.inter_dim
        if down_lds_bytes > 64 * 1024:
            return "prefill down activation buffer exceeds 64 KiB LDS"
        # The default down copy assigns one 16-byte atom to each of 256 threads.
        if down_lds_bytes % (256 * 16) != 0:
            return "prefill down requires BLOCK_M * inter_dim divisible by 4096"
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
        # Only runtime input validation needs the C++ enum exports.
        from aiter import QuantType

        if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
            raise ValueError("hidden_states must be 2D and MoE weights must be 3D")
        if topk_ids.ndim != 2 or topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("topk_ids must have shape [batch, topk]")
        experts, gateup_dim, hidden_dim = w1.shape
        model_dim, inter_dim = w2.shape[1], w2.shape[2]
        if w2.shape[0] != experts:
            raise ValueError("w1 and w2 must have the same expert count")
        if hidden_states.shape[1] != hidden_dim or gateup_dim != 2 * inter_dim:
            raise ValueError(
                "MoE input and gate/up dimensions do not match the weights"
            )
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


@cache
def _get_compiled_kernel_cached(
    device_cache_key,
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
    from aiter.ops.flydsl.kernels.moe_gemm_2stage import compile_gemm

    # Only a miss needs the requested device as the ambient build context.
    physical_device = device_cache_key[0]
    device = physical_device[0] if isinstance(physical_device, tuple) else None
    with device_context(device):
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
    *,
    device=None,
):
    # Resolve device and environment overrides before the outer cache lookup.
    return _get_compiled_kernel_cached(
        get_device_cache_key(device),
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
        act_quant_type_str,
        resolve_tile_k(BLOCK_TILE_SIZE_K),
        activation_str,
        swiglu_limit,
    )


_get_compiled_kernel.cache_clear = _get_compiled_kernel_cached.cache_clear
_get_compiled_kernel.cache_info = _get_compiled_kernel_cached.cache_info


def precompile_flydsl_moe(
    *,
    config_string: str,
    batch: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    weight_dtype: str,
    quant_type: str,
    activation: str,
    swiglu_limit: float | None = None,
    device=None,
) -> None:
    """Preload the baseline gfx942 whole-graph FlyDSL launchers without dispatch.

    Inputs/outputs are BF16 with FP8 E4M3FNUZ weights. Default HIP sorting and
    per-token quantization are outside this preload, as are alternate sorting
    backends. Per-tensor quantization and prefill reductions are included here.
    Swiglu limits are compile-time specializations: None/0 mean the legacy
    default 7.0, and each non-default limit must be precompiled separately
    before a cold RUN_ONLY invocation.
    """
    if weight_dtype != "fp8" or quant_type not in ("ptpc", "per_tensor"):
        raise ValueError(
            f"Unsupported whole-graph dtype/quant pair: {weight_dtype}/{quant_type}"
        )
    if activation not in ("silu", "swiglu"):
        raise ValueError(f"Unsupported whole-graph activation: {activation}")
    config = Config.from_string(config_string)
    problem = _Problem(
        batch=batch,
        experts=experts,
        gateup_dim=2 * inter_dim,
        hidden_dim=model_dim,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
        quant_type=quant_type,
    )
    reason = config.unsupported_reason(problem)
    if reason is not None:
        raise ValueError(f"Unsupported whole-graph config: {reason}")

    with device_context(device):
        target_arch = os.environ.get("ARCH") or os.environ.get("FLYDSL_GPU_ARCH")
        if (
            not target_arch
            and not flydsl_env.compile.compile_only
            and torch.cuda.is_available()
        ):
            target_arch = torch.cuda.get_device_properties(device).gcnArchName
        if not target_arch or target_arch.split(":", 1)[0] != "gfx942":
            raise ValueError("Whole-graph AOT requires a gfx942 compile target")

        # Pointer addresses and runtime grid/stream values do not specialize
        # these launchers. Match runtime pointer element types, never launch them.
        bf16 = flyc.from_c_void_p(fx.BFloat16, 0)
        byte = flyc.from_c_void_p(fx.Uint8, 0)
        int32 = flyc.from_c_void_p(fx.Int32, 0)
        float32 = flyc.from_c_void_p(fx.Float32, 0)
        stream = fx.Stream(None)
        alg = (
            "prefill_1x4"
            if config.use_prefill
            else ("batch1" if batch == 1 else "splitk")
        )
        for stage in ("gateup", "down"):
            is_gateup = stage == "gateup"
            if config.use_prefill:
                block_n = config.BLOCK_N if is_gateup else 128
            else:
                block_n = 32 if is_gateup and batch == 1 else 64
            kernel = _get_compiled_kernel(
                device=device,
                N=2 * inter_dim if is_gateup else model_dim,
                K=model_dim if is_gateup else inter_dim,
                weight_dtype_str=weight_dtype,
                quant_type_str=quant_type,
                TOPK=topk,
                BLOCK_TILE_SIZE_M=16 if alg == "batch1" else config.BLOCK_M,
                BLOCK_TILE_SIZE_N=block_n,
                BLOCK_TILE_SIZE_K=(
                    config.BLOCK_K if config.use_prefill and is_gateup else None
                ),
                stage=stage,
                alg=alg,
                E=None if alg == "batch1" else experts,
                act_quant_type_str=quant_type if config.use_prefill else None,
                activation_str=activation if is_gateup else "silu",
                swiglu_limit=swiglu_limit if is_gateup else None,
            )
            if alg == "batch1":
                args = (bf16, byte, bf16, int32, float32, float32, topk, stream)
            else:
                args = (
                    byte if config.use_prefill else bf16,
                    byte,
                    bf16,
                    int32,
                    float32,
                    int32,
                    int32,
                    float32,
                )
                if config.use_prefill:
                    args += (float32,)
                args += (batch, 1, stream)
            _preload_compiled(kernel, *args)

        if config.use_prefill:
            precompile_moe_reduction_kernels(topk, model_dim, device=device)
            if quant_type == "per_tensor":
                precompile_moe_quant_kernels(torch.float8_e4m3fnuz, device=device)
