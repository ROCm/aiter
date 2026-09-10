# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
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
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    invert_sorted_ids,
    sorted_sum,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942.common import (
    get_device_cache_key,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.moe_common import GateMode


@dataclass
class Config:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    use_prefill: bool
    use_batch1_algorithm: bool = False
    down_path: str = "default"
    GATEUP_BLOCK_M: int | None = None
    down_output_padding_bytes: int | None = None

    def to_string(self):
        base = (
            str(self.BLOCK_M)
            + "_"
            + str(self.BLOCK_N)
            + "_"
            + str(self.BLOCK_K)
            + "_"
            + str(self.use_prefill)
        )
        if self.use_batch1_algorithm:
            base += "_True"
        gateup_block_m = self.GATEUP_BLOCK_M or self.BLOCK_M
        if (
            self.down_path == "default"
            and gateup_block_m == self.BLOCK_M
            and self.down_output_padding_bytes is None
        ):
            return base
        padding = (
            "none"
            if self.down_output_padding_bytes is None
            else str(self.down_output_padding_bytes)
        )
        return f"{base}:{self.down_path}:{gateup_block_m}:{padding}"

    @classmethod
    def from_string(cls, data: str):
        extensions = data.split(":")
        if len(extensions) not in (1, 4):
            raise ValueError(f"Invalid config string: {data}")
        parts = extensions[0].split("_")
        if len(parts) not in (4, 5):
            raise ValueError(f"Invalid config string: {data}")

        def parse_bool(value: str) -> bool:
            if value == "True":
                return True
            if value == "False":
                return False
            raise ValueError(f"Invalid boolean value in config string: {value}")

        config = cls(
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
            parse_bool(parts[3]),
            use_batch1_algorithm=(parse_bool(parts[4]) if len(parts) == 5 else False),
        )
        if len(extensions) == 1:
            return config
        down_path, gateup_block_m, padding = extensions[1:4]
        if down_path not in ("default", "1x4_64x256", "2x4", "1x8"):
            raise ValueError(f"Invalid down path in config string: {data}")
        return cls(
            config.BLOCK_M,
            config.BLOCK_N,
            config.BLOCK_K,
            config.use_prefill,
            use_batch1_algorithm=config.use_batch1_algorithm,
            down_path=down_path,
            GATEUP_BLOCK_M=int(gateup_block_m),
            down_output_padding_bytes=(None if padding == "none" else int(padding)),
        )

    def unsupported_reason(self, problem: "_Problem") -> str | None:
        if self.use_batch1_algorithm:
            if self.use_prefill:
                return "direct route-wise algorithm cannot use prefill"
            if (
                self.down_path != "default"
                or (self.GATEUP_BLOCK_M or self.BLOCK_M) != self.BLOCK_M
                or self.down_output_padding_bytes is not None
            ):
                return (
                    "direct route-wise algorithm does not support extended down configs"
                )
            if not 2 <= problem.batch <= 8:
                return f"direct route-wise algorithm requires 2 to 8 tokens, got {problem.batch}"
            if (self.BLOCK_M, self.BLOCK_N, self.BLOCK_K) != (16, 16, 16):
                return "direct route-wise algorithm uses the fixed legacy tile"
            if problem.quant_type == "mxfp4":
                if problem.hidden_dim % 512 != 0:
                    return (
                        f"MXFP4 gateup K={problem.hidden_dim} must be divisible by 512"
                    )
                if problem.inter_dim % 128 != 0:
                    return f"MXFP4 down K={problem.inter_dim} must be divisible by 128"
            return None
        if not self.use_prefill:
            if problem.batch > 256:
                return f"decode path supports at most 256 tokens, got {problem.batch}"
            if self.BLOCK_M != 16:
                return "decode path requires BLOCK_M=16"
            if problem.batch == 1 and (self.BLOCK_N, self.BLOCK_K) != (16, 16):
                return "batch1 uses the fixed legacy decode tile"
            block_n = 64 if self.BLOCK_N == 16 else self.BLOCK_N
            block_k = 64 if self.BLOCK_K == 16 else self.BLOCK_K
            if self.down_path != "default":
                return "decode path requires the default down kernel"
            if problem.quant_type == "mxfp4" and block_n != 64:
                return "MXFP4 decode requires BLOCK_N=64"
            if block_n not in (64, 128):
                return f"decode BLOCK_N must be 64 or 128, got {block_n}"
            if block_k != 64:
                return f"decode BLOCK_K must be 64, got {block_k}"
            if problem.gateup_dim % block_n != 0:
                return f"gateup_dim={problem.gateup_dim} is not divisible by BLOCK_N={block_n}"
            if problem.model_dim % block_n != 0:
                return f"model_dim={problem.model_dim} is not divisible by BLOCK_N={block_n}"
            if problem.hidden_dim % (4 * block_k) != 0:
                return f"hidden_dim={problem.hidden_dim} is not divisible by 4*BLOCK_K={4 * block_k}"
            if problem.inter_dim % block_k != 0:
                return f"inter_dim={problem.inter_dim} is not divisible by BLOCK_K={block_k}"
            if problem.quant_type == "mxfp4":
                if problem.hidden_dim % 512 != 0:
                    return (
                        f"MXFP4 gateup K={problem.hidden_dim} must be divisible by 512"
                    )
                if problem.inter_dim % 128 != 0:
                    return f"MXFP4 down K={problem.inter_dim} must be divisible by 128"
            return None
        if problem.quant_type == "mxfp4":
            return "MXFP4 does not support the prefill algorithm"
        gateup_block_m = self.GATEUP_BLOCK_M or self.BLOCK_M
        if problem.gateup_dim % self.BLOCK_N != 0:
            return f"gateup_dim={problem.gateup_dim} is not divisible by BLOCK_N={self.BLOCK_N}"
        if problem.hidden_dim % self.BLOCK_K != 0:
            return f"hidden_dim={problem.hidden_dim} is not divisible by BLOCK_K={self.BLOCK_K}"
        if (problem.hidden_dim // self.BLOCK_K) % 2 != 0:
            return "gateup requires an even number of BLOCK_K tiles"
        if self.down_path == "default":
            activation_bytes = (
                self.BLOCK_M
                * problem.inter_dim
                * (2 if problem.quant_type == "no" else 1)
            )
            if activation_bytes > 64 * 1024:
                return f"default down path requires {activation_bytes} bytes of LDS"
            return None
        if problem.quant_type not in ("ptpc", "per_tensor"):
            return "specialized down paths require FP8 weights"
        if self.down_path == "1x8" and problem.quant_type != "per_tensor":
            return "1x8 requires per-tensor weight and activation scales"
        down_tile_n = {
            "1x4_64x256": 256,
            "2x4": 256,
            "1x8": 512,
        }[self.down_path]
        if problem.model_dim % down_tile_n != 0:
            return f"model_dim={problem.model_dim} is not divisible by down tile N={down_tile_n}"
        if problem.inter_dim % 64 != 0:
            return f"inter_dim={problem.inter_dim} is not divisible by 64"
        if self.down_path == "1x4_64x256":
            if self.BLOCK_M != 64 or gateup_block_m != 64:
                return "1x4_64x256 requires down/gateup BLOCK_M=64"
            scale_bytes = 256 * 4 if problem.quant_type == "ptpc" else 0
            lds_bytes = 64 * problem.inter_dim + scale_bytes + 4 * 16 * 64 * 2
        elif self.down_path == "2x4":
            if self.BLOCK_M != 128 or gateup_block_m != 64:
                return "2x4 requires down BLOCK_M=128 and gateup BLOCK_M=64"
            lds_bytes = 2 * 64 * problem.inter_dim + 8 * 16 * 64 * 2
        else:
            if self.BLOCK_M != 64 or gateup_block_m != 64:
                return "1x8 requires down/gateup BLOCK_M=64"
            lds_bytes = 64 * problem.inter_dim + 8 * 16 * 64 * 2
        if lds_bytes > 64 * 1024:
            return f"down path requires {lds_bytes} bytes of LDS"
        return None


def get_tune_config_unsupported_reason(
    config_string: str,
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    quant_type: QuantType,
) -> str | None:
    quant_type_string = {
        QuantType.No: "no",
        QuantType.per_Token: "ptpc",
        QuantType.per_Tensor: "per_tensor",
        QuantType.per_1x32: "mxfp4",
    }.get(quant_type)
    if quant_type_string is None:
        return f"unsupported quant_type: {quant_type}"
    problem = _Problem(
        batch=token,
        experts=expert,
        gateup_dim=inter_dim * 2,
        hidden_dim=model_dim,
        model_dim=model_dim,
        inter_dim=inter_dim,
        topk=topk,
        quant_type=quant_type_string,
    )
    return Config.from_string(config_string).unsupported_reason(problem)


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
        if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
            raise ValueError("hidden_states must be 2D and MoE weights must be 3D")
        if topk_ids.ndim != 2 or topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("topk_ids must have shape [batch, topk]")
        experts, gateup_dim, hidden_dim = w1.shape
        if w2.shape[0] != experts:
            raise ValueError("w1 and w2 must have the same expert count")
        model_dim, inter_dim = w2.shape[1], w2.shape[2]
        if w1.dtype == torch.float4_e2m1fn_x2:
            hidden_dim *= 2
            inter_dim *= 2
        if gateup_dim != 2 * inter_dim:
            raise ValueError(
                f"w1 gate-up dim {gateup_dim} must equal 2 * w2 inter dim {inter_dim}"
            )
        if hidden_states.shape[1] != hidden_dim or model_dim != hidden_dim:
            raise ValueError(
                "hidden_states, w1 input, and w2 output dimensions must match"
            )
        quant_type_string = {
            QuantType.No: "no",
            QuantType.per_Token: "ptpc",
            QuantType.per_Tensor: "per_tensor",
            QuantType.per_1x32: "mxfp4",
        }.get(quant_type)
        if quant_type_string is None:
            raise RuntimeError(f"Unsupported quant_type: {quant_type}")
        return cls(
            batch=int(hidden_states.shape[0]),
            experts=experts,
            gateup_dim=gateup_dim,
            hidden_dim=hidden_dim,
            model_dim=model_dim,
            inter_dim=inter_dim,
            topk=topk_ids.shape[1],
            quant_type=quant_type_string,
        )


def get_tune_space(batch: int | None = None, *, include_prefill: bool = True):
    configs = [
        # Legacy decode configs map 16/16 to the original 64x64 split-K tile.
        Config(16, 16, 16, False),
    ]
    if include_prefill:
        configs.extend(
            [
                Config(64, 256, 128, True),
                Config(64, 128, 256, True),
                Config(64, 128, 128, True),
            ]
        )
    if batch is not None and 2 <= batch <= 8:
        configs.insert(1, Config(16, 16, 16, False, use_batch1_algorithm=True))
    configs.extend(
        Config(16, block_n, block_k, False)
        for block_n in (64, 128)
        for block_k in (64,)
        if (block_n, block_k) != (64, 64)
    )
    if include_prefill:
        for block_k in (128, 256):
            for gateup_block_n in (128, 256):
                configs.extend(
                    [
                        Config(
                            64,
                            gateup_block_n,
                            block_k,
                            True,
                            down_path="1x4_64x256",
                            GATEUP_BLOCK_M=64,
                            down_output_padding_bytes=128,
                        ),
                        Config(
                            128,
                            gateup_block_n,
                            block_k,
                            True,
                            down_path="2x4",
                            GATEUP_BLOCK_M=64,
                            down_output_padding_bytes=0,
                        ),
                    ]
                )
            configs.append(
                Config(
                    64,
                    128,
                    block_k,
                    True,
                    down_path="1x8",
                    GATEUP_BLOCK_M=64,
                    down_output_padding_bytes=0,
                )
            )
    return [config.to_string() for config in configs]


@cache
def _get_compiled_kernel_cached(
    device,
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
    USE_ATOMIC_WRITE=True,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
):
    """Cache-compiled flydsl kernel via compile_gemm."""
    del device
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
        USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
        act_quant_type=act_quant_type_str,
        activation=activation_str,
        swiglu_limit=swiglu_limit,
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
    )


def _get_compiled_kernel(*args, **kwargs):
    return _get_compiled_kernel_cached(get_device_cache_key(), *args, **kwargs)


_get_compiled_kernel.cache_clear = _get_compiled_kernel_cached.cache_clear
_get_compiled_kernel.cache_info = _get_compiled_kernel_cached.cache_info


_TORCH_TO_FX = {
    torch.uint8: fx.Uint8,
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
}
if hasattr(torch, "float8_e8m0fnu"):
    _TORCH_TO_FX[torch.float8_e8m0fnu] = fx.Uint8
if hasattr(torch, "float4_e2m1fn_x2"):
    _TORCH_TO_FX[torch.float4_e2m1fn_x2] = fx.Uint8


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


def _validate_mxfp4_inputs(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    problem: _Problem,
) -> None:
    if not w1.is_contiguous() or not w2.is_contiguous():
        raise ValueError("MXFP4 weights must be contiguous preshuffled tensors")
    if w1_scale is None or w2_scale is None:
        raise ValueError("MXFP4 weights require both E8M0 scale tensors")
    scale_dtypes = {torch.uint8}
    if hasattr(torch, "float8_e8m0fnu"):
        scale_dtypes.add(torch.float8_e8m0fnu)
    for name, scale in (("w1_scale", w1_scale), ("w2_scale", w2_scale)):
        if scale.dtype not in scale_dtypes:
            raise ValueError(f"{name} must use an E8M0/uint8 dtype, got {scale.dtype}")
        if not scale.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    gate_groups = ((problem.hidden_dim // 32 + 7) // 8) * 8
    down_groups = ((problem.inter_dim // 32 + 7) // 8) * 8
    required_w1_scales = problem.experts * problem.gateup_dim * gate_groups
    required_w2_scales = problem.experts * problem.model_dim * down_groups
    if w1_scale.numel() < required_w1_scales:
        raise ValueError(
            f"w1_scale has {w1_scale.numel()} entries; expected at least {required_w1_scales}"
        )
    if w2_scale.numel() < required_w2_scales:
        raise ValueError(
            f"w2_scale has {w2_scale.numel()} entries; expected at least {required_w2_scales}"
        )


def _activation_scalars(
    activation: str,
    situ_beta: float,
    situ_linear_beta: float,
    activation_limit: float | None,
) -> tuple[float, float, float, float, float]:
    if activation == "situv2":
        beta = float(situ_beta)
        linear_beta = float(situ_linear_beta)
        if beta <= 0.0 or linear_beta <= 0.0:
            raise ValueError(
                "situ_beta and situ_linear_beta must be positive, "
                f"got {beta}/{linear_beta}"
            )
    else:
        beta = linear_beta = 1.0
    if activation == "swiglu":
        limit = float(activation_limit) if activation_limit else 7.0
    else:
        limit = float(activation_limit) if activation_limit else float("inf")
    return beta, 1.0 / beta, linear_beta, 1.0 / linear_beta, limit


def _gateup_output(hidden_states: torch.Tensor, problem: _Problem):
    return torch.empty(
        [problem.batch, problem.topk, problem.inter_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


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
) -> None:
    """Compile all launchers reachable through one whole-graph tuned config."""
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
    unsupported_reason = config.unsupported_reason(problem)
    if unsupported_reason is not None:
        raise ValueError(f"Unsupported whole-graph config: {unsupported_reason}")

    if (weight_dtype, quant_type) not in (
        ("bf16", "no"),
        ("fp8", "ptpc"),
        ("fp8", "per_tensor"),
        ("fp4", "mxfp4"),
    ):
        raise ValueError(
            f"Unsupported whole-graph dtype/quant pair: {weight_dtype}/{quant_type}"
        )

    bf16 = torch.empty(1, dtype=torch.bfloat16)
    byte = torch.empty(1, dtype=torch.uint8)
    int32 = torch.empty(1, dtype=torch.int32)
    float32 = torch.empty(1, dtype=torch.float32)
    is_fp8 = weight_dtype == "fp8"
    is_mxfp4 = weight_dtype == "fp4"
    weight = bf16 if weight_dtype == "bf16" else byte
    weight_scale = byte if is_mxfp4 else float32
    activation_scalars = _activation_scalars(activation, 1.0, 1.0, None)

    def compile_launcher(executable, *args):
        if (
            os.environ.get("COMPILE_ONLY") == "1"
            and getattr(executable, "_cf", None) is not None
        ):
            return
        _run_compiled(executable, *args)

    def compile_kernel(*, stage, alg, block_m, block_n, **kwargs):
        return _get_compiled_kernel(
            N=2 * inter_dim if stage == "gateup" else model_dim,
            K=model_dim if stage == "gateup" else inter_dim,
            weight_dtype_str=weight_dtype,
            quant_type_str=quant_type,
            TOPK=topk,
            BLOCK_TILE_SIZE_M=block_m,
            BLOCK_TILE_SIZE_N=block_n,
            stage=stage,
            alg=alg,
            E=None if alg == "batch1" else experts,
            activation_str=activation if stage == "gateup" else "silu",
            **kwargs,
        )

    if config.use_prefill:
        gateup_block_m = config.GATEUP_BLOCK_M or config.BLOCK_M
        gateup = compile_kernel(
            stage="gateup",
            alg="prefill_1x4",
            block_m=gateup_block_m,
            block_n=config.BLOCK_N,
            act_quant_type_str=quant_type,
            BLOCK_TILE_SIZE_K=config.BLOCK_K,
            METADATA_TILE_SIZE_M=config.BLOCK_M,
        )
        compile_launcher(
            gateup,
            _ptr(byte if is_fp8 else bf16),
            _ptr(weight),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(int32),
            _ptr(int32),
            _ptr(weight_scale),
            _ptr(float32),
            batch,
            1,
            *activation_scalars,
            0,
        )

        down_block_n = {
            "default": 128,
            "1x4_64x256": 256,
            "2x4": 256,
            "1x8": 512,
        }[config.down_path]
        down = compile_kernel(
            stage="down",
            alg="prefill_1x4",
            block_m=config.BLOCK_M,
            block_n=down_block_n,
            USE_ATOMIC_WRITE=False,
            act_quant_type_str=quant_type,
            down_path=config.down_path,
            down_output_padding_bytes=config.down_output_padding_bytes,
        )
        down_args = (
            _ptr(byte if is_fp8 else bf16),
            _ptr(weight),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(int32),
            _ptr(int32),
            _ptr(weight_scale),
            _ptr(float32),
            batch,
            1,
        )
        if config.down_path == "default":
            down_args += activation_scalars
        compile_launcher(down, *down_args, 0)
        return

    use_batch1_algorithm = batch == 1 or config.use_batch1_algorithm
    if use_batch1_algorithm:
        fused_down_clear = is_mxfp4 and batch > 1
        gate_layouts = (False, True) if is_mxfp4 else (True,)
        gate_block_n = 64 if is_mxfp4 and batch >= 4 else 32
        for gate_up_interleaved in gate_layouts:
            gateup = compile_kernel(
                stage="gateup",
                alg="batch1",
                block_m=16,
                block_n=gate_block_n,
                mxfp4_gate_up_interleaved=gate_up_interleaved,
                fused_down_clear=fused_down_clear,
            )
            compile_launcher(
                gateup,
                _ptr(bf16),
                _ptr(weight),
                _ptr(bf16),
                _ptr(int32),
                _ptr(bf16 if fused_down_clear else float32),
                _ptr(weight_scale),
                batch,
                *activation_scalars,
                0,
            )
        down = compile_kernel(
            stage="down",
            alg="batch1",
            block_m=16,
            block_n=32 if fused_down_clear else 64,
        )
        compile_launcher(
            down,
            _ptr(bf16),
            _ptr(weight),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(weight_scale),
            batch,
            *activation_scalars,
            0,
        )
        return

    block_n = 64 if config.BLOCK_N == 16 else config.BLOCK_N
    block_k = 64 if config.BLOCK_K == 16 else config.BLOCK_K
    gate_layouts = (False, True) if is_mxfp4 else (True,)
    for gate_up_interleaved in gate_layouts:
        gateup = compile_kernel(
            stage="gateup",
            alg="splitk",
            block_m=config.BLOCK_M,
            block_n=block_n,
            BLOCK_TILE_SIZE_K=block_k,
            mxfp4_gate_up_interleaved=gate_up_interleaved,
        )
        compile_launcher(
            gateup,
            _ptr(bf16),
            _ptr(weight),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(int32),
            _ptr(int32),
            _ptr(weight_scale),
            batch,
            1,
            *activation_scalars,
            0,
        )
    down = compile_kernel(
        stage="down",
        alg="splitk",
        block_m=config.BLOCK_M,
        block_n=block_n,
        BLOCK_TILE_SIZE_K=block_k,
    )
    compile_launcher(
        down,
        _ptr(bf16),
        _ptr(weight),
        _ptr(bf16),
        _ptr(int32),
        _ptr(float32),
        _ptr(int32),
        _ptr(int32),
        _ptr(weight_scale),
        batch,
        1,
        *activation_scalars,
        0,
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
    situ_beta: float,
    situ_linear_beta: float,
):
    gateup_block_m = config.GATEUP_BLOCK_M or config.BLOCK_M
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
        BLOCK_TILE_SIZE_M=gateup_block_m,
        BLOCK_TILE_SIZE_N=config.BLOCK_N,
        BLOCK_TILE_SIZE_K=config.BLOCK_K,
        stage="gateup",
        alg="prefill_1x4",
        E=problem.experts,
        act_quant_type_str=act_quant_type_str,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
        METADATA_TILE_SIZE_M=config.BLOCK_M,
    )
    task_num = int(sorted_expert_ids.shape[0])
    activation_scalars = _activation_scalars(
        activation_str, situ_beta, situ_linear_beta, swiglu_limit
    )
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
        *activation_scalars,
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

    output_padding_bytes = config.down_output_padding_bytes
    output_row_size = problem.model_dim + (
        output_padding_bytes // hidden_states.element_size()
        if output_padding_bytes is not None
        else 0
    )
    gemm2_out = torch.empty(
        [sorted_expert_ids.shape[0] * config.BLOCK_M, output_row_size],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    down_tile_n = {
        "default": 128,
        "1x4_64x256": 256,
        "2x4": 256,
        "1x8": 512,
    }[config.down_path]
    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=down_tile_n,
        stage="down",
        alg="prefill_1x4",
        E=problem.experts,
        USE_ATOMIC_WRITE=False,
        act_quant_type_str=problem.quant_type,
        down_path=config.down_path,
        down_output_padding_bytes=output_padding_bytes,
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
    if config.down_path == "default":
        _launch(down_kernel, *down_args, *activation_scalars)
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
    sorted_sum(problem.topk, problem.model_dim, output_padding_bytes)(
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
    weight_dtype_str = (
        "bf16" if w1.dtype == torch.bfloat16 else "fp4" if is_mxfp4 else "fp8"
    )
    force_batch1_path = problem.batch > 1
    fused_down_clear = is_mxfp4 and force_batch1_path
    topk_weight = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )
    gemm1_out = _gateup_output(hidden_states, problem)
    output_factory = torch.empty if fused_down_clear else torch.zeros
    cur_out = output_factory(
        [problem.batch, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
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
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
    )
    activation_scalars = _activation_scalars(
        activation_str, situ_beta, situ_linear_beta, swiglu_limit
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        topk_ids,
        cur_out if fused_down_clear else topk_weight,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        problem.batch,
        *activation_scalars,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=32 if is_mxfp4 and force_batch1_path else 64,
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
        problem.batch,
        *activation_scalars,
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
    weight_dtype_str = (
        "bf16"
        if w1.dtype == torch.bfloat16
        else "fp4" if w1.dtype == torch.float4_e2m1fn_x2 else "fp8"
    )
    block_n = 64 if config.BLOCK_N == 16 else config.BLOCK_N
    block_k = 64 if config.BLOCK_K == 16 else config.BLOCK_K
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
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        BLOCK_TILE_SIZE_K=block_k,
        stage="gateup",
        alg="splitk",
        E=problem.experts,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
    )
    activation_scalars = _activation_scalars(
        activation_str, situ_beta, situ_linear_beta, swiglu_limit
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
        *activation_scalars,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        BLOCK_TILE_SIZE_K=block_k,
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
        *activation_scalars,
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
    config = Config.from_string(config_string)
    gate_mode = GateMode(gate_mode)
    if gate_mode not in (GateMode.SEPARATED, GateMode.INTERLEAVE):
        raise RuntimeError(
            f"Unsupported gate mode for the whole-graph backend: {gate_mode.value}"
        )
    is_bf16 = (
        w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and quant_type == QuantType.No
    )
    is_fp8 = (
        w1.dtype == torch.float8_e4m3fnuz
        and w2.dtype == torch.float8_e4m3fnuz
        and quant_type in (QuantType.per_Token, QuantType.per_Tensor)
    )
    is_mxfp4 = (
        w1.dtype == torch.float4_e2m1fn_x2
        and w2.dtype == torch.float4_e2m1fn_x2
        and quant_type == QuantType.per_1x32
        and get_gfx() == "gfx950"
    )
    sit_uv2 = getattr(ActivationType, "Situv2", None)
    supported_activations = (ActivationType.Silu, ActivationType.Swiglu)
    if sit_uv2 is not None:
        supported_activations += (sit_uv2,)
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation not in supported_activations
        or not (is_bf16 or is_fp8 or is_mxfp4)
    ):
        raise RuntimeError("Unsupported input for the gfx942 FlyDSL MoE backend")
    if (is_fp8 or is_mxfp4) and (w1_scale is None or w2_scale is None):
        raise ValueError("Quantized weights require both w1_scale and w2_scale")
    if gate_mode == GateMode.INTERLEAVE and not is_mxfp4:
        raise RuntimeError(
            "The whole-graph backend supports interleaved gate/up weights only "
            "for MXFP4"
        )
    if not getattr(w1, "is_shuffled", False) or not getattr(w2, "is_shuffled", False):
        raise RuntimeError(
            "The whole-graph backend requires preshuffled w1 and w2 tensors"
        )

    activation_str = (
        "situv2"
        if sit_uv2 is not None and activation == sit_uv2
        else "swiglu" if activation == ActivationType.Swiglu else "silu"
    )
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
            config,
            problem,
            activation_str,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
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
            gate_mode == GateMode.INTERLEAVE,
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
            gate_mode == GateMode.INTERLEAVE,
        )
    raise RuntimeError(f"Unsupported batch-size {problem.batch}")


def run_flydsl_moe_gfx942_impl(
    request: FusedMoeRequest,
    config_string: str,
) -> torch.Tensor:
    if (
        request.doweight_stage1
        or request.bias1 is not None
        or request.bias2 is not None
    ):
        raise RuntimeError(
            "The FlyDSL whole-graph backend does not support bias or doweight_stage1"
        )
    if request.hidden_pad or request.intermediate_pad:
        raise RuntimeError(
            "The FlyDSL whole-graph backend does not support padded dimensions"
        )
    if (
        request.q_dtype_w == torch.float4_e2m1fn_x2
        and request.q_dtype_a != torch.bfloat16
    ):
        raise RuntimeError("The MXFP4 whole-graph backend requires BF16 activations")
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
        1.0 if request.beta is None else float(request.beta),
        1.0 if request.linear_beta is None else float(request.linear_beta),
        GateMode.SEPARATED if request.gate_mode is None else request.gate_mode,
    )
