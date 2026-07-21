# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from typing import Any, Optional
from functools import cache

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from dataclasses import dataclass
import flydsl.compiler as flyc
import flydsl.expr as fx


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
    """Launch a flydsl JIT-compiled kernel using flyc.compile pattern.
    Appends the current CUDA stream as the last argument."""
    stream = torch.cuda.current_stream()
    prepared_args = [
        _ptr(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    _run_compiled(kernel_fn, *prepared_args, stream)


from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    sorted_sum,
    invert_sorted_ids,
)


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


def fused_moe_gfx942(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config_string: str,
) -> Optional[torch.Tensor]:

    # decode kernel configs from kernel name
    kcfgs = Config.from_string(config_string)

    B = int(hidden_states.shape[0])
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation != ActivationType.Silu
        or w1.dtype != torch.float8_e4m3fnuz
        or w2.dtype != torch.float8_e4m3fnuz
    ):
        raise RuntimeError("Unsupported input for fused_moe_gfx942")

    if quant_type != QuantType.per_Token and quant_type != QuantType.per_Tensor:
        raise RuntimeError(f"Unsupported quant_type: {quant_type}")

    qtype_str = "ptpc" if quant_type == QuantType.per_Token else "per_tensor"

    E, N1, K1 = w1.shape
    N2, K2 = w2.shape[1], w2.shape[2]
    TOPK = topk_ids.shape[1]
    assert N1 == 2 * K2

    topk_w_f32 = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )

    gemm1_out = torch.empty(
        [B, TOPK, N1 // 2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    if kcfgs.use_prefill:
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = (
            moe_sorting(
                topk_ids,
                topk_weight,
                E,
                N2,  # reduce dim is same with output dim
                hidden_states.dtype,
                kcfgs.BLOCK_M,
                None,
                None,
                0,
            )
        )
        # weight dtype: native-fp8 or bf16 prefill_1x4 gateup.
        weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
        # native-fp8 prefill folds a per-token activation scale (ptpc) into the dequant;
        # default to ptpc act for both per_Token/per_Tensor weights.
        act_quant_type_str = "ptpc"

        DOWN_BLOCK_TILE_SIZE_N = 128

        # prefill_1x4 gateup input: native-fp8 needs a quantized activation plus its per-token
        # scale; bf16 passes the activation through with a dummy scale (unused by bf16 path).
        if weight_dtype_str == "fp8":
            quant_func = (
                aiter.get_hip_quant(aiter.QuantType.per_Token)
                if quant_type == QuantType.per_Token
                else _quant_per_tensor
            )
            gateup_in, a_scale = quant_func(
                hidden_states,
                scale=None,
                quant_dtype=w1.dtype,
                num_rows=None,
            )
            if quant_type == QuantType.per_Tensor:
                a_scale = a_scale.repeat(B, 1).contiguous()
            a_scale = a_scale.to(torch.float32).contiguous()
        else:
            gateup_in = hidden_states
            a_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

        # Compile and launch gateup kernel (prefill_1x4)
        gateup_kernel = _get_compiled_kernel(
            N=N1,
            K=K1,
            weight_dtype_str=weight_dtype_str,
            quant_type_str=qtype_str,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=kcfgs.BLOCK_M,
            BLOCK_TILE_SIZE_N=kcfgs.BLOCK_N,
            BLOCK_TILE_SIZE_K=None,  # kcfgs.BLOCK_K,
            stage="gateup",
            alg="prefill_1x4",
            E=E,
            act_quant_type_str=act_quant_type_str,
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
            (
                w1_scale
                if w1_scale is not None
                else torch.empty(0, device=hidden_states.device)
            ),
            a_scale,
            B,
            task_num,
        )

        if weight_dtype_str == "fp8":
            assert qtype_str in ["ptpc", "per_tensor"]
            down_in, down_in_scale = quant_func(
                gemm1_out.view(B * TOPK, -1),
                scale=None,
                quant_dtype=w2.dtype,
                num_rows=None,
            )
        else:
            down_in = gemm1_out
            down_in_scale = torch.empty(
                1, dtype=torch.float32, device=hidden_states.device
            )

        gemm2_out = torch.empty(
            [sorted_expert_ids.shape[0] * kcfgs.BLOCK_M, N2],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Compile and launch down kernel (splitk; consumes the bf16 gateup output directly,
        # no inter-stage quantization).
        down_kernel = _get_compiled_kernel(
            N=N2,
            K=K2,
            weight_dtype_str=weight_dtype_str,
            quant_type_str=qtype_str,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=kcfgs.BLOCK_M,
            BLOCK_TILE_SIZE_N=DOWN_BLOCK_TILE_SIZE_N,
            stage="down",
            alg="prefill_1x4",
            E=E,
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
            (
                w2_scale
                if w2_scale is not None
                else torch.empty(0, device=hidden_states.device)
            ),
            down_in_scale,
            B,
            task_num,
        )
        loc_ids = torch.empty([B, TOPK], dtype=torch.int32, device=hidden_states.device)

        invert_sorted_ids(TOPK)(sorted_ids, loc_ids, sorted_ids.shape[0], B)
        sorted_sum(TOPK, N2)(loc_ids, gemm2_out, cur_out, B)

        return cur_out

    if B == 1:
        assert N1 == 2 * K2
        cur_out = torch.zeros(
            [1, N2], dtype=hidden_states.dtype, device=hidden_states.device
        )
        # Stage 1: batch1 gateup
        gateup_batch1 = _get_compiled_kernel(
            N=N1,
            K=K1,
            weight_dtype_str="fp8",
            quant_type_str=qtype_str,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=16,
            BLOCK_TILE_SIZE_N=32,
            stage="gateup",
            alg="batch1",
            E=None,
        )
        _launch(
            gateup_batch1,
            hidden_states,
            w1,
            gemm1_out,
            topk_ids,
            topk_w_f32,
            (
                w1_scale
                if w1_scale is not None
                else torch.empty(0, device=hidden_states.device)
            ),
            TOPK,
        )
        # Stage 2: batch1 down
        down_batch1 = _get_compiled_kernel(
            N=N2,
            K=K2,
            weight_dtype_str="fp8",
            quant_type_str=qtype_str,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=16,
            BLOCK_TILE_SIZE_N=64,
            stage="down",
            alg="batch1",
            E=None,
        )
        _launch(
            down_batch1,
            gemm1_out,
            w2,
            cur_out,
            topk_ids,
            topk_w_f32,
            (
                w2_scale
                if w2_scale is not None
                else torch.empty(0, device=hidden_states.device)
            ),
            TOPK,
        )
        return cur_out

    elif 2 <= B <= 32:
        BLOCK_M = kcfgs.BLOCK_M
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = (
            moe_sorting(
                topk_ids,
                topk_weight,
                E,
                K1,
                hidden_states.dtype,
                BLOCK_M,
                expert_mask,
                num_local_tokens,
                moe_sorting_dispatch_policy,
            )
        )
        grid = int(sorted_expert_ids.shape[0])
        if B * TOPK <= E:
            grid = B * TOPK

        # Stage 1: batch gateup (splitk)
        gateup_kernel = _get_compiled_kernel(
            N=N1,
            K=K1,
            weight_dtype_str="fp8",
            quant_type_str=qtype_str,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_M,
            BLOCK_TILE_SIZE_N=64,
            stage="gateup",
            alg="splitk",
            E=E,
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
            (
                w1_scale
                if w1_scale is not None
                else torch.empty(0, device=hidden_states.device)
            ),
            B,
            grid,
        )

        # Stage 2: down (splitk)
        BLOCK_TILE_SIZE_N = 64
        down_kernel = _get_compiled_kernel(
            N=N2,
            K=K2,
            weight_dtype_str="fp8",
            quant_type_str=qtype_str,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_M,
            BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
            stage="down",
            alg="splitk",
            E=E,
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
            (
                w2_scale
                if w2_scale is not None
                else torch.empty(0, device=hidden_states.device)
            ),
            B,
            grid,
        )
        return cur_out

    else:
        raise RuntimeError(f"Unsupported batch-size {B}")
