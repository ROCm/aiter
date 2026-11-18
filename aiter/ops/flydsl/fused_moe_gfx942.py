# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    sorted_sum,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.quant import per_group_quant_hip

_ACTIVATIONS = {
    ActivationType.Silu: "silu",
    ActivationType.Swiglu: "swiglu",
    ActivationType.SwigluStep: "swiglustep",
}

# Block-scaled FP8 takes the checkpoint's FP32 block scales, b in {128, 64} per
# dim (64 when vLLM refines 128x128 blocks for a TP shard that starts mid-block):
# w1 [E, 2 * ceil(inter_dim / b), ceil(model_dim / b)] with gate and up blocked
# separately, w2 [E, ceil(model_dim / b), ceil(inter_dim / b)]. The kernels read
# them as they are on every call, and prefill quantizes activations per (token,
# K group).
_BLOCK_QUANT_TYPES = (QuantType.per_1x128, QuantType.per_128x128)
# A dim of at most 64 is one block either way; it reads as 128.
_BLOCK_SIZES = (128, 64)


def _blocks(n, block):
    return -(-n // block)


def _scale_blocks(scale, experts, rows, k, parts=1):
    """(block_n, block_k) of FP32 scales [experts, parts * ceil(rows / parts /
    block_n), ceil(k / block_k)] with supported block sizes, else None. With
    parts=2 the two row halves (gate, up) are blocked separately."""
    if (
        scale is None
        or scale.dtype != torch.float32
        or scale.dim() != 3
        or scale.shape[0] != experts
    ):
        return None
    sizes = tuple(
        next((b for b in _BLOCK_SIZES if n * _blocks(dim, b) == count), None)
        for count, dim, n in (
            (scale.shape[1], rows // parts, parts),
            (scale.shape[2], k, 1),
        )
    )
    return None if None in sizes else sizes


def _accepted_shape(experts, rows, k, parts=1):
    dims = (
        "|".join(str(n * _blocks(d, b)) for b in _BLOCK_SIZES)
        for d, n in ((rows // parts, parts), (k, 1))
    )
    return f"[{experts}, {', '.join(dims)}]"


def _describe(scale):
    if scale is None:
        return "None"
    return f"{scale.dtype} {list(scale.shape)}"


def block_scale_groups(inter_dim, hidden_dim, w1_block_k=128, w2_block_k=128):
    """K groups (Gate/Up, Down) the prefill kernels scale and quantize
    activations by.

    Gate/Up uses the w1 scale block width when it divides hidden_dim, so each
    checkpoint block is scaled once, and 64 otherwise. The prefill Down reads one group per K core, 128 wide unless inter_dim
    ends in a half block. Groups never straddle w2 scale blocks, so 64-wide
    blocks get 64-wide Down groups; with inter_dim % 128 == 0 that leaves only
    decode.
    """
    gateup = w1_block_k if hidden_dim % w1_block_k == 0 else 64
    return gateup, min(128 if inter_dim % 128 == 0 else 64, w2_block_k)


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
        # The batch-1 kernels have a fixed tile, so any other tile would be misreported.
        tile = (self.BLOCK_M, self.BLOCK_N, self.BLOCK_K)
        if self.use_batch1_algorithm and tile != (16, 16, 16):
            raise ValueError("Direct decoding uses the 16_16_16 tile")

    def to_string(self):
        # Only direct decoding adds the fifth tile field, so existing strings are unchanged.
        batch1 = "_True" if self.use_batch1_algorithm else ""
        return (
            f"{self.BLOCK_M}_{self.BLOCK_N}_{self.BLOCK_K}_{self.use_prefill}{batch1}:"
            f"{self.down_path}:{self.down_output_padding_bytes}"
        )

    @classmethod
    def from_string(cls, data: str):
        parts = data.split(":")
        # Configs tuned before the Down suffix existed use the default Down path.
        if len(parts) == 1:
            parts += ["default", "0"]
        tile, down_path, padding = parts
        fields = tile.split("_")
        if len(fields) == 4:
            fields.append("False")
        block_m, block_n, block_k, use_prefill, use_batch1_algorithm = fields
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
        if self.use_batch1_algorithm and not 2 <= problem.batch <= 8:
            return "direct decoding requires 2 to 8 tokens"
        # Batch 1 runs the batch-1 kernels for any tile; tune it only as 16_16_16.
        tile = (self.BLOCK_M, self.BLOCK_N, self.BLOCK_K)
        if not self.use_prefill and problem.batch == 1 and tile != (16, 16, 16):
            return "batch 1 uses the 16_16_16 config"
        if self.down_path != "default":
            if self.BLOCK_M != 64:
                return "specialized Down paths require unchanged M64 Gate/Up metadata"
            # The native tuner's empty quant_type is a shape-only probe. The
            # runtime wrapper checks actual FP8 weight and activation support.
            if problem.quant_type not in ("", "ptpc", "per_tensor", "block"):
                return "specialized Down paths require FP8 weights"
            if self.down_path == "8x1_compact":
                if problem.inter_dim not in (192, 256, 384, 512, 640):
                    return "compact Down requires inter_dim in {192,256,384,512,640}"
                if problem.model_dim < 512:
                    return "compact Down requires model_dim >= 512"
                if not 0 < problem.experts <= 2048:
                    return "compact Down requires 1 to 2048 experts"
            scale_bytes = 256 * 4 if problem.quant_type == "ptpc" else 0
            if 64 * problem.inter_dim + scale_bytes + 4 * 16 * 64 * 2 > 64 * 1024:
                return "specialized M64 Down exceeds the 64 KiB LDS capacity"
        if problem.quant_type == "block" and self.use_prefill:
            if self.down_path != "1x4_64x256":
                return "block-scaled prefill requires the 1x4_64x256 Down path"
            # Each Gate/Up K tile must hold whole scale groups.
            if self.BLOCK_K % problem.scale_k1 != 0:
                return "block-scaled Gate/Up scale group must divide BLOCK_K"
            if problem.scale_k2 != (128 if problem.inter_dim % 128 == 0 else 64):
                return (
                    "block-scaled Down needs one scale group per 1x4_64x256 K core; "
                    "64-wide w2 scale blocks need inter_dim % 128 != 0"
                )
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
    scale_k1: int = 0
    scale_k2: int = 0
    # Checkpoint (block_n, block_k) of the w1 and w2 block scales.
    scale_block1: tuple[int, int] | None = None
    scale_block2: tuple[int, int] | None = None

    @classmethod
    def from_inputs(
        cls,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_type: QuantType,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
    ):
        experts, gateup_dim, hidden_dim = w1.shape
        model_dim, inter_dim = w2.shape[1], w2.shape[2]
        assert gateup_dim == 2 * inter_dim
        scale_k1 = scale_k2 = 0
        w1_blocks = w2_blocks = None
        if quant_type in _BLOCK_QUANT_TYPES:
            # Gate and up rows are blocked separately, as checkpoints store them.
            w1_blocks = _scale_blocks(w1_scale, experts, gateup_dim, hidden_dim, 2)
            w2_blocks = _scale_blocks(w2_scale, experts, model_dim, inter_dim)
            if w1_blocks is None or w2_blocks is None:
                raise ValueError(
                    "block-scaled FP8 needs FP32 checkpoint block scales, b in "
                    "{128, 64}: w1 [E, 2 * ceil(inter_dim / b), ceil(model_dim / b)] "
                    f"= {_accepted_shape(experts, gateup_dim, hidden_dim, 2)} and "
                    "w2 [E, ceil(model_dim / b), ceil(inter_dim / b)] = "
                    f"{_accepted_shape(experts, model_dim, inter_dim)}, got "
                    f"{_describe(w1_scale)} and {_describe(w2_scale)}"
                )
            if hidden_dim % 64 != 0 or inter_dim % 64 != 0:
                raise ValueError(
                    f"block-scaled FP8 needs K dims divisible by 64, got "
                    f"{hidden_dim=} {inter_dim=}"
                )
            scale_k1, scale_k2 = block_scale_groups(
                inter_dim, hidden_dim, w1_blocks[1], w2_blocks[1]
            )
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
                QuantType.per_1x128: "block",
                QuantType.per_128x128: "block",
            }[quant_type],
            scale_k1=scale_k1,
            scale_k2=scale_k2,
            scale_block1=w1_blocks,
            scale_block2=w2_blocks,
        )


def default_config(token: int) -> str:
    """Config for a token bucket with no tuned row: the batch-1 kernels at 2
    tokens, the 16-row split-K decode tiles to 16, the 128-wide ones to 256 and
    the prefill pipeline above."""
    if token == 2:
        return Config(16, 16, 16, False, use_batch1_algorithm=True).to_string()
    if token <= 16:
        return Config(16, 16, 16, False).to_string()
    if token <= 256:
        return Config(16, 128, 64, False).to_string()
    return Config(64, 128, 128, True).to_string()


def get_tune_space():
    prefill = [(256, 128), (128, 256), (128, 128)]
    configs = [
        Config(16, 16, 16, False),
        Config(16, 16, 16, False, use_batch1_algorithm=True),
        Config(16, 128, 64, False),
    ]
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
    scale_k=None,
    scale_block=None,
    has_num_local_tokens=False,
):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import compile_gemm

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
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        fused_down_clear=fused_down_clear,
        scale_k=scale_k,
        scale_block=scale_block,
        has_num_local_tokens=has_num_local_tokens,
    )


def _scale_kwargs(problem: _Problem, stage: str):
    """K group and checkpoint (block_n, block_k) of a block-scaled stage."""
    if problem.quant_type != "block":
        return {}
    if stage == "gateup":
        return {"scale_k": problem.scale_k1, "scale_block": problem.scale_block1}
    return {"scale_k": problem.scale_k2, "scale_block": problem.scale_block2}


def _launch(kernel_fn, *args):
    stream = torch.cuda.current_stream()
    prepared_args = [
        _ptr(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    _run_compiled(kernel_fn, *prepared_args, stream)


def _quant_per_tensor(x, scale=None, quant_dtype=torch.float8_e4m3fn, num_rows=None):
    assert scale is None

    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    xq = torch.empty_like(x, dtype=quant_dtype)
    if num_rows is None:
        flydsl_absmax()(x, amax)
    else:
        # Rows at and past num_rows (CUDA-graph padding) hold stale values.
        valid = torch.arange(x.shape[0], device=x.device)[:, None] < num_rows
        amax.copy_(torch.where(valid, x.abs(), 0).amax())
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
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config: Config,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
):
    # Opus also returns each route's sorted row; merged repeated expert IDs get -1.
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        _,
        _,
        loc_ids,
    ) = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.model_dim,
        hidden_states.dtype,
        config.BLOCK_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
        # sorted_sum writes every output row, so the sort need not clear one.
        accumulate=False,
        output_aux=AUX_SORT_OPUS,
    )
    cur_out = torch.empty(
        [problem.batch, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    act_quant_type_str = problem.quant_type
    if problem.quant_type == "block":
        quant_func = per_group_quant_hip
    elif quant_type == QuantType.per_Token:
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_Token)
    else:
        quant_func = _quant_per_tensor

    def group_kwargs(stage):
        if problem.quant_type != "block":
            return {}
        return {"group_size": _scale_kwargs(problem, stage)["scale_k"]}

    if weight_dtype_str == "fp8":
        gateup_in, a_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=w1.dtype,
            # Per-token scales are row-local; a per-tensor one must skip padding.
            num_rows=num_local_tokens if quant_type == QuantType.per_Tensor else None,
            **group_kwargs("gateup"),
        )
        a_scale = a_scale.to(torch.float32).contiguous()
    else:
        gateup_in = hidden_states
        a_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

    gemm1_out = _gateup_output(hidden_states, problem)
    if quant_type == QuantType.per_Tensor:
        # Gate/Up skips routes sorting drops, but the Down absmax reads every row.
        gemm1_out.zero_()
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
        **_scale_kwargs(problem, "gateup"),
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
            # Sorting pads each expert with a route that writes past the count.
            num_rows=(
                num_local_tokens * problem.topk
                if quant_type == QuantType.per_Tensor and num_local_tokens is not None
                else None
            ),
            **group_kwargs("down"),
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
        BLOCK_TILE_SIZE_N=128,
        stage="down",
        **_scale_kwargs(problem, "down"),
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
        from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm2_8x1_compact import (
            allocate_task_buffers,
        )

        # The split policy and its buffer capacities must use the same CU count.
        cu_count = get_cu_num()
        full_tasks, tail_tasks, counts = allocate_task_buffers(
            sorted_expert_ids, problem.experts, cu_count
        )
        _launch(
            down_kernel,
            *down_args,
            full_tasks,
            tail_tasks,
            counts,
            full_tasks.shape[0],
            tail_tasks.shape[0],
            cu_count,
        )
    else:
        _launch(down_kernel, *down_args)

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
    num_local_tokens: torch.Tensor | None = None,
):
    # No sort reads the count here: both kernels take it and skip the rows at
    # and past it, as they skip caller-marked (-1) pad rows.
    has_num_local_tokens = num_local_tokens is not None
    if has_num_local_tokens and num_local_tokens.dtype != torch.int32:
        num_local_tokens = num_local_tokens.to(torch.int32)
    # Without a count the kernels never read this slot.
    count_arg = num_local_tokens if has_num_local_tokens else topk_ids
    topk_weight = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )
    gemm1_out = _gateup_output(hidden_states, problem)
    # Gate/Up zeroes the output itself when the output row matches its input row.
    fused_down_clear = problem.model_dim == problem.hidden_dim
    allocate_output = torch.empty if fused_down_clear else torch.zeros
    cur_out = allocate_output(
        [problem.batch, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    route_count = problem.batch * problem.topk
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=32,
        stage="gateup",
        **_scale_kwargs(problem, "gateup"),
        alg="batch1",
        E=None,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
        fused_down_clear=fused_down_clear,
        has_num_local_tokens=has_num_local_tokens,
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        topk_ids,
        cur_out,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        count_arg,
        route_count,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=64,
        stage="down",
        **_scale_kwargs(problem, "down"),
        alg="batch1",
        E=None,
        has_num_local_tokens=has_num_local_tokens,
    )
    _launch(
        down_kernel,
        gemm1_out,
        w2,
        cur_out,
        topk_ids,
        topk_weight,
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device),
        count_arg,
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

    block_n = 64 if config.BLOCK_N == 16 else config.BLOCK_N
    gemm1_out = _gateup_output(hidden_states, problem)
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        stage="gateup",
        **_scale_kwargs(problem, "gateup"),
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
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        stage="down",
        **_scale_kwargs(problem, "down"),
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
    """Whole MoE (sort, Gate/Up, activation, Down, reduction) with FlyDSL on gfx942.

    Block-scaled FP8 (per_128x128 or per_1x128) takes the checkpoint's FP32
    block scales w1_scale [E, 2 * ceil(inter_dim / b), ceil(model_dim / b)]
    (gate blocks, then up blocks) and w2_scale [E, ceil(model_dim / b),
    ceil(inter_dim / b)], b in {128, 64} per dim (64 when vLLM refines checkpoint
    blocks for a TP shard). The kernels read them directly on every call, so
    in-place updates (weight reload) take effect without any conversion step.
    """
    # num_local_tokens: rows at and past it (CUDA-graph padding) run no route
    # and their output rows are unspecified, as in fused_moe. The sorted paths
    # hand it to moe_sorting; the batch-1 kernels read it themselves.
    if num_local_tokens is not None and not (
        isinstance(num_local_tokens, torch.Tensor) and num_local_tokens.is_cuda
    ):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend takes num_local_tokens as a CUDA tensor"
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
    is_bf16 = (
        w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and quant_type == QuantType.No
    )
    is_fp8 = (
        w1.dtype == torch.float8_e4m3fnuz
        and w2.dtype == torch.float8_e4m3fnuz
        and quant_type
        in (QuantType.per_Token, QuantType.per_Tensor, *_BLOCK_QUANT_TYPES)
    )
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation not in _ACTIVATIONS
        or not (is_bf16 or is_fp8)
    ):
        raise RuntimeError("Unsupported input for the gfx942 FlyDSL MoE backend")
    if is_fp8 and (w1_scale is None or w2_scale is None):
        raise ValueError("FP8 weights require both w1_scale and w2_scale")
    if is_bf16 and (w1_scale is not None or w2_scale is not None):
        raise NotImplementedError("BF16 weights do not support weight scales")

    if activation == ActivationType.SwigluStep and (
        not isinstance(swiglu_limit, (int, float)) or swiglu_limit <= 0
    ):
        raise ValueError(
            f"SwigluStep requires a positive swiglu_limit, got {swiglu_limit!r}"
        )

    activation_str = _ACTIVATIONS[activation]
    problem = _Problem.from_inputs(
        hidden_states, w1, w2, topk_ids, quant_type, w1_scale, w2_scale
    )
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
            num_local_tokens,
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
    if request.gate_mode not in (None, "separated"):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend only supports separated gate weights"
        )
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
    quant_dtype = (
        torch.bfloat16 if request.w1.dtype == torch.bfloat16 else torch.float8_e4m3fnuz
    )
    if request.q_dtype_a not in (None, quant_dtype):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph activation dtype must match the weight mode"
        )
    if request.q_dtype_w not in (None, quant_dtype):
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
    )
