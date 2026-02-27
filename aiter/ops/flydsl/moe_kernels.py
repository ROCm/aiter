# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management utilities.

Provides:

**Kernel name utilities**

- ``flydsl_kernel_name``         -- construct kernel name from compile parameters
- ``parse_flydsl_kernel_name``   -- parse kernel name back to parameters
- ``get_flydsl_stage1_kernels``  -- enumerate valid stage1 configurations
- ``get_flydsl_stage2_kernels``  -- enumerate valid stage2 configurations

**Low-level compile helpers**  (return raw ``exe`` callable)

- ``compile_flydsl_moe_stage1``  -- compile stage1 kernel
- ``compile_flydsl_moe_stage2``  -- compile stage2 kernel

**High-level public API**  (FlashInfer-style, handles alloc / dtype / shuffle)

- ``flydsl_moe_stage1``          -- fused gate+up GEMM (stage1)
- ``flydsl_moe_stage2``          -- down-projection GEMM (stage2)
"""

import functools
import re
from typing import Dict, Optional, Tuple

import torch

_KERNEL_NAME_RE = re.compile(
    r"flydsl_moe(\d+)_a(\w+)_w(\w+)_(\w+)_t(\d+)x(\d+)x(\d+)(?:_(\w+))?"
)


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
) -> str:
    """Construct a FlyDSL MOE kernel name encoding compile parameters.

    Format: flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}[_{mode}]

    Args:
        stage: 1 or 2
        a_dtype: activation dtype ("fp8", "fp4", "fp16", etc.)
        b_dtype: weight dtype ("fp4", "fp8", "fp16", etc.)
        out_dtype: output dtype ("bf16", "f16", "fp8", etc.)
        tile_m: M tile size (block_m)
        tile_n: N tile size
        tile_k: K tile size
        mode: optional mode suffix for stage2 ("reduce", "atomic")
    """
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    return name


def parse_flydsl_kernel_name(name: str) -> Optional[Dict]:
    """Parse a FlyDSL kernel name into its component parameters.

    Returns None if the name is not a valid FlyDSL kernel name.
    """
    if not name or not name.startswith("flydsl_moe"):
        return None
    m = _KERNEL_NAME_RE.match(name)
    if not m:
        return None
    return {
        "stage": int(m.group(1)),
        "a_dtype": m.group(2),
        "b_dtype": m.group(3),
        "out_dtype": m.group(4),
        "tile_m": int(m.group(5)),
        "tile_n": int(m.group(6)),
        "tile_k": int(m.group(7)),
        "mode": m.group(8) or "",
    }


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return a dict of kernelName -> params for all supported FlyDSL stage1 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [256] if is_fp4 else [128]
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [16, 32, 64]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                name = flydsl_kernel_name(1, a_dtype, b_dtype, out_dtype, tm, tn, tk)
                kernels[name] = {
                    "stage": 1,
                    "a_dtype": a_dtype,
                    "b_dtype": b_dtype,
                    "out_dtype": out_dtype,
                    "tile_m": tm,
                    "tile_n": tn,
                    "tile_k": tk,
                    "MPerBlock": tm,
                }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return a dict of kernelName -> params for all supported FlyDSL stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128]
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [32, 64]
    modes = ["atomic", "reduce"]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    name = flydsl_kernel_name(
                        2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                    )
                    kernels[name] = {
                        "stage": 2,
                        "a_dtype": a_dtype,
                        "b_dtype": b_dtype,
                        "out_dtype": out_dtype,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "mode": mode,
                        "MPerBlock": tm,
                    }
    return kernels


def compile_flydsl_moe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
):
    """Compile FlyDSL stage1 kernel (or return cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

        return compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            use_cshuffle_epilog=(out_dtype == "fp8"),
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
        )


def compile_flydsl_moe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
):
    """Compile FlyDSL stage2 kernel (or return cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        return compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm2

        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
        )


# ---------------------------------------------------------------------------
# Private: compiled kernel closures  (similar to FlashInfer _get_compiled_kernel)
# ---------------------------------------------------------------------------


@functools.cache
def _get_compiled_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
):
    """Compile and cache a stage1 kernel, return a ``tensor_api`` closure."""
    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
    )
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * 2 if is_fp4 else inter_dim
    _k_in = model_dim

    def tensor_api(
        out: torch.Tensor,
        a: torch.Tensor,
        w: torch.Tensor,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        sorted_ids: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        size_expert_ids_in: int,
    ) -> None:
        if is_fp4:
            empty_bias = torch.empty(0, device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream().cuda_stream
            exe(
                out,
                a,
                w,
                a_scale,
                w_scale,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                empty_bias,
                token_num,
                _n_in,
                _k_in,
                size_expert_ids_in,
                stream,
            )
        else:
            exe(
                out,
                a,
                w,
                a_scale,
                w_scale,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                token_num,
                _n_in,
                _k_in,
                size_expert_ids_in,
            )

    return tensor_api


@functools.cache
def _get_compiled_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
):
    """Compile and cache a stage2 kernel, return a ``tensor_api`` closure."""
    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
    )
    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    reduce_exe = None
    if not accumulate:
        from .kernels.moe_gemm_2stage import compile_moe_reduction

        reduce_exe = compile_moe_reduction(
            topk=topk,
            model_dim=model_dim,
            dtype_str=out_dtype,
        )

    def tensor_api(
        out: torch.Tensor,
        a: torch.Tensor,
        w: torch.Tensor,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        sorted_ids: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        blocks: int,
    ) -> None:
        if accumulate:
            target = out
        else:
            target = torch.empty(
                (token_num * topk * model_dim,),
                device=out.device,
                dtype=out.dtype,
            )

        if is_fp4:
            empty_bias = torch.empty(0, device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream().cuda_stream
            exe(
                target,
                a,
                w,
                a_scale,
                w_scale,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                empty_bias,
                token_num,
                _n_in,
                _k_in,
                blocks,
                stream,
            )
        else:
            exe(
                target,
                a,
                w,
                a_scale,
                w_scale,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                token_num,
                _n_in,
                _k_in,
                blocks,
            )

        if not accumulate:
            stream = torch.cuda.current_stream().cuda_stream
            reduce_exe(
                target.view(token_num, topk, model_dim),
                out,
                token_num,
                stream,
            )

    return tensor_api


# ---------------------------------------------------------------------------
# Public API  (similar to FlashInfer rmsnorm_fp4quant)
# ---------------------------------------------------------------------------

_DTYPE_TO_A_STR = {
    torch.float4_e2m1fn_x2: "fp4",
    torch.float8_e4m3fnuz: "fp8",
    torch.float8_e4m3fn: "fp8",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


def _infer_a_dtype(t: torch.Tensor) -> str:
    s = _DTYPE_TO_A_STR.get(t.dtype)
    if s is None:
        raise ValueError(f"Unsupported activation dtype: {t.dtype}")
    return s


def flydsl_moe_stage1(
    a: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused gate+up GEMM (MOE stage1) using FlyDSL.

    Parameters
    ----------
    a : torch.Tensor
        Activation tensor, shape ``(token_num, model_dim)``.
    w1 : torch.Tensor
        Gate+up weight, shape ``(E, 2*inter_dim, model_dim)`` (pre-shuffled).
    sorted_token_ids : torch.Tensor
        Sorted token IDs from moe_sorting.
    sorted_expert_ids : torch.Tensor
        Sorted expert IDs (one per M-tile block).
    num_valid_ids : torch.Tensor
        Number of valid token IDs.
    out : torch.Tensor, optional
        Output tensor ``(token_num, topk, inter_dim)``.
        If ``None``, will be allocated automatically.
    topk : int
        Number of experts per token.
    tile_m, tile_n, tile_k : int
        Tile sizes for the kernel.
    a_dtype : str
        Activation dtype (``"fp8"``, ``"fp4"``, ``"fp16"``, etc.).
    b_dtype : str
        Weight dtype (``"fp4"``, ``"fp8"``, etc.).
    out_dtype : str
        Output dtype (``"bf16"``, ``"f16"``).
    w1_scale : torch.Tensor, optional
        Weight scale factors.
    a1_scale : torch.Tensor, optional
        Activation scale factors.
    sorted_weights : torch.Tensor, optional
        Per-token expert weights (for weighted MOE).

    Returns
    -------
    torch.Tensor
        Stage1 output of shape ``(token_num, topk, inter_dim)``.
    """
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    if out is None:
        out = torch.empty(
            (token_num, topk, inter_dim), dtype=torch_out_dtype, device=a.device
        )

    dev = a.device
    flat_a_scale = (
        a1_scale.view(-1) if a1_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w1_scale.view(-1) if w1_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    tensor_api = _get_compiled_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
    )
    tensor_api(
        out.view(-1),
        a.view(-1),
        w1.view(-1),
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        token_num,
        sorted_expert_ids.shape[0],
    )

    return out


def flydsl_moe_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2) using FlyDSL.

    Supports ``atomic`` and ``reduce`` modes.  For reduce mode a temporary
    buffer is allocated internally and a reduction kernel produces the final
    output, avoiding atomic contention.

    Parameters
    ----------
    inter_states : torch.Tensor
        Intermediate activations from stage1, shape ``(token_num, topk, inter_dim)``.
    w2 : torch.Tensor
        Down-projection weight, shape ``(E, model_dim, inter_dim)``,
        must be pre-shuffled via ``shuffle_weight(w2, layout=(16,16))``.
    sorted_token_ids : torch.Tensor
        Sorted token IDs from moe_sorting.
    sorted_expert_ids : torch.Tensor
        Sorted expert IDs.
    num_valid_ids : torch.Tensor
        Number of valid token IDs.
    out : torch.Tensor, optional
        Output tensor ``(token_num, model_dim)``.
        If ``None``, will be allocated (zeroed for atomic mode).
    topk : int
        Number of experts per token.
    tile_m, tile_n, tile_k : int
        Tile sizes for the kernel.
    a_dtype : str
        Activation dtype (``"fp8"``, ``"fp4"``, ``"fp16"``, etc.).
    b_dtype : str
        Weight dtype (``"fp4"``, ``"fp8"``, etc.).
    out_dtype : str
        Output dtype (``"bf16"``, ``"f16"``).
    mode : str
        Execution mode: ``"atomic"`` (default) or ``"reduce"``.
    w2_scale : torch.Tensor, optional
        Weight scale factors (pre-shuffled via ``e8m0_shuffle`` for fp4).
    a2_scale : torch.Tensor, optional
        Activation scale factors.
    sorted_weights : torch.Tensor, optional
        Per-token expert weights.

    Returns
    -------
    torch.Tensor
        Stage2 output of shape ``(token_num, model_dim)``.
    """
    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    accumulate = mode != "reduce"

    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    if out is None:
        out = torch.zeros(
            (token_num, model_dim), dtype=torch_out_dtype, device=inter_states.device
        )
    elif accumulate:
        out.zero_()

    dev = inter_states.device
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.zeros(
            sorted_token_ids.shape, dtype=torch.float32, device=dev
        )
    )

    tensor_api = _get_compiled_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
    )
    tensor_api(
        out,
        inter_states,
        w2,
        a2_scale,
        w2_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        token_num,
        int(sorted_expert_ids.numel()),
    )

    return out
