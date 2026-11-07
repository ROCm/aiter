# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
from torch import Tensor

from aiter.ops.enum import QuantType

from ..jit.core import compile_ops
from ..utility import dtypes
from .fused_qk_norm_rope_cache_quant import _fused_qk_rmsnorm


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant",
    fc_name="fused_qk_rmsnorm_group_quant",
    develop=True,
)
def _fused_qk_rmsnorm_group_quant_kernel(
    q_out_quantized: Tensor | None = None,
    q_out_scale: Tensor | None = None,
    q: Tensor | None = None,
    q_weight: Tensor | None = None,
    q_epsilon: float = 1e-6,
    q_out_unquantized: Tensor | None = None,
    k_out: Tensor | None = None,
    q_res_out: Tensor | None = None,
    k: Tensor | None = None,
    k_weight: Tensor | None = None,
    k_epsilon: float | None = None,
    q_residual: Tensor | None = None,
    group_size: int = 128,
    transpose_scale: bool = False,
    gemma_norm: bool = False,
    amax_floor: float = 0.0,
) -> None: ...


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant",
    fc_name="fused_qk_rmsnorm_per_token_quant",
    develop=True,
)
def _fused_qk_rmsnorm_per_token_quant_kernel(
    q_out_quantized: Tensor,
    q_out_scale: Tensor,
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    q_out_unquantized: Tensor | None = None,
    k_out: Tensor | None = None,
    q_res_out: Tensor | None = None,
    k: Tensor | None = None,
    k_weight: Tensor | None = None,
    k_epsilon: float | None = None,
    q_residual: Tensor | None = None,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops(
    "module_fused_qk_rmsnorm_group_quant",
    fc_name="fused_qk_rmsnorm_mxfp8_kv_rope_quant",
    develop=True,
)
def _fused_qk_rmsnorm_mxfp8_kv_rope_quant_kernel(
    q_out: Tensor,
    q_out_scale: Tensor,
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    kv: Tensor,
    kv_weight: Tensor,
    kv_epsilon: float,
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
    rope_dim: int,
    amax_floor: float = 0.0,
    kv_out: Tensor | None = None,
    qat: Tensor | None = None,
    values: Tensor | None = None,
    scales: Tensor | None = None,
    pool: Tensor | None = None,
    pool_scale: Tensor | None = None,
    batch_ids: Tensor | None = None,
    cu_seqlens_q: Tensor | None = None,
    slots: Tensor | None = None,
    ring_start: int = 0,
    ring_slots: int = 1,
    slot_rows: int = 0,
    ring_stride: int = 1,
    run_rows: int = 0,
) -> None: ...


def fused_qk_rmsnorm_mxfp8_kv_rope_quant(
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    kv: Tensor,
    kv_weight: Tensor,
    kv_epsilon: float,
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
    *,
    rope_dim: int,
    amax_floor: float = 0.0,
    q_out: Tensor | None = None,
    q_out_scale: Tensor | None = None,
    kv_out: Tensor | None = None,
    qat: Tensor | None = None,
    values: Tensor | None = None,
    scales: Tensor | None = None,
    ring=None,
) -> tuple[Tensor, Tensor]:
    """Q: RMSNorm -> MXFP8 (fp8 e4m3 + uint8 e8m0 per 32), as triton
    `fused_dual_rmsnorm_mxfp8_quant`. KV: RMSNorm -> bf16, then the KV half of ATOM's
    `rope_quant_window` on that row (GPT-J RoPE on the last `rope_dim` columns, MXFP8 with
    step 2^ceil(log2(amax / 448))), into whichever of `qat` (dequantized bf16 row),
    `values` / `scales` (fp8 + e8m0 rows of any stride) and `ring` (`(pool, step, window)`,
    as rope_quant_window takes it) are given. `ring`'s pool is either `(data, scale)`, an
    fp8 plane [rows, width] and an e8m0 plane [rows, width / 32], or a bf16 [rows, width]
    tensor; `window`'s geometry counts rows. `kv_out` keeps the normed bf16 kv row.
    Returns `(q_out, q_out_scale)`.
    """
    m, kq = q.shape
    if not (0 < rope_dim <= kv.shape[-1] and rope_dim % 16 == 0):
        raise ValueError(
            f"rope_dim must be a positive multiple of 16, at most kv width; got {rope_dim}"
        )
    if q_out is None:
        q_out = q.new_empty((m, kq), dtype=dtypes.fp8)
    if q_out_scale is None:
        q_out_scale = q.new_empty((m, kq // 32), dtype=torch.uint8)
    pool, step, window = ring or (None, None, None)
    pool, pool_scale = pool if isinstance(pool, tuple) else (pool, None)
    if pool_scale is None and pool is not None and pool.dtype != torch.bfloat16:
        raise ValueError("A quantized ring needs separate (data, scale) planes")
    _fused_qk_rmsnorm_mxfp8_kv_rope_quant_kernel(
        q_out,
        q_out_scale,
        q,
        q_weight,
        q_epsilon,
        kv,
        kv_weight,
        kv_epsilon,
        cos,
        sin,
        positions,
        rope_dim,
        amax_floor,
        kv_out,
        qat,
        values,
        scales,
        pool,
        pool_scale,
        None if step is None else step.batch_ids,
        None if step is None else step.cu_seqlens_q,
        None if step is None else step.slots,
        0 if window is None else int(window.ring_start),
        1 if window is None else int(window.ring_slots),
        0 if window is None else int(window.slot_rows),
        1 if window is None else int(window.ring_stride),
        0 if window is None else int(window.run_rows),
    )
    return q_out, q_out_scale


def fused_qk_rmsnorm_group_quant(
    q_out_quantized: Tensor | None = None,
    q_out_scale: Tensor | None = None,
    q: Tensor | None = None,
    q_weight: Tensor | None = None,
    q_epsilon: float = 1e-6,
    q_out_unquantized: Tensor | None = None,
    k_out: Tensor | None = None,
    q_res_out: Tensor | None = None,
    k: Tensor | None = None,
    k_weight: Tensor | None = None,
    k_epsilon: float | None = None,
    q_residual: Tensor | None = None,
    group_size: int = 128,
    transpose_scale: bool = False,
    gemma_norm: bool = False,
    amax_floor: float = 0.0,
) -> None:
    """A uint8 / float8_e8m0fnu q_out_scale with fp8 q_out_quantized selects MXFP8
    (group_size=32, OCP e8m0 rounding as aiter triton `_mxfp8_quant_op`, each group's
    amax floored at `amax_floor`)."""
    # No-quant mode: when q_out_scale is None we only do RMSNorm and write to q_out_unquantized.
    no_quant = q_out_scale is None
    if no_quant:
        if q_out_unquantized is None:
            raise ValueError(
                "fused_qk_rmsnorm_group_quant: q_out_unquantized must be provided "
                "when q_out_scale is None (no-quant mode)"
            )
    else:
        if group_size <= 0:
            raise ValueError(
                "fused_qk_rmsnorm_group_quant requires group_size > 0; "
                "use fused_qk_rmsnorm_per_token_quant for per-token quant"
            )
        if q_out_quantized is None:
            raise ValueError(
                "fused_qk_rmsnorm_group_quant: q_out_quantized must be provided "
                "when q_out_scale is provided (quant mode)"
            )
        if q_out_quantized.dtype not in (dtypes.fp8, dtypes.fp4x2):
            raise ValueError(
                "fused_qk_rmsnorm_group_quant currently supports fp8/fp4x2 output quant only; "
                f"got {q_out_quantized.dtype}"
            )
        if q_out_quantized.dtype == dtypes.fp4x2:
            if transpose_scale:
                raise ValueError(
                    "fused_qk_rmsnorm_group_quant fp4x2 currently does not support transpose_scale=True"
                )
            n1 = q.size(1)
            if n1 % 2 != 0:
                raise ValueError(
                    f"q.size(1) must be even for fp4x2 packed output, got {n1}"
                )
            expected_packed = n1 // 2
            if q_out_quantized.size(1) != expected_packed:
                raise ValueError(
                    f"fp4x2 q_out_quantized.size(1) should be {expected_packed} "
                    f"(n1//2 packed), got {q_out_quantized.size(1)}"
                )

    _fused_qk_rmsnorm_group_quant_kernel(
        q_out_quantized,
        q_out_scale,
        q,
        q_weight,
        q_epsilon,
        q_out_unquantized,
        k_out,
        q_res_out,
        k,
        k_weight,
        k_epsilon,
        q_residual,
        group_size,
        transpose_scale,
        gemma_norm,
        amax_floor,
    )


def fused_qk_rmsnorm_per_token_quant(
    q_out_quantized: Tensor,
    q_out_scale: Tensor,
    q: Tensor,
    q_weight: Tensor,
    q_epsilon: float,
    q_out_unquantized: Tensor | None = None,
    k_out: Tensor | None = None,
    q_res_out: Tensor | None = None,
    k: Tensor | None = None,
    k_weight: Tensor | None = None,
    k_epsilon: float | None = None,
    q_residual: Tensor | None = None,
    gemma_norm: bool = False,
) -> None:
    if q_out_quantized.dtype != dtypes.fp8:
        raise ValueError(
            "fused_qk_rmsnorm_per_token_quant currently supports fp8 output quant only; "
            f"got {q_out_quantized.dtype}"
        )
    if q_out_scale.dim() != 2 or q_out_scale.size(1) != 1:
        raise ValueError(
            "fused_qk_rmsnorm_per_token_quant expects q_out_scale with shape [m, 1]"
        )

    _fused_qk_rmsnorm_per_token_quant_kernel(
        q_out_quantized,
        q_out_scale,
        q,
        q_weight,
        q_epsilon,
        q_out_unquantized,
        k_out,
        q_res_out,
        k,
        k_weight,
        k_epsilon,
        q_residual,
        gemma_norm,
    )


def fused_qk_rmsnorm(
    q_out_quantized: Tensor | None = None,
    q_out_scale: Tensor | None = None,
    q: Tensor | None = None,
    q_weight: Tensor | None = None,
    q_epsilon: float = 1e-6,
    q_out_unquantized: Tensor | None = None,
    k_out: Tensor | None = None,
    q_res_out: Tensor | None = None,
    k: Tensor | None = None,
    k_weight: Tensor | None = None,
    k_epsilon: float | None = None,
    q_residual: Tensor | None = None,
    gemma_norm: bool = False,
    quant_type: QuantType | None = QuantType.No,
    group_size: int | None = None,
    transpose_scale: bool = False,
) -> None:
    # Centralized interface
    if quant_type == QuantType.No:
        _fused_qk_rmsnorm(
            q_out_quantized, q, q_weight, q_epsilon, k_out, k, k_weight, k_epsilon
        )
        return
    elif quant_type == QuantType.per_Tensor:
        raise NotImplementedError("fused_qk_rmsnorm + per_tensor quant not supported")
    elif quant_type == QuantType.per_Token:
        fused_qk_rmsnorm_per_token_quant(
            q_out_quantized,
            q_out_scale,
            q,
            q_weight,
            q_epsilon,
            q_out_unquantized,
            k_out,
            q_res_out,
            k,
            k_weight,
            k_epsilon,
            q_residual,
            gemma_norm,
        )
        return
    elif group_size is None:
        if quant_type in [
            QuantType.per_1x128,
            QuantType.per_128x128,
            QuantType.per_256x128,
            QuantType.per_1024x128,
        ]:
            group_size = 128
        elif quant_type in [QuantType.per_1x32]:
            group_size = 32
    fused_qk_rmsnorm_group_quant(
        q_out_quantized,
        q_out_scale,
        q,
        q_weight,
        q_epsilon,
        q_out_unquantized,
        k_out,
        q_res_out,
        k,
        k_weight,
        k_epsilon,
        q_residual,
        group_size,
        transpose_scale,
        gemma_norm,
    )
