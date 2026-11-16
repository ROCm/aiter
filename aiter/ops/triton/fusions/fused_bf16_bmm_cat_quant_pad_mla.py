# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_bf16_bmm_cat_quant_pad_mla import (
    _fused_bf16_bmm_cat_quant_pad_mla_kernel,
)
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

_LOGGER = AiterTritonLogger()


def _get_config(M: int, N: int, K: int, B: int) -> tuple[dict, bool]:
    return get_gemm_config("FUSED-BF16-BMM-CAT-QUANT-PAD-MLA", M, N, K, B=B)


def fused_bf16_bmm_cat_quant_pad_mla(
    q_nope: torch.Tensor,
    w_uk_t: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    num_padded_heads: int | None = None,
    out: torch.Tensor | None = None,
    fp8_max: float | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    MLA decode query prep in one launch: the bf16 W_UK absorb BMM, concat with
    the roped q_pe, static per-tensor FP8 quantization, and head padding by
    duplication. Equivalent to

        ql_nope = torch.bmm(q_nope, w_uk_t).transpose(0, 1)       # (M, QH, L)
        q = torch.cat((ql_nope, q_pe), dim=-1)                    # (M, QH, L + R)
        q = (q.float() * (1 / q_scale)).clamp(-fp8_max, fp8_max).to(fp8)
        q = q.repeat_interleave(num_padded_heads // QH, dim=1)

    Tuned configs exist only for gfx950. On other archs, pass config.

    Args:
        q_nope (torch.Tensor): (QH, M, P) bf16, may be a strided view.
        w_uk_t (torch.Tensor): (QH, P, L) bf16, may be a strided view.
        q_pe (torch.Tensor): (M, QH, R) bf16 with RoPE already applied.
        q_scale (torch.Tensor): Single-element fp32 query scale.
        num_padded_heads (Optional[int]): Head count of the output, a multiple of
            QH. Defaults to QH (no padding).
        out (Optional[torch.Tensor]): Pre-allocated (M, num_padded_heads, L + R)
            fp8 e4m3 output.
        fp8_max (Optional[float]): Clamp bound. Defaults to the max of out.dtype;
            vLLM clamps fnuz at 224, so pass its bound to match it.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M/N/K,
            GROUP_SIZE_M, num_warps, num_stages, waves_per_eu,
            matrix_instr_nonkdim, cache_modifier). Defaults to the
            FUSED-BF16-BMM-CAT-QUANT-PAD-MLA config for the shape.

    Returns:
        torch.Tensor: (M, num_padded_heads, L + R) fp8.
    """
    _LOGGER.info(
        "FUSED_BF16_BMM_CAT_QUANT_PAD_MLA: q_nope=%s w_uk_t=%s q_pe=%s heads=%s",
        tuple(q_nope.shape),
        tuple(w_uk_t.shape),
        tuple(q_pe.shape),
        num_padded_heads,
    )
    qh, m, k = q_nope.shape
    qh_w, k_w, n = w_uk_t.shape
    m_pe, qh_pe, d_pe = q_pe.shape
    assert qh == qh_w == qh_pe, f"head mismatch: {qh}, {qh_w}, {qh_pe}"
    assert k == k_w, f"BMM K mismatch: {k} vs {k_w}"
    assert m == m_pe, f"token mismatch: {m} vs {m_pe}"
    assert (
        q_nope.dtype == w_uk_t.dtype == q_pe.dtype == torch.bfloat16
    ), "q_nope, w_uk_t and q_pe must be bf16"
    assert q_scale.numel() == 1, "q_scale must be a single per-tensor scale"
    if num_padded_heads is None:
        num_padded_heads = qh
    assert (
        num_padded_heads % qh == 0
    ), f"num_padded_heads ({num_padded_heads}) must be a multiple of {qh}"

    if out is None:
        out = torch.empty(
            (m, num_padded_heads, n + d_pe),
            dtype=get_fp8_e4m3_dtype(),
            device=q_nope.device,
        )
    assert out.shape == (m, num_padded_heads, n + d_pe), f"bad out shape {out.shape}"
    assert out.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ), f"out must be fp8 e4m3, got {out.dtype}"
    if fp8_max is None:
        fp8_max = torch.finfo(out.dtype).max

    if config is None:
        config, _ = _get_config(m, n, k, qh)

    num_pid_m = triton.cdiv(m, config["BLOCK_SIZE_M"])
    num_pid_n = triton.cdiv(n, config["BLOCK_SIZE_N"])
    grid = (qh * num_pid_m * num_pid_n + qh * num_pid_m,)

    _fused_bf16_bmm_cat_quant_pad_mla_kernel[grid](
        q_nope,
        w_uk_t,
        q_pe,
        q_scale,
        out,
        m,
        n,
        k,
        qh,
        *q_nope.stride(),
        *w_uk_t.stride(),
        *q_pe.stride(),
        *out.stride(),
        FP8_MAX=fp8_max,
        D_PE=d_pe,
        BLOCK_D_PE=triton.next_power_of_2(d_pe),
        DUP=num_padded_heads // qh,
        **config,
    )
    return out
