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


def _get_rope_cache_config(M: int, N: int, K: int, B: int) -> tuple[dict, bool]:
    # Own family: the RoPE and KV-cache programs change the best BMM tiles.
    return get_gemm_config("FUSED-BF16-BMM-ROPE-CAT-QUANT-PAD-CACHE-MLA", M, N, K, B=B)


def _launch(
    q_nope: torch.Tensor,
    w_uk_t: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    num_padded_heads: int | None,
    out: torch.Tensor | None,
    fp8_max: float | None,
    config: dict | None,
    kv_c: torch.Tensor | None = None,
    k_pe: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    is_neox: bool = False,
) -> torch.Tensor:
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

    rope_and_cache = kv_cache is not None
    if rope_and_cache:
        cache_block_size = kv_cache.shape[1]
        cache_max = torch.finfo(kv_cache.dtype).max
        rope_cache_strides = (
            *cos_sin_cache.stride(),
            *kv_c.stride(),
            *k_pe.stride(),
            *kv_cache.stride(),
        )
        kv_programs = m
    else:
        cache_block_size, cache_max, rope_cache_strides, kv_programs = (
            0,
            0.0,
            9 * (0,),
            0,
        )

    if config is None:
        get_config = _get_rope_cache_config if rope_and_cache else _get_config
        config, _ = get_config(m, n, k, qh)

    num_pid_m = triton.cdiv(m, config["BLOCK_SIZE_M"])
    num_pid_n = triton.cdiv(n, config["BLOCK_SIZE_N"])
    grid = (qh * num_pid_m * num_pid_n + qh * num_pid_m + kv_programs,)

    _fused_bf16_bmm_cat_quant_pad_mla_kernel[grid](
        q_nope,
        w_uk_t,
        q_pe,
        q_scale,
        out,
        positions,
        cos_sin_cache,
        slot_mapping,
        kv_c,
        k_pe,
        kv_cache,
        k_scale,
        m,
        n,
        k,
        qh,
        cache_block_size,
        *q_nope.stride(),
        *w_uk_t.stride(),
        *q_pe.stride(),
        *out.stride(),
        *rope_cache_strides,
        FP8_MAX=fp8_max,
        CACHE_MAX=cache_max,
        D_PE=d_pe,
        BLOCK_D_PE=triton.next_power_of_2(d_pe),
        BLOCK_N_KV=triton.next_power_of_2(n),
        DUP=num_padded_heads // qh,
        ROPE_AND_CACHE=rope_and_cache,
        IS_NEOX=is_neox,
        **config,
    )
    return out


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
    return _launch(
        q_nope, w_uk_t, q_pe, q_scale, num_padded_heads, out, fp8_max, config
    )


def fused_bf16_bmm_rope_cat_quant_pad_cache_mla(
    q_nope: torch.Tensor,
    w_uk_t: torch.Tensor,
    q_pe: torch.Tensor,
    q_scale: torch.Tensor,
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    num_padded_heads: int | None = None,
    out: torch.Tensor | None = None,
    fp8_max: float | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    MLA decode query prep and KV-cache write in one launch. It is
    fused_bf16_bmm_cat_quant_pad_mla with the RoPE of q_pe and k_pe and the fp8
    cache write of vLLM's concat_and_cache_mla_rope_fused added:

        q_pe, k_pe = rope(positions, q_pe, k_pe)       # only tokens with slot >= 0
        kv_cache[slot] = fp8([kv_c | k_pe] / k_scale)  # only tokens with slot >= 0
        return fused_bf16_bmm_cat_quant_pad_mla(q_nope, w_uk_t, q_pe, ...)

    The RoPE output is bit-exact to the vLLM HIP kernel. The roped q_pe and k_pe
    are not written back. All tokens must be decode tokens. Tuned configs exist
    only for gfx950. On other archs, pass config.

    Args:
        q_nope, w_uk_t, q_scale, num_padded_heads, out, fp8_max: As in
            fused_bf16_bmm_cat_quant_pad_mla.
        config (Optional[dict]): Kernel tuning parameters, with the keys of
            fused_bf16_bmm_cat_quant_pad_mla. Defaults to the
            FUSED-BF16-BMM-ROPE-CAT-QUANT-PAD-CACHE-MLA config for the shape.
        q_pe (torch.Tensor): (M, QH, R) bf16 before RoPE. R is a power of two.
        kv_c (torch.Tensor): (M, L) bf16 compressed KV, L = w_uk_t.shape[-1].
        k_pe (torch.Tensor): (M, R) bf16 key pe before RoPE.
        kv_cache (torch.Tensor): (num_blocks, block_size, L + R) fp8 e4m3 cache.
        slot_mapping (torch.Tensor): (M,) contiguous int cache slot of each
            token. A token with a negative slot is padding.
        k_scale (torch.Tensor): Single-element fp32 cache scale.
        positions (torch.Tensor): (M,) contiguous int position of each token.
        cos_sin_cache (torch.Tensor): (max_position, R) bf16, the cos half then
            the sin half.
        is_neox (bool): NeoX pairs (i, i + R/2) if True, else GPT-J pairs
            (2i, 2i + 1).

    Returns:
        torch.Tensor: (M, num_padded_heads, L + R) fp8.
    """
    _LOGGER.info(
        "FUSED_BF16_BMM_ROPE_CAT_QUANT_PAD_CACHE_MLA: q_nope=%s w_uk_t=%s q_pe=%s "
        "kv_cache=%s heads=%s",
        tuple(q_nope.shape),
        tuple(w_uk_t.shape),
        tuple(q_pe.shape),
        tuple(kv_cache.shape),
        num_padded_heads,
    )
    m, _, d_pe = q_pe.shape
    n = w_uk_t.shape[-1]
    assert d_pe == triton.next_power_of_2(d_pe), f"RoPE dim {d_pe} not a power of 2"
    assert kv_c.shape == (m, n), f"bad kv_c shape {kv_c.shape}"
    assert k_pe.shape == (m, d_pe), f"bad k_pe shape {k_pe.shape}"
    assert (
        kv_c.dtype == k_pe.dtype == cos_sin_cache.dtype == torch.bfloat16
    ), "kv_c, k_pe and cos_sin_cache must be bf16"
    assert cos_sin_cache.shape[-1] == d_pe, "cos_sin_cache must have R columns"
    assert (
        kv_cache.dim() == 3 and kv_cache.shape[-1] == n + d_pe
    ), f"bad kv_cache shape {kv_cache.shape}"
    assert kv_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ), f"kv_cache must be fp8 e4m3, got {kv_cache.dtype}"
    assert k_scale.numel() == 1, "k_scale must be a single per-tensor scale"
    for name, t in (("positions", positions), ("slot_mapping", slot_mapping)):
        assert t.shape == (m,) and t.is_contiguous(), f"bad {name} shape {t.shape}"

    return _launch(
        q_nope,
        w_uk_t,
        q_pe,
        q_scale,
        num_padded_heads,
        out,
        fp8_max,
        config,
        kv_c=kv_c,
        k_pe=k_pe,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        k_scale=k_scale,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )
