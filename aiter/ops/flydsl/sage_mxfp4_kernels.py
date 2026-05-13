# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Sage MXFP4 Attention APIs (CDNA / gfx950).

Drop-in wrappers for the Triton MXFP4 sage attention path:
  - ``flydsl_sage_attn_mxfp4_func``     — replacement for ``fav3_sage_mxfp4_func``
  - ``fav3_sage_mxfp4_flydsl_wrapper``  — replacement for ``fav3_sage_mxfp4_wrapper``

The high-precision wrapper reuses the Triton ``sage_quant_mxfp4`` quantizer
verbatim — only the attention forward kernel is FlyDSL.

Architecture: gfx950 only (FP4 MFMA). head_dim=128 only initially.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

from aiter.utility.dtypes import fp8 as _fp8_dtype
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant_mxfp4 as _triton_sage_quant_mxfp4,
)

from .kernels.sage_attn_mxfp4_cdna import build_sage_attn_mxfp4_cdna_module


__all__ = [
    "flydsl_sage_attn_mxfp4_func",
    "fav3_sage_mxfp4_flydsl_wrapper",
]


_KERNEL_BLOCK_M_DEFAULT = 256
_KERNEL_BLOCK_N_DEFAULT = 128


def _check_gfx950(device: torch.device) -> str:
    try:
        arch = torch.cuda.get_device_properties(device.index).gcnArchName
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not arch_base.startswith("gfx950"):
        raise ValueError(
            f"flydsl_sage_attn_mxfp4_func requires gfx950 (FP4 MFMA), got {arch!r}"
        )
    return arch_base


@lru_cache(maxsize=64)
def _get_kernel(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    use_bias: bool,
    waves_per_eu: int,
    block_m: int,
    block_n: int,
):
    return build_sage_attn_mxfp4_cdna_module(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        use_bias=use_bias,
        waves_per_eu=waves_per_eu,
        block_m=block_m,
        block_n=block_n,
    )


def flydsl_sage_attn_mxfp4_func(
    q: torch.Tensor,                # uint8 packed FP4 (already quantized)
    k: torch.Tensor,                # uint8 packed FP4
    v: torch.Tensor,                # FP8
    q_descale: torch.Tensor,        # uint8 e8m0, [B, S, Hq, D//32] (BSHD)
    k_descale: torch.Tensor,        # uint8 e8m0, [B, S, Hkv, D//32] (BSHD)
    v_descale: torch.Tensor,        # f32, [B, Hkv, D]
    bias: Optional[torch.Tensor] = None,  # f32, [B, Hq, Q_NUM_BLKS, Sk] or None
    causal: bool = False,
    layout: str = "bshd",
    waves_per_eu: int = 2,
    block_m: Optional[int] = None,
    block_n: int = _KERNEL_BLOCK_N_DEFAULT,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Drop-in for ``fav3_sage_mxfp4_func``. Returns bf16 output, same
    BSHD/BHSD layout as q.

    q, k: shape ``[..., head_dim_bytes]`` where ``head_dim_bytes = head_dim/2``
        (FP4 packs 2 elements per byte).
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_sage_attn_mxfp4_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"q/k/v must be on the same device: q={q.device} k={k.device} v={v.device}"
        )
    if q.dtype != torch.uint8 or k.dtype != torch.uint8:
        raise ValueError(
            f"MXFP4 Q/K must be uint8 (packed FP4); got q.dtype={q.dtype} k.dtype={k.dtype}"
        )
    if q_descale.dtype != torch.uint8 or k_descale.dtype != torch.uint8:
        raise ValueError(
            f"q_descale/k_descale must be uint8 (e8m0); "
            f"got {q_descale.dtype}/{k_descale.dtype}"
        )
    if v_descale.dtype != torch.float32:
        raise ValueError(f"v_descale must be float32, got {v_descale.dtype}")
    if q.dim() != 4:
        raise ValueError(f"expected 4D tensor, got rank {q.dim()} ({tuple(q.shape)})")
    if layout not in ("bshd", "bhsd"):
        raise ValueError(f"layout must be 'bshd' or 'bhsd', got {layout!r}")

    _check_gfx950(q.device)

    # Dimension mapping
    if layout == "bshd":
        batch, seq_q, num_q_heads, head_dim_bytes = q.shape
        _, seq_k, num_kv_heads, _ = k.shape
        _, seq_k_v, _, head_dim_v = v.shape
    else:
        batch, num_q_heads, seq_q, head_dim_bytes = q.shape
        _, num_kv_heads, seq_k, _ = k.shape
        _, _, seq_k_v, head_dim_v = v.shape

    head_dim = head_dim_bytes * 2  # FP4 packs 2 elem/byte

    if head_dim != 128:
        raise ValueError(
            f"flydsl MXFP4 currently restricted to head_dim=128, got {head_dim}"
        )
    if seq_k != seq_k_v:
        raise ValueError(
            f"k seq_len ({seq_k}) must match v seq_len ({seq_k_v})"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    use_bias = bias is not None
    if use_bias and bias.dtype != torch.float32:
        raise ValueError(f"bias must be float32, got {bias.dtype}")

    if block_m is None:
        # Auto-pick BLOCK_M for occupancy (mirror sage v1).
        try:
            cu_count = torch.cuda.get_device_properties(
                q.device.index
            ).multi_processor_count
        except Exception:
            cu_count = 256
        grid_at_bm256 = batch * num_q_heads * ((seq_q + 255) // 256)
        block_m = 128 if grid_at_bm256 < cu_count else 256

    # Permute BHSD → BSHD (canonical for the kernel)
    if layout == "bhsd":
        q_b = q.permute(0, 2, 1, 3).contiguous()
        k_b = k.permute(0, 2, 1, 3).contiguous()
        v_b = v.permute(0, 2, 1, 3).contiguous()
        q_d = q_descale.permute(0, 2, 1, 3).contiguous()
        k_d = k_descale.permute(0, 2, 1, 3).contiguous()
    else:
        q_b = q.contiguous()
        k_b = k.contiguous()
        v_b = v.contiguous()
        q_d = q_descale.contiguous()
        k_d = k_descale.contiguous()

    # Pad seq_q to multiple of block_m
    seq_q_pad = ((seq_q + block_m - 1) // block_m) * block_m
    n_pad_q = seq_q_pad - seq_q
    if n_pad_q > 0:
        q_b = torch.nn.functional.pad(q_b, (0, 0, 0, 0, 0, n_pad_q))
        # Pad q_descale similarly along seq dim (1)
        q_d = torch.nn.functional.pad(q_d, (0, 0, 0, 0, 0, n_pad_q))

    # bf16 output, BSHD shape
    o = torch.empty(
        (batch, seq_q_pad, num_q_heads, head_dim),
        dtype=torch.bfloat16, device=q.device,
    )

    # Bias: pass dummy 1-byte tensor when None to keep kernel signature uniform
    if use_bias:
        bias_t = bias.contiguous()
    else:
        bias_t = torch.zeros(1, dtype=torch.float32, device=q.device)

    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError(
                f"`stream` must be on {q.device}, got {launch_stream.device}"
            )

        exe = _get_kernel(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
            use_bias=use_bias,
            waves_per_eu=waves_per_eu,
            block_m=block_m,
            block_n=block_n,
        )
        exe(
            q_b.reshape(-1),
            k_b.reshape(-1),
            v_b.reshape(-1),
            o.reshape(-1),
            q_d.reshape(-1),
            k_d.reshape(-1),
            v_descale,
            batch,
            seq_q_pad,
            seq_k,
            stream=launch_stream,
        )

    if n_pad_q > 0:
        o = o[:, :seq_q, :, :].contiguous()

    if layout == "bhsd":
        o = o.permute(0, 2, 1, 3).contiguous()

    return o


def fav3_sage_mxfp4_flydsl_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    layout: str = "bshd",
    q_smooth: bool = False,
    hadamard_rotation: bool = True,
    config: Optional[dict] = None,
    R: Optional[torch.Tensor] = None,
    BLOCK_R: int = 128,
    block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Drop-in for ``fav3_sage_mxfp4_wrapper``.

    Quantizes via the **Triton** ``sage_quant_mxfp4`` (untouched), then runs
    the FlyDSL MXFP4 attention kernel.
    """
    if not hadamard_rotation:
        raise NotImplementedError(
            "hadamard_rotation=False is not supported in the FlyDSL MXFP4 path"
        )
    if block_lut is not None:
        raise NotImplementedError(
            "block-sparse path not supported in FlyDSL MXFP4"
        )
    for tensor, name in zip([q, k, v], ["q", "k", "v"]):
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                f"Expected high-precision for {name}, got {tensor.dtype}"
            )

    if config is None:
        config = {
            "BLOCK_M": _KERNEL_BLOCK_M_DEFAULT,
            "BLOCK_N": _KERNEL_BLOCK_N_DEFAULT,
            "waves_per_eu": 2,
        }

    fp8_type = _fp8_dtype
    fp8_max = torch.finfo(fp8_type).max

    (q_q, q_d, k_q, k_d, v_q, v_d, delta_s) = _triton_sage_quant_mxfp4(
        q, k, v, fp8_type, fp8_max,
        BLKQ=config["BLOCK_M"], BLKK=64,
        layout=layout, R=R, BLOCK_R=BLOCK_R, q_smoothing=q_smooth,
    )

    return flydsl_sage_attn_mxfp4_func(
        q=q_q, k=k_q, v=v_q,
        q_descale=q_d, k_descale=k_d, v_descale=v_d,
        bias=delta_s,
        causal=causal, layout=layout,
        block_m=config["BLOCK_M"],
        block_n=config["BLOCK_N"],
        waves_per_eu=config["waves_per_eu"],
    )
