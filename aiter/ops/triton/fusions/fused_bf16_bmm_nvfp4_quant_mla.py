# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = (
    "gfx1250",
    "gfx1250-strict",
)
_CONFIG_NAME = "FUSED_BF16_BMM_NVFP4_QUANT_MLA"
# R1 MLA: the NoPE head dim P and the KV LoRA rank L.
D_NOPE = 128
KV_LORA = 512


def _get_config() -> dict:
    # Both supported archs read the gfx1250 tree; there is no gfx1250-strict one.
    cfg_dir = resolve_config_dir(
        "fusions", _CONFIG_NAME, backend="gluon", arch="gfx1250"
    )
    return dict(load_config_json(f"{cfg_dir}/DEFAULT.json")["any"])


def fused_bf16_bmm_nvfp4_quant_mla(
    q_nope: torch.Tensor,
    w_uk: torch.Tensor,
    q_nope_fp4: torch.Tensor | None = None,
    q_aux: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absorb W_UK into the NoPE query and quantize it into the R1 NVFP4 MLA
    query planes, in one launch.

    The result is the same bytes as a BF16 ``q_nope @ w_uk^T`` per head
    (``batched_gemm_bf16``) followed by the NoPE part of
    ``aiter.ops.mla_nvfp4.mla_nvfp4_quantize_q``: per row, a block of 16
    values gets the E4M3 scale of amax / 6 at the gain 6.72, and E2M1 codes
    over that scale. That holds while every value over its block's divisor
    stays below 2**25, that is for every absorbed value below about 2.2e9 in
    magnitude (only a scale saturated at 448 lets the quotient pass 6). Past
    that the codes saturate to 6.0, where the scalar producer's nearest-code
    search, comparing FP32 distances that round alike, keeps 4.0. A block
    that holds an Inf gets other bytes.

    Args:
        q_nope (torch.Tensor): ``(B, H, 128)`` BF16. The last dim must have
            unit stride. The other strides must be multiples of 8 elements.
        w_uk (torch.Tensor): ``(H, 512, 128)`` BF16, contiguous.
        q_nope_fp4 (Optional[torch.Tensor]): ``(B, H, 256)`` uint8,
            contiguous: the E2M1 codes, two per byte, low nibble first.
        q_aux (Optional[torch.Tensor]): ``(B, H, 128)`` uint8, contiguous.
            The kernel writes the 32 NoPE scales in bytes 64..95 and does not
            change the other bytes.

    Returns:
        tuple: ``(q_nope_fp4, q_aux)``.

    gfx1250 only. One workgroup per (head, split) reuses W_UK operands in
    registers; there are about as many workgroups as CUs. Graph-safe.
    """
    _LOGGER.info(
        "FUSED_BF16_BMM_NVFP4_QUANT_MLA: q_nope=%s w_uk=%s",
        tuple(q_nope.shape),
        tuple(w_uk.shape),
    )
    assert (
        get_arch() in _GLUON_SUPPORTED_ARCHS
    ), f"needs one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
    assert q_nope.dim() == 3 and q_nope.shape[2] == D_NOPE, "q_nope must be (B, H, 128)"
    batch, heads, _ = q_nope.shape
    assert w_uk.shape == (heads, KV_LORA, D_NOPE), "w_uk must be (H, 512, 128)"
    assert q_nope.dtype == torch.bfloat16 and w_uk.dtype == torch.bfloat16
    assert (
        q_nope.stride(2) == 1
        and q_nope.stride(0) % 8 == 0
        and q_nope.stride(1) % 8 == 0
    )
    assert q_nope.data_ptr() % 16 == 0, "q_nope rows must be 16-byte aligned"
    assert w_uk.is_contiguous(), "w_uk must be contiguous"
    assert w_uk.data_ptr() % 16 == 0, "w_uk must be 16-byte aligned"
    if q_nope_fp4 is None:
        q_nope_fp4 = torch.empty(
            (batch, heads, KV_LORA // 2), dtype=torch.uint8, device=q_nope.device
        )
    if q_aux is None:
        q_aux = torch.empty(
            (batch, heads, 128), dtype=torch.uint8, device=q_nope.device
        )
    assert (
        q_nope_fp4.shape == (batch, heads, KV_LORA // 2)
        and q_nope_fp4.dtype == torch.uint8
    )
    assert q_aux.shape == (batch, heads, 128) and q_aux.dtype == torch.uint8
    assert q_nope.is_cuda and all(
        t.device == q_nope.device for t in (w_uk, q_nope_fp4, q_aux)
    ), "q_nope, w_uk, q_nope_fp4 and q_aux must be on one GPU"
    assert q_nope_fp4.is_contiguous() and q_aux.is_contiguous()
    # The TDM stores need 16-byte bases; a contiguous view can start anywhere.
    assert (
        q_nope_fp4.data_ptr() % 16 == 0 and q_aux.data_ptr() % 16 == 0
    ), "q_nope_fp4 and q_aux must be 16-byte aligned"
    if batch == 0 or heads == 0:
        return q_nope_fp4, q_aux

    from aiter.ops.triton._gluon_kernels.gfx1250.fusions.fused_bf16_bmm_nvfp4_quant_mla import (
        _fused_bf16_bmm_nvfp4_quant_mla_kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
        create_wmma_layouts,
    )

    config = _get_config()
    block_m, num_warps = config["BLOCK_M"], config["num_warps"]
    block_n = KV_LORA // 2
    warp_bases = tuple(tuple(base) for base in config["WARP_BASES"])
    with torch.cuda.device(q_nope.device):
        splits = max(1, get_num_sms() // heads)
        tiles = triton.cdiv(triton.cdiv(batch, block_m), splits)
        wmma, op_a, op_b = create_wmma_layouts(num_warps, warp_bases)
        sh_a, sh_b = create_shared_layouts(block_m, block_n, D_NOPE, "TN", 16)
        grid = (heads, triton.cdiv(triton.cdiv(batch, block_m), tiles))
        _fused_bf16_bmm_nvfp4_quant_mla_kernel[grid](
            q_nope,
            w_uk,
            q_nope_fp4,
            q_aux[..., 64:],
            batch,
            KV_LORA,
            D_NOPE,
            tiles,
            q_nope.stride(1),
            q_nope.stride(0),
            w_uk.stride(0),
            w_uk.stride(1),
            q_nope_fp4.stride(1),
            q_nope_fp4.stride(0),
            q_aux.stride(1),
            q_aux.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=D_NOPE,
            SHARED_LAYOUT_A=sh_a,
            SHARED_LAYOUT_B=sh_b,
            WMMA_LAYOUT=wmma,
            WARP_BASES=warp_bases,
            OPERAND_LAYOUT_A=op_a,
            OPERAND_LAYOUT_B=op_b,
            num_warps=num_warps,
        )
    return q_nope_fp4, q_aux
