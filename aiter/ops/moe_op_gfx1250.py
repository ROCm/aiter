# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 PoC MoE ASM: MXFP4/MXFP8 activations and MXFP4 weights.

These kernels use 16-row gate/up blocks and a fixed SwiGLU limit of 10.
Use the packing helpers below; the row-interleaved FlyDSL weight layout is
different. Routing tensors follow ``moe_sorting_fwd``'s packed token/slot ABI.
"""

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.asm_guard import require_gfx1250_asm
from ..utility import dtypes
from .shuffle import shuffle_mxfp8fp4_b, shuffle_mxfp8fp4_scale

__all__ = [
    "moe_stage1_asm",
    "moe_stage2_asm",
    "shuffle_moe_input_scale_gfx1250",
    "shuffle_moe_weights_gfx1250",
]


@compile_ops(
    "module_moe_stage1_asm", fc_name="moe_stage1_gfx1250_asm", ffi_type="ctypes"
)
def _moe_stage1_asm(
    A: Tensor,
    W: Tensor,
    AScale: Tensor,
    WScale: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    out_scale: Tensor | None,
    topk: int,
    block_m: int,
    kernelName: str | None = None,
) -> None: ...


@compile_ops(
    "module_moe_stage2_asm", fc_name="moe_stage2_gfx1250_asm", ffi_type="ctypes"
)
def _moe_stage2_asm(
    A: Tensor,
    W: Tensor,
    AScale: Tensor,
    WScale: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    sorted_weights: Tensor,
    out: Tensor,
    block_m: int,
    kernelName: str | None = None,
) -> None: ...


def _output(shape, dtype, device, supplied, name):
    if supplied is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        tuple(supplied.shape) != tuple(shape)
        or supplied.dtype != dtype
        or supplied.device != device
        or not supplied.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous with shape {tuple(shape)}, dtype {dtype}, device {device}"
        )
    return supplied


def _check_activation(A, W, ndim):
    if A.ndim != ndim or W.ndim != 3:
        raise ValueError(f"A must be {ndim}D and W must be [experts, N, K/2]")
    if A.dtype not in (dtypes.fp4x2, torch.uint8, torch.float8_e4m3fn):
        raise ValueError("A must contain packed MXFP4 or MXFP8 E4M3FN values")


def moe_stage1_asm(
    A: Tensor,
    W: Tensor,
    AScale: Tensor,
    WScale: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    block_m: int = 32,
    dtype: torch.dtype = dtypes.bf16,
    kernelName: str = "",
    *,
    y: Tensor | None = None,
    out_scale: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Gate/up GEMM followed by ``silu(min(gate,10)) * clamp(up,-10,10)``.

    ``A`` is row-major ``[T,D/2]`` MXFP4 or ``[T,D]`` MXFP8 E4M3FN.
    ``W [E,2*I,D/2]`` and ``WScale [E,2*I,D/32]`` must be packed with
    ``shuffle_moe_weights_gfx1250(..., gate_up=True)``. ``AScale`` is E8M0
    gathered by sorted token and packed with ``shuffle_moe_input_scale_gfx1250``.

    ``block_m`` must equal the sorting block size (16, 32 or 64).
    ``sorted_token_ids`` stores ``token | (slot << 24)``; a padding token has
    ``token >= T``. ``sorted_expert_ids`` has one local expert ID per block,
    and ``num_valid_ids[0]`` is the padded sorted row count. All are GPU int32.
    Default dispatch selects a compatible N384/N512 kernel; ``kernelName``
    selects an exact symbol from ``hsa/gfx1250/moe_stage1/moe_stage1.csv``.

    BF16 returns ``[T,topk,I]``. With MXFP8 input, ``dtype=dtypes.fp4x2``
    (or uint8) returns ``(packed [T,topk,I/2], scale [pad32(capacity),I/32])``.
    The output scales already use sorted-row n32k4 layout and can be passed
    directly to ``moe_stage2_asm`` with the SAME routing and block_m.
    Fused quantization uses the PoC formula ``round_pow2(amax)/4`` with a
    1.5 power-of-two rounding threshold, then FP4 RNE conversion.
    It requires I%128==0. Unrouted outputs are zero; unused scales are 1.
    ``y`` and ``out_scale`` optionally supply reusable output buffers.
    """
    require_gfx1250_asm("moe_stage1_asm")
    _check_activation(A, W, 2)
    if dtype not in (dtypes.bf16, dtypes.fp4x2, torch.uint8):
        raise ValueError("stage1 output dtype must be BF16 or packed MXFP4")
    fq = dtype != dtypes.bf16
    if fq and A.dtype != torch.float8_e4m3fn:
        raise ValueError("fused MXFP4 output requires MXFP8 input")
    if W.shape[1] % 32:
        raise ValueError("W must have 2*I rows, with I divisible by 16")
    inter = W.shape[1] // 2
    if fq and inter % 128:
        raise ValueError("fused MXFP4 output requires inter_dim%128==0")
    if not fq and out_scale is not None:
        raise ValueError("BF16 stage1 must not receive out_scale")
    out = _output(
        (A.shape[0], topk, inter // 2 if fq else inter), dtype, A.device, y, "y"
    )
    if fq:
        rows = (sorted_token_ids.numel() + 31) // 32 * 32
        out_scale = _output(
            (rows, inter // 32), dtypes.fp8_e8m0, A.device, out_scale, "out_scale"
        )
    _moe_stage1_asm(
        A,
        W,
        AScale,
        WScale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        out_scale,
        topk,
        block_m,
        kernelName or None,
    )
    return (out, out_scale) if fq else out


def moe_stage2_asm(
    A: Tensor,
    W: Tensor,
    AScale: Tensor,
    WScale: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    sorted_weights: Tensor,
    block_m: int = 32,
    kernelName: str = "",
    *,
    y: Tensor | None = None,
) -> Tensor:
    """Down GEMM, multiply by routing weights, and reduce to BF16 ``[T,D]``.

    ``A`` is row-major ``[T,topk,I/2]`` MXFP4 or ``[T,topk,I]`` MXFP8.
    Pack ``W [E,D,I/2]`` and ``WScale [E,D,I/32]`` with
    ``shuffle_moe_weights_gfx1250(..., gate_up=False)``. ``AScale`` is gathered
    by both token and slot and packed with ``shuffle_moe_input_scale_gfx1250``;
    stage1 fused MXFP4 output scales are already in this layout.

    Routing and block_m follow ``moe_stage1_asm``. ``sorted_weights`` is a
    GPU FP32 vector parallel to ``sorted_token_ids``. Each expert contribution
    is rounded to BF16 before BF16 atomic accumulation, so topk>1 results may
    differ across launches. This API does not promise bitwise determinism.
    The output, including a supplied ``y``, is cleared on the current stream
    on EVERY call; graph replay does not accumulate previous results.
    """
    require_gfx1250_asm("moe_stage2_asm")
    _check_activation(A, W, 3)
    out = _output((A.shape[0], W.shape[1]), dtypes.bf16, A.device, y, "y")
    _moe_stage2_asm(
        A,
        W,
        AScale,
        WScale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        sorted_weights,
        out,
        block_m,
        kernelName or None,
    )
    return out


def _interleave_blocks(tensor):
    experts, rows, cols = tensor.shape
    # [gate0..gateI, up0..upI] -> [gate0..15, up0..15, gate16..31, ...].
    return (
        tensor.reshape(experts, 2, rows // 32, 16, cols)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(experts, rows, cols)
    )


def shuffle_moe_weights_gfx1250(
    W: Tensor, WScale: Tensor, *, gate_up: bool = False
) -> tuple[Tensor, Tensor]:
    """Pack logical MXFP4 weights and E8M0 scales for the gfx1250 MoE ASM.

    Shapes are preserved: ``W [E,N,K/2]``, ``WScale [E,N,K/32]``.
    For stage1, pass raw ``[gate; up]`` weights with ``gate_up=True`` to
    interleave 16-row blocks BEFORE the weight/scale tile shuffles.
    Stage2 uses ``gate_up=False``. This is an offline weight preparation step.
    """
    if W.ndim != 3 or WScale.ndim != 3:
        raise ValueError("W and WScale must be 3D expert tensors")
    if W.dtype not in (dtypes.fp4x2, torch.uint8):
        raise ValueError("W must contain packed MXFP4 bytes")
    if WScale.dtype not in (dtypes.fp8_e8m0, torch.uint8):
        raise ValueError("WScale must contain E8M0 bytes")
    experts, n, packed_k = W.shape
    if n % 32 or packed_k % 64:
        raise ValueError("weight packing requires N%32==0 and K%128==0")
    if tuple(WScale.shape) != (experts, n, packed_k // 16):
        raise ValueError("WScale must have shape [E,N,K/32] matching W [E,N,K/2]")
    if W.device != WScale.device:
        raise ValueError("W and WScale must be on the same device")
    w = W.view(torch.uint8).contiguous()
    s = WScale.view(torch.uint8).contiguous()
    if gate_up:
        w, s = _interleave_blocks(w), _interleave_blocks(s)
    w = shuffle_mxfp8fp4_b(w.reshape(experts * n, packed_k))
    s = shuffle_mxfp8fp4_scale(s.reshape(experts * n, packed_k // 16))
    return w.reshape(W.shape).view(W.dtype), s.reshape(WScale.shape).view(WScale.dtype)


def shuffle_moe_input_scale_gfx1250(scale: Tensor, sorted_token_ids: Tensor) -> Tensor:
    """Gather logical E8M0 scales in sorted order and pack them as n32k4.

    Stage1 takes ``scale [T,K/32]``; stage2 takes ``[T,topk,K/32]`` and uses
    BOTH parts of ``token | (slot << 24)``. Invalid/padding IDs use scale 1.
    The result is ``[round_up(sorted_token_ids.numel(),32),K/32]``. Capacity
    determines allocation, so no device-to-host count read is needed and this
    helper can be captured in a graph. Reuse with unchanged routing only.
    """
    if scale.ndim not in (2, 3) or scale.shape[-1] % 4:
        raise ValueError("scale must be [T,K/32] or [T,topk,K/32], with K%128==0")
    if scale.dtype not in (dtypes.fp8_e8m0, torch.uint8):
        raise ValueError("scale must contain E8M0 bytes")
    if sorted_token_ids.ndim != 1 or sorted_token_ids.dtype != torch.int32:
        raise ValueError("sorted_token_ids must be a 1D int32 tensor")
    if scale.device != sorted_token_ids.device:
        raise ValueError("scale and sorted_token_ids must be on the same device")
    raw = scale.view(torch.uint8).reshape(-1, scale.shape[-1])
    ids = sorted_token_ids.to(torch.int64)
    token = ids & 0xFFFFFF
    valid = token < scale.shape[0]
    row = token
    if scale.ndim == 3:
        slot = (ids >> 24) & 0xFF
        valid = valid & (slot < scale.shape[1])
        row = token * scale.shape[1] + slot
    if raw.shape[0] == 0:
        gathered = torch.full(
            (ids.numel(), scale.shape[-1]), 0x7F, dtype=torch.uint8, device=scale.device
        )
    else:
        gathered = raw.index_select(0, row.clamp(max=raw.shape[0] - 1))
        gathered = torch.where(valid[:, None], gathered, 0x7F)
    return shuffle_mxfp8fp4_scale(gathered).view(scale.dtype)
