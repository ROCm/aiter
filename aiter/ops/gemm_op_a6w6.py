# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import os

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from aiter import logger

from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..jit.utils.chip_info import get_gfx_runtime as get_gfx
from ..utility import dtypes

# The mxfp6 (E2M3, per-1x32 blockscale) asm gemm shares the a4w4 kernarg ABI.
# Its packed operand/scale layouts are produced by the helpers below and must
# match exactly what the `f6gemm_dmabig_kernel_func` kernel consumes.
_KERNEL_NAME = "f6gemm_dmabig_kernel_func"
_SAFE_FALLBACK_KERNEL_NAME = "f6gemm_dmabig_swz0_kernel_func"
_TILE = 256
_K_TILE = 128
_SCALE_GROUP_SIZE = 32
_PADK = 2  # K-padding steps (of 128) baked into the packed A/B layout
_PACKED_TILE_BYTES = 24576
_SCALE_TILE_BYTES = 1024
_PACK_LAYOUT = "mxfp6_c0c1_256_padk2"
_SHORT_K_SWIZZLE_LIMIT = 48 * _K_TILE
_GROUPED_SWIZZLE_MAX_M = 16 * 32 * _TILE
_GROUPED_SWIZZLE_MAX_N = 64 * _TILE
_BATCHED_PACK_MIN_ELEMENTS = 32 * 1024 * 1024
_BATCHED_PACK_BLOCK_M = 64
_BATCHED_PACK_K_BLOCKS = _K_TILE // _SCALE_GROUP_SIZE
_QUANT_BACKEND = (
    os.environ.get("AITER_MXFP6_QUANT_BACKEND", "").strip().lower() or "auto"
)
if _QUANT_BACKEND not in {"auto", "hip", "triton"}:
    raise ValueError(
        "AITER_MXFP6_QUANT_BACKEND must be one of auto, hip, or triton, "
        f"got {_QUANT_BACKEND!r}"
    )
try:
    _IS_GFX950 = torch.cuda.is_available() and get_gfx() == "gfx950"
except (KeyError, RuntimeError):
    _IS_GFX950 = False
_TUNED_CONFIG_KEY_COLUMNS = ("gfx", "cu_num", "M", "N", "K")
_TUNED_CONFIG_NUMERIC_COLUMNS = ("cu_num", "M", "N", "K", "splitK")
_TUNED_CONFIG_COLUMNS = frozenset(
    {
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "kernelName",
        "splitK",
    }
)


@compile_ops("module_quant", fc_name="quant_mxfp6_gemm_hip", develop=True)
def quant_mxfp6_gemm_hip_out(
    input: Tensor, packed: Tensor, packed_scale: Tensor
) -> None: ...


# ---------------------------------------------------------------------------
# host-side quantization + packing helpers
# ---------------------------------------------------------------------------
def _e2m3_table() -> np.ndarray:
    vals = np.empty(64, np.float32)
    for c in range(64):
        s = (c >> 5) & 1
        e = (c >> 3) & 3
        m = c & 7
        v = (m / 8.0) if e == 0 else (2.0 ** (e - 1)) * (1.0 + m / 8.0)
        vals[c] = -v if s else v
    return vals


E2M3 = _e2m3_table()
_POS = E2M3[0:32].copy()  # positive levels, monotonically increasing 0..7.5
_E2M3_MAX_EXP = 2  # floor(log2(7.5))

# ---------------------------------------------------------------------------
# Hadamard rotation along K (accuracy). MXFP6's per-1x32 e8m0 block
# scale loses precision when values in a block have very different magnitudes.
# A block-diagonal 32-point Walsh-Hadamard mixes each block before quantization.
# Because H is orthonormal, applying it to BOTH operands leaves the GEMM unchanged:
# (A@H)@(B@H)^T = A@(H@H^T)@B^T = A@B^T -- so it lives entirely in the quant
# path (fused into the Triton pack kernel; asm GEMM untouched).
#
# This is part of the packed-operand contract, not a runtime option: allowing
# callers to disable it independently for A and B can silently produce invalid
# GEMM results.


def _hadamard32_np() -> np.ndarray:
    H = np.ones((1, 1), np.float32)
    while H.shape[0] < _SCALE_GROUP_SIZE:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(float(_SCALE_GROUP_SIZE))).astype(np.float32)


_HAD32_NP = _hadamard32_np()
_HAD32_T: dict[torch.device, Tensor] = {}


def _had32_t(device: torch.device) -> Tensor:
    t = _HAD32_T.get(device)
    if t is None:
        t = torch.from_numpy(_HAD32_NP).to(device)
        _HAD32_T[device] = t
    return t


def _rotate_k32_torch(x: Tensor) -> Tensor:
    """Block-diagonal 32x32 Hadamard along the (contiguous) K axis. Rounds H to bf16
    to mirror the fused kernel's bf16 dot (both: bf16 operands, fp32 accumulate)."""
    R, K = x.shape
    h = _had32_t(x.device).to(torch.bfloat16).float()
    return (
        x.float().reshape(R, K // _SCALE_GROUP_SIZE, _SCALE_GROUP_SIZE) @ h
    ).reshape(R, K)


def _rotate_k32_np(x: np.ndarray) -> np.ndarray:
    R, K = x.shape
    return (
        x.reshape(R, K // _SCALE_GROUP_SIZE, _SCALE_GROUP_SIZE).astype(np.float32)
        @ _HAD32_NP
    ).reshape(R, K)


# ---------------------------------------------------------------------------
# fused Triton quantize+pack (bf16 -> mxfp6 codes + e8m0 scales, packed into the
# kernel's C0/C1 tile layout in ONE GPU pass). ~27x faster than the torch path,
# making per-call activation quantization cheap enough for inference.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _e2m3_dev(scaled):
        a = tl.minimum(tl.abs(scaled), 7.5)
        # Positive E2M3 levels are uniformly spaced within three regions:
        # [0, 2): 1/8, [2, 4): 1/4, [4, 7.5]: 1/2.  Encoding them directly
        # avoids per-element log2/exp2 while preserving the exact nearest-level
        # result of the exponent/mantissa formulation.
        quant_scale = tl.where(a < 2.0, 8.0, tl.where(a < 4.0, 4.0, 2.0))
        code_bias = tl.where(a < 2.0, 0.0, tl.where(a < 4.0, 8.0, 16.0))
        mag_code = tl.floor(a * quant_scale + 0.5) + code_bias
        return mag_code.to(tl.int32) + (scaled < 0.0).to(tl.int32) * 32

    @triton.jit
    def _quant_pack_kernel(
        x_ptr,
        a_ptr,
        s_ptr,
        M,
        NB,
        NK_PAD,
        stride_xm,
        h_ptr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        jb = pid % NB
        rblk = pid // NB
        rows = rblk * BLOCK_M + tl.arange(0, BLOCK_M)
        rm = rows < M
        r = rows[:, None]
        g8 = tl.arange(0, 8)[None, :]
        # coalesced contiguous [BM,32] load, then quantize all 32 and split into 4 phases
        xall = tl.load(
            x_ptr + r * stride_xm + jb * 32 + tl.arange(0, 32)[None, :],
            mask=rm[:, None],
            other=0.0,
        )
        # 32x32 Hadamard rotation FUSED in-register (no extra memory pass): mixes the
        # 32 K-elements of this block so mxfp6's per-block scale keeps precision.
        # bf16 MFMA (fp32 accum) -- fast, and plenty since the result feeds a 3-mantissa
        # E2M3 quant and the activation is already bf16.
        h = tl.load(
            h_ptr + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
        ).to(tl.bfloat16)
        xall = tl.dot(xall.to(tl.bfloat16), h)
        amax = tl.max(tl.abs(xall), 1)
        safe = tl.maximum(amax, 1e-30)
        se = tl.minimum(tl.maximum(tl.floor(tl.log2(safe)) - 2.0, -127.0), 127.0)
        se = tl.where(amax > 0.0, se, 0.0)
        e8 = (se + 127.0).to(tl.uint8)
        codes = _e2m3_dev(xall * tl.exp2(-se)[:, None])  # [BM,32]
        cc = tl.reshape(codes, [BLOCK_M, 8, 2, 2])
        lo, hi = tl.split(cc)  # lo=b0-bit {s0,s2}, hi {s1,s3}
        c0, c2 = tl.split(lo)  # phase 0, 2
        c1, c3 = tl.split(hi)  # phase 1, 3
        b0 = (c0 | (c1 << 6)) & 0xFF
        b1 = ((c1 >> 2) | (c2 << 4)) & 0xFF
        b2 = ((c2 >> 4) | (c3 << 2)) & 0xFF
        t = rows // 256
        rem = rows % 256
        rb = rem // 16
        r16 = rem % 16
        step = jb // 4
        kg = jb % 4
        blk = rb * 64 + (kg * 16 + r16)
        base = (t * NK_PAD + step) * 24576
        c0base = (base + blk * 16)[:, None]
        c1base = (base + 16384 + blk * 8)[:, None]
        p0 = 3 * g8 + 0
        p1 = 3 * g8 + 1
        p2 = 3 * g8 + 2
        tl.store(
            a_ptr + tl.where(p0 < 16, c0base + p0, c1base + (p0 - 16)),
            b0.to(tl.uint8),
            mask=rm[:, None],
        )
        tl.store(
            a_ptr + tl.where(p1 < 16, c0base + p1, c1base + (p1 - 16)),
            b1.to(tl.uint8),
            mask=rm[:, None],
        )
        tl.store(
            a_ptr + tl.where(p2 < 16, c0base + p2, c1base + (p2 - 16)),
            b2.to(tl.uint8),
            mask=rm[:, None],
        )
        su = rem // 128
        sub = (rem % 128) // 16
        scaddr = (t * NK_PAD + step) * 1024 + su * 512 + kg * 128 + r16 * 8 + sub
        tl.store(s_ptr + scaddr, e8, mask=rm)

    @triton.jit
    def _quant_pack_4block_kernel(
        x_ptr,
        a_ptr,
        s_ptr,
        M,
        NSTEP,
        NK_PAD,
        stride_xm,
        h_ptr,
        BLOCK_M: tl.constexpr,
    ):
        """Pack four adjacent 32-value blocks per program.

        Flattening [row, K-block] makes each row's full 128-value K tile
        contiguous in the load stream.  It also halves the program count versus
        the single-block kernel while retaining the exact same dot, quantization,
        and physical C0/C1 stores for every block.
        """
        pid = tl.program_id(0)
        step = pid % NSTEP
        rblk = pid // NSTEP
        flat = tl.arange(0, BLOCK_M * 4)
        rows = rblk * BLOCK_M + flat // 4
        kg = flat % 4
        rm = rows < M
        xall = tl.load(
            x_ptr
            + rows[:, None] * stride_xm
            + step * 128
            + kg[:, None] * 32
            + tl.arange(0, 32)[None, :],
            mask=rm[:, None],
            other=0.0,
        )
        h = tl.load(
            h_ptr + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
        ).to(tl.bfloat16)
        xall = tl.dot(xall.to(tl.bfloat16), h)
        amax = tl.max(tl.abs(xall), 1)
        safe = tl.maximum(amax, 1e-30)
        se = tl.minimum(tl.maximum(tl.floor(tl.log2(safe)) - 2.0, -127.0), 127.0)
        se = tl.where(amax > 0.0, se, 0.0)
        e8 = (se + 127.0).to(tl.uint8)
        codes = _e2m3_dev(xall * tl.exp2(-se)[:, None])
        cc = tl.reshape(codes, [BLOCK_M * 4, 8, 2, 2])
        lo, hi = tl.split(cc)
        c0, c2 = tl.split(lo)
        c1, c3 = tl.split(hi)
        b0 = (c0 | (c1 << 6)) & 0xFF
        b1 = ((c1 >> 2) | (c2 << 4)) & 0xFF
        b2 = ((c2 >> 4) | (c3 << 2)) & 0xFF
        t = rows // 256
        rem = rows % 256
        rb = rem // 16
        r16 = rem % 16
        blk = rb * 64 + (kg * 16 + r16)
        base = (t * NK_PAD + step) * 24576
        c0base = (base + blk * 16)[:, None]
        c1base = (base + 16384 + blk * 8)[:, None]
        g8 = tl.arange(0, 8)[None, :]
        p0 = 3 * g8 + 0
        p1 = 3 * g8 + 1
        p2 = 3 * g8 + 2
        tl.store(
            a_ptr + tl.where(p0 < 16, c0base + p0, c1base + (p0 - 16)),
            b0.to(tl.uint8),
            mask=rm[:, None],
        )
        tl.store(
            a_ptr + tl.where(p1 < 16, c0base + p1, c1base + (p1 - 16)),
            b1.to(tl.uint8),
            mask=rm[:, None],
        )
        tl.store(
            a_ptr + tl.where(p2 < 16, c0base + p2, c1base + (p2 - 16)),
            b2.to(tl.uint8),
            mask=rm[:, None],
        )
        su = rem // 128
        sub = (rem % 128) // 16
        scaddr = (t * NK_PAD + step) * 1024 + su * 512 + kg * 128 + r16 * 8 + sub
        tl.store(s_ptr + scaddr, e8, mask=rm)


def quant_mxfp6(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a [R, K] float array to mxfp6 (E2M3) codes + e8m0 block scales.

    Returns (codes[R, K] uint8 6-bit, scales[R, K//32] uint8 e8m0).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    R, K = x.shape
    assert K % _SCALE_GROUP_SIZE == 0, f"K must be a multiple of {_SCALE_GROUP_SIZE}"
    x = _rotate_k32_np(x)
    NB = K // _SCALE_GROUP_SIZE
    blk = x.reshape(R, NB, _SCALE_GROUP_SIZE)
    amax = np.abs(blk).max(axis=2)
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.where(amax > 0, amax, 1.0))).astype(np.int32)
    scale_exp = np.clip(exp - _E2M3_MAX_EXP, -127, 127)
    scale_exp = np.where(amax > 0, scale_exp, 0).astype(np.int32)
    scales = (scale_exp + 127).astype(np.uint8)

    scaled = blk / (2.0 ** scale_exp[:, :, None])
    mag = np.abs(scaled).ravel()
    idx = np.searchsorted(_POS, mag)
    idx = np.clip(idx, 1, 31)
    lo = idx - 1
    pick = np.where(np.abs(mag - _POS[idx]) < np.abs(mag - _POS[lo]), idx, lo)
    neg = (scaled.ravel() < 0).astype(np.int64)
    codes = (pick + neg * 32).astype(np.uint8).reshape(R, K)
    return codes, scales


def _pack32(blocks: np.ndarray) -> np.ndarray:
    """[N, 32] 6-bit codes -> [N, 24] bytes (little-endian 6-bit stream)."""
    g = blocks.reshape(-1, 8, 4).astype(np.uint16)
    b0 = (g[..., 0] | (g[..., 1] << 6)) & 0xFF
    b1 = ((g[..., 1] >> 2) | (g[..., 2] << 4)) & 0xFF
    b2 = ((g[..., 2] >> 4) | (g[..., 3] << 2)) & 0xFF
    out = np.stack([b0, b1, b2], axis=-1).astype(np.uint8)
    return out.reshape(blocks.shape[0], 24)


def pack_big(codes: np.ndarray, padK: int = _PADK) -> np.ndarray:
    """Re-tile [R, K] 6-bit codes into the kernel's C0/C1 sub-tile blob."""
    R, K = codes.shape
    assert R % _TILE == 0 and K % _K_TILE == 0, f"R%{_TILE} and K%{_K_TILE} required"
    nt, nk = R // _TILE, K // _K_TILE
    rb = np.repeat(np.arange(16), 64)
    L = np.tile(np.arange(64), 16)
    r16 = L % 16
    kg = L // 16
    local_row = rb * 16 + r16  # (1024,)

    t_ax = np.arange(nt)[:, None, None]
    s_ax = np.arange(nk)[None, :, None]
    row = t_ax * _TILE + local_row[None, None, :]  # (nt,1,1024)
    col = s_ax * _K_TILE + (kg * _SCALE_GROUP_SIZE)[None, None, :]  # (1,nk,1024)
    row_b, col_b = np.broadcast_arrays(row, col)  # (nt,nk,1024)
    blocks = codes[
        row_b[..., None],
        col_b[..., None] + np.arange(_SCALE_GROUP_SIZE),
    ]  # (nt,nk,1024,32)

    packed = _pack32(blocks.reshape(-1, _SCALE_GROUP_SIZE)).reshape(
        nt, nk, _SCALE_TILE_BYTES, 24
    )
    out = np.zeros((nt, nk + padK, _PACKED_TILE_BYTES), np.uint8)
    out[:, :nk, :16384] = packed[..., :16].reshape(nt, nk, 1024 * 16)
    out[:, :nk, 16384:] = packed[..., 16:].reshape(nt, nk, 1024 * 8)
    return np.ascontiguousarray(out.reshape(-1))


def pack_scale(S: np.ndarray, rows: int, padK: int = _PADK) -> np.ndarray:
    """Re-tile [R, K//32] e8m0 scales into the kernel's packed scale blob."""
    _R, NB = S.shape
    assert rows % _TILE == 0 and NB % 4 == 0, f"rows%{_TILE} and NB%4 required"
    nt, nk = rows // _TILE, NB // 4
    off = np.arange(_SCALE_TILE_BYTES)
    su = off // 512
    kg = (off % 512) // 128
    r16 = (off % 128) // 8
    sub = off % 8
    row_local = su * 128 + sub * 16 + r16  # (1024,)

    t_ax = np.arange(nt)[:, None, None]
    s_ax = np.arange(nk)[None, :, None]
    row = t_ax * _TILE + row_local[None, None, :]  # (nt,1,1024)
    block = s_ax * 4 + kg[None, None, :]  # (1,nk,1024)
    row_b, block_b = np.broadcast_arrays(row, block)
    vals = S[row_b, block_b]  # (nt,nk,1024)

    out = np.full((nt, nk + padK, _SCALE_TILE_BYTES), 127, np.uint8)
    out[:, :nk, :] = vals
    return np.ascontiguousarray(out.reshape(-1))


# ---------------------------------------------------------------------------
# GPU-native (torch) quantization + packing -- byte-identical to the numpy
# helpers above, but keeps everything on device so packing a GEMM operand
# costs microseconds instead of CPU-bound seconds.
# ---------------------------------------------------------------------------
_E2M3_T = None
_POS_T = None


def _tables(device: torch.device) -> tuple[Tensor, Tensor]:
    global _E2M3_T, _POS_T
    if _E2M3_T is None or _E2M3_T.device != device:
        _E2M3_T = torch.from_numpy(E2M3).to(device)
        _POS_T = torch.from_numpy(_POS).to(device)
    return _E2M3_T, _POS_T


def quant_mxfp6_torch(x: Tensor) -> tuple[Tensor, Tensor]:
    """torch/GPU version of quant_mxfp6: [R,K] float -> (codes uint8, scales uint8)."""
    if x.ndim != 2:
        raise ValueError(f"quant_mxfp6_torch expects a 2D [R, K] tensor, got {x.ndim}D")
    if x.shape[1] == 0 or x.shape[1] % _SCALE_GROUP_SIZE != 0:
        raise ValueError(
            "quant_mxfp6_torch requires K to be a positive multiple of "
            f"{_SCALE_GROUP_SIZE}, got {x.shape[1]}"
        )
    x = x.float()
    x = _rotate_k32_torch(x)
    R, K = x.shape
    NB = K // _SCALE_GROUP_SIZE
    blk = x.reshape(R, NB, _SCALE_GROUP_SIZE)
    amax = blk.abs().amax(dim=2)
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    exp = torch.floor(torch.log2(safe))
    scale_exp = torch.clamp(exp - _E2M3_MAX_EXP, -127, 127)
    scale_exp = torch.where(amax > 0, scale_exp, torch.zeros_like(scale_exp))
    scales = (scale_exp + 127).to(torch.uint8)

    scaled = blk / torch.pow(torch.tensor(2.0, device=x.device), scale_exp).unsqueeze(
        -1
    )
    # arithmetic E2M3 round-to-nearest (identical to the fused Triton _e2m3_dev)
    a = scaled.abs().clamp(max=7.5)
    isn = a >= 1.0
    ex = torch.floor(torch.log2(a.clamp(min=1.0))).clamp(max=2.0)
    base = torch.pow(torch.tensor(2.0, device=x.device), ex)
    step = base / 8.0
    mn = torch.floor((a - base) / step + 0.5)
    ms = torch.floor(a * 8.0 + 0.5)
    mag = torch.where(isn, (ex + 1.0) * 8.0 + mn, ms)
    codes = (
        (mag.to(torch.long) + (scaled < 0).to(torch.long) * 32)
        .to(torch.uint8)
        .reshape(R, K)
    )
    return codes, scales


def dequant_mxfp6_torch(codes: Tensor, scales: Tensor) -> Tensor:
    """Reconstruct fp32 values from mxfp6 codes + e8m0 scales (for reference)."""
    tab, _ = _tables(codes.device)
    v = tab[codes.long()]
    sf = torch.pow(torch.tensor(2.0, device=codes.device), scales.float() - 127)
    return v * sf.repeat_interleave(_SCALE_GROUP_SIZE, dim=1)


def _pack32_torch(blocks: Tensor) -> Tensor:
    g = blocks.reshape(-1, 8, 4).to(torch.int32)
    b0 = (g[..., 0] | (g[..., 1] << 6)) & 0xFF
    b1 = ((g[..., 1] >> 2) | (g[..., 2] << 4)) & 0xFF
    b2 = ((g[..., 2] >> 4) | (g[..., 3] << 2)) & 0xFF
    out = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return out.reshape(blocks.shape[0], 24)


def pack_big_torch(codes: Tensor, padK: int = _PADK) -> Tensor:
    dev = codes.device
    R, K = codes.shape
    nt, nk = R // _TILE, K // _K_TILE
    rb = torch.arange(16, device=dev).repeat_interleave(64)
    L = torch.arange(64, device=dev).repeat(16)
    r16 = L % 16
    kg = L // 16
    local_row = rb * 16 + r16
    t_ax = torch.arange(nt, device=dev).view(nt, 1, 1)
    s_ax = torch.arange(nk, device=dev).view(1, nk, 1)
    row = t_ax * _TILE + local_row.view(1, 1, _SCALE_TILE_BYTES)
    col = s_ax * _K_TILE + (kg * _SCALE_GROUP_SIZE).view(1, 1, _SCALE_TILE_BYTES)
    row_b, col_b = torch.broadcast_tensors(row, col)
    ar = torch.arange(_SCALE_GROUP_SIZE, device=dev)
    blocks = codes[row_b.unsqueeze(-1), col_b.unsqueeze(-1) + ar]  # (nt,nk,1024,32)
    packed = _pack32_torch(blocks.reshape(-1, _SCALE_GROUP_SIZE)).reshape(
        nt, nk, _SCALE_TILE_BYTES, 24
    )
    out = torch.zeros(
        (nt, nk + padK, _PACKED_TILE_BYTES), dtype=torch.uint8, device=dev
    )
    out[:, :nk, :16384] = packed[..., :16].reshape(nt, nk, 1024 * 16)
    out[:, :nk, 16384:] = packed[..., 16:].reshape(nt, nk, 1024 * 8)
    return out.reshape(-1).contiguous()


def pack_scale_torch(S: Tensor, rows: int, padK: int = _PADK) -> Tensor:
    dev = S.device
    _R, NB = S.shape
    nt, nk = rows // _TILE, NB // 4
    off = torch.arange(_SCALE_TILE_BYTES, device=dev)
    su = off // 512
    kg = (off % 512) // 128
    r16 = (off % 128) // 8
    sub = off % 8
    row_local = su * 128 + sub * 16 + r16
    t_ax = torch.arange(nt, device=dev).view(nt, 1, 1)
    s_ax = torch.arange(nk, device=dev).view(1, nk, 1)
    row = t_ax * _TILE + row_local.view(1, 1, _SCALE_TILE_BYTES)
    block = s_ax * 4 + kg.view(1, 1, 1024)
    row_b, block_b = torch.broadcast_tensors(row, block)
    vals = S[row_b, block_b]
    out = torch.full(
        (nt, nk + padK, _SCALE_TILE_BYTES),
        127,
        dtype=torch.uint8,
        device=dev,
    )
    out[:, :nk, :] = vals
    return out.reshape(-1).contiguous()


def _ceil(x: int, m: int) -> int:
    return (x + m - 1) // m * m


@functools.lru_cache(maxsize=8)
def _load_gemm_a6w6_configs(
    tuned_file: str,
) -> dict[tuple[str, int, int, int, int], dict[str, object]]:
    """Load and validate an A6W6 shape-tuning table."""
    if not os.path.exists(tuned_file):
        return {}
    try:
        configs = pd.read_csv(tuned_file)
    except pd.errors.EmptyDataError:
        return {}
    missing = _TUNED_CONFIG_COLUMNS - set(configs.columns)
    if missing:
        raise ValueError(
            f"{tuned_file} is missing required A6W6 columns: {sorted(missing)}"
        )
    if configs.empty:
        return {}

    configs = configs.copy()
    for column in _TUNED_CONFIG_NUMERIC_COLUMNS:
        configs[column] = pd.to_numeric(configs[column], errors="raise").astype(int)
    configs["gfx"] = configs["gfx"].astype(str).str.strip()
    configs["kernelName"] = configs["kernelName"].astype(str).str.strip()

    if (configs["gfx"] == "").any() or (configs["kernelName"] == "").any():
        raise ValueError(f"{tuned_file} contains an empty gfx or kernelName")
    if (configs["splitK"] != 0).any():
        raise ValueError("A6W6 tuned configs must use splitK=0")

    key_columns = list(_TUNED_CONFIG_KEY_COLUMNS)
    duplicate_rows = configs[configs.duplicated(key_columns, keep=False)]
    if not duplicate_rows.empty:
        raise ValueError(
            f"{tuned_file} contains duplicate A6W6 shape keys:\n"
            f"{duplicate_rows[key_columns + ['kernelName']].to_string(index=False)}"
        )
    return configs.set_index(key_columns).to_dict("index")


def clear_gemm_a6w6_config_cache() -> None:
    """Clear cached tuning data after a tuner updates the CSV."""
    _load_gemm_a6w6_configs.cache_clear()
    get_GEMM_A6W6_config.cache_clear()


def _default_gemm_a6w6_kernel(M: int, N: int, K: int) -> str:
    """Choose a safe untuned fallback for the physical launch shape.

    The optimized grouped-M kernel has compile-time swizzle bounds for short-K
    launches. Natural ordering has no such grid bound and is used outside them.
    """
    padM, padN, padK = _ceil(M, _TILE), _ceil(N, _TILE), _ceil(K, _K_TILE)
    short_k = padK <= _SHORT_K_SWIZZLE_LIMIT
    grouped_grid_in_bounds = (
        padM <= _GROUPED_SWIZZLE_MAX_M and padN <= _GROUPED_SWIZZLE_MAX_N
    )
    return (
        _KERNEL_NAME
        if not short_k or grouped_grid_in_bounds
        else _SAFE_FALLBACK_KERNEL_NAME
    )


@functools.lru_cache(maxsize=1024)
def get_GEMM_A6W6_config(
    M: int, N: int, K: int, tuned_file: str | None = None
) -> dict[str, object] | None:
    """Return an exact or physical-shape A6W6 tuning record."""
    tuned_file = tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_A6W6_FILE
    tuned_file = os.path.abspath(tuned_file)
    configs = _load_gemm_a6w6_configs(tuned_file)
    gfx, cu_num = get_gfx(), get_cu_num()
    candidates = [(M, N, K, "exact")]
    padded = (_ceil(M, _TILE), _ceil(N, _TILE), _ceil(K, _K_TILE))
    if padded != (M, N, K):
        candidates.append((*padded, "padded"))

    for candidate_M, candidate_N, candidate_K, match_kind in candidates:
        config = configs.get((gfx, cu_num, candidate_M, candidate_N, candidate_K))
        if config is not None:
            if AITER_LOG_TUNED_CONFIG:
                logger.info(
                    "A6W6 shape M:%s N:%s K:%s matched %s config "
                    "M:%s N:%s K:%s on %s/%s in %s: %s",
                    M,
                    N,
                    K,
                    match_kind,
                    candidate_M,
                    candidate_N,
                    candidate_K,
                    gfx,
                    cu_num,
                    tuned_file,
                    config["kernelName"],
                )
            return config

    if AITER_LOG_TUNED_CONFIG:
        logger.info(
            "A6W6 shape M:%s N:%s K:%s has no tuned config in %s; "
            "using the safe default kernel",
            M,
            N,
            K,
            tuned_file,
        )
    return None


def _select_gemm_a6w6_kernel(M: int, N: int, K: int, kernelName: str | None) -> str:
    if kernelName:
        return kernelName
    config = get_GEMM_A6W6_config(M, N, K)
    if config is not None:
        return str(config["kernelName"])
    return _default_gemm_a6w6_kernel(M, N, K)


def mxfp6_gemm_pack_size(rows: int, K: int) -> tuple[int, int]:
    """Return packed operand and scale element counts for quant_mxfp6_gemm."""
    padR, padK = _ceil(rows, _TILE), _ceil(K, _K_TILE)
    nt = padR // _TILE
    nk_pad = padK // _K_TILE + _PADK
    return (
        nt * nk_pad * _PACKED_TILE_BYTES,
        nt * nk_pad * _SCALE_TILE_BYTES,
    )


def quant_mxfp6_gemm_out(
    w: Tensor, packed: Tensor, packed_scale: Tensor
) -> tuple[Tensor, Tensor]:
    """Quantize + pack into caller-provided output buffers."""
    if w.ndim != 2:
        raise ValueError(
            f"quant_mxfp6_gemm_out expects a 2D [rows, K] tensor, got {w.ndim}D"
        )
    rows, K = w.shape
    padK = _ceil(K, _K_TILE)
    expected_packed, expected_scale = mxfp6_gemm_pack_size(rows, K)
    if packed.numel() != expected_packed or packed_scale.numel() != expected_scale:
        raise ValueError(
            "quant_mxfp6_gemm_out buffers have wrong size: "
            f"got ({packed.numel()}, {packed_scale.numel()}), "
            f"expected ({expected_packed}, {expected_scale})"
        )
    w = w.detach()
    hip_supported = (
        w.is_cuda and w.dtype in {torch.bfloat16, torch.float16} and _IS_GFX950
    )
    use_hip = hip_supported and _QUANT_BACKEND in {"auto", "hip"}
    if use_hip:
        quant_mxfp6_gemm_hip_out(w.contiguous(), packed, packed_scale)
        return packed, packed_scale
    if _QUANT_BACKEND == "hip":
        raise RuntimeError(
            "AITER_MXFP6_QUANT_BACKEND=hip requires a bf16/fp16 gfx950 CUDA/HIP tensor"
        )
    if _HAS_TRITON and w.is_cuda:
        x = w
        if padK != K:
            x = torch.nn.functional.pad(x, (0, padK - K))
        x = (
            x.contiguous()
        )  # rotation is fused inside the kernel (no fp32 [M,K] pre-pass)
        NB = padK // _SCALE_GROUP_SIZE
        NK_PAD = padK // _K_TILE + _PADK
        if _IS_GFX950 and rows * padK >= _BATCHED_PACK_MIN_ELEMENTS:
            BM = _BATCHED_PACK_BLOCK_M
            NSTEP = NB // _BATCHED_PACK_K_BLOCKS
            grid = ((rows + BM - 1) // BM * NSTEP,)
            _quant_pack_4block_kernel[grid](
                x,
                packed,
                packed_scale,
                rows,
                NSTEP,
                NK_PAD,
                x.stride(0),
                _had32_t(x.device),
                BLOCK_M=BM,
                num_warps=4,
            )
        else:
            BM = 128
            grid = ((rows + BM - 1) // BM * NB,)
            _quant_pack_kernel[grid](
                x,
                packed,
                packed_scale,
                rows,
                NB,
                NK_PAD,
                x.stride(0),
                _had32_t(x.device),
                BLOCK_M=BM,
            )
        return packed, packed_scale

    tmp_packed, tmp_scale = quant_mxfp6_gemm(w)
    packed.copy_(tmp_packed)
    packed_scale.copy_(tmp_scale)
    return packed, packed_scale


def quant_mxfp6_gemm(w: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize + pack a [rows, K] bf16/fp tensor for the a6w6 kernel.

    Rows are represented in a multiple-of-256 layout and K is zero-padded to a
    multiple of 128 so any GEMM shape maps onto the kernel's 256x256 / 128-K
    tiling. Row-padding slots and the two trailing ABI K-guard tiles are not
    consumed by the kernel and their contents are intentionally unspecified.

    Returns (packed uint8, packed_scale uint8) torch tensors on w.device.
    Works identically for both A and B operands. Runs entirely on the GPU.
    """
    if w.ndim != 2:
        raise ValueError(
            f"quant_mxfp6_gemm expects a 2D [rows, K] tensor, got {w.ndim}D"
        )
    rows, K = w.shape
    padR, padK = _ceil(rows, _TILE), _ceil(K, _K_TILE)
    w = w.detach()
    has_gpu_packer = w.is_cuda and (
        _HAS_TRITON
        or (
            _IS_GFX950
            and w.dtype in {torch.bfloat16, torch.float16}
            and _QUANT_BACKEND in {"auto", "hip"}
        )
    )
    if has_gpu_packer:
        NK_PAD = padK // _K_TILE + _PADK
        nt = padR // _TILE
        # The Triton packer writes every logical K tile.  The ASM bounds
        # accumulation with K; _PADK is addressable pipeline-guard spacing, not
        # data.  Leave guard and row-padding slots unspecified to avoid a fill
        # launch on every activation quantization in inference.
        packed = torch.empty(
            nt * NK_PAD * _PACKED_TILE_BYTES,
            dtype=torch.uint8,
            device=w.device,
        )
        packed_scale = torch.empty(
            (nt * NK_PAD * _SCALE_TILE_BYTES,),
            dtype=torch.uint8,
            device=w.device,
        )
        return quant_mxfp6_gemm_out(w, packed, packed_scale)
    # torch fallback (no triton / cpu)
    if padR != rows or padK != K:
        wp = torch.zeros((padR, padK), dtype=w.dtype, device=w.device)
        wp[:rows, :K] = w
        w = wp
    codes, scales = quant_mxfp6_torch(w)  # rotation applied inside quant_mxfp6_torch
    packed = pack_big_torch(codes)
    packed_scale = pack_scale_torch(scales, padR)
    return packed, packed_scale


# ---------------------------------------------------------------------------
# ctypes entrypoint (mirrors gemm_a4w4_asm structure)
# ---------------------------------------------------------------------------
@compile_ops(
    "module_gemm_a6w6_asm",
    fc_name="gemm_a6w6_asm",
    ffi_type="ctypes",
)
def _gemm_a6w6_asm(
    A: Tensor,  # packed mxfp6 blob
    B: Tensor,  # packed mxfp6 blob
    A_scale: Tensor,  # packed e8m0 blob
    B_scale: Tensor,  # packed e8m0 blob
    out: Tensor,  # Out:[M, N] bf16
    K: int,  # logical contraction dim
    kernelName: str | None = None,
    alpha: float = 1.0,
) -> None: ...


def gemm_a6w6_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str | None = None,
    alpha: float = 1.0,
) -> Tensor:
    if float(alpha) != 1.0:
        raise ValueError("gemm_a6w6 currently supports only alpha=1.0.")
    if not kernelName:
        if out.ndim != 2:
            raise ValueError("gemm_a6w6_asm expects a 2D [M, N] output tensor.")
        kernelName = _default_gemm_a6w6_kernel(*out.shape, K)
    _gemm_a6w6_asm(
        A,
        B,
        A_scale,
        B_scale,
        out,
        int(K),
        kernelName,
        float(alpha),
    )
    return out


def gemm_a6w6(
    A: Tensor,  # packed mxfp6 A (from quant_mxfp6_gemm)
    B: Tensor,  # packed mxfp6 B (from quant_mxfp6_gemm)
    A_scale: Tensor,  # packed A scales
    B_scale: Tensor,  # packed B scales
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = dtypes.bf16,
    alpha: float = 1.0,
    kernelName: str | None = None,
) -> Tensor:
    """A6W6 (mxfp6 E2M3, per-1x32 blockscale) GEMM: D = A * B^T.

    A/B and their scales must be pre-packed with `quant_mxfp6_gemm`. M/N/K are
    the logical (unpadded) dims. Unless ``kernelName`` explicitly overrides it,
    a shape-tuned kernel is selected before the launch is padded. The result is
    sliced back to [M, N].
    """
    if dtype != dtypes.bf16:
        raise ValueError(
            f"gemm_a6w6 currently supports only torch.bfloat16 output, got {dtype}."
        )
    if float(alpha) != 1.0:
        raise ValueError("gemm_a6w6 currently supports only alpha=1.0.")
    selected_kernel = _select_gemm_a6w6_kernel(M, N, K, kernelName)
    padM, padN, padK = _ceil(M, _TILE), _ceil(N, _TILE), _ceil(K, _K_TILE)
    out = torch.empty((padM, padN), dtype=dtype, device=A.device)
    gemm_a6w6_asm(A, B, A_scale, B_scale, out, padK, selected_kernel, alpha)
    if padM != M or padN != N:
        return out[:M, :N]
    return out
