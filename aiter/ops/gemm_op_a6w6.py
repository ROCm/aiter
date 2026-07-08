# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility import dtypes

# The mxfp6 (E2M3, per-1x32 blockscale) asm gemm shares the a4w4 kernarg ABI.
# Its packed operand/scale layouts are produced by the helpers below and must
# match exactly what the `f6gemm_dmabig_kernel_func` kernel consumes.
_KERNEL_NAME = "f6gemm_dmabig_kernel_func"
_TILE = 256
_PADK = 2  # K-padding steps (of 128) baked into the packed A/B layout


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
# fused Triton quantize+pack (bf16 -> mxfp6 codes + e8m0 scales, packed into the
# kernel's C0/C1 tile layout in ONE GPU pass). ~27x faster than the torch path,
# making per-call activation quantization cheap enough for inference.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _e2m3_dev(scaled):
        a = tl.minimum(tl.abs(scaled), 7.5)
        isn = a >= 1.0
        ex = tl.minimum(tl.floor(tl.log2(tl.maximum(a, 1.0))), 2.0)
        base = tl.exp2(ex)
        step = base / 8.0
        mn = tl.floor((a - base) / step + 0.5)
        ms = tl.floor(a * 8.0 + 0.5)
        mag_code = tl.where(isn, (ex + 1.0) * 8.0 + mn, ms)
        return mag_code.to(tl.int32) + (scaled < 0.0).to(tl.int32) * 32

    @triton.jit
    def _quant_pack_kernel(x_ptr, a_ptr, s_ptr, M, NB, NK_PAD, stride_xm, BLOCK_M: tl.constexpr):
        pid = tl.program_id(0)
        jb = pid % NB
        rblk = pid // NB
        rows = rblk * BLOCK_M + tl.arange(0, BLOCK_M)
        rm = rows < M
        r = rows[:, None]
        g8 = tl.arange(0, 8)[None, :]
        # coalesced contiguous [BM,32] load, then quantize all 32 and split into 4 phases
        xall = tl.load(x_ptr + r * stride_xm + jb * 32 + tl.arange(0, 32)[None, :],
                       mask=rm[:, None], other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(xall), 1)
        safe = tl.maximum(amax, 1e-30)
        se = tl.minimum(tl.maximum(tl.floor(tl.log2(safe)) - 2.0, -127.0), 127.0)
        se = tl.where(amax > 0.0, se, 0.0)
        e8 = (se + 127.0).to(tl.uint8)
        codes = _e2m3_dev(xall * tl.exp2(-se)[:, None])   # [BM,32]
        cc = tl.reshape(codes, [BLOCK_M, 8, 2, 2])
        lo, hi = tl.split(cc)        # lo=b0-bit {s0,s2}, hi {s1,s3}
        c0, c2 = tl.split(lo)        # phase 0, 2
        c1, c3 = tl.split(hi)        # phase 1, 3
        b0 = (c0 | (c1 << 6)) & 0xFF
        b1 = ((c1 >> 2) | (c2 << 4)) & 0xFF
        b2 = ((c2 >> 4) | (c3 << 2)) & 0xFF
        t = rows // 256; rem = rows % 256; rb = rem // 16; r16 = rem % 16
        step = jb // 4; kg = jb % 4
        blk = rb * 64 + (kg * 16 + r16)
        base = (t * NK_PAD + step) * 24576
        c0base = (base + blk * 16)[:, None]
        c1base = (base + 16384 + blk * 8)[:, None]
        p0 = 3 * g8 + 0; p1 = 3 * g8 + 1; p2 = 3 * g8 + 2
        tl.store(a_ptr + tl.where(p0 < 16, c0base + p0, c1base + (p0 - 16)), b0.to(tl.uint8), mask=rm[:, None])
        tl.store(a_ptr + tl.where(p1 < 16, c0base + p1, c1base + (p1 - 16)), b1.to(tl.uint8), mask=rm[:, None])
        tl.store(a_ptr + tl.where(p2 < 16, c0base + p2, c1base + (p2 - 16)), b2.to(tl.uint8), mask=rm[:, None])
        su = rem // 128; sub = (rem % 128) // 16
        scaddr = (t * NK_PAD + step) * 1024 + su * 512 + kg * 128 + r16 * 8 + sub
        tl.store(s_ptr + scaddr, e8, mask=rm)

def quant_mxfp6(x: np.ndarray):
    """Quantize a [R, K] float array to mxfp6 (E2M3) codes + e8m0 block scales.

    Returns (codes[R, K] uint8 6-bit, scales[R, K//32] uint8 e8m0).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    R, K = x.shape
    assert K % 32 == 0, "K must be a multiple of 32"
    NB = K // 32
    blk = x.reshape(R, NB, 32)
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
    assert R % 256 == 0 and K % 128 == 0, "R%256 and K%128 required"
    nt, nk = R // 256, K // 128
    rb = np.repeat(np.arange(16), 64)
    L = np.tile(np.arange(64), 16)
    r16 = L % 16
    kg = L // 16
    local_row = rb * 16 + r16  # (1024,)

    t_ax = np.arange(nt)[:, None, None]
    s_ax = np.arange(nk)[None, :, None]
    row = t_ax * 256 + local_row[None, None, :]  # (nt,1,1024)
    col = s_ax * 128 + (kg * 32)[None, None, :]  # (1,nk,1024)
    row_b, col_b = np.broadcast_arrays(row, col)  # (nt,nk,1024)
    blocks = codes[row_b[..., None], col_b[..., None] + np.arange(32)]  # (nt,nk,1024,32)

    packed = _pack32(blocks.reshape(-1, 32)).reshape(nt, nk, 1024, 24)
    out = np.zeros((nt, nk + padK, 24576), np.uint8)
    out[:, :nk, :16384] = packed[..., :16].reshape(nt, nk, 1024 * 16)
    out[:, :nk, 16384:] = packed[..., 16:].reshape(nt, nk, 1024 * 8)
    return np.ascontiguousarray(out.reshape(-1))


def pack_scale(S: np.ndarray, rows: int, padK: int = _PADK) -> np.ndarray:
    """Re-tile [R, K//32] e8m0 scales into the kernel's packed scale blob."""
    R, NB = S.shape
    assert rows % 256 == 0 and NB % 4 == 0, "rows%256 and NB%4 required"
    nt, nk = rows // 256, NB // 4
    off = np.arange(1024)
    su = off // 512
    kg = (off % 512) // 128
    r16 = (off % 128) // 8
    sub = off % 8
    row_local = su * 128 + sub * 16 + r16  # (1024,)

    t_ax = np.arange(nt)[:, None, None]
    s_ax = np.arange(nk)[None, :, None]
    row = t_ax * 256 + row_local[None, None, :]  # (nt,1,1024)
    block = s_ax * 4 + kg[None, None, :]  # (1,nk,1024)
    row_b, block_b = np.broadcast_arrays(row, block)
    vals = S[row_b, block_b]  # (nt,nk,1024)

    out = np.full((nt, nk + padK, 1024), 127, np.uint8)
    out[:, :nk, :] = vals
    return np.ascontiguousarray(out.reshape(-1))


# ---------------------------------------------------------------------------
# GPU-native (torch) quantization + packing -- byte-identical to the numpy
# helpers above, but keeps everything on device so packing a GEMM operand
# costs microseconds instead of CPU-bound seconds.
# ---------------------------------------------------------------------------
_E2M3_T = None
_POS_T = None


def _tables(device):
    global _E2M3_T, _POS_T
    if _E2M3_T is None or _E2M3_T.device != device:
        _E2M3_T = torch.from_numpy(E2M3).to(device)
        _POS_T = torch.from_numpy(_POS).to(device)
    return _E2M3_T, _POS_T


def quant_mxfp6_torch(x: Tensor):
    """torch/GPU version of quant_mxfp6: [R,K] float -> (codes uint8, scales uint8)."""
    x = x.float()
    R, K = x.shape
    NB = K // 32
    blk = x.reshape(R, NB, 32)
    amax = blk.abs().amax(dim=2)
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    exp = torch.floor(torch.log2(safe))
    scale_exp = torch.clamp(exp - _E2M3_MAX_EXP, -127, 127)
    scale_exp = torch.where(amax > 0, scale_exp, torch.zeros_like(scale_exp))
    scales = (scale_exp + 127).to(torch.uint8)

    scaled = blk / torch.pow(torch.tensor(2.0, device=x.device), scale_exp).unsqueeze(-1)
    # arithmetic E2M3 round-to-nearest (identical to the fused Triton _e2m3_dev)
    a = scaled.abs().clamp(max=7.5)
    isn = a >= 1.0
    ex = torch.floor(torch.log2(a.clamp(min=1.0))).clamp(max=2.0)
    base = torch.pow(torch.tensor(2.0, device=x.device), ex)
    step = base / 8.0
    mn = torch.floor((a - base) / step + 0.5)
    ms = torch.floor(a * 8.0 + 0.5)
    mag = torch.where(isn, (ex + 1.0) * 8.0 + mn, ms)
    codes = (mag.to(torch.long) + (scaled < 0).to(torch.long) * 32).to(torch.uint8).reshape(R, K)
    return codes, scales


def dequant_mxfp6_torch(codes: Tensor, scales: Tensor) -> Tensor:
    """Reconstruct fp32 values from mxfp6 codes + e8m0 scales (for reference)."""
    tab, _ = _tables(codes.device)
    v = tab[codes.long()]
    sf = torch.pow(torch.tensor(2.0, device=codes.device), scales.float() - 127)
    return v * sf.repeat_interleave(32, dim=1)


def _pack32_torch(blocks: Tensor) -> Tensor:
    g = blocks.reshape(-1, 8, 4).to(torch.int32)
    b0 = (g[..., 0] | (g[..., 1] << 6)) & 0xFF
    b1 = ((g[..., 1] >> 2) | (g[..., 2] << 4)) & 0xFF
    b2 = ((g[..., 2] >> 4) | (g[..., 3] << 2)) & 0xFF
    out = torch.stack([b0, b1, b2], dim=-1).to(torch.uint8)
    return out.reshape(blocks.shape[0], 24)


def pack_big_torch_128(codes: Tensor, padK: int = _PADK) -> Tensor:
    """128-N-row fp6 B tiles (12288 B/slab: C0 8192 + C1 4096) for the 256x128 db kernel."""
    dev = codes.device
    R, K = codes.shape
    nt, nk = R // 128, K // 128
    rb = torch.arange(8, device=dev).repeat_interleave(64)
    L = torch.arange(64, device=dev).repeat(8)
    r16 = L % 16
    kg = L // 16
    local_row = rb * 16 + r16
    t_ax = torch.arange(nt, device=dev).view(nt, 1, 1)
    s_ax = torch.arange(nk, device=dev).view(1, nk, 1)
    row = t_ax * 128 + local_row.view(1, 1, 512)
    col = s_ax * 128 + (kg * 32).view(1, 1, 512)
    row_b, col_b = torch.broadcast_tensors(row, col)
    ar = torch.arange(32, device=dev)
    blocks = codes[row_b.unsqueeze(-1), col_b.unsqueeze(-1) + ar]
    packed = _pack32_torch(blocks.reshape(-1, 32)).reshape(nt, nk, 512, 24)
    out = torch.zeros((nt, nk + padK, 12288), dtype=torch.uint8, device=dev)
    out[:, :nk, :8192] = packed[..., :16].reshape(nt, nk, 512 * 16)
    out[:, :nk, 8192:] = packed[..., 16:].reshape(nt, nk, 512 * 8)
    return out.reshape(-1).contiguous()


def pack_scale_torch_128(S: Tensor, rows: int, padK: int = _PADK) -> Tensor:
    """128-N-row e8m0 scale tiles (512 B/slab) for the 256x128 db kernel."""
    dev = S.device
    R, NB = S.shape
    nt, nk = rows // 128, NB // 4
    off = torch.arange(512, device=dev)
    kg = off // 128
    r16 = (off % 128) // 8
    sub = off % 8
    row_local = sub * 16 + r16
    t_ax = torch.arange(nt, device=dev).view(nt, 1, 1)
    s_ax = torch.arange(nk, device=dev).view(1, nk, 1)
    row = t_ax * 128 + row_local.view(1, 1, 512)
    block = s_ax * 4 + kg.view(1, 1, 512)
    row_b, block_b = torch.broadcast_tensors(row, block)
    out = torch.full((nt, nk + padK, 512), 127, dtype=torch.uint8, device=dev)
    out[:, :nk, :] = S[row_b, block_b]
    return out.reshape(-1).contiguous()


def quant_mxfp6_gemm_128(w: Tensor):
    """Pack a [N,K] weight to the 128-N-row fp6 B layout (for the a6w6 256x128 db kernel).
    Trailing pad tiles cover the db's 3-ahead B/scaleB prefetch."""
    rows, K = w.shape
    padR, padK = _ceil(rows, 128), _ceil(K, 128)
    w = w.detach()
    if padR != rows or padK != K:
        wp = torch.zeros((padR, padK), dtype=w.dtype, device=w.device)
        wp[:rows, :K] = w
        w = wp
    codes, scales = quant_mxfp6_torch(w)
    packedB = pack_big_torch_128(codes)
    scaleB = pack_scale_torch_128(scales, padR)
    padB = torch.zeros(2 * 12288, dtype=torch.uint8, device=packedB.device)
    padS = torch.full((2 * 512,), 127, dtype=torch.uint8, device=scaleB.device)
    return torch.cat([packedB, padB]), torch.cat([scaleB, padS])


def pack_big_torch(codes: Tensor, padK: int = _PADK) -> Tensor:
    dev = codes.device
    R, K = codes.shape
    nt, nk = R // 256, K // 128
    rb = torch.arange(16, device=dev).repeat_interleave(64)
    L = torch.arange(64, device=dev).repeat(16)
    r16 = L % 16
    kg = L // 16
    local_row = rb * 16 + r16
    t_ax = torch.arange(nt, device=dev).view(nt, 1, 1)
    s_ax = torch.arange(nk, device=dev).view(1, nk, 1)
    row = t_ax * 256 + local_row.view(1, 1, 1024)
    col = s_ax * 128 + (kg * 32).view(1, 1, 1024)
    row_b, col_b = torch.broadcast_tensors(row, col)
    ar = torch.arange(32, device=dev)
    blocks = codes[row_b.unsqueeze(-1), col_b.unsqueeze(-1) + ar]  # (nt,nk,1024,32)
    packed = _pack32_torch(blocks.reshape(-1, 32)).reshape(nt, nk, 1024, 24)
    out = torch.zeros((nt, nk + padK, 24576), dtype=torch.uint8, device=dev)
    out[:, :nk, :16384] = packed[..., :16].reshape(nt, nk, 1024 * 16)
    out[:, :nk, 16384:] = packed[..., 16:].reshape(nt, nk, 1024 * 8)
    return out.reshape(-1).contiguous()


def pack_scale_torch(S: Tensor, rows: int, padK: int = _PADK) -> Tensor:
    dev = S.device
    R, NB = S.shape
    nt, nk = rows // 256, NB // 4
    off = torch.arange(1024, device=dev)
    su = off // 512
    kg = (off % 512) // 128
    r16 = (off % 128) // 8
    sub = off % 8
    row_local = su * 128 + sub * 16 + r16
    t_ax = torch.arange(nt, device=dev).view(nt, 1, 1)
    s_ax = torch.arange(nk, device=dev).view(1, nk, 1)
    row = t_ax * 256 + row_local.view(1, 1, 1024)
    block = s_ax * 4 + kg.view(1, 1, 1024)
    row_b, block_b = torch.broadcast_tensors(row, block)
    vals = S[row_b, block_b]
    out = torch.full((nt, nk + padK, 1024), 127, dtype=torch.uint8, device=dev)
    out[:, :nk, :] = vals
    return out.reshape(-1).contiguous()


def _ceil(x, m):
    return (x + m - 1) // m * m


def mxfp6_gemm_pack_size(rows: int, K: int) -> tuple[int, int]:
    """Return packed operand and scale element counts for quant_mxfp6_gemm."""
    padR, padK = _ceil(rows, 256), _ceil(K, 128)
    nt = padR // 256
    nk_pad = padK // 128 + _PADK
    return nt * nk_pad * 24576, nt * nk_pad * 1024


def quant_mxfp6_gemm_out(w: Tensor, packed: Tensor, packed_scale: Tensor):
    """Quantize + pack into caller-provided output buffers."""
    rows, K = w.shape
    padR, padK = _ceil(rows, 256), _ceil(K, 128)
    expected_packed, expected_scale = mxfp6_gemm_pack_size(rows, K)
    if packed.numel() != expected_packed or packed_scale.numel() != expected_scale:
        raise ValueError(
            "quant_mxfp6_gemm_out buffers have wrong size: "
            f"got ({packed.numel()}, {packed_scale.numel()}), "
            f"expected ({expected_packed}, {expected_scale})"
        )
    w = w.detach()
    if _HAS_TRITON and w.is_cuda:
        x = w
        if padK != K:
            x = torch.nn.functional.pad(x, (0, padK - K))
        x = x.contiguous()
        NB = padK // 32
        NK_PAD = padK // 128 + _PADK
        BM = 128
        grid = ((rows + BM - 1) // BM * NB,)
        _quant_pack_kernel[grid](
            x, packed, packed_scale, rows, NB, NK_PAD, x.stride(0), BLOCK_M=BM
        )
        return packed, packed_scale

    tmp_packed, tmp_scale = quant_mxfp6_gemm(w)
    packed.copy_(tmp_packed)
    packed_scale.copy_(tmp_scale)
    return packed, packed_scale


def quant_mxfp6_gemm(w: Tensor):
    """Quantize + pack a [rows, K] bf16/fp tensor for the a6w6 kernel.

    Rows are zero-padded to a multiple of 256 and K to a multiple of 128 so
    that any GEMM shape maps onto the kernel's 256x256 / 128-K tiling. The
    padded region quantizes to zero and does not affect the result.

    Returns (packed uint8, packed_scale uint8) torch tensors on w.device.
    Works identically for both A and B operands. Runs entirely on the GPU.
    """
    rows, K = w.shape
    padR, padK = _ceil(rows, 256), _ceil(K, 128)
    w = w.detach()
    if _HAS_TRITON and w.is_cuda:
        x = w
        if padK != K:
            x = torch.nn.functional.pad(x, (0, padK - K))
        x = x.contiguous()
        NB = padK // 32
        NK_PAD = padK // 128 + _PADK
        nt = padR // 256
        # The Triton packer writes every tile that the a6w6 GEMM consumes for
        # the logical K extent.  Avoid full-buffer memset/fill here; those extra
        # launches are paid on every activation quantization in inference.
        packed = torch.empty(nt * NK_PAD * 24576, dtype=torch.uint8, device=w.device)
        packed_scale = torch.empty((nt * NK_PAD * 1024,), dtype=torch.uint8, device=w.device)
        return quant_mxfp6_gemm_out(w, packed, packed_scale)
    # torch fallback (no triton / cpu)
    if padR != rows or padK != K:
        wp = torch.zeros((padR, padK), dtype=w.dtype, device=w.device)
        wp[:rows, :K] = w
        w = wp
    codes, scales = quant_mxfp6_torch(w)
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
    kernelName: Optional[str] = None,
    alpha: float = 1.0,
) -> None: ...


def gemm_a6w6_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str = _KERNEL_NAME,
    alpha: float = 1.0,
) -> Tensor:
    if float(alpha) != 1.0:
        raise ValueError("gemm_a6w6 currently supports only alpha=1.0.")
    _gemm_a6w6_asm(
        A,
        B,
        A_scale,
        B_scale,
        out,
        int(K),
        kernelName if kernelName else None,
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
    kernelName: str = _KERNEL_NAME,
) -> Tensor:
    """A6W6 (mxfp6 E2M3, per-1x32 blockscale) GEMM: D = A * B^T.

    A/B and their scales must be pre-packed with `quant_mxfp6_gemm`. M/N/K are
    the logical (unpadded) dims; the kernel operates on the tile-padded extent
    and the result is sliced back to [M, N].
    """
    if float(alpha) != 1.0:
        raise ValueError("gemm_a6w6 currently supports only alpha=1.0.")
    padM, padN, padK = _ceil(M, 256), _ceil(N, 256), _ceil(K, 128)
    out = torch.empty((padM, padN), dtype=dtype, device=A.device)
    gemm_a6w6_asm(A, B, A_scale, B_scale, out, padK, kernelName, alpha)
    if padM != M or padN != N:
        return out[:M, :N]
    return out


_A_DB_PAD_BYTES = 2 * 24576  # trailing pad tiles for the db kernel's 3-ahead A prefetch


def quant_mxfp6_gemm_act_db(x: Tensor):
    """Pack a fp6 activation for the 256x128 db kernel (f6gemm_db_dmabig_kernel_func):
    identical to `quant_mxfp6_gemm` plus trailing zero tiles so the kernel's 3-ahead
    A prefetch cannot fault past the last M-tile."""
    packed, scale = quant_mxfp6_gemm(x)
    pad = torch.zeros(_A_DB_PAD_BYTES, dtype=torch.uint8, device=packed.device)
    return torch.cat([packed, pad]), scale


