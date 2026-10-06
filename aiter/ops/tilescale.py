# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tilescale: an MX operand layout for GEMMs that stage both operands through LDS in 256-row tiles.

GEMM convention ``C[M, N] = A[M, K] @ B[N, K]^T``. Each operand is ``(codes, scales)``; "rows" are M for role A and
N for role B. Storage rows are padded to a multiple of 256 (padded rows hold zero codes and scale byte 127), K is a
multiple of 256.

Codes
  * FP4 (E2M1) ``"row"``: plain row-major ``[R, K/2]`` bytes, low nibble = even k. One 32-value group is 16 bytes.
  * FP4 (E2M1) ``"k128"``: the same bytes K128-blocked, ``[R/16, K/128, 16, 64]``: every 16-row x 128-K block is one
    contiguous KiB. For kernels that stage K128 per step, where a row's 64 bytes are half a cache line.
  * FP6 (E2M3): a 32-value group is 24 bytes, value i at bits 6i..6i+5 little-endian. Bytes 0-15 of every group
    form plane C0, bytes 16-23 plane C1; both K128-blocked, in one buffer: C0 ``[R/16, K/128, 16, 64]`` then
    C1 ``[R/32, K/128, 32, 32]``, ``R*K*3/4`` bytes in total.

Scales: one E8M0 byte per 1x32 block, ``R*K/32`` bytes, in a slab tiled per 256 rows. The byte position of
(row, k-block) is ``ts_scale_byte``; it depends on the operand role, on the parity of K/256 and, for role B only, on
an optional 4-row interleave (``ilv``) that kernels folding the C store into the K loop use (``ts_b_ilv``).

Not part of the layout: the Hadamard rotation, the scale rule and the rounding of the quantizer. ``TilescaleMeta``
records them for tests; GEMM entry points take raw tensors and ints.
"""

from dataclasses import dataclass

import torch

TILESCALE_VERSION = 1
TILE = 256

FP4 = "fp4"
FP6 = "fp6"


@dataclass(frozen=True)
class TilescaleMeta:
    """What a tilescale operand holds beyond its bytes (informational; kernels take raw tensors)."""

    fmt: str  # FP4 / FP6
    role: str  # "a" / "b"
    ilv: int = 0  # scale interleave, role B only
    fp4_codes: str = "row"  # FP4 only: "row" / "k128"
    hadamard: str = "none"  # quantizer detail: "none" / "h16" / "h32"
    scale_rule: str = "rceil"
    rounding: str = "rn"  # "rn" / "sr"
    version: int = TILESCALE_VERSION


def _ceil(x: int, m: int) -> int:
    return -(-x // m) * m


def ts_b_ilv(K: int) -> int:
    """Scale interleave of role B for kernels that fold the C store into the K loop: 4 when K spans an even
    number (at least 4) of 256-blocks, else 0. Kernels that do not fold the store use 0."""
    kb = -(-K // 256)
    return 4 if kb >= 4 and kb % 2 == 0 else 0


def ts_code_bytes(rows: int, K: int, fmt: str) -> int:
    rp = _ceil(rows, TILE)
    return rp * K // 2 if fmt == FP4 else rp * K * 3 // 4


def ts_scale_bytes(rows: int, K: int) -> int:
    return _ceil(rows, TILE) * K // 32


def ts_scale_byte(row: torch.Tensor, kblk: torch.Tensor, *, is_b: bool, K: int, ilv: int = 0) -> torch.Tensor:
    """Byte offset in the scale slab of (row, 32-wide k-block ``kblk``); broadcasting int64 tensors."""
    if ilv not in (0, 4) or (ilv and not is_b):
        raise ValueError(f"ilv must be 0, or 4 for role B (is_b={is_b}, ilv={ilv})")
    if K % TILE:
        raise ValueError(f"K must be a multiple of {TILE}, got {K}")
    row = torch.as_tensor(row, dtype=torch.int64)
    kblk = torch.as_tensor(kblk, dtype=torch.int64)
    kk = K // 256  # K256 pairs
    ku_shift = 1 if kk % 2 == 0 else 0
    nw_shift = ku_shift + 1
    kdw, g = kblk >> 2, kblk & 3
    kh, rem = kdw >> nw_shift, kdw & ((1 << nw_shift) - 1)
    u, lo = rem >> 1, rem & 1
    grp, loc = row >> 6, row & 63
    if is_b:
        r_region = (grp & 3) >> 1
        wi = (grp >> 2) * 2 + (grp & 1)
    else:
        wi = grp >> 1
        r_region = grp & 1
    r = (loc >> 2) if ilv else (loc & 15)
    t = (loc & 3) if ilv else (loc >> 4)
    last = r_region * 2 + lo
    base = ((wi * kk + (kh << ku_shift)) * 64 + r) * 4
    return (base + u * 256 + g * 64 + last) * 4 + t


def _scale_map(rows: int, K: int, is_b: bool, ilv: int, device) -> torch.Tensor:
    rp = _ceil(rows, TILE)
    r = torch.arange(rp, device=device).view(-1, 1)
    k = torch.arange(K // 32, device=device).view(1, -1)
    return ts_scale_byte(r, k, is_b=is_b, K=K, ilv=ilv)


def pack_scales_ref(scales: torch.Tensor, *, is_b: bool, ilv: int = 0) -> torch.Tensor:
    """``[R, K/32]`` E8M0 (uint8) -> the tilescale slab (uint8, ``ts_scale_bytes(R, K)``)."""
    R, kb = scales.shape
    K = kb * 32
    rp = _ceil(R, TILE)
    full = torch.full((rp, kb), 127, dtype=torch.uint8, device=scales.device)
    full[:R] = scales.view(torch.uint8)
    out = torch.empty(rp * kb, dtype=torch.uint8, device=scales.device)
    out[_scale_map(R, K, is_b, ilv, scales.device).reshape(-1)] = full.reshape(-1)
    return out


def unpack_scales_ref(slab: torch.Tensor, rows: int, K: int, *, is_b: bool, ilv: int = 0) -> torch.Tensor:
    m = _scale_map(rows, K, is_b, ilv, slab.device)
    return slab.view(torch.uint8).reshape(-1)[m][:rows].contiguous()


def _pad_rows(codes: torch.Tensor) -> torch.Tensor:
    R = codes.shape[0]
    rp = _ceil(R, TILE)
    if rp == R:
        return codes
    return torch.cat([codes, codes.new_zeros(rp - R, *codes.shape[1:])])


def _blocked(x: torch.Tensor, rg: int, width: int) -> torch.Tensor:
    """[R, K*width/128] -> [R/rg, K/128, rg, width] flattened."""
    R, cols = x.shape
    nb = cols // width
    return x.view(R // rg, rg, nb, width).permute(0, 2, 1, 3).reshape(-1)


def _unblocked(x: torch.Tensor, R: int, rg: int, width: int, nb: int) -> torch.Tensor:
    return x.view(R // rg, nb, rg, width).permute(0, 2, 1, 3).reshape(R, nb * width)


def pack_fp4_codes_ref(codes: torch.Tensor, layout: str = "row") -> torch.Tensor:
    """``[R, K]`` E2M1 codes (uint8, 0-15) -> tilescale FP4 bytes ``[R_pad, K/2]`` (``layout`` "row" / "k128")."""
    c = _pad_rows(codes.to(torch.uint8))
    b = (c[:, 0::2] & 15) | ((c[:, 1::2] & 15) << 4)
    if layout == "row":
        return b.contiguous()
    if layout == "k128":
        return _blocked(b, 16, 64).view(b.shape)
    raise ValueError(f"unknown FP4 code layout {layout!r}")


def unpack_fp4_codes_ref(buf: torch.Tensor, rows: int, K: int, layout: str = "row") -> torch.Tensor:
    rp = _ceil(rows, TILE)
    b = buf.view(torch.uint8).reshape(rp, K // 2)
    if layout == "k128":
        b = _unblocked(b.reshape(-1), rp, 16, 64, K // 128)
    elif layout != "row":
        raise ValueError(f"unknown FP4 code layout {layout!r}")
    return torch.stack((b & 15, b >> 4), -1).reshape(rp, K)[:rows].contiguous()


def pack_fp6_codes_ref(codes: torch.Tensor) -> torch.Tensor:
    """``[R, K]`` E2M3 codes (uint8, 0-63) -> the tilescale FP6 buffer (uint8, ``R_pad*K*3/4``: C0 then C1)."""
    c = _pad_rows(codes.to(torch.int32) & 63)
    R, K = c.shape
    v = c.view(R, K // 4, 4)
    w = v[..., 0] | v[..., 1] << 6 | v[..., 2] << 12 | v[..., 3] << 18
    by = torch.stack((w & 255, (w >> 8) & 255, (w >> 16) & 255), -1).to(torch.uint8).view(R, K // 32, 24)
    c0 = by[..., :16].reshape(R, K // 2)
    c1 = by[..., 16:].reshape(R, K // 4)
    return torch.cat((_blocked(c0, 16, 64), _blocked(c1, 32, 32)))


def unpack_fp6_codes_ref(buf: torch.Tensor, rows: int, K: int) -> torch.Tensor:
    rp = _ceil(rows, TILE)
    flat = buf.view(torch.uint8).reshape(-1)
    n0 = rp * K // 2
    c0 = _unblocked(flat[:n0], rp, 16, 64, K // 128).view(rp, K // 32, 16)
    c1 = _unblocked(flat[n0 : n0 + rp * K // 4], rp, 32, 32, K // 128).view(rp, K // 32, 8)
    by = torch.cat((c0, c1), -1).to(torch.int32).view(rp, K // 4, 3)
    w = by[..., 0] | by[..., 1] << 8 | by[..., 2] << 16
    v = torch.stack([(w >> (6 * i)) & 63 for i in range(4)], -1)
    return v.reshape(rp, K)[:rows].to(torch.uint8).contiguous()


def fp6_planes(buf: torch.Tensor, rows: int, K: int):
    """Views of a tilescale FP6 buffer as the (C0, C1) planes kernels take: ``[R_pad, K/2]``, ``[R_pad, K/4]``."""
    rp = _ceil(rows, TILE)
    flat = buf.view(torch.uint8).reshape(-1)
    n0 = rp * K // 2
    return flat[:n0].view(rp, K // 2), flat[n0 : n0 + rp * K // 4].view(rp, K // 4)


# ── reference values (tests) ───────────────────────────────────────────────────

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M3 = [m / 8 for m in range(8)] + [(1 + m / 8) * 2.0**e for e in range(3) for m in range(8)]


def _grid(fmt: str, device) -> torch.Tensor:
    return torch.tensor(_E2M1 if fmt == FP4 else _E2M3, dtype=torch.float64, device=device)


def code_values(codes: torch.Tensor, fmt: str) -> torch.Tensor:
    """E2M1 / E2M3 codes -> float64 values."""
    sbit = 3 if fmt == FP4 else 5
    c = codes.to(torch.int64)
    mag = _grid(fmt, codes.device)[c & ((1 << sbit) - 1)]
    return torch.where((c >> sbit) & 1 == 1, -mag, mag)


def quant_mx_ref(x: torch.Tensor, fmt: str):
    """Reference MX quantizer along the last dim: RCEIL scales (ceil(log2(amax / max))), round to nearest (ties to
    the even code). Returns (codes ``[R, K]`` uint8, scales ``[R, K/32]`` uint8)."""
    R, K = x.shape
    grid = _grid(fmt, x.device)
    sbit = 3 if fmt == FP4 else 5
    xb = x.double().view(R, K // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.ceil(torch.log2(amax / grid[-1])), torch.full_like(amax, -127)).clamp(-127, 127)
    y = (xb / torch.exp2(e)).abs()
    hi = torch.searchsorted(grid, y.contiguous()).clamp(1, len(grid) - 1)
    lo = hi - 1
    dl, dh = y - grid[lo], grid[hi] - y
    idx = torch.where((dh < dl) | ((dh == dl) & (hi % 2 == 0)), hi, lo)
    codes = idx | ((xb < 0).to(torch.int64) << sbit)
    return codes.view(R, K).to(torch.uint8), (e.view(R, K // 32) + 127).to(torch.uint8)


def dequant_ref(codes: torch.Tensor, scales: torch.Tensor, fmt: str) -> torch.Tensor:
    """``[R, K]`` codes and ``[R, K/32]`` E8M0 scales -> float64 ``[R, K]``."""
    R, K = codes.shape
    v = code_values(codes, fmt).view(R, K // 32, 32)
    return (v * torch.exp2(scales.to(torch.float64) - 127).unsqueeze(-1)).view(R, K)
