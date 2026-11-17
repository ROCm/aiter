"""MX quantization + operand packing for the FlyDSL MXFP6 MoE (torch, GPU).

E2M3 numerics are the standard OCP MX recipe (bitwise identical to a plain torch reference): E8M0 scale per 32
along K, scale = 2^ceil(log2(amax / 7.5)), round-half-even onto the E2M3 grid, clamp 7.5.
E3M2 and FP8 (E4M3, for A8W6) use the same recipe with their own max / mantissa bits.

MFMA operand contract (V_MFMA_SCALE_F32_16X16X128_F8F6F4, measured in tests/test_mfma_scale.py):
lane l: row r = l % 16, group g = l // 16.
  FP6: the lane's 6 dwords hold K[32g, 32g+32) of its row, element j at bits [6j, 6j+6).
  FP8: 8 dwords = K[16g, 16g+16) then K[64+16g, 64+16g+16).
  Scale of (row r, K-block b of one 128-step) in lane 16b + r.

Storage formats produced here (K = reduction dim, KS = K/128 steps, KQ = ceil(KS/4)):
  rows  "A" operand: [R, K * bits / 8] bytes, row-major, already in per-128 MFMA order
        (FP6: identical to plain packing; FP8: each 128-block permuted to the split-half order)
        so lane (r, g) loads 24 (FP6) / 32 (FP8) contiguous bytes at 128-step s:
        offset s * 16 * bits + g * 4 * bits.
  A scales: [R, KQ, 4] int32, byte j of dword (row, q, b) = E8M0 of block 4*(4q+j)+b.
  "B" weights (FP6, preshuffled): [E, N/BN, KS, BN/16, 64 lanes, 24 B] (each fragment = 1536 B
        contiguous); N-fragment order optionally gate/up interleaved for the SwiGLU epilogue.
  B scales: [E, N/16, KQ, 64] int32, byte j of lane (16b + r) = block 4*(4q+j)+b of row r.
"""

from __future__ import annotations

import torch

BLOCK = 32
FORMATS = {
    # name: (max, mantissa bits, exponent bias, bits per element, min normal exponent)
    "e2m3": (7.5, 3, 1, 6),
    "e3m2": (28.0, 2, 3, 6),
    "e4m3": (448.0, 3, 7, 8),
}


def quant_mx(x: torch.Tensor, fmt: str = "e2m3"):
    """x [..., K] -> (codes uint8 [..., K], scale exponents uint8 [..., K/32] (E8M0 biased 127)).

    Scale 2^ceil(log2(amax / max)), round-half-even onto the grid, clamp; done in fp32."""
    vmax, mbits, bias, _ = FORMATS[fmt]
    k = x.shape[-1]
    b = x.float().reshape(*x.shape[:-1], k // BLOCK, BLOCK)
    amax = b.abs().amax(-1, keepdim=True)
    e = torch.ceil(torch.log2((amax / vmax).clamp(min=2.0**-126)))
    e = torch.where(amax > 0, e, torch.full_like(e, -127.0))
    scale = torch.exp2(e)
    u = (b / scale).abs().clamp(max=vmax)
    min_norm_exp = 1 - bias
    exp = torch.floor(torch.log2(u.clamp(min=2.0**min_norm_exp)))
    step = torch.exp2(exp - mbits)
    r = (torch.round(u / step) * step).clamp(max=vmax)
    # encode magnitude r -> (ebits, mbits) code
    rexp = torch.floor(torch.log2(r.clamp(min=2.0**min_norm_exp)))
    sub = r < 2.0**min_norm_exp
    efield = torch.where(sub, torch.zeros_like(rexp), rexp + bias)
    mfield = torch.where(
        sub, r / 2.0 ** (min_norm_exp - mbits), (r / torch.exp2(rexp) - 1.0) * 2**mbits
    )
    mag = (efield.to(torch.int32) << mbits) | torch.round(mfield).to(torch.int32)
    nbits = {"e2m3": 6, "e3m2": 6, "e4m3": 8}[fmt]
    sign = (b < 0).to(torch.int32) << (nbits - 1)
    codes = (sign | mag).to(torch.uint8).reshape(x.shape)
    sexp = (e.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8)
    return codes, sexp


def dequant_mx(codes: torch.Tensor, sexp: torch.Tensor, fmt: str = "e2m3") -> torch.Tensor:
    vmax, mbits, bias, nbits = FORMATS[fmt]
    c = codes.to(torch.int32)
    s = (c >> (nbits - 1)) & 1
    ef = (c >> mbits) & ((1 << (nbits - 1 - mbits)) - 1)
    mf = c & ((1 << mbits) - 1)
    v = torch.where(
        ef == 0,
        mf.float() * 2.0 ** (1 - bias - mbits),
        (1.0 + mf.float() / 2**mbits) * torch.exp2(ef.float() - bias),
    )
    v = torch.where(s == 1, -v, v)
    k = codes.shape[-1]
    sc = torch.exp2(sexp.float() - 127.0).repeat_interleave(BLOCK, -1)
    return v * sc.reshape(*codes.shape[:-1], k)


def pack6(codes: torch.Tensor) -> torch.Tensor:
    """uint8 6-bit codes [..., K] -> packed bytes [..., K*3/4] (element j at bits 6j, LE)."""
    c = codes.to(torch.int32).reshape(*codes.shape[:-1], -1, 4)
    w = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    out = torch.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], -1)
    return out.to(torch.uint8).reshape(*codes.shape[:-1], codes.shape[-1] * 3 // 4)


_FP8_PERM = None


def fp8_kperm(device) -> torch.Tensor:
    """Index permutation of a 128-block into FP8 MFMA lane order (split halves)."""
    idx = []
    for g in range(4):
        idx += list(range(16 * g, 16 * g + 16)) + list(range(64 + 16 * g, 64 + 16 * g + 16))
    return torch.tensor(idx, device=device, dtype=torch.long)


def pack_rows(codes: torch.Tensor, fmt: str) -> torch.Tensor:
    """codes [R, K] -> "A" row storage [R, K*bits/8] uint8 in MFMA K order."""
    if FORMATS[fmt][3] == 6:
        return pack6(codes)
    R, K = codes.shape
    perm = fp8_kperm(codes.device)
    return codes.reshape(R, K // 128, 128)[:, :, perm].reshape(R, K).contiguous()


def kq(K: int) -> int:
    return (K // 128 + 3) // 4


def pack_a_scales(sexp: torch.Tensor) -> torch.Tensor:
    """sexp [R, K/32] uint8 -> [R, KQ, 4] int32: dword (q, b) byte j = E8M0 of block 4*(4q+j)+b, so one row's
    scales for a 4-step quad are 16 contiguous bytes (a single 16-B DMA per row and quad)."""
    R, nb = sexp.shape
    KS = nb // 4
    KQ = (KS + 3) // 4
    s = torch.full((R, KQ * 4, 4), 127, dtype=torch.uint8, device=sexp.device)
    s[:, :KS, :] = sexp.reshape(R, KS, 4)  # [R, step, b]
    s = s.reshape(R, KQ, 4, 4).permute(0, 1, 3, 2).contiguous()  # [R, q, b, j]
    return s.view(torch.int32).reshape(R, KQ, 4)


def unpack_a_scales(packed: torch.Tensor, KS: int) -> torch.Tensor:
    """Inverse of pack_a_scales for the KS valid steps: [R, KQ, 4] int32 -> [R, KS, 4 b] uint8."""
    R, KQ = packed.shape[0], packed.shape[1]
    s = packed.contiguous().view(torch.uint8).reshape(R, KQ, 4, 4).permute(0, 1, 3, 2)  # [R, q, j, b]
    return s.reshape(R, KQ * 4, 4)[:, :KS, :]


def preshuffle_b(codes6: torch.Tensor, sexp: torch.Tensor, interleave_gate_up: bool = False, bn: int | None = None):
    """Weights [E, N, K] (6-bit codes) + scales [E, N, K/32] -> (frags uint8, scales int32).

    interleave_gate_up: N = 2*I with gate rows first; reorder 16-row fragments as
    gate0, up0, gate1, up1, ... so a wave's fragment pair (2p, 2p+1) is (gate, up)."""
    E, N, K = codes6.shape
    KS, KQ = K // 128, kq(K)
    if interleave_gate_up:
        I = N // 2
        g = codes6[:, :I].reshape(E, I // 16, 16, K)
        u = codes6[:, I:].reshape(E, I // 16, 16, K)
        codes6 = torch.stack([g, u], 2).reshape(E, N, K)
        gs = sexp[:, :I].reshape(E, I // 16, 16, -1)
        us = sexp[:, I:].reshape(E, I // 16, 16, -1)
        sexp = torch.stack([gs, us], 2).reshape(E, N, -1)
    p = pack6(codes6)  # [E, N, K*3/4]
    p = p.reshape(E, N // 16, 16, KS, 4, 24).permute(0, 1, 3, 4, 2, 5).contiguous()
    frags = p.reshape(E, N // 16, KS, 64 * 24)
    if bn is not None:
        # N-block-major [E, N/bn, KS, bn/16, 1536]: a block's fragments for one K step are contiguous
        frags = frags.reshape(E, N // bn, bn // 16, KS, 64 * 24).permute(0, 1, 3, 2, 4).contiguous()
    s = torch.full((E, N, KQ * 4, 4), 127, dtype=torch.uint8, device=codes6.device)
    s[:, :, :KS, :] = sexp.reshape(E, N, KS, 4)  # [E, n, step, b]
    s = s.reshape(E, N // 16, 16, KQ, 4, 4)  # [E, nf, r, q, j, b]
    s = s.permute(0, 1, 3, 5, 2, 4).contiguous()  # [E, nf, q, b, r, j] -> lane = 16b + r
    scales = s.view(torch.int32).reshape(E, N // 16, KQ, 64)
    return frags, scales
