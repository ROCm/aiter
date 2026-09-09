# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host reference for the QRInt4 wire codecs and the two all-reduce schedules.

A bit-faithful torch model of what the kernels do, used two ways:

* as an oracle for the packed words, so a kernel round-trip can be checked
  *bit for bit* rather than only through its reconstruction -- a half-swap in
  the INT6 ``hi2`` lane pairing reconstructs plausibly and would survive a
  reconstruction-only check;
* as a no-GPU predictor of SQNR, so the codec and the schedule can be sized
  before any FlyDSL is written.

Everything here mirrors ``qr_int4_kernel``: group-16 signed E4M3 scales that
encode the group *extremum* (not the step), a codec whose extremum maps to the
most negative code, and packed-fp16 arithmetic in the accumulate path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# One 16 B atom is 8 values; two threads (16 values) share one E4M3 byte.
GROUP = 16


@dataclass(frozen=True)
class CodecRef:
    name: str
    bits: int
    bias: int
    qmin: float
    qmax: float

    @property
    def step(self) -> float:
        """Decoding factor applied to the E4M3 extremum: -1/8 or -1/32."""
        return -1.0 / float(self.bias)


INT4_REF = CodecRef(name="int4", bits=4, bias=8, qmin=-8.0, qmax=7.0)
INT6_REF = CodecRef(name="int6", bits=6, bias=32, qmin=-32.0, qmax=31.0)
CODECS_REF = {c.name: c for c in (INT4_REF, INT6_REF)}


def f32_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """``_f32_to_e4m3``: 1 sign + 4 exp (bias 7, e=0 still implicit 1) + 3 mant.

    Not OCP E4M3 -- e=0 encodes ``(1+m/8)*2^-7`` rather than a denormal, which
    is what keeps typical extrema in range. Byte 0 is +0.
    """
    x = x.to(torch.float32)
    is_z = x == 0.0
    sign = torch.where(x < 0.0, 0x80, 0).to(torch.int32)
    bits = x.abs().view(torch.int32)
    e = ((bits >> 23) & 255) - 127
    mant = bits & 0x7FFFFF
    m3 = (mant + (1 << 19)) >> 20
    carry = m3 == 8
    e = e + carry.to(torch.int32)
    m3 = torch.where(carry, torch.zeros_like(m3), m3)
    e4 = (e + 7).clamp(0, 15)
    byte = sign | (e4 << 3) | (m3 & 7)
    return torch.where(is_z, torch.zeros_like(byte), byte)


def e4m3_to_f32(b: torch.Tensor) -> torch.Tensor:
    is_z = b == 0
    sign = (b & 0x80) != 0
    e4 = (b >> 3) & 15
    m3 = b & 7
    mag = (((e4 + 120) << 23) | (m3 << 20)).view(torch.float32)
    signed = torch.where(sign, -mag, mag)
    return torch.where(is_z, torch.zeros_like(signed), signed)


def _signed_extremum(g: torch.Tensor) -> torch.Tensor:
    """Largest-magnitude value of each group, keeping its sign (``_pair_signed_ext_f16``)."""
    mx = g.max(dim=-1).values
    mn = g.min(dim=-1).values
    return torch.where(mx.abs() > mn.abs(), mx, mn)


def quantize(x: torch.Tensor, codec: CodecRef) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-16 quantize *x* (``[..., n]``, n % GROUP == 0).

    Returns ``(q, e)`` -- unsigned codes in ``[0, 2**bits)`` shaped like *x*,
    and the E4M3 scale byte per group.

    The kernel works in fp16 from the group extremum onward, so this does too:
    the ``x * (1/d)`` product and its rounding are the codec's, not f32's.
    """
    g = x.reshape(*x.shape[:-1], -1, GROUP)
    e = f32_to_e4m3(_signed_extremum(g.to(torch.float32)))
    d = e4m3_to_f32(e) * codec.step
    # Match the kernel's guard, and its fp16 ceiling: 1/d is materialised as
    # fp16 there, so a huge reciprocal saturates rather than becoming Inf.
    enc = (1.0 / (d + 1e-7)).clamp(-65504.0, 65504.0)
    enc16 = enc.to(torch.float16).unsqueeze(-1)
    w = (g.to(torch.float16) * enc16).clamp(codec.qmin, codec.qmax)
    q = torch.round(w).to(torch.int32) + codec.bias
    return q.reshape(x.shape), e


def dequantize(q: torch.Tensor, e: torch.Tensor, codec: CodecRef) -> torch.Tensor:
    """Inverse of :func:`quantize`, in fp16 as the kernel's dequant path is."""
    d = (e4m3_to_f32(e) * codec.step).to(torch.float16)
    qg = q.reshape(*q.shape[:-1], -1, GROUP)
    deq = (qg - codec.bias).to(torch.float16) * d.unsqueeze(-1)
    return deq.reshape(q.shape).to(torch.float16)


def roundtrip(x: torch.Tensor, codec: CodecRef) -> torch.Tensor:
    q, e = quantize(x, codec)
    return dequantize(q, e, codec)


# --- wire packing ---------------------------------------------------------
#
# Thread t of a 256-thread block owns values [8t, 8t+8). Value i of a thread
# lands in nibble i of that thread's i32; the fp16x2 pairing means element
# ``2j`` goes to bits ``[4j..4j+3]`` and element ``2j+1`` to ``[16+4j..]``.


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """``[..., 256, 8]`` codes -> ``[..., 256]`` i32, one per thread."""
    lo, hi = q[..., 0::2], q[..., 1::2]
    out = torch.zeros(q.shape[:-1], dtype=torch.int32, device=q.device)
    for j in range(4):
        out |= (lo[..., j] & 0xF) << (4 * j)
        out |= (hi[..., j] & 0xF) << (16 + 4 * j)
    return out


def pack_int6(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., 256, 8]`` codes -> (``lo4`` ``[..., 256]``, ``hi2`` ``[..., 128]``).

    ``lo4`` is byte-identical in layout to :func:`pack_int4`'s output. ``hi2``
    holds 16 dense bits per thread, and threads ``2t`` / ``2t+1`` share i32
    ``t`` -- low half from the even thread.
    """
    lo, hi = q[..., 0::2], q[..., 1::2]
    lo4 = torch.zeros(q.shape[:-1], dtype=torch.int32, device=q.device)
    raw = torch.zeros(q.shape[:-1], dtype=torch.int32, device=q.device)
    for j in range(4):
        lo4 |= (lo[..., j] & 0xF) << (4 * j)
        lo4 |= (hi[..., j] & 0xF) << (16 + 4 * j)
        raw |= ((lo[..., j] >> 4) & 0x3) << (2 * j)
        raw |= ((hi[..., j] >> 4) & 0x3) << (16 + 2 * j)
    # Compact the two 8-bit halves into 16 dense bits, then merge the lane pair.
    c = (raw & 0xFF) | ((raw >> 8) & 0xFF00)
    hi2 = (c[..., 0::2] & 0xFFFF) | ((c[..., 1::2] & 0xFFFF) << 16)
    return lo4, hi2


def unpack_int4(lo4: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((*lo4.shape, 8), dtype=torch.int32, device=lo4.device)
    for j in range(4):
        out[..., 2 * j] = (lo4 >> (4 * j)) & 0xF
        out[..., 2 * j + 1] = (lo4 >> (16 + 4 * j)) & 0xF
    return out


def unpack_int6(lo4: torch.Tensor, hi2: torch.Tensor) -> torch.Tensor:
    c = torch.stack([hi2 & 0xFFFF, (hi2 >> 16) & 0xFFFF], dim=-1).reshape(lo4.shape)
    raw = (c & 0xFF) | ((c & 0xFF00) << 8)
    out = torch.zeros((*lo4.shape, 8), dtype=torch.int32, device=lo4.device)
    for j in range(4):
        out[..., 2 * j] = ((lo4 >> (4 * j)) & 0xF) | (((raw >> (2 * j)) & 0x3) << 4)
        out[..., 2 * j + 1] = ((lo4 >> (16 + 4 * j)) & 0xF) | (
            ((raw >> (16 + 2 * j)) & 0x3) << 4
        )
    return out


# --- schedules ------------------------------------------------------------


def allreduce_mesh(xs: list[torch.Tensor], codec: CodecRef) -> torch.Tensor:
    """Reduce-scatter to the owner, then all-gather. Two quantizations deep.

    Each rank quantizes its own contribution once; the owner sums the decoded
    contributions in fp16 and quantizes the sum once for the gather.
    """
    acc = torch.zeros_like(xs[0], dtype=torch.float16)
    for x in xs:
        acc = acc + roundtrip(x, codec)
    return roundtrip(acc, codec).to(torch.float32)


def allreduce_ring(
    xs: list[torch.Tensor], rs_codec: CodecRef, ag_codec: CodecRef
) -> torch.Tensor:
    """``2(N-1)`` hops. The RS lap requantizes the running partial each hop.

    Modelled per chunk, for one chunk, which is all the error depends on: rank
    ``r`` starts the chain for chunk ``r`` and every other rank adds into it in
    ring order. The all-gather lap forwards bytes verbatim, so it contributes
    exactly one quantization -- the one at the last reduce.
    """
    n = len(xs)
    owner = 0
    # Op 1: the chain's first sender quantizes its own contribution.
    src = (owner + 1) % n
    acc = roundtrip(xs[src], rs_codec)
    # Ops 2..N-1: dequantize the partial, add ours, requantize.
    for hop in range(2, n):
        src = (owner + hop) % n
        acc = roundtrip(acc + xs[src].to(torch.float16), rs_codec)
    # Op N: last reduce. The owner's own copy of this chunk is exact; every
    # other rank sees it through the single all-gather quantization.
    final = acc + xs[owner].to(torch.float16)
    return roundtrip(final, ag_codec).to(torch.float32)


def sqnr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    mse = float(((got.to(torch.float64) - ref.to(torch.float64)) ** 2).mean())
    pwr = float((ref.to(torch.float64) ** 2).mean())
    if mse <= 0.0:
        return float("inf")
    if pwr <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(pwr / mse)
