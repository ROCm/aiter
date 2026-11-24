# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Weights of MegaMoEGfx942: the standard aiter a16w4 preshuffle -> the fd2 form the
kernels read, once per layer at load time (no rank communication).

Input, this rank's E experts of one layer:
  w1        [E, 2I, H/2]  fp4x2 / uint8, rows [gate (I); up (I)], preshuffled with
                          ``shuffle_weight_a16w4(w1, 16, False)``
  w2        [E, H, I/2]   fp4x2 / uint8, ``shuffle_weight_a16w4(w2, 16, False)``
  w*_scale  e8m0 per 1x32 K group, ``shuffle_scale_a16w4(s, E, False)``
Output, still four tensors (``prepare_weights``):
  w1, w2    the nibbles k0..k7 of each dword as bytes (k_j | k_{j+4} << 4), -0 -> +0
  w*_scale  per 32-K group min(channel max code - group code, 11) * 8, followed by the
            per-output-channel max e8m0 codes (int32 [E, 2I] / [E, H]; ``tail_exp``)
The kernels' FP8 operands are then exactly the input's MXFP4 values: a group within
2^11 of its output channel's largest scale decodes exactly, and the decode flushes a
group further below to 0, so such a group must hold only zeros (ValueError otherwise).

a8w8 (``prepare_weights_a8w8``): fp8 W with f32 128x128 block scales.
  w1        [E, 2I, H]    float8_e4m3fnuz, rows [gate (I); up (I)], in aiter's fp8 MoE
                          preshuffle ``shuffle_weight(w, layout=(16, 16))``
  w2        [E, H, I]     float8_e4m3fnuz, ``shuffle_weight(w, layout=(16, 16))``
  w*_scale  f32 [E, N/128, K/128], the dequant scale of each 128x128 block
Output: the same W, which the K loop reads as they are (so other readers of the
preshuffle can share them); a block whose scale is 0 has its W zeroed in place, and in
the returned f32 scales the largest scale of its 128-row block row (1 if that row is all
zero), so the K loop's ratios of consecutive scales stay within the checked span.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr

from ..tensor_shim import _run_compiled
from .mem import _pointer


def _scale_index(row, group, groups):
    # shuffle_scale_a16w4: [rows/32, groups/8, 4, 16, 2, 2].
    return (
        (row // 32) * (groups * 32)
        + (group // 8) * 256
        + (group % 4) * 64
        + (row % 16) * 4
        + ((group // 4) % 2) * 2
        + (row % 32) // 16
    )


@functools.cache
def _channel_exp_launcher(rows, cols, device):
    """Launcher of the channel-exponent kernel for ``rows`` x ``cols`` weights on
    ``device`` (a compiled launcher runs on one device only)."""
    groups = cols // 32
    rows_per_wave = 4 if cols >= 4096 else 16
    partitions = 64 // rows_per_wave
    if groups % partitions:
        raise ValueError("unsupported scale group geometry")
    rounds = groups // partitions
    rows_per_block = 4 * rows_per_wave

    @flyc.kernel(known_block_size=(256, 1, 1))
    def kernel(scales: fx.Int64, output: fx.Int64):
        tid = fx.Int32(gpu.thread_id("x"))
        row = (
            fx.Int32(gpu.block_id("x")) * rows_per_block
            + (tid // 64) * rows_per_wave
            + tid % rows_per_wave
        )
        part = (tid % 64) // rows_per_wave
        if row < rows:
            largest = fx.Int32(0)
            for j in range(rounds):
                group = part + j * partitions
                code = fx.ptr_load(
                    fx.add_offset(
                        _pointer(scales, fx.Uint8, 1),
                        fx.Int64(_scale_index(row, group, groups)),
                    )
                )
                largest = fx.max(largest, code.to(fx.Int32) & 255)
            for bit in range_constexpr(partitions.bit_length() - 1):
                largest = fx.max(largest, largest.shuffle_xor(rows_per_wave << bit, 64))
            if part == 0:
                fx.ptr_store(largest, fx.add_offset(_pointer(output, fx.Int32, 4), row))

    @flyc.jit
    def launch(scales: fx.Int64, output: fx.Int64, stream: fx.Stream):
        kernel(scales, output).launch(
            grid=((rows + rows_per_block - 1) // rows_per_block,),
            block=(256,),
            stream=stream,
        )

    return launch


def channel_exponents(packed, scales):
    """int32 [E, N]: per output channel, the largest e8m0 code of the preshuffled
    ``scales`` of ``packed`` [E, N, K/2]."""
    experts, n, kp = packed.shape
    result = torch.empty((experts, n), device=packed.device, dtype=torch.int32)
    launch = _channel_exp_launcher(experts * n, kp * 2, packed.device.index)
    _run_compiled(
        launch,
        scales.data_ptr(),
        result.data_ptr(),
        torch.cuda.current_stream(packed.device),
    )
    return result


def _fast_decode(w_u8):
    """Preshuffled W bytes -> the fd2 nibble order: per dword the nibbles k0..k7 as bytes
    (k_j | k_{j+4} << 4), and -0 (0x8) -> +0."""
    b = w_u8.reshape(-1, 4)
    nib = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(-1, 8)  # k0..k7
    nib = torch.where(nib == 8, torch.zeros_like(nib), nib)
    return (nib[:, :4] | (nib[:, 4:] << 4)).contiguous().view(-1)


def _predelta(w_u8, scale_u8, exp):
    """fd2 scales from the standard preshuffled W / e8m0 scale bytes: scale byte ->
    min(max_code - code, 11) * 8. The kernels decode a group with max_code - code > 11
    to 0, so its W must be all zero (ValueError otherwise). ``exp``:
    ``channel_exponents`` of the original scales.

    W bytes are (N/16, K/128, klane 4, nlane 16, 16 B): 16 B = the 32-K group
    kg = k128*4 + klane of column n = n16*16 + nlane. Scale bytes are (N/32, K/256,
    klane 4, nlane 16, kpack 2, npack 2): column n32*32 + npack*16 + nlane, group
    k256*8 + kpack*4 + klane."""
    NT = exp.numel()  # all experts' output channels
    K = w_u8.numel() * 2 // NT
    G = K // 32
    assert K % 256 == 0 and NT % 32 == 0 and scale_u8.numel() == NT * G
    s6 = scale_u8.reshape(NT // 32, K // 256, 4, 16, 2, 2)
    code = s6.permute(0, 5, 3, 1, 4, 2).reshape(NT, G).to(torch.int32)
    max_code = exp.reshape(NT, 1).to(torch.int32)
    assert torch.equal(code.amax(-1, keepdim=True), max_code), "channel max mismatch"
    diff = max_code - code
    d8 = (diff.clamp(max=11) * 8).to(torch.uint8)
    s = d8.view(NT // 32, 2, 16, K // 256, 2, 4).permute(0, 3, 5, 2, 4, 1)
    flushed = (diff > 11).view(NT // 16, 16, K // 128, 4).permute(0, 2, 3, 1)
    groups = w_u8.reshape(NT // 16, K // 128, 4, 16, 16)[flushed]
    lost = int((groups & 0x77).ne(0).any(-1).sum())  # nonzero FP4 magnitudes
    if lost:
        raise ValueError(
            f"{lost} 32-K groups hold nonzero weights more than 11 binades below their "
            "output channel's largest scale; the kernels' FP8 decode would flush them"
        )
    return s.contiguous().view(-1)


def prepare_weights(w1, w1_scale, w2, w2_scale):
    """One layer's experts in the standard preshuffle -> ``(w1, w1_scale, w2, w2_scale)``
    in fd2 form (see the module docstring). w1 [E, 2I, H/2], w2 [E, H, I/2]."""
    E, n2, hk = w1.shape
    if w2.shape[0] != E or w2.shape[1] != hk * 2 or w2.shape[2] * 2 != n2 // 2:
        raise ValueError(f"w1 {tuple(w1.shape)} and w2 {tuple(w2.shape)} do not match")
    out = []
    for w, s in ((w1, w1_scale), (w2, w2_scale)):
        # channel_exponents reads the scales through a raw pointer on W's GPU.
        if w.device.type != "cuda" or s.device != w.device:
            raise ValueError(
                f"W on {w.device}, its scales on {s.device}: expected one GPU"
            )
        w_u8 = w.contiguous().view(torch.uint8)
        s_u8 = s.contiguous().view(torch.uint8).view(-1)
        rows, k = w_u8.shape[0] * w_u8.shape[1], w_u8.shape[2] * 2
        if rows % 32 or k % 256 or s_u8.numel() != rows * k // 32:
            raise ValueError(
                f"W {tuple(w.shape)} with {s_u8.numel()} scale bytes: expected N a "
                "multiple of 32, K of 256, and one e8m0 byte per 32 K values"
            )
        exp = channel_exponents(w_u8, s_u8)  # from the original scales
        s_u8 = _predelta(w_u8, s_u8, exp)
        tail = exp.contiguous().view(-1).view(torch.uint8)
        out += [_fast_decode(w_u8).view(w_u8.shape), torch.cat([s_u8, tail])]
    return tuple(out)


def tail_exp(w, scale):
    """The int32 [E, N] per-channel exponents at the tail of the fd2 ``scale`` of ``w``."""
    experts, channels = w.shape[:2]
    tail = experts * channels * 4
    return scale.view(torch.uint8)[-tail:].view(torch.int32).view(experts, channels)


def _zero_unscaled(w, scale):
    """Check fp8 W [E, N, K] in the (16, 16) preshuffle ([N/16, K/32, 2, 16 rows, 16 B])
    and its block scales; zero (in place) the W of blocks with a zero scale."""
    E, N, K = w.shape
    if N % 128 or K % 128 or tuple(scale.shape) != (E, N // 128, K // 128):
        raise ValueError(
            f"w {tuple(w.shape)} / scale {tuple(scale.shape)}: expected N, K multiples "
            "of 128 and one scale per 128x128 block"
        )
    if not w.is_contiguous():
        raise ValueError("w must be contiguous")
    if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise ValueError("block scales must be finite and non-negative")
    # The K loop keeps acc in units of the current 128-K group's scale (x's per-128
    # scales on top), so the nonzero scales along K of one 128-row block must stay within
    # 2^60 of each other for acc to stay finite in fp32.
    pos = scale > 0
    hi = torch.where(pos, scale, torch.zeros_like(scale)).amax(dim=-1)
    lo = torch.where(pos, scale, torch.full_like(scale, float("inf"))).amin(dim=-1)
    if bool((hi > lo * 2.0**60).any()):
        raise ValueError(
            "the block scales along K of a 128-row block span more than 2^60"
        )
    zero = (scale == 0).repeat_interleave(8, dim=1)  # [E, N/16, K/128]
    if bool(zero.any()):
        # a 128-K chunk of 16 rows: 4 x 32-K groups of 2 x 16 rows x 16 B
        w.view(torch.uint8).view(E, N // 16, K // 128, 2048)[zero] = 0


def prepare_weights_a8w8(w1, w1_scale, w2, w2_scale):
    """One layer's experts in aiter's fp8 (16, 16) preshuffle -> ``(w1, w1_scale, w2,
    w2_scale)`` as the kernels read them: the same W (zero-scale blocks zeroed in
    place), f32 scales without zeros (see the module docstring). w1 [E, 2I, H], w2
    [E, H, I]."""
    E, n2, h = w1.shape
    if tuple(w2.shape) != (E, h, n2 // 2):
        raise ValueError(f"w1 {tuple(w1.shape)} and w2 {tuple(w2.shape)} do not match")
    out = []
    for w, s in ((w1, w1_scale), (w2, w2_scale)):
        if w.dtype != torch.float8_e4m3fnuz:
            raise TypeError(f"{w.dtype} weights: expected float8_e4m3fnuz")
        s = s.to(torch.float32).contiguous()
        _zero_unscaled(w, s)
        # A stand-in of 1 for a zero scale would overflow the K loop's ratio next to a
        # tiny scale; the largest scale of the block row keeps it in the checked span.
        row_max = s.amax(dim=-1, keepdim=True)
        s = torch.where(s > 0, s, torch.where(row_max > 0, row_max, 1.0))
        out += [w, s]
    return tuple(out)
