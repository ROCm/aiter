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
  w1, w2    the nibbles k0..k7 of each dword as bytes (k_j | k_{j+4} << 4), -0 -> +0;
            the W of groups that decode to 0 (scale code <= channel max - 12) zeroed
  w*_scale  per 32-K group min(channel max code - group code, 11) * 8, followed by the
            per-output-channel max e8m0 codes (int32 [E, 2I] / [E, H]; ``tail_exp``)
Decoding the output gives the same bits as decoding the input.

a8w8 (``prepare_weights_a8w8``): fp8 W with f32 128x128 block scales.
  w1        [E, 2I, H]    float8_e4m3fnuz, rows [gate (I); up (I)], in aiter's fp8 MoE
                          preshuffle ``shuffle_weight(w, layout=(16, 16))``
  w2        [E, H, I]     float8_e4m3fnuz, ``shuffle_weight(w, layout=(16, 16))``
  w*_scale  f32 [E, N/128, K/128], the dequant scale of each 128x128 block
Output: W in the fd8 order the K loop streams (per 16 rows and 128-K chunk, the 32-K
group of each row in lane order: [N/16, K/128, klane 4, nlane 16, 32 B]); a block whose
scale is 0 has its W zeroed (the kernels treat a zero scale as 1); the scales as is.
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
    """fd2 scales from the standard preshuffled W / e8m0 scale bytes (before
    ``_fast_decode``): scale byte -> min(max_code - code, 11) * 8, and the W of flushed
    groups (max_code - code > 11, which the kernels decode to 0) set to 0. ``exp``:
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
    w = w_u8.reshape(NT // 16, K // 128, 4, 16, 16).clone()
    w[flushed] = 0
    return w.view(w_u8.shape), s.contiguous().view(-1)


def prepare_weights(w1, w1_scale, w2, w2_scale):
    """One layer's experts in the standard preshuffle -> ``(w1, w1_scale, w2, w2_scale)``
    in fd2 form (see the module docstring). w1 [E, 2I, H/2], w2 [E, H, I/2]."""
    E, n2, hk = w1.shape
    if w2.shape[0] != E or w2.shape[1] != hk * 2 or w2.shape[2] * 2 != n2 // 2:
        raise ValueError(f"w1 {tuple(w1.shape)} and w2 {tuple(w2.shape)} do not match")
    out = []
    for w, s in ((w1, w1_scale), (w2, w2_scale)):
        w_u8 = w.contiguous().view(torch.uint8)
        s_u8 = s.contiguous().view(torch.uint8).view(-1)
        exp = channel_exponents(w_u8, s_u8)  # from the original scales
        w_u8, s_u8 = _predelta(w_u8, s_u8, exp)
        tail = exp.contiguous().view(-1).view(torch.uint8)
        out += [_fast_decode(w_u8).view(w_u8.shape), torch.cat([s_u8, tail])]
    return tuple(out)


def tail_exp(w, scale):
    """The int32 [E, N] per-channel exponents at the tail of the fd2 ``scale`` of ``w``."""
    experts, channels = w.shape[:2]
    tail = experts * channels * 4
    return scale.view(torch.uint8)[-tail:].view(torch.int32).view(experts, channels)


def _fd8(w, scale):
    """fp8 W [E, N, K] in the (16, 16) preshuffle ([N/16, K/32, 2, 16 rows, 16 B]) ->
    fd8 ([N/16, K/128, klane 4, nlane 16, 32 B]); blocks with a zero scale zeroed."""
    E, N, K = w.shape
    if N % 128 or K % 128 or tuple(scale.shape) != (E, N // 128, K // 128):
        raise ValueError(
            f"w {tuple(w.shape)} / scale {tuple(scale.shape)}: expected N, K multiples "
            "of 128 and one scale per 128x128 block"
        )
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
    w8 = w.contiguous().view(torch.uint8).view(E, N // 16, K // 128, 4, 2, 16, 16)
    w8 = w8.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
    zero = (scale == 0).repeat_interleave(8, dim=1)  # [E, N/16, K/128]
    if bool(zero.any()):
        w8[zero] = 0
    return w8.view(E, N, K).view(w.dtype)


def prepare_weights_a8w8(w1, w1_scale, w2, w2_scale):
    """One layer's experts in aiter's fp8 (16, 16) preshuffle -> ``(w1, w1_scale, w2,
    w2_scale)`` in fd8 form (see the module docstring). w1 [E, 2I, H], w2 [E, H, I]."""
    E, n2, h = w1.shape
    if tuple(w2.shape) != (E, h, n2 // 2):
        raise ValueError(f"w1 {tuple(w1.shape)} and w2 {tuple(w2.shape)} do not match")
    out = []
    for w, s in ((w1, w1_scale), (w2, w2_scale)):
        if w.dtype != torch.float8_e4m3fnuz:
            raise TypeError(f"{w.dtype} weights: expected float8_e4m3fnuz")
        s = s.to(torch.float32).contiguous()
        out += [_fd8(w, s), s]
    return tuple(out)
