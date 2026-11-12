# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Torch reference and inputs for DeepSeek V4.1's RoPE / MXFP8 / window seam.

``rope_quant_window_ref`` is a byte-exact port of ATOM's Triton ``rope_quant_window``:
GPT-J RoPE on the last ``rope_dim`` columns in fp32 (a NoPE pair rotated by cos 1 / sin 0),
rounded to bf16, then MXFP8 per 32 with step ``2^ceil(log2(max(amax, 1e-10) / 448))``.
"""

import random

import torch

from aiter import dtypes

FP8 = dtypes.fp8
E8 = torch.float8_e8m0fnu
D, RD, MAXPOS, RING = 512, 64, 65536, 8
SENTINEL = 0x5A
dev = "cuda"


def rope_tables(rope_dim, max_pos, base=10000.0):
    inv_freq = 1.0 / base ** (
        torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim
    )
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return freqs.cos().to(dev, torch.bfloat16), freqs.sin().to(dev, torch.bfloat16)


COS, SIN = rope_tables(RD, MAXPOS)


def _rotate(x, cos, sin):
    """GPT-J pairs of fp32 rows x [N, 2P] by cos / sin [N, P], as two products and a sum."""
    even, odd = x[:, 0::2], x[:, 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(1)


def _rope_quant_rows(rows, pos, rope_dim):
    """Rotate and MXFP8-quantize bf16 rows [N, dim]: (fp8 bytes, e8m0 codes, steps)."""
    n, dim = rows.shape
    cos = torch.ones(n, dim // 2, device=dev)
    sin = torch.zeros(n, dim // 2, device=dev)
    cos[:, (dim - rope_dim) // 2 :] = COS[pos].float()
    sin[:, (dim - rope_dim) // 2 :] = SIN[pos].float()
    y = _rotate(rows.float(), cos, sin).to(torch.bfloat16).float()
    y = y.view(n, dim // 32, 32)
    amax = y.abs().amax(-1).clamp_min(1e-10) * (1.0 / 448.0)
    bits = amax.view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    step = (code << 23).view(torch.float32)
    q = (y / step[..., None]).clamp(-448.0, 448.0).to(FP8)
    return q.view(n, dim), code.to(torch.uint8), step


def _dequant(q, step):
    n, dim = q.shape
    q = q.float().view(n, dim // 32, 32) * step[..., None]
    return q.view(n, dim).to(torch.bfloat16)


def in_ref_steps(x, ref_codes, codes=None):
    """x [N, dim] in units of the step 2^(ref_code - 127) of each group of 32, with x
    first dequantized by its own e8m0 codes when given. An MXFP8 output then lands within
    one e4m3 step of the exact rows in those units whatever code its rounding chose, while
    a code off by one everywhere doubles or halves it."""
    shift = 127.0 - ref_codes.view(torch.uint8).float()
    if codes is not None:
        shift += codes.view(torch.uint8).float() - 127.0
    return x.float() * torch.exp2(shift).repeat_interleave(32, dim=-1)


def rope_quant_window_ref(
    query,
    kv,
    positions,
    *,
    rope_dim,
    qat=None,
    values=None,
    scales=None,
    ring=None,
    query_fp8=None,
):
    """Torch port of ATOM's Triton ``rope_quant_window``; writes the same sinks."""
    tokens, heads, dim = query.shape
    pos = positions.long()
    if heads:
        p = pos.repeat_interleave(heads)
        if query_fp8 is not None:
            qv, qs = query_fp8
            q8, codes, _ = _rope_quant_rows(query.reshape(-1, dim), p, rope_dim)
            qv.copy_(q8.view(tokens, heads, dim))
            qs.view(torch.uint8).copy_(codes.view(tokens, heads, -1))
        else:
            tail = query[..., dim - rope_dim :].reshape(-1, rope_dim)
            rotated = _rotate(tail.float(), COS[p].float(), SIN[p].float())
            query[..., dim - rope_dim :] = rotated.to(query.dtype).view(
                tokens, heads, rope_dim
            )
    if kv is None or (qat is None and values is None and ring is None):
        return
    q8, codes, step = _rope_quant_rows(kv, pos, rope_dim)
    if qat is not None:
        qat.copy_(_dequant(q8, step))
    if values is not None:
        values.copy_(q8)
        scales.view(torch.uint8).copy_(codes)
    if ring is not None:
        # A padding row (batch -1) owns no request, and a row the ring has scrolled past
        # has nowhere to land: only the last ring_slots tokens of a request are written.
        pool, step_info, w = ring
        batch = step_info.batch_ids.long()
        cu = step_info.cu_seqlens_q.long()
        token = torch.arange(tokens, device=dev)
        live = (batch >= 0) & (token >= cu[(batch + 1).clamp_min(0)] - w.ring_slots)
        t = token[live]
        ring_pos = pos[live] % w.ring_slots
        row = (
            step_info.slots.long()[batch[live]] * w.slot_rows
            + w.ring_start
            + (ring_pos // w.ring_stride) * w.run_rows
            + ring_pos % w.ring_stride
        )
        if isinstance(pool, tuple):
            data, scale = pool
            data.view(torch.uint8)[row] = q8[t].view(torch.uint8)
            scale.view(torch.uint8)[row] = codes[t]
        else:
            pool[row] = _dequant(q8[t], step[t])


def with_zeros(t):
    t.view(-1)[::97] = 0.0
    t.view(-1)[1::89] = -0.0
    return t


class Step:
    """A step of up to three requests plus one padding row (batch -1)."""

    def __init__(self, tokens, idx64):
        real = tokens - 1 if tokens > 1 else 1
        nreq = min(3, real)
        bounds = (
            [0] + sorted(random.sample(range(1, real), nreq - 1)) + [real]
            if nreq > 1
            else [0, real]
        )
        dt = torch.int64 if idx64 else torch.int32
        batch = torch.full((tokens,), -1, dtype=torch.int64)
        pos = torch.zeros(tokens, dtype=torch.int64)
        for r in range(nreq):
            lo, hi = bounds[r], bounds[r + 1]
            batch[lo:hi] = r
            pos[lo:hi] = random.randint(0, 3000) + torch.arange(hi - lo)
        self.batch_ids = batch.to(dev, torch.int32)
        self.cu_seqlens_q = torch.tensor(bounds, device=dev, dtype=dt)
        self.slots = torch.tensor(random.sample(range(6), nreq), device=dev, dtype=dt)
        self.positions = pos.to(dev, dt)


class PaddingStep:
    """A step of padding rows only: no request, so cu_seqlens_q is [0] and slots is empty."""

    def __init__(self, tokens, idx64):
        dt = torch.int64 if idx64 else torch.int32
        self.batch_ids = torch.full((tokens,), -1, device=dev, dtype=torch.int32)
        self.cu_seqlens_q = torch.zeros(1, device=dev, dtype=dt)
        self.slots = torch.zeros(0, device=dev, dtype=dt)
        self.positions = torch.randint(0, 3000, (tokens,), device=dev).to(dt)


def window(start=4096, ring_stride=RING, run_rows=RING):
    """Ring geometry in rows: position p of a slot lands on run p // ring_stride, row
    p % ring_stride of it, runs run_rows apart."""
    w = type("Window", (), {})()
    w.ring_start, w.ring_slots = start, RING
    w.ring_stride, w.run_rows = ring_stride, run_rows
    w.slot_rows = (RING // ring_stride) * run_rows
    return w


def sinks(mode, tokens, heads):
    """Zeroed outputs for a mode, a '+'-joined subset of: qfp8 (query MXFP8, else rotated in
    place), qat, vs (values / scales), rows (values / scales packed in one row), ringp (fp8 /
    e8m0 ring planes, contiguous), ringps (the planes as strided slices, the scale rows not
    dword-aligned, the ring rows in gapped runs) and ringh (bf16 pool)."""
    o = {}
    if "qfp8" in mode:
        o["qv"] = torch.zeros(tokens, heads, D, device=dev, dtype=FP8)
        o["qs"] = torch.zeros(tokens, heads, D // 32, device=dev, dtype=E8)
    if "ringps" in mode:
        o["w"] = window(4099, ring_stride=2, run_rows=3)
        rows = o["w"].ring_start + 6 * o["w"].slot_rows
        o["pool"] = torch.zeros(rows, D + 64, device=dev, dtype=torch.uint8)
        o["pool_s"] = torch.zeros(rows, D // 32 + 4, device=dev, dtype=torch.uint8)
    elif "ringp" in mode:
        o["w"] = window()
        rows = o["w"].ring_start + 6 * o["w"].slot_rows
        o["pool"] = torch.zeros(rows, D, device=dev, dtype=torch.uint8)
        o["pool_s"] = torch.zeros(rows, D // 32, device=dev, dtype=torch.uint8)
    if "ringh" in mode:
        o["w"] = window()
        rows = o["w"].ring_start + 6 * o["w"].slot_rows
        o["pool"] = torch.zeros(rows, D, device=dev, dtype=torch.bfloat16)
    if "rows" in mode:
        o["rows"] = torch.zeros(tokens, D + D // 32, device=dev, dtype=torch.uint8)
    if "vs" in mode:
        o["v"] = torch.zeros(tokens, D, device=dev, dtype=FP8)
        o["s"] = torch.zeros(tokens, D // 32, device=dev, dtype=E8)
    if "qat" in mode:
        o["qat"] = torch.zeros(tokens, D, device=dev, dtype=torch.bfloat16)
    # A nonzero sentinel, the same in the reference's sinks, so a store that never lands
    # shows even where the right byte is 0; rows nobody writes still match.
    for k, t in o.items():
        if k != "w":
            t.view(torch.uint8).fill_(SENTINEL)
    return o


def sink_kwargs(o, step):
    kw = {}
    if "qv" in o:
        kw["query_fp8"] = (o["qv"], o["qs"])
    if "pool_s" in o:
        data, scale = o["pool"], o["pool_s"]
        if data.shape[1] != D:
            data, scale = data[:, 32 : 32 + D], scale[:, 3 : 3 + D // 32]
        kw["ring"] = ((data.view(FP8), scale.view(E8)), step, o["w"])
    elif "pool" in o:
        kw["ring"] = (o["pool"], step, o["w"])
    if "rows" in o:
        kw["values"] = o["rows"][:, :D].view(FP8)
        kw["scales"] = o["rows"][:, D:].view(E8)
    if "v" in o:
        kw["values"], kw["scales"] = o["v"], o["s"]
    if "qat" in o:
        kw["qat"] = o["qat"]
    return kw


def byte_diff(ref, got):
    """{output: differing bytes} over the outputs that differ."""
    out = {}
    for k in ref:
        if k == "w":
            continue
        n = int((ref[k].view(torch.uint8) != got[k].view(torch.uint8)).sum().item())
        if n:
            out[k] = n
    return out
