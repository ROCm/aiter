# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Kimi-K3 AttnRes for gfx1250, r9: drop-in op + whole-forward schedule (kernels: _gluon_kernels/.../attn_res_r9.py).
#   attn_res(...)        drop-in for vllm.models.kimi_k3.amd.ops.attn_res.attn_res
#   attn_res_produce()   a call that also writes the fp32 block mixtures + stats of the next K calls (group head)
#   attn_res_consume()   a call that reads one mixture row + stats instead of the blocks (group member)
#   AttnResSchedule      the model's call list; runs of n >= 5 calls with the same n are mix-ahead groups
import functools

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.attention.attn_res_r9 import (
    _attn_res_kernel,
    _attn_res_produce_kernel,
    _attn_res_consume_kernel,
    _attn_res_logits_kernel,
)

LDS_PER_CU = 320 * 1024
AHEAD_K = {5: 23, 6: 23, 7: 23, 8: 17}  # mixtures per group (one group per n in K3.1: 24 or 18 calls)

# Default launches (k3). Plain calls by n: kernel ("plain" | "produce" = produce kernel without mixtures), LDS token
# slots, early slot release, warps, CTAs per CU (0 = as many as fit, <= 4).
PLAIN_CFG = {
    1: dict(kernel="produce", nslot=1, early=1, num_warps=4, cpc=lambda T, fp8: 3 if fp8 else 2),
    3: dict(kernel="plain", nslot=1, early=1, num_warps=4, cpc=lambda T, fp8: 2),
    4: dict(kernel="plain", nslot=1, early=1, num_warps=4, cpc=lambda T, fp8: 4 if T >= 4096 else 2),
}
PLAIN_DEFAULT = dict(kernel="plain", nslot=2, early=0, num_warps=None, cpc=lambda T, fp8: 0)  # n = 0, 2 (n >= 5 alone)
PRODUCE_CFG = dict(nslot=2, num_warps=8, cpc=0)
CONSUME_CFG = dict(nslot=1, num_warps=4, cpc=lambda has_delta: 3 if has_delta else 4)

# r9 overrides, tuned in the 187-call forward (r9/tune.py). Tier by T: 2 = T >= 4096, 1 = 1024..4095, 0 = 512..1023,
# 3 = below 512 (k3). Keys: n<n>[f|b] plain call by n and fp8/bf16 output, cons[f|b] group member, h<n> group head.
# Values: ns (slots), ea (early release), nw (warps), cpc (CTAs/CU), ub (Membar-only barriers), pl (WMMA logits
# kernel + head mixing from them; the value is the logits kernel's warps), fexp (bare v_exp_f32 in the mixture loop).
_HEADS = {f"h{n}": dict(pl=8, fexp=1, ub=1) for n in (5, 6, 7, 8)}  # logits kernel (8 warps) + PL head
_HEADS4 = {f"h{n}": dict(pl=4, fexp=1, ub=1) for n in (5, 6, 7, 8)}
# fmt: off
TUNE = {
    2: dict(n1f=dict(ub=1), n1b=dict(ub=1), n2f=dict(ns=1, ea=1, cpc=2, ub=1), n2b=dict(ns=1, ea=1, cpc=3, ub=1),
            n3f=dict(ub=1), n3b=dict(ub=1), n4f=dict(cpc=2, ub=1), n4b=dict(cpc=2, ub=1), **_HEADS),
    1: dict(n2f=dict(ns=1, ea=1, nw=4, cpc=2, ub=1), n2b=dict(nw=4, ub=1), n3f=dict(ub=1), n3b=dict(ub=1),
            n4f=dict(ub=1), n4b=dict(ub=1), consb=dict(cpc=3), **_HEADS),
    0: dict(n1f=dict(ub=1), n2f=dict(nw=4), n2b=dict(nw=4), n3f=dict(nw=8, ub=1), n3b=dict(ub=1), n4f=dict(ub=1),
            n4b=dict(ub=1), consf=dict(nw=8), **_HEADS4),
}
# fmt: on
TUNE[3] = {}  # T < 512 (decode-like): exactly k3


def _tune(key, T, fp8=None):
    t = TUNE[2 if T >= 4096 else 1 if T >= 1024 else 0 if T >= 512 else 3]
    return t.get(key + ("" if fp8 is None else "f" if fp8 else "b"), t.get(key, {}))


@functools.lru_cache(maxsize=None)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_WSPLIT = {}


def _w_split(w):
    """[2K, H] bf16 = (W_hi; W_lo) with W_hi + W_lo == w exactly (w = norm_w * qk_w has <= 16 significant bits)"""
    key = (w.data_ptr(), w.shape[0])
    if key not in _WSPLIT:
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        assert torch.equal(hi.float() + lo.float(), w), "w does not split exactly into two bf16"
        _WSPLIT[key] = (torch.cat([hi, lo]).contiguous(), w)
    return _WSPLIT[key][0]


def _out(prefix, quant_dtype):
    out = torch.empty_like(prefix, dtype=quant_dtype or prefix.dtype, memory_format=torch.contiguous_format)
    scale = (
        torch.empty((prefix.shape[0], 1), device=prefix.device, dtype=torch.float32)
        if quant_dtype is not None
        else None
    )
    qmax = float(torch.finfo(quant_dtype).max) if quant_dtype is not None else 0.0
    return out, scale, qmax


def _pieces(nch, chunk):
    """power-of-two column pieces covering nch chunks (7168 = 4096 + 2048 + 1024)"""
    ps, pw, start = [], [], 0
    for bit in reversed(range(nch.bit_length())):
        if nch >> bit & 1:
            ps.append(start)
            pw.append(chunk << bit)
            start += chunk << bit
    return ps, pw


def _block_tiles(num_blocks, hidden, nch, chunk):
    """block tile: one [NBP, BD] tile, or (8 rows) column pieces so that two tokens fit in LDS"""
    nbp = triton.next_power_of_2(max(num_blocks, 1))
    bd = triton.next_power_of_2(hidden)
    if nbp * bd * 2 * 2 > LDS_PER_CU // 2:
        return nbp, bd, _pieces(nch, chunk)
    return nbp, bd, ([0], [bd])


def _check(prefix, delta, blocks, num_blocks, block_write_idx, quant_dtype):
    assert (
        prefix.stride(-1) == 1
        and blocks.stride(-1) == 1
        and blocks.stride(1) == prefix.shape[1]
        and blocks.shape[1] == 8
    )
    assert delta is None or delta.stride(-1) == 1
    assert block_write_idx < 0 or block_write_idx >= num_blocks  # K3.1 writes the next free slot
    assert quant_dtype in (None, torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _launch_plain(
    prefix,
    delta,
    blocks,
    nw_,
    qk_,
    onw_,
    n,
    widx,
    eps,
    oeps,
    quant_dtype,
    store_prefix,
    nslot,
    early,
    num_warps,
    cpc,
    ub=0,
):
    T, H = prefix.shape
    out, scale, qmax = _out(prefix, quant_dtype)
    nbp = triton.next_power_of_2(max(n, 1))
    if num_warps is None:
        num_warps = 4 if (nbp <= 4 and T >= 1024) else 8
    epl = 32 // num_warps
    chunk = 32 * epl * num_warps
    nch = triton.cdiv(H, chunk)
    nbp, bd, (ps, pw) = _block_tiles(n, H, nch, chunk)
    lds = nslot * 2 * ((1 + (delta is not None)) * bd + (nbp * sum(pw) if n else 0))
    cpc = cpc or max(1, min(4, LDS_PER_CU // lds))
    grid = min(T, cpc * _num_cus(prefix.device.index or 0))
    _attn_res_kernel[(grid,)](
        prefix,
        delta,
        blocks,
        nw_,
        qk_,
        onw_,
        out,
        scale,
        T,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        out.stride(0),
        eps,
        oeps,
        HIDDEN=H,
        NUM_BLOCKS=n,
        BLOCK_WRITE_IDX=max(widx, 0),
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=widx >= 0,
        STORE_PREFIX=store_prefix,
        APPLY_OUTPUT_NORM=onw_ is not None,
        QUANT_MAX=qmax,
        NBP=nbp,
        EPL=epl,
        C=chunk,
        NCH=nch,
        BD=bd,
        PS=tuple(ps),
        PW=tuple(pw),
        NSLOT=nslot,
        NUM_WARPS=num_warps,
        EARLY=early,
        UB=ub,
        num_warps=num_warps,
    )
    return out if scale is None else (out, scale)


def _logits(blocks, w_next, n, T, H, sw=256, nw=8):
    """partial logits plog [T, SP, 8, 32] of a group head (WMMA kernel, one 256-column slice per CTA)"""
    k = w_next.shape[0]
    s = H // sw
    sp = triton.next_power_of_2(s)
    assert blocks.stride(0) == 8 * H and blocks.is_contiguous()
    plog = torch.empty((T, sp, 32, 8), device=blocks.device, dtype=torch.float32)
    grid = s * max(1, min(triton.cdiv(T, 2 * nw), _num_cus(blocks.device.index or 0) // s))
    _attn_res_logits_kernel[(grid,)](
        blocks,
        _w_split(w_next),
        plog,
        T,
        k,
        blocks.stride(0),
        HIDDEN=H,
        NUM_BLOCKS=n,
        S=s,
        SW=sw,
        NUM_WARPS=nw,
        SP=sp,
        num_warps=nw,
    )
    return plog, s, sp


def attn_res_produce(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    w_next=None,
    quant_dtype=None,
    store_prefix=True,
    nslot=PRODUCE_CFG["nslot"],
    num_warps=PRODUCE_CFG["num_warps"],
    ctas_per_cu=PRODUCE_CFG["cpc"],
    early=0,
    ub=0,
):
    """A regular call; with w_next [K, H] fp32 (norm_weight * qk_weight of the next K calls) it also returns the
    mixtures mix [T, K, H] fp32 and stats [T, K, 2] (max logit, sum of exp). Returns (out[, scale][, mix, stats])."""
    _check(prefix, delta, blocks, num_blocks, block_write_idx, quant_dtype)
    T, H = prefix.shape
    ahead = w_next is not None and w_next.shape[0] > 0
    assert not ahead or (num_blocks > 0 and w_next.dtype == torch.float32 and w_next.is_contiguous())
    k = w_next.shape[0] if ahead else 0
    out, scale, qmax = _out(prefix, quant_dtype)
    if ahead:
        mix = torch.empty((T, k, H), device=prefix.device, dtype=torch.float32)
        stats = torch.empty((T, k, 2), device=prefix.device, dtype=torch.float32)
    else:
        mix = stats = torch.empty(1, device=prefix.device, dtype=torch.float32)  # unused
    head = _tune(f"h{num_blocks}", T) if ahead else {}
    plog, s, sp = (stats, 28, 32)
    if head.get("pl"):
        plog, s, sp = _logits(blocks, w_next, num_blocks, T, H, nw=head["pl"])
    epl = 32 // num_warps
    chunk = 32 * epl * num_warps
    nch = triton.cdiv(H, chunk)
    nbp, bd, (ps, pw) = _block_tiles(num_blocks, H, nch, chunk)
    lds = nslot * 2 * ((1 + (delta is not None)) * bd + (nbp * sum(pw) if num_blocks else 0))
    cpc = ctas_per_cu or max(1, min(4, LDS_PER_CU // lds))
    grid = min(T, cpc * _num_cus(prefix.device.index or 0))
    _attn_res_produce_kernel[(grid,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        scale,
        w_next if ahead else norm_weight,
        mix,
        stats,
        plog,
        T,
        k,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        out.stride(0),
        eps,
        output_norm_eps,
        HIDDEN=H,
        NUM_BLOCKS=num_blocks,
        BLOCK_WRITE_IDX=max(block_write_idx, 0),
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        STORE_PREFIX=store_prefix,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        QUANT_MAX=qmax,
        NBP=nbp,
        EPL=epl,
        C=chunk,
        NCH=nch,
        BD=bd,
        PS=tuple(ps),
        PW=tuple(pw),
        NSLOT=nslot,
        AHEAD=ahead,
        NUM_WARPS=num_warps,
        EARLY=early,
        UB=head.get("ub", ub),
        PL=1 if head.get("pl") else 0,
        S=s,
        SP=sp,
        FEXP=head.get("fexp", 0),
        num_warps=num_warps,
    )
    res = (out,) if scale is None else (out, scale)
    return res + ((mix, stats) if ahead else ())


def attn_res_consume(
    prefix,
    delta,
    blocks,
    mix,
    stats,
    norm_weight,
    qk_weight,
    output_norm_weight,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
    store_prefix=True,
    nslot=CONSUME_CFG["nslot"],
    num_warps=CONSUME_CFG["num_warps"],
    ctas_per_cu=None,
):
    """A call on one mixture row mix [T, H] + stats [T, 2] (views of attn_res_produce's outputs) instead of the
    blocks; block_write_idx >= 0 also writes bf16(prefix + delta) into blocks[:, block_write_idx]."""
    T, H = prefix.shape
    c = _tune("cons", T, quant_dtype is not None)
    nslot, num_warps, ctas_per_cu = c.get("ns", nslot), c.get("nw", num_warps), c.get("cpc", ctas_per_cu)
    assert prefix.stride(-1) == 1 and (delta is None or delta.stride(-1) == 1)
    assert mix.shape == prefix.shape and mix.stride(-1) == 1 and stats.stride(-1) == 1
    out, scale, qmax = _out(prefix, quant_dtype)
    epl = 32 // num_warps
    chunk = 32 * epl * num_warps
    assert H % chunk == 0
    nch = H // chunk
    ps, pw = _pieces(nch, chunk)
    pbd = triton.next_power_of_2(H)  # prefix / delta / mixture rows as one padded tile each
    cpc = ctas_per_cu or CONSUME_CFG["cpc"](delta is not None)
    assert cpc * nslot * ((1 + (delta is not None)) * 2 + 4) * pbd <= LDS_PER_CU, "LDS"
    grid = min(T, cpc * _num_cus(prefix.device.index or 0))
    _attn_res_consume_kernel[(grid,)](
        prefix,
        delta,
        blocks,
        mix,
        stats,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        scale,
        T,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        mix.stride(0),
        stats.stride(0),
        out.stride(0),
        eps,
        output_norm_eps,
        HIDDEN=H,
        BLOCK_WRITE_IDX=max(block_write_idx, 0),
        WRITE_BLOCK=block_write_idx >= 0,
        HAS_DELTA=delta is not None,
        STORE_PREFIX=store_prefix,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        QUANT_MAX=qmax,
        EPL=epl,
        C=chunk,
        NCH=nch,
        PS=tuple(ps),
        PW=tuple(pw),
        NSLOT=nslot,
        PBD=pbd,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return out if scale is None else (out, scale)


def attn_res(
    prefix,
    delta,
    blocks,
    norm_weight,
    qk_weight,
    output_norm_weight,
    num_blocks,
    block_write_idx,
    eps,
    output_norm_eps,
    *,
    quant_dtype=None,
    store_prefix=True,
):
    """Drop-in for the vLLM op (plus store_prefix=False to skip a dead prefix store). Returns out or (out, scale)."""
    _check(prefix, delta, blocks, num_blocks, block_write_idx, quant_dtype)
    T = prefix.shape[0]
    if T == 0:
        out, scale, _ = _out(prefix, quant_dtype)
        return out if scale is None else (out, scale)
    fp8 = quant_dtype is not None
    cfg = PLAIN_CFG.get(num_blocks, PLAIN_DEFAULT)
    c = _tune(f"n{num_blocks}", T, fp8)
    nslot, early = c.get("ns", cfg["nslot"]), c.get("ea", cfg["early"])
    nw, cpc, ub = c.get("nw", cfg["num_warps"]), c.get("cpc", cfg["cpc"](T, fp8)), c.get("ub", 0)
    args = (
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
    )
    if cfg["kernel"] == "produce":
        res = attn_res_produce(
            *args,
            quant_dtype=quant_dtype,
            store_prefix=store_prefix,
            nslot=nslot,
            num_warps=nw or 4,
            ctas_per_cu=cpc,
            early=early,
            ub=ub,
        )
        return res[0] if quant_dtype is None else res
    return _launch_plain(*args, quant_dtype, store_prefix, nslot, early, nw, cpc, ub)


class AttnResSchedule:
    """calls: the model's AttnRes calls in order, dicts with n, norm_weight, qk_weight, output_norm_weight,
    block_write_idx, eps, output_norm_eps, quant_dtype, store_prefix (False on block-write and final calls).
    Runs of n >= 5 calls with the same n are split into groups of AHEAD_K[n] + 1 calls; schedule(i, prefix,
    delta, blocks) runs call i. use_ahead=False runs every call as a plain call."""

    def __init__(self, calls, use_ahead=True, ahead_k=None):
        self.calls = calls
        ahead_k = ahead_k or AHEAD_K
        self.role = ["plain"] * len(calls)  # plain | produce | consume
        self.w_next = [None] * len(calls)
        self.src = [None] * len(calls)  # consume: (producer call index, mixture slot)
        i = 0
        while i < len(calls):
            n, j = calls[i]["n"], i
            while j < len(calls) and calls[j]["n"] == n:
                j += 1
            g = i
            while use_ahead and n in ahead_k and g < j:
                e = min(j, g + ahead_k[n] + 1)
                if e - g >= 2:
                    self.role[g] = "produce"
                    self.w_next[g] = torch.stack(
                        [calls[c]["norm_weight"].float() * calls[c]["qk_weight"].float() for c in range(g + 1, e)]
                    ).contiguous()
                    for c in range(g + 1, e):
                        self.role[c], self.src[c] = "consume", (g, c - g - 1)
                g = e
            i = j
        self._mix = {}

    def __call__(self, i, prefix, delta, blocks):
        c = self.calls[i]
        w = (c["norm_weight"], c["qk_weight"], c["output_norm_weight"])
        kw = dict(quant_dtype=c["quant_dtype"], store_prefix=c["store_prefix"])
        if self.role[i] == "plain":
            return attn_res(
                prefix, delta, blocks, *w, c["n"], c["block_write_idx"], c["eps"], c["output_norm_eps"], **kw
            )
        if self.role[i] == "produce":
            res = attn_res_produce(
                prefix,
                delta,
                blocks,
                *w,
                c["n"],
                c["block_write_idx"],
                c["eps"],
                c["output_norm_eps"],
                w_next=self.w_next[i],
                **kw,
            )
            self._mix[i] = res[-2:]
            return res[0] if c["quant_dtype"] is None else (res[0], res[1])
        g, s = self.src[i]
        mix, stats = self._mix[g]
        out = attn_res_consume(
            prefix,
            delta,
            blocks,
            mix[:, s],
            stats[:, s],
            *w,
            c["block_write_idx"],
            c["eps"],
            c["output_norm_eps"],
            **kw,
        )
        if s == self.w_next[g].shape[0] - 1:
            del self._mix[g]  # last consumer of the group
        return out
