# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf tests for the gfx1250 varlen ASM forward kernels:
fmha_fwd_with_sink_varlen_asm (BF16) and fmha_fwd_fp8_varlen_asm (FP8 D192x128).

BF16  public API:  aiter.flash_attn_varlen_func          (the path the model calls)
      ops layer:   aiter.fmha_fwd_with_sink_varlen_asm    (low-level, packed/varlen)
FP8   public API:  aiter.flash_attn_varlen_fp8_pertensor_func,
                   aiter.flash_attn_varlen_func(..., q/k/v_descale=...)
      ops layer:   aiter.fmha_fwd_fp8_varlen_asm          (per-tensor fp32 descales,
                   qk 192 / v 128, bf16 output; its tables and tolerances are in
                   the FP8 section below; --dtype picks bf16 / fp8)

Built to the aiter op-test standard (see .claude/skills/aiter-op-test): mirror
test_quant.py — @benchmark + run_perftest candidate loop, a torch reference,
per-candidate us / TFLOPS / TB/s / err, a markdown summary table per test
function, and a __main__ guard so the module is importable.

Layout (packed THD; batch folded into the token axis):
    q   : (total_q, nheads,   hdim_q)
    k   : (total_k, nheads_k, hdim_q)
    v   : (total_k, nheads_k, hdim_v)
    out : (total_q, nheads,   hdim_v)
    cu_seqlens_q / cu_seqlens_k : int32 [batch+1] cumulative

Head dims: (64, 64) and (128, 128) are the symmetric kernels; (192, 128) is the
asymmetric D192x128 one (qk_head_dim=192, v_head_dim=128); (256, 256) is D256.
D192x128 and D256 tile KV by 128 instead of 256.

Sink convention (same as the fixed-batch path): `sink` ([q_head_num] fp32) is a
per-Q-head logit in the scaled domain, a zero-value virtual KV column passed
verbatim.  D64 kernels read it; D128, D192x128 and D256 kernels ignore it (pass
None).

KV-length constraint (mask=0 only): the non-causal D64/D128 kernels require
per-sequence kv_seqlen that is a multiple of 256.  D192x128 and D256 have no such
constraint (sub_K=128 with the border mask compiled in).

Strided inputs: D192x128 and D256 take q/k/v/out views (e.g. v = kv[..., 128:]);
they are checked against the reference and bit-for-bit against contiguous inputs
(BF16 and FP8 alike).
"""

import argparse
import functools
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes, per_tensor_quant
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.test_common import assertAllclose, benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# .co files only ship for gfx1250 (hsa/gfx1250/fmha_fwd_bf16_varlen/*.co).
SUPPORTED_GFX = ["gfx1250"]


# ---------------------------------------------------------------------------
# Reference (fp32 math, cast back).  Not timed, not in the table.
# ---------------------------------------------------------------------------


def _attn_one(q, k, v, *, is_causal, sink):
    """Single-sequence attention reference (no batch dim).

    q: (sq, hq, d)  k: (sk, hk, d)  v: (sk, hk, dv) -> out (sq, hq, dv), lse (sq, hq).
    A row that sees no key (sk == 0, or causal with sq > sk) gets O = 0 and
    LSE = -inf (unless a sink logit gives it one).
    """
    sq, hq, d = q.shape
    sk, hk, _ = k.shape
    if sk == 0 and sink is None:
        out = torch.zeros((sq, hq, v.shape[-1]), dtype=q.dtype, device=q.device)
        return out, torch.full((sq, hq), float("-inf"), device=q.device)
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale
    if is_causal:
        row = torch.arange(sq, device=q.device)[:, None]
        col = torch.arange(sk, device=q.device)[None, :]
        masked = col > (row + (sk - sq))  # bottom-right aligned causal mask
        scores = scores.masked_fill(masked[None], float("-inf"))
    max_attn = scores.max(dim=-1).values
    if sink is not None:
        sink_hs = sink.float()[:, None].expand(hq, sq)
        max_total = torch.maximum(max_attn, sink_hs)
    else:
        max_total = max_attn
    # -inf only on rows that see no key: subtract 0 there, so exp gives 0, the
    # sum 0, O 0 and the LSE log(0) = -inf.
    max_total = torch.where(torch.isinf(max_total), 0.0, max_total)
    denom = torch.exp(scores - max_total.unsqueeze(-1)).sum(dim=-1)
    if sink is not None:
        denom = denom + torch.exp(sink_hs - max_total)
    probs = torch.exp(scores - max_total.unsqueeze(-1)) / torch.where(
        denom > 0, denom, 1.0
    ).unsqueeze(-1)
    out = torch.einsum("hqk,khd->qhd", probs, vf).to(q.dtype)
    lse = (torch.log(denom) + max_total).transpose(0, 1)
    return out, lse


def run_torch(q, k, v, cu_q, cu_k, *, is_causal, sink):
    """Packed-THD reference: loop over batches, slice via cu_seqlens."""
    total_q, hq, _ = q.shape
    dv = v.shape[-1]
    batch = cu_q.numel() - 1
    out = torch.empty((total_q, hq, dv), dtype=q.dtype, device=q.device)
    lse = torch.empty((total_q, hq), dtype=dtypes.fp32, device=q.device)
    cuq, cuk = cu_q.tolist(), cu_k.tolist()
    for b in range(batch):
        q0, q1 = cuq[b], cuq[b + 1]
        k0, k1 = cuk[b], cuk[b + 1]
        if q1 == q0:
            continue
        ob, lb = _attn_one(q[q0:q1], k[k0:k1], v[k0:k1], is_causal=is_causal, sink=sink)
        out[q0:q1] = ob
        lse[q0:q1] = lb
    return out, lse


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def make_varlen_packed(seqlens: list[int], hq, hk, d, dv, init="randn", seed=0):
    """Build packed THD q/k/v + cu_seqlens for the given per-batch seqlens.

    Equal q/k seqlens per batch (standard varlen self-attention).
    init: "randn" or "const0.25".
    """
    torch.manual_seed(seed)
    cu = torch.tensor(
        [0] + list(torch.tensor(seqlens).cumsum(0).tolist()), dtype=dtypes.i32
    )
    total = int(cu[-1].item())
    q = torch.randn(total, hq, d, dtype=dtypes.bf16)
    k = torch.randn(total, hk, d, dtype=dtypes.bf16)
    v = torch.randn(total, hk, dv, dtype=dtypes.bf16)
    if init == "const0.25":
        q.fill_(0.25)
        k.fill_(0.25)
        v.fill_(0.25)
    elif init != "randn":
        raise ValueError(f"unknown init pattern: {init!r}")
    return q, k, v, cu


def _d64_sink(hq):
    """Per-head sink logits in [0.5, 2.0] (scaled domain), varied across heads."""
    return torch.linspace(0.5, 2.0, hq, dtype=dtypes.fp32)


def run_kernel(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_seqlen_q,
    *,
    scale,
    is_causal,
    sink=None,
    via="public",
    out=None,
    max_seqlen_k=None,
):
    """Return (out, lse) with lse shaped (total_q, nheads) to match run_torch.
    max_seqlen_k defaults to max_seqlen_q (equal q/k seqlens).

    via="public" → aiter.flash_attn_varlen_func (the model path); lse comes back
                   (nheads, total_q) and is transposed here.
    via="ops"    → aiter.fmha_fwd_with_sink_varlen_asm; lse is (total_q, nheads, 1).
    """
    if via == "public":
        r = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_q if max_seqlen_k is None else max_seqlen_k,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
            sink_ptr=sink,
            out=out,
        )
        return r[0], r[1].transpose(0, 1).contiguous()
    if via == "ops":
        out, lse = aiter.fmha_fwd_with_sink_varlen_asm(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            scale,
            is_causal,
            True,
            sink=sink,
            out=out,
        )
        return out, lse.squeeze(-1)
    raise ValueError(f"unknown via={via!r}")


def _flops_bytes(seqlens, hq, hk, d, dv, is_causal, total, esz):
    """Attention roofline numerators summed over the packed batches."""
    flops = sum(2.0 * hq * s * s * (d + dv) for s in seqlens)  # 2 GEMMs (QK^T, PV)
    if is_causal:
        flops /= 2.0
    # q (d) + o (dv) per q head, k (d) + v (dv) per kv head
    nbytes = (total * hq * (d + dv) + total * hk * (d + dv)) * esz
    return flops, nbytes


# ---------------------------------------------------------------------------
# Shape tables
# ---------------------------------------------------------------------------

# Correctness shapes (torch reference feasible).  hq=64; hk=8 (D64) / 4 (others).
# The symmetric D64/D128 kernels tile KV by 256 and their non-causal (mask=0)
# variants require every kv_seqlen % 256 == 0 (filtered); D192x128 and D256 tile
# by 128 with a border mask, so unaligned lengths are in scope for them too.
# (hdim_q, hdim_v, hq, hk, seqlens)
_CORRECTNESS_SHAPES = [
    (64, 64, 64, 8, [256]),
    (128, 128, 64, 4, [256]),
    (192, 128, 64, 4, [256]),
    (64, 64, 64, 8, [128, 256, 384]),  # mixed (some unaligned) -> causal only
    (128, 128, 64, 4, [128, 256, 384]),
    (192, 128, 64, 4, [128, 256, 384]),
    (64, 64, 64, 8, [100, 200, 300]),  # unaligned
    (128, 128, 64, 4, [100, 200, 300]),
    (192, 128, 64, 4, [100, 200, 300]),
    (64, 64, 64, 8, [256, 512]),  # 256-aligned (causal AND mask=0)
    (128, 128, 64, 4, [256, 512]),
    (192, 128, 64, 4, [256, 512]),
    (64, 64, 64, 8, [256, 512, 768]),
    (128, 128, 64, 4, [256, 512, 768]),
    (64, 64, 64, 8, [512, 1024]),
    (128, 128, 64, 4, [512, 1024]),
    (192, 128, 64, 4, [512, 1024]),
    (256, 256, 64, 4, [256]),
    (256, 256, 64, 4, [128, 256, 384]),
    (256, 256, 64, 4, [100, 200, 300]),
    (256, 256, 64, 4, [256, 512]),
    (256, 256, 64, 4, [512, 1024]),
    (256, 256, 16, 16, [2048]),
    # kv heads x mean kv length > 2^17: the host launches without remap_xy (and without
    # double_q when non-causal); the shapes above all take remap_xy
    (256, 256, 64, 64, [2304]),
    (256, 256, 64, 64, [1000, 3000, 2500]),
]

# Perf-only shapes (torch ref O(s^2) infeasible at 16384/32768).
# (hdim_q, hdim_v, hq, hk, seqlens)
_VARLEN_PERF_SHAPES = [
    (64, 64, 64, 8, [4096, 4096]),
    (128, 128, 64, 4, [2048, 2048]),
    (128, 128, 64, 4, [16384]),
    (64, 64, 64, 8, [32768]),
    (192, 128, 64, 4, [2048, 2048]),
    (192, 128, 64, 4, [16384]),
    (256, 256, 64, 4, [2048, 2048]),
    (256, 256, 16, 16, [16384]),
    (256, 256, 16, 16, [32768]),
    (256, 256, 16, 2, [4096, 4096, 4096, 4096]),
    (256, 256, 32, 32, [16000] * 4),  # mi455x-opbench attn/uniform/B4L16000/h32d256
    (256, 256, 32, 1, [16384]),  # MQA
    (256, 256, 64, 64, [1024] * 16),
]


def _kv_alignment_ok(hdim_q, seqlens):
    """Non-causal eligibility for the per-head-dim KV tile.

    D64/D128 use sub_K=256 and need every kv_seqlen to be a multiple of it.
    D192x128 and D256 use sub_K=128 and compile the border mask in, so any length
    works.
    """
    if hdim_q in (192, 256):
        return True
    return all(s % 256 == 0 for s in seqlens)


# ---------------------------------------------------------------------------
# Test functions (one markdown table each).
# ---------------------------------------------------------------------------


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm(
    hdim_q, hdim_v, hq, hk, seqlens, is_causal, init, d64_sink=True
):
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, hdim_q, hdim_v, init=init)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(hdim_q)
    # D64 without a sink: the wrapper passes the kernel a -inf sink.
    sink = _d64_sink(hq) if hdim_q == 64 and d64_sink else None

    ref_out, ref_lse = run_torch(q, k, v, cu, cu, is_causal=is_causal, sink=sink)

    total = q.shape[0]
    flops, nbytes = _flops_bytes(
        seqlens, hq, hk, hdim_q, hdim_v, is_causal, total, q.element_size()
    )

    # The model calls the public dispatcher (flash_attn_varlen_func) → asm path.
    # It sends D64 without a sink to another backend, so that case calls the ops.
    via = "public" if hdim_q != 64 or d64_sink else "ops"
    candidates = {
        "asm": lambda: run_kernel(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q,
            scale=scale,
            is_causal=is_causal,
            sink=sink,
            via=via,
        )
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (out, lse), us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err(O)"] = assertAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} O d={hdim_q}x{hdim_v} c={is_causal}",
        )
        ret[f"{name} err(LSE)"] = assertAllclose(
            ref_lse.to(dtypes.fp32),
            lse.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} LSE d={hdim_q}x{hdim_v} c={is_causal}",
        )
    return ret


def _nan_buffer(shape, dtype):
    """NaN-filled buffer, so a read outside the view shows up (fp8: 0x7f, the
    e4m3fn NaN)."""
    if dtype == dtypes.fp8:
        return torch.full(shape, 0x7F, dtype=torch.uint8).view(dtype)
    return torch.full(shape, float("nan"), dtype=dtype)


def _as_view(x, head_pad=0, token_pad=0, base_off=0):
    """Strided copy of `x` in a NaN buffer; odd pads/offset give rows aligned
    only to the element size (2 B for bf16, 1 B for fp8)."""
    t, h, d = x.shape
    row = h * (d + head_pad) + token_pad
    buf = _nan_buffer((base_off + t * row,), x.dtype)
    view = buf.as_strided((t, h, d), (row, d + head_pad, 1), base_off)
    view.copy_(x)
    return view


def _strided_inputs(layout, q, k, v):
    """Views with the same values as q/k/v for one of the strided layouts."""
    t, hq, dq = q.shape
    hk, dv = k.size(1), v.size(2)
    if layout == "kv_split":
        # DeepSeek-style: v is the second half of a [.., k_nope | v] projection.
        kv = _nan_buffer((v.size(0), hk, dv + dv), v.dtype)
        kv[..., dv:] = v
        return q, k, kv[..., dv:]
    if layout == "fused_qkv":
        # One row per token holds every q, k and v head back to back.
        assert k.size(0) == t, "fused_qkv needs equal q and k token counts"
        row = _nan_buffer((t, hq * dq + hk * dq + hk * dv), q.dtype)
        qv = row[:, : hq * dq].view(t, hq, dq)
        kvv = row[:, hq * dq : hq * dq + hk * dq].view(t, hk, dq)
        vv = row[:, hq * dq + hk * dq :].view(t, hk, dv)
        qv.copy_(q)
        kvv.copy_(k)
        vv.copy_(v)
        return qv, kvv, vv
    if layout == "padded":
        return (
            _as_view(q, head_pad=64, token_pad=8),
            _as_view(k, head_pad=16, token_pad=24),
            _as_view(v, head_pad=128),
        )
    if layout == "odd":
        # Odd-element strides and offsets: rows start only element-size aligned.
        return (
            _as_view(q, head_pad=1, token_pad=3),
            _as_view(k, head_pad=1, token_pad=1),
            _as_view(v, head_pad=3, token_pad=1, base_off=1),
        )
    raise ValueError(f"unknown layout {layout!r}")


# Runs per candidate before timing; every run must match the dense result bit
# for bit (an intermittent race shows up as one differing run).
_STRIDED_REPEATS = 5
_STRIDED_VIAS = ["public", "ops"]


def _strided_vias(q, is_causal):
    """The candidates checked bit for bit against the ops run: `public` only when
    flash_attn_varlen_func hands the call to the ASM kernel, i.e. not for the calls
    it offers to FlyDSL first (e.g. D192x128 with AITER_ENABLE_EXPERIMENTAL)."""
    if aiter.ops.mha._prefer_flydsl_over_gfx1250_bf16_varlen_asm(q, is_causal):
        return [v for v in _STRIDED_VIAS if v != "public"]
    return _STRIDED_VIAS


def _same(res, dense):
    return torch.equal(res[0], dense[0]) and torch.equal(res[1], dense[1])


def _record(ret, name, fn, dense, ref, flops, nbytes, msg):
    """Check `fn` against the dense run (repeated) and the reference, then time it."""
    same = all(_same(fn(), dense) for _ in range(_STRIDED_REPEATS))
    assert same, f"{msg} via={name}: differs from dense"
    (out, lse), us = run_perftest(fn)
    ret[f"{name} us"] = us
    ret[f"{name} TFLOPS"] = flops / us / 1e6
    ret[f"{name} TB/s"] = nbytes / us / 1e6
    ret[f"{name} == dense"] = same
    ret[f"{name} err(O)"] = assertAllclose(
        ref[0].to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"{msg} via={name} O",
    )
    ret[f"{name} err(LSE)"] = assertAllclose(
        ref[1].to(dtypes.fp32),
        lse.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"{msg} via={name} LSE",
    )


def _strided_setup(hdim_q, hdim_v, hq, hk, seqlens, is_causal):
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, hdim_q, hdim_v)
    kw = {"scale": 1.0 / math.sqrt(hdim_q), "is_causal": is_causal}
    ref = run_torch(q, k, v, cu, cu, is_causal=is_causal, sink=None)
    dense = run_kernel(q, k, v, cu, cu, max(seqlens), via="ops", **kw)
    flops, nbytes = _flops_bytes(
        seqlens, hq, hk, hdim_q, hdim_v, is_causal, q.size(0), q.element_size()
    )
    return q, k, v, cu, kw, ref, dense, flops, nbytes


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_strided(
    hdim_q, hdim_v, hq, hk, seqlens, is_causal, layout
):
    """D192x128 / D256 with strided q/k/v views."""
    q, k, v, cu, kw, ref, dense, flops, nbytes = _strided_setup(
        hdim_q, hdim_v, hq, hk, seqlens, is_causal
    )
    qs, ks, vs = _strided_inputs(layout, q, k, v)
    candidates = {
        via: functools.partial(
            run_kernel, qs, ks, vs, cu, cu, max(seqlens), via=via, **kw
        )
        for via in _strided_vias(q, is_causal)
    }
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _record(ret, name, fn, dense, ref, flops, nbytes, f"strided {layout}")
    return ret


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_strided_out(
    hdim_q, hdim_v, hq, hk, seqlens, is_causal, width
):
    """D192x128 / D256 writing into a preallocated `out` that is a [..., :hdim_v]
    view of a wider NaN buffer (width hdim_v + 1: rows only 2-byte aligned)."""
    q, k, v, cu, kw, ref, dense, flops, nbytes = _strided_setup(
        hdim_q, hdim_v, hq, hk, seqlens, is_causal
    )
    wides = {
        via: torch.full((q.size(0), hq, width), float("nan"), dtype=q.dtype)
        for via in _strided_vias(q, is_causal)
    }
    candidates = {
        via: functools.partial(
            run_kernel,
            q,
            k,
            v,
            cu,
            cu,
            max(seqlens),
            via=via,
            out=wide[..., :hdim_v],
            **kw,
        )
        for via, wide in wides.items()
    }
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _record(ret, name, fn, dense, ref, flops, nbytes, f"strided out w={width}")
        wide = wides[name]
        in_place = fn()[0].data_ptr() == wide.data_ptr()
        pad_untouched = bool(wide[..., hdim_v:].isnan().all().item())
        assert in_place and pad_untouched, f"w={width} via={name}: out not in place"
        ret[f"{name} out in place"] = in_place
        ret[f"{name} pad untouched"] = pad_untouched
    return ret


def _unsupported_kv(case, k, v):
    """k/v views the D192x128 / D256 kernels cannot address."""
    if case == "k token stride 2^24 B":
        tok = (1 << 24) // k.element_size()
        d = k.size(-1)
        buf = torch.empty((len(k) - 1) * tok + k.size(1) * d, dtype=k.dtype)
        k_far = buf.as_strided(k.shape, (tok, d, 1))
        k_far.copy_(k)
        return k_far, v
    if case == "k/v head stride 0":
        # All kv heads share one head (expanded MQA): pass hk=1 instead.
        return k[:, :1].expand_as(k), v[:, :1].expand_as(v)
    raise ValueError(f"unknown case {case!r}")


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_unsupported_stride(
    hdim_q, hdim_v, case, is_causal
):
    """Unsupported k/v strides: the public path must use another backend, and a
    direct op call must be refused by the C++ entry.

    Not timed: the backend that serves these calls is not the one under test."""
    seqlens, hq, hk = [2], 4, 2
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, hdim_q, hdim_v)
    k, v = _unsupported_kv(case, k, v)
    scale = 1.0 / math.sqrt(hdim_q)
    ref_out, _ = run_torch(q, k, v, cu, cu, is_causal=is_causal, sink=None)

    mha = aiter.ops.mha
    calls = []
    asm = mha._fmha_fwd_with_sink_varlen_asm

    def spy(*args, **kwargs):
        calls.append(1)
        return asm(*args, **kwargs)

    mha._fmha_fwd_with_sink_varlen_asm = spy
    try:
        try:
            aiter.fmha_fwd_with_sink_varlen_asm(
                q, k, v, cu, cu, max(seqlens), scale, is_causal, True
            )
            ops_refused = False
        except RuntimeError as e:  # raised by the C++ stride check
            ops_refused = "strides out of range" in str(e)
        ops_calls = len(calls)
        out, _ = run_kernel(
            q, k, v, cu, cu, max(seqlens), scale=scale, is_causal=is_causal
        )
    finally:
        mha._fmha_fwd_with_sink_varlen_asm = asm
    assert ops_refused, f"{case}: C++ accepted an unsupported stride"
    assert len(calls) == ops_calls, f"{case}: public path called ASM"
    return {
        "gfx": get_gfx(),
        "ops refused": ops_refused,
        "public asm calls": len(calls) - ops_calls,
        "err(O)": assertAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{case} -> other backend, c={is_causal}",
        ),
    }


def _visible_pairs(sq, sk, is_causal):
    """(query, key) pairs attention computes; causal is bottom-right aligned, so
    row i sees min(sk, max(0, i + 1 + sk - sq)) keys."""
    if not is_causal:
        return sq * sk
    return sum(min(sk, max(0, i + 1 + sk - sq)) for i in range(sq))


def _asm_calls(fn):
    """Run fn() with a spy on the bf16 varlen ctypes entry; returns the launches."""
    mha = aiter.ops.mha
    asm = mha._fmha_fwd_with_sink_varlen_asm
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return asm(*args, **kwargs)

    mha._fmha_fwd_with_sink_varlen_asm = spy
    try:
        fn()
    finally:
        mha._fmha_fwd_with_sink_varlen_asm = asm
    return len(calls)


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_qk(
    hdim_q, hdim_v, hq, hk, seqlens_q, seqlens_k, is_causal
):
    """D192x128 / D256 with separate q and k lengths: q < k (chunked prefill),
    q > k (causal rows that see no key: O = 0, LSE = -inf), sequences without keys
    or queries.  Both entries reach the ASM kernel and agree bit for bit (`public`
    is skipped for calls flash_attn_varlen_func offers to FlyDSL first)."""
    torch.manual_seed(0)
    q = torch.randn(sum(seqlens_q), hq, hdim_q, dtype=dtypes.bf16)
    k = torch.randn(sum(seqlens_k), hk, hdim_q, dtype=dtypes.bf16)
    v = torch.randn(sum(seqlens_k), hk, hdim_v, dtype=dtypes.bf16)
    cu_q, cu_k = _cu(seqlens_q), _cu(seqlens_k)
    max_q, max_k = max(seqlens_q), max(seqlens_k)
    kw = {
        "scale": 1.0 / math.sqrt(hdim_q),
        "is_causal": is_causal,
        "max_seqlen_k": max_k,
    }
    ref = run_torch(q, k, v, cu_q, cu_k, is_causal=is_causal, sink=None)
    dense = run_kernel(q, k, v, cu_q, cu_k, max_q, via="ops", **kw)
    pairs = sum(_visible_pairs(a, b, is_causal) for a, b in zip(seqlens_q, seqlens_k))
    flops = 2.0 * hq * pairs * (hdim_q + hdim_v)
    nbytes = (q.numel() + k.numel() + v.numel() + q.size(0) * hq * hdim_v) * 2
    candidates = {
        via: functools.partial(run_kernel, q, k, v, cu_q, cu_k, max_q, via=via, **kw)
        for via in _strided_vias(q, is_causal)
    }
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        assert _asm_calls(fn) == 1, f"via={name} did not reach the ASM kernel"
        _record(ret, name, fn, dense, ref, flops, nbytes, f"q/k lengths c={is_causal}")
    return ret


# (hq, hk, seqlens_q, seqlens_k): one sequence, q < k, q > k, mixed varlen, short
# sequences, no keys at all, a sequence without queries and one without keys.
_QK_SHAPES = [
    (16, 4, [1000], [1000]),
    (8, 8, [768], [2048]),
    (8, 8, [300], [65]),
    (16, 4, [77, 513, 256], [130, 1000, 65]),
    (8, 2, [1, 64, 255, 257], [1, 64, 255, 257]),
    (8, 2, [64, 300], [0, 0]),
    (8, 8, [0, 512, 77], [100, 512, 0]),
]


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_d64_qk(
    hq, hk, seqlens_q, seqlens_k, is_causal, d64_sink
):
    """D64 through the ops wrapper with separate q and k lengths, q > k.  Causal rows
    that see no key must give O = 0 and LSE = -inf (the sink logit with a sink) on
    every element; without a sink the wrapper's -inf sink meets a -inf row maximum
    there."""
    torch.manual_seed(0)
    q = torch.randn(sum(seqlens_q), hq, 64, dtype=dtypes.bf16)
    k = torch.randn(sum(seqlens_k), hk, 64, dtype=dtypes.bf16)
    v = torch.randn(sum(seqlens_k), hk, 64, dtype=dtypes.bf16)
    cu_q, cu_k = _cu(seqlens_q), _cu(seqlens_k)
    sink = _d64_sink(hq) if d64_sink else None
    ref_out, ref_lse = run_torch(q, k, v, cu_q, cu_k, is_causal=is_causal, sink=sink)
    fn = functools.partial(
        run_kernel,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(seqlens_q),
        scale=1.0 / math.sqrt(64),
        is_causal=is_causal,
        sink=sink,
        via="ops",
    )
    (out, lse), us = run_perftest(fn)
    pairs = sum(_visible_pairs(a, b, is_causal) for a, b in zip(seqlens_q, seqlens_k))
    msg = f"D64 q/k lengths c={is_causal} sink={d64_sink}"
    # Rows without a key by position (bottom-right aligned; every k length here is
    # non-zero): a sink makes the reference LSE finite on them too.
    no_key = torch.cat(
        [
            torch.arange(a) < (a - b if is_causal else 0)
            for a, b in zip(seqlens_q, seqlens_k)
        ]
    )
    ret = {
        "gfx": get_gfx(),
        "rows without a key": int(no_key.sum()),
        "asm us": us,
        "asm TFLOPS": 2.0 * hq * pairs * (64 + 64) / us / 1e6,
        "asm err(O)": assertAllclose(
            ref_out.float(), out.float(), rtol=1e-2, atol=1e-2, msg=f"{msg} O"
        ),
        "asm err(LSE)": assertAllclose(
            ref_lse, lse.float(), rtol=1e-2, atol=1e-2, msg=f"{msg} LSE"
        ),
    }
    # assertAllclose tolerates a few bad elements; rows without a key must match on
    # every element
    assert not out[no_key].any(), f"{msg}: O != 0 without keys"
    ok = torch.isclose(lse[no_key], ref_lse[no_key], rtol=1e-2, atol=1e-2)
    assert ok.all(), f"{msg}: LSE differs from the reference without keys"
    return ret


# (hq, hk, seqlens_q, seqlens_k) for test_fmha_fwd_with_sink_varlen_asm_d64_qk: two
# sequences whose q and k offsets differ; causal, the first 44 rows of each see no key.
_D64_QK_SHAPES = [(64, 8, [300, 300], [256, 256])]


# Attention options _select_gfx1250_fmha_fwd_varlen_asm checks the same way for
# both kernels.  The gate tests use one sequence of 2 query and 2 key tokens, so
# max_seqlen_k is 2.  name -> (gate arguments, accepted)
_ASM_GATE_OPTION_CASES = {
    "causal": ({"causal": True}, True),
    "causal, right window 0": ({"causal": True, "window_size_right": 0}, True),
    "left window >= max_seqlen_k": ({"window_size_left": 2}, True),
    "non-causal, right window 0": ({"window_size_right": 0}, False),
    "sliding window": ({"window_size_left": 1}, False),
    "sink tokens": ({"sink_size": 1}, False),
    "dropout": ({"dropout_p": 0.1}, False),
    "soft-cap": ({"logits_soft_cap": 30.0}, False),
    "no query rows": ({"max_seqlen_q": 0}, False),
}


def _asm_gate_args(case):
    """Attention-option arguments of the gate for `case`: non-causal plain
    attention unless the case says otherwise."""
    args = {
        "causal": False,
        "max_seqlen_q": 2,
        "max_seqlen_k": 2,
        "window_size_left": -1,
        "window_size_right": -1,
        "sink_size": 0,
        "dropout_p": 0.0,
        "logits_soft_cap": 0.0,
        "bias": None,
        "alibi_slopes": None,
        "block_table": None,
        "cu_seqlens_q_padded": None,
        "cu_seqlens_k_padded": None,
        "return_softmax": False,
    }
    args.update(_ASM_GATE_OPTION_CASES.get(case, ({}, None))[0])
    return args


# name -> served by the bf16 kernel (_select_gfx1250_fmha_fwd_varlen_asm)
_BF16_GATE_CASES = {
    "D192x128": True,
    "D128": True,
    "D64 with sink logits": True,
    "D64 without sink logits": False,
    "D128 with sink logits": False,
    "D192x128 with sink logits": False,
    "D256": True,
    "D256 with sink logits": False,
    # Ahead of maybe_contiguous (the pre-router): a non-dense last dim is fixed
    # later for q, but `out` is used as given.
    "D256, q last-dim stride 2, ahead of maybe_contiguous": True,
    "D256, out last-dim stride 2, ahead of maybe_contiguous": False,
    "head dim 96": False,
    "fp32 descales": False,
    "zero kv heads": False,
    **{c: ok for c, (_, ok) in _ASM_GATE_OPTION_CASES.items()},
}


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_gate(case):
    """The gate every dispatch site asks must pick the bf16 kernel for exactly what
    it serves.  Option cases use D192x128.  Not timed: nothing is launched."""
    if case.startswith("D64"):
        d, dv = 64, 64
    elif case.startswith("D128"):
        d, dv = 128, 128
    elif case.startswith("D256"):
        d, dv = 256, 256
    elif case == "head dim 96":
        d, dv = 96, 96
    else:
        d, dv = 192, 128
    q, k, v, _ = make_varlen_packed([2], 4, 4, d, dv)
    if case == "zero kv heads":
        k, v = k[:, :0], v[:, :0]
    out = None
    late = "ahead of maybe_contiguous" in case
    if late:
        # same shape, last-dim stride 2
        strided2 = lambda x, dl: x.new_empty(x.size(0), x.size(1), dl, 2)[..., 0]
        if ", q last-dim" in case:
            q = strided2(q, d)
        else:
            out = strided2(q, dv)
    sink = torch.zeros(4, dtype=dtypes.fp32) if "with sink" in case else None
    descale = torch.ones(1, dtype=dtypes.fp32) if case == "fp32 descales" else None
    mha = aiter.ops.mha
    routed = mha._GFX1250_VARLEN_ASM_BF16 == mha._select_gfx1250_fmha_fwd_varlen_asm(
        q,
        k,
        v,
        out,
        dense_last_dim_later=late,
        sink_ptr=sink,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
        **_asm_gate_args(case),
    )
    assert routed == _BF16_GATE_CASES[case], f"{case}: gate returned {routed}"
    return {"gfx": get_gfx(), "routed to asm": routed}


# (hdim_q, hdim_v, is_causal, requires_grad).  A call that needs gradients goes to
# the ASM forward only for head dims the gfx1250 ASM backward has (not D256).
# Causal, as non-causal D64 / D128 are offered to FlyDSL first.
_GRAD_ROUTE_CASES = [
    (128, 128, True, False),
    (128, 128, True, True),
    (192, 128, True, True),
    (256, 256, True, False),
    (256, 256, True, True),
    (256, 256, False, True),
]


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_grad_route(hdim_q, hdim_v, is_causal, grad):
    """ASM forward launches of a flash_attn_varlen_func call with and without
    gradients; with gradients the backward must run.  Not timed."""
    q, k, v, cu = make_varlen_packed([300, 77], 4, 4, hdim_q, hdim_v)
    for x in (q, k, v):
        x.requires_grad_(grad)
    mha = aiter.ops.mha
    has_bwd = (hdim_q, hdim_v) in mha._GFX1250_BWD_HDIMS
    flydsl_first = mha._prefer_flydsl_over_gfx1250_bf16_varlen_asm(q, is_causal)
    if not flydsl_first and (has_bwd or not grad):
        expect = 1  # the pre-router hands it to the ASM forward
    elif grad:
        # FlyDSL takes no call that needs gradients: Triton with ENABLE_CK=0, else
        # FlashAttnVarlenFunc, whose forward is the ASM
        expect = 1 if mha.ENABLE_CK else 0
    else:
        expect = None  # offered to FlyDSL first: not checked
    res = []
    with torch.enable_grad():
        calls = _asm_calls(
            lambda: res.append(
                run_kernel(
                    q, k, v, cu, cu, 300, scale=hdim_q**-0.5, is_causal=is_causal
                )
            )
        )
        msg = f"D{hdim_q}x{hdim_v} causal={is_causal} grad={grad}"
        assert expect is None or calls == expect, f"{msg}: {calls} ASM launches"
        ret = {"gfx": get_gfx(), "ASM forward launches": calls}
        if grad:
            res[0][0].float().sum().backward()
            ok = all(
                x.grad is not None and bool(torch.isfinite(x.grad).all())
                for x in (q, k, v)
            )
            assert ok, f"{msg}: backward gave no or non-finite gradients"
            ret["backward ok"] = ok
    return ret


_STRIDED_LAYOUTS = [
    "kv_split",
    "fused_qkv",
    "padded",
    "odd",
]
# out buffer width minus hdim_v for the strided-out table (1: 2-byte-aligned rows)
_STRIDED_OUT_PADS = [64, 1]
# (hdim_q, hdim_v) of the kernels that take strided q/k/v/out
_STRIDED_HDIMS = [(192, 128), (256, 256)]
# (hq, hk, seqlens): GQA, unaligned and mixed lengths
_STRIDED_SHAPES = [
    (16, 4, [129, 1000, 333]),
    (32, 32, [2048]),
]


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_perf(
    hdim_q, hdim_v, hq, hk, seqlens, is_causal, init
):
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, hdim_q, hdim_v, init=init)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(hdim_q)
    sink = _d64_sink(hq) if hdim_q == 64 else None

    total = q.shape[0]
    flops, nbytes = _flops_bytes(
        seqlens, hq, hk, hdim_q, hdim_v, is_causal, total, q.element_size()
    )

    # asm: the ASM kernel itself (ops entry); public: what flash_attn_varlen_func
    # runs (FlyDSL for the calls _prefer_flydsl_over_gfx1250_bf16_varlen_asm hands
    # it); flydsl: FlyDSL's gfx1250 kernel called directly, when it takes the call.
    run = functools.partial(
        run_kernel,
        q,
        k,
        v,
        cu,
        cu,
        max_seqlen_q,
        scale=scale,
        is_causal=is_causal,
        sink=sink,
    )
    candidates = {
        "asm": functools.partial(run, via="ops"),
        "public": functools.partial(run, via="public"),
    }
    if flydsl_flash_attn_varlen_fwd is not None and sink is None:
        fly = functools.partial(
            flydsl_flash_attn_varlen_fwd,
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen_q,
            max_seqlen_q,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
        )
        if fly() is not None:  # None: FlyDSL does not take this configuration
            candidates["flydsl"] = fly

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


# ---------------------------------------------------------------------------
# FP8 D192x128 (fmha_fwd_fp8_varlen_asm): fp8 e4m3fn q/k/v with per-tensor
# fp32 descales, bf16 output.
#
# Candidates (all must reach the ASM kernel and agree bit for bit):
#   public -> aiter.flash_attn_varlen_fp8_pertensor_func
#   varlen -> aiter.flash_attn_varlen_func(..., q/k/v_descale=...)  (with LSE)
#   ops    -> aiter.fmha_fwd_fp8_varlen_asm
#
# Reference: fp32 on the dequantized fp8 inputs, emulating the kernel's online
# softmax block by block (as the poc host's pemu=2): keys in blocks of the
# kernel's KV tile, query rows in groups of one wave; the first block, or a block
# whose max exceeds the running max by more than FP8_TAU (exp2 units) in any row
# of the group, moves the whole group's max and rescales; P = exp(S - running max)
# is rounded to e4m3 and the row sum is the sum of the rounded P (so is the LSE).
# What is left between kernel and reference is the summation order and the
# reduced precision of S inside the fp8 WMMA, which now and then puts a P on the
# other side of an e4m3 rounding boundary.
#   * O: an element is "bad" past FP8_ATOL + FP8_RTOL * |ref|.  One such flip can
#     shift a whole row with few visible keys, so the check counts rows (a token
#     of a head) holding bad elements and allows a few per thousand; a layout /
#     mask / rescale bug breaks whole 16-row tiles or every head.
#   * cos >= FP8_COS_MIN over the whole output.
#   * LSE: within FP8_LSE_EMU_TOL of the emulated one, for all but as many rows as
#     O may have bad ones (a flipped P moves the sum of a short row); rows that see
#     no key are -inf.  The distance to the exact log-sum-exp is only reported:
#     it is the rounding of P, which has no fixed bound -- P far below the row's
#     max fall into e4m3's subnormal range or round to zero, and a row whose
#     weight sits mostly in such P keeps a correspondingly smaller sum.
# ---------------------------------------------------------------------------

try:  # FlyDSL's gfx1250 fp8 kernel, a reference point in the fp8 perf table
    from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_fwd
except Exception:  # noqa: BLE001 -- optional; the table just goes without it
    flydsl_flash_attn_varlen_fwd = None

FP8_ATOL, FP8_RTOL = 4e-2, 1e-2
FP8_COS_MIN = 0.9999
FP8_COS_MIN_EXACT = 0.999  # against the exact softmax: the rounding of P alone moves O
FP8_LSE_EMU_TOL = 1e-2
FP8_TAU = 8.0  # the kernel's lazy-rescale threshold (log2 units)
_FP8_VIAS = ["public", "varlen", "ops"]


def _cu(seqlens):
    return torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=dtypes.i32
    )


def _quant(x):
    """Per-tensor fp8 quantization; an empty tensor (no keys at all) gets descale 1."""
    if x.numel() == 0:
        return x.to(dtypes.fp8), torch.ones(1, dtype=dtypes.fp32)
    return per_tensor_quant(x, quant_dtype=dtypes.fp8)


def make_fp8_varlen(seqlens_q, seqlens_k, hq, hk, seed=0):
    """Packed fp8 q/k/v quantized from randn with per-tensor descales."""
    torch.manual_seed(seed)
    tq, tk = sum(seqlens_q), sum(seqlens_k)
    q8, qd = _quant(torch.randn(tq, hq, 192, dtype=dtypes.bf16))
    k8, kd = _quant(torch.randn(tk, hk, 192, dtype=dtypes.bf16))
    v8, vd = _quant(torch.randn(tk, hk, 128, dtype=dtypes.bf16))
    return q8, k8, v8, qd, kd, vd, _cu(seqlens_q), _cu(seqlens_k)


def _emulate_block(s, scale_log2e, causal_vis, vf):
    """Kernel-order online softmax for one sequence: s (h, sq, sk) natural-log
    scores with -inf masked, vf (h, sk, d) values.  Returns O (sq, h, d), LSE of
    the rounded P (h, sq)."""
    h, sq, sk = s.shape
    g = (sq + 63) // 64
    pad = g * 64 - sq
    s = torch.nn.functional.pad(s, (0, 0, 0, pad), value=float("-inf"))
    m = torch.full((h, g * 64), float("-inf"))
    l = torch.zeros(h, g * 64)
    o = torch.zeros(h, g * 64, vf.size(-1))
    tau = FP8_TAU * math.log(2.0)  # FP8_TAU (exp2 units) as natural log
    for k0 in range(0, sk, 64):
        sb = s[:, :, k0 : k0 + 64]
        bm = sb.amax(-1)
        exceed = (bm > m + tau) & torch.isfinite(bm)
        upd = (
            exceed.view(h, g, 64)
            .any(-1, keepdim=True)
            .expand(h, g, 64)
            .reshape(h, g * 64)
        )
        if k0 == 0:
            upd = torch.ones_like(upd)
        m_new = torch.where(upd, torch.maximum(m, bm), m)
        alpha = torch.exp(m - m_new).nan_to_num(0.0)
        alpha = torch.where(torch.isfinite(m), alpha, torch.zeros_like(alpha))
        l, o = l * alpha, o * alpha[..., None]
        m = m_new
        p = torch.exp(sb - m[..., None]).nan_to_num(0.0)
        p = (
            torch.where(torch.isfinite(m)[..., None], p, torch.zeros_like(p))
            .to(dtypes.fp8)
            .float()
        )
        l = l + p.sum(-1)
        o = o + torch.einsum("hqk,hkd->hqd", p, vf[:, k0 : k0 + 64])
    o = (o / l[..., None]).nan_to_num(0.0)
    lse = torch.where(l > 0, m + torch.log(l), torch.full_like(l, float("-inf")))
    return o[:, :sq].transpose(0, 1), lse[:, :sq]


def run_torch_fp8(q8, k8, v8, qd, kd, vd, cu_q, cu_k, *, is_causal, exact=False):
    """Reference on the dequantized inputs: out (total_q, hq, 128) fp32, lse
    (total_q, hq) fp32, and the exact LSE (total_q, hq).  Bottom-right causal;
    rows without keys get O = 0, LSE = -inf.  Emulates the kernel's online softmax
    (see above); exact=True gives the exact softmax instead (O and LSE)."""
    total_q, hq, d = q8.shape
    hk = k8.size(1)
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros(total_q, hq, v8.size(-1), dtype=dtypes.fp32)
    lse = torch.full((total_q, hq), float("-inf"), dtype=dtypes.fp32)
    lse_exact = torch.full((total_q, hq), float("-inf"), dtype=dtypes.fp32)
    cuq, cuk = cu_q.tolist(), cu_k.tolist()
    for b in range(len(cuq) - 1):
        q0, q1, k0, k1 = cuq[b], cuq[b + 1], cuk[b], cuk[b + 1]
        sq, sk = q1 - q0, k1 - k0
        if sq == 0 or sk == 0:
            continue
        qf = q8[q0:q1].float() * qd
        kf = (k8[k0:k1].float() * kd).repeat_interleave(hq // hk, dim=1)
        vf = (v8[k0:k1].float() * vd).repeat_interleave(hq // hk, dim=1)
        s = torch.einsum("qhd,khd->hqk", qf, kf) * scale
        if is_causal:
            row = torch.arange(sq)[:, None]
            col = torch.arange(sk)[None, :]
            s = s.masked_fill((col > row + (sk - sq))[None], float("-inf"))
        le = torch.logsumexp(s, dim=-1)  # (hq, sq)
        lse_exact[q0:q1] = le.transpose(0, 1)
        if exact:
            p = torch.exp(s - le.unsqueeze(-1)).nan_to_num(0.0)
            out[q0:q1] = torch.einsum("hqk,khd->qhd", p, vf)
            lse[q0:q1] = le.transpose(0, 1)
        else:
            ob, lb = _emulate_block(s, scale, None, vf.transpose(0, 1))
            out[q0:q1] = ob
            lse[q0:q1] = lb.transpose(0, 1)
    return out, lse, lse_exact


def run_kernel_fp8(q8, k8, v8, qd, kd, vd, cu_q, cu_k, max_q, max_k, *, is_causal, via):
    """Returns (out, lse (total_q, nheads)); lse is None for `public`, which
    does not return it."""
    scale = 1.0 / math.sqrt(q8.size(-1))
    if via == "public":
        out = aiter.flash_attn_varlen_fp8_pertensor_func(
            q8,
            k8,
            v8,
            qd,
            kd,
            vd,
            cu_q,
            cu_k,
            max_q,
            max_k,
            causal=is_causal,
            softmax_scale=scale,
        )
        return out, None
    if via == "varlen":
        r = aiter.flash_attn_varlen_func(
            q8,
            k8,
            v8,
            cu_q,
            cu_k,
            max_q,
            max_k,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
            q_descale=qd,
            k_descale=kd,
            v_descale=vd,
        )
        return r[0], r[1].transpose(0, 1)
    if via == "ops":
        out, lse = aiter.fmha_fwd_fp8_varlen_asm(
            q8,
            k8,
            v8,
            qd,
            kd,
            vd,
            cu_q,
            cu_k,
            max_q,
            max_k,
            softmax_scale=scale,
            is_causal=is_causal,
            return_lse=True,
        )
        return out, lse.squeeze(-1)
    raise ValueError(f"unknown via={via!r}")


def _fp8_asm_calls(fn):
    """Run fn() with a spy on the fp8 ctypes entry; returns the number of launches."""
    mha = aiter.ops.mha
    asm = mha._fmha_fwd_fp8_varlen_asm
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return asm(*args, **kwargs)

    mha._fmha_fwd_fp8_varlen_asm = spy
    try:
        fn()
    finally:
        mha._fmha_fwd_fp8_varlen_asm = asm
    return len(calls)


def _fp8_flops_bytes(seqlens_q, seqlens_k, hq, hk, is_causal):
    """2 GEMMs over the visible (q, k) pairs (bottom-right causal); fp8 q/k/v
    read once, bf16 O and fp32 LSE written once."""

    def pairs(sq, sk):
        if not is_causal:
            return sq * sk
        if sk >= sq:
            return sq * (sk - sq) + sq * (sq + 1) / 2
        return sk * (sk + 1) / 2

    flops = sum(
        2.0 * hq * pairs(a, b) * (192 + 128) for a, b in zip(seqlens_q, seqlens_k)
    )
    tq, tk = sum(seqlens_q), sum(seqlens_k)
    nbytes = tq * hq * 192 + tk * hk * (192 + 128) + tq * hq * (128 * 2 + 4)
    return flops, nbytes


def check_fp8_against_ref(out, lse, ref, msg, exact=False):
    """Asserts the fp8 tolerances above against ref = (O, emulated LSE, exact LSE);
    returns the error summary columns.  exact: ref's O is the exact softmax (cos
    against FP8_COS_MIN_EXACT, as the rounding of P alone moves O from it) and
    there is no emulated LSE to check."""
    o, r = out.float(), ref[0]
    assert not o.isnan().any(), f"{msg}: NaN in output"
    bad = (o - r).abs() > FP8_ATOL + FP8_RTOL * r.abs()
    rows = bad.shape[0] * bad.shape[1]
    bad_rows = int(bad.any(dim=-1).sum().item())
    if r.abs().max() == 0:  # no row sees a key: the output must be exactly 0
        cos = 1.0 if o.abs().max() == 0 else 0.0
    else:
        cos = torch.nn.functional.cosine_similarity(
            o.flatten(), r.flatten(), dim=0
        ).item()
    res = {"max_abs": (o - r).abs().max().item(), "bad rows": bad_rows, "cos": cos}
    assert bad_rows <= max(4, rows // 1000), f"{msg}: {bad_rows} of {rows} rows off"
    assert cos >= (FP8_COS_MIN_EXACT if exact else FP8_COS_MIN), f"{msg}: cos {cos}"
    if lse is not None:
        fin = torch.isfinite(ref[2])
        assert torch.equal(torch.isfinite(lse), fin), f"{msg}: LSE finite mask differs"
        # a row that sees no key must merge with weight 0: -inf, never NaN / +inf
        assert bool(
            (lse[~fin] == float("-inf")).all()
        ), f"{msg}: LSE of a row without keys is not -inf"
        # distance to the exact log-sum-exp: reported, not checked (see above)
        res["lse err"] = (
            (lse[fin] - ref[2][fin]).abs().max().item() if fin.any() else 0.0
        )
        if exact:
            return res
        d = (lse[fin] - ref[1][fin]).abs()
        off = int((d > FP8_LSE_EMU_TOL).sum().item())
        assert off <= max(
            4, rows // 1000
        ), f"{msg}: LSE of {off} rows off from the emulated one (max {d.max().item()})"
        res["lse vs emu"] = d.max().item() if fin.any() else 0.0
        res["lse rows off"] = off
    return res


def _record_fp8(ret, name, fn, ref, flops, nbytes, msg):
    """Time one candidate and add its us / TFLOPS / TB/s / err columns."""
    (out, lse), us = run_perftest(fn)
    ret[f"{name} us"] = us
    ret[f"{name} TFLOPS"] = flops / us / 1e6
    ret[f"{name} TB/s"] = nbytes / us / 1e6
    ret[f"{name} err"] = checkAllclose(
        ref[0], out.to(dtypes.fp32), rtol=FP8_RTOL, atol=FP8_ATOL, msg=f"{msg} {name}"
    )
    return out, lse


@benchmark()
def test_fmha_fwd_fp8_varlen_asm(hq, hk, seqlens_q, seqlens_k, is_causal, layout):
    """FP8 D192x128: every entry point on dense or strided q/k/v vs the reference;
    the strided runs must match the dense one bit for bit."""
    q8, k8, v8, qd, kd, vd, cu_q, cu_k = make_fp8_varlen(seqlens_q, seqlens_k, hq, hk)
    max_q, max_k = max(seqlens_q), max(seqlens_k)
    ref = run_torch_fp8(q8, k8, v8, qd, kd, vd, cu_q, cu_k, is_causal=is_causal)
    dense = run_kernel_fp8(
        q8, k8, v8, qd, kd, vd, cu_q, cu_k, max_q, max_k, is_causal=is_causal, via="ops"
    )
    if layout != "dense":
        q8, k8, v8 = _strided_inputs(layout, q8, k8, v8)
    flops, nbytes = _fp8_flops_bytes(seqlens_q, seqlens_k, hq, hk, is_causal)
    msg = f"fp8 hq={hq} hk={hk} q={seqlens_q} k={seqlens_k} c={is_causal} {layout}"

    candidates = {
        via: functools.partial(
            run_kernel_fp8,
            q8,
            k8,
            v8,
            qd,
            kd,
            vd,
            cu_q,
            cu_k,
            max_q,
            max_k,
            is_causal=is_causal,
            via=via,
        )
        for via in _FP8_VIAS
    }
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        assert _fp8_asm_calls(fn) == 1, f"{msg} via={name}: not one ASM launch"
        for _ in range(
            _STRIDED_REPEATS
        ):  # an intermittent race shows up as one differing run
            out, lse = fn()
            assert torch.equal(out, dense[0]), f"{msg} via={name}: O differs from dense"
            assert lse is None or torch.equal(
                lse, dense[1]
            ), f"{msg} via={name}: LSE differs"
        _record_fp8(ret, name, fn, ref, flops, nbytes, msg)
    ret.update(check_fp8_against_ref(*dense, ref, msg))
    return ret


def merge_attn_states(o_a, lse_a, o_b, lse_b):
    """Combine two partial attentions over disjoint key sets (vLLM
    merge_attn_states math, fp32).  o: (t, h, d); lse: (t, h)."""
    m = torch.maximum(lse_a, lse_b)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w_a, w_b = torch.exp(lse_a - m), torch.exp(lse_b - m)
    s = w_a + w_b
    o = (o_a.float() * w_a[..., None] + o_b.float() * w_b[..., None]) / s[..., None]
    return o.nan_to_num(0.0), m + torch.log(s)


def _gather_tokens(x, cu, lo, hi):
    """Pack tokens [start + lo[b], start + hi[b]) of every sequence of x."""
    idx = [torch.arange(s + a, s + b) for s, a, b in zip(cu.tolist()[:-1], lo, hi)]
    return x.index_select(0, torch.cat(idx).to(torch.long))


def _kernel_part(q8, k8, v8, qd, kd, vd, cu_q, cu_k, max_q, max_k, causal):
    return run_kernel_fp8(
        q8, k8, v8, qd, kd, vd, cu_q, cu_k, max_q, max_k, is_causal=causal, via="ops"
    )


def _reference_part(q8, k8, v8, qd, kd, vd, cu_q, cu_k, max_q, max_k, causal):
    o, lse, _ = run_torch_fp8(q8, k8, v8, qd, kd, vd, cu_q, cu_k, is_causal=causal)
    return o, lse


def run_chunked_fp8(
    q8,
    k8,
    v8,
    qd,
    kd,
    vd,
    cu_q,
    cu_k,
    prefix_lens,
    chunk_lens,
    n_ctx,
    part=_kernel_part,
):
    """MLA-style chunked prefill: the cached prefix in n_ctx non-causal pieces
    plus the new tokens causally over themselves, merged with their LSEs.  A
    piece may hold no key of a sequence; that partial result is O = 0, LSE = -inf.
    part=_reference_part runs the same steps on the reference."""
    max_q = max(chunk_lens)
    ends = [p + c for p, c in zip(prefix_lens, chunk_lens)]
    o, lse = part(
        q8,
        _gather_tokens(k8, cu_k, prefix_lens, ends),
        _gather_tokens(v8, cu_k, prefix_lens, ends),
        qd,
        kd,
        vd,
        cu_q,
        cu_q,
        max_q,
        max_q,
        True,
    )
    for i in range(n_ctx):
        lo = [p * i // n_ctx for p in prefix_lens]
        hi = [p * (i + 1) // n_ctx for p in prefix_lens]
        lens = [b - a for a, b in zip(lo, hi)]
        o_p, lse_p = part(
            q8,
            _gather_tokens(k8, cu_k, lo, hi),
            _gather_tokens(v8, cu_k, lo, hi),
            qd,
            kd,
            vd,
            cu_q,
            _cu(lens),
            max_q,
            max(max(lens), 1),
            False,
        )
        o, lse = merge_attn_states(o, lse, o_p, lse_p)
    if part is _kernel_part:
        o = o.to(dtypes.bf16)  # the model keeps the merged output in bf16
    return o, lse


@benchmark()
def test_fmha_fwd_fp8_varlen_asm_chunked(hq, hk, prefix_lens, chunk_lens, n_ctx):
    """FP8 chunked prefill: one causal call (one_call) vs prefix pieces + new tokens
    merged by LSE (merged, the merge in torch so its us is not a kernel time).
    exact-prefix (a check, not a candidate) merges the kernel's new-token part with
    an exact fp32 prefix part, as when the other part comes from an exact-LSE backend.
    References: one_call against the emulated one-call softmax, merged against the
    same pieces and merge done on the reference, exact-prefix against the exact
    softmax."""
    seqlens_k = [p + c for p, c in zip(prefix_lens, chunk_lens)]
    args = make_fp8_varlen(chunk_lens, seqlens_k, hq, hk)
    q8, k8, v8, qd, kd, vd, cu_q, cu_k = args
    ref_one = run_torch_fp8(*args, is_causal=True)
    ref_exact = run_torch_fp8(*args, is_causal=True, exact=True)
    ref_merged = (
        *run_chunked_fp8(*args, prefix_lens, chunk_lens, n_ctx, part=_reference_part),
        ref_one[2],
    )
    refs = {"one_call": ref_one, "merged": ref_merged}
    flops, nbytes = _fp8_flops_bytes(chunk_lens, seqlens_k, hq, hk, True)
    msg = f"fp8 chunked hq={hq} prefix={prefix_lens} chunk={chunk_lens} n_ctx={n_ctx}"

    candidates = {
        "one_call": functools.partial(
            run_kernel_fp8,
            *args,
            max(chunk_lens),
            max(seqlens_k),
            is_causal=True,
            via="ops",
        ),
        "merged": functools.partial(
            run_chunked_fp8, *args, prefix_lens, chunk_lens, n_ctx
        ),
    }
    ret = {"gfx": get_gfx()}
    res = {}
    for name, fn in candidates.items():
        res[name] = _record_fp8(ret, name, fn, refs[name], flops, nbytes, msg)
        for key, val in check_fp8_against_ref(
            *res[name], refs[name], f"{msg} {name}"
        ).items():
            ret[f"{name} {key}"] = val
    ret["merged vs one_call max_abs"] = (
        (res["merged"][0].float() - res["one_call"][0].float()).abs().max().item()
    )

    zeros = [0] * len(prefix_lens)
    o_new, lse_new = run_kernel_fp8(
        q8,
        _gather_tokens(k8, cu_k, prefix_lens, seqlens_k),
        _gather_tokens(v8, cu_k, prefix_lens, seqlens_k),
        qd,
        kd,
        vd,
        cu_q,
        cu_q,
        max(chunk_lens),
        max(chunk_lens),
        is_causal=True,
        via="ops",
    )
    o_pre, lse_pre, _ = run_torch_fp8(
        q8,
        _gather_tokens(k8, cu_k, zeros, prefix_lens),
        _gather_tokens(v8, cu_k, zeros, prefix_lens),
        qd,
        kd,
        vd,
        cu_q,
        _cu(prefix_lens),
        is_causal=False,
        exact=True,
    )
    mixed = merge_attn_states(o_new, lse_new, o_pre, lse_pre)
    for key, val in check_fp8_against_ref(
        *mixed, ref_exact, f"{msg} exact-prefix", exact=True
    ).items():
        ret[f"exact-prefix {key}"] = val
    return ret


# name -> served by the fp8 kernel (_select_gfx1250_fmha_fwd_varlen_asm)
_FP8_GATE_CASES = {
    "supported": True,
    "per-head descale": False,
    "fp64 descale": False,
    "no descale": False,
    "head dim 128": False,
    "bf16 q": False,
    "k token stride 2^24 B": False,
    "fp32 out": False,
    "sink logits": False,
    "descale on cpu": False,
    "zero kv heads": False,
    **{c: ok for c, (_, ok) in _ASM_GATE_OPTION_CASES.items()},
}


@benchmark()
def test_fmha_fwd_fp8_varlen_asm_gate(case):
    """The gate every dispatch site asks must pick the fp8 kernel for exactly what
    it serves.  Not timed: nothing is launched."""
    q8, k8, v8, qd, kd, vd, _, _ = make_fp8_varlen([2], [2], 4, 4)
    out, sink = None, None
    if case == "per-head descale":
        qd = qd.expand(4).contiguous()
    elif case == "fp64 descale":
        qd = qd.double()
    elif case == "no descale":
        qd = None
    elif case == "head dim 128":
        q8, k8 = q8[..., :128], k8[..., :128]
    elif case == "bf16 q":
        q8 = q8.to(dtypes.bf16)
    elif case == "k token stride 2^24 B":
        far = torch.empty((1 << 24) + 4 * 192, dtype=k8.dtype)
        k8 = far.as_strided(k8.shape, (1 << 24, 192, 1))
    elif case == "fp32 out":
        out = torch.empty(2, 4, 128, dtype=dtypes.fp32)
    elif case == "sink logits":
        sink = torch.zeros(4, dtype=dtypes.fp32)
    elif case == "descale on cpu":
        qd = qd.cpu()
    elif case == "zero kv heads":
        k8, v8 = k8[:, :0], v8[:, :0]
    mha = aiter.ops.mha
    routed = mha._GFX1250_VARLEN_ASM_FP8 == mha._select_gfx1250_fmha_fwd_varlen_asm(
        q8,
        k8,
        v8,
        out,
        sink_ptr=sink,
        q_descale=qd,
        k_descale=kd,
        v_descale=vd,
        **_asm_gate_args(case),
    )
    assert routed == _FP8_GATE_CASES[case], f"{case}: gate returned {routed}"
    return {"gfx": get_gfx(), "routed to asm": routed}


# name -> the C++ entry's error for a direct call with that operand
_FP8_OPS_REFUSED_CASES = {
    "descale on cpu": "must be on the same GPU as q",
    "zero kv heads": "head counts must be positive",
}


@benchmark()
def test_fmha_fwd_fp8_varlen_asm_ops_refused(case):
    """A direct call with an operand the kernel cannot use must be refused by the
    C++ entry rather than launched.  Not timed: nothing is launched."""
    q8, k8, v8, qd, kd, vd, cu_q, cu_k = make_fp8_varlen([2], [2], 4, 4)
    if case == "descale on cpu":
        qd = qd.cpu()
    elif case == "zero kv heads":
        k8, v8 = k8[:, :0], v8[:, :0]
    try:
        aiter.fmha_fwd_fp8_varlen_asm(q8, k8, v8, qd, kd, vd, cu_q, cu_k, 2, 2)
        refused = False
    except RuntimeError as e:
        refused = _FP8_OPS_REFUSED_CASES[case] in str(e)
    assert refused, f"{case}: the C++ entry did not refuse it"
    return {"gfx": get_gfx(), "ops refused": refused}


@benchmark()
def test_fmha_fwd_fp8_varlen_asm_perf(batch, seqlen_q, seqlen_k, hq, hk, is_causal):
    """FP8 perf only (the O(s^2) reference is infeasible at these lengths).

    fp8 asm is the kernel under test (public entry point).  For comparison:
    bf16 asm -- aiter.flash_attn_varlen_func on the same values in bf16 (the bf16
    D192x128 ASM on gfx1250); flydsl fp8 -- FlyDSL's gfx1250 fp8 kernel on the same
    fp8 values and descales.  TFLOPS use the same FLOP count for all three; TB/s
    the fp8 traffic."""
    seqlens_q, seqlens_k = [seqlen_q] * batch, [seqlen_k] * batch
    args = make_fp8_varlen(seqlens_q, seqlens_k, hq, hk)
    q8, k8, v8, qd, kd, vd, cu_q, cu_k = args
    scale = 1.0 / math.sqrt(192)
    flops, nbytes = _fp8_flops_bytes(seqlens_q, seqlens_k, hq, hk, is_causal)
    q16, k16, v16 = (
        (x.float() * d).to(dtypes.bf16) for x, d in ((q8, qd), (k8, kd), (v8, vd))
    )
    candidates = {
        "fp8 asm": functools.partial(
            run_kernel_fp8, *args, seqlen_q, seqlen_k, is_causal=is_causal, via="public"
        ),
        "bf16 asm": functools.partial(
            aiter.flash_attn_varlen_func,
            q16,
            k16,
            v16,
            cu_q,
            cu_k,
            seqlen_q,
            seqlen_k,
            softmax_scale=scale,
            causal=is_causal,
        ),
    }
    if flydsl_flash_attn_varlen_fwd is not None:
        fly = functools.partial(
            flydsl_flash_attn_varlen_fwd,
            q8,
            k8,
            v8,
            cu_q,
            cu_k,
            seqlen_q,
            seqlen_k,
            softmax_scale=scale,
            causal=is_causal,
            q_descale=qd,
            k_descale=kd,
            v_descale=vd,
        )
        if fly() is not None:  # None: FlyDSL does not take this configuration
            candidates["flydsl fp8"] = fly
    ret = {"gfx": get_gfx()}
    assert (
        _fp8_asm_calls(candidates["fp8 asm"]) == 1
    ), "fp8 asm did not reach the ASM kernel"
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


# (hq, hk, seqlens_q, seqlens_k): one sequence, GQA, q < k (chunked prefill),
# q > k (causal rows that see no key), mixed varlen, short sequences.
_FP8_CORRECTNESS_SHAPES = [
    (16, 16, [1024], [1024]),
    (16, 4, [1000], [1000]),
    (8, 8, [768], [2048]),
    (8, 8, [300], [65]),
    (16, 4, [77, 513, 256], [130, 1000, 65]),
    (8, 2, [1, 64, 255, 257], [1, 64, 255, 257]),
    (32, 32, [2048, 2048], [2048, 2048]),
    (
        8,
        2,
        [64, 300],
        [0, 0],
    ),  # no keys at all (a chunked-prefill piece no sequence reaches)
    (8, 8, [0, 512, 77], [100, 512, 0]),  # a sequence without queries, one without keys
]
# dense plus the strided layouts above (fused_qkv needs equal q and k lengths)
_FP8_LAYOUTS = ["dense"] + _STRIDED_LAYOUTS
# (hq, hk, prefix_lens, chunk_lens, n_ctx); prefix 0 leaves a sequence without
# keys in every prefix piece
_FP8_CHUNKED_SHAPES = [
    (16, 16, [2048], [512], 1),
    (16, 16, [3000], [1000], 3),
    (8, 2, [1000, 4096, 77], [300, 1024, 129], 2),
    (8, 2, [0, 1500, 77], [256, 300, 129], 2),
    (8, 2, [0, 0], [256, 129], 1),  # the prefix piece holds no key of any sequence
    (16, 16, [6144], [2048], 2),
]
# (batch, seqlen_q, seqlen_k, hq, hk); 96 heads with long sequences is the target
# (1 x 12288 / 14336 the typical shape), the rest for reference
_FP8_PERF_SHAPES = [
    (1, 12288, 14336, 96, 96),
    (1, 16384, 16384, 96, 96),
    (1, 2048, 14336, 96, 96),
    (1, 16384, 16384, 32, 32),
    (1, 16384, 16384, 64, 4),
    (4, 4096, 4096, 128, 128),
    (12, 1024, 1024, 128, 128),
]


def summarize(name, rows):
    aiter.logger.info(
        "fmha_fwd_with_sink_varlen_asm %s summary (markdown):\n%s",
        name,
        pd.DataFrame(rows).to_markdown(index=False),
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fmha_fwd_with_sink_varlen_asm unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--head_dim",
        type=int,
        nargs="*",
        choices=[64, 128, 192, 256],
        default=[64, 128, 192, 256],
        help="qk head dim(s) to test; 192 selects the D192x128 kernels "
        "(default: 64 128 192 256)",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="causal mode(s): 0=non-causal 1=causal (default: 0 1)",
    )
    parser.add_argument(
        "--init",
        type=str,
        nargs="*",
        choices=["randn", "const0.25"],
        default=["randn", "const0.25"],
        help="q/k/v init pattern(s) (default: randn const0.25); bf16 only",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="*",
        choices=["bf16", "fp8"],
        default=["bf16", "fp8"],
        help="q/k/v dtype(s): bf16 (fmha_fwd_with_sink_varlen_asm) and/or fp8\n"
        "(fmha_fwd_fp8_varlen_asm, D192x128 only) (default: bf16 fp8)",
    )
    args = parser.parse_args()
    causal_modes = [bool(c) for c in args.causal]

    if "bf16" in args.dtype:
        # ---- correctness + perf table ----
        df = []
        for hdim_q, hdim_v, hq, hk, seqlens in _CORRECTNESS_SHAPES:
            if hdim_q not in args.head_dim:
                continue
            for is_causal, init in itertools.product(causal_modes, args.init):
                if not is_causal and not _kv_alignment_ok(hdim_q, seqlens):
                    continue
                for d64_sink in (True, False) if hdim_q == 64 else (True,):
                    df.append(
                        test_fmha_fwd_with_sink_varlen_asm(
                            hdim_q, hdim_v, hq, hk, seqlens, is_causal, init, d64_sink
                        )
                    )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "fmha_fwd_with_sink_varlen_asm correctness summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        summarize(
            "bf16 gate",
            [test_fmha_fwd_with_sink_varlen_asm_gate(c) for c in _BF16_GATE_CASES],
        )
        summarize(
            "bf16 routing of calls that need gradients",
            [
                test_fmha_fwd_with_sink_varlen_asm_grad_route(*c)
                for c in _GRAD_ROUTE_CASES
                if c[0] in args.head_dim
            ],
        )
        if 64 in args.head_dim:
            summarize(
                "D64 q/k lengths",
                [
                    test_fmha_fwd_with_sink_varlen_asm_d64_qk(hq, hk, sq, sk, c, s)
                    for (hq, hk, sq, sk), c, s in itertools.product(
                        _D64_QK_SHAPES, causal_modes, (True, False)
                    )
                ],
            )

        # ---- D192x128 / D256: strided q/k/v/out, separate q and k lengths ----
        strided_hdims = [hd for hd in _STRIDED_HDIMS if hd[0] in args.head_dim]
        if strided_hdims:
            summarize(
                "strided q/k/v",
                [
                    test_fmha_fwd_with_sink_varlen_asm_strided(
                        dq, dv, hq, hk, seqlens, c, layout
                    )
                    for (dq, dv), (hq, hk, seqlens), c, layout in itertools.product(
                        strided_hdims, _STRIDED_SHAPES, causal_modes, _STRIDED_LAYOUTS
                    )
                ],
            )
            summarize(
                "strided out",
                [
                    test_fmha_fwd_with_sink_varlen_asm_strided_out(
                        dq, dv, hq, hk, seqlens, c, dv + pad
                    )
                    for (dq, dv), (hq, hk, seqlens), c, pad in itertools.product(
                        strided_hdims, _STRIDED_SHAPES, causal_modes, _STRIDED_OUT_PADS
                    )
                ],
            )
            summarize(
                "unsupported stride",
                [
                    test_fmha_fwd_with_sink_varlen_asm_unsupported_stride(
                        dq, dv, case, c
                    )
                    for (dq, dv), case, c in itertools.product(
                        strided_hdims,
                        ["k token stride 2^24 B", "k/v head stride 0"],
                        causal_modes,
                    )
                ],
            )
            summarize(
                "q/k lengths",
                [
                    test_fmha_fwd_with_sink_varlen_asm_qk(dq, dv, hq, hk, sq, sk, c)
                    for (dq, dv), (hq, hk, sq, sk), c in itertools.product(
                        strided_hdims, _QK_SHAPES, causal_modes
                    )
                ],
            )

        # ---- perf-only table (large shapes; ref infeasible) ----
        df = []
        for hdim_q, hdim_v, hq, hk, seqlens in _VARLEN_PERF_SHAPES:
            if hdim_q not in args.head_dim:
                continue
            for is_causal, init in itertools.product(causal_modes, args.init):
                df.append(
                    test_fmha_fwd_with_sink_varlen_asm_perf(
                        hdim_q, hdim_v, hq, hk, seqlens, is_causal, init
                    )
                )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "fmha_fwd_with_sink_varlen_asm perf summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    # ---- FP8 D192x128 (fmha_fwd_fp8_varlen_asm) ----
    if "fp8" in args.dtype and 192 in args.head_dim:
        summarize(
            "fp8 D192x128",
            [
                test_fmha_fwd_fp8_varlen_asm(hq, hk, sq, sk, c, layout)
                for (hq, hk, sq, sk), c, layout in itertools.product(
                    _FP8_CORRECTNESS_SHAPES, causal_modes, _FP8_LAYOUTS
                )
                if layout != "fused_qkv" or sq == sk  # one row holds q, k and v
            ],
        )
        if True in causal_modes:  # chunked prefill is causal
            summarize(
                "fp8 D192x128 chunked prefill",
                [test_fmha_fwd_fp8_varlen_asm_chunked(*s) for s in _FP8_CHUNKED_SHAPES],
            )
        summarize(
            "fp8 D192x128 gate",
            [test_fmha_fwd_fp8_varlen_asm_gate(c) for c in _FP8_GATE_CASES],
        )
        summarize(
            "fp8 D192x128 ops refused",
            [
                test_fmha_fwd_fp8_varlen_asm_ops_refused(c)
                for c in _FP8_OPS_REFUSED_CASES
            ],
        )
        summarize(
            "fp8 D192x128 perf",
            [
                test_fmha_fwd_fp8_varlen_asm_perf(*s, c)
                for s, c in itertools.product(_FP8_PERF_SHAPES, causal_modes)
            ],
        )


if __name__ == "__main__":
    main()
