# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf tests for fmha_fwd_with_sink_varlen_asm (BF16 ASM, gfx1250).

Public API:  aiter.flash_attn_varlen_func          (the path the model calls)
Ops layer:   aiter.fmha_fwd_with_sink_varlen_asm    (low-level, packed/varlen)

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
asymmetric D192x128 one (qk_head_dim=192, v_head_dim=128), which tiles KV by 128
instead of 256.

Sink convention (same as the fixed-batch path): `sink` ([q_head_num] fp32) is a
per-Q-head logit in the scaled domain, a zero-value virtual KV column passed
verbatim.  D64 kernels read it; D128 and D192x128 kernels ignore it (pass None).

KV-length constraint (mask=0 only): the non-causal D64/D128 kernels require
per-sequence kv_seqlen that is a multiple of 256.  D192x128 has no such
constraint (sub_K=128 with the border mask compiled in).

Strided inputs: D192x128 takes q/k/v/out views (e.g. v = kv[..., 128:]); they are
checked against the reference and bit-for-bit against contiguous inputs.
"""

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# .co files only ship for gfx1250 (hsa/gfx1250/fmha_fwd_bf16_varlen/*.co).
SUPPORTED_GFX = ["gfx1250"]


# ---------------------------------------------------------------------------
# Reference (fp32 math, cast back).  Not timed, not in the table.
# ---------------------------------------------------------------------------


def _attn_one(q, k, v, *, is_causal, sink):
    """Single-sequence attention reference (no batch dim).

    q: (sq, hq, d)  k: (sk, hk, d)  v: (sk, hk, dv) -> out (sq, hq, dv), lse (sq, hq).
    """
    sq, hq, d = q.shape
    sk, hk, _ = k.shape
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
    denom = torch.exp(scores - max_total.unsqueeze(-1)).sum(dim=-1)
    if sink is not None:
        denom = denom + torch.exp(sink_hs - max_total)
    probs = torch.exp(scores - max_total.unsqueeze(-1)) / denom.unsqueeze(-1)
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
    q, k, v, cu_q, cu_k, max_seqlen_q, *, scale, is_causal, sink=None, via="public"
):
    """Return (out, lse) with lse shaped (total_q, nheads) to match run_torch.

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
            max_seqlen_q,  # equal q/k seqlens in these tests
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
            sink_ptr=sink,
        )
        return r[0], r[1].transpose(0, 1).contiguous()
    if via == "ops":
        out, lse = aiter.fmha_fwd_with_sink_varlen_asm(
            q, k, v, cu_q, cu_k, max_seqlen_q, scale, is_causal, True, sink=sink
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

# Correctness shapes (torch reference feasible).  hq=64; hk=8 (D64) / 4 (D128/D192x128).
# The symmetric D64/D128 kernels tile KV by 256 and their non-causal (mask=0)
# variants require every kv_seqlen % 256 == 0 (filtered); D192x128 tiles by 128
# with a border mask, so unaligned lengths are in scope for it too.
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
]


def _kv_alignment_ok(hdim_q, seqlens):
    """Non-causal eligibility for the per-head-dim KV tile.

    D64/D128 use sub_K=256 and need every kv_seqlen to be a multiple of it.
    D192x128 uses sub_K=128 and compiles the border mask in, so any length works.
    """
    if hdim_q == 192:
        return True
    return all(s % 256 == 0 for s in seqlens)


# ---------------------------------------------------------------------------
# Test functions (one markdown table each).
# ---------------------------------------------------------------------------


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm(
    hdim_q, hdim_v, hq, hk, seqlens, is_causal, init
):
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, hdim_q, hdim_v, init=init)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(hdim_q)
    sink = _d64_sink(hq) if hdim_q == 64 else None

    ref_out, ref_lse = run_torch(q, k, v, cu, cu, is_causal=is_causal, sink=sink)

    total = q.shape[0]
    flops, nbytes = _flops_bytes(
        seqlens, hq, hk, hdim_q, hdim_v, is_causal, total, q.element_size()
    )

    # The model calls the public dispatcher (flash_attn_varlen_func) → asm path.
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
            via="public",
        )
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (out, lse), us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err(O)"] = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} O d={hdim_q}x{hdim_v} c={is_causal}",
        )
        ret[f"{name} err(LSE)"] = checkAllclose(
            ref_lse.to(dtypes.fp32),
            lse.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} LSE d={hdim_q}x{hdim_v} c={is_causal}",
        )
    return ret


def _as_view(x, head_pad=0, token_pad=0, base_off=0):
    """Strided copy of `x` in a NaN buffer; odd pads/offset give 2-byte-aligned rows."""
    t, h, d = x.shape
    row = h * (d + head_pad) + token_pad
    buf = torch.full((base_off + t * row,), float("nan"), dtype=x.dtype)
    view = buf.as_strided((t, h, d), (row, d + head_pad, 1), base_off)
    view.copy_(x)
    return view


def _strided_inputs(layout, q, k, v):
    """Views with the same values as q/k/v for one of the strided layouts."""
    t, hq, dq = q.shape
    hk, dv = k.size(1), v.size(2)
    if layout == "kv_split":
        # DeepSeek-style: v is the second half of a [.., k_nope | v] projection.
        kv = torch.full((t, hk, dv + dv), float("nan"), dtype=v.dtype)
        kv[..., dv:] = v
        return q, k, kv[..., dv:]
    if layout == "fused_qkv":
        # One row per token holds every q, k and v head back to back.
        row = torch.full((t, hq * dq + hk * dq + hk * dv), float("nan"), dtype=q.dtype)
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
        # Odd-element strides and offsets: rows start only 2-byte aligned.
        return (
            _as_view(q, head_pad=1, token_pad=3),
            _as_view(k, head_pad=1, token_pad=1),
            _as_view(v, head_pad=3, token_pad=1, base_off=1),
        )
    raise ValueError(f"unknown layout {layout!r}")


# Runs per case; every run must match the dense result bit for bit.
_STRIDED_REPEATS = 5


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_strided(hq, hk, seqlens, is_causal, layout, via):
    """D192x128 with strided q/k/v, or a strided `out` for the out_view layouts."""
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, 192, 128)
    max_seqlen_q = max(seqlens)
    scale = 1.0 / math.sqrt(192)
    ref_out, ref_lse = run_torch(q, k, v, cu, cu, is_causal=is_causal, sink=None)
    dense_out, dense_lse = run_kernel(
        q, k, v, cu, cu, max_seqlen_q, scale=scale, is_causal=is_causal, via=via
    )

    ret = {"gfx": get_gfx()}
    same_o = same_lse = in_place = pad_untouched = True
    for _ in range(_STRIDED_REPEATS):
        if layout.startswith("out_view"):
            # out is a view of a wider NaN buffer (width 129: 2-byte-aligned rows).
            width = 129 if layout == "out_view_odd" else 192
            wide = torch.full((q.size(0), hq, width), float("nan"), dtype=q.dtype)
            out_view = wide[..., :128]
            out, lse = aiter.fmha_fwd_with_sink_varlen_asm(
                q, k, v, cu, cu, max_seqlen_q, scale, is_causal, True, out=out_view
            )
            lse = lse.squeeze(-1)
            in_place &= out.data_ptr() == out_view.data_ptr()
            pad_untouched &= bool(wide[..., 128:].isnan().all().item())
        else:
            qs, ks, vs = _strided_inputs(layout, q, k, v)
            out, lse = run_kernel(
                qs,
                ks,
                vs,
                cu,
                cu,
                max_seqlen_q,
                scale=scale,
                is_causal=is_causal,
                via=via,
            )
        same_o &= torch.equal(out, dense_out)
        same_lse &= torch.equal(lse, dense_lse)
    assert same_o and same_lse, f"strided {layout} via={via}: differs from dense"
    assert in_place and pad_untouched, f"{layout}: out not written in place"
    ret["== dense (O)"] = same_o
    ret["== dense (LSE)"] = same_lse
    if layout.startswith("out_view"):
        ret["out in place"] = in_place
        ret["pad untouched"] = pad_untouched
    ret["err(O)"] = checkAllclose(
        ref_out.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"strided {layout} via={via} O c={is_causal}",
    )
    ret["err(LSE)"] = checkAllclose(
        ref_lse.to(dtypes.fp32),
        lse.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"strided {layout} via={via} LSE c={is_causal}",
    )
    return ret


def _unsupported_kv(case, k, v):
    """k/v views the D192x128 kernel cannot address."""
    if case == "k token stride 2^24 B":
        tok = (1 << 24) // k.element_size()
        buf = torch.empty((len(k) - 1) * tok + k.size(1) * 192, dtype=k.dtype)
        k_far = buf.as_strided(k.shape, (tok, 192, 1))
        k_far.copy_(k)
        return k_far, v
    if case == "k/v head stride 0":
        # All kv heads share one head (expanded MQA): pass hk=1 instead.
        return k[:, :1].expand_as(k), v[:, :1].expand_as(v)
    raise ValueError(f"unknown case {case!r}")


@benchmark()
def test_fmha_fwd_with_sink_varlen_asm_unsupported_stride(case, is_causal):
    """Unsupported k/v strides: the public path must use another backend, and a
    direct op call must be refused by the C++ entry."""
    seqlens, hq, hk = [2], 4, 2
    q, k, v, cu = make_varlen_packed(seqlens, hq, hk, 192, 128)
    k, v = _unsupported_kv(case, k, v)
    scale = 1.0 / math.sqrt(192)
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
        "ops refused": ops_refused,
        "public asm calls": len(calls) - ops_calls,
        "err(O)": checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{case} -> other backend, c={is_causal}",
        ),
    }


def test_fmha_fwd_with_sink_varlen_asm_overlapping_out():
    """An out whose heads overlap (head stride 64 < 128) must be refused."""
    q, k, v, cu = make_varlen_packed([2], 4, 2, 192, 128)
    t, hq = q.shape[:2]
    buf = torch.empty(t * hq * 64 + 64, dtype=q.dtype)
    out = buf.as_strided((t, hq, 128), (hq * 64, 64, 1))
    try:
        aiter.fmha_fwd_with_sink_varlen_asm(
            q, k, v, cu, cu, 2, 1.0 / math.sqrt(192), False, True, out=out
        )
        refused = False
    except RuntimeError as e:
        refused = "must not overlap" in str(e)
    assert refused, "C++ accepted an overlapping out"


_STRIDED_LAYOUTS = [
    "kv_split",
    "fused_qkv",
    "padded",
    "odd",
    "out_view",
    "out_view_odd",
]
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
            via="public",
        )
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


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
        choices=[64, 128, 192],
        default=[64, 128, 192],
        help="qk head dim(s) to test; 192 selects the D192x128 kernels "
        "(default: 64 128 192)",
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
        help="q/k/v init pattern(s) (default: randn const0.25)",
    )
    args = parser.parse_args()
    causal_modes = [bool(c) for c in args.causal]

    # ---- correctness + perf table ----
    df = []
    for hdim_q, hdim_v, hq, hk, seqlens in _CORRECTNESS_SHAPES:
        if hdim_q not in args.head_dim:
            continue
        for is_causal, init in itertools.product(causal_modes, args.init):
            if not is_causal and not _kv_alignment_ok(hdim_q, seqlens):
                continue
            df.append(
                test_fmha_fwd_with_sink_varlen_asm(
                    hdim_q, hdim_v, hq, hk, seqlens, is_causal, init
                )
            )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fmha_fwd_with_sink_varlen_asm correctness summary (markdown):\n%s",
        df.to_markdown(index=False),
    )

    # ---- D192x128 strided q/k/v/out ----
    if 192 in args.head_dim:
        df = []
        for (hq, hk, seqlens), is_causal, layout in itertools.product(
            _STRIDED_SHAPES, causal_modes, _STRIDED_LAYOUTS
        ):
            for via in ["ops"] if layout.startswith("out_view") else ["public", "ops"]:
                df.append(
                    test_fmha_fwd_with_sink_varlen_asm_strided(
                        hq, hk, seqlens, is_causal, layout, via
                    )
                )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "fmha_fwd_with_sink_varlen_asm D192x128 strided summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        df = pd.DataFrame(
            [
                test_fmha_fwd_with_sink_varlen_asm_unsupported_stride(case, is_causal)
                for case, is_causal in itertools.product(
                    ["k token stride 2^24 B", "k/v head stride 0"], causal_modes
                )
            ]
        )
        aiter.logger.info(
            "fmha_fwd_with_sink_varlen_asm D192x128 unsupported-stride summary "
            "(markdown):\n%s",
            df.to_markdown(index=False),
        )
        test_fmha_fwd_with_sink_varlen_asm_overlapping_out()

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


if __name__ == "__main__":
    main()
