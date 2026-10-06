# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL flash attention: gfx1201 bf16/f16 and gfx950 fp8 correctness and timing sweeps.

Timings use warm, fixed buffers; TB/s is logical traffic, not measured HBM bandwidth.
The fp8 reference dequantizes the same e4m3fn tensors the kernel reads, so the error
measured is the kernel's, not the quantizer's.
"""

import argparse
import itertools
import os
import sys
from functools import partial
from itertools import pairwise

import pandas as pd
import torch
import torch.nn.functional as F

# Plain-script CI invocation must select this checkout, not another editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.quant import per_tensor_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest

FP8_DTYPE = torch.float8_e4m3fn
FP8_REL_ERR = 8e-2
FP8_MIN_COS = 0.98
FP8_LSE_REL_ERR = 1e-4
FP8_UNIFORM_RANGE = (-1.0, 1.0)
FP8_SEED = 123


def visible_pairs(q, k, causal):
    # Bottom-right-aligned causal: query i sees k - q + i + 1 keys.
    if not causal:
        return q * k
    return q * k - q * (q - 1) // 2 if k >= q else k * (k + 1) // 2


# ---------------------------------------------------------------- gfx1201 bf16/f16


def _ref_sdpa_bshd(q, k, v, causal=False):
    """SDPA reference with BSHD inputs/outputs."""
    out_bhsd = F.scaled_dot_product_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        is_causal=causal,
    )
    return out_bhsd.transpose(1, 2).contiguous()


# (dtype, causal, batch, seq_len, num_heads, head_dim)
RDNA_CASES = [
    # Aligned production-like Wan2.1 1.3B shape, padded to a multiple of 128.
    (dtypes.bf16, False, 1, 32768, 12, 128),
    (dtypes.bf16, False, 2, 1024, 8, 128),
    # Unaligned shape exercises the auto-padding path: 32760 -> 32768.
    (dtypes.bf16, False, 1, 32760, 12, 128),
    (dtypes.fp16, False, 1, 32768, 12, 128),
    (dtypes.bf16, True, 2, 4096, 8, 128),
]


@benchmark()
def test_fmha_rdna(dtype, causal, batch, seq_len, num_heads, head_dim):
    from aiter.ops.flydsl import flydsl_flash_attn_func

    g = torch.Generator(device="cuda").manual_seed(0)
    shape = (batch, seq_len, num_heads, head_dim)
    q, k, v = (
        torch.randn(shape, generator=g, dtype=dtype, device="cuda") for _ in range(3)
    )
    want = _ref_sdpa_bshd(q, k, v, causal)
    candidates = {"flydsl": lambda: flydsl_flash_attn_func(q, k, v, causal=causal)}

    flops = 4 * batch * num_heads * head_dim * visible_pairs(seq_len, seq_len, causal)
    nbytes = 4 * q.numel() * q.element_size()
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        got = fn()
        assert got.shape == want.shape and got.dtype == dtype
        # bf16 attention is noisy; cosine is the correctness signal, not elementwise.
        cos = F.cosine_similarity(
            got.float().reshape(-1, head_dim),
            want.float().reshape(-1, head_dim),
            dim=1,
        )
        assert cos.min().item() > 0.99, f"{name}: min_cos={cos.min().item():.6f}"
        assert cos.mean().item() > 0.999, f"{name}: mean_cos={cos.mean().item():.6f}"
        if seq_len % 128:
            # Padded zero keys must not move the output: the production-shape gate.
            assert cos.min().item() > 0.9999, f"{name}: min_cos={cos.min().item():.6f}"
        _, us = run_perftest(fn, num_rotate_args=1)
        assert us > 0, f"{name}: empty timing"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = 1 - cos.min().item()
    return ret


# ---------------------------------------------------------------- gfx950 fp8


def _fp8_quant(x):
    """Per-tensor e4m3fn quantization: descale = amax / fp8_max, floored at 1e-12."""
    fp8_max = torch.finfo(FP8_DTYPE).max
    descale = (x.abs().amax().float() / fp8_max).clamp(min=1e-12)
    y, scale = per_tensor_quant(x, scale=descale, quant_dtype=FP8_DTYPE)
    return y.contiguous(), scale.contiguous()


def _fp8_dequant(x, descale):
    return x.to(torch.float32) * descale.to(torch.float32)


def _fp8_rel_err(got, ref, floor=0.0):
    scale = max(ref.abs().max().item(), floor)
    err = (got - ref).abs().max().item()
    return (err / scale if scale > 0 else err), err, scale


def _ref_attention(q, k, v, causal, softmax_scale=None):
    """fp32 SDPA over BSHD/THD-with-batch inputs, GQA-aware, bottom-right causal.

    ``F.scaled_dot_product_attention(is_causal=True)`` aligns the mask top-left,
    which differs from this kernel (and from aiter's documented convention) as
    soon as Sq != Skv, so the mask is built explicitly from ``delta``.
    """
    q_t, k_t, v_t = (t.transpose(1, 2).float() for t in (q, k, v))
    nh_q, nh_kv = q_t.shape[1], k_t.shape[1]
    if nh_q != nh_kv:
        rep = nh_q // nh_kv
        k_t = k_t.repeat_interleave(rep, dim=1)
        v_t = v_t.repeat_interleave(rep, dim=1)
    Sq, D = q_t.shape[2], q_t.shape[3]
    Skv = k_t.shape[2]
    scale = D**-0.5 if softmax_scale is None else softmax_scale
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale
    if causal:
        delta = Skv - Sq
        q_idx = torch.arange(Sq, device=q.device).view(-1, 1)
        k_idx = torch.arange(Skv, device=q.device).view(1, -1)
        scores = scores.masked_fill(k_idx > q_idx + delta, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    # A fully-masked row softmaxes to NaN; the kernel writes zeros there.
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v_t).transpose(1, 2)


def _ref_lse(q, k, causal, softmax_scale=None):
    """fp64 log-sum-exp of the same logits ``_ref_attention`` softmaxes, as [B, H, Sq]."""
    q_t, k_t = (t.transpose(1, 2).double() for t in (q, k))
    nh_q, nh_kv = q_t.shape[1], k_t.shape[1]
    if nh_q != nh_kv:
        k_t = k_t.repeat_interleave(nh_q // nh_kv, dim=1)
    Sq, D = q_t.shape[2], q_t.shape[3]
    Skv = k_t.shape[2]
    scale = D**-0.5 if softmax_scale is None else softmax_scale
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * scale
    if causal:
        delta = Skv - Sq
        q_idx = torch.arange(Sq, device=q.device).view(-1, 1)
        k_idx = torch.arange(Skv, device=q.device).view(1, -1)
        scores = scores.masked_fill(k_idx > q_idx + delta, float("-inf"))
    return torch.logsumexp(scores, dim=-1).float()


def _assert_lse_matches(got, ref):
    """Compare LSE against the fp64 reference, treating fully-masked rows exactly.

    A row that sees no key at all (bottom-right causal with Skv < Sq, or a
    zero-length varlen KV entry) has LSE = -inf, and ``-inf - -inf`` is NaN, so
    those rows are asserted as an exact bit match instead of a tolerance.
    """
    dead = torch.isinf(ref) & (ref < 0)
    assert not torch.isnan(got).any(), (
        f"LSE has {int(torch.isnan(got).sum())} NaN entries "
        f"({int((torch.isnan(got) & dead).sum())} of them on fully-masked rows)"
    )
    assert torch.equal(got[dead], ref[dead]), (
        f"{int((got[dead] != float('-inf')).sum())} of {int(dead.sum())} "
        "fully-masked rows are not -inf"
    )
    live = ~dead
    if live.any():
        rel, err, scale = _fp8_rel_err(got[live], ref[live], floor=1.0)
        assert rel < FP8_LSE_REL_ERR, (
            f"fp8 LSE gate: rel_err={rel:.3e} (< {FP8_LSE_REL_ERR}), "
            f"abs_err={err:.3e}, |lse|max={scale:.3e}"
        )


def _case(**kw):
    base = {
        "causal": (False, True),
        "varlen": False,
        "B": 1,
        "S": 1,
        "Skv": None,
        "vl_q": None,
        "vl_kv": None,
        "H": 12,
        "HKV": None,
        "D": 192,
        "Dv": 128,
        "splits": None,
        "out_dtype": dtypes.bf16,
        "dist": "uniform",
        "k_scale": 1.0,
        "scale": None,
        "block_m": None,
        "entry": "kernel",
        "checks": (),
    }
    base.update(kw)
    return base


VL_Q = {1: [2614], 2: [1024, 1590], 3: [1024, 512, 1078], 4: [1024, 512, 256, 822]}
VL_KV = {
    1: [16384],
    2: [8192, 8192],
    3: [8192, 4096, 4096],
    4: [8192, 4096, 2048, 2048],
}
SPLIT_MODES = {"dense": 1, "auto": None}
FP8_CASES = {}


def _add(name, **kw):
    assert name not in FP8_CASES, name
    FP8_CASES[name] = _case(**kw)


for S in (4096, 8192):
    for H, HKV in ((16, 1), (32, 8), (12, 12)):
        _add(f"gqa_dense_s{S}_h{H}_kv{HKV}", S=S, H=H, HKV=HKV, D=128)
for H, HKV in ((16, 1), (32, 8)):
    _add(
        f"gqa_varlen_h{H}_kv{HKV}",
        varlen=True,
        vl_q=VL_Q[2],
        vl_kv=VL_KV[2],
        H=H,
        HKV=HKV,
        D=128,
    )
# fp16 output buffer: the same kernel with a non-default out dtype.
_add("out_f16_dense", S=4096, H=16, HKV=1, D=128, splits=1, out_dtype=dtypes.fp16)
_add(
    "out_f16_varlen",
    varlen=True,
    vl_q=VL_Q[2],
    vl_kv=VL_KV[2],
    H=16,
    HKV=1,
    D=128,
    splits=1,
    out_dtype=dtypes.fp16,
)
for mode, sp in SPLIT_MODES.items():
    for B, S in ((1, 4096), (2, 4096), (3, 4096), (4, 4096), (1, 8192)):
        _add(f"d192_dense_b{B}_s{S}_{mode}", B=B, S=S, splits=sp)
    for B in (1, 2, 3, 4):
        _add(f"d192_varlen_b{B}_{mode}", varlen=True, vl_q=VL_Q[B], splits=sp)
        _add(
            f"d192_varlen_cross_b{B}_{mode}",
            varlen=True,
            vl_q=VL_Q[B],
            vl_kv=VL_KV[B],
            splits=sp,
        )
# The 128/128 pair isolates the split from the head-dim pair.
for D in (128, 192):
    for sp in (2, 4, 8, 16):
        _add(f"split_kv_d{D}_n{sp}", S=8192, D=D, splits=sp)
for B in (1, 2, 3, 4):
    _add(f"split_kv_batched_b{B}", B=B, S=8192, splits=4)
for sq, skv, sp in ((512, 16384, 8), (2614, 16384, 8), (1024, 32768, 16)):
    _add(f"split_kv_cross_s{sq}_kv{skv}_n{sp}", S=sq, Skv=skv, splits=sp)
# Short cached-prefix queries are eligible only without causal masking.
for sq, skv, sp in (
    (1, 257, 4),
    (70, 16384, 2),
    (70, 16384, 4),
    (70, 16384, 16),
    (70, 16384, None),
    (127, 11008, 4),
    (129, 257, 4),
    (255, 513, 4),
    (383, 11008, 4),
):
    _add(
        f"split_kv_short_q_s{sq}_kv{skv}_n{sp}",
        S=sq,
        Skv=skv,
        splits=sp,
        causal=(False,),
    )
for S, sp in ((4096, 2), (8192, 4), (16384, 8), (32768, 16)):
    _add(f"split_kv_long_s{S}_n{sp}", S=S, splits=sp)
for Dv in (64, 96, 128, 160, 192):
    _add(f"v_head_dim_{Dv}", S=512, H=4, D=128, Dv=Dv, causal=(False,))
for S in (1, 385, 1000, 4097):
    _add(f"ragged_s{S}", B=2, S=S, H=4, causal=(True,))
# Every output row must be written (the output buffer is NaN-prefilled): the shapes
# leave a tail in the split combine grid.
for B, S, H in ((1, 4097, 1), (1, 4097, 3), (1, 2050, 7), (1, 8193, 1)):
    _add(f"auto_split_tail_s{S}_h{H}", B=B, S=S, H=H, D=128, causal=(False,))
# Varlen split-K must not mix batches inside a combine wave.
for S, H, Dv in ((385, 1, 128), (385, 3, 128), (1155, 1, 128), (386, 1, 64)):
    _add(
        f"varlen_split_batch_s{S}_h{H}_dv{Dv}",
        varlen=True,
        vl_q=[S] * 8,
        H=H,
        Dv=Dv,
        splits=2,
        causal=(False,),
    )
# Softmax normalisation under peaked attention: k_scale widens the score range, which
# is invisible on near-uniform attention. Lazy and eager rescale are both candidates.
for S in (1024, 12288):
    for ks in (1.0, 200.0):
        _add(
            f"peaked_s{S}_k{ks:g}",
            B=2,
            S=S,
            H=8,
            D=128,
            dist="normal",
            k_scale=ks,
            causal=(False,),
        )
# Public wrappers must route fp8 to the gfx950 kernel.
for sc in (None, 0.37):
    _add(f"dispatch_batch_sc{sc}", B=2, S=1024, H=8, D=128, scale=sc, entry="batch")
    _add(
        f"dispatch_varlen_sc{sc}",
        varlen=True,
        vl_q=[400, 1648],
        H=8,
        D=128,
        scale=sc,
        entry="varlen",
    )
# The accuracy gate must track the kernel, not the magnitude of the inputs.
for dist in ("uniform", "normal", "normal_x8"):
    _add(f"scale_invariant_{dist}", S=2048, H=8, D=128, dist=dist, checks=("lse",))
for i, (causal, B, S, Skv, H, HKV, D, Dv, sp) in enumerate(
    [
        (True, 2, 256, 256, 8, 8, 128, 128, None),
        (False, 2, 256, 256, 8, 8, 128, 128, None),
        (True, 1, 512, 1024, 16, 2, 128, 128, None),
        (True, 1, 384, 384, 12, 12, 192, 128, None),
        (True, 1, 128, 512, 8, 8, 192, 192, None),
        (True, 1, 1024, 4096, 8, 1, 128, 128, 4),
        (False, 2, 512, 512, 8, 8, 128, 128, 2),
        (True, 1, 512, 2048, 12, 12, 192, 128, 4),
        (False, 1, 70, 16384, 12, 12, 192, 128, 4),
        (False, 1, 70, 16384, 12, 12, 192, 128, None),
        # Skv < Sq: bottom-right causal leaves the leading Sq-Skv rows with no
        # visible key, so their LSE must be -inf rather than NaN.
        (True, 1, 1024, 128, 8, 8, 128, 128, None),
        (True, 1, 2048, 256, 16, 4, 128, 128, None),
        (True, 1, 1024, 128, 8, 8, 192, 128, None),
        (True, 1, 1024, 128, 8, 8, 128, 128, 4),
    ]
):
    _add(
        f"lse_dense_{i}",
        causal=(causal,),
        B=B,
        S=S,
        Skv=Skv,
        H=H,
        HKV=HKV,
        D=D,
        Dv=Dv,
        splits=sp,
        checks=("lse",),
    )
for i, (vl_q, vl_kv, H, HKV, D, Dv, sp) in enumerate(
    [
        ([300, 500], [300, 500], 8, 8, 128, 128, None),
        ([300, 500], [1024, 2048], 12, 2, 192, 128, None),
        ([512, 512], [4096, 4096], 8, 8, 128, 128, 4),
    ]
):
    _add(
        f"lse_varlen_{i}",
        varlen=True,
        vl_q=vl_q,
        vl_kv=vl_kv,
        H=H,
        HKV=HKV,
        D=D,
        Dv=Dv,
        splits=sp,
        checks=("lse",),
    )
# Dead rows (Skv < Sq) must be -inf on every tile/split configuration. The race that
# wrote NaN in a varying handful of rows is per-launch, hence the repeat loop.
for S, Skv in ((1024, 128), (2048, 256)):
    for bm in (128, 256):
        for sp in (1, 4):
            _add(
                f"dead_rows_s{S}_kv{Skv}_bm{bm}_n{sp}",
                S=S,
                Skv=Skv,
                H=32,
                D=128,
                splits=sp,
                block_m=bm,
                causal=(True,),
                checks=("lse", "dead_lse"),
            )
# MLA chunked prefill produces seqlen_kv == 0 entries routinely, and the downstream
# merge consumes LSE, so NaN here spreads across the whole layer.
for i, (vl_q, vl_kv) in enumerate(
    [
        ([256, 128, 64], [512, 0, 300]),
        ([256, 128], [0, 384]),
        ([256, 128], [0, 0]),
    ]
):
    _add(
        f"zero_kv_{i}",
        varlen=True,
        vl_q=vl_q,
        vl_kv=vl_kv,
        H=8,
        D=128,
        Dv=128,
        checks=("lse", "zero_kv"),
    )
# Runtime softmax_scale: updates O and LSE on the cached launcher, leaves the
# descales untouched, and the default scale is bit-identical to an explicit D**-0.5.
for i, (S, Skv, HKV, D, sp) in enumerate(
    [
        (512, 512, 12, 128, 1),
        (70, 2048, 12, 192, 1),  # cached-chunk prefill
        (512, 2048, 2, 192, 4),  # split-K + GQA
        (512, 128, 12, 192, 1),  # fully masked leading rows when causal
        (512, 128, 12, 192, 4),  # empty splits
    ]
):
    _add(
        f"scale_dense_{i}",
        S=S,
        Skv=Skv,
        HKV=HKV,
        D=D,
        splits=sp,
        dist="normal",
        scale=0.37,
        checks=("lse", "scale"),
    )
for sp in (1, 4, None):
    for tag, cuq, cukv, causal in (
        ("a", [0, 512, 582, 838], [0, 2048, 2048, 2176], (False, True)),
        ("zero_kv", [0, 70, 71, 198, 198], [0, 16384, 16384, 16641, 16769], (False,)),
    ):
        _add(
            f"scale_varlen_{tag}_n{sp}",
            varlen=True,
            vl_q=[b - a for a, b in pairwise(cuq)],
            vl_kv=[b - a for a, b in pairwise(cukv)],
            splits=sp,
            dist="normal",
            scale=0.37,
            causal=causal,
            checks=("lse", "scale"),
        )
# A custom runtime scalar needs no scale-preparation kernel during graph capture.
_add(
    "scale_graph_s512",
    S=512,
    scale=0.137,
    causal=(True,),
    checks=("lse", "graph"),
)
_add(
    "scale_graph_s70_kv16384",
    S=70,
    Skv=16384,
    splits=4,
    scale=0.137,
    causal=(False,),
    checks=("lse", "graph"),
)
# A caller-supplied out= comes back filled and is the same tensor.
for sc in (None, 0.37):
    _add(
        f"out_returned_sc{sc}",
        B=2,
        S=512,
        H=8,
        D=128,
        scale=sc,
        causal=(False,),
        checks=("out_returned",),
    )


def _uniform(shape, c):
    if c["dist"] == "uniform":
        return torch.empty(shape, dtype=dtypes.bf16, device="cuda").uniform_(
            *FP8_UNIFORM_RANGE
        )
    x = torch.randn(shape, dtype=dtypes.bf16, device="cuda")
    return x * 8 if c["dist"] == "normal_x8" else x


def _cu(lens):
    return [0] + list(itertools.accumulate(lens))


def make_fp8_inputs(c):
    torch.manual_seed(FP8_SEED)
    H, D, Dv = c["H"], c["D"], c["Dv"]
    HKV = c["HKV"] or H
    if c["varlen"]:
        vl_q = c["vl_q"]
        vl_kv = c["vl_kv"] if c["vl_kv"] is not None else vl_q
        cuq, cukv = _cu(vl_q), _cu(vl_kv)
        # k/v still need a real allocation when every entry is empty.
        tkv = max(cukv[-1], 1)
        shapes = ((cuq[-1], H, D), (tkv, HKV, D), (tkv, HKV, Dv))
    else:
        B, S = c["B"], c["S"]
        Skv = S if c["Skv"] is None else c["Skv"]
        shapes = ((B, S, H, D), (B, Skv, HKV, D), (B, Skv, HKV, Dv))
    q_bf, k_bf, v_bf = (_uniform(s, c) for s in shapes)
    k_bf = k_bf * c["k_scale"]
    q, qs = _fp8_quant(q_bf)
    k, ks = _fp8_quant(k_bf)
    v, vs = _fp8_quant(v_bf)
    return q, k, v, {"q_descale": qs, "k_descale": ks, "v_descale": vs}


def fp8_reference(c, causal, q, k, v, descales, scale):
    """fp32 torch attention and fp64 LSE over the dequantized tensors."""
    qd, kd, vd = (
        _fp8_dequant(t, descales[n])
        for t, n in ((q, "q_descale"), (k, "k_descale"), (v, "v_descale"))
    )
    if not c["varlen"]:
        return _ref_attention(qd, kd, vd, causal, scale), _ref_lse(
            qd, kd, causal, scale
        )
    vl_q = c["vl_q"]
    vl_kv = c["vl_kv"] if c["vl_kv"] is not None else vl_q
    cuq, cukv = _cu(vl_q), _cu(vl_kv)
    ref = torch.zeros(q.shape[:-1] + (c["Dv"],), dtype=torch.float32, device="cuda")
    lse = torch.full((c["H"], cuq[-1]), float("-inf"), device="cuda")
    for b in range(len(vl_q)):
        if vl_kv[b] == 0:
            continue  # no softmax support: O == 0 and LSE == -inf
        sl_q, sl_kv = slice(cuq[b], cuq[b + 1]), slice(cukv[b], cukv[b + 1])
        args = (qd[sl_q].unsqueeze(0), kd[sl_kv].unsqueeze(0), vd[sl_kv].unsqueeze(0))
        ref[sl_q] = _ref_attention(*args, causal, scale).squeeze(0)
        lse[:, sl_q] = _ref_lse(args[0], args[1], causal, scale)[0]
    return ref, lse


def compare(want, got, name):
    atol = FP8_REL_ERR * want.abs().max().item()
    got = got.float()
    assert not torch.isnan(
        got
    ).any(), (
        f"{name}: {int(torch.isnan(got).any(-1).sum())} output rows were never written"
    )
    err = checkAllclose(want, got, rtol=0, atol=atol, tol_err_ratio=0, msg=name)
    cos = F.cosine_similarity(
        got.reshape(-1, want.shape[-1]), want.reshape(-1, want.shape[-1]), dim=1
    )
    # Fully masked or empty-KV rows are zero in both; cosine is undefined there.
    both_zero = (got == 0).all(-1).reshape(-1) & (want == 0).all(-1).reshape(-1)
    cos = cos.masked_fill(both_zero, 1.0)
    assert err == 0 and cos.min().item() > FP8_MIN_COS, (
        f"{name}: fp8 gate: {err:.3%} elements exceed atol={atol:.3e}, "
        f"min_cos={cos.min().item():.5f} (> {FP8_MIN_COS})"
    )
    return err


@benchmark()
def test_fmha_fp8(name, causal):
    from aiter.ops.flydsl.fmha_kernels import (
        flydsl_flash_attn_batch_func,
        flydsl_flash_attn_varlen_func,
    )
    from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
        flydsl_flash_attn_fp8_func,
    )

    c = FP8_CASES[name]
    H, D, Dv = c["H"], c["D"], c["Dv"]
    HKV = c["HKV"] or H
    q, k, v, descales = make_fp8_inputs(c)
    scale = c["scale"]
    ref, ref_lse = fp8_reference(
        c, causal, q, k, v, descales, D**-0.5 if scale is None else scale
    )
    want_lse = "lse" in c["checks"]
    # NaN-prefilled so an unwritten row fails the comparison.
    out = torch.full(
        q.shape[:-1] + (Dv,), float("nan"), device="cuda", dtype=c["out_dtype"]
    )
    kw = dict(causal=causal, softmax_scale=scale, **descales)
    if c["varlen"]:
        vl_q = c["vl_q"]
        vl_kv = c["vl_kv"] if c["vl_kv"] is not None else vl_q
        cuq, cukv = _cu(vl_q), _cu(vl_kv)
        cu_q = torch.tensor(cuq, dtype=torch.int32, device="cuda")
        cu_kv = torch.tensor(cukv, dtype=torch.int32, device="cuda")
        vkw = {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_kv": cu_kv,
            "max_seqlen_q": max(vl_q),
            "max_seqlen_kv": max(max(vl_kv), 1),
            "cross_seqlen": vl_q != vl_kv,
        }
    else:
        vkw = {}
    kernel_kw = dict(
        kw,
        **vkw,
        num_kv_heads=HKV,
        num_kv_splits=c["splits"],
        fp8_block_m=c["block_m"],
        return_lse=want_lse,
    )
    if c["entry"] == "kernel":
        # Lazy and eager rescale lift P differently; both must match the same reference.
        candidates = {
            tag: partial(
                flydsl_flash_attn_fp8_func,
                q,
                k,
                v,
                out=out,
                dualwave_swp_lazy_rescale=lazy,
                **kernel_kw,
            )
            for tag, lazy in (("lazy", True), ("eager", False))
        }
    elif c["entry"] == "batch":
        candidates = {
            "dispatch": lambda: flydsl_flash_attn_batch_func(q, k, v, **kw),
        }
    else:
        candidates = {
            "dispatch": lambda: flydsl_flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_kv,
                vkw["max_seqlen_q"],
                vkw["max_seqlen_kv"],
                **kw,
            ),
        }

    if c["varlen"]:
        pairs = sum(visible_pairs(a, b, causal) for a, b in zip(vl_q, vl_kv))
    else:
        skv = c["S"] if c["Skv"] is None else c["Skv"]
        pairs = c["B"] * visible_pairs(c["S"], skv, causal)
    flops = 2 * H * (D + Dv) * pairs
    tq = q.shape[0] if c["varlen"] else q.shape[0] * q.shape[1]
    tkv = k.shape[0] if c["varlen"] else k.shape[0] * k.shape[1]
    nbytes = tq * H * D + tkv * HKV * (D + Dv) + out.numel() * out.element_size()

    ret = {"gfx": get_gfx_runtime()}
    for tag, fn in candidates.items():
        got = fn()
        got_lse = None
        if isinstance(got, tuple):
            got, got_lse = got
        assert got is not None, "gfx950 fp8 must not fall through to CK/Triton"
        torch.cuda.synchronize()
        assert got.shape == ref.shape and got.dtype == c["out_dtype"]
        err = compare(ref, got, tag)
        if got_lse is not None:
            want_shape = (H, q.shape[0]) if c["varlen"] else (c["B"], H, c["S"])
            assert got_lse.shape == want_shape and got_lse.dtype == torch.float32
            _assert_lse_matches(got_lse, ref_lse)
        _, us = run_perftest(fn, num_rotate_args=1)
        assert us > 0, f"{tag}: empty timing"
        ret[f"{tag} us"] = us
        ret[f"{tag} TFLOPS"] = flops / us / 1e6
        ret[f"{tag} TB/s"] = nbytes / us / 1e6
        ret[f"{tag} err"] = err

    if c["entry"] != "kernel":
        return ret
    checks = c["checks"]
    plain = dict(kernel_kw, return_lse=False)
    if "lse" in checks:
        # RETURN_LSE is a compile-time trait: asking for LSE must not perturb O.
        o_lse, _ = flydsl_flash_attn_fp8_func(
            q, k, v, **dict(kernel_kw, return_lse=True)
        )
        o_only = flydsl_flash_attn_fp8_func(q, k, v, **plain)
        assert torch.equal(o_lse, o_only), "return_lse must not perturb O"
    if "dead_lse" in checks:
        S, Skv = c["S"], c["Skv"]
        for _ in range(4):
            # Sentinel-filled so "never written" is distinguishable from "wrote NaN".
            lse = torch.full((1, H, S), 1.2345e-7, dtype=torch.float32, device="cuda")
            o, lse = flydsl_flash_attn_fp8_func(
                q, k, v, **dict(kernel_kw, return_lse=True, lse=lse)
            )
            torch.cuda.synchronize()
            dead = lse[:, :, : S - Skv]
            assert (dead == float("-inf")).all(), (
                f"{int((dead != float('-inf')).sum())} of {dead.numel()} "
                f"fully-masked rows are not -inf (NaN={int(torch.isnan(dead).sum())}, "
                f"unwritten={int((dead == 1.2345e-7).sum())})"
            )
            assert torch.isfinite(
                lse[:, :, S - Skv :]
            ).all(), "live rows must be finite"
            assert (
                o[:, : S - Skv].float() == 0
            ).all(), "fully-masked rows of O are zero"
    if "zero_kv" in checks:
        for b, n_kv in enumerate(vl_kv):
            if n_kv:
                continue
            o_b = out[cuq[b] : cuq[b + 1]].float()
            assert (o_b == 0).all(), f"entry {b} has non-zero O"
            _, lse_chk = flydsl_flash_attn_fp8_func(
                q, k, v, **dict(kernel_kw, return_lse=True)
            )
            assert (lse_chk[:, cuq[b] : cuq[b + 1]] == float("-inf")).all()
    if "scale" in checks:
        saved = [d.clone() for d in descales.values()]
        o_def = flydsl_flash_attn_fp8_func(
            q, k, v, **dict(kernel_kw, softmax_scale=None, return_lse=True)
        )
        o_exp = flydsl_flash_attn_fp8_func(
            q, k, v, **dict(kernel_kw, softmax_scale=D**-0.5, return_lse=True)
        )
        assert torch.equal(o_def[0], o_exp[0]) and torch.equal(o_def[1], o_exp[1])
        for d, s in zip(descales.values(), saved):
            assert torch.equal(d, s), "softmax_scale must not mutate descales"
    if "out_returned" in checks:
        fresh = torch.empty_like(out)
        got = flydsl_flash_attn_fp8_func(q, k, v, **dict(kernel_kw, out=fresh))
        torch.cuda.synchronize()
        no_out = flydsl_flash_attn_fp8_func(q, k, v, **kernel_kw)
        assert got is fresh, "out must be returned, not a copy"
        assert torch.equal(fresh, no_out)
    if "graph" in checks:
        graph_kw = dict(kernel_kw, return_lse=True)
        flydsl_flash_attn_fp8_func(q, k, v, **graph_kw)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out, g_lse = flydsl_flash_attn_fp8_func(q, k, v, **graph_kw)
        q_descale = descales["q_descale"]
        saved_q = q_descale.clone()
        for factor in (0.5, 2.0):
            q_descale.mul_(factor)
            graph.replay()
            expected, expected_lse = flydsl_flash_attn_fp8_func(q, k, v, **graph_kw)
            torch.cuda.synchronize()
            assert torch.equal(g_out, expected) and torch.equal(g_lse, expected_lse)
        q_descale.copy_(saved_q)
    return ret


def main():
    arch, build = get_gfx_runtime(), get_gfx()
    if build != arch or arch not in ("gfx1201", "gfx950"):
        aiter.logger.warning(
            "FlyDSL flash attention requires gfx1201 or gfx950; skipping "
            "(build=%s, attached=%s)",
            build,
            arch,
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL flash attention correctness and warm-buffer timing sweep",
    )
    parser.add_argument("--causal", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="gfx950 fp8 case names to sweep (default: all)",
    )
    args = parser.parse_args()
    causals = [bool(c) for c in args.causal]
    if arch == "gfx1201":
        label = "gfx1201 bf16/f16"
        rows = [test_fmha_rdna(*c) for c in RDNA_CASES if c[1] in causals]
    else:
        from aiter.ops.flydsl.unified_attention_kernels import is_flydsl_available

        if not is_flydsl_available(torch.cuda.current_device()):
            aiter.logger.warning("FlyDSL fp8 attention requires FlyDSL; skipping")
            return
        label = "gfx950 fp8"
        selected = args.cases or list(FP8_CASES)
        unknown = [n for n in selected if n not in FP8_CASES]
        if unknown:
            parser.error(f"unknown cases: {unknown}")
        rows = [
            test_fmha_fp8(n, c)
            for n, c in itertools.product(selected, causals)
            if c in FP8_CASES[n]["causal"]
        ]
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", label, df.to_markdown(index=False))
    aiter.logger.info("PASS: all %s cases; all candidate timings non-zero", label)


if __name__ == "__main__":
    main()
