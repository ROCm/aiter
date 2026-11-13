# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FHMoE (fused routed + native MXFP8 shared expert) on the MiniMax-M3.1 TP4 per-rank profile.

Profile: hidden 6144, inter 768, 128 routed MXFP4 experts + a zero dummy row (E=129),
top-4 routed + the shared expert as id 128 (top-5), Swiglu (alpha 1.702, limit 7),
routed weights x2.0 (routed_scaling_factor folded into the routing weights), shared 1.0.

Variants (``-v``):
  a8: GateMode.INTERLEAVE (GUGU routed w13), MXFP8 activations: routed a8w4, shared a8w8.
  a4: GateMode.SEPARATED  (GGUU routed w13), MXFP4 activations: routed a4w4, shared a4w8.

Gates (exact, reduce mode = AITER_FLYDSL_FORCE_REDUCE=1):
  P0  prequantised-xn input == bf16-xn input (a8), routed [:128] view bytes == E=128 shuffle
  G1  FHMoE route rows [n,5,H][:, :4] == routed-only E=128/top-4 route rows (same kernels)
  G2  FHMoE reduce output == bf16(sum_s f32(route_row_s)) in slot order
  G5  reduce output repeatable (4 runs); atomic run-to-run max abs reported
Report only: G3 shared slot vs ATOM's dense MXFP8 MLP (if ATOM importable), G4 final vs
bf16(bf16(2 r) + s), rel L2 vs an f64 reference.
"""

import argparse
import contextlib
import itertools
import os

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight, shuffle_scale, shuffle_weight
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]  # FHMoE is gfx950-only (fh:72)
H, INTER, E, TOPK = 6144, 768, 128, 4
SID = E  # shared expert id = dummy routed row
RSF, LIMIT, ALPHA = 2.0, 7.0, 1.702
FP4 = dtypes.fp4x2
GATE = {"a8": "interleave", "a4": "separated"}
QA = {"a8": dtypes.fp8, "a4": FP4}


# ------------------------------------------------------------------ weights
def _quant_fp4(experts, rows, cols, mag, seed):
    """[experts, rows, cols/2] fp4x2 + e8m0 [experts*rows, cols/32]; chunked bf16 generation."""
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    q = torch.empty((experts, rows, cols // 2), dtype=torch.uint8)
    s = torch.empty((experts * rows, cols // 32), dtype=torch.uint8)
    g = torch.Generator(device="cuda")
    for e in range(0, experts, 16):
        k = min(16, experts - e)
        g.manual_seed(seed + e)
        w = torch.randn((k, rows, cols), dtype=dtypes.bf16, generator=g).mul_(mag)
        qq, ss = quant(w, quant_dtype=FP4)
        q[e : e + k] = qq.view(k, rows, cols // 2).view(torch.uint8)
        s[e * rows : (e + k) * rows] = ss.view(torch.uint8).view(k * rows, -1)
    return q, s


def _quant_fp8(rows, cols, mag, seed):
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn((rows, cols), dtype=dtypes.bf16, generator=g).mul_(mag)
    q, s = per_1x32_f8_scale_f8_quant(w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
    return q.view(rows, cols), s.view(torch.uint8).view(rows, cols // 32)


class RawM3:
    """Unshuffled tensors: routed experts 0..127 + zero dummy row 128 (+ scale 0x7F), shared MXFP8 expert."""

    def __init__(self, seed=123):
        w1, s1 = _quant_fp4(E, 2 * INTER, H, H**-0.5, seed)
        w2, s2 = _quant_fp4(E, H, INTER, INTER**-0.5, seed + 7)
        self.w1 = torch.cat([w1, torch.zeros_like(w1[:1])])
        self.w2 = torch.cat([w2, torch.zeros_like(w2[:1])])
        self.s1 = torch.cat([s1, torch.full_like(s1[: 2 * INTER], 0x7F)])
        self.s2 = torch.cat([s2, torch.full_like(s2[:H], 0x7F)])
        self.sw1, self.ss1 = _quant_fp8(2 * INTER, H, H**-0.5, seed + 11)  # [gate; up]
        self.sw2, self.ss2 = _quant_fp8(H, INTER, INTER**-0.5, seed + 13)


def _mark(t, gu):
    t.is_shuffled = True
    t.gate_up_interleave = gu
    return t


class LayerM3:
    """One layer in the layout of a variant: routed (E=129) + shared, all preshuffled.

    Shared preshuffle follows test_fhmoe._make_shared_weights(interleave=a8):
    a8: shuffle_weight_a16w4 / shuffle_scale_a16w4 (gate/up interleaved);
    a4: shuffle_weight(16,16) / shuffle_scale = the bytes ATOM's dense MLP already holds (weights).
    """

    def __init__(self, raw: RawM3, variant: str):
        gu = variant == "a8"
        e = E + 1
        self.w1 = _mark(moe_shuffle_weight(raw.w1.view(FP4), experts_cnt=e, is_guinterleave=gu, gate_up=True), gu)
        self.w2 = _mark(moe_shuffle_weight(raw.w2.view(FP4), experts_cnt=e, is_guinterleave=gu, gate_up=False), gu)
        s1, s2 = raw.s1.view(dtypes.fp8_e8m0), raw.s2.view(dtypes.fp8_e8m0)
        if gu:
            self.w1s = moe_shuffle_scale(s1, e, is_guinterleave=True, gate_up=True)
            self.w2s = moe_shuffle_scale(s2, e, is_guinterleave=True, gate_up=False)
        else:
            self.w1s, self.w2s = shuffle_scale(s1), shuffle_scale(s2)
        sw1 = raw.sw1.view(1, 2 * INTER, H)
        sw2 = raw.sw2.view(1, H, INTER)
        ss1, ss2 = raw.ss1.view(dtypes.fp8_e8m0), raw.ss2.view(dtypes.fp8_e8m0)
        self.sw1 = _mark(shuffle_weight(sw1, layout=(16, 16), is_guinterleave=gu, gate_up=gu), gu)
        self.sw2 = _mark(shuffle_weight(sw2, layout=(16, 16), is_guinterleave=gu, gate_up=False), gu)
        if gu:
            self.ss1 = shuffle_scale(ss1, experts_cnt=1, is_guinterleave=True, gate_up=True)
            self.ss2 = shuffle_scale(ss2, experts_cnt=1, is_guinterleave=True, gate_up=False)
        else:
            self.ss1, self.ss2 = shuffle_scale(ss1), shuffle_scale(ss2)

    def routed128(self):
        """The first 128 experts as views of the 129-expert tensors (what a 128-expert consumer would bind)."""
        w1 = _mark(self.w1[:E], self.w1.gate_up_interleave)
        w2 = _mark(self.w2[:E], self.w2.gate_up_interleave)
        n1 = self.w1s.numel() * E // (E + 1)
        n2 = self.w2s.numel() * E // (E + 1)
        return w1, w2, self.w1s.view(-1)[:n1].view(-1, self.w1s.shape[-1]), self.w2s.view(-1)[:n2].view(-1, self.w2s.shape[-1])

    def clone(self):
        c = object.__new__(LayerM3)
        for k, v in self.__dict__.items():
            t = v.clone()
            for a in ("is_shuffled", "gate_up_interleave"):
                if hasattr(v, a):
                    setattr(t, a, getattr(v, a))
            setattr(c, k, t)
        return c


# ------------------------------------------------------------------ routing / inputs
def make_routing(n, seed):
    """Same draw as the baseline (softmax(randn) top-4, renormalised); FH weights = [2w, 1], ids = [ids, 128]."""
    g = torch.Generator(device="cuda").manual_seed(1000 * seed + n)
    w, ids = torch.softmax(torch.randn((n, E), generator=g), dim=-1).topk(TOPK, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).float().contiguous()
    ids = ids.to(torch.int32).contiguous()
    w5 = torch.cat([w * RSF, torch.ones_like(w[:, :1])], 1).contiguous()
    ids5 = torch.cat([ids, torch.full_like(ids[:, :1], SID)], 1).contiguous()
    return w, ids, w5, ids5


def make_x(n):
    g = torch.Generator(device="cuda").manual_seed(77 + n)
    return torch.randn((n, H), dtype=dtypes.bf16, generator=g)


def quant_x8(x):
    from aiter.ops.quant import per_1x32_mx_quant_hip

    return per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)


# ------------------------------------------------------------------ calls
@contextlib.contextmanager
def reduce_mode(on):
    prev = os.environ.get("AITER_FLYDSL_FORCE_REDUCE")
    os.environ["AITER_FLYDSL_FORCE_REDUCE"] = "1" if on else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("AITER_FLYDSL_FORCE_REDUCE")
        else:
            os.environ["AITER_FLYDSL_FORCE_REDUCE"] = prev


def fh_call(variant, L: LayerM3, x, w5, ids5, xs=None):
    """FHMoE: routed + shared, one call. xs = e8m0 scales of a prequantised MXFP8 x (a8 only)."""
    kw = dict(a1_scale=xs) if xs is not None else {}
    return fused_moe(x, L.w1, L.w2, w5, ids5, activation=ActivationType.Swiglu, quant_type=QuantType.per_1x32,
                     w1_scale=L.w1s, w2_scale=L.w2s, swiglu_limit=LIMIT, gate_mode=GATE[variant], dtype=dtypes.bf16,
                     shared_w1=L.sw1, shared_w2=L.sw2, shared_w1_scale=L.ss1, shared_w2_scale=L.ss2,
                     shared_expert_id=SID, **kw)


def routed_call(variant, w1, w2, w1s, w2s, x, w, ids, xs=None):
    """Plain routed-only fused_moe (no shared expert); kernels come from whatever row is active."""
    kw = dict(a1_scale=xs) if xs is not None else {}
    return fused_moe(x, w1, w2, w, ids, activation=ActivationType.Swiglu, quant_type=QuantType.per_1x32,
                     w1_scale=w1s, w2_scale=w2s, swiglu_limit=LIMIT, gate_mode=GATE[variant],
                     quant_dtype_a=QA[variant], dtype=dtypes.bf16, **kw)


def fh_row(variant, n):
    """The tuned_fhmoe.csv row (kernelName1/2, block_m) this request resolves to."""
    from aiter.fhmoe import _fhmoe_config_file
    from aiter.fused_moe import get_padded_M

    df = pd.read_csv(_fhmoe_config_file())
    df = df[(df.model_dim == H) & (df.inter_dim == INTER) & (df.expert == E + 1) & (df.topk == TOPK + 1)
            & (df.q_dtype_a == str(QA[variant])) & (df.token == get_padded_M(n))]
    assert len(df) == 1, f"expected one M3.1 {variant} row for n={n}, got {len(df)}"
    return df.iloc[0]


@contextlib.contextmanager
def pinned_routed_row(variant, n, kn1, kn2, block_m=32, experts=E, topk=TOPK):
    """Pin (kn1, kn2) for the plain fused_moe E/top-k key in-process; restores any production row."""
    import aiter.fused_moe as FM
    from aiter.fused_moe import get_padded_M
    from aiter.jit.utils.chip_info import get_cu_num

    if FM.cfg_2stages is None:
        FM.get_2stage_cfgs(1, H, INTER, E, TOPK, dtypes.bf16, dtypes.bf16, FP4, QuantType.per_1x32, True,
                           ActivationType.Swiglu, False, 0, 0)
    key = ("gfx950", get_cu_num(), get_padded_M(n), H, INTER, experts, topk, "ActivationType.Swiglu", "torch.bfloat16",
           str(QA[variant]), str(FP4), "QuantType.per_1x32", 1, 0)
    old = FM.cfg_2stages[0].get(key)
    FM.cfg_2stages[0][key] = dict(block_m=block_m, ksplit=0, us1=0.0, kernelName1=kn1, err1="0.0%", us2=0.0,
                                  kernelName2=kn2, err2="0.0%", us=0.0, run_1stage=0, xbf16=0, flat=0, tflops=0, bw=0,
                                  _tag="")
    FM.get_2stage_cfgs.cache_clear()
    try:
        yield
    finally:
        if old is None:
            FM.cfg_2stages[0].pop(key, None)
        else:
            FM.cfg_2stages[0][key] = old
        FM.get_2stage_cfgs.cache_clear()


@contextlib.contextmanager
def capture_route_rows(store):
    """Record the [tokens*topk, H] route-row buffer that reduce mode hands to _run_moe_reduction."""
    from aiter.ops.flydsl import moe_kernels as MK

    orig = MK._run_moe_reduction

    def spy(target, out, token_num, topk, model_dim, *a, **k):
        store.append(target.reshape(-1)[: token_num * topk * model_dim].view(token_num, topk, model_dim).clone())
        return orig(target, out, token_num, topk, model_dim, *a, **k)

    MK._run_moe_reduction = spy
    try:
        yield
    finally:
        MK._run_moe_reduction = orig


# ------------------------------------------------------------------ references
def _dequant(q, s):
    if q.dtype in (FP4, torch.uint8) and q.shape[-1] * 2 == s.shape[-1] * 32:
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
        b = q.view(torch.uint8)
        v = torch.stack([lut[(b & 0xF).long()], lut[(b >> 4).long()]], -1).flatten(-2)
    else:
        v = q.view(dtypes.fp8).float()
    sc = torch.exp2(s.view(torch.uint8).float() - 127.0)
    return (v.view(*v.shape[:-1], -1, 32) * sc.view(*v.shape[:-1], -1, 1)).flatten(-2)


def _swi(g, u):
    g = g.clamp(max=LIMIT)
    u = u.clamp(-LIMIT, LIMIT)
    return g * torch.sigmoid(ALPHA * g) * (u + 1.0)


@torch.no_grad()
def ref64(raw: RawM3, x, w, ids):
    """f64: exact bf16 x, dequantised weights, no activation quantisation. -> (routed (unscaled), shared)."""
    xd, n = x.double(), x.shape[0]
    s1, s2 = raw.s1.view(E + 1, 2 * INTER, -1), raw.s2.view(E + 1, H, -1)
    routed = torch.zeros((n, H), dtype=torch.float64)
    for e in torch.unique(ids).tolist():
        t, k = (ids == e).nonzero(as_tuple=True)
        h = xd[t] @ _dequant(raw.w1[e], s1[e]).double().T
        a = _swi(h[:, :INTER], h[:, INTER:])
        routed.index_add_(0, t, (a @ _dequant(raw.w2[e], s2[e]).double().T) * w[t, k].double()[:, None])
    h = xd @ _dequant(raw.sw1, raw.ss1).double().T
    shared = _swi(h[:, :INTER], h[:, INTER:]) @ _dequant(raw.sw2, raw.ss2).double().T
    return routed, shared


def rel_l2(a, ref):
    a, ref = a.double(), ref.double()
    return float(((a - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())


def ulp_stats(a, b):
    """% of differing bf16 elements and the max distance in bf16 ulps (ordered-int metric) over |b| >= 1e-2
    (elements near zero, e.g. the all-zero / denormal P0 rows, make ulp distances meaningless)."""
    ia = a.contiguous().view(torch.int16).int()
    ib = b.contiguous().view(torch.int16).int()
    oa = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ob = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    d = (oa - ob).abs()
    big = b.float().abs() >= 1e-2
    return float((d > 0).float().mean() * 100), int(d[big].max()) if big.any() else 0


def atom_shared(raw: RawM3, xq, xs):
    """ATOM's dense native MXFP8 shared MLP (MiniMaxM3MLP math) on the same quantised x; None if ATOM is absent."""
    try:
        from atom.model_ops.blockscale import native_quant_linear
        from atom.model_ops.swiglu_oai import swiglu_oai_split
    except ImportError:
        return None
    w1 = shuffle_weight(raw.sw1, layout=(16, 16))
    w2 = shuffle_weight(raw.sw2, layout=(16, 16))
    gu = native_quant_linear(xq, w1, raw.ss1.view(dtypes.fp8_e8m0), x_scale=xs, weight_group_rows=1,
                             weight_preshuffled=True, dtype=dtypes.bf16)
    a = swiglu_oai_split(gu, alpha=ALPHA, beta=1.0, limit=LIMIT)
    return native_quant_linear(a, w2, raw.ss2.view(dtypes.fp8_e8m0), x_scale=None, weight_group_rows=1,
                               weight_preshuffled=True, dtype=dtypes.bf16)


# ------------------------------------------------------------------ the test
_CACHE = {}


def _layers(variant):
    if "raw" not in _CACHE:
        _CACHE["raw"] = RawM3()
    if variant not in _CACHE:
        _CACHE[variant] = LayerM3(_CACHE["raw"], variant)
    return _CACHE["raw"], _CACHE[variant]


@benchmark()
def test_p0_layout(variant):
    """Byte checks: 129-expert tensors viewed [:128] == the E=128 shuffle (what mono / a 128-expert consumer binds);
    a4 shared weight == ATOM's preshuffled dense-MLP weight (zero-copy candidate)."""
    raw, L = _layers(variant)
    gu = variant == "a8"
    w1, w2, w1s, w2s = L.routed128()
    r1 = moe_shuffle_weight(raw.w1[:E].view(FP4), experts_cnt=E, is_guinterleave=gu, gate_up=True)
    r2 = moe_shuffle_weight(raw.w2[:E].view(FP4), experts_cnt=E, is_guinterleave=gu, gate_up=False)
    s1, s2 = raw.s1[: E * 2 * INTER].view(dtypes.fp8_e8m0), raw.s2[: E * H].view(dtypes.fp8_e8m0)
    if gu:
        q1 = moe_shuffle_scale(s1, E, is_guinterleave=True, gate_up=True)
        q2 = moe_shuffle_scale(s2, E, is_guinterleave=True, gate_up=False)
    else:
        q1, q2 = shuffle_scale(s1), shuffle_scale(s2)
    eq = lambda a, b: a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    ret = {"w13[:128] bytes": eq(w1, r1), "w2[:128] bytes": eq(w2, r2),
           "w13 scale[:128] bytes": eq(w1s, q1), "w2 scale[:128] bytes": eq(w2s, q2)}
    atom_w1 = shuffle_weight(raw.sw1, layout=(16, 16))
    ret["shared w13 == ATOM preshuffled bytes"] = eq(L.sw1.view(2 * INTER, H), atom_w1)
    ret["shared w13 scale shape"] = str(tuple(L.ss1.shape))
    return ret


@benchmark()
def test_fhmoe_m3(n, seed, variant):
    raw, L = _layers(variant)
    x = make_x(n)
    if n >= 4:  # P0 edge rows: all-zero, denormal-scale, amax exactly a power of two
        x[1] = 0
        x[2] = x[2] * 1e-38
        x[3, :32] = torch.tensor([4.0] + [1.0] * 31)
    w, ids, w5, ids5 = make_routing(n, seed)
    xq = xs = None
    if variant == "a8":
        xq, xs = quant_x8(x)
    xin = xq if variant == "a8" else x
    ret = {"gfx": get_gfx()}
    row = fh_row(variant, n)
    kn1, kn2 = row["kernelName1"], row["kernelName2"]
    ret["kn1"] = kn1

    # ---- reduce mode: G1 / G2 / G5 / P0
    rows_fh, rows_r = [], []
    with reduce_mode(True):
        with capture_route_rows(rows_fh):
            out_r = fh_call(variant, L, xin, w5, ids5, xs)
        reps = [fh_call(variant, L, xin, w5, ids5, xs) for _ in range(3)]
        w1, w2, w1s, w2s = L.routed128()
        with pinned_routed_row(variant, n, kn1, kn2), capture_route_rows(rows_r):
            r_only = routed_call(variant, w1, w2, w1s, w2s, xin, w * RSF, ids, xs)
        if variant == "a8":
            out_bf16_in = fh_call(variant, L, x, w5, ids5)
    torch.cuda.synchronize()
    fh_rows = rows_fh[-1]
    ret["G1 routed rows bit-exact"] = bool(torch.equal(fh_rows[:, :TOPK], rows_r[-1]))
    acc = torch.zeros((n, H), dtype=torch.float32)
    for k in range(TOPK + 1):
        acc = acc + fh_rows[:, k].float()
    ret["G2 combine bit-exact"] = bool(torch.equal(out_r, acc.to(dtypes.bf16)))
    ret["G5 reduce repeatable"] = all(torch.equal(out_r, r) for r in reps)
    if variant == "a8":
        ret["P0 xq-in == bf16-in"] = bool(torch.equal(out_r, out_bf16_in))

    # ---- atomic mode: run-to-run spread
    with reduce_mode(False):
        outs_a = [fh_call(variant, L, xin, w5, ids5, xs) for _ in range(4)]
    ret["atomic r2r max abs"] = max(float((o.float() - outs_a[0].float()).abs().max()) for o in outs_a)
    ret["atomic vs reduce max abs"] = float((outs_a[0].float() - out_r.float()).abs().max())

    # ---- report-only numerics
    routed64, shared64 = ref64(raw, x, w, ids)
    total64 = RSF * routed64 + shared64
    ret["relL2 FH vs f64"] = rel_l2(out_r, total64)
    ret["relL2 shared slot vs f64"] = rel_l2(fh_rows[:, TOPK], shared64)
    ret["relL2 routed (2r) vs f64"] = rel_l2(acc - fh_rows[:, TOPK].float(), RSF * routed64)
    xq8, xs8 = (xq, xs) if variant == "a8" else quant_x8(x)
    s_atom = atom_shared(raw, xq8, xs8)
    if s_atom is not None:
        pct, mu = ulp_stats(fh_rows[:, TOPK], s_atom)
        ret["G3 slot4 vs ATOM %diff"] = pct
        ret["G3 slot4 vs ATOM max ulp"] = mu
        ret["G3 slot4 vs ATOM relL2"] = rel_l2(fh_rows[:, TOPK], s_atom.float())
        ret["relL2 ATOM shared vs f64"] = rel_l2(s_atom, shared64)
        u = (r_only.float() + s_atom.float()).to(dtypes.bf16)
        pct, mu = ulp_stats(out_r, u)
        ret["G4 final vs bf16(2r)+s %diff"] = pct
        ret["G4 max ulp"] = mu

    # ---- warm (single layer) timing, both modes; the cold 16-layer numbers come from the bench script
    for mode in (False, True):
        with reduce_mode(mode):
            _, us = run_perftest(fh_call, variant, L, xin, w5, ids5, xs, num_iters=50, num_warmup=5)
        ret[f"FH {'reduce' if mode else 'atomic'} warm us"] = us
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FHMoE M3.1 test unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__)
    parser.add_argument("-n", "--tokens", type=int, nargs="*", default=[1, 4, 16, 32, 64, 96, 128, 192, 256])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("-v", "--variant", nargs="*", default=["a8", "a4"], choices=["a8", "a4"])
    args = parser.parse_args()
    df = pd.DataFrame([test_p0_layout(v) for v in args.variant])
    aiter.logger.info("FHMoE M3.1 layout byte checks (markdown):\n%s", df.to_markdown(index=False))
    for variant in args.variant:
        df = pd.DataFrame([test_fhmoe_m3(n, s, variant) for n, s in itertools.product(args.tokens, args.seeds)])
        aiter.logger.info("FHMoE M3.1 %s summary (markdown):\n%s", variant, df.to_markdown(index=False))


if __name__ == "__main__":
    main()
