# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`gated_mha_bwd`: the backward of the D_QK = 192 / D_V = 128 attention on the OPUS
gfx950 kernels, with the output gate, out = attention(q, k, v) * sigmoid(gate), and
without it.

Each case runs the OPUS forward, gates its output in fp32 where the case has a gate, and
calls the op on that output and LSE. dq, dk, dv and dgate are compared with fp32
autograd of a torch attention (`_ref_bwd`): the relative L2 error must stay within
PARITY_REL_L2 over each output and over each of its heads. Each row (token, head) is
compared with `_faithful_bwd`, which forms D from the bf16 `out` as the kernels do: its
error must stay within LOCAL_ROW_ERR of the row's norm, floored at a hundredth of its
head's RMS row norm. A wrong row or head is thus not diluted.

The cases cover sequence lengths from one token, around the kernels' tile and block
edges, to 16384 tokens at 96 heads (65536 tokens and batches of 131072 with
AITER_GATED_MHA_BWD_FULL_SIZE=1), equal-length batches and varlen mixes, key sequences
of other lengths than the queries', sequences without queries and no query at all, head
counts around the scheduling groups of 16 and 32 heads that the kernels deal over the 8
XCDs, causal and unmasked attention, other softmax scales, max_seqlen bounds above the
lengths, views of every input (q, k, v, out, dout, gate, lse and cu_seqlens: strided,
sliced, unaligned, broadcast and overlapping), token strides at and past the kernels'
2^24 limit, gate and dout as [T, H * 128] and [T, H, 128], KV heads shared by several
query heads (GQA / MQA), run-to-run determinism, torch.compile with fullgraph=True, the
ops' schemas, the exports and the input checks.
"""

import itertools
import os
import sys

import pytest
import torch

import aiter
import aiter.ops.opus._gated_mha_bwd as _OP_MODULE
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import fmha_fwd_bf16_opus_varlen_fwd
from aiter.ops.opus import gated_mha_bwd

D_QK = 192
D_V = 128
SCALE = D_QK**-0.5
# Bound on the relative L2 error of the bf16 gradients against fp32 autograd.
PARITY_REL_L2 = 2e-2
# Bound on a row's (token, head) L2 error against `_faithful_bwd`, over the row's norm
# floored at a hundredth of its head's RMS row norm.
LOCAL_ROW_ERR = 0.1
# The cases of 32768 tokens and more need about 100 GB for their fp32 references.
FULL_SIZE = os.environ.get("AITER_GATED_MHA_BWD_FULL_SIZE", "0") == "1"


def _is_gfx950():
    try:
        return get_gfx() == "gfx950"
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _is_gfx950(), reason="gfx950 only")

_OUTPUTS = ("dq", "dk", "dv", "dgate")
# Largest relative L2 error of each output over the module's cases, with its case.
_worst = {}


@pytest.fixture(autouse=True, scope="module")
def _error_summary():
    yield
    print(f"\naiter: {os.path.dirname(aiter.__file__)}")
    print("largest relative L2 errors against fp32:")
    for name, (err, case) in _worst.items():
        print(f"  {name:22s} {err:.3e}  {case}")


def _case_ids(cases):
    """b<sequences>_s<length>_h<heads>, or the shortest and longest length of a varlen
    mix."""
    ids = []
    for seqlens, heads in cases:
        lo, hi = min(seqlens), max(seqlens)
        span = f"s{hi}" if lo == hi else f"min{lo}_max{hi}"
        ids.append(f"b{len(seqlens)}_{span}_h{heads}")
    return ids


def _inputs(seqlens, heads, seed=0):
    """Random bf16 q, k, v, gate and dout of `heads` heads, with the sequences' bounds."""
    torch.manual_seed(seed)
    cu = torch.tensor(
        [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device="cuda"
    )
    t = int(cu[-1])
    bf = torch.bfloat16
    q = torch.randn(t, heads, D_QK, device="cuda", dtype=bf)
    k = torch.randn(t, heads, D_QK, device="cuda", dtype=bf)
    v = torch.randn(t, heads, D_V, device="cuda", dtype=bf)
    gate = 2 * torch.randn(t, heads * D_V, device="cuda", dtype=bf)
    dout = torch.randn(t, heads * D_V, device="cuda", dtype=bf)
    return cu, max(seqlens), q, k, v, gate, dout


def _ref_attention(q, k, v, gate, cu, causal, cu_k=None, scale=SCALE):
    """fp32 per-sequence attention: gated output [T, H, D_V] and LSE [H, T]. `cu_k`
    bounds the keys' sequences where they differ from the queries'; the causal mask is
    aligned to the bottom right, and a query that sees no key has a zero output."""
    q, k, v = (x.float() for x in (q, k, v))
    cu_k = cu if cu_k is None else cu_k
    heads = q.shape[1]
    out = torch.empty(q.shape[0], heads, D_V, device=q.device)
    lse = torch.empty(heads, q.shape[0], device=q.device)
    for b in range(cu.numel() - 1):
        rows = slice(int(cu[b]), int(cu[b + 1]))
        keys = slice(int(cu_k[b]), int(cu_k[b + 1]))
        sq, sk = rows.stop - rows.start, keys.stop - keys.start
        scores = torch.einsum("qhd,khd->hqk", q[rows], k[keys]) * scale
        mask = torch.zeros(sq, sk, dtype=torch.bool, device=q.device)
        if causal:
            mask = torch.ones_like(mask).triu(1 + sk - sq)
        blind = mask.all(-1, keepdim=True)
        scores = scores.masked_fill(mask & ~blind, float("-inf"))
        lse[:, rows] = torch.logsumexp(scores, dim=-1).masked_fill(
            blind[:, 0], float("-inf")
        )
        p = scores.softmax(-1).masked_fill(blind, 0.0)
        out[rows] = torch.einsum("hqk,khd->qhd", p, v[keys])
    if gate is not None:
        out = out * gate.float().view(out.shape).sigmoid()
    return out, lse


def _ref_bwd(q, k, v, gate, dout, cu, causal, cu_k=None, scale=SCALE):
    """fp32 autograd of the gated attention: d(q), d(k), d(v), d(gate); without a gate,
    the first three."""
    qf, kf, vf = (x.detach().float().requires_grad_() for x in (q, k, v))
    gf = None if gate is None else gate.detach().float().requires_grad_()
    out, _ = _ref_attention(qf, kf, vf, gf, cu, causal, cu_k, scale)
    out.backward(dout.float().view(out.shape))
    grads = (qf.grad, kf.grad, vf.grad)
    return grads if gate is None else (*grads, gf.grad)


def _rel_err(a, ref):
    ref = ref.float()
    return ((a.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _assert_equal(a, b, what):
    differ = [
        n for n, x, y in zip(_OUTPUTS, a, b) if x is not None and not torch.equal(x, y)
    ]
    assert not differ, f"{what}: {differ} differ"


def _fwd_bwd(
    q,
    k,
    v,
    gate,
    dout,
    cu,
    smax,
    causal,
    cu_k=None,
    smax_k=None,
    scale=SCALE,
    views=None,
    bounds=None,
    seen=None,
):
    """dq, dk, dv and dgate (None without a gate) of `gated_mha_bwd`, on the OPUS
    forward's output, gated in fp32, and its LSE. `views` maps "out", "lse", "cu" and
    "cu_k" to functions that give the op a view of that argument; `bounds` replaces
    (max_seqlen_q, max_seqlen_k) of the op's call; `seen`, a dict, gets the `out` the op
    was given."""
    cu_k, smax_k = cu if cu_k is None else cu_k, smax if smax_k is None else smax_k
    out, lse = fmha_fwd_bf16_opus_varlen_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        scale,
        causal,
        cu,
        cu_k,
        smax,
        smax_k,
        return_lse=True,
    )
    if gate is not None:
        out = (out.float() * gate.float().reshape(out.shape).sigmoid()).to(out.dtype)
    args = {"out": out, "lse": lse, "cu": cu, "cu_k": cu_k}
    for name, make in (views or {}).items():
        args[name] = make(args[name])
    if seen is not None:
        seen["out"] = args["out"]
    bq, bk = (smax, smax_k) if bounds is None else bounds
    return gated_mha_bwd(
        q,
        k,
        v,
        args["out"],
        dout,
        args["lse"],
        args["cu"],
        args["cu_k"],
        bq,
        bk,
        scale,
        causal,
        gate,
    )


def _as_rows(x):
    """`x` as [T, heads, d]: a gradient of q, k or v as it is, one of the gate by D_V."""
    return x.reshape(x.shape[0], -1, x.shape[-1] if x.dim() == 3 else D_V).float()


def _head_error(a, r):
    """The largest relative L2 error of a head."""
    a, r = _as_rows(a), _as_rows(r)
    diff, ref = (a - r).pow(2).sum(-1), r.pow(2).sum(-1)  # [T, heads]
    return (diff.sum(0) / ref.sum(0).clamp_min(1e-30)).sqrt().max().item()


def _row_error(a, r):
    """The largest L2 error of a row (token, head) over its reference's norm, floored at
    a hundredth of its head's RMS row norm (a row of 0 stays 0, as dq of a query that
    sees one key)."""
    a, r = _as_rows(a), _as_rows(r)
    diff, ref = (a - r).pow(2).sum(-1), r.pow(2).sum(-1)  # [T, heads]
    if not diff.numel():
        return 0.0
    floor = torch.maximum(ref, 1e-4 * ref.mean(0)).clamp_min(1e-30)
    return (diff / floor).sqrt().max().item()


def _faithful_bwd(q, k, v, out, gate, dout, cu, causal, heads=None, scale=SCALE):
    """fp32 gradients from the op's own inputs as the kernels form them: D = rowsum(dout
    * out) of the bf16 `out` the op is given, and dO = dout * sigmoid(gate) rounded to
    bf16. Against fp32 autograd a row whose dS cancels (a query that sees few keys)
    differs by the rounding of `out` in D; against this it carries only the op's own
    rounding, so it bounds each row. One cu_seqlens for queries and keys and as many KV
    heads as query heads: dq, dk, dv and dgate (None without a gate) of `heads` (all by
    default)."""
    t, h = q.shape[:2]
    heads = list(range(h)) if heads is None else list(heads)
    dout3, out3 = dout.reshape(t, h, D_V).float(), out.reshape(t, h, D_V).float()
    dgate = None
    if gate is None:
        d_o = dout3
    else:
        s = gate.reshape(t, h, D_V).float().sigmoid()
        d_o = (dout3 * s).to(torch.bfloat16).float()
        dgate = (dout3 * out3 * (1.0 - s))[:, heads]
    d_row = (dout3 * out3).sum(-1)  # [T, H]
    n = len(heads)
    dq = torch.zeros(t, n, D_QK, device=q.device)
    dk = torch.zeros(t, n, D_QK, device=q.device)
    dv = torch.zeros(t, n, D_V, device=q.device)
    chunk = 2048
    for start, stop in itertools.pairwise(cu.tolist()):
        s_len = stop - start
        for i, hh in enumerate(heads):
            qh, kh = q[start:stop, hh].float(), k[start:stop, hh].float()
            vh, doh = v[start:stop, hh].float(), d_o[start:stop, hh]
            for c0 in range(0, s_len, chunk):
                c1 = min(c0 + chunk, s_len)
                sc = qh[c0:c1] @ kh.T * scale
                if causal:
                    rows = torch.arange(c0, c1, device=q.device)[:, None]
                    cols = torch.arange(s_len, device=q.device)[None]
                    sc = sc.masked_fill(cols > rows, float("-inf"))
                p = (sc - sc.logsumexp(-1, keepdim=True)).exp()
                dp = doh[c0:c1] @ vh.T
                ds = p * (dp - d_row[start + c0 : start + c1, hh, None]) * scale
                dq[start + c0 : start + c1, i] = ds @ kh
                dk[start:stop, i] += ds.T @ qh[c0:c1]
                dv[start:stop, i] += p.T @ doh[c0:c1]
    return dq, dk, dv, dgate


def _assert_parity(ours, ref, names=_OUTPUTS, scales=None, faithful=None):
    """Relative L2 error of each of `ours` against `ref` within PARITY_REL_L2, over each
    output and each of its heads; with `faithful` (`_faithful_bwd`), every row within
    LOCAL_ROW_ERR of it. `scales` replaces the reference norms as the errors'
    denominators, and leaves out the head bounds."""
    case = os.environ.get("PYTEST_CURRENT_TEST", "").split("::")[-1].split(" ")[0]
    failed = []
    for i, (name, a, r) in enumerate(zip(names, ours, ref)):
        a = a.reshape(r.shape)
        head = row = 0.0
        if scales is None:
            err = _rel_err(a, r)
            head = _head_error(a, r)
        else:
            err = ((a.float() - r.float()).norm() / scales[i]).item()
        if faithful is not None and faithful[i] is not None:
            row = _row_error(a, faithful[i].reshape(r.shape))
        print(f"  {name:22s} rel_err {err:.3e}  head {head:.3e}  row {row:.3e}")
        for key, value in ((name, err), (f"{name} head", head), (f"{name} row", row)):
            if value > _worst.get(key, (0.0, ""))[0]:
                _worst[key] = (value, case)
        # Not `err > bound`, which a NaN would pass.
        if not (
            err <= PARITY_REL_L2 and head <= PARITY_REL_L2 and row <= LOCAL_ROW_ERR
        ):
            failed.append(name)
    assert (
        not failed
    ), f"errors above {PARITY_REL_L2} (output, head) or {LOCAL_ROW_ERR} (row): {failed}"


def _check(seqlens, heads, causal):
    cu, smax, q, k, v, gate, dout = _inputs(seqlens, heads)
    seen = {}
    ours = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal, seen=seen)
    _assert_parity(
        ours,
        _ref_bwd(q, k, v, gate, dout, cu, causal),
        faithful=_faithful_bwd(q, k, v, seen["out"], gate, dout, cu, causal),
    )


# Around the 32-row tiles, the 48 rows of a wave and the 192-row blocks of the dK / dV
# pass, and around other powers of two.
_EDGE_LENS = [n + d for n in (32, 48, 64, 96, 128, 192, 256) for d in (-1, 0, 1)]
_EDGE_LENS += [383, 385, 511, 513]
# The dQ pass's blocks are of 256 queries, a sequence's last ones of 192: lengths where
# the split of a sequence changes.
_EDGE_LENS += [n + d for n in (320, 448, 576, 640, 832) for d in (0, 1)]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 9, 33])
@pytest.mark.parametrize("seqlen", _EDGE_LENS)
def test_gated_mha_bwd_edge_lengths(seqlen, heads, causal):
    """One short sequence, its length at or next to a tile or block edge."""
    _check([seqlen], heads, causal)


def _dp_term_norms(q, k, v, gate, dout, cu, causal):
    """Norms of dq and dk in fp32 with dS = P dP for dS = P (dP - D): the half of each
    gradient that its D half cancels."""
    q, k, v = q.float(), k.float(), v.float()
    d_o = dout.float().view(v.shape) * gate.float().view(v.shape).sigmoid()
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    for b in range(cu.numel() - 1):
        rows = slice(int(cu[b]), int(cu[b + 1]))
        s = rows.stop - rows.start
        scores = torch.einsum("qhd,khd->hqk", q[rows], k[rows]) * SCALE
        if causal:
            mask = torch.ones(s, s, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        ds = scores.softmax(-1) * torch.einsum("qhd,khd->hqk", d_o[rows], v[rows])
        dq[rows] = torch.einsum("hqk,khd->qhd", ds, k[rows]) * SCALE
        dk[rows] = torch.einsum("hqk,qhd->khd", ds, q[rows]) * SCALE
    return dq.norm(), dk.norm()


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("seqlen", [1, 2])
def test_gated_mha_bwd_one_and_two_tokens(seqlen, batch, causal):
    """Sequences of one or two tokens. A query that sees one key has P = 1 and dS = P
    (dP - D) = 0, and two keys leave dS = +-P0 P1 (dP0 - dP1): dq and dk are zero or
    what is left of two terms that cancel, while the kernels' D, the rowsum of dout
    times the bf16 `out`, carries that output's rounding. Their errors against fp32
    autograd are therefore measured against the dP half of each gradient instead of the
    reference's own norm; dv and dgate do not cancel and are compared as usual."""
    cu, smax, q, k, v, gate, dout = _inputs([seqlen] * batch, 9)
    dq, dk, dv, dgate = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal)
    ref_dq, ref_dk, ref_dv, ref_dgate = _ref_bwd(q, k, v, gate, dout, cu, causal)
    _assert_parity(
        (dq, dk),
        (ref_dq, ref_dk),
        ("dq / dP half", "dk / dP half"),
        _dp_term_norms(q, k, v, gate, dout, cu, causal),
    )
    _assert_parity((dv, dgate), (ref_dv, ref_dgate), ("dv", "dgate"))


# Up to 20 dQ blocks and 27 dK / dV blocks a head, lengths off the 32-row tiles
# included, from a partial scheduling group of heads to several whole ones.
_MEDIUM = [
    ([1000], 24),
    ([2048], 24),
    ([3000], 128),
    ([4096], 96),
    ([4097], 24),
    ([5000], 24),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _MEDIUM, ids=_case_ids(_MEDIUM))
def test_gated_mha_bwd_medium_lengths(seqlens, heads, causal):
    _check(seqlens, heads, causal)


# Few heads keep the fp32 reference affordable: H * s * s * 4 bytes a temporary.
_LARGE = [
    ([8192], 1),
    ([8192], 4),
    ([8192], 16),
    ([8192, 8192], 3),
    ([16384], 1),
    ([16384], 2),
    ([16384], 4),
    ([16384, 8192, 4096, 2048], 2),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _LARGE, ids=_case_ids(_LARGE))
def test_gated_mha_bwd_large(seqlens, heads, causal):
    _check(seqlens, heads, causal)


def _ref_bwd_per_head(q, k, v, gate, dout, cu, causal, heads):
    """`_ref_bwd` one sequence and one head at a time, which bounds its temporaries to s
    * s * 4 bytes each: d(q), d(k), d(v) and d(gate) of `heads` as [T, len(heads), ...].
    """
    t, h = q.shape[:2]
    gate, dout = gate.view(t, h, -1), dout.view(t, h, -1)
    bounds = cu.tolist()
    per_seq = []
    for start, stop in itertools.pairwise(bounds):
        rows = slice(start, stop)
        cu_seq = torch.tensor([0, stop - start], dtype=cu.dtype, device=cu.device)
        dq, dk, dv, dgate = zip(
            *(
                _ref_bwd(
                    q[rows, i : i + 1],
                    k[rows, i : i + 1],
                    v[rows, i : i + 1],
                    gate[rows, i],
                    dout[rows, i],
                    cu_seq,
                    causal,
                )
                for i in heads
            )
        )
        per_seq.append(
            (
                torch.cat(dq, 1),
                torch.cat(dk, 1),
                torch.cat(dv, 1),
                torch.stack(dgate, 1),
            )
        )
    return tuple(torch.cat(x) for x in zip(*per_seq))


# The first and last heads of scheduling groups and a few in between.
_SOME_HEADS = (0, 1, 15, 16, 47, 64, 80, 95)
# (seqlens, heads, heads the fp32 reference is computed for; None: all). The K3 head
# count from 8192 tokens to batches of 131072, where q and k and their gradients are 4.5
# GiB each, and its share of a GPU at tensor parallelism 4 and 8 (24 and 12 heads).
_FULL_SIZE = [
    ([8192], 96, None),
    ([8192], 24, None),
    ([8192], 12, None),
    ([16384], 96, None),
    ([16384], 24, None),
    ([16384], 12, None),
    ([8192] * 4, 64, None),
    ([32768], 96, None),
    ([65536], 96, _SOME_HEADS),
    ([32768] * 4, 96, _SOME_HEADS),
    ([65536] * 2, 96, _SOME_HEADS),
]


def _full_size_param(case, case_id):
    """The case, skipped from 32768 tokens on unless AITER_GATED_MHA_BWD_FULL_SIZE=1."""
    big = max(case[0]) >= 32768
    skip = pytest.mark.skipif(
        big and not FULL_SIZE, reason="AITER_GATED_MHA_BWD_FULL_SIZE"
    )
    return pytest.param(*case, id=case_id, marks=skip)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlens,heads,ref_heads",
    [
        _full_size_param(case, case_id)
        for case, case_id in zip(_FULL_SIZE, _case_ids([c[:2] for c in _FULL_SIZE]))
    ],
)
def test_gated_mha_bwd_full_size(seqlens, heads, ref_heads, causal):
    """Production-size problems against fp32 autograd taken one head at a time, on all
    heads or, for the largest, on `_SOME_HEADS`."""
    cu, smax, q, k, v, gate, dout = _inputs(seqlens, heads)
    seen = {}
    ours = list(_fwd_bwd(q, k, v, gate, dout, cu, smax, causal, seen=seen))
    ours[3] = ours[3].view(q.shape[0], heads, -1)
    idx = list(range(heads) if ref_heads is None else ref_heads)
    ref = _ref_bwd_per_head(q, k, v, gate, dout, cu, causal, idx)
    faithful = _faithful_bwd(q, k, v, seen["out"], gate, dout, cu, causal, idx)
    _assert_parity([x[:, idx] for x in ours], ref, faithful=faithful)


# Equal-length batches and varlen mixes of 2 to 32 sequences: one-token sequences
# between longer ones, every edge length at an unaligned token offset, and one long
# sequence with many short ones (its blocks are dispatched with theirs).
_BATCHES = [
    ([513] * 2, 96),
    ([1000] * 4, 96),
    ([257] * 16, 96),
    ([512] * 32, 96),
    ([1, 1000], 96),
    ([700, 1, 1900, 300], 96),
    ([1, 2, *_EDGE_LENS], 96),
    ([4096] + [17] * 15, 96),
    ([1] * 7 + [2500] + [1] * 8, 24),
    ([100, 3000] + [64] * 14, 24),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _BATCHES, ids=_case_ids(_BATCHES))
def test_gated_mha_bwd_batches(seqlens, heads, causal):
    _check(seqlens, heads, causal)


# Whole and partial scheduling groups (16 heads, 32 with a causal mask), and heads per
# XCD from none to several.
_HEAD_COUNTS = [1, 2, 3, 7, 8, 9, 12, 16, 17, 24, 33, 96, 128]
# Several blocks of both passes a sequence, with a one-token sequence in between.
_SEQLENS = [600, 1, 257, 1000]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "seqlens", [_SEQLENS, [1024] * 2], ids=["b4_min1_max1000", "b2_s1024"]
)
@pytest.mark.parametrize("heads", _HEAD_COUNTS)
def test_gated_mha_bwd_head_counts(heads, seqlens, causal):
    _check(seqlens, heads, causal)


def _within(x, front):
    """A view equal to `x`: the last dim of a wider tensor from element `front` on."""
    n = x.shape[-1]
    wide = torch.randn(*x.shape[:-1], front + n + 8, device=x.device, dtype=x.dtype)
    wide[..., front : front + n] = x
    return wide[..., front : front + n]


def _every_other(x, dim):
    """A view equal to `x`: every other index along `dim` of a tensor twice as long."""
    shape = list(x.shape)
    shape[dim] *= 2
    index = [slice(None)] * x.dim()
    index[dim] = slice(1, None, 2)
    if x.is_floating_point():
        wide = torch.randn(shape, device=x.device, dtype=x.dtype)
    else:
        wide = torch.full(shape, -7, device=x.device, dtype=x.dtype)
    wide[tuple(index)] = x
    return wide[tuple(index)]


def _shifted(x, front):
    """A contiguous copy of `x` that starts `front` elements into its storage."""
    flat = torch.empty(front + x.numel() + 8, device=x.device, dtype=x.dtype)
    view = flat[front : front + x.numel()].view(x.shape)
    view.copy_(x)
    return view


def _transposed(x):
    """A copy of the 2-D `x` laid out column-major."""
    return x.t().contiguous().t()


# Views of the contiguous inputs, by argument: q, k, v, gate and dout are the op's
# inputs, out, lse, cu and cu_k (cu_seqlens_q, cu_seqlens_k) the forward's outputs and
# bounds given to the op. 16-byte aligned rows with strides that are multiples of 8
# elements are read in place; the op copies the others (an element offset off 16 bytes,
# a column-major LSE, strided cu_seqlens).
_LAYOUTS = {
    "q_slice": {"q": lambda x: _within(x, 8)},
    "q_slice_unaligned": {"q": lambda x: _within(x, 1)},
    "q_contiguous_unaligned": {"q": lambda x: _shifted(x, 4)},
    "q_every_other_head": {"q": lambda x: _every_other(x, 1)},
    "q_every_other_token": {"q": lambda x: _every_other(x, 0)},
    "k_slice": {"k": lambda x: _within(x, 32)},
    "k_slice_unaligned": {"k": lambda x: _within(x, 3)},
    "k_contiguous_unaligned": {"k": lambda x: _shifted(x, 2)},
    "k_every_other_head": {"k": lambda x: _every_other(x, 1)},
    "k_every_other_token": {"k": lambda x: _every_other(x, 0)},
    "v_slice": {"v": lambda x: _within(x, 0)},
    "v_slice_unaligned": {"v": lambda x: _within(x, 1)},
    "v_contiguous_unaligned": {"v": lambda x: _shifted(x, 1)},
    "v_every_other_head": {"v": lambda x: _every_other(x, 1)},
    "gate_dout_slices": {
        "gate": lambda x: _within(x, 8),
        "dout": lambda x: _within(x, 8),
    },
    "gate_dout_unaligned": {
        "gate": lambda x: _within(x, 1),
        "dout": lambda x: _within(x, 3),
    },
    "gate_dout_contiguous_unaligned": {
        "gate": lambda x: _shifted(x, 4),
        "dout": lambda x: _shifted(x, 1),
    },
    "gate_dout_3d": {
        "gate": lambda x: x.view(x.shape[0], -1, D_V),
        "dout": lambda x: x.view(x.shape[0], -1, D_V),
    },
    "out_slice": {"out": lambda x: _within(x, 8)},
    "out_every_other_head": {"out": lambda x: _every_other(x, 1)},
    "out_contiguous_unaligned": {"out": lambda x: _shifted(x, 3)},
    "lse_slice": {"lse": lambda x: _within(x, 5)},
    "lse_column_major": {"lse": _transposed},
    "cu_offset": {"cu": lambda x: _shifted(x, 1), "cu_k": lambda x: _shifted(x, 3)},
    "cu_every_other": {"cu": lambda x: _every_other(x, 0)},
    "all_views": {
        "q": lambda x: _every_other(x, 1),
        "k": lambda x: _within(x, 32),
        "v": lambda x: _every_other(x, 0),
        "gate": lambda x: _within(x, 8),
        "dout": lambda x: _within(x, 1),
        "out": lambda x: _every_other(x, 0),
        "lse": lambda x: _within(x, 3),
        "cu_k": lambda x: _every_other(x, 0),
    },
}
_OUTPUT_ARGS = ("out", "lse", "cu", "cu_k")


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_gated_mha_bwd_input_layouts(layout, causal):
    """Strided, sliced, unaligned and reshaped views of every input give the gradients
    of contiguous inputs bit for bit."""
    seqlens, heads = [300, 1, 700], 17
    cu, smax, q, k, v, gate, dout = _inputs(seqlens, heads)
    x = {"q": q, "k": k, "v": v, "gate": gate, "dout": dout}
    assert all(t.is_contiguous() for t in x.values())
    plain = _fwd_bwd(**x, cu=cu, smax=smax, causal=causal)
    makers = _LAYOUTS[layout]
    views = {n: make(x[n]) for n, make in makers.items() if n not in _OUTPUT_ARGS}
    for name, view in views.items():
        assert view is not x[name] and torch.equal(view.reshape(x[name].shape), x[name])
    late = {n: make for n, make in makers.items() if n in _OUTPUT_ARGS}

    def checked(make):
        def view(t):
            out = make(t)
            assert out is not t and torch.equal(out, t)
            return out

        return view

    late = {n: checked(make) for n, make in late.items()}
    ours = list(_fwd_bwd(**{**x, **views}, cu=cu, smax=smax, causal=causal, views=late))
    # dgate takes the gate's shape, [T, H * 128] or [T, H, 128].
    assert ours[3].shape == views.get("gate", gate).shape
    ours[3] = ours[3].reshape(plain[3].shape)
    _assert_equal(ours, plain, f"{layout} against contiguous inputs")
    _assert_parity(ours, _ref_bwd(q, k, v, gate, dout, cu, causal))


def _broadcast_tokens(x):
    """`x`'s first token for every token: a view with token stride 0."""
    return x[:1].expand(x.shape)


def _broadcast_heads(x):
    """`x`'s first head for every head: a view with head stride 0 ([T, H, d] only)."""
    return x[:, :1].expand(x.shape)


def _overlapping_rows(x):
    """A [T, H, d] view whose tokens overlap: token stride 8 elements, rows of d."""
    t, h, d = x.shape
    flat = torch.randn(8 * t + h * d + 8, device=x.device, dtype=x.dtype)
    return flat.as_strided((t, h, d), (8, d, 1))


# Views the kernels cannot read in place, by argument: broadcast tokens (token stride 0)
# and overlapping rows are copied; broadcast heads (head stride 0) are read in place.
_BROADCAST = {
    "q_tokens": {"q": _broadcast_tokens},
    "k_tokens": {"k": _broadcast_tokens},
    "v_tokens": {"v": _broadcast_tokens},
    "dout_tokens": {"dout": _broadcast_tokens},
    "gate_tokens": {"gate": _broadcast_tokens},
    "qkv_dout_tokens": {
        "q": _broadcast_tokens,
        "k": _broadcast_tokens,
        "v": _broadcast_tokens,
        "dout": _broadcast_tokens,
    },
    "q_heads": {"q": _broadcast_heads},
    "kv_heads": {"k": _broadcast_heads, "v": _broadcast_heads},
    "q_overlapping": {"q": _overlapping_rows},
    "kv_overlapping": {"k": _overlapping_rows, "v": _overlapping_rows},
}


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("layout", _BROADCAST)
def test_gated_mha_bwd_broadcast_inputs(layout, causal, gated):
    """Broadcast (zero-stride) and overlapping views give the gradients of their dense
    copies bit for bit, and match fp32 autograd: the kernels never read them through
    buffers sized by their token stride. With one token's q, k or v for all, dq and dk
    are sums that cancel to about 0 (equal keys or values make dS sum to 0 over them),
    so there only dv and dgate are held to fp32 autograd."""
    seqlens, heads = [300, 1, 700], 9
    cu, smax, q, k, v, gate, dout = _inputs(seqlens, heads)
    x = {"q": q, "k": k, "v": v, "gate": gate, "dout": dout}
    x = {**x, **{n: make(x[n]) for n, make in _BROADCAST[layout].items()}}
    dense = {n: t.clone(memory_format=torch.contiguous_format) for n, t in x.items()}
    if not gated:
        x["gate"] = dense["gate"] = None
    ours = _fwd_bwd(**x, cu=cu, smax=smax, causal=causal)
    plain = _fwd_bwd(**dense, cu=cu, smax=smax, causal=causal)
    _assert_equal(ours, plain, f"{layout} against dense copies")
    ref = _ref_bwd(*(dense[n] for n in ("q", "k", "v", "gate", "dout")), cu, causal)
    keep = [i for i in range(len(ref)) if i >= 2 or "tokens" not in layout]
    _assert_parity(
        [ours[i] for i in keep], [ref[i] for i in keep], [_OUTPUTS[i] for i in keep]
    )


def _ref_bwd_shared_kv(q, k, v, gate, dout, cu, causal, cu_k=None):
    """fp32 autograd as `_ref_bwd`, each head of `k` and `v` shared by the next
    q.shape[1] / k.shape[1] query heads: d(q), d(k), d(v) of the KV heads, d(gate);
    without a gate, the first three."""
    qf, kf, vf = (x.detach().float().requires_grad_() for x in (q, k, v))
    gf = None if gate is None else gate.detach().float().requires_grad_()
    group = q.shape[1] // k.shape[1]
    out, _ = _ref_attention(
        qf,
        kf.repeat_interleave(group, dim=1),
        vf.repeat_interleave(group, dim=1),
        gf,
        cu,
        causal,
        cu_k,
    )
    out.backward(dout.float().view(out.shape))
    grads = (qf.grad, kf.grad, vf.grad)
    return grads if gate is None else (*grads, gf.grad)


# (query heads, KV heads)
_SHARED_KV = [
    (16, 4),
    (24, 8),
    (33, 11),
    (64, 16),
    (96, 24),
    (12, 1),
    (96, 1),
    (128, 1),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "heads,kv_heads", _SHARED_KV, ids=[f"hq{h}_hk{k}" for h, k in _SHARED_KV]
)
def test_gated_mha_bwd_shared_kv(heads, kv_heads, causal):
    """GQA / MQA: K and V of fewer heads than Q, their gradients per KV head."""
    cu, smax, q, _, _, gate, dout = _inputs(_SEQLENS, heads)
    k = torch.randn(q.shape[0], kv_heads, D_QK, device="cuda", dtype=q.dtype)
    v = torch.randn(q.shape[0], kv_heads, D_V, device="cuda", dtype=q.dtype)
    ours = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal)
    _assert_parity(ours, _ref_bwd_shared_kv(q, k, v, gate, dout, cu, causal))


# (query lengths, key lengths) of the sequences: keys of other lengths than the
# queries'.
_CROSS = [
    ([100], [300]),
    ([300], [100]),
    ([257, 1, 40], [64, 500, 40]),
    ([1000, 513], [513, 2000]),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize(
    "lens_q,lens_k", _CROSS, ids=[f"q{max(a)}_k{max(b)}" for a, b in _CROSS]
)
def test_gated_mha_bwd_cross_lengths(lens_q, lens_k, gated, causal):
    """Query and key sequences of different lengths; the causal mask is aligned to the
    bottom right, so a query may see no key."""
    heads = 9
    cu, smax, q, _, _, gate, dout = _inputs(lens_q, heads)
    cu_k, smax_k, _, k, v, _, _ = _inputs(lens_k, heads, seed=1)
    gate = gate if gated else None
    ours = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal, cu_k, smax_k)
    ref = _ref_bwd(q, k, v, gate, dout, cu, causal, cu_k)
    _assert_parity([x for x in ours if x is not None], ref)


_DETERMINISM = [([1000, 3000, 513], 96), ([8192], 33)]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _DETERMINISM, ids=_case_ids(_DETERMINISM))
def test_gated_mha_bwd_deterministic(seqlens, heads, causal):
    """No atomics: repeated runs give the same bits."""
    cu, smax, q, k, v, gate, dout = _inputs(seqlens, heads)
    first = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal)
    for _ in range(3):
        again = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal)
        _assert_equal(again, first, "run to run")


_NO_GATE = [([257], 9), ([300, 1, 700], 17), ([2048], 96)]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("seqlens,heads", _NO_GATE, ids=_case_ids(_NO_GATE))
def test_gated_mha_bwd_no_gate(seqlens, heads, causal):
    """No gate: the backward of the attention alone."""
    cu, smax, q, k, v, _, dout = _inputs(seqlens, heads)
    *ours, dgate = _fwd_bwd(q, k, v, None, dout, cu, smax, causal)
    assert dgate is None
    _assert_parity(ours, _ref_bwd(q, k, v, None, dout, cu, causal), _OUTPUTS[:3])


def _junk_allocator():
    """Leaves NaN in the blocks the caching allocator hands out next, so that an output
    the op leaves unwritten shows."""
    junk = torch.full((1 << 26,), float("nan"), device="cuda")
    del junk


# (query lengths, key lengths): sequences without queries, and no query at all.
_NO_QUERIES = [
    ([0], [300]),
    ([0, 0], [1, 700]),
    ([0, 40, 0], [64, 100, 257]),
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("heads,kv_heads", [(9, 9), (16, 4)], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "lens_q,lens_k", _NO_QUERIES, ids=[f"q{a}_k{b}" for a, b in _NO_QUERIES]
)
def test_gated_mha_bwd_no_queries(lens_q, lens_k, heads, kv_heads, gated, causal):
    """Keys that no query sees get zero gradients, also with no query at all
    (max_seqlen_q 0), where the kernels have nothing to run on."""
    bf = torch.bfloat16
    torch.manual_seed(0)
    cu = torch.tensor(
        [0, *itertools.accumulate(lens_q)], dtype=torch.int32, device="cuda"
    )
    cu_k = torch.tensor(
        [0, *itertools.accumulate(lens_k)], dtype=torch.int32, device="cuda"
    )
    t, tk = sum(lens_q), sum(lens_k)
    q = torch.randn(t, heads, D_QK, device="cuda", dtype=bf)
    k = torch.randn(tk, kv_heads, D_QK, device="cuda", dtype=bf)
    v = torch.randn(tk, kv_heads, D_V, device="cuda", dtype=bf)
    gate = 2 * torch.randn(t, heads * D_V, device="cuda", dtype=bf) if gated else None
    dout = torch.randn(t, heads * D_V, device="cuda", dtype=bf)
    group = heads // kv_heads
    k_rep, v_rep = (x.repeat_interleave(group, dim=1) for x in (k, v))
    out, lse = _ref_attention(q, k_rep, v_rep, gate, cu, causal, cu_k)
    _junk_allocator()
    ours = gated_mha_bwd(
        q,
        k,
        v,
        out.to(bf),
        dout,
        lse,
        cu,
        cu_k,
        max(lens_q),
        max(lens_k),
        SCALE,
        causal,
        gate,
    )
    assert ours[1].shape == k.shape and ours[2].shape == v.shape
    assert (gate is None) == (ours[3] is None)
    if t == 0:
        assert ours[0].shape == q.shape
        assert ours[1].eq(0).all() and ours[2].eq(0).all()
        return
    ref = _ref_bwd_shared_kv(q, k, v, gate, dout, cu, causal, cu_k)
    _assert_parity([x for x in ours if x is not None], ref)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.int16])
def test_gated_mha_bwd_cu_seqlens_dtype(dtype):
    """cu_seqlens of another dtype than int32 are refused, not converted."""
    cu, smax, q, k, v, gate, dout = _inputs([40], 3)
    out, lse = fmha_fwd_bf16_opus_varlen_fwd(
        q, k, v, SCALE, True, cu, cu, smax, smax, return_lse=True
    )
    for args in ((cu.to(dtype), cu), (cu, cu.to(dtype))):
        with pytest.raises(ValueError, match="int32"):
            gated_mha_bwd(q, k, v, out, dout, lse, *args, smax, smax, SCALE, True, gate)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("scale", [0.03, 0.125, 0.3])
def test_gated_mha_bwd_softmax_scale(scale, causal):
    """Other scales than 1 / sqrt(192): P's exponent and dq and dk carry it."""
    seqlens, heads = [300, 1, 700], 17
    cu, smax, q, k, v, gate, dout = _inputs(seqlens, heads)
    ours = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal, scale=scale)
    _assert_parity(ours, _ref_bwd(q, k, v, gate, dout, cu, causal, scale=scale))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "lens_q,lens_k,bounds",
    [
        ([300, 1, 700], None, (701, 700)),
        ([300, 1, 700], None, (2000, 5000)),
        ([257, 1, 40], [64, 500, 40], (600, 1200)),
        ([300, 1, 700], None, (2**31 - 1, 2**31 - 1)),
    ],
    ids=["q_plus_1", "far_above", "cross_lengths", "int_max"],
)
def test_gated_mha_bwd_max_seqlen_bounds(lens_q, lens_k, bounds, causal):
    """max_seqlen_q and max_seqlen_k above the longest sequences' lengths give the
    gradients of the exact bounds bit for bit: the blocks past a sequence end at once.
    """
    heads = 9
    cu, smax, q, k, v, gate, dout = _inputs(lens_q, heads)
    cu_k, smax_k = cu, smax
    if lens_k is not None:
        cu_k, smax_k, _, k, v, _, _ = _inputs(lens_k, heads, seed=1)
    exact = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal, cu_k, smax_k)
    above = _fwd_bwd(q, k, v, gate, dout, cu, smax, causal, cu_k, smax_k, bounds=bounds)
    _assert_equal(above, exact, f"bounds {bounds} against the exact ones")


_ROW_STRIDE_LIMIT = 1 << 24


def _token_stride(x, stride):
    """A copy of `x` [T, H, d] with token stride `stride` elements (at least H * d)."""
    t, h, d = x.shape
    flat = torch.empty((t - 1) * stride + h * d, device=x.device, dtype=x.dtype)
    view = flat.as_strided((t, h, d), (stride, d, 1))
    view.copy_(x)
    return view


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "stride,tokens",
    [(_ROW_STRIDE_LIMIT, 70), (_ROW_STRIDE_LIMIT + 8, 4)],
    ids=["at_limit", "past_limit"],
)
def test_gated_mha_bwd_token_stride_limit(stride, tokens, causal):
    """Token strides up to 2^24 elements, the limit of the attention kernel's 32-bit
    offsets within a tile, are read in place, past it copied; both give the bits of
    dense inputs. At the limit 70 tokens span the dQ pass's 64-row reads (2.3 GB a
    tensor)."""
    cu, smax, q, k, v, _, dout = _inputs([tokens], 1)
    x = {"q": q, "k": k, "v": v, "dout": dout.view(tokens, 1, D_V)}
    views = {n: _token_stride(t, stride) for n, t in x.items()}
    for view in views.values():
        assert _OP_MODULE._strides_ok(view) == (stride <= _ROW_STRIDE_LIMIT)
    ours = _fwd_bwd(**views, gate=None, cu=cu, smax=smax, causal=causal)
    plain = _fwd_bwd(**x, gate=None, cu=cu, smax=smax, causal=causal)
    del views
    _assert_equal(ours, plain, f"token stride {stride} against dense inputs")


def test_gated_mha_bwd_binding_checks():
    """The attention kernel's binding refuses a token stride past 2^24 elements and
    tensors without heads, which the op never passes it."""
    bf = torch.bfloat16
    cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")

    def call(q, k, v):
        t, h = q.shape[:2]
        d_o = torch.zeros(t, h, D_V, device="cuda", dtype=bf)
        lse = torch.zeros(h, t, device="cuda")
        grads = [
            torch.empty(t, h, n, device="cuda", dtype=bf) for n in (D_QK, D_QK, D_V)
        ]
        _OP_MODULE._fmha_bwd_bf16_opus_bwd(
            q, k, v, d_o, lse, lse.clone(), *grads, cu, cu, 4, 4, False, SCALE
        )

    q = torch.zeros(4, 1, D_QK, device="cuda", dtype=bf)
    k, v = torch.zeros_like(q), torch.zeros(4, 1, D_V, device="cuda", dtype=bf)
    call(q, k, v)  # dense: accepted
    with pytest.raises(RuntimeError, match="token stride"):
        call(_token_stride(q, _ROW_STRIDE_LIMIT + 8), k, v)
    empty = torch.zeros(4, 0, D_QK, device="cuda", dtype=bf)
    with pytest.raises(RuntimeError, match="at least one head"):
        call(empty, empty, torch.zeros(4, 0, D_V, device="cuda", dtype=bf))


def test_gated_mha_bwd_op_schemas():
    """The three kernel ops mark only the tensors they write, so that torch.compile
    neither copies nor writes back their inputs."""
    writes = {
        "_gated_mha_bwd_pre_gate": ["d_o", "dgate", "d"],
        "_gated_mha_bwd_pre": ["d"],
        "_gated_mha_bwd_attn": ["dq", "dk", "dv"],
    }
    for name, expected in writes.items():
        schema = getattr(torch.ops.aiter, name).default._schema
        written = [
            a.name
            for a in schema.arguments
            if a.alias_info is not None and a.alias_info.is_write
        ]
        assert written == expected, f"{name}: {schema}"
        assert not schema.returns, f"{name} returns {schema.returns}"


def test_gated_mha_bwd_exported():
    """The op is aiter.gated_mha_bwd and aiter.ops.opus.gated_mha_bwd; its module is
    aiter.ops.opus._gated_mha_bwd."""
    import aiter.ops.opus._gated_mha_bwd as module
    from aiter.ops import opus

    assert "gated_mha_bwd" in opus.__all__
    assert opus.gated_mha_bwd is gated_mha_bwd is aiter.gated_mha_bwd
    assert callable(gated_mha_bwd) and module.gated_mha_bwd is gated_mha_bwd


@pytest.mark.parametrize(
    "heads,kv_heads,gated",
    [(9, 9, True), (9, 9, False), (16, 4, True)],
    ids=["mha_gated", "mha_no_gate", "gqa_gated"],
)
def test_gated_mha_bwd_compiled(heads, kv_heads, gated):
    """torch.compile(fullgraph=True) traces the op into one graph around its three
    kernel ops and gives the eager gradients bit for bit."""
    import torch._dynamo

    cu, smax, q, _, _, gate, dout = _inputs([300, 1, 700], heads)
    k = torch.randn(q.shape[0], kv_heads, D_QK, device="cuda", dtype=q.dtype)
    v = torch.randn(q.shape[0], kv_heads, D_V, device="cuda", dtype=q.dtype)
    gate = gate if gated else None
    out, lse = fmha_fwd_bf16_opus_varlen_fwd(
        q, k, v, SCALE, True, cu, cu, smax, smax, return_lse=True
    )
    if gate is not None:
        out = (out.float() * gate.float().view(out.shape).sigmoid()).to(out.dtype)
    args = (q, k, v, out, dout, lse, cu, cu, smax, smax, SCALE, True, gate)
    eager = gated_mha_bwd(*args)

    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    traced = torch.compile(gated_mha_bwd, backend=capture, fullgraph=True)(*args)
    _assert_equal(traced, eager, "the traced graph against eager")
    targets = {str(n.target) for g in graphs for n in g.graph.nodes}
    pre = "aiter._gated_mha_bwd_pre_gate" if gated else "aiter._gated_mha_bwd_pre"
    for op in (pre, "aiter._gated_mha_bwd_attn"):
        assert any(op in t for t in targets), f"{op} not in the graph: {targets}"
    torch._dynamo.reset()
    compiled = torch.compile(gated_mha_bwd, backend="aot_eager", fullgraph=True)(*args)
    _assert_equal(compiled, eager, "aot_eager against eager")


def test_gated_mha_bwd_needs_gfx950(monkeypatch):
    """The code object is gfx950's: on another device the op raises a clear error. The
    architecture is that of q's device, not the build target."""
    cu, smax, q, k, v, gate, dout = _inputs([40], 3)
    out, lse = fmha_fwd_bf16_opus_varlen_fwd(
        q, k, v, SCALE, True, cu, cu, smax, smax, return_lse=True
    )
    seen = []

    def arch(device):
        seen.append(device)
        return "gfx942"

    monkeypatch.setattr(_OP_MODULE, "get_gfx_runtime_for_device", arch)
    with pytest.raises(RuntimeError, match="gfx950"):
        gated_mha_bwd(q, k, v, out, dout, lse, cu, cu, smax, smax, SCALE, True, gate)
    assert seen and all(torch.device(d) == q.device for d in seen)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
