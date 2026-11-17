# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + performance tests for fmha_fwd_with_sink_asm (BF16 ASM, gfx1250).

Public API:  aiter.flash_attn_func              (the path the model calls)
Ops layer:   aiter.fmha_fwd_with_sink_asm       (low-level)

Built to the aiter op-test standard (see .claude/skills/aiter-op-test): mirror
test_quant.py — @benchmark + run_perftest candidate loop, a torch reference,
per-candidate us / TFLOPS / TB/s / err, a markdown summary table per test
function, and a __main__ guard so the module is importable.

Sink convention
---------------
`sink` ([q_head_num] fp32) is a per-Q-head logit in the SAME scaled domain as
Q·K^T * softmax_scale; it acts as a zero-value virtual KV column and is passed
to the kernel verbatim (no host-side scaling).  D64 kernels read it; D128 and
D256 kernels ignore it (pass None).

Candidates: `asm` calls the ASM kernel through the ops layer; `public` is what
flash_attn_func runs, which offers the call to FlyDSL first, so it may be served
by another backend.

Layout: the API only accepts bshd shape ([b, s, h, d]).  To exercise the
kernel's stride handling for sbhd / bhsd memory, qkv are allocated in the
chosen `layout` and `permute()`d to a bshd-shaped non-contiguous view.
"""

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.test_common import assertAllclose, benchmark, run_perftest

torch.set_default_device("cuda")

# Every card these ASM kernels are built/validated for.  The .co files only
# ship for gfx1250 (hsa/gfx1250/fmha_fwd_bf16/*.co); on any other arch the
# kernel launch raises 'no kernel for arch=...'.
SUPPORTED_GFX = ["gfx1250"]


# ---------------------------------------------------------------------------
# Reference (fp32 math, cast back).  Not timed, not in the table.
# ---------------------------------------------------------------------------


def run_torch(q, k, v, *, is_causal, sink=None):
    """bshd-in / bshd-out attention reference, sink optional.

    scores = (Q·K^T) * scale,  scale = 1/sqrt(d);  lse returned in fp32.
    sink (optional): [hq] fp32 per-head logit in the scaled domain (a zero-value
    KV column appended to the scaled scores).
    """
    b, sq, hq, d = q.shape
    _, sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("bshd,bkhd->bhsk", qf, kf) * scale
    if is_causal:
        # bottom-right aligned; with sq > sk the first sq - sk rows see no key
        m = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=q.device), sk - sq + 1
        )
        scores = scores.masked_fill(m, float("-inf"))
    max_attn, _ = scores.max(dim=-1)
    if sink is not None:
        sink_bhs = sink.float()[None, :, None].expand(b, hq, sq)
        max_total = torch.maximum(max_attn, sink_bhs)
    else:
        max_total = max_attn
    # -inf only on rows that see no key: subtract 0 there, so O = 0, LSE = -inf
    max_total = torch.where(torch.isinf(max_total), 0.0, max_total)
    denom = torch.exp(scores - max_total.unsqueeze(-1)).sum(dim=-1)
    if sink is not None:
        denom = denom + torch.exp(sink_bhs - max_total)
    probs = torch.exp(scores - max_total.unsqueeze(-1)) / torch.where(
        denom > 0, denom, 1.0
    ).unsqueeze(-1)
    out = torch.einsum("bhsk,bkhd->bshd", probs, vf).to(q.dtype)
    lse = torch.log(denom) + max_total
    return out, lse


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def make_qkv_bshd(layout, sq, sk, batch, hq, hk, d, init="randn", dtype=dtypes.bf16):
    """Allocate (q, k, v) in `layout` memory, return bshd-shaped views.

    layout: 0 = bshd (contiguous), 1 = bhsd view, 2 = sbhd view.  The kernel
    reads strides directly so non-contiguous bshd views (layout 1/2) are valid.
    init:   "randn" (random normal) or "const0.25" (every element = 0.25).
    """
    if layout == 0:
        q = torch.randn(batch, sq, hq, d, dtype=dtype)
        k = torch.randn(batch, sk, hk, d, dtype=dtype)
        v = torch.randn(batch, sk, hk, d, dtype=dtype)
    elif layout == 1:
        q = torch.randn(batch, hq, sq, d, dtype=dtype).permute(0, 2, 1, 3)
        k = torch.randn(batch, hk, sk, d, dtype=dtype).permute(0, 2, 1, 3)
        v = torch.randn(batch, hk, sk, d, dtype=dtype).permute(0, 2, 1, 3)
    elif layout == 2:
        q = torch.randn(sq, batch, hq, d, dtype=dtype).permute(1, 0, 2, 3)
        k = torch.randn(sk, batch, hk, d, dtype=dtype).permute(1, 0, 2, 3)
        v = torch.randn(sk, batch, hk, d, dtype=dtype).permute(1, 0, 2, 3)
    else:
        raise ValueError(f"unsupported layout={layout}")
    if init == "const0.25":
        # In-place fill is layout-agnostic (works on non-contiguous views).
        q.fill_(0.25)
        k.fill_(0.25)
        v.fill_(0.25)
    elif init != "randn":
        raise ValueError(f"unknown init pattern: {init!r}")
    return q, k, v


def _d64_sink(hq):
    """Per-head sink logits in [0.5, 2.0] (scaled domain), varied across heads."""
    return torch.linspace(0.5, 2.0, hq, dtype=dtypes.fp32)


def run_kernel(q, k, v, *, scale, is_causal, sink=None, via="public"):
    """via="public" → aiter.flash_attn_func (the model path); "ops" → low-level."""
    if via == "public":
        r = aiter.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=is_causal,
            return_lse=True,
            sink_ptr=sink,
        )
        return r[0], r[1]
    if via == "ops":
        return aiter.fmha_fwd_with_sink_asm(q, k, v, scale, is_causal, True, sink=sink)
    raise ValueError(f"unknown via={via!r}")


def _flops_bytes(batch, hq, hk, sq, sk, d, is_causal, esz):
    """Attention roofline numerators: 2 GEMMs (QK^T, PV), HBM traffic q+k+v+o."""
    flops = 4.0 * batch * hq * sq * sk * d  # 2*(2*M*N*K) over the two matmuls
    if is_causal:
        flops /= 2.0
    nbytes = (
        2 * batch * sq * hq * d  # q read + o write
        + 2 * batch * sk * hk * d  # k + v read
    ) * esz
    return flops, nbytes


# ---------------------------------------------------------------------------
# Shape tables
# ---------------------------------------------------------------------------

# Correctness shapes (torch reference is feasible here).  hq=64; hk=8 for D64
# and hk=4 for D128 (GQA ratios 8 / 16).  Non-causal (mask=0) D64/D128 kernels
# require sk % 256 == 0 — non-aligned sk rows are causal-only for them (filtered
# in the sweep); D256 tiles KV by 128 with a border mask and takes any sk.
# (head_dim, hq, hk, sq, sk, batch)
_CORRECTNESS_SHAPES = [
    (64, 64, 8, 128, 2048, 1),  # D64  aligned
    (64, 64, 8, 128, 2048, 2),
    (64, 64, 8, 130, 2048, 1),  # D64  q-unaligned (sq not mult of 128)
    (64, 64, 8, 128, 2300, 1),  # D64  kv-unaligned (sk not mult of 256) -> causal
    (64, 64, 8, 300, 256, 2),  # D64  sq > sk: causal rows that see no key
    (128, 64, 4, 128, 2048, 1),  # D128 aligned
    (128, 64, 4, 128, 2048, 2),
    (128, 64, 4, 130, 2048, 1),  # D128 q-unaligned
    (128, 64, 4, 128, 2300, 1),  # D128 kv-unaligned -> causal
    (64, 64, 8, 8192, 8192, 1),  # D64  perf-sized, aligned
    (128, 64, 4, 4096, 4096, 1),  # D128 perf-sized, aligned
    (256, 16, 4, 128, 2048, 1),  # D256 aligned
    (256, 16, 4, 128, 2048, 2),
    (256, 16, 4, 130, 2048, 1),  # D256 q-unaligned
    (256, 16, 4, 128, 2300, 1),  # D256 kv-unaligned (non-causal too)
    (256, 16, 4, 1000, 300, 2),  # D256 sq > sk: causal rows that see no key
    (256, 16, 4, 300, 1000, 3),  # D256 sq < sk
    (256, 16, 16, 4096, 4096, 1),  # D256 perf-sized
    (256, 64, 64, 2304, 2304, 1),  # D256 kv heads x kv length > 2^17: no remap_xy
]

# Perf-only shapes; sq == sk, sbhd.
# The torch reference is O(s^2) in memory (e.g. 32768 -> ~256 GB of scores) so
# these are timed but NOT correctness-checked here (use _CORRECTNESS_SHAPES).
# (head_dim, hq, hk, seqlen, batch)
_PERF_SHAPES = [
    (64, 64, 8, 1024, 1),
    (64, 64, 8, 4096, 1),
    (64, 64, 8, 8192, 1),
    (64, 64, 8, 16384, 1),
    (64, 64, 8, 32768, 1),
    (128, 64, 4, 1024, 1),
    (128, 64, 4, 2048, 1),
    (128, 64, 4, 4096, 1),
    (128, 64, 4, 8192, 1),
    (128, 64, 4, 16384, 1),
    (256, 16, 16, 4096, 1),
    (256, 16, 16, 16384, 1),
    (256, 32, 4, 8192, 1),
    (256, 32, 32, 16000, 4),  # mi455x-opbench attn/uniform/B4L16000/h32d256
    (256, 32, 1, 16384, 1),  # MQA
    (256, 64, 64, 1024, 16),
]


# ---------------------------------------------------------------------------
# Test functions (one markdown table each).  @benchmark logs the call args as
# the table's left columns and merges the returned metric dict.
# ---------------------------------------------------------------------------


@benchmark()
def test_fmha_fwd_with_sink_asm(
    head_dim, hq, hk, sq, sk, batch, is_causal, layout, init, d64_sink=True
):
    torch.manual_seed(0)
    q, k, v = make_qkv_bshd(layout, sq, sk, batch, hq, hk, head_dim, init=init)
    scale = 1.0 / math.sqrt(head_dim)
    # D64 -> non-zero sink (exercises ENABLE_SINK), or none (the wrapper passes
    # the kernel a -inf sink); D128 -> kernel ignores it.
    sink = _d64_sink(hq) if head_dim == 64 and d64_sink else None

    ref_out, ref_lse = run_torch(q, k, v, is_causal=is_causal, sink=sink)

    flops, nbytes = _flops_bytes(
        batch, hq, hk, sq, sk, head_dim, is_causal, q.element_size()
    )

    candidates = {
        name: lambda via=via: run_kernel(
            q, k, v, scale=scale, is_causal=is_causal, sink=sink, via=via
        )
        for name, via in (("asm", "ops"), ("public", "public"))
    }

    ret = {"gfx": get_gfx()}
    # Rows that see no key (causal, sq > sk): O = 0, LSE = -inf, or the sink logit
    # with a sink.  The ASM LSE is checked on every row; flash_attn_func may run
    # FlyDSL, Triton or CK, which can write LSE = 0 there (Triton; FlyDSL with a
    # sink), so the public LSE is checked on rows that see a key.  Those rows are
    # found by position (bottom-right aligned): a sink makes the reference LSE
    # finite on every row.
    has_key = (torch.arange(sq) >= (sq - sk if is_causal else 0)).expand(batch, hq, sq)
    no_key = ~has_key
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
            msg=f"{name} O d={head_dim} c={is_causal}",
        )
        ret[f"{name} err(LSE)"] = assertAllclose(
            ref_lse.to(dtypes.fp32),
            lse.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} LSE d={head_dim} c={is_causal}",
            mask=has_key if name == "public" else None,
        )
        # assertAllclose tolerates a few bad elements; rows without a key must match on
        # every element: O = 0, LSE = -inf (no sink) or the sink logit
        if name == "asm" and no_key.any():
            assert not out.permute(0, 2, 1, 3)[no_key].any(), "asm O != 0 without keys"
            ok = torch.isclose(lse[no_key], ref_lse[no_key], rtol=1e-2, atol=1e-2)
            assert ok.all(), "asm LSE differs from the reference without keys"
    return ret


# ---------------------------------------------------------------------------
# Batch gate, driven through _flash_attn_forward (flash_attn_func reaches it only
# with CK built), with every forward entry replaced by a stand-in that raises
# instead of launching.
# ---------------------------------------------------------------------------

_FWD_ENTRIES = (
    "mha_fwd_native_splitkv",
    "fmha_fwd_with_sink_asm",
    "fmha_fwd_mxfp8_asm",
    "fmha_fwd_bf16_opus_fwd",
    "fmha_v3_fwd",
    "mha_fwd",
)


class _Served(Exception):
    """Raised by a forward-entry stand-in; args[0] is the entry name."""


def _batch_route(q, k, v, *, causal, window, sink=None):
    """The forward entry _flash_attn_forward calls for these arguments."""
    mha = aiter.ops.mha
    saved = {n: getattr(mha, n) for n in _FWD_ENTRIES if hasattr(mha, n)}

    def stand_in(name):
        def call(*args, **kwargs):
            raise _Served(name)

        return call

    try:
        for n in saved:
            setattr(mha, n, stand_in(n))
        mha._flash_attn_forward(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=q.size(-1) ** -0.5,
            causal=causal,
            window_size_left=window[0],
            window_size_right=window[1],
            sink_size=0,
            bias=None,
            alibi_slopes=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            return_lse=True,
            return_softmax=False,
            sink_ptr=sink,
            num_splits=1,
        )
    except _Served as served:
        return served.args[0]
    finally:
        for n, f in saved.items():
            setattr(mha, n, f)
    raise AssertionError("_flash_attn_forward called no forward entry")


# name -> (served by the gfx1250 bf16 ASM kernel, head dim, causal, window), with
# 4/4 heads and 256 tokens.  CK window convention: >= seqlen_k is unlimited, and
# causal ignores the right window.
_BATCH_GATE_CASES = {
    "D256": (True, 256, False, (-1, -1)),
    "D256 causal": (True, 256, True, (-1, -1)),
    "D128": (True, 128, False, (-1, -1)),
    "D64 with sink logits": (True, 64, False, (-1, -1)),
    "D64 without sink logits": (False, 64, False, (-1, -1)),
    "D128 with sink logits": (False, 128, False, (-1, -1)),
    "D256 with sink logits": (False, 256, False, (-1, -1)),
    "D256 causal, window (-1, 0)": (True, 256, True, (-1, 0)),
    "D256 window (-1, 256)": (True, 256, False, (-1, 256)),
    "D256 window (-1, 0)": (False, 256, False, (-1, 0)),
    "D256 window (0, -1)": (False, 256, False, (0, -1)),
    "D256 causal, window (0, -1)": (False, 256, True, (0, -1)),
    "D128 window (-1, 0)": (False, 128, False, (-1, 0)),
    "D256 k token stride 2^24 B": (False, 256, False, (-1, -1)),
    # sbhd q, batch 2, 32 heads: the dense output fmha_fwd_with_sink_asm allocates
    # has a batch stride of sq x 32 x 512 B
    "D256 sbhd q, sq 65536 (output batch stride 2^30 B)": (True, 256, False, (-1, -1)),
    "D256 sbhd q, sq 131072 (output batch stride 2^31 B)": (
        False,
        256,
        False,
        (-1, -1),
    ),
}


@benchmark()
def test_fmha_fwd_with_sink_asm_gate(case):
    """The batch gate must pick the ASM kernel for exactly what it serves.  Shapes
    the kernel cannot address are built on the meta device: nothing is launched."""
    expect, d, causal, window = _BATCH_GATE_CASES[case]
    q = torch.randn(1, 256, 4, d, dtype=dtypes.bf16)
    k = torch.randn(1, 256, 4, d, dtype=dtypes.bf16)
    v = torch.randn(1, 256, 4, d, dtype=dtypes.bf16)
    if case == "D256 k token stride 2^24 B":
        tok = (1 << 24) // 2
        k = torch.empty_strided(
            (1, 256, 4, d), (256 * tok, tok, d, 1), dtype=dtypes.bf16, device="meta"
        )
    if "sbhd q" in case:
        sq = 65536 if "sq 65536" in case else 131072
        q = torch.empty(sq, 2, 32, d, dtype=dtypes.bf16, device="meta")
        q = q.permute(1, 0, 2, 3)
        k = torch.empty(2, 128, 32, d, dtype=dtypes.bf16, device="meta")
        v = torch.empty(2, 128, 32, d, dtype=dtypes.bf16, device="meta")
    sink = None
    if "with sink" in case:
        sink = torch.zeros(q.size(2), dtype=dtypes.fp32, device=q.device)
    served = _batch_route(q, k, v, causal=causal, window=window, sink=sink)
    routed = served == "fmha_fwd_with_sink_asm"
    assert routed == expect, f"{case}: served by {served}"
    return {"gfx": get_gfx(), "served by": served}


# (batch, seqlen_q) without queries
_EMPTY_SHAPES = [(0, 128), (2, 0)]


@benchmark()
def test_fmha_fwd_with_sink_asm_empty(head_dim, batch, sq):
    """No queries (batch 0 or seqlen_q 0): the ops entry returns without a launch."""
    q = torch.randn(batch, sq, 4, head_dim, dtype=dtypes.bf16)
    k = torch.randn(batch, 256, 4, head_dim, dtype=dtypes.bf16)
    v = torch.randn(batch, 256, 4, head_dim, dtype=dtypes.bf16)
    out, lse = run_kernel(
        q,
        k,
        v,
        scale=head_dim**-0.5,
        is_causal=False,
        sink=_d64_sink(4) if head_dim == 64 else None,
        via="ops",
    )
    ok = out.shape == (batch, sq, 4, head_dim) and lse.shape == (batch, 4, sq)
    assert ok, f"D{head_dim} batch={batch} sq={sq}: out {out.shape}, lse {lse.shape}"
    return {"gfx": get_gfx(), "ok": ok}


@benchmark()
def test_fmha_fwd_with_sink_asm_perf(head_dim, hq, hk, sq, sk, batch, is_causal, init):
    torch.manual_seed(0)
    q, k, v = make_qkv_bshd(2, sq, sk, batch, hq, hk, head_dim, init=init)
    scale = 1.0 / math.sqrt(head_dim)
    sink = _d64_sink(hq) if head_dim == 64 else None

    flops, nbytes = _flops_bytes(
        batch, hq, hk, sq, sk, head_dim, is_causal, q.element_size()
    )

    candidates = {
        name: lambda via=via: run_kernel(
            q, k, v, scale=scale, is_causal=is_causal, sink=sink, via=via
        )
        for name, via in (("asm", "ops"), ("public", "public"))
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


def summarize(name, rows):
    aiter.logger.info(
        "fmha_fwd_with_sink_asm %s summary (markdown):\n%s",
        name,
        pd.DataFrame(rows).to_markdown(index=False),
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fmha_fwd_with_sink_asm unsupported on %s; skipping", get_gfx()
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
        choices=[64, 128, 256],
        default=[64, 128, 256],
        help="head dim(s) to test (default: 64 128 256)",
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
        "-l",
        "--layout",
        type=int,
        nargs="*",
        choices=[0, 1, 2],
        default=[2],
        help="memory layout(s): 0=bshd 1=bhsd 2=sbhd (default: 2)",
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

    summarize(
        "correctness",
        [
            test_fmha_fwd_with_sink_asm(d, hq, hk, sq, sk, b, c, layout, init, s)
            for (d, hq, hk, sq, sk, b), c, layout, init in itertools.product(
                _CORRECTNESS_SHAPES, causal_modes, args.layout, args.init
            )
            # non-causal (mask=0) D64 / D128 kernels need sk % 256 == 0
            if d in args.head_dim and (c or sk % 256 == 0 or d == 256)
            # D64 also runs without a sink (the ops wrapper passes a -inf one)
            for s in ((True, False) if d == 64 else (True,))
        ],
    )
    summarize(
        "gate",
        [
            test_fmha_fwd_with_sink_asm_gate(case)
            for case, (_, d, _, _) in _BATCH_GATE_CASES.items()
            if d in args.head_dim
        ],
    )
    summarize(
        "empty-input",
        [
            test_fmha_fwd_with_sink_asm_empty(d, b, sq)
            for d, (b, sq) in itertools.product(args.head_dim, _EMPTY_SHAPES)
        ],
    )
    # perf-only: the torch reference is infeasible at these sizes
    summarize(
        "perf",
        [
            test_fmha_fwd_with_sink_asm_perf(d, hq, hk, s, s, b, c, init)
            for (d, hq, hk, s, b), c, init in itertools.product(
                _PERF_SHAPES, causal_modes, args.init
            )
            if d in args.head_dim
        ],
    )


if __name__ == "__main__":
    main()
