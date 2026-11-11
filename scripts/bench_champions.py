#!/usr/bin/env python3
"""Correctness + perf harness for the standalone FlyDSL fp8 paged-prefill champions vs the incumbent/Gluon/Triton.

One process = one (kernel, shape) point (or any subset), so points can be submitted independently:
  ENABLE_CK=0 FLYDSL_RUNTIME_ENABLE_CACHE=0 PYTHONPATH=$W gpuq --num-gpus=1 python3 -u scripts/bench_champions.py \
      --kernel exact --shape trace_M32768 --mode both --out bench_out/x/exact__trace_M32768.json
  python3 scripts/bench_champions.py --merge bench_out/champions/<ts>      # -> summary.md
  python3 scripts/bench_champions.py --compare <dirA> <dirB>               # per-shape flydsl time ratio B/A (port-step A/B)

Perf knobs: --causal {1,0,both} (default 1), --group {4,8,16,all} (default 16; Q heads = 4*group, 4 KV heads, D128),
  --out-dtype {bf16,f16,both} (default bf16), --tag <label> (stored in every row). Champions are causal/G16/bf16 only and
  record "unsupported" otherwise; the incumbent 'flydsl' runs everything; gluon/triton record whatever they do.

Kernels (--kernel, comma list, default all):
  exact    champion_exact.attention(q,k,v,out,cu_seqlens_q,seqused_k,block_table,q_descale,k_descale,v_descale,
                                    max_seqlen_q,max_seqlen_k,softmax_scale)   EXP_APPROX=0, causal only
  noslack  champion_noslack.attention(same)                                     EXP_APPROX=16, causal only
  flydsl   incumbent: aiter.ops.unified_attention.unified_attention(**kw, backend=None) (perf; as bench_fua_rebaseline)
           / backend="flydsl" (correctness; as op_tests/test_unified_attention.py)
  gluon, triton   unified_attention(**kw, backend="gluon"|"triton")

Reuse, not reinvention:
  perf   inputs: generate_data(shuffled_kv_cache=...) via bench_fua_rebaseline.build(); timing: bench_fua_rebaseline.time_us
         (one-call CUDA graph, 512 MB flush before each sample, WARMUP=10, SAMPLES=31, median; eager fallback recorded).
  correctness inputs: op_tests/test_unified_attention.py make_case / shuffle_kv_cache / ref_paged_attn (fp32 reference on
         the linear cache, descales applied); tolerance = that file's measure(): atol = FP8_REL_ERR[bf16]=7.7e-2 *
         max|ref|, rtol=0, zero tolerated mismatches (column `pass`). Column `pass_tritontol` additionally applies
         the triton test's atol=rtol=1.5e-1 with 1% tolerated mismatches. NaN anywhere in the output fails.
  Poison variant: V and K tails of partial pages and ALL unused pages (16 appended) are filled with fp8 NaN bytes (0x7F);
         the output buffer is pre-filled with NaN so unwritten rows fail.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import math
import os
import statistics
import sys

PAGE, D, H, HKV = 64, 128, 64, 4
NUM_BLOCKS = 65536
WARMUP, SAMPLES = 10, 31
FLUSH_BYTES = 512 << 20
FP8_REL_ERR = 7.7e-2  # op_tests/test_unified_attention.py FP8_REL_ERR[bf16]
GROUPS = (4, 8, 16)
KERNELS = ["exact", "noslack", "flydsl", "gluon", "triton"]
ALIASES = {
    "champion_exact": "exact",
    "champion_noslack": "noslack",
    "incumbent": "flydsl",
}
CHAMP_MODULE = {"exact": "champion_exact", "noslack": "champion_noslack"}


def split_equal(m, n):
    base, rem = divmod(m, n)
    return [base + 1] * rem + [base] * (n - rem)


# (M, nseq, kv_mult): bench_fua_rebaseline.py PREFILL_SHAPES
TRACE = [
    (32768, 42, 2.9),
    (29262, 38, 2.9),
    (29261, 37, 2.9),
    (25613, 42, 3.6),
    (25610, 39, 3.4),
    (25609, 38, 3.3),
    (11795, 64, 4.5),
    (11793, 64, 4.4),
    (8303, 64, 6.6),
    (8294, 64, 6.6),
    (8291, 64, 6.6),
    (8290, 64, 6.6),
    (3697, 44, 10.9),
    (3694, 41, 10.1),
    (3693, 40, 10.0),
]


def perf_shapes():
    s = {}
    for m, n, mult in TRACE:
        q = split_equal(m, n)
        s[f"trace_M{m}"] = (q, [max(1, round(x * mult)) for x in q])
    # "ticket" shapes: bench_fua_ab.py prefill cases (the rebaseline script has none; see report)
    s["ticket_M32768"] = ([781] * 8 + [780] * 34, [2124] * 8 + [2122] * 34)
    s["ticket_M3693"] = ([93] * 13 + [92] * 27, [837] * 13 + [828] * 27)
    s["ticket_dense_1x8192"] = ([8192], [8192])
    s["ticket_dense_4x2048"] = ([2048] * 4, [2048] * 4)
    return s


def ragged_B(B, seed=1000):
    """check_champion_edges.py Check 1: B ragged sequences, last q_len != first (B==64 last-sequence write)."""
    import random

    rnd = random.Random(seed + B)
    qls = [rnd.randint(1, 40) for _ in range(B)]
    if qls[-1] == qls[0]:
        qls[-1] = qls[0] % 40 + 1
    return qls, [q + rnd.randint(0, 300) for q in qls]


# correctness shapes: op_tests/test_unified_attention.py PREFILL_CASES (group 16 only) + edge cases.
# value = (qls, kvs) or (qls, kvs, opts); opts keys: group (4/8/16), dtype ("bf16"/"f16" out), causals (tuple of bool),
# garbage_bt (extra block-table columns and unused tail entries filled with garbage page ids).
def corr_shapes():
    s = {
        "c_base": ([512, 256], [512, 256]),
        "c_small_sq": ([64, 64], [2048, 2048]),
        "c_varlen": ([320, 300, 200, 150], [2048, 5000, 8192, 3000]),
        "c_varlen_causal": ([93, 80, 71, 60], [157, 100, 90, 70]),
        "c_ragged_g16": ([93, 17, 41, 9], [157, 0, 70, 1]),
        "c_partial_page": ([700], [1000]),
    }
    q = split_equal(3693, 40)
    s["c_trace_M3693"] = (q, [max(1, round(x * 10.0)) for x in q])
    for B in (63, 64, 65, 128, 129):
        s[f"c_B{B}"] = ragged_B(B)
    # uniform q_len (q_len*B == M) vs ragged batch with the same M and B
    s["c_uniform_q"] = ([128] * 8, [1024 + 64 * i for i in range(8)])
    s["c_ragged_sameM"] = (
        [300, 20, 200, 100, 128, 60, 150, 66],
        [1024 + 64 * i for i in range(8)],
    )
    s["c_uniform_q_partial"] = ([100] * 6, [1000, 777, 130, 1024, 65, 1])
    # non-causal with partial tail pages (poison variant puts NaN in the tails)
    s["c_noncausal_partial"] = ([100, 64, 37], [1000, 777, 129], {"causals": (False,)})
    s["c_noncausal_long"] = ([320, 300], [5000, 3001], {"causals": (False,)})
    for g in (4, 8):
        s[f"c_g{g}_varlen"] = (
            [320, 300, 200, 150],
            [2048, 5000, 8192, 3000],
            {"group": g},
        )
        s[f"c_g{g}_partial"] = ([93, 17, 41, 9], [157, 100, 70, 65], {"group": g})
    s["c_f16_varlen"] = (
        [320, 300, 200, 150],
        [2048, 5000, 8192, 3000],
        {"dtype": "f16"},
    )
    s["c_f16_g4_partial"] = (
        [93, 17, 41, 9],
        [157, 100, 70, 65],
        {"group": 4, "dtype": "f16"},
    )
    # empty KV (run for champions too; result recorded)
    s["c_empty_kv_mixed"] = ([64, 32, 48], [0, 200, 0])
    s["c_empty_kv_all"] = ([64, 32], [0, 0])
    # q_len > kv_len (delta < 0); the router may decline, recorded as status
    s["c_q_gt_kv"] = ([200, 100, 64], [50, 100, 10])
    # block tables: minimally narrow (width == ceil(kv/64) for every row) and padded with garbage
    s["c_bt_narrow"] = ([128, 128], [1024, 1024])
    s["c_bt_narrow_partial"] = ([128, 128], [1000, 961])
    s["c_bt_garbage"] = ([93, 80, 71, 60], [157, 100, 90, 700], {"garbage_bt": True})
    s["c_bt_garbage_noncausal"] = (
        [93, 80, 71, 60],
        [157, 100, 90, 700],
        {"garbage_bt": True, "causals": (False,)},
    )
    # nonzero cu_seqlens_q[0]: not covered (unified_attention API / test_unified_attention.py always use cu_q[0]==0
    # and index q rows from cu_q[0]; no documented support), so skipped.
    return s


def all_shapes():
    s = dict(corr_shapes())
    s.update(perf_shapes())
    return s


def geomean(xs):
    xs = [x for x in xs if not math.isnan(x) and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


# ---------------------------------------------------------------- calls
def make_call(kernel, kw, for_corr=False):
    import aiter.ops.unified_attention as ua

    if kernel in CHAMP_MODULE:
        import importlib

        mod = importlib.import_module(CHAMP_MODULE[kernel])
        return lambda: mod.attention(
            kw["q"],
            kw["k"],
            kw["v"],
            kw["out"],
            kw["cu_seqlens_q"],
            kw["seqused_k"],
            kw["block_table"],
            kw["q_descale"],
            kw["k_descale"],
            kw["v_descale"],
            kw["max_seqlen_q"],
            kw["max_seqlen_k"],
            kw["softmax_scale"],
        )
    backend = {
        "flydsl": "flydsl" if for_corr else None,
        "gluon": "gluon",
        "triton": "triton",
    }[kernel]
    return lambda: ua.unified_attention(**kw, backend=backend)


def champion_unsupported(kw):
    """Reason the standalone champion cannot take this call, else None (it asserts or is causal-only)."""
    import torch

    if not kw["causal"]:
        return "unsupported: causal-only kernel"
    if kw["q"].shape[1] != 4 * 16:
        return f"unsupported: group {kw['q'].shape[1] // HKV} (G16 only)"
    if kw["out"].dtype != torch.bfloat16:
        return f"unsupported: out dtype {kw['out'].dtype} (bf16 only)"
    if -(-int(kw["max_seqlen_k"]) // PAGE) > 2048:
        return "unsupported: max_seqlen_k > 131072 (block-table LDS staging)"
    if kw["block_table"].stride(1) != 1:
        return "unsupported: block_table stride(1) != 1"
    return None


# ---------------------------------------------------------------- perf
DT = {}
_flush = None


def time_us(fn):
    """bench_fua_rebaseline.time_us verbatim: median single-call graph replay after a 512 MB flush."""
    import torch

    global _flush
    if _flush is None:
        _flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    cap_stream = torch.cuda.Stream()
    with torch.cuda.stream(cap_stream):
        fn()
    torch.cuda.synchronize()
    mode = "graph"
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cap_stream):
            fn()
        run = g.replay
    except Exception:  # noqa: BLE001
        torch.cuda.synchronize()
        mode, run = "eager", fn
    for _ in range(WARMUP):
        _flush.zero_()
        run()
    ts = []
    for _ in range(SAMPLES):
        _flush.zero_()
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    return statistics.median(ts), mode


def build_perf(qls, kvs, shuffled, group=16, out_dtype="bf16"):
    """bench_fua_rebaseline.build(); heads = 4*group."""
    from aiter.ops.triton.utils.types import e4m3_dtype
    from op_tests.triton_tests.attention.test_unified_attention import generate_data

    data = generate_data(
        seq_lens=list(zip(qls, kvs)),
        num_blocks=NUM_BLOCKS,
        block_size=PAGE,
        head_size=D,
        num_heads=(HKV * group, HKV),
        q_dtype=e4m3_dtype,
        kv_dtype=e4m3_dtype,
        out_dtype=DT[out_dtype],
        shuffled_kv_cache=shuffled,
        device="cuda",
    )
    (
        _q,
        _ko,
        _vo,
        kc,
        vc,
        _s,
        out,
        cuq,
        kvl,
        mq,
        mk,
        scale,
        win,
        bt,
        mqq,
        _qs,
        qd,
        kd,
        vd,
        _os,
    ) = data
    return {
        "q": mqq,
        "k": kc,
        "v": vc,
        "out": out,
        "cu_seqlens_q": cuq,
        "max_seqlen_q": mq,
        "seqused_k": kvl,
        "max_seqlen_k": mk,
        "softmax_scale": scale,
        "causal": True,
        "window_size": win,
        "block_table": bt,
        "softcap": 0,
        "q_descale": qd,
        "k_descale": kd,
        "v_descale": vd,
        "shuffled_kv_cache": shuffled,
    }


def flydsl_route_name(kw):
    """Which kernel the default route runs (spies, as bench_fua_rebaseline.kernel_name)."""
    from unittest import mock

    import torch

    import aiter.ops.flydsl.unified_attention_kernels as uak
    import aiter.ops.unified_attention as ua

    seen = {}
    real = uak.flydsl_unified_attention

    def spy(*a, **k):
        r = real(*a, **k)
        seen["served"] = r is not None
        return r

    with mock.patch.object(uak, "flydsl_unified_attention", spy):
        ua.unified_attention(**kw, backend=None)
    torch.cuda.synchronize()
    return (
        "flydsl_prefill_or_mixed"
        if seen.get("served")
        else "declined_fallback_triton_gluon"
    )


def run_perf(
    kernels, shape, qls, kvs, layouts, causals=(1,), groups=(16,), dtypes=("bf16",)
):
    import torch

    rows = []
    for layout, group, dt in itertools.product(layouts, groups, dtypes):
        kw = build_perf(qls, kvs, layout == "shuffled", group, dt)
        for causal in causals:
            kw["causal"] = bool(causal)
            for k in kernels:
                row = {
                    "kind": "perf",
                    "shape": shape,
                    "kernel": k,
                    "layout": layout,
                    "causal": int(causal),
                    "group": group,
                    "out_dtype": dt,
                    "M": sum(qls),
                    "nseq": len(qls),
                }
                try:
                    why = champion_unsupported(kw) if k in CHAMP_MODULE else None
                    if why:
                        row["status"] = why
                    else:
                        if k == "flydsl":
                            row["kernel_name"] = flydsl_route_name(kw)
                        kw["out"].zero_()
                        fn = make_call(k, kw)
                        fn()
                        torch.cuda.synchronize()
                        row["us"], row["timing_mode"] = time_us(fn)
                        row["status"] = "ok"
                except Exception as ex:  # noqa: BLE001
                    row["status"] = f"error: {type(ex).__name__}: {ex}"[:200]
                    torch.cuda.synchronize()
                print({k2: v for k2, v in row.items()}, flush=True)
                rows.append(row)
        del kw
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- correctness
def run_corr(kernels, shape, qls, kvs, opts, layouts, causals, poisons):
    import torch

    from op_tests.test_unified_attention import (
        make_case,
        ref_paged_attn,
        shuffle_kv_cache,
    )

    group, dt = opts.get("group", 16), opts.get("dtype", "bf16")
    heads = HKV * group
    causals = [c for c in causals if c in opts.get("causals", (True, False))]
    rows = []
    for causal in causals:
        case = make_case(qls, kvs, DT[dt], causal=causal, group=group)
        want = ref_paged_attn(
            case["q"],
            case["k"],
            case["v"],
            qls,
            kvs,
            case["block_table"],
            case["softmax_scale"],
            q_descale=case["q_descale"],
            k_descale=case["k_descale"],
            v_descale=case["v_descale"],
            causal=int(causal),
        )
        want = torch.nan_to_num(
            want.float()
        )  # empty-KV rows: kernels write zeros (op test does the same)
        atol = FP8_REL_ERR * want.abs().max().item()
        for poison, layout in itertools.product(poisons, layouts):
            k, v = case["k"].clone(), case["v"].clone()
            if poison:

                def pad(t):
                    return torch.cat(
                        [
                            t.view(torch.uint8),
                            torch.full(
                                (16, *t.shape[1:]),
                                0x7F,
                                dtype=torch.uint8,
                                device="cuda",
                            ),
                        ]
                    ).view(t.dtype)

                for t in (k, v):
                    tv = t.view(torch.uint8)
                    for row_i, n in enumerate(kvs):
                        if n % PAGE:
                            tv[
                                int(case["block_table"][row_i, n // PAGE]), n % PAGE :
                            ] = 0x7F
                k, v = pad(k), pad(
                    v
                )  # 16 unused trailing pages, never in the block table
            if layout == "shuffled":
                k, v = shuffle_kv_cache(k, v)
            total = sum(qls)
            kw = dict(case, k=k, v=v, shuffled_kv_cache=layout == "shuffled")
            if opts.get("garbage_bt"):
                # 3 extra columns plus every entry beyond the pages a row needs: ids of poisoned unused pages when
                # poison (a read there shows as NaN), else random valid ids.
                n_pages = case["k"].shape[0]
                bt = case["block_table"]
                need = torch.tensor(
                    [-(-n // PAGE) for n in kvs], device="cuda"
                ).unsqueeze(1)
                lo, hi = (n_pages, n_pages + 16) if poison else (0, n_pages)
                gb = torch.randint(
                    lo,
                    hi,
                    (bt.shape[0], bt.shape[1] + 3),
                    device="cuda",
                    dtype=torch.int32,
                )
                cols = torch.arange(gb.shape[1], device="cuda").unsqueeze(0)
                gb[:, : bt.shape[1]] = torch.where(
                    cols[:, : bt.shape[1]] < need, bt, gb[:, : bt.shape[1]]
                )
                kw["block_table"] = gb.contiguous()
            outs = {}
            for kernel in dict.fromkeys(
                list(kernels) + ["triton"]
            ):  # triton output is the cross-check
                row = {
                    "kind": "corr",
                    "shape": shape,
                    "kernel": kernel,
                    "layout": layout,
                    "causal": int(causal),
                    "poison": int(poison),
                    "group": group,
                    "out_dtype": dt,
                    "M": total,
                    "nseq": len(qls),
                    "atol": atol,
                    "reported": kernel in kernels,
                }
                try:
                    why = champion_unsupported(kw) if kernel in CHAMP_MODULE else None
                    if why:
                        row["status"] = why
                    else:
                        storage = torch.full(
                            (total + 128, heads, D),
                            float("nan"),
                            device="cuda",
                            dtype=DT[dt],
                        )
                        kw["out"] = storage[:total]
                        make_call(kernel, kw, for_corr=True)()
                        torch.cuda.synchronize()
                        got = kw["out"].float()
                        outs[kernel] = got
                        d = (got - want).abs()
                        nan = bool(torch.isnan(got).any().item())
                        d = torch.nan_to_num(d, nan=float("inf"))
                        row.update(
                            max_abs_err=d.max().item(),
                            mean_abs_err=d.mean().item(),
                            has_nan=nan,
                            canary_intact=bool(
                                torch.isnan(storage[total:]).all().item()
                            ),
                            mismatch_frac=(d > atol).float().mean().item(),
                            mismatch_frac_tritontol=(d > 0.15 + 0.15 * want.abs())
                            .float()
                            .mean()
                            .item(),
                        )
                        row["pass"] = bool(
                            not nan
                            and row["mismatch_frac"] == 0
                            and row["canary_intact"]
                        )
                        row["pass_tritontol"] = bool(
                            not nan and row["mismatch_frac_tritontol"] <= 0.01
                        )
                        row["status"] = "ok"
                except Exception as ex:  # noqa: BLE001
                    msg = f"{type(ex).__name__}: {ex}"[:200]
                    row["status"] = (
                        "unsupported: "
                        if "unavailable or does not support" in msg
                        else "error: "
                    ) + msg
                    torch.cuda.synchronize()
                rows.append(row)
            for r in rows:  # vs-triton column for rows of this variant
                if (
                    (r["shape"], r["layout"], r["causal"], r["poison"])
                    == (shape, layout, int(causal), int(poison))
                    and r["kernel"] in outs
                    and "triton" in outs
                ):
                    r["max_abs_err_vs_triton"] = (
                        (outs[r["kernel"]] - outs["triton"]).abs().max().item()
                    )
            for r in rows[-len(set(list(kernels) + ["triton"])) :]:
                print({a: b for a, b in r.items()}, flush=True)
    return [r for r in rows if r["reported"]]


# ---------------------------------------------------------------- merge
def fmt(x, p=1):
    return "-" if x is None or math.isnan(x) else f"{x:.{p}f}"


def merge(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(f) as fh:
                rows += json.load(fh)["rows"]
        except Exception as ex:  # noqa: BLE001
            print(f"skip {f}: {ex}", file=sys.stderr)
    tags = sorted({r["tag"] for r in rows if r.get("tag")})
    out = [
        "# bench_champions summary",
        f"dir: {d}  jobs rows: {len(rows)}"
        + (f"  tag: {','.join(tags)}" if tags else ""),
        "",
    ]
    perf = [r for r in rows if r["kind"] == "perf"]

    def key(r):
        return (
            r["layout"],
            r.get("causal", 1),
            r.get("group", 16),
            r.get("out_dtype", "bf16"),
        )

    for layout, causal, group, odt in sorted({key(r) for r in perf}):
        pr = [r for r in perf if key(r) == (layout, causal, group, odt)]
        layout = f"{layout}, causal={causal}, group={group}, out={odt}"
        shapes = list(dict.fromkeys(r["shape"] for r in pr))
        tab = {s: {r["kernel"]: r for r in pr if r["shape"] == s} for s in shapes}

        def us(s, k, _tab=tab):
            return _tab[s].get(k, {}).get("us", float("nan"))

        out += [
            f"## Perf, layout={layout} (us, median of {SAMPLES}, cache-defeated graph replay)",
            "",
            (
                "| shape | M | exact | noslack | flydsl | gluon | triton | exact/flydsl | exact/gluon | exact/triton "
                "| noslack/flydsl | noslack/gluon | noslack/triton |"
            ),
            "|" + "---|" * 13,
        ]
        ratios = {}
        for s in shapes:
            m = next(iter(tab[s].values()))["M"]
            cells = [fmt(us(s, k)) for k in KERNELS]
            rs = []
            for c in ("exact", "noslack"):
                for b in ("flydsl", "gluon", "triton"):
                    r = us(s, c) / us(s, b)
                    ratios.setdefault((c, b), []).append(r)
                    rs.append(fmt(r, 3))
            out.append(
                f"| {s} | {m} | " + " | ".join(cells) + " | " + " | ".join(rs) + " |"
            )
        gm = [
            fmt(geomean(ratios.get((c, b), [])), 3)
            for c in ("exact", "noslack")
            for b in ("flydsl", "gluon", "triton")
        ]
        out += [
            "| **geomean** | | | | | | | " + " | ".join(gm) + " |",
            "",
            "ratio = champion_us / comparator_us (<1 champion faster); '-' = missing/unsupported/error.",
            "",
        ]
        bad = [
            f"{r['shape']}/{r['kernel']}: {r['status']}"
            for r in pr
            if r["status"] != "ok"
        ]
        if bad:
            out += ["Not ok: " + "; ".join(bad), ""]
        names = sorted(
            {f"{r['shape']}={r['kernel_name']}" for r in pr if r.get("kernel_name")}
        )
        if names:
            out += ["Incumbent route: " + ", ".join(names), ""]
    corr = [r for r in rows if r["kind"] == "corr"]
    if corr:
        out += [
            "## Correctness",
            "",
            (
                "pass = no NaN, canary intact, zero elements > 7.7e-2*max|ref| (op_tests/test_unified_attention.py); "
                "pass_tritontol = <=1% elements beyond atol=rtol=0.15. Worst-case errors over shapes/variants."
            ),
            "",
            (
                "| kernel | variants ok | pass | pass_tritontol | unsupported | error | NaN | worst max_abs_err "
                "| worst mean_abs_err | worst vs triton |"
            ),
            "|" + "---|" * 10,
        ]
        for k in KERNELS:
            kr = [r for r in corr if r["kernel"] == k]
            if not kr:
                continue
            ok = [r for r in kr if r["status"] == "ok"]

            def w(key, _ok=ok):
                return max([r[key] for r in _ok if key in r], default=float("nan"))

            out.append(
                f"| {k} | {len(ok)}/{len(kr)} | {sum(r['pass'] for r in ok)} | "
                f"{sum(r['pass_tritontol'] for r in ok)} | {sum(r['status'].startswith('unsupported') for r in kr)} "
                f"| {sum(r['status'].startswith('error') for r in kr)} | {sum(r['has_nan'] for r in ok)} | "
                f"{fmt(w('max_abs_err'), 4)} | {fmt(w('mean_abs_err'), 5)} | {fmt(w('max_abs_err_vs_triton'), 4)} |"
            )
        out += [
            "",
            "### Failures / errors / unsupported (non-pass rows)",
            "",
            "| kernel | shape | layout | causal | poison | status | max_abs | mean_abs | nan | mismatch_frac |",
            "|" + "---|" * 10,
        ]
        for r in sorted(
            corr,
            key=lambda r: (
                r["kernel"],
                r["shape"],
                r["layout"],
                r["causal"],
                r["poison"],
            ),
        ):
            if r["status"] == "ok" and r["pass"]:
                continue
            out.append(
                f"| {r['kernel']} | {r['shape']} | {r['layout']} | {r['causal']} | {r['poison']} | "
                f"{r['status']} | {fmt(r.get('max_abs_err'), 4)} | {fmt(r.get('mean_abs_err'), 5)} | "
                f"{r.get('has_nan', '-')} | {fmt(r.get('mismatch_frac'), 5)} |"
            )
    text = "\n".join(out) + "\n"
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write(text)
    csvp = os.path.join(d, "all_rows.csv")
    keys = sorted({k for r in rows for k in r})
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(text)
    print(f"wrote {os.path.join(d, 'summary.md')} and {csvp}")


def load_rows(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(f) as fh:
                rows += json.load(fh)["rows"]
        except Exception as ex:  # noqa: BLE001
            print(f"skip {f}: {ex}", file=sys.stderr)
    return rows


def compare(da, db):
    """Per-shape incumbent time ratio B/A (<1: B faster) per (layout, causal, group, dtype), plus geomean."""

    def table(d):
        return {
            (
                r["layout"],
                r.get("causal", 1),
                r.get("group", 16),
                r.get("out_dtype", "bf16"),
                r["shape"],
            ): r
            for r in load_rows(d)
            if r["kind"] == "perf" and r["kernel"] == "flydsl"
        }

    ta, tb = table(da), table(db)

    def tag(t):
        return ",".join(sorted({r.get("tag", "") for r in t.values()} - {""})) or "-"

    out = [
        f"# flydsl compare  A={da} (tag {tag(ta)})  B={db} (tag {tag(tb)})",
        "ratio = B_us / A_us (<1 B faster)",
        "",
    ]
    for var in sorted({k[:4] for k in ta} | {k[:4] for k in tb}):
        out += [
            f"## layout={var[0]} causal={var[1]} group={var[2]} out={var[3]}",
            "",
            "| shape | A us | B us | B/A |",
            "|---|---|---|---|",
        ]
        rs = []
        for k in [k for k in ta if k[:4] == var] + [
            k for k in tb if k[:4] == var and k not in ta
        ]:
            a, b = ta.get(k, {}), tb.get(k, {})
            ua, ub = a.get("us", float("nan")), b.get("us", float("nan"))
            r = ub / ua
            rs.append(r)
            note = (
                ""
                if a.get("status", "ok") == "ok" and b.get("status", "ok") == "ok"
                else f" ({a.get('status')} / {b.get('status')})"
            )
            out.append(f"| {k[4]} | {fmt(ua)} | {fmt(ub)} | {fmt(r, 4)}{note} |")
        out += [f"| **geomean** | | | {fmt(geomean(rs), 4)} |", ""]
    print("\n".join(out))


def pick(spec, names, what):
    if not spec:
        return list(names)
    keys = list(names)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < len(keys):
            out.append(keys[int(tok)])
        elif tok in names:
            out.append(tok)
        elif tok.endswith("*"):
            out += [k for k in keys if k.startswith(tok[:-1])]
        else:
            raise SystemExit(f"unknown {what} {tok!r}; choices: {keys}")
    return list(dict.fromkeys(out))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--kernel", help=f"comma list of {KERNELS} (default all)")
    p.add_argument(
        "--shape",
        help="comma list of shape names, indices into --list, or prefix* (default: all shapes "
        "of the chosen mode: c_* for correctness, trace_*/ticket_* for perf)",
    )
    p.add_argument("--mode", choices=("correctness", "perf", "both"), default="both")
    p.add_argument(
        "--layout",
        choices=("shuffled", "linear", "both"),
        default="both",
        help="KV layout; perf default is shuffled only when 'both' is not asked explicitly (see --perf-layout)",
    )
    p.add_argument(
        "--perf-layout",
        choices=("shuffled", "linear", "both"),
        default="shuffled",
        help="perf layouts (default shuffled: the production layout, as bench_fua_rebaseline)",
    )
    p.add_argument(
        "--out", help="result file (.json rows; .csv also written next to it)"
    )
    p.add_argument(
        "--merge", metavar="DIR", help="merge DIR/*.json into DIR/summary.md and exit"
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        help="print flydsl B/A time ratios and exit",
    )
    p.add_argument(
        "--causal",
        choices=("1", "0", "both"),
        default="1",
        help="perf causal mode (default 1)",
    )
    p.add_argument(
        "--group",
        choices=("4", "8", "16", "all"),
        default="16",
        help="perf Q heads per KV head (Q heads = 4*group, 4 KV heads); default 16",
    )
    p.add_argument(
        "--out-dtype",
        choices=("bf16", "f16", "both"),
        default="bf16",
        help="perf output dtype",
    )
    p.add_argument("--tag", default="", help="label stored in every row")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.merge:
        return merge(a.merge)
    if a.compare:
        return compare(*a.compare)
    shapes_all = all_shapes()
    if a.list:
        for i, (n, (q, k, *_)) in enumerate(shapes_all.items()):
            print(i, n, f"M={sum(q)} nseq={len(q)}")
        return
    kernels = [
        ALIASES.get(x, x) for x in (a.kernel.split(",") if a.kernel else KERNELS)
    ]
    bad = [k for k in kernels if k not in KERNELS]
    if bad:
        p.error(f"unknown kernel {bad}; choices {KERNELS}")
    if not a.out:
        p.error("--out required")
    if os.environ.get("ENABLE_CK") != "0":
        p.error("need ENABLE_CK=0")
    import torch

    DT.update(bf16=torch.bfloat16, f16=torch.float16)
    causals = {"1": (1,), "0": (0,), "both": (1, 0)}[a.causal]
    groups = GROUPS if a.group == "all" else (int(a.group),)
    dtypes = ("bf16", "f16") if a.out_dtype == "both" else (a.out_dtype,)
    corr_layouts = ["linear", "shuffled"] if a.layout == "both" else [a.layout]
    perf_layouts = (
        ["shuffled", "linear"] if a.perf_layout == "both" else [a.perf_layout]
    )
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def flush():
        with open(a.out, "w") as f:
            json.dump(
                {
                    "argv": sys.argv,
                    "device": torch.cuda.get_device_name(),
                    "rows": rows,
                },
                f,
                indent=1,
            )
        if a.out.endswith(".json"):
            keys = sorted({k for r in rows for k in r})
            with open(a.out[:-5] + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, restval="")
                w.writeheader()
                w.writerows(rows)

    print(
        f"dev={torch.cuda.get_device_name()} HIP={os.environ.get('HIP_VISIBLE_DEVICES')}",
        flush=True,
    )
    cs, ps = corr_shapes(), perf_shapes()
    for s_name in pick(a.shape, shapes_all, "shape"):
        if s_name in cs and a.mode in ("correctness", "both"):
            qls, kvs, *o = cs[s_name]
            new = run_corr(
                kernels,
                s_name,
                qls,
                kvs,
                o[0] if o else {},
                corr_layouts,
                (True, False),
                (False, True),
            )
            rows += [dict(r, tag=a.tag) for r in new]
            flush()
        if s_name in ps and a.mode in ("perf", "both"):
            new = run_perf(
                kernels, s_name, *ps[s_name], perf_layouts, causals, groups, dtypes
            )
            rows += [dict(r, tag=a.tag) for r in new]
            flush()
    flush()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
