# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf-only benchmark for the FlyDSL m32x8 MHA forward kernel on gfx1250.

This is the performance companion to ``op_tests/test_mha_flydsl.py`` — it drives
the same public interface (``flash_attn_func`` / ``flash_attn_varlen_func`` in
``aiter/ops/mha.py``) but computes NO reference and checks NO correctness. It only
times the kernel and reports latency / TFLOPS / TB-s.

The m32x8 kernel (``fmha_fwd_prefill_a16w16_m32x8.py``, dispatched as ``*_m32x8``) serves the
D_qk in {128,192,256} / D_v=128 bf16 path, gated behind ``AITER_ENABLE_EXPERIMENTAL=1``.
Without that env var the public wrappers fall through to CK, so this harness asserts
the gate is on and spies on the dispatch entry to fail loudly on a silent fallthrough.

Two modes:

1. Detailed args — drive one exact shape from the CLI:
     AITER_ENABLE_EXPERIMENTAL=1 python op_tests/perf_mha_flydsl.py \\
         --layout thd -b 2 -sq 4095 -sk 4095 -nh 16 -nhkv 4 -c

2. Predefined cases — replay curated shapes by id:
     AITER_ENABLE_EXPERIMENTAL=1 python op_tests/perf_mha_flydsl.py --case 0 4
     AITER_ENABLE_EXPERIMENTAL=1 python op_tests/perf_mha_flydsl.py --list   # show, don't run

seqlen ranges 0..2^17; the cases sprinkle in odd (non-power-of-two) lengths. hdim is
fixed at 128/128 for now. Defaults: --warmup 2 --repeat 5.
"""

import argparse
import contextlib
import math
import os
import sys

# The m32x8 (hdim 128/128) path is gated behind AITER_ENABLE_EXPERIMENTAL; this
# harness targets exactly that kernel, so enable it automatically. Must be set
# BEFORE aiter is imported so both the Python gate (is_experimental_enabled reads
# os.environ live) and the C++ side see it. Force it on rather than setdefault:
# the test cannot run the kernel under test without it.
os.environ["AITER_ENABLE_EXPERIMENTAL"] = "1"

import pandas as pd
import torch

# NOTE: aiter is imported lazily inside main()/run_perf so `--list` (and --help)
# stay cheap — importing aiter triggers the heavy JIT module build/load.

DEFAULT_HDIM_QK = 128
DEFAULT_HDIM_V = 128


# ============================================================================
# Timing + dispatch verification
# ============================================================================


def _time_fn(fn, warmup, repeat):
    """Average latency (ms) over ``repeat`` runs after ``warmup`` un-timed runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(repeat):
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    return sum(latencies) / len(latencies)


@contextlib.contextmanager
def _verify_dispatch(layout):
    """Spy on the m32x8 kernel entry so a silent fall-through to CK is a hard error.

    The public wrappers only route the 128/128 bf16 path to the FlyDSL m32x8 kernel
    when ``AITER_ENABLE_EXPERIMENTAL=1`` and the shape is supported; otherwise they
    return None and the caller runs CK, which would quietly report perf for the wrong
    kernel. Patch the innermost entry with a counting wrapper and assert it fired.
    """
    import aiter.ops.flydsl.fmha_kernels as fk

    name = "flash_attn_varlen_m32x8" if layout == "thd" else "flash_attn_batch_m32x8"
    orig = getattr(fk, name)
    calls = {"n": 0}

    def _spy(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    setattr(fk, name, _spy)
    try:
        yield calls
    finally:
        setattr(fk, name, orig)
    assert calls["n"] > 0, (
        f"m32x8 kernel ({name}) never dispatched for layout={layout}; the call fell "
        f"through to CK. Run with AITER_ENABLE_EXPERIMENTAL=1 and hdim 128/128."
    )


# ============================================================================
# Roofline helpers
# ============================================================================


def _fwd_flops(B, sq, sk, Hq, d_qk, d_v, causal):
    """Forward FLOPs: QK^T + PV over B uniform-length sequences; causal halves it."""
    f = B * Hq * (2 * sq * sk * d_qk + 2 * sq * sk * d_v)
    return f // 2 if causal else f


def _fwd_bytes(B, sq, sk, Hq, Hkv, d_qk, d_v, lse):
    """HBM traffic: read Q/K/V, write O (all bf16 = 2B); +LSE (fp32 = 4B) when returned."""
    total_q = B * sq
    total_k = B * sk
    elems = (
        total_q * Hq * d_qk  # Q
        + total_k * Hkv * d_qk  # K
        + total_k * Hkv * d_v  # V
        + total_q * Hq * d_v  # O
    )
    nbytes = elems * 2
    if lse:
        nbytes += total_q * Hq * 4
    return nbytes


def _tflops(flop, ms):
    return float("inf") if ms <= 0 else flop / ms / 1e9


def _tbps(nbytes, ms):
    return float("inf") if ms <= 0 else nbytes / ms / 1e6


def _make_sink(nheads_q, device):
    """Per-head fp32 sink logits (scaled-score domain), spread across heads."""
    return torch.linspace(-2.0, 5.0, nheads_q, dtype=torch.float32, device=device)


# ============================================================================
# Case runner
# ============================================================================


def run_perf(case, warmup, repeat, fix_init=None):
    """Build bf16 inputs for one case, time the kernel, return a summary row.

    ``case`` keys: layout, B, sq, sk, Hq, Hkv, d_qk, d_v, causal, lse, window, sink.
    ``fix_init``: None -> random-normal q/k/v; a float -> every q/k/v element set
    to that constant. A constant fill keeps the operand bits from toggling, so the
    data buses draw little dynamic power and the clock does not throttle -- a
    stable, data-independent perf ceiling (vs random, whose bit flips can pull
    clocks down).
    """
    from aiter.ops.mha import flash_attn_func, flash_attn_varlen_func

    device = torch.device("cuda")
    torch.manual_seed(42)

    layout = case["layout"]
    B, sq, sk = case["B"], case["sq"], case["sk"]
    Hq = case["Hq"]
    Hkv = case.get("Hkv") or Hq
    d_qk = case.get("d_qk", DEFAULT_HDIM_QK)
    d_v = case.get("d_v", DEFAULT_HDIM_V)
    causal = case.get("causal", False)
    lse = case.get("lse", False)
    window = tuple(case.get("window", (-1, -1)))
    sink = case.get("sink", False)

    assert Hq % Hkv == 0, f"nheads_q={Hq} must be a multiple of nheads_kv={Hkv}"

    # Always measure OUR m32x8 kernel. Experimental must be ON for 128 (the PR3039
    # gfx1250 ASM gate in mha.py owns 128/128-causal unless experimental yields it to
    # FlyDSL) and for 256 (FlyDSL router only takes 256 when experimental). For 192 it
    # must be OFF: 192 reaches our kernel by default, and exp=1 routes it to the d192
    # sibling instead. So the only no-exp case is 192.
    os.environ["AITER_ENABLE_EXPERIMENTAL"] = "0" if d_qk == 192 else "1"

    scale = 1.0 / math.sqrt(d_qk)
    sink_t = _make_sink(Hq, device) if sink else None

    if layout == "thd":
        total_q, total_k = B * sq, B * sk
        q = torch.randn(total_q, Hq, d_qk, dtype=torch.bfloat16, device=device)
        k = torch.randn(total_k, Hkv, d_qk, dtype=torch.bfloat16, device=device)
        v = torch.randn(total_k, Hkv, d_v, dtype=torch.bfloat16, device=device)
        if fix_init is not None:
            q.fill_(fix_init)
            k.fill_(fix_init)
            v.fill_(fix_init)
        cu_q = torch.arange(0, (B + 1) * sq, sq, dtype=torch.int32, device=device)
        cu_k = torch.arange(0, (B + 1) * sk, sk, dtype=torch.int32, device=device)

        def _run():
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_q,
                cu_k,
                sq,
                sk,
                softmax_scale=scale,
                causal=causal,
                window_size=window,
                return_lse=lse,
                sink_ptr=sink_t,
            )

    elif layout == "bshd":
        q = torch.randn(B, sq, Hq, d_qk, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, sk, Hkv, d_qk, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, sk, Hkv, d_v, dtype=torch.bfloat16, device=device)
        if fix_init is not None:
            q.fill_(fix_init)
            k.fill_(fix_init)
            v.fill_(fix_init)

        def _run():
            return flash_attn_func(
                q,
                k,
                v,
                softmax_scale=scale,
                causal=causal,
                window_size=window,
                return_lse=lse,
                sink_ptr=sink_t,
            )

    else:
        raise ValueError(f"unknown layout {layout!r} (expected 'thd' or 'bshd')")

    with _verify_dispatch(layout):
        avg_ms = _time_fn(_run, warmup, repeat)
        _run()  # one more inside the spy to confirm dispatch

    flop = _fwd_flops(B, sq, sk, Hq, d_qk, d_v, causal)
    nbytes = _fwd_bytes(B, sq, sk, Hq, Hkv, d_qk, d_v, lse)
    avg_us = avg_ms * 1000
    tflops = _tflops(flop, avg_ms)
    tbps = _tbps(nbytes, avg_ms)

    tag = (
        f"{layout} B={B} H={Hq}/{Hkv} d={d_qk}/{d_v} sq={sq} sk={sk} "
        f"causal={causal} lse={lse} window={window} sink={sink}"
    )
    print(
        f"  [{tag}] avg: {avg_ms:.3f}ms ({avg_us:.1f} us)  "
        f"{tflops:.1f} TFLOPS  {tbps:.1f} TB/s"
    )

    return {
        "id": case.get("id", ""),
        "layout": layout,
        "B": B,
        "Hq": Hq,
        "Hkv": Hkv,
        "d_qk": d_qk,
        "d_v": d_v,
        "sq": sq,
        "sk": sk,
        "causal": causal,
        "lse": lse,
        "window": str(window),
        "sink": sink,
        "us": round(avg_us, 2),
        "TFLOPS": round(tflops, 2),
        "TB/s": round(tbps, 2),
    }


# ============================================================================
# Predefined cases — seqlen spans tiny..2^17, with odd (non-power-of-two) lengths
# sprinkled in. Every base shape below is replayed at D_qk in {128, 192, 256} (D_v=128).
# ============================================================================

_BASE_CASES = [
    # --- thd (varlen) ---
    {
        "layout": "thd",
        "B": 1,
        "sq": 128,
        "sk": 128,
        "Hq": 8,
        "desc": "tiny square, single tile",
    },
    {
        "layout": "thd",
        "B": 1,
        "sq": 341,
        "sk": 341,
        "Hq": 8,
        "desc": "odd single-tile-ish square",
    },
    {
        "layout": "thd",
        "B": 2,
        "sq": 713,
        "sk": 713,
        "Hq": 16,
        "causal": True,
        "desc": "odd multi-tile, causal, multi-batch",
    },
    {
        "layout": "thd",
        "B": 1,
        "sq": 4095,
        "sk": 4095,
        "Hq": 16,
        "Hkv": 4,
        "causal": True,
        "desc": "odd 4k, causal, GQA 4x",
    },
    {"layout": "thd", "B": 1, "sq": 1, "sk": 4096, "Hq": 8, "desc": "decode-like sq=1"},
    {
        "layout": "thd",
        "B": 1,
        "sq": 897,
        "sk": 4097,
        "Hq": 8,
        "causal": True,
        "desc": "odd sq!=sk, causal (chunked-prefill-like)",
    },
    {
        "layout": "thd",
        "B": 1,
        "sq": 2048,
        "sk": 2048,
        "Hq": 8,
        "lse": True,
        "desc": "return_lse",
    },
    {
        "layout": "thd",
        "B": 1,
        "sq": 2048,
        "sk": 2048,
        "Hq": 8,
        "window": (256, 256),
        "desc": "sliding window (256,256)",
    },
    {
        "layout": "thd",
        "B": 1,
        "sq": 2048,
        "sk": 2048,
        "Hq": 8,
        "sink": True,
        "desc": "attention sink",
    },
    {
        "layout": "thd",
        "B": 1,
        "sq": 131071,
        "sk": 131071,
        "Hq": 4,
        "Hkv": 1,
        "causal": True,
        "desc": "odd ~2^17 long-context, causal, MQA",
    },
    {
        "layout": "thd",
        "B": 4,
        "sq": 16384,
        "sk": 16384,
        "Hq": 8,
        "causal": True,
        "desc": "16k square, causal, multi-batch (Hq==Hkv)",
    },
    # --- bshd (batch) ---
    {
        "layout": "bshd",
        "B": 2,
        "sq": 512,
        "sk": 512,
        "Hq": 8,
        "desc": "square, multi-batch",
    },
    {
        "layout": "bshd",
        "B": 1,
        "sq": 4095,
        "sk": 4095,
        "Hq": 8,
        "causal": True,
        "desc": "odd 4k, causal",
    },
    {
        "layout": "bshd",
        "B": 2,
        "sq": 1024,
        "sk": 1024,
        "Hq": 32,
        "Hkv": 8,
        "causal": True,
        "desc": "GQA 4x, causal",
    },
    {
        "layout": "bshd",
        "B": 1,
        "sq": 2048,
        "sk": 2048,
        "Hq": 8,
        "lse": True,
        "desc": "return_lse",
    },
    {
        "layout": "bshd",
        "B": 1,
        "sq": 2048,
        "sk": 2048,
        "Hq": 8,
        "window": (128, 0),
        "desc": "causal-shaped window (128,0)",
    },
    {
        "layout": "bshd",
        "B": 1,
        "sq": 2048,
        "sk": 2048,
        "Hq": 8,
        "sink": True,
        "desc": "attention sink",
    },
    {
        "layout": "bshd",
        "B": 1,
        "sq": 32769,
        "sk": 32769,
        "Hq": 8,
        "causal": True,
        "desc": "odd long-context, causal",
    },
]

# Replay every base shape at D_qk in {128, 192, 256} (D_v=128), D_qk set explicitly on each.
# Case ids: 128 block = [0, N), 192 = [N, 2N), 256 = [2N, 3N) where N = len(_BASE_CASES).
CASES = [dict(_c, d_qk=_dqk, d_v=128) for _dqk in (128, 192, 256) for _c in _BASE_CASES]
for _i, _c in enumerate(CASES):
    _c["id"] = _i


def _print_case_list():
    rows = [
        {
            "id": c["id"],
            "layout": c["layout"],
            "B": c["B"],
            "Hq": c["Hq"],
            "Hkv": c.get("Hkv") or c["Hq"],
            "d_qk": c.get("d_qk", DEFAULT_HDIM_QK),
            "d_v": c.get("d_v", DEFAULT_HDIM_V),
            "sq": c["sq"],
            "sk": c["sk"],
            "causal": c.get("causal", False),
            "lse": c.get("lse", False),
            "window": str(tuple(c.get("window", (-1, -1)))),
            "sink": c.get("sink", False),
            "desc": c.get("desc", ""),
        }
        for c in CASES
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def _parse_window(s):
    if s is None:
        return (-1, -1)
    parts = [int(x) for x in s.split(",")]
    assert len(parts) == 2, f"--window expects 'left,right', got {s!r}"
    return (parts[0], parts[1])


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Perf-only benchmark for the FlyDSL m32x8 MHA kernel (gfx1250, "
        "bf16, hdim 128/128).\nNo reference / no correctness check.",
    )
    parser.add_argument(
        "--layout",
        choices=["thd", "bshd"],
        default="thd",
        help="Layout for detailed-args mode (default thd).",
    )
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "-sq",
        "--seqlen_q",
        type=int,
        default=None,
        help="Query seqlen. Setting -sq and -sk triggers detailed mode.",
    )
    parser.add_argument(
        "-sk",
        "--seqlen_kv",
        type=int,
        default=None,
        help="Key/value seqlen. Setting -sq and -sk triggers detailed mode.",
    )
    parser.add_argument("-nh", "--nheads", type=int, default=8, help="Query heads.")
    parser.add_argument(
        "-nhkv",
        "--nheads_kv",
        type=int,
        default=None,
        help="KV heads (GQA). Must divide --nheads. Default = --nheads.",
    )
    parser.add_argument(
        "-dqk",
        "--hdim_qk",
        type=int,
        default=DEFAULT_HDIM_QK,
        help="Head dim of q/k (only 128 supported for now).",
    )
    parser.add_argument(
        "-dv",
        "--hdim_v",
        type=int,
        default=DEFAULT_HDIM_V,
        help="Head dim of v (only 128 supported for now).",
    )
    parser.add_argument("-c", "--causal", action="store_true", help="Causal mask.")
    parser.add_argument("-l", "--lse", action="store_true", help="Return LSE.")
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Sliding window 'left,right' (default full attention).",
    )
    parser.add_argument("-s", "--sink", action="store_true", help="Attention sink.")
    parser.add_argument(
        "--case",
        type=int,
        nargs="+",
        default=None,
        help="Run predefined case id(s), e.g. --case 0 4.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List predefined cases and exit (no execution).",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="Warmup iters (default 2)."
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="Timed iters (default 5)."
    )
    parser.add_argument(
        "--fix-init",
        type=float,
        nargs="?",
        const=0.25,
        default=None,
        metavar="VALUE",
        help="Fill q/k/v with a constant instead of random normal (default when "
        "the flag is absent). Bare --fix-init uses 0.25; --fix-init VALUE uses "
        "VALUE. A constant fill stops the operand buses toggling, so the clock "
        "does not throttle -- a stable, data-independent perf ceiling.",
    )
    args = parser.parse_args()

    if args.list:
        _print_case_list()
        return

    # Import aiter only now — kept out of the module top so --list / --help don't
    # pay the JIT module build/load cost.
    import aiter
    from aiter.jit.core import is_experimental_enabled

    if aiter.get_gfx() != "gfx1250":
        print(f"Skipping: perf test requires gfx1250 (current: {aiter.get_gfx()})")
        sys.exit(0)

    # The 128/128 m32x8 path is gated behind experimental (forced on at module top);
    # without it the public wrappers silently run CK. Assert the invariant held in case
    # the env was clobbered after import.
    assert (
        is_experimental_enabled()
    ), "AITER_ENABLE_EXPERIMENTAL must be enabled for the m32x8 (hdim 128/128) path"

    if args.case is not None:
        selected = []
        for cid in args.case:
            assert 0 <= cid < len(CASES), f"--case {cid} out of range [0,{len(CASES)})"
            selected.append(CASES[cid])
    elif args.seqlen_q is not None and args.seqlen_kv is not None:
        Hkv = args.nheads_kv if args.nheads_kv is not None else args.nheads
        selected = [
            {
                "layout": args.layout,
                "B": args.batch,
                "sq": args.seqlen_q,
                "sk": args.seqlen_kv,
                "Hq": args.nheads,
                "Hkv": Hkv,
                "d_qk": args.hdim_qk,
                "d_v": args.hdim_v,
                "causal": args.causal,
                "lse": args.lse,
                "window": _parse_window(args.window),
                "sink": args.sink,
            }
        ]
    else:
        parser.error(
            "nothing to run: pass --case ID [ID ...], or detailed args including "
            "both -sq and -sk (see --help). Use --list to see predefined cases."
        )

    init_desc = "random" if args.fix_init is None else f"fixed={args.fix_init}"
    print("=" * 70)
    print(f"FlyDSL m32x8 MHA perf (gfx1250, bf16, init={init_desc})")
    print("=" * 70)

    rows = []
    for case in selected:
        try:
            rows.append(run_perf(case, args.warmup, args.repeat, args.fix_init))
        except Exception as e:  # noqa: BLE001 - per-case guard, continue suite
            print(f"  [id={case.get('id', '?')} {case['layout']}] ERROR: {e}")
            import traceback

            traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows)
        aiter.logger.info("flydsl_mha_perf summary:\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
