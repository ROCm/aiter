# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B performance harness: FlyDSL vs Triton FP8 MQA logits (gfx942).

Times Triton plus one or more FlyDSL kernel *variants* (kernel versions) on
identical inputs and prints a side-by-side table (time, TFLOP/s, verify, and
each impl's speedup vs Triton).

The FlyDSL kernel ships several versions, registered in
``aiter.ops.flydsl.kernels.fp8_mqa_logits`` (``KERNEL_VARIANTS``): ``"mfma"`` is
the baseline, ``"scalar"`` the correctness-first fallback, and new versions can
be added there. Pick which to benchmark with ``--flydsl-variants`` (default:
just the baseline). Each becomes its own column, e.g. ``flydsl:mfma``.

Examples:
    # default DeepSeek-ish shape, baseline FlyDSL variant vs Triton
    python op_tests/op_benchmarks/triton/bench_flydsl_vs_triton_fp8_mqa_logits.py

    # compare two FlyDSL variants against Triton on a custom shape, with parity
    python .../bench_flydsl_vs_triton_fp8_mqa_logits.py \
        --flydsl-variants mfma,scalar \
        --seq_q_l 1024 --seq_kv_l 4096 --num_heads_q 64 --head_dim 128 --verify
"""
import argparse
import functools
import os
import subprocess
import sys
from datetime import datetime, timezone

# Make the script runnable directly (python path/to/bench.py) by putting the
# repo root on sys.path, so `import aiter` / `import op_tests` resolve without
# requiring PYTHONPATH or an installed aiter package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.flydsl import is_flydsl_available
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
)


def calculate_tflops(start_inds, end_inds, num_heads_q, head_dim, time_ms):
    time_s = time_ms * 1e-3
    start_inds = start_inds.to("cpu").numpy()
    end_inds = end_inds.to("cpu").numpy()
    total_flops = 0.0
    for i in range(len(start_inds)):
        total_flops += 2.0 * num_heads_q * head_dim * (end_inds[i] - start_inds[i])
    return total_flops / (time_s * 1e12)


def _make_inputs(batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim):
    s_q = batch_size * seq_q_l
    s_k = batch_size * seq_kv_l

    q = torch.randn(s_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    # Round-trip kv through fp8 so the bf16 `kv` used by the torch reference
    # matches what the kernels actually consume (mirrors the unit test).
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads_q, device="cuda", dtype=torch.float32)

    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.zeros(s_q, dtype=torch.int, device="cuda")
    arange_q = torch.arange(seq_q_l, dtype=torch.int, device="cuda")
    for b in range(batch_size):
        qs = b * seq_q_l
        kvs = b * seq_kv_l
        ks[qs : qs + seq_q_l] = kvs
        ke[qs : qs + seq_q_l] = kvs + (seq_kv_l - seq_q_l) + arange_q + 1

    q_fp8 = q.to(e4m3_dtype)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    # `q`/`kv` (bf16) are returned for the torch reference; the rest feed the
    # actual kernels.
    return q, kv, q_fp8, kv_fp8, scales, weights, ks, ke


def _time_ms(func, warmup=25, rep=100):
    """Return median time in ms, or None if the implementation isn't ready."""
    try:
        func()  # one eager call to surface NotImplementedError / compile errors
    except NotImplementedError:
        return None
    torch.cuda.synchronize()
    return triton.testing.do_bench(func, warmup=warmup, rep=rep)


# Perf-oriented preset sweep (DeepSeek-style square + rectangular shapes).
# Each entry: (batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim).
# Covers both head counts the ticket calls out (H in {64, 128}) at D=128,
# square + rectangular windows, plus a non-aligned s_kv "tail" shape.
PRESET_SHAPES = [
    # H = 64
    (1, 1024, 1024, 64, 128),
    (1, 2048, 2048, 64, 128),
    (1, 4096, 4096, 64, 128),
    (1, 1024, 4096, 64, 128),
    (1, 4096, 16384, 64, 128),
    (1, 4096, 16000, 64, 128),  # non-aligned s_kv (tail)
    # H = 128
    (1, 1024, 1024, 128, 128),
    (1, 2048, 2048, 128, 128),
    (1, 4096, 4096, 128, 128),
    (1, 1024, 4096, 128, 128),
    (1, 4096, 16384, 128, 128),
    (1, 4096, 16000, 128, 128),  # non-aligned s_kv (tail)
]


FLYDSL_PREFIX = "flydsl:"


def _is_flydsl_impl(name):
    return name.startswith(FLYDSL_PREFIX)


def _flydsl_variant_of(name):
    """Return the variant tag for a 'flydsl:<variant>' impl name."""
    return name[len(FLYDSL_PREFIX):]


def _impl_label(name):
    """Short, column-friendly label for an impl name (e.g. 'triton', 'fly:mfma')."""
    if name == "triton":
        return "triton"
    if _is_flydsl_impl(name):
        return "fly:" + _flydsl_variant_of(name)
    return name


def _available_variants():
    """Return ``(tuple_of_variant_tags, default_tag)`` or ``(None, None)``.

    ``None`` when FlyDSL isn't importable (so the bench can still run Triton-only).
    """
    if not is_flydsl_available():
        return None, None
    from aiter.ops.flydsl import (
        FP8_MQA_LOGITS_VARIANTS,
        FP8_MQA_LOGITS_DEFAULT_VARIANT,
    )

    return tuple(FP8_MQA_LOGITS_VARIANTS), FP8_MQA_LOGITS_DEFAULT_VARIANT


def _select_impls(which, flydsl_variants):
    """Return an ordered ``{impl_name: fn}`` for the requested selection.

    ``which`` is the impl-family selection ('all'/'triton'/'flydsl').
    ``flydsl_variants`` is the resolved list of FlyDSL kernel-version tags to
    benchmark; each becomes its own impl named ``flydsl:<variant>`` bound to that
    variant via ``functools.partial``.
    """
    flydsl_fn = None
    available_variants = ()
    if is_flydsl_available():
        from aiter.ops.flydsl import (
            flydsl_fp8_mqa_logits,
            FP8_MQA_LOGITS_VARIANTS,
        )

        flydsl_fn = flydsl_fp8_mqa_logits
        available_variants = tuple(FP8_MQA_LOGITS_VARIANTS)

    want_triton = which in ("all", "triton")
    want_flydsl = which in ("all", "flydsl")

    if want_flydsl and flydsl_fn is None:
        if which == "flydsl":
            raise SystemExit(
                "[error] --impl flydsl requested but flydsl is unavailable."
            )
        print("[warn] flydsl unavailable -- benchmarking Triton only.")
        want_flydsl = False

    impls = {}
    if want_triton:
        impls["triton"] = triton_logits
    if want_flydsl:
        for v in flydsl_variants:
            if v not in available_variants:
                raise SystemExit(
                    f"[error] unknown FlyDSL variant {v!r}; available: "
                    f"{list(available_variants)}."
                )
            impls[FLYDSL_PREFIX + v] = functools.partial(flydsl_fn, variant=v)

    if not impls:
        raise SystemExit("[error] no implementations selected.")
    return impls


def run(args):
    impls = _select_impls(args.impl, args.flydsl_variants)
    shapes = _resolve_shapes(args)

    rows = []
    total = len(shapes)
    for n, (idx, (batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim)) in enumerate(
        shapes, start=1
    ):
        shape = argparse.Namespace(
            batch_size=batch_size,
            seq_q_l=seq_q_l,
            seq_kv_l=seq_kv_l,
            num_heads_q=num_heads_q,
            head_dim=head_dim,
            clean_logits=args.clean_logits,
        )
        # Progress to stderr so it doesn't interleave with the final table.
        print(f"[{n}/{total}] {_fmt_shape(shape)}", file=sys.stderr, flush=True)
        rows.append(_run_one(idx, impls, shape, args))

    _print_table(rows, impls, args)

    if args.output:
        _write_markdown(args.output, rows, impls, args)
        print(f"\n[wrote] {args.output}", file=sys.stderr, flush=True)


def _run_one(idx, impls, shape, args):
    """Time + (optionally) verify one case; return a row dict for the table."""
    q, kv, q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        shape.batch_size, shape.seq_q_l, shape.seq_kv_l, shape.num_heads_q,
        shape.head_dim,
    )

    def make_func(fn):
        return lambda: fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)

    times, tflops = {}, {}
    for name, fn in impls.items():
        t = _time_ms(make_func(fn), args.warmup, args.rep)
        times[name] = t
        tflops[name] = (
            calculate_tflops(ks, ke, shape.num_heads_q, shape.head_dim, t)
            if t is not None else None
        )

    verify = (
        _verify(impls, q, kv, q_fp8, kv_fp8, scales, weights, ks, ke, shape)
        if args.verify else {name: "N/A" for name in impls}
    )

    # Per-impl speedup vs the Triton baseline (one column per FlyDSL variant).
    base = times.get("triton")
    speedups = {}
    for name in impls:
        if name == "triton":
            continue
        t = times.get(name)
        speedups[name] = f"{base / t:.2f}x" if base and t else "-"

    return {
        "idx": idx,
        "shape": shape,
        "times": times,
        "tflops": tflops,
        "verify": verify,
        "speedups": speedups,
    }


def _verify(impls, q, kv, q_fp8, kv_fp8, scales, weights, ks, ke, shape):
    """Grade every implementation against the pure-PyTorch reference.

    ``ref_fp8_mqa_logits`` (the DeepGEMM-derived torch implementation, also used
    by the unit test) is the ground truth so Triton and FlyDSL are each
    validated independently. Returns {impl_name: "PASS"|"FAIL"|"SKIP"}.
    """
    from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
        calc_diff,
        ref_fp8_mqa_logits,
    )

    ref, _ = ref_fp8_mqa_logits(
        q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
    )
    status = {}
    for name, fn in impls.items():
        try:
            out = fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)
        except NotImplementedError:
            status[name] = "SKIP"
            continue
        m = (ref == float("-inf")) | (out == float("-inf"))
        diff = calc_diff(out.masked_fill(m, 0), ref.masked_fill(m, 0))
        status[name] = "PASS" if diff < 1e-3 else "FAIL"
    return status


def _fmt_shape(shape):
    return (
        f"bs{shape.batch_size} {shape.seq_q_l}x{shape.seq_kv_l} "
        f"H{shape.num_heads_q} D{shape.head_dim}"
    )


def _git_commit():
    """Return the current short+long git commit hash, or None if unavailable."""
    try:
        full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ) != 0
        return full + (" (dirty)" if dirty else "")
    except Exception:
        return None


def _gpu_info():
    """Return a list of (label, value) describing the GPU / runtime environment."""
    info = []
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.append(("GPU", props.name))
        arch = getattr(props, "gcnArchName", None)
        if arch:
            info.append(("Arch", arch))
        info.append(("GPU count", str(torch.cuda.device_count())))
        info.append(
            ("VRAM", f"{props.total_memory / (1024 ** 3):.1f} GiB")
        )
    else:
        info.append(("GPU", "none (CUDA/HIP not available)"))
    info.append(("torch", torch.__version__))
    info.append(("triton", triton.__version__))
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if hip:
        info.append(("HIP", hip))
    info.append(("flydsl", "available" if is_flydsl_available() else "unavailable"))
    return info


def _markdown_table(rows, impls):
    """Return the comparison table as a GitHub-flavored Markdown string.

    One ``<impl>_ms / <impl>_TFLOP/s / <impl>_vrf`` column trio per impl, plus a
    ``<impl>_vs-tri`` speedup column for each non-Triton impl when Triton is run.
    """
    names = list(impls)
    has_tri = "triton" in names
    others = [n for n in names if n != "triton"]

    headers = ["idx", "shape"]
    for n in names:
        lbl = _impl_label(n)
        headers += [f"{lbl}_ms", f"{lbl}_TFLOP/s", f"{lbl}_vrf"]
    if has_tri:
        for n in others:
            headers += [f"{_impl_label(n)}_vs-tri"]

    def cell(val, fmt):
        return "N/A" if val is None else format(val, fmt)

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out = [str(r["idx"]), _fmt_shape(r["shape"])]
        for n in names:
            out += [
                cell(r["times"].get(n), ".4f"),
                cell(r["tflops"].get(n), ".2f"),
                r["verify"].get(n, "N/A"),
            ]
        if has_tri:
            for n in others:
                out += [r["speedups"].get(n, "-")]
        lines.append("| " + " | ".join(out) + " |")
    return "\n".join(lines)


def _write_markdown(path, rows, impls, args):
    """Write the comparison table + environment/git metadata as Markdown."""
    commit = _git_commit() or "unknown"
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    parts = [
        "# FlyDSL vs Triton FP8 MQA Logits benchmark",
        "",
        f"- Generated: {when}",
        f"- Git commit: `{commit}`",
        f"- Verify: {'on' if args.verify else 'off'}"
        f" (warmup={args.warmup}, rep={args.rep})",
        "",
        "## Environment",
        "",
    ]
    for label, value in _gpu_info():
        parts.append(f"- {label}: {value}")
    parts += ["", "## Results", "", _markdown_table(rows, impls), ""]

    base_dir = os.path.dirname(path)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(path, "w") as f:
        f.write("\n".join(parts))


def _print_table(rows, impls, args):
    """Print one row per case across all shapes in a single table.

    Columns are generated per impl (``<lbl>_ms``, ``<lbl>_TFLOP/s``,
    ``<lbl>_vrf``) plus a per-non-Triton-impl ``<lbl>_vs-tri`` speedup column;
    each is widened to fit its header so variant labels don't truncate.
    """
    names = list(impls)
    has_tri = "triton" in names
    others = [n for n in names if n != "triton"]

    cols = [("idx", 5), ("shape", 26)]
    for n in names:
        lbl = _impl_label(n)
        cols += [
            (f"{lbl}_ms", max(12, len(lbl) + 4)),
            (f"{lbl}_TFLOP/s", max(13, len(lbl) + 10)),
            (f"{lbl}_vrf", max(9, len(lbl) + 5)),
        ]
    if has_tri:
        for n in others:
            lbl = _impl_label(n)
            cols += [(f"{lbl}_vs-tri", max(11, len(lbl) + 8))]

    header = "".join(f"{name:>{w}}" for name, w in cols)
    print()
    print(header)
    print("-" * len(header))

    def cell(val, fmt):
        return "N/A" if val is None else format(val, fmt)

    # Column widths in the same order as the per-row values below.
    data_widths = [w for _, w in cols[2:]]

    for r in rows:
        vals = []
        for n in names:
            vals += [
                cell(r["times"].get(n), ".4f"),
                cell(r["tflops"].get(n), ".2f"),
                r["verify"].get(n, "N/A"),
            ]
        if has_tri:
            for n in others:
                vals.append(r["speedups"].get(n, "-"))
        out = [f"{str(r['idx']):>5}", f"{_fmt_shape(r['shape']):>26}"]
        out += [f"{v:>{w}}" for v, w in zip(vals, data_widths)]
        print("".join(out))


# Names of the per-shape manual override flags (no defaults -- all or nothing).
_MANUAL_SHAPE_FLAGS = ("batch_size", "num_heads_q", "head_dim", "seq_q_l", "seq_kv_l")


def _resolve_shapes(args):
    """Resolve the shape selection into a list of (idx, (bs, s_q, s_kv, H, D)).

    ``idx`` is the 1-based case number shown in the table and accepted by
    ``--shape-index`` (converted to a 0-based offset into PRESET_SHAPES here).

    Three mutually-exclusive modes:
      * default (no shape args)      -> the full PRESET_SHAPES sweep
      * --shape-index N              -> a single preset shape by 1-based index
      * manual flags (all required)  -> a single custom shape
    """
    manual = {f: getattr(args, f) for f in _MANUAL_SHAPE_FLAGS}
    any_manual = any(v is not None for v in manual.values())

    if args.shape_index is not None and any_manual:
        raise SystemExit(
            "[error] --shape-index and manual shape flags are mutually exclusive."
        )

    if args.shape_index is not None:
        if not (1 <= args.shape_index <= len(PRESET_SHAPES)):
            raise SystemExit(
                f"[error] --shape-index must be in [1, {len(PRESET_SHAPES)}]; "
                f"got {args.shape_index}. Use --list to see the preset shapes."
            )
        zero_based = args.shape_index - 1
        return [(args.shape_index, PRESET_SHAPES[zero_based])]

    if any_manual:
        missing = [f for f, v in manual.items() if v is None]
        if missing:
            raise SystemExit(
                "[error] manual shape requires all of "
                f"{list(_MANUAL_SHAPE_FLAGS)}; missing: {missing}."
            )
        # "M" marks a manual (non-preset) shape -- not re-runnable by index.
        return [
            (
                "M",
                (
                    manual["batch_size"],
                    manual["seq_q_l"],
                    manual["seq_kv_l"],
                    manual["num_heads_q"],
                    manual["head_dim"],
                ),
            )
        ]

    return [(i + 1, s) for i, s in enumerate(PRESET_SHAPES)]


def main():
    p = argparse.ArgumentParser(
        description="FlyDSL vs Triton FP8 MQA Logits A/B benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shape selection (3 mutually-exclusive modes; see _resolve_shapes).
    p.add_argument(
        "--shape-index",
        type=int,
        default=None,
        help="run a single shape from the preset sweep by 1-based index (see --list)",
    )
    # Manual override flags -- NO defaults; supplying any requires all of them.
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_heads_q", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--seq_q_l", type=int, default=None)
    p.add_argument("--seq_kv_l", type=int, default=None)

    p.add_argument(
        "--impl",
        choices=["all", "triton", "flydsl"],
        default="all",
        help="which implementation family(ies) to run",
    )
    p.add_argument(
        "--flydsl-variants",
        type=str,
        default=None,
        metavar="V1,V2,...",
        help=(
            "comma-separated FlyDSL kernel-version tags to benchmark, each as its "
            "own column (default: the baseline variant). See --list-variants."
        ),
    )
    p.add_argument(
        "--list-variants",
        action="store_true",
        help="list the available FlyDSL kernel variants and exit",
    )
    p.add_argument("--clean_logits", type=int, default=1)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=100)
    p.add_argument(
        "--verify",
        action="store_true",
        help="check each impl against the torch reference (calc_diff < 1e-3)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="FILE",
        help="write the comparison table (Markdown) + GPU/git metadata to FILE",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list the preset shapes (with their --shape-index) and exit",
    )
    args = p.parse_args()

    if args.list:
        print("preset shapes (index: bs, s_q, s_kv, H, D):")
        for i, s in enumerate(PRESET_SHAPES):
            print(f"  {i + 1}: {s}")
        return

    avail, default = _available_variants()
    if args.list_variants:
        if avail is None:
            print("FlyDSL is unavailable; no variants to list.")
        else:
            print("FlyDSL kernel variants (default marked *):")
            for v in avail:
                print(f"  {'*' if v == default else ' '} {v}")
        return

    # Resolve the requested FlyDSL variants (comma list), defaulting to the
    # baseline. Validation against what's actually registered happens in
    # _select_impls so an unavailable-FlyDSL run still works for --impl triton.
    if args.flydsl_variants:
        args.flydsl_variants = [
            v.strip() for v in args.flydsl_variants.split(",") if v.strip()
        ]
    else:
        args.flydsl_variants = [default] if default is not None else []

    run(args)


if __name__ == "__main__":
    main()
