"""Sweep unified-attention config overrides to justify a table change.

Each (shape, config) point is measured in its own subprocess. Measuring several
Triton configs sequentially inside one process penalizes whichever config runs
later by a few percent, which is enough to invert the sign of a small win, and
it lets two configs that hash to the same kernel name (``waves_per_eu``
variants do) misreport each other's register and LDS usage.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton

import aiter.ops.triton.attention.unified_attention as unified_attention_mod
from op_tests.op_benchmarks.triton.bench_unified_attention import make_inputs
from op_tests.op_benchmarks.triton.utils.argparse import get_parser

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RECORD_PREFIX = "BENCH_RECORD "

PRESETS = {
    "gemma4-full": {
        "hq": 32,
        "hk": 4,
        "d": 512,
        "block_size": 64,
        "shapes": [
            (1, 16384, 32768, "fp8"),
            (1, 16384, 16384, "fp8"),
            (2, 8192, 16384, "fp8"),
            (4, 4096, 8192, "fp8"),
            (1, 8192, 8192, "fp8"),
            (1, 2048, 4096, "fp8"),
            (1, 16384, 32768, "bf16"),
            (2, 8192, 16384, "bf16"),
        ],
    },
}

_base_get_config = unified_attention_mod.get_unified_attention_config
_override: dict = {}


def _patched_get_config(op, params, backend="triton", arch=None):
    config = _base_get_config(op, params, backend=backend, arch=arch)
    if op == "attn_2d" and backend == "triton" and _override:
        config.update(_override)
    return config


unified_attention_mod.get_unified_attention_config = _patched_get_config


def kernel_resources():
    """Report register, spill and LDS usage of every compiled attn_2d kernel."""
    from aiter.ops.triton._triton_kernels.attention.unified_attention import (
        kernel_unified_attention_2d,
    )

    found = []
    seen = set()

    def visit(obj, depth=0):
        if depth > 4 or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, "n_regs") and hasattr(obj, "metadata"):
            meta = obj.metadata
            found.append(
                {
                    "name": getattr(meta, "name", None),
                    "n_regs": getattr(obj, "n_regs", None),
                    "n_spills": getattr(obj, "n_spills", None),
                    "lds_bytes": getattr(meta, "shared", None),
                    "num_warps": getattr(meta, "num_warps", None),
                    "num_stages": getattr(meta, "num_stages", None),
                }
            )
            return
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value, depth + 1)
        elif hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                visit(value, depth + 1)

    try:
        visit(kernel_unified_attention_2d.device_caches)
    except AttributeError:
        return []
    return found


def build_callable(args):
    """Build the closure to time, plus the output buffer it writes into."""
    torch.manual_seed(20)
    fp8 = args.dtype == "fp8"
    max_blocks_per_seq = (args.sk + args.block_size - 1) // args.block_size
    num_blocks = max(args.b * max_blocks_per_seq * 2, 2048)
    inputs = make_inputs(
        seq_lens=[(args.sq, args.sk)] * args.b,
        num_heads=(args.hq, args.hk),
        head_size_qk=args.d,
        head_size_v=args.d,
        block_size=args.block_size,
        num_blocks=num_blocks,
        fp8_q=fp8,
        fp8_kv=fp8,
        fp8_output=False,
        out_scale_value=1.0,
    )

    def fn():
        return unified_attention_mod.unified_attention(
            q=inputs["q_fp8"] if fp8 else inputs["query"],
            k=inputs["k_fp8"] if fp8 else inputs["key_cache"],
            v=inputs["v_fp8"] if fp8 else inputs["value_cache"],
            out=inputs["output"],
            cu_seqlens_q=inputs["cu_query_lens"],
            seqused_k=inputs["kv_lens"],
            max_seqlen_q=inputs["max_query_len"],
            max_seqlen_k=inputs["max_kv_len"],
            softmax_scale=inputs["scale"],
            causal=True,
            window_size=(-1, -1),
            block_table=inputs["block_tables"],
            softcap=0,
            q_descale=inputs["q_descale"],
            k_descale=inputs["k_descale"],
            v_descale=inputs["v_descale"],
            output_scale=None,
            backend="triton",
        )

    return fn, inputs


def attention_tflops(args, ms):
    causal_positions = args.sq * args.sk - (args.sq**2 - args.sq) / 2
    flops = args.b * causal_positions * args.hq * 2 * args.d * 2
    return flops / ms * 1e-9


def run_worker(args):
    """Measure exactly one (shape, config) point and print it as JSON."""
    global _override
    _override = json.loads(args.override)

    fn, inputs = build_callable(args)
    fn()
    torch.cuda.synchronize()

    if args.cudagraph:
        ms = triton.testing.do_bench_cudagraph(fn)
    else:
        ms = triton.testing.do_bench(
            fn, warmup=args.warmup, rep=args.rep, return_mode="median"
        )

    record = {
        "tag": args.tag,
        "override": _override,
        "shape": shape_key(args),
        "ms": round(ms, 4),
        "tflops": round(attention_tflops(args, ms), 1),
        "kernels": kernel_resources(),
    }

    if args.check:
        measured = inputs["output"].detach().clone().to(torch.float32)
        _override = {}
        inputs["output"].zero_()
        fn()
        torch.cuda.synchronize()
        reference = inputs["output"].to(torch.float32)
        diff = (measured - reference).abs()
        ref_absmax = float(reference.abs().max())
        record["check"] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "max_abs_over_ref_absmax": float(diff.max()) / max(ref_absmax, 1e-9),
            "ref_absmax": ref_absmax,
            "allclose_2e-2": bool(torch.allclose(measured, reference, atol=2e-2)),
        }

    print(_RECORD_PREFIX + json.dumps(record), flush=True)


def shape_key(args):
    return f"{args.dtype} b{args.b} sq{args.sq} sk{args.sk}"


def launch_worker(args, shape, override, tag):
    b, sq, sk, dtype = shape
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "-b",
        str(b),
        "-sq",
        str(sq),
        "-sk",
        str(sk),
        "-hq",
        str(args.hq),
        "-hk",
        str(args.hk),
        "-d",
        str(args.d),
        "-block_size",
        str(args.block_size),
        "--dtype",
        dtype,
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--override",
        json.dumps(override),
        "--tag",
        tag,
    ]
    if args.cudagraph:
        cmd.append("--cudagraph")
    if args.check:
        cmd.append("--check")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)

    for line in proc.stdout.splitlines():
        if line.startswith(_RECORD_PREFIX):
            return json.loads(line[len(_RECORD_PREFIX) :])
    return {
        "tag": tag,
        "override": override,
        "shape": f"{dtype} b{b} sq{sq} sk{sk}",
        "ms": None,
        "error": (proc.stderr.strip().splitlines() or ["no record emitted"])[-1][:300],
    }


def run_in_process(args, shape, override, tag):
    point = argparse.Namespace(**vars(args))
    point.b, point.sq, point.sk, point.dtype = shape
    point.override = json.dumps(override)
    point.tag = tag

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        run_worker(point)
    line = next(
        ln for ln in buffer.getvalue().splitlines() if ln.startswith(_RECORD_PREFIX)
    )
    return json.loads(line[len(_RECORD_PREFIX) :])


def format_table(variants, rows):
    baseline = variants[0][0]
    header = f"{'shape':<26}" + "".join(f"{name:>14}" for name, _ in variants)
    header += "".join(f"{name + ' x':>14}" for name, _ in variants[1:])
    lines = [header, "-" * len(header)]
    for shape_name, by_tag in rows:
        cells = ""
        for name, _ in variants:
            ms = by_tag.get(name, {}).get("ms")
            cells += f"{ms:>14.4f}" if ms is not None else f"{'err':>14}"
        for name, _ in variants[1:]:
            base_ms = by_tag.get(baseline, {}).get("ms")
            ms = by_tag.get(name, {}).get("ms")
            if base_ms and ms:
                cells += f"{base_ms / ms:>14.4f}"
            else:
                cells += f"{'-':>14}"
        lines.append(f"{shape_name:<26}" + cells)
    return "\n".join(lines)


def parse_variant(spec, index):
    """Split a ``NAME=JSON`` config spec, naming it positionally if unnamed."""
    if "=" in spec and not spec.lstrip().startswith("{"):
        name, body = spec.split("=", 1)
    else:
        name, body = f"cfg{index}", spec
    return name.strip(), json.loads(body)


def run_sweep(args):
    preset = PRESETS[args.preset] if args.preset else None
    if preset:
        for key in ("hq", "hk", "d", "block_size"):
            if not getattr(args, key):
                setattr(args, key, preset[key])
        shapes = preset["shapes"]
    else:
        if not (args.b and args.sq and args.hq and args.hk and args.d):
            raise ValueError(
                "Without --preset, pass -b -sq -hq -hk -d (and optionally -sk)"
            )
        shapes = [(args.b, args.sq, args.sk or args.sq, args.dtype)]

    variants = [parse_variant(spec, i) for i, spec in enumerate(args.config)]

    rows = []
    records = []
    runner = run_in_process if args.in_process else launch_worker
    for shape in shapes:
        by_tag = {}
        for tag, override in variants:
            record = runner(args, shape, override, tag)
            records.append(record)
            by_tag[tag] = record
            print(_RECORD_PREFIX + json.dumps(record), flush=True)
        rows.append((by_tag[variants[0][0]]["shape"], by_tag))

    print()
    print(format_table(variants, rows))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "records": records}, f, indent=2)
        print(f"\nwrote {args.json}")


def parse_args(args=None):
    parser = get_parser(kernel_name="Unified Attention config tuning")
    parser.add_argument("-b", type=int, default=0, help="Batch size")
    parser.add_argument("-hq", type=int, default=0, help="Query heads")
    parser.add_argument("-hk", type=int, default=0, help="KV heads")
    parser.add_argument("-sq", type=int, default=0, help="Query sequence length")
    parser.add_argument("-sk", type=int, default=0, help="Key sequence length")
    parser.add_argument("-d", type=int, default=0, help="Head size")
    parser.add_argument("-block_size", type=int, default=0, help="KV page size")
    parser.add_argument(
        "--dtype", type=str, default="fp8", choices=["fp8", "bf16"], help="Q/K/V dtype"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESETS),
        help="Named shape set; supplies -hq -hk -d -block_size unless overridden",
    )
    parser.add_argument(
        "--config",
        action="append",
        metavar="NAME=JSON",
        help="A named attn_2d config override, repeatable. The first is the "
        "baseline every later one is divided by. Example: "
        "--config 'split={\"SPLIT_UNMASKED_LOOP\": true}'",
    )
    parser.add_argument("--warmup", type=int, default=200, help="do_bench warmup (ms)")
    parser.add_argument("--rep", type=int, default=800, help="do_bench rep (ms)")
    parser.add_argument(
        "--cudagraph", action="store_true", help="Time with do_bench_cudagraph"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Also compare each config's output against the untouched table",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Skip per-point subprocess isolation. Faster, but biases whichever "
        "config runs later and is not fit for reporting a small win",
    )
    parser.add_argument("--json", type=str, default=None, help="Write raw records here")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--override", type=str, default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--tag", type=str, default="", help=argparse.SUPPRESS)
    return parser.parse_args(args=args)


def main(args=None) -> None:
    args = parse_args(args=args)
    if args.worker:
        run_worker(args)
        return
    if not args.config:
        args.config = ["baseline={}"]
    run_sweep(args)


if __name__ == "__main__":
    sys.exit(main())
