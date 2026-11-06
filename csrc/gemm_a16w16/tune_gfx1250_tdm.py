# SPDX-License-Identifier: MIT
"""Tune the source HGEMM config space through AITER's policy='tdm' entry.

python -m csrc.gemm_a16w16.tune_gfx1250_tdm --all-bias --output /tmp/tdm.csv
The output uses the AITER_CONFIG_GEMM_BF16 CSV schema. No default table is replaced.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

SHAPES = (
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (256, 4096, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
    (2048, 4096, 4096),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 4096, 8192),
    (8192, 8192, 8192),
    (8, 7168, 2048),
    (32, 384, 7168),
    (32, 14336, 4096),
    (16, 28672, 4096),
    (4096, 256, 4096),
)
FIELDS = (
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
    "libtype",
    "solidx",
    "splitK",
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
)


def _compile_init():
    import os

    os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["ARCH"] = "gfx1250"
    torch.set_num_threads(1)


def _compile(job):
    index, shape, config, bias = job
    try:
        import flydsl.compiler as fc
        import flydsl.expr as fx

        from aiter.ops.flydsl.kernels.gemm_a16w16_tdm_gfx1250 import build_gemm_gfx1250

        m, n, k = shape
        f = build_gemm_gfx1250(
            2,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            config["stages"],
            config["m_waves"],
            config["n_waves"],
            config["group_m"],
            bias,
            True,
            16,
            16,
            32,
            config["split_k"],
        )
        p = fc.from_c_void_p(fx.BFloat16, 0)
        fc.compile(f, p, p, p, p, p, m, n, k, k, k, n, fx.Stream(0))
        return index, None
    except Exception as exc:  # noqa: BLE001 - persist per-candidate failures
        return index, f"{type(exc).__name__}: {exc}"


def _run(a, w, out, bias, config):
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

    return flydsl_hgemm(
        a,
        w,
        out,
        bias,
        **{k: v for k, v in config.items() if k != "use_half_tile_interleaved"},
    )


def _time(fn, calls, rounds):
    # Prewarm outside capture; event samples are mean us/call within a HIP graph.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / calls)
    graph.reset()
    return samples


def tune_shape(shape, use_bias, args):
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.flydsl.gemm_a16w16_tdm_policy import get_gfx1250_tdm_configs
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm_kernel_name

    m, n, k = shape
    configs = get_gfx1250_tdm_configs(
        m, n, k, torch.bfloat16, enable_split_k=not args.no_split_k
    )
    print("SHAPE", shape, "bias", use_bias, "configs", len(configs), flush=True)
    jobs = [(i, shape, c, use_bias) for i, c in enumerate(configs)]
    with ProcessPoolExecutor(
        max_workers=args.compile_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_compile_init,
    ) as pool:
        compiled = list(pool.map(_compile, jobs))
    errors = {i: err for i, err in compiled if err}
    print("COMPILED", len(configs) - len(errors), "failed", len(errors), flush=True)
    torch.manual_seed(args.seed)
    # Quantized source-like random inputs, not zeros. Exactly representable
    # products bound reference error; winners also get a fresh normal-data check.
    a = (torch.randint(-4, 5, (m, k), device="cuda").float() / 16).bfloat16()
    w = (torch.randint(-4, 5, (n, k), device="cuda").float() / 16).bfloat16()
    bias = (torch.rand(n, device="cuda") + 0.5).bfloat16() if use_bias else None
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    # Source rounds each split partial, then sum/output, then bias.
    refs = {}
    for sk in {c["split_k"] for c in configs}:
        part = k // sk
        partial = [
            (
                a[:, i * part : (i + 1) * part].float()
                @ w[:, i * part : (i + 1) * part].float().t()
            )
            .bfloat16()
            .float()
            for i in range(sk)
        ]
        ref = sum(partial).bfloat16()
        refs[sk] = ref if bias is None else (ref.float() + bias.float()).bfloat16()
    torch.cuda.synchronize()
    candidates, records = [], []
    for i, cfg in enumerate(configs):
        record = {"index": i, "config": cfg}
        if i in errors:
            record["compile_error"] = errors[i]
            records.append(record)
            continue
        try:
            # Fail on a mismatched precompile/runtime identity instead of
            # silently recompiling thousands of candidates in this process.
            previous_run_only = os.environ.get("FLYDSL_RUNTIME_RUN_ONLY")
            os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
            try:
                _run(a, w, out, bias, cfg)
            finally:
                if previous_run_only is None:
                    os.environ.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
                else:
                    os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = previous_run_only
            torch.cuda.synchronize()
            torch.testing.assert_close(out, refs[cfg["split_k"]], atol=0, rtol=0)
            samples = _time(lambda cfg=cfg: _run(a, w, out, bias, cfg), 8, 3)
            med = statistics.median(samples)
            record.update(us=med, samples_us=samples, correct=True)
            candidates.append((med, i))
        except Exception as exc:
            if "no usable AOT cache" in str(exc):
                raise RuntimeError(
                    "precompile cache mismatch; do not edit kernel/dependency "
                    "sources while tuning; restart with --resume"
                ) from exc
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if i % 100 == 0:
            print("PROGRESS", shape, use_bias, i, len(configs), flush=True)
    if not candidates:
        raise RuntimeError(f"no correct configs for {shape}, bias={use_bias}")
    finalists = sorted(candidates)[: min(8, len(candidates))]
    # Retime all finalists, not only the fastest observation.
    final = []
    for _, i in finalists:
        cfg = configs[i]
        samples = _time(lambda cfg=cfg: _run(a, w, out, bias, cfg), 20, 9)
        final.append((statistics.median(samples), i, samples))
    med, index, samples = min(final)
    cfg = configs[index]
    torch.manual_seed(args.seed + 1)
    aa = (torch.randn((m, k), device="cuda") / 8).bfloat16()
    ww = (torch.randn((n, k), device="cuda") / 8).bfloat16()
    part = k // cfg["split_k"]
    ref = sum(
        (
            aa[:, i * part : (i + 1) * part].float()
            @ ww[:, i * part : (i + 1) * part].float().t()
        )
        .bfloat16()
        .float()
        for i in range(cfg["split_k"])
    ).bfloat16()
    if bias is not None:
        ref = (ref.float() + bias.float()).bfloat16()
    _run(aa, ww, out, bias, cfg)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)
    name = flydsl_hgemm_kernel_name(
        dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        config=cfg,
        has_bias=use_bias,
        target_gfx=get_gfx(),
    )
    row = dict(
        zip(
            FIELDS,
            (
                get_gfx(),
                torch.cuda.get_device_properties(0).multi_processor_count,
                m,
                n,
                k,
                use_bias,
                str(torch.bfloat16),
                str(torch.bfloat16),
                False,
                False,
                "flydsl",
                index,
                cfg["split_k"],
                med,
                name,
                0.0,
                2 * m * n * k / med / 1e6,
                ((m * k + n * k + m * n) * 2) / med / 1e3,
            ),
        )
    )
    evidence = {
        "shape": shape,
        "bias": use_bias,
        "seed": args.seed,
        "candidates": records,
        "finalists": final,
        "selected": row,
        "normal_validation": "passed; atol=rtol=0.05",
    }
    args.details.mkdir(parents=True, exist_ok=True)
    (args.details / f"{m}_{n}_{k}_bias{int(use_bias)}.json").write_text(
        json.dumps(evidence, indent=2)
    )
    print("WINNER", json.dumps(row), flush=True)
    return row


def _resume_keys(args):
    """Reuse only complete, matching rows backed by per-candidate evidence."""
    if not args.output.exists():
        return set()
    if not args.resume:
        raise ValueError("output exists; use --resume or choose a new path")
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.flydsl.gemm_a16w16_tdm_policy import get_gfx1250_tdm_configs

    keys = set()
    with args.output.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("existing output has a different CSV schema")
        for row in reader:
            m, n, k = (int(row[name]) for name in ("M", "N", "K"))
            if row["bias"] not in ("True", "False"):
                raise ValueError("invalid bias field in existing output")
            bias = row["bias"] == "True"
            key = (m, n, k, bias)
            if key in keys:
                raise ValueError(f"duplicate completed row: {key}")
            evidence = json.loads(
                (args.details / f"{m}_{n}_{k}_bias{int(bias)}.json").read_text()
            )
            expected = get_gfx1250_tdm_configs(
                m, n, k, torch.bfloat16, enable_split_k=not args.no_split_k
            )
            if (
                row["gfx"] != get_gfx()
                or int(row["cu_num"])
                != torch.cuda.get_device_properties(0).multi_processor_count
                or evidence["seed"] != args.seed
                or evidence["shape"] != [m, n, k]
                or evidence["bias"] != bias
                or [c["config"] for c in evidence["candidates"]] != expected
                or any(str(evidence["selected"][name]) != row[name] for name in FIELDS)
            ):
                raise ValueError(f"resume evidence/settings mismatch: {key}")
            keys.add(key)
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, default=Path("/tmp/aiter_tdm_tuning"))
    parser.add_argument("--compile-workers", type=int, default=16)
    parser.add_argument("--all-bias", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-split-k", action="store_true")
    parser.add_argument("--seed", type=int, default=1106)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shape", nargs=3, type=int)
    args = parser.parse_args()
    if args.compile_workers < 1 or not 0 <= args.shard < args.shards:
        parser.error("invalid worker/shard settings")
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ("gfx1250", "gfx1250-strict"):
        parser.error("requires gfx1250")
    torch.set_num_threads(1)
    shapes = [tuple(args.shape)] if args.shape else SHAPES[args.shard :: args.shards]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = _resume_keys(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    existed = args.output.exists()
    with args.output.open("a" if existed else "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not existed:
            writer.writeheader()
        for shape in shapes:
            for use_bias in ([False, True] if args.all_bias else [False]):
                if (*shape, use_bias) in completed:
                    print("RESUME_SKIP", shape, use_bias, flush=True)
                    continue
                writer.writerow(tune_shape(shape, use_bias, args))
                f.flush()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
