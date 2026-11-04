"""
Benchmark the fused MLA decode query prep (fused_bf16_bmm_cat_quant_pad_mla)
against the unfused path vLLM runs ahead of the padded asm MLA decode:
torch.bmm (W_UK absorb), the compiled cat + static fp8 quant, and
repeat_interleave.

With --rope-cache, every provider also does the RoPE and the fp8 KV-cache
write. unfused then runs torch.bmm, AITER's fused_qk_rope_concat_and_cache_mla
(RoPE + cat + fp8 quant + cache) and repeat_interleave. fused_rope_cache runs
fused_bf16_bmm_rope_cat_quant_pad_cache_mla. fused is not available in this
mode: AITER has no kernel that does only the RoPE and the cache write.

Inputs rotate through a pool so each call reads its W_UK^T from HBM, as each
decode layer does with its own weight.
"""

import json
from collections import Counter

import torch
import triton

from aiter.ops.cache import fused_qk_rope_concat_and_cache_mla
from aiter.ops.triton.fusions.fused_bf16_bmm_cat_quant_pad_mla import (
    _get_config,
    _get_rope_cache_config,
    fused_bf16_bmm_cat_quant_pad_mla,
    fused_bf16_bmm_rope_cat_quant_pad_cache_mla,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    positive_int,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
    run_input_generator,
)
from op_tests.triton_tests.fusions.test_fused_bf16_bmm_cat_quant_pad_mla import (
    D_NOPE,
    KV_LORA,
    cat_quant_fp8,
    compiled_cat_quant_fp8,
    generate_inputs,
    generate_rope_cache_inputs,
    vllm_fp8_max,
)

PROVIDERS = ["unfused", "fused"]
ROPE_CACHE_PROVIDERS = ["unfused", "fused_rope_cache"]
FIELDS = ("q_nope", "w_uk_t", "q_pe", "q_scale", "buf")
ROPE_CACHE_FIELDS = FIELDS + (
    "kv_c",
    "k_pe",
    "cos_sin_cache",
    "positions",
    "slot_mapping",
    "kv_cache",
    "k_scale",
    "q_out",
)


def fields(args):
    return ROPE_CACHE_FIELDS if args.rope_cache else FIELDS


def make_input_gen(M, provider, args):
    def gen():
        q_nope, w_uk_t, q_pe = generate_inputs(M, args.qh)
        q_scale = torch.ones((), dtype=torch.float32, device="cuda")
        fp8 = get_fp8_e4m3_dtype()
        q_dim = w_uk_t.shape[-1] + q_pe.shape[-1]
        if provider == "unfused":
            buf = q_nope.new_empty((M, args.qh, w_uk_t.shape[-1]))
        else:
            buf = torch.empty((M, args.padded_heads, q_dim), dtype=fp8, device="cuda")
        if not args.rope_cache:
            return q_nope, w_uk_t, q_pe, q_scale, buf
        rope_cache = generate_rope_cache_inputs(M)
        k_scale = torch.ones((), dtype=torch.float32, device="cuda")
        # The unpadded fp8 query fused_qk_rope_concat_and_cache_mla writes.
        q_out_shape = (M, args.qh, q_dim) if provider == "unfused" else (0,)
        q_out = torch.empty(q_out_shape, dtype=fp8, device="cuda")
        return q_nope, w_uk_t, q_pe, q_scale, buf, *rope_cache, k_scale, q_out

    return gen


def make_fn(M, provider, args, cat_quant):
    fp8_max = vllm_fp8_max()
    dup = args.padded_heads // args.qh
    config = None
    if args.config:
        get_config = (
            _get_rope_cache_config if provider == "fused_rope_cache" else _get_config
        )
        config, _ = get_config(M, KV_LORA, D_NOPE, args.qh)
        config.update(args.config)

    cache_dtype = get_fp8_e4m3_dtype()
    if provider == "unfused" and args.rope_cache:

        def fn(
            q_nope,
            w_uk_t,
            q_pe,
            q_scale,
            buf,
            kv_c,
            k_pe,
            cos_sin_cache,
            positions,
            slot_mapping,
            kv_cache,
            k_scale,
            q_out,
        ):
            torch.bmm(q_nope, w_uk_t, out=buf.transpose(0, 1))
            cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
            fused_qk_rope_concat_and_cache_mla(
                buf,
                q_pe,
                kv_c,
                k_pe,
                kv_cache.view(cache_dtype),
                q_out,
                slot_mapping,
                k_scale,
                q_scale,
                positions,
                cos_cache,
                sin_cache,
                args.is_neox,
                True,
            )
            return q_out.repeat_interleave(dup, dim=1)

    elif provider == "unfused":

        def fn(q_nope, w_uk_t, q_pe, q_scale, buf):
            torch.bmm(q_nope, w_uk_t, out=buf.transpose(0, 1))
            q = cat_quant(buf, q_pe, q_scale, fp8_max)
            return q.repeat_interleave(dup, dim=1)

    elif provider == "fused":

        def fn(q_nope, w_uk_t, q_pe, q_scale, buf):
            return fused_bf16_bmm_cat_quant_pad_mla(
                q_nope,
                w_uk_t,
                q_pe,
                q_scale,
                args.padded_heads,
                out=buf,
                fp8_max=fp8_max,
                config=config,
            )

    else:

        def fn(
            q_nope,
            w_uk_t,
            q_pe,
            q_scale,
            buf,
            kv_c,
            k_pe,
            cos_sin_cache,
            positions,
            slot_mapping,
            kv_cache,
            k_scale,
            q_out,
        ):
            return fused_bf16_bmm_rope_cat_quant_pad_cache_mla(
                q_nope,
                w_uk_t,
                q_pe,
                q_scale,
                kv_c,
                k_pe,
                kv_cache.view(cache_dtype),
                slot_mapping,
                k_scale,
                positions,
                cos_sin_cache,
                args.is_neox,
                args.padded_heads,
                out=buf,
                fp8_max=fp8_max,
                config=config,
            )

    return fn


def print_kernels(M, args, cat_quant, calls=4):
    """Kernels each provider launches over a few calls, to compare with the vLLM
    trace names. Every kernel should show up once per call."""
    for provider in args.providers:
        inputs = run_input_generator(make_input_gen(M, provider, args), fields(args))
        fn = make_fn(M, provider, args, cat_quant)
        fn(**inputs)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(calls):
                fn(**inputs)
            torch.cuda.synchronize()
        names = Counter(
            evt.name
            for evt in prof.events()
            if evt.device_type == torch.autograd.DeviceType.CUDA
        )
        print(f"{provider} (M={M}, {calls} calls):")
        if not names:
            print("    no GPU kernels recorded")
        for name, count in names.items():
            print(f"    {count:3d}x {name[:160]}")


def bench_fn(M, provider, args, cat_quant):
    torch.manual_seed(0)
    rot_bufs = generate_rotating_buffers_pool(
        [make_input_gen(M, provider, args)],
        [fields(args)],
        **get_rot_bufs_kwargs(args),
    )
    ms = do_bench_aiter_triton(
        make_fn(M, provider, args, cat_quant),
        rot_bufs,
        **get_do_bench_graph_kwargs(args),
    )
    return ms * 1e3


def run_benchmark(args):
    cat_quant = compiled_cat_quant_fp8 if args.compile else cat_quant_fp8
    if args.print_kernels:
        for M in args.M:
            print_kernels(M, args, cat_quant)
        return

    benchmark = triton.testing.Benchmark(
        x_names=["M"],
        x_vals=args.M,
        line_arg="provider",
        line_vals=args.providers,
        line_names=args.providers,
        ylabel="us",
        plot_name=(
            f"{get_caller_name_no_ext()}_qh{args.qh}_pad{args.padded_heads}"
            + ("_rope_cache" if args.rope_cache else "")
        ),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(M, provider):
        return bench_fn(M, provider, args, cat_quant)

    df = bench.run(print_data=True, return_df=True)[0]
    if args.o:
        df.to_csv(f"{benchmark.plot_name}.csv", index=False)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_bf16_bmm_cat_quant_pad_mla")
    parser.add_argument(
        "-M",
        type=positive_int,
        nargs="+",
        default=[1, 16, 32, 64, 128, 256],
        help="Decode tokens",
    )
    parser.add_argument(
        "--rope-cache",
        action="store_true",
        default=False,
        help="Also do the RoPE and the fp8 KV-cache write in every provider",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=sorted(set(PROVIDERS + ROPE_CACHE_PROVIDERS)),
        default=None,
        help=f"Providers to time or list (default {PROVIDERS}, "
        f"{ROPE_CACHE_PROVIDERS} with --rope-cache)",
    )
    parser.add_argument("--qh", type=positive_int, default=8, help="Heads per rank")
    parser.add_argument(
        "--padded-heads",
        type=positive_int,
        default=16,
        help="Head count of the padded MLA query",
    )
    parser.add_argument(
        "--is-neox",
        action="store_true",
        default=False,
        help="NeoX RoPE pairs with --rope-cache (default GPT-J, as Kimi-K2)",
    )
    parser.add_argument(
        "--config",
        type=json.loads,
        default={},
        help="JSON overrides of the tuned config, for example '{\"BLOCK_SIZE_N\": 64}'",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="Run the unfused cat + quant eagerly instead of through torch.compile",
    )
    parser.add_argument(
        "--print-kernels",
        action="store_true",
        default=False,
        help="Print the kernels each provider launches instead of timing",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )
    parsed = parser.parse_args(args=args)
    if parsed.padded_heads % parsed.qh:
        parser.error("--padded-heads must be a multiple of --qh")
    allowed = ROPE_CACHE_PROVIDERS if parsed.rope_cache else PROVIDERS
    if parsed.providers is None:
        parsed.providers = allowed
    elif not set(parsed.providers) <= set(allowed):
        parser.error(
            f"providers {allowed} only {'with' if parsed.rope_cache else 'without'} "
            "--rope-cache"
        )
    return parsed


def main(args: list[str] | None = None):
    run_benchmark(parse_args(args))


if __name__ == "__main__":
    main()
