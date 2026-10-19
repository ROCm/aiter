"""
Benchmark for the fused MLA BMM + RoPE + concat + KV-cache write kernels
(fused_fp4_bmm_rope_cat_and_cache_mla and fused_fp8_bmm_rope_cat_and_cache_mla).
"""

import torch
import triton

from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
    fused_fp4_bmm_rope_cat_and_cache_mla,
    fused_fp8_bmm_rope_cat_and_cache_mla,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import str_to_torch_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    positive_int,
    pow2_int,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    generate_batched_gemm_a16w8_inputs,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a16wfp4 import (
    generate_batched_gemm_a16wfp4_inputs,
)
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs


def get_benchmark_shapes(args):
    """Return [(T, QH, KH), ...] for the current CLI args."""
    return [(t, args.qh_per_kh * kh, kh) for kh in args.kh for t in args.T]


def bench_fused_bmm_rope_kv_cache_fn(T, QH, KH, metric, args):
    dtype = torch.bfloat16
    device = "cuda"
    # Default BMM shape (N x K), from the unit tests: fp4 512 x 128, fp8 128 x 512.
    test_n, test_k = (512, 128) if args.bmm == "fp4" else (128, 512)
    kv_lora_rank = args.kv_lora_rank or test_n
    nope_dim = args.qk_nope_dim or test_k
    rope_dim = args.rope_dim
    cache_dtype = str_to_torch_dtype[args.cache_dtype]
    torch.manual_seed(0)

    # Same input construction as the unit test, per BMM dtype.
    if args.bmm == "fp4":
        _, w_k, _, w_k_scale, _ = generate_batched_gemm_a16wfp4_inputs(
            QH, T, kv_lora_rank, nope_dim, dtype, layout="TN", output=False
        )
        q = torch.randn((T, QH, nope_dim + rope_dim), dtype=dtype, device=device)
        q_nope, q_pe = q.split((nope_dim, rope_dim), dim=-1)
        q_nope = q_nope.transpose(0, 1)
    else:
        q_nope, w_k, w_k_scale, _, _ = generate_batched_gemm_a16w8_inputs(
            QH,
            T,
            kv_lora_rank,
            nope_dim,
            dtype,
            has_bias=False,
            output=False,
            layout="TN",
            transpose_bm=True,
        )
        q_pe = torch.randn((T, QH, rope_dim), dtype=dtype, device=device)

    _, _, _, _, _, positions, _, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH // KH,
        rope_dim,
        cached=True,
        reuse_freqs_front_part=args.reuse_freqs_front_part,
        nope=False,
        pos=True,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    k_lora = torch.randn((T, KH, kv_lora_rank), dtype=dtype, device=device)
    k_pe = torch.randn((T, KH, rope_dim), dtype=dtype, device=device)
    kv_cache = torch.zeros(
        (max(16384, T), KH, kv_lora_rank + rope_dim), dtype=cache_dtype, device=device
    )
    k_scale = torch.ones((), dtype=torch.float32, device=device)
    slot_mapping = torch.randperm(T, device=device)

    if args.bmm == "fp4":

        def fn():
            return fused_fp4_bmm_rope_cat_and_cache_mla(
                q_nope,
                w_k,
                w_k_scale,
                q_pe,
                k_lora,
                k_pe,
                kv_cache,
                slot_mapping,
                positions,
                cos,
                sin,
                k_scale=k_scale,
                is_neox=args.neox,
                num_decode_toks_for_zeros=T,
            )

    else:

        def fn():
            return fused_fp8_bmm_rope_cat_and_cache_mla(
                q_nope,
                w_k,
                w_k_scale,
                q_pe,
                k_lora,
                k_pe,
                kv_cache,
                slot_mapping,
                positions,
                cos,
                sin,
                group_size=128,
                k_scale=k_scale,
                is_neox=args.neox,
                q_out_dtype=dtype,
                num_decode_toks_for_zeros=T,
            )

    # do_bench_cudagraph runs fn on a new stream that does not wait for the inputs above,
    # so without this the kernel can index with a slot_mapping/positions not yet written.
    torch.cuda.synchronize()
    ms = triton.testing.do_bench_cudagraph(fn, rep=args.rep)
    if metric == "time":
        return ms
    if metric == "throughput":
        flops = 2 * QH * T * kv_lora_rank * nope_dim
        return flops / (ms * 1e-3) * 1e-12  # TFLOPS

    # Bytes: weights, q_nope/q_pe, k_lora/k_pe reads; q_out, decode_q_pe_out, zeros,
    # k_pe_out, kv_cache writes (positions, cos/sin, slot_mapping are negligible).
    elem = 2
    if args.bmm == "fp4":
        w_bytes = QH * kv_lora_rank * (nope_dim // 2 + nope_dim // 32)
    else:
        w_bytes = QH * kv_lora_rank * nope_dim
    q_bytes = T * QH * (nope_dim + rope_dim) * elem
    q_out_bytes = T * QH * (kv_lora_rank + rope_dim + rope_dim + kv_lora_rank) * elem
    k_bytes = T * KH * (kv_lora_rank + 2 * rope_dim) * elem
    cache_bytes = T * KH * (kv_lora_rank + rope_dim) * kv_cache.element_size()
    mem = w_bytes + q_bytes + q_out_bytes + k_bytes + cache_bytes
    return mem / (ms * 1e-3) * 1e-9  # GB/s


def run_benchmark(args):
    units = {"time": "ms", "throughput": "TFLOPS", "bandwidth": "GB/s"}
    benchmark = triton.testing.Benchmark(
        x_names=["T", "QH", "KH"],
        x_vals=get_benchmark_shapes(args),
        line_arg="metric",
        line_vals=[args.metric],
        line_names=[args.metric],
        ylabel=units[args.metric],
        plot_name=f"{get_caller_name_no_ext()}_{args.bmm}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(T, QH, KH, metric):
        return bench_fused_bmm_rope_kv_cache_fn(T, QH, KH, metric, args)

    df = bench_fn.run(print_data=True, return_df=True)[0]
    if args.o:
        df.to_csv(f"{benchmark.plot_name}.csv", index=False)


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="fused_bmm_rope_kv_cache")
    parser.set_defaults(metric="time")
    parser.add_argument(
        "-T",
        type=positive_int,
        nargs="+",
        default=[1, 2, 32, 2048],
        help="Number of tokens",
    )
    parser.add_argument(
        "--kh",
        type=positive_int,
        nargs="+",
        default=[1, 8],
        help="Number of KV heads",
    )
    parser.add_argument(
        "--qh-per-kh",
        type=positive_int,
        default=16,
        help="Query heads per KV head",
    )
    parser.add_argument(
        "--kv-lora-rank",
        type=pow2_int,
        default=None,
        help="Latent dim the BMM projects q_nope to, BMM N; unset: 512 fp4, 128 fp8",
    )
    parser.add_argument(
        "--qk-nope-dim",
        type=positive_int,
        default=None,
        help="q_nope head dim, BMM K; unset: 128 fp4, 512 fp8",
    )
    parser.add_argument(
        "--rope-dim",
        type=pow2_int,
        default=128,
        help="RoPE head dim",
    )
    parser.add_argument(
        "--bmm",
        type=str,
        default="fp4",
        choices=["fp4", "fp8"],
        help="BMM weight dtype: MXFP4 (fused_fp4_...) or FP8 (fused_fp8_...)",
    )
    parser.add_argument(
        "--cache-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8e4m3"],
        help="KV cache dtype",
    )
    parser.add_argument(
        "--neox", action="store_true", default=False, help="NeoX-style RoPE"
    )
    parser.add_argument(
        "--reuse-freqs-front-part",
        action="store_true",
        default=False,
        help="cos/sin hold rope_dim // 2 frequencies",
    )
    parser.add_argument(
        "--rep",
        type=positive_int,
        default=20,
        help="do_bench_cudagraph measurement budget in ms",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        default=False,
        help="Write performance results to CSV file",
    )
    parsed_args = parser.parse_args(args=args)
    for name in ("rope_dim", "kv_lora_rank"):
        if (getattr(parsed_args, name) or 2) < 2:
            parser.error(f"--{name.replace('_', '-')} must be >= 2")
    nope_dim = parsed_args.qk_nope_dim
    # The fused fp4 kernel masks its K tail with the packed width, so a q_nope dim
    # that isn't a power of 2 >= 64 can give a wrong BMM (or fail to compile),
    # depending on the config's BLOCK_SIZE_K.
    if (
        parsed_args.bmm == "fp4"
        and nope_dim is not None
        and (nope_dim < 64 or nope_dim & (nope_dim - 1))
    ):
        parser.error("--qk-nope-dim must be a power of 2 >= 64 with --bmm fp4")
    return parsed_args


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)
    if parsed_args.bmm == "fp4" and not arch_info.is_fp4_avail():
        print("MXFP4 is not available on this architecture")
        return
    if parsed_args.print_vgpr:
        print("Retrieving VGPR usage for fused_bmm_rope_kv_cache kernels...")
        print_vgpr(lambda: run_benchmark(parsed_args), get_caller_name_no_ext())
        return
    run_benchmark(parsed_args)


if __name__ == "__main__":
    main()
