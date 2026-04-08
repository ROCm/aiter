import itertools
import sys

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.triton_tests.attention.test_unified_attention import ref_paged_attn

FP8_TYPE = e4m3_dtype
FP8_MAX = torch.finfo(FP8_TYPE).max


def default_benchmark_configs():
    batch_sizes = [1, 4, 8]
    n_heads = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [8192]
    head_dim = 128
    v_head_dim = head_dim
    configs = list(itertools.product(batch_sizes, n_heads, seq_len_q, seq_len_k))
    return [(bs, nh, nh, sq, sk, head_dim, v_head_dim) for bs, nh, sq, sk in configs]


def quantize_to_fp8(tensor):
    """Per-tensor symmetric FP8 quantization. Returns (quantized, descale)."""
    abs_max = tensor.abs().amax().clamp(min=1e-9)
    descale = (abs_max / FP8_MAX).to(torch.float32).unsqueeze(0).cuda()
    quantized = (tensor * (FP8_MAX / abs_max)).to(FP8_TYPE)
    return quantized, descale


def make_inputs(
    seq_lens,
    num_heads,
    head_size_qk,
    head_size_v,
    block_size,
    num_blocks,
    fp8_q,
    fp8_kv,
    fp8_output,
    out_scale_value,
):
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)
    scale = head_size_qk**-0.5

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size_qk,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_qk,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_v,
        dtype=torch.bfloat16,
        device="cuda",
    )

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    cu_query_lens = torch.tensor(
        [0] + query_lens,
        dtype=torch.int32,
        device="cuda",
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    query_lens_t = torch.tensor(query_lens, dtype=torch.int32, device="cuda")

    q_fp8, q_descale = quantize_to_fp8(query) if fp8_q else (None, None)
    k_fp8, k_descale = quantize_to_fp8(key_cache) if fp8_kv else (None, None)
    v_fp8, v_descale = quantize_to_fp8(value_cache) if fp8_kv else (None, None)

    out_scale = None
    out_dtype = torch.bfloat16
    if fp8_output:
        out_scale = torch.tensor([out_scale_value], dtype=torch.float32, device="cuda")
        out_dtype = FP8_TYPE

    output = torch.empty(
        cu_query_lens[-1].item(),
        num_query_heads,
        head_size_v,
        dtype=out_dtype,
        device="cuda",
    )

    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        out_scale=out_scale,
        output=output,
        cu_query_lens=cu_query_lens,
        kv_lens=kv_lens,
        query_lens=query_lens_t,
        block_tables=block_tables,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        scale=scale,
    )


def _mode_label(args):
    parts = []
    if args.fp8:
        parts.append("fp8")
    elif args.fp8_kv:
        parts.append("fp8_kv")
    else:
        parts.append("bf16")
    if args.fp8_output:
        parts.append("fp8out")
    return "_".join(parts) + "_fwd"


def run_benchmark(custom, args):
    torch.manual_seed(20)

    any_fp8 = args.fp8 or args.fp8_kv or args.fp8_output
    label = _mode_label(args)

    def create_configs():
        hk = args.hq if not args.hk else args.hk
        sk = args.sq if not args.sk else args.sk
        head_size = 128 if not args.d else args.d
        head_size_v = head_size if not args.dv else args.dv
        decode_p = args.decode

        x_names = [
            "BATCH",
            "HQ",
            "HK",
            "N_CTX_Q",
            "N_CTX_K",
            "D_HEAD",
            "D_HEAD_V",
            "DECODE_P",
        ]

        if isinstance(args.sq, list):
            batch_size = len(args.sq)
        elif isinstance(args.sk, list):
            batch_size = len(args.sk)
        else:
            batch_size = args.b if args.b else 1

        if custom:
            x_vals_list = [
                (batch_size, args.hq, hk, args.sq, sk, head_size, head_size_v)
            ]
        else:
            x_vals_list = default_benchmark_configs()

        x_vals_list = [(*v, decode_p) for v in x_vals_list]

        unit = {"time": "ms", "throughput": "TFLOPS", "bandwidth": "GB/s"}[args.metric]

        return [
            triton.testing.Benchmark(
                x_names=x_names,
                x_vals=x_vals_list,
                line_arg="provider",
                line_vals=[label],
                line_names=[label],
                styles=[("red", "-")],
                ylabel=unit,
                plot_name=f"bench_unified_attention_{label}",
                args={},
            )
        ]

    @triton.testing.perf_report(create_configs())
    def bench_fn(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        DECODE_P,
        provider,
    ):
        varlen = not args.equal_seqlens

        if isinstance(N_CTX_Q, list):
            seqlens_q = torch.tensor(N_CTX_Q, dtype=torch.int32, device="cuda")
        elif varlen:
            seqlens_q = torch.randint(
                1, N_CTX_Q + 1, (BATCH,), dtype=torch.int32, device="cuda"
            )
        else:
            seqlens_q = torch.full((BATCH,), N_CTX_Q, dtype=torch.int32, device="cuda")

        if isinstance(N_CTX_K, list):
            seqlens_k = torch.tensor(N_CTX_K, dtype=torch.int32, device="cuda")
        elif varlen:
            seqlens_k = torch.randint(
                1, N_CTX_K + 1, (BATCH,), dtype=torch.int32, device="cuda"
            )
        else:
            seqlens_k = torch.full((BATCH,), N_CTX_K, dtype=torch.int32, device="cuda")

        seqlens_k = torch.maximum(seqlens_k, seqlens_q)

        if DECODE_P > 0.0:
            num_decode = int(round(DECODE_P * BATCH))
            if num_decode > 0:
                decode_idx = torch.randperm(BATCH, device=seqlens_q.device)[:num_decode]
                seqlens_q[decode_idx] = 1

        block_size = args.block_size if args.block_size else 512
        max_num_blocks_per_seq = (seqlens_k.max().item() + block_size - 1) // block_size
        min_required_blocks = BATCH * max_num_blocks_per_seq
        num_blocks = (
            args.num_blocks if args.num_blocks else max(min_required_blocks * 4, 2048)
        )

        inputs = make_inputs(
            seq_lens=list(zip(seqlens_q.tolist(), seqlens_k.tolist())),
            num_heads=(HQ, HK),
            head_size_qk=D_HEAD,
            head_size_v=D_HEAD_V,
            block_size=block_size,
            num_blocks=num_blocks,
            fp8_q=args.fp8,
            fp8_kv=args.fp8 or args.fp8_kv,
            fp8_output=args.fp8_output,
            out_scale_value=args.out_scale,
        )

        q_tensor = inputs["q_fp8"] if args.fp8 else inputs["query"]
        k_tensor = inputs["k_fp8"] if (args.fp8 or args.fp8_kv) else inputs["key_cache"]
        v_tensor = (
            inputs["v_fp8"] if (args.fp8 or args.fp8_kv) else inputs["value_cache"]
        )

        window_size = (
            (args.sliding_window - 1, 0)
            if args.sliding_window is not None
            else (-1, -1)
        )

        def fn():
            return unified_attention(
                q=q_tensor,
                k=k_tensor,
                v=v_tensor,
                out=inputs["output"],
                cu_seqlens_q=inputs["cu_query_lens"],
                seqused_k=inputs["kv_lens"],
                max_seqlen_q=inputs["max_query_len"],
                max_seqlen_k=inputs["max_kv_len"],
                softmax_scale=inputs["scale"],
                causal=True,
                window_size=window_size,
                block_table=inputs["block_tables"],
                softcap=0,
                q_descale=inputs["q_descale"],
                k_descale=inputs["k_descale"],
                v_descale=inputs["v_descale"],
                output_scale=inputs["out_scale"],
            )

        ms = triton.testing.do_bench_cudagraph(fn)

        if args.test:
            fn()
            ref_output = ref_paged_attn(
                query=inputs["query"],
                key_cache=inputs["key_cache"],
                value_cache=inputs["value_cache"],
                query_lens=inputs["query_lens"],
                kv_lens=inputs["kv_lens"],
                block_tables=inputs["block_tables"],
                scale=inputs["scale"],
                sliding_window=args.sliding_window,
                soft_cap=None,
                q_descale=inputs["q_descale"],
                k_descale=inputs["k_descale"],
                v_descale=inputs["v_descale"],
                output_scale=inputs["out_scale"],
                out_dtype=inputs["output"].dtype,
            )
            if any_fp8:
                atol, rtol = 1.5e-1, 1.5e-1
            else:
                atol, rtol = 1.5e-2, 1e-2
            out_f32 = inputs["output"].to(torch.float32)
            ref_f32 = ref_output.to(torch.float32)
            max_diff = torch.max(torch.abs(out_f32 - ref_f32)).item()
            shape_str = f"(B={BATCH}, HQ={HQ}, HK={HK}, SQ={N_CTX_Q}, SK={N_CTX_K}, D={D_HEAD}, DV={D_HEAD_V})"
            try:
                torch.testing.assert_close(out_f32, ref_f32, atol=atol, rtol=rtol)
                print(f"  PASS {shape_str}  max_diff={max_diff:.6f}")
            except AssertionError as e:
                print(f"  FAIL {shape_str}  max_diff={max_diff:.6f}")
                print(f"    {e}")

        cu_query_lens = inputs["cu_query_lens"]
        num_contexts = len(cu_query_lens) - 1
        total_flops = 0.0
        for i in range(num_contexts):
            sq = (cu_query_lens[i + 1] - cu_query_lens[i]).item()
            sk = seqlens_k[i].item()
            valid = sq * sk - ((sq**2 - sq) / 2)
            total_flops += valid * HQ * (D_HEAD + D_HEAD_V) * 2.0

        total_q = cu_query_lens[-1].item()
        total_k = seqlens_k.sum().item()
        q_bytes = total_q * HQ * D_HEAD * q_tensor.element_size()
        k_bytes = total_k * HK * D_HEAD * k_tensor.element_size()
        v_bytes = total_k * HK * D_HEAD_V * v_tensor.element_size()
        o_bytes = total_q * HQ * D_HEAD_V * inputs["output"].element_size()
        mem = q_bytes + k_bytes + v_bytes + o_bytes

        if args.metric == "time":
            return ms
        elif args.metric == "throughput":
            return total_flops / ms * 1e-9
        elif args.metric == "bandwidth":
            return mem / ms * 1e-6
        else:
            raise ValueError(f"Unknown metric: {args.metric}")

    bench_fn.run(None, print_data=True)


def parse_int_or_list(value):
    if "," in value:
        return [int(x) for x in value.split(",")]
    return int(value)


def parse_args():
    parser = get_parser(kernel_name="Unified Attention")

    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument(
        "-sq",
        type=parse_int_or_list,
        default=0,
        help="Query sequence length (single int or comma-separated list)",
    )
    parser.add_argument(
        "-sk",
        type=parse_int_or_list,
        default=0,
        help="Key sequence length (single int or comma-separated list, defaults to sq if 0)",
    )
    parser.add_argument("-d", type=int, default=0, help="Q/K head size")
    parser.add_argument("-dv", type=int, default=0, help="V head size (defaults to -d)")
    parser.add_argument(
        "-num_blocks", type=int, default=0, help="KV cache blocks (0=auto)"
    )
    parser.add_argument("-block_size", type=int, default=0, help="KV cache block size")
    parser.add_argument(
        "-test",
        action="store_true",
        default=False,
        help="Verify correctness against reference implementation for each shape",
    )
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="Use equal sequence lengths (no varlen); default is random varlen",
    )
    parser.add_argument(
        "-fp8",
        action="store_true",
        default=False,
        help="Quantize Q, K, V to FP8 e4m3 with per-tensor descales",
    )
    parser.add_argument(
        "-fp8_kv",
        action="store_true",
        default=False,
        help="Quantize only K, V to FP8 e4m3 (Q stays bf16)",
    )
    parser.add_argument(
        "-fp8_output",
        action="store_true",
        default=False,
        help="Output tensor in FP8 with output_scale",
    )
    parser.add_argument(
        "-out_scale",
        type=float,
        default=1.0,
        help="Output scale factor when -fp8_output is set (default: 1.0)",
    )
    parser.add_argument(
        "-decode",
        nargs="?",
        const=1.0,
        default=0.0,
        type=float,
        metavar="P",
        help="Portion of decode samples (seqlen_q=1) in batch; omit P for all=1.0",
    )
    parser.add_argument(
        "-sliding_window",
        type=int,
        default=None,
        help="Sliding window size (default: disabled)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.fp8 and args.fp8_kv:
        raise ValueError(
            "-fp8 already quantizes K/V; -fp8_kv is redundant. Use one or the other."
        )

    custom_config = False

    if args.hq or args.hk or args.d or args.dv:
        custom_config = True
        if not args.dv:
            args.dv = args.d
        assert (
            args.b and args.hq and args.sq and args.d and args.dv
        ), "Custom config requires -b, -hq, -sq, -d (and optionally -dv)"

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    sys.exit(main())
