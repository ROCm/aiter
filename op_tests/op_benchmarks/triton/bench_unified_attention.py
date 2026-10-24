import argparse
import functools
import itertools
import sys

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
)
from op_tests.triton_tests.attention.test_unified_attention import (
    ref_paged_attn,
    shuffle_kv_cache,
)

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
    descale = (abs_max / FP8_MAX).to(torch.float32).unsqueeze(0)
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

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "q_fp8": q_fp8,
        "k_fp8": k_fp8,
        "v_fp8": v_fp8,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "out_scale": out_scale,
        "output": output,
        "cu_query_lens": cu_query_lens,
        "kv_lens": kv_lens,
        "query_lens": query_lens_t,
        "block_tables": block_tables,
        "max_query_len": max_query_len,
        "max_kv_len": max_kv_len,
        "scale": scale,
    }


def _make_seqlens(BATCH, N_CTX_Q, N_CTX_K, varlen, DECODE_P):
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
        num_decode = round(DECODE_P * BATCH)
        if num_decode > 0:
            decode_idx = torch.randperm(BATCH, device=seqlens_q.device)[:num_decode]
            seqlens_q[decode_idx] = 1

    return seqlens_q, seqlens_k


def _generate_ua_pool_inputs(
    BATCH,
    N_CTX_Q,
    N_CTX_K,
    HQ,
    HK,
    D_HEAD,
    D_HEAD_V,
    DECODE_P,
    varlen,
    block_size,
    num_blocks,
    fp8_q,
    fp8_kv,
    fp8_output,
    out_scale_value,
    shuffled_kv_cache,
):
    """Build one full request batch: random per-sequence lengths, a fresh KV
    cache, and (optionally) its FP8/shuffled views.

    Pure setup, not measured. Each call is one rotating-buffer pool entry:
    independent random sequence lengths and cache contents, so replaying
    through the pool during timing defeats cache residency and avoids biasing
    the measurement on a single lucky/unlucky varlen draw.
    """
    seqlens_q, seqlens_k = _make_seqlens(BATCH, N_CTX_Q, N_CTX_K, varlen, DECODE_P)

    inputs = make_inputs(
        seq_lens=list(zip(seqlens_q.tolist(), seqlens_k.tolist())),
        num_heads=(HQ, HK),
        head_size_qk=D_HEAD,
        head_size_v=D_HEAD_V,
        block_size=block_size,
        num_blocks=num_blocks,
        fp8_q=fp8_q,
        fp8_kv=fp8_kv,
        fp8_output=fp8_output,
        out_scale_value=out_scale_value,
    )

    q = inputs["q_fp8"] if fp8_q else inputs["query"]
    # The kernel reads the (optionally) shuffled layout; k_ref/v_ref stay
    # unshuffled for the reference correctness check.
    k_ref = inputs["k_fp8"] if fp8_kv else inputs["key_cache"]
    v_ref = inputs["v_fp8"] if fp8_kv else inputs["value_cache"]
    if shuffled_kv_cache:
        k, v = shuffle_kv_cache(k_ref, v_ref)
    else:
        k, v = k_ref, v_ref

    return (
        q,
        k,
        v,
        k_ref,
        v_ref,
        inputs["output"],
        inputs["cu_query_lens"],
        inputs["kv_lens"],
        inputs["query_lens"],
        inputs["block_tables"],
        inputs["q_descale"],
        inputs["k_descale"],
        inputs["v_descale"],
        inputs["out_scale"],
        inputs["max_query_len"],
        inputs["max_kv_len"],
        inputs["scale"],
    )


_UA_POOL_FIELDS = [
    "q",
    "k",
    "v",
    "k_ref",
    "v_ref",
    "output",
    "cu_seqlens_q",
    "seqused_k",
    "query_lens",
    "block_table",
    "q_descale",
    "k_descale",
    "v_descale",
    "output_scale",
    "max_seqlen_q",
    "max_seqlen_k",
    "softmax_scale",
]


def _make_unified_attention_fn(window_size, shuffled_kv_cache, backend):
    """Parameterized unified_attention callable for do_bench_aiter_triton.

    Argument names match a rotating-buffer pool entry's keys (``**_``
    absorbs the ones the kernel call does not need).
    """

    def kernel_fn(
        q,
        k,
        v,
        output,
        cu_seqlens_q,
        seqused_k,
        block_table,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        **_,
    ):
        unified_attention(
            q=q,
            k=k,
            v=v,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            block_table=block_table,
            softcap=0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            output_scale=output_scale,
            shuffled_kv_cache=shuffled_kv_cache,
            backend=backend,
        )

    return kernel_fn


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
    if args.shuffled_kv_cache:
        parts.append("shuffled")
    return "_".join(parts) + "_fwd"


def run_benchmark(custom, args):
    torch.manual_seed(20)

    any_fp8 = args.fp8 or args.fp8_kv or args.fp8_output
    label = _mode_label(args)
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

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

        # Fixed across pool entries: sized off the sweep point's upper bound
        # (N_CTX_K), not any one entry's random draw, so block_tables stays a
        # consistent shape across the whole rotating pool.
        n_ctx_k_max = max(N_CTX_K) if isinstance(N_CTX_K, list) else N_CTX_K
        block_size = args.block_size if args.block_size else 512
        max_num_blocks_per_seq = (n_ctx_k_max + block_size - 1) // block_size
        min_required_blocks = BATCH * max_num_blocks_per_seq
        num_blocks = (
            args.num_blocks if args.num_blocks else max(min_required_blocks * 4, 2048)
        )

        if args.shuffled_kv_cache:
            assert D_HEAD == D_HEAD_V, (
                "Shuffled KV cache requires equal Q/K and V head sizes"
            )
            assert block_size >= 16 and block_size & (block_size - 1) == 0, (
                "Shuffled KV cache requires a power-of-2 block size >= 16"
            )

        window_size = (
            (args.sliding_window - 1, 0)
            if args.sliding_window is not None
            else (-1, -1)
        )

        pool_gen = functools.partial(
            _generate_ua_pool_inputs,
            BATCH,
            N_CTX_Q,
            N_CTX_K,
            HQ,
            HK,
            D_HEAD,
            D_HEAD_V,
            DECODE_P,
            varlen,
            block_size,
            num_blocks,
            args.fp8,
            args.fp8 or args.fp8_kv,
            args.fp8_output,
            args.out_scale,
            args.shuffled_kv_cache,
        )
        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_UA_POOL_FIELDS], **kwargs_rot_bufs
        )
        kernel_fn = _make_unified_attention_fn(
            window_size, args.shuffled_kv_cache, args.backend
        )

        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)

        sample = rot_bufs[0]
        if args.test:
            kernel_fn(**sample)
            ref_output = ref_paged_attn(
                query=sample["q"],
                key_cache=sample["k_ref"],
                value_cache=sample["v_ref"],
                query_lens=sample["query_lens"],
                kv_lens=sample["seqused_k"],
                block_tables=sample["block_table"],
                scale=sample["softmax_scale"],
                sliding_window=args.sliding_window,
                soft_cap=None,
                q_descale=sample["q_descale"],
                k_descale=sample["k_descale"],
                v_descale=sample["v_descale"],
                output_scale=sample["output_scale"],
                out_dtype=sample["output"].dtype,
            )
            if any_fp8:
                atol, rtol = 1.5e-1, 1.5e-1
            else:
                atol, rtol = 1.5e-2, 1e-2
            out_f32 = sample["output"].to(torch.float32)
            ref_f32 = ref_output.to(torch.float32)
            max_diff = torch.max(torch.abs(out_f32 - ref_f32)).item()
            shape_str = f"(B={BATCH}, HQ={HQ}, HK={HK}, SQ={N_CTX_Q}, SK={N_CTX_K}, D={D_HEAD}, DV={D_HEAD_V})"
            try:
                torch.testing.assert_close(out_f32, ref_f32, atol=atol, rtol=rtol)
                print(f"  PASS {shape_str}  max_diff={max_diff:.6f}")
            except AssertionError as e:
                print(f"  FAIL {shape_str}  max_diff={max_diff:.6f}")
                print(f"    {e}")

        # Metrics are computed from one representative pool entry -- the
        # original single-buffer benchmark was equally a single random draw.
        cu_query_lens = sample["cu_seqlens_q"]
        seqlens_k = sample["seqused_k"]
        num_contexts = len(cu_query_lens) - 1
        # Kernel disables windowing for non-positive values (SLIDING_WINDOW > 0).
        window = (
            args.sliding_window
            if args.sliding_window is not None and args.sliding_window > 0
            else None
        )
        total_flops = 0.0
        total_k = 0
        for i in range(num_contexts):
            sq = (cu_query_lens[i + 1] - cu_query_lens[i]).item()
            sk = seqlens_k[i].item()
            if window is None:
                valid = sq * sk - ((sq**2 - sq) / 2)
                kv_read = sk
            else:
                # With a sliding window each of the sq (causal) queries attends to
                # at most `window` keys, so count windowed causal pairs instead of
                # the full triangle. Query i (0-indexed within the block) attends
                # to min(sk - sq + i + 1, window) keys.
                idx = torch.arange(sq, device=seqlens_k.device)
                valid = torch.clamp(sk - sq + 1 + idx, max=window).sum().item()
                # Keys streamed from HBM = union of all query windows.
                kv_read = min(sk, sq - 1 + window)
            total_flops += valid * HQ * (D_HEAD + D_HEAD_V) * 2.0
            total_k += kv_read

        total_q = cu_query_lens[-1].item()
        q_bytes = total_q * HQ * D_HEAD * sample["q"].element_size()
        k_bytes = total_k * HK * D_HEAD * sample["k"].element_size()
        v_bytes = total_k * HK * D_HEAD_V * sample["v"].element_size()
        o_bytes = total_q * HQ * D_HEAD_V * sample["output"].element_size()
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


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
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

    parser.add_argument(
        "--shuffled_kv_cache",
        "-shuffled_kv_cache",
        action="store_true",
        default=False,
        help="Pre-shuffle the KV cache into the layout the gluon kernels expect",
    )

    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    args = parse_args(args=args)

    if args.fp8 and args.fp8_kv:
        raise ValueError(
            "-fp8 already quantizes K/V; -fp8_kv is redundant. Use one or the other."
        )

    custom_config = False

    if args.hq or args.hk or args.d or args.dv:
        custom_config = True
        if not args.dv:
            args.dv = args.d
        assert args.b and args.hq and args.sq and args.d and args.dv, (
            "Custom config requires -b, -hq, -sq, -d (and optionally -dv)"
        )

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    sys.exit(main())
