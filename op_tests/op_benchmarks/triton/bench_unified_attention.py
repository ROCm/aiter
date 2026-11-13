import argparse
import functools
import itertools
import sys

import torch
import triton

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.shuffle import shuffle_scale_batched
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_unified_attention import (
    dynamic_nvfp4_quant_kv_cache,
    ref_paged_attn,
    shuffle_kv_cache,
)
from op_tests.triton_tests.quant.test_quant_mxfp4 import (
    torch_dequant_nvfp4,
    torch_dynamic_mxfp4_quant,
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


def nvfp4_quant_dequant(tensor):
    """NVFP4 (e2m1 values, one e4m3 scale per 16 along the last dim) round trip: the values a kernel reading
    the NVFP4 tensor sees, for the reference."""
    shape = tensor.shape
    packed, scales = torch_dynamic_mxfp4_quant(
        tensor.reshape(-1, shape[-1]), is_nvfp4=True
    )
    return torch_dequant_nvfp4(packed, scales, out_dtype=torch.float32).reshape(shape)


def nvfp4_quant_shuffle_cache(cache, scale_preshuffle):
    """One NVFP4 cache in the kernels' shuffled layout, as dynamic_nvfp4_quant_kv_cache builds it, with the scales
    preshuffled in scale_preshuffle-row groups (dynamic_nvfp4_quant_kv_cache uses 128, so pages of 128).
    """
    from op_tests.triton_tests.attention import test_unified_attention as T

    _num_blocks, block_size, num_kv_heads, head_size = cache.shape
    x = cache.view(-1, block_size, num_kv_heads, head_size).permute(0, 2, 1, 3)
    data, scales = torch_dynamic_mxfp4_quant(x, is_nvfp4=True)
    scale_k = head_size // 16
    scales = shuffle_scale_batched(
        scales.view(-1, block_size, scale_k), preshuffle_factor=scale_preshuffle
    ).view(-1, num_kv_heads, block_size * scale_k)
    data = T.shuffle_weight(data, arch="gfx950").view(
        -1, num_kv_heads, block_size * head_size // 2
    )
    return (
        torch.cat([data.view(torch.uint8), scales.view(torch.uint8)], dim=-1)
        .contiguous()
        .view(-1, num_kv_heads, block_size, head_size // 2 + scale_k)
    )


def nvfp4_quant_kv_cache_chunked(key_cache, value_cache, blocks_per_chunk=16384):
    """dynamic_nvfp4_quant_kv_cache over chunks of blocks (pages quantize and shuffle independently), so its
    fp32 temporaries stay bounded on large caches."""
    outs = []
    for c in (key_cache, value_cache):
        num_blocks = c.shape[0]
        out = None
        for b0 in range(0, num_blocks, blocks_per_chunk):
            if c.shape[1] >= 128:
                kq, vq = dynamic_nvfp4_quant_kv_cache(
                    c[b0 : b0 + blocks_per_chunk], c[b0 : b0 + blocks_per_chunk]
                )
            else:
                # 64-token pages (A8K4V8's K): scales preshuffled per page
                kq = vq = nvfp4_quant_shuffle_cache(
                    c[b0 : b0 + blocks_per_chunk], c.shape[1]
                )
            if out is None:
                out = torch.empty(
                    (num_blocks,) + kq.shape[1:], dtype=kq.dtype, device=kq.device
                )
            out[b0 : b0 + kq.shape[0]] = kq
            del kq, vq
        outs.append(out)
    return outs[0], outs[1]


def nvfp4_dequant(packed, scales):
    """fp32 values of an already-packed NVFP4 tensor (``(..., D // 2)`` e2m1
    pairs, ``(..., D // 16)`` e4m3 scales), e.g. a pool entry's Q, for the
    reference."""
    *lead, half_d = packed.shape
    return torch_dequant_nvfp4(
        packed.reshape(-1, half_d),
        scales.reshape(-1, scales.shape[-1]),
        out_dtype=torch.float32,
    ).reshape(*lead, half_d * 2)


def _make_ua_cache(
    num_kv_heads,
    head_size_qk,
    head_size_v,
    block_size,
    num_blocks,
    fp8_kv,
    shuffled_kv_cache,
    nvfp4_kv=False,
    nvfp4_ref=True,
    nvfp4_v=True,
):
    """Paged KV cache: key_cache/value_cache at num_blocks capacity.

    unified_attention reads this through block_table, which for most
    sequences in a batch only ever indexes a handful of the num_blocks rows.
    Built once per shape (fixed, shared across the whole pool) instead of once
    per pool entry: regenerating the full capacity for every rotating buffer
    entry would pay for, and count toward sizing, bytes most calls never touch.
    """
    torch.cuda.empty_cache()
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

    # The kernel reads the (optionally) shuffled, (optionally) fp8 layout;
    # k_ref/v_ref stay bf16 and unshuffled for the reference correctness
    # check -- the timed kernel_fn never takes them as arguments at all.
    k_ref, v_ref = key_cache, value_cache
    k_descale = v_descale = None
    if fp8_kv:
        k_ref, k_descale = quantize_to_fp8(key_cache)
        v_ref, v_descale = quantize_to_fp8(value_cache)

    if nvfp4_kv:
        # NVFP4: K/V packed e2m1 + per-16 e4m3 scales, already in the shuffled
        # cache layout (the only one the kernels read). The reference uses the
        # dequantized values: fp32 copies of the whole cache, so only built
        # for a correctness check (nvfp4_ref).
        k, v = nvfp4_quant_kv_cache_chunked(key_cache, value_cache)
        k_descale = torch.ones(1, dtype=torch.float32, device="cuda")
        if nvfp4_v:
            v_ref = nvfp4_quant_dequant(value_cache) if nvfp4_ref else None
            v_descale = torch.ones(1, dtype=torch.float32, device="cuda")
        else:
            # A8K4V8: V stays fp8 (v_ref, v_descale from fp8_kv above) in the
            # A8W8 shuffled layout; the reference reads it with its descale.
            v = shuffle_kv_cache(k_ref, v_ref)[1]
        k_ref = nvfp4_quant_dequant(key_cache) if nvfp4_ref else None
    elif shuffled_kv_cache:
        k, v = shuffle_kv_cache(k_ref, v_ref)
    else:
        k, v = k_ref, v_ref
    softmax_scale = head_size_qk**-0.5
    return k, v, k_ref, v_ref, k_descale, v_descale, softmax_scale


_UA_FIXED_FIELDS = [
    "k",
    "v",
    "k_ref",
    "v_ref",
    "k_descale",
    "v_descale",
    "softmax_scale",
]


def _make_ua_rotating_inputs(
    seq_lens,
    num_heads,
    head_size_qk,
    head_size_v,
    block_size,
    num_blocks,
    fp8_q,
    fp8_output,
    out_scale_value,
    nvfp4_q=False,
):
    """Per-pool-entry inputs: query and everything shaped by this entry's
    (possibly varlen) sequence lengths. The paged KV cache itself is fixed
    -- see _make_ua_cache.
    """
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size_qk,
        dtype=torch.bfloat16,
        device="cuda",
    )

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randperm(
        num_seqs * max_num_blocks_per_seq, dtype=torch.int32, device="cuda"
    ).reshape(num_seqs, max_num_blocks_per_seq)
    cu_query_lens = torch.tensor(
        [0] + query_lens,
        dtype=torch.int32,
        device="cuda",
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    query_lens_t = torch.tensor(query_lens, dtype=torch.int32, device="cuda")

    q_fp8, q_descale = quantize_to_fp8(query) if fp8_q else (None, None)
    q = q_fp8 if fp8_q else query
    # A4W4: Q in NVFP4 with its per-16 scales (the reference dequantizes it,
    # see nvfp4_dequant)
    q_scales = None
    if nvfp4_q:
        q, q_scales = torch_dynamic_mxfp4_quant(
            query.view(-1, head_size_qk), is_nvfp4=True
        )
        q = q.view(-1, num_query_heads, head_size_qk // 2)
        q_scales = q_scales.view(-1, num_query_heads, head_size_qk // 16)

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

    return (
        q,
        output,
        cu_query_lens,
        kv_lens,
        query_lens_t,
        block_tables,
        q_descale,
        q_scales,
        out_scale,
        max_query_len,
        max_kv_len,
    )


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
    seqlens_q,
    seqlens_k,
    HQ,
    HK,
    D_HEAD,
    D_HEAD_V,
    block_size,
    num_blocks,
    fp8_q,
    fp8_output,
    out_scale_value,
    nvfp4_q=False,
):
    """Build one rotating-buffer pool entry: query, block_tables and the
    per-sequence length tensors for the given seqlens_q/seqlens_k. The paged
    KV cache is fixed -- see _make_ua_cache.

    Every entry of a shape gets the same sequence lengths (drawn once in
    run_benchmark); only tensor contents rotate. Varying lengths per entry
    would change max_seqlen_q/max_seqlen_k, which select the attention config
    and 2D/3D kernel, so entries other than the warmed-up one would compile
    during graph capture. It would also make pool sizing depend on a single
    random draw.

    Pure setup, not measured.
    """
    return _make_ua_rotating_inputs(
        seq_lens=list(zip(seqlens_q.tolist(), seqlens_k.tolist())),
        num_heads=(HQ, HK),
        head_size_qk=D_HEAD,
        head_size_v=D_HEAD_V,
        block_size=block_size,
        num_blocks=num_blocks,
        fp8_q=fp8_q,
        fp8_output=fp8_output,
        out_scale_value=out_scale_value,
        nvfp4_q=nvfp4_q,
    )


_UA_POOL_FIELDS = [
    "q",
    "output",
    "cu_seqlens_q",
    "seqused_k",
    "query_lens",
    "block_table",
    "q_descale",
    "q_scales",
    "output_scale",
    "max_seqlen_q",
    "max_seqlen_k",
]


def _make_unified_attention_fn(
    window_size, shuffled_kv_cache, equal_seqlen_q, fp8_pv, backend, causal=True
):
    """Parameterized unified_attention callable for RotatingBufferBench.bench.

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
        q_scales,
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
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            softcap=0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            q_scales=q_scales,
            force_fp8_pv_on_fp4_kv_cache=fp8_pv,
            output_scale=output_scale,
            shuffled_kv_cache=shuffled_kv_cache,
            equal_seqlen_q=equal_seqlen_q,
            backend=backend,
        )

    return kernel_fn


def _mode_label(args):
    parts = []
    if args.nvfp4:
        parts.append("a4w4_nvfp4")
    elif args.k4v8:
        parts.append("a8k4v8")
    elif args.nvfp4_kv:
        parts.append("a8w4_nvfp4_kv")
    if args.fp8_pv:
        parts.append("fp8pv")
    elif args.fp8:
        parts.append("fp8")
    elif args.fp8_kv:
        parts.append("fp8_kv")
    else:
        parts.append("bf16")
    if args.fp8_output:
        parts.append("fp8out")
    if args.shuffled_kv_cache:
        parts.append("shuffled")
    if args.non_causal:
        parts.append("noncausal")
    return "_".join(parts) + "_fwd"


def run_benchmark(custom, args):
    torch.manual_seed(20)

    any_fp8 = args.fp8 or args.fp8_kv or args.fp8_output or args.nvfp4 or args.nvfp4_kv
    any_nvfp4 = args.nvfp4 or args.nvfp4_kv
    label = _mode_label(args)
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

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

        # Fixed across pool entries: sized off the batch's upper bound
        # (N_CTX_K), not any one entry's random draw, so block_tables stays a
        # consistent shape across the whole rotating pool.
        n_ctx_k_max = max(N_CTX_K) if isinstance(N_CTX_K, list) else N_CTX_K
        block_size = args.block_size if args.block_size else 512
        max_num_blocks_per_seq = (n_ctx_k_max + block_size - 1) // block_size
        num_blocks = max(BATCH * max_num_blocks_per_seq, 2048)

        # NVFP4 K/V is built straight into its own shuffled layout (see
        # nvfp4_quant_kv_cache_chunked), so these checks are for shuffle_kv_cache.
        if args.shuffled_kv_cache and not any_nvfp4:
            assert (
                D_HEAD == D_HEAD_V
            ), "Shuffled KV cache requires equal Q/K and V head sizes"
            assert (
                block_size >= 16 and block_size & (block_size - 1) == 0
            ), "Shuffled KV cache requires a power-of-2 block size >= 16"

        window_size = (
            (args.sliding_window - 1, 0)
            if args.sliding_window is not None
            else (-1, -1)
        )

        # Draw sequence lengths once per shape and reuse them for every pool
        # entry -- see _generate_ua_pool_inputs.
        seqlens_q, seqlens_k = _make_seqlens(BATCH, N_CTX_Q, N_CTX_K, varlen, DECODE_P)
        pool_gen = functools.partial(
            _generate_ua_pool_inputs,
            seqlens_q,
            seqlens_k,
            HQ,
            HK,
            D_HEAD,
            D_HEAD_V,
            block_size,
            num_blocks,
            args.fp8,
            args.fp8_output,
            args.out_scale,
            args.nvfp4,
        )
        fixed_gen = functools.partial(
            _make_ua_cache,
            HK,
            D_HEAD,
            D_HEAD_V,
            block_size,
            num_blocks,
            # NVFP4 K/V needs no fp8 copy (args.fp8 only quantizes Q there);
            # A8K4V8 needs the fp8 V
            (args.fp8 and not any_nvfp4) or args.fp8_kv or args.k4v8,
            args.shuffled_kv_cache,
            nvfp4_kv=any_nvfp4,
            # fp32 dequantized cache copies: only for a correctness check
            nvfp4_ref=args.test,
            nvfp4_v=not args.k4v8,
        )
        rot_bufs = bencher.build_pool(
            [pool_gen],
            [_UA_POOL_FIELDS],
            inp_gen_fixed_fns=[fixed_gen],
            inp_fixed_keys=[_UA_FIXED_FIELDS],
        )
        kernel_fn = _make_unified_attention_fn(
            window_size,
            args.shuffled_kv_cache,
            args.equal_seqlen_q,
            args.fp8_pv,
            args.backend,
            causal=not args.non_causal,
        )

        if args.equal_seqlen_q:
            assert (
                seqlens_q == seqlens_q[0]
            ).all(), "-equal_seqlen_q needs equal query lengths"

        ms = bencher.bench(kernel_fn)

        sample = rot_bufs[0]
        if args.test:
            kernel_fn(**sample)
            ref_q, ref_q_descale = sample["q"], sample["q_descale"]
            # the NVFP4 reference values already carry the q descale
            if args.nvfp4:
                ref_q, ref_q_descale = (
                    nvfp4_dequant(sample["q"], sample["q_scales"]),
                    None,
                )
            elif any_nvfp4:
                ref_q, ref_q_descale = (
                    sample["q"].to(torch.float32) * sample["q_descale"],
                    None,
                )
            ref_output = ref_paged_attn(
                query=ref_q,
                key_cache=sample["k_ref"],
                value_cache=sample["v_ref"],
                query_lens=sample["query_lens"],
                kv_lens=sample["seqused_k"],
                block_tables=sample["block_table"],
                scale=sample["softmax_scale"],
                sliding_window=args.sliding_window,
                soft_cap=None,
                q_descale=ref_q_descale,
                k_descale=sample["k_descale"],
                v_descale=sample["v_descale"],
                output_scale=sample["output_scale"],
                out_dtype=sample["output"].dtype,
                causal=0 if args.non_causal else 1,
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

        # main() turns a window of 0 or less into None
        window = args.sliding_window

        def _entry_flops_bytes(entry):
            cu_query_lens = entry["cu_seqlens_q"]
            seqlens_k = entry["seqused_k"]
            num_contexts = len(cu_query_lens) - 1
            flops = 0.0
            total_k = 0
            for i in range(num_contexts):
                sq = (cu_query_lens[i + 1] - cu_query_lens[i]).item()
                sk = seqlens_k[i].item()
                if args.non_causal:
                    valid = sq * sk
                    kv_read = sk
                elif window is None:
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
                flops += valid * HQ * (D_HEAD + D_HEAD_V) * 2.0
                total_k += kv_read

            total_q = cu_query_lens[-1].item()
            # NVFP4: half a byte per value plus one e4m3 scale per 16 values
            nvfp4_bytes = lambda d: d // 2 + d // 16
            q_bytes = (
                total_q
                * HQ
                * (
                    nvfp4_bytes(D_HEAD)
                    if args.nvfp4
                    else D_HEAD * entry["q"].element_size()
                )
            )
            k_bytes = (
                total_k
                * HK
                * (
                    nvfp4_bytes(D_HEAD)
                    if any_nvfp4
                    else D_HEAD * entry["k"].element_size()
                )
            )
            v_bytes = (
                total_k
                * HK
                * (
                    nvfp4_bytes(D_HEAD_V)
                    if any_nvfp4 and not args.k4v8
                    else D_HEAD_V * entry["v"].element_size()
                )
            )
            o_bytes = total_q * HQ * D_HEAD_V * entry["output"].element_size()
            return flops, q_bytes + k_bytes + v_bytes + o_bytes

        # Every pool entry shares the same sequence lengths, so each carries
        # the same work; entry 0's FLOPs/bytes stand for every captured call.
        total_flops, mem = _entry_flops_bytes(sample)

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
        "-equal_seqlen_q",
        action="store_true",
        default=False,
        help="Tell unified_attention every seq has the same query length "
        "(spec-decode verify); needs -equal_seqlens",
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
        "-nvfp4_kv",
        action="store_true",
        default=False,
        help="A8W4: K, V in NVFP4 (e2m1 + per-16 e4m3 scales), Q in FP8; needs -block_size 128",
    )
    parser.add_argument(
        "-k4v8",
        action="store_true",
        default=False,
        help="A8K4V8: Q in FP8, K in NVFP4, V in FP8 (shuffled); needs -block_size 128",
    )
    parser.add_argument(
        "-nvfp4",
        action="store_true",
        default=False,
        help="A4W4: Q, K, V in NVFP4; needs -block_size 128",
    )
    parser.add_argument(
        "-fp8_pv",
        action="store_true",
        default=False,
        help="With -nvfp4 / -nvfp4_kv: force_fp8_pv_on_fp4_kv_cache (PV as an FP8 WMMA)",
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
    parser.add_argument(
        "--non_causal",
        "-non_causal",
        action="store_true",
        default=False,
        help="Let each query token attend to all keys of its sequence",
    )

    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    args = parse_args(args=args)

    if args.fp8 and args.fp8_kv:
        raise ValueError(
            "-fp8 already quantizes K/V; -fp8_kv is redundant. Use one or the other."
        )
    # A window of 0 or less means no window.
    if args.sliding_window is not None and args.sliding_window <= 0:
        args.sliding_window = None
    if args.non_causal and args.sliding_window is not None:
        raise ValueError("-non_causal does not support -sliding_window.")

    if args.fp8_pv and not (args.nvfp4 or args.nvfp4_kv):
        raise ValueError(
            "-fp8_pv applies to an NVFP4 KV cache only (-nvfp4 / -nvfp4_kv)."
        )
    if args.k4v8:
        if args.nvfp4 or args.nvfp4_kv or args.fp8_pv:
            raise ValueError(
                "-k4v8 is its own KV format; drop -nvfp4 / -nvfp4_kv / -fp8_pv."
            )
        args.nvfp4_kv = True
    if args.nvfp4 or args.nvfp4_kv:
        if args.nvfp4 and args.nvfp4_kv:
            raise ValueError(
                "-nvfp4 already quantizes K/V; use one of -nvfp4 / -nvfp4_kv."
            )
        if args.fp8_kv:
            raise ValueError("-fp8_kv conflicts with NVFP4 K/V.")
        if args.block_size != 128 and not (args.k4v8 and args.block_size == 64):
            # the gfx1250 kernels' NVFP4 V-scale layout assumes 128-token pages; an fp8 V (A8K4V8) allows 64
            raise ValueError("NVFP4 K/V needs -block_size 128 (A8K4V8: 64 or 128).")
        # A8W4 runs Q in FP8; NVFP4 K/V lives only in the shuffled layout
        args.fp8 = args.nvfp4_kv
        args.shuffled_kv_cache = True

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
