import argparse
import functools
import itertools
import sys
from collections.abc import Callable
from typing import NamedTuple

import torch
import triton

from aiter.ops.triton.attention.unified_attention import (
    UnifiedAttentionPartials,
    unified_attention,
)
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
    fp4pt_quant_v_cache,
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


def nvfp4_quant_cache_chunked(cache, blocks_per_chunk=16384):
    """One NVFP4 cache in the kernels' shuffled layout (dynamic_nvfp4_quant_kv_cache), over chunks of blocks
    (pages quantize and shuffle independently), so its fp32 temporaries stay bounded on large caches.
    """
    num_blocks = cache.shape[0]
    out = None
    for b0 in range(0, num_blocks, blocks_per_chunk):
        chunk = cache[b0 : b0 + blocks_per_chunk]
        if cache.shape[1] >= 128:
            q = dynamic_nvfp4_quant_kv_cache(chunk, chunk)[0]
        else:
            # 64-token pages (A8K4V8's K): scales preshuffled per page
            q = nvfp4_quant_shuffle_cache(chunk, cache.shape[1])
        if out is None:
            out = torch.empty(
                (num_blocks,) + q.shape[1:], dtype=q.dtype, device=q.device
            )
        out[b0 : b0 + q.shape[0]] = q
        del q
    return out


def fp4pt_quant_v_cache_chunked(value_cache, ref, blocks_per_chunk=8192):
    """A8K4V4PT's fp4pt V cache (fp4pt_quant_v_cache: e2m1 in the NVFP4-V layout + one fp32 scale per token
    and KV head at each page row's tail) over chunks of blocks, so the fp32 temporaries stay bounded on large
    caches; with ref, also the fp32 dequantized copy (e2m1 * scale) for the reference.
    """
    num_blocks = value_cache.shape[0]
    v = v_ref = None
    for b0 in range(0, num_blocks, blocks_per_chunk):
        vq, vr = fp4pt_quant_v_cache(value_cache[b0 : b0 + blocks_per_chunk], ref=ref)
        if v is None:
            v = torch.empty(
                (num_blocks,) + vq.shape[1:], dtype=torch.uint8, device=vq.device
            )
            if ref:
                v_ref = torch.empty(
                    (num_blocks,) + vr.shape[1:], dtype=vr.dtype, device=vr.device
                )
        v[b0 : b0 + vq.shape[0]] = vq.view(torch.uint8)
        if ref:
            v_ref[b0 : b0 + vr.shape[0]] = vr
        del vq, vr
    return v.view(torch.float4_e2m1fn_x2), v_ref


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


# Per-format input preparation. A format pairs a Q preparation (per rotating-pool entry, from the bf16 query)
# with a KV preparation (once per shape, from the bf16 K / V caches [num_blocks, block_size, num_kv_heads, d]).
#
# Q preps return (q, q_descale, q_scales): the kernel's Q, its per-tensor descale (fp8) and its per-16 e4m3
# scales (NVFP4).
# KV preps return a KVCache: the caches the kernel reads (k, v), the reference's caches (k_ref, v_ref: the values
# the kernel sees, for ref_paged_attn; None for the NVFP4 / fp4pt ones unless ref, as they are fp32 copies of the
# whole cache) and the per-tensor descales the kernel and the reference both take.


class KVCache(NamedTuple):
    k: torch.Tensor
    v: torch.Tensor
    k_ref: torch.Tensor | None
    v_ref: torch.Tensor | None
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None


def _prep_q_bf16(query):
    return query, None, None


def _prep_q_fp8(query):
    q, q_descale = quantize_to_fp8(query)
    return q, q_descale, None


def _prep_q_nvfp4(query):
    num_tokens, num_heads, head_size = query.shape
    q, q_scales = torch_dynamic_mxfp4_quant(query.view(-1, head_size), is_nvfp4=True)
    return (
        q.view(num_tokens, num_heads, head_size // 2),
        None,
        q_scales.view(num_tokens, num_heads, head_size // 16),
    )


def _ref_q(q, q_descale, q_scales):
    """The reference's (query, q_descale) for a prepared Q: an NVFP4 Q dequantized, else as is (ref_paged_attn
    applies q_descale)."""
    if q_scales is not None:
        return nvfp4_dequant(q, q_scales), None
    return q, q_descale


def _unit_descale():
    return torch.ones(1, dtype=torch.float32, device="cuda")


def _prep_kv_bf16(key_cache, value_cache, shuffled, ref):
    """a16w16: bf16 K / V, optionally shuffled."""
    k, v = (
        shuffle_kv_cache(key_cache, value_cache)
        if shuffled
        else (key_cache, value_cache)
    )
    return KVCache(k, v, key_cache, value_cache, None, None)


def _prep_kv_fp8(key_cache, value_cache, shuffled, ref):
    """a16w8 / a8w8: FP8 e4m3 K / V with per-tensor descales, optionally shuffled."""
    k8, k_descale = quantize_to_fp8(key_cache)
    v8, v_descale = quantize_to_fp8(value_cache)
    k, v = shuffle_kv_cache(k8, v8) if shuffled else (k8, v8)
    return KVCache(k, v, k8, v8, k_descale, v_descale)


def _prep_kv_nvfp4(key_cache, value_cache, shuffled, ref):
    """a8w4 / a4w4: NVFP4 K / V (e2m1 + per-16 e4m3 scales), built in the shuffled layout."""
    return KVCache(
        nvfp4_quant_cache_chunked(key_cache),
        nvfp4_quant_cache_chunked(value_cache),
        nvfp4_quant_dequant(key_cache) if ref else None,
        nvfp4_quant_dequant(value_cache) if ref else None,
        _unit_descale(),
        _unit_descale(),
    )


def _prep_kv_k4v8(key_cache, value_cache, shuffled, ref):
    """a8k4v8: NVFP4 K (shuffled), FP8 V with its per-tensor descale in the A8W8 shuffled layout."""
    v8, v_descale = quantize_to_fp8(value_cache)
    return KVCache(
        nvfp4_quant_cache_chunked(key_cache),
        shuffle_kv_cache(v8, v8)[1],
        nvfp4_quant_dequant(key_cache) if ref else None,
        v8,
        _unit_descale(),
        v_descale,
    )


def _prep_kv_k4v4pt(key_cache, value_cache, shuffled, ref):
    """a8k4v4pt: NVFP4 K (shuffled), fp4pt V (e2m1 in the NVFP4-V layout + one fp32 scale per token and KV head
    at each page row's tail, passed as torch.float4_e2m1fn_x2). v_descale stays 1: the scales are in the cache.
    """
    v, v_ref = fp4pt_quant_v_cache_chunked(value_cache, ref=ref)
    return KVCache(
        nvfp4_quant_cache_chunked(key_cache),
        v,
        nvfp4_quant_dequant(key_cache) if ref else None,
        v_ref,
        _unit_descale(),
        _unit_descale(),
    )


# Bytes per (token, head) of a d-wide row, for the bandwidth metric
def _bf16_bytes(d):
    return 2 * d


def _fp8_bytes(d):
    return d


def _nvfp4_bytes(d):
    # half a byte per value plus one e4m3 scale per 16 values
    return d // 2 + d // 16


def _fp4pt_bytes(d):
    # half a byte per value plus one fp32 scale per token and head
    return d // 2 + 4


class KVFormat(NamedTuple):
    desc: str
    prep_q: Callable
    prep_kv: Callable
    q_bytes: Callable[[int], int]
    k_bytes: Callable[[int], int]
    v_bytes: Callable[[int], int]
    # page sizes the format allows (None: any)
    block_sizes: tuple[int, ...] | None = None
    # an NVFP4 K cache: built straight into the kernels' shuffled layout
    nvfp4_k: bool = False
    # -fp8_pv (force_fp8_pv_on_fp4_kv_cache) applies
    fp8_pv: bool = False


KV_FORMATS = {
    "a16w16": KVFormat(
        "bf16 Q, K, V",
        _prep_q_bf16,
        _prep_kv_bf16,
        _bf16_bytes,
        _bf16_bytes,
        _bf16_bytes,
    ),
    "a16w8": KVFormat(
        "bf16 Q, FP8 e4m3 K / V (per-tensor descales)",
        _prep_q_bf16,
        _prep_kv_fp8,
        _bf16_bytes,
        _fp8_bytes,
        _fp8_bytes,
    ),
    "a8w8": KVFormat(
        "FP8 e4m3 Q, K, V (per-tensor descales)",
        _prep_q_fp8,
        _prep_kv_fp8,
        _fp8_bytes,
        _fp8_bytes,
        _fp8_bytes,
    ),
    "a8w4": KVFormat(
        "FP8 Q, NVFP4 K / V (e2m1 + per-16 e4m3 scales); -block_size 128",
        _prep_q_fp8,
        _prep_kv_nvfp4,
        _fp8_bytes,
        _nvfp4_bytes,
        _nvfp4_bytes,
        block_sizes=(128,),
        nvfp4_k=True,
        fp8_pv=True,
    ),
    "a4w4": KVFormat(
        "NVFP4 Q, K, V; -block_size 128",
        _prep_q_nvfp4,
        _prep_kv_nvfp4,
        _nvfp4_bytes,
        _nvfp4_bytes,
        _nvfp4_bytes,
        block_sizes=(128,),
        nvfp4_k=True,
        fp8_pv=True,
    ),
    "a8k4v8": KVFormat(
        "FP8 Q, NVFP4 K, FP8 V (shuffled); -block_size 64 or 128",
        _prep_q_fp8,
        _prep_kv_k4v8,
        _fp8_bytes,
        _nvfp4_bytes,
        _fp8_bytes,
        block_sizes=(64, 128),
        nvfp4_k=True,
    ),
    "a8k4v4pt": KVFormat(
        "FP8 Q, NVFP4 K, fp4pt V (e2m1 + one fp32 scale per token and KV head at the page tail); "
        "-block_size 128",
        _prep_q_fp8,
        _prep_kv_k4v4pt,
        _fp8_bytes,
        _nvfp4_bytes,
        _fp4pt_bytes,
        block_sizes=(128,),
        nvfp4_k=True,
    ),
}


def _make_ua_cache(
    kv_format: KVFormat,
    num_kv_heads,
    head_size_qk,
    head_size_v,
    block_size,
    num_blocks,
    shuffled_kv_cache,
    ref=True,
):
    """Paged KV cache at num_blocks capacity, in kv_format (its prep_kv), plus the softmax scale. ref: build the
    reference's fp32 copies of an NVFP4 / fp4pt cache (only for a correctness check).

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
    kv = kv_format.prep_kv(key_cache, value_cache, shuffled_kv_cache, ref)
    return (*kv, head_size_qk**-0.5)


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
    prep_q,
    fp8_output,
    out_scale_value,
):
    """Per-pool-entry inputs: query (in the format's prep_q) and everything shaped by this entry's (possibly
    varlen) sequence lengths. The paged KV cache itself is fixed -- see _make_ua_cache.
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

    q, q_descale, q_scales = prep_q(query)

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
    prep_q,
    fp8_output,
    out_scale_value,
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
        prep_q=prep_q,
        fp8_output=fp8_output,
        out_scale_value=out_scale_value,
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
    window_size,
    shuffled_kv_cache,
    equal_seqlen_q,
    fp8_pv,
    backend,
    skip_reduce=False,
    segm_tiles=None,
):
    """Parameterized unified_attention callable for RotatingBufferBench.bench.

    Argument names match a rotating-buffer pool entry's keys (``**_``
    absorbs the ones the kernel call does not need). skip_reduce and
    segm_tiles (gfx1250 split rebalance) leave only the attention kernel.
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
        return unified_attention(
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
            q_scales=q_scales,
            force_fp8_pv_on_fp4_kv_cache=fp8_pv,
            output_scale=output_scale,
            shuffled_kv_cache=shuffled_kv_cache,
            equal_seqlen_q=equal_seqlen_q,
            backend=backend,
            skip_reduce=skip_reduce,
            segm_tiles=segm_tiles,
        )

    return kernel_fn


def _partials_bytes(entry, partials, num_query_heads):
    """Bytes the split-KV attention kernel writes: for each (query token, query
    head) of a sequence, one segm_output row, segm_max and segm_expsum for each
    segment that covers KV. A segment past the sequence's KV writes nothing.
    """
    seg_out = partials.segm_output
    num_segments = seg_out.shape[2]
    tile = partials.tile_size
    row_bytes = (
        seg_out.shape[-1] * seg_out.element_size()
        + partials.segm_max.element_size()
        + partials.segm_expsum.element_size()
    )
    rebalance_tiles = (
        int(partials.segm_tiles.item()) if partials.segm_tiles is not None else None
    )
    cu_q = entry["cu_seqlens_q"].tolist()
    total = 0
    for i, seq_len in enumerate(entry["seqused_k"].tolist()):
        tiles = triton.cdiv(seq_len, tile)
        if tiles == 0:
            continue
        # the kernel's tiles_per_segment: the rebalance's, else an even split
        per_segment = rebalance_tiles or triton.cdiv(seq_len, num_segments * tile)
        live = triton.cdiv(tiles, per_segment)
        total += (cu_q[i + 1] - cu_q[i]) * num_query_heads * live * row_bytes
    return total


def _mode_label(args):
    parts = [args.kv_format]
    if args.fp8_pv:
        parts.append("fp8pv")
    if args.fp8_output:
        parts.append("fp8out")
    if args.shuffled_kv_cache:
        parts.append("shuffled")
    return "_".join(parts) + "_fwd"


def run_benchmark(custom, args):
    torch.manual_seed(20)

    fmt = KV_FORMATS[args.kv_format]
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
        if args.shuffled_kv_cache and not fmt.nvfp4_k:
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
            fmt.prep_q,
            args.fp8_output,
            args.out_scale,
        )
        fixed_gen = functools.partial(
            _make_ua_cache,
            fmt,
            HK,
            D_HEAD,
            D_HEAD_V,
            block_size,
            num_blocks,
            args.shuffled_kv_cache,
            # fp32 dequantized cache copies: only for a correctness check
            ref=args.test,
        )
        rot_bufs = bencher.build_pool(
            [pool_gen],
            [_UA_POOL_FIELDS],
            inp_gen_fixed_fns=[fixed_gen],
            inp_fixed_keys=[_UA_FIXED_FIELDS],
        )
        make_fn = functools.partial(
            _make_unified_attention_fn,
            window_size,
            args.shuffled_kv_cache,
            args.equal_seqlen_q,
            args.fp8_pv,
            args.backend,
        )
        full_fn = make_fn()

        if args.equal_seqlen_q:
            assert (
                seqlens_q == seqlens_q[0]
            ).all(), "-equal_seqlen_q needs equal query lengths"

        sample = rot_bufs[0]
        # The attention kernel alone: one untimed skip_reduce call gives the
        # split's partials and, on gfx1250, the split rebalance's segm_tiles.
        # The timed calls take that segm_tiles, so they skip the rebalance kernel.
        partials = None
        kernel_fn = full_fn
        if args.only_bench_attn:
            assert (
                args.sliding_window is None
            ), "--only_bench_attn counts the partials without a sliding window"
            result = make_fn(skip_reduce=True)(**sample)
            if isinstance(result, UnifiedAttentionPartials):
                partials = result
                kernel_fn = make_fn(skip_reduce=True, segm_tiles=result.segm_tiles)

        ms = bencher.bench(kernel_fn)

        if args.test:
            full_fn(**sample)
            ref_q, ref_q_descale = _ref_q(
                sample["q"], sample["q_descale"], sample["q_scales"]
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
            )
            if args.kv_format != "a16w16" or args.fp8_output:
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

        # Kernel disables windowing for non-positive values (SLIDING_WINDOW > 0).
        window = (
            args.sliding_window
            if args.sliding_window is not None and args.sliding_window > 0
            else None
        )

        def _entry_flops_bytes(entry):
            cu_query_lens = entry["cu_seqlens_q"]
            seqlens_k = entry["seqused_k"]
            num_contexts = len(cu_query_lens) - 1
            flops = 0.0
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
                flops += valid * HQ * (D_HEAD + D_HEAD_V) * 2.0
                total_k += kv_read

            total_q = cu_query_lens[-1].item()
            q_bytes = total_q * HQ * fmt.q_bytes(D_HEAD)
            k_bytes = total_k * HK * fmt.k_bytes(D_HEAD)
            v_bytes = total_k * HK * fmt.v_bytes(D_HEAD_V)
            if partials is None:
                o_bytes = total_q * HQ * D_HEAD_V * entry["output"].element_size()
            else:
                o_bytes = _partials_bytes(entry, partials, HQ)
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
        "-kv_format",
        choices=list(KV_FORMATS),
        default="a16w16",
        help="Q / KV cache format: "
        + "; ".join(f"{name}: {fmt.desc}" for name, fmt in KV_FORMATS.items()),
    )
    parser.add_argument(
        "-fp8_pv",
        action="store_true",
        default=False,
        help="With -kv_format a8w4 / a4w4: force_fp8_pv_on_fp4_kv_cache (PV as an FP8 WMMA)",
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
        "--only_bench_attn",
        "-only_bench_attn",
        action="store_true",
        default=False,
        help="Time only the attention kernel: skip the reduce and reuse the "
        "gfx1250 split rebalance's segm_tiles from an untimed call; bytes count "
        "the partials it writes. Default: split rebalance + attention + reduce, "
        "with end-to-end bytes",
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

    fmt = KV_FORMATS[args.kv_format]
    if args.fp8_pv and not fmt.fp8_pv:
        raise ValueError("-fp8_pv applies to -kv_format a8w4 / a4w4 only.")
    if fmt.block_sizes is not None and args.block_size not in fmt.block_sizes:
        sizes = " or ".join(map(str, fmt.block_sizes))
        raise ValueError(f"-kv_format {args.kv_format} needs -block_size {sizes}.")
    if fmt.nvfp4_k:
        # an NVFP4 K cache exists only in the shuffled layout
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
