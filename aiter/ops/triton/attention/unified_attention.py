# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
from typing import NamedTuple

import os
import torch
import triton

from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    attention_output_gate_rows,
    unified_attention_3d_split_rebalance_kernel,
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import get_arch
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.common_utils import max_addressable_bytes
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils.unified_attention_utils import (
    get_dtype_str,
    get_unified_attention_config,
)

# gfx1250
try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_3d import (
        _unified_attention_gluon_kernel_3d as _unified_attention_kernel_3d_gfx1250,
    )
except:  # noqa: E722
    _unified_attention_kernel_3d_gfx1250 = None
try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_2d import (
        _unified_attention_gluon_kernel_2d as _unified_attention_kernel_2d_gfx1250,
    )
except:  # noqa: E722
    _unified_attention_kernel_2d_gfx1250 = None
try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_reduce import (
        reduce_segments_gluon as _reduce_segments_kernel_gfx1250,
    )
except:  # noqa: E722
    _reduce_segments_kernel_gfx1250 = None

# gfx950
try:
    from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
        _unified_attention_gluon_kernel as _unified_attention_kernel_gfx950,
    )
except:  # noqa: E722
    _unified_attention_kernel_gfx950 = None

# Max NUM_SEGMENTS the gluon reduce holds in-thread; larger split counts fall back to the Triton reduce_segments.
_GLUON_REDUCE_MAX_SEGMENTS = 8
# gfx1250, short queries (speculative-decode verify, q_len = 1 + num_spec): the 3D
# split-KV kernel beats the 2D one while the batch has at most this many query rows
# (tokens * query heads) per CU; past that the 2D grid fills the GPU.
_GFX12_3D_MAX_SEQLEN_Q = 16
_GFX12_3D_MAX_QUERY_ROWS_PER_CU = 8
# Split rebalance for the gfx1250 gluon 3d kernel (decode and spec-decode verify),
# always on: every KV segment covers the same number of tiles, so a long sequence gets
# more segments instead of longer ones. Cutting every sequence into the same number of
# segments leaves, with mixed context lengths, the longest sequences' segments running
# long after the rest are done: Qwen3.5 (16 q / 2 kv heads, d 256, fp8, 4-token verify),
# 64 sequences of 4k-191k tokens: 5.5 TB/s, against 16.7 for 64 x 64k.
# The segment size is computed on the device each launch from the batch's total KV,
# for about _GFX12_UA_SPLIT_REBALANCE programs, clamped to
# [_GFX12_UA_SPLIT_REBALANCE_MIN_TILES, _GFX12_UA_SPLIT_REBALANCE_MAX_TILES] tiles, so
# it follows the live batch under a CUDA graph too. The grid covers
# _GFX12_UA_SPLIT_REBALANCE_MAX_KV_LEN tokens at the minimum segment size, since a graph
# keeps its capture's grid; spare segments exit at once.
_GFX12_UA_SPLIT_REBALANCE = 2048
_GFX12_UA_SPLIT_REBALANCE_MIN_TILES = 16
_GFX12_UA_SPLIT_REBALANCE_MAX_TILES = 256
_GFX12_UA_SPLIT_REBALANCE_MAX_KV_LEN = 262144
# attn_3d config keys that steer use_2d_kernel on gfx1250 rather than the launch:
# MAX_QUERY_ROWS_PER_CU overrides the short-query bound above, and
# PREFILL_MAX_2D_PRGMS / PREFILL_MIN_KV / PREFILL_MIN_KV_PER_PRGM opt an entry
# into the 3D split for prefill over a long cached prefix, and PREFILL_FILL_TIERS
# raises that cap for uneven batches when the caller passes attn_work (see
# use_2d_kernel).
_GFX12_ROUTING_KEYS = (
    "MAX_QUERY_ROWS_PER_CU",
    "PREFILL_MAX_2D_PRGMS",
    "PREFILL_MIN_KV",
    "PREFILL_MIN_KV_PER_PRGM",
    "PREFILL_FILL_TIERS",
    "PREFILL_REF_BLOCK_M",
)
# gfx1250: the output gate inside the 2d kernel's epilogue (its gate tile TDM-loaded
# before the attention loop) from this longest context on; below it, and with
# AITER_UA2D_FUSED_EPILOGUE=0, the gate is the separate attention_output_gate_rows
# launch after the 2d kernel. Short prefills (478-8192 new tokens, no prefix) measured
# 2-6 us slower with the gate in the epilogue than with the separate launch.
_UA2D_FUSED_EPILOGUE = os.environ.get("AITER_UA2D_FUSED_EPILOGUE", "1") == "1"
_UA2D_FUSED_EPILOGUE_MIN_KV = int(
    os.environ.get("AITER_UA2D_FUSED_EPILOGUE_MIN_KV", "16384")
)

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250", "gfx1250-strict")
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64

_GLUON_SUPPORTED_ARCHS = ("gfx1250", "gfx1250-strict", "gfx950")

# LDS the gfx950 KV double buffer may occupy; a wider tile falls back to one buffer.
_GFX950_LDS_BUDGET = 160 * 1024


def _is_gluon_available():
    return any(supported in DEVICE_ARCH for supported in _GLUON_SUPPORTED_ARCHS)


def _split_launch_config(config):
    # NUM_SEGMENTS inherits the caller's max_seqlen_k type when the context
    # length bounds the split, so a numpy max_seqlen_k makes it a numpy scalar.
    # Gluon constexpr arguments must be native Python scalars; numpy ones fail
    # during TTIR lowering once a runtime shape selects the 3D path.
    return int(config["NUM_SEGMENTS"]), int(config["TILE_SIZE"])


class _UAParams(NamedTuple):
    # tensors
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    cu_seqlens_q: torch.Tensor  # [num_seqs + 1], kernels' query_start_len_ptr
    seqused_k: torch.Tensor  # [num_seqs], kernels' seq_lens_ptr
    block_table: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]

    # scalars
    softmax_scale: float
    softcap: float
    causal: bool
    sliding_window: int  # kernels' SLIDING_WINDOW, i.e. 1 + window_size[0]
    max_seqlen_q: int
    max_seqlen_k: int

    # shapes
    num_tokens: int  # q.shape[0]: queries summed over all seqs
    num_query_heads: int
    num_kv_heads: int
    num_queries_per_kv: int
    head_size: int  # logical, i.e. already doubled for fp4-packed q
    head_size_v: int  # v's head size, which may differ from q/k's
    num_seqs: int
    total_num_q_blocks: int  # query-block upper bound at the default BLOCK_Q
    num_2d_prgms: int  # total_num_q_blocks * num_kv_heads

    # kv cache layout
    num_blocks: int
    block_size: int  # kv page size, kernels' BLOCK_SIZE
    k_width: int
    scale_k_width: int
    block_scales_size: int  # elements sharing one quantization scale

    # dtypes and modes
    q_dtype: torch.dtype
    kv_cache_dtype: torch.dtype  # the K cache's
    # the V cache's; differs from kv_cache_dtype only for A8K4V8 (NVFP4 K, fp8 V)
    v_cache_dtype: torch.dtype
    all_decode: bool  # max_seqlen_q == 1
    # equal_seqlen_q and not all_decode: speculative-decode verify, every seq
    # holds max_seqlen_q queries
    spec_decode: bool
    shuffled_kv_cache: bool
    use_alibi_slopes: bool  # alibi_slopes is not None
    use_qq_bias: bool  # qq_bias is not None

    # device
    num_sms: int  # CU count; occupancy targets are derived from it
    target_num_prgms: int  # num_sms * 4: the target the heuristics aim at

    # optional inputs
    # sum over seqs of new tokens x mean attended context (causal query-key pairs),
    # from the caller's host-side lengths; 0 when not given
    attn_work: int = 0
    sinks: torch.Tensor | None = None
    alibi_slopes: torch.Tensor | None = None
    qq_bias: torch.Tensor | None = None
    q_scales: torch.Tensor | None = None  # fp4 per-block query scales
    q_descale: torch.Tensor | None = None
    k_descale: torch.Tensor | None = None
    v_descale: torch.Tensor | None = None
    output_scale: torch.Tensor | None = None
    skip_reduce: bool = False
    # NVFP4 KV cache only: PV as an FP8 WMMA (P and the upcast V in e4m3) instead of BF16
    force_fp8_pv_on_fp4_kv_cache: bool = False


def _gfx950_gluon_supported(params: _UAParams):
    """GFX950 Gluon supported configs."""
    if params.shuffled_kv_cache and (
        params.block_size & (params.block_size - 1)
        or params.block_size < params.k_width
    ):
        return False

    return (
        DEVICE_ARCH == "gfx950"
        and _unified_attention_kernel_gfx950 is not None
        # softcap hits an AMDGPU backend assert (GCNRewritePartialRegUses) on Triton 3.8
        and not params.softcap
        and not params.use_qq_bias
        and not params.use_alibi_slopes
        and params.head_size <= 256
        and triton.next_power_of_2(params.head_size) == params.head_size
        and params.head_size_v == params.head_size
        and params.q_dtype in (e4m3_dtype, torch.bfloat16)
        and params.q_dtype == params.kv_cache_dtype
    )


def _gfx950_split_params(params: _UAParams):
    """Size the split by the gluon grid when a sequence spans several query blocks."""
    q_rows = params.max_seqlen_q * params.num_queries_per_kv
    if params.all_decode or (q_rows <= 128 and params.sliding_window <= 0):
        return params
    config = get_unified_attention_config("attn_3d", params, backend="gluon")
    block_m = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    if q_rows <= 128:
        block_m = max(block_m, triton.next_power_of_2(q_rows))
    block_q = block_m // params.num_queries_per_kv
    num_q_blocks = params.num_tokens // block_q + params.num_seqs
    return params._replace(num_2d_prgms=num_q_blocks * params.num_kv_heads)


_SW_SPLIT_MAX_Q = 16


def _gfx12_split_rebalance_tiles(
    seqused_k: torch.Tensor, num_kv_heads: int, tile_size: int
) -> torch.Tensor:
    """The split rebalance's tiles per segment, [1] int32, in one single-program
    Triton launch (unified_attention_3d_split_rebalance_kernel):
      clamp(ceil(sum_i ceil(seqused_k[i] / tile_size) * num_kv_heads
                 / _GFX12_UA_SPLIT_REBALANCE),
            _GFX12_UA_SPLIT_REBALANCE_MIN_TILES, _GFX12_UA_SPLIT_REBALANCE_MAX_TILES).

    No host read of device data and a fixed grid, so it captures into a CUDA graph
    and recomputes from the live seqused_k on replay.
    """
    assert seqused_k.dim() == 1, "seqused_k must be 1-D [num_seqs]"
    segm_tiles = torch.empty(1, dtype=torch.int32, device=seqused_k.device)
    num_seqs = seqused_k.shape[0]
    unified_attention_3d_split_rebalance_kernel[(1,)](
        seqused_k,
        segm_tiles,
        num_seqs,
        seqused_k.stride(0),
        num_kv_heads,
        TILE_SIZE=tile_size,
        PROGRAMS=_GFX12_UA_SPLIT_REBALANCE,
        MIN_TILES=_GFX12_UA_SPLIT_REBALANCE_MIN_TILES,
        MAX_TILES=_GFX12_UA_SPLIT_REBALANCE_MAX_TILES,
        BLOCK=min(1024, max(64, triton.next_power_of_2(num_seqs))),
        num_warps=4,
    )
    return segm_tiles


def _gfx12_split_rebalance(params: _UAParams, tile_size: int):
    """The split rebalance's grid width (NUM_SEGMENTS) and its tiles per segment.

    Tiles per segment is a device tensor: the batch's KV tiles over every KV head,
    over _GFX12_UA_SPLIT_REBALANCE, clamped to [MIN_TILES, MAX_TILES], from one
    Triton program, so a CUDA graph recomputes it from the live context lengths on
    replay.
    """
    segm_tiles = _gfx12_split_rebalance_tiles(
        params.seqused_k, params.num_kv_heads, tile_size
    )
    kv_cover = max(int(params.max_seqlen_k), _GFX12_UA_SPLIT_REBALANCE_MAX_KV_LEN)
    num_segments = triton.next_power_of_2(
        triton.cdiv(kv_cover, _GFX12_UA_SPLIT_REBALANCE_MIN_TILES * tile_size)
    )
    return num_segments, segm_tiles


def _gfx950_split_grid(params: _UAParams):
    """Decode, and batches too small to fill the GPU, take the split grid."""
    if params.all_decode:
        return True
    # the split carves only the window, so a full table splits like a trimmed one
    if params.sliding_window > 0 and not (
        params.max_seqlen_q <= _SW_SPLIT_MAX_Q
        and params.max_seqlen_q * params.num_queries_per_kv <= 128
    ):
        return False
    # unsplit, the 2d grid is faster
    config = get_unified_attention_config(
        "kv_split", _gfx950_split_params(params), backend="gluon"
    )
    return config["NUM_SEGMENTS"] > 1


def _gfx12_2d_prgms(params: _UAParams, block_m: int | None = None):
    """Programs the gfx1250 2D kernel would launch (KV heads x query blocks), and
    the queries a block covers; at the attn_2d entry's BLOCK_M unless given."""
    if block_m is None:
        config = get_unified_attention_config("attn_2d", params, backend="gluon")
        block_m = config["BLOCK_M"]
    block_m = max(block_m, triton.next_power_of_2(params.num_queries_per_kv))
    block_q = block_m // params.num_queries_per_kv
    prgms = params.num_kv_heads * (params.num_tokens // block_q + params.num_seqs)
    return prgms, block_q


def _gfx12_prefill_fill(params: _UAParams, prgms: int, block_q: int):
    """Share of a 2D prefill launch's time that is real work: the caller's
    attn_work over (programs x the longest walk, one block over the longest
    context). Near 1 the grid is evenly loaded; low when a long-context seq adds
    a few tokens next to a big chunk over a short context. None without the hint."""
    if params.attn_work <= 0:
        return None
    return (params.attn_work * params.num_kv_heads) / (
        prgms * block_q * params.max_seqlen_k
    )


def use_2d_kernel(params: _UAParams, backend: str = "triton"):
    # if IS_DEVICE_ARCH_GFX12, use 3D for decode, and for short (spec-decode verify)
    # queries while their rows are too few to fill the 2D grid; 2D otherwise, unless
    # the attn_3d entry opts prefill over a long cached prefix into the 3D split.
    if IS_DEVICE_ARCH_GFX12:
        if params.sliding_window > 0:
            return True
        if params.all_decode:
            return False
        if params.max_seqlen_q <= _GFX12_3D_MAX_SEQLEN_Q and is_3d_gluon_available(
            params, backend
        ):
            # the split rebalance keeps the 3d grid full at any batch, where 2d runs
            # one program per sequence over its whole context
            return False
        route = get_unified_attention_config("attn_3d", params, backend="gluon")
        if params.max_seqlen_q <= _GFX12_3D_MAX_SEQLEN_Q:
            rows_cap = route.get(
                "MAX_QUERY_ROWS_PER_CU", _GFX12_3D_MAX_QUERY_ROWS_PER_CU
            )
            return (
                params.num_tokens * params.num_query_heads > rows_cap * params.num_sms
            )
        # With few KV heads (TP) a prefill chunk can launch too few 2D programs to
        # fill the GPU, each streaming the whole prefix; the 3D kernel splits the
        # KV range instead. It pays while each of those programs has a long run of
        # keys to split, i.e. the prefix dominates the chunk.
        max_prgms = route.get("PREFILL_MAX_2D_PRGMS", 0)
        if max_prgms == 0:
            return True
        prgms, block_q = _gfx12_2d_prgms(params, route.get("PREFILL_REF_BLOCK_M"))
        # A 2D call lasts as long as its longest program, one block walking the
        # longest context. With the caller's work hint, fill = work / (programs x
        # that walk) says how much of the grid's time is real work: a long-context
        # seq adding a few tokens next to a big chunk over a short context leaves
        # most programs idle, and the 3D split pays above the plain program cap.
        fill = _gfx12_prefill_fill(params, prgms, block_q)
        if fill is not None:
            for max_fill, tier_prgms in route.get("PREFILL_FILL_TIERS", ()):
                if fill < max_fill:
                    max_prgms = max(max_prgms, tier_prgms)
        min_kv = max(
            route.get("PREFILL_MIN_KV", 0),
            route.get("PREFILL_MIN_KV_PER_PRGM", 0) * prgms,
        )
        return prgms >= max_prgms or params.max_seqlen_k < min_kv

    # The gfx950 Gluon kernel is one kernel with a decode and a prefill grid,
    # and it owns its KV split, so decode and split batches take the 3d path
    # (a single segment simply drops the split axis) and the rest the 2d one.
    if backend == "gluon" and _gfx950_gluon_supported(params):
        return not _gfx950_split_grid(params)

    if params.head_size >= 512 and not get_arch().is_rdna and not params.all_decode:
        return True

    gate = params.target_num_prgms
    if DEVICE_ARCH == "gfx1100":
        # On gfx1100 the 3D split-KV kernel only pays off while the 2D launch
        # is too small to fill the machine; past that the extra segment traffic
        # and the reduce pass cost more than the added parallelism. The
        # measured crossover is cu_count / 4, and target_num_prgms is
        # cu_count * 4, hence the // 16.
        gate = params.target_num_prgms // 16

    return (
        (params.sliding_window > 0)
        or (params.max_seqlen_k <= 512)
        or (params.num_2d_prgms > gate)
    )


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
    # every seq has the same query length (max_seqlen_q), e.g. speculative-decode
    # verify; lets the kernels locate a query block without searching cu_seqlens_q
    equal_seqlen_q: bool = False,
    # optional: the batch's causal query-key pairs, sum_i q_i * (k_i - q_i / 2) over
    # its seqs, computed by the caller on the host. Lets the gfx1250 router see how
    # evenly a prefill's work spreads over the 2D grid; never read on the GPU.
    attn_work: int | None = None,
    # NVFP4 (uint8 e2m1) KV cache only: run PV as an FP8 WMMA, with P and the upcast,
    # scaled V in e4m3, instead of a BF16 WMMA. Same for A8W4 and A4W4; ignored otherwise.
    force_fp8_pv_on_fp4_kv_cache: bool = False,
    # backend
    backend: str | None = None,  # "triton" | "gluon"
    # Epilogue of the Qwen3-Next / Qwen3.5 attention: out * sigmoid(output_gate),
    # output_gate [num_tokens, num_query_heads * head_size_v] (row-strided, e.g. a
    # slice of the QKV-gate projection), bitwise the unfused sigmoid-mul. With
    # output_rows ([R] int32/int64), out has R rows: out[r] is query token
    # output_rows[r] (every token still attends and its KV is read as usual).
    output_gate: torch.Tensor | None = None,
    output_rows: torch.Tensor | None = None,
):
    assert causal, "Only causal attention is supported"

    if backend is None:
        backend = "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"
    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]

    q_dtype = q.dtype
    kv_cache_dtype = k.dtype
    num_tokens, num_query_heads, head_size = q.shape

    if sinks is not None:
        assert sinks.shape[0] == num_query_heads, "Sinks must be num_query_heads size"

    epilogue = output_gate is not None or output_rows is not None
    final_out = out
    if epilogue:
        assert not skip_reduce, "the output epilogue needs the reduce"
        assert out.dtype in (torch.bfloat16, torch.float16) and output_scale is None
        assert out.stride(-1) == 1 and out.stride(-2) == out.shape[-1]
        if output_gate is not None:
            assert output_gate.dim() == 2 and output_gate.stride(1) == 1
            assert output_gate.shape == (num_tokens, num_query_heads * out.shape[-1])
            assert output_gate.dtype == out.dtype
        if output_rows is not None:
            assert output_rows.dim() == 1 and output_rows.is_contiguous()
            assert out.shape[0] == output_rows.shape[0]
            # The kernels write every token; the epilogue picks the rows.
            out = torch.empty(
                (num_tokens, *out.shape[1:]), dtype=out.dtype, device=out.device
            )

    BLOCK_SCALES_SIZE = 16
    if q_dtype == torch.uint8:
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        head_size = head_size * 2

    if shuffled_kv_cache:
        SCALE_K_WIDTH = 4
        if kv_cache_dtype == torch.uint8:
            num_blocks, num_kv_heads, block_size, _ = k.shape
            K_WIDTH = 16
            SCALE_K = head_size // 16
            SCALE_K_WIDTH = (
                min(16, triton.next_power_of_2(SCALE_K)) if SCALE_K >= 4 else SCALE_K
            )
        else:
            # key_cache: num_blocks, num_kv_heads, head_size // x, block_size, x
            # value_cache: num_blocks, num_kv_heads, block_size // x, head_size, x
            num_blocks, num_kv_heads, _, block_size, K_WIDTH = k.shape
    else:
        # key_cache and value_cache: num_blocks, block_size, num_kv_heads, head_size
        num_blocks, block_size, num_kv_heads, _ = k.shape
        K_WIDTH = 16 if kv_cache_dtype == e4m3_dtype else 8
        SCALE_K_WIDTH = 4

    if shuffled_kv_cache:
        # A shuffled tile is exactly one page (the kernels index the block table
        # per tile and read TILE_SIZE * HEAD_SIZE_PADDED contiguous elements), so
        # TILE_SIZE is pinned to block_size and has to be a power of 2 for the
        # tl.arange over the tile. Non-shuffled pages have no such constraint:
        # there the block table is indexed per token, so a tile may straddle pages.
        assert block_size & (block_size - 1) == 0, (
            "Unified Attention with pre-shuffled KV cache requires a power-of-2 "
            f"page, got block_size={block_size}"
        )
        # The shuffled kernels lay V out with HEAD_SIZE, and a shuffled v does not
        # expose its head size as the last dim, so compare per-page element counts.
        if k.dtype == torch.uint8 and v.dtype != torch.uint8:
            # A8K4V8: k is [num_blocks, num_kv_heads, block_size, head_size // 2 + head_size // 16]
            v_page_numel = k.shape[1] * k.shape[2] * head_size
        else:
            v_page_numel = k.shape[1:].numel()
        assert v.shape[1:].numel() == v_page_numel, (
            "Unified Attention with pre-shuffled KV cache requires the value head "
            "size to match the query/key head size"
        )

    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads
    # only the plain layout exposes v's head size as its last dimension; a
    # shuffled value cache carries the vectorization width there instead.
    head_size_v = head_size if shuffled_kv_cache else v.shape[-1]
    assert out.shape[-1] == head_size_v, (
        f"out last dim must be the value head size {head_size_v}, "
        f"got {out.shape[-1]}"
    )

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    ALL_DECODE = int(max_seqlen_q) == 1
    # With equal_seqlen_q, seq i's queries are q rows [i, i + 1) * max_seqlen_q.
    # Trailing pad seqs may have no rows behind them so long as their seqused_k
    # is 0: the launches stop at the last seq with rows, and both kernels return
    # on an empty KV before they touch q or out.
    SPEC_DECODE = equal_seqlen_q and not ALL_DECODE
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads

    # build parameters
    params = _UAParams(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        softcap=softcap,
        causal=causal,
        sliding_window=SLIDING_WINDOW,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_tokens=num_tokens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        head_size_v=head_size_v,
        num_seqs=num_seqs,
        total_num_q_blocks=total_num_q_blocks,
        num_2d_prgms=num_2d_prgms,
        num_blocks=num_blocks,
        block_size=block_size,
        k_width=K_WIDTH,
        scale_k_width=SCALE_K_WIDTH,
        block_scales_size=BLOCK_SCALES_SIZE,
        q_dtype=q_dtype,
        kv_cache_dtype=kv_cache_dtype,
        v_cache_dtype=v.dtype,
        all_decode=ALL_DECODE,
        spec_decode=SPEC_DECODE,
        attn_work=int(attn_work or 0),
        shuffled_kv_cache=shuffled_kv_cache,
        use_alibi_slopes=use_alibi_slopes,
        use_qq_bias=use_qq_bias,
        num_sms=cu_count,
        target_num_prgms=target_num_prgms,
        sinks=sinks,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        q_scales=q_scales,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        output_scale=output_scale,
        skip_reduce=skip_reduce,
        force_fp8_pv_on_fp4_kv_cache=force_fp8_pv_on_fp4_kv_cache,
    )

    # if batch contains a prefill
    if use_2d_kernel(params, backend):
        # The gfx1250 Gluon 2d kernel only handles bf16/fp8 q+kv (with optional
        # sinks / output_scale / shuffled_kv_cache)
        use_gluon_2d = is_2d_gluon_available(params, backend)
        # the gate in the gfx1250 2d kernel's own epilogue (no row selection)
        fused_2d_gate = (
            output_gate is not None
            and use_gluon_2d
            and DEVICE_ARCH == "gfx1250"
            and _2d_gate_fusable(params, output_rows)
        )
        if use_gluon_2d:
            if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
                _unified_attention_2d_gfx1250(
                    params, output_gate=output_gate if fused_2d_gate else None
                )
            elif DEVICE_ARCH == "gfx950":
                _unified_attention_2d_gfx950(params)
            else:
                assert False, f"No gluon subwrapper for {DEVICE_ARCH}"
        else:
            _unified_attention_2d_triton(params)
        if epilogue and not fused_2d_gate:
            _output_epilogue(out, final_out, output_gate, output_rows)
    else:
        use_gluon_3d = is_3d_gluon_available(params, backend)
        split_params = params
        if use_gluon_3d and DEVICE_ARCH == "gfx950":
            split_params = _gfx950_split_params(params)
        config = get_unified_attention_config(
            "kv_split", split_params, backend="gluon" if use_gluon_3d else "triton"
        )
        NUM_SEGMENTS, CONFIG_TILE_SIZE = _split_launch_config(config)
        if (
            use_gluon_3d
            and IS_DEVICE_ARCH_GFX12
            and not (params.all_decode or params.spec_decode)
            and config.get("FILL_MIN_SEGMENTS")
        ):
            # the derived split bounds the grid at BLOCK_M 16, which on an uneven
            # batch leaves it a few segments; the work hint says when to widen it
            ref = get_unified_attention_config("attn_3d", params, backend="gluon")
            prgms_bq = _gfx12_2d_prgms(params, ref.get("PREFILL_REF_BLOCK_M"))
            fill = _gfx12_prefill_fill(params, *prgms_bq)
            if fill is not None:
                limit = triton.cdiv(params.max_seqlen_k, CONFIG_TILE_SIZE)
                for max_fill, min_segments in config["FILL_MIN_SEGMENTS"]:
                    if fill < max_fill:
                        NUM_SEGMENTS = max(NUM_SEGMENTS, min(min_segments, limit))
        segm_tiles = None
        if use_gluon_3d and DEVICE_ARCH == "gfx1250":
            NUM_SEGMENTS, segm_tiles = _gfx12_split_rebalance(params, CONFIG_TILE_SIZE)
        # The triton kernels read a shuffled tile as one contiguous page, so
        # there the tile is the page. The gfx950 gluon loader gathers per token
        # and keeps its tuned tile.
        if shuffled_kv_cache and not use_gluon_3d:
            TILE_SIZE = block_size
        else:
            TILE_SIZE = CONFIG_TILE_SIZE

        if NUM_SEGMENTS > 1:
            segm_output = torch.empty(
                q.shape[0],
                num_query_heads,
                NUM_SEGMENTS,
                triton.next_power_of_2(head_size_v),
                dtype=torch.float32,
                device=q.device,
            )
            segm_max = torch.empty(
                q.shape[0],
                num_query_heads,
                NUM_SEGMENTS,
                dtype=torch.float32,
                device=q.device,
            )
            segm_expsum = torch.empty(
                q.shape[0],
                num_query_heads,
                NUM_SEGMENTS,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            segm_output = out
            segm_max = out  # dummy ptr
            segm_expsum = out  # dummy ptr

        if use_gluon_3d:
            if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
                _unified_attention_3d_gfx1250(
                    params,
                    segm_output,
                    segm_max,
                    segm_expsum,
                    NUM_SEGMENTS,
                    TILE_SIZE,
                    segm_tiles=segm_tiles,
                )
            elif DEVICE_ARCH == "gfx950":
                _unified_attention_3d_gfx950(
                    params,
                    segm_output,
                    segm_max,
                    segm_expsum,
                    NUM_SEGMENTS,
                    TILE_SIZE,
                )
            else:
                assert False, f"No gluon subwrapper for {DEVICE_ARCH}"
        else:
            _unified_attention_3d_triton(
                params,
                segm_output,
                segm_max,
                segm_expsum,
                NUM_SEGMENTS,
                TILE_SIZE,
            )

        if NUM_SEGMENTS == 1:
            if epilogue:
                _output_epilogue(segm_output, final_out, output_gate, output_rows)
                return final_out
            return segm_output
        elif skip_reduce:
            # with a sliding window the segments carve only the window's tiles
            # (split_window_tiles), not [0, seq_len)
            return segm_output, segm_max, segm_expsum

        # Gluon reduce (one workgroup/token, in-wave segment merge); valid for all-decode with small split counts, else the Triton reduce_segments.
        # The output epilogue is the Triton reduce's.
        use_gluon_reduce = (
            is_reduce_gluon_available(params, NUM_SEGMENTS, backend)
            and backend == "gluon"
            and not epilogue
        )
        if use_gluon_reduce:
            if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
                _reduce_segments_gfx1250(
                    params,
                    segm_output,
                    segm_max,
                    segm_expsum,
                    NUM_SEGMENTS,
                    TILE_SIZE,
                    segm_tiles=segm_tiles,
                )
            else:
                assert False, f"No gluon subwrapper for {DEVICE_ARCH}"
        else:
            _reduce_segments_triton(
                params._replace(out=final_out),
                segm_output,
                segm_max,
                segm_expsum,
                NUM_SEGMENTS,
                TILE_SIZE,
                output_gate=output_gate,
                output_rows=output_rows,
                segm_tiles=segm_tiles,
            )
    return final_out


def _2d_gate_fusable(params: _UAParams, output_rows) -> bool:
    """Whether the gfx1250 2d kernel applies the output gate in its epilogue: every row
    (no row selection), a power-of-2 GQA group whose output columns split into
    128-column groups, and a context from _UA2D_FUSED_EPILOGUE_MIN_KV on."""
    nq = params.num_queries_per_kv
    return (
        output_rows is None
        and _UA2D_FUSED_EPILOGUE
        and int(params.max_seqlen_k) >= _UA2D_FUSED_EPILOGUE_MIN_KV
        and nq & (nq - 1) == 0
        and (nq * params.head_size_v) % 256 == 0
        and params.head_size == params.head_size_v
    )


def _output_epilogue(out, final_out, output_gate, output_rows):
    """The output epilogue of reduce_segments for an attention that ends without it:
    final_out[r] = out[rows[r]] * sigmoid(gate[rows[r]]), in one launch."""
    width = out.shape[-2] * out.shape[-1]
    num_rows = final_out.shape[0]
    if num_rows == 0:
        return
    block = min(1024, triton.next_power_of_2(width))
    attention_output_gate_rows[(num_rows, triton.cdiv(width, block))](
        final_out,
        out,
        output_gate,
        output_rows,
        out.stride(0),
        output_gate.stride(0) if output_gate is not None else 0,
        final_out.stride(0),
        WIDTH=width,
        BLOCK=block,
        num_warps=4,
        # the roundings of the production _gated_pointwise_kernel, which runs unfused
        enable_fp_fusion=False,
    )


def is_2d_gluon_available(params: _UAParams, backend: str):
    use_gluon = backend == "gluon" and _is_gluon_available()

    # arch-specific gates
    use_gluon_arch = False
    if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
        use_gluon_arch = (
            _unified_attention_kernel_2d_gfx1250 is not None
            and not params.softcap
            and not params.use_qq_bias
            and not params.use_alibi_slopes
            and (
                params.q_dtype == params.kv_cache_dtype
                # NVFP4 KV: fp8 Q (A8W4) or NVFP4 Q (A4W4), shuffled 128-token pages
                or (
                    params.kv_cache_dtype == torch.uint8
                    and params.q_dtype.itemsize == 1
                )
            )
            and (
                params.kv_cache_dtype != torch.uint8
                or (
                    params.shuffled_kv_cache
                    and (
                        params.block_size == 128
                        # A8K4V8 (fp8 V) also runs 64-token pages
                        or (params.block_size == 64 and params.v_cache_dtype != torch.uint8)
                    )
                )
            )
            and params.head_size_v == params.head_size
        )
    elif DEVICE_ARCH == "gfx950":
        use_gluon_arch = _gfx950_gluon_supported(params)

    return use_gluon and use_gluon_arch


def is_3d_gluon_available(params: _UAParams, backend: str):
    use_gluon = backend == "gluon" and _is_gluon_available()

    # arch-specific gates
    use_gluon_arch = False
    if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
        use_gluon_arch = (
            _unified_attention_kernel_3d_gfx1250 is not None
            and params.shuffled_kv_cache
        )
    elif DEVICE_ARCH == "gfx950":
        use_gluon_arch = _gfx950_gluon_supported(params)

    return use_gluon and use_gluon_arch


def is_reduce_gluon_available(params: _UAParams, NUM_SEGMENTS, backend: str):
    use_gluon = backend == "gluon" and _is_gluon_available()

    head_size_padded = triton.next_power_of_2(params.head_size)
    gluon_num_warps = 8 if params.num_query_heads % 8 == 0 else 4

    # arch-specific gates
    use_gluon_arch = False
    if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
        use_gluon_arch = (
            _reduce_segments_kernel_gfx1250 is not None
            and params.all_decode
            and NUM_SEGMENTS <= _GLUON_REDUCE_MAX_SEGMENTS
            and head_size_padded % 32 == 0
            and params.num_query_heads % gluon_num_warps == 0
            and params.head_size_v == params.head_size
        )

    return use_gluon and use_gluon_arch


def _unified_attention_2d_triton(params: _UAParams):
    if params.shuffled_kv_cache and (
        params.q_dtype == e4m3_dtype and params.kv_cache_dtype == e4m3_dtype
    ):
        assert (
            params.block_size >= 32
        ), "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"

    config = get_unified_attention_config("attn_2d", params, backend="triton")
    config["BLOCK_M"] = max(
        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)
    )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    assert config["BLOCK_Q"] >= 1
    if params.shuffled_kv_cache:
        config["TILE_SIZE"] = params.block_size
    if params.all_decode:
        total_num_q_blocks = params.num_seqs
    else:
        total_num_q_blocks = params.num_tokens // config["BLOCK_Q"] + params.num_seqs

    kernel_unified_attention_2d[
        (
            params.num_kv_heads,
            total_num_q_blocks,
        )
    ](
        output_ptr=params.out,
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        alibi_slopes_ptr=params.alibi_slopes,
        qq_bias_ptr=params.qq_bias,
        scale=params.softmax_scale,
        q_descale_ptr=params.q_descale,
        k_descale_ptr=params.k_descale,
        v_descale_ptr=params.v_descale,
        out_scale_ptr=params.output_scale,
        softcap=params.softcap,
        num_query_heads=params.num_query_heads,
        num_queries_per_kv=params.num_queries_per_kv,
        block_table_stride=params.block_table.stride(0),
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        qq_bias_stride_0=params.qq_bias.stride(0) if params.use_qq_bias else 0,
        BLOCK_SIZE=params.block_size,
        HEAD_SIZE=params.head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size),
        V_HEAD_SIZE=params.head_size_v,
        V_HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size_v),
        USE_ALIBI_SLOPES=params.use_alibi_slopes,
        USE_QQ_BIAS=params.use_qq_bias,
        USE_SOFTCAP=(params.softcap > 0),
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        query_start_len_ptr=params.cu_seqlens_q,
        num_seqs=params.num_seqs,
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        **config,
        enable_fp_fusion=True,
    )


def _unified_attention_3d_triton(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
):
    config = get_unified_attention_config("attn_3d", params, backend="triton")
    config["BLOCK_M"] = max(
        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)
    )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    assert config["BLOCK_Q"] >= 1

    if params.all_decode:
        total_num_q_blocks = params.num_seqs
    else:
        total_num_q_blocks = params.num_tokens // config["BLOCK_Q"] + params.num_seqs

    kernel_unified_attention_3d[
        (total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)
    ](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        alibi_slopes_ptr=params.alibi_slopes,
        qq_bias_ptr=params.qq_bias,
        scale=params.softmax_scale,
        q_descale_ptr=params.q_descale,
        k_descale_ptr=params.k_descale,
        v_descale_ptr=params.v_descale,
        out_scale_ptr=(
            params.output_scale
            if (params.output_scale is not None and NUM_SEGMENTS == 1)
            else None
        ),
        softcap=params.softcap,
        num_query_heads=params.num_query_heads,
        num_queries_per_kv=params.num_queries_per_kv,
        block_table_stride=params.block_table.stride(0),
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        qq_bias_stride_0=params.qq_bias.stride(0) if params.use_qq_bias else 0,
        BLOCK_SIZE=params.block_size,
        HEAD_SIZE=params.head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size),
        V_HEAD_SIZE=params.head_size_v,
        V_HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size_v),
        USE_ALIBI_SLOPES=params.use_alibi_slopes,
        USE_QQ_BIAS=params.use_qq_bias,
        USE_SOFTCAP=(params.softcap > 0),
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        query_start_len_ptr=params.cu_seqlens_q,
        num_seqs=params.num_seqs,
        ALL_DECODE=params.all_decode,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        IS_Q_FP8=(params.q_dtype == e4m3_dtype),
        IS_KV_FP8=(params.kv_cache_dtype == e4m3_dtype),
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        TILE_SIZE=TILE_SIZE,
        **config,
        enable_fp_fusion=True,
    )


def _reduce_segments_triton(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
    output_gate=None,
    output_rows=None,
    segm_tiles=None,
):
    head_size_padded = triton.next_power_of_2(params.head_size)
    config = get_unified_attention_config("reduce", params, backend="triton")
    num_rows = params.num_tokens if output_rows is None else output_rows.shape[0]
    if num_rows == 0:
        return
    v_head_size_padded = triton.next_power_of_2(params.head_size_v)
    num_d_blocks = triton.cdiv(
        v_head_size_padded, config.get("BLOCK_D") or v_head_size_padded
    )

    reduce_segments[(num_rows, params.num_query_heads, num_d_blocks)](
        output_ptr=params.out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=params.seqused_k,
        num_seqs=params.num_seqs,
        num_query_heads=params.num_query_heads,
        out_scale_ptr=params.output_scale,
        segm_tiles_ptr=segm_tiles,
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        block_table_stride=params.block_table.stride(0),
        HEAD_SIZE=params.head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        V_HEAD_SIZE=params.head_size_v,
        V_HEAD_SIZE_PADDED=triton.next_power_of_2(params.head_size_v),
        query_start_len_ptr=params.cu_seqlens_q,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        TILE_SIZE=TILE_SIZE,
        SLIDING_WINDOW=params.sliding_window,
        BLOCK_Q=None,
        gate_ptr=output_gate,
        gate_stride_0=output_gate.stride(0) if output_gate is not None else None,
        rows_ptr=output_rows,
        **config,
        enable_fp_fusion=True,
    )


"""
Below are arch-specific Gluon kernel wrappers.
unified_attention() picks one per kernel through the is_*_gluon_available() gates,
so the wrappers for a given kernel follows the signature below:
```
_unified_attention_2d_{arch}(
    params: _UAParams,
): ...

_unified_attention_3d_{arch}(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
): ...

_reduce_segments_{arch}(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
): ...
```
"""


def _unified_attention_2d_gfx1250(params: _UAParams, output_gate=None):
    """
    Internal wrapper for the gfx1250 gluon kernel.

    loop_variant:
        0=plain double buffered version,
        1=2-stage version,
        2=4-stage version
    """
    assert params.softcap == 0, "Softcap is not supported"

    config = get_unified_attention_config("attn_2d", params, backend="gluon")
    override = config.pop("FILL_OVERRIDE", None)
    if override and not (params.all_decode or params.spec_decode):
        # Over a long context, a batch the hint shows unevenly loaded runs faster on
        # a wider block and tile; an even one (a single long sequence) does not.
        ref = get_unified_attention_config("attn_3d", params, backend="gluon")
        fill = _gfx12_prefill_fill(
            params, *_gfx12_2d_prgms(params, ref.get("PREFILL_REF_BLOCK_M"))
        )
        if (
            fill is not None
            and fill < override["max_fill"]
            and params.max_seqlen_k >= override["min_kv"]
        ):
            config.update(override["config"])
    NUM_SEQS = params.num_seqs
    TILE_SIZE = config["TILE_SIZE"]
    BLOCK_M = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    num_warps = config["num_warps"]
    loop_variant = config["LOOP_VARIANT"]
    # Short queries (speculative-decode verify): size the block to the query rows
    # instead of padding them to the prefill BLOCK_M, one 16-row WMMA tile per warp.
    q_rows = triton.next_power_of_2(params.max_seqlen_q * params.num_queries_per_kv)
    if q_rows < BLOCK_M:
        BLOCK_M = max(q_rows, 16)
        num_warps = BLOCK_M // 16
        loop_variant = 0

    # Non-shuffled KV can't use TDM gather (KV layout), so a tile is one page
    if not params.shuffled_kv_cache or TILE_SIZE < params.block_size:
        TILE_SIZE = params.block_size
    num_kv_blocks = TILE_SIZE // params.block_size
    assert (
        num_kv_blocks & (num_kv_blocks - 1) == 0
    ), f"TILE_SIZE={TILE_SIZE} must be a power-of-2 multiple of PAGE_SIZE={params.block_size}"

    # the loop variants other than 0 mask at most twice at the end of the loop,
    # and need a tile wider than 32
    query_span = (BLOCK_M - 1) // params.num_queries_per_kv + 1
    if query_span > TILE_SIZE or TILE_SIZE <= 32:
        loop_variant = 0

    num_buffers = config["NUM_BUFFERS"]
    waves_per_eu = config["waves_per_eu"]
    kv_nvfp4 = params.kv_cache_dtype == torch.uint8
    if kv_nvfp4:
        num_buffers = 2
        if params.v_cache_dtype == torch.uint8:
            # an NVFP4 V (A8W4, BF16 PV) runs the double-buffered loop variant 0 with one 128-token page per tile
            TILE_SIZE = params.block_size
            loop_variant = 0
        elif loop_variant not in (0, 1, 2):
            # NVFP4 K with an fp8 V (A8K4V8): loop variants 0, 1 and 2, tiles of one or more pages
            loop_variant = 1
        # the NVFP4 path (packed K, scale layouts; the V upcast for an NVFP4 V) needs the whole VGPR file:
        # one wave per SIMD (2 is 28x slower for A8K4V8 prefill, ~20x for A8W4 decode)
        waves_per_eu = 1

    BLOCK_Q = BLOCK_M // params.num_queries_per_kv
    assert BLOCK_Q >= 1
    # spec decode launches the exact block count, so the kernel needs no search
    SEQLEN_Q = int(params.max_seqlen_q) if params.spec_decode else 1
    if params.all_decode:
        total_query_blocks = NUM_SEQS
    elif params.spec_decode:
        # only the seqs with q rows behind them; the trailing pad seqs of a
        # DP-padded batch have none (and seqused_k == 0)
        total_query_blocks = min(NUM_SEQS, params.num_tokens // SEQLEN_Q) * triton.cdiv(
            SEQLEN_Q, BLOCK_Q
        )
    else:
        total_query_blocks = params.num_tokens // BLOCK_Q + NUM_SEQS

    # buffer ops need the tensor to fit a 32-bit offset; gfx1250 loads through TDM
    MAX_INT32 = 2**31 - 1
    USE_STORE_BUFFER_OP = params.out.nelement() * params.out.element_size() <= MAX_INT32
    _unified_attention_kernel_2d_gfx1250[(params.num_kv_heads, total_query_blocks)](
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        output_ptr=params.out,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        query_start_len_ptr=params.cu_seqlens_q,
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        k_descale_ptr=params.k_descale,
        v_descale_ptr=params.v_descale,
        q_descale_ptr=params.q_descale,
        out_scale_ptr=params.output_scale,
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        num_blocks=params.num_blocks,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        block_table_stride=params.block_table.stride(0),
        num_seqs=NUM_SEQS,
        SCALE=params.softmax_scale,
        NUM_QUERY_HEADS=params.num_query_heads,
        NUM_KV_HEADS=params.num_kv_heads,
        BLOCK_SIZE=params.block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=params.head_size,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        ARCH_NAME=DEVICE_ARCH,
        waves_per_eu=waves_per_eu,
        USE_LOAD_BUFFER_OP=False,
        USE_STORE_BUFFER_OP=USE_STORE_BUFFER_OP,
        num_warps=num_warps,
        ALL_DECODE=params.all_decode,
        SPEC_DECODE=params.spec_decode,
        SEQLEN_Q=SEQLEN_Q,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        CAUSAL=params.causal,
        # useful for debugging when needed
        REMOVE_INDIRECT_ACCESS=False,
        NUM_BUFFERS=num_buffers,
        LOOP_VARIANT=loop_variant,
        Q_NVFP4=params.q_dtype == torch.uint8,
        KV_NVFP4=kv_nvfp4,
        V_NVFP4=params.v_cache_dtype == torch.uint8,
        query_scales_ptr=params.q_scales,
        query_scales_stride_0=(
            params.q_scales.stride(0) if params.q_scales is not None else 0
        ),
        query_scales_stride_1=(
            params.q_scales.stride(1) if params.q_scales is not None else 0
        ),
        gate_ptr=output_gate,
        gate_stride_0=output_gate.stride(0) if output_gate is not None else None,
        GATE=output_gate is not None,
        enable_fp_fusion=True,
    )


def _unified_attention_3d_gfx1250(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
    segm_tiles=None,
):
    # its split carves [0, seq_len), unlike reduce_segments for a sliding window;
    # use_2d_kernel keeps every sliding-window call on the 2d grid here
    assert params.sliding_window <= 0, "gfx1250 split-KV has no sliding-window carve"
    NUM_BLOCKS_GATHER_PER_TILE = 1
    QUERY_DTYPE = get_dtype_str(params.q_dtype)
    KV_CACHE_DTYPE = get_dtype_str(params.kv_cache_dtype)
    # A8K4V8: NVFP4 K, fp8 V
    V_CACHE_DTYPE = (
        get_dtype_str(params.v_cache_dtype)
        if params.v_cache_dtype != params.kv_cache_dtype
        else ""
    )
    config = get_unified_attention_config("attn_3d", params, backend="gluon")
    for key in _GFX12_ROUTING_KEYS:
        config.pop(key, None)
    config["BLOCK_M"] = max(
        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)
    )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    assert config["BLOCK_Q"] >= 1

    # spec decode launches the exact block count, so the kernel needs no search
    SEQLEN_Q = int(params.max_seqlen_q) if params.spec_decode else 1
    if params.all_decode:
        total_num_q_blocks = params.num_seqs
    elif params.spec_decode:
        # only the seqs with q rows behind them, as in the 2d launch
        total_num_q_blocks = min(
            params.num_seqs, params.num_tokens // SEQLEN_Q
        ) * triton.cdiv(SEQLEN_Q, config["BLOCK_Q"])
    else:
        total_num_q_blocks = params.num_tokens // config["BLOCK_Q"] + params.num_seqs

    _unified_attention_kernel_3d_gfx1250[
        (total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)
    ](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=params.q,
        query_scales_ptr=params.q_scales,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        alibi_slopes_ptr=params.alibi_slopes,
        qq_bias_ptr=params.qq_bias,
        q_scale_ptr=params.q_descale,
        k_scale_ptr=params.k_descale,
        v_scale_ptr=params.v_descale,
        out_scale_ptr=(
            params.output_scale
            if (params.output_scale is not None and NUM_SEGMENTS == 1)
            else None
        ),
        segm_tiles_ptr=segm_tiles,
        softcap=params.softcap,
        num_seqs=params.num_seqs,
        num_blocks=params.num_blocks,
        block_table_stride=params.block_table.stride(0),
        max_num_blocks_per_seq=params.block_table.shape[1],
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        query_scales_stride_0=(
            params.q_scales.stride(0) if params.q_scales is not None else 0
        ),
        query_scales_stride_1=(
            params.q_scales.stride(1) if params.q_scales is not None else 0
        ),
        qq_bias_stride_0=params.qq_bias.stride(0) if params.use_qq_bias else 0,
        BLOCK_SIZE=params.block_size,
        HEAD_SIZE=params.head_size,
        USE_ALIBI_SLOPES=params.use_alibi_slopes,
        USE_QQ_BIAS=params.use_qq_bias,
        USE_SOFTCAP=(params.softcap > 0),
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        query_start_len_ptr=params.cu_seqlens_q,
        SCALE=params.softmax_scale,
        NUM_QUERY_HEADS=params.num_query_heads,
        NUM_KV_HEADS=params.num_kv_heads,
        ALL_DECODE=params.all_decode,
        SPEC_DECODE=params.spec_decode,
        SEQLEN_Q=SEQLEN_Q,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        K_WIDTH=params.k_width,
        SCALE_K_WIDTH=params.scale_k_width,
        FP8_PV=params.force_fp8_pv_on_fp4_kv_cache
        and params.v_cache_dtype == torch.uint8,
        V_CACHE_DTYPE=V_CACHE_DTYPE,
        USE_STORE_TDM=True,
        # experiment knob: NVFP4 KV + BF16 PV processed in 2 head-dim halves
        HEAD_SIZE_SPLIT=(
            int(os.environ.get("AITER_UA_FP4_HEAD_SPLIT", "1"))
            if params.v_cache_dtype == torch.uint8 and not params.force_fp8_pv_on_fp4_kv_cache
            else 1
        ),
        WARP_SIZE=WARP_SIZE,
        NUM_BLOCKS_GATHER_PER_TILE=NUM_BLOCKS_GATHER_PER_TILE,
        QUERY_DTYPE=QUERY_DTYPE,
        KV_CACHE_DTYPE=KV_CACHE_DTYPE,
        BLOCK_SCALES_SIZE=params.block_scales_size,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
        TILE_SIZE=TILE_SIZE,
        **config,
        enable_fp_fusion=True,
    )


def _reduce_segments_gfx1250(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
    segm_tiles=None,
):
    head_size_padded = triton.next_power_of_2(params.head_size)
    gluon_num_warps = 8 if params.num_query_heads % 8 == 0 else 4
    config = get_unified_attention_config("reduce", params, backend="gluon")

    _reduce_segments_kernel_gfx1250[(params.num_tokens,)](
        output_ptr=params.out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=params.seqused_k,
        num_query_heads=params.num_query_heads,
        out_scale_ptr=params.output_scale,
        segm_tiles_ptr=segm_tiles,
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        H=params.num_query_heads,
        S=NUM_SEGMENTS,
        D=params.head_size,
        D_PAD=head_size_padded,
        TILE_SIZE=TILE_SIZE,
        IS_FP8_OUT=(params.out.dtype == e4m3_dtype),
        FP8_MIN=torch.finfo(e4m3_dtype).min,
        FP8_MAX=torch.finfo(e4m3_dtype).max,
        NUM_WARPS=gluon_num_warps,
        num_warps=gluon_num_warps,
        **config,
        enable_fp_fusion=True,
    )


def _unified_attention_gfx950(
    params: _UAParams,
    config: dict,
    BLOCK_M: int,
    TILE_SIZE: int,
    num_warps: int,
    NUM_SEGMENTS: int = 1,
    partial_acc=None,
    partial_m=None,
    partial_l=None,
):
    """Single launch site for the gfx950 gluon kernel.

    One kernel serves both grids: ALL_DECODE selects the decode program order
    and, when NUM_SEGMENTS > 1, a third grid axis over the KV split. A split
    launch writes fp32 partials instead of the output, which reduce_segments
    then merges. With a sliding window both carve only the window's tiles.

    With shuffled_kv_cache the caches must be in the layout shuffle_kv_cache
    produces (op_tests/triton_tests/attention/test_unified_attention.py):

    K = [num_blocks, num_kv_heads, head_size // W, block_size, W]
    V = [num_blocks, num_kv_heads, block_size // W, head_size, W],
      where
      W = 16 // element_size.

    The W run has to sit on each dot's reduction axis,
    so K groups head_size and V groups tokens. The triton backend reads the same
    layout off the cache shape.
    """
    BLOCK_Q = BLOCK_M // params.num_queries_per_kv
    assert BLOCK_Q >= 1

    if params.all_decode:
        total_query_blocks = params.num_seqs
        grid = (total_query_blocks, params.num_kv_heads)
    else:
        total_query_blocks = params.num_tokens // BLOCK_Q + params.num_seqs
        grid = (params.num_kv_heads, total_query_blocks)
    if NUM_SEGMENTS > 1:
        grid = grid + (NUM_SEGMENTS,)

    # k and v are both staged, twice over for a double buffer; drop to one
    # buffer rather than overrun LDS.
    NUM_BUFFERS = config["NUM_BUFFERS"]
    if 4 * TILE_SIZE * params.head_size * params.k.element_size() > _GFX950_LDS_BUDGET:
        NUM_BUFFERS = 1

    # buffer ops need the tensor to fit a 32-bit offset. k and v can be strided
    # views of one packed cache, so use the span of the view, not its size.
    MAX_INT32 = 2**31 - 1
    USE_LOAD_BUFFER_OP = (
        max(max_addressable_bytes(params.k), max_addressable_bytes(params.v))
        <= MAX_INT32
    )
    USE_STORE_BUFFER_OP = params.out.nelement() * params.out.element_size() <= MAX_INT32

    _unified_attention_kernel_gfx950[grid](
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=params.v,
        sink_ptr=params.sinks,
        output_ptr=params.out,
        block_tables_ptr=params.block_table,
        seq_lens_ptr=params.seqused_k,
        query_start_len_ptr=params.cu_seqlens_q,
        query_stride_0=params.q.stride(0),
        query_stride_1=params.q.stride(1),
        output_stride_0=params.out.stride(0),
        output_stride_1=params.out.stride(1),
        k_descale_ptr=params.k_descale,
        v_descale_ptr=params.v_descale,
        q_descale_ptr=params.q_descale,
        out_scale_ptr=params.output_scale,
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        num_blocks=params.num_blocks,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=params.v.stride(0),
        stride_v_cache_1=params.v.stride(1),
        stride_v_cache_2=params.v.stride(2),
        stride_v_cache_3=params.v.stride(3),
        block_table_stride=params.block_table.stride(0),
        num_seqs=params.num_seqs,
        SCALE=params.softmax_scale,
        SOFTCAP=params.softcap,
        USE_SOFTCAP=(params.softcap > 0),
        NUM_QUERY_HEADS=params.num_query_heads,
        NUM_KV_HEADS=params.num_kv_heads,
        BLOCK_SIZE=params.block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=params.head_size,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        MFMA_DIM=config["MFMA_DIM"],
        ARCH_NAME=DEVICE_ARCH,
        waves_per_eu=config["waves_per_eu"],
        USE_LOAD_BUFFER_OP=USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP=USE_STORE_BUFFER_OP,
        num_warps=num_warps,
        ALL_DECODE=params.all_decode,
        CAUSAL=params.causal,
        # useful for debugging when needed
        REMOVE_INDIRECT_ACCESS=False,
        NUM_BUFFERS=NUM_BUFFERS,
        SHUFFLED_KV_CACHE=params.shuffled_kv_cache,
        KV_SHUFFLE_WIDTH=params.k_width,
        NUM_SPLITS=NUM_SEGMENTS,
        partial_m_ptr=partial_m,
        partial_l_ptr=partial_l,
        partial_acc_ptr=partial_acc,
        enable_fp_fusion=True,
    )


def _unified_attention_2d_gfx950(params: _UAParams):
    """Prefill and mixed batches: one program per (kv head, query block)."""

    config = get_unified_attention_config("attn_2d", params, backend="gluon")
    BLOCK_M = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    _unified_attention_gfx950(
        params,
        config,
        BLOCK_M=BLOCK_M,
        TILE_SIZE=config["TILE_SIZE"],
        num_warps=config["num_warps"],
    )


def _unified_attention_3d_gfx950(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
):
    """Decode and split batches: one program per (query block, kv head[, KV split]).

    num_warps is not tuned directly here; BLOCK_M rows are split MFMA_DIM to a
    warp, so the warp count follows the query group the launch has to cover.
    """

    config = get_unified_attention_config("attn_3d", params, backend="gluon")
    MFMA_DIM = config["MFMA_DIM"]
    BLOCK_M = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    q_rows = triton.next_power_of_2(params.max_seqlen_q * params.num_queries_per_kv)
    if not params.all_decode and q_rows <= 128:
        # short queries: one block covers the sequence
        BLOCK_M = max(BLOCK_M, q_rows)
    # An unsplit launch that already leaves CUs idle can afford a wider query
    # block, which buys back the warps the missing split axis would have used.
    if NUM_SEGMENTS == 1 and params.num_2d_prgms <= 2 * params.num_sms:
        BLOCK_M = max(config.get("WIDE_BLOCK_M_MULT", 1) * MFMA_DIM, BLOCK_M)

    split = NUM_SEGMENTS > 1
    _unified_attention_gfx950(
        params,
        config,
        BLOCK_M=BLOCK_M,
        TILE_SIZE=TILE_SIZE,
        num_warps=BLOCK_M // MFMA_DIM,
        NUM_SEGMENTS=NUM_SEGMENTS,
        partial_acc=segm_output if split else None,
        partial_m=segm_max if split else None,
        partial_l=segm_expsum if split else None,
    )
