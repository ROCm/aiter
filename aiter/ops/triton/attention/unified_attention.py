# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import math
import json
import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    attention_output_gate_rows,
    gfx12_token_split_tiles,
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
    reduce_splits_heads,
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
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_reduce import (
        reduce_segments_split_gluon as _reduce_segments_split_kernel_gfx1250,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_reduce import (
        reduce_segments_split_oneshot_gluon as _reduce_segments_split_oneshot_kernel_gfx1250,
    )
except:  # noqa: E722
    _reduce_segments_kernel_gfx1250 = None
    _reduce_segments_split_kernel_gfx1250 = None
    _reduce_segments_split_oneshot_kernel_gfx1250 = None

# gfx950
try:
    from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
        _unified_attention_gluon_kernel as _unified_attention_kernel_gfx950,
    )
except:  # noqa: E722
    _unified_attention_kernel_gfx950 = None

# Max NUM_SEGMENTS the gluon reduce holds in-thread; larger split counts fall back to the Triton reduce_segments.
_GLUON_REDUCE_MAX_SEGMENTS = 8
# gfx1250, short queries (speculative-decode verify, q_len = 1 + num_spec): at most
# this many query tokens a sequence. Such a batch takes the 3D split-KV kernel while it
# has at most the attn_3d entry's MAX_QUERY_ROWS_PER_CU query rows (tokens * query
# heads) per CU; without one, while it splits each sequence's KV at least
# _GFX12_3D_MIN_SEGMENTS ways. A 2D program covers a whole GQA group of a sequence, so the 2D grid is
# sequences * KV heads programs; one walk over the longest context bounds it. The 3D
# kernel pays for its split in one-page tiles, partial outputs and a reduce. Measured
# on Qwen3.5 (16 q / 2 kv heads, d 256, 4-token verify, 28k-token mean context):
# 8 segments 2.2x faster, 4 segments 18-27% faster, 2 segments 1.5x slower.
_GFX12_3D_MAX_SEQLEN_Q = 16
_GFX12_3D_MIN_SEGMENTS = 4
# gfx1250 3D kernel, a speculative-decode verify (equal_seqlen_q) wider than one query
# block of the tuned attn_3d BLOCK_M, under the token split: one workgroup takes up to
# _GFX12_3D_QPACK_ROWS query rows (token x head) of a sequence, split over warps of 32
# rows (16 while a segment has at most 4 such workgroups), against one copy of each
# KV tile in LDS, where every block was a workgroup streaming the same KV. A sequence
# still spanning several workgroups has them on one XCD (one L2). A single sequence of
# at most 3 blocks keeps its blocks as workgroups (packed it measured slower) and only
# takes the XCD placement. Bitwise the same output (each warp runs a block's per-row
# instructions). Qwen3.6 A8K4V8 (16 q / 2 kv heads, d 256, page 128), 78k-145k-token
# contexts, token split MIN/MAX_TILES 16/128, us/call with the reduce, bs 1/3/6:
# Q8 27.4/61.1/120.7 -> 23.8/40.4/72.4, Q6 22.7/55.6/89.4 -> 20.9/39.6/70.5,
# Q16 44.6/99.3/189.8 -> 27.7/64.1/120.2 (Q4, one block: 23.3/35.7/64.3).
# A ragged verify (equal_seqlen_q=False) keeps its blocks: packing its short sequences'
# blocks with a long one's masked warps measured slower. Off: AITER_UA3D_QPACK=0.
_GFX12_3D_QPACK = os.environ.get("AITER_UA3D_QPACK", "1") == "1"
_GFX12_3D_QPACK_ROWS = 64
# gfx1250 3D decode (one query block a sequence): the hardware sends workgroups to the 8 XCDs
# round robin by dispatch id. With the sequences a multiple of 8, all segments of a sequence go
# to one XCD, and an XCD with long sequences can get more than its 128 resident programs.
# AITER_UA3D_XCD_ROTATE=1 rotates the sequences by the segment index, so that the segments of a
# sequence go to all XCDs. Each program does the same work: the output is bitwise the same.
_GFX12_3D_XCD_ROTATE = os.environ.get("AITER_UA3D_XCD_ROTATE", "0") == "1"
# gfx1250 3D kernel, a prefill chunk that the attn_3d entry routes to the 3D split: the
# entry's PREFILL_SPLIT_* keys size its workgroups and KV split (_gfx12_prefill_split)
# in place of the decode token split. Off: AITER_UA3D_PREFILL_SPLIT=0.
_GFX12_3D_PREFILL_SPLIT = os.environ.get("AITER_UA3D_PREFILL_SPLIT", "1") == "1"
# gfx1250 2D kernel, a prefill whose grid leaves CUs idle: split the KV range when the
# attn_2d entry has SPLIT_* keys (_gfx12_2d_split). Off: AITER_UA2D_PREFILL_SPLIT=0.
_GFX12_2D_PREFILL_SPLIT = os.environ.get("AITER_UA2D_PREFILL_SPLIT", "1") == "1"
# gfx1250 2D kernel, loop variant 2: fixed reference max (skips the row max, alpha and the accumulator rescale;
# rounds differently, so off by default). AITER_UA_2D_CONFIG='{"FM_FIXED_MAX": true, ...}' sets it per launch.
_UA2D_FIXED_MAX = os.environ.get("AITER_UA_2D_FIXED_MAX", "0") == "1"
# gfx1250 2D kernel, prefill of several sequences: dispatch the programs longest sequence first (AITER_UA_2D_LPT=1).
# The default order walks the q blocks of the batch backwards (last sequence first), so a long-context sequence
# early in the batch runs in the last wave while CUs idle. ua2d_seq_order (one small kernel a launch) ranks the
# sequences by seq_len and writes one (seq, q block, start, len) row per program; the kernel's per-program work is
# unchanged, so the output is bitwise the same. Off by default.
_UA2D_LPT = os.environ.get("AITER_UA_2D_LPT", "0") == "1"
# AITER_UA_FP4PT_NO_SPECIALIZE=1: the gfx1250 2D kernel takes the KV pool's page count (num_blocks) as a runtime,
# never-specialized i32 instead of a constexpr. num_blocks only bounds the TDM descriptors' rows (the loaders clamp
# the page ids to the sequence's own pages), but a constexpr keys the compile, and a served pool's count comes from
# the memory profile: it differs per DP rank and per restart, so every prefill shape a fresh server meets (BLOCK_M,
# NUM_SPLITS, GATE) recompiles for seconds and the on-disk cache never covers it. Any KV format (A8K4V4PT, A8K4V8).
_GFX12_UA_RUNTIME_NUM_BLOCKS = os.environ.get("AITER_UA_FP4PT_NO_SPECIALIZE", "0") == "1"
# Opt-in floor on the gfx1250 gluon 3d KV split (decode and spec-decode verify).
# The split is sized from the launch's program count, so a CUDA graph captured
# at a batch wider than the live sequences (DP padding) splits too few ways.
# A floor pays off on long contexts and costs on short ones, which the split
# cannot tell apart under a graph (max_seqlen_k is the model's max), so it is
# left to the deployment. Qwen3.5 d 256, 2 kv heads, 40k-190k-token contexts,
# 19-32 live of 24/32 captured: 64 is 5-23% faster than the heuristic's 16-32.
_GFX12_3D_SEGMENTS_FLOOR = int(os.environ.get("AITER_GFX12_MIN_SEGMENTS", "0"))
# Opt-in token split for the gfx1250 gluon 3d kernel (decode and spec-decode verify):
# every KV segment covers the same number of tiles, so a long sequence gets more
# segments instead of longer ones. The default cuts every sequence into the same
# number of segments, and with mixed context lengths the longest sequences' segments
# run long after the rest are done: Qwen3.5 (16 q / 2 kv heads, d 256, fp8, 4-token
# verify), 64 sequences of 4k-191k tokens: 5.5 TB/s, against 16.7 for 64 x 64k.
# The segment size is computed on the device each launch from the batch's total KV,
# for about _GFX12_TOKEN_SPLIT_PROGRAMS programs, so it follows the live batch under
# a CUDA graph too. The grid covers _GFX12_TOKEN_SPLIT_MAX_KV tokens at the minimum
# segment size, since a graph keeps its capture's grid; spare segments exit at once.
_GFX12_TOKEN_SPLIT = os.environ.get("AITER_GFX12_TOKEN_SPLIT", "0") == "1"
_GFX12_TOKEN_SPLIT_PROGRAMS = int(
    os.environ.get("AITER_GFX12_TOKEN_SPLIT_PROGRAMS", "2048")
)
_GFX12_TOKEN_SPLIT_MIN_TILES = int(
    os.environ.get("AITER_GFX12_TOKEN_SPLIT_MIN_TILES", "32")
)
_GFX12_TOKEN_SPLIT_MAX_TILES = int(
    os.environ.get("AITER_GFX12_TOKEN_SPLIT_MAX_TILES", "256")
)
_GFX12_TOKEN_SPLIT_MAX_KV = int(
    os.environ.get("AITER_GFX12_TOKEN_SPLIT_MAX_KV", "262144")
)
# A8K4V4PT (fp4pt V cache) takes half-size segments by default: its 3D loop issues more per KV tile (the per-token
# scale TDM load and wait, P * s, the fp4 V LDS loads), so at decode's low occupancy (one single-wave CTA per SIMD,
# a few hundred working CTAs) shortening each CTA's chain of tiles wins. Qwen3.5 16 / 2 heads, d 256, q 4, 4 x 60k
# tokens: MIN_TILES 8 vs 16 is 1.22x for fp4pt and 0.82x for A8K4V8 (2026-11-17), so every other format keeps
# _GFX12_TOKEN_SPLIT_MIN_TILES.
_GFX12_TOKEN_SPLIT_MIN_TILES_FP4PT = int(
    os.environ.get("AITER_GFX12_TOKEN_SPLIT_MIN_TILES_FP4PT", "8")
)


def _gfx12_token_split_min_tiles(v_cache_dtype=None) -> int:
    """The token split's minimum tiles per segment for a V cache dtype (None: the default)."""
    if v_cache_dtype is not None and _is_fp4pt_v(v_cache_dtype):
        return _GFX12_TOKEN_SPLIT_MIN_TILES_FP4PT
    return _GFX12_TOKEN_SPLIT_MIN_TILES
# Opt-in: compute the token split's tiles per segment in one Triton program instead
# of the nine torch ops (add, floor-divide, sum, multiply, add, floor-divide, clamp,
# cast) that run before every 3d launch -- bitwise the same int32, still recomputed
# from the live seqused_k on every CUDA-graph replay. 13 calls a Qwen3.5 MTP-3 decode
# step (10 layers + 3 draft passes): 117 kernels become 13.
_GFX12_TOKEN_SPLIT_FUSED = os.environ.get("AITER_GFX12_TOKEN_SPLIT_FUSED", "0") == "1"
# Opt-in (or unified_attention(split_reduce=True)): the gfx1250 decode / spec-decode
# reduce of a split wider than the in-wave gluon reduce takes (the token split's 128-256
# segments) runs reduce_segments_split_gluon instead of the Triton reduce_segments:
# one program per (query token, head), the sequence from the token without a search,
# only the live segments read, their partials by TDM through double-buffered LDS.
# Qwen3.5 c=1024 verify (340 tokens, 16 heads, d 256, 10k-150k-token contexts,
# 128-segment grid, ~12 live): 21 -> 7.7 us. Not bitwise the Triton reduce (another
# fp32 summation order): ~0.005% of the bf16 outputs one rounding step apart.
_GFX12_SPLIT_REDUCE = os.environ.get("AITER_GFX12_SPLIT_REDUCE", "0") == "1"
# gfx1250: the output gate inside the 2d kernel's epilogue (its gate tile TDM-loaded
# before the attention loop) from this longest context on; below it, and with
# AITER_UA2D_FUSED_EPILOGUE=0, the gate is the separate attention_output_gate_rows
# launch after the 2d kernel. Short prefills (478-8192 new tokens, no prefix) measured
# 2-6 us slower with the gate in the epilogue than with the separate launch.
_UA2D_FUSED_EPILOGUE = os.environ.get("AITER_UA2D_FUSED_EPILOGUE", "1") == "1"
_UA2D_FUSED_EPILOGUE_MIN_KV = int(
    os.environ.get("AITER_UA2D_FUSED_EPILOGUE_MIN_KV", "16384")
)
# (SB segments a step, NUM_BUF LDS buffers, num_warps) of reduce_segments_split_gluon
_GFX12_SPLIT_REDUCE_CONFIG = tuple(
    int(x)
    for x in os.environ.get("AITER_GFX12_SPLIT_REDUCE_CONFIG", "8,2,1").split(",")
)
# Opt-in (or unified_attention(split_reduce_oneshot=True)), with the split reduce:
# reduce_segments_split_oneshot_gluon in its place, bitwise the same output. Its loads
# (the first CHUNK segments' partials, the statistics, the output gate) are all in
# flight at once instead of a chain of dependent round trips.
_GFX12_SPLIT_REDUCE_ONESHOT = (
    os.environ.get("AITER_GFX12_SPLIT_REDUCE_ONESHOT", "0") == "1"
)
# (CHUNK segments a pass loads, SPEC of them before the context length, num_warps)
_GFX12_SPLIT_REDUCE_ONESHOT_CONFIG = tuple(
    int(x)
    for x in os.environ.get("AITER_GFX12_SPLIT_REDUCE_ONESHOT_CONFIG", "64,32,2").split(
        ","
    )
)


def gfx12_token_split_constants(v_cache_dtype=None):
    """None when the gfx1250 token split is off, else the (PROGRAMS, MIN_TILES,
    MAX_TILES) of its tiles per segment: clamp(cdiv(sum_i cdiv(seqused_k[i],
    tile_size) * num_kv_heads, PROGRAMS), MIN_TILES, MAX_TILES), int32. For a
    caller that computes it itself and passes it as `unified_attention`'s
    ``gfx12_token_split_tiles``: pass the V cache's dtype, since MIN_TILES
    depends on it (fp4pt V: _GFX12_TOKEN_SPLIT_MIN_TILES_FP4PT)."""
    if not (_GFX12_TOKEN_SPLIT and DEVICE_ARCH == "gfx1250"):
        return None
    return (
        _GFX12_TOKEN_SPLIT_PROGRAMS,
        _gfx12_token_split_min_tiles(v_cache_dtype),
        _GFX12_TOKEN_SPLIT_MAX_TILES,
    )


# attn_3d config keys that steer use_2d_kernel on gfx1250 rather than the launch:
# MAX_QUERY_ROWS_PER_CU sets the short-query bound above, and
# PREFILL_MAX_2D_PRGMS / PREFILL_MIN_KV / PREFILL_MIN_KV_PER_PRGM opt an entry
# into the 3D split for prefill over a long cached prefix, up to PREFILL_MAX_KV
# (see use_2d_kernel).
_GFX12_ROUTING_KEYS = (
    "MAX_QUERY_ROWS_PER_CU",
    "PREFILL_MAX_2D_PRGMS",
    "PREFILL_MIN_KV",
    "PREFILL_MIN_KV_PER_PRGM",
    "PREFILL_MAX_KV",
    "PREFILL_SPLIT_ROWS",
    "PREFILL_SPLIT_WARPS",
    "PREFILL_SPLIT_PROGRAMS",
    "PREFILL_SPLIT_MAX_SEGMENTS",
)

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64

_GLUON_SUPPORTED_ARCHS = ("gfx1250", "gfx950")

# A8K4V4PT: fp8 Q, NVFP4 K (as A8K4V8) and an "fp4pt" V, selected by passing the V cache as
# torch.float4_e2m1fn_x2 (never by a plain uint8 V, which means NVFP4 V). A V page row (one
# block and KV head) is [block_size * head_size / 2 B e2m1, shuffled exactly as NVFP4 V |
# block_size fp32 scales, one per token], viewed as [num_blocks, num_kv_heads, block_size,
# head_size // 2 + 4]: the value of token t, dim d is e2m1[t, d] * scale[t] (* v_descale).
# gfx1250 Gluon kernels only, 128-token pages: the 3D kernel for decode / spec-decode verify and
# for prefill over a long cached prefix (the attn_3d PREFILL_MAX_2D_PRGMS route), the 2D kernel
# for the rest of prefill (loop variants 0 / 1 / 2, tiles of one or two pages, the fused output
# gate). No sliding window.
FP4PT_V_DTYPE = torch.float4_e2m1fn_x2
_FP4PT_BLOCK_SIZE = 128
# The fp4pt PV variant (V_PT_MODE, the 2D and 3D kernels alike), by AITER_UA_FP4PT_PV_MODE:
#   "p" / 1: SCALE_ON_P, upcast V to bf16, scale P's columns in fp32, bf16 PV WMMA (the backup)
#   "fp8" / 2: PV_FP8FP4, P * s in fp8 x the e2m1 V in a scaled WMMA, no V upcast (the default)
# The default is PV_FP8FP4: the fewest VALU (no V upcast), 1/4 of the bf16 PV WMMAs, ~300 fewer
# VGPRs than SCALE_ON_P, and 1.11-1.39x A8K4V8's decode speed at the Qwen shapes (SCALE_ON_P:
# 0.88-1.28x); its P is fp8, as A8K4V8's. (Mode 0, scaling the upcast V rows, was dropped.)
_FP4PT_PV_MODES = {"p": 1, "fp8": 2, "1": 1, "2": 2}
_FP4PT_PV_MODE_DEFAULT = "fp8"


def _fp4pt_pv_mode() -> int:
    mode = os.environ.get("AITER_UA_FP4PT_PV_MODE", _FP4PT_PV_MODE_DEFAULT).lower()
    assert (
        mode in _FP4PT_PV_MODES
    ), f"AITER_UA_FP4PT_PV_MODE={mode!r}: one of {sorted(_FP4PT_PV_MODES)}"
    return _FP4PT_PV_MODES[mode]


def _is_fp4pt_v(dtype) -> bool:
    return dtype == FP4PT_V_DTYPE


# AITER_UA_FP4PT_LOG=1: one line per process the first time each fp4pt route (2d / 3d, PV mode) launches
_FP4PT_LOGGED = set()


def _fp4pt_log(route: str, mode: int, **info):
    if os.environ.get("AITER_UA_FP4PT_LOG", "0") != "1" or (route, mode) in _FP4PT_LOGGED:
        return
    _FP4PT_LOGGED.add((route, mode))
    print(f"[aiter UA fp4pt] pid {os.getpid()} route {route} V_PT_MODE={mode} {info}", flush=True)


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
    if params.all_decode or params.max_seqlen_q * params.num_queries_per_kv <= 128:
        return params
    config = get_unified_attention_config("attn_3d", params, backend="gluon")
    block_m = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    block_q = block_m // params.num_queries_per_kv
    num_q_blocks = params.num_tokens // block_q + params.num_seqs
    return params._replace(num_2d_prgms=num_q_blocks * params.num_kv_heads)


def _gfx12_split_params(params: _UAParams):
    """Size the split by the exact 3d grid of a spec-decode verify launch.

    `num_2d_prgms` bounds the query blocks at the default BLOCK_Q plus one per
    sequence, the search grid's slack. A spec-decode launch has exactly one row
    of blocks per sequence at the 3d config's BLOCK_Q (one block for a 4-token
    verify of a GQA-8 group), so the bound is 3x the grid and the split it
    yields too narrow: a CUDA graph captured at 96 sequences split 2 ways, which
    is slower than the 2d kernel, where 8 ways is 1.3-2.1x faster than 2d at
    48-80 live sequences. An entry with a tuned MAX_QUERY_ROWS_PER_CU keeps the
    split it was tuned with.
    """
    if not params.spec_decode:
        return params
    config = get_unified_attention_config("attn_3d", params, backend="gluon")
    if "MAX_QUERY_ROWS_PER_CU" in config:
        return params
    block_m = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    block_q = block_m // params.num_queries_per_kv
    num_q_blocks = min(
        params.num_seqs, params.num_tokens // params.max_seqlen_q
    ) * triton.cdiv(params.max_seqlen_q, block_q)
    return params._replace(num_2d_prgms=num_q_blocks * params.num_kv_heads)


def _gfx12_token_split_tiles_torch(
    seqused_k: torch.Tensor, num_kv_heads: int, tile_size: int, min_tiles: int | None = None
) -> torch.Tensor:
    """Tiles per segment as plain torch ops (nine kernels): [1] int32."""
    if min_tiles is None:
        min_tiles = _GFX12_TOKEN_SPLIT_MIN_TILES
    tiles = (seqused_k + (tile_size - 1)) // tile_size
    want = tiles.sum() * num_kv_heads
    want = (want + (_GFX12_TOKEN_SPLIT_PROGRAMS - 1)) // _GFX12_TOKEN_SPLIT_PROGRAMS
    return (
        want.clamp(min_tiles, _GFX12_TOKEN_SPLIT_MAX_TILES)
        .to(torch.int32)
        .reshape(1)
    )


def _gfx12_token_split_tiles_fused(
    seqused_k: torch.Tensor, num_kv_heads: int, tile_size: int, min_tiles: int | None = None
) -> torch.Tensor:
    """`_gfx12_token_split_tiles_torch` in one single-program Triton launch.

    Bitwise the same [1] int32. No host read of device data and a fixed grid, so
    it captures into a CUDA graph and recomputes from the live seqused_k on replay.
    """
    if min_tiles is None:
        min_tiles = _GFX12_TOKEN_SPLIT_MIN_TILES
    if seqused_k.dim() != 1:
        return _gfx12_token_split_tiles_torch(seqused_k, num_kv_heads, tile_size, min_tiles)
    segm_tiles = torch.empty(1, dtype=torch.int32, device=seqused_k.device)
    num_seqs = seqused_k.shape[0]
    gfx12_token_split_tiles[(1,)](
        seqused_k,
        segm_tiles,
        num_seqs,
        seqused_k.stride(0),
        num_kv_heads,
        TILE_SIZE=tile_size,
        PROGRAMS=_GFX12_TOKEN_SPLIT_PROGRAMS,
        MIN_TILES=min_tiles,
        MAX_TILES=_GFX12_TOKEN_SPLIT_MAX_TILES,
        BLOCK=min(1024, max(64, triton.next_power_of_2(num_seqs))),
        num_warps=4,
    )
    return segm_tiles


def _gfx12_token_split(params: _UAParams, tile_size: int, precomputed=None):
    """The token split's grid width (NUM_SEGMENTS) and its tiles per segment.

    Tiles per segment is a device tensor: the batch's KV tiles over every KV head,
    over _GFX12_TOKEN_SPLIT_PROGRAMS, clamped to [MIN_TILES, MAX_TILES]. Device ops
    only (nine torch ops, or one Triton program under AITER_GFX12_TOKEN_SPLIT_FUSED=1),
    so a CUDA graph recomputes it from the live context lengths on replay.

    ``precomputed``: the caller's ``(segm_tiles, tile_size, num_kv_heads)``
    (`unified_attention`'s ``gfx12_token_split_tiles``), used when it was computed
    for this launch's tile size and KV heads; otherwise it is computed here. The
    minimum tiles per segment, and so the grid width, depend on the V cache's dtype
    (_gfx12_token_split_min_tiles); a caller precomputing the tiles has to use
    gfx12_token_split_constants(<the V dtype>) to match.
    """
    min_tiles = _gfx12_token_split_min_tiles(params.v_cache_dtype)
    if precomputed is not None:
        tiles, at_tile, at_heads = precomputed
        if at_tile == tile_size and at_heads == params.num_kv_heads:
            kv_cover = max(int(params.max_seqlen_k), _GFX12_TOKEN_SPLIT_MAX_KV)
            num_segments = triton.next_power_of_2(
                triton.cdiv(kv_cover, min_tiles * tile_size)
            )
            return num_segments, tiles
    if _GFX12_TOKEN_SPLIT_FUSED:
        segm_tiles = _gfx12_token_split_tiles_fused(
            params.seqused_k, params.num_kv_heads, tile_size, min_tiles
        )
    else:
        segm_tiles = _gfx12_token_split_tiles_torch(
            params.seqused_k, params.num_kv_heads, tile_size, min_tiles
        )
    kv_cover = max(int(params.max_seqlen_k), _GFX12_TOKEN_SPLIT_MAX_KV)
    num_segments = triton.next_power_of_2(
        triton.cdiv(kv_cover, min_tiles * tile_size)
    )
    return num_segments, segm_tiles


def _gfx950_split_grid(params: _UAParams):
    """Decode, and batches too small to fill the GPU, take the split grid."""
    if params.all_decode:
        return True
    if params.sliding_window > 0:
        return False
    # unsplit, the 2d grid is faster
    config = get_unified_attention_config(
        "kv_split", _gfx950_split_params(params), backend="gluon"
    )
    return config["NUM_SEGMENTS"] > 1


def _gfx12_2d_prgms(params: _UAParams):
    """Programs the gfx1250 2D kernel would launch: KV heads x query blocks."""
    config = get_unified_attention_config("attn_2d", params, backend="gluon")
    block_m = max(config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv))
    block_q = block_m // params.num_queries_per_kv
    return params.num_kv_heads * (params.num_tokens // block_q + params.num_seqs)


def _gfx12_prefill_2d(params: _UAParams, route: dict):
    """With few KV heads (TP) a prefill chunk can launch too few 2D programs to
    fill the GPU, each streaming the whole prefix; the 3D kernel splits the KV
    range instead. It pays while each of those programs has a long run of keys
    to split, i.e. the prefix dominates the chunk."""
    max_prgms = route.get("PREFILL_MAX_2D_PRGMS", 0)
    if max_prgms == 0 or params.max_seqlen_k >= route.get("PREFILL_MAX_KV", math.inf):
        return True
    prgms = _gfx12_2d_prgms(params)
    min_kv = max(
        route.get("PREFILL_MIN_KV", 0),
        route.get("PREFILL_MIN_KV_PER_PRGM", 0) * prgms,
    )
    return prgms >= max_prgms or params.max_seqlen_k < min_kv


def _gfx12_3d_prefill(params: _UAParams, route: dict):
    """Whether a gfx1250 3D launch is a prefill chunk that takes the attn_3d entry's
    PREFILL_SPLIT_* split instead of the decode token split."""
    return (
        _GFX12_3D_PREFILL_SPLIT
        and _GFX12_TOKEN_SPLIT
        and "PREFILL_SPLIT_ROWS" in route
        and not params.all_decode
        and not params.spec_decode
        and params.max_seqlen_q > _GFX12_3D_MAX_SEQLEN_Q
        # PV_WARPS_M: NVFP4 V lays its scales out over the warps of the head dim
        and params.v_cache_dtype != torch.uint8
    )


def _gfx12_prefill_split(params: _UAParams, route: dict, tile_size: int):
    """(NUM_SEGMENTS, segm_tiles) of a prefill chunk. The token split sizes its
    segments for one query block a sequence; a chunk has many, so it would launch
    thousands of small workgroups and reduce up to 128 segments. Here: the power of 2
    of segments that brings the grid of PREFILL_SPLIT_ROWS-row query blocks nearest
    PREFILL_SPLIT_PROGRAMS workgroups, each an even share of the longest context."""
    block_q = route["PREFILL_SPLIT_ROWS"] // params.num_queries_per_kv
    q_blocks = params.num_tokens // block_q + params.num_seqs
    want = route["PREFILL_SPLIT_PROGRAMS"] / (q_blocks * params.num_kv_heads)
    tiles = triton.cdiv(int(params.max_seqlen_k), tile_size)
    segments = 2 ** max(1, round(math.log2(max(want, 1.0))))
    segments = min(
        segments,
        route["PREFILL_SPLIT_MAX_SEGMENTS"],
        triton.next_power_of_2(tiles),
    )
    # cdiv(sum of the contexts' tiles, segments) on the device: covers the longest
    # context, and a Triton launch keeps the call replayable (AITER_UA_FAST_LAUNCH)
    segm_tiles = torch.empty(1, dtype=torch.int32, device=params.q.device)
    gfx12_token_split_tiles[(1,)](
        params.seqused_k,
        segm_tiles,
        params.num_seqs,
        params.seqused_k.stride(0),
        params.num_kv_heads,
        TILE_SIZE=tile_size,
        PROGRAMS=segments * params.num_kv_heads,
        MIN_TILES=1,
        MAX_TILES=1 << 30,
        BLOCK=min(1024, max(64, triton.next_power_of_2(params.num_seqs))),
        num_warps=4,
    )
    return segments, segm_tiles


def use_2d_kernel(params: _UAParams, backend: str = "triton"):
    # if IS_DEVICE_ARCH_GFX12, use 3D for decode, and for short (spec-decode verify)
    # queries while their rows are too few to fill the 2D grid (a tuned rows cap)
    # or the KV split is wide enough to pay for itself; 2D otherwise, unless the
    # attn_3d entry opts prefill over a long cached prefix into the 3D split.
    if IS_DEVICE_ARCH_GFX12:
        if params.sliding_window > 0:
            return True
        if params.all_decode:
            return False
        if (
            _GFX12_TOKEN_SPLIT
            and params.max_seqlen_q <= _GFX12_3D_MAX_SEQLEN_Q
            and is_3d_gluon_available(params, backend)
        ):
            # the token split keeps the 3d grid full at any batch, where 2d runs
            # one program per sequence over its whole context
            return False
        route = get_unified_attention_config("attn_3d", params, backend="gluon")
        if params.max_seqlen_q > _GFX12_3D_MAX_SEQLEN_Q:
            return _gfx12_prefill_2d(params, route)
        rows_cap = route.get("MAX_QUERY_ROWS_PER_CU")
        if rows_cap is not None:
            return (
                params.num_tokens * params.num_query_heads > rows_cap * params.num_sms
            )
        gluon_3d = is_3d_gluon_available(params, backend)
        if gluon_3d and params.spec_decode:
            # The exact grid: 2d runs one program per query block and KV head, and
            # past one per CU it fills the GPU and the split only adds traffic
            # (192 sequences: 2d 25-35% faster than a 4-way split).
            return _gfx12_split_params(params).num_2d_prgms > params.num_sms
        # the split the 3d launch would make
        split = get_unified_attention_config(
            "kv_split", params, backend="gluon" if gluon_3d else "triton"
        )
        return split["NUM_SEGMENTS"] < _GFX12_3D_MIN_SEGMENTS

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
    # NVFP4 (uint8 e2m1) KV cache only: run PV as an FP8 WMMA, with P and the upcast,
    # scaled V in e4m3, instead of a BF16 WMMA. Same for A8W4 and A4W4; ignored otherwise.
    force_fp8_pv_on_fp4_kv_cache: bool = False,
    # gfx1250 token split only (AITER_GFX12_TOKEN_SPLIT=1): (segm_tiles, tile_size,
    # num_kv_heads), the [1] int32 tiles per segment the caller computed from this
    # call's seqused_k (see gfx12_token_split_constants), so that it is not
    # computed here. Ignored unless tile_size and num_kv_heads are this launch's.
    gfx12_token_split_tiles=None,
    # backend
    backend: str | None = None,  # "triton" | "gluon"
    # Epilogue of the Qwen3-Next / Qwen3.5 attention: out * sigmoid(output_gate),
    # output_gate [num_tokens, num_query_heads * head_size_v] (row-strided, e.g. a
    # slice of the QKV-gate projection), bitwise the unfused sigmoid-mul. With
    # output_rows ([R] int32/int64), out has R rows: out[r] is query token
    # output_rows[r] (every token still attends and its KV is read as usual).
    output_gate: torch.Tensor | None = None,
    output_rows: torch.Tensor | None = None,
    # gfx1250: the split-KV reduce of decode / spec-decode verify (equal_seqlen_q)
    # through reduce_segments_split_gluon; None follows AITER_GFX12_SPLIT_REDUCE.
    split_reduce: bool | None = None,
    # with the split reduce: reduce_segments_split_oneshot_gluon (bitwise the same
    # output, its loads in one round trip); None follows AITER_GFX12_SPLIT_REDUCE_ONESHOT.
    split_reduce_oneshot: bool | None = None,
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
        if k.dtype == torch.uint8 and _is_fp4pt_v(v.dtype):
            # A8K4V4PT: v is [num_blocks, num_kv_heads, block_size, head_size // 2 + 4] bytes
            v_page_numel = k.shape[1] * k.shape[2] * (head_size // 2 + 4)
        elif k.dtype == torch.uint8 and v.dtype != torch.uint8:
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

    if _is_fp4pt_v(params.v_cache_dtype):
        _check_fp4pt(params, backend)

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
            if DEVICE_ARCH == "gfx1250":
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
        elif use_gluon_3d and DEVICE_ARCH == "gfx1250":
            split_params = _gfx12_split_params(params)
        config = get_unified_attention_config(
            "kv_split", split_params, backend="gluon" if use_gluon_3d else "triton"
        )
        NUM_SEGMENTS, CONFIG_TILE_SIZE = _split_launch_config(config)
        if (
            _GFX12_3D_SEGMENTS_FLOOR > 1
            and use_gluon_3d
            and DEVICE_ARCH == "gfx1250"
            and NUM_SEGMENTS < _GFX12_3D_SEGMENTS_FLOOR
        ):
            # never more segments than KV tiles in the longest context
            tiles = triton.cdiv(int(max_seqlen_k), CONFIG_TILE_SIZE)
            NUM_SEGMENTS = max(
                NUM_SEGMENTS,
                triton.next_power_of_2(min(_GFX12_3D_SEGMENTS_FLOOR, tiles)),
            )
        segm_tiles = None
        if _GFX12_TOKEN_SPLIT and use_gluon_3d and DEVICE_ARCH == "gfx1250":
            route = get_unified_attention_config("attn_3d", params, backend="gluon")
            if _gfx12_3d_prefill(params, route):
                NUM_SEGMENTS, segm_tiles = _gfx12_prefill_split(
                    params, route, CONFIG_TILE_SIZE
                )
            else:
                NUM_SEGMENTS, segm_tiles = _gfx12_token_split(
                    params, CONFIG_TILE_SIZE, gfx12_token_split_tiles
                )
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
            if DEVICE_ARCH == "gfx1250":
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
            return segm_output, segm_max, segm_expsum

        # Gluon reduce (one workgroup/token, in-wave segment merge); valid for all-decode with small split counts, else the Triton reduce_segments.
        # The output epilogue is the Triton reduce's.
        use_gluon_reduce = (
            is_reduce_gluon_available(params, NUM_SEGMENTS, backend)
            and backend == "gluon"
            and not epilogue
        )
        if split_reduce is None:
            split_reduce = _GFX12_SPLIT_REDUCE
        if split_reduce_oneshot is None:
            split_reduce_oneshot = _GFX12_SPLIT_REDUCE_ONESHOT
        if (
            split_reduce
            and not use_gluon_reduce
            and is_reduce_split_gluon_available(params, NUM_SEGMENTS, backend)
        ):
            _reduce_segments_split_gfx1250(
                params._replace(out=final_out),
                segm_output,
                segm_max,
                segm_expsum,
                NUM_SEGMENTS,
                TILE_SIZE,
                segm_tiles=segm_tiles,
                output_gate=output_gate,
                output_rows=output_rows,
                oneshot=split_reduce_oneshot,
            )
        elif use_gluon_reduce:
            if DEVICE_ARCH == "gfx1250":
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
                segm_tiles=segm_tiles,
                output_gate=output_gate,
                output_rows=output_rows,
            )
    return final_out


def _check_fp4pt(params: _UAParams, backend: str):
    """A8K4V4PT (fp4pt V) runs only on the gfx1250 Gluon 2D / 3D kernels: reject what would route elsewhere."""
    if params.sliding_window > 0:
        raise NotImplementedError("A8K4V4PT (fp4pt V cache) has no sliding window")
    if not (
        IS_DEVICE_ARCH_GFX12
        and is_3d_gluon_available(params, backend)
        and is_2d_gluon_available(params, backend)
    ):
        raise NotImplementedError(
            "A8K4V4PT (fp4pt V cache) runs only on the gfx1250 Gluon kernels "
            "(backend='gluon', shuffled_kv_cache=True, fp8 Q, NVFP4 K)"
        )
    assert params.kv_cache_dtype == torch.uint8 and params.q_dtype == e4m3_dtype, (
        "A8K4V4PT needs an NVFP4 (uint8) K cache and an fp8 Q, got "
        f"K {params.kv_cache_dtype}, Q {params.q_dtype}"
    )
    assert (
        params.block_size == _FP4PT_BLOCK_SIZE
    ), f"A8K4V4PT needs {_FP4PT_BLOCK_SIZE}-token pages, got block_size={params.block_size}"
    assert not (params.softcap or params.use_alibi_slopes or params.use_qq_bias)
    assert params.head_size_v == params.head_size
    v = params.v
    assert v.dim() == 4 and v.shape[2] == params.block_size, (
        "A8K4V4PT v must be [num_blocks, num_kv_heads, block_size, head_size // 2 + 4], "
        f"got {tuple(v.shape)}"
    )
    # the kernel's TDM descriptors address a (block, kv head) row by stride(1) alone
    assert v.stride(3) == 1 and v.stride(2) == v.shape[3]
    assert v.stride(1) == v.shape[2] * v.shape[3] and v.stride(0) == v.shape[
        1
    ] * v.stride(1)


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
    if DEVICE_ARCH == "gfx1250":
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
    if DEVICE_ARCH == "gfx1250":
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
    if DEVICE_ARCH == "gfx1250":
        use_gluon_arch = (
            _reduce_segments_kernel_gfx1250 is not None
            and params.all_decode
            and NUM_SEGMENTS <= _GLUON_REDUCE_MAX_SEGMENTS
            and head_size_padded % 32 == 0
            and params.num_query_heads % gluon_num_warps == 0
            and params.head_size_v == params.head_size
        )

    return use_gluon and use_gluon_arch


def is_reduce_split_gluon_available(params: _UAParams, NUM_SEGMENTS, backend: str):
    """reduce_segments_split_gluon: gfx1250, every sequence the same query length
    (decode or spec-decode verify), head size = value head size, a power of 2."""
    _, _, num_warps = _GFX12_SPLIT_REDUCE_CONFIG
    d = params.head_size_v
    return (
        DEVICE_ARCH == "gfx1250"
        and backend == "gluon"
        and _reduce_segments_split_kernel_gfx1250 is not None
        and (params.all_decode or params.spec_decode)
        and d == params.head_size
        and d & (d - 1) == 0
        and d % (32 * num_warps) == 0
        and triton.next_power_of_2(NUM_SEGMENTS) == NUM_SEGMENTS
    )


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
    segm_tiles=None,
    output_gate=None,
    output_rows=None,
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
        BLOCK_Q=None,
        gate_ptr=output_gate,
        gate_stride_0=output_gate.stride(0) if output_gate is not None else 0,
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


def _2d_full_causal_prefill(params: _UAParams) -> bool:
    """Causal prefill with full attention (no window, no sinks), the scope of the fixed-max loop."""
    return (
        not params.all_decode
        and not params.spec_decode
        and params.causal
        and params.sliding_window == 0
        and params.sinks is None
    )


@triton.jit
def ua2d_seq_order(
    query_start_len_ptr,  # [num_seqs + 1]
    seq_lens_ptr,  # [num_seqs]
    order_ptr,  # out, int32 [4 + 4 * q blocks]
    num_seqs,
    BLOCK_Q: tl.constexpr,
    N: tl.constexpr,  # >= num_seqs, power of 2
    CHUNK: tl.constexpr,
):
    """Longest-first program order of a prefill for the gfx1250 2D kernel (SEQ_ORDER). Program i (one per
    sequence) ranks sequence i by seq_len (descending; ties: batch order) and writes the rows of its q blocks,
    heaviest (last) first, from its first virtual block on: row v = (seq, q block, query start, query len).
    order[0] = the number of q blocks."""
    i = tl.program_id(0)
    j = tl.arange(0, N)
    valid = j < num_seqs
    start = tl.load(query_start_len_ptr + j, mask=valid, other=0)
    stop = tl.load(query_start_len_ptr + j + 1, mask=valid, other=0)
    nblk = tl.where(valid, (stop - start + BLOCK_Q - 1) // BLOCK_Q, 0)
    w = tl.load(seq_lens_ptr + j, mask=valid, other=0)
    w_i = tl.load(seq_lens_ptr + i)
    before = ((w > w_i) | ((w == w_i) & (j < i))) & valid
    vstart = tl.sum(tl.where(before, nblk, 0), axis=0)
    s_i = tl.load(query_start_len_ptr + i)
    len_i = tl.load(query_start_len_ptr + i + 1) - s_i
    nb_i = (len_i + BLOCK_Q - 1) // BLOCK_Q
    if i == 0:
        tl.store(order_ptr, tl.sum(nblk, axis=0))
    for c in range(0, nb_i, CHUNK):
        b = c + tl.arange(0, CHUNK)
        m = b < nb_i
        row = order_ptr + 4 + (vstart + b) * 4
        tl.store(row, tl.zeros([CHUNK], tl.int32) + i, mask=m)
        tl.store(row + 1, nb_i - 1 - b, mask=m)
        tl.store(row + 2, tl.zeros([CHUNK], tl.int32) + s_i, mask=m)
        tl.store(row + 3, tl.zeros([CHUNK], tl.int32) + len_i, mask=m)


def _gfx12_2d_seq_order(params: _UAParams, block_q: int, q_blocks: int):
    """Device-side longest-first program order of a multi-sequence prefill (AITER_UA_2D_LPT), or None."""
    if not _UA2D_LPT or params.all_decode or params.spec_decode or params.num_seqs < 2:
        return None
    order = torch.empty(4 + 4 * q_blocks, dtype=torch.int32, device=params.q.device)
    ua2d_seq_order[(params.num_seqs,)](
        params.cu_seqlens_q,
        params.seqused_k,
        order,
        params.num_seqs,
        BLOCK_Q=block_q,
        N=max(32, triton.next_power_of_2(params.num_seqs)),
        CHUNK=256,
        num_warps=4,
    )
    return order


def _gfx12_2d_split(params: _UAParams, config: dict):
    """(KV splits, config) of a gfx1250 2D prefill launch. An attn_2d entry with
    SPLIT_* keys splits a prefill over a context of at least SPLIT_MIN_KV into the
    largest power of 2 of KV ranges that keeps the grid of SPLIT_BLOCK_M-row query
    blocks within SPLIT_PROGRAMS workgroups (at most SPLIT_MAX, at least
    SPLIT_MIN_TILES tiles a range), and launches the SPLIT_* shape; one split and the
    entry's own shape otherwise."""
    target = config.get("SPLIT_PROGRAMS")
    if (
        not (_GFX12_2D_PREFILL_SPLIT and target)
        or params.all_decode
        or params.spec_decode
        or params.sliding_window > 0
        or int(params.max_seqlen_k) < config["SPLIT_MIN_KV"]
        or params.head_size != params.head_size_v
        # reduce_splits_heads: power-of-2 head blocks and head size
        or params.num_query_heads & (params.num_query_heads - 1)
        or params.head_size & (params.head_size - 1)
    ):
        return 1, config
    block_q = config["SPLIT_BLOCK_M"] // params.num_queries_per_kv
    query_blocks = params.num_tokens // block_q + params.num_seqs
    want = target / (params.num_kv_heads * query_blocks)
    if want < 2:
        return 1, config
    tiles = triton.cdiv(int(params.max_seqlen_k), config["SPLIT_TILE_SIZE"])
    splits = min(
        2 ** int(math.log2(want)),
        config["SPLIT_MAX"],
        triton.next_power_of_2(max(1, tiles // config["SPLIT_MIN_TILES"])),
    )
    if splits < 2:
        return 1, config
    return splits, dict(
        config,
        BLOCK_M=config["SPLIT_BLOCK_M"],
        num_warps=config["SPLIT_NUM_WARPS"],
        TILE_SIZE=config["SPLIT_TILE_SIZE"],
        LOOP_VARIANT=config["SPLIT_LOOP_VARIANT"],
    )


def _unified_attention_2d_gfx1250(params: _UAParams, output_gate=None):
    """
    Internal wrapper for the gfx1250 gluon kernel.

    loop_variant:
        0=plain double buffered version,
        1=2-stage version,
        2=4-stage version
    """
    assert params.softcap == 0, "Softcap is not supported"

    config = dict(get_unified_attention_config("attn_2d", params, backend="gluon"))
    long_kv = config.pop("LONG_KV_OVERRIDE", None)
    if (
        long_kv
        and os.environ.get("AITER_UA_2D_LONG_KV", "1") == "1"
        and params.max_seqlen_k >= long_kv["min_kv"]
        and _2d_full_causal_prefill(params)
    ):
        # causal full-attention prefill over a long key range: the entry's long-KV launch and FM_* options
        config.update(long_kv["config"])
    env_cfg = os.environ.get("AITER_UA_2D_CONFIG")
    if env_cfg:
        # tuning / debugging: JSON merged into the resolved attn_2d config
        config.update(json.loads(env_cfg))
    num_splits, config = _gfx12_2d_split(params, config)
    # fixed-max loop options (FM_*), passed to the kernel as they are; AITER_UA_2D_FIXED_MAX=1 turns the fixed
    # reference max on for causal full-attention prefill on loop variant 2 (off by default: not bitwise)
    fm_options = {k: config.pop(k) for k in list(config) if k.startswith("FM_")}
    if _UA2D_FIXED_MAX and "FM_FIXED_MAX" not in fm_options and _2d_full_causal_prefill(params):
        fm_options["FM_FIXED_MAX"] = True
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

    if _is_fp4pt_v(params.v_cache_dtype) and _fp4pt_pv_mode() == 1:
        # A8K4V4PT SCALE_ON_P holds V upcast to bf16 (TILE_SIZE x head_size, 512 VGPRs a lane for 128 keys at
        # d 256) next to P and the accumulator: one-page tiles and at most 16 query rows a warp, 64 a block, keep
        # it out of scratch (compile-only, Qwen 16 / 2 heads d 256: BLOCK_M 128 or 256-key tiles spill 0.3-2.7 KB)
        TILE_SIZE = params.block_size
        if BLOCK_M > 64:
            BLOCK_M = 64
        num_warps = max(num_warps, BLOCK_M // 16)
    # the loop variants other than 0 mask at most twice at the end of the loop,
    # and need a tile wider than 32
    query_span = (BLOCK_M - 1) // params.num_queries_per_kv + 1
    if query_span > TILE_SIZE or TILE_SIZE <= 32:
        loop_variant = 0

    num_buffers = config["NUM_BUFFERS"]
    waves_per_eu = config["waves_per_eu"]
    kv_nvfp4 = params.kv_cache_dtype == torch.uint8
    v_fp4pt = _is_fp4pt_v(params.v_cache_dtype)
    if kv_nvfp4:
        num_buffers = 2
        if params.v_cache_dtype == torch.uint8:
            # an NVFP4 V (A8W4, BF16 PV) runs the double-buffered loop variant 0 with one 128-token page per tile
            TILE_SIZE = params.block_size
            loop_variant = 0
        elif loop_variant not in (0, 1, 2):
            # NVFP4 K with an fp8 V (A8K4V8) or an fp4pt V (A8K4V4PT): loop variants 0, 1 and 2, tiles of one
            # or more pages
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

    if num_splits > 1:
        # fp32 partials in the reduce_segments layout; the reduce applies the output
        # scale and the gate
        segm_output = torch.empty(
            (params.num_tokens, params.num_query_heads, num_splits, params.head_size),
            dtype=torch.float32,
            device=params.q.device,
        )
        segm_max = torch.empty(
            (params.num_tokens, params.num_query_heads, num_splits),
            dtype=torch.float32,
            device=params.q.device,
        )
        segm_expsum = torch.empty(
            (params.num_tokens, params.num_query_heads, num_splits),
            dtype=torch.float32,
            device=params.q.device,
        )
        reduce_gate, output_gate = output_gate, None
    else:
        segm_output = segm_max = segm_expsum = None

    # buffer ops need the tensor to fit a 32-bit offset; gfx1250 loads through TDM
    MAX_INT32 = 2**31 - 1
    USE_STORE_BUFFER_OP = params.out.nelement() * params.out.element_size() <= MAX_INT32
    # A8K4V4PT: the kernel reads the fp4pt V cache as bytes; V_FP4PT says how
    v_cache = params.v.view(torch.uint8) if v_fp4pt else params.v
    if v_fp4pt:
        _fp4pt_log("2d", _fp4pt_pv_mode(), v=tuple(params.v.shape), tile=TILE_SIZE, block_m=BLOCK_M)
    seq_order = _gfx12_2d_seq_order(params, BLOCK_Q, total_query_blocks)
    seq_order_args = {} if seq_order is None else dict(SEQ_ORDER=True, seq_order_ptr=seq_order)
    _unified_attention_kernel_2d_gfx1250[
        (params.num_kv_heads, total_query_blocks, num_splits)
    ](
        query_ptr=params.q,
        key_cache_ptr=params.k,
        value_cache_ptr=v_cache,
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
        out_scale_ptr=params.output_scale if num_splits == 1 else None,
        USE_SINKS=(params.sinks is not None),
        SLIDING_WINDOW=params.sliding_window,
        # AITER_UA_FP4PT_NO_SPECIALIZE: a constant 0 keys the compile, num_blocks_rt carries the count
        num_blocks=0 if _GFX12_UA_RUNTIME_NUM_BLOCKS else params.num_blocks,
        NUM_BLOCKS_RUNTIME=_GFX12_UA_RUNTIME_NUM_BLOCKS,
        num_blocks_rt=int(params.num_blocks) if _GFX12_UA_RUNTIME_NUM_BLOCKS else None,
        stride_k_cache_0=params.k.stride(0),
        stride_k_cache_1=params.k.stride(1),
        stride_k_cache_2=params.k.stride(2),
        stride_k_cache_3=params.k.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
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
        V_FP4PT=v_fp4pt,
        V_PT_MODE=_fp4pt_pv_mode() if v_fp4pt else 2,
        query_scales_ptr=params.q_scales,
        query_scales_stride_0=params.q_scales.stride(0) if params.q_scales is not None else 0,
        query_scales_stride_1=params.q_scales.stride(1) if params.q_scales is not None else 0,
        gate_ptr=output_gate,
        gate_stride_0=output_gate.stride(0) if output_gate is not None else 0,
        GATE=output_gate is not None,
        NUM_SPLITS=num_splits,
        partial_m_ptr=segm_max,
        partial_l_ptr=segm_expsum,
        partial_acc_ptr=segm_output,
        enable_fp_fusion=True,
        **fm_options,
        **seq_order_args,
    )
    if num_splits > 1:
        # all heads of a token in few programs (~16 x 256 x 4 splits a program)
        block_h = min(params.num_query_heads, max(1, 64 // num_splits))
        reduce_splits_heads[
            (params.num_tokens, triton.cdiv(params.num_query_heads, block_h))
        ](
            params.out,
            segm_output,
            segm_max,
            segm_expsum,
            params.output_scale,
            reduce_gate,
            reduce_gate.stride(0) if reduce_gate is not None else 0,
            params.out.stride(0),
            params.out.stride(1),
            NUM_QUERY_HEADS=params.num_query_heads,
            NUM_SPLITS=num_splits,
            HEAD_SIZE=params.head_size,
            BLOCK_H=block_h,
            num_warps=4,
        )


def _gfx12_3d_qpack(params: _UAParams, config, token_split: bool):
    """(BLOCK_M, num_warps, XCD_REMAP) of a gfx1250 3d launch, from the tuned attn_3d
    entry's BLOCK_M / num_warps (see _GFX12_3D_QPACK); the entry's own, without the
    XCD placement, outside a token-split spec-decode verify of several blocks."""
    block_m, num_warps = config["BLOCK_M"], config.get("num_warps", 1)
    unchanged = (block_m, num_warps, False)
    G = params.num_queries_per_kv
    if (
        not (_GFX12_3D_QPACK and token_split and params.spec_decode)
        or num_warps != 1
        # NVFP4 V lays its scales out over the warps of the head dim
        or params.v_cache_dtype == torch.uint8
        or block_m % G
        or block_m < 16
    ):
        return unchanged
    seqlen_q = int(params.max_seqlen_q)
    blocks = triton.cdiv(seqlen_q, block_m // G)
    if blocks == 1:
        return unchanged
    live_seqs = min(params.num_seqs, params.num_tokens // seqlen_q)
    if live_seqs <= 1 and blocks <= 3:
        return block_m, 1, True
    rows = max(
        block_m, min(triton.next_power_of_2(blocks * block_m), _GFX12_3D_QPACK_ROWS)
    )
    wgs_per_seq = triton.cdiv(seqlen_q, rows // G)
    # few workgroups a segment: shorter warps cut each KV tile's latency; more: 32-row
    # warps read each tile from LDS half as often a row
    warp_rows = 16 if live_seqs * params.num_kv_heads * wgs_per_seq <= 4 else 32
    warp_rows = min(max(warp_rows, block_m), rows)
    return rows, rows // warp_rows, wgs_per_seq > 1


def _unified_attention_3d_gfx1250(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
    segm_tiles=None,
):
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
    prefill = segm_tiles is not None and _gfx12_3d_prefill(params, config)
    if prefill:
        prefill_pack = (config["PREFILL_SPLIT_ROWS"], config["PREFILL_SPLIT_WARPS"])
    for key in _GFX12_ROUTING_KEYS:
        config.pop(key, None)
    config["BLOCK_M"] = max(
        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)
    )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    assert config["BLOCK_Q"] >= 1
    # A8K4V4PT: the kernel reads the fp4pt V cache as bytes; V_CACHE_DTYPE "fp4pt" says how
    v_cache = params.v
    V_PT_MODE = 2  # unused unless V is fp4pt
    if _is_fp4pt_v(params.v_cache_dtype):
        v_cache = v_cache.view(torch.uint8)
        V_PT_MODE = _fp4pt_pv_mode()
        _fp4pt_log("3d", V_PT_MODE, v=tuple(params.v.shape), block_m=config["BLOCK_M"])
    tuned_warps = config.get("num_warps", 1)
    if prefill:
        # a prefill chunk's workgroups (see _gfx12_prefill_split), warps of whole rows
        assert tuned_warps == 1 and prefill_pack[0] % params.num_queries_per_kv == 0
        config["BLOCK_M"], config["num_warps"] = prefill_pack
        xcd_remap = False
    else:
        config["BLOCK_M"], config["num_warps"], xcd_remap = _gfx12_3d_qpack(
            params, config, segm_tiles is not None
        )
    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv
    # a packed workgroup's warps own query rows in PV too, not head-dim columns
    pv_warps_m = config["num_warps"] > tuned_warps

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
        value_cache_ptr=v_cache,
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
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
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
        V_PT_MODE=V_PT_MODE,
        PV_WARPS_M=pv_warps_m,
        XCD_REMAP=xcd_remap,
        XCD_ROTATE=_GFX12_3D_XCD_ROTATE,
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


def _reduce_segments_split_gfx1250(
    params: _UAParams,
    segm_output,
    segm_max,
    segm_expsum,
    NUM_SEGMENTS,
    TILE_SIZE,
    segm_tiles=None,
    output_gate=None,
    output_rows=None,
    oneshot=False,
):
    num_rows = params.num_tokens if output_rows is None else output_rows.shape[0]
    if num_rows == 0:
        return
    sb, num_buf, num_warps = _GFX12_SPLIT_REDUCE_CONFIG
    sb = min(sb, NUM_SEGMENTS)
    if oneshot and _reduce_segments_split_oneshot_kernel_gfx1250 is not None:
        chunk, spec, oneshot_warps = _GFX12_SPLIT_REDUCE_ONESHOT_CONFIG
        chunk = max(sb, min(chunk, NUM_SEGMENTS) // sb * sb)
        if params.head_size_v % (32 * oneshot_warps) == 0:
            _reduce_segments_split_oneshot_kernel_gfx1250[
                (num_rows, params.num_query_heads)
            ](
                params.out,
                segm_output,
                segm_max,
                segm_expsum,
                params.seqused_k,
                params.num_seqs,
                segm_tiles,
                params.output_scale,
                output_gate,
                output_gate.stride(0) if output_gate is not None else 0,
                output_rows,
                params.out.stride(0),
                params.out.stride(1),
                H=params.num_query_heads,
                S=NUM_SEGMENTS,
                D=params.head_size_v,
                TILE_SIZE=TILE_SIZE,
                SEQLEN_Q=1 if params.all_decode else int(params.max_seqlen_q),
                SB=sb,
                CHUNK=chunk,
                SPEC=min(spec, chunk) // sb * sb,
                NUM_WARPS=oneshot_warps,
                IS_FP8_OUT=(params.out.dtype == e4m3_dtype),
                FP8_MIN=torch.finfo(e4m3_dtype).min,
                FP8_MAX=torch.finfo(e4m3_dtype).max,
                num_warps=oneshot_warps,
            )
            return
    _reduce_segments_split_kernel_gfx1250[(num_rows, params.num_query_heads)](
        params.out,
        segm_output,
        segm_max,
        segm_expsum,
        params.seqused_k,
        params.num_seqs,
        segm_tiles,
        params.output_scale,
        output_gate,
        output_gate.stride(0) if output_gate is not None else 0,
        output_rows,
        params.out.stride(0),
        params.out.stride(1),
        H=params.num_query_heads,
        S=NUM_SEGMENTS,
        D=params.head_size_v,
        TILE_SIZE=TILE_SIZE,
        SEQLEN_Q=1 if params.all_decode else int(params.max_seqlen_q),
        SB=sb,
        NUM_BUF=num_buf,
        NUM_WARPS=num_warps,
        IS_FP8_OUT=(params.out.dtype == e4m3_dtype),
        FP8_MIN=torch.finfo(e4m3_dtype).min,
        FP8_MAX=torch.finfo(e4m3_dtype).max,
        num_warps=num_warps,
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
    then merges.

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


# AITER_UA_FAST_LAUNCH=1: record-and-replay host fast path (see unified_attention_utils.py).
if __import__("os").environ.get("AITER_UA_FAST_LAUNCH", "1") == "1":
    from aiter.ops.triton.utils.unified_attention_utils import wrap_fast_launch

    unified_attention = wrap_fast_launch(unified_attention)
