# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 lightning-indexer decode block-score kernel (FlyDSL).

Replaces ATOM's `_decode_index_score_tiled_kernel`. For each request b, each
128-token page p, and each (query row, index head) pair, produce

    score[head, row, p] = max over t in [0,128) of
                              dot(K[page, t, :], Q[row, head, :]) * scale

with a per-query causal cutoff.

Why this is faster than the Triton kernel it replaces
-----------------------------------------------------
Not because of MFMA -- Triton already uses tl.dot -- and not because of
pipelining, which Triton already does well. The Triton inner loop spends
240 v_max + 112 v_mov_b32_dpp against 8 v_mfma, plus two LDS round trips and
barriers, all of it in `tl.max(qk, axis=0)` (measured from the ISA).

The cause is layout. Triton picks mfma_32x32x16, which spreads the token axis
-- the axis being reduced -- across lanes, so a 128-token max needs DPP,
permlane and LDS. Choosing 16x16x32 with A=K (M=token) instead puts 4
consecutive tokens in *one lane's* 4 accumulators, measured on gfx950 with
positional codes exact in bf16 and asymmetric enough that a transposed or
mis-gathered mapping could not pass unnoticed. The page reduction then becomes
register-local elementwise max, and only the final fold across g needs two
shuffle_xor steps. In the default config there is no LDS and no barrier at all.

Lane mapping (measured on gfx950; every score test asserts it transitively,
since a wrong A, B or C mapping cannot reproduce the torch oracle),
lane = tid % 64, g = lane // 16, u = lane % 16:

    A operand[v] = A[u, 8*g + v]     v = 0..7     A = K, M = token
    B operand[v] = B[u, 8*g + v]     v = 0..7     B = Q, N = feature
    C accum[r]   = C[4*g + r, u]     r = 0..3

so a lane holds tokens {4g+r} at feature u -- exactly the fold we want.

Occupancy is not the lever -- measured, not assumed
---------------------------------------------------
At pages_per_wave=1 this runs at 128 VGPRs = exactly 4 waves/SIMD, which looks
like a classic occupancy cliff. It is not one -- the two knobs aimed straight at
it both fail. (The page loop does eventually get to 96 VGPRs and 5 waves/SIMD,
but as a side effect of unrolling, not by freeing Q; see `resolve_config`.)
Swept at b32_q8_s128k/512k/1M:

  feat_waves=2  128 -> 46 VGPRs, 4 -> 11 waves/SIMD, no spills, no LDS
                ... and 3-7% SLOWER at long sequence.
  q_to_lds      frees Q's 32 VGPRs and the scheduler immediately spends them
                on more load hoisting: 128 -> 136, i.e. one wave *worse*.
                Within noise on time (+-1%).

feat_waves does not cost HBM bandwidth -- TCC_EA0_RDREQ_sum is flat at 4.10 M
across feat_waves 1 vs 2, so the waves sharing a page really are deduplicated
(TCC_HIT_sum 1.48 M -> 3.01 M). What it costs is L2 *requests*: 27% more of
them for the same bytes, and that is the slowdown. The kernel is 5% off the
pure-gather floor and memory bound; more waves cannot help a kernel that is
already waiting on HBM, they just add pressure in front of it.

Both knobs are kept, defaulted off, so the next person sweeps instead of
re-deriving. feat_waves=2 does win ~5% at short sequence (b32_q8_s8k fp8),
where the kernel is launch- rather than bandwidth-bound.

Short sequence is a different machine -- and needs a different knob
--------------------------------------------------------------------
Total waves = total pages / pages_per_wave, independent of how the grid is
sliced, so at b16_q1_s8k that is 1024 waves against 4096 wave slots: three
quarters of the GPU idle, and 26-36% of peak bandwidth. The only way to add
waves is to split the work *inside* a page. feat_waves does that on the
feature axis, but the axis is F = S*H columns wide and one MFMA tile is 16 of
them -- at S=1 H=4 there is exactly one tile and the knob is illegal. Which is
precisely the shape that is short of waves.

token_waves splits the page's 8 token panels instead. Always available (8
panels, any shape), and unlike feat_waves the co-resident waves read *disjoint*
K, so it does not inflate L2 requests. It costs one LDS round trip and one
barrier per page. Measured, 3 trials, shuffled fp8 (flyS -> best token_waves):

  decode_b16_s8k       17.3 -> 16.1 (tw4)   -7%   1024 waves, F = 4
  serve_b32_q1_s128k  157.4 -> 154.9 (tw4)  -2%   q=1, feat_waves illegal
  decode_b64_s8k       34.1 -> 33.4 (tw2)   -2%
  serve_b32_q8_s128k  159.3 -> 159.5 (tw2)   0%   already wave-saturated
  serve_b50_q8_s100k  181.3 -> 183.3 (tw2)  +1%   tw4 is +17%, much worse

So: worth setting at q=1 (or any shape with fewer pages than wave slots),
worth nothing at q=8 long sequence, and actively harmful at tw4 once the
machine is already full. Auto-selection is limited to the much narrower
native-layout Q1 regime in resolve_config; explicit 1 keeps it off.

Where the bandwidth actually is, and what is left
-------------------------------------------------
Achieved read bandwidth, best config per shape, median of 40 with L2 flushed:

  fp8    decode_b16_s8k    0.02 GB  0.99 TB/s     bf16  0.03 GB  1.85
         decode_b64_s8k    0.07     2.83                0.13     4.05
         spec_b50_q4_s100k 0.64     4.98                1.28     5.69
         serve_b32_q1_s128k 0.52    5.43                1.05     5.84
         serve_b32_q8_s128k 0.52    4.77                1.05     5.77
         ragged8x_b32_q8   0.52     4.70                1.05     5.63

Absolute numbers drift 3-5% between processes on this box (the floor and the
Triton column drift with them), so compare configs inside one run, not across
runs. Every knob decision recorded here was made from a paired run.

Read that table down the GB column, not across the dtype column: bandwidth is a
function of *footprint*, and fp8 looks slow only because it moves half the
bytes for the same page count -- twice the page-id loads, epilogues and wave
launches per byte delivered. It is not a defect in the fp8 path.

The ceiling is 6.3-6.4 TB/s, not 8. Measured on this part with L2 flushed:
torch.add(a, a, out=b) (1 read + 1 write) reaches 6.31, and a pure contiguous
int32 read reaches 6.41 -- but only non-temporally; at the default cache policy
the same read tops out at 5.47. So serve_b32_q1_s128k/bf16 at 6.05 is 94% of
what any kernel gets on this silicon, and the arithmetic there is fully hidden.

An earlier revision of this paragraph claimed ~5.2 TB/s was the ceiling and that
"nothing simple reads faster than this kernel". Both were wrong, and wrong in
the same way: every reference they were measured against -- including the
bench's own gather floor -- allocated the stream in L1. See `nt_k`.

Remaining levers, in the order they are worth trying:
  1. The q1-vs-q8 gap: 6.17 vs 5.43 TB/s bf16, 5.58 vs 4.75 fp8, same bytes and
     same page count. q1 is at the memory ceiling and q8 is not, so at F=32 the
     MFMA has stopped hiding under the loads. This is now the largest lever at
     the production point and it is an arithmetic problem, not a memory one.
     `resolve_config`'s page loop closed about a third of it (q8 bf16 was 5.11,
     fp8 4.30) by raising occupancy to 5 waves/SIMD; the rest is still open,
     but it is not the causal mask. Deleting the mask outright -- v_cmp and
     v_cndmask gone, wrong answers, timing only -- is worth 0.5% bf16 and 2.2%
     fp8 at serve_b32_q8, and moves q1 by as much, so it is inside drift. A
     scalar fast path for pages entirely below the cutoff would recover less
     than that, and only on the pages that are not the boundary page. Dead end.
  2. Scatter: already small. identity vs randperm block table is 8% either
     dtype, so a locality-aware paged allocator would buy at most that.
  3. Short sequence: bounded by wave supply, and token_waves caps at 4 because
     WAVES == 4. THREADS=512 would allow token_waves=8 (one panel per wave)
     and 2x the waves again at b16_q1_s8k. Untested.
"""

import math
from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    _to_raw,
    buf_copy_atom,
    ptr_arg,
    ptr_buf_tensor,
)

WAVE = 64
WAVES = 4  # waves per CTA
THREADS = WAVE * WAVES
PAGE = 128  # SPARSE_BLOCK_SIZE
HEAD_DIM = 128  # the only supported head dim; asserted on the host
MFMA_M = MFMA_N = 16
MFMA_K = 32
PANELS = PAGE // MFMA_M  # 8 token panels per page
KSTEPS = HEAD_DIM // MFMA_K  # 4 k-steps per dot
LOG2E = 1.4426950409
NEG_INF = float("-inf")

# rocdl.sched_group_barrier instruction-class masks (rocdl._SCHED_MASK_INT_TO_KW).
_SCHED_MFMA = 8
_SCHED_VMEM_RD = 32


@dataclass(frozen=True)
class IndexScoreConfig:
    """Tunables, in the shape aiter/ops/flydsl/gemm_kernels.py uses.

    A note on axis names, because they are easy to get backwards here. This
    kernel puts A=K (MFMA M = token) and B=Q (MFMA N = feature = tok*H+head),
    so the *query* axis -- what you would call M in a plain GEMM, and what
    `spec_decode * num_kv_head` sizes -- is N in this kernel. `feat_waves` is
    therefore the TILE_N knob, and `q_to_lds` is this kernel's `b_to_lds`.

    feat_waves     waves of the CTA that split the feature axis. Each of them
                   re-reads the whole page, so this buys parallelism at the
                   cost of duplicate L2 requests.
    token_waves    waves of the CTA that split a page's 8 token panels. Same
                   parallelism as feat_waves without the duplicate reads --
                   each wave loads a different slice of the page -- paid for
                   with one LDS round trip and one barrier per page to combine
                   the partial maxes. feat_waves * token_waves waves cover one
                   page; the remaining WAVES // (feat_waves * token_waves)
                   cover distinct pages. 0 selects the narrow automatic split
                   in resolve_config; explicit 1/2/4 always win.
    pages_per_wave pages one wave walks back to back, pipelined across the
                   boundary. 0 means pick it from the launch bounds; see
                   resolve_config, which every entry point calls first.
    q_to_lds       stage Q through LDS instead of holding it in VGPRs.
    shuffled       K cache is pre-shuffled into the kernel's load order.
    waves_per_eu   occupancy floor passed to the backend; 0 leaves it unset.
                   Nothing here has ever wanted it. It was re-swept after the
                   page loop brought the kernel to 96-101 VGPRs, i.e. once 5
                   waves/SIMD was already free: serve_b32_q8 bf16 goes 184.2 ->
                   215.3 (wpe 5) -> 316.2 (wpe 6), and the one point that gains
                   (q1 fp8, -2%) is inside drift. Asking for occupancy the
                   allocator has not offered only costs.
    nt_k           CDNA cache policy for the K loads, as the raw aux field:
                   0 = default (allocate in L1), 2 = NT, 3 = SC0|NT, the
                   `GROUP_NT` the rest of aiter uses for streamed reads (see
                   csrc/include/aiter_opus_plus.h). Bit 0 is sc0, bit 1 is nt;
                   NT means "L1 miss evict, L2 hit stream".

                   K is read once per launch and never revisited, so allocating
                   it in L1 costs a line fill per access and buys nothing -- and
                   the fills, not DRAM, are what cap the load path. On gfx950 a
                   32000-page random gather runs 3.29 -> 5.02 TB/s (fp8) once
                   this is non-zero. An int rather than a bool so the null
                   hypothesis stays measurable and so 2 vs 3 can be swept; the
                   same knob in mega_moe (mega_moe_stage1.py) is an int for the
                   same reason. Note NT is known to *hurt* fp32 operands
                   elsewhere in aiter (worse cache-line utilisation); K here is
                   always 1 or 2 bytes, so that gate does not apply.

                   2 vs 3 was swept and is a null: at serve_b32_s128k, median of
                   40, bf16 171.7/203.9 (nt=2) against 171.6/203.5 (nt=3) and
                   fp8 93.4/117.5 against 93.8/117.0 -- every delta under 0.5%
                   and the sign inconsistent across the four points. The sc0 bit
                   buys nothing here; only the nt bit matters. Left at 2 because
                   it is the smaller claim, not because 3 was shown worse.

                   -1 (the default) resolves to 2, except on the served
                   decode path (see resolve_config), where it resolves to 0.
                   Those NT numbers were taken with a read-modify-write flush
                   in front of every launch, which leaves the cache full of
                   dirty lines that only an allocating load pays for; timed
                   instead as 57 layers' pools back to back, the default
                   policy is 3-7% faster at every served batch size.

    sched          instruction-scheduling hint for the page loop. -1 picks it
                   from the query shape (see resolve_config); 0 leaves the
                   backend alone. 1/2/3 emit `rocdl.iglp_opt(0/1/2)` at the top
                   of the loop body -- LLVM's canned MFMA/VMEM interleavings,
                   written for FA-shaped loops. 4 spells the interleave out with
                   `sched_group_barrier`, one VMEM group per MFMA group.

    spread         work-map layout. 0 hands out fixed chunks of `work_chunk`
                   pages; 1 spreads each step's live pages evenly over the CUs,
                   1..4 pages per CTA and at most one per wave (see
                   make_spread_work_map). Needs one page per wave and no
                   intra-page split. -1 resolves to 1 on the served decode path
                   when pages_per_wave resolves to 1, else 0.

    cp_world       ranks a context-parallel indexer splits the blocks over, and
    cp_rank        which one this launch is. The default 1/0 is the whole
                   context on one rank, i.e. every expression below folds away.

                   Under CP each rank owns the round-robin subset
                   `block % cp_world == cp_rank`, so logical block p of this
                   launch is global block `p * cp_world + cp_rank`. That
                   remapping is the entire difference: the global id addresses
                   the block table and positions the causal mask, while the
                   score is stored at the local p, giving the compacted
                   [H, tokens, ceil(blocks/world)] shard the CP selector reads.

                   Both are constexpr rather than runtime arguments because
                   `num_pages` divides by cp_world; constexpr makes that a
                   multiply-shift instead of an integer divide in the prologue.
                   The consequence is one compiled kernel per rank, which is why
                   `kernel_name` carries them.

    There is deliberately no maxnreg knob: --amdgpu-num-vgpr is reachable
    through compile_hints, but 64 and 96 both left this kernel at 128 VGPRs,
    i.e. it does nothing here. waves_per_eu is the channel that works.
    """

    feat_waves: int = 1
    token_waves: int = 0  # 0 = auto; explicit 1/2/4 are never overridden
    pages_per_wave: int = 0  # 0 = auto, resolved from the launch bounds
    q_to_lds: bool = False
    shuffled: bool = False
    waves_per_eu: int = 0
    nt_k: int = -1  # -1 = auto: 0 on the served decode path, else 2
    sched: int = -1  # -1 = auto, resolved from the query shape
    cp_world: int = 1  # 1 = no context parallelism; see above
    cp_rank: int = 0
    spread: int = -1  # -1 = auto: 1 on the served one-page-per-wave path, else 0


def feature_tiles(S: int, H: int) -> int:
    """Number of 16-wide feature tiles needed to cover F = S*H columns."""
    return (S * H + MFMA_N - 1) // MFMA_N


def _tiling(S: int, H: int, cfg: "IndexScoreConfig"):
    """(tiles, tiles per wave, tiles rounded up to a whole number of waves).

    The third is what buffers are sized on: when feat_waves does not divide FT
    the last feature group owns tiles past the end, and letting them exist
    (columns >= F are already masked at store time) is cheaper than special
    casing the tail.
    """
    ft = feature_tiles(S, H)
    ftw = (ft + cfg.feat_waves - 1) // cfg.feat_waves
    return ft, ftw, ftw * cfg.feat_waves


def q_lds_bytes(S: int, H: int, cfg: "IndexScoreConfig") -> int:
    """LDS held by the Q staging buffer, 0 when Q stays in registers.

    One 16 B fragment per (feature tile, k-step, lane), which is exactly
    FT*16 features x 128 head-dim elements with nothing duplicated.
    """
    if not cfg.q_to_lds:
        return 0
    return _tiling(S, H, cfg)[2] * KSTEPS * WAVE * 16


def reduce_lds_bytes(S: int, H: int, cfg: "IndexScoreConfig") -> int:
    """LDS held by the cross-wave partial-max exchange, 0 when token_waves == 1.

    One fp32 per (wave, wave-local feature tile, lane). Keeping all 64 lanes
    rather than just the g == 0 writers makes the write contiguous and the read
    lane-local, and it is only WAVES*FTW*64*4 B -- 2 KB at S=8 H=4.
    """
    if cfg.token_waves <= 1:
        return 0
    return WAVES * _tiling(S, H, cfg)[1] * WAVE * 4


def work_chunk(cfg: IndexScoreConfig) -> int:
    """Pages one CTA covers -- the granularity make_work_map hands out.

    Waves not spent on a page's feature or token axis each take their own page,
    and each of those loops `pages_per_wave` times.
    """
    if not cfg.pages_per_wave:
        raise ValueError("pages_per_wave is unresolved; call resolve_config first")
    # An explicit page depth disables auto token splitting.
    return (WAVES // (cfg.feat_waves * (cfg.token_waves or 1))) * cfg.pages_per_wave


def _grid_chunks(max_block: int, cfg: IndexScoreConfig) -> int:
    """Grid columns per request. A spread map needs one spare row per request:
    it may cut up to one CTA per request boundary, and the spare column keeps
    four pages per CTA enough even at full capacity."""
    chunk = work_chunk(cfg)
    return (max_block + chunk - 1) // chunk + (1 if cfg.spread > 0 else 0)


def _spread_fits(batch: int, max_block: int, cfg: IndexScoreConfig) -> bool:
    """A spread row packs the first page in 16 bits and the page count in the
    top byte of the length word, so both have to fit; the map kernel unrolls
    the batch."""
    return (
        batch <= SPREAD_MAX_BATCH
        and max_block <= 0xFFFF
        and max_block * cfg.cp_world * PAGE <= 0xFFFFFF
    )


# The spread map is built by one small kernel with the request loop unrolled,
# so it is limited to batches this size (the served ladder stops at 4).
SPREAD_MAX_BATCH = 64
MAP_THREADS = 256

# Chunks the packed work map can address. A row names its chunk in 16 bits, so
# a grid wider than this cannot be packed at all -- which is also what bounds
# `max_block` once the chunk is one page.
_MAX_CHUNKS = 0x10000


# Oversubscription the page loop is allowed to spend. Below this many CTAs the
# machine is short of waves and a longer per-wave loop only makes it shorter;
# above it, the loop is free and buys memory-level parallelism. 4x mirrors
# _auto_num_splits in mqa_logits/fp8_mqa_logits.py, which tuned the same
# trade-off on the same class of kernel.
_CTA_OVERSUBSCRIBE = 4


def _auto_token_split(cfg: IndexScoreConfig) -> bool:
    """Only tune the untouched decode path; CP does not alter the gate.

    `shuffled` is in scope. It changes the *order* the K bytes are read in, not
    how much work there is, so the depth tables and the spread map -- which are
    about filling CUs -- carry over unchanged. Leaving it out cost a factor of
    two at small launches: a 128-page case fell back to the capacity estimate's
    4 pages/wave, i.e. 8 CTAs on 256 CUs, and measured 9.6 us against the
    native layout's 3.9. The one thing that does not carry over is the cache
    policy, which the layout inverts; see `nt_k` at the resolve site.
    """
    return (
        cfg.token_waves == 0
        and cfg.pages_per_wave == 0
        and cfg.feat_waves == 1
        and not cfg.q_to_lds
        and cfg.waves_per_eu == 0
        and cfg.nt_k == -1
        and cfg.sched == -1
        and cfg.spread == -1
    )


# pages_per_wave for the served MiniMax-M3 decode launches, keyed by (S, H):
# (largest graph batch, depth) steps, then 4 beyond the last one.
_SERVED_DECODE_DEPTH = {
    (4, 1): ((4, 1), (16, 2)),  # TP4 MTP: Q4, one index head
    (8, 4): ((2, 1), (4, 2), (16, 3)),  # Q8, four index heads
}
# Every other one-feature-tile shape (S*H <= 16) was swept too -- S1H1, S4H4,
# S1H4 behave within a few percent of S4H1 at every batch -- so they share its
# table.
_ONE_TILE_DECODE_DEPTH = _SERVED_DECODE_DEPTH[(4, 1)]


def _decode_depth_table(S: int, H: int):
    if (S, H) in _SERVED_DECODE_DEPTH:
        return _SERVED_DECODE_DEPTH[(S, H)]
    if S and H and feature_tiles(S, H) == 1:
        return _ONE_TILE_DECODE_DEPTH
    return None


def _served_decode(cfg: IndexScoreConfig, S: int, H: int, max_block: int) -> bool:
    """A decode launch with a measured depth table, no CP. The tables were
    measured at max_block 8192; past 0xFFFF blocks their shallow depths would
    not even pack, so such capacities keep the estimate."""
    return (
        _auto_token_split(cfg)
        and _decode_depth_table(S, H) is not None
        and cfg.cp_world == 1
        and max_block <= 0xFFFF
    )


def _served_decode_depth(batch: int, S: int = 4, H: int = 1) -> int:
    """pages_per_wave for a served decode launch, by graph batch size."""
    for limit, depth in _decode_depth_table(S, H):
        if batch <= limit:
            return depth
    return 4


def _occupancy_depth_cap(
    batch: int, max_block: int, cfg: IndexScoreConfig, device=None
) -> int:
    """Deepest pages_per_wave that still hands every CU a CTA.

    The served depth tables above are keyed on batch alone, but depth only pays
    off when there are enough pages to fill the machine at that depth. A full
    batch of *short* requests has both: a large batch and very little work.
    `batch * max_block` is the graph's page capacity and so an upper bound on
    the live pages, which makes this cap safe in the direction that matters --
    a batch that really is long keeps its tabled depth, and only a launch that
    provably cannot fill one CU round gets trimmed.

    Without it a short-context full batch strands half the GPU: bs32 at an 8K
    context is 2048 pages, which at the tabled depth 4 is 16 pages per CTA,
    i.e. 128 CTAs on 256 CUs. Measured, 12 layers replayed, S4H1 fp8:

      bs32 8K (2048 pages)   depth 4 12.64 us   depth 2  8.68   depth 1  9.04
      bs16 8K (1024 pages)   depth 2  6.80      depth 1  5.82
      bs8  8K ( 512 pages)   depth 2  6.63      depth 1  5.44
      bs64 8K (4096 pages)   depth 4 14.49      depth 2 15.21   (cap: 4, kept)

    The non-served auto path below already guards occupancy this way; this is
    the same guard for the tabled path.
    """
    per_wave = WAVES // (cfg.feat_waves * max(cfg.token_waves, 1))
    return max(1, (batch * max_block) // (per_wave * _cu_count(device)))


def _cu_count(device=None) -> int:
    import torch

    from aiter.jit.utils.chip_info import get_cu_num

    if device is None:
        return get_cu_num()
    return torch.cuda.get_device_properties(device).multi_processor_count


def resolve_config(
    batch: int,
    max_block: int,
    cfg: IndexScoreConfig,
    S: int = 0,
    H: int = 0,
    device=None,
):
    """Fill in `cfg`'s shape-dependent fields. Idempotent; explicit wins.

    Three fields are auto, each only at its sentinel: `token_waves` and
    `pages_per_wave` at 0, and `sched` at -1. Explicit token_waves=1 preserves
    the original unsplit behavior. Auto token splitting selects four waves and
    depth one only for native-layout Q1, one feature tile, and batch*local_blocks
    <= device CU count, with all other tuning knobs untouched. Up to twice
    that page count, two token waves fill the first CTA scheduling round;
    beyond it the split resolves to one. CP bounds are local.
    `sched` needs the query shape, so pass S and H if
    you use that sentinel; without them it stays unresolved and the kernel
    treats it as off.

    pages_per_wave
    ~~~~~~~~~~~~~~
    Looping a wave over several pages keeps the next page's loads in flight
    across the page boundary, so the wave never drains to vmcnt(0) at the seam.
    That is worth 5-12% at long sequence -- but it also divides the CTA count by
    the same factor, and at short sequence there were not enough CTAs to fill
    the GPU to begin with. Measured at s8k it costs up to +71% (decode_b16 fp8
    17.0 -> 29.1 us), while at s128k it gains 9% (serve_b32_q8 fp8 122.6 ->
    111.5). So the knob is chosen by how much work there is, not by shape name.

    Part of the gain is not the pipeline at all: the unrolled loop lets the
    compiler reuse the Q and accumulator registers across pages, so VGPRs fall
    128 -> 99 (ppw=2) -> 96 (ppw=4) and occupancy rises 4 -> 5 waves/SIMD, with
    no spills at any of the three. That is the same 128 -> 96 the abandoned
    `q_to_lds` plan was chasing, obtained here for free.

    The ladder stops at 4 on purpose. 8 and 16 were swept twice over the long
    cases; the ordering reproduces exactly but has no single predictor. At
    serve_b32_q8, fp8 wants 16 (105.1 against 111.0 at 4) while bf16 wants 4
    (186.8 against 193.7 at 16). At serve_b50_q8, bf16 wants 8 (236.1) and fp8
    wants 16, with 8 the *worst* of the three (145.7). ragged8x fp8 wants 8.
    Every rule that fits the long cases -- halve the CTA target for fp8, double
    the depth for fp8 -- breaks decode_b64_s8k, where fp8 at 2 already gives up
    ~15%. The ISA is clean at every depth (no spills, 5 waves/SIMD, 99-101
    VGPRs), so this is memory parallelism against CTA supply with a third term,
    most likely I$: at 16 the body is 1024 MFMA and 264 loads of straight-line
    code. Left at 4, which is never more than 4% off the per-row best; pass an
    explicit pages_per_wave if you are tuning one fixed shape.

    The estimate uses only `batch` and `max_block`, never the contents of
    seq_lens: the grid is a launch-time bound that has to stay valid across a
    cudagraph replay with different lengths. For a ragged batch that over-counts
    (the holes are counted as work), which biases towards the longer loop; every
    ragged case measured still wants the longest one, so the bias is harmless.

    served decode
    ~~~~~~~~~~~~~
    The capacity estimate above is wrong for the launch MiniMax-M3 actually
    serves: TP4 decode with MTP gives S=4, H=1, no CP, and max_block is the
    1M-token capacity (8192), so it always picked 2 (batch 1) or 4 pages per
    wave. Real lengths are far shorter -- captured from an agentic cc=4 run,
    median 110K tokens, and 97% of steps are graph batch 1 or 2 -- so that
    left 20-450 live CTAs, one wave per SIMD, each walking a 16-32 panel chain
    that ATT showed to be pure K latency. The grid has to stay graph-stable,
    so the depth is chosen from the graph batch alone, and the auto cache
    policy goes to 0 (see nt_k). Timed as 57 layers' pools back to back,
    served lengths, weighted mean us per graph batch, fp8:

      batch    old auto   ppw 1   ppw 2   ppw 3   ppw 4   Triton
      1          8.87      7.21    8.57     -      -       7.82
      2         19.83     14.11   15.98     -      -      16.37
      4         20.60     14.90   16.11     -      -      25.32
      8         49.08     44.96   44.89   44.47   45.97   61.12
      16        92.69     87.05   84.93   85.85   86.67  111.30
      32       157.96    154.61  148.98  147.24  147.64  204.00
      128      687.18    678.01  656.67  647.15  641.87  874.92

    (batch 1-4: captured steps, nt 0; 8+: synthetic full batches with
    captured lengths, nt 0 except "old auto"). Over the captured step mix
    that is 1.28x the old auto and 1.14x Triton. Inside batch 1 the long
    steps (~466K) would still prefer 4 (-4%), which a batch-only rule
    cannot see. Measured at max_block 8192 only; other shapes keep the
    capacity estimate.

    Where that resolves to one page per wave, the map resolves to spread
    (see make_spread_work_map): the grid is still fixed by the graph batch,
    but each step's live pages are dealt out so no CU ends up holding a
    spilled full CTA. Captured steps, layers bench, weighted over the step
    mix: S4 H1 9.81 -> 9.56 us, S8 H4 9.92 -> 9.64.

    S=8, H=4 (two feature tiles) gets its own table from the same harness
    and lengths. nt 1 ties nt 0 and nt 3 ties nt 2, which is ~9% slower;
    sched 0/1, q_to_lds and feat_waves 2 (-30%) do not help. Weighted mean us,
    nt 0 except "old auto" (ppw 2 at batch 1, else 4; nt 2):

      batch    old auto   ppw 1   ppw 2   ppw 3   ppw 4   ppw 8   Triton
      1          9.20      7.99    8.95   10.95   12.30     -     12.13
      2         19.84     15.72   16.38   15.63   17.20     -     28.42
      4         18.58     15.67   14.33   14.28   16.44     -     39.85
      8         49.82     53.57   47.87   46.85   47.04     -    113.76
      16        96.27    108.56   91.36   89.19   89.84     -    231.76
      32       200.93    224.95  191.98  183.25  181.97     -    456.05
      64       366.28       -    351.26     -    324.82  320.98  890.39
      128      782.80       -    728.38     -    678.39  671.71 1565.10

    (batch 1-4: captured steps; 8+: synthetic full batches; 64/128 from a
    separate run.) Batch 2 is a tie between 1 and 3 (3 wins captured steps
    by 0.6%, 1 wins full batches by 3.6%); 8 stays out of the ladder for the
    reasons above.

    sched
    ~~~~~
    iglp_opt(0) pays exactly when a wave has more than one feature tile, and
    only then. Two runs of the full case list, shuffled, median of 40, sched 0
    vs 1, as a percentage:

      q=8 (FT=2)   serve_b32_q8 -1.4/-0.8   serve_b50_q8 -3.1/-1.1
                   ragged2x -2.3/-1.9       ragged8x -1.3/-0.5   (bf16/fp8)
      q=4 (FT=1)   spec_b50_q4 +0.3/+0.6    spec_b16_q4 0/+5.3
      q=1 (FT=1)   serve_b32_q1 -0.2/-0.1   decode_b16 +0.6/+3.3

    The sign tracks FT, not sequence length or dtype, and there is a mechanism:
    at FT=2 a panel is 8 MFMAs against 2-4 loads and there is something to
    interleave; at FT=1 it is 4 MFMAs and the hint only perturbs a schedule that
    was already right. So the gate is FT > 1. Costs 5 VGPRs (96 -> 101), which
    at ppw=4 still leaves 5 waves/SIMD and no spills.

    Variants 2 and 3 (iglp_opt(1)/(2)) and the hand-written sched_group_barrier
    interleave were measured at the same point and all lose: at serve_b32_q8 fp8,
    111.4 (variant 0) against 111.6 / 114.2 / 119.0.
    """
    auto_tokens = _auto_token_split(cfg)
    served = _served_decode(cfg, S, H, max_block)
    if served and S == 1 and batch * ((max_block + 1) // 2) <= _cu_count(device):
        # Q1 launches small enough for the token split below keep it: that
        # regime was measured separately and the depth tables were not.
        served = False
    if cfg.nt_k < 0:
        # The native layout wants K in L1 and the shuffled layout does not, and
        # the reason is the same fact seen from two sides. Natively one K
        # instruction takes 64 B from each of 16 rows, so every 128 B line is
        # requested twice; L1 turns the second into a hit, and NT would send it
        # back to L2 -- NT measures 0.94-1.00x there, i.e. it only ever loses.
        # Shuffled, one instruction is 1024 contiguous bytes and each line is
        # requested exactly once, so there is no second access to lose and the
        # fill buys nothing: NT is 1.10-1.22x. Neither half pays alone --
        # coalescing at the default policy is only 1.00-1.05x -- which is why
        # this flips with the layout rather than with the launch. Measured over
        # 12-layer replays, bs1-32 x 8K-512K, both S4H1 and S8H4; the shuffled
        # arm peaks at 6.69 TB/s against a 6.3-6.4 TB/s native ceiling.
        cfg = replace(cfg, nt_k=0 if served and not cfg.shuffled else 2)
    if cfg.token_waves == 0:
        cfg = replace(cfg, token_waves=1)
    if cfg.pages_per_wave and cfg.sched >= 0:
        return _resolve_spread(cfg, served, batch, max_block)

    if not cfg.pages_per_wave and served:
        cfg = replace(
            cfg,
            pages_per_wave=min(
                _served_decode_depth(batch, S, H),
                _occupancy_depth_cap(batch, max_block, cfg, device),
            ),
        )
    elif not cfg.pages_per_wave:
        cu = _cu_count(device)
        # One page per CTA fills otherwise missing CU scheduling slots. Only
        # native-layout Q1/one-tile launches were measured for this auto path.
        if auto_tokens and S == 1 and 0 < H <= MFMA_N:
            if batch * max_block <= cu:
                cfg = replace(cfg, token_waves=WAVES)
            elif batch * ((max_block + 1) // 2) <= cu:
                # Two pages/CTA retain one scheduling round while halving each
                # wave's panel chain. Beyond one round the LDS exchange loses.
                cfg = replace(cfg, token_waves=2)
        target = cu * _CTA_OVERSUBSCRIBE
        best = 1
        for ppw in (2, 4):
            chunk = (WAVES // (cfg.feat_waves * cfg.token_waves)) * ppw
            ctas = batch * ((max_block + chunk - 1) // chunk)
            if ctas >= target:
                best = ppw
        cfg = replace(cfg, pages_per_wave=best)

    if cfg.sched < 0 and S and H:
        cfg = replace(cfg, sched=1 if _tiling(S, H, cfg)[1] > 1 else 0)
    return _resolve_spread(cfg, served, batch, max_block)


def _resolve_spread(cfg: IndexScoreConfig, served: bool, batch: int, max_block: int):
    if cfg.spread >= 0:
        return cfg
    on = (
        served
        and cfg.pages_per_wave == 1
        and cfg.token_waves == 1
        and _spread_fits(batch, max_block, cfg)
    )
    return replace(cfg, spread=int(on))


def selection_filter(S: int, H: int, cfg: IndexScoreConfig, arch=None) -> bool:
    """Is this config legal for this shape? Mirrors gemm_kernels.selection_filter."""
    # Check the auto token split as its fallback geometry; its Q1 branch is
    # also legal (one feature tile, four token waves).
    if cfg.token_waves == 0:
        cfg = replace(cfg, token_waves=1)
    if cfg.pages_per_wave < 0 or cfg.feat_waves < 1 or cfg.token_waves < 1:
        return False
    # The two intra-page splits have to partition the CTA's waves between them,
    # and there is no point handing a wave fewer than one tile or one panel.
    if WAVES % (cfg.feat_waves * cfg.token_waves):
        return False
    if cfg.feat_waves > feature_tiles(S, H) or PANELS % cfg.token_waves:
        return False
    if cfg.waves_per_eu and not 1 <= cfg.waves_per_eu <= 10:
        return False
    if not -1 <= cfg.nt_k <= 3:  # -1 is the auto sentinel, see resolve_config
        return False
    if not -1 <= cfg.sched <= 4:  # -1 is the auto sentinel, see resolve_config
        return False
    if cfg.cp_world < 1 or not 0 <= cfg.cp_rank < cfg.cp_world:
        return False
    if not -1 <= cfg.spread <= 1:  # -1 is the auto sentinel, see resolve_config
        return False
    if cfg.spread == 1 and (
        cfg.pages_per_wave > 1 or cfg.token_waves > 1 or cfg.feat_waves > 1
    ):
        return False  # a spread CTA holds at most one page per wave
    if cfg.q_to_lds or cfg.token_waves > 1:
        from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes

        need = q_lds_bytes(S, H, cfg) + reduce_lds_bytes(S, H, cfg)
        if need > get_lds_capacity_bytes((arch or get_gfx()).split(":", 1)[0]):
            return False
    return True


def kernel_name(S: int, H: int, fp8: bool, cfg: IndexScoreConfig) -> str:
    """Config -> kernel name. Non-default knobs append a suffix, so the default
    config keeps the name it had before the knobs existed."""
    name = f"m3_index_score_S{S}_H{H}_{'fp8' if fp8 else 'bf16'}_L{cfg.pages_per_wave}"
    if cfg.shuffled:
        name += "_shuf"
    if cfg.feat_waves > 1:
        name += f"_fw{cfg.feat_waves}"
    if cfg.token_waves > 1:
        name += f"_tw{cfg.token_waves}"
    if cfg.q_to_lds:
        name += "_qlds"
    if cfg.waves_per_eu:
        name += f"_wpe{cfg.waves_per_eu}"
    if cfg.nt_k != 2:  # 2 is the default policy, so only the others are marked
        name += f"_kc{cfg.nt_k}"
    if cfg.sched > 0:  # 0 and the unresolved -1 both mean "no hint"
        name += f"_sc{cfg.sched}"
    if cfg.spread > 0:
        name += "_sp"
    if cfg.cp_world > 1:
        # The rank is part of the name, not just the world: ranks sharing a
        # node share the JIT cache directory, and they compile *different*
        # kernels (the block remap folds the rank in as a constant).
        name += f"_cp{cfg.cp_world}r{cfg.cp_rank}"
    return name


def build_index_score(S: int, H: int, fp8: bool, cfg: IndexScoreConfig):
    """Compile a score kernel specialised on (S, H, cache dtype, config).

    S = max query tokens per request (num_spec + 1), H = index heads.
    F = S*H is the feature count, laid out as column n = tok*H + head.
    """
    if not selection_filter(S, H, cfg, arch="gfx950"):
        raise ValueError(f"illegal config for S={S} H={H}: {cfg}")
    pages_per_wave = cfg.pages_per_wave
    shuffled = cfg.shuffled
    q_to_lds = cfg.q_to_lds
    nt_k = cfg.nt_k
    sched = cfg.sched
    # Context-parallel block remap. At 1/0 every use below is the identity and
    # the emitted code is byte-identical to the non-CP kernel.
    CP_WORLD = cfg.cp_world
    CP_RANK = cfg.cp_rank
    SPREAD = cfg.spread > 0

    F = S * H
    # FT feature tiles in total; FTW of them per wave; FT_PAD = FTW*feat_waves
    # is what the LDS buffer is sized on (see _tiling). FEAT_WAVES waves split
    # the feature axis, the remaining PAGE_WAVES cover distinct pages.
    _ft, FTW, FT_PAD = _tiling(S, H, cfg)
    FEAT_WAVES = cfg.feat_waves
    TOKEN_WAVES = cfg.token_waves
    # FEAT_WAVES x TOKEN_WAVES waves cooperate on one page; PAGE_WAVES groups
    # of those cover distinct pages.
    PAGE_WAVES = WAVES // (FEAT_WAVES * TOKEN_WAVES)
    PANELS_W = PANELS // TOKEN_WAVES  # token panels one wave walks
    CHUNK = PAGE_WAVES * pages_per_wave
    # One lane's K fragment for a given (panel, load): 8 elements for bf16
    # (16 B, one k-step) or 16 for fp8 (16 B, two k-steps). Both are 16 B, so
    # a shuffled page is WAVE*16 = 1024 B per load slot either way.
    CHUNK_ELEMS = 16 if fp8 else 8
    # K load instructions per panel. bf16 issues one dwordx4 per k-step; fp8
    # issues one per *pair* of k-steps, so both dtypes move 16 B per
    # instruction and fp8 halves the instruction count along with the bytes.
    K_LOADS = KSTEPS // 2 if fp8 else KSTEPS
    # Q staging buffer: one 16 B fragment per (tile, k-step, lane). Laid out so
    # the hot-loop read is a single ds_read_b128 with the 64 lanes covering
    # 1024 contiguous bytes, which is conflict-free.
    Q_LDS_SLOTS = FT_PAD * KSTEPS * WAVE if q_to_lds else 0
    # Cross-wave partial-max exchange, one fp32 per (wave, tile, lane).
    RED_SLOTS = WAVES * FTW * WAVE if TOKEN_WAVES > 1 else 0

    # Built from only the arrays this config uses, so the default config still
    # reports group_segment_fixed_size = 0 -- no LDS and no barrier at all.
    _fields = {}
    if Q_LDS_SLOTS:
        _fields["q"] = fx.Array[fx.Uint8, Q_LDS_SLOTS * 16, 16]
    if RED_SLOTS:
        _fields["red"] = fx.Array[fx.Float32, RED_SLOTS, 16]
    SharedStorage = (
        fx.struct(type("SharedStorage", (), {"__annotations__": _fields}))
        if _fields
        else None
    )

    @flyc.kernel(
        name=kernel_name(S, H, fp8, cfg),
        known_block_size=[THREADS, 1, 1],
    )
    def score_kernel(
        arg_q: fx.Pointer,
        arg_k: fx.Pointer,
        arg_score: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_work: fx.Pointer,
        i32_batch: fx.Int32,
        i32_stride_q_n: fx.Int32,
        i32_stride_q_h: fx.Int32,
        i32_stride_k_blk: fx.Int32,
        i32_stride_k_pos: fx.Int32,
        i32_stride_s_h: fx.Int32,
        i32_stride_s_b: fx.Int32,
        i32_stride_s_k: fx.Int32,
        i32_stride_bt_b: fx.Int32,
        f32_scale: fx.Float32,
    ):
        """One CTA scores CHUNK pages of request b.

            prologue   ids and strides, Q staged (registers or LDS), and every
                       page id this wave will touch, fetched in one batch
            main loop  for each of pages_per_wave pages, for each of its 8
                       panels: load K, MFMA against Q, causal-mask, fold into
                       a per-feature running max
            epilogue   (per page) fold that max across g, scale, store one
                       value per feature

        Everything is constexpr-unrolled, so "loop" here means the shape of
        the emitted code, not a branch.

        Wave decomposition: wave w takes fw = w % FEAT_WAVES (which slice of
        the feature axis) and pw = w // FEAT_WAVES (which page slot). A wave
        still owns whole pages, so the token reduction stays register-local and
        the only barrier in the kernel is the one after the Q fill -- placed in
        the prologue, ahead of every divergent branch.
        """
        # ==================== PROLOGUE ====================
        # The grid is a rectangle sized by max_block, but a ragged batch only
        # fills part of it. Rather than map (b, c) = (block.x, block.y) and let
        # the holes fall wherever the lengths put them, the CTA looks up which
        # (b, c) the n-th *dispatched* block owns. make_work_map() packs the
        # real work into [0, total) so every hole lands past the end, in one
        # contiguous run. This is the whole optimisation: holes bunched at the
        # tail cost ~0.2 us per 1000, holes interleaved with real work along the
        # dispatch axis cost ~7 us per 1000, because they consume launch slots
        # and retire before the machine can build up K loads in flight.
        #
        # block.x is the fast dispatch axis, so n is monotonic in launch order.
        n = fx.Int32(gpu.block_id("y")) * i32_batch + fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)
        g = lane // fx.Int32(16)
        u = lane % fx.Int32(16)
        # wave = fw + FEAT_WAVES*tw + FEAT_WAVES*TOKEN_WAVES*pw, so the waves
        # sharing a page are adjacent and their K addresses issue together.
        if const_expr(FEAT_WAVES * TOKEN_WAVES == 1):
            fw = tw = fx.Int32(0)
            pw = wave
        else:
            fw = (
                wave % fx.Int32(FEAT_WAVES)
                if const_expr(FEAT_WAVES > 1)
                else fx.Int32(0)
            )
            tw = (wave // fx.Int32(FEAT_WAVES)) % fx.Int32(TOKEN_WAVES)
            pw = wave // fx.Int32(FEAT_WAVES * TOKEN_WAVES)
        # This wave's slice of the page's token axis, as the three offsets the
        # body needs. All three fold to zero at TOKEN_WAVES == 1, which is why
        # the default config's addressing is unchanged.
        tw_tok = tw * fx.Int32(PANELS_W * MFMA_M)  # first token of the slice
        tw_slot = tw * fx.Int32(PANELS_W * K_LOADS * WAVE)  # shuffled load slot

        # Read both operand arrays as i32 and bitcast: one 8-element fragment is
        # 16 B (bf16) or 8 B (fp8), so it lands as a single dwordx4 / dwordx2
        # instead of 8 scalar loads. Every fragment base is 8-element aligned
        # (all strides here are multiples of 8), which those widths require.
        q_buf = ptr_buf_tensor(arg_q, fx.Int32)
        # Indexed in 16 B units, not i32s: every K access is a dwordx4, and the
        # copy atom that carries the cache policy needs the width in the layout.
        s_buf = ptr_buf_tensor(arg_score, fx.Float32)
        bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
        wm_buf = ptr_buf_tensor(arg_work, fx.Int32)

        # Two adjacent dwords, so this is the same single cache line -- and the
        # same single memory round trip -- that reading seq_lens[b] used to be.
        # Entry n is (packed, seq_len); a hole is (0, 0), which makes num_pages
        # zero and lets the existing tail guard below retire the CTA. No new
        # branch, and the clamp on the page id keeps every address legal.
        wm = fx.add_offset(fx.get_iter(wm_buf), n * fx.Int32(2))
        packed = fx.Int32(wm.load(T.i32))
        seq_len = fx.Int32(fx.add_offset(wm, fx.Int32(1)).load(T.i32))
        b = (packed >> fx.Int32(16)) & fx.Int32(0xFFFF)
        c = packed & fx.Int32(0xFFFF)
        if const_expr(SPREAD):
            # A spread row names its first page rather than a chunk, and
            # carries its page count (1..4) in the length word's top byte.
            span = (seq_len >> fx.Int32(24)) & fx.Int32(0xFF)
            seq_len = seq_len & fx.Int32(0xFFFFFF)
        # Pages this request actually owns. Grid is sized from max_seq_len, so
        # trailing waves find nothing to do.
        num_pages = (seq_len + fx.Int32(PAGE - 1)) // fx.Int32(PAGE)
        if const_expr(CP_WORLD > 1):
            # ...and of those, the ones this rank owns. Round-robin, so rank r
            # holds global blocks r, r+W, r+2W, ...: that is ceil((n - r) / W)
            # of them, clamped at zero for a request too short to reach r.
            # make_work_map applies the identical formula when it packs, and the
            # two must agree -- a mismatch is not a crash but a shard whose
            # chunks are numbered against a different length.
            avail = num_pages - fx.Int32(CP_RANK)
            zero_p = fx.Int32(0)
            avail = (avail < zero_p).select(zero_p, avail)
            num_pages = (avail + fx.Int32(CP_WORLD - 1)) // fx.Int32(CP_WORLD)

        def global_block(p):
            """Global block id of this rank's logical block `p`.

            The block table is indexed globally and the causal cutoff is a
            global token position, so both go through here; the score store
            does not, because the shard is stored compacted at local `p`.
            """
            if const_expr(CP_WORLD == 1 and CP_RANK == 0):
                return p
            return p * fx.Int32(CP_WORLD) + fx.Int32(CP_RANK)

        # -- feature decode: which (tok, head) does this lane's column carry? --
        # Column f = 16*ft + u, and f = tok*H + head. H is constexpr so these
        # fold to shifts. causal_len depends on tok, hence on u: it is a
        # per-lane value and must not be hoisted.
        #
        # Two flavours, because the feature split gives a wave only part of the
        # axis. `feature_of` takes the wave-local tile index i in [0, FTW) and
        # is what the main loop and the epilogue use; `feature_of_all` takes a
        # global tile index and is used only by the LDS fill, where the CTA
        # cooperatively stages *every* tile regardless of who will read it.
        feat_base = (
            fx.Int32(0) if const_expr(FEAT_WAVES == 1) else fw * fx.Int32(FTW * MFMA_N)
        )

        def feature_of(i):
            return feat_base + fx.Int32(MFMA_N * i) + u

        def feature_of_all(ft):
            return fx.Int32(MFMA_N * ft) + u

        def tok_head_of(i, all_tiles=False):
            f = feature_of_all(i) if all_tiles else feature_of(i)
            tok = f // fx.Int32(H)
            return tok, f - tok * fx.Int32(H)

        # -- where in Q does one fragment live? -------------------------------
        # Lane needs Q[row(u), head(u), 32*ks + 8*g + v], v = 0..7.
        # Strides come from the host: a CP parity test passes q[:, h:h+1],
        # whose token stride is still that of the full tensor.
        # Which 8 head-dim elements does lane (g, ks) carry? The dot runs over
        # all 128 and the kernel controls both gathers, so any bijection
        # (g, ks, v) -> k works provided Q and K use the same one. Pick the one
        # that makes every load instruction read 64 contiguous bytes:
        #
        #     byte offset of load j, lane g  =  64*j + 16*g
        #
        # so the four g-lanes tile one full 64 B cache line. bf16 needs 4 such
        # loads per panel (256 B row), fp8 needs 2 (128 B row) -- fp8 moves half
        # the bytes in half the instructions, which is the whole point of fp8.
        #
        # In elements that is k = 32*ks + 8*g for bf16, and for fp8 one 16 B
        # load covers 16 elements = two k-steps, so
        # k = 64*(ks//2) + 16*g + 8*(ks%2).
        #
        # This replaced an earlier fp8 map of k = 32*g + 8*ks. That one also
        # merged two k-steps into one dwordx4, but at byte 32*g + 16*i: the four
        # lanes landed 32 B apart, so each instruction used only half of every
        # cache line it touched and fp8 never reached bf16's bandwidth.
        #
        # `ks` is a Python int on the register path (so this folds to a
        # constant) and an Int32 on the LDS fill path, where each wave stages a
        # different k-step and the step index is therefore the wave id.
        def k_offset(ks):
            if const_expr(isinstance(ks, int)):
                if const_expr(fp8):
                    return (
                        fx.Int32(64 * (ks // 2))
                        + g * fx.Int32(16)
                        + fx.Int32(8 * (ks % 2))
                    )
                return fx.Int32(ks * MFMA_K) + g * fx.Int32(8)
            if const_expr(fp8):
                return (
                    (ks // fx.Int32(2)) * fx.Int32(64)
                    + g * fx.Int32(16)
                    + (ks % fx.Int32(2)) * fx.Int32(8)
                )
            return ks * fx.Int32(MFMA_K) + g * fx.Int32(8)

        def load_q_frag(i, ks, all_tiles=False):
            """One lane's 16 B of Q for (feature tile, k-step), from gmem."""
            tok, head = tok_head_of(i, all_tiles)
            row = b * fx.Int32(S) + tok
            # Columns past F are padding; clamp them to row 0 so the load stays
            # in bounds, then drop the result at store time.
            f = feature_of_all(i) if all_tiles else feature_of(i)
            in_range = f < fx.Int32(F)
            row = in_range.select(row, fx.Int32(0))
            head = in_range.select(head, fx.Int32(0))
            off = row * i32_stride_q_n + head * i32_stride_q_h + k_offset(ks)
            # 8 bf16 = 16 B = one dwordx4.
            return fx.Vector(
                fx.add_offset(fx.get_iter(q_buf), off >> fx.Int32(1)).load(
                    T.vec(4, T.i32)
                )
            )

        def as_bf16_frag(raw16):
            t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            t.store(raw16.bitcast(fx.BFloat16))
            return t

        # One allocation covers both users (Q staging and the token-wave
        # partial-max exchange); SharedStorage only carries the arrays this
        # config actually asked for.
        lds = (
            fx.SharedAllocator().allocate(SharedStorage).peek()
            if const_expr(SharedStorage is not None)
            else None
        )

        # -- stage Q: registers for the whole page loop, or once through LDS --
        if const_expr(q_to_lds):
            # Q depends only on block_id.x, so a per-wave register copy is four
            # identical copies of the same object -- 32 VGPRs each at S=8 H=4,
            # which is exactly what sits between this kernel and the next
            # occupancy bracket. Stage it once in LDS instead and ds_read it.
            #
            # The fill is a perfect assignment with no division: slot index
            # ci = t*THREADS + tid decomposes as lane = ci % WAVE,
            # ks = (ci // WAVE) % KSTEPS, ft = ci // (WAVE*KSTEPS), and since
            # THREADS == WAVE*KSTEPS those collapse to lane = lane, ks = wave,
            # ft = t. So thread `tid` on iteration t stages tile t, k-step
            # `wave`, its own lane -- the same (g, u) the gmem math already
            # assumes, and exactly FT_PAD iterations with no remainder.
            for ft in range_constexpr(FT_PAD):
                raw = load_q_frag(ft, wave, all_tiles=True)
                off = fx.Int32(ft * KSTEPS * WAVE * 16) + (
                    wave * fx.Int32(WAVE * 16) + lane * fx.Int32(16)
                )
                dst = fx.Tensor(
                    fx.make_view(
                        fx.recast_iter(
                            fx.Uint8, fx.add_offset(lds.q.ptr, fx.make_int_tuple(off))
                        ),
                        fx.make_layout(16, 1),
                    )
                )
                dst.store(raw.bitcast(fx.Uint8))
            # Must stay here: every branch below is wave-divergent.
            gpu.barrier()

            q_read_base = lane * fx.Int32(16)
            if const_expr(FEAT_WAVES > 1):
                q_read_base = q_read_base + fw * fx.Int32(FTW * KSTEPS * WAVE * 16)

            def q_operand(i, ks):
                off = q_read_base + fx.Int32((i * KSTEPS + ks) * WAVE * 16)
                src = fx.Tensor(
                    fx.make_view(
                        fx.recast_iter(
                            fx.Uint8, fx.add_offset(lds.q.ptr, fx.make_int_tuple(off))
                        ),
                        fx.make_layout(16, 1),
                    )
                )
                return as_bf16_frag(src.load())

        else:
            q_frag = [
                [as_bf16_frag(load_q_frag(i, ks)) for ks in range_constexpr(KSTEPS)]
                for i in range_constexpr(FTW)
            ]

            def q_operand(i, ks):
                return q_frag[i][ks]

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_N, MFMA_K, fx.BFloat16))
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(NEG_INF)

        # -- page ids: one batch, up front -----------------------------------
        # Every page id this wave will need, as independent loads with no
        # dependant between them, so they pipeline. The old code issued one per
        # page and waited on it before it could form that page's K address -- a
        # serial round trip repeated once per page, 32000 times at the
        # production shape, and ATT put 52.5% of latency in s_waitcnt.
        #
        # Pages are strided by PAGE_WAVES so that consecutive page slots cover
        # consecutive pages, keeping a wavefront's loads inside one span of
        # block_table. With FEAT_WAVES > 1 the waves sharing a page slot are
        # adjacent (pw = wave // FEAT_WAVES), so their identical K addresses
        # issue at the same time from the same CU and L1 serves the duplicates
        # -- which is why K is re-read rather than staged through LDS.
        #
        # Out-of-range entries are clamped rather than predicated: the value
        # only forms an address, and the store it would feed is masked off
        # below. Clamping uses cmp+select, not maximumf/minimumf -- those are
        # *float* ops and would compare these indices as bit patterns.
        # `limit` is where this CTA's pages end: the request's end, or for a
        # spread row the end of its own span.
        if const_expr(SPREAD):
            first = c
            end = c + span
            limit = (end < num_pages).select(end, num_pages)
        else:
            first = c * fx.Int32(CHUNK)
            limit = num_pages
        pages = [
            first + fx.Int32(j * PAGE_WAVES) + pw
            for j in range_constexpr(pages_per_wave)
        ]
        zero = fx.Int32(0)
        nm1 = limit - fx.Int32(1)
        last = (nm1 < zero).select(zero, nm1)

        def load_page_id(p):
            """Physical page holding logical block p of this request.

            The table is not sharded -- every rank sees the whole of it -- so
            this indexes with the global id.
            """
            # Empty CP shards may have no column for CP_RANK at all. They
            # still form descriptors (and token-split waves execute barriers),
            # so clamp their speculative load to the always-present column 0.
            column = (num_pages > zero).select(global_block(p), zero)
            return fx.Int32(
                fx.add_offset(fx.get_iter(bt_buf), b * i32_stride_bt_b + column).load(
                    T.i32
                )
            )

        page_ids = [load_page_id((pp < last).select(pp, last)) for pp in pages]

        def page_buffer(page):
            # Widen BEFORE multiplication. A pool can exceed the descriptor's
            # 4 GiB range; only within-page offsets belong in the buffer index.
            page_bytes = fx.Int64(i32_stride_k_blk) * fx.Int64(1 if fp8 else 2)
            address = fx.Int64(fx.ptrtoint(arg_k)) + fx.Int64(page) * page_bytes
            ptr = fx.inttoptr(arg_k.type, address)
            return ptr_buf_tensor(
                ptr, fx.Int32, unit_elems=4, num_records_bytes=fx.Int32(page_bytes)
            )

        page_buffers = [page_buffer(page) for page in page_ids]

        # -- K fragment access ------------------------------------------------
        # Issue and consume are split so a whole panel's loads (and the next
        # panel's, see the main loop) can be in flight at once. Issuing one load
        # and immediately waiting on it left the floor kernel ahead of us: at
        # the production shape this kernel was at 55% of measured HBM peak, and
        # the gap was memory-level parallelism, not arithmetic.
        # K is streamed: every byte is read exactly once per launch and never
        # revisited, so letting it allocate in L1 pays a line fill per access for
        # no reuse. Those fills, not DRAM, are what held the gather to ~3.3 TB/s
        # -- an isolated 32000-page gather at this exact shape went 3.29 -> 5.02
        # TB/s (fp8) purely from this bit. Q, the block table and the work map
        # keep the default policy: they are small and genuinely re-read.
        k_atom = buf_copy_atom(16, fx.Int32, cache_modifier=nt_k)

        def load_k16(page_buf, unit):
            """One 16 B K access at 16 B-unit index `unit`, via the k_atom.

            Goes through a copy atom rather than a plain `.load()` because the
            cache-policy field lives on the atom -- it is the only channel
            FlyDSL exposes for it. Issue and consume stay split: the fragment is
            registers, so the compiler still sinks the s_waitcnt to first use
            and a whole panel's loads remain in flight.
            """
            src = fx.slice(page_buf, (unit, None))
            frag = fx.make_fragment_like(src)
            fx.copy(k_atom, src, frag)
            return fx.Vector(fx.memref_load_vec(frag))

        def issue_k(page, panel, i):
            """Start load i of K[page, 16*panel + u, :], per the k_offset map.

            `panel` is this wave's *local* panel index; tw_slot / tw_tok shift
            it onto the wave's own slice of the page's token axis.
            """
            if const_expr(shuffled):
                # Pre-shuffled cache: the 16 B chunk each lane wants for
                # (panel, i) has already been placed at lane index, so one
                # instruction reads WAVE*16 = 1024 contiguous bytes instead of
                # 16 chunks scattered one row-stride apart. Addressing no longer
                # involves u, the row stride, or k_offset at all.
                base = (
                    fx.Int32((panel * K_LOADS + i) * WAVE) + tw_slot + lane
                ) * fx.Int32(CHUNK_ELEMS)
                # base is in cache elements; >>4 (fp8) / >>3 (bf16) turns it into
                # a 16 B unit index. Both are exact: CHUNK_ELEMS is one 16 B
                # chunk by construction, and the strides above it are multiples.
                shift = fx.Int32(4) if const_expr(fp8) else fx.Int32(3)
                return load_k16(page, base >> shift)
            tok_row = fx.Int32(MFMA_M * panel) + tw_tok + u
            base = tok_row * i32_stride_k_pos
            if const_expr(fp8):
                # k_offset(2i) = 64*i + 16*g and k_offset(2i+1) = that + 8, so
                # the two k-steps are 16 contiguous bytes: one dwordx4, and the
                # four g-lanes tile exactly one 64 B line.
                base = base + fx.Int32(64 * i) + g * fx.Int32(16)
                return load_k16(page, base >> fx.Int32(4))
            base = base + fx.Int32(i * MFMA_K) + g * fx.Int32(8)
            return load_k16(page, base >> fx.Int32(3))

        def convert_k(raws, ks):
            """Widen the raw load holding k-step ks into a bf16 A-fragment.

            fp8 is widened here rather than fed to a native fp8 MFMA: decode
            deliberately lifts K to Q's precision instead of rounding Q down,
            and matching that is what makes this a rewrite of the existing
            operator rather than a different one.
            """
            t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            if const_expr(fp8):
                # Load ks//2 holds k-steps 2n and 2n+1 in dwords [0:2] and
                # [2:4]. Two v_cvt_scalef32_pk_bf16_fp8 per dword; the cache is
                # unit-scale, hence scale 1.0.
                raw = raws[ks // 2]
                lo = 2 * (ks % 2)
                one = _to_raw(fx.Float32(1.0))
                pairs = [
                    fx.Vector(
                        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                            T.vec(2, T.bf16),
                            _to_raw(fx.Int32(raw[lo + d])),
                            one,
                            bool(half),
                        )
                    )
                    for d in range_constexpr(2)
                    for half in range_constexpr(2)
                ]
                t.store(
                    fx.Vector.from_elements(
                        [
                            fx.BFloat16(pairs[i][j])
                            for i in range_constexpr(4)
                            for j in range_constexpr(2)
                        ],
                        fx.BFloat16,
                    )
                )
            else:
                t.store(raws[ks].bitcast(fx.BFloat16))
            return t

        # ==================== MAIN LOOP ====================
        def score_panel(p, panel, raws, run_max):
            """MFMA one 16-token panel against this wave's feature tiles.

            Consumes K loads issued earlier (`raws`) and updates the caller's
            per-feature running max in place. This is the whole arithmetic body
            of the kernel: everything else is address math and plumbing.
            """
            accs = []
            for i in range_constexpr(FTW):
                acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                acc.store(zero4)
                accs.append(acc)
            for ks in range_constexpr(KSTEPS):
                kf = convert_k(raws, ks)
                for i in range_constexpr(FTW):
                    fx.gemm(mma_atom, accs[i], kf, q_operand(i, ks), accs[i])

            # This lane holds tokens 16*panel + 4*g + r of the wave's slice,
            # i.e. tw_tok further along the page's token axis. The page's base
            # token is a *global* position -- the cutoff it is compared against
            # counts the whole context, not this rank's share of it.
            # The scale is NOT applied here: it is positive, so it commutes with
            # max, and folding it in once in the epilogue turns PANELS*FT*4 = 64
            # multiplies per page into FT = 2.
            tok_base = (
                global_block(p) * fx.Int32(PAGE)
                + fx.Int32(MFMA_M * panel)
                + tw_tok
                + g * fx.Int32(4)
            )
            for i in range_constexpr(FTW):
                tok, _ = tok_head_of(i)
                causal_len = seq_len - fx.Int32(S) + tok + fx.Int32(1)
                v = fx.Vector(fx.memref_load_vec(accs[i]))
                for r in range_constexpr(4):
                    ok = (tok_base + fx.Int32(r)) < causal_len
                    run_max[i] = run_max[i].maximumf(
                        ok.select(fx.Float32(v[r]), neg_inf)
                    )

        def page_loop(p, page, carry, nxt_page):
            """Run this wave's panels of one page. Returns the next page's carry.

            At TOKEN_WAVES > 1 a wave walks only PANELS_W = 8 // TOKEN_WAVES of
            the page's panels; the others are another wave's, and the partial
            maxes meet in LDS in the epilogue.

            `carry` holds the next panel's K, issued while the *previous* page
            was still doing MFMAs -- so the wait for it is already paid by the
            time we get here. Symmetrically, at the last panel we issue
            `nxt_page`'s opening loads instead of idling and hand them back, so
            the wave never drains and refills at a page boundary.

            FP8 keeps three panels in flight, capped by the wave's token
            slice, and carries that window across pages. Scheduling boundaries
            prevent LLVM from collapsing it back to the one-panel source
            schedule. BF16 retains the original one-panel software queue.
            """
            if const_expr(1 <= sched <= 3):
                fx.rocdl.iglp_opt(sched - 1)
            run_max = [neg_inf for _ in range_constexpr(FTW)]
            if const_expr(fp8):
                # A scheduling boundary makes the three-panel window real;
                # changing only the source queue depth lets LLVM undo it.
                depth = min(3, PANELS_W)
                queue = list(carry) if carry is not None else []
                for panel in range_constexpr(len(queue), depth):
                    queue.append(
                        [issue_k(page, panel, i) for i in range_constexpr(K_LOADS)]
                    )
                out_carry = []
                for panel in range_constexpr(PANELS_W):
                    raws = queue.pop(0)
                    fx.rocdl.sched_barrier(0)
                    score_panel(p, panel, raws, run_max)
                    fx.rocdl.sched_barrier(0)
                    future = panel + depth
                    if const_expr(future < PANELS_W):
                        queue.append(
                            [issue_k(page, future, i) for i in range_constexpr(K_LOADS)]
                        )
                    elif nxt_page is not None:
                        out_carry.append(
                            [
                                issue_k(nxt_page, future - PANELS_W, i)
                                for i in range_constexpr(K_LOADS)
                            ]
                        )
            else:
                queue = (
                    list(carry)
                    if carry is not None
                    else [[issue_k(page, 0, i) for i in range_constexpr(K_LOADS)]]
                )
                out_carry = []

                for panel in range_constexpr(PANELS_W):
                    raws = queue.pop(0)
                    # Issue panel n+1's loads before consuming panel n's, so they
                    # are in flight during the MFMAs rather than waited on.
                    if const_expr(panel + 1 < PANELS_W):
                        queue.append(
                            [
                                issue_k(page, panel + 1, i)
                                for i in range_constexpr(K_LOADS)
                            ]
                        )
                    elif nxt_page is not None:
                        out_carry.append(
                            [issue_k(nxt_page, 0, i) for i in range_constexpr(K_LOADS)]
                        )
                    score_panel(p, panel, raws, run_max)

            if const_expr(sched == 4):
                # Ask for the order the source already has -- one panel's loads,
                # then one panel's MFMAs -- rather than whatever the scheduler
                # hoisted to. Group id 0 throughout: these are one pipeline, and
                # the calls are read in order.
                for _ in range_constexpr(PANELS_W):
                    fx.rocdl.sched_group_barrier(_SCHED_VMEM_RD, K_LOADS, 0)
                    fx.rocdl.sched_group_barrier(_SCHED_MFMA, FTW * KSTEPS, 0)

            return run_max, out_carry

        # ==================== EPILOGUE ====================
        def fold_g(run_max):
            """Fold each feature's max across g: lanes {u, u+16, u+32, u+48}.

            XOR 1/2/4/8 would mix *features*, not tokens. Afterwards
            every lane holds the tile's max for feature `u`, replicated over g.
            """
            out = []
            for i in range_constexpr(FTW):
                m = run_max[i]
                for sh in (16, 32):
                    m = m.maximumf(m.shuffle_xor(sh, WAVE))
                out.append(m)
            return out

        def reduce_tokens(run_max):
            """Combine the TOKEN_WAVES partial maxes for this page through LDS.

            Each of the waves splitting the page's token axis has a max over its
            own panels only; they have to meet somewhere, and LDS is the only
            place waves meet. Cheap because it happens once per page, after the
            g-fold, on FTW values -- not once per panel.

            Every lane writes, not just the g == 0 holders, so the write is one
            contiguous 256 B run per (wave, tile) and the read is lane-local.
            """
            if const_expr(TOKEN_WAVES == 1):
                return run_max
            red = lds.red.ptr
            if const_expr(pages_per_wave > 1):
                # The slots are reused every page: do not overwrite them until
                # the previous page's readers are through.
                gpu.barrier()
            base = wave * fx.Int32(FTW * WAVE) + lane
            for i in range_constexpr(FTW):
                fx.ptr_store(run_max[i], red + (base + fx.Int32(i * WAVE)))
            gpu.barrier()
            out = []
            for i in range_constexpr(FTW):
                m = run_max[i]
                for t in range_constexpr(TOKEN_WAVES):
                    # Same fw (same features) and same pw (same page), other tw.
                    w_other = (
                        fw
                        + fx.Int32(FEAT_WAVES * t)
                        + pw * fx.Int32(FEAT_WAVES * TOKEN_WAVES)
                    )
                    m = m.maximumf(
                        fx.ptr_load(
                            red
                            + (w_other * fx.Int32(FTW * WAVE) + fx.Int32(i * WAVE))
                            + lane
                        )
                    )
                out.append(m)
            return out

        def store_page(p, run_max, valid):
            """Reduce one page's FTW scores across waves, scale them, and store.

            `valid` says whether page p really exists. It gates the store, not
            the work: page ids are clamped in the prologue, so an out-of-range
            page reads real memory and computes a value nobody keeps. Branching
            around the main loop instead would need a second copy of it for the
            guarded case, and that copy cost more (1068 vs 622 instructions per
            page, measured) than the <=CHUNK-1 wasted pages it would save.
            """
            folded = reduce_tokens(fold_g(run_max))
            for i in range_constexpr(FTW):
                # Scale folded in here rather than per accumulator: it is
                # positive so max(s*x) == s*max(x), and -inf * s is still -inf,
                # so a fully masked page still stores -inf.
                m = folded[i] * f32_scale

                tok, head = tok_head_of(i)
                row = b * fx.Int32(S) + tok
                addr = head * i32_stride_s_h + row * i32_stride_s_b + p * i32_stride_s_k
                # One writer per feature; padding columns write nothing (this
                # also covers the tiles a ragged feature split hands the last
                # wave), and neither do pages past the end of this request.
                is_writer = (g == fx.Int32(0)) & (feature_of(i) < fx.Int32(F))
                if const_expr(TOKEN_WAVES > 1):
                    # All TOKEN_WAVES waves now hold the same value; pick one.
                    is_writer = is_writer & (tw == fx.Int32(0))
                if valid is not None:
                    is_writer = is_writer & valid

                def _store(_a=addr, _v=m):
                    fx.add_offset(fx.get_iter(s_buf), _a).store(_v)

                @flyc.jit
                def _guarded(_p=is_writer, _w=_store):
                    if _p:
                        _w()

                _guarded()

        # Deferring these stores to the end of the wave -- holding all
        # pages_per_wave results live and emitting them back to back -- was
        # tried and is a wash (paired A/B, +-1% with inconsistent sign across
        # six shape/dtype pairs, though it does move VGPRs 124 -> 108 without
        # changing the occupancy bucket). The write bursts a wave issues are
        # already far apart in time relative to the K stream between them, so
        # clustering them buys nothing; what did pay was making each store
        # cover a full cache line, which is alloc_score's job.
        #
        # ==================== DRIVER ====================
        # One straight-line body over this wave's pages, with the K pipeline
        # carried across each boundary. The only branch is the one below: a wave
        # whose *first* page is out of range skips everything, which is the case
        # worth branching on -- a short request in a ragged batch would
        # otherwise stream whole chunks of K it can never use.
        def body():
            carry = None
            for j in range_constexpr(pages_per_wave):
                nxt = page_buffers[j + 1] if j + 1 < pages_per_wave else None
                run_max, carry = page_loop(pages[j], page_buffers[j], carry, nxt)
                # With one page per wave and the guard below in place, that
                # guard already proved the page in range and the extra mask
                # would be dead.
                valid = (
                    None
                    if const_expr(pages_per_wave == 1 and TOKEN_WAVES == 1)
                    else pages[j] < limit
                )
                store_page(pages[j], run_max, valid)

        if const_expr(TOKEN_WAVES > 1):
            # No skip guard: it is wave-divergent, and store_page's barrier has
            # to be reached by every wave in the CTA. The skipped waves instead
            # run the body on a clamped page and drop the result at the store.
            body()
        else:

            @flyc.jit
            def _maybe(_pred=(pages[0] < limit), _w=body):
                if _pred:
                    _w()

            _maybe()

    @flyc.jit
    def launch(
        arg_q: fx.Pointer,
        arg_k: fx.Pointer,
        arg_score: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_work: fx.Pointer,
        i32_stride_q_n: fx.Int32,
        i32_stride_q_h: fx.Int32,
        i32_stride_k_blk: fx.Int32,
        i32_stride_k_pos: fx.Int32,
        i32_stride_s_h: fx.Int32,
        i32_stride_s_b: fx.Int32,
        i32_stride_s_k: fx.Int32,
        i32_stride_bt_b: fx.Int32,
        f32_scale: fx.Float32,
        i32_batch: fx.Int32,
        i32_chunks: fx.Int32,
        stream: fx.Stream,
    ):
        score_kernel(
            arg_q,
            arg_k,
            arg_score,
            arg_bt,
            arg_work,
            i32_batch,
            i32_stride_q_n,
            i32_stride_q_h,
            i32_stride_k_blk,
            i32_stride_k_pos,
            i32_stride_s_h,
            i32_stride_s_b,
            i32_stride_s_k,
            i32_stride_bt_b,
            f32_scale,
        ).launch(
            grid=(fx.Int64(i32_batch), fx.Int64(i32_chunks), 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    # Freeing VGPRs is not the same as spending them on occupancy: the
    # scheduler reinvests them in more load hoisting and lands back where it
    # started (q_to_lds alone measured 128 -> 136 VGPRs, i.e. one wave *worse*).
    # This is the channel that actually forces the issue.
    if cfg.waves_per_eu:
        launch.compile_hints["waves_per_eu"] = cfg.waves_per_eu

    return launch, CHUNK


_CACHE = {}


def _get(S, H, fp8, cfg, device):
    key = (S, H, fp8, cfg, device)  # IndexScoreConfig is frozen, hence hashable
    if key not in _CACHE:
        _CACHE[key] = build_index_score(S, H, fp8, cfg)
    return _CACHE[key]


def shuffle_cache(cache):
    """Reorder a [pages, 128, 128] K cache into the kernel's shuffled layout.

    Element (row, k) moves to the slot the lane that wants it will read:

        load slot j = panel * K_LOADS + i   (panel = row // 16)
        lane        = 16 * g + u            (u = row % 16)
        position    = (j * WAVE + lane) * CHUNK_ELEMS + v

    where (g, v) come from inverting the kernel's k_offset map. Derived from
    that map rather than written independently, because the two must agree
    exactly -- a mismatch here is silent wrong numbers, not a crash.

    This is a one-off cost paid when the cache is written, not per decode step.
    """
    import torch

    fp8 = cache.dtype != torch.bfloat16
    npages = cache.shape[0]
    chunk_elems = 16 if fp8 else 8
    k_loads = KSTEPS // 2 if fp8 else KSTEPS

    dev = cache.device
    row = torch.arange(PAGE, device=dev)
    kk = torch.arange(HEAD_DIM, device=dev)
    # For each (row, k), which lane/slot/offset does it belong to?
    panel, u = row // MFMA_M, row % MFMA_M
    if fp8:
        # k = 64*i + 16*g + 8*(ks%2), v in [0,16)
        i = kk // 64
        g = (kk % 64) // 16
        v = kk % 16
    else:
        # k = 32*ks + 8*g, v in [0,8)
        i = kk // MFMA_K
        g = (kk % MFMA_K) // 8
        v = kk % 8
    slot = panel[:, None] * k_loads + i[None, :]
    fx_lane = 16 * g[None, :] + u[:, None]
    dst = (slot * WAVE + fx_lane) * chunk_elems + v[None, :]

    flat = cache.reshape(npages, PAGE * HEAD_DIM)
    out = torch.empty_like(flat)
    out.scatter_(1, dst.reshape(1, -1).expand(npages, -1), flat)
    return out.reshape(cache.shape)


def make_work_map(seq_lens, max_block, chunk, out=None, world: int = 1, rank: int = 0):
    """Dispatch order -> work item, packed so the holes all land at the end.

    The grid is `batch x ceil(max_block/chunk)`, sized by the longest request.
    A ragged batch leaves holes in that rectangle, and where the holes sit
    decides what they cost: bunched together at the end they are ~0.2 us per
    1000, interleaved with real work along the dispatch axis they are ~7 us per
    1000, because each takes a launch slot and retires before the machine can
    build up K loads in flight. Measured at b32_q8_s128k fp8, a batch with one
    4x-length outlier ran 371 us against the uniform 159 us on identical bytes.

    So: number the real work 0..total-1 and give item n to the n-th dispatched
    CTA. Row n of the result is

        [0] = (b << 16) | c    request, and which chunk of it; 0 for a hole
        [1] = seq_lens[b]      the kernel needs the length anyway, and keeping
                               it here holds the lookup to one cache line

    A hole reads as (0, 0), i.e. seq_len 0, so the kernel's existing tail guard
    retires it -- no extra branch, and page ids clamp to a legal address.

    Every step is a device op on `seq_lens`, so this is cudagraph-capturable and
    stays correct for whatever lengths a replay supplies. Call it once per
    decode step; the lengths do not change between layers.

    `world`/`rank` describe a context-parallel shard, and `max_block` is then
    the LOCAL bound `ceil(global_blocks / world)`. Only the per-request chunk
    count changes: a rank holds `ceil((nblk - rank) / world)` of the blocks, so
    that is what it is given chunks for. Row [1] stays the full `seq_lens[b]`,
    because the kernel still needs the global length for the causal cutoff and
    re-derives its own local count from it with this same formula -- the two
    are deliberately the same arithmetic in two places, and must stay so.
    """
    import torch

    if world < 1 or not 0 <= rank < world or chunk < 1 or max_block < 1:
        raise ValueError("invalid shard, chunk or max_block")
    _validate_tensor(seq_lens, "seq_lens", (torch.int32,), 1, seq_lens.device)
    if seq_lens.stride() != (1,):
        raise ValueError("seq_lens must be contiguous")
    batch = seq_lens.shape[0]
    chunks = (max_block + chunk - 1) // chunk
    if batch > 0xFFFF or chunks > _MAX_CHUNKS or batch * chunks * 8 > 0xFFFFFFFF:
        raise ValueError("work_map: packed IDs or address span out of range")
    if out is not None:
        _validate_map(out, batch * chunks, seq_lens.device, exact=True)
    if out is None:
        out = torch.empty(
            (batch * chunks, 2), dtype=torch.int32, device=seq_lens.device
        )

    lens = seq_lens.to(torch.int32)
    nblk = (lens + (PAGE - 1)) // PAGE
    if world > 1:
        nblk = (
            (nblk - rank)
            .clamp_(min=0)
            .add_(world - 1)
            .div_(world, rounding_mode="floor")
        )
    nch = (nblk + (chunk - 1)) // chunk  # work items this request owns
    cum = torch.cumsum(nch, 0, dtype=torch.int32)  # inclusive
    n = torch.arange(batch * chunks, dtype=torch.int32, device=seq_lens.device)
    # First i with cum[i] > n owns item n; it is `batch` once n runs past the
    # end. Requests of length zero have cum[i] == cum[i-1] and are skipped.
    b = torch.searchsorted(cum, n, right=True).to(torch.int32)
    live = b < batch
    bc = torch.where(live, b, torch.zeros_like(b)).long()
    c = n - (cum - nch)[bc]  # cum - nch is the exclusive prefix
    zero = torch.zeros_like(n)
    out[:, 0] = torch.where(live, (bc.to(torch.int32) << 16) | c, zero)
    out[:, 1] = torch.where(live, lens[bc], zero)
    return out


def make_spread_work_map(seq_lens, rows, cu, out=None, world: int = 1, rank: int = 0):
    """The one-page-per-wave map, with the last CU round spread thin.

    make_work_map gives every CTA a full chunk of four pages, which is fine
    until the live CTAs spill past a multiple of the CU count: the spill lands
    as a second full CTA on a few CUs and the launch waits for them. Measured
    at batch 1 in the served layers bench, 1024 -> 1028 pages (256 -> 257 CTAs)
    costs 5.4 -> 6.7 us, while 1024 -> 1025 (the extra CTA holding one page)
    costs nothing; 2048 -> 2052 is 8.4 -> 9.9. It is the pages that pile onto
    one CU, not the CTA count -- eight-wave CTAs (half as many, same pages per
    CU) and fewer, deeper CTAs were both measured and both lose.

    So a step needing r CU rounds keeps r - 1 rounds of full four-page CTAs
    and spreads what is left over the last round's CTAs, 1..4 pages each and
    the larger ones first, so no CU holds more than ceil(pages / cu) plus
    rounding. Spreading every round evenly instead balances as well but pays
    a CTA prologue per page or two: -6% at S8 H4, 3300 pages.

    Row n is [0] = (b << 16) | first page, [1] = seq_len | (pages << 24); holes
    are (0, 0) and packed at the tail as before. The full CTAs come first in
    dispatch order, taken from the front of the batch; then each request's
    remainder, over a share of the last round proportional to its size. Each
    request can round its share up by one CTA, so one slot per live request
    is held back. `rows` is the grid, with one spare row per request (see
    _grid_chunks), which is what keeps four pages per CTA enough at full
    capacity. Device ops only: cudagraph-capturable.
    """
    import torch

    batch = seq_lens.shape[0]
    dev = seq_lens.device
    if out is None:
        out = torch.empty((rows, 2), dtype=torch.int32, device=dev)
    else:
        _validate_map(out, rows, dev, exact=True)
    lens = seq_lens.to(torch.int32)
    nblk = (lens + (PAGE - 1)) // PAGE
    if world > 1:
        nblk = (
            (nblk - rank)
            .clamp_(min=0)
            .add_(world - 1)
            .div_(world, rounding_mode="floor")
        )
    nblk = nblk.to(torch.int64)
    total = nblk.sum()
    live_reqs = (nblk > 0).sum()
    rounds = ((total + 4 * live_reqs + 4 * cu - 1) // (4 * cu)).clamp(min=1)
    # Full four-page CTAs for every round but the last, front of the batch first.
    fits = nblk // 4
    fits_cum = torch.cumsum(fits, 0)
    full = torch.minimum(fits_cum[-1], (rounds - 1) * cu)
    a = torch.minimum((full - (fits_cum - fits)).clamp(min=0), fits)
    # The rest, spread over what is left of the last round.
    rest = nblk - 4 * a
    rest_total = total - 4 * full
    thin = torch.minimum(rounds * cu - full - live_reqs, rest_total)
    thin = torch.minimum(thin, rows - full - live_reqs).clamp(min=1)
    k = (rest * thin + rest_total.clamp(min=1) - 1) // rest_total.clamp(min=1)
    a_cum = torch.cumsum(a, 0)
    k_cum = torch.cumsum(k, 0)

    n = torch.arange(rows, dtype=torch.int64, device=dev)
    in_full = n < full
    bf = torch.searchsorted(a_cum, n, right=True).clamp(max=batch - 1)
    first_f = 4 * (n - (a_cum - a)[bf])
    nt = n - full
    bt = torch.searchsorted(k_cum, nt, right=True)
    live = in_full | ((nt >= 0) & (bt < batch))
    bt = bt.clamp(max=batch - 1)
    it = nt - (k_cum - k)[bt]
    kt = k[bt].clamp(min=1)
    base, extra = rest[bt] // kt, rest[bt] % kt
    first_t = 4 * a[bt] + it * base + torch.minimum(it, extra)
    b = torch.where(in_full, bf, bt)
    first = torch.where(in_full, first_f, first_t)
    span = torch.where(in_full, 4, base + (it < extra).to(torch.int64))
    out[:, 0] = torch.where(live, (b << 16) | first, 0).to(torch.int32)
    out[:, 1] = torch.where(live, lens[b].to(torch.int64) | (span << 24), 0).to(
        torch.int32
    )
    return out


def build_spread_map(batch: int, world: int = 1, rank: int = 0):
    """make_spread_work_map as one kernel: one thread per row, batch unrolled.

    The torch version is ~40 small ops -- ~1 ms of host launches per step when
    built eagerly, ~190 us of GPU time inside a graph, against a scorer that
    spends ~0.5 ms per step over all 57 layers. Every thread redoes the O(batch)
    per-request prefix arithmetic for its own row instead of sharing it, which
    at batch <= SPREAD_MAX_BATCH is cheaper than a second pass or a barrier.
    Bit-identical to make_spread_work_map; the tests hold the two together.
    """
    if not 1 <= batch <= SPREAD_MAX_BATCH:
        raise ValueError(f"spread map batch must be in [1, {SPREAD_MAX_BATCH}]")
    name = f"m3_index_spread_map_B{batch}"
    if world > 1:
        name += f"_cp{world}r{rank}"

    @flyc.kernel(name=name, known_block_size=[MAP_THREADS, 1, 1])
    def map_kernel(
        arg_lens: fx.Pointer, arg_out: fx.Pointer, i32_rows: fx.Int32, i32_cu: fx.Int32
    ):
        n = fx.Int32(gpu.block_id("x")) * fx.Int32(MAP_THREADS) + fx.Int32(
            gpu.thread_id("x")
        )
        lens_buf = ptr_buf_tensor(arg_lens, fx.Int32)
        out_buf = ptr_buf_tensor(arg_out, fx.Int32)
        zero, one, four = fx.Int32(0), fx.Int32(1), fx.Int32(4)

        def lo(x, y):
            return (x < y).select(x, y)

        def hi(x, y):
            return (x < y).select(y, x)

        lens = [
            fx.Int32(fx.add_offset(fx.get_iter(lens_buf), fx.Int32(b)).load(T.i32))
            for b in range_constexpr(batch)
        ]
        nblk = []
        for b in range_constexpr(batch):
            pages = (lens[b] + fx.Int32(PAGE - 1)) // fx.Int32(PAGE)
            if const_expr(world > 1):
                pages = (hi(pages - fx.Int32(rank), zero) + fx.Int32(world - 1)) // (
                    fx.Int32(world)
                )
            nblk.append(pages)

        total, live, fits_total = zero, zero, zero
        for b in range_constexpr(batch):
            total = total + nblk[b]
            live = live + (nblk[b] > zero).select(one, zero)
            fits_total = fits_total + nblk[b] // four
        cu = i32_cu
        rounds = hi((total + four * live + four * cu - one) // (four * cu), one)
        full = lo(fits_total, (rounds - one) * cu)
        rest_total = total - four * full
        thin = lo(lo(rounds * cu - full - live, rest_total), i32_rows - full - live)
        thin = hi(thin, one)
        denom = hi(rest_total, one)

        nt = n - full
        fits_before, a_before, k_before = zero, zero, zero
        hit, hit_b, hit_first, hit_span, hit_len = zero, zero, zero, zero, zero
        for b in range_constexpr(batch):
            fits = nblk[b] // four
            a = lo(hi(full - fits_before, zero), fits)
            fits_before = fits_before + fits
            rest = nblk[b] - four * a
            # rest * thin can pass 2**31 (65535 pages x ~1M rows): widen.
            k = fx.Int32(
                (fx.Int64(rest) * fx.Int64(thin) + fx.Int64(denom) - fx.Int64(1))
                // fx.Int64(denom)
            )
            in_full = (n >= a_before) & (n < a_before + a)
            in_thin = (nt >= k_before) & (nt < k_before + k)
            it = nt - k_before
            kk = hi(k, one)
            base = rest // kk
            extra = rest - base * kk
            first_t = four * a + it * base + lo(it, extra)
            span_t = base + (it < extra).select(one, zero)
            first = in_full.select(four * (n - a_before), first_t)
            span = in_full.select(four, span_t)
            take = in_full | in_thin
            hit = take.select(one, hit)
            hit_b = take.select(fx.Int32(b), hit_b)
            hit_first = take.select(first, hit_first)
            hit_span = take.select(span, hit_span)
            hit_len = take.select(lens[b], hit_len)
            a_before = a_before + a
            k_before = k_before + k

        live_row = hit > zero
        row0 = live_row.select((hit_b << fx.Int32(16)) | hit_first, zero)
        row1 = live_row.select(hit_len | (hit_span << fx.Int32(24)), zero)

        def _store(_n=n, _r0=row0, _r1=row1):
            base = fx.get_iter(out_buf)
            fx.add_offset(base, _n * fx.Int32(2)).store(_r0)
            fx.add_offset(base, _n * fx.Int32(2) + fx.Int32(1)).store(_r1)

        @flyc.jit
        def _guarded(_p=(n < i32_rows), _w=_store):
            if _p:
                _w()

        _guarded()

    @flyc.jit
    def launch(
        arg_lens: fx.Pointer,
        arg_out: fx.Pointer,
        i32_rows: fx.Int32,
        i32_cu: fx.Int32,
        i32_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        map_kernel(arg_lens, arg_out, i32_rows, i32_cu).launch(
            grid=(fx.Int64(i32_blocks), 1, 1),
            block=(MAP_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def build_chunk_map(batch: int, chunk: int, world: int = 1, rank: int = 0):
    """make_work_map as one kernel: one thread per row, batch unrolled.

    Same reason as :func:`build_spread_map` -- the torch version is a dozen
    small ops whose cost is launch floor, ~50-60 us of GPU time inside a graph
    against a scorer that can be 4 us -- and the same shape of answer: each
    thread redoes the O(batch) prefix sum for its own row.
    Bit-identical to `make_work_map`; the tests hold the two together.
    """
    if not 1 <= batch <= SPREAD_MAX_BATCH:
        raise ValueError(f"chunk map batch must be in [1, {SPREAD_MAX_BATCH}]")
    name = f"m3_index_chunk_map_B{batch}_C{chunk}"
    if world > 1:
        name += f"_cp{world}r{rank}"

    @flyc.kernel(name=name, known_block_size=[MAP_THREADS, 1, 1])
    def map_kernel(arg_lens: fx.Pointer, arg_out: fx.Pointer, i32_rows: fx.Int32):
        n = fx.Int32(gpu.block_id("x")) * fx.Int32(MAP_THREADS) + fx.Int32(
            gpu.thread_id("x")
        )
        lens_buf = ptr_buf_tensor(arg_lens, fx.Int32)
        out_buf = ptr_buf_tensor(arg_out, fx.Int32)
        zero, one = fx.Int32(0), fx.Int32(1)

        hit, hit_b, hit_c, hit_len = zero, zero, zero, zero
        before = zero
        for b in range_constexpr(batch):
            length = fx.Int32(
                fx.add_offset(fx.get_iter(lens_buf), fx.Int32(b)).load(T.i32)
            )
            pages = (length + fx.Int32(PAGE - 1)) // fx.Int32(PAGE)
            if const_expr(world > 1):
                avail = pages - fx.Int32(rank)
                avail = (avail < zero).select(zero, avail)
                pages = (avail + fx.Int32(world - 1)) // fx.Int32(world)
            nch = (pages + fx.Int32(chunk - 1)) // fx.Int32(chunk)
            take = (n >= before) & (n < before + nch)
            hit = take.select(one, hit)
            hit_b = take.select(fx.Int32(b), hit_b)
            hit_c = take.select(n - before, hit_c)
            hit_len = take.select(length, hit_len)
            before = before + nch

        live = hit > zero
        row0 = live.select((hit_b << fx.Int32(16)) | hit_c, zero)
        row1 = live.select(hit_len, zero)

        def _store(_n=n, _r0=row0, _r1=row1):
            base = fx.get_iter(out_buf)
            fx.add_offset(base, _n * fx.Int32(2)).store(_r0)
            fx.add_offset(base, _n * fx.Int32(2) + fx.Int32(1)).store(_r1)

        @flyc.jit
        def _guarded(_p=(n < i32_rows), _w=_store):
            if _p:
                _w()

        _guarded()

    @flyc.jit
    def launch(
        arg_lens: fx.Pointer,
        arg_out: fx.Pointer,
        i32_rows: fx.Int32,
        i32_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        map_kernel(arg_lens, arg_out, i32_rows).launch(
            grid=(fx.Int64(i32_blocks), 1, 1),
            block=(MAP_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _run_chunk_map(seq_lens, rows, chunk, out, world, rank):
    import torch

    batch = seq_lens.shape[0]
    key = ("chunk_map", batch, chunk, world, rank, seq_lens.device.index)
    if key not in _CACHE:
        _CACHE[key] = build_chunk_map(batch, chunk, world, rank)
    with torch.cuda.device(seq_lens.device):
        _run_compiled(
            _CACHE[key],
            ptr_arg(seq_lens, fx.Int32),
            ptr_arg(out, fx.Int32),
            rows,
            (rows + MAP_THREADS - 1) // MAP_THREADS,
            torch.cuda.current_stream(seq_lens.device).cuda_stream,
        )
    return out


def _run_spread_map(seq_lens, rows, cu, out, world, rank):
    import torch

    batch = seq_lens.shape[0]
    key = ("spread_map", batch, world, rank, seq_lens.device.index)
    if key not in _CACHE:
        _CACHE[key] = build_spread_map(batch, world, rank)
    with torch.cuda.device(seq_lens.device):
        _run_compiled(
            _CACHE[key],
            ptr_arg(seq_lens, fx.Int32),
            ptr_arg(out, fx.Int32),
            rows,
            cu,
            (rows + MAP_THREADS - 1) // MAP_THREADS,
            torch.cuda.current_stream(seq_lens.device).cuda_stream,
        )
    return out


def work_map_size(
    batch: int, max_block: int, S: int = 0, H: int = 0, cfg=None, *, device=None
) -> int:
    """Rows `make_work_map` will produce -- i.e. the grid, one row per CTA.

    The buffer itself is `[rows, 2]` int32; a caller sizing a persistent one for
    a cudagraph must use work_map_capacity (exact grid size is non-monotonic) and hand
    `make_work_map` a `buf[:rows]` slice, which stays packed and so stays a legal
    kernel argument.

    Exists so that caller does not re-derive the chunk size: it depends on the
    resolved `pages_per_wave`, which depends on the CU count, so a duplicated
    copy would go wrong the first time this runs on a different chip.

    Under context parallelism `max_block` is the local bound, so this is the
    shard's grid -- roughly 1/world of the unsharded one. Pass S/H to enable
    automatic token splitting. Omitting them keeps the legacy unsplit map;
    score_flydsl accepts that exact map with the matching fallback geometry.
    """
    cfg = cfg or IndexScoreConfig()
    _validate_bounds(batch, max_block, cfg, resolved=False)
    cfg = resolve_config(batch, max_block, cfg, S, H, device=device)
    return _validate_bounds(batch, max_block, cfg)


def work_map_capacity(
    max_batch: int, max_block: int, S: int = 0, H: int = 0, cfg=None
) -> int:
    """Persistent rows for every batch/block bound within the envelope.

    Use the smallest CTA chunk reachable by automatic depth/token splitting,
    independent of CU count. Explicit configurations retain their tighter
    geometry-specific envelope.

    Include Q1 even when S is a larger query bound: serving callers can reuse a
    maximum-query allocation for decode replays, and asking with the largest S
    and launching Q1 is the natural mistake -- it costs 4x on the buffer (128 x
    8192 blocks: 262144 rows against 262144*4) and buys that the mistake still
    works. Nothing in aiter or ATOM pre-allocates this yet, so the cost is
    currently notional and the safety is the point.

    Not, however, where `max_block` cannot pack at one page per chunk: there
    the block count is itself what runs into the packing limit, and the token
    split that would need it cannot fire anyway -- `resolve_config` only takes
    it when `batch * max_block` fits the CU count, which at that width it never
    does. Widening there rejected bounds this used to accept (`max_block`
    262144 at batch 128), for a geometry no launch can reach.
    """
    cfg = cfg or IndexScoreConfig()
    _validate_bounds(max_batch, max_block, cfg, resolved=False)
    auto = _auto_token_split(cfg)
    split = auto and max_block <= _MAX_CHUNKS
    geometry = replace(
        cfg,
        token_waves=WAVES if split else (cfg.token_waves or 1),
        pages_per_wave=cfg.pages_per_wave or 1,
        nt_k=max(cfg.nt_k, 0),  # cache policy never changes the geometry
        spread=max(cfg.spread, 0),
    )
    rows = _validate_bounds(max_batch, max_block, geometry)
    # ...or the spread map the served path may resolve to instead, which only
    # batches up to SPREAD_MAX_BATCH can (its rows grow with the batch).
    spread_batch = min(max_batch, SPREAD_MAX_BATCH)
    if auto and _spread_fits(spread_batch, max_block, cfg):
        spread = replace(geometry, token_waves=1, pages_per_wave=1, spread=1)
        rows = max(rows, _validate_bounds(spread_batch, max_block, spread))
    return rows


def build_work_map(seq_lens, max_block, S: int = 0, H: int = 0, cfg=None, out=None):
    """`make_work_map` with the chunk size resolved for you.

    This is the entry point a serving caller wants. `make_work_map` takes the
    chunk size as a number, which means a caller that builds the map in one
    place and launches the kernel in another has to resolve the config twice and
    keep the two agreeing -- and a disagreement is not a crash, it is a map the
    kernel indexes with the wrong stride.

    `out` may be larger than needed (a persistent worst-case buffer sized by
    `work_map_capacity`); it is sliced down here, and a short one is an error rather
    than a silently truncated grid.
    """
    import torch

    _validate_tensor(seq_lens, "seq_lens", (torch.int32,), 1, seq_lens.device)
    cfg = cfg or IndexScoreConfig()
    _validate_bounds(seq_lens.shape[0], max_block, cfg, resolved=False)
    cfg = resolve_config(
        seq_lens.shape[0],
        max_block,
        cfg,
        S,
        H,
        device=seq_lens.device,
    )
    _validate_bounds(seq_lens.shape[0], max_block, cfg)
    if (
        S
        and H
        and not selection_filter(
            S,
            H,
            cfg,
            arch=torch.cuda.get_device_properties(seq_lens.device).gcnArchName,
        )
    ):
        raise ValueError(f"illegal config for S={S} H={H}: {cfg}")
    rows = seq_lens.shape[0] * _grid_chunks(max_block, cfg)
    if out is not None:
        _validate_map(out, rows, seq_lens.device)
        out = out[:rows]
    return _make_map(seq_lens, max_block, cfg, rows, out)


def _make_map(seq_lens, max_block, cfg, rows, out=None):
    # The shard comes from the same cfg the kernel will be compiled against, so
    # the map and the kernel cannot disagree about which blocks this rank owns.
    import torch

    batch = seq_lens.shape[0]
    if batch and batch <= SPREAD_MAX_BATCH and seq_lens.stride() == (1,):
        if out is None:
            out = torch.empty((rows, 2), dtype=torch.int32, device=seq_lens.device)
        else:
            _validate_map(out, rows, seq_lens.device, exact=True)
        if cfg.spread > 0:
            cu = torch.cuda.get_device_properties(seq_lens.device).multi_processor_count
            return _run_spread_map(seq_lens, rows, cu, out, cfg.cp_world, cfg.cp_rank)
        return _run_chunk_map(
            seq_lens, rows, work_chunk(cfg), out, cfg.cp_world, cfg.cp_rank
        )
    if cfg.spread > 0:  # no torch fallback: the spread row packing is kernel-only
        raise ValueError(
            f"spread map needs a contiguous batch of 1..{SPREAD_MAX_BATCH}"
        )
    return make_work_map(
        seq_lens,
        max_block,
        work_chunk(cfg),
        out=out,
        world=cfg.cp_world,
        rank=cfg.cp_rank,
    )


def _span(t):
    """Elements addressed from data_ptr(), including padding (not storage size)."""
    return 1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride()))


def _validate_tensor(t, name, dtype, ndim, device, alignment=4, bounded=True):
    import torch

    if not isinstance(t, torch.Tensor) or t.dtype not in dtype or t.ndim != ndim:
        raise ValueError(f"{name}: invalid dtype or rank")
    if t.device != device or t.device.type != "cuda":
        raise ValueError(f"{name}: expected CUDA tensor on {device}")
    if any(n < 1 for n in t.shape) or any(
        st <= 0 or st > 0x7FFFFFFF for st in t.stride()
    ):
        raise ValueError(f"{name}: empty shape or unsupported stride")
    if t.data_ptr() % alignment:
        raise ValueError(f"{name}: base must be {alignment}-byte aligned")
    if bounded and (_span(t) > 0x7FFFFFFF or _span(t) * t.element_size() > 0xFFFFFFFF):
        raise ValueError(f"{name}: strided address span exceeds 32-bit addressing")


def _validate_map(out, rows, device, exact=False):
    import torch

    _validate_tensor(out, "work_map", (torch.int32,), 2, device)
    if out.shape[1] != 2 or out.stride() != (2, 1) or out.shape[0] < rows:
        raise ValueError(f"work_map: expected packed [at least {rows}, 2] int32")
    if exact and out.shape[0] != rows:
        raise ValueError(f"work_map: expected exact grid slice [{rows}, 2]")


def _validate_bounds(batch, max_block, cfg, *, resolved=True) -> int:
    """Validate packed/address arithmetic and return the exact grid row count."""
    integer_fields = (
        cfg.feat_waves,
        cfg.token_waves,
        cfg.pages_per_wave,
        cfg.cp_world,
        cfg.cp_rank,
        cfg.sched,
        cfg.nt_k,
        cfg.waves_per_eu,
        cfg.spread,
    )
    if any(not isinstance(value, int) for value in integer_fields):
        raise ValueError("config geometry and policy fields must be integers")
    if (
        not (0 if resolved else -1) <= cfg.nt_k <= 3
        or not -1 <= cfg.sched <= 4
        or not 0 <= cfg.waves_per_eu <= 10
        or not (0 if resolved else -1) <= cfg.spread <= 1
    ):
        raise ValueError("invalid cache, scheduling or occupancy policy")
    if not isinstance(batch, int) or not 0 <= batch <= 0xFFFF:
        raise ValueError("batch must be in [0, 65535] for packed request IDs")
    if not isinstance(max_block, int) or max_block < 1:
        raise ValueError("max_block must be positive")
    if (
        cfg.feat_waves < 1
        or cfg.token_waves < (1 if resolved else 0)
        or WAVES % (cfg.feat_waves * (cfg.token_waves or 1))
    ):
        raise ValueError("invalid wave split")
    if (
        cfg.pages_per_wave < 0
        or cfg.cp_world < 1
        or not 0 <= cfg.cp_rank < cfg.cp_world
    ):
        raise ValueError("invalid page depth or CP shard")
    if not resolved:
        return 0  # Geometry only; auto depth must be resolved before packing.
    # Includes rounded tail lanes and causal token arithmetic, before masking.
    geometry = replace(cfg, pages_per_wave=cfg.pages_per_wave or 1)
    chunk = work_chunk(geometry)
    chunks = _grid_chunks(max_block, geometry)
    if cfg.spread > 0 and not _spread_fits(batch, max_block, cfg):
        raise ValueError(
            "spread: batch, max_block or its token span exceeds the map packing"
        )
    if chunks > _MAX_CHUNKS or batch * chunks * 8 > 0xFFFFFFFF:
        raise ValueError("work_map: packed chunk or address range exceeded")
    if ((chunks * chunk) * cfg.cp_world + cfg.cp_rank) * PAGE > 0x7FFFFFFF:
        raise ValueError("max_block: causal token address exceeds int32")
    return batch * chunks


def _validate_metadata(
    idx_q,
    cache,
    S,
    H,
    max_block,
    block_table=None,
    cfg=None,
    seq_lens=None,
    out=None,
    work_map=None,
):
    """Metadata only: no device reads, synchronization, or per-layer allocations.

    Contents are a caller contract: 0 <= seq_lens <= INT32_MAX-127,
    each length fits max_block's local CP bound and the global block table,
    and every referenced physical page ID is in [0, cache.shape[0]). Even
    zero-length requests need a valid clamped table entry at column zero.
    A supplied map must be rebuilt when lengths change, with the same config.
    """
    import torch

    if not isinstance(S, int) or not isinstance(H, int) or S < 1 or H < 1:
        raise ValueError("S and H must be positive integers")
    device = idx_q.device
    _validate_tensor(idx_q, "idx_q", (torch.bfloat16,), 3, device, 16)
    _validate_tensor(
        cache, "cache", (torch.bfloat16, torch.float8_e4m3fn), 3, device, 16, False
    )
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx950":
        raise ValueError("index score requires gfx950")
    if idx_q.shape[1:] != (H, HEAD_DIM) or idx_q.shape[0] % S:
        raise ValueError("idx_q: expected [batch*S, H, 128]")
    if cache.shape[1:] != (PAGE, HEAD_DIM) or cache.shape[0] > 0x7FFFFFFF:
        raise ValueError("cache: expected [pages, 128, 128] with int32 page IDs")
    for t, name in ((idx_q, "idx_q"), (cache, "cache")):
        if t.stride(2) != 1 or any(st * t.element_size() % 16 for st in t.stride()[:2]):
            raise ValueError(f"{name}: rows and heads/pages must be 16-byte aligned")
        if (
            t.stride(1) < HEAD_DIM
            or t.stride(0) < (t.shape[1] - 1) * t.stride(1) + HEAD_DIM
        ):
            raise ValueError(f"{name}: overlapping rows are unsupported")
    if cache.stride(0) * cache.element_size() > 0x7FFFFFFF:
        raise ValueError("cache: a single page span exceeds int32")
    batch = idx_q.shape[0] // S
    cfg = cfg or IndexScoreConfig()
    auto_tokens = _auto_token_split(cfg)
    _validate_bounds(batch, max_block, cfg, resolved=False)
    cfg = resolve_config(batch, max_block, cfg, S, H, device=device)
    # Legacy build_work_map(seq_lens, max_block) has no query dimensions and
    # therefore uses unsplit geometry. Preserve those supplied maps, but only
    # for the auto branch: an explicit split must still require its exact grid.
    if (
        auto_tokens
        and cfg.token_waves > 1
        and isinstance(work_map, torch.Tensor)
        and work_map.ndim == 2
        and work_map.shape[0] == batch * ((max_block + WAVES - 1) // WAVES)
        and work_map.shape[0] != batch * max_block
    ):
        cfg = replace(cfg, token_waves=1)
    rows = _validate_bounds(batch, max_block, cfg)
    if cfg.shuffled and cache.stride(1) != HEAD_DIM:
        raise ValueError("shuffled cache must be packed within each page")
    if not selection_filter(S, H, cfg, arch="gfx950"):
        raise ValueError(f"illegal config for S={S} H={H}: {cfg}")
    if block_table is not None:
        _validate_tensor(block_table, "block_table", (torch.int32,), 2, device)
        if (
            block_table.shape[0] != batch
            or block_table.stride(1) != 1
            or block_table.stride(0) < block_table.shape[1]
        ):
            raise ValueError("block_table: invalid shape or layout")
    if seq_lens is not None:
        _validate_tensor(seq_lens, "seq_lens", (torch.int32,), 1, device)
        if seq_lens.shape != (batch,) or seq_lens.stride() != (1,):
            raise ValueError("seq_lens: expected contiguous [batch]")
    if out is not None:
        _validate_tensor(out, "out", (torch.float32,), 3, device)
        if out.shape != (H, batch * S, max_block):
            raise ValueError("out: expected [H, batch*S, max_block]")
        # Both production layouts, including non-overlapping aligned padding.
        st = out.stride()
        contiguous = st[2] == 1 and st[1] >= max_block and st[0] >= batch * S * st[1]
        feature = st[0] == 1 and st[1] >= H and st[2] >= batch * S * st[1]
        if not (contiguous or feature):
            raise ValueError("out: expected page- or feature-contiguous layout")
    elif H * batch * S * max_block * 4 > 0xFFFFFFFF:
        raise ValueError("out: address span exceeds 32-bit addressing")
    if work_map is not None:
        _validate_map(work_map, rows, device, exact=True)
    return batch, cfg


def index_score_supported(
    idx_q,
    cache,
    S: int,
    H: int,
    max_block: int,
    block_table=None,
    cfg=None,
    *,
    seq_lens=None,
    out=None,
    work_map=None,
) -> bool:
    """Metadata-only predicate sharing the execution entry point's validation."""
    try:
        _validate_metadata(
            idx_q, cache, S, H, max_block, block_table, cfg, seq_lens, out, work_map
        )
        return True
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        RuntimeError,
    ):
        return False


# Feature count from which the transposed score layout starts paying.
TRANSPOSE_MIN_F = 16


def alloc_score(batch, S, H, max_block, device):
    """Allocate the score output in whichever layout the kernel writes fastest.

    The shape is [H, batch*S, max_block] either way and only the strides differ,
    so this is invisible to every consumer -- both selectors take the three
    score strides as arguments -- except in the time it takes.

    Why it matters: the epilogue stores one fp32 per feature per page, so with
    F = S*H features the contiguous layout puts those F lanes on F different
    cache lines `max_block` floats apart, which at the production point is 32
    lines spread over ~3 MB. That store costs 15% of the kernel while carrying
    0.8% of its traffic, so no amount of byte accounting finds it. Making the
    feature axis contiguous packs the same values into ONE full 128 B line per
    store. Measured on MI355X at b32_q8_s128k, kernel only:

        bf16 190 -> 173 us (5.51 -> 6.05 TB/s),  fp8 108 -> 96 us

    against a 162 us floor for a kernel that computes the scores and throws
    them away. What is left of that gap is the write volume itself, not the
    pattern: folding four pages onto one address recovers most of it, but
    coalescing the CTA's whole chunk into 2 KB recovers none.

    The consumer pays for this -- its page axis goes from contiguous to
    total_q*H*4 = 4 KB strided, measured at +5.7% on the Triton selector, ~2.6
    us against the 12 us saved. It also loses eligibility for the aiter
    selector, which needs `score.view(rows, max_block)`; at the shapes where
    this layout wins that selector already declines on its LDS budget, but a
    caller with a small max_block should check rather than assume.

    Below F = 16 the store is at most 16 bytes, there is no scatter left to fix,
    and the wider address arithmetic loses ~2% -- so those shapes stay
    contiguous.
    """
    import torch

    total_q = batch * S
    if S * H >= TRANSPOSE_MIN_F:
        # [max_block, total_q, H] viewed as [H, total_q, max_block]: strides
        # (1, H, total_q*H), i.e. feature-contiguous.
        return torch.empty(
            (max_block, total_q, H), dtype=torch.float32, device=device
        ).permute(2, 1, 0)
    return torch.empty((H, total_q, max_block), dtype=torch.float32, device=device)


def score_flydsl(
    idx_q,
    cache,
    block_table,
    seq_lens,
    S,
    H,
    sm_scale,
    max_block,
    out=None,
    cfg=None,
    work_map=None,
    **cfg_kwargs,
):
    """Host entry point. Mirrors the signature the op_test drives.

    Tunables come in as an IndexScoreConfig, or as loose keyword arguments
    naming its fields (`shuffled=True`, `feat_waves=2`, ...) for callers that
    only want to flip one.
    """
    import torch

    if cfg is None:
        cfg = IndexScoreConfig(**cfg_kwargs)
    elif cfg_kwargs:
        raise TypeError("pass either cfg or its fields as keywords, not both")

    batch, cfg = _validate_metadata(
        idx_q, cache, S, H, max_block, block_table, cfg, seq_lens, out, work_map
    )
    if block_table is None or seq_lens is None:
        raise ValueError("block_table and seq_lens are required")
    scaled = sm_scale * LOG2E
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        raise ValueError("sm_scale * LOG2E must be finite and positive in FP32")
    fp8 = cache.dtype == torch.float8_e4m3fn

    if out is None:
        out = alloc_score(batch, S, H, max_block, idx_q.device)

    launch, _chunk = _get(S, H, fp8, cfg, idx_q.device.index)
    # Grid depends only on launch-time bounds, never on seq_lens contents, so
    # it stays valid across a cudagraph replay with different lengths.
    chunks = _grid_chunks(max_block, cfg)
    # Per *step*, not per layer -- a serving caller should hoist this out and
    # pass it in, since every layer of a step sees the same lengths.
    if work_map is None:
        work_map = _make_map(seq_lens, max_block, cfg, batch * chunks)

    with torch.cuda.device(idx_q.device):
        _run_compiled(
            launch,
            ptr_arg(idx_q, fx.BFloat16),
            ptr_arg(cache, fx.Float8E4M3FN if fp8 else fx.BFloat16),
            ptr_arg(out, fx.Float32),
            ptr_arg(block_table, fx.Int32),
            ptr_arg(work_map, fx.Int32),
            idx_q.stride(0),
            idx_q.stride(1),
            cache.stride(0),
            cache.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            block_table.stride(0),
            float(sm_scale * LOG2E),
            batch,
            chunks,
            torch.cuda.current_stream(idx_q.device).cuda_stream,
        )
    return out
