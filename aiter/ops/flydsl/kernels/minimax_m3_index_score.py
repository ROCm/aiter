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
consecutive tokens in *one lane's* 4 accumulators, verified on gfx950 by
op_tests/test_flydsl_mfma_lane_mapping.py. The page reduction then becomes
register-local elementwise max, and only the final fold across g needs two
shuffle_xor steps. In the default config there is no LDS and no barrier at all.

Lane mapping (measured, and asserted by op_tests/test_flydsl_mfma_lane_mapping.py),
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
machine is already full. Defaulted off; sweep it per shape.

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
                   cover distinct pages.
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

    sched          instruction-scheduling hint for the page loop. -1 picks it
                   from the query shape (see resolve_config); 0 leaves the
                   backend alone. 1/2/3 emit `rocdl.iglp_opt(0/1/2)` at the top
                   of the loop body -- LLVM's canned MFMA/VMEM interleavings,
                   written for FA-shaped loops. 4 spells the interleave out with
                   `sched_group_barrier`, one VMEM group per MFMA group.

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
    token_waves: int = 1
    pages_per_wave: int = 0  # 0 = auto, resolved from the launch bounds
    q_to_lds: bool = False
    shuffled: bool = False
    waves_per_eu: int = 0
    nt_k: int = 2
    sched: int = -1  # -1 = auto, resolved from the query shape
    cp_world: int = 1  # 1 = no context parallelism; see above
    cp_rank: int = 0


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
    if cfg.token_waves == 1:
        return 0
    return WAVES * _tiling(S, H, cfg)[1] * WAVE * 4


def work_chunk(cfg: IndexScoreConfig) -> int:
    """Pages one CTA covers -- the granularity make_work_map hands out.

    Waves not spent on a page's feature or token axis each take their own page,
    and each of those loops `pages_per_wave` times.
    """
    if not cfg.pages_per_wave:
        raise ValueError("pages_per_wave is unresolved; call resolve_config first")
    return (WAVES // (cfg.feat_waves * cfg.token_waves)) * cfg.pages_per_wave


# Oversubscription the page loop is allowed to spend. Below this many CTAs the
# machine is short of waves and a longer per-wave loop only makes it shorter;
# above it, the loop is free and buys memory-level parallelism. 4x mirrors
# _auto_num_splits in mqa_logits/fp8_mqa_logits.py, which tuned the same
# trade-off on the same class of kernel.
_CTA_OVERSUBSCRIBE = 4


def resolve_config(
    batch: int, max_block: int, cfg: IndexScoreConfig, S: int = 0, H: int = 0
):
    """Fill in `cfg`'s shape-dependent fields. Idempotent; explicit wins.

    Two fields are auto, each only when left at its sentinel: `pages_per_wave`
    at 0, and `sched` at -1. `sched` needs the query shape, so pass S and H if
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
    if cfg.pages_per_wave and cfg.sched >= 0:
        return cfg

    if not cfg.pages_per_wave:
        from aiter.jit.utils.chip_info import get_cu_num

        target = get_cu_num() * _CTA_OVERSUBSCRIBE
        best = 1
        for ppw in (2, 4):
            chunk = (WAVES // (cfg.feat_waves * cfg.token_waves)) * ppw
            ctas = batch * ((max_block + chunk - 1) // chunk)
            if ctas >= target:
                best = ppw
        cfg = replace(cfg, pages_per_wave=best)

    if cfg.sched < 0 and S and H:
        cfg = replace(cfg, sched=1 if _tiling(S, H, cfg)[1] > 1 else 0)
    return cfg


def selection_filter(S: int, H: int, cfg: IndexScoreConfig) -> bool:
    """Is this config legal for this shape? Mirrors gemm_kernels.selection_filter."""
    # 0 is the "auto" sentinel and legal everywhere; resolve_config only ever
    # picks a value this same filter would accept.
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
    if not -1 <= cfg.sched <= 4:  # -1 is the auto sentinel, see resolve_config
        return False
    if cfg.cp_world < 1 or not 0 <= cfg.cp_rank < cfg.cp_world:
        return False
    if cfg.q_to_lds or cfg.token_waves > 1:
        from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes

        need = q_lds_bytes(S, H, cfg) + reduce_lds_bytes(S, H, cfg)
        if need > get_lds_capacity_bytes(get_gfx().split(":", 1)[0]):
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
    if not selection_filter(S, H, cfg):
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

    F = S * H
    # FT feature tiles in total; FTW of them per wave; FT_PAD = FTW*feat_waves
    # is what the LDS buffer is sized on (see _tiling). FEAT_WAVES waves split
    # the feature axis, the remaining PAGE_WAVES cover distinct pages.
    FT, FTW, FT_PAD = _tiling(S, H, cfg)
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
        k4_buf = ptr_buf_tensor(arg_k, fx.Int32, unit_elems=4)
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
        b = packed >> fx.Int32(16)
        c = packed & fx.Int32(0xFFFF)
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
        pages = [
            c * fx.Int32(CHUNK) + fx.Int32(j * PAGE_WAVES) + pw
            for j in range_constexpr(pages_per_wave)
        ]
        zero = fx.Int32(0)
        nm1 = num_pages - fx.Int32(1)
        last = (nm1 < zero).select(zero, nm1)

        def load_page_id(p):
            """Physical page holding logical block p of this request.

            The table is not sharded -- every rank sees the whole of it -- so
            this indexes with the global id.
            """
            return fx.Int32(
                fx.add_offset(
                    fx.get_iter(bt_buf), b * i32_stride_bt_b + global_block(p)
                ).load(T.i32)
            )

        page_ids = [load_page_id((pp < last).select(pp, last)) for pp in pages]

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

        def load_k16(unit):
            """One 16 B K access at 16 B-unit index `unit`, via the k_atom.

            Goes through a copy atom rather than a plain `.load()` because the
            cache-policy field lives on the atom -- it is the only channel
            FlyDSL exposes for it. Issue and consume stay split: the fragment is
            registers, so the compiler still sinks the s_waitcnt to first use
            and a whole panel's loads remain in flight.
            """
            src = fx.slice(k4_buf, (unit, None))
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
                base = page * i32_stride_k_blk + (
                    fx.Int32((panel * K_LOADS + i) * WAVE) + tw_slot + lane
                ) * fx.Int32(CHUNK_ELEMS)
                # base is in cache elements; >>4 (fp8) / >>3 (bf16) turns it into
                # a 16 B unit index. Both are exact: CHUNK_ELEMS is one 16 B
                # chunk by construction, and the strides above it are multiples.
                shift = fx.Int32(4) if const_expr(fp8) else fx.Int32(3)
                return load_k16(base >> shift)
            tok_row = fx.Int32(MFMA_M * panel) + tw_tok + u
            base = page * i32_stride_k_blk + tok_row * i32_stride_k_pos
            if const_expr(fp8):
                # k_offset(2i) = 64*i + 16*g and k_offset(2i+1) = that + 8, so
                # the two k-steps are 16 contiguous bytes: one dwordx4, and the
                # four g-lanes tile exactly one 64 B line.
                base = base + fx.Int32(64 * i) + g * fx.Int32(16)
                return load_k16(base >> fx.Int32(4))
            base = base + fx.Int32(i * MFMA_K) + g * fx.Int32(8)
            return load_k16(base >> fx.Int32(3))

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

            The software depth is one panel and stays that way: pf=2/4/8 were
            measured and emitted byte-identical ISA (24 loads, maxvmcnt 8) --
            the scheduler hoists to its own depth regardless of the source
            order, so a deeper queue here only made the Python harder to read.
            """
            if const_expr(1 <= sched <= 3):
                fx.rocdl.iglp_opt(sched - 1)
            run_max = [neg_inf for _ in range_constexpr(FTW)]
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
                        [issue_k(page, panel + 1, i) for i in range_constexpr(K_LOADS)]
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
                nxt = page_ids[j + 1] if j + 1 < pages_per_wave else None
                run_max, carry = page_loop(pages[j], page_ids[j], carry, nxt)
                # With one page per wave and the guard below in place, that
                # guard already proved the page in range and the extra mask
                # would be dead.
                valid = (
                    None
                    if const_expr(pages_per_wave == 1 and TOKEN_WAVES == 1)
                    else pages[j] < num_pages
                )
                store_page(pages[j], run_max, valid)

        if const_expr(TOKEN_WAVES > 1):
            # No skip guard: it is wave-divergent, and store_page's barrier has
            # to be reached by every wave in the CTA. The skipped waves instead
            # run the body on a clamped page and drop the result at the store.
            body()
        else:

            @flyc.jit
            def _maybe(_pred=(pages[0] < num_pages), _w=body):
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


def _get(S, H, fp8, cfg):
    key = (S, H, fp8, cfg)  # IndexScoreConfig is frozen, hence hashable
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

    assert world >= 1 and 0 <= rank < world, f"bad shard {rank}/{world}"
    batch = seq_lens.shape[0]
    chunks = (max_block + chunk - 1) // chunk
    assert batch <= 0xFFFF and chunks <= 0x10000, "b and c must pack into 32 bits"
    if out is None:
        out = torch.empty((batch * chunks, 2), dtype=torch.int32, device=seq_lens.device)

    lens = seq_lens.to(torch.int32)
    nblk = (lens + (PAGE - 1)) // PAGE
    if world > 1:
        nblk = (nblk - rank).clamp_(min=0).add_(world - 1).div_(world, rounding_mode="floor")
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


def work_map_size(batch: int, max_block: int, S: int = 0, H: int = 0, cfg=None) -> int:
    """Rows `make_work_map` will produce -- i.e. the grid, one row per CTA.

    The buffer itself is `[rows, 2]` int32; a caller sizing a persistent one for
    a cudagraph should allocate the worst case (max batch, max blocks) and hand
    `make_work_map` a `buf[:rows]` slice, which stays packed and so stays a legal
    kernel argument.

    Exists so that caller does not re-derive the chunk size: it depends on the
    resolved `pages_per_wave`, which depends on the CU count, so a duplicated
    copy would go wrong the first time this runs on a different chip.

    Under context parallelism `max_block` is the local bound, so this is the
    shard's grid -- roughly 1/world of the unsharded one.
    """
    cfg = resolve_config(batch, max_block, cfg or IndexScoreConfig(), S, H)
    chunk = work_chunk(cfg)
    return batch * ((max_block + chunk - 1) // chunk)


def build_work_map(seq_lens, max_block, S: int = 0, H: int = 0, cfg=None, out=None):
    """`make_work_map` with the chunk size resolved for you.

    This is the entry point a serving caller wants. `make_work_map` takes the
    chunk size as a number, which means a caller that builds the map in one
    place and launches the kernel in another has to resolve the config twice and
    keep the two agreeing -- and a disagreement is not a crash, it is a map the
    kernel indexes with the wrong stride.

    `out` may be larger than needed (a persistent worst-case buffer sized by
    `work_map_size`); it is sliced down here, and a short one is an error rather
    than a silently truncated grid.
    """
    cfg = resolve_config(seq_lens.shape[0], max_block, cfg or IndexScoreConfig(), S, H)
    chunk = work_chunk(cfg)
    if out is not None:
        rows = seq_lens.shape[0] * ((max_block + chunk - 1) // chunk)
        assert out.shape[0] >= rows, f"work map buffer holds {out.shape[0]} < {rows}"
        out = out[:rows]
    # The shard comes from the same cfg the kernel will be compiled against, so
    # the map and the kernel cannot disagree about which blocks this rank owns.
    return make_work_map(
        seq_lens, max_block, chunk, out=out, world=cfg.cp_world, rank=cfg.cp_rank
    )


def index_score_supported(
    idx_q, cache, S: int, H: int, max_block: int, block_table=None, cfg=None
) -> bool:
    """Can `score_flydsl` serve this call?

    `score_flydsl` enforces its envelope with asserts, which is right for a test
    but wrong for a dispatcher that has a working fallback. This is the same
    envelope phrased as a question, so a caller can pick the other path without
    catching AssertionError.

    Cheap but not free (it resolves a config, which reads the CU count) --
    memoize it on the shape if it is on a per-layer path.
    """
    import torch

    try:
        if idx_q.dtype != torch.bfloat16 or idx_q.shape[2] != HEAD_DIM:
            return False
        # `fp8 = cache.dtype != bfloat16` downstream, so anything else that is
        # not e4m3 would be silently reinterpreted rather than rejected.
        if cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            return False
        if cache.ndim != 3 or cache.shape[1] != PAGE or cache.shape[2] != HEAD_DIM:
            return False
        if cache.stride(2) != 1 or idx_q.stride(2) != 1:
            return False
        if block_table is not None and (
            block_table.dtype != torch.int32 or block_table.stride(1) != 1
        ):
            return False
        if S < 1 or H < 1 or max_block < 1:
            return False
        batch = idx_q.shape[0] // S
        if batch < 1 or idx_q.shape[0] != batch * S or idx_q.shape[1] != H:
            return False
        cfg = resolve_config(batch, max_block, cfg or IndexScoreConfig(), S, H)
        if not selection_filter(S, H, cfg):
            return False
        # make_work_map packs (request, chunk) into one int32.
        chunk = work_chunk(cfg)
        chunks = (max_block + chunk - 1) // chunk
        return batch <= 0xFFFF and chunks <= 0x10000
    except (AttributeError, IndexError, ValueError, ZeroDivisionError):
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

    assert idx_q.shape[2] == HEAD_DIM, f"head_dim must be {HEAD_DIM}"
    assert idx_q.dtype == torch.bfloat16
    assert cache.stride(2) == 1 and idx_q.stride(2) == 1
    fp8 = cache.dtype != torch.bfloat16
    batch = seq_lens.shape[0]

    # Before _get, so the cache key is the concrete config and not the sentinel,
    # and before make_work_map, which needs the resolved chunk size. A caller
    # passing its own work_map must have resolved with the same bounds -- which
    # it will have, since the chunk size depends only on batch and max_block.
    cfg = resolve_config(batch, max_block, cfg, S, H)

    if out is None:
        out = alloc_score(batch, S, H, max_block, idx_q.device)

    launch, chunk = _get(S, H, fp8, cfg)
    # Grid depends only on launch-time bounds, never on seq_lens contents, so
    # it stays valid across a cudagraph replay with different lengths.
    chunks = (max_block + chunk - 1) // chunk
    # Per *step*, not per layer -- a serving caller should hoist this out and
    # pass it in, since every layer of a step sees the same lengths.
    if work_map is None:
        work_map = make_work_map(
            seq_lens, max_block, chunk, world=cfg.cp_world, rank=cfg.cp_rank
        )

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
        torch.cuda.current_stream().cuda_stream,
    )
    return out
