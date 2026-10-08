# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 lightning-indexer decode block-score kernel (FlyDSL).

Replaces ATOM's `_decode_index_score_tiled_kernel`. For each request b, each
128-token page p, and each (query row, index head) pair, produce

    score[head, row, p] = max over t in [0,128) of
                              dot(K[page, t, :], Q[row, head, :]) * scale

with a per-query causal cutoff.

Layout
------
The MFMA tile is 16x16xK with A = K (M = token) and B = Q (N = feature), which
is what makes the page reduction cheap: with lane = tid % 64, g = lane // 16
and u = lane % 16,

    A operand[v] = A[u, 8*g + v]     v = 0..7     A = K, M = token
    B operand[v] = B[u, 8*g + v]     v = 0..7     B = Q, N = feature
    C accum[r]   = C[4*g + r, u]     r = 0..3

so a lane holds tokens {4g+r} at feature u. The 128-token max is therefore a
register-local elementwise max, and only the final fold across g needs two
shuffle_xor steps. In the default config there is no LDS and no barrier at all.
A 32x32 tile would spread the reduced axis across lanes instead and need DPP,
permlane and LDS to close it.

Tunables are `IndexScoreConfig`; `resolve_config` fills in the automatic ones
and is the only place that decides between them.

gfx942 (CDNA3) alongside gfx950 (CDNA4)
---------------------------------------
Three things differ between the generations, and ArchTraits carries all of
them as compile-time constants:

  1. `mfma_f32_16x16x32_bf16` is gfx950+. gfx942's widest 16x16 bf16 tile is
     `mfma_f32_16x16x16bf16_1k`, so `ksteps` doubles to 8 and each lane's
     MFMA fragment halves to 4 bf16.
  2. The fp8 path widens K to bf16 with `v_cvt_scalef32_pk_bf16_fp8`, which is
     CDNA4. gfx942 goes through `v_cvt_pk_f32_fp8` and then to bf16. The
     detour is exact -- e4m3's 3 mantissa and 4 exponent bits both fit bf16 --
     so it costs instructions, not accuracy.
  3. gfx942's fp8 is `e4m3fnuz` (bias 8, no inf) against gfx950's OCP
     `e4m3fn` (bias 7). Each chip's conversion instruction speaks its own
     chip's format, and `dtypes.fp8` names which that is, so nothing here
     reinterprets one as the other.

The invariant that makes this one kernel rather than two is in `ArchTraits`.
A gfx942 binary has never been run on gfx942 silicon; see the module's tests
for what is and is not covered.
"""

import math
from dataclasses import dataclass, replace
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

# The 8-bit float this chip speaks: OCP e4m3fn on gfx950, e4m3fnuz on gfx942.
# They are not relabelings of each other, so the dtype, the widening
# instruction and the MFMA all have to agree on which one it is.
from aiter import dtypes
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
TOK_TILES = PAGE // MFMA_M  # 8 token tiles per page
LOG2E = 1.4426950409
NEG_INF = float("-inf")

# The widest 16x16 bf16 MFMA each supported generation has. This single number
# is the whole ISA difference the kernel body sees; ArchTraits derives the
# rest, and `k_loads`, `chunk_elems` and the shuffled layout come out the same
# on both -- see ArchTraits for why that is not a coincidence.
_MFMA_K_BY_ARCH = {"gfx950": 32, "gfx942": 16}
SUPPORTED_ARCHS = tuple(_MFMA_K_BY_ARCH)
DEFAULT_ARCH = "gfx950"

# Every K and Q access is one 16 B dwordx4; the element counts below follow.
ACCESS_BYTES = 16
# The four lane groups g = lane // 16 tile the k axis of one access block, so a
# lane holds mfma_k // LANE_GROUPS of the k axis per k-step.
LANE_GROUPS = 4


@dataclass(frozen=True)
class ArchTraits:
    """Compile-time constants for one (architecture, cache dtype) pair.

    All of it is derived from `mfma_k`, the k of this generation's 16x16 bf16
    MFMA: 32 on gfx950 (CDNA4's `mfma_f32_16x16x32_bf16`) and 16 on gfx942
    (CDNA3's `mfma_f32_16x16x16bf16_1k`).

    A lane holds `lane_k = mfma_k // 4` of the k axis per k-step, so one k-step
    is `lane_k * elem_bytes` bytes of cache and a 16 B access covers
    `k_per_load` whole k-steps. Those are packed contiguously per lane by the
    k-axis permutation below, which is why this stays one dwordx4 everywhere:

        k_offset(ks, g) = (ks // k_per_load) * block_k     # which access block
                        + g * lane_block                   # which lane group
                        + (ks % k_per_load) * lane_k       # which k-step in it

    This permutes the k axis, and A and B are both indexed through it, so the
    dot product's sum is reordered and not otherwise changed. Q's layout is
    ours to choose and K's is the shuffled layout's contract, so both follow it.

    The consequence worth stating, because it is what makes gfx942 a small
    change: `block_k` and `lane_block` depend on `mfma_k * k_per_load` and
    `lane_k * k_per_load`, and both products are invariant -- halving mfma_k
    doubles k_per_load. So K's addressing, `k_loads`, `chunk_elems`, `q_loads`
    and the entire shuffled layout are *identical* across the two generations.
    Only `ksteps` (the MFMA loop count) and `lane_k` (the fragment width) move.
    """

    arch: str
    fp8: bool
    mfma_k: int  # k of this arch's 16x16 bf16 MFMA
    lane_k: int  # k elements one lane holds per k-step = MFMA fragment width
    ksteps: int  # MFMA k-steps per 128-wide dot = the inner loop count
    k_per_load: int  # whole k-steps covered by one 16 B K access
    k_loads: int  # 16 B K accesses per token tile
    chunk_elems: int  # cache elements in 16 B
    q_per_load: int  # whole k-steps covered by one 16 B Q access
    q_loads: int  # 16 B Q accesses per feature tile
    block_k: int  # k positions one access block spans, over all four lane groups
    lane_block: int  # k positions one lane group takes inside that block
    # CDNA4 widens fp8 straight to bf16 (`v_cvt_scalef32_pk_bf16_fp8`); CDNA3
    # has to go via f32 (`v_cvt_pk_f32_fp8`). The detour is exact, so this
    # costs instructions and not accuracy -- see convert_k.
    fp8_to_bf16_direct: bool

    def k_offset(self, ks, g):
        """Host-side twin of the kernel's k-axis map; see the class docstring.

        Kept here so `shuffle_cache` and the tests invert the same arithmetic
        the kernel emits instead of a second copy of it -- a disagreement here
        is silent wrong numbers, not a crash.
        """
        return (
            (ks // self.k_per_load) * self.block_k
            + g * self.lane_block
            + (ks % self.k_per_load) * self.lane_k
        )


@cache
def arch_traits(arch: str = DEFAULT_ARCH, fp8: bool = False) -> ArchTraits:
    """Resolve one (arch, dtype) pair to its compile-time constants.

    Memoised: pure, and the host path asks for it on every launch.
    """
    if arch not in _MFMA_K_BY_ARCH:
        raise ValueError(f"unsupported arch {arch!r}; expected {SUPPORTED_ARCHS}")
    mfma_k = _MFMA_K_BY_ARCH[arch]
    lane_k = mfma_k // LANE_GROUPS
    elem_bytes = 1 if fp8 else 2
    k_per_load = ACCESS_BYTES // (lane_k * elem_bytes)
    # Q is bf16 whatever the cache is, so it may span fewer k-steps per access
    # than K does -- but never more, or it would read past the access block.
    q_per_load = min(k_per_load, ACCESS_BYTES // (lane_k * 2))
    tr = ArchTraits(
        arch=arch,
        fp8=fp8,
        mfma_k=mfma_k,
        lane_k=lane_k,
        ksteps=HEAD_DIM // mfma_k,
        k_per_load=k_per_load,
        k_loads=(HEAD_DIM // mfma_k) // k_per_load,
        chunk_elems=ACCESS_BYTES // elem_bytes,
        q_per_load=q_per_load,
        q_loads=(HEAD_DIM // mfma_k) // q_per_load,
        block_k=mfma_k * k_per_load,
        lane_block=lane_k * k_per_load,
        fp8_to_bf16_direct=arch == "gfx950",
    )
    # The invariants the kernel body relies on, asserted once rather than
    # re-derived at each use.
    assert tr.lane_block * LANE_GROUPS == tr.block_k  # lane groups tile a block
    assert tr.k_loads * tr.k_per_load == tr.ksteps  # no partial K access
    assert tr.q_loads * tr.q_per_load == tr.ksteps  # no partial Q access
    assert tr.q_loads * WAVE == THREADS  # the Q LDS fill is a perfect assignment
    return tr


def _fp8_t(arch: str):
    """The FlyDSL fp8 element type this architecture's `dtypes.fp8` names.

    gfx942 is e4m3FNUZ and gfx950 is e4m3fn; the two differ in exponent bias, so
    binding one as the other silently reads every value at the wrong scale.
    """
    return fx.Float8E4M3FN if arch == "gfx950" else fx.Float8E4M3FNUZ


def fragment_helpers(tr: ArchTraits, g):
    """Per-lane-group MFMA fragment builders, shared by both index scorers.

    `g` is the caller's lane-group expression. Returns
    (q_load_offset, as_bf16_frag, convert_k); see each for its contract and
    ArchTraits for the map they implement.

    Shared rather than copied because the decode and prefill kernels read the
    same cache through the same k-axis permutation: a divergence between two
    copies would be silent wrong numbers, not a crash.
    """
    # Q accesses per K access block: 1 when a Q access spans a whole block, 2
    # when two Q accesses share one. Folding the floor division into this lets
    # both branches stay a single multiply-add.
    q_split = tr.k_per_load // tr.q_per_load
    q_block = tr.block_k // q_split
    q_half = tr.lane_block // q_split

    def q_load_offset(j):
        """k position of this lane group's 16 B Q access `j`.

        Equal to `tr.k_offset(j * q_per_load, g)`. `j` is a Python int on the
        register path, where it folds to a constant, and an Int32 on an LDS
        fill path, where the access index is the wave id.
        """
        if const_expr(q_split == 1):
            if const_expr(isinstance(j, int)):
                return fx.Int32(j * q_block) + g * fx.Int32(tr.lane_block)
            return j * fx.Int32(q_block) + g * fx.Int32(tr.lane_block)
        if const_expr(isinstance(j, int)):
            return (
                fx.Int32((j // q_split) * tr.block_k)
                + g * fx.Int32(tr.lane_block)
                + fx.Int32((j % q_split) * q_half)
            )
        return (
            (j // fx.Int32(q_split)) * fx.Int32(tr.block_k)
            + g * fx.Int32(tr.lane_block)
            + (j % fx.Int32(q_split)) * fx.Int32(q_half)
        )

    def as_bf16_frag(raw16, sub=0):
        """MFMA B-fragment `sub` of a 16 B Q access.

        One access holds q_per_load k-steps of lane_k bf16 each, laid out back
        to back by q_load_offset, so k-step `sub` is the slice
        [sub*lane_k, +lane_k). On gfx950 q_per_load is 1 and this is the whole
        16 B; on gfx942 it is one of two halves.
        """
        t = fx.make_rmem_tensor(fx.make_layout(tr.lane_k, 1), fx.BFloat16)
        wide = raw16.bitcast(fx.BFloat16)
        if const_expr(tr.q_per_load == 1):
            t.store(wide)
        else:
            t.store(
                fx.Vector.from_elements(
                    [
                        fx.BFloat16(wide[sub * tr.lane_k + v])
                        for v in range_constexpr(tr.lane_k)
                    ],
                    fx.BFloat16,
                )
            )
        return t

    def convert_k(raws, ks):
        """Widen the raw access holding k-step ks into a bf16 A-fragment.

        fp8 is widened here rather than fed to a native fp8 MFMA, because the
        operator this replaces lifts K to Q's precision instead of rounding Q
        down. That is a numerics contract, not a performance choice.

        One access holds k_per_load k-steps of lane_k elements each, laid out
        back to back by the k_offset map, so this takes slice
        `ks % k_per_load`. On gfx950 bf16 that slice is the whole access and
        this is a bitcast.
        """
        raw = raws[ks // tr.k_per_load]
        sub = ks % tr.k_per_load
        t = fx.make_rmem_tensor(fx.make_layout(tr.lane_k, 1), fx.BFloat16)
        if const_expr(not tr.fp8):
            wide = raw.bitcast(fx.BFloat16)
            if const_expr(tr.k_per_load == 1):
                t.store(wide)
            else:
                t.store(
                    fx.Vector.from_elements(
                        [
                            fx.BFloat16(wide[sub * tr.lane_k + v])
                            for v in range_constexpr(tr.lane_k)
                        ],
                        fx.BFloat16,
                    )
                )
            return t
        t.store(
            fx.Vector.from_elements(
                fp8_dwords_to_bf16(tr, raw, sub * (tr.lane_k // 4), tr.lane_k // 4),
                fx.BFloat16,
            )
        )
        return t

    return q_load_offset, as_bf16_frag, convert_k


def fp8_dwords_to_bf16(tr: ArchTraits, raw, lo, dwords):
    """`dwords` packed-fp8 dwords of `raw` from index `lo`, as 4*dwords bf16.

    Split out of `convert_k` so an fp8 Q can be widened by exactly the
    conversion K is widened by. Two copies would be silent wrong numbers rather
    than a crash -- the same reason `fragment_helpers` is shared between the two
    scorers at all. Module level rather than another `fragment_helpers` return
    value so no existing caller has to change its unpacking.
    """
    one = _to_raw(fx.Float32(1.0))
    elems = []
    for d in range_constexpr(dwords):
        word = _to_raw(fx.Int32(raw[lo + d]))
        for half in range_constexpr(2):
            if const_expr(tr.fp8_to_bf16_direct):
                # The cache is unit-scale, hence scale 1.0.
                pair = fx.Vector(
                    fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                        T.vec(2, T.bf16), word, one, bool(half)
                    )
                )
                elems += [fx.BFloat16(pair[0]), fx.BFloat16(pair[1])]
            else:
                # CDNA3 has no fp8->bf16, so go through f32. The detour is
                # exact and the rounding mode is irrelevant: e4m3 carries 3
                # mantissa bits and 4 exponent bits, both of which fit bf16's
                # 7 and 8, so nothing is rounded.
                pair = fx.Vector(
                    fx.rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), word, bool(half))
                )
                elems += [
                    fx.BFloat16(fx.Float32(pair[0])),
                    fx.BFloat16(fx.Float32(pair[1])),
                ]
    return elems


def bf16_dwords_to_fp8(raw, dwords):
    """`dwords` dwords of bf16 in `raw` as dwords//2 packed-fp8 dwords.

    The exact inverse direction of `fp8_dwords_to_bf16`, and the same rounding
    the prefill scorer applies to Q. Rounding Q down rather than lifting K up
    is a DIFFERENT numerics contract from this kernel's default -- see
    `IndexScoreConfig.precision`.
    """
    wide = raw.bitcast(fx.BFloat16)
    words = []
    for d in range_constexpr(dwords // 2):
        w = _to_raw(fx.Int32(0))
        for h in range_constexpr(2):
            w = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                _to_raw(fx.Float32(wide[4 * d + 2 * h])),
                _to_raw(fx.Float32(wide[4 * d + 2 * h + 1])),
                w,
                bool(h),
            )
        words.append(fx.Int32(w))
    return words


# rocdl.sched_group_barrier instruction-class masks (rocdl._SCHED_MASK_INT_TO_KW).
_SCHED_MFMA = 8
_SCHED_VMEM_RD = 32


@dataclass(frozen=True)
class IndexScoreConfig:
    """Tunables, in the shape aiter/ops/flydsl/gemm_kernels.py uses.

    A note on axis names, because they are easy to get backwards here. This
    kernel puts A=K (MFMA M = token) and B=Q (MFMA N = feature = tok*H+head),
    so the *query* axis -- what you would call M in a plain GEMM, and what
    `spec_decode * num_kv_head` sizes -- is N in this kernel. Every wave owns
    the whole of that axis; the CTA's waves divide the BLOCK axis instead.

    waves_per_tok  waves of the CTA that split a page's 8 token tiles. They
                   each load a different slice of the page, so unlike splitting
                   the FEATURE axis this duplicates no reads -- it is paid for
                   with one LDS round trip and one barrier per page to combine
                   the partial maxes. waves_per_tok waves cover one page; the
                   remaining WAVES // waves_per_tok cover distinct pages. 0
                   selects the narrow automatic split in resolve_config;
                   explicit 1/2/4 always win.

                   There used to be a `waves_per_feat` beside this, splitting
                   the feature axis (F = S*H columns) the same way. It is gone:
                   every wave re-read the whole page under it, so it bought
                   parallelism with duplicate K traffic, and this kernel runs
                   at ~97% of HBM peak with a 4-8% L2 hit rate -- doubling K
                   traffic is the worst trade available here. It also had
                   almost no room to act: F is 4-32 columns for MiniMax-M3, so
                   feature_tiles is 1 except at S=8, and `resolve_config` never
                   raised it above 1 on any shape.
    pages_per_wave pages one wave walks back to back, pipelined across the
                   boundary. 0 means pick it from the launch bounds; see
                   resolve_config, which every entry point calls first.
    shuffled       K cache is pre-shuffled into the kernel's load order.
    waves_per_eu   occupancy floor passed to the backend; 0 leaves it unset,
                   which is what every shape measured here wants.
    nt_k           CDNA cache policy for the K loads, as the raw aux field:
                   0 = default (allocate in L1), 2 = NT, 3 = SC0|NT (the
                   `GROUP_NT` the rest of aiter uses for streamed reads, see
                   csrc/include/aiter_opus_plus.h). Bit 0 is sc0, bit 1 is nt.
                   -1 resolves per layout; see resolve_config.
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
    cp_rank        which one this launch is; 1/0 is the whole context on one
                   rank and folds every CP expression away. Each rank owns the
                   round-robin subset `block % cp_world == cp_rank`, so logical
                   block p here is global block `p * cp_world + cp_rank`. The
                   global id addresses the block table and positions the causal
                   mask; the score is stored at the local p, giving the
                   compacted [H, tokens, ceil(blocks/world)] shard the CP
                   selector reads. Both are constexpr, so there is one compiled
                   kernel per rank and `kernel_name` carries them.

    precision     which dtype the dot is computed in. THIS CHANGES RESULTS.

                  "bf16" (default) keeps the contract this kernel has always
                  had: an fp8 cache is widened up to Q's bf16 and the MFMA is
                  bf16, matching the operator being replaced (see `convert_k`).

                  "fp8" instead rounds Q down to the cache's fp8 and uses the
                  fp8 MFMA -- what the prefill scorer does, and what Triton's
                  `tl.dot(q.to(k.dtype), k)` does. Same 16x16 tile and the same
                  k=32, so it buys no arithmetic; what it removes is the
                  widening in the inner loop, which runs per (page, tok_tile,
                  k-step).

                  Scores differ between the two. The selector consumes them as
                  a top-k ranking, so a tie reordered near the cut can change
                  which blocks are chosen; that is a decision for the caller,
                  which is why this is opt-in and never resolved automatically.

                  gfx950 only. There the fp8 MFMA's k is 32, the same as the
                  bf16 one, so `lane_k` and the whole k-axis map -- and
                  therefore the shuffled cache layout -- are unchanged. On
                  gfx942 the bf16 MFMA is k=16 and the fp8 one k=32, so the map
                  would have to differ and the shuffled cache would no longer
                  be the same bytes. Rejected there rather than silently
                  reshaping a layout two kernels share.
    """

    waves_per_tok: int = 0  # 0 = auto; explicit 1/2/4 are never overridden
    pages_per_wave: int = 0  # 0 = auto, resolved from the launch bounds
    shuffled: bool = False
    fp8_q: int = 0  # Q arrives fp8, not bf16; set from idx_q.dtype, not tuned
    precision: str = "bf16"  # "bf16" | "fp8"; see the docstring above
    waves_per_eu: int = 0
    nt_k: int = -1  # -1 = auto: 0 on the served decode path, else 2
    sched: int = -1  # -1 = auto, resolved from the query shape
    cp_world: int = 1  # 1 = no context parallelism; see above
    cp_rank: int = 0
    spread: int = -1  # -1 = auto: 1 on the served one-page-per-wave path, else 0


def _cp_blocks(pages, world: int, rank: int, zero):
    """This rank's share of `pages` blocks, as an fx expression.

    Round-robin: rank r owns global blocks r, r+world, r+2*world, ..., i.e.
    ceil(max(pages - r, 0) / world) of them. `zero` is the caller's Int32 zero,
    which all three kernels using this already have in scope. The clamp is
    cmp+select, never maximumf -- that is a float op and would compare these
    indices as bit patterns.

    The kernel and both work-map builders each need this number and must agree
    on it: a divergence is not a crash, it is a shard whose chunks are numbered
    against a different length. Hence one copy, with `_cp_blocks_torch` as the
    host twin.
    """
    if world == 1:
        return pages
    avail = pages - fx.Int32(rank)
    avail = (avail < zero).select(zero, avail)
    return (avail + fx.Int32(world - 1)) // fx.Int32(world)


def _cp_blocks_torch(nblk, world: int, rank: int):
    """`_cp_blocks` on a torch int32 tensor, in place. See it for the formula."""
    if world == 1:
        return nblk
    return (
        (nblk - rank).clamp_(min=0).add_(world - 1).div_(world, rounding_mode="floor")
    )


def feature_tiles(S: int, H: int) -> int:
    """Number of 16-wide feature tiles needed to cover F = S*H columns."""
    return (S * H + MFMA_N - 1) // MFMA_N


def _tiles_per_wave(S: int, H: int, cfg: "IndexScoreConfig") -> int:
    """Feature tiles one wave owns -- all of them; the axis is not split."""
    return feature_tiles(S, H)


def reduce_lds_bytes(S: int, H: int, cfg: "IndexScoreConfig") -> int:
    """LDS held by the cross-wave partial-max exchange, 0 when waves_per_tok == 1.

    One fp32 per (wave, wave-local feature tile, lane). Keeping all 64 lanes
    rather than just the g == 0 writers makes the write contiguous and the read
    lane-local, and it is only WAVES*FEAT_TILES_PER_WAVE*64*4 B -- 2 KB at S=8 H=4.
    """
    if cfg.waves_per_tok <= 1:
        return 0
    return WAVES * _tiles_per_wave(S, H, cfg) * WAVE * 4


def work_chunk(cfg: IndexScoreConfig) -> int:
    """Pages one CTA covers -- the granularity make_work_map hands out.

    Waves not spent on a page's feature or token axis each take their own page,
    and each of those loops `pages_per_wave` times.
    """
    if not cfg.pages_per_wave:
        raise ValueError("pages_per_wave is unresolved; call resolve_config first")
    # An explicit page depth disables auto token splitting.
    return (WAVES // (cfg.waves_per_tok or 1)) * cfg.pages_per_wave


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


def _auto_tok_split(cfg: IndexScoreConfig) -> bool:
    """Only tune the untouched decode path; CP does not alter the gate.

    `shuffled` is deliberately in scope: it changes the *order* K is read in,
    not how much work there is, so the depth tables and the spread map -- both
    about filling CUs -- carry over. The cache policy is the one thing that
    does not; see `nt_k` at the resolve site.
    """
    return (
        cfg.waves_per_tok == 0
        and cfg.pages_per_wave == 0
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

    batch, max_block  launch bounds; never seq_lens contents, because the grid
                      has to stay valid across a cudagraph replay
    cfg               config whose sentinel fields are to be resolved
    S, H              query shape; `sched` needs them and stays off without
    device            for the CU count; None asks the current device

    Five fields are auto, each only at its sentinel. An explicit value is kept
    as given.

    nt_k            (-1) follows the K layout; the two are one fact seen from
                    two sides, argued at the assignment below.
    pages_per_wave  (0) deeper keeps loads in flight across the page seam but
                    divides the CTA count, so it is chosen from how much work
                    a launch has. A *served* decode launch takes it from the
                    graph batch, since the grid must stay graph-stable and the
                    real lengths sit far below the capacity `max_block`
                    reports; everything else estimates CTA supply from
                    `batch * max_block`, which over-counts a ragged batch and
                    so errs towards the longer loop.
    waves_per_tok     (0) splitting a page's token axis only helps where there
                    are too few pages to fill the machine. Gated to
                    native-layout Q1 with one feature tile.
    sched           (-1) iglp_opt(0) pays exactly when a wave has more than
                    one feature tile, so that is the gate.
    spread          (-1) resolved by `_resolve_spread`.
    """
    auto_tokens = _auto_tok_split(cfg)
    # A decode launch with a measured depth table and no CP. The tables were
    # measured at max_block 8192; past 0xFFFF blocks their shallow depths would
    # not even pack, so those capacities fall through to the estimate.
    served = (
        auto_tokens
        and _decode_depth_table(S, H) is not None
        and cfg.cp_world == 1
        and max_block <= 0xFFFF
    )
    if served and S == 1 and batch * ((max_block + 1) // 2) <= _cu_count(device):
        # Q1 launches small enough for the token split below keep it: that
        # regime was measured separately and the depth tables were not.
        served = False
    if cfg.nt_k < 0:
        # The policy follows the layout, because the two are one fact seen from
        # two sides. Natively one K instruction takes 64 B from each of 16 rows,
        # so every 128 B line is requested twice and L1 turns the second into a
        # hit -- NT would send it back to L2. Shuffled, one instruction is 1024
        # contiguous bytes and each line is requested exactly once, so there is
        # no second access to lose and the L1 fill buys nothing.
        cfg = replace(cfg, nt_k=0 if served and not cfg.shuffled else 2)
    if cfg.waves_per_tok == 0:
        cfg = replace(cfg, waves_per_tok=1)
    if not cfg.pages_per_wave and served:
        # The tabled depth, capped at the deepest that still hands every CU a
        # CTA. The table is keyed on batch alone, but depth only pays when
        # there are enough pages to fill the machine at that depth, and a full
        # batch of *short* requests has a large batch and very little work.
        # `batch * max_block` is the graph's page capacity and so an upper
        # bound on live pages, which makes the cap safe in the direction that
        # matters: a genuinely long batch keeps its tabled depth and only a
        # launch that provably cannot fill one CU round is trimmed.
        depth = 4
        for limit, tabled in _decode_depth_table(S, H):
            if batch <= limit:
                depth = tabled
                break
        per_wave = WAVES // max(cfg.waves_per_tok, 1)
        cap = max(1, (batch * max_block) // (per_wave * _cu_count(device)))
        cfg = replace(cfg, pages_per_wave=min(depth, cap))
    elif not cfg.pages_per_wave:
        cu = _cu_count(device)
        # One page per CTA fills otherwise missing CU scheduling slots. Only
        # native-layout Q1/one-tile launches were measured for this auto path.
        if auto_tokens and S == 1 and 0 < H <= MFMA_N:
            if batch * max_block <= cu:
                cfg = replace(cfg, waves_per_tok=WAVES)
            elif batch * ((max_block + 1) // 2) <= cu:
                # Two pages/CTA retain one scheduling round while halving each
                # wave's tok_tile chain. Beyond one round the LDS exchange loses.
                cfg = replace(cfg, waves_per_tok=2)
        target = cu * _CTA_OVERSUBSCRIBE
        best = 1
        for ppw in (2, 4):
            chunk = (WAVES // cfg.waves_per_tok) * ppw
            ctas = batch * ((max_block + chunk - 1) // chunk)
            if ctas >= target:
                best = ppw
        cfg = replace(cfg, pages_per_wave=best)

    if cfg.sched < 0 and S and H:
        cfg = replace(cfg, sched=1 if _tiles_per_wave(S, H, cfg) > 1 else 0)
    return _resolve_spread(cfg, served, batch, max_block)


def _resolve_spread(cfg: IndexScoreConfig, served: bool, batch: int, max_block: int):
    if cfg.spread >= 0:
        return cfg
    on = (
        served
        and cfg.pages_per_wave == 1
        and cfg.waves_per_tok == 1
        and _spread_fits(batch, max_block, cfg)
    )
    return replace(cfg, spread=int(on))


def selection_filter(S: int, H: int, cfg: IndexScoreConfig, arch=None) -> bool:
    """Is this config legal for this shape? Mirrors gemm_kernels.selection_filter."""
    # Check the auto token split as its fallback geometry; its Q1 branch is
    # also legal (one feature tile, four token waves).
    if cfg.waves_per_tok == 0:
        cfg = replace(cfg, waves_per_tok=1)
    if cfg.pages_per_wave < 0 or cfg.waves_per_tok < 1:
        return False
    # The two intra-page splits have to partition the CTA's waves between them,
    # and there is no point handing a wave fewer than one tile or one tok_tile.
    if WAVES % cfg.waves_per_tok:
        return False
    if TOK_TILES % cfg.waves_per_tok:
        return False
    if cfg.waves_per_eu and not 1 <= cfg.waves_per_eu <= 10:
        return False
    # `precision` reaches an MFMA whose fp8 form is CDNA4-only. Checked here as
    # well as in `build_index_score` so `index_score_supported` cannot report a
    # config as dispatchable and then have `score_flydsl` raise on it.
    if cfg.precision not in ("bf16", "fp8"):
        return False
    if cfg.precision == "fp8" and arch is not None and arch != "gfx950":
        return False
    if not -1 <= cfg.nt_k <= 3:  # -1 is the auto sentinel, see resolve_config
        return False
    if not -1 <= cfg.sched <= 4:  # -1 is the auto sentinel, see resolve_config
        return False
    if cfg.cp_world < 1 or not 0 <= cfg.cp_rank < cfg.cp_world:
        return False
    if not -1 <= cfg.spread <= 1:  # -1 is the auto sentinel, see resolve_config
        return False
    if cfg.spread == 1 and (cfg.pages_per_wave > 1 or cfg.waves_per_tok > 1):
        return False  # a spread CTA holds at most one page per wave
    if cfg.waves_per_tok > 1:
        from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes

        need = reduce_lds_bytes(S, H, cfg)
        if need > get_lds_capacity_bytes((arch or get_gfx()).split(":", 1)[0]):
            return False
    return True


def kernel_name(
    S: int, H: int, fp8: bool, cfg: IndexScoreConfig, arch: str = DEFAULT_ARCH
) -> str:
    """Config -> kernel name. Non-default knobs append a suffix, so the default
    config keeps the name it had before the knobs existed."""
    name = f"m3_index_score_S{S}_H{H}_{'fp8' if fp8 else 'bf16'}_L{cfg.pages_per_wave}"
    if arch != DEFAULT_ARCH:
        # A different MFMA generation is a different binary from the same
        # config, so it cannot share a JIT cache entry with gfx950's.
        name += f"_{arch}"
    if cfg.shuffled:
        name += "_shuf"
    if cfg.fp8_q:
        name += "_q8"
    if cfg.precision != "bf16":
        name += f"_p{cfg.precision}"
    if cfg.waves_per_tok > 1:
        name += f"_tw{cfg.waves_per_tok}"
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


def build_index_score(
    S: int, H: int, fp8: bool, cfg: IndexScoreConfig, arch: str = DEFAULT_ARCH
):
    """Compile a score kernel specialised on (S, H, cache dtype, config, arch).

    S = max query tokens per request (num_spec + 1), H = index heads.
    F = S*H is the feature count, laid out as column n = tok*H + head.

    `arch` selects the MFMA generation through ArchTraits. It is a parameter
    rather than a lookup of the running device so the gfx942 shapes can be
    built and numerically checked on a gfx950 box -- CDNA4 kept both the
    16x16x16 MFMA and `v_cvt_pk_f32_fp8`, so the only part of the gfx942 path
    that genuinely needs gfx942 silicon is whether the binary loads there.
    """
    if not selection_filter(S, H, cfg, arch=arch):
        raise ValueError(f"illegal config for S={S} H={H}: {cfg}")
    tr = arch_traits(arch, fp8)
    # gfx942's fp8 is e4m3FNUZ (see aiter.utility.dtypes.defaultDtypes) and
    # gfx950's is e4m3fn. They differ in exponent bias, so binding one as the
    # other does not fail -- it reads every value at the wrong scale. The host
    # already hands over whichever `dtypes.fp8` names, so only the FlyDSL-side
    # element type has to follow. Same selection as pa_decode_kernel.py:220.
    FP8_T = _fp8_t(arch)
    # Compute the dot in fp8 instead of widening K to bf16. Changes results;
    # see IndexScoreConfig.precision.
    if cfg.precision not in ("bf16", "fp8"):
        raise ValueError(f"precision must be 'bf16' or 'fp8', got {cfg.precision!r}")
    FP8_COMPUTE = cfg.precision == "fp8"
    if FP8_COMPUTE and not fp8:
        raise ValueError("precision='fp8' needs an fp8 cache")
    if FP8_COMPUTE and arch != "gfx950":
        raise ValueError(
            "precision='fp8' is gfx950 only: there the fp8 MFMA's k matches the "
            "bf16 one, so the k-axis map and the shuffled layout are unchanged"
        )
    # Q is fp8 in memory. On the bf16 path it is widened back on load (fewer Q
    # bytes, one more conversion); on the fp8 path it is already what the MFMA
    # wants and the conversion disappears entirely. See `load_q_frag`.
    FP8_Q = cfg.fp8_q > 0
    pages_per_wave = cfg.pages_per_wave
    shuffled = cfg.shuffled
    nt_k = cfg.nt_k
    sched = cfg.sched
    # Context-parallel block remap. At 1/0 every use below is the identity and
    # the emitted code is byte-identical to the non-CP kernel.
    CP_WORLD = cfg.cp_world
    CP_RANK = cfg.cp_rank
    SPREAD = cfg.spread > 0

    F = S * H
    # FEAT_TILES feature tiles in total, and every wave owns all of them --
    # the feature axis is not divided between waves. The CTA's waves divide the
    # block axis: WAVES_PER_TOK of them share one page by splitting its token
    # tiles, and PAGE_SLOTS groups of those cover distinct pages.
    FEAT_TILES_PER_WAVE = _tiles_per_wave(S, H, cfg)
    WAVES_PER_TOK = cfg.waves_per_tok
    PAGE_SLOTS = WAVES // WAVES_PER_TOK
    TOK_TILES_PER_WAVE = TOK_TILES // WAVES_PER_TOK  # token tiles one wave walks
    CHUNK = PAGE_SLOTS * pages_per_wave
    # One lane's K access for a given (tok_tile, load) is 16 elements for fp8 or 8
    # for bf16 -- 16 B either way, so a shuffled page is WAVE*16 = 1024 B per
    # load slot. How many k-steps that covers is the arch's business, not this
    # layout's, which is why the shuffled cache is the same on both.
    CHUNK_ELEMS = tr.chunk_elems
    # K load instructions per tok_tile: 4 for bf16, 2 for fp8, on both
    # architectures. gfx942 halves mfma_k and doubles the k-steps one access
    # covers, and the two cancel -- see ArchTraits.
    K_LOADS = tr.k_loads
    KSTEPS = tr.ksteps  # MFMA k-steps per dot: 4 on gfx950, 8 on gfx942
    # The kernel's only LDS: the cross-wave partial-max exchange, one fp32 per
    # (wave, tile, lane). Zero unless the token axis is split.
    RED_SLOTS = WAVES * FEAT_TILES_PER_WAVE * WAVE if WAVES_PER_TOK > 1 else 0

    # Built from only the arrays this config uses, so the default config still
    # reports group_segment_fixed_size = 0 -- no LDS and no barrier at all.
    SharedStorage = (
        fx.struct(
            type(
                "SharedStorage",
                (),
                {"__annotations__": {"red": fx.Array[fx.Float32, RED_SLOTS, 16]}},
            )
        )
        if RED_SLOTS
        else None
    )

    @flyc.kernel(
        name=kernel_name(S, H, fp8, cfg, arch),
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

        arg_q       [batch*S, H, 128] bf16 query
        arg_k       [pages, 128, 128] bf16 or fp8 index key cache
        arg_score   [H, batch*S, max_block] fp32, written one value per
                    (feature, page)
        arg_bt      [batch, >= max_block] int32 block table
        arg_work    [rows, 2] int32 work map; see make_work_map
        i32_batch   requests, i.e. the grid's x extent
        i32_stride_*  element strides of the tensor named by the suffix
        f32_scale   score multiplier, already folded with LOG2E by the host

        The three stages below are:

            prologue   ids and strides, Q staged in registers, and every
                       page id this wave will touch, fetched in one batch
            main loop  per page, per 16-token tile: load K, MFMA against Q,
                       causal-mask, fold into a per-feature running max
            epilogue   per page, fold that max across g and across the token
                       waves, scale, store one value per feature

        Everything is constexpr-unrolled, so "loop" means the shape of the
        emitted code, not a branch.

        Wave decomposition: wave w takes tw (which slice of the page's token
        axis) and pw (which page slot). At WAVES_PER_TOK == 1 a wave owns whole pages, the
        token reduction stays register-local, and the kernel has no LDS and no
        barrier at all; above it the per-page epilogue exchanges partial maxes
        and is the only place a barrier appears.
        """
        # ==================== PROLOGUE ====================
        # The CTA looks up which (b, c) the n-th *dispatched* block owns rather
        # than taking (block.x, block.y) directly, so that a ragged batch's
        # holes all land past the end of the work; see make_work_map. block.x
        # is the fast dispatch axis, so n is monotonic in launch order.
        n = fx.Int32(gpu.block_id("y")) * i32_batch + fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)
        g = lane // fx.Int32(16)
        u = lane % fx.Int32(16)
        # wave = tw + WAVES_PER_TOK*pw, so the waves sharing a page are
        # adjacent and their K addresses issue together.
        if const_expr(WAVES_PER_TOK == 1):
            tw = fx.Int32(0)
            pw = wave
        else:
            tw = wave % fx.Int32(WAVES_PER_TOK)
            pw = wave // fx.Int32(WAVES_PER_TOK)
        # This wave's slice of the page's token axis. Both fold to zero at
        # WAVES_PER_TOK == 1, so the default config's addressing is unchanged.
        tw_tok = tw * fx.Int32(TOK_TILES_PER_WAVE * MFMA_M)  # first token of the slice
        tw_slot = tw * fx.Int32(
            TOK_TILES_PER_WAVE * K_LOADS * WAVE
        )  # shuffled load slot

        # Q is read as i32 and bitcast: one 8-element fragment is 16 B (bf16) or
        # 8 B (fp8), so it lands as a single dwordx4 / dwordx2 instead of 8
        # scalar loads. Every fragment base is 8-element aligned, which those
        # widths require.
        q_buf = ptr_buf_tensor(arg_q, fx.Int32)
        s_buf = ptr_buf_tensor(arg_score, fx.Float32)
        bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
        wm_buf = ptr_buf_tensor(arg_work, fx.Int32)

        # Work-map entry n is two adjacent dwords, (packed, seq_len), so it is
        # one cache line and one round trip. A hole reads as (0, 0), which makes
        # num_pages zero and lets the tail guard retire the CTA with no extra
        # branch; the clamp on the page id keeps every address legal.
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
            # ...and of those, the ones this rank owns; see `_cp_blocks`.
            num_pages = _cp_blocks(num_pages, CP_WORLD, CP_RANK, fx.Int32(0))

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
        # fold to shifts. causal_len depends on tok, hence on u, so it is a
        # per-lane value and must not be hoisted.
        #
        def feature_of(i):
            return fx.Int32(MFMA_N * i) + u

        def tok_head_of(i):
            f = feature_of(i)
            tok = f // fx.Int32(H)
            return tok, f - tok * fx.Int32(H)

        # -- where in Q does one fragment live? -------------------------------
        # Any bijection (g, ks, v) -> k works provided Q and K use the same
        # one; the one picked has access j of lane g start at byte 64*j + 16*g,
        # so the four g-lanes tile one full 64 B cache line. In elements that
        # is ArchTraits.k_offset. Q is addressed by *access* index rather than
        # k-step: one 16 B access holds q_per_load whole k-steps.
        q_load_offset, as_bf16_frag, convert_k = fragment_helpers(tr, g)

        def load_q_frag(i, j):
            """One lane's 16 B of Q for (feature tile i, Q access j), from gmem."""
            tok, head = tok_head_of(i)
            row = b * fx.Int32(S) + tok
            # Columns past F are padding; clamp them to row 0 so the load stays
            # in bounds, then drop the result at store time.
            f = feature_of(i)
            in_range = f < fx.Int32(F)
            row = in_range.select(row, fx.Int32(0))
            head = in_range.select(head, fx.Int32(0))
            off = row * i32_stride_q_n + head * i32_stride_q_h + q_load_offset(j)
            # Returns this access in the COMPUTE dtype: 4 dwords of bf16, or
            # 2 dwords of fp8. Four combinations, and only one of them needs a
            # conversion in each direction -- matching input to compute costs
            # nothing, crossing costs one convert on a value that is hoisted
            # out of the page loop anyway.
            if const_expr(FP8_Q):
                # 8 fp8 = 8 B = one dwordx2.
                raw8 = fx.Vector(
                    fx.add_offset(fx.get_iter(q_buf), off >> fx.Int32(2)).load(
                        T.vec(2, T.i32)
                    )
                )
                if const_expr(FP8_COMPUTE):
                    return raw8  # already exactly what the fp8 MFMA wants
                # Widen back up: this scorer's default MFMA is bf16 and lifts K
                # to Q's precision rather than rounding Q down (see
                # `convert_k`). So on the bf16 path an fp8 Q saves read bytes
                # and ADDS a conversion; it does not remove one.
                return fx.Vector.from_elements(
                    fp8_dwords_to_bf16(tr, raw8, 0, 2), fx.BFloat16
                ).bitcast(fx.Int32)
            # 8 bf16 = 16 B = one dwordx4.
            raw16 = fx.Vector(
                fx.add_offset(fx.get_iter(q_buf), off >> fx.Int32(1)).load(
                    T.vec(4, T.i32)
                )
            )
            if const_expr(FP8_COMPUTE):
                # Round Q down to the cache's fp8, hoisted out of the page loop.
                return fx.Vector.from_elements(bf16_dwords_to_fp8(raw16, 4), fx.Int32)
            return raw16

        # Only the token-wave partial-max exchange needs LDS, and only when
        # there is more than one token wave; otherwise the kernel has none and
        # reports group_segment_fixed_size = 0.
        lds = (
            fx.SharedAllocator().allocate(SharedStorage).peek()
            if const_expr(SharedStorage is not None)
            else None
        )

        # -- stage Q in registers, held for the whole page loop ---------------
        # One 16 B gmem access per (tile, Q access), hoisted so the extra
        # k-steps gfx942 needs cost register slicing and not extra loads.
        q_raw = [
            [load_q_frag(i, j) for j in range_constexpr(tr.q_loads)]
            for i in range_constexpr(FEAT_TILES_PER_WAVE)
        ]

        def as_q_frag(raw, sub):
            """One MFMA B-fragment in the compute dtype.

            On the fp8 path a Q access is 8 fp8 = exactly one k-step (q_per_load
            is 1 for an fp8 cache on gfx950), so the slice is the whole access
            and this is a bitcast.
            """
            if const_expr(not FP8_COMPUTE):
                return as_bf16_frag(raw, sub)
            t = fx.make_rmem_tensor(fx.make_layout(tr.lane_k, 1), FP8_T)
            t.store(raw.bitcast(FP8_T))
            return t

        q_frag = [
            [
                as_q_frag(q_raw[i][ks // tr.q_per_load], ks % tr.q_per_load)
                for ks in range_constexpr(KSTEPS)
            ]
            for i in range_constexpr(FEAT_TILES_PER_WAVE)
        ]

        mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA_M, MFMA_N, tr.mfma_k, FP8_T)
            if const_expr(FP8_COMPUTE)
            # Same 16x16 tile and the same k on gfx950, so the fp8 form buys no
            # arithmetic. What it buys is deleting `convert_k` from the inner
            # loop, which runs per (page, tok_tile, k-step).
            else fx.rocdl.MFMA(MFMA_M, MFMA_N, tr.mfma_k, fx.BFloat16)
        )
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(NEG_INF)

        # -- page ids and their buffer descriptors, one batch up front --------
        # All of this wave's page ids, issued with no dependant between them so
        # they pipeline rather than costing a serial round trip per page. Pages
        # are strided by PAGE_SLOTS so consecutive slots cover consecutive
        # pages, keeping a wavefront's loads inside one span of block_table.
        #
        # `limit` is where this CTA's pages end: the request's end, or for a
        # spread row the end of its own span. Out-of-range entries are clamped
        # rather than predicated -- the value only forms an address and the
        # store it would feed is masked off in the epilogue. The clamp is
        # cmp+select, never maximumf/minimumf, which are float ops and would
        # compare these indices as bit patterns.
        if const_expr(SPREAD):
            first = c
            end = c + span
            limit = (end < num_pages).select(end, num_pages)
        else:
            first = c * fx.Int32(CHUNK)
            limit = num_pages
        pages = [
            first + fx.Int32(j * PAGE_SLOTS) + pw
            for j in range_constexpr(pages_per_wave)
        ]
        zero = fx.Int32(0)
        nm1 = limit - fx.Int32(1)
        last = (nm1 < zero).select(zero, nm1)

        # Widen BEFORE multiplication: a pool can exceed the descriptor's
        # 4 GiB range, and only within-page offsets belong in the index.
        page_bytes = fx.Int64(i32_stride_k_blk) * fx.Int64(1 if fp8 else 2)
        k_base = fx.Int64(fx.ptrtoint(arg_k))
        page_buffers = []
        for pp in pages:
            p_clamped = (pp < last).select(pp, last)
            # The block table is not sharded -- every rank sees all of it -- so
            # it is indexed with the global block id. An empty CP shard may have
            # no column for CP_RANK at all, yet still forms descriptors (and its
            # token-split waves execute barriers), so clamp that speculative
            # load to the always-present column 0.
            column = (num_pages > zero).select(global_block(p_clamped), zero)
            page = fx.Int32(
                fx.add_offset(fx.get_iter(bt_buf), b * i32_stride_bt_b + column).load(
                    T.i32
                )
            )
            address = k_base + fx.Int64(page) * page_bytes
            page_buffers.append(
                ptr_buf_tensor(
                    fx.inttoptr(arg_k.type, address),
                    fx.Int32,
                    unit_elems=4,
                    num_records_bytes=fx.Int32(page_bytes),
                )
            )

        # -- K fragment access ------------------------------------------------
        # K is streamed: every byte is read once per launch and never revisited,
        # so `nt_k` keeps it from allocating in L1 and paying a line fill per
        # access for no reuse. Q, the block table and the work map keep the
        # default policy -- they are small and genuinely re-read.
        k_atom = buf_copy_atom(16, fx.Int32, cache_modifier=nt_k)

        def load_k16(page_buf, unit):
            """One 16 B K access at 16 B-unit index `unit`.

            Goes through a copy atom rather than a plain `.load()` because the
            cache-policy field lives on the atom. Issue and consume stay split:
            the fragment is registers, so the compiler sinks the s_waitcnt to
            first use and a whole tok_tile's loads remain in flight.
            """
            src = fx.slice(page_buf, (unit, None))
            frag = fx.make_fragment_like(src)
            fx.copy(k_atom, src, frag)
            return fx.Vector(fx.memref_load_vec(frag))

        def issue_k(page, tok_tile, i):
            """Start load i of K[page, 16*tok_tile + u, :], per the k_offset map.

            `tok_tile` is this wave's *local* tok_tile index; tw_slot / tw_tok shift
            it onto the wave's own slice of the page's token axis.
            """
            if const_expr(shuffled):
                # Pre-shuffled cache: the 16 B chunk each lane wants for
                # (tok_tile, i) already sits at lane index, so one instruction
                # reads WAVE*16 = 1024 contiguous bytes instead of 16 chunks a
                # row-stride apart. Addressing involves neither u, the row
                # stride, nor k_offset.
                base = (
                    fx.Int32((tok_tile * K_LOADS + i) * WAVE) + tw_slot + lane
                ) * fx.Int32(CHUNK_ELEMS)
            else:
                # Access block i starts at k = i*block_k and lane group g takes
                # lane_block of it, so the block is one dwordx4 per lane and the
                # four g-lanes tile exactly one 64 B line. Both constants are
                # arch-independent; see ArchTraits.
                tok_row = fx.Int32(MFMA_M * tok_tile) + tw_tok + u
                base = (
                    tok_row * i32_stride_k_pos
                    + fx.Int32(i * tr.block_k)
                    + g * fx.Int32(tr.lane_block)
                )
            # `base` is in cache elements; the shift turns it into a 16 B unit
            # index, exactly -- CHUNK_ELEMS is one 16 B chunk by construction
            # and every stride above it is a multiple.
            shift = fx.Int32(4) if const_expr(fp8) else fx.Int32(3)
            return load_k16(page, base >> shift)

        def k_operand(raws, ks):
            """One MFMA A-fragment of K in the compute dtype.

            The bf16 path widens (`convert_k`). The fp8 path does not: a lane's
            k-step is lane_k fp8, an access holds k_per_load of them back to
            back by the k_offset map, so the fragment is already in the raw and
            this is a slice. That deleted widening is the whole point of
            `precision='fp8'` -- it runs per (page, tok_tile, k-step), unlike
            anything on the Q side.
            """
            if const_expr(not FP8_COMPUTE):
                return convert_k(raws, ks)
            raw = raws[ks // tr.k_per_load]
            dwords = tr.lane_k // 4
            lo = (ks % tr.k_per_load) * dwords
            t = fx.make_rmem_tensor(fx.make_layout(tr.lane_k, 1), FP8_T)
            t.store(
                fx.Vector.from_elements(
                    [fx.Int32(raw[lo + d]) for d in range_constexpr(dwords)],
                    fx.Int32,
                ).bitcast(FP8_T)
            )
            return t

        # ==================== MAIN LOOP ====================
        def score_tok_tile(p, tok_tile, raws, run_max):
            """MFMA one 16-token tile against this wave's feature tiles.

            p        logical block index of the page being scored
            tok_tile    the wave's local tok_tile index within that page
            raws     K loads issued earlier, K_LOADS of them
            run_max  per-feature running max, updated in place

            This is the whole arithmetic body of the kernel; everything else is
            address math and plumbing.
            """
            accs = []
            for i in range_constexpr(FEAT_TILES_PER_WAVE):
                acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                acc.store(zero4)
                accs.append(acc)
            for ks in range_constexpr(KSTEPS):
                kf = k_operand(raws, ks)
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    fx.gemm(mma_atom, accs[i], kf, q_frag[i][ks], accs[i])

            # This lane holds tokens 16*tok_tile + 4*g + r of the wave's slice,
            # i.e. tw_tok further along the page's token axis. The page's base
            # token is a *global* position: the cutoff it is compared against
            # counts the whole context, not this rank's share of it.
            #
            # The scale is not applied here. It is positive, so it commutes
            # with max, and folding it in once in the epilogue turns
            # TOK_TILES*FEAT_TILES*4 multiplies per page into FEAT_TILES.
            tok_base = (
                global_block(p) * fx.Int32(PAGE)
                + fx.Int32(MFMA_M * tok_tile)
                + tw_tok
                + g * fx.Int32(4)
            )
            for i in range_constexpr(FEAT_TILES_PER_WAVE):
                tok, _ = tok_head_of(i)
                causal_len = seq_len - fx.Int32(S) + tok + fx.Int32(1)
                v = fx.Vector(fx.memref_load_vec(accs[i]))
                for r in range_constexpr(4):
                    ok = (tok_base + fx.Int32(r)) < causal_len
                    run_max[i] = run_max[i].maximumf(
                        ok.select(fx.Float32(v[r]), neg_inf)
                    )

        # A wave whose first page is out of range skips everything: a short
        # request in a ragged batch would otherwise stream whole chunks of K it
        # can never use. At WAVES_PER_TOK > 1 there is no guard, because it is
        # wave-divergent and the epilogue's barrier has to be reached by every
        # wave in the CTA; those waves run on a clamped page and drop the
        # result at the store. A Python `True` folds at trace time, so that
        # case emits no branch at all.
        _live = True if const_expr(WAVES_PER_TOK > 1) else (pages[0] < limit)

        if _live:
            carry = None
            for j in range_constexpr(pages_per_wave):
                p = pages[j]
                page = page_buffers[j]
                nxt_page = page_buffers[j + 1] if j + 1 < pages_per_wave else None

                # ---------- MAIN LOOP: K pipeline + MFMA over one page --------
                # At WAVES_PER_TOK > 1 a wave walks only TOK_TILES_PER_WAVE of the page's
                # tok_tiles; the rest belong to other waves and the partial maxes
                # meet in LDS in the epilogue.
                if const_expr(1 <= sched <= 3):
                    fx.rocdl.iglp_opt(sched - 1)
                run_max = [neg_inf for _ in range_constexpr(FEAT_TILES_PER_WAVE)]
                if const_expr(fp8):
                    # Three tok_tiles in flight, capped by the wave's token slice,
                    # carried across pages. The scheduling boundaries are what
                    # keep the window from collapsing back to one tok_tile.
                    depth = min(3, TOK_TILES_PER_WAVE)
                    queue = list(carry) if carry is not None else []
                    for tok_tile in range_constexpr(len(queue), depth):
                        queue.append(
                            [
                                issue_k(page, tok_tile, i)
                                for i in range_constexpr(K_LOADS)
                            ]
                        )
                    carry = []
                    for tok_tile in range_constexpr(TOK_TILES_PER_WAVE):
                        raws = queue.pop(0)
                        fx.rocdl.sched_barrier(0)
                        score_tok_tile(p, tok_tile, raws, run_max)
                        fx.rocdl.sched_barrier(0)
                        future = tok_tile + depth
                        if const_expr(future < TOK_TILES_PER_WAVE):
                            queue.append(
                                [
                                    issue_k(page, future, i)
                                    for i in range_constexpr(K_LOADS)
                                ]
                            )
                        elif nxt_page is not None:
                            carry.append(
                                [
                                    issue_k(nxt_page, future - TOK_TILES_PER_WAVE, i)
                                    for i in range_constexpr(K_LOADS)
                                ]
                            )
                else:
                    # One tok_tile deep.
                    queue = (
                        list(carry)
                        if carry is not None
                        else [[issue_k(page, 0, i) for i in range_constexpr(K_LOADS)]]
                    )
                    carry = []
                    for tok_tile in range_constexpr(TOK_TILES_PER_WAVE):
                        raws = queue.pop(0)
                        # Issue tok_tile n+1 before consuming tok_tile n, so it is in
                        # flight during the MFMAs rather than waited on.
                        if const_expr(tok_tile + 1 < TOK_TILES_PER_WAVE):
                            queue.append(
                                [
                                    issue_k(page, tok_tile + 1, i)
                                    for i in range_constexpr(K_LOADS)
                                ]
                            )
                        elif nxt_page is not None:
                            carry.append(
                                [
                                    issue_k(nxt_page, 0, i)
                                    for i in range_constexpr(K_LOADS)
                                ]
                            )
                        score_tok_tile(p, tok_tile, raws, run_max)

                if const_expr(sched == 4):
                    # Ask for the order the source already has: one tok_tile's
                    # loads, then one tok_tile's MFMAs. Group id 0 throughout --
                    # these are one pipeline and the calls are read in order.
                    for _ in range_constexpr(TOK_TILES_PER_WAVE):
                        fx.rocdl.sched_group_barrier(_SCHED_VMEM_RD, K_LOADS, 0)
                        fx.rocdl.sched_group_barrier(
                            _SCHED_MFMA, FEAT_TILES_PER_WAVE * KSTEPS, 0
                        )

                # ---------- EPILOGUE: fold, scale, store one value/feature ----
                # Fold each feature's max across g, i.e. lanes {u, u+16, u+32,
                # u+48}. XOR 1/2/4/8 would mix features rather than tokens.
                # Afterwards every lane holds the tile's max for feature u,
                # replicated over g.
                folded = []
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    m = run_max[i]
                    for sh in (16, 32):
                        m = m.maximumf(m.shuffle_xor(sh, WAVE))
                    folded.append(m)

                if const_expr(WAVES_PER_TOK > 1):
                    # The waves splitting the page's token axis each hold a max
                    # over their own tok_tiles only, and LDS is the only place
                    # waves meet. Once per page on FEAT_TILES_PER_WAVE values, not per tok_tile.
                    # Every lane writes, so the write is one contiguous 256 B
                    # run per (wave, tile) and the read is lane-local.
                    red = lds.red.ptr
                    if const_expr(pages_per_wave > 1):
                        # Slots are reused every page: do not overwrite them
                        # until the previous page's readers are through.
                        gpu.barrier()
                    base = wave * fx.Int32(FEAT_TILES_PER_WAVE * WAVE) + lane
                    for i in range_constexpr(FEAT_TILES_PER_WAVE):
                        fx.ptr_store(folded[i], red + (base + fx.Int32(i * WAVE)))
                    gpu.barrier()
                    reduced = []
                    for i in range_constexpr(FEAT_TILES_PER_WAVE):
                        m = folded[i]
                        for t in range_constexpr(WAVES_PER_TOK):
                            # Same pw (same page), other tw.
                            w_other = fx.Int32(t) + pw * fx.Int32(WAVES_PER_TOK)
                            m = m.maximumf(
                                fx.ptr_load(
                                    red
                                    + (
                                        w_other * fx.Int32(FEAT_TILES_PER_WAVE * WAVE)
                                        + fx.Int32(i * WAVE)
                                    )
                                    + lane
                                )
                            )
                        reduced.append(m)
                    folded = reduced

                # Whether page p really exists. This gates the store, not the
                # work: page ids are clamped in the prologue, so an out-of-range
                # page reads real memory and computes a value nobody keeps. At
                # one page per wave the branch below has already proved the page
                # in range and the mask would be dead.
                valid = (
                    None
                    if const_expr(pages_per_wave == 1 and WAVES_PER_TOK == 1)
                    else p < limit
                )
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    # Scale folded in here rather than per accumulator: it is
                    # positive so max(s*x) == s*max(x), and -inf * s is still
                    # -inf, so a fully masked page still stores -inf.
                    m = folded[i] * f32_scale
                    tok, head = tok_head_of(i)
                    row = b * fx.Int32(S) + tok
                    addr = (
                        head * i32_stride_s_h
                        + row * i32_stride_s_b
                        + p * i32_stride_s_k
                    )
                    # One writer per feature. Padding columns write nothing --
                    # which also covers the tiles a ragged feature split hands
                    # the last wave -- and neither do pages past this request.
                    is_writer = (g == fx.Int32(0)) & (feature_of(i) < fx.Int32(F))
                    if const_expr(WAVES_PER_TOK > 1):
                        # All WAVES_PER_TOK waves hold the same value; pick one.
                        is_writer = is_writer & (tw == fx.Int32(0))
                    if valid is not None:
                        is_writer = is_writer & valid

                    if is_writer:
                        fx.add_offset(fx.get_iter(s_buf), addr).store(m)

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
    # scheduler reinvests them in more load hoisting. This is the channel that
    # actually forces the issue.
    if cfg.waves_per_eu:
        launch.compile_hints["waves_per_eu"] = cfg.waves_per_eu

    return launch, CHUNK


_CACHE = {}


def _get(S, H, fp8, cfg, device, arch=DEFAULT_ARCH):
    # IndexScoreConfig is frozen, hence hashable. The arch is in the key
    # because it changes the emitted code, not just where it runs.
    key = (S, H, fp8, cfg, device, arch)
    if key not in _CACHE:
        _CACHE[key] = build_index_score(S, H, fp8, cfg, arch)
    return _CACHE[key]


def shuffle_cache(cache):
    """Reorder a [pages, 128, 128] K cache into the kernel's shuffled layout.

    Element (row, k) moves to the slot the lane that wants it will read:

        load slot j = tok_tile * K_LOADS + i   (tok_tile = row // 16)
        lane        = 16 * g + u            (u = row % 16)
        position    = (j * WAVE + lane) * CHUNK_ELEMS + v

    where (g, v) come from inverting the kernel's k_offset map. Derived from
    that map rather than written independently, because the two must agree
    exactly -- a mismatch here is silent wrong numbers, not a crash.

    The layout is the same on gfx950 and gfx942 (see ArchTraits for why), so
    the producer -- which lives in another component -- does not have to know
    which chip will read it. Paid once when the cache is written.
    """
    import torch

    if cache.dtype not in (torch.bfloat16, dtypes.fp8):
        raise ValueError(f"cache: expected bfloat16 or {dtypes.fp8}")
    fp8 = cache.dtype != torch.bfloat16
    npages = cache.shape[0]
    tr = arch_traits(DEFAULT_ARCH, fp8)
    chunk_elems = tr.chunk_elems
    k_loads = tr.k_loads

    dev = cache.device
    row = torch.arange(PAGE, device=dev)
    kk = torch.arange(HEAD_DIM, device=dev)
    # For each (row, k), which lane/slot/offset does it belong to?
    tok_tile, u = row // MFMA_M, row % MFMA_M
    # Invert k_offset: an access block is block_k wide and lane group g owns
    # lane_block of it, so k = i*block_k + g*lane_block + v with v in
    # [0, chunk_elems). That is k = 64*i + 16*g + v for fp8 and
    # k = 32*i + 8*g + v for bf16, on either architecture.
    i = kk // tr.block_k
    g = (kk % tr.block_k) // tr.lane_block
    v = kk % tr.lane_block
    slot = tok_tile[:, None] * k_loads + i[None, :]
    fx_lane = 16 * g[None, :] + u[:, None]
    dst = (slot * WAVE + fx_lane) * chunk_elems + v[None, :]

    flat = cache.reshape(npages, PAGE * HEAD_DIM)
    out = torch.empty_like(flat)
    out.scatter_(1, dst.reshape(1, -1).expand(npages, -1), flat)
    return out.reshape(cache.shape)


def make_work_map(seq_lens, max_block, chunk, out=None, world: int = 1, rank: int = 0):
    """Dispatch order -> work item, packed so the holes all land at the end.

    seq_lens    [batch] int32 context lengths
    max_block   launch bound on blocks per request; LOCAL bound under CP
    chunk       pages one CTA covers, i.e. `work_chunk` for this config
    out         optional destination, sliced down if larger than needed
    world/rank  context-parallel shard, 1/0 for none

    The grid is `batch x ceil(max_block/chunk)`, sized by the longest request,
    so a ragged batch leaves holes in that rectangle, and where they sit is
    what they cost: each takes a launch slot and retires before the machine can
    build up K loads in flight. So number the real work 0..total-1 and give
    item n to the n-th dispatched CTA. Row n is

        [0] = (b << 16) | c    request, and which chunk of it; 0 for a hole
        [1] = seq_lens[b]      the kernel needs the length anyway, and keeping
                               it here holds the lookup to one cache line

    A hole reads as (0, 0), i.e. seq_len 0, so the kernel's existing tail guard
    retires it -- no extra branch, and page ids clamp to a legal address.

    Every step is a device op on `seq_lens`, so this is cudagraph-capturable
    and stays correct for whatever lengths a replay supplies. Call it once per
    decode step; the lengths do not change between layers.

    Under CP only the per-request chunk count changes -- a rank is given chunks
    for `_cp_blocks` of the blocks. Row [1] stays the *global* `seq_lens[b]`,
    because the kernel needs it for the causal cutoff and re-derives its own
    local count from it.
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
    else:
        out = torch.empty(
            (batch * chunks, 2), dtype=torch.int32, device=seq_lens.device
        )

    lens = seq_lens.to(torch.int32)
    nblk = (lens + (PAGE - 1)) // PAGE
    nblk = _cp_blocks_torch(nblk, world, rank)
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

    seq_lens    [batch] int32 context lengths
    rows        grid rows to fill; see _grid_chunks
    cu          CU count the spread is balanced against
    out         optional destination, sliced down if larger than needed
    world/rank  context-parallel shard, 1/0 for none

    make_work_map gives every CTA a full chunk of four pages, which spills as
    a second full CTA onto a few CUs once the live CTAs pass a multiple of the
    CU count. What costs is the pages piling onto one CU, not the CTA count --
    an extra CTA holding a single page is free.

    So a step needing r CU rounds keeps r - 1 rounds of full four-page CTAs
    and spreads what is left over the last round's CTAs, 1..4 pages each and
    the larger ones first, so no CU holds more than ceil(pages / cu) plus
    rounding. Spreading every round evenly balances as well but pays a CTA
    prologue per page or two.

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
    nblk = _cp_blocks_torch(nblk, world, rank)
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

    The torch version is ~40 small ops, whose launch cost dwarfs the scorer it
    prepares for. Every thread redoes the O(batch) per-request prefix
    arithmetic for its own row; at batch <= SPREAD_MAX_BATCH that is cheaper
    than a second pass or a barrier. Bit-identical to `make_spread_work_map`;
    the tests hold the two together.
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
                pages = _cp_blocks(pages, world, rank, zero)
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

        if n < i32_rows:
            base = fx.get_iter(out_buf)
            fx.add_offset(base, n * fx.Int32(2)).store(row0)
            fx.add_offset(base, n * fx.Int32(2) + fx.Int32(1)).store(row1)

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

    The torch version is a dozen small ops whose cost is launch floor, which a
    scorer this short cannot absorb. Each thread redoes the O(batch) prefix sum
    for its own row rather than sharing it. Bit-identical to `make_work_map`;
    the tests hold the two together.
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
                pages = _cp_blocks(pages, world, rank, zero)
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

        if n < i32_rows:
            base = fx.get_iter(out_buf)
            fx.add_offset(base, n * fx.Int32(2)).store(row0)
            fx.add_offset(base, n * fx.Int32(2) + fx.Int32(1)).store(row1)

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


def _run_map(key, build, seq_lens, rows, out, extra=()):
    """Launch a cached map-builder kernel. `extra` is its per-builder argument."""
    import torch

    if key not in _CACHE:
        _CACHE[key] = build()
    with torch.cuda.device(seq_lens.device):
        _run_compiled(
            _CACHE[key],
            ptr_arg(seq_lens, fx.Int32),
            ptr_arg(out, fx.Int32),
            rows,
            *extra,
            (rows + MAP_THREADS - 1) // MAP_THREADS,
            torch.cuda.current_stream(seq_lens.device).cuda_stream,
        )
    return out


def _run_chunk_map(seq_lens, rows, chunk, out, world, rank):
    batch = seq_lens.shape[0]
    return _run_map(
        ("chunk_map", batch, chunk, world, rank, seq_lens.device.index),
        lambda: build_chunk_map(batch, chunk, world, rank),
        seq_lens,
        rows,
        out,
    )


def _run_spread_map(seq_lens, rows, cu, out, world, rank):
    batch = seq_lens.shape[0]
    return _run_map(
        ("spread_map", batch, world, rank, seq_lens.device.index),
        lambda: build_spread_map(batch, world, rank),
        seq_lens,
        rows,
        out,
        (cu,),
    )


def work_map_size(
    batch: int, max_block: int, S: int = 0, H: int = 0, cfg=None, *, device=None
) -> int:
    """Rows `make_work_map` will produce -- i.e. the grid, one row per CTA.

    The buffer is `[rows, 2]` int32. A caller sizing a persistent one for a
    cudagraph must use `work_map_capacity` instead, because the exact grid size
    is non-monotonic in the bounds, and hand `make_work_map` a `buf[:rows]`
    slice -- still packed, so still a legal kernel argument.

    Exists so a caller does not re-derive the chunk size: it depends on the
    resolved `pages_per_wave`, which depends on the CU count, so a second copy
    would go wrong the first time this runs on a different chip.

    Under context parallelism `max_block` is the local bound, so this is the
    shard's grid. Pass S/H to enable automatic token splitting; omitting them
    keeps the legacy unsplit map, which `score_flydsl` accepts with the
    matching fallback geometry.
    """
    cfg = cfg or IndexScoreConfig()
    _validate_bounds(batch, max_block, cfg, resolved=False)
    cfg = resolve_config(batch, max_block, cfg, S, H, device=device)
    return _validate_bounds(batch, max_block, cfg)


def work_map_capacity(
    max_batch: int, max_block: int, S: int = 0, H: int = 0, cfg=None
) -> int:
    """Persistent rows covering every batch/block bound within the envelope.

    Sizes for the smallest CTA chunk automatic depth/token splitting can reach,
    independent of CU count, so the buffer stays valid on any chip. An explicit
    configuration keeps its own tighter geometry.

    The envelope deliberately includes Q1 even when S is a larger query bound:
    a serving caller may reuse a maximum-query allocation for decode replays,
    and asking with the largest S then launching Q1 is the natural mistake.
    It is not widened where `max_block` cannot pack at one page per chunk --
    there the block count is what hits the packing limit, and the token split
    that would need it cannot fire anyway, so widening would only reject
    bounds this accepts today for a geometry no launch can reach.
    """
    cfg = cfg or IndexScoreConfig()
    _validate_bounds(max_batch, max_block, cfg, resolved=False)
    auto = _auto_tok_split(cfg)
    split = auto and max_block <= _MAX_CHUNKS
    geometry = replace(
        cfg,
        waves_per_tok=WAVES if split else (cfg.waves_per_tok or 1),
        pages_per_wave=cfg.pages_per_wave or 1,
        nt_k=max(cfg.nt_k, 0),  # cache policy never changes the geometry
        spread=max(cfg.spread, 0),
    )
    rows = _validate_bounds(max_batch, max_block, geometry)
    # ...or the spread map the served path may resolve to instead, which only
    # batches up to SPREAD_MAX_BATCH can (its rows grow with the batch).
    spread_batch = min(max_batch, SPREAD_MAX_BATCH)
    if auto and _spread_fits(spread_batch, max_block, cfg):
        spread = replace(geometry, waves_per_tok=1, pages_per_wave=1, spread=1)
        rows = max(rows, _validate_bounds(spread_batch, max_block, spread))
    return rows


def build_work_map(seq_lens, max_block, S: int = 0, H: int = 0, cfg=None, out=None):
    """`make_work_map` with the chunk size resolved for you.

    The entry point a serving caller wants: `make_work_map` takes the chunk
    size as a number, so a caller that builds the map in one place and launches
    in another would have to resolve the config twice and keep the two
    agreeing. A disagreement there is not a crash, it is a map the kernel
    indexes with the wrong stride.

    `out` may be larger than needed -- a persistent worst-case buffer from
    `work_map_capacity` -- and is sliced down here; a short one is an error
    rather than a silently truncated grid.
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
        cfg.waves_per_tok,
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
    if cfg.waves_per_tok < (1 if resolved else 0) or WAVES % (cfg.waves_per_tok or 1):
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
    # Arch first: on an unsupported arch every other message would be a
    # symptom, and the accepted fp8 flavour depends on the answer.
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"index score requires one of {SUPPORTED_ARCHS}, got {arch}")
    # Q may arrive bf16 or already fp8 -- this used to accept bf16 only while
    # the pointer was bound as bf16 unconditionally, so an fp8 Q was a silent
    # misread. 16 B alignment covers both widths.
    _validate_tensor(idx_q, "idx_q", (torch.bfloat16, dtypes.fp8), 3, device, 16)
    _validate_tensor(cache, "cache", (torch.bfloat16, dtypes.fp8), 3, device, 16, False)
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
    auto_tokens = _auto_tok_split(cfg)
    _validate_bounds(batch, max_block, cfg, resolved=False)
    cfg = resolve_config(batch, max_block, cfg, S, H, device=device)
    # Legacy build_work_map(seq_lens, max_block) has no query dimensions and
    # therefore uses unsplit geometry. Preserve those supplied maps, but only
    # for the auto branch: an explicit split must still require its exact grid.
    if (
        auto_tokens
        and cfg.waves_per_tok > 1
        and isinstance(work_map, torch.Tensor)
        and work_map.ndim == 2
        and work_map.shape[0] == batch * ((max_block + WAVES - 1) // WAVES)
        and work_map.shape[0] != batch * max_block
    ):
        cfg = replace(cfg, waves_per_tok=1)
    rows = _validate_bounds(batch, max_block, cfg)
    if cfg.shuffled and cache.stride(1) != HEAD_DIM:
        raise ValueError("shuffled cache must be packed within each page")
    if not selection_filter(S, H, cfg, arch=arch):
        raise ValueError(f"illegal config for S={S} H={H}: {cfg}")
    # The other half of the precision check: `selection_filter` never sees the
    # cache, so the dtype half has to live where the tensor does.
    if cfg.precision == "fp8" and cache.dtype == torch.bfloat16:
        raise ValueError("precision='fp8' needs an fp8 cache")
    if block_table is not None:
        _validate_tensor(block_table, "block_table", (torch.int32,), 2, device)
        if (
            block_table.shape[0] != batch
            or block_table.shape[1] < max_block
            or block_table.stride(1) != 1
            or block_table.stride(0) < block_table.shape[1]
        ):
            raise ValueError("block_table: invalid shape or layout")
        # Under context parallelism the table is indexed by the GLOBAL block
        # `p * cp_world + cp_rank`, so a rank with a non-empty shard needs
        # columns well past `max_block` -- which is its LOCAL page count. The
        # width that needs is a function of seq_lens, and this path is
        # deliberately metadata-only (reading seq_lens means a device sync on
        # every call), so the table width stays a caller contract; see `cp_world`.
        #
        # The worst case `(max_block - 1) * cp_world + cp_rank` is NOT usable as
        # a check here. A caller whose shard is empty has no local page to state
        # and passes the global count instead, and the kernel clamps its
        # speculative block-table load to column 0 -- legal at any width, and
        # pinned by `test_empty_cp_shard_narrow_table`. Rejecting on the worst
        # case would turn that supported call into an argument error.
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
    return batch, cfg, arch


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

    Returns [H, batch*S, max_block] fp32 either way; only the strides differ,
    so this is invisible to every consumer -- both selectors take the three
    score strides as arguments -- except in the time it takes.

    Why it matters: the epilogue stores one fp32 per feature per page, so with
    F = S*H features the contiguous layout puts those F lanes on F different
    cache lines `max_block` floats apart. Making the feature axis contiguous
    instead packs the same values into one full 128 B line per store.

    The consumer pays a little for it -- its page axis goes from contiguous to
    total_q*H*4 strided -- and it loses eligibility for the aiter selector,
    which needs `score.view(rows, max_block)`. At the shapes where this layout
    wins, that selector already declines on its LDS budget, but a caller with a
    small max_block should check rather than assume.

    Below F = 16 the store is at most 16 bytes, so there is no scatter left to
    fix and those shapes stay contiguous.
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
    naming its fields (`shuffled=True`, `waves_per_tok=2`, ...) for callers that
    only want to flip one.
    """
    import torch

    if cfg is None:
        cfg = IndexScoreConfig(**cfg_kwargs)
    elif cfg_kwargs:
        raise TypeError("pass either cfg or its fields as keywords, not both")

    batch, cfg, arch = _validate_metadata(
        idx_q, cache, S, H, max_block, block_table, cfg, seq_lens, out, work_map
    )
    if block_table is None or seq_lens is None:
        raise ValueError("block_table and seq_lens are required")
    scaled = sm_scale * LOG2E
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        raise ValueError("sm_scale * LOG2E must be finite and positive in FP32")
    fp8 = cache.dtype == dtypes.fp8
    # Q's dtype decides how the pointer is bound; _validate_metadata has
    # already rejected anything that is neither.
    fp8_q = idx_q.dtype == dtypes.fp8
    cfg = replace(cfg, fp8_q=int(fp8_q))

    if out is None:
        out = alloc_score(batch, S, H, max_block, idx_q.device)

    launch, _chunk = _get(S, H, fp8, cfg, idx_q.device.index, arch)
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
            ptr_arg(idx_q, _fp8_t(arch) if fp8_q else fx.BFloat16),
            # `dtypes.fp8` has already matched both tensors against the flavour
            # this chip speaks, so the FlyDSL element type has to name the same
            # one: e4m3fn on gfx950, e4m3fnuz on gfx942. See `_fp8_t` -- the two
            # biases differ, and binding the wrong one is silently wrong values,
            # not a failure.
            ptr_arg(cache, _fp8_t(arch) if fp8 else fx.BFloat16),
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
            float(scaled),
            batch,
            chunks,
            torch.cuda.current_stream(idx_q.device).cuda_stream,
        )
    return out
