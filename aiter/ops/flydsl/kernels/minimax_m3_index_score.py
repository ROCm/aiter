# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 lightning-indexer decode block-score kernel (FlyDSL, gfx950).

For each request b, each 128-token page p and each (query row, index head),

    score[head, row, p] = max over t in [0,128) of
                              dot(K[page, t, :], Q[row, head, :]) * scale

with a per-query causal cutoff. K is the fp8 index cache
`fused_qknorm_idxrqknorm` writes; Q is bf16.

The MFMA tile is 16x16x32 with A = K (M = token) and B = Q (N = feature), which
is what makes the page reduction cheap: with lane = tid % 64, g = lane // 16
and u = lane % 16,

    A operand[v] = A[u, 8*g + v]     v = 0..7     A = K, M = token
    B operand[v] = B[u, 8*g + v]     v = 0..7     B = Q, N = feature
    C accum[r]   = C[4*g + r, u]     r = 0..3

so a lane holds tokens {4g+r} at feature u. The 128-token max is therefore a
register-local elementwise max, and only the final fold across g needs two
shuffle_xor steps; at one wave per page there is no LDS and no barrier at all.
A 32x32 tile would spread the reduced axis across lanes instead and need DPP,
permlane and LDS to close it.

K is widened fp8 -> bf16 and the MFMA is bf16, because the operator this
replaces lifts K to Q's precision rather than rounding Q down. That is a
numerics contract: the selector consumes these scores as a top-k ranking, and
a tie reordered near the cut changes which blocks are chosen.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import triton
import triton.language as tl
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

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

# The widest 16x16 bf16 MFMA each supported generation has -- the whole ISA
# ---------------------------------------------------------------------------
# gfx950 fp8 geometry
# ---------------------------------------------------------------------------
# One arch, one cache dtype, so what used to be an (arch, dtype) lookup is
# eight constants. They describe the k-axis map: access block j spans BLOCK_K
# of the head dim and lane group g takes LANE_BLOCK of it, so the four
# g-lanes tile one 64 B line. The prefill scorer reads the same cache and
# must agree.
MFMA_K = 32  # k per MFMA: 16x16x32 bf16, the widest gfx950 has
LANE_K = 8  # k values one lane holds for one k-step
KSTEPS = 4  # MFMA k-steps per 128-wide dot
K_PER_LOAD = 2  # k-steps one 16 B K access covers
K_LOADS = 2  # K accesses per 16-token tile
Q_LOADS = 4  # Q accesses per feature tile
BLOCK_K = 64  # k an access block spans across the four lane groups
LANE_BLOCK = 16  # k one lane group takes inside that block
ACCESS_BYTES = 16


def fp8_dwords_to_bf16(raw, lo, dwords):
    """`dwords` packed-fp8 dwords of `raw` from index `lo`, as 4*dwords bf16.

    Split out of `convert_k` so an fp8 Q is widened by exactly the conversion K
    is widened by; two copies would be silent wrong numbers, not a crash. The
    cache is unit-scale, hence scale 1.0.
    """
    one = _to_raw(fx.Float32(1.0))
    elems = []
    for d in range_constexpr(dwords):
        word = _to_raw(fx.Int32(raw[lo + d]))
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), word, one, bool(half)
                )
            )
            elems += [fx.BFloat16(pair[0]), fx.BFloat16(pair[1])]
    return elems


def fragment_helpers(g):
    """Per-lane-group MFMA fragment builders, shared with the prefill scorer.

    `g` is the caller's lane-group expression. Shared rather than copied: both
    scorers read the same cache through the same k-axis permutation, and two
    copies that drifted would be silent wrong numbers, not a crash.
    """
    # Q accesses per K access block. Q covers one k-step per access, K two.
    Q_SPLIT = K_PER_LOAD
    Q_HALF = LANE_BLOCK // Q_SPLIT

    def q_load_offset(j):
        """k position of this lane group's 16 B Q access `j`. `j` is a python
        int on the register path and an Int32 on an LDS fill path."""
        if const_expr(isinstance(j, int)):
            return (
                fx.Int32((j // Q_SPLIT) * BLOCK_K)
                + g * fx.Int32(LANE_BLOCK)
                + fx.Int32((j % Q_SPLIT) * Q_HALF)
            )
        return (
            (j // fx.Int32(Q_SPLIT)) * fx.Int32(BLOCK_K)
            + g * fx.Int32(LANE_BLOCK)
            + (j % fx.Int32(Q_SPLIT)) * fx.Int32(Q_HALF)
        )

    def as_bf16_frag(raw16, sub=0):
        """MFMA B-fragment of a 16 B Q access. One access is exactly one
        k-step here, so `sub` is accepted and ignored."""
        t = fx.make_rmem_tensor(fx.make_layout(LANE_K, 1), fx.BFloat16)
        t.store(raw16.bitcast(fx.BFloat16))
        return t

    def convert_k(raws, ks):
        """Widen the raw access holding k-step `ks` into a bf16 A-fragment.

        fp8 is widened rather than fed to a native fp8 MFMA because the
        operator this replaces lifts K to Q's precision instead of rounding Q
        down. That is a numerics contract, not a performance choice.
        """
        t = fx.make_rmem_tensor(fx.make_layout(LANE_K, 1), fx.BFloat16)
        t.store(
            fx.Vector.from_elements(
                fp8_dwords_to_bf16(
                    raws[ks // K_PER_LOAD],
                    (ks % K_PER_LOAD) * (LANE_K // 4),
                    LANE_K // 4,
                ),
                fx.BFloat16,
            )
        )
        return t

    return q_load_offset, as_bf16_frag, convert_k


def bf16_dwords_to_fp8(raw, dwords):
    """`dwords` dwords of bf16 in `raw` as dwords//2 packed-fp8 dwords.

    The rounding the prefill scorer applies to Q, shared with it. Rounding Q
    down rather than lifting K up is a DIFFERENT numerics contract from this
    kernel's default -- see `IndexScoreConfig.precision`.
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


# pages_per_wave for the served MiniMax-M3 decode launches, keyed by (S, H):
# (largest graph batch, depth) steps, then 4 beyond the last one.
_SERVED_DECODE_DEPTH = {
    (4, 1): ((4, 1), (16, 2)),  # TP4 MTP: Q4, one index head
    (8, 4): ((2, 1), (4, 2), (16, 3)),  # Q8, four index heads
}
# Every other one-feature-tile shape (S*H <= 16) was swept and behaves within a
# few percent of S4H1 at every batch, so they share its table.
_ONE_TILE_DECODE_DEPTH = _SERVED_DECODE_DEPTH[(4, 1)]


def _decode_depth_table(S: int, H: int):
    if (S, H) in _SERVED_DECODE_DEPTH:
        return _SERVED_DECODE_DEPTH[(S, H)]
    if S and H and feature_tiles(S, H) == 1:
        return _ONE_TILE_DECODE_DEPTH
    return None


def _cp_blocks(pages, world: int, rank: int, zero):
    """This rank's share of `pages` blocks: ceil(max(pages - r, 0) / world).

    The clamp is cmp+select, never maximumf -- a float op would compare these
    indices as bit patterns. The kernel and both work-map builders must agree
    on this number, or the shard's chunks are numbered against a different
    length, so the scorer and both plan kernels share this one formula.
    """
    if world == 1:
        return pages
    avail = pages - fx.Int32(rank)
    avail = (avail < zero).select(zero, avail)
    return (avail + fx.Int32(world - 1)) // fx.Int32(world)


def feature_tiles(S: int, H: int) -> int:
    """Number of 16-wide feature tiles needed to cover F = S*H columns."""
    return (S * H + MFMA_N - 1) // MFMA_N


def reduce_lds_bytes(S: int, H: int, waves_per_tok: int) -> int:
    """LDS held by the cross-wave partial-max exchange, 0 at waves_per_tok 1.

    One fp32 per (wave, tile, lane): keeping all 64 lanes rather than the g == 0
    writers makes the write contiguous and the read lane-local, for 2 KB at
    S=8 H=4."""
    if waves_per_tok <= 1:
        return 0
    return WAVES * feature_tiles(S, H) * WAVE * 4


def work_chunk(geom) -> int:
    """Pages one CTA covers -- the granularity make_work_map hands out."""
    return (WAVES // geom["waves_per_tok"]) * geom["pages_per_wave"]


def _grid_chunks(max_block: int, geom) -> int:
    """Grid columns per request. A spread map needs one spare row per request:
    it may cut up to one CTA per request boundary, and the spare column keeps
    four pages per CTA enough at full capacity."""
    chunk = work_chunk(geom)
    return (max_block + chunk - 1) // chunk + (1 if geom["spread"] else 0)


def _spread_fits(batch: int, max_block: int, cp_world: int = 1) -> bool:
    """A spread row packs the first page in 16 bits and the page count in the
    top byte of the length word, so both have to fit; the map kernel unrolls
    the batch."""
    return (
        batch <= SPREAD_MAX_BATCH
        and max_block <= 0xFFFF
        and max_block * cp_world * PAGE <= 0xFFFFFF
    )


# The spread map is built by one small kernel with the request loop unrolled,
# so it is limited to batches this size (the served ladder stops at 4).
SPREAD_MAX_BATCH = 64

# Chunks the packed work map can address. A row names its chunk in 16 bits, so
# a grid wider than this cannot be packed at all -- which is also what bounds
# `max_block` once the chunk is one page.
_MAX_CHUNKS = 0x10000


# Oversubscription the page loop may spend: below this the machine is short of
# waves and a longer per-wave loop only makes it shorter. Mirrors
# _auto_num_splits in mqa_logits/fp8_mqa_logits.py.
_CTA_OVERSUBSCRIBE = 4


def _cu_count(device=None) -> int:
    """CU count, with `CU_NUM` winning over the device.

    `work_map_size` has no tensor and so resolves with `device=None`, while
    `build_work_map` resolves from `seq_lens.device`. If the two read different
    counts the geometry differs and the buffer one sizes is rejected by the
    other -- with `CU_NUM=64` on a 256-CU part that is 64 rows against 128. So
    the override is honoured on both paths, not just the device-less one.
    """
    import os

    import torch

    from aiter.jit.utils.chip_info import get_cu_num

    if device is None or os.environ.get("CU_NUM"):
        return get_cu_num()
    return torch.cuda.get_device_properties(device).multi_processor_count


def geometry(batch, max_block, S, H, cp_world=1, cp_rank=0, device=None):
    """Pick the launch geometry from the BOUNDS -- never seq_lens contents, so
    the grid stays valid across a cudagraph replay. Returns a plain dict.

    nt_k            one K instruction takes 64 B from each of 16 rows, so every
                    128 B line is requested twice and L1 turns the second into a
                    hit where NT would send it back to L2. Only the served path,
                    whose reuse distance is short, keeps L1.
    pages_per_wave  deeper keeps loads in flight across the page seam but
                    divides the CTA count. Served launches take it from a
                    measured table keyed on the graph batch; the rest estimate
                    CTA supply from `batch * max_block`, which over-counts a
                    ragged batch and so errs towards the longer loop.
    waves_per_tok   splitting a page's token axis helps only where there are too
                    few pages to fill the machine. Unlike splitting the FEATURE
                    axis it duplicates no reads, costing one LDS round trip and
                    one barrier per page.
    spread          what costs is pages piling onto one CU, not the CTA count:
                    an extra CTA holding one page is nearly free, holding four
                    is not.
    """
    if not isinstance(batch, int) or not 0 <= batch <= 0xFFFF:
        raise ValueError("batch must be in [0, 65535] for packed request IDs")
    if not isinstance(max_block, int) or max_block < 1:
        raise ValueError("max_block must be positive")
    if cp_world < 1 or not 0 <= cp_rank < cp_world:
        raise ValueError("invalid CP shard")
    # A decode launch with a measured depth table and no CP. Past 0xFFFF blocks
    # those shallow depths would not pack, so those capacities fall through to
    # the estimate.
    served = (
        _decode_depth_table(S, H) is not None and cp_world == 1 and max_block <= 0xFFFF
    )
    if served and S == 1 and batch * ((max_block + 1) // 2) <= _cu_count(device):
        # Q1 launches small enough for the token split below keep it: that
        # regime was measured separately and the depth tables were not.
        served = False
    waves_per_tok, spread = 1, 0
    if served:
        # Past the tabled batches a shallower loop wins while the request is
        # short: most of a page's cost is the K stream, and a deep per-wave loop
        # just serialises more of it behind one CTA. That is worth 5-19% across
        # max_block 48-128.
        #
        # Above max_block 256 the depth stops mattering -- swept to 8192, all of
        # 1, 2 and 4 land within 2% of each other. By then the kernel is reading
        # at 90-95% of this part's HBM ceiling, so there is nothing for the loop
        # shape to win. 4 is kept there because it is what shipped; 2 would do
        # as well.
        depth = 2 if max_block <= 256 else 4
        for limit, tabled in _decode_depth_table(S, H):
            if batch <= limit:
                depth = tabled
                break
        # The tabled depth, capped at the deepest that still hands every CU a
        # CTA: the table is keyed on batch alone, but a full batch of *short*
        # requests has very little work.
        cap = max(1, (batch * max_block) // (WAVES * _cu_count(device)))
        pages_per_wave = min(depth, cap)
        spread = int(pages_per_wave == 1 and _spread_fits(batch, max_block, cp_world))
    else:
        cu = _cu_count(device)
        # One page per CTA fills otherwise missing CU scheduling slots. Only
        # Q1/one-tile launches were measured for this path.
        if S == 1 and 0 < H <= MFMA_N:
            if batch * max_block <= cu:
                waves_per_tok = WAVES
            elif batch * ((max_block + 1) // 2) <= cu:
                # Two pages/CTA keep one scheduling round while halving each
                # wave's tok_tile chain. Beyond one round the LDS exchange loses.
                waves_per_tok = 2
        pages_per_wave = 1
        for ppw in (2, 4):
            if batch * -(-max_block // ((WAVES // waves_per_tok) * ppw)) >= (
                cu * _CTA_OVERSUBSCRIBE
            ):
                pages_per_wave = ppw
    if waves_per_tok > 1:
        from aiter.jit.utils.chip_info import get_lds_capacity_bytes

        if reduce_lds_bytes(S, H, waves_per_tok) > get_lds_capacity_bytes("gfx950"):
            waves_per_tok = 1
    return {
        "waves_per_tok": waves_per_tok,
        "pages_per_wave": pages_per_wave,
        "nt_k": 0 if served else 2,
        "spread": spread,
        "cp_world": cp_world,
        "cp_rank": cp_rank,
    }


def kernel_name(S: int, H: int, geom) -> str:
    """Geometry -> kernel name. The default launch keeps the name it had before
    any of these knobs existed, so its JIT cache entries are not orphaned."""
    name = f"m3_index_score_S{S}_H{H}_fp8_L{geom['pages_per_wave']}"
    if geom["waves_per_tok"] > 1:
        name += f"_tw{geom['waves_per_tok']}"
    if geom["nt_k"] != 2:  # 2 is the default policy; only mark the others
        name += f"_kc{geom['nt_k']}"
    if geom["spread"]:
        name += "_sp"
    if geom["cp_world"] > 1:
        # The rank is in the name, not just the world: ranks sharing a node
        # share the JIT cache directory and compile DIFFERENT kernels, since
        # the block remap folds the rank in as a constant.
        name += f"_cp{geom['cp_world']}r{geom['cp_rank']}"
    return name


def build_index_score(S: int, H: int, geom):
    """Compile a score kernel specialised on (S, H, geometry, Q dtype).

    S = max query tokens per request (num_spec + 1), H = index heads, and
    F = S*H is the feature count, laid out as column n = tok*H + head.
    """
    pages_per_wave = geom["pages_per_wave"]
    nt_k = geom["nt_k"]
    # Context-parallel block remap. At 1/0 every use below is the identity and
    # the emitted code is byte-identical to the non-CP kernel.
    CP_WORLD = geom["cp_world"]
    CP_RANK = geom["cp_rank"]
    SPREAD = bool(geom["spread"])
    F = S * H
    # Every wave owns every feature tile; the CTA's waves divide the block
    # axis instead -- WAVES_PER_TOK of them share one page by splitting its
    # token tiles, and PAGE_SLOTS groups of those cover distinct pages.
    FEAT_TILES_PER_WAVE = feature_tiles(S, H)
    WAVES_PER_TOK = geom["waves_per_tok"]
    PAGE_SLOTS = WAVES // WAVES_PER_TOK
    TOK_TILES_PER_WAVE = TOK_TILES // WAVES_PER_TOK  # token tiles one wave walks
    CHUNK = PAGE_SLOTS * pages_per_wave
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
        name=kernel_name(S, H, geom),
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

        arg_q      [batch*S, H, 128] bf16 or fp8 query
        arg_k      [pages, 128, 128] fp8 index key cache
        arg_score  [H, batch*S, max_block] fp32, one per (feature, page)
        arg_bt     [batch, >= max_block] int32 block table
        arg_work   [rows, 2] int32 work map; see make_work_map
        f32_scale  score multiplier, already folded with LOG2E by the host

        Everything is constexpr-unrolled, so "loop" is the shape of the emitted
        code, not a branch. Wave w takes tw (slice of the page's token axis) and
        pw (page slot); at WAVES_PER_TOK == 1 a wave owns whole pages, the
        reduction stays register-local and there is no LDS and no barrier at
        all. Above it the per-page epilogue exchanges partial maxes, the only
        place a barrier appears.
        """
        # ==================== PROLOGUE ====================
        # The CTA looks up which (b, c) the n-th *dispatched* block owns rather
        # than taking (block.x, block.y), so a ragged batch's holes all land
        # past the end of the work; see make_work_map. block.x is the fast
        # dispatch axis, so n is monotonic in launch order.
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
        # Q is read as i32 and bitcast so a fragment lands as one dwordx4 or
        # dwordx2, not 8 scalar loads; every base is 8-element aligned.
        q_buf = ptr_buf_tensor(arg_q, fx.Int32)
        s_buf = ptr_buf_tensor(arg_score, fx.Float32)
        bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
        wm_buf = ptr_buf_tensor(arg_work, fx.Int32)
        # Work-map entry n is two adjacent dwords, (packed, seq_len): one cache
        # line, one round trip. A hole reads as (0, 0), so num_pages is zero and
        # the tail guard retires the CTA with no extra branch.
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
        # Pages this request owns; the grid is sized from max_seq_len.
        num_pages = (seq_len + fx.Int32(PAGE - 1)) // fx.Int32(PAGE)
        if const_expr(CP_WORLD > 1):
            # ...and of those, the ones this rank owns; see `_cp_blocks`.
            num_pages = _cp_blocks(num_pages, CP_WORLD, CP_RANK, fx.Int32(0))

        def global_block(p):
            """Global block id of this rank's logical block `p`."""
            if const_expr(CP_WORLD == 1 and CP_RANK == 0):
                return p
            return p * fx.Int32(CP_WORLD) + fx.Int32(CP_RANK)

        # -- feature decode: which (tok, head) does this lane's column carry? --
        # Column f = 16*ft + u and f = tok*H + head, with H constexpr. Note
        # causal_len depends on tok, hence on u: it must not be hoisted.
        def feature_of(i):
            return fx.Int32(MFMA_N * i) + u

        def tok_head_of(i):
            f = feature_of(i)
            tok = f // fx.Int32(H)
            return tok, f - tok * fx.Int32(H)

        # Any bijection (g, ks, v) -> k works provided Q and K use the SAME
        # one; this one is ArchTraits.k_offset, which starts access j of lane
        # group g at byte 64*j + 16*g so the four g-lanes tile one 64 B line.
        # Q is addressed by access index, not k-step.
        q_load_offset, as_bf16_frag, convert_k = fragment_helpers(g)

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
            # Q is bf16, which is also the compute dtype, so there is no
            # conversion on this path at all.
            return fx.Vector(
                fx.add_offset(fx.get_iter(q_buf), off >> fx.Int32(1)).load(
                    T.vec(4, T.i32)
                )
            )

        # Only the token-wave partial-max exchange needs LDS; otherwise the
        # kernel has none and reports group_segment_fixed_size = 0.
        lds = (
            fx.SharedAllocator().allocate(SharedStorage).peek()
            if const_expr(SharedStorage is not None)
            else None
        )
        # -- stage Q in registers, held for the whole page loop ---------------
        # One 16 B gmem access per (tile, Q access), hoisted so the extra
        # k-steps gfx942 needs cost register slicing, not loads.
        q_raw = [
            [load_q_frag(i, j) for j in range_constexpr(Q_LOADS)]
            for i in range_constexpr(FEAT_TILES_PER_WAVE)
        ]

        q_frag = [
            [as_bf16_frag(q_raw[i][ks // 1], ks % 1) for ks in range_constexpr(KSTEPS)]
            for i in range_constexpr(FEAT_TILES_PER_WAVE)
        ]
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(MFMA_M, MFMA_N, MFMA_K, fx.BFloat16))
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(NEG_INF)
        # -- page ids and their buffer descriptors, one batch up front --------
        # Issued with no dependant between them so they pipeline instead of costing
        # a serial round trip per page. `limit` is where this CTA's pages end: the
        # request's end, or a spread row's own span. Out-of-range entries are
        # clamped rather than predicated -- the value only forms an address and the
        # store it feeds is masked off -- and the clamp is cmp+select, never
        # maximumf, which would compare these indices as bit patterns.
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
        page_bytes = fx.Int64(i32_stride_k_blk)
        k_base = fx.Int64(fx.ptrtoint(arg_k))
        page_buffers = []
        for pp in pages:
            p_clamped = (pp < last).select(pp, last)
            # The block table is not sharded, so it is indexed with the global
            # block id. An empty CP shard may have no column for CP_RANK at
            # all, yet still forms descriptors (and its token-split waves
            # execute barriers), so clamp that speculative load to column 0.
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
        # K is streamed -- every byte read once per launch -- so `nt_k` keeps it
        # from allocating in L1 for no reuse. Q, the block table and the work
        # map keep the default policy.
        k_atom = buf_copy_atom(16, fx.Int32, cache_modifier=nt_k)

        def load_k16(page_buf, unit):
            """One 16 B K access at 16 B-unit index `unit`.

            A copy atom rather than a plain `.load()` because the cache-policy
            field lives on the atom. Issue and consume stay split: the compiler
            sinks the s_waitcnt to first use, so a whole tok_tile's loads
            remain in flight."""
            src = fx.slice(page_buf, (unit, None))
            frag = fx.make_fragment_like(src)
            fx.copy(k_atom, src, frag)
            return fx.Vector(fx.memref_load_vec(frag))

        def issue_k(page, tok_tile, i):
            """Start load i of K[page, 16*tok_tile + u, :], per the k_offset
            map. `tok_tile` is this wave's *local* index; tw_slot / tw_tok
            shift it onto the wave's own slice of the token axis."""
            # Access block i starts at k = i*block_k and lane group g takes
            # lane_block of it, so the four g-lanes tile one 64 B line.
            tok_row = fx.Int32(MFMA_M * tok_tile) + tw_tok + u
            base = (
                tok_row * i32_stride_k_pos
                + fx.Int32(i * BLOCK_K)
                + g * fx.Int32(LANE_BLOCK)
            )
            # `base` counts fp8 elements; 16 of them are one 16 B access, and
            # every stride above that is a multiple, so the shift is exact.
            return load_k16(page, base >> fx.Int32(4))

        # ==================== MAIN LOOP ====================
        def score_tok_tile(p, tok_tile, raws, run_max):
            """MFMA one 16-token tile against this wave's feature tiles, into
            the per-feature running max. `p` is the logical block index and
            `tok_tile` the wave's local index within that page. The whole
            arithmetic body of the kernel; the rest is addressing."""
            accs = []
            for i in range_constexpr(FEAT_TILES_PER_WAVE):
                acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                acc.store(zero4)
                accs.append(acc)
            for ks in range_constexpr(KSTEPS):
                kf = convert_k(raws, ks)
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    fx.gemm(mma_atom, accs[i], kf, q_frag[i][ks], accs[i])
            # This lane holds tokens 16*tok_tile + 4*g + r of the wave's slice.
            # The page's base token is a *global* position: the cutoff counts
            # the whole context, not this rank's share. The scale is applied in
            # the epilogue instead; being positive, it commutes with max.
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

        # A wave whose first page is out of range skips everything. At
        # WAVES_PER_TOK > 1 there is no guard -- it would be wave-divergent and every
        # wave must reach the epilogue's barrier -- so those waves run on a clamped
        # page and drop the result at the store.
        _live = True if const_expr(WAVES_PER_TOK > 1) else (pages[0] < limit)
        if _live:
            carry = None
            for j in range_constexpr(pages_per_wave):
                p = pages[j]
                page = page_buffers[j]
                nxt_page = page_buffers[j + 1] if j + 1 < pages_per_wave else None
                # ---------- MAIN LOOP: K pipeline + MFMA over one page --------
                run_max = [neg_inf for _ in range_constexpr(FEAT_TILES_PER_WAVE)]
                # Three tok_tiles in flight, capped by the wave's token slice
                # and carried across pages, consumed before the queue refills.
                # The scheduling boundaries below are what keep that window
                # from collapsing back to one tok_tile.
                depth = min(3, TOK_TILES_PER_WAVE)
                queue = list(carry) if carry is not None else []
                for tok_tile in range_constexpr(len(queue), depth):
                    queue.append(
                        [issue_k(page, tok_tile, i) for i in range_constexpr(K_LOADS)]
                    )
                carry = []
                for tok_tile in range_constexpr(TOK_TILES_PER_WAVE):
                    raws = queue.pop(0)
                    future = tok_tile + depth
                    fx.rocdl.sched_barrier(0)
                    score_tok_tile(p, tok_tile, raws, run_max)
                    fx.rocdl.sched_barrier(0)
                    if const_expr(future < TOK_TILES_PER_WAVE):
                        queue.append(
                            [issue_k(page, future, i) for i in range_constexpr(K_LOADS)]
                        )
                    elif nxt_page is not None:
                        carry.append(
                            [
                                issue_k(nxt_page, future - TOK_TILES_PER_WAVE, i)
                                for i in range_constexpr(K_LOADS)
                            ]
                        )
                # ---------- EPILOGUE: fold, scale, store one value/feature ----
                # Fold across g, i.e. lanes {u, u+16, u+32, u+48}; XOR 1/2/4/8
                # would mix features rather than tokens.
                folded = []
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    m = run_max[i]
                    for sh in (16, 32):
                        m = m.maximumf(m.shuffle_xor(sh, WAVE))
                    folded.append(m)
                if const_expr(WAVES_PER_TOK > 1):
                    # The token waves each hold a max over their own tok_tiles
                    # and LDS is the only place waves meet. Every lane writes,
                    # so the write is one contiguous 256 B run per (wave, tile)
                    # and the read is lane-local.
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
                # Whether page p really exists. Gates the store, not the work:
                # ids are clamped in the prologue, so an out-of-range page
                # computes a value nobody keeps. At one page per wave the
                # branch above proved it and the mask would be dead.
                valid = (
                    None
                    if const_expr(pages_per_wave == 1 and WAVES_PER_TOK == 1)
                    else p < limit
                )
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    # Scale folded in here rather than per accumulator: it is
                    # positive so max(s*x) == s*max(x), and -inf * s is -inf,
                    # so a fully masked page still stores -inf.
                    m = folded[i] * f32_scale
                    tok, head = tok_head_of(i)
                    row = b * fx.Int32(S) + tok
                    addr = (
                        head * i32_stride_s_h
                        + row * i32_stride_s_b
                        + p * i32_stride_s_k
                    )
                    # One writer per feature; padding columns and pages past
                    # this request write nothing.
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

    return launch, CHUNK


_CACHE = {}


def _get(S, H, geom, device):
    key = (S, H, tuple(sorted(geom.items())), device)
    if key not in _CACHE:
        _CACHE[key] = build_index_score(S, H, geom)
    return _CACHE[key]


@triton.jit
def _plan_spread_map(
    lengths,
    out,
    batch,
    rows,
    cu,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Spread a ragged batch over CU rounds, so no CU draws a deep run.

    What costs is pages piling onto one CU, not the CTA count: an extra CTA
    holding one page is nearly free, holding four is not. So the batch is cut
    into runs of at most four pages, as few rows as will still spread the deep
    requests, and the rest apportioned by integer prefixes -- which is what
    keeps rounding from overfilling the row budget.

    One program per row tile, vectorised over the batch as a [BLOCK_N, BLOCK_B]
    tile. `_spread_fits` bounds the batch at SPREAD_MAX_BATCH, so one tile
    always covers it.
    """
    b = tl.arange(0, BLOCK_B)
    in_batch = b < batch
    length = tl.load(lengths + b, in_batch, other=0)
    pages = (tl.maximum(length, 0) + 127) // 128
    if WORLD > 1:  # block p of this rank is global block p*WORLD + RANK
        pages = (tl.maximum(pages - RANK, 0) + WORLD - 1) // WORLD
    pages = tl.where(in_batch, pages, 0)

    total = tl.sum(pages, 0)
    live = tl.sum((pages > 0).to(tl.int32), 0)
    fits = pages // 4
    rounds = tl.maximum((total + 4 * live + 4 * cu - 1) // (4 * cu), 1)
    full = tl.minimum(tl.sum(fits, 0), (rounds - 1) * cu)
    rest_total = total - 4 * full
    thin = tl.minimum(rounds * cu - full - live, rest_total)
    thin = tl.maximum(tl.minimum(thin, rows - full - live), 1)
    denom = tl.maximum(rest_total, 1)

    # Deep requests give up whole 4-page runs first, greedily in request order,
    # so the exclusive prefix of the runs taken is just min(full, fits_before).
    fits_before = tl.cumsum(fits, 0) - fits
    a = tl.minimum(full, fits_before + fits) - tl.minimum(full, fits_before)
    a_before = tl.minimum(full, fits_before)
    rest = pages - 4 * a
    # rest * thin can pass 2**31 (65535 pages x ~1M rows): widen.
    k = ((rest.to(tl.int64) * thin + denom - 1) // denom).to(tl.int32)
    k_before = tl.cumsum(k, 0) - k

    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    nt = n - full
    in_full = (n[:, None] >= a_before[None, :]) & (n[:, None] < (a_before + a)[None, :])
    in_thin = (nt[:, None] >= k_before[None, :]) & (
        nt[:, None] < (k_before + k)[None, :]
    )
    take = in_full | in_thin

    it = nt[:, None] - k_before[None, :]
    kk = tl.maximum(k, 1)[None, :]
    base = (rest // kk.reshape(BLOCK_B))[None, :]
    extra = rest[None, :] - base * kk
    first_t = 4 * a[None, :] + it * base + tl.minimum(it, extra)
    span_t = base + (it < extra).to(tl.int32)
    first = tl.where(in_full, 4 * (n[:, None] - a_before[None, :]), first_t)
    span = tl.where(in_full, 4, span_t)

    hit = tl.sum(take.to(tl.int32), 1)
    hit_b = tl.sum(tl.where(take, b[None, :], 0), 1)
    hit_first = tl.sum(tl.where(take, first, 0), 1)
    hit_span = tl.sum(tl.where(take, span, 0), 1)
    hit_len = tl.sum(tl.where(take, length[None, :], 0), 1)

    # A hole is (0, 0), which the scorer's tail guard retires with no branch.
    row = n < rows
    alive = (hit > 0) & row
    tl.store(out + n * 2, tl.where(alive, (hit_b << 16) | hit_first, 0), row)
    tl.store(out + n * 2 + 1, tl.where(alive, hit_len | (hit_span << 24), 0), row)


def _run_spread_map(seq_lens, rows, cu, out, world, rank):
    import torch

    batch = seq_lens.shape[0]
    with torch.cuda.device(seq_lens.device):
        _plan_spread_map[(max(1, -(-rows // 256)),)](
            seq_lens,
            out,
            batch,
            rows,
            cu,
            WORLD=world,
            RANK=rank,
            BLOCK_B=triton.next_power_of_2(max(batch, 1)),
            BLOCK_N=256,
        )
    return out


@triton.jit
def _plan_chunk_map(
    lengths,
    out,
    batch,
    rows,
    CHUNK: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dispatch order -> work item, packed so the holes land at the end.

    One program per (request, row tile). The exclusive prefix of the request's
    rows is summed with a runtime loop over the batch rather than an unrolled
    one, so there is no batch bound and no second implementation for large
    batches -- which is what the FlyDSL builder this replaces needed.
    """
    b = tl.program_id(0)
    first = 0
    for base in range(0, b, BLOCK_B):
        idx = base + tl.arange(0, BLOCK_B)
        length = tl.load(lengths + idx, idx < b, other=0)
        pages = (tl.maximum(length, 0) + 127) // 128
        if WORLD > 1:  # block p of this rank is global block p*WORLD + RANK
            pages = (tl.maximum(pages - RANK, 0) + WORLD - 1) // WORLD
        first += tl.sum((pages + CHUNK - 1) // CHUNK, 0)

    length = tl.load(lengths + b)
    pages = (tl.maximum(length, 0) + 127) // 128
    if WORLD > 1:
        pages = (tl.maximum(pages - RANK, 0) + WORLD - 1) // WORLD
    count = (pages + CHUNK - 1) // CHUNK

    c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n = first + c
    live = (c < count) & (n < rows)
    # Row n is [0] = (b << 16) | c and [1] = the GLOBAL length, one cache line;
    # the kernel re-derives its local page count from the length.
    tl.store(out + n * 2, (b << 16) | c, live)
    tl.store(out + n * 2 + 1, length, live)


@triton.jit
def _zero_rows(out, rows, BLOCK_N: tl.constexpr):
    """A hole reads as (0, 0), which the scorer's tail guard retires with no
    extra branch. Written first so the plan above can leave them alone."""
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(out + n * 2, 0, n < rows)
    tl.store(out + n * 2 + 1, 0, n < rows)


def _run_chunk_map(seq_lens, rows, chunks, chunk, out, world, rank):
    """Launch the plan above over (request, row tile)."""
    import torch

    batch = seq_lens.shape[0]
    with torch.cuda.device(seq_lens.device):
        _zero_rows[(max(1, -(-rows // 1024)),)](out, rows, BLOCK_N=1024)
        if batch:
            _plan_chunk_map[(batch, max(1, -(-chunks // 256)))](
                seq_lens,
                out,
                batch,
                rows,
                CHUNK=chunk,
                WORLD=world,
                RANK=rank,
                BLOCK_B=256,
                BLOCK_N=256,
            )
    return out


def work_map_size(
    batch: int, max_block: int, S: int, H: int, *, device=None, **cp
) -> int:
    """Rows the work map will have: the grid, one row per CTA.

    A cudagraph caller sizes its persistent buffer with `work_map_capacity`
    instead -- the exact grid is non-monotonic in the bounds -- and passes a
    `buf[:rows]` slice. S/H must match the `score_flydsl` call the map is for:
    they change the geometry, and a map built for a different one is a map the
    kernel would index with the wrong stride.
    """
    return _validate_bounds(
        batch, max_block, geometry(batch, max_block, S, H, device=device, **cp)
    )


def work_map_capacity(max_batch: int, max_block: int, S: int, H: int, **cp):
    """Persistent rows covering every batch/block bound within the envelope.

    Sizes for the smallest CTA chunk `geometry` can reach, independent of CU
    count, so one buffer stays valid on any chip and across a replay. The exact
    grid is non-monotonic in the bounds, which is why a cudagraph caller sizes
    with this and then hands `work_map` a `buf[:rows]` slice.
    """
    world = cp.get("cp_world", 1)
    base = {"nt_k": 2, "cp_world": world, "cp_rank": cp.get("cp_rank", 0)}
    # The token split is only reachable where max_block still packs at one page
    # per chunk; beyond that the split that would need these rows cannot fire.
    rows = _validate_bounds(
        max_batch,
        max_block,
        {
            **base,
            "waves_per_tok": WAVES if max_block <= _MAX_CHUNKS else 1,
            "pages_per_wave": 1,
            "spread": 0,
        },
    )
    # ...or the spread map the served path may resolve to instead, whose rows
    # grow with the batch, so only batches it can pack are considered.
    spread_batch = min(max_batch, SPREAD_MAX_BATCH)
    if _spread_fits(spread_batch, max_block, world):
        rows = max(
            rows,
            _validate_bounds(
                spread_batch,
                max_block,
                {**base, "waves_per_tok": 1, "pages_per_wave": 1, "spread": 1},
            ),
        )
    return rows


def build_work_map(seq_lens, max_block, S: int, H: int, out=None, **cp):
    """The work map with the geometry resolved for you.

    Resolving it here and in `score_flydsl` separately would mean keeping two
    answers in step, and a disagreement is not a crash -- it is a map the
    kernel indexes with the wrong stride. `out` may be a larger persistent
    buffer and is sliced down; a short one is an error, not a truncated grid.
    """
    import torch

    _validate_tensor(seq_lens, "seq_lens", (torch.int32,), 1, seq_lens.device)
    geom = geometry(seq_lens.shape[0], max_block, S, H, device=seq_lens.device, **cp)
    rows = seq_lens.shape[0] * _grid_chunks(max_block, geom)
    _validate_bounds(seq_lens.shape[0], max_block, geom)
    if out is not None:
        _validate_map(out, rows, seq_lens.device)
        out = out[:rows]
    return _make_map(seq_lens, max_block, geom, rows, out)


def _make_map(seq_lens, max_block, geom, rows, out=None):
    # The shard comes from the same geometry the kernel is compiled against, so
    # the map and the kernel cannot disagree about which blocks this rank owns.
    import torch

    world, rank = geom["cp_world"], geom["cp_rank"]
    if seq_lens.stride() != (1,):
        raise ValueError("seq_lens must be contiguous")
    if out is None:
        out = torch.empty((rows, 2), dtype=torch.int32, device=seq_lens.device)
    else:
        _validate_map(out, rows, seq_lens.device, exact=True)
    if geom["spread"]:
        # `_spread_fits`, which `geometry` consults, already bounds the batch
        # the spread builder unrolls over.
        cu = _cu_count(seq_lens.device)
        return _run_spread_map(seq_lens, rows, cu, out, world, rank)
    return _run_chunk_map(
        seq_lens,
        rows,
        _grid_chunks(max_block, geom),
        work_chunk(geom),
        out,
        world,
        rank,
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


SUPPORTED_ARCHS = ("gfx950",)


def _validate_arch(device):
    """Both scorers' arch gate.

    The device TYPE is checked before the driver query: `get_device_properties`
    initialises the driver, so on a machine with no GPU the probe raised
    `RuntimeError: No HIP GPUs are available` instead of a rejection -- which
    `index_score_supported`, documented as metadata-only, is the one entry a
    dispatcher may call anywhere.
    """
    import torch

    if device.type != "cuda":
        raise ValueError(f"index score requires CUDA tensors, got {device.type}")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"index score requires {'/'.join(SUPPORTED_ARCHS)}, got {arch}"
        )


def _validate_rows(t, name):
    """The row/head stride rule for a [rows, heads, 128] bf16 or fp8 operand.

    Only stride(0)/stride(1) reach the kernel and every innermost index is an
    element offset, so a non-unit feature stride silently scores the wrong
    features. Alignment matters too, not just overlap: the loaders turn an
    element offset into a dword offset and vector-load it, so an odd row or
    head stride drops a feature. Both scorers hold the same contract.
    """
    if (
        t.stride(2) != 1
        or t.stride(1) < HEAD_DIM
        or t.stride(0) < (t.shape[1] - 1) * t.stride(1) + HEAD_DIM
    ):
        raise ValueError(
            f"{name}: feature stride must be 1 and neither heads nor rows may "
            f"overlap, got stride {t.stride()} for shape {tuple(t.shape)}"
        )
    if any(st * t.element_size() % 16 for st in t.stride()[:2]):
        raise ValueError(
            f"{name}: rows and heads must be 16-byte aligned, got stride "
            f"{t.stride()} for a {t.element_size()}-byte element"
        )


def _validate_map(out, rows, device, exact=False):
    import torch

    _validate_tensor(out, "work_map", (torch.int32,), 2, device)
    if out.shape[1] != 2 or out.stride() != (2, 1) or out.shape[0] < rows:
        raise ValueError(f"work_map: expected packed [at least {rows}, 2] int32")
    if exact and out.shape[0] != rows:
        raise ValueError(f"work_map: expected exact grid slice [{rows}, 2]")


def _validate_bounds(batch, max_block, geom) -> int:
    """Validate the packed/address arithmetic and return the exact grid rows."""
    if any(not isinstance(v, int) for v in geom.values()):
        raise ValueError("geometry fields must be integers")
    if not isinstance(batch, int) or not 0 <= batch <= 0xFFFF:
        raise ValueError("batch must be in [0, 65535] for packed request IDs")
    if geom["waves_per_tok"] < 1 or WAVES % geom["waves_per_tok"]:
        raise ValueError("invalid wave split")
    if geom["pages_per_wave"] < 1 or not 0 <= geom["nt_k"] <= 3:
        raise ValueError("invalid page depth or cache policy")
    if geom["cp_world"] < 1 or not 0 <= geom["cp_rank"] < geom["cp_world"]:
        raise ValueError("invalid CP shard")
    chunk = work_chunk(geom)
    chunks = _grid_chunks(max_block, geom)
    if geom["spread"] and not _spread_fits(batch, max_block, geom["cp_world"]):
        raise ValueError("spread: batch, max_block or its token span cannot be packed")
    if chunks > _MAX_CHUNKS or batch * chunks * 8 > 0xFFFFFFFF:
        raise ValueError("work_map: packed chunk or address range exceeded")
    # Includes the rounded tail lanes and the causal token arithmetic.
    if ((chunks * chunk) * geom["cp_world"] + geom["cp_rank"]) * PAGE > 0x7FFFFFFF:
        raise ValueError("max_block: causal token address exceeds int32")
    return batch * chunks


def _validate_metadata(
    idx_q,
    cache,
    S,
    H,
    max_block,
    block_table=None,
    seq_lens=None,
    out=None,
    work_map=None,
    **cp,
):
    """Metadata only: no device reads, synchronisation or allocations.

    Contents are a caller contract: 0 <= seq_lens <= INT32_MAX-127, each length
    fits max_block's local CP bound and the global block table, and every
    referenced page id is in [0, cache.shape[0]). Even a zero-length request
    needs a valid clamped table entry at column zero. A supplied map must be
    rebuilt whenever lengths change.
    """
    import torch

    if not isinstance(S, int) or not isinstance(H, int) or S < 1 or H < 1:
        raise ValueError("S and H must be positive integers")
    device = idx_q.device
    _validate_arch(device)
    # Q is bf16 and the cache is fp8: this scores what
    # `fused_qknorm_idxrqknorm` writes, and the launch binds those two types.
    _validate_tensor(idx_q, "idx_q", (torch.bfloat16,), 3, device, 16)
    _validate_tensor(cache, "cache", (dtypes.fp8,), 3, device, 16, False)
    if idx_q.shape[1:] != (H, HEAD_DIM) or idx_q.shape[0] % S:
        raise ValueError("idx_q: expected [batch*S, H, 128]")
    if cache.shape[1:] != (PAGE, HEAD_DIM) or cache.shape[0] > 0x7FFFFFFF:
        raise ValueError("cache: expected [pages, 128, 128] with int32 page IDs")
    for t, name in ((idx_q, "idx_q"), (cache, "cache")):
        _validate_rows(t, name)
    batch = idx_q.shape[0] // S
    geom = geometry(batch, max_block, S, H, device=device, **cp)
    rows = _validate_bounds(batch, max_block, geom)
    if block_table is not None:
        _validate_tensor(block_table, "block_table", (torch.int32,), 2, device)
        if (
            block_table.shape[0] != batch
            or block_table.shape[1] < max_block
            or block_table.stride(1) != 1
            or block_table.stride(0) < block_table.shape[1]
        ):
            raise ValueError("block_table: invalid shape or layout")
        # Under CP the table is indexed by the GLOBAL block
        # `p * cp_world + cp_rank`, so a rank with a non-empty shard needs columns
        # well past `max_block`, its LOCAL count. That width is a function of
        # seq_lens, which this metadata-only path never reads, so it stays a caller
        # contract -- and the worst case is not usable as a check, since a rank with
        # an empty shard legitimately passes the global count
        # (`test_empty_cp_shard_narrow_table`).
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
    return batch, geom


def index_score_supported(
    idx_q,
    cache,
    S: int,
    H: int,
    max_block: int,
    block_table,
    *,
    seq_lens,
    sm_scale=1.0,
    out=None,
    work_map=None,
    **cp,
) -> bool:
    """Metadata-only predicate sharing the entry point's validation.

    It has to answer for everything `score_flydsl` would reject, or a
    dispatcher that trusts it routes here and takes a raise instead of a
    fallback. `_validate_metadata` skips `block_table`/`seq_lens` when they are
    None -- the entry point requires them -- and never sees the scale, so both
    are checked here.
    """
    if block_table is None or seq_lens is None:
        return False  # explicit None, which the entry point also refuses
    try:
        _validate_metadata(
            idx_q, cache, S, H, max_block, block_table, seq_lens, out, work_map, **cp
        )
        scaled = sm_scale * LOG2E
        return math.isfinite(scaled) and 2**-149 <= scaled <= (2 - 2**-23) * 2**127
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return False


# Feature count from which the transposed score layout starts paying.
TRANSPOSE_MIN_F = 16


def alloc_score(batch, S, H, max_block, device):
    """Allocate the scores in whichever layout the kernel writes fastest.

    [H, batch*S, max_block] fp32 either way; only the strides differ, which
    every consumer takes as arguments. The epilogue stores one fp32 per feature
    per page, so a contiguous page axis puts those F = S*H lanes on F different
    cache lines, where a contiguous FEATURE axis packs them into one 128 B line.
    The consumer then pays a strided page axis and loses eligibility for the
    aiter selector (it needs `score.view(rows, max_block)`), which at these
    shapes already declines on its LDS budget -- but a caller with a small
    max_block should check rather than assume. Below F = 16 the store is at
    most 16 B, so those shapes stay contiguous.

    Filled with -inf rather than left as `empty`: a request shorter than
    `max_block` never stores to its tail pages, so on a ragged batch a third of
    the buffer can stay whatever was in it. A caller passing its own `out` owes
    it the same sentinel.
    """
    import torch

    total_q = batch * S
    if S * H >= TRANSPOSE_MIN_F:
        # [max_block, total_q, H] viewed as [H, total_q, max_block]: strides
        # (1, H, total_q*H), i.e. feature-contiguous.
        return torch.full(
            (max_block, total_q, H), NEG_INF, dtype=torch.float32, device=device
        ).permute(2, 1, 0)
    return torch.full(
        (H, total_q, max_block), NEG_INF, dtype=torch.float32, device=device
    )


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
    work_map=None,
    cp_world: int = 1,
    cp_rank: int = 0,
):
    """Host entry point.

    `cp_world`/`cp_rank` shard the blocks round robin: this rank owns
    `block % cp_world == cp_rank`, so its logical block p is global block
    `p * cp_world + cp_rank`. The global id addresses the block table and
    positions the causal mask; the score is stored at the local p, giving the
    compacted [H, tokens, ceil(blocks/world)] shard the CP selector reads.
    """
    import torch

    batch, geom = _validate_metadata(
        idx_q,
        cache,
        S,
        H,
        max_block,
        block_table,
        seq_lens,
        out,
        work_map,
        cp_world=cp_world,
        cp_rank=cp_rank,
    )
    if block_table is None or seq_lens is None:
        raise ValueError("block_table and seq_lens are required")
    scaled = sm_scale * LOG2E
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        raise ValueError("sm_scale * LOG2E must be finite and positive in FP32")
    if out is None:
        out = alloc_score(batch, S, H, max_block, idx_q.device)
    launch, _chunk = _get(S, H, geom, idx_q.device.index)
    # The grid depends only on launch-time bounds, never on seq_lens contents,
    # so it stays valid across a cudagraph replay with different lengths.
    chunks = _grid_chunks(max_block, geom)
    # Per *step*, not per layer -- a serving caller should hoist this out and
    # pass it in, since every layer of a step sees the same lengths.
    if work_map is None:
        work_map = _make_map(seq_lens, max_block, geom, batch * chunks)
    with torch.cuda.device(idx_q.device):
        _run_compiled(
            launch,
            ptr_arg(idx_q, fx.BFloat16),
            ptr_arg(cache, fx.Float8E4M3FN),
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
