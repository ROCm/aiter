# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Delayed mHC seam with RMSNorm, fused tile form -- the fused pass and the finish pass.

For `streams` residual streams R, the previous seam's gates post, comb and pre, and this seam's
weight fn:

    R'_j   = bf16(post_j * y + sum_i comb[i][j] * R_i)
    logits = (fn @ R') * rsqrt(mean(R'^2) + rms_eps) * hc_scale + hc_base
    x1     = bf16(sum_j pre_j * R'_j)
    out    = bf16(x1 * rsqrt(mean(x1^2) + norm_eps) * norm_weight)

and the gates it hands the next seam, off the same logits:

    pre'_j  = sigmoid(logits_pre_j) + hc_pre_eps
    post'_j = sigmoid(logits_post_j) * hc_post_mult_value
    comb'   = sinkhorn(softmax_row(logits_comb), hc_sinkhorn_eps, sinkhorn_repeat)

The gate projection runs inside the pass that produces `R'`, against the value still in
registers. Reading `residual_out` back for it instead would cost a second `tokens * streams *
hidden` of traffic.

Neither row mean closes inside a workgroup once the hidden axis is split, so the fused pass
publishes per (segment, token) one unscaled logit per gate row and the two sums of squares, and a
finish pass folds them, forms the gates and rescales the staged collapse.

Inputs/outputs: the streamed tensors are flat and addressed through the layout named for them.
`stage` carries the bf16-rounded un-normalised collapse from the fused pass to the finish pass;
it is the value the contract squares and rescales, so the round trip through it is exact.

Paths:
  - `has_post` off: the residual passes through unmixed, so the post-mix coefficients are never
    read and `residual_out` is the caller's `residual`, left unwritten.
  - `identity_pre` on: the pre band is the identity on stream 0, so `pre_mix` is never read and
    `x1` is `R'_0`.

Gate packing, which a caller must match exactly and cannot derive from the signatures:
  - `fn` is `(gate_rows(streams), streams * hidden)` fp32, its rows ordered `pre | post | comb`
    -- `streams` pre rows, then `streams` post rows, then the `streams * streams` comb rows in
    row-major `(i, j)` order.
  - `hc_scale` is three fp32, `(pre, post, comb)`, applied to the matching band of rows.
  - `hc_base` is one fp32 per projection row, in the same order as `fn`'s rows.

Constraints:
  - `streams` is `STREAMS`: the gate chain closes the comb with fixed DPP controls, which is a
    4x4 matrix on one 16-lane row and no other shape. `check_config` refuses anything else.
  - A 64-lane wave. A token owns `SPLIT_SLOTS` consecutive lanes and the comb sits in the lower
    16-lane DPP row of that group; whether the map survives a 32-lane wave is untested.
  - The fused launch's `n_blk` is `ceil(tokens / block_m)` for the `block_m` it was built with;
    the finish launch's is `finish_blocks(tokens)`. Every scratch element the finish pass reads
    is then written by the fused pass on the same call, which lets all three buffers go unzeroed.
  - The fused pass's tile geometry: see `check_config`.
"""

import hashlib
from functools import cache
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.act import sigmoid_f32
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.kernels_common import LOG2E, ceildiv, get_warp_size

WAVE_SIZE = get_warp_size()
BLOCK = 4 * WAVE_SIZE
# bf16 columns one post-mix thread owns per stream: 8 is a 16 B access, the widest issued.
VEC = 8
MFMA_MN = 16
MFMA_K = 16
# The gate rows padded to two 16-wide MFMA N tiles. The atom asserts M == N, so 24 is not a
# tile width; the eight padding rows read past `fn` and the buffer descriptor returns zeros.
PAD_ROWS = 2 * MFMA_MN
# Every wave covers the whole output tile and they split the contraction instead: splitting an
# output axis would cost operand copies a tile this small cannot amortise. The price is a fold of
# the partials, once per segment through LDS.
WAVES_K = BLOCK // WAVE_SIZE
# The token tile is a whole number of MFMA M tiles, and the chunk a whole number of the k extent
# the split gives the tiled MMA.
BLOCK_M_GRAIN = MFMA_MN
CHUNK_GRAIN = WAVES_K * MFMA_K
# bf16 elements of padding on each row of the staged tile. The MFMA A fragment reads lane `l` at
# row `l % MFMA_MN`, so sixteen lanes read sixteen rows at one column: with an unpadded stride
# that is a multiple of the bank period they all land on one bank.
TILE_PAD = 8
# The LDS a workgroup may hold and still leave two of them resident on a 64 KB part, which is
# what decides how wide the accumulator fold may be.
FOLD_BUDGET = 32 * 1024
# fp32 per (segment, token) of the published logit row: one per gate row, then padding to the
# 128 B line the finish pass gathers.
SPLIT_SLOTS = PAD_ROWS
# Tokens one finish workgroup owns: exactly enough that its published lines give every lane of
# the block one value to fold.
TOKENS_PER_BLOCK = BLOCK // SPLIT_SLOTS
# DPP controls. The comb's sixteen entries sit in one 16-lane DPP row of the fold's lane map, so
# a row of the matrix is four consecutive lanes and a column is a stride-4 class inside that row;
# neither move leaves the row, so a group's reduction never reads another token's lanes.
_QUAD_PERM_XOR1 = 0xB1  # quad_perm:[1,0,3,2] -- partner within a pair
_QUAD_PERM_XOR2 = (
    0x4E  # quad_perm:[2,3,0,1] -- partner pair; the two together sum a group of 4
)
_ROW_ROR4 = 0x124
_ROW_ROR8 = (
    0x128  # ror 8 then ror 4 sums the four lanes of a stride-4 class in a 16-lane row
)
# The CU count a caller passes when it does not know the part's. Only ever read as how many
# rounds of the part a grid is deep, so a wrong value is a geometry choice and not a wrong answer.
DEFAULT_CU_COUNT = 304
# Rounds of the part below which the finish pass's gate chain is still on the critical path, and
# so worth a column step of its own rather than a seat in a block that also carries traffic.
GATE_STEP_ROUNDS = 8


# Residual streams the gate lane map is specialised for.
STREAMS = 4
# A floor, not the part's: the smallest arch this runs on, so taking it never admits a geometry
# that does not fit. gfx950 holds 160 KB and a caller there should pass its own.
DEFAULT_LDS_BYTES = 64 * 1024
# Every index the kernels compute is int32, which caps the elements they can address. The same cap
# keeps the split-K partials, stored through a buffer descriptor, inside its 32-bit byte offset.
MAX_FLAT_ELEMENTS = 2**31 - 1


def _require(cond: bool, msg: str, exc: type[Exception] = ValueError) -> None:
    """Build-time legality check that survives `python -O`, which strips `assert`."""
    if not cond:
        raise exc(msg)


def gate_rows(streams: int) -> int:
    """Rows of the `fn` gate projection: `streams` pre + `streams` post + `streams^2` comb."""
    return 2 * streams + streams * streams


def _rcp(value: Any) -> Any:
    """`v_rcp_f32`: ~1 ulp, against the ~10-instruction IEEE expansion a `/` lowers to."""
    return fx.Float32(rocdl.rcp(T.f32, fx.Float32(value)))


def _exp2(value: Any) -> Any:
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(value).ir_value()))


def _dpp(value: Any, ctrl: int) -> Any:
    """Move `value` across lanes by a DPP control, with `old = 0` and `bound_ctrl` set so the
    compiler can fold the move into the consuming VALU op. Every control used here stays inside a
    16-lane row, and the fold gives a token a whole 32-lane group, so the bound value is never
    observed and a dead token's group never contributes to a live one's."""
    src_bits = fx.Float32(value).bitcast(fx.Int32)
    moved = update_dpp_i32(0, src_bits, ctrl, 0xF, 0xF, True)
    return fx.Int32(moved).bitcast(fx.Float32)


def _row_sum(value: Any) -> Any:
    partial = value + _dpp(value, _QUAD_PERM_XOR1)
    return partial + _dpp(partial, _QUAD_PERM_XOR2)


def _col_sum(value: Any) -> Any:
    partial = value + _dpp(value, _ROW_ROR8)
    return partial + _dpp(partial, _ROW_ROR4)


def _row_max(value: Any) -> Any:
    first = _dpp(value, _QUAD_PERM_XOR1)
    partial = (value > first).select(value, first)
    second = _dpp(partial, _QUAD_PERM_XOR2)
    return (partial > second).select(partial, second)


def _offset(layout: Any, *coord: Any) -> Any:
    """The flat element index `layout` gives `coord`.

    Apply it to the part of a coordinate that is uniform across the block and add the divergent
    part afterwards, so the uniform arithmetic stays in scalar registers.
    """
    return fx.Int32(fx.get_scalar(fx.crd2idx(fx.make_coord(*coord), layout)))


def _narrow_round(value: Any, fast: bool) -> tuple[Any, Any]:
    """Round one fp32 to bf16 and widen it back, as the pair this kernel needs of every element.

    gfx942 has no f32->bf16 convert instruction, so `fptrunc` expands inline, and `fast_fp_math`
    then folds `fpext(fptrunc x)` back to `x` -- which deletes the rounding the contract puts
    between the post-mix and the collapse, and between the collapse and the RMSNorm's sum of
    squares. `fast` applies the round-to-nearest-even bias in integer ops instead: bit-identical
    to `fptrunc` on every finite input, invisible to the fold, and free of the NaN-quieting
    compare and select the expansion carries. Widening is then a mask of the same biased word.
    """
    if not fast:
        narrow = fx.BFloat16(value)
        return narrow, fx.Float32(narrow)
    bits = fx.Float32(value).bitcast(fx.Uint32)
    biased = bits + (bits >> fx.Int32(16) & fx.Uint32(1)) + fx.Uint32(0x7FFF)
    return fx.Uint16(biased >> fx.Int32(16)).bitcast(fx.BFloat16), (
        biased & fx.Uint32(0xFFFF0000)
    ).bitcast(fx.Float32)


def _load_bf16(ptr: Any, offset: Any, count: int, nt: bool) -> Any:
    return fx.generic_load(
        fx.add_offset(ptr, offset),
        dtype=fx.BFloat16,
        count=count,
        nontemporal=nt,
    )


def _store_bf16(ptr: Any, offset: Any, values: list[Any], nt: bool) -> None:
    fx.generic_store(
        fx.add_offset(ptr, offset),
        fx.Vector.from_elements(values, fx.BFloat16),
        nontemporal=nt,
    )


def _load_f32(ptr: Any, offset: Any) -> Any:
    return fx.generic_load(fx.add_offset(ptr, offset), dtype=fx.Float32)


def _store_f32(ptr: Any, offset: Any, value: Any) -> None:
    fx.generic_store(fx.add_offset(ptr, offset), fx.Float32(value))


def _rows_view(ptr: Any, rows: int, cols: int, live_rows: int, elem_bytes: int) -> Any:
    """Re-view an fp32 pointer argument as a row-major `(rows, cols)` buffer tensor.

    `rows` may overhang the argument's real row count: the descriptor's record limit is the live
    extent, so the hardware returns zeros past it. That is the whole story of the second N tile,
    with no mask in the loop and no padded copy of the weight on the host.
    """
    return fx.Tensor(
        fx.make_view(
            fx.rocdl.make_buffer_ptr(
                ptr, num_records_bytes=live_rows * cols * elem_bytes
            ),
            fx.make_layout((rows, cols), (cols, 1)),
        )
    )


def check_config(
    tokens: int,
    hidden: int,
    streams: int,
    block_m: int,
    chunk_c: int,
    splits: int,
    *,
    lds_bytes: int = DEFAULT_LDS_BYTES,
    fast_fp_math: bool = True,
    fast_narrow: bool = True,
) -> None:
    """Reject a geometry this module's own code does not admit, raising `ValueError`.

    Besides `streams == STREAMS` and the int32 flat-index cap, the fused pass's tile geometry:
      - `VEC` divides `chunk_c` and `chunk_c` divides `hidden`; `splits` divides the chunk count.
      - `chunk_c` is a multiple of `CHUNK_GRAIN`, the k extent the wave split gives the tiled MMA.
      - `chunk_c / VEC`, the lanes sharing a token, is a power of two within a wave: both sums
        close in a shuffle butterfly over them.
      - `block_m * chunk_c / VEC` is a multiple of `BLOCK`, so the post-mix slots divide evenly
        over the block and every slot is a live token column.
      - `block_m` is a multiple of the `MFMA_MN`-token MFMA M tile, and no larger than `BLOCK`.
      - The staged A tile and the accumulator fold fit `lds_bytes`; `smem_bytes` is their total,
        and `fold_slots` is what it trades against.
      - `fast_fp_math` needs `fast_narrow`: the blanket flag folds `fpext(fptrunc x)` away, and
        that round trip is the contract.
    """
    _require(
        streams == STREAMS,
        f"the gate lane map is specialised for {STREAMS} residual streams, got {streams}",
    )
    _require(
        fast_narrow or not fast_fp_math,
        "fast_fp_math folds the bf16 narrow-widen round trip the contract requires; it needs fast_narrow",
    )
    rows = gate_rows(streams)
    _require(
        rows <= PAD_ROWS and rows + 2 <= SPLIT_SLOTS,
        f"the {rows} gate rows and the two sums of squares must fit the {SPLIT_SLOTS}-wide published line",
    )
    _require(
        block_m > 0 and block_m % BLOCK_M_GRAIN == 0 and block_m <= BLOCK,
        f"block_m must be a positive multiple of {BLOCK_M_GRAIN}, the MFMA M tile, and at most {BLOCK}, got {block_m}",
    )
    _require(
        chunk_c > 0 and hidden % chunk_c == 0,
        f"chunk_c {chunk_c} must be positive and divide hidden {hidden}",
    )
    _require(chunk_c % VEC == 0, f"the {VEC}-wide access must divide chunk_c {chunk_c}")
    _require(
        chunk_c % CHUNK_GRAIN == 0,
        f"the waves split the contraction, so chunk_c must be a multiple of {CHUNK_GRAIN}, got {chunk_c}",
    )
    groups = chunk_c // VEC
    _require(
        groups & (groups - 1) == 0 and groups <= WAVE_SIZE,
        f"the {groups} lanes sharing a token (chunk_c / {VEC}) must be a power of two within a wave",
    )
    _require(
        (block_m * chunk_c // VEC) % BLOCK == 0,
        f"block_m * chunk_c / {VEC} must be a multiple of {BLOCK} so the post-mix slots divide over the block",
    )
    _require(
        splits >= 1 and (hidden // chunk_c) % splits == 0,
        f"splits {splits} must divide the {hidden // chunk_c} chunks of the hidden axis",
    )
    smem = smem_bytes(block_m, chunk_c)
    _require(
        smem <= lds_bytes,
        f"the {smem} B staging tile and reduction scratch exceed the {lds_bytes} B LDS budget",
    )
    _require(
        tokens * streams * hidden <= MAX_FLAT_ELEMENTS,
        f"flat int32 indexing caps tokens * streams * hidden at {MAX_FLAT_ELEMENTS}",
    )


def _tile_bytes(block_m: int, chunk_c: int) -> int:
    """The staged A tile, which is all the fold width does not change."""
    return block_m * (chunk_c + TILE_PAD) * 2


def fold_slots(block_m: int, chunk_c: int) -> int:
    """Slots the cross-wave accumulator fold holds at once: one per wave where the scratch fits
    `FOLD_BUDGET`, otherwise two rounds of two with the first round's sum carried in registers.
    """
    return (
        WAVES_K
        if _tile_bytes(block_m, chunk_c) + WAVES_K * block_m * PAD_ROWS * 4
        <= FOLD_BUDGET
        else 2
    )


def gate_coeffs(streams: int, has_post: bool) -> int:
    """Per-token gate coefficients the post-mix stages: `post`, `comb` and `pre` where the post
    block runs, `pre` alone where it does not. `pre` is the last `streams` either way, so the
    read index does not fork on `has_post`.
    """
    return gate_rows(streams) if has_post else streams


def smem_bytes(block_m: int, chunk_c: int) -> int:
    """The bf16 A tile of one stream's chunk plus the scratch the waves fold their partial
    accumulators through, which the staged gate coefficients share, so they add nothing.
    """
    return (
        _tile_bytes(block_m, chunk_c)
        + fold_slots(block_m, chunk_c) * block_m * PAD_ROWS * 4
    )


def finish_blocks(tokens: int) -> int:
    """Workgroups the finish pass wants: one per group of `TOKENS_PER_BLOCK` tokens."""
    return ceildiv(max(tokens, 1), TOKENS_PER_BLOCK)


def finish_col_steps(hidden: int) -> int:
    """Column steps the finish pass sweeps the hidden axis in, one `VEC` group per lane."""
    return ceildiv(hidden // VEC, BLOCK)


def gate_step(tokens: int, hidden: int, cu_count: int = DEFAULT_CU_COUNT) -> bool:
    """Whether the finish pass reserves a leading column step for the gates alone.

    The Sinkhorn is a dependent VALU chain that `TOKENS_PER_BLOCK` lanes carry while the rest of
    the block idles, so a rescale sharing the block waits it out; a step of its own removes that
    wait and costs one extra block per token group, which only pays while the grid is shallow
    enough for one block's latency to matter.

    `cu_count` is the part's, which the caller knows and this module does not; the default is a
    full gfx942 and wrong on a partitioned or masked part.
    """
    return (
        finish_blocks(tokens) * finish_col_steps(hidden) <= GATE_STEP_ROUNDS * cu_count
    )


def _consts_tag(*values: float) -> str:
    """Short digest of the constants the finish pass compiles in, so variants that differ only
    in them still get distinct kernel names."""
    return hashlib.sha1(repr(tuple(float(v) for v in values)).encode()).hexdigest()[:8]


def part_elements(tokens: int, splits: int, block_m: int) -> int:
    """Elements of the published logit lines.

    Indexed by token tile rather than by token, so a workgroup's slab is a whole number of lines
    and the accumulator stores into it with no tile index and no bound; the lines of a tile's dead
    tokens are written and never read.
    """
    return ceildiv(max(tokens, 1), block_m) * splits * block_m * SPLIT_SLOTS


def sums_elements(tokens: int, splits: int) -> int:
    """Elements of the published sums of squares: `sum(R'^2)` and `sum(x1^2)` per (segment, token)."""
    return splits * max(tokens, 1) * 2


@cache
def compile_mhc_seam_fused(
    *,
    streams: int,
    hidden: int,
    has_post: bool,
    identity_pre: bool,
    block_m: int,
    chunk_c: int,
    splits: int,
    fast_fp_math: bool,
    fast_narrow: bool,
    load_nt: bool,
    store_nt: bool,
    stage_nt: bool,
    waves_per_eu: int,
) -> Any:
    """The fused pass: the post-mix, the collapse, the gate projection, and both sums of squares.

    A workgroup owns `block_m` tokens by one hidden segment and sweeps the segment in
    `chunk_c`-wide chunks. The streams of a chunk are post-mixed together -- every output stream
    reads all input streams -- and then published one at a time into an LDS tile that the matrix
    core reads as the MFMA A operand, which is the only way the post-mix's thread map (a thread
    owns `VEC` contiguous columns of one token) reaches the fragment's (a lane holds one token's
    one k element).

    `fn` needs no such staging: in B-fragment order a lane's elements are contiguous in k, so it
    is read straight from global, once, in the fp32 the caller supplies, and the two bf16 limbs
    the matrix core takes are formed from it in registers. It arrives as the
    `(gate_rows(streams), streams * hidden)` weight, re-viewed `PAD_ROWS` tall so the matrix
    core's second N tile reads its padding from past the weight's record limit and gets zeros.
    """
    # Fixed DPP controls make the gate chain a 4x4 comb and no other shape; every other value
    # compiles and returns a wrong `comb_out`, so it is refused rather than left to a host.
    _require(
        streams == STREAMS,
        f"the gate lane map is specialised for {STREAMS} residual streams, got {streams}",
    )
    _require(
        fast_narrow or not fast_fp_math,
        "fast_fp_math folds the bf16 narrow-widen round trip the contract requires; it needs fast_narrow",
    )
    rows = gate_rows(streams)
    wide = streams * hidden
    chunks = hidden // chunk_c
    seg_chunks = chunks // splits
    groups = chunk_c // VEC
    slots = block_m * groups // BLOCK
    # Weight elements one lane holds of the B fragment: the k tiles of a chunk, by the N tiles, by
    # what one lane takes of an atom.
    b_frag = (
        (chunk_c // (WAVES_K * MFMA_K))
        * (PAD_ROWS // MFMA_MN)
        * (MFMA_MN * MFMA_K // WAVE_SIZE)
    )
    folds = fold_slots(block_m, chunk_c)
    fold_rounds = WAVES_K // folds
    # Butterfly steps that close one token's sums across the `groups` lanes sharing it. They are
    # consecutive and aligned, so an xor of a mask below `groups` never leaves the token.
    butterfly = groups.bit_length() - 1
    _require(
        groups == 1 << butterfly and groups <= WAVE_SIZE,
        f"the {groups} lanes sharing a token must be a power of two within a wave",
    )
    coeffs = gate_coeffs(streams, has_post)
    _require(
        coeffs <= folds * PAD_ROWS,
        f"the {coeffs} staged coefficients must fit a token's {folds * PAD_ROWS} words of accumulator scratch",
    )
    # Where `pre` starts in a staged row; `post` is at 0 and `comb` follows it.
    pre_base = coeffs - streams

    @fx.struct
    class SharedStorage:
        a: fx.Array[fx.BFloat16, block_m * (chunk_c + TILE_PAD)]
        acc: fx.Array[fx.Float32, folds * block_m * PAD_ROWS]

    mode = ("post" if has_post else "nopost") + ("_idpre" if identity_pre else "")
    name = (
        f"mhc_seam_fused_n{streams}_h{hidden}_bm{block_m}_c{chunk_c}_s{splits}"
        f"_{mode}_w{waves_per_eu}"
        + ("_ffm" if fast_fp_math else "")
        + ("_fnr" if fast_narrow else "")
        + ("_ntl" if load_nt else "")
        + ("_nts" if store_nt else "")
        + ("_ntg" if stage_nt else "")
    )

    @flyc.kernel(name=name, known_block_size=[BLOCK, 1, 1])
    def mhc_seam_fused_kernel(
        residual: fx.Pointer,  # (T, streams, H) bf16
        sublayer_out: fx.Pointer,  # (T, H) bf16                   [has_post]
        post_layer_mix: fx.Pointer,  # (T, streams) fp32           [has_post]
        comb_res_mix: fx.Pointer,  # (T, streams, streams) fp32    [has_post]
        pre_mix: fx.Pointer,  # (T, streams) fp32          [not identity_pre]
        fn: fx.Pointer,  # (gate_rows, streams * H) fp32
        residual_out: fx.Pointer,  # (T, streams, H) bf16
        stage: fx.Pointer,  # (T, H) bf16
        part: fx.Pointer,  # (T/block_m, splits, block_m, SPLIT_SLOTS) fp32
        sums: fx.Pointer,  # (splits, T, 2) fp32
        n_tok: fx.Int32,
    ) -> None:
        tid = fx.thread_idx.x
        wave = tid // fx.Int32(WAVE_SIZE)
        # The hidden segment is the grid's fast axis: workgroups are dispatched round-robin over
        # the part's XCDs, so a die then holds one segment's slice of `fn` rather than every
        # segment's.
        seg = fx.block_idx.x
        tile = fx.block_idx.y
        token0 = tile * block_m

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_a_flat = lds.a.view(fx.make_layout(block_m * (chunk_c + TILE_PAD), 1))
        s_a = lds.a.view(fx.make_layout((block_m, chunk_c), (chunk_c + TILE_PAD, 1)))
        s_acc = lds.acc.view(
            fx.make_layout(
                (folds, block_m, PAD_ROWS), (block_m * PAD_ROWS, PAD_ROWS, 1)
            )
        )
        s_acc_flat = lds.acc.view(fx.make_layout(folds * block_m * PAD_ROWS, 1))
        # The staged gates sit on the accumulator scratch rather than scratch of their own: they
        # are dead by the end of the chunk loop, which is the first thing the fold's barrier
        # separates them from, so the two never want the words at once.
        s_gates = lds.acc.view(fx.make_layout((block_m, coeffs), (coeffs, 1)))

        res_major = fx.make_layout(
            (n_tok, streams, hidden), (streams * hidden, hidden, 1)
        )  # residual, residual_out
        token_major = fx.make_layout(
            (n_tok, hidden), (hidden, 1)
        )  # sublayer_out, stage
        per_stream = fx.make_layout(
            (n_tok, streams), (streams, 1)
        )  # post_layer_mix, pre_mix
        per_pair = fx.make_layout(
            (n_tok, streams, streams), (streams * streams, streams, 1)
        )  # comb_res_mix
        sums_major = fx.make_layout((splits, n_tok, 2), (n_tok * 2, 2, 1))

        # ── Matrix core ──
        g_fn = _rows_view(fn, PAD_ROWS, wide, rows, 4)
        mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA_MN, MFMA_MN, MFMA_K, fx.BFloat16)
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((1, 1, WAVES_K), (1, 1, 1))
        )
        thr_mma = tiled_mma.thr_slice(tid)
        lds_copy = fx.make_copy_atom(
            fx.UniversalCopy(MFMA_MN * MFMA_K // WAVE_SIZE * 16), fx.BFloat16
        )
        # The fragment is bf16 and the weight fp32: the same (lane -> k) map, twice the bytes, so
        # one access carries what both limbs are made of.
        wgt_copy = fx.make_copy_atom(
            fx.rocdl.BufferCopy(MFMA_MN * MFMA_K // WAVE_SIZE * 32), fx.Float32
        )
        acc_copy = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        thr_a = fx.make_tiled_copy_A(lds_copy, tiled_mma).get_slice(tid)
        thr_b = fx.make_tiled_copy_B(wgt_copy, tiled_mma).get_slice(tid)
        thr_c = fx.make_tiled_copy_C(acc_copy, tiled_mma).get_slice(tid)
        my_acc = fx.slice(s_acc, (wave % fx.Int32(folds), None, None))

        # Indexed by tile, so this slab is a whole number of lines and the store needs no tile
        # index.
        g_part = fx.Tensor(
            fx.make_view(
                fx.add_offset(
                    fx.rocdl.make_buffer_ptr(part),
                    (tile * fx.Int32(splits) + seg) * fx.Int32(block_m * SPLIT_SLOTS),
                ),
                fx.make_layout(block_m * SPLIT_SLOTS, 1),
            )
        )

        t_fn = fx.flat_divide(g_fn, fx.make_tile(PAD_ROWS, chunk_c))[
            None, None, 0, None
        ]
        frag_a = thr_mma.make_fragment_A(s_a)
        frag_hi = thr_mma.make_fragment_B(t_fn[None, None, 0])
        frag_lo = thr_mma.make_fragment_B(t_fn[None, None, 0])
        frag_wide = fx.make_fragment_like(frag_hi, fx.Float32.ir_type)
        frag_c = thr_mma.make_fragment_C(my_acc)
        frag_c.fill(0)

        # Every partition is applied once, to the chunk-invariant prefix of its coordinate; the
        # chunk loop only selects the k tile. A ThrCopy referenced inside a dynamic loop is taken
        # for loop-carried state and cannot be reconstructed, so none may appear in the body.
        p_a = thr_a.partition_S(s_a)
        p_fn = thr_b.partition_S(t_fn)
        p_acc = thr_c.partition_S(my_acc)
        cf_a = thr_a.retile(frag_a)
        cf_wide = thr_b.retile(frag_wide)
        cf_c = thr_c.retile(frag_c)

        def stage_gates(out_gates: Any) -> None:
            """Put this tile's per-token gate coefficients in LDS, once per workgroup.

            They do not vary with the chunk, but the chunk loop stores to `residual_out` and
            `stage`, which the compiler cannot prove disjoint from the three gate tensors, so it
            will not hoist the loads itself: left in the body they are most of the vector loads a
            post-mix slot issues, re-read on every one of the segment's chunks.
            """
            for q in range_constexpr(ceildiv(block_m * coeffs, BLOCK)):
                idx = fx.Int32(q * BLOCK) + tid
                if idx < fx.Int32(block_m * coeffs):
                    token_local = idx // fx.Int32(coeffs)
                    slot_i = idx - token_local * fx.Int32(coeffs)
                    token = token0 + token_local
                    safe = (token < n_tok).select(token, fx.Int32(0))
                    # `pre_base` is 0 without the post block, so one expression serves both.
                    if const_expr(identity_pre):
                        pre_value = (slot_i == fx.Int32(pre_base)).select(
                            fx.Float32(1.0), fx.Float32(0.0)
                        )
                    if const_expr(not identity_pre):
                        # Both arms of a select are evaluated, so the index is clamped: lanes
                        # below the band would otherwise read before the tensor's base.
                        pre_value = _load_f32(
                            pre_mix,
                            _offset(per_stream, safe, 0)
                            + (slot_i >= fx.Int32(pre_base)).select(
                                slot_i - fx.Int32(pre_base), fx.Int32(0)
                            ),
                        )
                    if const_expr(has_post):
                        # `select` issues both operands' loads, so each index is clamped into
                        # its own tensor's row: the last token's row ends the allocation.
                        is_post = slot_i < fx.Int32(streams)
                        is_comb = (slot_i >= fx.Int32(streams)) & (
                            slot_i < fx.Int32(pre_base)
                        )
                        out_gates[token_local, slot_i] = is_post.select(
                            _load_f32(
                                post_layer_mix,
                                _offset(per_stream, safe, 0)
                                + is_post.select(slot_i, fx.Int32(0)),
                            ),
                            (slot_i < fx.Int32(pre_base)).select(
                                _load_f32(
                                    comb_res_mix,
                                    _offset(per_pair, safe, 0, 0)
                                    + is_comb.select(
                                        slot_i - fx.Int32(streams), fx.Int32(0)
                                    ),
                                ),
                                pre_value,
                            ),
                        )
                    if const_expr(not has_post):
                        out_gates[token_local, slot_i] = pre_value

        def post_mix_chunk(
            chunk: Any, sq: list[Any], sqc: list[Any]
        ) -> tuple[list[Any], list[Any], list[Any]]:
            """Post-mix and collapse one chunk; publish the new residual and the un-normalised
            collapse, carry both sums of squares, and hand the rounded streams on."""
            held = []
            for g in range_constexpr(slots):
                slot = fx.Int32(g * BLOCK) + tid
                token_local = slot // fx.Int32(groups)
                group = slot - token_local * fx.Int32(groups)
                token = token0 + token_local
                live = token < n_tok
                safe = live.select(token, fx.Int32(0))
                col = chunk * fx.Int32(chunk_c) + group * fx.Int32(VEC)

                res = [
                    _load_bf16(
                        residual, _offset(res_major, safe, i, 0) + col, VEC, load_nt
                    )
                    for i in range_constexpr(streams)
                ]
                pre = [
                    s_gates[token_local, fx.Int32(pre_base + j)]
                    for j in range_constexpr(streams)
                ]
                if const_expr(has_post):
                    post = [
                        s_gates[token_local, fx.Int32(j)]
                        for j in range_constexpr(streams)
                    ]
                    comb = [
                        [
                            s_gates[token_local, fx.Int32(streams + i * streams + j)]
                            for j in range_constexpr(streams)
                        ]
                        for i in range_constexpr(streams)
                    ]
                    y = [
                        fx.Float32(e)
                        for e in _load_bf16(
                            sublayer_out,
                            _offset(token_major, safe, 0) + col,
                            VEC,
                            load_nt,
                        )
                    ]
                    src = [
                        [fx.Float32(res[i][e]) for e in range_constexpr(VEC)]
                        for i in range_constexpr(streams)
                    ]

                collapse = [fx.Float32(0.0) for _ in range_constexpr(VEC)]
                squares = fx.Float32(0.0)
                mixed = []
                for j in range_constexpr(streams):
                    narrowed, wide_j = [], []
                    for e in range_constexpr(VEC):
                        if const_expr(has_post):
                            value = post[j] * y[e]
                            for i in range_constexpr(streams):
                                value = value + comb[i][j] * src[i][e]
                            rounded, widened = _narrow_round(value, fast_narrow)
                        else:
                            rounded, widened = res[j][e], fx.Float32(res[j][e])
                        narrowed.append(rounded)
                        wide_j.append(widened)
                    # No post-mix aliases `residual_out` onto `residual`, so the store would
                    # rewrite the value it just read.
                    if const_expr(has_post) and live:
                        _store_bf16(
                            residual_out,
                            _offset(res_major, safe, j, 0) + col,
                            narrowed,
                            store_nt,
                        )
                    for e in range_constexpr(VEC):
                        collapse[e] = collapse[e] + pre[j] * wide_j[e]
                        squares = squares + wide_j[e] * wide_j[e]
                    mixed.append(narrowed)
                held.append(mixed)

                # `x1` is rounded to bf16 before it is squared and before it is rescaled, so the
                # round trip through `stage` is the value the contract names.
                rounds = [
                    _narrow_round(collapse[e], fast_narrow)
                    for e in range_constexpr(VEC)
                ]
                staged = [rounds[e][0] for e in range_constexpr(VEC)]
                if live:
                    _store_bf16(
                        stage, _offset(token_major, safe, 0) + col, staged, stage_nt
                    )
                collapse_squares = fx.Float32(0.0)
                for e in range_constexpr(VEC):
                    collapse_squares = collapse_squares + rounds[e][1] * rounds[e][1]
                sq[g] = sq[g] + squares
                sqc[g] = sqc[g] + collapse_squares
            return held, sq, sqc

        def project_chunk(chunk: Any, held: list[Any], tile_lds: Any) -> None:
            """Contract the chunk's rounded streams against `fn`, one stream at a time.

            The A tile holds a single stream, which is what keeps `chunk_c` at the width the
            bandwidth phase wants inside the LDS budget.
            """
            for j in range_constexpr(streams):
                fx.gpu.barrier()
                for g in range_constexpr(slots):
                    slot = fx.Int32(g * BLOCK) + tid
                    token_local = slot // fx.Int32(groups)
                    group = slot - token_local * fx.Int32(groups)
                    live = (token0 + token_local) < n_tok
                    base = token_local * fx.Int32(
                        chunk_c + TILE_PAD
                    ) + group * fx.Int32(VEC)
                    for e in range_constexpr(VEC):
                        # A dead token's row must contract to zero: its logits are published to a
                        # line the finish pass never reads, but the line is still written.
                        tile_lds[base + e] = fx.BFloat16(
                            live.select(fx.Float32(held[g][j][e]), fx.Float32(0.0))
                        )
                fx.gpu.barrier()
                k_tile = fx.Int32(j * chunks) + chunk
                fx.copy(lds_copy, p_a, cf_a)
                fx.copy(wgt_copy, p_fn[None, None, None, k_tile], cf_wide)
                # Two limbs carry about sixteen mantissa bits, which is what the gate tolerances
                # need and a single bf16 limb misses by an order. Formed here rather than read as
                # two bf16 planes because that halves the loop's memory instructions, which is
                # what it is bound by; the bytes are the same either way. The narrowing is the
                # integer form because `fpext(fptrunc v)` is exactly the round trip
                # `fast_fp_math` folds away, and `v - hi` needs the fp32 value to survive it.
                wide = frag_wide.load()
                limbs = [_narrow_round(wide[e], True) for e in range_constexpr(b_frag)]
                frag_hi.store(
                    fx.Vector.from_elements(
                        [limbs[e][0] for e in range_constexpr(b_frag)], fx.BFloat16
                    )
                )
                frag_lo.store(
                    fx.Vector.from_elements(
                        [
                            _narrow_round(wide[e] - limbs[e][1], True)[0]
                            for e in range_constexpr(b_frag)
                        ],
                        fx.BFloat16,
                    )
                )
                fx.gemm(tiled_mma, frag_c, frag_a, frag_hi, frag_c)
                fx.gemm(tiled_mma, frag_c, frag_a, frag_lo, frag_c)

        # ── Chunk loop ──
        stage_gates(s_gates)
        fx.gpu.barrier()
        sq = [fx.Float32(0.0) for _ in range_constexpr(slots)]
        sqc = [fx.Float32(0.0) for _ in range_constexpr(slots)]
        for c in range(seg_chunks):
            chunk = seg * fx.Int32(seg_chunks) + fx.Int32(c)
            held, sq, sqc = post_mix_chunk(chunk, sq, sqc)
            project_chunk(chunk, held, s_a_flat)

        # ── Publish ──
        # Each wave holds a partial over its share of the contraction, published in `fold_rounds`
        # rounds of `folds` with the running sum carried in registers between them -- which is
        # what lets a wide token tile hold fewer slots than it has waves. The padding columns go
        # out too: the weight's absent rows contracted to zero, so the finish pass reads `rows`.
        fold_steps = block_m * PAD_ROWS // BLOCK
        carried = [fx.Float32(0.0) for _ in range_constexpr(fold_steps)]
        for r in range_constexpr(fold_rounds):
            fx.gpu.barrier()
            if (wave >= fx.Int32(r * folds)) & (wave < fx.Int32((r + 1) * folds)):
                fx.copy(acc_copy, cf_c, p_acc)
            fx.gpu.barrier()
            for g in range_constexpr(fold_steps):
                i = fx.Int32(g * BLOCK) + tid
                for h in range_constexpr(folds):
                    carried[g] = (
                        carried[g] + s_acc_flat[fx.Int32(h * block_m * PAD_ROWS) + i]
                    )
        for g in range_constexpr(fold_steps):
            g_part[fx.Int32(g * BLOCK) + tid] = carried[g]
        # The `groups` lanes that share a token are consecutive and aligned, so both sums close
        # in a butterfly inside the wave rather than through LDS and a barrier.
        width = fx.Int32(WAVE_SIZE)
        for g in range_constexpr(slots):
            total_sq, total_sqc = sq[g], sqc[g]
            for step in range_constexpr(butterfly):
                mask = fx.Int32(1 << step)
                total_sq = total_sq + fx.Float32(
                    fx.gpu.shuffle_xor(total_sq, mask, width)
                )
                total_sqc = total_sqc + fx.Float32(
                    fx.gpu.shuffle_xor(total_sqc, mask, width)
                )
            token = token0 + (fx.Int32(g * BLOCK) + tid) // fx.Int32(groups)
            if (tid % fx.Int32(groups) == fx.Int32(0)) & (token < n_tok):
                base = _offset(sums_major, seg, token, 0)
                _store_f32(sums, base, total_sq)
                _store_f32(sums, base + 1, total_sqc)

    @flyc.jit
    def launch_mhc_seam_fused(
        residual: fx.Pointer,
        sublayer_out: fx.Pointer,
        post_layer_mix: fx.Pointer,
        comb_res_mix: fx.Pointer,
        pre_mix: fx.Pointer,
        fn: fx.Pointer,
        residual_out: fx.Pointer,
        stage: fx.Pointer,
        part: fx.Pointer,
        sums: fx.Pointer,
        n_tok: fx.Int32,
        n_blk: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        mhc_seam_fused_kernel(
            residual,
            sublayer_out,
            post_layer_mix,
            comb_res_mix,
            pre_mix,
            fn,
            residual_out,
            stage,
            part,
            sums,
            n_tok,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(
            grid=(splits, n_blk, 1),
            block=BLOCK,
            stream=stream,
        )

    if fast_fp_math:
        launch_mhc_seam_fused.compile_hints = {
            **launch_mhc_seam_fused.compile_hints,
            "fast_fp_math": True,
        }
    launch_mhc_seam_fused.kernel_name = name
    return launch_mhc_seam_fused


@cache
def compile_mhc_seam_finish(
    *,
    streams: int,
    hidden: int,
    block_m: int,
    splits: int,
    rms_eps: float,
    norm_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    fast_fp_math: bool,
    fast_narrow: bool,
    norm_load_nt: bool,
    norm_store_nt: bool,
    waves_per_eu: int,
    gate_only: bool,
) -> Any:
    """The finish pass: fold the segments' partials, form the three gates, rescale the collapse.

    A workgroup owns `TOKENS_PER_BLOCK` tokens, one `SPLIT_SLOTS`-lane group per token, so every
    lane folds one published value. The grid's slow axis is the rescale's column steps
    (`finish_col_steps`); with `gate_only` a leading step carries the gate chain alone
    (`gate_step`).
    """
    # Fixed DPP controls make the gate chain a 4x4 comb and no other shape; every other value
    # compiles and returns a wrong `comb_out`, so it is refused rather than left to a host.
    _require(
        streams == STREAMS,
        f"the gate lane map is specialised for {STREAMS} residual streams, got {streams}",
    )
    rows = gate_rows(streams)
    wide = streams * hidden
    sq_slot, sqc_slot = rows, rows + 1
    inv_wide = 1.0 / float(wide)
    inv_hidden = 1.0 / float(hidden)
    line = block_m * SPLIT_SLOTS
    # Column steps of the rescale field, the grid's slow axis rather than a loop: the token count
    # alone leaves the pass short of grid.
    col_steps = finish_col_steps(hidden)
    # Column steps the rescale is shifted by: one where the gates take a step of their own.
    shift = 1 if gate_only else 0

    @fx.struct
    class SharedStorage:
        total: fx.Array[fx.Float32, TOKENS_PER_BLOCK * SPLIT_SLOTS]

    consts = _consts_tag(
        rms_eps, norm_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value
    )
    name = (
        f"mhc_seam_finish_n{streams}_h{hidden}_bm{block_m}_s{splits}_w{waves_per_eu}"
        f"_r{sinkhorn_repeat}_{consts}"
        + ("_gate" if gate_only else "")
        + ("_ffm" if fast_fp_math else "")
        + ("_fnr" if fast_narrow else "")
        + ("_ntl" if norm_load_nt else "")
        + ("_nts" if norm_store_nt else "")
    )

    @flyc.kernel(name=name, known_block_size=[BLOCK, 1, 1])
    def mhc_seam_finish_kernel(
        part: fx.Pointer,  # (T/block_m, splits, block_m, SPLIT_SLOTS) fp32
        sums: fx.Pointer,  # (splits, T, 2) fp32
        hc_scale: fx.Pointer,  # (3,) fp32
        hc_base: fx.Pointer,  # (gate_rows,) fp32
        stage: fx.Pointer,  # (T, H) bf16, the un-normalised collapse
        norm_weight: fx.Pointer,  # (H,) bf16
        layer_input: fx.Pointer,  # (T, H) bf16, `stage` rescaled in place
        post_out: fx.Pointer,  # (T, streams) fp32
        comb_out: fx.Pointer,  # (T, streams, streams) fp32
        next_pre: fx.Pointer,  # (T, streams) fp32
        n_tok: fx.Int32,
    ) -> None:
        tid = fx.thread_idx.x
        step = fx.block_idx.y
        token0 = fx.block_idx.x * fx.Int32(TOKENS_PER_BLOCK)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        total = lds.total.view(fx.make_layout(TOKENS_PER_BLOCK * SPLIT_SLOTS, 1))

        token_major = fx.make_layout((n_tok, hidden), (hidden, 1))  # stage, layer_input
        per_stream = fx.make_layout(
            (n_tok, streams), (streams, 1)
        )  # post_out, next_pre
        per_pair = fx.make_layout(
            (n_tok, streams, streams), (streams * streams, streams, 1)
        )  # comb_out
        sums_major = fx.make_layout((splits, n_tok, 2), (n_tok * 2, 2, 1))
        fold_major = fx.make_layout(
            (TOKENS_PER_BLOCK, SPLIT_SLOTS), (SPLIT_SLOTS, 1)
        )  # the folded lines

        # ── Issue the rescale's loads ──
        # Ahead of the fold, because they depend on neither it nor the barrier, and a compiler
        # will not move a load across `s_barrier`. Left where they read, a rescale block pays the
        # gather's round trip and the staged collapse's one after the other; clamping the address
        # instead of branching puts them in flight together.
        col_g = ((step - fx.Int32(shift)) * fx.Int32(BLOCK) + tid) * fx.Int32(VEC)
        live = (step >= fx.Int32(shift)) & (col_g < fx.Int32(hidden))
        safe_col = live.select(col_g, fx.Int32(0))
        weight = [fx.Float32(e) for e in _load_bf16(norm_weight, safe_col, VEC, False)]
        staged = [
            _load_bf16(
                stage,
                _offset(
                    token_major,
                    ((token0 + fx.Int32(s)) < n_tok).select(
                        token0 + fx.Int32(s), fx.Int32(0)
                    ),
                    0,
                )
                + safe_col,
                VEC,
                norm_load_nt,
            )
            for s in range_constexpr(TOKENS_PER_BLOCK)
        ]

        # ── Fold the segments ──
        # One lane per published value of the block's whole token group, so every lane is in the
        # gather. A workgroup owning a single token would leave it to 26 lanes and make the rest
        # wait on their latency at a barrier.
        local = tid // fx.Int32(SPLIT_SLOTS)
        col = tid - local * fx.Int32(SPLIT_SLOTS)
        token = token0 + local
        safe = (token < n_tok).select(token, fx.Int32(0))
        tile = safe // fx.Int32(block_m)
        row_in_tile = safe - tile * fx.Int32(block_m)
        # Lane -> projection row, so the gather lands each value where the gates want it: the
        # comb's rows 8..23 on the group's lanes 0..15, which is one whole 16-lane DPP row, then
        # pre 0..3 on 16..19 and post 4..7 on 20..23. The two sum slots are their own lane either
        # way, so the permutation leaves `sq_slot` and `sqc_slot` -- and the rescale's read of
        # them -- exactly where they were.
        row = (col < fx.Int32(streams * streams)).select(
            col + fx.Int32(2 * streams), col - fx.Int32(streams * streams)
        )
        row = (col < fx.Int32(rows)).select(row, col)
        # Only the gate step folds the whole line. A rescale step needs one value per token --
        # `sum(x1^2)` -- so it reads that column alone: `ceil(splits / MFMA_MN)` loads per lane
        # and one 16-lane DPP reduction, against the full line's gather.
        if step == fx.Int32(0):
            if col < fx.Int32(rows):
                folded = fx.Float32(0.0)
                for seg in range_constexpr(splits):
                    folded = folded + _load_f32(
                        part,
                        (tile * fx.Int32(splits) + fx.Int32(seg)) * fx.Int32(line)
                        + row_in_tile * fx.Int32(SPLIT_SLOTS)
                        + row,
                    )
                total[tid] = folded
            if (col >= fx.Int32(rows)) & (col < fx.Int32(rows + 2)):
                folded = fx.Float32(0.0)
                for seg in range_constexpr(splits):
                    folded = folded + _load_f32(
                        sums, _offset(sums_major, seg, safe, 0) + col - fx.Int32(rows)
                    )
                total[tid] = folded
            # A group is `SPLIT_SLOTS` lanes and the fold fills `rows + 2` of them. The rest are
            # read unconditionally below and carried through the gate chain, so they are written
            # rather than left holding whatever the last call put there: under `fast_fp_math` a
            # stale NaN propagates through the chain silently before the guarded stores drop it.
            if col >= fx.Int32(rows + 2):
                total[tid] = fx.Float32(0.0)
        else:
            # The group's lower 16 lanes are one DPP row, so a row sum and a column sum over it
            # close the whole 16-lane reduction; the segments are dealt round-robin across them.
            if col < fx.Int32(MFMA_MN):
                narrow = fx.Float32(0.0)
                for seg in range_constexpr(ceildiv(splits, MFMA_MN)):
                    idx = col + fx.Int32(seg * MFMA_MN)
                    narrow = narrow + (idx < fx.Int32(splits)).select(
                        _load_f32(
                            sums,
                            _offset(sums_major, 0, safe, 0)
                            + (idx < fx.Int32(splits)).select(idx, fx.Int32(0))
                            * n_tok
                            * fx.Int32(2)
                            + fx.Int32(1),
                        ),
                        fx.Float32(0.0),
                    )
                narrow = _col_sum(_row_sum(narrow))
                if col == fx.Int32(0):
                    total[local * fx.Int32(SPLIT_SLOTS) + fx.Int32(sqc_slot)] = narrow
        fx.gpu.barrier()

        # ── Gates ──
        # On the lane map the fold already produced -- a 32-lane group per token, the projection
        # row on the lane -- in the leading column step alone, since the other steps would only
        # duplicate it.
        #
        # The reduction is written as DPP rather than through the algebra, which has no term for
        # a cross-lane move. Collapsed into one lane each Sinkhorn half-step is eight dependent
        # reductions and sixteen multiplies issued by one live lane; on the lanes it is two DPP
        # moves, a `v_rcp_f32` and a multiply, and where the token count leaves the part empty
        # there is no co-resident wave to hide the difference.
        if (step == fx.Int32(0)) & (token < n_tok):
            mine = total[tid]
            rstd = fmath.rsqrt(
                total[local * fx.Int32(SPLIT_SLOTS) + fx.Int32(sq_slot)] * inv_wide
                + rms_eps
            )
            scale = (col < fx.Int32(streams * streams)).select(
                _load_f32(hc_scale, 2),
                (col < fx.Int32(streams * streams + streams)).select(
                    _load_f32(hc_scale, 0), _load_f32(hc_scale, 1)
                ),
            )
            logit = mine * rstd * scale + _load_f32(
                hc_base, (col < fx.Int32(rows)).select(row, fx.Int32(0))
            )

            # Sigmoid on every lane; only the pre and post lanes reach a store.
            sig = sigmoid_f32(logit)

            comb = _exp2((logit - _row_max(logit)) * LOG2E)
            comb = comb * _rcp(_row_sum(comb)) + hc_sinkhorn_eps
            comb = comb * _rcp(_col_sum(comb) + hc_sinkhorn_eps)
            for _ in range_constexpr(sinkhorn_repeat - 1):
                comb = comb * _rcp(_row_sum(comb) + hc_sinkhorn_eps)
                comb = comb * _rcp(_col_sum(comb) + hc_sinkhorn_eps)

            comb_row = _offset(per_pair, token0, 0, 0) + local * fx.Int32(
                streams * streams
            )
            stream_row = _offset(per_stream, token0, 0) + local * fx.Int32(streams)
            if col < fx.Int32(streams * streams):
                _store_f32(comb_out, comb_row + col, comb)
            if (col >= fx.Int32(streams * streams)) & (
                col < fx.Int32(streams * streams + streams)
            ):
                _store_f32(
                    next_pre,
                    stream_row + col - fx.Int32(streams * streams),
                    sig + hc_pre_eps,
                )
            if (col >= fx.Int32(streams * streams + streams)) & (col < fx.Int32(rows)):
                _store_f32(
                    post_out,
                    stream_row + col - fx.Int32(streams * streams + streams),
                    sig * hc_post_mult_value,
                )

        # ── RMSNorm rescale of the staged collapse ──
        # The token is a compile-time repeat and the column a dynamic sweep, so neither the row
        # scale nor the row origin is recomputed per group.
        if live:
            for s in range_constexpr(TOKENS_PER_BLOCK):
                row_token = token0 + fx.Int32(s)
                if row_token < n_tok:
                    norm_rstd = fmath.rsqrt(
                        total[_offset(fold_major, s, sqc_slot)] * inv_hidden + norm_eps
                    )
                    _store_bf16(
                        layer_input,
                        _offset(token_major, row_token, 0) + col_g,
                        [
                            _narrow_round(
                                fx.Float32(staged[s][e]) * norm_rstd * weight[e],
                                fast_narrow,
                            )[0]
                            for e in range_constexpr(VEC)
                        ],
                        norm_store_nt,
                    )

    @flyc.jit
    def launch_mhc_seam_finish(
        part: fx.Pointer,
        sums: fx.Pointer,
        hc_scale: fx.Pointer,
        hc_base: fx.Pointer,
        stage: fx.Pointer,
        norm_weight: fx.Pointer,
        layer_input: fx.Pointer,
        post_out: fx.Pointer,
        comb_out: fx.Pointer,
        next_pre: fx.Pointer,
        n_tok: fx.Int32,
        n_blk: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        mhc_seam_finish_kernel(
            part,
            sums,
            hc_scale,
            hc_base,
            stage,
            norm_weight,
            layer_input,
            post_out,
            comb_out,
            next_pre,
            n_tok,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(
            grid=(n_blk, col_steps + shift, 1),
            block=BLOCK,
            stream=stream,
        )

    if fast_fp_math:
        launch_mhc_seam_finish.compile_hints = {
            **launch_mhc_seam_finish.compile_hints,
            "fast_fp_math": True,
        }
    launch_mhc_seam_finish.kernel_name = name
    return launch_mhc_seam_finish
