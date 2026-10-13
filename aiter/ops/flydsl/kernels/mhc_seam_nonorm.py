# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Delayed mHC seam without the RMSNorm -- the fused pass and the gates finish pass.

The `NORM="separate"` path of `flydsl_mhc_seam`: the contract of `mhc_seam.py`, except that
`layer_input` leaves un-normalised and the caller applies the RMSNorm after it. Only the
post-mix seam with a carried pre gate is served; there are no `has_post` / `identity_pre`
paths. Helpers and constants shared with `mhc_seam.py` are imported from it.

For the four residual streams R, the previous seam's gates post, comb and pre, and this
seam's weight fn:

    R'_j   = bf16(post_j * y + sum_h comb[h][j] * R_h)          -> residual_out
    x1     = bf16(sum_j pre_j * R'_j)                            -> layer_input
    mixes  = (fn @ flatten(R')) * rsqrt(mean(R'^2) + rms_eps)    -> the 24 gate logits

and the gates it hands the next seam, off the same logits:

    pre'_j  = sigmoid(mixes_pre_j * hc_scale[0] + hc_base) + hc_pre_eps
    post'_j = sigmoid(mixes_post_j * hc_scale[1] + hc_base) * hc_post_mult_value
    comb'   = sinkhorn(softmax_row(mixes_comb * hc_scale[2] + hc_base))

A workgroup owns `block_m` tokens of one hidden segment and sweeps it in `chunk_c`-column
chunks: it post-mixes the chunk, writes both bf16 outputs, and publishes the new residual into
an LDS tile one stream at a time, where the four waves contract their share of its columns
against `fn` with a one-wave tiled `MFMA(16, 16, 16, bf16)`.

`fn` is carried as two bf16 limbs, about sixteen mantissa bits between them, which is what
the gate tolerances need. The host stacks the limbs' 24 rows into 48 row-products, three
16-wide N tiles, and preshuffles them into MFMA B lane order
(`mhc_seam_kernels._pack_fn_nonorm`). A row's two limbs land in different accumulators, and
the tail adds them while it folds the four waves' partials.

Neither the RMS nor the logits close inside a workgroup once the hidden axis is split, so the
fused pass publishes per (segment, token) the unscaled logits and the segment's sum of
squares, and the finish pass folds them and forms the gates, at `splits == 1` too.

Gate packing, as in `mhc_seam.py`: `fn` is `(gate_rows(STREAMS), STREAMS * hidden)` fp32 with
rows `pre | post | comb` (comb row-major in `(i, j)`), `hc_scale` is `(pre, post, comb)`, and
`hc_base` has one fp32 per row in `fn`'s order.

Launch contract: the token count is a kernarg, so one build serves every token count with the
same geometry. The fused grid is `(ceil(tokens / block_m), splits)` and the finish grid
`finish_blocks(tokens)`; every scratch element the finish pass reads is then written by the
fused pass of the same call, so the scratch needs no zeroing. The tile geometry: see
`check_config`.
"""

from functools import cache
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr

from aiter.ops.flydsl.kernels.act import sigmoid_f32
from aiter.ops.flydsl.kernels.kernels_common import LOG2E, ceildiv
from aiter.ops.flydsl.kernels.mhc_seam import (
    BLOCK,
    CHUNK_GRAIN,
    DEFAULT_LDS_BYTES,
    MAX_FLAT_ELEMENTS,
    MFMA_K,
    MFMA_MN,
    STREAMS,
    VEC,
    WAVE_SIZE,
    _col_sum,
    _consts_tag,
    _exp2,
    _load_bf16,
    _load_f32,
    _narrow_round,
    _rcp,
    _require,
    _row_max,
    _row_sum,
    _store_bf16,
    _store_f32,
    gate_rows,
)

N = STREAMS
ROWS = gate_rows(N)  # 24 gate projection rows: pre, post, then the 4x4 comb
# Per-token post, comb and carried-pre coefficients the post-mix reads, staged in LDS.
GATE_COEFFS = N + N * N + N
# The two limbs' 24 rows stacked are 48 row-products, exactly three 16-wide N tiles with
# no padding. Limb `l` row `r` is slot `24 * l + r`.
PACK_ROWS = 2 * ROWS
N_TILES = PACK_ROWS // MFMA_MN
WAVES = BLOCK // WAVE_SIZE  # four, which split the projection's k range between them
# LDS row padding in bf16 elements: each token row starts two banks after the one above,
# so the A fragment's sixteen-row read spreads over the banks.
TILE_PAD = 4
# One split's per-token partial: the 24 projection logits and the segment's sum of
# squares, padded to a 128 B line so a finishing lane reads exactly one line per split.
SPLIT_SLOTS = 32
# Lanes the finish pass spends on one token: the 4x4 comb needs a whole 16-lane DPP row,
# then pre on 16..19, post on 20..23 and the sum of squares on 24.
LANES_PER_TOKEN = 32


def smem_bytes(block_m: int, chunk_c: int) -> int:
    """Bytes of the fused pass's LDS union: the staged tile and gate coefficients the
    chunk loop uses, or the partials and slot sums the tail folds, whichever is larger.
    """
    staged = ceildiv(block_m * (chunk_c + TILE_PAD), 2) * 4 + block_m * GATE_COEFFS * 4
    folded = (WAVES * block_m * PACK_ROWS + block_m * (chunk_c // VEC)) * 4
    return max(staged, folded)


def part_elements(tokens: int, splits: int) -> int:
    """fp32 the fused pass hands the finish pass: one line per (segment, token)."""
    return splits * max(tokens, 1) * SPLIT_SLOTS


def finish_blocks(tokens: int) -> int:
    """Grid extent of the finish launch, which the host passes as its `n_blk`."""
    return ceildiv(max(tokens, 1), BLOCK // LANES_PER_TOKEN)


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

    `lds_bytes` is the part's LDS budget, which the caller knows and this module does not.
    """
    _require(
        streams == STREAMS,
        f"the gate lane map is specialised for {STREAMS} residual streams, got {streams}",
    )
    _require(
        fast_narrow or not fast_fp_math,
        "fast_fp_math folds the bf16 narrow-widen round trip the contract requires; "
        "it needs fast_narrow",
    )
    _require(
        block_m >= MFMA_MN and block_m % MFMA_MN == 0,
        f"block_m must be a positive multiple of the {MFMA_MN}-token MFMA M tile, "
        f"got {block_m}",
    )
    _require(
        block_m <= BLOCK,
        f"the gate tail maps one thread per token, so block_m must not exceed {BLOCK}, "
        f"got {block_m}",
    )
    _require(
        chunk_c >= CHUNK_GRAIN and chunk_c % CHUNK_GRAIN == 0,
        f"the four waves split each chunk's k range, so chunk_c must be a positive "
        f"multiple of {CHUNK_GRAIN}, got {chunk_c}",
    )
    _require(hidden % chunk_c == 0, f"chunk_c {chunk_c} must divide hidden {hidden}")
    _require(
        (block_m * (chunk_c // VEC)) % BLOCK == 0,
        f"block_m * chunk_c / {VEC} must be a multiple of {BLOCK}, so the post-mix slots "
        "divide evenly over the block",
    )
    _require(
        splits >= 1 and (hidden // chunk_c) % splits == 0,
        f"splits {splits} must divide the {hidden // chunk_c} chunks of the hidden sweep",
    )
    smem = smem_bytes(block_m, chunk_c)
    _require(
        smem <= lds_bytes,
        f"the {smem} B LDS union exceeds the {lds_bytes} B budget",
    )
    _require(
        tokens * streams * hidden <= MAX_FLAT_ELEMENTS,
        f"flat int32 indexing caps tokens * streams * hidden at {MAX_FLAT_ELEMENTS}",
    )


def _post_mixed(
    post_g: list, comb_g: list, y: list, res_f: list, j: int, e: int
) -> Any:
    """One post-mixed fp32 element: `post[j] * y + sum_h comb[h][j] * R[h]`."""
    mixed = post_g[j] * y[e]
    for i in range(len(comb_g)):
        mixed = mixed + comb_g[i][j] * res_f[i][e]
    return mixed


def _sq_total(sq_scratch: Any, tid: Any, groups_per_token: int) -> Any:
    """`sum R'^2` over this token's column groups, as the chunk loop left them in LDS."""
    total = fx.Float32(0.0)
    for p in range_constexpr(groups_per_token):
        total = total + sq_scratch[tid * fx.Int32(groups_per_token) + fx.Int32(p)]
    return total


@cache
def compile_mhc_seam_fused_nonorm(
    *,
    hidden: int,
    block_m: int,
    chunk_c: int,
    splits: int,
    load_nt: bool,
    load_nt_x: bool,
    store_nt: bool,
    fast_fp_math: bool,
    fast_narrow: bool,
    waves_per_eu: int,
) -> Any:
    """The fused pass: post-mix, collapse, gate projection and sum of squares, published
    per (segment, token) for `compile_mhc_seam_finish_nonorm` to fold.

    `load_nt` hints the four-stream `residual` read and `load_nt_x` the `sublayer_out`
    read: the two want opposite answers in part of the token band. `store_nt` hints both
    writes, which nothing in the seam reads back.
    """
    check_config(
        0,
        hidden,
        STREAMS,
        block_m,
        chunk_c,
        splits,
        lds_bytes=1 << 30,
        fast_fp_math=fast_fp_math,
        fast_narrow=fast_narrow,
    )
    # A workgroup owns one of `splits` equal hidden segments, so the grid is `splits`
    # times wider. The post-mix and its two outputs are elementwise in the hidden column,
    # so the split leaves them alone.
    seg_chunks = hidden // chunk_c // splits
    groups_per_token = chunk_c // VEC
    slots = (block_m * groups_per_token) // BLOCK
    wide = N * hidden
    stride = chunk_c + TILE_PAD
    # The tile holds one stream, so each chunk is published and contracted in `N` rounds,
    # and wave `w` contracts the `w`-th `kspan` columns of each.
    kspan = chunk_c // WAVES
    k_steps = kspan // MFMA_K
    pre_base = N + N * N

    # 16 B alignment, not the element's: a lane's A fragment is one 8 B read, and without
    # provable alignment it splits into narrower ones, which multiplies the LDS reads and
    # the VGPRs holding them.
    @fx.struct
    class Staged:
        tile: fx.Array[fx.BFloat16, block_m * stride, 16]
        gates: fx.Array[fx.Float32, block_m * GATE_COEFFS, 16]

    @fx.struct
    class Folded:
        parts: fx.Array[fx.Float32, WAVES * block_m * PACK_ROWS, 16]
        sq: fx.Array[fx.Float32, slots * BLOCK, 16]

    # The tail's partials take the words of the tile and the gate coefficients, which are
    # both dead once the chunk loop's last barrier has passed.
    @fx.union
    class SharedStorage:
        staged: Staged
        folded: Folded

    name = (
        f"mhc_seam_fused_nonorm_h{hidden}_bm{block_m}_c{chunk_c}_s{splits}_w{waves_per_eu}"
        + ("_ffm" if fast_fp_math else "")
        + ("_fnr" if fast_narrow else "")
        + ("_ntl" if load_nt else "")
        + ("_ntx" if load_nt_x else "")
        + ("_nts" if store_nt else "")
    )

    @flyc.kernel(name=name, known_block_size=[BLOCK, 1, 1])
    def mhc_seam_fused_nonorm_kernel(
        residual: fx.Pointer,  # (T, 4, H) bf16
        sublayer_out: fx.Pointer,  # (T, H) bf16
        post_layer_mix: fx.Pointer,  # (T, 4) fp32
        comb_res_mix: fx.Pointer,  # (T, 4, 4) fp32
        pre_mix: fx.Pointer,  # (T, 4) fp32
        fn_packed: fx.Pointer,  # the packed limbs, bf16
        residual_out: fx.Pointer,  # (T, 4, H) bf16
        layer_input: fx.Pointer,  # (T, H) bf16
        part: fx.Pointer,  # part_elements(T, splits) fp32
        n_tok: fx.Int32,
    ) -> None:
        tid = fx.thread_idx.x
        t0 = fx.block_idx.x * fx.Int32(block_m)
        # `seg` is this workgroup's hidden segment, `chunk0` its first chunk.
        seg = fx.block_idx.y
        chunk0 = seg * fx.Int32(seg_chunks)
        lane = tid % fx.Int32(WAVE_SIZE)
        wave = tid // fx.Int32(WAVE_SIZE)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        tile = storage.staged.tile.peek().view(
            fx.make_layout((block_m, stride), (stride, 1))
        )
        gates = storage.staged.gates.peek().view(
            fx.make_layout((block_m, GATE_COEFFS), (GATE_COEFFS, 1))
        )
        parts = storage.folded.parts.peek().view(
            fx.make_layout(
                (WAVES, block_m, PACK_ROWS), (block_m * PACK_ROWS, PACK_ROWS, 1)
            )
        )
        sq_scratch = storage.folded.sq.peek().view(fx.make_layout(slots * BLOCK, 1))

        # Each wave contracts its own column span of the staged tile against the matching
        # k range of `fn`, so the tiled MMA is one wave wide and every partition is sliced
        # by `lane`.
        mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(MFMA_MN, MFMA_MN, MFMA_K, fx.BFloat16)
        )
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane)
        # A lane's share of an A or B atom is four bf16 adjacent in k: one 8 B access.
        # `fn` takes a plain global copy rather than a buffer-resource one: the descriptor's
        # extra VGPRs push the smallest token tile across an allocation step and cost a wave
        # of occupancy.
        lds_copy = fx.make_copy_atom(fx.UniversalCopy(64), fx.BFloat16)
        wgt_copy = fx.make_copy_atom(fx.UniversalCopy(64), fx.BFloat16)
        acc_copy = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
        thr_a = fx.make_tiled_copy_A(lds_copy, tiled_mma).get_slice(lane)
        thr_b = fx.make_tiled_copy_B(wgt_copy, tiled_mma).get_slice(lane)
        thr_c = fx.make_tiled_copy_C(acc_copy, tiled_mma).get_slice(lane)

        span = fx.make_view(
            fx.add_offset(fx.get_iter(tile), wave * fx.Int32(kspan)),
            fx.make_layout((block_m, kspan), (stride, 1)),
        )
        t_a = fx.flat_divide(span, fx.make_tile(block_m, MFMA_K))[None, None, 0, None]
        # The caller's preshuffle, read through the logical (row-product, k) coordinates:
        # element (n, k) is at
        # `(k // 16) * 768 + (n // 16) * 256 + (k % 16 // 4) * 64 + (n % 16) * 4 + k % 4`.
        g_fn = fx.Tensor(
            fx.make_view(
                fn_packed,
                fx.make_layout(
                    ((16, N_TILES), (4, 4, wide // MFMA_K)),
                    ((4, 256), (1, 64, 16 * PACK_ROWS)),
                ),
            )
        )
        t_fn = fx.flat_divide(g_fn, fx.make_tile(PACK_ROWS, MFMA_K))[
            None, None, 0, None
        ]
        my_parts = fx.slice(parts, (wave, None, None))
        frag_a = thr_mma.make_fragment_A(t_a[None, None, 0])
        frag_b = thr_mma.make_fragment_B(t_fn[None, None, 0])
        frag_c = thr_mma.make_fragment_C(my_parts)
        frag_c.fill(0)
        # Every partition is taken here, outside the dynamic loops: a ThrCopy referenced
        # inside one is taken for loop-carried state and cannot be reconstructed.
        p_a = thr_a.partition_S(t_a)
        p_fn = thr_b.partition_S(t_fn)
        p_c = thr_c.partition_S(my_parts)
        cf_a = thr_a.retile(frag_a)
        cf_b = thr_b.retile(frag_b)
        cf_c = thr_c.retile(frag_c)

        # Coefficients 0..3 are `post`, 4..19 `comb`, 20..23 the carried `pre`. Every arm
        # of a select is evaluated, so each band's index is clamped into its own band
        # rather than allowed to run off either end of its tensor.
        for q in range_constexpr(ceildiv(block_m * GATE_COEFFS, BLOCK)):
            idx = fx.Int32(q * BLOCK) + tid
            if idx < fx.Int32(block_m * GATE_COEFFS):
                gate_t = idx // fx.Int32(GATE_COEFFS)
                slot_i = idx % fx.Int32(GATE_COEFFS)
                token = t0 + gate_t
                safe = (token < n_tok).select(token, fx.Int32(0))
                in_post = slot_i < fx.Int32(N)
                in_comb = slot_i < fx.Int32(N + N * N)
                post_i = in_post.select(slot_i, fx.Int32(0))
                comb_i = in_post.select(
                    fx.Int32(0),
                    in_comb.select(slot_i - fx.Int32(N), fx.Int32(N * N - 1)),
                )
                pre_i = in_comb.select(fx.Int32(0), slot_i - fx.Int32(N + N * N))
                gates[gate_t, slot_i] = in_post.select(
                    _load_f32(post_layer_mix, safe * N + post_i),
                    in_comb.select(
                        _load_f32(comb_res_mix, safe * (N * N) + comb_i),
                        _load_f32(pre_mix, safe * N + pre_i),
                    ),
                )

        mix_t, mix_col, live, safe_token = [], [], [], []
        for g in range_constexpr(slots):
            slot = fx.Int32(g * BLOCK) + tid
            local_t = slot // fx.Int32(groups_per_token)
            mix_t.append(local_t)
            mix_col.append((slot % fx.Int32(groups_per_token)) * fx.Int32(VEC))
            token = t0 + local_t
            in_range = token < n_tok
            live.append(in_range)
            safe_token.append(in_range.select(token, fx.Int32(0)))
        sq = [fx.Float32(0.0) for _ in range_constexpr(slots)]
        fx.gpu.barrier()

        for chunk in range(seg_chunks):
            chunk_i = chunk0 + chunk
            col0 = chunk_i * fx.Int32(chunk_c)
            # The four streams' narrowed values are held until their publish round, since
            # the tile takes one stream at a time and the post-mix runs once.
            held = []
            for g in range_constexpr(slots):
                base = safe_token[g] * fx.Int32(wide) + col0 + mix_col[g]
                flat = safe_token[g] * fx.Int32(hidden) + col0 + mix_col[g]
                res = [
                    _load_bf16(residual, base + fx.Int32(i * hidden), VEC, load_nt)
                    for i in range_constexpr(N)
                ]
                pre_g = [
                    gates[mix_t[g], fx.Int32(pre_base + j)] for j in range_constexpr(N)
                ]
                acc_y = [fx.Float32(0.0) for _ in range_constexpr(VEC)]
                y = [
                    fx.Float32(e)
                    for e in _load_bf16(sublayer_out, flat, VEC, load_nt_x)
                ]
                post_g = [gates[mix_t[g], fx.Int32(j)] for j in range_constexpr(N)]
                comb_g = [
                    [
                        gates[mix_t[g], fx.Int32(N + i * N + j)]
                        for j in range_constexpr(N)
                    ]
                    for i in range_constexpr(N)
                ]
                res_f = [
                    [fx.Float32(res[i][e]) for e in range_constexpr(VEC)]
                    for i in range_constexpr(N)
                ]

                # One stream at a time, arithmetic fully ahead of the stores: the AMDGPU
                # load-store vectorizer merges only stores adjacent in program order.
                per_stream = []
                for j in range_constexpr(N):
                    narrowed = []
                    for e in range_constexpr(VEC):
                        value, widened = _narrow_round(
                            _post_mixed(post_g, comb_g, y, res_f, j, e), fast_narrow
                        )
                        sq[g] = sq[g] + widened * widened
                        acc_y[e] = acc_y[e] + pre_g[j] * widened
                        narrowed.append(value)
                    if live[g]:
                        _store_bf16(
                            residual_out,
                            base + fx.Int32(j * hidden),
                            narrowed,
                            store_nt,
                        )
                    per_stream.append(narrowed)
                held.append(per_stream)
                if live[g]:
                    _store_bf16(
                        layer_input,
                        flat,
                        [
                            _narrow_round(acc_y[e], fast_narrow)[0]
                            for e in range_constexpr(VEC)
                        ],
                        store_nt,
                    )

            for r in range_constexpr(N):
                for g in range_constexpr(slots):
                    for e in range_constexpr(VEC):
                        tile[mix_t[g], mix_col[g] + fx.Int32(e)] = held[g][r][e]
                fx.gpu.barrier()
                k_tile_base = (
                    fx.Int32(r * (hidden // MFMA_K))
                    + chunk_i * fx.Int32(chunk_c // MFMA_K)
                    + wave * fx.Int32(kspan // MFMA_K)
                )
                for k_step in range(k_steps):
                    fx.copy(lds_copy, p_a[None, None, None, k_step], cf_a)
                    # One B fragment feeds every M tile, so a taller token tile reads `fn`
                    # fewer times per token. A row's two limbs are in different N tiles.
                    fx.copy(
                        wgt_copy, p_fn[None, None, None, k_tile_base + k_step], cf_b
                    )
                    fx.gemm(tiled_mma, frag_c, frag_a, frag_b, frag_c)
                fx.gpu.barrier()

        fx.copy(acc_copy, cf_c, p_c)
        for g in range_constexpr(slots):
            sq_scratch[fx.Int32(g * BLOCK) + tid] = sq[g]
        fx.gpu.barrier()

        # Tail: one thread per live token folds the four waves' partials and publishes
        # this segment's logits and sum of squares unscaled -- the RMS is over the whole
        # hidden axis, so no workgroup can apply it. A row's two limbs sit in different N
        # tiles at slots `r` and `ROWS + r`, so the limb sum joins the cross-wave fold.
        out_t = t0 + tid
        if (tid < fx.Int32(block_m)) & (out_t < n_tok):
            slot0 = (seg * n_tok + out_t) * fx.Int32(SPLIT_SLOTS)
            for r in range_constexpr(ROWS):
                folded = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for sl in (r, ROWS + r):
                        folded = folded + parts[fx.Int32(w), tid, fx.Int32(sl)]
                _store_f32(part, slot0 + fx.Int32(r), folded)
            _store_f32(
                part,
                slot0 + fx.Int32(ROWS),
                _sq_total(sq_scratch, tid, groups_per_token),
            )

    @flyc.jit
    def launch_mhc_seam_fused_nonorm(
        residual: fx.Pointer,
        sublayer_out: fx.Pointer,
        post_layer_mix: fx.Pointer,
        comb_res_mix: fx.Pointer,
        pre_mix: fx.Pointer,
        fn_packed: fx.Pointer,
        residual_out: fx.Pointer,
        layer_input: fx.Pointer,
        part: fx.Pointer,
        n_tok: fx.Int32,
        n_blk: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        mhc_seam_fused_nonorm_kernel(
            residual,
            sublayer_out,
            post_layer_mix,
            comb_res_mix,
            pre_mix,
            fn_packed,
            residual_out,
            layer_input,
            part,
            n_tok,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(grid=(n_blk, splits, 1), block=BLOCK, stream=stream)

    # Sound only with `fast_narrow`, which keeps the post-mix's bf16 round trip out of
    # `fpext(fptrunc x) -> x`; `check_config` enforces the pairing.
    if fast_fp_math:
        launch_mhc_seam_fused_nonorm.compile_hints = {
            **launch_mhc_seam_fused_nonorm.compile_hints,
            "fast_fp_math": True,
        }
    launch_mhc_seam_fused_nonorm.kernel_name = name
    return launch_mhc_seam_fused_nonorm


@cache
def compile_mhc_seam_finish_nonorm(
    *,
    hidden: int,
    splits: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    fast_fp_math: bool = True,
) -> Any:
    """Fold the fused pass's per-segment partials and run the gates.

    Only the reduction the hidden split makes cross-workgroup lives here: the sum of
    squares that sets the RMS scale, and the 24 projection logits. Both bf16 outputs are
    elementwise in the hidden column, so the fused pass finishes those itself.

    One token spreads over `LANES_PER_TOKEN` lanes -- one lane per projection row, the 4x4
    comb on a single 16-lane DPP row -- so each Sinkhorn half-step is two DPP adds, a
    reciprocal and a multiply per lane instead of 16 serial sums and 16 multiplies in one
    thread.
    """
    _require(splits >= 1, f"splits must be at least 1, got {splits}")
    inv_wide = 1.0 / float(N * hidden)
    tokens_per_block = BLOCK // LANES_PER_TOKEN
    name = (
        f"mhc_seam_finish_nonorm_h{hidden}_s{splits}_r{sinkhorn_repeat}_"
        + _consts_tag(rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value)
        + ("_ffm" if fast_fp_math else "")
    )

    @flyc.kernel(name=name, known_block_size=[BLOCK, 1, 1])
    def mhc_seam_finish_nonorm_kernel(
        part: fx.Pointer,  # part_elements(T, splits) fp32
        hc_scale: fx.Pointer,  # (3,) fp32
        hc_base: fx.Pointer,  # (24,) fp32
        post_out: fx.Pointer,  # (T, 4) fp32
        comb_out: fx.Pointer,  # (T, 4, 4) fp32
        next_pre: fx.Pointer,  # (T, 4) fp32
        n_tok: fx.Int32,
    ) -> None:
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(LANES_PER_TOKEN)
        raw_token = fx.block_idx.x * fx.Int32(tokens_per_block) + tid // fx.Int32(
            LANES_PER_TOKEN
        )
        live = raw_token < n_tok
        # The arithmetic runs unconditionally against a clamped token so the three stores
        # are the kernel's only branches: a value computed in an enclosing `if` body and
        # read from a nested one comes back permuted, so nesting is not available here.
        t = live.select(raw_token, n_tok - fx.Int32(1))

        # Lane -> projection row: comb rows 8..23 on lanes 0..15, pre 0..3 on 16..19, post
        # 4..7 on 20..23, the sum of squares on 24; lanes 25..31 are inert.
        row = (lane < fx.Int32(16)).select(lane + fx.Int32(8), lane - fx.Int32(16))
        row = (lane < fx.Int32(24)).select(row, lane)

        total = fx.Float32(0.0)
        squares = fx.Float32(0.0)
        for s in range_constexpr(splits):
            base = (fx.Int32(s) * n_tok + t) * fx.Int32(SPLIT_SLOTS)
            total = total + _load_f32(part, base + row)
            # The sum of squares shares the lane's own 128 B window, so this is an L1 hit
            # and saves a cross-lane broadcast of lane 24.
            squares = squares + _load_f32(part, base + fx.Int32(ROWS))

        rstd = fmath.rsqrt(squares * inv_wide + rms_eps)
        s_pre = _load_f32(hc_scale, fx.Int32(0))
        s_post = _load_f32(hc_scale, fx.Int32(1))
        s_comb = _load_f32(hc_scale, fx.Int32(2))
        scale = (lane < fx.Int32(16)).select(
            s_comb, (lane < fx.Int32(20)).select(s_pre, s_post)
        )
        bias_row = (row < fx.Int32(ROWS)).select(row, fx.Int32(0))
        logit = total * rstd * scale + _load_f32(hc_base, bias_row)

        # Sigmoid on every lane; only 16..23 reach a store.
        sig = sigmoid_f32(logit)

        # Softmax across each comb row, then the `2 * repeat - 1` alternating half-steps,
        # opening on the column pass as the reference does.
        comb = _exp2((logit - _row_max(logit)) * LOG2E)
        comb = comb * _rcp(_row_sum(comb)) + hc_sinkhorn_eps
        comb = comb * _rcp(_col_sum(comb) + hc_sinkhorn_eps)
        for _ in range_constexpr(sinkhorn_repeat - 1):
            comb = comb * _rcp(_row_sum(comb) + hc_sinkhorn_eps)
            comb = comb * _rcp(_col_sum(comb) + hc_sinkhorn_eps)

        if live & (lane < fx.Int32(16)):
            _store_f32(comb_out, t * fx.Int32(N * N) + lane, comb)
        if live & (lane >= fx.Int32(16)) & (lane < fx.Int32(20)):
            _store_f32(
                next_pre, t * fx.Int32(N) + lane - fx.Int32(16), sig + hc_pre_eps
            )
        if live & (lane >= fx.Int32(20)) & (lane < fx.Int32(24)):
            _store_f32(
                post_out,
                t * fx.Int32(N) + lane - fx.Int32(20),
                sig * hc_post_mult_value,
            )

    @flyc.jit
    def launch_mhc_seam_finish_nonorm(
        part: fx.Pointer,
        hc_scale: fx.Pointer,
        hc_base: fx.Pointer,
        post_out: fx.Pointer,
        comb_out: fx.Pointer,
        next_pre: fx.Pointer,
        n_tok: fx.Int32,
        n_blk: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        mhc_seam_finish_nonorm_kernel(
            part, hc_scale, hc_base, post_out, comb_out, next_pre, n_tok
        ).launch(grid=(n_blk, 1, 1), block=BLOCK, stream=stream)

    if fast_fp_math:
        launch_mhc_seam_finish_nonorm.compile_hints = {
            **launch_mhc_seam_finish_nonorm.compile_hints,
            "fast_fp_math": True,
        }
    launch_mhc_seam_finish_nonorm.kernel_name = name
    return launch_mhc_seam_finish_nonorm
