# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Short-context Gluon (gfx950) body for paged decode attention with a sigmoid
output gate and an optional group-128 FP8 epilogue.

Same op, same contract and same outputs as the long-context body in
`paged_attention_output_gate.py`; a different decomposition of the same work.
The launcher picks between them on the caller's capacity hint, because the two
cross over at real context 32768: below it this body is up to 24% faster
(bs=1, 8K), above it the long body pulls ahead (~4% at 64K).

What is different here:

* **Four producer waves split QK by channel, not by head.** Each wave computes a
  64-channel partial of every head's score, and `_head_softmax` sums the four
  partials through LDS. The MFMA is the native 4x64x64 shape, so four query
  heads are not padded to a 16-row tile the way `HEAD_TILE = 16` pads them in
  the long body. That is where the low-batch win comes from, and it is also why
  `HEADS` has to be a multiple of 4 here (4, 8 or 16) instead of anything in
  1..16 -- the launcher sends the rest to the long body.
* **`ROWS` is a `gl.constexpr`.** The batch size selects the K/V staging
  strategy: four rows or more alternate K and V through one shared tile, while
  smaller batches keep separate tiles so the V transfer overlaps QK and the
  successor K overlaps PV. Batch 32 additionally keeps two upcoming page-index
  tiles in registers. None of this is a batch *table* -- every arm is derived
  from `ROWS` arithmetically, so non-power-of-two captures (12, 24, 40, 48, 56)
  compile and run like any other size.
* **The sigmoid is staged by the producer.** A dedicated gate CTA (partition
  zero at batch 32) writes `sigmoid(gate)` into the `gated` output buffer, and
  `_finish_group` reads it back as the epilogue's first operand, off the
  reduction's dependency chain. The buffer is scratch until `_short_finish`
  overwrites it, so the FP8 outputs must never alias it -- see `QUANTIZE`.
* **`_short_finish` is per (row, head-half).** The quant group is the CTA too,
  but the grid is `(rows, HEADS*2)` rather than `(rows*HEADS, 2)`, and the
  reduction width is chosen from the live length among six compiled arms.

Deviations from the Artemis source this was ported from, all deliberate:

* **No `MAX_CONTEXT`.** The original clamps the live length to a `gl.constexpr`
  capacity in both kernels, which truncates the context if the caller passes a
  capacity that is too small, and puts capacity into the compilation key. The
  live length here comes from `kv_indptr` and is never clamped, matching the
  long body. The one place capacity was load-bearing -- proving a page-index
  buffer load fits a 32-bit offset -- is proved host-side and arrives as
  `NARROW_PAGES`.
* **`QUANTIZE`.** The original always writes the FP8 output and its scales. When
  the epilogue is skipped the launcher has no FP8 buffers to pass, so those
  stores must be compiled out rather than aimed at the `gated` buffer the
  sigmoid was staged in.
* **`NARROW_RECORDS`.** The original asserts the split-record offsets fit in 32
  bits host-side and then unconditionally uses unsigned 32-bit addressing for
  the narrow reductions. The proof arrives as a constexpr instead, so an
  oversized launch falls back to 64-bit addressing rather than wrapping.

**Numerics are contract**, identical to the long body: attention -> bf16 ->
fp32, sigmoid -> bf16 -> fp32, product -> bf16, and only then the group absmax.

Launcher: aiter/ops/triton/attention/paged_attention_output_gate.py
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Must be `gl.constexpr(...)`, not a `: gl.constexpr` annotation -- Triton only
# lets a @jit body read globals instantiated the first way.
HEAD_DIM = gl.constexpr(256)
QUANT_GROUP = gl.constexpr(128)


@gluon.jit
def _cache_offsets(
    slots,
    columns,
    ROW_STRIDE: gl.constexpr,
    DIM_STRIDE: gl.constexpr,
    NARROW_OFFSETS: gl.constexpr,
):
    """Element offsets of one gathered K/V tile, in the narrowest legal width."""
    if NARROW_OFFSETS:
        # The host proves the complete element offset fits, including the last
        # channel. Zero-extend only after unsigned offset arithmetic.
        return (
            slots[:, None].to(gl.uint32) * ROW_STRIDE
            + columns.to(gl.uint32) * DIM_STRIDE
        ).to(gl.int64)
    return slots[:, None].to(gl.int64) * ROW_STRIDE + columns.to(gl.int64) * DIM_STRIDE


@gluon.jit
def _load_page_indices(
    indices_ptr, begin, positions, length, NARROW_PAGES: gl.constexpr
):
    """Page slots for one tile, clamped to the live length rather than masked."""
    offsets = gl.minimum(positions, length - 1)
    if NARROW_PAGES:
        # Relative offsets cover only one context, not the entire page table.
        # The host proves that this byte range fits a signed buffer offset.
        return gl.amd.cdna4.buffer_load(
            indices_ptr + begin.to(gl.int64), offsets.to(gl.int32)
        )
    return gl.load(indices_ptr + begin + offsets)


@gluon.jit
def _prefetch_cache(
    cache_smem,
    cache_ptr,
    page_slots,
    tokens,
    dims,
    length,
    ROW_STRIDE: gl.constexpr,
    DIM_STRIDE: gl.constexpr,
    NARROW_OFFSETS: gl.constexpr,
    IS_VALUE: gl.constexpr,
    VALUE_CACHE_POLICY: gl.constexpr,
):
    """Issue one asynchronous gather of a BLOCK-token K or V tile into LDS.

    Writes pre-swizzled values through a linear alias; the MFMA consumers use
    the original descriptor to recover the logical tile. XOR preserves each
    aligned 16-element contiguous group.
    """
    vector: gl.constexpr = 16
    base = ((dims[None, :] // vector) ^ (tokens[:, None] & 15)) * vector
    columns = gl.multiple_of(base, [1, vector]) + dims[None, :] % vector
    columns = gl.max_contiguous(columns, [1, vector])
    linear: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    copy_smem = cache_smem._reinterpret(layout=linear)
    ptrs = cache_ptr + _cache_offsets(
        page_slots, columns, ROW_STRIDE, DIM_STRIDE, NARROW_OFFSETS
    )
    if IS_VALUE:
        if VALUE_CACHE_POLICY >= 16:
            # Only stream V when the working set is too big to stay resident
            # but small enough that the gather is still the bottleneck.
            working_tokens = length * VALUE_CACHE_POLICY
            stream_value = (working_tokens > 32768) & (working_tokens <= 131072)
        else:
            stream_value = VALUE_CACHE_POLICY == 1
        if stream_value:
            gl.amd.cdna4.async_copy.global_load_to_shared(
                copy_smem, ptrs, cache_modifier=".cg"
            )
        else:
            gl.amd.cdna4.async_copy.global_load_to_shared(copy_smem, ptrs)
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(copy_smem, ptrs)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _sigmoid_bf16(value):
    """The gate's activation, rounded at the boundary the contract specifies."""
    return (1.0 / (1.0 + gl.exp(-value.to(gl.float32)))).to(gl.bfloat16)


@gluon.jit
def _quant_scale(absmax, FP8_MAX: gl.constexpr):
    """`absmax / FP8_MAX`, correctly rounded, then floored away from zero."""
    # absmax is a BF16 value widened to FP32. Compensate the constant reciprocal
    # product to preserve its correctly rounded FP32 quotient.
    inverse_max = gl.div_rn(1.0, FP8_MAX)
    quotient = absmax * inverse_max
    remainder = gl.fma(-quotient, FP8_MAX, absmax)
    quotient = gl.fma(remainder, inverse_max, quotient)
    quotient = gl.where(absmax == float("inf"), absmax, quotient)
    return gl.maximum(quotient, 1.0e-10)


@gluon.jit
def _split_record(
    row,
    head,
    part,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    SPLIT_MAJOR: gl.constexpr,
):
    """Index of one (row, head, split) scratch record, in the chosen order."""
    if SPLIT_MAJOR:
        return (row * SPLITS + part) * HEADS + head
    return (row * HEADS + head) * SPLITS + part


@gluon.jit
def _head_softmax(
    score_smem,
    maximum,
    denominator,
    start,
    length,
    key_scale,
    scale,
    HEADS: gl.constexpr,
    BLOCK: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    """One online-softmax step over a BLOCK-token tile of scores.

    Each head owner combines the four disjoint QK channel reductions the
    producer waves left in LDS, then updates the running maximum, denominator
    and correction.
    """
    s0 = gl.amd.cdna4.async_copy.load_shared_relaxed(score_smem.slice(0, HEADS), LAYOUT)
    s1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        score_smem.slice(HEADS, HEADS), LAYOUT
    )
    s2 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        score_smem.slice(2 * HEADS, HEADS), LAYOUT
    )
    s3 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        score_smem.slice(3 * HEADS, HEADS), LAYOUT
    )
    scores = (s0 + s1) + (s2 + s3)
    scores = (scores * key_scale) * scale
    positions = start + gl.arange(0, BLOCK, layout=gl.SliceLayout(0, LAYOUT))
    valid = positions[None, :] < length
    scores = gl.where(valid, scores, -float("inf"))
    next_maximum = gl.maximum(maximum, gl.max(scores, 1))
    correction = gl.exp(maximum - next_maximum)
    weights = gl.where(valid, gl.exp(scores - next_maximum[:, None]), 0.0)
    denominator = denominator * correction + gl.sum(weights, 1)
    return next_maximum, denominator, correction, weights


_partials_repr = make_kernel_repr(
    "_paged_attention_output_gate_short_partials",
    ["HEADS", "SPLITS", "ROWS", "DEDICATED_GATE", "SPLIT_MAJOR"],
)


@gluon.jit(repr=_partials_repr)
def _short_attention_partials(
    query_ptr,
    key_ptr,
    value_ptr,
    indptr_ptr,
    indices_ptr,
    key_scale_ptr,
    partial_ptr,
    stats_ptr,
    gate_ptr,
    gated_ptr,
    scale,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    QUERY_ROW_STRIDE: gl.constexpr,
    QUERY_HEAD_STRIDE: gl.constexpr,
    QUERY_DIM_STRIDE: gl.constexpr,
    KEY_ROW_STRIDE: gl.constexpr,
    KEY_DIM_STRIDE: gl.constexpr,
    VALUE_ROW_STRIDE: gl.constexpr,
    VALUE_DIM_STRIDE: gl.constexpr,
    KEY_NARROW: gl.constexpr,
    VALUE_NARROW: gl.constexpr,
    NARROW_PAGES: gl.constexpr,
    USE_SCALES: gl.constexpr,
    DEDICATED_GATE: gl.constexpr,
    GATE_ROW_STRIDE: gl.constexpr,
    GATE_DIM_STRIDE: gl.constexpr,
    ROWS: gl.constexpr,
    SPLIT_MAJOR: gl.constexpr,
):
    """grid = (tokens, SPLITS + DEDICATED_GATE). One split, plus the gate CTA."""
    row = gl.program_id(0)
    # Put the dedicated gate CTA first for the four-row producer grid.
    first_gate: gl.constexpr = DEDICATED_GATE and ROWS == 4
    part = gl.program_id(1) - (1 if first_gate else 0)
    begin = gl.load(indptr_ptr + row)
    # The live length, never clamped: one binary serves any context.
    length = gl.load(indptr_ptr + row + 1) - begin
    first = part * BLOCK

    # Every row stages its sigmoid, including empty rows. Batch 32 shares
    # partition zero; the other shapes use an independent gate CTA. The staging
    # buffer is the `gated` output, which `_finish` overwrites afterwards.
    gate_part: gl.constexpr = -1 if first_gate else SPLITS if DEDICATED_GATE else 0
    if part == gate_part:
        gate_layout: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
        channel = gl.arange(0, HEADS * HEAD_DIM, layout=gate_layout)
        raw_gate = gl.load(
            gate_ptr + row * GATE_ROW_STRIDE + channel * GATE_DIM_STRIDE
        ).to(gl.float32)
        gl.store(gated_ptr + row * HEADS * HEAD_DIM + channel, _sigmoid_bf16(raw_gate))

    active = first < length
    if DEDICATED_GATE:
        if first_gate:
            active = active & (part >= 0)
        else:
            active = active & (part < SPLITS)

    # Inactive records are never read by the finish kernel. In particular, a
    # shrinking graph replay does not need to clear previously active records.
    if active:
        single_buffer: gl.constexpr = ROWS >= 4
        staggered: gl.constexpr = not single_buffer
        two_step_pages: gl.constexpr = ROWS == 32
        buffer_pages: gl.constexpr = ROWS == 8 and NARROW_PAGES
        # Small alternating tiles and the largest batch stream V directly;
        # intermediate batches retain their working-set-dependent policy.
        value_cache_policy: gl.constexpr = (
            1 if ROWS >= 64 or (ROWS >= 8 and ROWS < 16) else ROWS if ROWS >= 16 else 0
        )
        vector: gl.constexpr = 16
        lanes_d: gl.constexpr = 16
        load_layout: gl.constexpr = gl.BlockedLayout(
            [1, vector], [64 // lanes_d, lanes_d], [4, 1], [1, 0]
        )
        # Native four-row instructions avoid padding four query heads to 16.
        # QK uses the batch axis to give each wave 64 of the 256 channels.
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[4, 64, 64],
            transposed=False,
            warps_per_cta=[1, 4],
        )
        qk_mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[4, 64, 64],
            transposed=False,
            warps_per_cta=[4, 1, 1],
        )
        qk_a: gl.constexpr = gl.DotOperandLayout(0, qk_mma, 4)
        qk_b: gl.constexpr = gl.DotOperandLayout(1, qk_mma, 4)
        qk_scores: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        qk_head: gl.constexpr = gl.SliceLayout(1, qk_scores)
        dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 4)
        dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 4)
        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(vector, 1, 16, [1, 0])
        dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, load_layout))
        tokens = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, load_layout))
        # Each query element belongs to just one wave; load directly into its
        # dot-operand registers without an initial shared-memory exchange.
        qb = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, qk_a)))
        qh = gl.arange(0, HEADS, layout=gl.SliceLayout(0, gl.SliceLayout(2, qk_a)))
        qd = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, qk_a)))
        query = gl.load(
            query_ptr
            + row * QUERY_ROW_STRIDE
            + qh[None, :, None] * QUERY_HEAD_STRIDE
            + (qb[:, None, None] * 64 + qd[None, None, :]) * QUERY_DIM_STRIDE
        )
        # The Artemis original requires both dequant scales; make them optional
        # the way the long body does, so `key_scale_ptr` need not be real.
        key_scale = 1.0
        if USE_SCALES:
            key_scale = gl.load(key_scale_ptr)

        key_smem = gl.allocate_shared_memory(
            key_ptr.dtype.element_ty, (BLOCK, HEAD_DIM), shared_layout
        )
        if single_buffer:
            # Deliberate aliasing: the loop fences every K/V buffer reuse.
            value_smem = key_smem
        else:
            value_smem = gl.allocate_shared_memory(
                value_ptr.dtype.element_ty, (BLOCK, HEAD_DIM), shared_layout
            )
        score_smem = gl.allocate_shared_memory(
            gl.float32,
            (4 * HEADS, BLOCK),
            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        )
        probability_smem = gl.allocate_shared_memory(
            gl.float16,
            (HEADS, BLOCK),
            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        )
        correction_smem = gl.allocate_shared_memory(
            gl.float32,
            (HEADS,),
            gl.SwizzledSharedLayout(1, 1, 1, [0]),
        )
        maximum = gl.full((HEADS,), -float("inf"), gl.float32, qk_head)
        denominator = gl.full((HEADS,), 0.0, gl.float32, qk_head)
        numerator = gl.full((HEADS, HEAD_DIM), 0.0, gl.float32, mma)

        # FP16 preserves enough of the attention probabilities' range here; the
        # scores have already been shifted by the running maximum.
        pv_dtype: gl.constexpr = gl.float16
        # Start K first. Separate tiny-batch tiles permit the V transfer to
        # overlap QK, and successor K to overlap the current softmax/PV.
        page_slots = _load_page_indices(
            indices_ptr, begin, first + tokens, length, buffer_pages
        )
        _prefetch_cache(
            key_smem,
            key_ptr,
            page_slots,
            tokens,
            dims,
            length,
            KEY_ROW_STRIDE,
            KEY_DIM_STRIDE,
            KEY_NARROW,
            False,
            value_cache_policy,
        )

        if two_step_pages:
            # Only fetch a successor that belongs to this live row. The same
            # graph can grow to capacity, shrink, or replay with empty rows.
            next_pages = page_slots
            if first + SPLITS * BLOCK < length:
                next_pages = _load_page_indices(
                    indices_ptr,
                    begin,
                    first + SPLITS * BLOCK + tokens,
                    length,
                    buffer_pages,
                )

        for start in range(first, length, SPLITS * BLOCK):
            gl.amd.cdna4.async_copy.wait_group(0)
            # Complete every wave's copies before any cross-wave K reads.
            gl.barrier()
            key = gl.amd.cdna4.async_copy.load_shared_relaxed(
                key_smem.reshape((BLOCK, 4, 64)).permute((1, 2, 0)), qk_b
            ).to(gl.bfloat16)
            if two_step_pages:
                next_start = start + SPLITS * BLOCK
                future_pages = next_pages
                if next_start + SPLITS * BLOCK < length:
                    future_pages = _load_page_indices(
                        indices_ptr,
                        begin,
                        next_start + SPLITS * BLOCK + tokens,
                        length,
                        buffer_pages,
                    )

            if single_buffer:
                # Complete all K readers before V overwrites the same LDS.
                gl.barrier()
            _prefetch_cache(
                value_smem,
                value_ptr,
                page_slots,
                tokens,
                dims,
                length,
                VALUE_ROW_STRIDE,
                VALUE_DIM_STRIDE,
                VALUE_NARROW,
                True,
                value_cache_policy,
            )

            if not two_step_pages:
                next_start = start + SPLITS * BLOCK
                next_pages = page_slots
                if next_start < length:
                    next_pages = _load_page_indices(
                        indices_ptr, begin, next_start + tokens, length, buffer_pages
                    )

            score_partials = gl.amd.cdna4.mfma(
                query, key, gl.full((4, HEADS, BLOCK), 0.0, gl.float32, qk_mma)
            )
            score_smem.store(score_partials.reshape((4 * HEADS, BLOCK)))
            gl.barrier()
            if staggered and next_start < length:
                # Score publication fences K readers before successor reuse.
                page_slots = next_pages
                _prefetch_cache(
                    key_smem,
                    key_ptr,
                    page_slots,
                    tokens,
                    dims,
                    length,
                    KEY_ROW_STRIDE,
                    KEY_DIM_STRIDE,
                    KEY_NARROW,
                    False,
                    value_cache_policy,
                )

            next_maximum, denominator, correction, weights = _head_softmax(
                score_smem,
                maximum,
                denominator,
                start,
                length,
                key_scale,
                scale,
                HEADS,
                BLOCK,
                qk_scores,
            )
            probability_smem.store(weights.to(pv_dtype))
            correction_smem.store(correction)
            if staggered:
                # The V transfer precedes successor K in the async queue.
                if next_start < length:
                    gl.amd.cdna4.async_copy.wait_group(1)
                else:
                    gl.amd.cdna4.async_copy.wait_group(0)
            elif single_buffer:
                gl.amd.cdna4.async_copy.wait_group(0)
            # Publish probabilities/corrections and, for alternating staging,
            # all V copies. Prior score readers are also complete at this fence.
            gl.barrier()
            value = gl.amd.cdna4.async_copy.load_shared_relaxed(value_smem, dot_b).to(
                pv_dtype
            )
            pv_correction = gl.amd.cdna4.async_copy.load_shared_relaxed(
                correction_smem, gl.SliceLayout(1, mma)
            )
            if not single_buffer:
                numerator = numerator * pv_correction[:, None]
            weights = gl.amd.cdna4.async_copy.load_shared_relaxed(
                probability_smem, dot_a
            )
            if single_buffer and next_start < length:
                # Complete every V reader before the next K copy starts.
                gl.barrier()
                page_slots = next_pages
                if two_step_pages:
                    next_pages = future_pages
                _prefetch_cache(
                    key_smem,
                    key_ptr,
                    page_slots,
                    tokens,
                    dims,
                    length,
                    KEY_ROW_STRIDE,
                    KEY_DIM_STRIDE,
                    KEY_NARROW,
                    False,
                    value_cache_policy,
                )
            if single_buffer:
                numerator = numerator * pv_correction[:, None]
            numerator = gl.amd.cdna4.mfma(weights, value, numerator)
            maximum = next_maximum

        out_head = gl.arange(0, HEADS, layout=gl.SliceLayout(1, mma))
        out_dim = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, mma))
        record = _split_record(row, out_head, part, HEADS, SPLITS, SPLIT_MAJOR)
        gl.store(partial_ptr + record[:, None] * HEAD_DIM + out_dim[None, :], numerator)
        # Each head owner emits its FP32 maximum/denominator pair in one
        # explicitly vectorized store, without a cross-wave redistribution.
        stats_layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, 1], [1, 0])
        stats_head_layout: gl.constexpr = gl.SliceLayout(1, stats_layout)
        stats_head = gl.arange(0, HEADS, layout=stats_head_layout)
        stats_field = gl.arange(0, 2, layout=gl.SliceLayout(0, stats_layout))
        stats_record = _split_record(row, stats_head, part, HEADS, SPLITS, SPLIT_MAJOR)
        stats_pair = gl.join(
            gl.convert_layout(maximum, stats_head_layout),
            gl.convert_layout(denominator, stats_head_layout),
        )
        gl.store(
            stats_ptr + stats_record[:, None] * 2 + stats_field[None, :], stats_pair
        )


@gluon.jit
def _finish_group(
    partial_ptr,
    stats_ptr,
    length,
    value_scale_ptr,
    gated_ptr,
    quantized_ptr,
    scales_ptr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    FP8_MAX: gl.constexpr,
    FINISH_WARPS: gl.constexpr,
    SPLIT_MAJOR: gl.constexpr,
    USE_SCALES: gl.constexpr,
    QUANTIZE: gl.constexpr,
    NARROW_RECORDS: gl.constexpr,
    REDUCE_SPLITS: gl.constexpr,
    ALL_ACTIVE: gl.constexpr = False,
):
    """Reduce REDUCE_SPLITS split records of one 128-wide quant group and emit.

    The group IS the CTA, which is what makes the group absmax a plain
    intra-CTA reduction.
    """
    # Unsigned, combined offsets shorten the small-reduction address chain. The
    # host proves that every partial element offset fits in 32 bits; without
    # that proof this falls back to 64-bit addressing rather than wrapping.
    narrow: gl.constexpr = REDUCE_SPLITS <= 16 and NARROW_RECORDS
    if narrow:
        row = gl.program_id(0).to(gl.uint32)
        group = gl.program_id(1).to(gl.uint32)
        head = group >> 1
        half = group & 1
    else:
        row = gl.program_id(0)
        group = gl.program_id(1)
        head = group // 2
        half = group % 2
    if FINISH_WARPS == 4:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [1, 4], [1, 0])
    elif FINISH_WARPS == 2:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 2], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 4], [2, 32], [FINISH_WARPS, 1], [1, 0]
        )
    s = gl.arange(0, REDUCE_SPLITS, layout=gl.SliceLayout(1, layout))
    d = gl.arange(0, QUANT_GROUP, layout=gl.SliceLayout(0, layout))
    record = _split_record(row, head, s, HEADS, SPLITS, SPLIT_MAJOR)
    if ALL_ACTIVE:
        valid = gl.full((REDUCE_SPLITS,), True, gl.int1, gl.SliceLayout(1, layout))
    else:
        valid = s * BLOCK < length
    channel = group * QUANT_GROUP + d

    # Fetch independent payloads before the softmax reduction dependency chain.
    # `gated_ptr` currently holds the sigmoid the producer staged there.
    output_offset = row * HEADS * HEAD_DIM + channel
    sigmoid = gl.load(gated_ptr + output_offset).to(gl.float32)
    value_scale = 1.0
    if USE_SCALES:
        value_scale = gl.load(value_scale_ptr)
    if (
        REDUCE_SPLITS >= 64
        or (FINISH_WARPS == 2 and REDUCE_SPLITS == 32)
        or SPLITS == 16
    ):
        # Coalesce statistics independently from numerator ownership, then
        # gather corrections into each wave's channel-oriented layout.
        # Numerator reductions remain wave-local; only quantization's group
        # maximum crosses waves.
        stats_layout: gl.constexpr = gl.SliceLayout(
            1, gl.BlockedLayout([1, 1], [64, 1], [1, FINISH_WARPS], [0, 1])
        )
        stats_split = gl.arange(0, REDUCE_SPLITS, layout=stats_layout)
        stats_record = _split_record(row, head, stats_split, HEADS, SPLITS, SPLIT_MAJOR)
        if ALL_ACTIVE:
            stats_valid = gl.full((REDUCE_SPLITS,), True, gl.int1, stats_layout)
        else:
            stats_valid = stats_split * BLOCK < length
        maxima = gl.load(
            stats_ptr + stats_record * 2, mask=stats_valid, other=-float("inf")
        )
        denominators = gl.load(
            stats_ptr + stats_record * 2 + 1, mask=stats_valid, other=0.0
        )
    else:
        stats_valid = valid
        maxima = gl.load(stats_ptr + record * 2, mask=valid, other=-float("inf"))
        denominators = gl.load(stats_ptr + record * 2 + 1, mask=valid, other=0.0)
    if narrow:
        partial_offset = (
            (record[:, None] * HEAD_DIM + half * QUANT_GROUP + d[None, :])
            .to(gl.uint32)
            .to(gl.int64)
        )
        partial_raw = gl.load(
            partial_ptr + partial_offset, mask=valid[:, None], other=0.0
        )
    else:
        partial_raw = gl.load(
            partial_ptr + record[:, None] * HEAD_DIM + half * QUANT_GROUP + d[None, :],
            mask=valid[:, None],
            other=0.0,
        )
    maximum = gl.max(maxima, 0)
    # Emptiness is determined by the live row, not by a floating sentinel.
    maximum = gl.where(length > 0, maximum, 0.0)
    correction = gl.where(stats_valid, gl.exp(maxima - maximum), 0.0)
    denominator = gl.sum(denominators * correction, 0)
    if (
        REDUCE_SPLITS >= 64
        or (FINISH_WARPS == 2 and REDUCE_SPLITS == 32)
        or SPLITS == 16
    ):
        if FINISH_WARPS == 1:
            # Static wave-local redistribution avoids indexed LDS shuffles.
            correction = gl.convert_layout(correction, gl.SliceLayout(1, layout))
        else:
            correction = gl.gather(correction, s, 0)
    numerator = gl.sum(partial_raw.to(gl.float32) * correction[:, None], 0)
    attention = (
        gl.where(denominator > 0, numerator * (value_scale / denominator), 0.0)
        .to(gl.bfloat16)
        .to(gl.float32)
    )

    # Preserve all three BF16 rounding boundaries in the output contract.
    gated = (attention * sigmoid).to(gl.bfloat16).to(gl.float32)
    # The cross-wave absmax is issued before any store, so the `gated` store is
    # not sitting in front of it. Keep it that way.
    if QUANTIZE:
        scale = _quant_scale(gl.max(gl.abs(gated), 0), FP8_MAX)
        # Share one FP32 reciprocal across the entire quantization group.
        inverse_scale = 1.0 / scale
        quantized = gl.clamp(gated * inverse_scale, -FP8_MAX, FP8_MAX)
    gl.store(gated_ptr + output_offset, gated)
    if QUANTIZE:
        # Compiled out when there is no FP8 output. That is what keeps
        # `quantized_ptr`/`scales_ptr` from having to be distinct buffers from
        # the `gated` one the sigmoid was staged in.
        gl.store(quantized_ptr + output_offset, quantized)
        gl.store(scales_ptr + row * HEADS * 2 + group, scale)


_finish_repr = make_kernel_repr(
    "_paged_attention_output_gate_short_finish",
    ["HEADS", "SPLITS", "FINISH_WARPS", "QUANTIZE", "SPLIT_MAJOR"],
)


@gluon.jit(repr=_finish_repr)
def _short_finish(
    partial_ptr,
    stats_ptr,
    indptr_ptr,
    value_scale_ptr,
    gated_ptr,
    quantized_ptr,
    scales_ptr,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    BLOCK: gl.constexpr,
    FP8_MAX: gl.constexpr,
    FINISH_WARPS: gl.constexpr,
    SPLIT_MAJOR: gl.constexpr,
    USE_SCALES: gl.constexpr,
    QUANTIZE: gl.constexpr,
    NARROW_RECORDS: gl.constexpr,
):
    """grid = (tokens, HEADS * 2). One program per 128-wide quant group."""
    # The launch geometry fixes the scratch pitch; the live length bounds this
    # reduction and is reloaded on every changed-input graph replay.
    row = gl.program_id(0)
    length = gl.load(indptr_ptr + row + 1) - gl.load(indptr_ptr + row)
    # Keep tensor arguments separate from constexpr configuration: ordinary
    # Gluon tuples would turn integer configuration fields into device values.
    args = (
        partial_ptr,
        stats_ptr,
        length,
        value_scale_ptr,
        gated_ptr,
        quantized_ptr,
        scales_ptr,
    )
    config: gl.constexpr = (
        HEADS,
        SPLITS,
        BLOCK,
        FP8_MAX,
        FINISH_WARPS,
        SPLIT_MAJOR,
        USE_SCALES,
        QUANTIZE,
        NARROW_RECORDS,
    )
    if length > (SPLITS - 1) * BLOCK:
        _finish_group(*args, *config, REDUCE_SPLITS=SPLITS, ALL_ACTIVE=True)
    elif SPLITS > 8 and length <= 8 * BLOCK:
        _finish_group(*args, *config, REDUCE_SPLITS=8)
    elif SPLITS > 16 and length <= 16 * BLOCK:
        if FINISH_WARPS == 4:
            # A dense prefix contains only fresh records, including its tail.
            if length > 15 * BLOCK:
                _finish_group(*args, *config, REDUCE_SPLITS=16, ALL_ACTIVE=True)
            else:
                _finish_group(*args, *config, REDUCE_SPLITS=16)
        else:
            _finish_group(*args, *config, REDUCE_SPLITS=16)
    elif SPLITS > 32 and length <= 32 * BLOCK:
        _finish_group(*args, *config, REDUCE_SPLITS=32)
    elif SPLITS > 64 and length <= 64 * BLOCK:
        _finish_group(*args, *config, REDUCE_SPLITS=64)
    else:
        _finish_group(*args, *config, REDUCE_SPLITS=SPLITS)
