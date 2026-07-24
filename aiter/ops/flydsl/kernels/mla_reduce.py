# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the MLA decode reduce/combine epilogue.

Faithful port of the HIP kernel ``kn_mla_reduce_v1`` / ``kn_mla_reduce_v1_ps``
(``csrc/kernels/mla/reduce.cu``). Stage-2 of split-KV MLA decode: merges per-split
partial outputs ``O_i`` (fp32) weighted by ``exp(LSE_i - LSE_max)`` (online softmax)
into the final output (bf16/fp16), and optionally the merged LSE.

Layout / contract (matches the HIP kernel):
  partial_output : fp32 [max_partial_row, H, Dv]  contiguous
  partial_lse    : fp32 [max_partial_row, H]       contiguous
  reduce_indptr  : i32  [num_reduce_tile + 1]      CSR over splits
  reduce_partial_map : i32 [reduce_indptr[-1]]     split -> partial row (gather)
  reduce_final_map   : i32 [num_reduce_tile, 2]    {q_start, q_end} (optional)
  final_output   : bf16/fp16 [bs, H, Dv]           runtime strides
  final_lse      : fp32 [bs, H]                    (optional)

Each work item = (head, q-pos-group, reduce-tile). A ``NUM_THREADS``-thread
block owns one (seq, head) output row; thread t owns
``VEC = Dv // NUM_THREADS`` contiguous floats.

Two launch modes (mirrors the HIP kernel):
  * grid-launch (default): 3-D grid (H, NTG, num_reduce_tile), one block per work item.
  * persistent (``persistent=True``): 1-D grid of ``num_cu * PS_GRID_MULT`` blocks that
    grid-stride over the flat work index, matching ``kn_mla_reduce_v1_ps``. This
    collapses the dispatch when ``num_reduce_tile`` is large but few tiles are
    active (sparse serving grids), eliminating launch/dispatch latency.
"""

import enum
import functools
import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, rocdl
from flydsl.expr import math as fly_math
from flydsl.expr.typing import T

_LOG2E = math.log2(math.e)
fm_fast = "fast"

# Matches MlaReduceKernelV1Traits (reduce.cu:13)
# NUM_THREADS may be 64, 128, or 256. It controls the block size and per-thread
# accumulator width; it must divide Dv.
NUM_THREADS = int(os.environ.get("MLA_NUM_THREADS", "256"))
WARP = 64
OCC = 8
MASSIVE_THR = 4  # kMassiveThreshold

# GRP is the power-of-two pipeline depth for the M64 and M256/MLDS tiers.
GRP_M256 = int(os.environ.get("MLA_GRP_M256", "16"))
GRP_M64 = int(os.environ.get("MLA_GRP_M64", "8"))

# A runtime M64 sub-split uses a deeper pipeline above the threshold without
# changing CUDA-graph topology. Set the threshold to zero to disable it.
M64_HI_THR = int(os.environ.get("MLA_M64_HI_THR", "8"))
M64_HI_GRP = int(os.environ.get("MLA_M64_HI_GRP", "16"))
# Persistent grid multiplier; the grid-stride loop covers all work items.
PS_GRID_MULT = 1


def _out_numeric_t(out_dtype: str):
    """Resolve the public FlyDSL numeric output type."""
    if out_dtype in ("bf16", "bfloat16"):
        return fx.BFloat16
    if out_dtype in ("fp16", "f16", "float16", "half"):
        return fx.Float16
    raise ValueError(f"Unsupported out_dtype: {out_dtype}")


def _vector_elements(value, dtype, width: int):
    """Expose a raw vector value through FlyDSL's public vector interface."""
    vector = fx.Vector(value, shape=width, dtype=dtype)
    return [vector[i] for i in fx.range_constexpr(width)]


def _make_vector(elements, dtype):
    """Build a vector with FlyDSL's public value-semantics API."""
    return fx.Vector.from_elements(elements, dtype=dtype)


def _pointer_view(ptr, dtype, shape, stride):
    """Build a typed layout view from an opaque pointer-launch ABI argument."""
    typed_ptr = fx.recast_iter(dtype, ptr)
    return fx.make_view(typed_ptr, fx.make_layout(shape, stride))


def _pointer_buffer_tensor(ptr, dtype, shape, stride):
    """Build a public buffer-backed tensor view from a pointer ABI argument."""
    return fx.rocdl.make_buffer_tensor(_pointer_view(ptr, dtype, shape, stride))


def _pointer_buffer_resource(ptr, dtype, shape, stride):
    """Build a raw buffer descriptor for a documented low-level MLA boundary."""
    return buffer_ops.create_buffer_resource(
        _pointer_view(ptr, dtype, shape, stride), max_size=True
    )


def _exp(x, use_exp2=True):
    """exp(x) via the hardware v_exp_f32 (exp2(x*log2e))."""
    if fx.const_expr(use_exp2):
        return fx.rocdl.exp2(T.f32, (x * _LOG2E).ir_value())
    return fly_math.exp(x, fastmath=fm_fast)


def _log(x):
    """natural log; mlir math dialect lowers to the device log."""
    return fly_math.log(x, fastmath=fm_fast)


def _is_zero_or_nan(value):
    """Preserve the ordered-zero or unordered-NaN reduction invalidity test."""
    value = fx.Float32(value)
    return (value == fx.Float32(0.0)) | fly_math.isnan(value)


# Massive-path sub-tiers (mirror MlaReduceProblemSize, reduce.cu:121).
class Tier(enum.Enum):
    SIMPLE = "simple"  # register online-softmax, num_splits in {2,3}
    M64 = "m64"  # <=64 splits, 1 LSE per lane in warp0
    M256 = "m256"  # <=256 splits, up to 4 LSE per lane in warp0
    MLDS = "mlds"  # >256 splits, 4 in regs + overflow in LDS
    ALL = "all"  # all bodies; device-side runtime branch on n_splits (production)


# LSE registers per warp0 lane (None = LDS-backed overflow).
_TIER_NLSE = {
    Tier.SIMPLE: 0,
    Tier.M64: 1,
    Tier.M256: 4,
    Tier.MLDS: None,
    Tier.ALL: None,
}
LDS_MAX_SPLITS = 304  # >= MI300X CU count; compile-time LDS cap
# Pad LDS tails so masked vector reads stay within one allocation.
LDS_PADDED_SPLITS = ((LDS_MAX_SPLITS + WARP - 1) // WARP) * WARP
NLSE_MLDS = (LDS_MAX_SPLITS + WARP - 1) // WARP  # ceil(304/64) = 5

# Production occupancy hint.
_DEFAULT_WAVES_PER_EU = 4


def select_tier(num_splits: int) -> Tier:
    if num_splits < MASSIVE_THR:
        return Tier.SIMPLE
    if num_splits <= 64:
        return Tier.M64
    if num_splits <= 256:
        return Tier.M256
    return Tier.MLDS


def should_use_persistent_launch(
    *,
    H: int,
    max_seqlen_q: int,
    num_reduce_tile: int,
    num_cu: int,
) -> bool:
    """Use HIP's work threshold to select grid or persistent dispatch."""
    tot_work = H * max_seqlen_q * num_reduce_tile
    ps_grid_size = num_cu * OCC * 2
    return tot_work > ps_grid_size


@functools.lru_cache(maxsize=256)
def compile_mla_reduce(
    *,
    H: int,
    Dv: int,
    out_dtype: str = "bf16",
    tier: Tier = Tier.SIMPLE,
    persistent: bool = False,
    output_lse: bool = False,
    use_reduce_final_map: bool = True,
    waves_per_eu: int = _DEFAULT_WAVES_PER_EU,
    use_exp2: bool = True,
    disable_guards: bool = False,
    adaptive: bool = False,
    low_direct_pmap_thr: int = 0,
):
    """Compile an MLA reduce kernel for fixed (H, Dv, out_dtype, tier).

    tier selects the per-work-item algorithm. ``Tier.ALL`` (production) emits
    every body and branches on device ``n_splits`` per tile (mirrors HIP).
    Other tiers compile a single body for isolated tests. ``persistent`` selects
    the grid-stride launch (kn_mla_reduce_v1_ps); otherwise a 3-D grid launch.

    ``adaptive`` (mutually exclusive with ``persistent``) emits the direct 3-D
    body but the launcher sizes grid-z to ``num_final_rows`` (the host-known
    active-tile count = decode batch), NOT ``num_reduce_tile``. This avoids the
    persistent grid-stride loop, CSR-sentinel traversal, and LDS-reuse barrier.
    Correct because ``process_work_item`` self-guards empty/OOB tiles, and
    active tiles are the CSR prefix ``[0, num_final_rows)`` on the decode path.

    disable_guards: test-only compile-time knob that skips gather/store bounds
    guards so the suite can run a pre-fix kernel in-process. The production
    wrapper (mla_reduce_kernels.py) never threads this (defaults False).
    """
    assert (
        Dv % NUM_THREADS == 0
    ), f"Dv ({Dv}) must be divisible by NUM_THREADS ({NUM_THREADS})"
    assert tier in _TIER_NLSE, f"bad tier {tier}"
    VEC = Dv // NUM_THREADS
    is_runtime_tier = tier == Tier.ALL
    is_massive = tier != Tier.SIMPLE
    if tier == Tier.MLDS:
        NLSE = NLSE_MLDS
    elif tier == Tier.ALL:
        NLSE = 0  # unused; runtime dispatch picks NLSE per branch
    else:
        NLSE = _TIER_NLSE[tier]

    kernel_value_attrs = (
        {"rocdl.waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )
    # WPE must be in compile hints because the JIT cache ignores value attrs.
    kernel_compile_hints = (
        {"waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )

    # Separate 16-byte aligned fields preserve the wide LDS load contract.
    @fx.struct
    class SharedStorage:
        pmap: fx.Array[fx.Int32, LDS_PADDED_SPLITS, 16]

        if is_massive:
            lse_scale: fx.Array[fx.Float32, LDS_PADDED_SPLITS, 16]

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def mla_reduce_kernel(
        partial_output: fx.Pointer,  # fp32 [row, H, Dv]
        partial_lse: fx.Pointer,  # fp32 [row, H]
        reduce_indptr: fx.Pointer,  # i32  [tiles+1]
        reduce_partial_map: fx.Pointer,  # i32  [nsplits_total]
        reduce_final_map: fx.Pointer,  # i32  [tiles, 2]
        final_output: fx.Pointer,  # out  [bs, H, Dv]
        final_lse: fx.Pointer,  # fp32 [bs, H]
        stride_s_o: fx.Int32,  # final_output row stride (elements)
        stride_h_o: fx.Int32,  # final_output head stride (elements)
        max_splits: fx.Int32,  # = num_cu (LSE distribution stride)
        num_reduce_tile: fx.Int32,  # CSR sentinel at indptr[num_reduce_tile]
        num_partial_rows: fx.Int32,  # partial_output row count (gather bounds)
        num_final_rows: fx.Int32,  # final_output row count (store q-range)
        num_thread_group: fx.Int32,  # NTG (persistent unflatten / seq stride)
    ):
        out_numeric_t = _out_numeric_t(out_dtype)
        c_VEC = VEC

        def load_o_elems(row_idx, head_idx, col_idx):
            """Load VEC fp32 elements as a python list of scalars."""
            offset = row_idx * (H * Dv) + head_idx * Dv + col_idx
            if fx.const_expr(VEC == 1):
                return [
                    fx.Float32(
                        buffer_ops.buffer_load(
                            partial_output_rsrc, offset, vec_width=1, dtype=T.f32
                        )
                    )
                ]
            v = buffer_ops.buffer_load(
                partial_output_rsrc, offset, vec_width=VEC, dtype=T.f32
            )
            return _vector_elements(v, fx.Float32, VEC)

        def store_o_elems(off_idx, elems_f32):
            """Cast VEC fp32 scalars to out_t and emit one vector buffer store.

            Truncate each element to out_t first, then pack into the out_t vector
            so the store lowers to buffer_store_dwordx2/4 instead of a per-element
            (scalarized) buffer_store_short.
            """
            if fx.const_expr(VEC == 1):
                buffer_ops.buffer_store(
                    elems_f32[0].to(out_numeric_t), final_output_rsrc, off_idx
                )
                return
            out_vec = _make_vector(
                [elems_f32[i].to(out_numeric_t) for i in fx.range_constexpr(VEC)],
                out_numeric_t,
            )
            buffer_ops.buffer_store(out_vec, final_output_rsrc, off_idx)

        tid = fx.thread_idx.x
        col = tid * c_VEC
        lane = tid % WARP
        wave = tid // WARP

        # FlyDSL 0.2.4 boundary: the runtime gather needs a VEC-wide buffer
        # load; the view/copy API cannot preserve this buffer ISA for VEC=1..8.
        partial_output_rsrc = _pointer_buffer_resource(
            partial_output,
            fx.Float32,
            (num_partial_rows, H, Dv),
            (H * Dv, Dv, 1),
        )
        # FlyDSL 0.2.4 boundary: runtime output strides require a packed buffer
        # store; a layout copy would scalarize the bf16/fp16 vector write.
        final_output_rsrc = _pointer_buffer_resource(
            final_output,
            out_numeric_t,
            (num_final_rows, H, Dv),
            (stride_s_o, stride_h_o, 1),
        )
        g_pl = _pointer_buffer_tensor(
            partial_lse,
            fx.Float32,
            (num_partial_rows, H),
            (H, 1),
        )
        g_indptr = _pointer_buffer_tensor(
            reduce_indptr,
            fx.Int32,
            (num_reduce_tile + 1,),
            (1,),
        )
        g_pmap = _pointer_buffer_tensor(
            reduce_partial_map,
            fx.Int32,
            (num_reduce_tile * max_splits,),
            (1,),
        )
        g_fmap = _pointer_buffer_tensor(
            reduce_final_map,
            fx.Int32,
            (num_reduce_tile, 2),
            (2, 1),
        )
        g_flse = _pointer_buffer_tensor(
            final_lse,
            fx.Float32,
            (num_final_rows, H),
            (H, 1),
        )

        # FlyDSL 0.2.4 does not expose the descriptor behind a buffer tensor.
        # Keep this resource only for wave-uniform s_buffer_load CSR sentinels.
        indptr_rsrc = _pointer_buffer_resource(
            reduce_indptr,
            fx.Int32,
            (num_reduce_tile + 1,),
            (1,),
        )

        def _scalar_indptr(idx):
            """Load a wave-uniform CSR pointer through the scalar memory path."""
            val = buffer_ops.buffer_load(
                indptr_rsrc, idx, vec_width=1, dtype=T.i32, is_scalar=True
            )
            return fx.Int32(rocdl.readfirstlane(T.i32, val))

        c_H = fx.Int32(H)
        last = _scalar_indptr(num_reduce_tile)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_pmap = lds.pmap
        if fx.const_expr(is_massive):
            lds_scale = lds.lse_scale

        def store_lse_scale(split_idx, value):
            lds_scale[split_idx] = value

        def store_lse_value(lse, row, head_idx, value):
            lse[row, head_idx] = value

        def process_work_item(head, block_idx, tile, ntg):
            """Reduce one (head, q-pos-group, tile) work item into final_output.

            Self-guards: tiles at the CSR sentinel, n_splits<=1, or out-of-range
            q-start run zero seq iterations (ub_seq clamp) and never touch the
            gather/store paths, so callers may invoke this for any work index.
            """
            t0 = g_indptr[tile]
            t1 = g_indptr[tile + 1]
            n_splits = t1 - t0

            def stage_pmap():
                """Stage reduce_partial_map[t0:t1] to LDS once per work item.

                This is the normal high-split path (mirrors reduce.cu:431-438).
                The direct path bypasses the staging barrier for low split counts.
                """
                for split_i in range(tid, n_splits, fx.Int32(NUM_THREADS), init=None):
                    split_i32 = fx.Int32(split_i)
                    lds_pmap[split_i32] = g_pmap[t0 + split_i32]
                fx.gpu.barrier()

            # Collapse the sequence bound so inactive tiles never gather or store.
            has_work = (n_splits > fx.Int32(1)) & (t0 != last)

            if fx.const_expr(use_reduce_final_map):
                q_start = g_fmap[tile, 0]
                q_end = g_fmap[tile, 1]
            else:
                stage_pmap()
                row0_idx0 = lds_pmap[0]
                row1_idx0 = lds_pmap[1]
                qo_len = row1_idx0 - row0_idx0
                q_start = tile * qo_len
                q_end = (tile + 1) * qo_len

            if fx.const_expr(not disable_guards):
                q_valid = (q_start >= fx.Int32(0)) & (q_start < num_final_rows)
                has_work = has_work & q_valid
                q_end_oob = q_end > num_final_rows
                q_end = q_end_oob.select(num_final_rows, q_end)

            seq0 = q_start + block_idx
            ub_seq = has_work.select(q_end, seq0)

            def row_from_pmap(pmap_i32, local_seq):
                """Bounds-guarded row index from an already-loaded pmap value.

                Shared by the scalar gather_row (LDS read per split) and the
                vectorized group loader (one ds_read_b128 per group), so both
                paths apply the identical clamp/mask.
                """
                row_i32 = pmap_i32 + local_seq
                if fx.const_expr(not disable_guards):
                    in_bounds = (row_i32 >= fx.Int32(0)) & (row_i32 < num_partial_rows)
                    safe_row_i32 = in_bounds.select(row_i32, fx.Int32(0))
                else:
                    in_bounds = fx.Int32(0) == fx.Int32(0)
                    safe_row_i32 = row_i32
                return safe_row_i32, in_bounds

            def pmap_value(split_i32, direct_pmap: bool = False):
                if fx.const_expr(direct_pmap):
                    # Clamp the absolute pmap load address for masked tail slots.
                    # The contribution is later zeroed by the split-valid guard.
                    in_split = split_i32 < n_splits
                    safe_split = in_split.select(split_i32, fx.Int32(0))
                    return g_pmap[t0 + safe_split]
                return lds_pmap[split_i32]

            def gather_row(split_i32, local_seq, direct_pmap: bool = False):
                pmap = pmap_value(split_i32, direct_pmap)
                return row_from_pmap(pmap, local_seq)

            def load_split_o(split_i32, local_seq, direct_pmap: bool = False):
                row_idx, in_bounds = gather_row(split_i32, local_seq, direct_pmap)
                loaded = load_o_elems(row_idx, head, col)
                zero = fx.Float32(0.0)
                return [in_bounds.select(v, zero) for v in loaded]

            def load_split_lse(split_i32, local_seq, direct_pmap: bool = False):
                row_idx, in_bounds = gather_row(split_i32, local_seq, direct_pmap)
                lse = g_pl[row_idx, head]
                neg_inf = fx.Float32(float("-inf"))
                return in_bounds.select(lse, neg_inf)

            def store_result(seq, out_elems):
                store_off = seq * stride_s_o + head * stride_h_o + tid * c_VEC
                store_o_elems(store_off, out_elems)

            def store_lse(seq, max_lse, sum_e):
                if fx.const_expr(output_lse):
                    bad = _is_zero_or_nan(sum_e)
                    lse_val = _log(sum_e) + max_lse
                    inf = fx.Float32(float("inf"))
                    final_lse_val = bad.select(inf, lse_val)
                    if tid == fx.Int32(0):
                        store_lse_value(g_flse, seq, head, final_lse_val)

            # Runtime range without carried state lets scheduling overlap the
            # split-loop VMEM loads with compute.
            def emit_simple_body(seq_i32, local_seq, direct_pmap: bool = False):
                o0 = load_split_o(fx.Int32(0), local_seq, direct_pmap)
                lse0 = load_split_lse(fx.Int32(0), local_seq, direct_pmap)

                init = [o0[i].ir_value() for i in fx.range_constexpr(VEC)]
                init += [
                    lse0.ir_value(),
                    fx.Float32(1.0).ir_value(),
                ]

                results = init
                for s, state in range(fx.Int32(1), n_splits, fx.Int32(1), init=init):
                    regs = [state[i] for i in fx.range_constexpr(VEC)]
                    max_lse = state[VEC]
                    sum_e = state[VEC + 1]
                    os = load_split_o(fx.Int32(s), local_seq, direct_pmap)
                    lse = load_split_lse(fx.Int32(s), local_seq, direct_pmap)
                    new_max = fx.Float32(max_lse).maximumf(lse)
                    old = _exp(fx.Float32(max_lse) - new_max, use_exp2)
                    new = _exp(lse - new_max, use_exp2)
                    new_regs = [
                        (fx.Float32(regs[i]) * old + os[i] * new).ir_value()
                        for i in fx.range_constexpr(VEC)
                    ]
                    results = yield new_regs + [
                        new_max.ir_value(),
                        (fx.Float32(sum_e) * old + new).ir_value(),
                    ]

                regs = [results[i] for i in fx.range_constexpr(VEC)]
                max_lse = results[VEC]
                sum_e = results[VEC + 1]
                inv = fx.rocdl.rcp(T.f32, fx.Float32(sum_e).ir_value())
                out_elems = [fx.Float32(regs[i]) * inv for i in fx.range_constexpr(VEC)]
                store_result(seq_i32, out_elems)
                store_lse(seq_i32, max_lse, sum_e)

            def emit_massive_body(
                seq_i32,
                local_seq,
                nlse: int,
                grp_override=None,
                direct_pmap: bool = False,
            ):
                """Warp0 LSE reduce -> lds_scale -> barrier -> accumulate.

                ``grp_override`` forces the accumulate GRP (loads-in-flight)
                independent of the nlse-derived default, so the runtime M64
                sub-split can give high-split M64 tiles a deeper pipeline
                (GRP=16) while low-split tiles keep GRP=8 (fewer wasted masked
                lanes) -- a per-shape choice the single GRP_M64 constant can't
                make.
                """
                neg_inf = fx.Float32(float("-inf"))
                zero_f = fx.Float32(0.0)
                if wave == fx.Int32(0):
                    local_lses = []
                    max_lse = neg_inf
                    for j in fx.range_constexpr(nlse):
                        split_idx = lane + fx.Int32(j * WARP)
                        in_rng = split_idx < n_splits
                        safe = in_rng.select(split_idx, fx.Int32(0))
                        lse_j = load_split_lse(safe, local_seq, direct_pmap)
                        lse_j = in_rng.select(lse_j, neg_inf)
                        local_lses.append(lse_j)
                        max_lse = fx.Float32(max_lse).maximumf(lse_j)
                    for off in [32, 16, 8, 4, 2, 1]:
                        peer = fx.Float32(max_lse).shuffle_xor(
                            fx.Int32(off), fx.Int32(WARP)
                        )
                        max_lse = fx.Float32(max_lse).maximumf(peer)
                    sum_e = fx.Float32(0.0)
                    for j in fx.range_constexpr(nlse):
                        sum_e = sum_e + _exp(local_lses[j] - max_lse, use_exp2)
                    for off in [32, 16, 8, 4, 2, 1]:
                        peer = fx.Float32(sum_e).shuffle_xor(
                            fx.Int32(off), fx.Int32(WARP)
                        )
                        sum_e = sum_e + peer
                    bad = _is_zero_or_nan(sum_e)
                    inf = fx.Float32(float("inf"))
                    global_lse = bad.select(inf, _log(sum_e) + max_lse)
                    for j in fx.range_constexpr(nlse):
                        split_idx = lane + fx.Int32(j * WARP)
                        in_rng = split_idx < n_splits
                        sc = _exp(local_lses[j] - global_lse, use_exp2)
                        store_lse_scale(split_idx, in_rng.select(sc, zero_f))
                    if fx.const_expr(output_lse) and lane == fx.Int32(0):
                        store_lse_value(g_flse, seq_i32, head, global_lse)

                # Keep GRP output loads in flight while computing the prior group.
                # Tail gathers use slot zero and the scale select zeros invalid
                # contributions, so no OOB row reaches the buffer descriptor.
                if fx.const_expr(grp_override is not None):
                    GRP = grp_override
                else:
                    GRP = GRP_M256 if nlse >= 4 else GRP_M64
                _grp_shift = GRP.bit_length() - 1

                def load_os_group(base_i32):
                    """Gather rows and issue output loads before the scale barrier.

                    Vectorized LDS: base = i*GRP is 16B-aligned so the pmap read
                    is a wide ds_read_b128. Returns (os_list, valids); the scale
                    fold (which also absorbs the OOB guard) happens in
                    load_scales.

                    Tail safety: OOB lanes (slot >= n_splits) read stale slots, so
                    the pmap value is substituted with slot-0's (always staged,
                    since the massive body only runs when n_splits > 1) and
                    row-clamped -- no stale value ever reaches the gather address,
                    even with guards disabled.
                    """
                    if fx.const_expr(direct_pmap):
                        pmap0 = g_pmap[t0]
                    else:
                        # `Array.view` vector access does not retain the required
                        # ds_read_b128 lowering in FlyDSL 0.2.4.
                        pmap_v = fx.ptr_load(
                            fx.add_offset(lds_pmap.ptr, base_i32),
                            T.vec(GRP, T.i32),
                        )
                        pmap_elements = _vector_elements(pmap_v, fx.Int32, GRP)
                        pmap0 = pmap_elements[0]
                    os_list = []
                    valids = []
                    for j in fx.range_constexpr(GRP):
                        split_j = base_i32 + j
                        in_split = split_j < n_splits
                        if fx.const_expr(direct_pmap):
                            safe_split = in_split.select(split_j, fx.Int32(0))
                            pmap_raw = g_pmap[t0 + safe_split]
                        else:
                            pmap_raw = pmap_elements[j]
                        pmap_j = in_split.select(pmap_raw, pmap0)
                        row_idx, in_bounds = row_from_pmap(pmap_j, local_seq)
                        os_raw = load_o_elems(row_idx, head, col)
                        valid = in_split & in_bounds
                        os_list.append(os_raw)
                        valids.append(valid)
                    return os_list, valids

                def load_scales(base_i32, valids):
                    """scale phase: wide ds_read_b128 of lds_scale + select-to-0
                    on invalid splits. This select IS the accumulate's OOB guard:
                    sc=0 zeroes the split's contribution, and the os read is
                    pmap0-substituted + row-clamped to a finite value, so no
                    separate float mask is needed. Must run after the scale
                    barrier."""
                    # See the pmap boundary above: keep the wide LDS read as a
                    # direct vector load to preserve ds_read_b128.
                    scale_v = fx.ptr_load(
                        fx.add_offset(lds_scale.ptr, base_i32),
                        T.vec(GRP, T.f32),
                    )
                    scs = []
                    scale_elements = _vector_elements(scale_v, fx.Float32, GRP)
                    for j in fx.range_constexpr(GRP):
                        scs.append(valids[j].select(scale_elements[j], zero_f))
                    return scs

                def load_group(base_i32):
                    """Combined pmap+scale load for the steady-state loop (both
                    LDS regions are valid past the barrier)."""
                    os_list, valids = load_os_group(base_i32)
                    scs = load_scales(base_i32, valids)
                    return os_list, scs

                # Hoist group zero's output loads before the scale barrier.
                os_g0, valid_g0 = load_os_group(fx.Int32(0))

                fx.gpu.barrier()

                sc_g0 = load_scales(fx.Int32(0), valid_g0)
                init_os = []
                for j in fx.range_constexpr(GRP):
                    init_os += [os_g0[j][k].ir_value() for k in fx.range_constexpr(VEC)]
                init_regs = [zero_f.ir_value() for _ in fx.range_constexpr(VEC)]
                init_sc = [sc_g0[j].ir_value() for j in fx.range_constexpr(GRP)]
                init_state = init_os + init_regs + init_sc

                # Carry the final group into the epilogue.
                n_minus1 = n_splits - fx.Int32(1)
                num_iters = n_minus1 >> fx.Int32(_grp_shift)
                stop_i32 = num_iters * fx.Int32(GRP)
                _sbase = (GRP + 1) * VEC

                def unpack(state):
                    os_g = [
                        [state[j * VEC + k] for k in fx.range_constexpr(VEC)]
                        for j in fx.range_constexpr(GRP)
                    ]
                    regs = [state[GRP * VEC + k] for k in fx.range_constexpr(VEC)]
                    sc_g = [state[_sbase + j] for j in fx.range_constexpr(GRP)]
                    return os_g, regs, sc_g

                def accumulate(regs, os_g, sc_g):
                    new_regs = list(regs)
                    for j in fx.range_constexpr(GRP):
                        new_regs = [
                            fx.Float32(new_regs[k])
                            + fx.Float32(os_g[j][k]) * fx.Float32(sc_g[j])
                            for k in fx.range_constexpr(VEC)
                        ]
                    return new_regs

                for i, state in range(
                    fx.Int32(0), stop_i32, fx.Int32(GRP), init=init_state
                ):
                    os_g, regs, sc_g = unpack(state)
                    i_i32 = fx.Int32(i)
                    # Prefetch the next group before accumulating the carried one.
                    n_os, n_sc = load_group(i_i32 + fx.Int32(GRP))
                    new_regs = accumulate(regs, os_g, sc_g)
                    next_os_flat = []
                    for j in fx.range_constexpr(GRP):
                        next_os_flat += [
                            n_os[j][k].ir_value() for k in fx.range_constexpr(VEC)
                        ]
                    next_sc = [n_sc[j].ir_value() for j in fx.range_constexpr(GRP)]
                    results = yield (
                        next_os_flat + [r.ir_value() for r in new_regs] + next_sc
                    )

                os_g, regs, sc_g = unpack(results)
                out_elems = accumulate(regs, os_g, sc_g)
                store_result(seq_i32, out_elems)

            def dispatch_tier_body(seq_i32, local_seq, direct_pmap: bool = False):
                """Block-uniform, side-effect-only runtime tier dispatch."""
                if fx.const_expr(is_runtime_tier):
                    if n_splits < fx.Int32(MASSIVE_THR):
                        emit_simple_body(seq_i32, local_seq, direct_pmap)
                    elif n_splits <= fx.Int32(64):
                        if fx.const_expr(M64_HI_THR > 0):
                            # Runtime high-split M64 uses the deeper pipeline.
                            if n_splits > fx.Int32(M64_HI_THR):
                                emit_massive_body(
                                    seq_i32,
                                    local_seq,
                                    1,
                                    grp_override=M64_HI_GRP,
                                    direct_pmap=direct_pmap,
                                )
                            else:
                                emit_massive_body(
                                    seq_i32,
                                    local_seq,
                                    1,
                                    direct_pmap=direct_pmap,
                                )
                        else:
                            emit_massive_body(
                                seq_i32, local_seq, 1, direct_pmap=direct_pmap
                            )
                    elif n_splits <= fx.Int32(256):
                        emit_massive_body(
                            seq_i32, local_seq, 4, direct_pmap=direct_pmap
                        )
                    else:
                        emit_massive_body(
                            seq_i32,
                            local_seq,
                            NLSE_MLDS,
                            direct_pmap=direct_pmap,
                        )
                elif fx.const_expr(not is_massive):
                    emit_simple_body(seq_i32, local_seq, direct_pmap)
                else:
                    emit_massive_body(seq_i32, local_seq, NLSE, direct_pmap=direct_pmap)

            def run_seq_loop(direct_pmap: bool = False):
                for seq in range(seq0, ub_seq, ntg, init=None):
                    seq_i32 = fx.Int32(seq)
                    local_seq = seq_i32 - q_start
                    dispatch_tier_body(seq_i32, local_seq, direct_pmap)

            if fx.const_expr(low_direct_pmap_thr > 0 and use_reduce_final_map):
                # Block-uniform side-effect dispatch; keep staging barrier placement explicit.
                if n_splits <= fx.Int32(low_direct_pmap_thr):
                    run_seq_loop(direct_pmap=True)
                else:
                    stage_pmap()
                    run_seq_loop(direct_pmap=False)
            else:
                if fx.const_expr(use_reduce_final_map):
                    stage_pmap()
                run_seq_loop(direct_pmap=False)

        if fx.const_expr(persistent and not adaptive):
            # Grid-stride persistent launch terminates at the CSR sentinel.
            tot_work = c_H * num_thread_group * num_reduce_tile
            grid_stride = fx.grid_dim.x
            work_idx = fx.block_idx.x
            while work_idx < tot_work:
                head = work_idx % c_H
                temp_idx = work_idx // c_H
                block_idx = temp_idx % num_thread_group
                tile = temp_idx // num_thread_group

                # Preserve the wave-uniform scalar CSR load.
                tile_start = _scalar_indptr(tile)
                is_past_end = tile_start == last
                if tile_start != last:
                    process_work_item(head, block_idx, tile, num_thread_group)
                    # Fence before reusing LDS for the next work item.
                    fx.gpu.barrier()

                work_idx = is_past_end.select(tot_work, work_idx + grid_stride)
        else:
            head = fx.block_idx.x
            block_idx = fx.block_idx.y  # q-pos group (NTG)
            tile = fx.block_idx.z
            ntg = fx.grid_dim.y
            process_work_item(head, block_idx, tile, ntg)

    default_stream = fx.Stream(None)

    @flyc.jit
    def launch_mla_reduce(
        partial_output: fx.Pointer,
        partial_lse: fx.Pointer,
        reduce_indptr: fx.Pointer,
        reduce_partial_map: fx.Pointer,
        reduce_final_map: fx.Pointer,
        final_output: fx.Pointer,
        final_lse: fx.Pointer,
        stride_s_o: fx.Int32,
        stride_h_o: fx.Int32,
        max_splits: fx.Int32,
        num_reduce_tile: fx.Int32,
        max_seqlen_q: fx.Int32,
        num_partial_rows: fx.Int32,
        num_final_rows: fx.Int32,
        stream: fx.Stream = default_stream,
    ):
        idx_tiles = num_reduce_tile
        idx_H = fx.Int32(H)
        idx_ntg = max_seqlen_q
        if fx.const_expr(persistent and not adaptive):
            _ps_mult = int(os.environ.get("MLA_PS_GRID_MULT", PS_GRID_MULT))
            ps_grid = max_splits * fx.Int32(_ps_mult)
            grid = (ps_grid, 1, 1)
        elif fx.const_expr(adaptive):
            # Grid-z covers only the active CSR prefix.
            grid = (idx_H, idx_ntg, num_final_rows)
        else:
            grid = (idx_H, idx_ntg, idx_tiles)
        mla_reduce_kernel(
            partial_output,
            partial_lse,
            reduce_indptr,
            reduce_partial_map,
            reduce_final_map,
            final_output,
            final_lse,
            stride_s_o,
            stride_h_o,
            max_splits,
            num_reduce_tile,
            num_partial_rows,
            num_final_rows,
            max_seqlen_q,
            value_attrs=kernel_value_attrs,
        ).launch(
            grid=grid,
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    launch_mla_reduce.compile_hints = dict(kernel_compile_hints)
    return launch_mla_reduce


# Split-K scratch is cached for CUDA-graph replay; active tiles form a CSR prefix.
_SPLITK_FACTOR_DEFAULT = 16
_SPLITK_MIN_SPLITS_DEFAULT = 64
_SPLITK_MAX_ACTIVE_TILES_DEFAULT = 64


def splitk_enabled() -> bool:
    """Opt-in gate for the cooperative split-K reduction (default OFF)."""
    return os.environ.get("AITER_MLA_REDUCE_SPLITK", "0") == "1"


def plan_splitk(*, active_tiles, H, max_seqlen_q, max_splits, num_cu):
    """Decide split-K engagement + factor from HOST-visible metadata only
    (capture-safe: no device read in the launch path).

    Returns ``(engage: bool, K: int, num_slots: int)``. ``num_slots`` is the
    number of (tile, head) combine slots = ``active_tiles * H``; the partial
    kernel launches ``num_slots * K`` blocks, the combine ``num_slots``.
    """
    if not splitk_enabled():
        return False, 1, 0
    factor = int(os.environ.get("MLA_SPLITK_FACTOR", _SPLITK_FACTOR_DEFAULT))
    min_splits = int(
        os.environ.get("MLA_SPLITK_MIN_SPLITS", _SPLITK_MIN_SPLITS_DEFAULT)
    )
    max_active = int(
        os.environ.get("MLA_SPLITK_MAX_ACTIVE_TILES", _SPLITK_MAX_ACTIVE_TILES_DEFAULT)
    )
    if max_seqlen_q != 1:
        return False, 1, 0  # decode-only prototype (NTG == 1)
    if max_splits < min_splits:
        return False, 1, 0  # not enough splits to amortize the extra kernel
    if active_tiles <= 0 or active_tiles > max_active:
        return False, 1, 0
    base_blocks = active_tiles * H
    if base_blocks >= num_cu:
        return False, 1, 0  # grid already saturated; split-K would not help
    K = max(2, min(factor, max_splits))
    # Do not oversubscribe far past the CU count (each partial should keep
    # enough splits to be worth a block).
    while K > 2 and base_blocks * K > 2 * num_cu:
        K //= 2
    return True, int(K), int(base_blocks)


def derive_actual_max_splits(reduce_indptr) -> int:
    """Return ``max_t(reduce_indptr[t+1] - reduce_indptr[t])`` from the CSR.

    MUST be called at **planning time** (outside any CUDA-graph capture region),
    after ``get_mla_metadata_v1`` has filled ``reduce_indptr`` on device.
    """
    if reduce_indptr.numel() < 2:
        return 0
    diffs = reduce_indptr[1:] - reduce_indptr[:-1]
    return int(diffs.max().item())


def plan_splitk_capture_safe(
    *,
    num_final_rows,
    H,
    max_seqlen_q,
    num_kv_splits,
    num_cu,
    actual_max_splits=None,
):
    """Capture-safe, DEFAULT-ABLE split-K plan from HOST-ONLY values.

    Returns ``(engage, K, num_slots)`` deciding split-K purely from values known
    on the host at launch, with NO device read (unlike :func:`plan_splitk`, which
    reads ``active_tiles``/``max_splits`` off the CSR via ``.item()`` and thus
    cannot run under CUDA-graph capture):

    * ``num_final_rows`` = ``final_output.size(0)`` = the decode batch size = the
      number of active reduce tiles (active tiles are the CSR prefix ``[0, bs)``),
      so ``base_slots = num_final_rows * H`` is a capture-safe count of active
      (tile, head) combine slots -- and the grid/scratch derived from it are fixed
      for a given CUDA-graph capture (one graph per batch-size bucket).
    * ``num_kv_splits`` is the host split budget (= ``max_split_per_batch`` on the
      aiter decode dispatch): an upper bound on every tile's actual ``n_splits``.
      On the persistent decode path it is often a fixed capacity (~304) unrelated
      to the current context length.
    * ``actual_max_splits`` (optional): the true ``max_t(n_splits)`` over active
      tiles from :func:`derive_actual_max_splits`. When set, engagement and K
      sizing use it instead of ``num_kv_splits``; when ``None``, the plan uses
      ``num_kv_splits`` only.

    Engage only when the grid is otherwise IDLE (few active tiles ->
    ``base_slots < num_cu``) AND ``engage_splits >= min_splits`` (default 64),
    where ``engage_splits = actual_max_splits if provided else num_kv_splits``.
    Large-batch decode (grid already saturated) keeps the single-kernel path.

    CORRECTNESS is device-adaptive and independent of this plan: the partial
    kernel reads each tile's real ``n_splits`` on-device and reduces its own
    contiguous chunk ``[j*ceil(n/K), ...)``; chunks past the end are empty and the
    combine weights them by ``exp(-inf)=0``. So the SAME captured (grid, K,
    scratch) stays correct across replays whose per-tile split counts differ
    (fixed batch bucket, varying context) -- the capture-safety property the
    opt-in ``plan_splitk`` cannot provide.
    """
    factor = int(os.environ.get("MLA_SPLITK_FACTOR", _SPLITK_FACTOR_DEFAULT))
    min_splits = int(
        os.environ.get("MLA_SPLITK_MIN_SPLITS", _SPLITK_MIN_SPLITS_DEFAULT)
    )
    if max_seqlen_q != 1:
        return False, 1, 0  # decode-only prototype (NTG == 1)
    engage_splits = (
        int(num_kv_splits) if actual_max_splits is None else int(actual_max_splits)
    )
    if engage_splits < min_splits:
        return False, 1, 0  # perf heuristic: too few splits to amortize combine
    base_slots = num_final_rows * H
    if base_slots <= 0 or base_slots >= num_cu:
        return False, 1, 0  # grid already saturated; split-K would not help
    K = max(2, min(factor, engage_splits))
    # Keep the partial grid within ~2 waves of the CU count.
    while K > 2 and base_slots * K > 2 * num_cu:
        K //= 2
    return True, int(K), int(base_slots)


@functools.lru_cache(maxsize=64)
def _get_splitk_scratch(num_slots: int, K: int, Dv: int, device_index: int):
    """Pre-allocate (once) the split-K scratch: partial weighted-output
    accumulators + (m_j, l_j) per partial. Cached so CUDA-graph replays reuse
    the same device buffers (capture-safe). Zero-initialized (partials fully
    overwrite their own rows every launch)."""
    import torch

    dev = torch.device("cuda", device_index)
    sk_acc = torch.zeros(num_slots * K, Dv, dtype=torch.float32, device=dev)
    sk_ml = torch.zeros(num_slots * K, 2, dtype=torch.float32, device=dev)
    return sk_acc, sk_ml


@functools.lru_cache(maxsize=64)
def compile_mla_reduce_splitk(
    *,
    H: int,
    Dv: int,
    out_dtype: str = "bf16",
    K: int = 8,
    output_lse: bool = False,
    waves_per_eu: int = _DEFAULT_WAVES_PER_EU,
):
    """Compile the split-K partial + combine kernels for fixed (H, Dv, K).

    Returns ``(launch_partial, launch_combine)``: two ``@flyc.jit`` launchers
    the host invokes in sequence (partial writes scratch, combine reads it and
    writes ``final_output``). ``use_reduce_final_map`` is always True here (the
    combine needs the per-tile q-range).
    """
    assert Dv % NUM_THREADS == 0
    VEC = Dv // NUM_THREADS
    kernel_value_attrs = (
        {"rocdl.waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )
    # Match the normal launcher: WPE must participate in the FlyDSL cache key,
    # not only the traced kernel attributes.
    kernel_compile_hints = (
        {"waves_per_eu": int(waves_per_eu)} if waves_per_eu >= 1 else {}
    )
    default_stream = fx.Stream(None)

    def _f32_vec_load(rsrc, offset):
        """Keep VEC-wide dynamic scratch loads as buffer ISA in FlyDSL 0.2.4.

        Copy atoms cannot preserve the required vector transaction for every
        supported VEC width (1, 2, 4, or 8).
        """
        if fx.const_expr(VEC == 1):
            return [
                fx.Float32(
                    buffer_ops.buffer_load(rsrc, offset, vec_width=1, dtype=T.f32)
                )
            ]
        v = buffer_ops.buffer_load(rsrc, offset, vec_width=VEC, dtype=T.f32)
        return _vector_elements(v, fx.Float32, VEC)

    def _f32_vec_store(rsrc, offset, elems):
        """Keep VEC-wide dynamic scratch stores as one buffer transaction."""
        if fx.const_expr(VEC == 1):
            buffer_ops.buffer_store(elems[0], rsrc, offset)
            return
        vec = _make_vector(elems, fx.Float32)
        buffer_ops.buffer_store(vec, rsrc, offset)

    def _o3d_vec_load(rsrc, row_idx, head_idx, col_idx):
        """Keep the runtime 3-D gather on buffer_load for VEC-wide ISA parity."""
        offset = row_idx * fx.Int32(H * Dv) + head_idx * fx.Int32(Dv) + col_idx
        if fx.const_expr(VEC == 1):
            return [
                fx.Float32(
                    buffer_ops.buffer_load(rsrc, offset, vec_width=1, dtype=T.f32)
                )
            ]
        v = buffer_ops.buffer_load(rsrc, offset, vec_width=VEC, dtype=T.f32)
        return _vector_elements(v, fx.Float32, VEC)

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def sk_partial_kernel(
        partial_output: fx.Pointer,  # fp32 [row, H, Dv]
        partial_lse: fx.Pointer,  # fp32 [row, H]
        reduce_indptr: fx.Pointer,  # i32  [tiles+1]
        reduce_partial_map: fx.Pointer,  # i32  [nsplits_total]
        sk_acc: fx.Pointer,  # fp32 [num_slots*K, Dv]  (out)
        sk_ml: fx.Pointer,  # fp32 [num_slots*K, 2]   (out: m_j, l_j)
        num_partial_rows: fx.Int32,
    ):
        c_VEC = VEC
        tid = fx.thread_idx.x
        col = tid * c_VEC
        sk_grid = fx.grid_dim.x

        partial_output_rsrc = _pointer_buffer_resource(
            partial_output,
            fx.Float32,
            (num_partial_rows, H, Dv),
            (H * Dv, Dv, 1),
        )
        sk_acc_rsrc = _pointer_buffer_resource(
            sk_acc,
            fx.Float32,
            (sk_grid, Dv),
            (Dv, 1),
        )
        g_pl = _pointer_buffer_tensor(
            partial_lse,
            fx.Float32,
            (num_partial_rows, H),
            (H, 1),
        )
        g_indptr = _pointer_buffer_tensor(
            reduce_indptr,
            fx.Int32,
            (sk_grid + 1,),
            (1,),
        )
        g_pmap = _pointer_buffer_tensor(
            reduce_partial_map,
            fx.Int32,
            (sk_grid * LDS_MAX_SPLITS,),
            (1,),
        )
        g_ml = _pointer_buffer_tensor(
            sk_ml,
            fx.Float32,
            (sk_grid, 2),
            (2, 1),
        )

        def store_partial_metadata(ml, row, max_lse, lse_acc):
            ml[row, 0] = max_lse
            ml[row, 1] = lse_acc

        c_H = fx.Int32(H)
        c_K = fx.Int32(K)
        w = fx.block_idx.x  # partial row = slot * K + j
        j = w % c_K
        slot = w // c_K
        tile = slot // c_H
        head = slot % c_H

        t0 = g_indptr[tile]
        t1 = g_indptr[tile + 1]
        n_splits = t1 - t0

        # contiguous split subset [lo, hi) for this partial
        chunk = (n_splits + K - 1) // c_K
        lo = j * chunk
        hi_full = lo + chunk
        over = hi_full > n_splits
        hi = over.select(n_splits, hi_full)

        neg_inf = fx.Float32(float("-inf"))
        zero_f = fx.Float32(0.0)

        init = [zero_f.ir_value() for _ in fx.range_constexpr(VEC)]
        init += [neg_inf.ir_value(), zero_f.ir_value()]

        results = init
        for s, state in range(lo, hi, fx.Int32(1), init=init):
            regs = [state[i] for i in fx.range_constexpr(VEC)]
            m = state[VEC]
            lse_acc = state[VEC + 1]
            s_i32 = fx.Int32(s)
            split_i32 = t0 + s_i32
            pmap_v = g_pmap[split_i32]
            in_b = (pmap_v >= fx.Int32(0)) & (pmap_v < num_partial_rows)
            safe_row = in_b.select(pmap_v, fx.Int32(0))
            os = _o3d_vec_load(partial_output_rsrc, safe_row, head, col)
            lse = g_pl[safe_row, head]
            lse = in_b.select(lse, neg_inf)
            new_m = fx.Float32(m).maximumf(lse)
            c_old = _exp(fx.Float32(m) - new_m, use_exp2=True)
            c_new = _exp(lse - new_m, use_exp2=True)
            new_regs = [
                (fx.Float32(regs[i]) * c_old + os[i] * c_new).ir_value()
                for i in fx.range_constexpr(VEC)
            ]
            new_l = fx.Float32(lse_acc) * c_old + c_new
            results = yield new_regs + [new_m.ir_value(), new_l.ir_value()]

        regs = [results[i] for i in fx.range_constexpr(VEC)]
        m = results[VEC]
        lse_acc = results[VEC + 1]
        _f32_vec_store(sk_acc_rsrc, w * Dv + col, regs)
        if tid == fx.Int32(0):
            store_partial_metadata(g_ml, w, m, lse_acc)

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def sk_combine_kernel(
        reduce_indptr: fx.Pointer,  # i32 [tiles + 1]
        reduce_final_map: fx.Pointer,  # i32 [tiles, 2] {q_start, q_end}
        sk_acc: fx.Pointer,  # fp32 [num_slots*K, Dv]
        sk_ml: fx.Pointer,  # fp32 [num_slots*K, 2]
        final_output: fx.Pointer,  # out [bs, H, Dv]
        final_lse: fx.Pointer,  # fp32 [bs, H]
        stride_s_o: fx.Int32,
        stride_h_o: fx.Int32,
        num_final_rows: fx.Int32,
    ):
        out_numeric_t = _out_numeric_t(out_dtype)
        c_VEC = VEC
        tid = fx.thread_idx.x
        col = tid * c_VEC
        sk_grid = fx.grid_dim.x

        sk_acc_rsrc = _pointer_buffer_resource(
            sk_acc,
            fx.Float32,
            (sk_grid * K, Dv),
            (Dv, 1),
        )
        # FlyDSL 0.2.4 boundary: combine writes use runtime output strides and
        # need the same packed bf16/fp16 buffer store as the normal path.
        final_output_rsrc = _pointer_buffer_resource(
            final_output,
            out_numeric_t,
            (num_final_rows, H, Dv),
            (stride_s_o, stride_h_o, 1),
        )
        g_indptr = _pointer_buffer_tensor(
            reduce_indptr,
            fx.Int32,
            (sk_grid + 1,),
            (1,),
        )
        g_fmap = _pointer_buffer_tensor(
            reduce_final_map,
            fx.Int32,
            (sk_grid, 2),
            (2, 1),
        )
        g_ml = _pointer_buffer_tensor(
            sk_ml,
            fx.Float32,
            (sk_grid * K, 2),
            (2, 1),
        )
        g_flse = _pointer_buffer_tensor(
            final_lse,
            fx.Float32,
            (num_final_rows, H),
            (H, 1),
        )

        def store_combined_lse(lse, row, head_idx, value):
            lse[row, head_idx] = value

        c_H = fx.Int32(H)
        c_K = fx.Int32(K)
        slot = fx.block_idx.x
        tile = slot // c_H
        head = slot % c_H
        base = slot * c_K

        t0 = g_indptr[tile]
        t1 = g_indptr[tile + 1]
        n_splits = t1 - t0
        # V1.2 metadata omits reduce_final_map entries for rows finalized by
        # stage 1 (n_splits <= 1). Keep those rows untouched: their map may
        # contain stale data from a previous decode/capture replay.
        if n_splits > fx.Int32(1):
            q_start = g_fmap[tile, 0]
            q_valid = (q_start >= fx.Int32(0)) & (q_start < num_final_rows)

            neg_inf = fx.Float32(float("-inf"))
            zero_f = fx.Float32(0.0)

            # global max over the K partials
            M = neg_inf
            for jj in fx.range_constexpr(K):
                prow = base + jj
                mj = g_ml[prow, 0]
                M = M.maximumf(mj)

            regs = [zero_f for _ in fx.range_constexpr(VEC)]
            den = zero_f
            for jj in fx.range_constexpr(K):
                prow = base + jj
                mj = g_ml[prow, 0]
                lj = g_ml[prow, 1]
                wj = _exp(mj - M, use_exp2=True)
                accj = _f32_vec_load(sk_acc_rsrc, prow * Dv + col)
                regs = [regs[i] + accj[i] * wj for i in fx.range_constexpr(VEC)]
                den = den + lj * wj

            den_ok = den > zero_f
            inv = den_ok.select(fx.rocdl.rcp(T.f32, den), zero_f)
            out_elems = [regs[i] * inv for i in fx.range_constexpr(VEC)]

            if q_valid:
                store_off = q_start * stride_s_o + head * stride_h_o + tid * VEC
                if fx.const_expr(VEC == 1):
                    buffer_ops.buffer_store(
                        out_elems[0].to(out_numeric_t),
                        final_output_rsrc,
                        store_off,
                    )
                else:
                    out_vec = _make_vector(
                        [
                            out_elems[i].to(out_numeric_t)
                            for i in fx.range_constexpr(VEC)
                        ],
                        out_numeric_t,
                    )
                    buffer_ops.buffer_store(out_vec, final_output_rsrc, store_off)
                if fx.const_expr(output_lse):
                    bad = _is_zero_or_nan(den)
                    inf = fx.Float32(float("inf"))
                    lse_val = bad.select(inf, _log(den) + M)
                    if tid == fx.Int32(0):
                        store_combined_lse(g_flse, q_start, head, lse_val)

    @flyc.jit
    def launch_partial(
        partial_output: fx.Pointer,
        partial_lse: fx.Pointer,
        reduce_indptr: fx.Pointer,
        reduce_partial_map: fx.Pointer,
        sk_acc: fx.Pointer,
        sk_ml: fx.Pointer,
        num_partial_rows: fx.Int32,
        sk_grid: fx.Int32,
        stream: fx.Stream = default_stream,
    ):
        sk_partial_kernel(
            partial_output,
            partial_lse,
            reduce_indptr,
            reduce_partial_map,
            sk_acc,
            sk_ml,
            num_partial_rows,
            value_attrs=kernel_value_attrs,
        ).launch(
            grid=(sk_grid, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def launch_combine(
        reduce_indptr: fx.Pointer,
        reduce_final_map: fx.Pointer,
        sk_acc: fx.Pointer,
        sk_ml: fx.Pointer,
        final_output: fx.Pointer,
        final_lse: fx.Pointer,
        stride_s_o: fx.Int32,
        stride_h_o: fx.Int32,
        num_final_rows: fx.Int32,
        sk_grid: fx.Int32,
        stream: fx.Stream = default_stream,
    ):
        sk_combine_kernel(
            reduce_indptr,
            reduce_final_map,
            sk_acc,
            sk_ml,
            final_output,
            final_lse,
            stride_s_o,
            stride_h_o,
            num_final_rows,
            value_attrs=kernel_value_attrs,
        ).launch(
            grid=(sk_grid, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    launch_partial.compile_hints = dict(kernel_compile_hints)
    launch_combine.compile_hints = dict(kernel_compile_hints)
    return launch_partial, launch_combine
