# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HCA-path compress + norm+rope+scatter kernels -- **gfx1250 (RDNA4, wave32)**.

Port of ``fused_compress_attn_hca.py`` (wave64) to gfx1250 wave32.
Key differences:
  - BLOCK_THREADS = 32 (wave32)
  - SLICE = 32 (head_dim elements per block)
  - VEC = SLICE_SZ / 32 (vs /64)
  - Kernel B: D=512 -> VEC=16, requires split load/store paths
  - Kernel names suffixed with "w32"

See ``fused_compress_attn_hca.py`` for the original wave64 documentation.
"""

import math
import os
from functools import lru_cache
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.arith import CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import Int32, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops

from .fused_compress_attn_common import (
    block_base_bytes_i64,
    emit_group_fp8_nm_asm_scatter,
    state_slot_byte_offset,
)
from .tensor_shim import _run_compiled

# gfx12 buffer cache-policy (CPol) SCOPE field, bits 3-4: device scope makes an
# access coherent at L2, bypassing the per-CU caches. This is the cheap, load-
# local way to read what sibling blocks wrote -- an agent-scope acquire/release
# fence expresses the same intent but lowers to a full L2 writeback.
_CPOL_SCOPE_DEV = 16

# Each boundary's arrival counter gets a cache line to itself. Packed one dword
# apart they share lines, and since the siblings of a boundary necessarily
# contend on their own dword, the packed layout drags 16 unrelated boundaries
# into that same contention at L2 -- which cost more than the fusion saves
# (prefill bs=32 Kernel A 7.1 -> 17.8us).
_SYNC_CTR_STRIDE_B = 128
_SYNC_CTR_STRIDE_F32 = _SYNC_CTR_STRIDE_B // 4

BLOCK_THREADS = 32  # 1 wave32 (RDNA4 / gfx1250)
SLICE = 32  # head_dim elements per block (grid-Y split)
_NEG_INF = float("-inf")
_LOG2E = math.log2(math.e)


# ============================================================================
# Kernel A: compress_forward with multi-wave LDS K-split
# ============================================================================


def _build_compress_forward_kernel(
    *,
    head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 64,
    ape_bf16: bool = False,
    fuse_epilogue: bool = False,
    epi_cfg: "_NormRopeScatterCfg | None" = None,
):
    """HCA compress_forward with K-axis parallelized across multiple waves.

    Architecture (multi-wave LDS K-split with per-thread VEC):
      - Grid:  (num_compress, NUM_SPLIT=head_dim/slice_size)
      - Block: BLOCK_THREADS = 64 * k_split_num_waves (8 waves on AMD).
      - Per block covers ``slice_size`` head_dim elements of one boundary.
      - Per thread owns ``VEC = slice_size / 64`` contiguous head_dim
        elements starting at lid*VEC within the block's slice.
      - K=ratio split across ``k_split_num_waves`` waves; each wave processes
        K_PER_WAVE = K/NW positions (= 16 for K=128, NW=8).
      - Per-wave local online-softmax -> (m_local, kv_local, w_local) lists
        of VEC values per thread.
      - LDS cross-wave reduction: only wave 0 active; each thread reads
        NW*VEC values from LDS, computes VEC reduced compressed values,
        writes them out via vector buffer_store.

    Tuning knobs:
      - ``k_split_num_waves`` (= NW): trades K-serial chain length for LDS
        reduce cost. Small N -> larger NW (more waves -> more CU coverage);
        large N -> smaller NW (less LDS overhead).
      - ``slice_size``: VEC width per thread. slice_size=64 -> VEC=1 scalar
        (more blocks per boundary -> small-N champion); slice_size=512 ->
        VEC=8 (1 block per boundary, v1-like -> large-N coalesced HBM).
      - ``ape_bf16``: read ape as bf16 instead of f32, halving the largest
        single source of VMEM traffic (see the traffic note). Selected
        automatically from ``ape.dtype`` by the public entry point.

    What actually bounds this kernel: how many loads the VGPR budget lets the
    unrolled K loop keep in flight. slice_size=64 with NW=4..8 is a deep local
    optimum; three plausible-looking directions all lose, and they lose in
    opposite directions, so do not retry them without new information:

      * Multiple softmax accumulators. The per-K chain (each step's exp()
        feeds the next step's rescale) looks like an ILP bottleneck, but
        round-robining K over N independent accumulator sets is catastrophic
        (prefill bs=64: 7.11us at N=1 -> 8.74 at 2 -> 17.76 at 4). The extra
        N*3*VEC live f32 values blow the budget ``waves_per_eu=8`` asks for
        and the scratch spills cost far more than the exposed latency.
      * Less traffic. Per block (slice_size=64 -> VEC=2, 128 K-iters) the
        kernel requests 16KB kv_in + 16KB score_in + 32KB ape, i.e. 73.7MB
        over 1152 blocks, and ape alone is 51% of it -- only 256KB of data,
        but every boundary reads all of it and f32 costs twice per element
        what the bf16 inputs do. Halving it via ``ape_bf16`` is worth 8% at
        prefill bs=256 yet *nothing* at prefill bs=64, which is the tell that
        the small case is not throughput bound.
      * More blocks. slice_size=32 doubles the grid to 2304 blocks (4.5 ->
        9.0 blocks/CU) and is 19% slower (7.21 -> 8.59us), because VEC=1
        narrows each lane's load to 2 bytes.

    Note the ceiling this is measured against: an elementwise probe sustains
    ~16 TB/s here, not the ~5.7 TB/s a torch reduction suggests (torch's
    reduce kernel itself only reaches 0.9 TB/s and is a bad reference). The
    small-case gap to that ceiling is exposed latency, not bandwidth.

    Phase 1 (state cache) is integrated by splitting each wave's K range at
    ``clamp(window_len, k_start, k_end)`` into a Phase 1 sub-loop reading
    kv_state + score_state (padded softmax when ``s < 0``) and a Phase 2
    sub-loop reading kv_in + score_in. Phase 2 in_row is clamped to >= 0
    so wasted reads in pure-Phase-1 iters stay in-bounds.
    """
    assert (
        head_dim % slice_size == 0
    ), f"head_dim={head_dim} must be divisible by slice_size={slice_size}"
    assert (
        slice_size % 32 == 0
    ), f"slice_size={slice_size} must be a multiple of 32 (wave width)"
    # VEC=16 (slice_size=512) is excluded: `_load_f32_vec`, which serves the
    # Phase 1 state-cache and ape reads, only splits up to 2x dwordx4.
    assert slice_size // 32 in (
        1,
        2,
        4,
        8,
    ), f"VEC={slice_size // 32} must be 1, 2, 4, or 8 (slice_size={slice_size})"
    assert (
        ratio % k_split_num_waves == 0
    ), f"K={ratio} must divide evenly across {k_split_num_waves} waves"
    assert state_size >= ratio, f"state_size={state_size} must be >= K={ratio}"
    assert not fuse_epilogue or epi_cfg is not None, "fuse_epilogue requires epi_cfg"
    D = head_dim
    K = ratio
    DIM_FULL = D
    SLICE_SZ = slice_size
    VEC = SLICE_SZ // BLOCK_THREADS  # per-lane head_dim element count
    NUM_SPLIT = D // SLICE_SZ
    NW = k_split_num_waves
    BLOCK_TH = BLOCK_THREADS * NW
    K_PER_WAVE = K // NW
    # Epilogue works in Kernel B's layout: one wave covers all D dims.
    VEC_EPI = D // BLOCK_THREADS

    # LDS layout: three independent fp32 arrays, each [NW * slice_size].
    LDS_M_ELEMS = NW * SLICE_SZ
    LDS_KV_ELEMS = NW * SLICE_SZ
    LDS_W_ELEMS = NW * SLICE_SZ

    @fx.struct
    class SharedStorage:
        lds_m: fx.Array[fx.Float32, LDS_M_ELEMS, 16]
        lds_kv: fx.Array[fx.Float32, LDS_KV_ELEMS, 16]
        lds_w: fx.Array[fx.Float32, LDS_W_ELEMS, 16]

    _kname = f"hca_compress_forward_w32_D{D}_R{ratio}_NW{NW}_SL{SLICE_SZ}_S{state_size}_flydsl"
    fm_fast = arith.FastMathFlags.fast

    @flyc.kernel(name=_kname, known_block_size=[BLOCK_TH, 1, 1])
    def kernel(
        kv_in: fx.Tensor,
        kv_in_row_stride: Int32,
        score_in: fx.Tensor,
        score_in_row_stride: Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,  # [num_slots, STATE_SIZE, DIM_FULL] f32
        kv_state_slot_stride: Int32,  # f32 elements
        kv_state_pos_stride: Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: Int32,
        score_state_pos_stride: Int32,
        state_slot_mapping: fx.Tensor,  # [bs] i32
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: Int32,
        # -- fused-epilogue operands; dummies when fuse_epilogue is off, which
        # keeps the launcher arity fixed across both builds --
        sync_ctr: fx.Tensor,  # [plan_capacity] f32 arrival counter
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: Int32,
        kv_cache_token_stride: Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: Int32,
        k_rope_buff: fx.Tensor,
        krope_block_stride: Int32,
        krope_token_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x
        sid = fx.block_idx.y
        tid = fx.thread_idx.x  # 0..BLOCK_TH-1

        c_zero_i32 = arith.constant(0, type=i32)
        c_neg_inf = arith.constant(_NEG_INF, type=f32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_log2e = arith.constant(_LOG2E, type=f32)

        def fexp_f32(x):
            return llvm.call_intrinsic(
                f32, "llvm.amdgcn.exp2.f32", [x * c_log2e], [], []
            )

        # Per-thread wave / lane (block-local); tid >= 0 -> unsigned div/rem
        # (divui/remui). Wrap back to Int32 for the signed i32 consumers.
        wid = fx.Int32((fx.Uint32(tid) // BLOCK_THREADS).ir_value())  # -> [0, NW)
        lid = fx.Int32((fx.Uint32(tid) % BLOCK_THREADS).ir_value())  # -> [0, 32)

        # -- Load plan row ----------------------------------------------
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_vec = fx.Vector(
            buffer_ops.buffer_load(plan_rsrc, fx.Int32(pid) * 4, vec_width=4, dtype=i32)
        )
        ragged_id = plan_vec[0]
        batch_id = plan_vec[1]
        position = plan_vec[2]
        window_len = plan_vec[3]

        # Sentinel-skip: run the whole body only for position >= 0, as a closure
        # under a runtime `if` (rewriter sees an opaque call -> scf.if).
        def _body():
            # Per-thread head_dim base: each thread owns VEC contiguous
            # elements starting at slice_base + lid * VEC.
            col_off_base = fx.Int32(sid) * SLICE_SZ + lid * VEC

            kv_in_rsrc = buffer_ops.create_buffer_resource(kv_in, max_size=True)
            score_in_rsrc = buffer_ops.create_buffer_resource(score_in, max_size=True)
            ape_rsrc = buffer_ops.create_buffer_resource(ape, max_size=True)

            def _state_rsrcs():
                """Build the state-cache descriptors, rebased onto this
                program's slot -- see ``state_slot_byte_offset``.

                Called from inside the Phase 1 loop body rather than hoisted up
                here on purpose. ``slot`` is a load indexed by the plan's
                batch_id, so it is a second memory round trip chained behind the
                plan load, and nothing outside Phase 1 needs it. Leaving it in
                the loop lets LICM sink it to the loop preheader, so a wave
                whose K range holds no state-cache positions -- the common case,
                and every wave when window_len is 0 -- never pays for it.
                """
                slot_map_rsrc = buffer_ops.create_buffer_resource(
                    state_slot_mapping, max_size=True
                )
                slot = buffer_ops.buffer_load(
                    slot_map_rsrc, batch_id, vec_width=1, dtype=i32
                )
                return (
                    buffer_ops.create_buffer_resource(
                        kv_state,
                        max_size=True,
                        base_byte_offset=state_slot_byte_offset(
                            slot, kv_state_slot_stride
                        ),
                    ),
                    buffer_ops.create_buffer_resource(
                        score_state,
                        max_size=True,
                        base_byte_offset=state_slot_byte_offset(
                            slot, score_state_slot_stride
                        ),
                    ),
                )

            def _load_bf16_vec_to_f32(rsrc, base_off_elems_i32):
                """Load VEC contiguous bf16 elements starting at
                ``base_off_elems_i32`` -> list of VEC f32 values.

                VEC=1: unaligned-safe scalar via dword + bit-extract.
                VEC>=2: vectorized i32 buffer_load + bitcast to bf16.
                """
                base_off = fx.Int32(base_off_elems_i32)
                # logical (unsigned) >> for the dword offset (base_off >= 0): fx
                # Int32 >> is arithmetic -> use Uint32 to keep shrui/v_lshrrev_b32.
                off_dw = fx.Int32((fx.Uint32(base_off_elems_i32) >> 1).ir_value())
                if const_expr(VEC == 1):
                    lane_in_dw = base_off & 1
                    raw_s = buffer_ops.buffer_load(rsrc, off_dw, vec_width=1, dtype=i32)
                    # logical shift for the hi-word extract too.
                    hi = fx.Int32((fx.Uint32(raw_s) >> 16).ir_value())
                    lo_or_hi = arith.select(
                        arith.cmpi(CmpIPredicate.eq, lane_in_dw.ir_value(), c_zero_i32),
                        raw_s,
                        hi.ir_value(),
                    )
                    lo16 = arith.andi(lo_or_hi, arith.constant(0xFFFF, type=i32))
                    lo16_v = fx.Vector.from_elements([lo16], dtype=fx.Int32)
                    bf16_pair = lo16_v.bitcast(fx.BFloat16)
                    # raw f32 for the explicit-fastmath float layer downstream.
                    return [bf16_pair[0].to(fx.Float32).ir_value()]
                else:
                    # base must be VEC-aligned (caller guarantees by
                    # col_off_base = sid*SLICE + lid*VEC, both multiples of VEC).
                    dwords = VEC // 2  # VEC bf16 = VEC*2 bytes
                    if const_expr(dwords == 1):
                        # buffer_load(vec_width=1) returns scalar i32; wrap
                        # into vec<1xi32> before bitcast to vec<2xbf16>.
                        raw_s = buffer_ops.buffer_load(
                            rsrc, off_dw, vec_width=1, dtype=i32
                        )
                        raw = fx.Vector.from_elements([raw_s], dtype=fx.Int32)
                    else:
                        raw = fx.Vector(
                            buffer_ops.buffer_load(
                                rsrc, off_dw, vec_width=dwords, dtype=i32
                            )
                        )
                    vec_bf16 = raw.bitcast(fx.BFloat16)
                    # raw f32 for the explicit-fastmath float layer downstream.
                    return [vec_bf16[i].to(fx.Float32).ir_value() for i in range(VEC)]

            def _load_f32_vec(rsrc, base_off_elems_i32):
                """Load VEC f32 (raw ir.Values) starting at base -> list of VEC."""
                if const_expr(VEC <= 4):
                    raw = buffer_ops.buffer_load(
                        rsrc, base_off_elems_i32, vec_width=VEC, dtype=f32
                    )
                    if const_expr(VEC == 1):
                        # vec_width=1 returns scalar, not 1-vec.
                        return [raw]
                    return [fx.Vector(raw)[i].ir_value() for i in range(VEC)]
                else:
                    # VEC == 8: AMD HW max is dwordx4 -> 2 loads.
                    assert VEC == 8
                    half = VEC // 2
                    r0 = fx.Vector(
                        buffer_ops.buffer_load(
                            rsrc, base_off_elems_i32, vec_width=half, dtype=f32
                        )
                    )
                    r1 = fx.Vector(
                        buffer_ops.buffer_load(
                            rsrc,
                            fx.Int32(base_off_elems_i32) + half,
                            vec_width=half,
                            dtype=f32,
                        )
                    )
                    return [r0[i].ir_value() for i in range(half)] + [
                        r1[i].ir_value() for i in range(half)
                    ]

            def _issue_phase2_loads(k_i32):
                """Phase 2 (ragged input) loads. Returns (kv_list, sc_list,
                ape_list) each of length VEC."""
                k = fx.Int32(k_i32)
                ape_row = fx.Int32((fx.Uint32(k_i32) % ratio).ir_value())
                in_row_raw = fx.Int32(ragged_id) - (fx.Int32(K - 1) - k)
                in_row = fx.max(in_row_raw, fx.Int32(0))
                base_in_off = in_row * fx.Int32(kv_in_row_stride) + col_off_base
                base_sc_off = in_row * fx.Int32(score_in_row_stride) + col_off_base
                base_ape_off = ape_row * DIM_FULL + col_off_base
                kv = _load_bf16_vec_to_f32(kv_in_rsrc, base_in_off)
                sc = _load_bf16_vec_to_f32(score_in_rsrc, base_sc_off)
                if const_expr(ape_bf16):
                    ape_v = _load_bf16_vec_to_f32(ape_rsrc, base_ape_off)
                else:
                    ape_v = _load_f32_vec(ape_rsrc, base_ape_off)
                return kv, sc, ape_v

            def _issue_phase1_loads(k_i32):
                """Phase 1 (state cache) loads. Returns (kv_list, sc_padded_list)
                each of length VEC. Score is -inf when s < 0."""
                s = (fx.Int32(position) - fx.Int32(K - 1) + fx.Int32(k_i32)).ir_value()
                is_pad = arith.cmpi(CmpIPredicate.slt, s, c_zero_i32)
                s_safe = fx.Int32(arith.select(is_pad, c_zero_i32, s))
                ring = fx.Int32((fx.Uint32(s_safe.ir_value()) % state_size).ir_value())
                # Slot term already folded into the descriptor base.
                base_kv_off = ring * fx.Int32(kv_state_pos_stride) + col_off_base
                base_sc_off = ring * fx.Int32(score_state_pos_stride) + col_off_base
                kv_state_rsrc, score_state_rsrc = _state_rsrcs()
                kv_list = _load_f32_vec(kv_state_rsrc, base_kv_off)
                sc_list = _load_f32_vec(score_state_rsrc, base_sc_off)
                sc_padded = [
                    arith.select(is_pad, c_neg_inf, sc_list[i]) for i in range(VEC)
                ]
                return kv_list, sc_padded

            def _softmax_step_padded(
                m_old_list, kv_old_list, w_old_list, score_k_list, kv_k_list
            ):
                """Padding-aware vector softmax step over VEC lanes. When
                score_k == -inf, w_k is forced to 0 (avoids NaN when m_old
                is also -inf). Safe in both Phase 1 (padding can occur) and
                Phase 2 (score finite -> pad-select branch is dead code).
                """
                new_m, new_kv, new_w = [], [], []
                for i in range_constexpr(VEC):
                    m_old = m_old_list[i]
                    kv_old = kv_old_list[i]
                    w_old = w_old_list[i]
                    score_k = score_k_list[i]
                    kv_k = kv_k_list[i]
                    m_new = fx.max(fx.Float32(m_old), fx.Float32(score_k)).ir_value()
                    is_first = arith.cmpf(CmpFPredicate.OEQ, m_old, c_neg_inf)
                    scale_active = fexp_f32(arith.subf(m_old, m_new))
                    scale_v = arith.select(is_first, c_zero_f32, scale_active)
                    wk_active = fexp_f32(arith.subf(score_k, m_new))
                    is_pad_score = arith.cmpf(CmpFPredicate.OEQ, score_k, c_neg_inf)
                    w_k = arith.select(is_pad_score, c_zero_f32, wk_active)
                    # Explicit fastmath float layer: fx `+`/`*` drop fastmath<fast>
                    # here (the rocdl-fastmath pass does not re-add it on gfx1250)
                    # -> ISA drift (fmac vs split add/mul). Kept raw arith.*FOp.
                    new_kv.append(
                        arith.AddFOp(
                            arith.MulFOp(kv_old, scale_v, fastmath=fm_fast).result,
                            arith.MulFOp(w_k, kv_k, fastmath=fm_fast).result,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_w.append(
                        arith.AddFOp(
                            arith.MulFOp(w_old, scale_v, fastmath=fm_fast).result,
                            w_k,
                            fastmath=fm_fast,
                        ).result
                    )
                    new_m.append(m_new)
                return new_m, new_kv, new_w

            # -- Wave's K range: [wid * K_PER_WAVE, (wid+1) * K_PER_WAVE) --
            k_start_i32 = wid * K_PER_WAVE
            k_end_i32 = k_start_i32 + K_PER_WAVE

            # Split point inside this wave's K range. Each wave sees a
            # window_len-dependent slice of Phase 1 followed by Phase 2.
            # Cases (`wl = window_len`):
            #   wl <= k_start:  pure Phase 2 (entire wave is input)
            #   wl >= k_end:    pure Phase 1 (entire wave is state cache)
            #   else:          mixed (Phase 1 in [k_start, wl), Phase 2 in [wl, k_end))
            # ``split`` = clamp(wl, k_start, k_end) gives the boundary;
            # both sub-loops are empty when their bound collapses, so any
            # of the three cases naturally falls out.
            # split = clamp(window_len, k_start, k_end) -> signed int max/min.
            split_i32 = fx.min(fx.max(window_len, k_start_i32), k_end_i32)

            # State is 3*VEC scalars: m_lane[VEC] + kv_lane[VEC] + w_lane[VEC].
            init_m = [c_neg_inf for _ in range(VEC)]
            init_kv = [c_zero_f32 for _ in range(VEC)]
            init_w = [c_zero_f32 for _ in range(VEC)]
            init_state = init_m + init_kv + init_w

            # Sub-loop 1: Phase 1 sub-range [k_start, split). Reads state
            # cache; padded softmax (score can be -inf).
            phase1_local = init_state
            for k_static, state in range(
                k_start_i32.ir_value(), split_i32.ir_value(), 1, init=init_state
            ):
                m_lane = list(state[0:VEC])
                kv_lane = list(state[VEC : 2 * VEC])
                w_lane = list(state[2 * VEC : 3 * VEC])
                k_i32 = fx.Int32(k_static)
                kv_v, sc_v = _issue_phase1_loads(k_i32)
                new_m, new_kv, new_w = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, sc_v, kv_v
                )
                phase1_local = yield list(new_m) + list(new_kv) + list(new_w)

            # Sub-loop 2: Phase 2 over the wave's WHOLE K range [k_start, k_end),
            # with the [k_start, split) prefix masked to -inf because Phase 1
            # already consumed it. A masked step is an exact no-op: score=-inf
            # forces w_k=0 and m_new=m_old, so the accumulator is scaled by
            # exp(0)=1 and incremented by 0.
            #
            # The point of covering the whole range instead of [split, k_end) is
            # the trip count: K_PER_WAVE is a compile-time constant, so this
            # unrolls and the scheduler hoists all K loads into one batch. With
            # runtime bounds it cannot, and the resulting per-iteration
            # load->softmax->load dependency chain is what throttled the kernel.
            #
            # Masked iters still issue their kv_in/score_in loads at a clamped
            # in_row -- in-bounds by construction, the same wasted-read tradeoff
            # the dynamic form already made for pure-Phase-1 waves.
            # Accumulator set 0 inherits Phase 1's state; the extra sets start
            # empty. Round-robining K over the sets is safe because the online
            # softmax is associative -- which set a given K lands in only
            # changes the order of the final merge.
            m_lane = list(phase1_local[0:VEC])
            kv_lane = list(phase1_local[VEC : 2 * VEC])
            w_lane = list(phase1_local[2 * VEC : 3 * VEC])
            for j in range_constexpr(K_PER_WAVE):
                k_i32 = k_start_i32 + j
                # Wave-uniform (split and k_start are both wave-uniform) -> SGPR
                # compare, one cndmask per element.
                is_consumed = arith.cmpi(
                    CmpIPredicate.slt, k_i32.ir_value(), split_i32.ir_value()
                )
                p2_kv, p2_sc, p2_ape = _issue_phase2_loads(k_i32)
                p2_score = [
                    arith.select(
                        is_consumed,
                        c_neg_inf,
                        arith.AddFOp(p2_sc[i], p2_ape[i], fastmath=fm_fast).result,
                    )
                    for i in range(VEC)
                ]
                m_lane, kv_lane, w_lane = _softmax_step_padded(
                    m_lane, kv_lane, w_lane, p2_score, p2_kv
                )

            m_local, kv_local, w_local = m_lane, kv_lane, w_lane

            # -- LDS write: each thread writes VEC entries per array --
            # Layout: per array, NW * SLICE_SZ fp32 entries; per-thread
            # base = wid * SLICE_SZ + lid * VEC; thread writes VEC values
            # at base+0, base+1, ..., base+VEC-1.
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_m_ptr = lds.lds_m.ptr
            lds_kv_ptr = lds.lds_kv.ptr
            lds_w_ptr = lds.lds_w.ptr
            lds_thread_base = wid * SLICE_SZ + lid * VEC
            for i in range_constexpr(VEC):
                idx_i = lds_thread_base + i
                fx.ptr_store(m_local[i], lds_m_ptr + idx_i)
                fx.ptr_store(kv_local[i], lds_kv_ptr + idx_i)
                fx.ptr_store(w_local[i], lds_w_ptr + idx_i)

            gpu.barrier()

            # -- Cross-wave reduction: only wave 0 reads and reduces --
            # Wave 0's 32 threads cover SLICE_SZ head_dim elements (VEC elements
            # per thread). For each owned element, the thread reads NW values
            # from LDS (one per K-split wave) and computes the global softmax.
            def _wave0():
                comp_list = []
                for i in range_constexpr(VEC):
                    lane_off = lid * VEC + i
                    # Global max across NW waves for this element.
                    m_g = fx.Float32(c_neg_inf)
                    m_arr = []
                    for w in range_constexpr(NW):
                        m_w = fx.ptr_load(lds_m_ptr + (lane_off + w * SLICE_SZ))
                        m_arr.append(m_w)
                        m_g = m_g.maximumf(m_w)

                    # Weighted sums (kv * scale_w) and (w * scale_w).
                    kv_sum = fx.Float32(0.0)
                    w_sum = fx.Float32(0.0)
                    for w in range_constexpr(NW):
                        idx_w = lane_off + w * SLICE_SZ
                        kv_w = fx.ptr_load(lds_kv_ptr + idx_w)
                        w_w = fx.ptr_load(lds_w_ptr + idx_w)
                        m_w = m_arr[w]
                        scale_w = fx.Float32(fexp_f32((m_w - m_g).ir_value()))
                        kv_sum = kv_sum + kv_w * scale_w
                        w_sum = w_sum + w_w * scale_w
                    rcp_w = fx.Float32(
                        llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [w_sum.ir_value()], [], []
                        )
                    )
                    comp_list.append(kv_sum * rcp_w)

                # -- Vectorized write of VEC f32 comp values --
                out_rsrc = buffer_ops.create_buffer_resource(
                    kv_compressed, max_size=True
                )
                out_off = (
                    fx.Int32(pid) * fx.Int32(kv_compressed_row_stride) + col_off_base
                )
                # When fused, a sibling block reads this slice back, so it has
                # to reach L2 (the device coherence point) rather than sit in
                # this CU's caches, which is where a default-scope store may
                # linger. Unfused, the kernel boundary flushes for us and the
                # default is both correct and faster.
                st_cpol = _CPOL_SCOPE_DEV if fuse_epilogue else 0
                if const_expr(VEC == 1):
                    buffer_ops.buffer_store(
                        comp_list[0].ir_value(), out_rsrc, out_off, None, st_cpol
                    )
                elif const_expr(VEC <= 4):
                    out_vec = fx.Vector.from_elements(comp_list, dtype=fx.Float32)
                    buffer_ops.buffer_store(
                        out_vec.ir_value(), out_rsrc, out_off, None, st_cpol
                    )
                else:
                    # VEC > 4: AMD HW max is dwordx4 -> split into Nx dwordx4 stores.
                    quarter = 4
                    n_chunks = VEC // quarter
                    for q in range_constexpr(n_chunks):
                        base = q * quarter
                        sv = fx.Vector.from_elements(
                            comp_list[base : base + quarter], dtype=fx.Float32
                        )
                        buffer_ops.buffer_store(
                            sv.ir_value(), out_rsrc, out_off + base, None, st_cpol
                        )

                if const_expr(fuse_epilogue):
                    _sync_and_fused_epilogue()

            # -- Cross-block arrival count + last-arriver epilogue ----------
            # The epilogue (RMSNorm + rope + scatter) reduces over ALL D dims,
            # but the row is produced by NUM_SPLIT sibling blocks, so it can
            # only run once the last of them has stored its slice. Each block
            # announces its arrival and whichever one finds itself last simply
            # continues into the epilogue. Nobody ever waits, so unlike a spin
            # on a flag this cannot deadlock regardless of scheduling order.
            #
            # As a standalone kernel this work launches only plan_capacity
            # waves -- far too few to hide its own dependent load chain, so it
            # costs its full ~2.9us latency at every size. Folded in here it
            # overlaps the compress work of the blocks still in flight, and
            # after the ~1.5us the sync itself costs the fused path nets out
            # 1.05-1.22x faster across prefill and decode.
            def _sync_and_fused_epilogue():
                c_one_f32 = arith.constant(1.0, type=f32)
                ctr_rsrc = buffer_ops.create_buffer_resource(sync_ctr, max_size=True)
                # Order the slice store above before the arrival announced
                # below -- they are independent VMEM ops, so without this the
                # atomic can retire first and the last arriver reads a row a
                # sibling has not landed yet.
                #
                # Workgroup scope is the cheap way to get it: it lowers to the
                # s_wait_storecnt that makes the store visible, and nothing
                # else. An agent-scope fence would also order it but is
                # unusable -- it adds a device-wide L2 writeback, and with
                # every block in the grid doing one they serialize (prefill
                # bs=64 Kernel A 7.4 -> 61us, bs=256 18 -> 345us). Device-scope
                # visibility comes from the SCOPE_DEV bits on the store above
                # and the readback below instead.
                llvm.fence(llvm.AtomicOrdering.release, syncscope="workgroup")
                # SCOPE_DEV on the atomic is what makes the election work at
                # all: gfx12 defaults atomics to CU scope, so without it each
                # CU counts in its own cached copy and several blocks -- or
                # none -- conclude they are last. The resulting corruption is
                # rare and moves around between runs, so check the `scope:` on
                # the atomic in the ISA rather than trusting a clean test run.
                # `aux` must be a Python int: the rocdl wrapper silently drops
                # one that is not, which is how this was missed the first time.
                #
                # Raw rocdl buffer atomics take a BYTE offset (unlike
                # buffer_ops, which takes elements).
                # Counting in f32 is exact for these magnitudes (NUM_SPLIT<=16).
                ctr_byte_off = (fx.Int32(pid) * _SYNC_CTR_STRIDE_B).ir_value()
                ticket = c_zero_f32
                if lid == 0:
                    ticket = rocdl.raw_ptr_buffer_atomic_fadd(
                        c_one_f32,
                        ctr_rsrc,
                        ctr_byte_off,
                        c_zero_i32,
                        aux=_CPOL_SCOPE_DEV,
                    )
                # Only lane 0 holds a real ticket; readfirstlane broadcasts it
                # in one instruction. It also makes `is_last` wave-uniform,
                # which the epilogue depends on -- its RMSNorm reduction
                # shuffles across all 32 lanes.
                arrivals = rocdl.readfirstlane(f32, ticket)
                is_last = fx.Float32(arrivals) > fx.Float32(
                    arith.constant(NUM_SPLIT - 1.5, type=f32)
                )

                def _epilogue():
                    # Put the counter back for the next launch / graph replay.
                    # Safe without extra sync: no sibling touches this boundary
                    # again until the next launch. Self-resetting matters --
                    # a per-call memset would cost more than the fusion saves.
                    if lid == 0:
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            arith.constant(-float(NUM_SPLIT), type=f32),
                            ctr_rsrc,
                            ctr_byte_off,
                            c_zero_i32,
                            aux=_CPOL_SCOPE_DEV,
                        )

                    # Whole-row readback in the epilogue's layout: 32 lanes x
                    # VEC_EPI f32 covers all D dims in this one wave.
                    #
                    # SCOPE_DEV is what makes reading the siblings' slices
                    # legal: it takes the load to L2, which is the device
                    # coherence point and where their write-through stores
                    # already landed. The textbook alternative -- an agent
                    # release/acquire fence pair -- is correct but unusable
                    # here: on gfx12 it lowers to a full L2 writeback, and with
                    # every block in the grid doing one they serialize
                    # catastrophically (prefill bs=64 Kernel A 7.4 -> 61us,
                    # and worse as the grid grows: bs=256 18 -> 345us).
                    row_rsrc = buffer_ops.create_buffer_resource(
                        kv_compressed, max_size=True
                    )
                    row_off = (
                        fx.Int32(pid) * fx.Int32(kv_compressed_row_stride)
                        + lid * VEC_EPI
                    )
                    row = []
                    for q in range_constexpr(VEC_EPI // 4):
                        r = fx.Vector(
                            buffer_ops.buffer_load(
                                row_rsrc,
                                row_off + q * 4,
                                vec_width=4,
                                dtype=f32,
                                cache_modifier=_CPOL_SCOPE_DEV,
                            )
                        )
                        row += [r[i].ir_value() for i in range_constexpr(4)]

                    _emit_norm_rope_scatter(
                        epi_cfg,
                        comp_lane=row,
                        lane=lid,
                        position=position,
                        batch_id=batch_id,
                        rms_weight=rms_weight,
                        cos_cache=cos_cache,
                        sin_cache=sin_cache,
                        kv_cache=kv_cache,
                        kv_cache_block_stride=kv_cache_block_stride,
                        kv_cache_token_stride=kv_cache_token_stride,
                        block_table=block_table,
                        block_table_seq_stride=block_table_seq_stride,
                        k_rope_buff=k_rope_buff,
                        krope_block_stride=krope_block_stride,
                        krope_token_stride=krope_token_stride,
                    )

                if is_last:
                    _epilogue()

            if wid == 0:
                _wave0()

        if fx.Int32(position) >= 0:
            _body()

    @flyc.jit
    def launch_hca_compress_forward(
        kv_in: fx.Tensor,
        kv_in_row_stride: fx.Int32,
        score_in: fx.Tensor,
        score_in_row_stride: fx.Int32,
        plan: fx.Tensor,
        kv_state: fx.Tensor,
        kv_state_slot_stride: fx.Int32,
        kv_state_pos_stride: fx.Int32,
        score_state: fx.Tensor,
        score_state_slot_stride: fx.Int32,
        score_state_pos_stride: fx.Int32,
        state_slot_mapping: fx.Tensor,
        ape: fx.Tensor,
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: fx.Int32,
        sync_ctr: fx.Tensor,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        k_rope_buff: fx.Tensor,
        krope_block_stride: fx.Int32,
        krope_token_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
        idx_s = fx.Int64(NUM_SPLIT)
        k = kernel(
            kv_in,
            kv_in_row_stride,
            score_in,
            score_in_row_stride,
            plan,
            kv_state,
            kv_state_slot_stride,
            kv_state_pos_stride,
            score_state,
            score_state_slot_stride,
            score_state_pos_stride,
            state_slot_mapping,
            ape,
            kv_compressed,
            kv_compressed_row_stride,
            sync_ctr,
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            block_table,
            block_table_seq_stride,
            k_rope_buff,
            krope_block_stride,
            krope_token_stride,
        )
        k.launch(
            grid=(idx_p, idx_s, 1),
            block=(BLOCK_TH, 1, 1),
            stream=stream,
        )

    return launch_hca_compress_forward


# ============================================================================
# Kernel B: norm + rope + scatter (BF16, per-row)
# ============================================================================


class _NormRopeScatterCfg(NamedTuple):
    """Compile-time geometry for the norm+rope+scatter epilogue."""

    D: int
    RD: int
    NOPE: int
    VEC: int
    ROPE_THREAD_LO: int
    PAIRS_PER_THREAD: int
    ratio: int
    k_per_block: int
    rms_weight_is_bf16: bool
    rms_eps: float
    quant: bool
    GROUP_SIZE_Q: int
    RTS: int
    log2_rts: int


def _norm_rope_scatter_cfg(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant: bool,
    quant_group_size: int,
) -> _NormRopeScatterCfg:
    """Validate + derive the epilogue geometry shared by Kernel B and the
    fused Kernel A epilogue."""
    D = head_dim
    RD = rope_head_dim
    NOPE = D - RD
    VEC = D // BLOCK_THREADS  # 16 for D=512 (wave32)
    assert D % BLOCK_THREADS == 0
    assert RD > 0 and RD % 2 == 0 and RD % VEC == 0

    # FP8 1xG e8m0 group-quant geometry (nope region only). GROUP_SIZE must
    # divide NOPE and be a multiple of VEC (a lane's VEC slice never crosses a
    # group).
    GROUP_SIZE_Q = quant_group_size
    assert (not quant) or (
        NOPE % GROUP_SIZE_Q == 0 and GROUP_SIZE_Q % VEC == 0
    ), f"quant: NOPE={NOPE} must be divisible by group={GROUP_SIZE_Q}, group%VEC==0"
    assert (not quant) or VEC % 4 == 0, f"quant: VEC={VEC} must be a multiple of 4"
    RTS = GROUP_SIZE_Q // VEC if quant else 1  # threads per group (=4 for G=64,VEC=16)
    return _NormRopeScatterCfg(
        D=D,
        RD=RD,
        NOPE=NOPE,
        VEC=VEC,
        ROPE_THREAD_LO=NOPE // VEC,
        PAIRS_PER_THREAD=VEC // 2,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        GROUP_SIZE_Q=GROUP_SIZE_Q,
        RTS=RTS,
        log2_rts=int(math.log2(RTS)) if quant else 0,
    )


def _emit_norm_rope_scatter(
    cfg: _NormRopeScatterCfg,
    *,
    comp_lane,
    lane,
    position,
    batch_id,
    rms_weight,
    cos_cache,
    sin_cache,
    kv_cache,
    kv_cache_block_stride,
    kv_cache_token_stride,
    block_table,
    block_table_seq_stride,
    k_rope_buff,
    krope_block_stride,
    krope_token_stride,
):
    """RMSNorm + GPT-J RoPE + paged scatter for one boundary, by one wave.

    Single source of truth for the epilogue: standalone Kernel B calls it with
    ``lane`` = its thread index, and the fused Kernel A calls it from the wave
    of whichever block arrived last for the boundary. ``comp_lane`` is that
    lane's VEC f32 slice of the compressed row; the caller decides where it
    came from (a global load in Kernel B, a post-sync readback when fused).

    Requires all BLOCK_THREADS lanes of the wave to be active -- the RMSNorm
    sum reduces across the wave via shuffle -- and ``position >= 0``.
    """
    (
        D,
        RD,
        NOPE,
        VEC,
        ROPE_THREAD_LO,
        PAIRS_PER_THREAD,
        ratio,
        k_per_block,
        rms_weight_is_bf16,
        rms_eps,
        quant,
        _GROUP_SIZE_Q,  # only reaches the ISA through RTS / log2_rts below
        RTS,
        log2_rts,
    ) = cfg

    f32 = T.f32
    i32 = T.i32
    fm_fast = arith.FastMathFlags.fast
    log2_block = int(math.log2(BLOCK_THREADS))
    c_eps = arith.constant(rms_eps, type=f32)
    c_inv_D = arith.constant(1.0 / D, type=f32)
    tid = lane
    tid_x_vec = fx.Int32(lane) * VEC

    def wave_reduce_add(w):
        # w is a raw f32 ir.Value; keep the explicit-fastmath add (fx `+`
        # drops fastmath<fast> -> ISA drift on gfx1250).
        for sh_exp in range_constexpr(log2_block):
            off = BLOCK_THREADS // (2 << sh_exp)
            peer = fx.Float32(w).shuffle_xor(off, BLOCK_THREADS).ir_value()
            w = arith.AddFOp(w, peer, fastmath=fm_fast).result
        return w

    # -- RMSNorm (wave reduce-add of squares / D + eps; rsqrt) --
    sq_local = arith.constant(0.0, type=f32)
    for i in range_constexpr(VEC):
        sq_local = arith.AddFOp(
            sq_local,
            arith.MulFOp(comp_lane[i], comp_lane[i], fastmath=fm_fast).result,
            fastmath=fm_fast,
        ).result
    sq_full = wave_reduce_add(sq_local)
    var = arith.MulFOp(sq_full, c_inv_D, fastmath=fm_fast).result
    rrms = fmath.rsqrt(
        arith.AddFOp(var, c_eps, fastmath=fm_fast).result, fastmath=fm_fast
    )

    # rms_weight load
    rmsw_rsrc = buffer_ops.create_buffer_resource(rms_weight, max_size=True)
    if const_expr(rms_weight_is_bf16):
        dwords = (VEC + 1) // 2
        # logical shift (tid_x_vec >= 0); fx Int32 >> is arithmetic.
        off_dw = fx.Int32((fx.Uint32(tid_x_vec.ir_value()) >> 1).ir_value())
        if const_expr(dwords == 1):
            raw_s = buffer_ops.buffer_load(rmsw_rsrc, off_dw, vec_width=1, dtype=i32)
            raw = fx.Vector.from_elements([raw_s], dtype=fx.Int32)
            vec_bf16 = raw.bitcast(fx.BFloat16)
            rmsw_lane = [
                vec_bf16[i].to(fx.Float32).ir_value() for i in range_constexpr(VEC)
            ]
        elif const_expr(dwords <= 4):
            raw = fx.Vector(
                buffer_ops.buffer_load(rmsw_rsrc, off_dw, vec_width=dwords, dtype=i32)
            )
            vec_bf16 = raw.bitcast(fx.BFloat16)
            rmsw_lane = [
                vec_bf16[i].to(fx.Float32).ir_value() for i in range_constexpr(VEC)
            ]
        else:
            # dwords > 4 (VEC=16 -> dwords=8): split into 2x dwordx4
            half_dw = 4
            half_bf16 = half_dw * 2
            rmsw_lane = []
            for chunk in range_constexpr(dwords // half_dw):
                r = buffer_ops.buffer_load(
                    rmsw_rsrc,
                    off_dw + chunk * half_dw,
                    vec_width=half_dw,
                    dtype=i32,
                )
                vbf16 = fx.Vector(r).bitcast(fx.BFloat16)
                rmsw_lane += [
                    vbf16[i].to(fx.Float32).ir_value()
                    for i in range_constexpr(half_bf16)
                ]
    else:
        if const_expr(VEC <= 4):
            raw = fx.Vector(
                buffer_ops.buffer_load(rmsw_rsrc, tid_x_vec, vec_width=VEC, dtype=f32)
            )
            rmsw_lane = [raw[i].ir_value() for i in range(VEC)]
        else:
            quarter = 4
            n_chunks = VEC // quarter
            rmsw_lane = []
            for q in range_constexpr(n_chunks):
                r = fx.Vector(
                    buffer_ops.buffer_load(
                        rmsw_rsrc,
                        tid_x_vec + q * quarter,
                        vec_width=quarter,
                        dtype=f32,
                    )
                )
                rmsw_lane += [r[i].ir_value() for i in range_constexpr(quarter)]

    normed_lane = [
        arith.MulFOp(
            arith.MulFOp(comp_lane[i], rrms, fastmath=fm_fast).result,
            rmsw_lane[i],
            fastmath=fm_fast,
        ).result
        for i in range(VEC)
    ]

    # -- GPT-J RoPE on RD tail -- (position >= 0 -> unsigned div)
    comp_pos_i32 = fx.Int32((fx.Uint32(position) // ratio).ir_value()) * ratio
    cos_rsrc = buffer_ops.create_buffer_resource(cos_cache, max_size=True)
    sin_rsrc = buffer_ops.create_buffer_resource(sin_cache, max_size=True)
    cos_row_base = comp_pos_i32 * (RD // 2)

    is_rope_t = arith.cmpi(
        CmpIPredicate.sge,
        tid.ir_value(),
        arith.constant(ROPE_THREAD_LO, type=i32),
    )
    rope_rel_raw = fx.Int32(tid) - ROPE_THREAD_LO
    rope_rel = fx.max(rope_rel_raw, fx.Int32(0))
    cs_lo = rope_rel * PAIRS_PER_THREAD

    if const_expr(PAIRS_PER_THREAD == 1):
        cos_b = buffer_ops.buffer_load(
            cos_rsrc, cos_row_base + cs_lo, vec_width=1, dtype=T.bf16
        )
        sin_b = buffer_ops.buffer_load(
            sin_rsrc, cos_row_base + cs_lo, vec_width=1, dtype=T.bf16
        )
        cos_vals = [fx.BFloat16(cos_b).to(fx.Float32).ir_value()]
        sin_vals = [fx.BFloat16(sin_b).to(fx.Float32).ir_value()]
    else:
        cos_vec = fx.Vector(
            buffer_ops.buffer_load(
                cos_rsrc,
                cos_row_base + cs_lo,
                vec_width=PAIRS_PER_THREAD,
                dtype=T.bf16,
            )
        )
        sin_vec = fx.Vector(
            buffer_ops.buffer_load(
                sin_rsrc,
                cos_row_base + cs_lo,
                vec_width=PAIRS_PER_THREAD,
                dtype=T.bf16,
            )
        )
        cos_vals = [
            cos_vec[i].to(fx.Float32).ir_value() for i in range(PAIRS_PER_THREAD)
        ]
        sin_vals = [
            sin_vec[i].to(fx.Float32).ir_value() for i in range(PAIRS_PER_THREAD)
        ]

    rotated_lane = list(normed_lane)
    for k in range_constexpr(PAIRS_PER_THREAD):
        e = normed_lane[2 * k]
        o = normed_lane[2 * k + 1]
        c = cos_vals[k]
        s = sin_vals[k]
        # NOTE: real part uses a non-fastmath subtract (default flags);
        # explicit-fastmath MulFOp/AddFOp for the rest (fx ops drop
        # fastmath<fast> on gfx1250 -> ISA drift).
        new_e = arith.subf(
            arith.MulFOp(e, c, fastmath=fm_fast).result,
            arith.MulFOp(o, s, fastmath=fm_fast).result,
        )
        new_o = arith.AddFOp(
            arith.MulFOp(e, s, fastmath=fm_fast).result,
            arith.MulFOp(o, c, fastmath=fm_fast).result,
            fastmath=fm_fast,
        ).result
        rotated_lane[2 * k] = new_e
        rotated_lane[2 * k + 1] = new_o

    # -- Paged scatter dest (shared by bf16 / fp8) --
    # position >= 0 (active guard) -> unsigned div/rem (divui/remui).
    ci = fx.Int32((fx.Uint32(position) // ratio).ir_value())
    block_in_seq = fx.Int32((fx.Uint32(ci.ir_value()) // k_per_block).ir_value())
    slot_in_block = fx.Int32((fx.Uint32(ci.ir_value()) % k_per_block).ir_value())
    bt_rsrc = buffer_ops.create_buffer_resource(block_table, max_size=True)
    bt_off = fx.Int32(batch_id) * fx.Int32(block_table_seq_stride) + block_in_seq
    physical_block = buffer_ops.buffer_load(bt_rsrc, bt_off, vec_width=1, dtype=i32)
    # The block term rides on the descriptor's base, not on the
    # 32-bit offset -- see `block_base_bytes_i64`.
    cache_base = slot_in_block * fx.Int32(kv_cache_token_stride)
    out_rsrc = buffer_ops.create_buffer_resource(
        kv_cache,
        max_size=True,
        base_byte_offset=block_base_bytes_i64(
            physical_block, kv_cache_block_stride, 1 if quant else 2
        ),
    )

    if const_expr(quant):
        # -- group_fp8 (V4 nm-asm) via shared emitter (wave32; same layout
        # as wave64 CSA/HCA -- single source of truth). The emitter lives
        # in _common and consumes raw ir.Values. --
        _krope_base = slot_in_block * fx.Int32(krope_token_stride)
        emit_group_fp8_nm_asm_scatter(
            normed_lane=normed_lane,
            rotated_lane=rotated_lane,
            lane=tid,
            is_rope_t=is_rope_t,
            cache_base=cache_base.ir_value(),
            out_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(kv_cache)))
            + fx.Int64(block_base_bytes_i64(physical_block, kv_cache_block_stride, 1)),
            krope_base=_krope_base.ir_value(),
            krope_base_i64=fx.Int64(fx.ptrtoint(fx.get_iter(k_rope_buff)))
            + fx.Int64(block_base_bytes_i64(physical_block, krope_block_stride, 2)),
            VEC=VEC,
            NOPE=NOPE,
            RTS=RTS,
            log2_rts=log2_rts,
            ROPE_THREAD_LO=ROPE_THREAD_LO,
            wave_width=BLOCK_THREADS,
        )
    else:
        # ---- BF16 single-buffer scatter (nope + rope contiguous) ----
        out_lane = [
            arith.select(is_rope_t, rotated_lane[i], normed_lane[i])
            for i in range_constexpr(VEC)
        ]
        cache_off = cache_base + tid_x_vec
        out_vec_t = T.vec(VEC, T.bf16)
        raw_vec = fx.Vector.from_elements(out_lane, dtype=fx.Float32)
        bf16_vec = raw_vec.truncf(out_vec_t)
        # logical shift (cache_off >= 0); fx Int32 >> is arithmetic.
        cache_off_dw = fx.Int32((fx.Uint32(cache_off.ir_value()) >> 1).ir_value())
        dwords = (VEC + 1) // 2
        bf16_as_i32 = bf16_vec.bitcast(fx.Int32)
        if const_expr(dwords == 1):
            buffer_ops.buffer_store(bf16_as_i32[0].ir_value(), out_rsrc, cache_off_dw)
        elif const_expr(dwords <= 4):
            buffer_ops.buffer_store(bf16_as_i32.ir_value(), out_rsrc, cache_off_dw)
        else:
            # dwords > 4 (VEC=16 -> dwords=8): split into 2x dwordx4.
            lo = fx.Vector.from_elements(
                [bf16_as_i32[i] for i in range(4)], dtype=fx.Int32
            )
            hi = fx.Vector.from_elements(
                [bf16_as_i32[i] for i in range(4, 8)], dtype=fx.Int32
            )
            buffer_ops.buffer_store(lo.ir_value(), out_rsrc, cache_off_dw)
            buffer_ops.buffer_store(hi.ir_value(), out_rsrc, cache_off_dw + 4)


def _build_norm_rope_scatter_kernel(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant: bool = False,
    quant_group_size: int = 64,
):
    """Build per-row RMSNorm + GPT-J RoPE + paged scatter for HCA (wave32).

    Reads kv_compressed[num_compress, head_dim] fp32 and the plan; for each
    boundary, normalizes / rotates / scatters into kv_cache.

    quant=False: BF16 single-buffer scatter (nope + rope in one kv_cache row).
    quant=True : FP8 nope (1xG e8m0 group-quant) + inline duplicated e8m0 scale into
                 kv_cache (V4 nm asm layout), rotated PE bf16 into a SEPARATE k_rope_buff
                 -- byte-identical to the C++ k_wave / fused_kv_compress_scatter output.
    """
    _cfg = _norm_rope_scatter_cfg(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
    )
    D = _cfg.D
    RD = _cfg.RD
    VEC = _cfg.VEC

    _kname = (
        f"hca_norm_rope_scatter_w32_D{D}_RD{RD}_R{ratio}_KB{k_per_block}"
        f"{'_rmsbf16' if rms_weight_is_bf16 else ''}{'_fp8' if quant else ''}_flydsl"
    )

    @flyc.kernel(name=_kname)
    def kernel(
        kv_compressed: fx.Tensor,  # [num_compress, head_dim] f32
        kv_compressed_row_stride: Int32,
        plan: fx.Tensor,  # [num_compress, 4] i32
        rms_weight: fx.Tensor,  # [head_dim] bf16 or f32
        cos_cache: fx.Tensor,  # [max_pos, RD/2] bf16
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,  # bf16: [NB,k_per_block,D]; fp8: [NB,k_per_block,entry] nope+scale
        kv_cache_block_stride: Int32,  # elements (bf16 or fp8/byte)
        kv_cache_token_stride: Int32,
        block_table: fx.Tensor,  # [bs, max_blocks_per_seq] i32
        block_table_seq_stride: Int32,
        k_rope_buff: fx.Tensor,  # fp8 only: paged [NB,k_per_block,RD] bf16 rope (dummy if !quant)
        krope_block_stride: Int32,
        krope_token_stride: Int32,
    ):
        f32 = T.f32
        i32 = T.i32

        pid = fx.block_idx.x
        tid = fx.thread_idx.x

        # -- Load plan row --
        plan_rsrc = buffer_ops.create_buffer_resource(plan, max_size=True)
        plan_vec = fx.Vector(
            buffer_ops.buffer_load(plan_rsrc, fx.Int32(pid) * 4, vec_width=4, dtype=i32)
        )
        batch_id = plan_vec[1]
        position = plan_vec[2]

        # Sentinel-skip: run the whole body only for position >= 0, as a closure
        # under a runtime `if` (rewriter sees an opaque call -> scf.if).
        tid_x_vec = fx.Int32(tid) * VEC

        # -- Load kv_compressed[pid, tid*VEC : tid*VEC + VEC] --
        # Deliberately outside the sentinel guard: the address needs only pid,
        # never the plan row. Under the guard it is control-dependent on the
        # plan load and can only issue once that returns, putting two full
        # memory round trips back to back in a kernel whose cost is pure
        # latency (flat in plan_capacity). Out here both loads issue together.
        # Sentinel rows then read a scratch row that is always allocated
        # (kv_compressed is [plan_capacity, D]) and still write nothing.
        kvc_rsrc = buffer_ops.create_buffer_resource(kv_compressed, max_size=True)
        base_off = fx.Int32(pid) * fx.Int32(kv_compressed_row_stride) + tid_x_vec
        # VEC ? {2, 4, 8, 16}: VEC <= 4 -> single dwordx{VEC}; VEC>4 -> Nx dwordx4.
        # comp_lane held as raw f32 ir.Values for the explicit-fastmath layer.
        if const_expr(VEC <= 4):
            raw = fx.Vector(
                buffer_ops.buffer_load(kvc_rsrc, base_off, vec_width=VEC, dtype=f32)
            )
            comp_lane = [raw[i].ir_value() for i in range(VEC)]
        else:
            quarter = 4
            n_chunks = VEC // quarter
            comp_lane = []
            for q in range_constexpr(n_chunks):
                r = fx.Vector(
                    buffer_ops.buffer_load(
                        kvc_rsrc,
                        base_off + q * quarter,
                        vec_width=quarter,
                        dtype=f32,
                    )
                )
                comp_lane += [r[i].ir_value() for i in range_constexpr(quarter)]

        def _body():
            _emit_norm_rope_scatter(
                _cfg,
                comp_lane=comp_lane,
                lane=tid,
                position=position,
                batch_id=batch_id,
                rms_weight=rms_weight,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
                kv_cache=kv_cache,
                kv_cache_block_stride=kv_cache_block_stride,
                kv_cache_token_stride=kv_cache_token_stride,
                block_table=block_table,
                block_table_seq_stride=block_table_seq_stride,
                k_rope_buff=k_rope_buff,
                krope_block_stride=krope_block_stride,
                krope_token_stride=krope_token_stride,
            )

        if fx.Int32(position) >= 0:
            _body()

    @flyc.jit
    def launch_hca_norm_rope_scatter(
        kv_compressed: fx.Tensor,
        kv_compressed_row_stride: fx.Int32,
        plan: fx.Tensor,
        rms_weight: fx.Tensor,
        cos_cache: fx.Tensor,
        sin_cache: fx.Tensor,
        kv_cache: fx.Tensor,
        kv_cache_block_stride: fx.Int32,
        kv_cache_token_stride: fx.Int32,
        block_table: fx.Tensor,
        block_table_seq_stride: fx.Int32,
        k_rope_buff: fx.Tensor,
        krope_block_stride: fx.Int32,
        krope_token_stride: fx.Int32,
        plan_capacity: fx.Int32,
        stream: fx.Stream,
    ):
        idx_p = fx.Int64(plan_capacity)
        k = kernel(
            kv_compressed,
            kv_compressed_row_stride,
            plan,
            rms_weight,
            cos_cache,
            sin_cache,
            kv_cache,
            kv_cache_block_stride,
            kv_cache_token_stride,
            block_table,
            block_table_seq_stride,
            k_rope_buff,
            krope_block_stride,
            krope_token_stride,
        )
        k.launch(
            grid=(idx_p, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_hca_norm_rope_scatter


# ============================================================================
# Cached compile + public API
# ============================================================================


# Fusing the epilogue into Kernel A removes a whole kernel launch and lets the
# epilogue's address resolution overlap other blocks' compress work. Opt-in
# while it soaks; set HCA_FUSE=0 to fall back to the 2-kernel path.
_FUSE_EPILOGUE = os.environ.get("HCA_FUSE", "1") == "1"
_SYNC_COUNTERS: dict[tuple[int, str], torch.Tensor] = {}


def _sync_counter(n: int, device) -> torch.Tensor:
    """Persistent zeroed f32 arrival counter for the fused epilogue.

    Cached rather than allocated per call because it must stay zeroed between
    launches, and a per-call zeroing kernel would cost more than the fusion
    saves. The last arriver for each boundary subtracts its slot back to zero,
    so the buffer is self-cleaning and safe to reuse across CUDAGraph replays.
    """
    key = (int(n), str(device))
    buf = _SYNC_COUNTERS.get(key)
    if buf is None:
        buf = torch.zeros(
            int(n) * _SYNC_CTR_STRIDE_F32, dtype=torch.float32, device=device
        )
        _SYNC_COUNTERS[key] = buf
    return buf


_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}


@lru_cache(maxsize=32)
def compile_hca_compress_forward_gfx1250(
    *,
    head_dim: int,
    ratio: int,
    state_size: int,
    k_split_num_waves: int = 8,
    slice_size: int = 64,
    ape_bf16: bool = False,
    fuse_epilogue: bool = False,
    epi_cfg: _NormRopeScatterCfg | None = None,
):
    """Build the HCA compress_forward launcher (multi-wave LDS K-split).

    Each wave handles K / ``k_split_num_waves`` K-positions; cross-wave LDS
    reduction merges per-wave softmax accumulators. Each iter selects
    between Phase 1 (state cache, ``k < window_len``) and Phase 2 (input)
    by splitting the wave's K range at ``clamp(window_len, k_start, k_end)``.

    ``slice_size`` controls per-thread vector width (VEC = slice_size / 64).
    Larger slice_size means each thread handles more head_dim elements per
    K-iter (wider buffer_load -> better HBM coalescing), but fewer blocks
    per boundary (NUM_SPLIT = head_dim / slice_size). slice_size=64 -> VEC=1
    (8 blocks/boundary, small-N champion); slice_size=512 -> VEC=8
    (1 block/boundary, v1-like HBM access, large-N champion).

    ``state_size`` is the ring-buffer modulo of ``kv_state.shape[1]`` (>= ratio).
    Cached per (head_dim, ratio, state_size, k_split_num_waves, slice_size) tuple.
    """
    launcher = _build_compress_forward_kernel(
        head_dim=head_dim,
        ratio=ratio,
        state_size=state_size,
        k_split_num_waves=k_split_num_waves,
        slice_size=slice_size,
        ape_bf16=ape_bf16,
        fuse_epilogue=fuse_epilogue,
        epi_cfg=epi_cfg,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


@lru_cache(maxsize=16)
def compile_hca_norm_rope_scatter_gfx1250(
    *,
    head_dim: int,
    rope_head_dim: int,
    ratio: int,
    k_per_block: int,
    rms_weight_is_bf16: bool,
    rms_eps: float,
    quant: bool = False,
    quant_group_size: int = 64,
):
    launcher = _build_norm_rope_scatter_kernel(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
    )
    launcher.compile_hints = dict(_DEFAULT_COMPILE_HINTS)
    return launcher


def flydsl_hca_compress_attn_gfx1250(
    *,
    kv_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    score_in: torch.Tensor,  # [num_q_tokens, head_dim] bf16
    kv_state: torch.Tensor,  # [num_slots, STATE_SIZE, head_dim] f32
    score_state: torch.Tensor,  # same shape as kv_state
    state_slot_mapping: torch.Tensor,  # [bs] i32
    plan_gpu: torch.Tensor,  # [num_compress, 4] i32
    ape: torch.Tensor,  # [ratio, head_dim] f32
    rms_weight: torch.Tensor,  # [head_dim] f32 or bf16
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    k_per_block: int,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    kv_compressed_scratch: torch.Tensor | None = None,
    quant: bool = False,
    k_rope_cache: torch.Tensor | None = None,
    quant_group_size: int = 64,
    k_split_num_waves: int | None = None,
    slice_size: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """HCA-only 2-kernel compress + norm+rope+scatter (V4-Pro Main path).

    Restrictions: ratio=128, overlap=False (implicit), head_dim=512 supported.

    Cache scatter dtype:
      * ``quant=False`` (default): BF16 single-buffer scatter -- nope + rope written
        contiguously into ``kv_cache`` [NB, k_per_block, head_dim] bf16.
      * ``quant=True``: FP8 1xG e8m0 group-quant. ``kv_cache`` is fp8
        [NB, k_per_block, entry] holding nope fp8 + inline duplicated e8m0 scale
        (V4 nm asm layout); rotated PE bf16 goes to ``k_rope_cache``
        [NB, k_per_block, rope_head_dim] bf16. Byte-identical to the C++
        ``fused_kv_compress_scatter`` k_wave output.

    Phase 1 (state cache) is enabled by passing real ``kv_state`` /
    ``score_state`` / ``state_slot_mapping``. When ``window_len > 0`` in
    the plan, the corresponding K iters are sourced from the state cache
    ring buffer instead of kv_in / score_in.

    When ``k_split_num_waves`` / ``slice_size`` are ``None`` (the default),
    the launcher auto-picks via :func:`hca_per_n_config` keyed on
    ``plan_gpu.shape[0]`` and ``kv_in.shape[0]`` (CUDAGraph-stable dispatch --
    see that function's docstring). Override only when bench-sweeping; the
    default matches the production tuning used by ATOM's compressor.
    """
    if k_split_num_waves is None or slice_size is None:
        from .fused_compress_attn_gfx1250 import hca_per_n_config_gfx1250

        auto_slice, auto_kw = hca_per_n_config_gfx1250(
            plan_gpu.shape[0], kv_in.shape[0]
        )
        if slice_size is None:
            slice_size = auto_slice
        if k_split_num_waves is None:
            k_split_num_waves = auto_kw
    # User-facing input validation -- must be ``raise`` not ``assert`` (asserts
    # are stripped under ``python -O``, which would let invalid inputs reach
    # the kernel and silently corrupt outputs / fault the GPU).
    if head_dim != 512:
        raise ValueError(f"HCA 2-kernel only supports head_dim=512, got {head_dim}")
    if ratio != 128:
        raise ValueError(f"HCA 2-kernel only supports ratio=128, got {ratio}")
    if kv_in.dim() != 2 or kv_in.shape[1] != head_dim:
        raise ValueError(f"kv_in shape {tuple(kv_in.shape)} != [*, {head_dim}]")
    if score_in.shape != kv_in.shape:
        raise ValueError(f"score_in shape {tuple(score_in.shape)} != kv_in")
    if kv_in.dtype != torch.bfloat16 or score_in.dtype != torch.bfloat16:
        raise TypeError(
            f"kv_in/score_in must be bf16; got {kv_in.dtype}/{score_in.dtype}"
        )
    if kv_in.stride(-1) != 1 or score_in.stride(-1) != 1:
        raise ValueError("kv_in/score_in inner stride must be 1")
    if kv_in.stride(0) % 2 != 0 or score_in.stride(0) % 2 != 0:
        raise ValueError(
            "kv_in/score_in row strides (bf16 elem) must be even for dword bitcast"
        )

    plan_capacity = plan_gpu.shape[0]
    if plan_capacity == 0:
        return

    if ape.shape != (ratio, head_dim) or ape.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError(
            f"ape shape {tuple(ape.shape)} dtype {ape.dtype} != "
            f"({ratio}, {head_dim}) f32/bf16"
        )
    if not ape.is_contiguous():
        raise ValueError("ape must be contiguous")

    # State cache validation.
    if kv_state.dim() != 3 or kv_state.shape[2] != head_dim:
        raise ValueError(
            f"kv_state shape {tuple(kv_state.shape)} != [*, *, {head_dim}]"
        )
    state_size = kv_state.shape[1]
    if state_size < ratio:
        raise ValueError(f"state_size={state_size} must be >= K={ratio}")
    if score_state.shape != kv_state.shape:
        raise ValueError("score_state shape != kv_state")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise TypeError("kv_state/score_state must be fp32")
    # Slot and ring strides are passed to the kernel and the descriptor is
    # rebased per slot, so the states may be strided views — a per-request
    # arena hands out a view whose slot stride is a whole entry. Only the
    # innermost dim must be unit stride: the kernel addresses it as
    # `col_off + lane`.
    if kv_state.stride(-1) != 1 or score_state.stride(-1) != 1:
        raise ValueError("kv_state/score_state inner stride must be 1")
    if state_slot_mapping.dim() != 1 or state_slot_mapping.dtype != torch.int32:
        raise ValueError("state_slot_mapping must be 1D int32")

    if quant:
        if kv_cache.dtype not in (torch.float8_e4m3fnuz, torch.float8_e4m3fn):
            raise TypeError(
                f"HCA fp8 kv_cache must be fp8 (e4m3fnuz/e4m3fn); got {kv_cache.dtype}"
            )
        if k_rope_cache is None:
            raise ValueError(
                "HCA fp8 path requires k_rope_cache (paged bf16 rope buffer)"
            )
        if k_rope_cache.dtype != torch.bfloat16:
            raise TypeError(f"k_rope_cache must be bf16; got {k_rope_cache.dtype}")
        if k_rope_cache.dim() != 3 or k_rope_cache.shape[2] != rope_head_dim:
            raise ValueError(
                f"k_rope_cache shape {tuple(k_rope_cache.shape)} != [NB, k_per_block, {rope_head_dim}]"
            )
        if k_rope_cache.stride(2) != 1:
            raise ValueError("k_rope_cache must be dense in the last dim")
    else:
        if kv_cache.dtype != torch.bfloat16:
            raise TypeError(f"HCA 2-kernel kv_cache must be bf16; got {kv_cache.dtype}")
    if block_tables.dtype != torch.int32:
        raise TypeError(f"block_tables must be int32; got {block_tables.dtype}")
    if not block_tables.is_contiguous():
        raise ValueError("block_tables must be contiguous")

    # Allocate kv_compressed scratch on demand.
    if kv_compressed_scratch is None:
        kv_compressed = torch.empty(
            (plan_capacity, head_dim),
            dtype=torch.float32,
            device=kv_in.device,
        )
    else:
        if kv_compressed_scratch.shape != (plan_capacity, head_dim):
            raise ValueError(
                f"kv_compressed_scratch shape {tuple(kv_compressed_scratch.shape)}"
                f" != ({plan_capacity}, {head_dim})"
            )
        if kv_compressed_scratch.dtype != torch.float32:
            raise TypeError("kv_compressed_scratch must be fp32")
        kv_compressed = kv_compressed_scratch

    # CRITICAL: must pass current_stream when stream is None. Stream(None) =
    # NULL/default stream, which during CUDA graph capture produces an empty
    # graph entry (kernel launches don't get recorded into the active graph),
    # so replay is a no-op -> HCA boundaries silently never fire in decode CG.
    # Match v1 single-kernel pattern (fused_compress_attn.py:1381).
    if stream is None:
        stream = torch.cuda.current_stream()
    stream_obj = Stream(stream)

    rms_weight_is_bf16 = rms_weight.dtype == torch.bfloat16
    # k_rope_buff is referenced only on the quant path; pass kv_cache as a dummy
    # (valid tensor, never read) when bf16 so the launcher arity stays fixed.
    if quant:
        krope_buf = k_rope_cache
        krope_bs = int(k_rope_cache.stride(0))
        krope_ts = int(k_rope_cache.stride(1))
    else:
        krope_buf = kv_cache
        krope_bs = 0
        krope_ts = 0

    epi_cfg = _norm_rope_scatter_cfg(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
    )
    compress_fn = compile_hca_compress_forward_gfx1250(
        head_dim=head_dim,
        ratio=ratio,
        state_size=int(state_size),
        k_split_num_waves=k_split_num_waves,
        slice_size=slice_size,
        ape_bf16=ape.dtype == torch.bfloat16,
        fuse_epilogue=_FUSE_EPILOGUE,
        epi_cfg=epi_cfg if _FUSE_EPILOGUE else None,
    )
    # Unfused builds ignore the counter but the launcher arity is fixed, so
    # hand them a 1-element dummy.
    sync_ctr = _sync_counter(plan_capacity if _FUSE_EPILOGUE else 1, kv_in.device)
    compress_args = (
        kv_in,
        int(kv_in.stride(0)),
        score_in,
        int(score_in.stride(0)),
        plan_gpu,
        kv_state,
        int(kv_state.stride(0)),
        int(kv_state.stride(1)),
        score_state,
        int(score_state.stride(0)),
        int(score_state.stride(1)),
        state_slot_mapping,
        ape,
        kv_compressed,
        int(kv_compressed.stride(0)),
        sync_ctr,
        rms_weight,
        cos_cache,
        sin_cache,
        kv_cache,
        int(kv_cache.stride(0)),
        int(kv_cache.stride(1)),
        block_tables,
        int(block_tables.stride(0)),
        krope_buf,
        krope_bs,
        krope_ts,
        int(plan_capacity),
        stream_obj,
    )
    _run_compiled(compress_fn, *compress_args)

    if _FUSE_EPILOGUE:
        # The compress kernel already scattered; no second kernel to launch.
        return

    norm_fn = compile_hca_norm_rope_scatter_gfx1250(
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        ratio=ratio,
        k_per_block=k_per_block,
        rms_weight_is_bf16=rms_weight_is_bf16,
        rms_eps=rms_eps,
        quant=quant,
        quant_group_size=quant_group_size,
    )
    norm_args = (
        kv_compressed,
        int(kv_compressed.stride(0)),
        plan_gpu,
        rms_weight,
        cos_cache,
        sin_cache,
        kv_cache,
        int(kv_cache.stride(0)),
        int(kv_cache.stride(1)),
        block_tables,
        int(block_tables.stride(0)),
        krope_buf,
        krope_bs,
        krope_ts,
        int(plan_capacity),
        stream_obj,
    )
    _run_compiled(norm_fn, *norm_args)
