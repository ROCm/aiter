# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Portions Apache-2.0, Copyright (c) 2025 FlyDSL Project Contributors

"""gfx950 DUALWAVE_SWP FP8 flash attention, vendored from FlyDSL.

Helpers live in ``flash_attn_dualwave_common.py``.

Upstream: FlyDSL ``kernels/attention/flash_attn_fp8_gfx950.py`` at tag v0.3.0
(5675194f). ``flydsl.kernels`` is not shipped in the wheel (``kernels/`` sits
outside ``python/``, which is what ``find_packages`` scans), so aiter vendors
these sources rather than importing them.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import (
    DualwaveFp8GemmHelper,
    DualwaveFp8KernelContext,
    DualwaveFp8KvGmemToLdsLoader,
    DualwaveFp8KvLdsToVgprLoader,
    DualwaveFp8PageIdLoader,
    DualwaveFp8QLoader,
    DualwaveFp8SoftmaxHelper,
    DualwaveFp8StoreHelper,
    DualwaveSplitKCombineContext,
    DualwaveSplitKCombineHelper,
    _make_dualwave_swp_fp8_traits,
    _sched_barrier_exp_pairs,
    _sched_barrier_pairs,
    dualwave_splitk_workspace_elems,  # noqa: F401
    waitcnt_vm_n,
)
from aiter.ops.flydsl.kernels.flash_attn_dualwave_common import dtype_to_elem_type
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def build_flash_attn_dualwave_swp_fp8_module(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    dualwave_swp_setprio=True,
    dualwave_swp_debug_lazy_counts=False,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    bn128=None,
    paged=False,
    prefetch_bound="none",
    pv_spread=False,
):
    """Build the gfx950 D=128 dual-wave flash-attention launcher.

    The dense path supports bf16/f16/fp8 QKV. ``varlen`` builds the packed
    variant: Q/O are ``[total_q, H, D]``, K/V are ``[total_kv, H_kv, D]``, and
    per-batch ranges come from int32 ``cu_seqlens_q`` / ``cu_seqlens_kv``.
    ``paged`` addresses KV through a block table instead of contiguously, with
    page size fixed at BLOCK_N=64. fp8 supports all three of varlen, paged and
    split-K, including in combination.

    ``prefetch_bound="clamp"`` bounds the ring's forward prefetch. It defaults
    off so the dense path stays bit-identical to upstream, and it is not a perf
    win -- see the comment at its use site."""
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(f"flash_attn_dualwave_swp requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}")
    if head_dim != 128:
        raise RuntimeError(f"flash_attn_dualwave_swp is D=128 only, got head_dim={head_dim}")
    if dtype_str not in ("bf16", "f16", "fp8"):
        raise RuntimeError(f"flash_attn_dualwave_swp supports bf16/f16/fp8 only, got dtype={dtype_str}")
    # fp8 split-K was rejected here. What blocked it was the combine pass
    # packing its output as DTYPE_STR (the input dtype) rather than the
    # kernel's fixed 2-byte output; see pack_output in the common module.
    # fp8 varlen was rejected here too. It needed the active guard the fp8
    # context was missing (see compute_active_guard in the common module) plus
    # the BN128 pipeline held on independently of varlen; with both, it works.
    #
    # The shallow (bn128=False) fp8 path still does not build at all -- its PV
    # dispatch expects an unpacked P pair that the body never produces -- so
    # fp8 stays on BN128 whatever varlen says, rather than inheriting the
    # upstream derivation that would turn it off here.
    if dtype_str == "fp8":
        if bn128 is False:
            raise RuntimeError("fp8 flash_attn has no working non-BN128 path (bn128=False)")
        bn128 = True

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0
    NUM_KV_SPLITS = int(num_kv_splits)
    assert NUM_KV_SPLITS >= 1
    if varlen and num_kv_splits and int(num_kv_splits) > 1:
        raise ValueError("varlen is not supported together with num_kv_splits > 1")

    # All compile-time tile/layout constants live in the fp8 traits object.
    traits = _make_dualwave_swp_fp8_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
        dualwave_swp_setprio=dualwave_swp_setprio,
        dualwave_swp_debug_lazy_counts=dualwave_swp_debug_lazy_counts,
        dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        num_kv_splits=num_kv_splits,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        bn128=bn128,
        paged=paged,
        prefetch_bound=prefetch_bound,
        pv_spread=pv_spread,
    )
    # Builder-level aliases used by SharedStorage and the launch/compile wrappers.
    SPLITK = traits.SPLITK
    BLOCK_M = traits.BLOCK_M
    BLOCK_SIZE = traits.BLOCK_SIZE
    HEAD_DIM = traits.HEAD_DIM
    NUM_HEADS_Q = traits.NUM_HEADS_Q
    DEFAULT_STRIDE_Q_N = traits.DEFAULT_STRIDE_Q_N
    DEFAULT_STRIDE_KV_N = traits.DEFAULT_STRIDE_KV_N
    PAGED = traits.PAGED
    _dualwave_swp_fp8_cache_tag = traits.cache_tag
    _lds_elem_dtype = dtype_to_elem_type(traits.DTYPE_STR)

    if const_expr(traits.PAGED):

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
            vt: fx.Array[fx.BFloat16, traits.VT_BF16_TOTAL, 16]
            q: fx.Array[_lds_elem_dtype, BLOCK_M * HEAD_DIM, 16]
            bt: fx.Array[fx.Int32, traits.PAGED_BT_LDS_SIZE, 16]

    else:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
            vt: fx.Array[fx.BFloat16, traits.VT_BF16_TOTAL, 16]
            q: fx.Array[_lds_elem_dtype, BLOCK_M * HEAD_DIM, 16]

    # BN128: two BLOCK_N=64 KV tiles per iteration, one merged softmax correction.
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_dualwave_swp_fp8_bn128_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        BlockTable: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
    ):
        ctx = DualwaveFp8KernelContext(
            traits,
            Q,
            K,
            V,
            O,
            DebugCounts,
            CuSeqQ,
            CuSeqKv,
            QDescale,
            KDescale,
            VDescale,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            BlockTable=BlockTable,
            block_table_stride=block_table_stride,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        if const_expr(traits.CAUSAL):
            ctx.init_causal_lpt_order()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_descale()
        ctx.init_tile_bounds()
        ctx.init_workspace_io()
        # After init_tile_bounds: the split-K guard reads split_nonempty.
        ctx.init_active_guard()

        q_loader = DualwaveFp8QLoader(ctx)
        gemm_helper = DualwaveFp8GemmHelper(ctx)
        softmax_helper = DualwaveFp8SoftmaxHelper(ctx)
        kv_gmem_to_lds = DualwaveFp8KvGmemToLdsLoader(ctx)
        kv_lds_to_regs = DualwaveFp8KvLdsToVgprLoader(ctx)
        output_store = DualwaveFp8StoreHelper(ctx)
        page_ids = DualwaveFp8PageIdLoader(ctx)

        # Stage the block table into LDS before any page id is read. Outside the
        # active guard below: it's a whole-CTA op, unlike the per-tile KV loads.
        if const_expr(traits.PAGED):
            page_ids.load_block_table_to_lds()
            rocdl.s_waitcnt(0)
            rocdl.s_barrier()

        BN = traits.BLOCK_N
        D_CHUNKS = traits.D_CHUNKS
        NPF = const_expr(traits.NUM_PREFETCH_K)
        t0 = ctx.split_t0
        t_end = ctx.split_t_end

        def _subtile_tail(v_s, v_v, v_o, l_row, m_new):
            v_s = softmax_helper.sub_m(v_s, m_new)
            v_p = softmax_helper.exp2(v_s, 0, 16)
            v_p = softmax_helper.exp2(v_p, 16, 16)
            for _ in range_constexpr(2):
                rocdl.sched_group_barrier(traits.SCHED_VALU_MASK, 8, 13)
                rocdl.sched_group_barrier(traits.SCHED_EXP_MASK, 16, 13)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = gemm_helper.cast_p_fp8_direct(v_p)
            v_o = gemm_helper.pv(v_p, v_v, v_o)
            v_o = softmax_helper.anchor_v_o(v_o)
            return v_o, l_row

        def _mask_sub(v_s, tile_idx):
            if const_expr(traits.CAUSAL):
                return v_s
            return softmax_helper.seq_pad_mask_if_needed(v_s, tile_idx)

        def _mask_pair(v_s_a, v_s_b, j):
            if const_expr(traits.CAUSAL):
                return softmax_helper.causal_mask_pair_if_needed(v_s_a, v_s_b, j)
            return v_s_a, v_s_b

        def _merge_tile_max(v_s_a, v_s_b):
            m_tile = softmax_helper.max2(softmax_helper.reduce_max(v_s_a), softmax_helper.reduce_max(v_s_b))
            if const_expr(traits.CAUSAL):
                m_tile = softmax_helper.floor_masked_max(m_tile)
            return m_tile

        def _main_body():
            kv_gmem_to_lds.load_k(t0 * BN, t0 % fx.Index(NPF))
            q_loader.stage_q_to_lds()
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()

            ctx.init_q_row()
            q_row = ctx.q_row

            q_wide = gemm_helper.load_q_wide() if const_expr(traits.QREG) else None

            kv_gmem_to_lds.load_k((t0 + 1) * BN, (t0 + 1) % fx.Index(NPF))
            kv_gmem_to_lds.load_v(t0 * BN, t0 % fx.Index(NPF))
            kv_gmem_to_lds.load_v((t0 + 1) * BN, (t0 + 1) % fx.Index(NPF))
            kv_gmem_to_lds.load_k((t0 + 2) * BN, (t0 + 2) % fx.Index(NPF))
            kv_gmem_to_lds.load_k((t0 + 3) * BN, (t0 + 3) % fx.Index(NPF))
            kv_gmem_to_lds.load_v((t0 + 2) * BN, (t0 + 2) % fx.Index(NPF))
            kv_gmem_to_lds.load_v((t0 + 3) * BN, (t0 + 3) % fx.Index(NPF))
            # The first loop iteration reads only K1, V0, V1 -- the first three
            # of the seven staged above, and vmcnt retires in issue order, so
            # vmcnt(4) covers exactly them. K2/K3/V2/V3 stay in flight; they are
            # not read until later iterations, each preceded by the full drain
            # at the end of the loop body. SMEM_D_RPT is 1 for fp8 d=128, so
            # every load_k/load_v above is exactly one DMA.
            #
            # The lgkm wait is separate and not optional: with VDMA off, V is
            # staged global->VGPR->LDS and only lgkmcnt covers that LDS store,
            # which the barrier below must see. Same pairing as the bf16 kernel.
            rocdl.s_waitcnt(traits.LGKMCNT_0_ONLY)
            waitcnt_vm_n(4 * ctx.NUM_DMA_K)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

            m_row = ctx.c_neg_inf
            l_row = ctx.c_zero_f
            v_o = [ctx.c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]

            NPF_I = const_expr(fx.Index(NPF))

            def _ring_wrap(x):
                return (x >= NPF_I).select(x - NPF_I, x)

            # The ring prefetches two iterations ahead unconditionally, so the
            # last iterations fetch tiles past t_end -- a fixed 4-tile overshoot
            # however long the range is, hence 1.29x KV traffic at M=1384 against
            # 1.016x at M=32768. Clamping re-reads a live tile from L2 instead.
            #
            # Measured 2026-08-06: this changes nothing at any shape. The kernel
            # is latency-bound, not bandwidth-bound, at every measured size, so
            # the overshoot was already absorbed by ring slack. Kept because it
            # is free and strictly less wasteful, not because it is a win; do not
            # expect it to help, and do not remove it expecting a regression.
            #
            # Clamping the tile INDEX rather than branching keeps the loop one
            # basic block. That matters: the loop carries a dense
            # sched_group_barrier schedule, and causal_mask_pair_if_needed
            # documents that an scf.if here splits it into five blocks and
            # breaks the QK/softmax/PV interleave. Same trick as
            # page_id_for_tile, for the same reason.
            t_last = t_end - fx.Index(1)

            def _pf_tile(x):
                if const_expr(traits.PREFETCH_BOUND != "clamp"):
                    return x
                return fx.Index((x < t_last).select(x, t_last))

            init_args = [m_row, l_row] + v_o + [t0 % fx.Index(NPF)]
            loop_results = init_args
            for j, loop_args in range(fx.Index(t0), t_end, fx.Index(2), init=init_args):
                m_row = loop_args[0]
                l_row = loop_args[1]
                v_o = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]

                a_buf = loop_args[2 + D_CHUNKS]
                b_buf = _ring_wrap(a_buf + fx.Index(1))
                nn_a_buf = _ring_wrap(a_buf + fx.Index(2))
                f_a_buf = _ring_wrap(a_buf + fx.Index(4))
                f_b_buf = _ring_wrap(a_buf + fx.Index(5))

                v_k_a = kv_lds_to_regs.load_k(a_buf)
                v_k_b = kv_lds_to_regs.load_k(b_buf)

                v_s_a = gemm_helper.qk(v_k_a, q_wide)
                v_s_a = _mask_sub(v_s_a, j)
                v_s_b = gemm_helper.qk(v_k_b, q_wide)
                v_s_b = _mask_sub(v_s_b, j + fx.Index(1))
                v_s_a, v_s_b = _mask_pair(v_s_a, v_s_b, j)

                v_v_a = kv_lds_to_regs.load_v(a_buf)

                pf_a = _pf_tile(j + fx.Index(4))
                pf_b = _pf_tile(j + fx.Index(5))
                kv_gmem_to_lds.load_k(pf_a * BN, f_a_buf)
                kv_gmem_to_lds.load_k(pf_b * BN, f_b_buf)
                kv_gmem_to_lds.load_v(pf_a * BN, f_a_buf)
                kv_gmem_to_lds.load_v(pf_b * BN, f_b_buf)

                m_tile = _merge_tile_max(v_s_a, v_s_b)
                v_o, m_new, l_row = softmax_helper.lazy_correct_o(v_o, m_row, l_row, m_tile)
                v_o = softmax_helper.anchor_v_o(v_o)

                v_o, l_row = _subtile_tail(v_s_a, v_v_a, v_o, l_row, m_new)
                v_v_b = kv_lds_to_regs.load_v(b_buf)
                v_o, l_row = _subtile_tail(v_s_b, v_v_b, v_o, l_row, m_new)
                m_row = m_new

                _sched_barrier_exp_pairs(traits, 8, 16, 11)
                _sched_barrier_pairs(traits, 8, 25, 11)
                rocdl.s_waitcnt(0)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

                loop_results = yield [m_row, l_row] + v_o + [nn_a_buf]
            m_row = loop_results[0]
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]

            inv_l_rcp = rocdl.rcp(T.f32, _raw(l_row))
            inv_l = ArithValue(fx.Float32(l_row) > ctx.c_zero_f).select(inv_l_rcp, ctx.c_zero_f)
            if const_expr(traits.FP8_PV):
                inv_l = ArithValue(inv_l) * ctx.vd_fp8
            softmax_helper.scale_o(v_o, inv_l)
            rocdl.s_barrier()
            if const_expr(not SPLITK):
                output_store.store_final_o(v_o, q_row)
            else:
                # Under split-K this workgroup owns one KV range, so it writes a
                # partial plus its (m, l) for the combine pass instead of the
                # final output. The fp8 body previously called store_final_o
                # unconditionally, leaving the workspace all zeros -- the
                # partial-store helpers existed but nothing called them.
                output_store.store_splitk_partial_o(v_o, m_row, l_row, q_row)

        # Skip workgroups whose Q block lies past their own sequence. The grid
        # is sized for the longest sequence, so under varlen the excess blocks
        # of every shorter one would otherwise write into the next packed
        # sequence's rows. `active` is None on the dense path, which keeps the
        # emitted code identical to before. Mirrors the bf16 kernel.
        if ctx.active is None:
            _main_body()
        else:

            @flyc.jit
            def _run_body_if_active():
                if ctx.active:
                    _main_body()

            _run_body_if_active()

        # Outside the active guard on purpose: a split whose KV range is empty
        # skips the body entirely but still has to mark its (m, l) slots, or the
        # combine pass reads stale workspace for that split.
        if const_expr(SPLITK):
            output_store.store_empty_split()

    # Combine kernel: out = sum_s w_s * O_s / sum_s w_s * l_s, w_s = exp2(m_s - m_max).
    # One wave row of 32 lanes covers a (b, h, s) row, 4 contiguous cols/lane.
    COMBINE_BLOCK = 256
    COMBINE_LANES_PER_ROW = traits.HEAD_DIM // 4
    COMBINE_ROWS_PER_BLOCK = COMBINE_BLOCK // COMBINE_LANES_PER_ROW

    @flyc.kernel(known_block_size=[COMBINE_BLOCK, 1, 1])
    def flash_attn_splitk_combine_kernel(
        O: fx.Tensor,  # noqa: E741
        WS: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stride_q_n: fx.Int32,
    ):
        ctx = DualwaveSplitKCombineContext(traits, O, WS, batch_size, seq_len, stride_q_n)
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_thread_mapping(COMBINE_ROWS_PER_BLOCK, COMBINE_LANES_PER_ROW)
        ctx.init_workspace()
        ctx.init_descriptors()

        combine = DualwaveSplitKCombineHelper(ctx)
        m_s, l_s = combine.load_ml_rows()
        m_max = combine.reduce_m_max(m_s)
        acc, den = combine.accumulate_splits(m_s, l_s, m_max)
        o_pack = combine.pack_output(acc, den)
        combine.store_output(o_pack)

    @flyc.jit
    def launch_flash_attn_dualwave_swp(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        DebugCounts: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        BlockTable: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        block_table_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _dualwave_swp_fp8_cache_tag
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + BLOCK_M - 1) // BLOCK_M
        if const_expr(SPLITK):
            grid_z = bs_idx * NUM_KV_SPLITS
        else:
            grid_z = bs_idx

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(daz)
            else None
        )
        flash_attn_dualwave_swp_fp8_bn128_kernel(
            Q,
            K,
            V,
            O,
            DebugCounts,
            CuSeqQ,
            CuSeqKv,
            QDescale,
            KDescale,
            VDescale,
            BlockTable,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            block_table_stride,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(NUM_HEADS_Q, num_q_blocks, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )
        if const_expr(SPLITK):
            combine_rows = bs_idx * NUM_HEADS_Q * sl_idx
            flash_attn_splitk_combine_kernel(O, DebugCounts, batch_size, seq_len, stride_q_n).launch(
                grid=(combine_rows // COMBINE_ROWS_PER_BLOCK, 1, 1),
                block=(COMBINE_BLOCK, 1, 1),
                stream=stream,
            )

    _dualwave_swp_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    def _launch(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,
        head_dim_runtime=None,
        debug_counts=None,
        *,
        seq_len_kv=None,
        workspace=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        block_table=None,
        block_table_stride=None,
        stream=None,
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        # seq_len_kv defaults to seq_len (self-attention / equal Q,KV lengths).
        if seq_len_kv is None:
            seq_len_kv = seq_len
        if SPLITK:
            if workspace is None:
                raise ValueError("num_kv_splits > 1 requires a fp32 workspace (see dualwave_splitk_workspace_elems)")
            debug_counts = workspace
        if debug_counts is None:
            debug_counts = O
        # Dense launches still pass valid tensors for the (unused) cu_seqlens slots;
        # the kernel only reads them under const_expr(VARLEN). Use O as a placeholder.
        if cu_seqlens_q is None:
            cu_seqlens_q = O
        if cu_seqlens_kv is None:
            cu_seqlens_kv = O
        # Per-tensor fp8 descales (shape-[1] fp32). The kernel only reads them on
        # the fp8 path; bf16/f16 launches pass O as an unused placeholder.
        if q_descale is None:
            q_descale = O
        if k_descale is None:
            k_descale = O
        if v_descale is None:
            v_descale = O
        # BlockTable is only read under const_expr(PAGED); use O as a placeholder
        # otherwise. block_table_stride defaults to 0 (unused without paging).
        if block_table is None:
            block_table = O
        if block_table_stride is None:
            block_table_stride = 0
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return _run_compiled(
                launch_flash_attn_dualwave_swp,
                Q,
                K,
                V,
                O,
                debug_counts,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_descale,
                k_descale,
                v_descale,
                block_table,
                batch_size,
                seq_len,
                seq_len_kv,
                stride_q_n,
                stride_kv_n,
                head_dim_runtime,
                block_table_stride,
                fx.Stream(stream),
            )

    def _compile(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,
        head_dim_runtime=None,
        debug_counts=None,
        *,
        seq_len_kv=None,
        workspace=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        block_table=None,
        block_table_stride=None,
        stream=None,
    ):
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        if seq_len_kv is None:
            seq_len_kv = seq_len
        if SPLITK:
            if workspace is None:
                raise ValueError("num_kv_splits > 1 requires a fp32 workspace (see dualwave_splitk_workspace_elems)")
            debug_counts = workspace
        if debug_counts is None:
            debug_counts = O
        if cu_seqlens_q is None:
            cu_seqlens_q = O
        if cu_seqlens_kv is None:
            cu_seqlens_kv = O
        if q_descale is None:
            q_descale = O
        if k_descale is None:
            k_descale = O
        if v_descale is None:
            v_descale = O
        if block_table is None:
            block_table = O
        if block_table_stride is None:
            block_table_stride = 0
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return flyc.compile(
                launch_flash_attn_dualwave_swp,
                Q,
                K,
                V,
                O,
                debug_counts,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_descale,
                k_descale,
                v_descale,
                block_table,
                batch_size,
                seq_len,
                seq_len_kv,
                stride_q_n,
                stride_kv_n,
                head_dim_runtime,
                block_table_stride,
                fx.Stream(stream),
            )

    _launch.compile = _compile

    return _launch
