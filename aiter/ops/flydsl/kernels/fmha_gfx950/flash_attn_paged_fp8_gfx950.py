# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""gfx950 paged FP8 flash attention with independent value dimensions."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.ops.flydsl.kernels.fmha_gfx950.common import load as _load
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_op_epilog import DualwaveFp8StoreHelper
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_op_gemm import DualwaveFp8GemmHelper
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_op_lds import (
    DualwaveFp8KvGmemToLdsLoader,
    DualwaveFp8KvLdsToVgprLoader,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_op_softmax import (
    DualwaveFp8SoftmaxHelper,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
    PAGED_FP8_BUFFER_LIMIT_BYTES,
    DualwaveFp8KernelContext,
    _make_paged_dualwave_swp_fp8_traits,
)
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled


def _query_bound_is_safe(ctx, upper_bound, maximum):
    # A proof must reject unordered comparisons and preserve overflow behavior.
    with fx.fastmath(None):
        safe = ((upper_bound - maximum) * ctx.c_logit_scale <= ctx.c_rescale_thr_f) & (
            ctx.c_logit_scale > 0
        )
    unsafe = safe == fx.Boolean(False)
    unsafe_lanes = fx.Int64(rocdl.ballot(fx.Int64.ir_type, unsafe.ir_value()))
    return unsafe_lanes == 0


def build_flash_attn_paged_fp8_module(
    num_heads,
    head_dim,
    value_head_dim=None,
    causal=True,
    dtype_str="bf16",
    num_kv_heads=None,
    waves_per_eu=2,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    rescale_threshold=8.0,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    paged_bn128=False,
    batch_interleave_group=1,
    page_size=64,
    cache_buffered=False,
    metadata_mode="block_table",
    has_last_page_lens=False,
    guard_output_rows=True,
):
    """Build the gfx950 packed-varlen paged FP8 attention launcher.

    Priority/stagger options are accepted for caller compatibility; this
    paired-page pipeline uses neither wave-priority nor staggered phases.
    All page-ID paths consume packed cu-seqlens, including B=1. Direct
    callers supply native cache tensors, flat contiguous Q/O, int32 metadata
    and one-element fp32 descales for both launch and explicit compilation;
    the public interface owns shape validation, copies and empty outputs.
    Only disable output-row guarding when the caller proves that every
    launched row's byte offset fits the hardware's 32-bit buffer offset.
    """
    gpu_arch = get_hip_arch()
    if value_head_dim is None:
        value_head_dim = head_dim

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(f"paged FP8 flash attention requires gfx950, got {gpu_arch}")
    native_cache_layout = (
        page_size == 1 and kv_cache_layout in ("linear", "linear3d")
    ) or (page_size in (16, 64, 1024) and kv_cache_layout == "vectorized")
    if (
        not paged
        or dtype_str != "fp8"
        or not causal
        or not varlen
        or not cross_seqlen
        or (head_dim, value_head_dim) not in ((128, 128), (192, 128), (192, 192))
        or not native_cache_layout
        or int(num_kv_splits) != 1
    ):
        raise RuntimeError(
            "paged FP8 flash_attn requires gfx950, causal packed-varlen cross-attention, "
            "page-16/64/1024 vectorized or page-1 linear/linear3d KV, (head_dim,value_head_dim) in "
            "{(128,128),(192,128),(192,192)}, "
            "and num_kv_splits=1"
        )

    if num_kv_heads is None:
        num_kv_heads = num_heads
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(
            "paged FP8 query and KV head counts must be positive, with query heads divisible by KV heads"
        )
    if paged_bn128 and page_size != 64:
        raise ValueError("paired page-ID loads require physical page size 64")
    if cache_buffered and page_size not in (1, 16):
        raise ValueError(
            "whole-cache buffer descriptors are reserved for page sizes 1/16"
        )
    if metadata_mode not in ("block_table", "csr"):
        raise ValueError("paged FP8 metadata mode must be block_table or csr")
    if metadata_mode == "csr" and page_size > 1 and not has_last_page_lens:
        raise ValueError("CSR pages larger than one require last-page lengths")
    batch_interleave_group = int(batch_interleave_group)
    if batch_interleave_group < 1:
        raise ValueError(
            f"batch_interleave_group must be positive, got {batch_interleave_group}"
        )
    traits = _make_paged_dualwave_swp_fp8_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        value_head_dim=value_head_dim,
        waves_per_eu=waves_per_eu,
        daz=daz,
        dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
        rescale_threshold=rescale_threshold,
        paired_page_ids=paged_bn128,
        batch_interleave_group=batch_interleave_group,
        page_size=page_size,
        kv_cache_layout=kv_cache_layout,
        cache_buffered=cache_buffered,
        metadata_mode=metadata_mode,
        has_last_page_lens=has_last_page_lens,
        guard_output_rows=guard_output_rows,
    )
    BLOCK_M = traits.BLOCK_M
    BLOCK_SIZE = traits.BLOCK_SIZE
    HEAD_DIM = traits.HEAD_DIM
    NUM_HEADS_Q = traits.NUM_HEADS_Q
    PAIRED_PAGE_IDS = bool(paged_bn128)
    BATCH_INTERLEAVE_GROUP = traits.BATCH_INTERLEAVE_GROUP
    DEFAULT_STRIDE_Q_N = traits.DEFAULT_STRIDE_Q_N
    DEFAULT_STRIDE_O_N = traits.NUM_HEADS_Q * traits.HEAD_DIM_V
    _dualwave_swp_fp8_cache_tag = traits.cache_tag

    @fx.struct
    class SharedStorage:
        kv: fx.Array[fx.Float8E4M3FN, traits.LDS_KV_TOTAL_SIZE, 16]
        vt: fx.Array[fx.BFloat16, traits.VT_BF16_TOTAL, 16]

    # BN128: two BLOCK_N=64 KV tiles per iteration, one merged softmax correction.
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_paged_fp8_bn128_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        CuSeqQ: fx.Tensor,
        KvMetadata: fx.Tensor,
        LastPageLens: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_o_n: fx.Int32,
        softmax_scale: fx.Float32,
    ):
        ctx = DualwaveFp8KernelContext(
            traits,
            Q,
            K,
            V,
            O,
            CuSeqQ=CuSeqQ,
            KvMetadata=KvMetadata,
            LastPageLens=LastPageLens,
            QDescale=QDescale,
            KDescale=KDescale,
            VDescale=VDescale,
            stride_q_n=stride_q_n,
            softmax_scale=softmax_scale,
            stride_o_n=DEFAULT_STRIDE_O_N if PAIRED_PAGE_IDS else stride_o_n,
            BlockTable=BlockTable,
            block_table_stride=block_table_stride,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_varlen_causal_lpt_order()
        ctx.init_descriptors()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_descale()
        ctx.init_tile_bounds()

        gemm_helper = DualwaveFp8GemmHelper(ctx)
        softmax_helper = DualwaveFp8SoftmaxHelper(ctx)
        kv_gmem_to_lds = DualwaveFp8KvGmemToLdsLoader(ctx)
        kv_lds_to_regs = DualwaveFp8KvLdsToVgprLoader(ctx)
        output_store = DualwaveFp8StoreHelper(ctx)

        BN = traits.BLOCK_N
        D_CHUNKS = traits.D_CHUNKS
        NPF = traits.NUM_PREFETCH_K
        BOUNDED_MAX = traits.HEAD_DIM == 128 and traits.DUALWAVE_SWP_LAZY_RESCALE
        FRAGMENT_PV = traits.HEAD_DIM == 192 and traits.HEAD_DIM_V == 128
        t0 = ctx.split_t0
        t_end = ctx.split_t_end

        def _load_v_first(buf_id):
            if const_expr(FRAGMENT_PV):
                return kv_lds_to_regs.load_v_fragment(buf_id, 0)
            return kv_lds_to_regs.load_v(buf_id)

        def _softmax_part(v_s, l_row, m_new):
            v_s = softmax_helper.sub_m(v_s, m_new)
            v_p = softmax_helper.exp2(v_s, 0)
            v_p = softmax_helper.exp2(v_p, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = gemm_helper.cast_p_fp8_direct(v_p)
            return v_p, l_row

        def _subtile_tail(v_s, v_v, v_o, l_row, m_new, buf_id):
            v_p, l_row = _softmax_part(v_s, l_row, m_new)
            # Keep the post-MFMA accumulators in SSA. Pinning them after each
            # subtile lengthens the paged schedule without reducing registers.
            if const_expr(FRAGMENT_PV):
                v_o = gemm_helper.preserve_accumulators(v_o)
                next_v = v_v
                # Keep K early on every page layout; retain only one current V
                # fragment and overlap the next LDS read with this fragment's PV.
                for dc in range_constexpr(D_CHUNKS):
                    current = next_v
                    if const_expr(dc + 1 < D_CHUNKS):
                        next_v = kv_lds_to_regs.load_v_fragment(buf_id, dc + 1)
                    v_o[dc] = gemm_helper._mfma_acc_fp8_wide(current, v_p, v_o[dc])
            else:
                v_o = gemm_helper.pv(v_p, v_v, v_o)
            return v_o, l_row

        def _correct_o(v_o, m_row, l_row, m_tile):
            if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                return softmax_helper.lazy_correct_o(v_o, m_row, l_row, m_tile)
            m_new, corr = softmax_helper.rescale_from_tile_max(m_row, m_tile)
            softmax_helper.scale_o(v_o, corr)
            return v_o, m_new, softmax_helper.apply_l_rescale(l_row, corr)

        def _merge_tile_max(v_s_a, v_s_b):
            m_tile = softmax_helper.reduce_max_pair(v_s_a, v_s_b)
            return softmax_helper.floor_masked_max(m_tile)

        page_t0, page_t1 = ctx.load_page_id_pair(t0 * BN)
        kv_gmem_to_lds.load_k(t0 * BN, t0 % NPF, page_id=page_t0)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()

        ctx.init_q_row()
        q_row = ctx.q_row

        q_wide = gemm_helper.load_q_wide()

        upper_bound = ctx.c_zero_f
        if const_expr(BOUNDED_MAX):
            # E4M3FN finite keys satisfy |K_i| <= 448. Bound every score by
            # 2 * 448 * sum(abs(Q_i)); factor two dominates FP32 summation error.
            max_key = fx.Vector.filled(8, 0x7E7E7E7E, fx.Int32)
            bound_tile = ctx.c_zero_v16f32
            for ws in range_constexpr(2):
                absolute_query = fx.Vector(q_wide[ws]) & fx.Int32(0x7F7F7F7F)
                bound_tile = gemm_helper._mfma_acc_fp8_wide(
                    max_key, absolute_query, bound_tile
                )
            with fx.fastmath(None):
                upper_bound = fx.Float32(
                    softmax_helper.anchor_scalar_f32(
                        fx.Vector(bound_tile)[0] * fx.Float32(2.0)
                    )
                )

        page_t2, page_t3 = ctx.load_page_id_pair((t0 + 2) * BN)
        kv_gmem_to_lds.load_k((t0 + 1) * BN, (t0 + 1) % NPF, page_id=page_t1)
        kv_gmem_to_lds.load_v(t0 * BN, t0 % NPF, page_id=page_t0)
        kv_gmem_to_lds.load_v((t0 + 1) * BN, (t0 + 1) % NPF, page_id=page_t1)
        kv_gmem_to_lds.load_k((t0 + 2) * BN, (t0 + 2) % NPF, page_id=page_t2)
        kv_gmem_to_lds.load_k((t0 + 3) * BN, (t0 + 3) % NPF, page_id=page_t3)
        if const_expr(traits.FP8_PV_SEGMENTED):
            kv_gmem_to_lds.load_v((t0 + 2) * BN, (t0 + 2) % NPF, page_id=page_t2)
            kv_gmem_to_lds.load_v((t0 + 3) * BN, (t0 + 3) % NPF, page_id=page_t3)
        else:
            next_v_a = kv_gmem_to_lds.load_v_source((t0 + 2) * BN, page_id=page_t2)
            next_v_b = kv_gmem_to_lds.load_v_source((t0 + 3) * BN, page_id=page_t3)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        m_row = ctx.c_neg_inf
        l_row = ctx.c_zero_f
        v_o = [ctx.c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]

        NPF_I = fx.Int64(NPF)

        def _ring_wrap(x):
            if const_expr(NPF == 8):
                return x & 7
            return (x >= NPF_I).select(x - NPF_I, x)

        init_args = [m_row, l_row] + v_o + [fx.Int64(t0) % NPF]
        if const_expr(not traits.FP8_PV_SEGMENTED):
            init_args += [next_v_a, next_v_b]
        loop_results = init_args
        next_v_arg_idx = 3 + D_CHUNKS

        def _iterate(
            j, loop_args, do_mask, skip_max=False, initialize=False, mask_v=True
        ):
            m_row = loop_args[0]
            l_row = loop_args[1]
            v_o = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]

            a_buf = fx.Int64(loop_args[2 + D_CHUNKS])
            b_buf = _ring_wrap(a_buf + 1)
            nn_a_buf = _ring_wrap(a_buf + 2)
            nn_b_buf = _ring_wrap(a_buf + 3)
            f_a_buf = _ring_wrap(a_buf + 4)
            f_b_buf = _ring_wrap(a_buf + 5)

            if const_expr(traits.FP8_PV_SEGMENTED):
                v_k_a = kv_lds_to_regs.load_k(a_buf)
                v_k_b = kv_lds_to_regs.load_k(b_buf)
                v_s_a = gemm_helper.qk(v_k_a, q_wide)
                v_s_b = gemm_helper.qk(v_k_b, q_wide)

                page_f_a, page_f_b = ctx.load_page_id_pair((j + 4) * BN)
                kv_gmem_to_lds.load_k((j + 4) * BN, f_a_buf, page_id=page_f_a)
                kv_gmem_to_lds.load_k((j + 5) * BN, f_b_buf, page_id=page_f_b)
                kv_gmem_to_lds.load_v(
                    (j + 4) * BN, f_a_buf, page_id=page_f_a, mask_padding=mask_v
                )
                kv_gmem_to_lds.load_v(
                    (j + 5) * BN, f_b_buf, page_id=page_f_b, mask_padding=mask_v
                )

                if const_expr(do_mask):
                    v_s_a, v_s_b = softmax_helper.causal_mask_pair_if_needed(
                        v_s_a, v_s_b, j
                    )
                m_tile = _merge_tile_max(v_s_a, v_s_b)
                v_o, m_new, l_row = _correct_o(v_o, m_row, l_row, m_tile)
                v_o = softmax_helper.anchor_v_o(v_o)

                v_p_a, l_row = _softmax_part(v_s_a, l_row, m_new)
                v_v_a = kv_lds_to_regs.load_v(a_buf)
                v_o = gemm_helper.pv(v_p_a, v_v_a, v_o)
                v_p_b, l_row = _softmax_part(v_s_b, l_row, m_new)
                v_v_b = kv_lds_to_regs.load_v(b_buf)
                v_o = gemm_helper.pv(v_p_b, v_v_b, v_o)
                m_row = m_new
                next_args = [m_row, l_row] + v_o + [nn_a_buf]
            else:
                next_v_a = loop_args[next_v_arg_idx]
                next_v_b = loop_args[next_v_arg_idx + 1]

                v_k_a = kv_lds_to_regs.load_k(a_buf)
                v_k_b = kv_lds_to_regs.load_k(b_buf)
                v_v_a = _load_v_first(a_buf)

                page_f_a, page_f_b = ctx.load_page_id_pair((j + 4) * BN)
                kv_gmem_to_lds.load_k((j + 4) * BN, f_a_buf, page_id=page_f_a)
                kv_gmem_to_lds.load_k((j + 5) * BN, f_b_buf, page_id=page_f_b)

                v_s_a = gemm_helper.qk(v_k_a, q_wide)
                kv_gmem_to_lds.store_v_source(
                    next_v_a, nn_a_buf, (j + 2) * BN, mask_padding=mask_v
                )
                v_f_a = kv_gmem_to_lds.load_v_source((j + 4) * BN, page_id=page_f_a)
                v_f_b = kv_gmem_to_lds.load_v_source((j + 5) * BN, page_id=page_f_b)
                v_s_b = gemm_helper.qk(v_k_b, q_wide)
                kv_gmem_to_lds.store_v_source(
                    next_v_b, nn_b_buf, (j + 3) * BN, mask_padding=mask_v
                )
                if const_expr(do_mask):
                    v_s_a, v_s_b = softmax_helper.causal_mask_pair_if_needed(
                        v_s_a, v_s_b, j
                    )
                m_new = m_row
                if const_expr(initialize):
                    # Zero O/l need no rescaling; avoid folding -inf through fast math.
                    with fx.fastmath(None):
                        m_new = _merge_tile_max(v_s_a, v_s_b)
                elif const_expr(not skip_max):
                    m_tile = _merge_tile_max(v_s_a, v_s_b)
                    v_o, m_new, l_row = _correct_o(v_o, m_row, l_row, m_tile)
                v_o = softmax_helper.anchor_v_o(v_o)

                v_o, l_row = _subtile_tail(v_s_a, v_v_a, v_o, l_row, m_new, a_buf)
                v_v_b = _load_v_first(b_buf)
                v_o, l_row = _subtile_tail(v_s_b, v_v_b, v_o, l_row, m_new, b_buf)
                m_row = m_new
                next_args = [m_row, l_row] + v_o + [nn_a_buf, v_f_a, v_f_b]

            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            return next_args

        # V128 stores packets j+2/j+3; segmented V stores j+4/j+5. Keep
        # byte masking out of iterations whose furthest packet is fully valid.
        if const_expr(traits.PAGE_SIZE == 1):
            v_prefix_end = fx.Int64(t_end)
        else:
            v_ahead = 4 if traits.FP8_PV_SEGMENTED else 2
            v_prefix_end = fx.max(
                fx.Int64(0), fx.Int64(ctx.seqlen_kv_v) // (2 * BN) * 2 - v_ahead
            )

        if const_expr(BOUNDED_MAX):
            first_end = fx.min(fx.Int64(t_end), fx.Int64(t0) + 2)
            for j, loop_args in range(
                fx.Int64(t0), first_end, fx.Int64(2), init=init_args
            ):
                next_args = _iterate(j, loop_args, True, initialize=True)
                loop_results = yield next_args
            sealed = _query_bound_is_safe(ctx, upper_bound, fx.Float32(loop_results[0]))
            fast_end = sealed.select(fx.Int64(t_end), first_end)
            fast_prefix_end = fx.min(fast_end, fx.max(first_end, v_prefix_end))
            first_state = loop_results
            # Waves can choose different loops, but pair order and barrier count agree.
            for j, loop_args in range(
                first_end, fast_prefix_end, fx.Int64(2), init=first_state
            ):
                next_args = _iterate(j, loop_args, True, skip_max=True, mask_v=False)
                loop_results = yield next_args
            fast_tail_state = loop_results
            for j, loop_args in range(
                fast_prefix_end, fast_end, fx.Int64(2), init=fast_tail_state
            ):
                next_args = _iterate(j, loop_args, True, skip_max=True)
                loop_results = yield next_args
            slow_start = sealed.select(fx.Int64(t_end), first_end)
            slow_prefix_end = fx.min(fx.Int64(t_end), fx.max(slow_start, v_prefix_end))
            slow_state = loop_results
            for j, loop_args in range(
                slow_start, slow_prefix_end, fx.Int64(2), init=slow_state
            ):
                next_args = _iterate(j, loop_args, True, mask_v=False)
                loop_results = yield next_args
            slow_tail_state = loop_results
            for j, loop_args in range(
                slow_prefix_end, fx.Int64(t_end), fx.Int64(2), init=slow_tail_state
            ):
                next_args = _iterate(j, loop_args, True)
                loop_results = yield next_args
        else:
            mask_prefix = (traits.HEAD_DIM, traits.HEAD_DIM_V) != (192, 128)
            prefix_end = v_prefix_end
            if const_expr(not mask_prefix):
                # This boundary is wave-uniform, not CTA-uniform. Both loops
                # must keep the same pair order and one rendezvous per pair.
                causal_prefix_end = (
                    fx.Int64(ctx.q_start_pos_i32 + ctx.delta_i32) // (2 * BN) * 2
                )
                prefix_end = fx.min(causal_prefix_end, v_prefix_end)
            prefix_end = fx.min(fx.Int64(t_end), fx.max(fx.Int64(t0), prefix_end))
            for j, loop_args in range(
                fx.Int64(t0), prefix_end, fx.Int64(2), init=init_args
            ):
                next_args = _iterate(j, loop_args, mask_prefix, mask_v=False)
                loop_results = yield next_args
            tail_init = loop_results
            for j, loop_args in range(
                prefix_end, fx.Int64(t_end), fx.Int64(2), init=tail_init
            ):
                next_args = _iterate(j, loop_args, True)
                loop_results = yield next_args
        m_row = loop_results[0]
        l_row = loop_results[1]
        v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]

        inv_l = softmax_helper.safe_l_inv(l_row)
        value_descale = _load(fx.get_iter(ctx.VDescale), dtype=fx.Float32, count=1)
        inv_l = inv_l * value_descale
        softmax_helper.scale_o(v_o, inv_l)
        rocdl.s_barrier()
        output_store.store_final_o(v_o, q_row)

    @flyc.jit
    def launch_flash_attn_dualwave_swp(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        CuSeqQ: fx.Tensor,
        KvMetadata: fx.Tensor,
        LastPageLens: fx.Tensor,
        BlockTable: fx.Tensor,
        block_table_stride: fx.Int32,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_o_n: fx.Int32,
        softmax_scale: fx.Float32,
        stream: fx.Stream,
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _dualwave_swp_fp8_cache_tag
        bs_idx = fx.Int64(batch_size)
        sl_idx = fx.Int64(seq_len)
        num_q_blocks = (sl_idx + BLOCK_M - 1) // BLOCK_M
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
        kernel_attrs = {
            "rocdl.waves_per_eu": waves_per_eu,
            "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
            "passthrough": passthrough_entries,
        }
        flash_attn_paged_fp8_bn128_kernel(
            Q,
            K,
            V,
            O,
            CuSeqQ,
            KvMetadata,
            LastPageLens,
            BlockTable,
            block_table_stride,
            QDescale,
            KDescale,
            VDescale,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_o_n,
            softmax_scale,
            value_attrs=kernel_attrs,
        ).launch(
            grid=(
                NUM_HEADS_Q * BATCH_INTERLEAVE_GROUP,
                num_q_blocks,
                grid_z // BATCH_INTERLEAVE_GROUP,
            ),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch_flash_attn_dualwave_swp.compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": True,
            "lsr-drop-solution": True,
            "disable-machine-sink": True,
        },
    }

    def _validate_paged_bn128_launch(seq_len_kv, block_table_stride):
        if not PAIRED_PAGE_IDS:
            return
        seq_len_kv = int(seq_len_kv)
        block_table_stride = int(block_table_stride)
        num_kv_pages = (seq_len_kv + traits.PAGE_SIZE - 1) // traits.PAGE_SIZE
        if num_kv_pages < 2 or num_kv_pages % 2 != 0:
            raise ValueError(
                "paged BN128 requires a positive even number "
                f"of KV pages; got seq_len_kv={seq_len_kv}, "
                f"page_size={traits.PAGE_SIZE}"
            )
        if metadata_mode == "block_table" and block_table_stride < num_kv_pages:
            raise ValueError(
                f"paged BN128 block table has too few entries: need {num_kv_pages}, got stride {block_table_stride}"
            )

    def _validate_batch_interleave_launch(batch_size):
        if int(batch_size) % BATCH_INTERLEAVE_GROUP != 0:
            raise ValueError(
                "paged FP8 batch size must be divisible by its interleave group: "
                f"batch_size={int(batch_size)}, group={BATCH_INTERLEAVE_GROUP}"
            )

    def _launch(
        Q,
        K,
        V,
        O,
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,
        stride_o_n=None,
        softmax_scale=None,
        *,
        seq_len_kv=None,
        cu_seqlens_q=None,
        kv_metadata=None,
        kv_last_page_lens=None,
        block_table=None,
        block_table_stride=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        stream=None,
        _compile_only=False,
    ):
        if (
            cache_buffered
            and max(K.numel() * K.element_size(), V.numel() * V.element_size())
            > PAGED_FP8_BUFFER_LIMIT_BYTES
        ):
            raise ValueError(
                "paged FP8 whole-cache buffer descriptor exceeds its byte limit"
            )
        if (
            cu_seqlens_q is None
            or kv_metadata is None
            or block_table is None
            or block_table_stride is None
            or q_descale is None
            or k_descale is None
            or v_descale is None
        ):
            raise ValueError(
                "paged FP8 flash_attn requires cu_seqlens_q, kv_metadata, block_table, "
                "block_table_stride, q_descale, k_descale, and v_descale"
            )
        if has_last_page_lens and kv_last_page_lens is None:
            raise ValueError("paged FP8 CSR launch requires kv_last_page_lens")
        for metadata in (cu_seqlens_q, kv_metadata, block_table, kv_last_page_lens):
            if (
                metadata is not None
                and metadata.numel() * metadata.element_size()
                > PAGED_FP8_BUFFER_LIMIT_BYTES
            ):
                raise ValueError(
                    "paged FP8 metadata exceeds the signed-int32 byte limit"
                )
        # stride_kv_n is accepted for compatibility; native cache layouts fix it.
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if stride_o_n is None:
            stride_o_n = DEFAULT_STRIDE_O_N
        if softmax_scale is None:
            softmax_scale = HEAD_DIM**-0.5
        if seq_len_kv is None:
            seq_len_kv = seq_len
        _validate_paged_bn128_launch(seq_len_kv, block_table_stride)
        _validate_batch_interleave_launch(batch_size)
        dispatch = _preload_compiled if _compile_only else _run_compiled
        return dispatch(
            launch_flash_attn_dualwave_swp,
            Q,
            K,
            V,
            O,
            cu_seqlens_q,
            kv_metadata,
            kv_last_page_lens if kv_last_page_lens is not None else cu_seqlens_q,
            block_table,
            block_table_stride,
            q_descale,
            k_descale,
            v_descale,
            batch_size,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_o_n,
            softmax_scale,
            fx.Stream(stream),
        )

    def _compile(*args, **kwargs):
        return _launch(*args, _compile_only=True, **kwargs)

    _launch.compile = _compile

    return _launch
