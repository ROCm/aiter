# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""gfx950 DUALWAVE_SWP FP8 flash attention."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.ops.flydsl.kernels.fmha_gfx950.op_combine import (
    DualwaveSplitKCombineContext,
    DualwaveSplitKCombineHelper,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.op_epilog import DualwaveFp8StoreHelper
from aiter.ops.flydsl.kernels.fmha_gfx950.op_gemm import DualwaveFp8GemmHelper
from aiter.ops.flydsl.kernels.fmha_gfx950.op_lds import (
    DualwaveFp8KvGmemToLdsLoader,
    DualwaveFp8KvLdsToVgprLoader,
    DualwaveFp8QLoader,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.op_softmax import DualwaveFp8SoftmaxHelper
from aiter.ops.flydsl.kernels.fmha_gfx950.pipeline import (
    DualwaveFp8KernelContext,
    _make_dualwave_swp_fp8_traits,
    _s_setprio,
    _stagger_extra_barrier_if_one,
    _waitcnt_vm_n,
    dualwave_fp8_dma_per_iter,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def build_flash_attn_dualwave_swp_fp8_module(
    num_heads,
    head_dim,
    head_dim_v=None,
    causal=True,
    num_kv_heads=None,
    daz=True,
    dualwave_swp_lazy_rescale=True,
    rescale_threshold=6.0,
    dualwave_swp_setprio=True,
    dualwave_swp_enable_stagger=True,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    block_m=256,
    batch_interleave_group=1,
    return_lse=False,
):
    """Build the gfx950 dual-wave fp8 launcher (dense, packed varlen, or split-K)."""
    gpu_arch = get_hip_arch()

    if not gpu_arch.startswith("gfx950"):
        raise RuntimeError(
            f"flash_attn_dualwave_swp requires gfx950+ (uses ds_read_tr16_b64), got {gpu_arch}"
        )
    if head_dim_v is None:
        head_dim_v = head_dim

    if num_kv_heads is None:
        num_kv_heads = num_heads
    assert num_heads % num_kv_heads == 0
    NUM_KV_SPLITS = int(num_kv_splits)
    assert NUM_KV_SPLITS >= 1

    # All compile-time tile/layout constants live in the fp8 traits object.
    traits = _make_dualwave_swp_fp8_traits(
        num_heads,
        num_kv_heads,
        head_dim,
        head_dim_v=head_dim_v,
        block_m=block_m,
        causal=causal,
        daz=daz,
        dualwave_swp_lazy_rescale=dualwave_swp_lazy_rescale,
        rescale_threshold=rescale_threshold,
        dualwave_swp_setprio=dualwave_swp_setprio,
        dualwave_swp_enable_stagger=dualwave_swp_enable_stagger,
        num_kv_splits=num_kv_splits,
        varlen=varlen,
        cross_seqlen=cross_seqlen,
        batch_interleave_group=batch_interleave_group,
        return_lse=return_lse,
    )
    # Builder-level aliases used by SharedStorage and the launch/compile wrappers.
    SPLITK = traits.SPLITK
    BLOCK_M = traits.BLOCK_M
    BLOCK_SIZE = traits.BLOCK_SIZE
    HEAD_DIM = traits.HEAD_DIM
    NUM_HEADS_Q = traits.NUM_HEADS_Q
    BATCH_INTERLEAVE_GROUP = traits.BATCH_INTERLEAVE_GROUP
    RETURN_LSE = traits.RETURN_LSE
    DEFAULT_STRIDE_Q_N = traits.DEFAULT_STRIDE_Q_N
    DEFAULT_STRIDE_O_N = traits.DEFAULT_STRIDE_O_N
    DEFAULT_STRIDE_KV_N = traits.DEFAULT_STRIDE_KV_N
    _dualwave_swp_fp8_cache_tag = traits.cache_tag
    _lds_elem_dtype = fx.Float8E4M3FN

    # fx.Array rejects a length of 0.
    _q_lds_elems = BLOCK_M * HEAD_DIM if traits.QLDS else 16

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
        vt: fx.Array[fx.BFloat16, traits.VT_BF16_TOTAL, 16]
        q: fx.Array[_lds_elem_dtype, _q_lds_elems, 16]

    # BN128: two BLOCK_N=64 KV tiles per iteration, one merged softmax correction.
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_dualwave_swp_fp8_bn128_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        Workspace: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        LSE: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        lse_stride_h: fx.Int32,
    ):
        ctx = DualwaveFp8KernelContext(
            traits,
            Q,
            K,
            V,
            O,
            Workspace,
            CuSeqQ,
            CuSeqKv,
            QDescale,
            KDescale,
            VDescale,
            LSE,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            lse_stride_h,
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

        q_loader = DualwaveFp8QLoader(ctx)
        gemm_helper = DualwaveFp8GemmHelper(ctx)
        softmax_helper = DualwaveFp8SoftmaxHelper(ctx)
        kv_gmem_to_lds = DualwaveFp8KvGmemToLdsLoader(ctx)
        kv_lds_to_regs = DualwaveFp8KvLdsToVgprLoader(ctx)
        output_store = DualwaveFp8StoreHelper(ctx)

        BN = traits.BLOCK_N
        D_CHUNKS = traits.D_CHUNKS
        NPF = const_expr(traits.NUM_PREFETCH_K)
        t0 = ctx.split_t0
        t_end = ctx.split_t_end

        def _pp_prio(v):
            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                _s_setprio(v)

        DMA_PER_ITER = const_expr(dualwave_fp8_dma_per_iter(traits))

        def _phase_bar():
            _waitcnt_vm_n(DMA_PER_ITER)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def _iter_end_bar():
            _waitcnt_vm_n(DMA_PER_ITER)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        def _softmax_part(v_s, l_row, m_new):
            v_s = softmax_helper.sub_m(v_s, m_new)
            v_p = softmax_helper.exp2(v_s, 0)
            v_p = softmax_helper.exp2(v_p, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = gemm_helper.cast_p_fp8_direct(v_p)
            return v_p, l_row

        def _pv_part(v_p, v_v, v_o):
            v_o = gemm_helper.pv(v_p, v_v, v_o)
            return softmax_helper.anchor_v_o(v_o)

        def _mask_sub(v_s, tile_idx):
            if const_expr(traits.CAUSAL):
                return v_s
            return softmax_helper.seq_pad_mask_if_needed(v_s, tile_idx)

        def _mask_pair(v_s_a, v_s_b, j):
            if const_expr(traits.CAUSAL):
                return softmax_helper.causal_mask_pair_if_needed(v_s_a, v_s_b, j)
            return v_s_a, v_s_b

        def _correct_o(v_o, m_row, l_row, m_tile):
            if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                return softmax_helper.lazy_correct_o(v_o, m_row, l_row, m_tile)
            m_new, corr = softmax_helper.rescale_from_tile_max(m_row, m_tile)
            softmax_helper.scale_o(v_o, corr)
            return v_o, m_new, softmax_helper.apply_l_rescale(l_row, corr)

        def _merge_tile_max(v_s_a, v_s_b):
            m_tile = softmax_helper.max2(
                softmax_helper.reduce_max(v_s_a), softmax_helper.reduce_max(v_s_b)
            )
            m_tile = softmax_helper.floor_masked_max(m_tile)
            return m_tile

        def _load_q_regs():
            ctx.init_q_row()
            return ctx.q_row, gemm_helper.load_q_wide()

        if const_expr(not traits.QLDS):
            q_row, q_wide = _load_q_regs()
        else:
            q_row, q_wide = None, None

        kv_gmem_to_lds.load_k(t0 * BN, t0 % fx.Index(NPF))
        if const_expr(traits.QLDS):
            q_loader.stage_q_to_lds()
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            q_row, q_wide = _load_q_regs()

        kv_gmem_to_lds.load_k((t0 + 1) * BN, (t0 + 1) % fx.Index(NPF))
        kv_gmem_to_lds.load_v(t0 * BN, t0 % fx.Index(NPF))
        kv_gmem_to_lds.load_v((t0 + 1) * BN, (t0 + 1) % fx.Index(NPF))
        kv_gmem_to_lds.load_k((t0 + 2) * BN, (t0 + 2) % fx.Index(NPF))
        kv_gmem_to_lds.load_k((t0 + 3) * BN, (t0 + 3) % fx.Index(NPF))
        kv_gmem_to_lds.load_v((t0 + 2) * BN, (t0 + 2) % fx.Index(NPF))
        kv_gmem_to_lds.load_v((t0 + 3) * BN, (t0 + 3) % fx.Index(NPF))
        if const_expr(traits.QLDS):
            rocdl.s_waitcnt(0)
        else:
            _waitcnt_vm_n(DMA_PER_ITER)
        rocdl.sched_barrier(0)
        rocdl.s_barrier()
        rocdl.sched_barrier(0)

        if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
            _stagger_extra_barrier_if_one(ctx.stagger_i32)
        _pp_prio(1)

        # Seed the running max at the same floor `floor_masked_max` clamps every
        # tile max to, not -inf: max(floor, m_tile) == max(-inf, m_tile) for any
        # tile the kernel can produce, but the kernel compiles with fast fp math
        # (nnan+ninf), under which a *compile-time* -inf reaching an arithmetic op
        # is poison. A q-block whose causal window is empty skips the tile loop
        # entirely and carries this seed straight into `m_row * c_logit_scale` in
        # the epilogue, so with -inf that multiply yields a garbage register and
        # the fully-masked row's LSE comes out NaN instead of -inf.
        m_row = ctx.c_neg_floor
        l_row = ctx.c_zero_f
        v_o = [ctx.c_zero_v16f32 for _ in range_constexpr(D_CHUNKS)]

        NPF_I = const_expr(fx.Index(NPF))

        def _ring_wrap(x):
            return (x >= NPF_I).select(x - NPF_I, x)

        init_args = [m_row, l_row] + v_o + [t0 % fx.Index(NPF)]
        loop_results = init_args
        for j, loop_args in range(fx.Index(t0), t_end, 2, init=init_args):
            m_row = loop_args[0]
            l_row = loop_args[1]
            v_o = [loop_args[2 + i] for i in range_constexpr(D_CHUNKS)]

            a_buf = loop_args[2 + D_CHUNKS]
            b_buf = _ring_wrap(a_buf + 1)
            nn_a_buf = _ring_wrap(a_buf + 2)
            f_a_buf = _ring_wrap(a_buf + 4)
            f_b_buf = _ring_wrap(a_buf + 5)

            v_k_a = kv_lds_to_regs.load_k(a_buf)
            v_k_b = kv_lds_to_regs.load_k(b_buf)

            v_s_a = gemm_helper.qk(v_k_a, q_wide)
            v_s_b = gemm_helper.qk(v_k_b, q_wide)
            v_v_a = kv_lds_to_regs.load_v(a_buf)

            kv_gmem_to_lds.load_k((j + 4) * BN, f_a_buf)
            kv_gmem_to_lds.load_k((j + 5) * BN, f_b_buf)
            kv_gmem_to_lds.load_v((j + 4) * BN, f_a_buf)
            kv_gmem_to_lds.load_v((j + 5) * BN, f_b_buf)

            _phase_bar()
            _pp_prio(0)
            v_s_a = _mask_sub(v_s_a, j)
            v_s_b = _mask_sub(v_s_b, j + 1)
            v_s_a, v_s_b = _mask_pair(v_s_a, v_s_b, j)
            m_tile = _merge_tile_max(v_s_a, v_s_b)
            v_o, m_new, l_row = _correct_o(v_o, m_row, l_row, m_tile)
            v_o = softmax_helper.anchor_v_o(v_o)
            v_p_a, l_row = _softmax_part(v_s_a, l_row, m_new)
            _phase_bar()
            _pp_prio(1)
            v_v_b = kv_lds_to_regs.load_v(b_buf)
            v_o = _pv_part(v_p_a, v_v_a, v_o)
            _phase_bar()
            _pp_prio(0)
            v_p_b, l_row = _softmax_part(v_s_b, l_row, m_new)
            m_row = m_new

            _pp_prio(1)
            v_o = _pv_part(v_p_b, v_v_b, v_o)
            _iter_end_bar()
            loop_results = yield [m_row, l_row] + v_o + [nn_a_buf]
        m_row = loop_results[0]
        l_row = loop_results[1]
        v_o = [loop_results[2 + i] for i in range_constexpr(D_CHUNKS)]

        inv_l_rcp = rocdl.rcp(T.f32, _raw(l_row))
        inv_l = fx.Float32(
            (fx.Float32(l_row) > ctx.c_zero_f).select(inv_l_rcp, ctx.c_zero_f)
        )
        inv_l = inv_l * ctx.vd_fp8
        softmax_helper.scale_o(v_o, inv_l)
        rocdl.s_barrier()
        if const_expr(not SPLITK):
            output_store.store_final_o(v_o, q_row, m_row, l_row)
        else:
            output_store.store_splitk_partial_o(v_o, m_row, l_row, q_row)
            output_store.store_empty_split()

    # Combine kernel: out = sum_s w_s * O_s / sum_s w_s * l_s, w_s = exp2(m_s - m_max).
    # One wave row of 32 lanes covers a (b, h, s) row, 4 contiguous cols/lane.
    COMBINE_BLOCK = 256
    COMBINE_LANES_PER_ROW = traits.HEAD_DIM_V // 4
    COMBINE_ROWS_PER_BLOCK = COMBINE_BLOCK // COMBINE_LANES_PER_ROW

    @flyc.kernel(known_block_size=[COMBINE_BLOCK, 1, 1])
    def flash_attn_splitk_combine_kernel(
        O: fx.Tensor,
        WS: fx.Tensor,
        CuSeqQ: fx.Tensor,
        LSE: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stride_o_n: fx.Int32,
        lse_stride_h: fx.Int32,
    ):
        ctx = DualwaveSplitKCombineContext(
            traits,
            O,
            WS,
            batch_size,
            seq_len,
            stride_o_n,
            CuSeqQ=CuSeqQ,
            LSE=LSE,
            lse_stride_h=lse_stride_h,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_thread_mapping(COMBINE_ROWS_PER_BLOCK, COMBINE_LANES_PER_ROW)
        ctx.init_workspace()
        ctx.init_descriptors()

        combine = DualwaveSplitKCombineHelper(ctx)
        m_s, l_s = combine.load_ml_rows()
        m_max = combine.reduce_m_max(m_s)
        acc, den = combine.accumulate_splits(m_s, l_s, m_max)
        if const_expr(RETURN_LSE):
            combine.store_lse(m_max, den)
        o_pack = combine.pack_output(acc, den)
        combine.store_output(o_pack)

    @flyc.jit
    def launch_flash_attn_dualwave_swp(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,
        Workspace: fx.Tensor,
        CuSeqQ: fx.Tensor,
        CuSeqKv: fx.Tensor,
        QDescale: fx.Tensor,
        KDescale: fx.Tensor,
        VDescale: fx.Tensor,
        LSE: fx.Tensor,
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_q_n: fx.Int32,
        stride_kv_n: fx.Int32,
        head_dim_runtime: fx.Int32,
        lse_stride_h: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008  framework idiom
    ):
        # Make shape/mode traits visible to the JIT cache key.
        _ = _dualwave_swp_fp8_cache_tag
        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_blocks = (sl_idx + BLOCK_M - 1) // BLOCK_M
        if const_expr(SPLITK):
            grid_z = bs_idx * NUM_KV_SPLITS
        elif const_expr(BATCH_INTERLEAVE_GROUP > 1):
            grid_z = bs_idx // BATCH_INTERLEAVE_GROUP
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
            Workspace,
            CuSeqQ,
            CuSeqKv,
            QDescale,
            KDescale,
            VDescale,
            LSE,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            lse_stride_h,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": f"{BLOCK_SIZE},{BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(NUM_HEADS_Q * BATCH_INTERLEAVE_GROUP, num_q_blocks, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )
        if const_expr(SPLITK):
            # One batch per y block keeps the combine kernel's O descriptor wave-uniform.
            combine_rows = NUM_HEADS_Q * sl_idx
            combine_blocks = (
                combine_rows + (COMBINE_ROWS_PER_BLOCK - 1)
            ) // COMBINE_ROWS_PER_BLOCK
            if const_expr(traits.HEAD_DIM_V == HEAD_DIM):
                stride_o_n = stride_q_n
            else:
                stride_o_n = fx.Int32(DEFAULT_STRIDE_O_N)
            flash_attn_splitk_combine_kernel(
                O, Workspace, CuSeqQ, LSE, batch_size, seq_len, stride_o_n, lse_stride_h
            ).launch(
                grid=(combine_blocks, bs_idx, 1),
                block=(COMBINE_BLOCK, 1, 1),
                stream=stream,
            )

    _dualwave_swp_llvm_options = {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
        "disable-machine-sink": True,
    }

    _dualwave_swp_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": _dualwave_swp_llvm_options,
    }

    def _prepare(
        Q,
        K,
        V,
        O,
        batch_size,
        seq_len,
        stride_kv_n=None,
        stride_q_n=None,
        head_dim_runtime=None,
        *,
        seq_len_kv=None,
        workspace=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        lse=None,
        lse_stride_h=0,
        stream=None,
    ):
        """Normalise the launch arguments shared by the run and compile paths."""
        if stride_kv_n is None:
            stride_kv_n = DEFAULT_STRIDE_KV_N
        if stride_q_n is None:
            stride_q_n = DEFAULT_STRIDE_Q_N
        if head_dim_runtime is None:
            head_dim_runtime = HEAD_DIM
        # seq_len_kv defaults to seq_len (self-attention / equal Q,KV lengths).
        if seq_len_kv is None:
            seq_len_kv = seq_len
        if BATCH_INTERLEAVE_GROUP > 1 and batch_size % BATCH_INTERLEAVE_GROUP:
            raise ValueError(
                f"flash_attn_dualwave_swp fp8: batch_interleave_group={BATCH_INTERLEAVE_GROUP} requires "
                f"batch_size divisible by it, got batch_size={batch_size}"
            )
        if SPLITK and workspace is None:
            raise ValueError(
                "num_kv_splits > 1 requires a fp32 workspace (see dualwave_splitk_workspace_elems)"
            )
        # O is bf16 and would be corrupted by the fp32 LSE stores. lse_stride_h
        # sizes the LSE buffer descriptor, so leaving it at 0 gives num_records=0
        # and the hardware silently drops every LSE store.
        if RETURN_LSE and (lse is None or not lse_stride_h):
            raise ValueError(
                "return_lse=True requires a fp32 lse tensor and a non-zero "
                f"lse_stride_h, got lse={'None' if lse is None else 'tensor'}, "
                f"lse_stride_h={lse_stride_h}"
            )
        ws = workspace if SPLITK else O
        return (
            Q,
            K,
            V,
            O,
            ws,
            O if cu_seqlens_q is None else cu_seqlens_q,
            O if cu_seqlens_kv is None else cu_seqlens_kv,
            O if q_descale is None else q_descale,
            O if k_descale is None else k_descale,
            O if v_descale is None else v_descale,
            O if lse is None else lse,
            batch_size,
            seq_len,
            seq_len_kv,
            stride_q_n,
            stride_kv_n,
            head_dim_runtime,
            lse_stride_h,
            fx.Stream(stream),
        )

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return _run_compiled(
                launch_flash_attn_dualwave_swp, *_prepare(*args, **kwargs)
            )

    def _compile(*args, **kwargs):
        with CompilationContext.compile_hints(_dualwave_swp_compile_hints):
            return flyc.compile(
                launch_flash_attn_dualwave_swp, *_prepare(*args, **kwargs)
            )

    _launch.compile = _compile

    return _launch
